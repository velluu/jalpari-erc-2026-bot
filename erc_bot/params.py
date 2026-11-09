"""Pick and place geometry as a parameter file (lane G, black-box optimisation in Gazebo).

  python3 -m erc_bot.mission --digit 4 --label red --params best.json

`--params FILE` selects the parametrised pick (`erc_bot.wall:ParamPick`, which also holds the wall-book manoeuvres)
and passes the file's `place` block to the scripted place as `--place-opt` values. Without `--params` the bot runs
exactly as before.

Every default comes from `erc_bot/params_default.json`, which holds today's bot values (erc_bot/stages.py, 2026-11-05
18:45). A parameter file overrides any subset, so `--params erc_bot/params_default.json` (or `{}`) is today's bot.
Nothing numeric is hard-coded here or in wall.py.

File layout (JSON, every section optional, unknown knobs rejected so a typo never silently runs a default):
  {"pick":     {knob: value},              scripted pick knobs, every row (per-row knobs take {"1": v, ...} or a scalar)
   "row":      {"1": {knob: value}, ...},  per-row overrides of the pick knobs
   "wall":     {knob: value},              wall-book manoeuvre (books pressed against a column side wall)
   "wall_row": {"1": {knob: value}, ...},  per-row overrides of the wall knobs
   "place":    {opt: value}}               ScriptedPlace opts (relook, stand, deep, release, gap, clear)

Pick knobs: torso, reach (per row), arm_y, clear_back (base stand-off before deploying), pregrasp, insert_past,
insert_steps, step_s, dz (per row, grasp height from the spine centre), dy (lateral, base left +), lift, lift_s,
extract, extract_s, retreat, open (gripper command before insertion, 0 open).
Wall knobs (lateral values signed away from the wall, yaw positive turns the tool nose toward the wall): enable,
detect_off, push, push_off, push_dz, push_yaw, push_start, push_dx, push_steps, push_s, push_sweep, push_back, g_off,
g_dx, g_dz, g_yaw, g_open, g_pre, g_insert, g_steps, g_step_s, g_lift, g_extract, base_off, reach_add, redetect.
"""
import copy
import json
import os

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params_default.json')
SECTIONS = ('pick', 'row', 'wall', 'wall_row', 'place')


def _rows(d):
    return {int(k): v for k, v in d.items()}


def defaults():
    d = json.load(open(DEFAULT_FILE))
    return {'pick': {k: (_rows(v) if isinstance(v, dict) else v) for k, v in d['pick'].items()},
            'wall': dict(d['wall']), 'place': dict(d['place'])}


def load(path):
    """Merged parameter dict: {'pick', 'row', 'wall', 'wall_row', 'place'}, defaults from params_default.json."""
    raw = json.load(open(path)) if isinstance(path, str) else (path or {})
    bad = set(raw) - set(SECTIONS) - {'note', 'id'}
    if bad:
        raise ValueError(f'params: unknown sections {sorted(bad)}')
    base = defaults()
    p = {'pick': copy.deepcopy(base['pick']), 'wall': copy.deepcopy(base['wall']), 'row': {}, 'wall_row': {},
         'place': dict(base['place'])}
    for k, v in raw.get('place', {}).items():
        if k not in base['place']:
            raise ValueError(f'params: unknown place opt {k}')
        p['place'][k] = v
    for sec in ('pick', 'wall'):
        for k, v in raw.get(sec, {}).items():
            if k not in base[sec]:
                raise ValueError(f'params: unknown {sec} knob {k}')
            p[sec][k] = {**p[sec][k], **_rows(v)} if isinstance(v, dict) and isinstance(p[sec][k], dict) else v
    for sec, known in (('row', base['pick']), ('wall_row', base['wall'])):
        for r, kv in raw.get(sec, {}).items():
            for k in kv:
                if k not in known:
                    raise ValueError(f'params: unknown {sec} knob {k}')
            p[sec][int(r)] = dict(kv)
    return p


def knobs(p, row, wall=False):
    """Flat knobs for one row: section defaults, then the per-row overrides, per-row dicts resolved."""
    sec, rsec = ('wall', 'wall_row') if wall else ('pick', 'row')
    k = dict(p[sec]); k.update(p[rsec].get(int(row), {}))
    return {n: (v[int(row)] if isinstance(v, dict) else v) for n, v in k.items()}


def stage_args(path, pick, place, pick_opt, place_opt):
    """mission.py --params: the parametrised pick, and the place opts from the file (explicit --*-opt win)."""
    p = load(path)
    if pick == 'erc_bot.stages:ScriptedPick':
        pick = 'erc_bot.wall:ParamPick'
    pick_opt = [f'params={path}'] + list(pick_opt)
    given = {o.split('=', 1)[0] for o in place_opt}
    place_opt = [f'{k}={v}' for k, v in p['place'].items() if k not in given] + list(place_opt)
    return pick, place, pick_opt, place_opt

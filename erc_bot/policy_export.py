"""Export a lane A checkpoint for the policy runner: weights as numpy (lane A's own erc_learn.export) plus a sidecar
JSON with what the runner needs to build the observation and time the commands (task, teacher or student, control
config, source run and step). The meta also carries lane A's env constants as the code stands at export time (env_constants): re-export an old run
only if its training code had the same constants. Runs where lane A's venv and code are (Z8 or HPC), never on the laptop:

  PYTHONPATH=~/erc/learn:~/erc/policy/bot ~/venvs/erclearn/bin/python -m erc_bot.policy_export \
      --run ~/erc/learn/runs/pick_t1 [--params params_000040894464.pkl] --out ~/erc/policy/exports/pick_t1_40M.npz [--check]

--check runs lane A's brax-vs-numpy comparison (erc_learn.export --check) and stores max |numpy - brax| in the meta.
"""
import argparse, hashlib, json, os, re, subprocess, sys


def env_constants(task, cfg):
    """What the runner must reproduce, read from lane A's env as the code stands at export time (C MuJoCo model,
    no GPU): the command range (lo, hi incl. the torso), the start postures, reach and torso per row, file md5s."""
    from mujoco import mjx
    put = mjx.put_model
    mjx.put_model = lambda m, impl=None: m
    try:
        from erc_learn import env_pick, env_place
        over = {}
        for o in cfg['args'].get('override', []):
            k, v = o.split('=', 1); over[k] = json.loads(v)
        E = env_place.PlaceEnv if task == 'place' else env_pick.PickEnv
        env = E(config_overrides=over or None, nenv_hint=1)
        out = {'lo': [float(x) for x in env._lo], 'hi': [float(x) for x in env._hi],
               'md5': {f: hashlib.md5(open(getattr(mod, '__file__'), 'rb').read()).hexdigest()
                       for f, mod in (('env_pick.py', env_pick), ('env_place.py', env_place))}}
        if task == 'place':
            out.update(q_hold=[float(x) for x in env._q_hold], torso_hold=float(env._torso_hold))
        else:
            out.update(q_pre={str(r + 1): [float(x) for x in env._qpre[r]] for r in range(4)},
                       torso={str(r + 1): float(env_pick.TORSO[r]) for r in range(4)},
                       reach={str(r + 1): float(env_pick.REACH[r]) for r in range(4)}, arm_y=float(env_pick.ARM_Y))
        return out
    finally:
        mjx.put_model = put


def from_lane_a(ec, task):
    """lane A's runs/<run>/env_constants.json (written at train start) -> the runner's meta 'env' layout."""
    out = {'lo': ec['lo'], 'hi': ec['hi'], 'md5': ec.get('md5'), 'written': ec.get('written')}
    if task == 'place':
        out.update(q_hold=ec['place']['q_hold'], torso_hold=ec['place']['torso'])
    else:
        out.update(q_pre={r: v['q_pre'] for r, v in ec['pick'].items()}, torso={r: v['torso'] for r, v in ec['pick'].items()},
                   reach={str(i + 1): x for i, x in enumerate(ec.get('pick_reach', [.55, .72, .75, .85]))}, arm_y=ec.get('arm_y', -0.24))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--params', default='params_latest.pkl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--env-json', default='', help='start postures and ranges as trained (overrides the run record and the code)')
    a = ap.parse_args()
    from erc_learn.export import export
    params = a.params if os.path.isabs(a.params) else os.path.join(a.run, a.params)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    d = export(params, a.out)
    cfg = json.load(open(os.path.join(a.run, 'config.json')))
    c = cfg['config']
    task = cfg['args'].get('task', 'pick')
    student = bool(cfg['args'].get('student')) or not c.get('privileged_actor', True)
    # what this export is: PPO (progress 'step') or DAgger (progress 'env_steps', 'iter', 'beta', 'teacher')
    trainer = 'dagger' if 'teacher' in cfg['args'] else 'ppo'
    m = re.search(r'params_(\d+)\.pkl', os.path.basename(params))
    step = int(m.group(1)) if m else None
    last = {}
    if os.path.exists(os.path.join(a.run, 'progress.jsonl')):
        lines = [l for l in open(os.path.join(a.run, 'progress.jsonl')) if l.strip()]
        last = json.loads(lines[-1]) if lines else {}
    if step is None:
        step = last.get('step', last.get('env_steps'))
    if step is None:
        sys.exit(f'{params}: cannot tell its training step (no params_<step>.pkl name, no step/env_steps in progress.jsonl)')
    meta = {'task': task, 'student': student, 'obs_size': int(d['obs_mean'].shape[0]), 'act_size': int(d['act_size']),
            'run': os.path.abspath(a.run), 'params': os.path.abspath(params), 'step': step,
            'params_md5': hashlib.md5(open(params, 'rb').read()).hexdigest(),
            'config': {k: c[k] for k in ('ctrl_dt', 'sim_dt', 'episode_length', 'action_scale', 'grip_close_time', 'base_v',
                                         'margin', 'latency_max_substeps', 'residual', 'residual_scale', 'expert_mode') if k in c},
            'overrides': cfg['args'].get('override', []), 'trainer': trainer,
            'progress_last': {k: last[k] for k in ('time', 'iter', 'beta', 'teacher', 'success_rate', 'successes', 'fails',
                                                   'eval/episode_success', 'eval/episode_success_any') if k in last}}
    if trainer == 'dagger':
        meta['teacher'] = cfg['args'].get('teacher')
    # start postures and command range AS TRAINED: the run's own record if lane A wrote one, else the code now, with
    # every lane A file changed since the run started listed (a stale start posture put place_t1 out of its training
    # distribution in Gazebo on 2026-11-05: 0 of 10 placed; the replica placed 12 of 12 from its true start)
    rec = os.path.join(a.run, 'env_constants.json')
    if os.path.exists(rec):
        meta['env'] = from_lane_a(json.load(open(rec)), task); meta['env_source'] = rec
    else:
        meta['env'] = env_constants(task, cfg); meta['env_source'] = 'lane A code at export time'
        import erc_learn
        root = os.path.dirname(os.path.dirname(os.path.abspath(erc_learn.__file__)))
        t0 = os.path.getmtime(os.path.join(a.run, 'config.json'))
        files = [os.path.join(root, 'erc_learn', f) for f in ('env_pick.py', 'env_place.py', 'build_model.py', 'scene.py')] + \
                [os.path.join(root, 'assets', f) for f in ('pregrasp.json', 'place_start.json', 'erc_pick_train.xml', 'erc_place_train.xml')]
        stale = [os.path.relpath(f, root) for f in files if os.path.exists(f) and os.path.getmtime(f) > t0]
        if stale:
            meta['env_stale'] = stale
            print(f'WARNING: {", ".join(stale)} changed after this run started: the start postures and ranges in the meta may '
                  f'not be the ones it trained on. Write runs/<run>/env_constants.json or pass --env-json.', file=sys.stderr)
    if a.env_json:
        meta['env'] = json.load(open(a.env_json)); meta['env_source'] = a.env_json; meta.pop('env_stale', None)
    if a.check:
        r = subprocess.run([sys.executable, '-m', 'erc_learn.export', '--params', params, '--out', a.out + '.chk.npz', '--check',
                            '--policy', cfg['args'].get('policy', '512,256,128')], capture_output=True, text=True)
        mm = re.search(r'max \|numpy - brax\| = ([0-9.eE+-]+)', r.stdout)
        meta['check_max_abs'] = float(mm.group(1)) if mm else (r.stdout + r.stderr)[-400:]
        try:
            os.remove(a.out + '.chk.npz')
        except OSError:
            pass
    json.dump(meta, open(os.path.splitext(a.out)[0] + '.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()

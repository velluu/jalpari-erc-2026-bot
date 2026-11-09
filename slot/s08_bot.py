#!/usr/bin/env python3
"""Slot step 8: the bot's mission on the real robot, slowly, stage by stage.

  python3 slot/s08_bot.py --digit 2 --label red --until BOOK            perception and base only: find column, approach, find book
  python3 slot/s08_bot.py --digit 2 --label red --until HOLDING         pick, stop with the book held
  python3 slot/s08_bot.py --digit 2 --label red                         full mission
  python3 slot/s08_bot.py --digit 2 --label red --pick pkg.mod:Class --pick-opt checkpoint=...   learned pick stage

Same code as `python3 -m erc_bot.mission`, with:
  --speed F   joint motions at F x the bot's own 60 % of each cap (default 0.5, so 30 % of the caps), base vmax x F
  --until S   stop cleanly right after stage S is logged (START COLUMN BOOK DEPLOY_ARM INSERT GRIPPER HOLDING
              CARRY_POSTURE BIN PLACE DONE ...); the robot holds where it is
  --step      wait for Enter at every stage (the operator stays 1.5 m away and watches; Ctrl-C stops)
mission.json lands in $SLOT_OUT/bot_<time>/, plus one verdict per stage reached.
"""
import argparse, os, sys, time
from common import Verdicts, out_dir, SIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=0.5)
    ap.add_argument('--until', default='')
    ap.add_argument('--step', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--force-gate', action='store_true', help='run a learned stage although slot/gate.py says NO-GO')
    ap.add_argument('--auto', action='store_true', help='choose scripted or learned stages from the gates, calibration from calib.py')
    a, rest = ap.parse_known_args()
    import json, glob
    choice = {}
    cal = {}
    try:
        cal = json.load(open(os.path.join(out_dir(), 'calibration.json')))
    except Exception:
        pass
    if a.auto:
        # learned stages only when every slot gate passed and a STUDENT export exists in $ERC_POLICIES; else scripted
        from gate import gate
        ok, miss = gate()
        pol = os.environ.get('ERC_POLICIES', '')
        students, dup = {}, []
        for j in sorted(glob.glob(os.path.join(pol, '*.json'))):
            try:
                m = json.load(open(j))
            except Exception:
                continue
            npz = j[:-5] + '.npz'
            if m.get('student') and m.get('task') in ('pick', 'place') and os.path.exists(npz):
                if m['task'] in students:
                    dup.append(m['task'])
                students[m['task']] = npz
        if dup:
            # never guess between two exports for one stage: final_build.sh ships one per task
            ok, miss = False, [f"more than one student {t} policy in the kit" for t in sorted(set(dup))]
        if ok and students:
            for task, npz in students.items():
                cls = 'LearnedPick' if task == 'pick' else 'LearnedPlace'
                rest += [f'--{task}', f'erc_bot.policy_runner:{cls}', f'--{task}-opt', f'checkpoint={npz}', f'--{task}-opt', 'fallback=scripted']
                if cal.get('grip_cal'):
                    rest += [f'--{task}-opt', f"grip_cal={cal['grip_cal']}"]
            choice = {'stages': 'learned ' + ', '.join(sorted(students)) + ' (scripted fallback on a policy failure)',
                      'why': 'every slot gate passed and student policies are in the kit'}
        else:
            choice = {'stages': 'scripted', 'why': ('no student policy in the kit' if ok else 'gates missing: ' + ' | '.join(miss))}
        print(f"[slot] AUTO CHOICE: {choice['stages']}, because {choice['why']}", flush=True)
    if cal.get('hold_threshold'):
        import erc_bot.stages as st_mod, erc_bot.mission as m_mod
        st_mod.SURE_HOLD = m_mod.SURE_HOLD = float(cal['hold_threshold'])
        print(f"[slot] hold threshold {cal['hold_threshold']} from calibration.json ({cal.get('why', {}).get('hold_threshold', '')})", flush=True)
    if cal.get('place_press') and not any(x.startswith('press=') for x in rest):
        rest += ['--place-opt', f"press={cal['place_press']}"]      # step 4's hang check: aim further under the estimate
        print(f"[slot] place press {cal['place_press']} from calibration.json ({cal.get('why', {}).get('hold', '')})", flush=True)
    learned = [x for i, x in enumerate(rest) if i > 0 and rest[i - 1] in ('--pick', '--place') and not x.startswith('erc_bot.stages:')]
    if learned and a.auto:
        a.force_gate = True       # chosen from the gates above
    if learned and not a.force_gate:
        from gate import gate
        ok, miss = gate()
        if not ok:
            print('FAIL s08_bot/gate: learned stage ' + ', '.join(learned) + ' blocked, slot checks missing:\n  ' + '\n  '.join(miss)
                  + '\n  -> run those steps first (sheet order), or the scripted stages only', flush=True)
            return 2
    import erc_bot.live as live
    import erc_bot.mission as mission
    live.SPEED_USE = 0.6 * a.speed
    go0, stage0 = mission.Bot.go, mission.Bot.stage

    def go(self, *args, **kw):
        kw['vmax'] = kw.get('vmax', 0.25) * a.speed
        return go0(self, *args, **kw)

    def stage(self, name, **kw):
        stage0(self, name, **kw)
        if a.step and sys.stdin.isatty():
            self.r.stop()
            input(f'[slot] stage {name} logged, Enter to continue, Ctrl-C to stop ')
        if a.until and name == a.until:
            raise live.Abort(f'slot gate: stopped after {name}')
    mission.Bot.go, mission.Bot.stage = go, stage
    out = os.path.join(out_dir(), f"bot_{a.tag or time.strftime('%H%M%S')}")
    if '--model' not in rest and os.environ.get('ERC_BOT_MODEL'):
        rest += ['--model', os.environ['ERC_BOT_MODEL']]      # kit_env.sh points it at the kit's det-v4
    argv = rest + ['--out', out] + (['--sim-time'] if SIM and '--sim-time' not in rest else [])
    print(f'[slot] erc_bot.mission {" ".join(argv)} at speed x{a.speed}, until {a.until or "the end"}', flush=True)
    rc = mission.main(argv)
    v = Verdicts('s08_bot' + (f'_{a.tag}' if a.tag else ''))
    import json
    try:
        m = json.load(open(os.path.join(out, 'mission.json')))
    except Exception as e:
        v.check('mission.json', False, str(e), 'the bot did not start: read its log above (interface not ready?)')
        return v.done()
    gated = a.until and m.get('reason', '').startswith('slot gate')
    ok = m.get('result') == 'done' or gated
    if choice:
        v.info(f"stages: {choice['stages']} ({choice['why']})")
    v.check('mission', ok, f"result {m.get('result')}, last stage {m.get('last_stage')} at {m.get('mission_time')} s, "
            f"reason {m.get('reason')!r}", 'note the reason and the last stage, the bag holds the rest. Rerun with --until at the '
            'stage before it, or move to the next item on the sheet', result=m.get('result'), last_stage=m.get('last_stage'),
            stage_t=m.get('stage_t'), choice=choice)
    for e in m.get('events', []):
        if e['stage'] in ('GRIPPER', 'BOOK', 'BIN', 'COLUMN', 'HOLD_MEASURED', 'APPROACH_ANCHOR', 'NORMAL_ANCHOR'):
            v.info(f"{e['stage']}: {({k: x for k, x in e.items() if k not in ('stage',)})}")
    # C4's laser anchors (lane H): /scan must see the shelf panel. Each APPROACH_ANCHOR either found a laser band ahead
    # (odom_travel vs laser_travel) or logged 'no laser band'; NORMAL_ANCHOR fits the shelf normal before the bin turn.
    anc = [e for e in m.get('events', []) if e['stage'] in ('APPROACH_ANCHOR', 'NORMAL_ANCHOR')]
    if anc:
        band = [e for e in anc if e['stage'] == 'APPROACH_ANCHOR' and e.get('why') != 'no laser band']
        noband = [e for e in anc if e['stage'] == 'APPROACH_ANCHOR' and e.get('why') == 'no laser band']
        norm = [e for e in anc if e['stage'] == 'NORMAL_ANCHOR']
        dis = [abs(e['laser_travel'] - e['odom_travel']) for e in band
               if e.get('laser_travel') is not None and e.get('odom_travel') is not None]
        v.check('laser anchors', bool(band) or bool([e for e in norm if e.get('applied')]),
                f"approach: laser band found {len(band)}, none {len(noband)}"
                + (f", |laser - odom travel| max {100 * max(dis):.1f} cm" if dis else '')
                + f"; shelf normal: applied {sum(1 for e in norm if e.get('applied'))} of {len(norm)}",
                'the laser never saw the shelf panel: the bot fell back to odometry (v5.9 behaviour). Note it for the lead; '
                'if a drive went wrong, rerun step 8 with ERC_BOT_ANCHOR=0 in front of the command', anchors=anc)
    v.done()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

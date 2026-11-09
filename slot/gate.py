#!/usr/bin/env python3
"""Learned-stage gate: a learned pick or place runs on the real robot only after these slot checks passed this session.

  python3 slot/gate.py          prints GO or NO-GO with the missing checks (reads $SLOT_OUT/verdicts.jsonl)

  1. clock offset            s00_preflight/clock offset
  2. joints 1 and 5 known    s01_urdf/arm 1 and 5 convention
  3. gripper on air and book s02_gripper_air/right close on air #1 and s02_gripper_book/close on book
  4. command latency         s04_latency right-arm action and topic rows (lane E sets action_scale from them)
s08_bot.py refuses --pick / --place other than the scripted stages until this says GO (--force-gate overrides).
"""
import json, os, sys

GATES = [('clock offset', lambda r: r['script'] == 's00_preflight' and r['check'] == 'clock offset'),
         ('URDF joints 1 and 5', lambda r: r['script'] == 's01_urdf' and r['check'] == 'arm 1 and 5 convention'),
         ('gripper on air', lambda r: r['script'] == 's02_gripper_air' and r['check'].startswith('right close on air')),
         ('gripper on a 3 cm book', lambda r: r['script'] == 's02_gripper_book' and r['check'] == 'close on book'),
         ('arm latency, action', lambda r: r['script'] == 's04_latency' and r['check'].startswith('arm_right_') and ':action:' in r['check']),
         ('arm latency, topic stream', lambda r: r['script'] == 's04_latency' and r['check'].startswith('arm_right_') and ':cl10:' in r['check'])]


def gate(path=None):
    path = path or os.path.join(os.environ.get('SLOT_OUT') or os.path.join(os.getcwd(), 'slot_out'), 'verdicts.jsonl')
    rows = []
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path) if l.strip()]
    missing = []
    for name, f in GATES:
        hits = [r for r in rows if f(r)]
        if not hits or not hits[-1]["ok"]:
            missing.append(f"{name}: {'not run' if not hits else 'FAIL ' + hits[-1].get('detail', '')[:120]}")
    return not missing, missing


if __name__ == '__main__':
    ok, miss = gate(sys.argv[1] if len(sys.argv) > 1 else None)
    print('GO: learned stages may run' if ok else 'NO-GO for learned stages:\n  ' + '\n  '.join(miss))
    sys.exit(0 if ok else 1)

#!/usr/bin/env python3
"""Turns this session's measurements into the calibration the bot uses, with the reason for every choice.

  python3 slot/calib.py        reads $SLOT_OUT/*.json, writes $SLOT_OUT/calibration.json and $SLOT_OUT/grip_cal.json

Choices (no human judgement needed):
  hold threshold   air and book readings separate by >= 0.03: threshold = their midpoint (the bot treats a reading below it
                   as holding). Otherwise 0.75 stays and the hold is confirmed by sight (the bot already does that).
  grip_cal.json    for the learned stages (lane E's GripCal): live [0, book, air] -> sim [0.069, 0.030, 0.0] when the book
                   reading separates from air, else not written (the runner keeps the sim's linear map)
  joints 1 and 5   nothing to choose: the bot and the runner read them from the robot's own URDF; recorded here
  latency          the right arm's action and stream numbers, recorded for the learned stages' action_scale
"""
import glob, json, os
import numpy as np
from common import out_dir, Verdicts, save

SIM_BOOK, SIM_OPEN = 0.030, 0.069     # replica finger value on a 3 cm book, sim fully open (erc-learn STATUS 17:40)


def load(name):
    try:
        return json.load(open(os.path.join(out_dir(), name)))
    except Exception:
        return None


def main():
    v = Verdicts('calib')
    cal, why = {'hold_threshold': 0.75}, {}
    air = load('gripper_air.json') or {}
    book = load('gripper_book.json') or {}
    a = [x['close']['final'] for k, x in air.items() if k.startswith('right_') and k[6:].isdigit()]
    b = book.get('close', {}).get('final')
    a_m = float(np.median(a)) if a else None
    cal.update(grip_air=a_m, grip_book=b)
    if a_m is not None and b is not None and a_m - b >= 0.03:
        cal['hold_threshold'] = round((a_m + b) / 2, 4)
        why['hold_threshold'] = f'air {a_m:.3f} and book {b:.3f} separate by {a_m - b:.3f}: midpoint'
        g = {'live': [0.0, round(b, 4), round(a_m, 4)], 'sim': [SIM_OPEN, SIM_BOOK, 0.0],
             'source': f'slot {os.path.basename(out_dir())}: right gripper air {a_m:.4f}, 3 cm book {b:.4f}'}
        save('grip_cal.json', g); cal['grip_cal'] = os.path.join(out_dir(), 'grip_cal.json')
        why['grip_cal'] = 'written from the measured air and book readings'
    else:
        low = min([x for x in (a_m, b) if x is not None], default=None)
        if low is not None and low - 0.03 < 0.75:
            # an empty close already reads below 0.75 (fingers stop early): 0.75 would call every miss a hold
            cal['hold_threshold'] = round(max(0.05, low - 0.03), 4)
            why['hold_threshold'] = (f'air {a_m} and book {b} do not separate by 0.03 and read below 0.78: threshold '
                                     f'{cal["hold_threshold"]} under both, so a miss never reads as a hold and the bot '
                                     'confirms a hold by sight')
        else:
            why['hold_threshold'] = (f'air {a_m} and book {b} do not separate by 0.03 (or one is missing): keep 0.75, '
                                     'the bot confirms a hold by sight')
        cal['grip_cal'] = None; why['grip_cal'] = 'not written: the learned stage keeps the sim map'
    u = load('urdf_limits.json') or {}
    cal['convention_1_5'] = u.get('convention_1_5'); why['convention_1_5'] = 'read from the live URDF by the bot and the runner'
    pre = load('preflight.json') or {}
    lat = load('latency.json') or {}
    rows = [r for r in lat.get('rows', []) if r['joint'].startswith('arm_right_')]
    def med(mode, key):
        x = [r[key] for r in rows if r['mode'] == mode and r.get(key) is not None]
        return round(float(np.median(x)), 4) if x else None
    cal['arm_latency'] = {m: {'latency_s': med(m, 'latency_s'), 'lag_s': med(m, 'lag_s'), 'v_ratio': med(m, 'v_ratio'),
                              'rejections': sum(r['rejections'] for r in rows if r['mode'] == m)} for m in ('action', 'topic', 'cl10', 'cl20', 'ol10', 'ol20')}
    h = book.get('hold') or {}
    cal['hold'] = {k: h.get(k) for k in ('dz_sight', 'est_hang', 'true_hang', 'hang_excess', 'tape_bottom', 'tool_z', 'seen', 'classes')}
    cal['place_press'] = h.get('place_press')
    if h.get('true_hang') is None:
        why['hold'] = f"no sight-vs-tape pair ({h.get('skipped') or h.get('error') or ('seen ' + str(h.get('seen')) if h else 'no hold check')})"
    elif h.get('place_press'):
        why['hold'] = (f"estimated hang {h['est_hang']} m vs tape {h['true_hang']} m exceeds +0.03 by {h['hang_excess']} m: "
                       f"place press raised to {h['place_press']} (step 8 passes --place-opt press=)")
    else:
        why['hold'] = f"estimated hang {h['est_hang']} m vs tape {h['true_hang']} m: within +0.03, place press stays 0.04"
    cal['why'] = why
    save('calibration.json', cal)
    v.check('calibration', True, f"hold threshold {cal['hold_threshold']} ({why['hold_threshold']}), grip_cal "
            f"{'written' if cal['grip_cal'] else 'not written'}, arm action lag {cal['arm_latency']['action']['lag_s']} s, "
            f"lane A stream at 10 Hz reaches {cal['arm_latency']['cl10']['v_ratio']} of its speed")
    return v.done()


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Slot step 5: how the checking layer rejects, on the head only, with goals that are harmless even if accepted.

  python3 slot/s05_reject.py

Each probe is chosen so that if the layer lets it through, the head only makes a small move that stays inside its
physical range (pan +-75 deg): a range probe targets 73.3 deg (inside the 0.05 rad margin zone), a speed probe moves
0.03 rad. Self-collision is not probed (an accepted probe would collide).
Per probe: accepted, error_code, error_string, the /erc/rejections text and its delay, and whether the head moved.
Writes reject.json to $SLOT_OUT.
"""
import math
import numpy as np
from common import start, finish, save, SIM, D

H = ['head_1_joint', 'head_2_joint']


def main():
    n, v = start('s05_reject')
    if not v.check('joints', n.wait_joints(H, 10.0), 'head joints present', 'rerun s00_preflight'):
        return finish(n, v)
    n.spin_for(0.5)
    home = [n.joints[h] for h in H]
    pan_edge = D(75) - 0.03                      # inside the margin zone (the layer's bound is 75 deg - 0.05 rad)
    out = []

    def probe(name, kind, points, joints=H, expect_reject=True):
        n.js_log = []
        q_start = {h: n.joints[h] for h in H}
        n_rej = len(n.rejections)
        t0 = n.now()
        if kind == 'action':
            r = n.send_goal('head_controller', joints, points, timeout=20.0)
            r.pop('handle', None)
        else:
            n.topic('head_controller').publish(n.traj(joints, points)); r = {'accepted': None, 'error_code': None, 'error_string': ''}
        n.spin_for(max(1.5, max(t for t, _ in points) + 1.0))
        rj = n.rejections[n_rej:]
        moved = max([abs(p.get(h, q_start[h]) - q_start[h]) for _, _, p in n.js_log for h in H] + [0.0])
        n.js_log = None
        rejected = bool(rj) or r.get('accepted') is False or (r.get('error_code') not in (None, 0))
        row = {'probe': name, 'kind': kind, 'accepted': r.get('accepted'), 'error_code': r.get('error_code'),
               'error_string': r.get('error_string'), 'rejection': rj[0][1] if rj else None,
               'rejection_delay_s': round(rj[0][0] - t0, 4) if rj else None, 'moved_rad': round(moved, 4),
               'rejected': rejected}
        out.append(row)
        how = ('not accepted' if r.get('accepted') is False else f"accepted then error_code {r.get('error_code')}") if kind == 'action' else 'topic'
        v.check(name, expect_reject is None or rejected == expect_reject,
                ('MEASURED ' if expect_reject is None else '') +
                f"{'rejected' if rejected else 'NOT rejected'} ({how}), reason {row['rejection'] or row['error_string']!r}, "
                f"after {row['rejection_delay_s']} s, head moved {moved:.3f} rad",
                ('the layer accepted it: the bot\'s own clamp (0.07 rad margin, 60 % of caps) is what keeps it safe, note it'
                 if expect_reject else 'a goal the bot relies on was refused: send reject.json to the lead before running the bot'),
                **{k: row[k] for k in ('accepted', 'error_code', 'rejection_delay_s', 'moved_rad')})
        n.move('head_controller', home, frac=0.4)
        n.spin_for(0.3)
        return row

    p0, t0 = home
    probe('range action (pan 73.3 deg)', 'action', [(max(3.0, abs(pan_edge - p0) / 0.3), [pan_edge, t0])])
    probe('speed action (0.03 rad in 0.02 s)', 'action', [(0.02, [p0 + 0.03, t0])])
    probe('range topic (pan 73.3 deg)', 'topic', [(max(3.0, abs(pan_edge - p0) / 0.3), [pan_edge, t0])])
    probe('speed topic (0.03 rad in 0.02 s)', 'topic', [(0.02, [p0 + 0.03, t0])])
    probe('zero time_from_start topic (0.002 rad)', 'topic', [(0.0, [p0 + 0.002, t0])])
    # lane A / E stream whole controllers; this records whether a partial goal is legal (JTC allow_partial_joints_goal)
    probe('one joint of two (pan only, action)', 'action', [(1.0, [p0 + 0.05])], joints=['head_1_joint'], expect_reject=None)
    probe('three points inside caps (action)', 'action', [(0.5, [p0 + 0.05, t0]), (1.0, [p0 + 0.10, t0]), (1.5, [p0 + 0.05, t0])],
          expect_reject=False)
    probe('three points, middle segment 1 rad/s (action)', 'action',
          [(0.5, [p0 + 0.05, t0]), (0.55, [p0 + 0.10, t0]), (1.5, [p0 + 0.05, t0])])
    n.move('head_controller', home, frac=0.4)
    save('reject.json', {'probes': out, 'sim': SIM})
    return finish(n, v)


if __name__ == '__main__':
    raise SystemExit(main())

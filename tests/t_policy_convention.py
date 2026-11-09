"""Joint-convention adapter test (erc_bot/policy_obs.Convention). Lane A trains in the sim URDF's coordinates (right
arm 1: -4.712 .. 0.524, 5: -1.571 .. 3.665, axis +z); the Guide's real right arm 1 and 5 are the negation. Pure numpy,
runs anywhere:

  python3 tests/t_policy_convention.py --urdf tiago_pro_sim.urdf

Cases: sim URDF (id), flipped axis with negated limits (neg), the Guide's table on the same axis (wrap mod 2 pi),
the same with negation forced, and URDFs that must fail loudly. For every accepted case: the live URDF's FK of the
live command equals the sim FK of the policy's posture, sim -> live -> sim is exact, every lane A start posture is
either inside the live window or reported unreachable, and ActionMapper's range is the mapped live window around the
start posture (torso 0.055 .. 0.295), never a clamp of joint 1 to +-0.45.
"""
import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from erc_bot.policy_fk import UrdfFK
from erc_bot import policy_obs as P


def variant(urdf, axis=None, limits=None, joints=('arm_right_1_joint', 'arm_right_5_joint')):
    """axis: 'flip' or an xyz string; limits: 'neg', 'shift' (+2 pi) or (dlo, dhi) offsets."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(urdf)
    for j in root.findall('joint'):
        if j.get('name') in joints:
            a = j.find('axis'); lim = j.find('limit')
            if axis == 'flip':
                a.set('xyz', ' '.join(str(-float(v) + 0.0) for v in a.get('xyz').split()))
            elif axis:
                a.set('xyz', axis)
            lo, hi = float(lim.get('lower')), float(lim.get('upper'))
            if limits == 'neg':
                lo, hi = -hi, -lo
            elif limits == 'shift':
                lo, hi = lo + 2 * math.pi, hi + 2 * math.pi
            elif limits:
                lo, hi = lo + limits[0], hi + limits[1]
            lim.set('lower', repr(lo)); lim.set('upper', repr(hi))
    return ET.tostring(root, encoding='unicode')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--urdf', required=True); a = ap.parse_args()
    sim_txt = open(a.urdf).read()
    sim = UrdfFK(sim_txt)
    rng = np.random.default_rng(1)
    ok = True
    starts = [(f'q_pre row {r}', np.r_[P.TORSO[r], P.Q_PRE[r]]) for r in (1, 2, 3, 4)] + [('q_hold', np.r_[P.TORSO_HOLD, P.Q_HOLD])]
    # name, urdf, force, expected modes (j1, j5) or None = must raise, expected unreachable starts
    cases = [('sim URDF', variant(sim_txt), None, ['id', 'id'], []),
             ('flipped axis + negated limits', variant(sim_txt, 'flip', 'neg'), None, ['neg', 'neg'], []),
             ('Guide table, same axis', variant(sim_txt, None, 'neg'), None, ['wrap', 'wrap'], ['q_pre row 4']),
             ('limits shifted by +2 pi', variant(sim_txt, None, 'shift'), None, ['wrap', 'wrap'], []),
             ('Guide table, negation forced', variant(sim_txt, None, 'neg'),
              {'arm_right_1_joint': 'neg', 'arm_right_5_joint': 'neg'}, ['neg', 'neg'], []),
             ('flipped axis, sim limits', variant(sim_txt, 'flip'), None, None, []),
             ('flipped axis, other limits', variant(sim_txt, 'flip', (0.3, 0.0)), None, None, []),
             ('axis not along z', variant(sim_txt, '0 1 0'), None, None, [])]
    for name, txt, force, want, want_unreach in cases:
        live = UrdfFK(txt)
        try:
            conv = P.Convention.from_fk(live, force=force)
        except P.ConventionError as e:
            res = want is None
            ok &= res
            print(f'{name:30s} ConventionError: {e}  {"PASS" if res else "FAIL"}')
            continue
        if want is None:
            print(f'{name:30s} accepted, expected a failure  FAIL'); ok = False; continue
        modes = [conv.modes[1], conv.modes[5]]
        unreach, fk_err, rt, outside, ranges = [], 0.0, 0.0, [], []
        for sname, q in starts:
            try:
                ql = conv.to_live(q, sname)
            except P.ConventionError:
                unreach.append(sname); continue
            rt = max(rt, float(np.abs(conv.to_sim(ql) - q).max()))
            for i, n in enumerate(P.JOINTS8[1:], 1):
                l, h = live.joints[n]['range']
                if not (l + 0.05 <= ql[i] <= h - 0.05):
                    outside.append(f'{sname} {n}={ql[i]:.3f}')
            lo, hi = conv.live_range_sim(live, around=q)
            mp = P.ActionMapper(live_lo=lo, live_hi=hi)
            ranges.append((sname, round(mp.lo[1], 3), round(mp.hi[1], 3), round(mp.lo[5], 3), round(mp.hi[5], 3)))
            ok_range = (mp.lo[1] <= q[1] <= mp.hi[1] and mp.lo[5] <= q[5] <= mp.hi[5] and abs(mp.lo[0] - 0.055) < 1e-9
                        and abs(mp.hi[0] - 0.295) < 1e-9 and mp.hi[1] - mp.lo[1] > 1.0)
            if not ok_range:
                outside.append(f'{sname} mapper range {ranges[-1]}')
            # the posture the robot takes on the live command (its own URDF) is the policy's posture (sim URDF);
            # only where the URDF is believed: a forced 'neg' on a same-axis URDF is the opposite assumption
            if not (force and any(conv.axis[i] > 0 and conv.modes[i] == 'neg' for i in (1, 5))):
                for _ in range(50):
                    qq = q + rng.uniform(-0.2, 0.2, 8); qq[0] = q[0]
                    try:
                        ql2 = conv.to_live(qq)
                    except P.ConventionError:
                        continue
                    T_ref = sim.pose('gripper_right_grasping_link', dict(zip(P.JOINTS8, qq)))
                    T_got = live.pose('gripper_right_grasping_link', dict(zip(P.JOINTS8, ql2)))
                    fk_err = max(fk_err, float(np.abs(T_ref - T_got).max()))
            T_obs = live.pose('gripper_right_grasping_link', conv.fk_q(q))
            fk_err = max(fk_err, float(np.abs(T_obs - sim.pose('gripper_right_grasping_link', dict(zip(P.JOINTS8, q)))).max()))
        res = modes == want and unreach == want_unreach and not outside and fk_err < 1e-9 and rt < 1e-9
        ok &= res
        print(f'{name:30s} modes {modes} (want {want}) | unreachable {unreach or "none"} | FK err {fk_err:.1e} | round trip {rt:.0e} | '
              f'outside live window {outside or "none"} | mapper j1/j5 ranges {[r[1:] for r in ranges[:2]]}  {"PASS" if res else "FAIL"}')
        warn = {k: v['warning'] for k, v in conv.report.items() if 'warning' in v}
        if warn:
            print(f'{"":30s} warnings: {warn}')
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

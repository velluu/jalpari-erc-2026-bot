"""Offline replay: place descent (v4 clearance) to tool z 0.87, then the retract (back 6 cm, up) and the park waypoint from each pose. python3 tests/t_replay_retract.py v4 1 2 3 4"""
import os, sys, numpy as np, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from erc_bot.kinematics import Kinematics, transform, transport_path
import math
D = math.radians
src = open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/erc_bot/live.py').read()
a = src.index('LIVE_LIMITS = {'); b = src.index('MARGIN = 0.07') + len('MARGIN = 0.07')
exec(src[a:b])
st = open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/erc_bot/stages.py').read()
for name in ('ROW_Z', 'TORSO', 'REACH', 'ARM_Y', 'GRASP_DZ'):
    line = next(l for l in st.splitlines() if l.startswith(name + ' ='))
    exec(line.split('#')[0])
from scipy.spatial.transform import Rotation
U = open(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/tests/data/tiago_pro_sim.urdf').read()
root = ET.fromstring(U)
par = {j.find('child').get('link'): j for j in root.findall('joint')}
def base_T_torso(z):
    T = np.eye(4); link = 'torso_lift_link'; chain = []
    while link != 'base_footprint':
        j = par[link]; chain.append(j); link = j.find('parent').get('link')
    for j in reversed(chain):
        o = j.find('origin'); xyz = [float(v) for v in o.get('xyz', '0 0 0').split()]; rpy = [float(v) for v in o.get('rpy', '0 0 0').split()]
        T = T @ transform(xyz, rpy)
        if j.get('type') == 'prismatic':
            a = np.array([float(v) for v in j.find('axis').get('xyz').split()]); M = np.eye(4); M[:3, 3] = a * z; T = T @ M
    return T
k = Kinematics(U, tip='gripper_right_grasping_link', limits=LIVE_LIMITS, margin=MARGIN)
mode = sys.argv[1] if len(sys.argv) > 1 else 'v3'
def solve(goal_base, q, torso, clear_z=None, clear_x=None, box=None, outer=None):
    tTb = np.linalg.inv(base_T_torso(torso))
    c = None
    if clear_z is not None:
        dx, dy, dz = tTb[0, 3], tTb[1, 3], tTb[2, 3]
        if box is not None:
            c = {'z': clear_z + dz, 'box': (box[0] + dx, box[1] + dx, box[2] + dy, box[3] + dy), 'outer': (outer[0] + dx, outer[1] + dx, outer[2] + dy, outer[3] + dy), 'idx': (1, 2, 3, 4, 5, 6, 7), 'n': 5}
        else:
            c = (clear_z + dz, (2, 3, 4, 5, 6), clear_x + dx)
    return k.solve(tTb @ goal_base, q, c)
def pts(q, torso):
    P = k.points(q); T = base_T_torso(torso)
    return (T[:3, :3] @ P.T).T + T[:3, 3]
q0 = np.zeros(7)
w = k.startup_waypoint(np.clip(q0, k.lower, k.upper)); q = k.compact(w)
rows = [int(a) for a in sys.argv[2:]] or [3, 4]
rim, floor = 0.95, 0.75
for row in rows:
    t = TORSO[row]; z = ROW_Z[row - 1] + GRASP_DZ[row]; reach = REACH[row]
    qq = q.copy()
    qq = solve(transform([reach - .09, ARM_Y, z]), qq, t)
    g = None
    for adv in np.linspace(0, 0.16, 13)[1:]:
        qq = solve(transform([reach - .09 + adv, ARM_Y, z]), qq, t)
    qq = solve(transform([reach + .07, ARM_Y, z + .025]), qq, t)
    qq = solve(transform([reach + .07 - .20, ARM_Y, z + .025]), qq, t)
    qgrasp = qq.copy()
    # base backs 0.28: the tool moves -0.28 in x in base terms is not true (arm fixed to base); torso up
    t = 0.345
    tip = (base_T_torso(t) @ k.fk(qq))[:3, 3]
    if tip[2] < 1.20:
        for gg in transport_path(tip):
            qq = solve(gg, qq, t)
    qcarry = qq.copy()
    stand, deep = 0.70, 0.12
    bx, by = stand, ARM_Y * 0 - 0.0  # bin centre in base: (stand, -ARM_Y*?)
    bx, by = stand, 0.24 * 0  # placeholder
    bin_c = np.array([stand, ARM_Y])
    p = np.array([bin_c[0] + deep, bin_c[1], rim + 0.18])
    clear = rim + 0.08
    r_link = 0.07
    box = (bin_c[0] - 0.28 + r_link, bin_c[0] + 0.28 - r_link, bin_c[1] - 0.155 + r_link, bin_c[1] + 0.155 - r_link)
    outer = (bin_c[0] - 0.28 - r_link, bin_c[0] + 0.28 + r_link, bin_c[1] - 0.155 - r_link, bin_c[1] + 0.155 + r_link)
    kw = dict(clear_z=clear, clear_x=bin_c[0] - 0.25 + 0.05) if mode == 'v3' else dict(clear_z=clear, box=box, outer=outer)
    qq = solve(transform(p), qq, t, **kw)
    zrel = floor + 0.005 + 0.115
    for zz in np.linspace(p[2], zrel, max(2, int(np.ceil((p[2] - zrel) / 0.04)) + 1))[1:]:
        p2 = p.copy(); p2[2] = zz
        try:
            qq = solve(transform(p2), qq, t, **kw)
        except RuntimeError as e:
            print('row', row, 'LOWER FAIL at', round(zz, 3), str(e)[-40:]); break
    tip = (base_T_torso(t) @ k.fk(qq))[:3, 3]
    print(f'row {row} released at tool z {tip[2]:.3f}')
    try:
        back = tip.copy(); back[0] -= 0.06
        qb = solve(transform(back), qq, t, **kw); print('   back ok')
    except RuntimeError as e:
        qb = None; print('   back FAIL', str(e)[-60:])
    for name, qs in (('from back', qb), ('from release', qq)):
        if qs is None: continue
        tp = (base_T_torso(t) @ k.fk(qs))[:3, 3]; up = tp.copy(); up[2] = p[2]
        try:
            qu = solve(transform(up), qs, t); print(f'   up {name} ok, joint step {np.abs(qu - qs).max():.2f}')
        except RuntimeError as e:
            qu = None; print(f'   up {name} FAIL', str(e)[-60:])
        for nm, qp in (('release/back pose', qs), ('up pose', qu)):
            if qp is None: continue
            try:
                w = k.startup_waypoint(np.clip(qp, k.lower, k.upper)); print(f'   waypoint from {name} {nm} ok, step {np.abs(w - qp).max():.2f}')
            except RuntimeError as e:
                print(f'   waypoint from {name} {nm} FAIL', e)

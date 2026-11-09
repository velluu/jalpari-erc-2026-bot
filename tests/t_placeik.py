"""Place IK feasibility with the thick-link clearance (no motion): bin centre at (stand, 0) in base_footprint, official
bin size, torso at the top. Prints, per tool height, whether a solution exists and the lowest arm sample outside the box.
  python3 tests/t_placeik.py [stand 0.70] [deep 0.12] [link_r 0.06] [clear 0.08]"""
import sys, numpy as np, rclpy
from erc_bot.live import Live, clamp
from erc_bot.arm import Arm, transform
stand, deep, rl, cl = [float(a) for a in (sys.argv[1:] + ['0.70', '0.12', '0.06', '0.08'][len(sys.argv) - 1:])][:4]
rclpy.init(); r = Live('t_placeik', sim_time=True); r.ready(); arm = Arm(r)
T = r.matrix('torso_lift_link', 'base_footprint'); T[2, 3] += r.joints['torso_lift_joint'] - clamp('torso_lift_joint', .345)
rim, floor = 0.95, 0.75
w = rl + 0.01
box = (stand - 0.28 + w, stand + 0.28 - w, -0.155 + w, 0.155 - w)
outer = (stand - 0.28 - w, stand + 0.28 + w, -0.155 - w, 0.155 + w)
clear = {'z': rim + cl + T[2, 3], 'box': (box[0] + T[0, 3], box[1] + T[0, 3], box[2] + T[1, 3], box[3] + T[1, 3]),
         'outer': (outer[0] + T[0, 3], outer[1] + T[0, 3], outer[2] + T[1, 3], outer[3] + T[1, 3]), 'idx': (1, 2, 3, 4, 5, 6, 7), 'n': 5}
seed = arm.q(arm.k.names)
for z in (rim + 0.18, 1.05, 1.0, 0.95, 0.92, 0.89, 0.87, 0.85, 0.82):
    g = transform([stand + deep, 0.0, z])
    try:
        q = arm.k.solve(T @ g, seed, clear); seed = q
        P = arm.k.points(q) - [0, 0, T[2, 3]]
        print(f'tool z {z:.2f}: ok  ' + ' '.join(f'{i}:({P[i,0]:.2f},{P[i,1]:.2f},{P[i,2]:.2f})' for i in range(1, 10)), flush=True)
    except RuntimeError as e:
        print(f'tool z {z:.2f}: FAIL {str(e)[-50:]}', flush=True)

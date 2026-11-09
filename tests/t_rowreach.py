"""IK feasibility of the scripted grasp path per row: for each base-to-spine distance, solve pre, deep, deep+lift and
back (the ScriptedPick plan) with the torso at the row's height. Prints the worst position error.
  python3 tests/t_rowreach.py <row> [insert_past]"""
import sys, numpy as np, rclpy
from erc_bot.live import Live, clamp
from erc_bot.arm import Arm, transform
from erc_bot.stages import ROW_Z, TORSO, ARM_Y
row = int(sys.argv[1]); past = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10; dzg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
rclpy.init(); r = Live('t_rowreach', sim_time=True); r.ready(); arm = Arm(r)
T = r.matrix('torso_lift_link', 'base_footprint'); T[2, 3] += r.joints['torso_lift_joint'] - clamp('torso_lift_joint', TORSO[row])   # torso at the row's height as sent
z = ROW_Z[row - 1] + dzg
for reach in np.arange(0.40, 0.90, 0.05):
    pre = transform([reach - .09, ARM_Y, z]); deep = .09 + past; seed = arm.q(arm.k.names); worst = 0; ok = True
    for dx, dz in ((0., 0.), (deep, 0.), (deep, .025), (-.05, .025)):
        g = pre.copy(); g[0, 3] += dx; g[2, 3] += dz
        try:
            seed = arm.k.solve(T @ g, seed)
        except RuntimeError as e:
            ok = False; worst = str(e)[-40:]; break
    print(f'row {row} reach {reach:.2f}: {"ok" if ok else "FAIL " + worst}', flush=True)

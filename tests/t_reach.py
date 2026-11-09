"""Arm shape for placing poses: joint origins in base_footprint for the tool at (x, -0.24, z), torso up."""
import sys, numpy as np, rclpy
from erc_bot.live import Live
from erc_bot.arm import Arm, transform
rclpy.init(); r = Live('t_reach', sim_time=True); r.ready(); arm = Arm(r)
T = r.matrix('torso_lift_link', 'base_footprint'); torso = r.joints['torso_lift_joint']
print('torso now', round(torso, 3))
for x in (0.6, 0.7, 0.8):
    for z in (1.10, 1.0, 0.95, 0.90, 0.87):
        try:
            q = arm.k.solve(T @ transform([x, -0.24, z]), arm.q(arm.k.names))
        except Exception as e:
            print(x, z, 'unreachable'); continue
        P = arm.k.points(q); Pb = (np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        print(x, z, ' '.join(f'({p[0]:.2f},{p[2]:.2f})' for p in Pb[-8:]))

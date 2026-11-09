"""Lowering test at the bin (dev sim, robot already near it): look, stand, lower the empty gripper step by step with the
elbow-clearance IK and print the chain heights. Run the scorer beside it to see collisions.
  python3 tests/t_lower.py <tool z at the bottom, base frame> [deep] [clear above rim]"""
import sys, math, numpy as np, rclpy
from erc_bot.live import Live
from erc_bot.mission import Bot
from erc_bot.arm import transform
from erc_bot.stages import ARM_Y
zbot = float(sys.argv[1]); deep = float(sys.argv[2]) if len(sys.argv) > 2 else 0.12; cl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
rclpy.init(); r = Live('t_lower', sim_time=True); r.ready()
b = Bot(r, '/opt/model/det_v4_736x1280.onnx', 1, 'red')
r.torso(.345, wait=False)
x, y, yaw = r.base_in_odom(); b.go(x - 0.6 * math.cos(yaw), y - 0.6 * math.sin(yaw), yaw, clearance=0.1)
best = None
for pan in (0.0, 0.6, -0.6, 1.2, -1.2):
    r.head(pan, -0.6); r.wait(0.3)
    pts, fl, rims = b.look_bin(frames=4)
    if len(pts) >= 3:
        best = b.bin_summary(pts, fl, rims); break
print('bin', np.round(best, 3), b.bin_info)
x, y, _ = r.base_in_odom(); to = best[:2] - [x, y]; u = to / np.linalg.norm(to); left = np.array([-u[1], u[0]])
st = best[:2] - 0.70 * u - ARM_Y * left
b.go(st[0], st[1], math.atan2(u[1], u[0]), tol=0.03, ytol=0.03, vmax=0.15, clearance=0.08)
r.torso(.345)
rim = b.bin_info['rim_z']
p = b.in_base(best); p[0] += deep; p[2] = rim + 0.18
b.arm.pose(transform(p), clear_z=rim + cl)
T = r.matrix('torso_lift_link', 'base_footprint')
for z in np.linspace(p[2], zbot, 7)[1:]:
    q = p.copy(); q[2] = z
    try:
        b.arm.pose(transform(q), duration=1.0, pos_tol=0.03, clear_z=rim + cl)
        P = b.arm.k.points(b.arm.q(b.arm.k.names)); Pb = P - [0, 0, T[2, 3]]
        print(f't {r.now() - b.t0:.1f} tool z {z:.3f}: ' + ' '.join(f'{i}:({Pb[i,0]:.2f},{Pb[i,2]:.2f})' for i in range(2, 10)), flush=True)
    except Exception as e:
        print(f't {r.now() - b.t0:.1f} tool z {z:.3f}: FAILED {e}', flush=True); break
r.wait(1.0)
b.arm.pose(transform(p), duration=1.5, pos_tol=0.04)

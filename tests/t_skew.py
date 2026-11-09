"""Clock-skew test (dev sim): this machine's clock pretends to be off the robot's by SKEW seconds (ERC_BOT_CLOCK_SKEW).
Camera frames must still pair and count as fresh, the laser guard must still see the scan, and a short base move
must happen. Also prints what the old stamp-based checks would have said.
  ERC_BOT_CLOCK_SKEW=5 python3 tests/t_skew.py"""
import math, os, sys, numpy as np, rclpy
from erc_bot.live import Live, stamp_s
skew = float(os.environ.get('ERC_BOT_CLOCK_SKEW', '0'))
rclpy.init(); r = Live('t_skew', sim_time=True); r.ready()
cams = 0; old_cam = 0
for _ in range(20):
    r.wait(0.2)
    cams += r.camera() is not None
    c = r.colour_q[-1][0]; old_cam += (r.now() - stamp_s(c.header)) <= 0.5
scan_ok = r.scan_points() is not None; old_scan = (r.now() - stamp_s(r.scan.header)) <= 1.0
x0, y0, _ = r.base_in_odom(); moved_cmds = 0
t0 = r.now()
while r.now() - t0 < 2.0:
    moved_cmds += r.velocity(0.0, 0.0, 0.3); r.tick()
r.stop(); r.wait(1.0)
yaw_moved = abs(math.atan2(math.sin(r.base_in_odom()[2]), math.cos(r.base_in_odom()[2])))
print(f'SKEW {skew:+.1f} s: offset est {r.clock_offset:+.3f} | camera fresh {cams}/20 (old rule {old_cam}/20) | '
      f'scan {scan_ok} (old rule {old_scan}) | twist accepted {moved_cmds}', flush=True)
x, y, yaw = r.base_in_odom(); t0 = r.now()
while r.now() - t0 < 1.5:
    r.velocity(0.15, 0.0, 0.0); r.tick()
r.stop(); r.wait(0.5)
x1, y1, _ = r.base_in_odom()
print(f'SKEW {skew:+.1f} s: forward move {math.hypot(x1 - x, y1 - y):.3f} m', flush=True)

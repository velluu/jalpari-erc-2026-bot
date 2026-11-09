#!/usr/bin/env python3
"""Relative base move on odometry with the bot's laser stop, for repositioning between sweeps.

  python3 slot/move.py --dx 0.5 [--dy 0] [--dyaw 0] [--speed 0.15]      metres / rad in the robot's current frame

Stops (and reports FAIL) if anything comes within 0.30 m of the footprint in the travel direction for 3 s.
"""
import argparse, math, time
from common import Verdicts, SIM
import rclpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dx', type=float, default=0.0); ap.add_argument('--dy', type=float, default=0.0)
    ap.add_argument('--dyaw', type=float, default=0.0); ap.add_argument('--speed', type=float, default=0.15)
    a = ap.parse_args()
    v = Verdicts('move')
    if a.speed > 0.3:
        v.check('speed', False, f'{a.speed} m/s', 'use 0.3 m/s or less in the slot'); return v.done()
    rclpy.init()
    import erc_bot.live as live
    L = live.Live('slot_move', sim_time=SIM)
    w0 = time.monotonic()
    while time.monotonic() - w0 < 30 and (L.odom is None or L.scan is None or L.now() <= 0.0):
        L.tick(0.05)          # sim time reads 0 until /clock arrives: a t0 of 0 ends the move at once (dry run 21:30)
    if L.odom is None:
        v.check('odom', False, 'no odometry', 'rerun s00_preflight'); return v.done()
    x, y, th = L.base_in_odom()
    c, s = math.cos(th), math.sin(th)
    goal = (x + c * a.dx - s * a.dy, y + s * a.dx + c * a.dy, th + a.dyaw)
    t0 = L.now(); blocked = None; why = 'timeout'
    timeout = 10 + 3 * (math.hypot(a.dx, a.dy) / a.speed + abs(a.dyaw) / 0.25)
    while L.now() - t0 < timeout:
        L.tick()
        bx, by, bth = L.base_in_odom()
        ex, ey = goal[0] - bx, goal[1] - by; eth = math.atan2(math.sin(goal[2] - bth), math.cos(goal[2] - bth))
        cc, ss = math.cos(bth), math.sin(bth); lx, ly = cc * ex + ss * ey, -ss * ex + cc * ey
        d = math.hypot(lx, ly)
        if d < 0.02 and abs(eth) < 0.02:
            why = 'arrived'; break
        vv = min(a.speed, 0.8 * d)
        vx, vy = (lx / d * vv, ly / d * vv) if d > 0.01 else (0.0, 0.0)
        if not L.velocity(vx, vy, max(-0.25, min(0.25, eth)), clearance=0.30):
            blocked = blocked or L.now()
            if L.now() - blocked > 3.0:
                why = 'blocked by the laser stop'; break
        else:
            blocked = None
    L.stop(); L.wait(0.3)
    bx, by, bth = L.base_in_odom()
    v.check('move', why == 'arrived', f'{why}, now ({bx:.3f}, {by:.3f}, {bth:.3f}) in odom, goal ({goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f})',
            'something is in the way: clear it or pick another position')
    L.destroy_node(); rclpy.shutdown()
    return v.done()


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Slot step 6: odometry drift over a short base move, checked against the laser (scan matching, no external tracker).

  python3 slot/s06_odom.py [--dist 0.30] [--side 0.20] [--turn 0.5] [--speed 0.10]

Legs (each driven on odom feedback at --speed m/s or 0.25 rad/s, laser stop at 0.35 m around the footprint, and only
when the scan shows at least dist + 0.5 m free that way): forward and back, left and back, turn left and back.
Before and after every leg the robot stands still for 1 s and a scan is taken; 2D ICP between scans gives the motion
the laser saw. Drift = odometry motion minus laser motion, per leg and for the closed loop (odometry says the robot is
back where it started, the laser says how far off it really is). Put a tape mark at the base's front edge too: the
loop error should show on the floor. Writes odom.json to $SLOT_OUT.
"""
import argparse, math, time
import numpy as np
from common import start, finish, save, SIM
from scipy.spatial import cKDTree


def se2(x, y, th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def un(T):
    return np.array([T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0])])


def icp(A, B, init, iters=60, trim=0.25):
    """Pose of scan B's frame in scan A's frame: A ~ R B + t. Returns (T, median residual, inlier fraction)."""
    tree = cKDTree(A)
    T = init.copy()
    for k in range(iters):
        Bt = B @ T[:2, :2].T + T[:2, 2]
        d, idx = tree.query(Bt)
        lim = max(0.05, trim * (1 - k / iters))
        m = d < lim
        if m.sum() < 30:
            break
        P, Q = Bt[m], A[idx[m]]
        mp, mq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1; R = Vt.T @ U.T
        dT = np.eye(3); dT[:2, :2] = R; dT[:2, 2] = mq - R @ mp
        T = dT @ T
        if np.linalg.norm(dT[:2, 2]) < 1e-5 and abs(math.atan2(R[1, 0], R[0, 0])) < 1e-6:
            break
    Bt = B @ T[:2, :2].T + T[:2, 2]
    d, _ = tree.query(Bt)
    return T, float(np.median(d)), float((d < 0.05).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dist', type=float, default=0.30); ap.add_argument('--side', type=float, default=0.20)
    ap.add_argument('--turn', type=float, default=0.5); ap.add_argument('--speed', type=float, default=0.10)
    a = ap.parse_args()
    n, v = start('s06_odom')
    import erc_bot.live as live
    L = live.Live('slot_odom', sim_time=SIM)
    w0 = time.monotonic()
    while time.monotonic() - w0 < 60 and (L.odom is None or L.scan is None or L.now() <= 0.0):
        L.tick(0.05)
    if not v.check('odom and scan', L.odom is not None and L.scan is not None, 'odometry and /scan arriving', 'rerun s00_preflight'):
        L.destroy_node(); return finish(n, v)

    def still_scan(secs=1.0):
        L.stop(); L.wait(secs)
        pts = []
        for _ in range(3):
            p = L.scan_points()
            if p is not None:
                pts.append(p)
            L.wait(0.15)
        P = np.concatenate(pts) if pts else None
        if P is not None:
            P = P[np.hypot(P[:, 0], P[:, 1]) < 8.0]
        return P, L.base_in_odom().copy()

    def free(direction, need):
        p = L.scan_points()
        if p is None:
            return 0.0
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        along = p @ d; side = np.abs(p[:, 0] * d[1] - p[:, 1] * d[0])
        ahead = along[(along > 0) & (side < 0.35)]
        return float(ahead.min() - 0.33) if len(ahead) else 25.0

    def drive(goal, timeout=60.0):
        """odom go-to (x, y, yaw), base frame P-control, laser stop. Returns why it stopped."""
        t0 = L.now(); blocked = None
        while L.now() - t0 < timeout:
            L.tick()
            x, y, th = L.base_in_odom()
            ex, ey = goal[0] - x, goal[1] - y; eth = math.atan2(math.sin(goal[2] - th), math.cos(goal[2] - th))
            c, s = math.cos(th), math.sin(th); lx, ly = c * ex + s * ey, -s * ex + c * ey
            dd = math.hypot(lx, ly)
            if dd < 0.01 and abs(eth) < 0.01:
                L.stop(); return 'arrived'
            vv = min(a.speed, 0.8 * dd)
            vx, vy = (lx / dd * vv, ly / dd * vv) if dd > 0.005 else (0.0, 0.0)
            wz = max(-0.25, min(0.25, 1.0 * eth))
            if not L.velocity(vx, vy, wz, clearance=0.35):
                blocked = blocked or L.now()
                if L.now() - blocked > 3.0:
                    L.stop(); return 'blocked by the laser stop'
            else:
                blocked = None
        L.stop(); return 'timeout'

    legs = [('forward', (a.dist, 0, 0), (1, 0)), ('back', (-a.dist, 0, 0), (-1, 0)),
            ('left', (0, a.side, 0), (0, 1)), ('right back', (0, -a.side, 0), (0, -1)),
            ('turn left', (0, 0, a.turn), None), ('turn back', (0, 0, -a.turn), None)]
    S0, P0 = still_scan()
    if not v.check('scan points', S0 is not None and len(S0) > 100, f'{0 if S0 is None else len(S0)} laser points within 8 m',
                   'too few laser returns for scan matching: the drift check needs walls or furniture in range. Drive anyway and '
                   'use the tape mark'):
        L.destroy_node(); return finish(n, v)
    rows = []
    S, P = S0, P0
    for name, (dx, dy, dth), dirn in legs:
        if dirn is not None:
            room = free(dirn, abs(dx) + abs(dy))
            if room < abs(dx) + abs(dy) + 0.5:
                v.check(f'leg {name}', False, f'only {room:.2f} m free that way', 'skipped: give the robot 1 m of space around it, '
                        'or run with smaller --dist / --side')
                continue
        g = se2(*P) @ se2(dx, dy, dth)
        why = drive(un(g))
        S2, P2 = still_scan()
        To = np.linalg.inv(se2(*P)) @ se2(*P2)                       # odometry: leg end in leg start frame
        Tl, res, inl = icp(S, S2, To)                                 # laser
        e = un(np.linalg.inv(Tl) @ To)
        row = {'leg': name, 'stop': why, 'odom': un(To).round(4).tolist(), 'laser': un(Tl).round(4).tolist(),
               'err_xy_m': round(float(math.hypot(e[0], e[1])), 4), 'err_yaw_rad': round(float(e[2]), 4),
               'icp_median_m': round(res, 4), 'icp_inliers': round(inl, 3)}
        rows.append(row)
        trav = f'{abs(un(To)[2]):.2f} rad' if dirn is None else f'{math.hypot(*un(To)[:2]):.2f} m'
        v.check(f'leg {name}', why == 'arrived' and inl > 0.5,
                f"odom {row['odom']} laser {row['laser']}: drift {row['err_xy_m'] * 100:.1f} cm / {math.degrees(row['err_yaw_rad']):.2f} deg "
                f"over {trav}, ICP median {res * 100:.1f} cm inliers {inl:.2f}",
                'stopped early or the scan match is poor (few static walls): trust the tape mark for this leg', **row)
        S, P = S2, P2
    if rows:
        Tl, res, inl = icp(S0, S, np.linalg.inv(se2(*P0)) @ se2(*P))
        To = np.linalg.inv(se2(*P0)) @ se2(*P)
        e = un(np.linalg.inv(Tl) @ To)
        loop = {'odom_end_in_start': un(To).round(4).tolist(), 'laser_end_in_start': un(Tl).round(4).tolist(),
                'err_xy_m': round(float(math.hypot(e[0], e[1])), 4), 'err_yaw_rad': round(float(e[2]), 4), 'icp_median_m': round(res, 4),
                'icp_inliers': round(inl, 3), 'path_m': round(2 * a.dist + 2 * a.side, 3), 'path_rad': round(2 * a.turn, 3)}
        v.check('loop', inl > 0.5, f"after {loop['path_m']} m and {loop['path_rad']} rad: odom-vs-laser {loop['err_xy_m'] * 100:.1f} cm, "
                f"{math.degrees(loop['err_yaw_rad']):.2f} deg (odom start offset {np.round(P0, 3).tolist()})",
                'poor scan match: read the tape mark', **loop)
    else:
        loop = None
    save('odom.json', {'legs': rows, 'loop': loop, 'odom_start': P0, 'sim': SIM})
    L.stop(); L.destroy_node()
    return finish(n, v)


if __name__ == '__main__':
    raise SystemExit(main())

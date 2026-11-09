"""Offline checks for lane H's re-anchoring (no sim): Shelf epochs, the laser shelf normal and the laser shelf range.
  python3 tests/t_anchor.py"""
import math, types
import numpy as np
from erc_bot.shelf import Shelf
from erc_bot.mission import Bot
from erc_bot.stages import ScriptedPick

ok = True
def check(name, cond, info=''):
    global ok
    ok &= bool(cond); print('PASS' if cond else 'FAIL', name, info)

# 1. Shelf epochs: the old sightings are 7 cm off; after new_epoch the fresh plate wins and the others shift with it.
S = Shelf()
for k in range(10):
    for d, x in ((1, 0.0), (2, 1.035), (3, 2.07)):
        S.add(d, [x, 3.0, 1.93])
before = S.plates()
check('no epoch: unchanged', np.allclose(before[2], [1.035, 3.0, 1.93]))
S.new_epoch()
for k in range(4):
    S.add(2, [1.035, 3.07, 1.93])
P = S.plates()
check('fresh plate', np.allclose(P[2], [1.035, 3.07, 1.93]), P[2])
check('others shifted', np.allclose(P[1], [0.0, 3.07, 1.93]) and np.allclose(P[3], [2.07, 3.07, 1.93]), (P[1], P[3]))
col = S.column(2, [1.0, 0.0])
check('column normal', np.allclose(col[1], [0, 1], atol=1e-6), col[1])

# 2. laser_normal: robot at odom yaw 0.3 facing away from the shelf; the real panel is rotated 5 deg from the old normal.
def fake_bot(pts_base, yaw):
    b = types.SimpleNamespace()
    b.r = types.SimpleNamespace(scan_points=lambda: pts_base, base_in_odom=lambda: np.array([0.5, 0.2, yaw]))
    return b
yaw = 0.3
n_old = np.array([math.cos(yaw + math.pi), math.sin(yaw + math.pi)])        # shelf behind the robot
dev = math.radians(5.0)
nb_true = np.array([math.cos(math.pi + dev), math.sin(math.pi + dev)])     # true normal in the base frame
tb = np.array([-nb_true[1], nb_true[0]])
rng = np.random.default_rng(1)
panel = np.array([1.7 * nb_true + s * tb for s in np.linspace(-2.5, 2.5, 300)]) + rng.normal(0, 0.005, (300, 2))
divs = np.array([1.4 * nb_true + s * tb + a * nb_true for s in (-1.55, -0.52, 0.52, 1.55) for a in np.linspace(0, 0.3, 8)])
legs = np.array([[0.8, 0.66], [0.8, -0.66], [1.5, 0.66], [1.5, -0.66]])     # table legs ahead
pts = np.vstack([panel, divs, legs])
res = Bot.laser_normal(fake_bot(pts, yaw), n_old)
n_true_odom = np.array([math.cos(yaw + math.pi + dev), math.sin(yaw + math.pi + dev)])
check('laser normal found', res is not None)
if res is not None:
    ang = math.degrees(math.acos(min(1, float(res[0] @ n_true_odom))))
    check('laser normal within 0.3 deg', ang < 0.3, f'err {ang:.3f} deg, dev {math.degrees(res[1]):.2f}, inliers {res[2]}, span {res[3]:.2f}, dist {res[4]:.3f}')
check('no line: None', Bot.laser_normal(fake_bot(legs, yaw), n_old) is None)

# 3. shelf_range: the panel 0.9 m ahead, then 0.68 m ahead.
pk = ScriptedPick.__new__(ScriptedPick)
def ahead(d):
    return np.array([[d, s] for s in np.linspace(-1.5, 1.5, 200)])
pk.r = types.SimpleNamespace(scan_points=lambda: ahead(0.9), base_in_odom=lambda: np.array([0.0, 0.0, 0.0]))
d0 = pk.shelf_range(np.array([1.0, 0.0]))
pk.r.scan_points = lambda: ahead(0.68)
d1 = pk.shelf_range(np.array([1.0, 0.0]))
check('shelf range', d0 and d1 and abs(d0[0] - 0.9) < 0.006 and abs(d1[0] - 0.68) < 0.006, (d0, d1))
print('ALL PASS' if ok else 'SOME FAILED')

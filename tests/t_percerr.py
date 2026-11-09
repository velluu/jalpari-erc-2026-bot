"""Perception error against ground truth: world->odom from the plates (they sit on the column centres), then every
detected book and plate is compared with the layout. Run in the dev sim with its layout.json."""
import json, sys, rclpy, numpy as np
from erc_bot.live import Live
from erc_bot.perception import Eyes
lay = json.load(open('/opt/results/dev/layout.json'))
face_x = lay['shelf'][0] - 0.245
plate_w = {d: np.array([face_x, lay['column_y'][c], 1.93]) for c, d in enumerate(lay['markers'])}
import math
from erc_bot.mission import Bot
rclpy.init(); r = Live('t_percerr', sim_time=True); r.ready()
bot = Bot(r, '/opt/model/det_v3_736x1280.onnx', int(sys.argv[1]) if len(sys.argv) > 1 else 1, 'red'); e = bot.eyes
bot.start(); plate, n, t_ = bot.find_column()
stand = plate[:2] - float(sys.argv[2] if len(sys.argv) > 2 else 1.15) * n
bot.go(stand[0], stand[1], math.atan2(n[1], n[0]))
pairs, books = [], []
for pan, tilt in ((0.0, 0.12), (0.0, 0.25), (0.0, 0.0), (0.0, -0.3), (0.0, -0.6)):
    r.head(pan, tilt); r.wait(0.5)
    for _ in range(4):
        dets, _ = e.look()
        for d in dets or []:
            if d['class'].startswith('marker'):
                pairs.append((plate_w[int(d['class'][-1])], d['p']))
            elif d['class'].startswith('book'):
                books.append(d)
        r.wait(0.1)
W = np.array([p[0][:2] for p in pairs]); O = np.array([p[1][:2] for p in pairs])
# 2D rigid fit odom = R world + t
mw, mo = W.mean(0), O.mean(0); H = (W - mw).T @ (O - mo); U, S, Vt = np.linalg.svd(H); R = Vt.T @ U.T
if np.linalg.det(R) < 0: Vt[1] *= -1; R = Vt.T @ U.T
t = mo - R @ mw
res = np.linalg.norm((W @ R.T + t) - O, axis=1)
print('plates used', len(pairs), 'fit residual mean', round(float(res.mean()), 4), 'yaw', round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 2))
Ri = R.T
errs = []
for d in books:
    pw = Ri @ (d['p'][:2] - t)
    col = [b for b in lay['books'] if (('white' in b['name'] and d['class'].split('_')[1] in ('FUTURE', 'ROBOT', 'LIBRARY', 'AGENT')) or b['name'].endswith(d['class'].split('_', 1)[1]))]
    if not col: continue
    b = min(col, key=lambda b: np.hypot(b['y'] - pw[1], b['z'] - d['p'][2]))
    errs.append((pw[0] - b['x'], pw[1] - b['y'], d['p'][2] - b['z'], d['class'], b['name'], d['z']))
E = np.array([e[:3] for e in errs])
print('books', len(errs), 'x (depth into shelf vs book centre) mean', np.round(E[:, 0].mean(), 3), 'y mean', np.round(E[:, 1].mean(), 4), 'y std', np.round(E[:, 1].std(), 4), 'z mean', np.round(E[:, 2].mean(), 3))
for e_ in sorted(errs, key=lambda e: abs(e[1]))[-8:]:
    print('  ', e_[3], e_[4], 'dx', round(e_[0], 3), 'dy', round(e_[1], 3), 'dz', round(e_[2], 3), 'range', round(e_[5], 2))

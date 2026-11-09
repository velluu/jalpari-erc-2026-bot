"""Book and plate 3D error against ground truth (robot world pose from gz, so no estimate is involved)."""
import json, subprocess, re, sys, math, rclpy, numpy as np
from scipy.spatial.transform import Rotation
from erc_bot.live import Live
from erc_bot.perception import Eyes
lay = json.load(open('/opt/results/dev/layout.json'))
face_x = lay['shelf'][0] - 0.245
plates = {d: np.array([face_x, lay['column_y'][c], 1.93]) for c, d in enumerate(lay['markers'])}
rclpy.init(); r = Live('t_proj', sim_time=True); r.ready(); e = Eyes(r, '/opt/model/det_v3_736x1280.onnx')
def world_T_base():
    out = subprocess.run(['gz', 'model', '-m', 'tiago_pro', '-p'], capture_output=True, text=True).stdout
    nums = [list(map(float, m.split())) for m in re.findall(r'\[([-\d\. e]+)\]', out)]
    W = np.eye(4); W[:3, :3] = Rotation.from_euler('xyz', nums[-1]).as_matrix(); W[:3, 3] = nums[-2]; return W
rows = []
for tilt in [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0.3,0.0,-0.3,-0.6').split(',')]:
    r.head(0.0, tilt); r.wait(0.8)
    W_B = world_T_base(); O_B = r.odom_T_base(); W_O = W_B @ np.linalg.inv(O_B)
    for _ in range(3):
        dets, _ = e.look()
        for d in dets or []:
            pw = (W_O @ np.r_[d['p'], 1])[:3]
            if d['class'].startswith('marker'):
                t = plates[int(d['class'][-1])]; rows.append(('plate', d['class'], pw - t, d['z']))
            else:
                key = d['class'].split('_', 1)[1]
                cand = [b for b in lay['books'] if b['name'].endswith('_' + key) or b['name'].endswith('_white_' + {'FUTURE': 'A', 'ROBOT': 'B', 'LIBRARY': 'C', 'AGENT': 'D'}.get(key, '?'))]
                if not cand: continue
                b = min(cand, key=lambda b: math.hypot(b['y'] - pw[1], b['z'] - pw[2]))
                rows.append(('book', d['class'], pw - np.array([b['x'], b['y'], b['z']]), d['z']))
        r.wait(0.1)
for kind in ('plate', 'book'):
    E = np.array([x[2] for x in rows if x[0] == kind])
    if len(E):
        print(kind, len(E), 'dx mean', round(E[:, 0].mean(), 3), 'dy mean', round(E[:, 1].mean(), 4), 'dy std', round(E[:, 1].std(), 4), 'dy max', round(np.abs(E[:, 1]).max(), 3), 'dz mean', round(E[:, 2].mean(), 3))

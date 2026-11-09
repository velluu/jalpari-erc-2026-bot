"""Bin detections against ground truth (gz poses): estimate in world vs the true bin centre, per head pose."""
import subprocess, re, sys, math, rclpy, numpy as np
from scipy.spatial.transform import Rotation
from erc_bot.live import Live
from erc_bot.perception import Eyes
rclpy.init(); r = Live('t_bin', sim_time=True); r.ready(); e = Eyes(r, '/opt/model/det_v3_736x1280.onnx')
def gz(name):
    out = subprocess.run(['gz', 'model', '-m', name, '-p'], capture_output=True, text=True).stdout
    nums = [list(map(float, m.split())) for m in re.findall(r'\[([-\d\. e]+)\]', out)]
    W = np.eye(4); W[:3, :3] = Rotation.from_euler('xyz', nums[-1]).as_matrix(); W[:3, 3] = nums[-2]; return W
W_O = gz('tiago_pro') @ np.linalg.inv(r.odom_T_base()); binw = gz('erc_collection_bin')[:3, 3]
rb = gz('tiago_pro')[:3, 3]
print('robot world', np.round(rb, 2), 'bin world', np.round(binw, 3), 'distance', round(float(np.linalg.norm(binw[:2] - rb[:2])), 2))
from erc_bot.mission import Bot
bot = Bot.__new__(Bot); bot.r = r
dist = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
bo = (np.linalg.inv(W_O) @ np.r_[binw, 1])[:3]; xo, yo, _ = r.base_in_odom()
u = (np.array([xo, yo]) - bo[:2]); u /= np.linalg.norm(u)
st = bo[:2] + dist * u
Bot.go(bot, st[0], st[1], math.atan2(-u[1], -u[0]))
rb = gz('tiago_pro')[:3, 3]; print('moved to', np.round(rb, 2), 'distance', round(float(np.linalg.norm(binw[:2] - rb[:2])), 2))
for pan in (0.0, 0.5, -0.5, 1.0, -1.0):
    for tilt in (-0.35, -0.6):
        r.head(pan, tilt); r.wait(0.4)
        dets, _ = e.look(['bin'], conf=0.25)
        for d in dets or []:
            pw = (W_O @ np.r_[d['p'], 1])[:3]
            print(f'pan {pan} tilt {tilt}: bin {d["score"]} box {[round(v) for v in d["box"]]} depth {d["z"]:.2f} est world {np.round(pw, 2)} err {np.round(pw - binw, 2)}')

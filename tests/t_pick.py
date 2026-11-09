"""Instrumented pick in the dev sim: true book vs estimate vs gripper (world frame, gz ground truth) at each step."""
import json, subprocess, re, sys, math, rclpy, numpy as np
from scipy.spatial.transform import Rotation
from erc_bot.live import Live, Abort
from erc_bot.mission import Bot
digit, label = int(sys.argv[1]), sys.argv[2]
rclpy.init(); r = Live('t_pick', sim_time=True); r.ready()
bot = Bot(r, '/opt/model/det_v3_736x1280.onnx', digit, label)
def gzpose(name):
    out = subprocess.run(['gz', 'model', '-m', name, '-p'], capture_output=True, text=True).stdout
    nums = [list(map(float, m.split())) for m in re.findall(r'\[([-\d\. e]+)\]', out)]
    W = np.eye(4); W[:3, :3] = Rotation.from_euler('xyz', nums[-1]).as_matrix(); W[:3, 3] = nums[-2]; return W
lay = json.load(open('/opt/results/dev/layout.json'))
col = lay['markers'].index(digit) + 1
target = [b['name'] for b in lay['books'] if b['column'] == col and b['name'].endswith('_' + label)][0]
def report(tag, est=None):
    W_O = gzpose('tiago_pro') @ np.linalg.inv(r.odom_T_base())
    bk = gzpose(target)[:3, 3]
    tool = (W_O @ r.odom_T_base() @ bot.arm.tool())[:3, 3]
    msg = f'{tag}: book true {np.round(bk, 3)} tool {np.round(tool, 3)} tool-book {np.round(tool - bk, 3)}'
    if est is not None:
        e = (W_O @ np.r_[est, 1])[:3]; msg += f' estimate {np.round(e, 3)} est-book {np.round(e - bk, 3)}'
    print(msg, flush=True)
orig_pose = bot.arm.pose
def pose(g, duration=None, **kw):
    a = orig_pose(g, duration, **kw); return a
bot.arm.pose = pose
bot.start(); plate, n, t = bot.find_column()
book, row, plate, n, t = bot.find_book(plate, n, t)
report('found', book)
orig_gripper = r.gripper
def gripper(closed, side='right'):
    report('before ' + ('close' if closed else 'open'), book)
    p = orig_gripper(closed, side); report('after ' + ('close' if closed else 'open') + f' pos {p:.3f}', book); return p
r.gripper = gripper
try:
    bot.pick(book, row, n)
    print('PICK OK, holding', bot.holding())
except Abort as e:
    print('PICK FAILED', e)

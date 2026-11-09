"""Full mission in the dev sim with a gz ground-truth trace at every stage (world frame).

  python3 tests/t_mission.py <digit> <label> [model] [layout.json]

Per stage: the robot's true pose, its odometry, and the odometry drift since START (world <- odom changing means the
odometry slipped). At BIN: the bin estimate mapped to world through the current world <- odom, against the true bin.
At RELEASE: the tool and the target book in world, against the bin. JSON trace to $OUT or /opt/results/dev/trace.json.
"""
import json, math, os, sys
import numpy as np
import rclpy
sys.path.insert(0, os.path.dirname(__file__))
from gztruth import GzTruth, yaw_of
from erc_bot.live import Live, Abort
from erc_bot.mission import Bot

digit, label = int(sys.argv[1]), sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else '/opt/model/det_v4_736x1280.onnx'
layout = json.load(open(sys.argv[4] if len(sys.argv) > 4 else '/opt/results/dev/layout.json'))
col = layout['markers'].index(digit) + 1
lab = label.lower()
target = [b['name'] for b in layout['books'] if b['column'] == col and (b['name'].endswith('_' + lab) or b['name'].endswith('_white_' + label.upper()))][0]
BIN = 'erc_collection_bin'

rclpy.init(); r = Live('t_mission', sim_time=True); r.ready()
g = GzTruth(); assert g.wait(('tiago_pro', BIN, target)), 'no gz poses'
bot = Bot(r, model, digit, label)
W0 = g.world_T_odom(r)
trace = {'target': target, 'bin_true': g.xyz(BIN).round(4).tolist(), 'stages': []}


def snap(name, **kw):
    W = g.world_T_odom(r)
    rb = g.pose('tiago_pro'); x, y, yaw = r.base_in_odom()
    d = np.linalg.inv(W0) @ W                    # drift of the odometry frame since START
    row = {'stage': name, 't': round(r.now() - bot.t0, 2), 'robot_world': [round(rb[0, 3], 3), round(rb[1, 3], 3), round(yaw_of(rb), 3)],
           'odom': [round(x, 3), round(y, 3), round(yaw, 3)], 'odom_drift_xy': round(float(np.hypot(d[0, 3], d[1, 3])), 3),
           'odom_drift_yaw': round(yaw_of(d), 4), 'book_world': g.xyz(target).round(3).tolist(), **kw}
    trace['stages'].append(row)
    print('TRUTH', json.dumps(row), flush=True)
    return W


orig_stage = bot.stage
def stage(name, **kw):
    orig_stage(name, **kw)
    extra = {}
    if name in ('BIN', 'BIN_RELOOK'):
        W = g.world_T_odom(r)
        est = (W @ np.r_[kw['p'], 1])[:3]
        extra = {'bin_est_world': est.round(3).tolist(), 'bin_err_xy': round(float(np.hypot(*(est[:2] - g.xyz(BIN)[:2]))), 3)}
    if name in ('RELEASE', 'LOWER_STOPPED'):
        tool = (g.pose('tiago_pro') @ bot.arm.tool())[:3, 3]
        b = g.xyz(BIN)
        T = r.matrix('torso_lift_link', 'base_footprint'); P = bot.arm.k.points(bot.arm.q(bot.arm.k.names))
        Pw = (g.pose('tiago_pro') @ np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        extra = {'tool_world': tool.round(3).tolist(), 'tool_minus_bin': (tool - b).round(3).tolist(),
                 'chain_minus_bin': (Pw - b).round(3).tolist()}
    snap(name, **extra)
bot.stage = stage

rc = 0
try:
    bot.run()
except Abort as e:
    print('FAILED', e, flush=True); trace['failed'] = str(e); rc = 1
r.wait(3.0)
snap('END', bin_true_now=g.xyz(BIN).round(3).tolist())
json.dump(trace, open(os.environ.get('OUT', '/opt/results/dev/trace.json'), 'w'), indent=1)
r.stop()
sys.exit(rc)

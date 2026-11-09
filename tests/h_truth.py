"""Lane H dev-sim truth logger (never used by the bot): every 0.25 s sim, the robot's gz world pose, the bot's odometry
as the bot sees it (/pert/... when the perturbation proxy runs, else the plain topic), the target book and the bin.
JSONL to argv[2]. Join with mission.json stage times (mission t = sim time - the bot's t0, logged as 'sim').
  python3 tests/h_truth.py <target book name> <out.jsonl> [odom topic]"""
import json, math, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
sys.path.insert(0, '/opt/bot/tests')
from gztruth import GzTruth, yaw_of

target, out = sys.argv[1], sys.argv[2]
topic = sys.argv[3] if len(sys.argv) > 3 else '/pert/mobile_base_controller/odom'
rclpy.init()
n = Node('h_truth')
st = {'clock': None, 'odom': None, 'raw': None}
n.create_subscription(Clock, '/clock', lambda m: st.__setitem__('clock', m.clock.sec + m.clock.nanosec * 1e-9), qos_profile_sensor_data)
def od(key):
    def f(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        st[key] = [p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))]
    return f
n.create_subscription(Odometry, topic, od('odom'), 20)
n.create_subscription(Odometry, '/mobile_base_controller/odom', od('raw'), 20)
g = GzTruth(); g.wait(('tiago_pro', 'erc_collection_bin', target), timeout=30)
f = open(out, 'w'); last = -1.0
while rclpy.ok():
    rclpy.spin_once(n, timeout_sec=0.05)
    t = st['clock']
    if t is None or t - last < 0.25:
        continue
    last = t
    try:
        rb = g.pose('tiago_pro')
        row = {'sim': round(t, 3), 'robot': [round(rb[0, 3], 4), round(rb[1, 3], 4), round(yaw_of(rb), 4)],
               'odom': None if st['odom'] is None else [round(v, 4) for v in st['odom']],
               'raw': None if st['raw'] is None else [round(v, 4) for v in st['raw']],
               'book': g.xyz(target).round(4).tolist(), 'bin': g.xyz('erc_collection_bin').round(4).tolist()}
    except KeyError:
        continue
    f.write(json.dumps(row) + '\n'); f.flush()

"""Shared plumbing for the hardware-slot scripts (lane F): node setup, recorders, verdicts, session folder.

Every script prints one verdict line per check, `PASS name: ...` or `FAIL name: ... -> what to do`, and appends it
with its numbers to $SLOT_OUT/verdicts.jsonl. Results go to $SLOT_OUT/<script>.json.

Live robot: source tiago_env.sh (domain 2, CycloneDDS). Against the erc-gym sim (live profile): ERC_SLOT_SIM=1, the
node then runs on /clock, so every time measured in the dry run is sim time.
Python 3.10 (Humble laptop) and 3.14 (HPC pixi) compatible, numpy 1.24 and 2.x.
"""
import json, math, os, sys, time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # erc_bot/ sits next to slot/ (erc-bot checkout or the kit)

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

SIM = os.environ.get('ERC_SLOT_SIM', '') == '1'
D = math.radians
# Interface Guide section 1 and 3 (checking layer): margins, speed caps, the Guide's joint table
CAP = {'torso_lift_joint': 0.035, 'head_1_joint': 0.5, 'head_2_joint': 0.5}
for _s in ('left', 'right'):
    CAP.update({f'arm_{_s}_{j}_joint': (0.60 if j <= 2 else 0.75) for j in range(1, 8)})
GUIDE = {'torso_lift_joint': (0.0, 0.35), 'head_1_joint': (D(-75), D(75)), 'head_2_joint': (D(-60), D(20)),
         'gripper_left_finger_joint': (0.0, 0.803), 'gripper_right_finger_joint': (0.0, 0.803),
         'arm_left_1_joint': (D(-270), D(30)), 'arm_right_1_joint': (D(-30), D(270)),
         'arm_left_5_joint': (D(-90), D(210)), 'arm_right_5_joint': (D(-210), D(90))}
for _s in ('left', 'right'):
    GUIDE.update({f'arm_{_s}_2_joint': (D(-140), D(65)), f'arm_{_s}_4_joint': (D(-140), D(65)),
                  f'arm_{_s}_3_joint': (D(-150), D(150)), f'arm_{_s}_7_joint': (D(-150), D(150)),
                  f'arm_{_s}_6_joint': (D(-108), D(172))})
MARGIN = 0.05
CONTROLLERS = {'arm_right_controller': [f'arm_right_{j}_joint' for j in range(1, 8)],
               'arm_left_controller': [f'arm_left_{j}_joint' for j in range(1, 8)],
               'torso_controller': ['torso_lift_joint'], 'head_controller': ['head_1_joint', 'head_2_joint'],
               'gripper_right_controller': ['gripper_right_finger_joint'], 'gripper_left_controller': ['gripper_left_finger_joint']}


def out_dir():
    d = os.environ.get('SLOT_OUT') or os.path.join(os.getcwd(), 'slot_out')
    os.makedirs(d, exist_ok=True)
    return d


def save(name, data):
    p = os.path.join(out_dir(), name)
    with open(p, 'w') as f:
        json.dump(data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    return p


class Verdicts:
    """Collects PASS / FAIL lines for one script; exit code 0 only if every check passed."""
    def __init__(self, script):
        self.script, self.items = script, []

    def check(self, name, ok, detail='', on_fail='', **numbers):
        line = f"{'PASS' if ok else 'FAIL'} {self.script}/{name}: {detail}" + ('' if ok or not on_fail else f'  -> {on_fail}')
        print(line, flush=True)
        self.items.append({'script': self.script, 'check': name, 'ok': bool(ok), 'detail': detail,
                           'on_fail': on_fail, **numbers})
        with open(os.path.join(out_dir(), 'verdicts.jsonl'), 'a') as f:
            f.write(json.dumps({'t_wall': time.strftime('%H:%M:%S'), **self.items[-1]},
                               default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
        return ok

    def info(self, text):
        print(f'     {self.script}: {text}', flush=True)

    def done(self):
        bad = [i['check'] for i in self.items if not i['ok']]
        print(f"== {self.script}: {len(self.items) - len(bad)} of {len(self.items)} checks passed"
              + (f", failed: {', '.join(bad)}" if bad else ''), flush=True)
        return 0 if not bad else 1


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def dur(t):
    t = max(0.0, float(t))
    return Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)) % 1000000000)


class SlotNode(Node):
    """Joint states with receive times (node clock), rejections, /erc/ action clients and topic publishers."""
    def __init__(self, name):
        from rclpy.parameter import Parameter
        super().__init__(name, parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, SIM)])
        self.joints, self.effort, self.vel = {}, {}, {}
        self.js_log = None          # list of (t_recv, t_header, {name: pos}) while recording
        self.js_n, self.js_first = 0, None
        self.rejections = []        # (t_recv, text)
        self.create_subscription(JointState, '/joint_states', self._on_js, qos_profile_sensor_data)
        self.create_subscription(String, '/erc/rejections', lambda m: self.rejections.append((self.now(), m.data)), 50)
        self._act, self._pub = {}, {}
        if SIM:                     # sim time reads 0 until the first /clock message: wait for it, or every timeout fires at once
            w0 = time.monotonic()
            while self.get_clock().now().nanoseconds == 0 and time.monotonic() - w0 < 30:
                rclpy.spin_once(self, timeout_sec=0.05)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_js(self, m):
        t = self.now()
        self.joints.update(zip(m.name, m.position))
        if len(m.effort) == len(m.name):
            self.effort.update(zip(m.name, m.effort))
        if len(m.velocity) == len(m.name):
            self.vel.update(zip(m.name, m.velocity))
        self.js_n += 1
        if self.js_first is None:
            self.js_first = t
        if self.js_log is not None:
            self.js_log.append((t, stamp(m.header), dict(zip(m.name, m.position))))

    def spin_for(self, seconds, dt=0.01):
        t0, w0 = self.now(), time.monotonic()
        while self.now() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=dt)
            if time.monotonic() - w0 > max(60.0, seconds * 40):     # sim clock stalled (RTF 0.15 on cpuq still passes)
                raise RuntimeError('clock stalled')

    def spin_until(self, pred, timeout, dt=0.01):
        t0, w0 = self.now(), time.monotonic()
        while self.now() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=dt)
            if pred():
                return True
            if time.monotonic() - w0 > max(60.0, timeout * 40):
                return False
        return pred()

    def wait_joints(self, names, timeout=10.0):
        return self.spin_until(lambda: all(n in self.joints for n in names), timeout)

    def action(self, ctrl):
        if ctrl not in self._act:
            self._act[ctrl] = ActionClient(self, FollowJointTrajectory, f'/erc/{ctrl}/follow_joint_trajectory')
        return self._act[ctrl]

    def topic(self, ctrl):
        if ctrl not in self._pub:
            self._pub[ctrl] = self.create_publisher(JointTrajectory, f'/erc/{ctrl}/joint_trajectory', 10)
        return self._pub[ctrl]

    def traj(self, names, points):
        """points: [(t_from_start, [q...]), ...]"""
        jt = JointTrajectory(); jt.joint_names = list(names)
        for t, q in points:
            p = JointTrajectoryPoint(); p.positions = [float(v) for v in q]; p.time_from_start = dur(t)
            jt.points.append(p)
        return jt

    def send_goal(self, ctrl, names, points, wait_result=True, timeout=30.0):
        """Action goal. Returns dict: accepted, error_code, error_string, t_send, t_accept, t_result, rejections."""
        cli = self.action(ctrl)
        if not cli.wait_for_server(timeout_sec=5.0):
            return {'accepted': False, 'error_code': None, 'error_string': 'action server not available', 't_send': self.now()}
        g = FollowJointTrajectory.Goal(); g.trajectory = self.traj(names, points)
        n_rej = len(self.rejections)
        t_send = self.now()
        fut = cli.send_goal_async(g)
        self.spin_until(fut.done, 10.0)
        out = {'t_send': t_send, 'accepted': None, 'error_code': None, 'error_string': ''}
        if not fut.done():
            out['error_string'] = 'no answer to send_goal within 10 s'
            return out
        gh = fut.result(); out['t_accept'] = self.now(); out['accepted'] = bool(gh.accepted)
        if gh.accepted and wait_result:
            rf = gh.get_result_async()
            self.spin_until(rf.done, timeout)
            if rf.done():
                r = rf.result().result
                out.update(error_code=int(r.error_code), error_string=r.error_string, t_result=self.now(),
                           status=int(rf.result().status))
            else:
                out['error_string'] = f'no result within {timeout} s'
        self.spin_for(0.3)
        out['rejections'] = [x for _, x in self.rejections[n_rej:]]
        out['handle'] = gh
        return out

    def move(self, ctrl, target, frac=0.4, minimum=1.0, timeout=60.0):
        """Safe one-point move inside the caps (frac of each speed cap), waits for the result."""
        names = CONTROLLERS[ctrl]
        cur = [self.joints[n] for n in names]
        t = max([abs(q - c) / (CAP[n] * frac) for n, q, c in zip(names, target, cur) if n in CAP] + [minimum])
        r = self.send_goal(ctrl, names, [(t, target)], timeout=t + timeout)
        r.pop('handle', None)
        return r


def guide_ok(name, q, margin=MARGIN, extra=0.02):
    lo, hi = GUIDE.get(name, (-math.inf, math.inf))
    m = 0.0 if 'gripper' in name else margin + extra
    return lo + m <= q <= hi - m


def start(name):
    rclpy.init()
    n = SlotNode(name)
    v = Verdicts(name)
    n._t0, n._w0 = n.now(), time.monotonic()
    return n, v


def timing(script, robot_s, wall_s):
    """Per-script durations: robot clock (sim time in the dry run, so it predicts the real robot) and wall clock."""
    with open(os.path.join(out_dir(), 'timings.jsonl'), 'a') as f:
        f.write(json.dumps({'script': script, 'robot_s': round(robot_s, 1), 'wall_s': round(wall_s, 1)}) + '\n')


def finish(node, v):
    try:
        for _ in range(10):                       # the node's /clock may not have been spun while another node worked
            rclpy.spin_once(node, timeout_sec=0.05)
        timing(v.script, node.now() - node._t0, time.monotonic() - node._w0)
    except Exception:
        pass
    rc = v.done()
    try:
        node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return rc


def ask(prompt, auto):
    """Operator confirmation. auto (dry run or --yes) answers yes."""
    if auto:
        print(f'[auto] {prompt} yes', flush=True)
        return True
    try:
        return input(f'{prompt} [y/N] ').strip().lower() in ('y', 'yes')
    except EOFError:
        return False

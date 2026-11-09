#!/usr/bin/env python3
"""(slot copy of erc-gym tools/head_sweep.py: sim-aware result timeout)
Raster head sweep over the shelf, for a hardware-session recording (run while tools/record_session.sh records).

Pans -pan..pan at each tilt at --speed rad/s through /erc/head_controller, inside the live checking layer (head
0.5 rad/s, 0.05 rad margin on pan +-75 deg and tilt -60..20 deg). head_2 negative looks down. Face the shelf from 1 to
2 m first. Against the erc-gym sim (live-robot profile) it runs unchanged.

  python3 head_sweep.py [--pan 0.6] [--tilts 0.15,-0.15,-0.45] [--speed 0.2]
"""
import argparse, os, sys, time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

JOINTS = ['head_1_joint', 'head_2_joint']
LIMITS = [(-1.309 + 0.05, 1.309 - 0.05), (-1.047 + 0.05, 0.349 - 0.05)]   # live limits less the 0.05 rad margin


def waypoints(pan, tilts):
    pts = []
    for i, t in enumerate(tilts):
        a, b = (-pan, pan) if i % 2 == 0 else (pan, -pan)
        pts += [(a, t), (b, t)]
    return pts + [(0.0, tilts[len(tilts) // 2])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pan', type=float, default=0.6, help='rad either side of straight ahead')
    ap.add_argument('--tilts', default='0.15,-0.15,-0.45', help='rad, one pan pass per tilt')
    ap.add_argument('--speed', type=float, default=0.2, help='rad/s per joint, live cap 0.5')
    ap.add_argument('--action', default='/erc/head_controller/follow_joint_trajectory')
    a = ap.parse_args()
    tilts = [float(x) for x in a.tilts.split(',')]
    pts = waypoints(a.pan, tilts)
    for q in pts:
        for v, (lo, hi), n in zip(q, LIMITS, JOINTS):
            if not lo <= v <= hi:
                sys.exit(f'{n}={v} outside the live range less margin [{lo:.3f}, {hi:.3f}]')
    if a.speed > 0.45:
        sys.exit('--speed above 0.45 rad/s leaves no room under the 0.5 rad/s cap')

    rclpy.init()
    node = Node('erc_gym_head_sweep')
    cur = {}
    node.create_subscription(JointState, '/joint_states', lambda m: cur.update(zip(m.name, m.position)), qos_profile_sensor_data)
    node.create_subscription(String, '/erc/rejections', lambda m: print(f'rejection: {m.data}'), 10)
    t_end = time.time() + 10
    while time.time() < t_end and not all(j in cur for j in JOINTS):
        rclpy.spin_once(node, timeout_sec=0.2)
    if not all(j in cur for j in JOINTS):
        sys.exit('no head joints on /joint_states after 10 s (tiago_env.sh sourced?)')

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINTS
    prev, t = [cur[j] for j in JOINTS], 0.0
    for q in pts:
        dq = max(abs(x - y) for x, y in zip(q, prev))
        t += max(dq / a.speed, 0.5)
        p = JointTrajectoryPoint(); p.positions = list(q)
        p.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
        goal.trajectory.points.append(p); prev = q
    client = ActionClient(node, FollowJointTrajectory, a.action)
    if not client.wait_for_server(timeout_sec=10):
        sys.exit(f'no action server {a.action}')
    print(f'sweep: {len(pts)} points, {t:.1f} s, pan +-{a.pan} rad, tilts {tilts}, {a.speed} rad/s', flush=True)
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=10)
    gh = fut.result()
    if gh is None or not gh.accepted:
        rclpy.spin_once(node, timeout_sec=1.0)        # print the rejection reason if one arrived
        sys.exit('goal rejected')
    res = gh.get_result_async()
    rclpy.spin_until_future_complete(node, res, timeout_sec=(2 * t + 30) * (8 if os.environ.get('ERC_SLOT_SIM') == '1' else 1))   # wall clock; cpuq sims run at RTF 0.15 to 0.4
    r = res.result()
    code = r.result.error_code if r is not None else None
    print(f'sweep done: error_code {code}' + (f' ({r.result.error_string})' if r is not None and r.result.error_string else ''))
    node.destroy_node(); rclpy.shutdown()
    sys.exit(0 if code == 0 else 1)


if __name__ == '__main__':
    main()

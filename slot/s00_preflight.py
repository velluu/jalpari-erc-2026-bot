#!/usr/bin/env python3
"""Slot step 0 (read-only): DDS environment, every topic and action the bot needs, their rates, the robot's clock.

  python3 slot/s00_preflight.py [--secs 4]

Writes preflight.json (rates, full topic and node list) to $SLOT_OUT.
"""
import argparse, os, time
import numpy as np
from common import start, finish, save, SIM, stamp, CONTROLLERS
import rclpy
from rclpy.time import Time as RTime
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

C = '/head_front_camera'
HERE_K = os.path.dirname(os.path.abspath(__file__))     # slot/ inside the kit: dds/ sits next to it
OFFSET_MAX = 0.15          # s; live.py camera max_age 0.5 s, minus camera pipeline age and margin
RMEM_MIN = 4194304       # bytes, lane C's laptop recipe (kit dds/60-erc-dds.conf): 0 Hz for a 3.7 MB frame at the 212992 default
# topic, type, nominal Hz (Interface Guide), qos
RATES = [('/joint_states', JointState, 50), ('/mobile_base_controller/odom', Odometry, 50), ('/scan', LaserScan, 15),
         ('/tf', TFMessage, 15), (f'{C}/color/image_raw/compressed', CompressedImage, 30),
         (f'{C}/aligned_depth_to_color/image_raw/compressedDepth', CompressedImage, 30),
         (f'{C}/color/camera_info', CameraInfo, 30)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--secs', type=float, default=4.0)
    ap.add_argument('--no-buftest', dest='buftest', action='store_false', help='skip the 20 s CycloneDDS receive-buffer test')
    ap.add_argument('--robot-epoch', action='store_true', help='print the robot clock (from one joint_states stamp) and exit')
    a = ap.parse_args()
    if a.robot_epoch:
        from common import SlotNode
        rclpy.init(); nn = SlotNode('slot_epoch'); box = {}
        nn.create_subscription(JointState, '/joint_states', lambda m: box.setdefault('t', stamp(m.header)), qos_profile_sensor_data)
        nn.spin_until(lambda: 't' in box, 10.0)
        print(f"{box['t']:.3f}" if 't' in box else 'none')
        return 0 if 't' in box else 1
    dom, rmw = os.environ.get('ROS_DOMAIN_ID', ''), os.environ.get('RMW_IMPLEMENTATION', '')
    n, v = start('s00_preflight')
    if not SIM:
        v.check('domain', dom == '2', f'ROS_DOMAIN_ID={dom or "unset"}', 'source tiago_dds_connect/tiago_env.sh in this terminal')
        v.check('rmw', rmw == 'rmw_cyclonedds_cpp', f'RMW_IMPLEMENTATION={rmw or "unset"}', 'source tiago_env.sh (CycloneDDS profile)')
        try:
            rmem = int(open('/proc/sys/net/core/rmem_max').read())
        except Exception:
            rmem = 0
        v.check('rmem_max', rmem >= RMEM_MIN, f'net.core.rmem_max {rmem} bytes (need {RMEM_MIN})',
                'CRITICAL, the bot loses camera frames and stops: sudo cp dds/60-erc-dds.conf /etc/sysctl.d/ && sudo sysctl --system, '
                'then source tiago_env.sh and kit_env.sh again in every terminal', rmem_max=rmem)
        uri = os.environ.get('CYCLONEDDS_URI', '')
        v.check('cyclone profile', 'cyclone_tiago_rcvbuf.xml' in uri, f'CYCLONEDDS_URI {uri}',
                'source kit_env.sh after tiago_env.sh (it switches to the 4 MB receive-buffer profile once rmem_max allows it)')
        if a.buftest:
            import subprocess, sys as _s
            bt = next((x for x in (os.path.join(os.path.dirname(HERE_K), 'dds', 'cyclone_buftest.py'),
                                   os.path.join(os.path.dirname(HERE_K), 'kit', 'dds', 'cyclone_buftest.py')) if os.path.exists(x)), '')
            if bt and os.path.exists(bt):
                env = dict(os.environ, ROS_DOMAIN_ID='87', SECS='8')     # off the robot's domain 2, laptop-local
                try:
                    o = subprocess.run([_s.executable, bt], env=env, capture_output=True, text=True, timeout=60).stdout.strip()
                except Exception as e:
                    o = f'error {e}'
                import re
                m = re.search(r'subscriber ([0-9.]+) Hz', o)
                hz = float(m.group(1)) if m else 0.0
                v.check('cyclone buffer test', hz >= 5.0, f'3.7 MB frames at 15 Hz to a best-effort reader on this laptop: {o or "no output"}',
                        'below 5 Hz: the receive buffer is still too small, recheck rmem_max and the profile (lane C measured '
                        '7.5 Hz with this profile at rmem_max 4194304, 0 Hz at 212992)', hz=hz)
        v.check('localhost_only', os.environ.get('ROS_LOCALHOST_ONLY', '0') in ('', '0'), 'ROS_LOCALHOST_ONLY unset',
                'unset ROS_LOCALHOST_ONLY (Guide section 1)')
    tf = Buffer(); TransformListener(tf, n)
    cnt, sizes, offs = {}, {}, {}
    for t, typ, _ in RATES:
        def cb(m, t=t):
            cnt[t] = cnt.get(t, 0) + 1
            if isinstance(m, CompressedImage):
                sizes.setdefault(t, []).append(len(m.data))
            if hasattr(m, 'header'):
                offs.setdefault(t, []).append(stamp(m.header) - n.now())
            elif t == '/tf' and m.transforms:
                offs.setdefault(t, []).append(stamp(m.transforms[0].header) - n.now())
        n.create_subscription(typ, t, cb, qos_profile_sensor_data)
    urdf = {}
    n.create_subscription(String, '/robot_description', lambda m: urdf.setdefault('x', m.data),
                          QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    n.spin_for(1.0)
    cnt.clear(); sizes.clear(); offs.clear()
    t0 = n.now(); n.spin_for(a.secs); el = n.now() - t0
    rates = {}
    for t, _, nom in RATES:
        hz = cnt.get(t, 0) / el
        rates[t] = round(hz, 2)
        floor = 0.6 * nom if not SIM else 0.5
        kb = (sum(sizes[t]) / len(sizes[t]) / 1000) if sizes.get(t) else None
        v.check(f'rate {t}', hz >= floor, f'{hz:.1f} Hz (nominal {nom})' + (f', {kb:.0f} kB per msg' if kb else ''),
                'no or slow data: ros2 topic list, check the Ethernet link and tiago_env.sh; if only images are slow, check '
                'ros2 topic bw and record with DEPTH=0', hz=round(hz, 2), nominal=nom, kb=kb)
    # robot clock vs ours. erc_bot/live.py compares robot header stamps with the laptop clock: camera() drops frames
    # older than 0.5 s, scan_points() scans older than 1.0 s, so an offset of about 0.5 s stops the bot (lane B is adding
    # compensation; until it lands the laptop clock has to match the robot's)
    js = offs.get('/joint_states', [])
    if js:
        off = float(np.median(js))
        ages = {t: round(float(np.median(o)), 3) for t, o in offs.items() if o}
        v.check('clock offset', abs(off) < OFFSET_MAX,
                f'robot stamp minus laptop receive time, median of {len(js)} joint_states: {off * 1000:+.0f} ms (pass below '
                f'{OFFSET_MAX * 1000:.0f} ms, the bot drops camera frames older than 500 ms by robot stamp). Per topic {ages}',
                'the bot will see every camera frame and scan as stale. Set the laptop clock to the robot clock: '
                '`sudo timedatectl set-ntp false` then `sudo date -s @$(python3 slot/s00_preflight.py --robot-epoch)`, rerun s00. '
                'Do not run s08_bot until this passes (unless lane B\'s offset compensation is in the kit, see kit VERSION)',
                offset_s=round(off, 4), per_topic=ages)
    else:
        v.check('clock offset', False, 'no joint_states to compare', 'rerun s00 once /joint_states arrives')
    n.spin_until(lambda: 'x' in urdf, 5.0)
    v.check('robot_description', 'x' in urdf, f"{len(urdf.get('x', ''))} chars", 'no /robot_description: the bot cannot build its IK, ask the organisers')
    acts = {}
    for c in CONTROLLERS:
        cli = n.action(c)
        acts[c] = cli.wait_for_server(timeout_sec=3.0)
    v.check('action servers', all(acts.values()), f"{sum(acts.values())} of {len(acts)} /erc/ follow_joint_trajectory up",
            f"down: {[c for c, ok in acts.items() if not ok]} - ask the organisers, the checking layer is not running", servers=acts)
    subs = {c: n.count_subscribers(f'/erc/{c}/joint_trajectory') for c in CONTROLLERS}
    v.check('trajectory topics', all(s > 0 for s in subs.values()), f'subscribers {subs}',
            'a /erc/<c>/joint_trajectory topic has no subscriber: streaming (learned stages) cannot work, use the action', subs=subs)
    rp = n.count_publishers('/erc/rejections')
    v.check('rejections topic', rp > 0, f'{rp} publisher(s) on /erc/rejections', 'rejection reasons will not be visible; continue')
    ok_tf = n.spin_until(lambda: tf.can_transform('odom', 'base_footprint', RTime())
                         and tf.can_transform('base_footprint', 'head_front_camera_color_optical_frame', RTime()), 5.0)
    v.check('tf', ok_tf, 'odom->base_footprint and base_footprint->head_front_camera_color_optical_frame',
            'TF incomplete: ros2 run tf2_tools view_frames, send frames.pdf')
    names = sorted(n.get_topic_names_and_types())
    nodes = sorted(f'{ns.rstrip("/")}/{nm}' for nm, ns in n.get_node_names_and_namespaces())
    raw = [t for t, _ in names if t.endswith('/controller_state') or t.endswith('/state')]
    v.info(f'{len(names)} topics, {len(nodes)} nodes, controller state topics visible: {raw[:8]}')
    save('preflight.json', {'rates': rates, 'actions': acts, 'subscribers': subs, 'topics': [[t, ty] for t, ty in names],
                            'nodes': nodes, 'env': {'ROS_DOMAIN_ID': dom, 'RMW': rmw}, 'sim': SIM, 'urdf_chars': len(urdf.get('x', ''))})
    return finish(n, v)


if __name__ == '__main__':
    raise SystemExit(main())

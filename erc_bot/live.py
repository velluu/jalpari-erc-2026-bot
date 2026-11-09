"""Live TIAGo Pro interface (ERC 2026 Live Phase Interface Guide): sensors, TF, /erc/ controllers, base, safety.

Everything the bot sends goes through the organisers' checking layer (/erc/<controller>/follow_joint_trajectory), so
every trajectory is timed against the live speed caps and clamped inside the live joint ranges before it leaves.
Nothing here uses a simulation-only topic: no contacts, no IMU, no map, no raw controllers.
"""
import math, os, time
from collections import deque
import numpy as np
import cv2
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from trajectory_msgs.msg import JointTrajectoryPoint
from scipy.spatial.transform import Rotation
from .kinematics import with_grasping_links

D = math.radians
# Interface Guide section 3: live ranges. The checking layer rejects anything within 0.05 of them, we keep 0.07.
# Arm joints 1 and 5 are left out on purpose: the Guide's values are the sim URDF's with left and right swapped, so
# the arm takes those two from the robot's own /robot_description (whatever robot it runs on).
LIVE_LIMITS = {
    'torso_lift_joint': (0.0, 0.35), 'head_1_joint': (D(-75), D(75)), 'head_2_joint': (D(-60), D(20)),
    'gripper_right_finger_joint': (0.0, 0.803), 'gripper_left_finger_joint': (0.0, 0.803),
}
for _s in ('left', 'right'):
    LIVE_LIMITS.update({f'arm_{_s}_2_joint': (D(-140), D(65)), f'arm_{_s}_4_joint': (D(-140), D(65)),
                        f'arm_{_s}_3_joint': (D(-150), D(150)), f'arm_{_s}_7_joint': (D(-150), D(150)),
                        f'arm_{_s}_6_joint': (D(-108), D(172))})
MARGIN = 0.07
SPEED = {'torso_lift_joint': 0.035, 'head_1_joint': 0.5, 'head_2_joint': 0.5}
for _s in ('left', 'right'):
    SPEED.update({f'arm_{_s}_{j}_joint': (0.60 if j <= 2 else 0.75) for j in range(1, 8)})
SPEED_USE = 0.6               # fraction of each cap the bot plans with (arms, torso)
SPEED_USE_HEAD = 0.9          # the head carries nothing and only looks: 0.45 of the 0.5 rad/s cap
ARM_RIGHT = [f'arm_right_{j}_joint' for j in range(1, 8)]
ARM_LEFT = [f'arm_left_{j}_joint' for j in range(1, 8)]
CONTROLLERS = {'arm_right_controller': ARM_RIGHT, 'arm_left_controller': ARM_LEFT, 'torso_controller': ['torso_lift_joint'],
               'head_controller': ['head_1_joint', 'head_2_joint'], 'gripper_right_controller': ['gripper_right_finger_joint'],
               'gripper_left_controller': ['gripper_left_finger_joint']}
GRIP_OPEN, GRIP_CLOSED = 0.0, 0.79


TORSO_MARGIN = 0.055          # metres: the checking layer's 0.05 plus 5 mm (the 0.07 rad joint margin cost 2 cm of torso at each end)


def clamp(name, q):
    lo, hi = LIVE_LIMITS.get(name, (-math.inf, math.inf))
    m = 0.0 if 'gripper' in name else TORSO_MARGIN if name == 'torso_lift_joint' else MARGIN
    return float(min(max(q, lo + m), hi - m))


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class Abort(RuntimeError):
    pass


class Live(Node):
    def __init__(self, name='erc_bot', sim_time=False):
        from rclpy.parameter import Parameter
        super().__init__(name, parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, bool(sim_time))])
        self.tf = Buffer(); TransformListener(self.tf, self)
        self.joints, self.joint_t = {}, -1.0
        self.colour_msg = self.depth_msg = self.info = self.scan = self.odom = None
        self.urdf = None
        self.urdf_injected = False         # True on the real robot: its URDF and TF have no grasping links
        self.rejections = []
        cam = '/head_front_camera'
        self.colour_q = deque(maxlen=20)      # recent (colour frame, receive time), matched to each depth frame by stamp
        # Freshness is judged by RECEIVE time on this machine's clock, never by now() minus a header stamp: the stamps are
        # the robot's clock, and an unsynced robot clock (seconds off) would make every frame look stale and hold the base.
        # Pairing colour with depth stays on stamps (both from the robot's clock). clock_offset is logged at ready().
        self.depth_recv = self.scan_recv = -1.0
        self.offsets = deque(maxlen=200)
        self.skew = float(os.environ.get('ERC_BOT_CLOCK_SKEW', '0'))   # test only: pretend this machine's clock is off
        self.create_subscription(CompressedImage, f'{cam}/color/image_raw/compressed', self.on_colour, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, os.environ.get('ERC_BOT_ALIGNED_DEPTH_TOPIC', f'{cam}/aligned_depth_to_color/image_raw/compressedDepth'), self.on_depth,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        # Depth fallback (lane D, d-depthreg): the 30 Oct real session had aligned depth at 0.02 Hz and the unaligned
        # depth/image_rect_raw at 13.7 Hz. ERC_BOT_DEPTHREG: 'auto' (default) registers the unaligned stream to colour only
        # when the aligned pair is missing and aligned depth arrives below ERC_BOT_DEPTHREG_MIN_HZ; '1' always; '0' never.
        # With aligned depth healthy, camera() returns exactly the aligned pair it returned before.
        self.depthreg = os.environ.get('ERC_BOT_DEPTHREG', 'auto')
        self.depthreg_min_hz = float(os.environ.get('ERC_BOT_DEPTHREG_MIN_HZ', '2.0'))
        self.depth_recvs = deque(maxlen=256); self.t_start = None
        self.raw_depth_msg = self.raw_info = None; self.raw_depth_recv = -1.0; self.T_cd = None; self.reg_said = False
        if self.depthreg != '0':
            if os.environ.get('ERC_BOT_DEPTHREG_RAW') == '1':      # sim test only: the gym publishes the unaligned stream raw
                from sensor_msgs.msg import Image
                self.create_subscription(Image, os.environ.get('ERC_BOT_DEPTHREG_TOPIC', f'{cam}/depth/image_rect_raw'), self.on_raw_depth,
                                         QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            else:
                self.create_subscription(CompressedImage, os.environ.get('ERC_BOT_DEPTHREG_TOPIC', f'{cam}/depth/image_rect_raw/compressedDepth'),
                                         self.on_raw_depth, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            self.create_subscription(CameraInfo, f'{cam}/depth/camera_info', lambda m: setattr(self, 'raw_info', m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, f'{cam}/color/camera_info', lambda m: setattr(self, 'info', m), qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joints, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/mobile_base_controller/odom', lambda m: setattr(self, 'odom', m), qos_profile_sensor_data)
        self.create_subscription(String, '/erc/rejections', lambda m: self.rejections.append((self.now(), m.data)), 10)
        self.create_subscription(String, '/robot_description', self.on_urdf,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status = self.create_publisher(String, '/erc_bot/status', 10)
        self.actions = {c: ActionClient(self, FollowJointTrajectory, f'/erc/{c}/follow_joint_trajectory') for c in CONTROLLERS}

    # ---- time and spinning ----
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9 + self.skew

    def tick(self, dt=0.02):
        if not rclpy.ok():
            raise Abort('ROS shutdown')
        try:
            rclpy.spin_once(self, timeout_sec=dt)
            self._take_errors = 0
        except RuntimeError as e:
            # rclpy's executor raises out of take_from_wait_list ('Unable to convert call argument') for one message
            # now and then (B-v53-crash8 s1003 1/blue after the place, B-sweep2 row 4 in the retract): drop that one
            # message and keep spinning; a persistent failure still ends the mission
            if 'convert call argument' not in str(e):
                raise
            self._take_errors = getattr(self, '_take_errors', 0) + 1
            self.get_logger().warn(f'dropped one message: {e} ({self._take_errors} in a row)')
            if self._take_errors > 50:
                raise

    def wait(self, seconds):
        t0, w0 = self.now(), time.monotonic()
        while self.now() - t0 < seconds:
            self.tick()
            if time.monotonic() - w0 > max(120.0, seconds * 60):    # a loaded host can run the sim at 0.05 x real time
                raise Abort('clock stalled')

    def say(self, text):
        self.get_logger().info(text)
        self.status.publish(String(data=text))

    def on_colour(self, m):
        self.colour_msg = m
        self.colour_q.append((m, self.now()))
        if self.t_start is None:
            self.t_start = self.now()

    def on_depth(self, m):
        self.depth_msg, self.depth_recv = m, self.now()
        self.depth_recvs.append(self.depth_recv)

    def on_raw_depth(self, m):
        self.raw_depth_msg, self.raw_depth_recv = m, self.now()

    def on_urdf(self, m):
        u = m.data
        self.urdf = with_grasping_links(u)
        self.urdf_injected = self.urdf != u

    def on_scan(self, m):
        self.scan, self.scan_recv = m, self.now()

    @property
    def clock_offset(self):
        """Median of (receive time - header stamp) over recent joint states: this clock minus the robot's, plus latency."""
        return float(np.median(self.offsets)) if self.offsets else None

    def on_joints(self, m):
        self.joints.update(zip(m.name, m.position))
        self.joint_t = self.now()
        self.offsets.append(self.joint_t - stamp_s(m.header))

    def ready(self, timeout=60.0):
        """Block until joints, camera, depth, scan, odom, URDF and every /erc/ action server are there."""
        w0 = time.monotonic()
        while time.monotonic() - w0 < timeout:
            self.tick(0.05)
            missing = [k for k, v in (('joints', self.joints), ('colour', self.colour_msg), ('depth', self.depth_msg or self._reg_ready(w0)), ('camera_info', self.info),
                                      ('scan', self.scan), ('odom', self.odom), ('robot_description', self.urdf)) if not v]
            if not missing and all(a.server_is_ready() for a in self.actions.values()):
                off = self.clock_offset
                self.say(f'interface ready, this clock minus robot stamps {off:+.3f} s (receive-time freshness, any offset works)'
                         if off is not None else 'interface ready')
                return
        raise Abort(f'interface not ready after {timeout:.0f} s, missing {missing}, '
                    f'action servers down {[c for c, a in self.actions.items() if not a.server_is_ready()]}')

    # ---- frames ----
    def matrix(self, target, source, stamp=None):
        t = self.tf.lookup_transform(target, source, Time() if stamp is None else Time.from_msg(stamp)).transform
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
        m[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
        return m

    def base_in_odom(self):
        """(x, y, yaw) of base_footprint in odom from the odometry message (TF odom->base runs at 15 Hz only)."""
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        return np.array([p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))])

    def odom_T_base(self):
        x, y, yaw = self.base_in_odom()
        m = np.eye(4); m[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]; m[:2, 3] = [x, y]
        return m

    # ---- camera ----
    def camera(self, max_age=0.5, max_dt=0.12):
        """Latest colour (BGR), aligned depth (m, 0 = none), K (fx, fy, cx, cy) and the optical frame in base_footprint at
        the colour stamp. None when the pair is stale or out of sync. Falls back to the registered unaligned depth (see
        __init__, ERC_BOT_DEPTHREG) when the aligned stream is slow or absent."""
        if self.depthreg == '1':
            return self._camera_registered(max_age, max_dt)
        out = self._camera_aligned(max_age, max_dt)
        if out is None and self.depthreg == 'auto' and self.aligned_slow():
            out = self._camera_registered(max_age, max_dt)
        return out

    def aligned_slow(self):
        """Aligned depth arrives below ERC_BOT_DEPTHREG_MIN_HZ over the last 10 s (judged after 3 s of listening)."""
        now = self.now()
        if self.t_start is None:
            self.t_start = now
        win = min(10.0, now - self.t_start)
        if win < 3.0:
            return False
        return sum(1 for t in self.depth_recvs if t > now - win) / win < self.depthreg_min_hz

    def _reg_ready(self, w0):
        """ready(): the unaligned stream may stand in for aligned depth ('1' at once, 'auto' after a grace period)."""
        if self.depthreg == '0' or self.raw_depth_msg is None or self.raw_info is None:
            return None
        grace = float(os.environ.get('ERC_BOT_DEPTHREG_GRACE', '10'))
        return True if self.depthreg == '1' or time.monotonic() - w0 > grace else None

    def _extrinsic(self):
        """Depth optical frame -> colour optical frame, from /tf_static, cached. ERC_BOT_DEPTHREG_EXTRINSIC=identity is for
        the gym only: its depth/image_rect_raw is resampled from the colour-pose image (no real baseline)."""
        if self.T_cd is None:
            if os.environ.get('ERC_BOT_DEPTHREG_EXTRINSIC') == 'identity':
                self.T_cd = np.eye(4)
            else:
                try:
                    self.T_cd = self.matrix(self.info.header.frame_id or 'head_front_camera_color_optical_frame',
                                            self.raw_info.header.frame_id or 'head_front_camera_depth_optical_frame')
                except TransformException:
                    return None
        return self.T_cd

    def _camera_registered(self, max_age, max_dt):
        from .depthreg import register, decode_compressed_depth
        d = self.raw_depth_msg
        if d is None or self.raw_info is None or not self.colour_q or self.info is None:
            return None
        td = stamp_s(d.header)
        c, rc = min(self.colour_q, key=lambda mr: abs(stamp_s(mr[0].header) - td))
        tc = stamp_s(c.header)
        if self.now() - rc > max_age or self.now() - self.raw_depth_recv > max_age + max_dt or abs(tc - td) > max_dt:
            return None
        bgr = cv2.imdecode(np.frombuffer(bytes(c.data), np.uint8), cv2.IMREAD_COLOR)
        if hasattr(d, 'encoding'):                                 # raw Image (sim test)
            if d.encoding == '16UC1':
                dep = np.frombuffer(bytes(d.data), np.uint16).reshape(d.height, d.width).astype(np.float32) * 0.001
            elif d.encoding == '32FC1':
                dep = np.nan_to_num(np.frombuffer(bytes(d.data), np.float32).reshape(d.height, d.width).copy())
            else:
                dep = None
        else:
            dep = decode_compressed_depth(d.data)
        T_cd = self._extrinsic()
        if bgr is None or dep is None or T_cd is None:
            return None
        Kd, K = self.raw_info.k, self.info.k
        reg = register(dep, (Kd[0], Kd[4], Kd[2], Kd[5]), T_cd, (K[0], K[4], K[2], K[5]), bgr.shape[:2])
        try:
            pose = self.matrix('base_footprint', self.info.header.frame_id or 'head_front_camera_color_optical_frame', c.header.stamp)
        except TransformException:
            try:
                pose = self.matrix('base_footprint', self.info.header.frame_id or 'head_front_camera_color_optical_frame')
            except TransformException:
                return None
        if not self.reg_said:
            self.reg_said = True
            self.say(f'depth: registered unaligned {dep.shape[1]}x{dep.shape[0]} to colour (mode {self.depthreg}, '
                     f'aligned {len(self.depth_recvs)} frames so far, extrinsic t {np.round(T_cd[:3, 3], 4).tolist()})')
        return bgr, reg, (K[0], K[4], K[2], K[5]), pose, tc

    def _camera_aligned(self, max_age=0.5, max_dt=0.12):
        d = self.depth_msg
        if d is None or not self.colour_q or self.info is None:
            return None
        td = stamp_s(d.header)
        c, rc = min(self.colour_q, key=lambda mr: abs(stamp_s(mr[0].header) - td))
        tc = stamp_s(c.header)
        if self.now() - rc > max_age or self.now() - self.depth_recv > max_age + max_dt or abs(tc - td) > max_dt:
            return None
        bgr = cv2.imdecode(np.frombuffer(bytes(c.data), np.uint8), cv2.IMREAD_COLOR)
        buf = bytes(d.data); k = buf.find(b'\x89PNG', 0, 64)
        dep = cv2.imdecode(np.frombuffer(buf[k:], np.uint8), cv2.IMREAD_UNCHANGED) if k >= 0 else None
        if bgr is None or dep is None:
            return None
        try:
            pose = self.matrix('base_footprint', self.info.header.frame_id or 'head_front_camera_color_optical_frame', c.header.stamp)
        except TransformException:
            try:
                pose = self.matrix('base_footprint', self.info.header.frame_id or 'head_front_camera_color_optical_frame')
            except TransformException:
                return None
        K = self.info.k
        return bgr, dep.astype(np.float32) * 0.001, (K[0], K[4], K[2], K[5]), pose, tc

    # ---- joints ----
    def duration_for(self, names, target, minimum=0.6):
        cur = [self.joints[n] for n in names]
        t = max([abs(q - c) / (SPEED[n] * (SPEED_USE_HEAD if n.startswith('head') else SPEED_USE)) for n, q, c in zip(names, target, cur)
                 if n in SPEED] + [minimum])
        return t

    def send(self, controller, target, duration=None, wait=True, tolerance=0.03, timeout=None, check=True, stall=None):
        """One-point trajectory through the checking layer. Clamped to the live ranges, timed under the speed caps.
        Waits for the measured joints to settle near the target (the gripper stops early on an object, so grippers
        settle when they stop moving). stall=S (arms): joints that stop moving for S s short of the target end the wait
        at once with an Abort (a contact), instead of the full duration + 8 s."""
        names = CONTROLLERS[controller]
        target = [clamp(n, q) for n, q in zip(names, target)]
        dur = max(duration or 0.0, self.duration_for(names, target))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        p = JointTrajectoryPoint(); p.positions = [float(v) for v in target]
        p.time_from_start = Duration(sec=int(dur), nanosec=int((dur % 1) * 1e9))
        goal.trajectory.points = [p]
        n_rej = len(self.rejections)
        fut = self.actions[controller].send_goal_async(goal)
        w0 = time.monotonic()
        while not fut.done():
            self.tick()
            if time.monotonic() - w0 > 10:
                raise Abort(f'{controller}: no answer from the checking layer')
        gh = fut.result()
        if not gh.accepted:
            self.wait(0.2)
            why = [r for _, r in self.rejections[n_rej:]]
            raise Abort(f'{controller} goal rejected: {why[-1] if why else "no reason published"}')
        res = gh.get_result_async()          # the checking layer accepts, then aborts a bad goal with error_code -1

        def failed():
            if not res.done():
                return None
            r = res.result().result
            if r.error_code != 0:
                why = [x for _, x in self.rejections[n_rej:]]
                return f'{controller} goal failed ({r.error_code}): {r.error_string or (why[-1] if why else "")}'
            return None
        if not wait:
            self.wait(0.15)
            if failed():
                raise Abort(failed())
            return target
        t0 = self.now(); last = None; still = None
        limit = timeout or (dur + 8.0)
        if controller == 'torso_controller':     # the simulated torso travels at about 0.0105 m/s whatever is commanded
            limit = max(limit, dur + abs(target[0] - self.joints['torso_lift_joint']) / 0.009 + 5.0)
        while self.now() - t0 < limit:
            self.tick()
            if failed() and not controller.startswith('gripper'):
                raise Abort(failed())
            if self.now() - t0 < dur * 0.9:
                continue
            cur = np.array([self.joints[n] for n in names])
            if controller.startswith('gripper'):
                if last is not None and np.max(np.abs(cur - last)) < 0.002:
                    still = still if still is not None else self.now()
                    if self.now() - still > 0.4:
                        return target
                else:
                    still = None
                last = cur
                continue
            if np.max(np.abs(cur - np.array(target))) < tolerance:
                return target
            if stall:                            # window test in sim time: per tick a slow arm moves < 1 mrad at low RTF
                if still is None or self.now() - still[0] >= stall:
                    if still is not None and np.max(np.abs(cur - still[1])) < 0.003:
                        break
                    still = (self.now(), cur)
        if check and not controller.startswith('gripper'):
            err = np.max(np.abs(np.array([self.joints[n] for n in names]) - np.array(target)))
            raise Abort(f'{controller} did not reach its target (max error {err:.3f} rad)')
        return target

    def gripper(self, closed, side='right'):
        """Open or close; returns the measured position after it stops (live scale, 0.0 open, about 0.79 closed)."""
        self.send(f'gripper_{side}_controller', [GRIP_CLOSED if closed else GRIP_OPEN], duration=1.5)
        return self.joints[f'gripper_{side}_finger_joint']

    def head(self, pan, tilt, wait=True):
        return self.send('head_controller', [pan, tilt], wait=wait, tolerance=0.06, check=False)

    def torso(self, z, wait=True):
        return self.send('torso_controller', [z], tolerance=0.012, wait=wait)

    # ---- base ----
    def stop(self):
        self.cmd.publish(Twist())

    def scan_points(self):
        """Merged scan as (N, 2) points in base_footprint (the scan frame is virtual_base_laser_link)."""
        s = self.scan
        if s is None or self.now() - self.scan_recv > 1.0:        # receive time: robot stamps may be off this clock
            return None
        r = np.asarray(s.ranges, np.float32)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment
        ok = np.isfinite(r) & (r > max(s.range_min, 0.05)) & (r < s.range_max)
        try:
            m = self.matrix('base_footprint', s.header.frame_id)
        except TransformException:
            m = np.eye(4)
        p = np.c_[r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])]
        return p @ m[:2, :2].T + m[:2, 3]

    def velocity(self, vx=0.0, vy=0.0, wz=0.0, clearance=0.30, ignore=None):
        """Base twist with a laser stop: nothing may lie within `clearance` of the footprint in the travel direction.
        `ignore(points) -> mask` can exempt known geometry (the shelf face while docking)."""
        if abs(vx) + abs(vy) > 1e-4:
            pts = self.scan_points()
            if pts is None:
                self.stop(); return False        # no fresh scan: hold still, the caller times out
            if ignore is not None and len(pts):
                pts = pts[~ignore(pts)]
            d = np.array([vx, vy]) / math.hypot(vx, vy)
            along = pts @ d; side = np.abs(pts[:, 0] * d[1] - pts[:, 1] * d[0])
            # TIAGo Pro omni base footprint about 0.6 x 0.5 m: points ahead within the half-width plus margin
            if np.any((along > 0) & (along < 0.33 + clearance) & (side < 0.30)):
                self.stop()
                return False
        t = Twist(); t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self.cmd.publish(t)
        return True

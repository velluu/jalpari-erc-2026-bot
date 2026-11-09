"""SIMULATION ONLY. Gazebo ground truth for the learned policies' privileged (teacher) inputs and for the parity
records: model poses from gz-transport, robot link contacts from the gz contact sensors, the bin contact sensor.
Never available on the live robot; a stage that uses it says so in its name and in mission.json.

Names follow erc-gym (scorer.py): robot 'tiago_pro', books 'book_col_<c>_row_<r>_<label>', bin 'erc_collection_bin',
shelf 'erc_shelf', table 'erc_table'. Gazebo world = lane A's replica world W.
"""
import threading, time
import numpy as np

WORLD = 'erc_world'
ROBOT, BIN, SHELF, TABLE = 'tiago_pro', 'erc_collection_bin', 'erc_shelf', 'erc_table'


def quat_R(q):                       # x y z w
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _model(s):
    return s.split('::', 1)[0]


def _link(s):
    p = s.split('::')
    return p[1] if len(p) > 1 else p[0]


class GzTruth:
    def __init__(self, world=WORLD):
        from gz.transport13 import Node
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.msgs10.contacts_pb2 import Contacts
        from gz.msgs10.world_stats_pb2 import WorldStatistics
        self.lock = threading.Lock()
        self.poses, self.t_pose = {}, None
        self.sim_t = None
        self.contact_t = {}            # (robot link, other model) -> last sim time
        self.bin_t = {}                # other model -> last sim time on the bin sensor
        self.n_contact_msgs = 0
        self.node = Node()
        self.node.subscribe(Pose_V, f'/world/{world}/dynamic_pose/info', self._on_pose)
        self.node.subscribe(Pose_V, f'/world/{world}/pose/info', self._on_pose)
        self.node.subscribe(WorldStatistics, f'/world/{world}/stats', self._on_stats)
        self.node.subscribe(Contacts, f'/world/{world}/model/{BIN}/link/collection_bin_base_link/sensor/bin_contact_sensor/contact', self._on_bin)
        topics, t0 = [], time.time()
        while time.time() - t0 < 20 and len(topics) < 30:
            topics = [t for t in self.node.topic_list() if t.startswith(f'/world/{world}/model/{ROBOT}/link/') and t.endswith('_contact/contact')]
            time.sleep(0.3)
        self.contact_topics = topics
        for t in topics:
            self.node.subscribe(Contacts, t, self._on_contacts)

    @staticmethod
    def _stamp(msg):
        s = msg.header.stamp
        return s.sec + s.nsec * 1e-9 if (s.sec or s.nsec) else None

    def _on_stats(self, msg):
        with self.lock:
            self.sim_t = msg.sim_time.sec + msg.sim_time.nsec * 1e-9

    def _on_pose(self, msg):
        t = self._stamp(msg)
        with self.lock:
            for p in msg.pose:
                n = p.name
                if n.startswith('book_col_') or n in (ROBOT, BIN, SHELF, TABLE) or n.startswith('number_marker'):
                    o = p.orientation
                    self.poses[n] = (np.array([p.position.x, p.position.y, p.position.z]), np.array([o.x, o.y, o.z, o.w]))
            if t is not None:
                self.t_pose = t

    def _on_contacts(self, msg):
        t = self._stamp(msg)
        with self.lock:
            t = t if t is not None else self.sim_t
            self.n_contact_msgs += 1
            for c in msg.contact:
                a, b = c.collision1.name, c.collision2.name
                if _model(b) == ROBOT:
                    a, b = b, a
                if _model(a) != ROBOT:
                    continue
                self.contact_t[(_link(a), _model(b))] = t

    def _on_bin(self, msg):
        t = self._stamp(msg)
        with self.lock:
            t = t if t is not None else self.sim_t
            for c in msg.contact:
                for n in (_model(c.collision1.name), _model(c.collision2.name)):
                    if n != BIN:
                        self.bin_t[n] = t

    # ---- queries ----
    def wait(self, names=(ROBOT,), timeout=20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if all(n in self.poses for n in names) and self.sim_t is not None:
                    return True
            time.sleep(0.05)
        return False

    def now(self):
        with self.lock:
            return self.sim_t

    def pose(self, name):
        with self.lock:
            p, q = self.poses[name]
        return quat_R(q), p.copy(), q.copy()

    def T(self, name):
        R, p, _ = self.pose(name)
        m = np.eye(4); m[:3, :3] = R; m[:3, 3] = p
        return m

    def books(self):
        with self.lock:
            return sorted(n for n in self.poses if n.startswith('book_col_'))

    def nearest_book(self, p_w):
        return min(self.books(), key=lambda n: float(np.linalg.norm(self.pose(n)[1] - np.asarray(p_w))))

    def world_T_odom(self, odom_T_base):
        return self.T(ROBOT) @ np.linalg.inv(odom_T_base)

    def touching(self, pred, other, window):
        """Any robot link matching pred(link) in contact with model `other` (or other(model) True) within window s."""
        t = self.now()
        with self.lock:
            items = list(self.contact_t.items())
        for (link, model), tc in items:
            if tc is None or t is None or t - tc > window:
                continue
            if (other(model) if callable(other) else model == other) and pred(link):
                return True
        return False

    def pick_contacts(self, target, others, window=0.06):
        """lane A pick sensors: padL padR inL inR (right fingertips / inner fingers on the target), shelf (any robot link
        on the shelf), other (any robot link on another book of the column), arm_book (right arm subtree on target)."""
        f = lambda key: (lambda l: l == key)
        c = [self.touching(f('gripper_right_fingertip_left_link'), target, window),
             self.touching(f('gripper_right_fingertip_right_link'), target, window),
             self.touching(f('gripper_right_inner_finger_left_link'), target, window),
             self.touching(f('gripper_right_inner_finger_right_link'), target, window),
             self.touching(lambda l: True, SHELF, window),
             self.touching(lambda l: True, lambda m: m in others, window),
             self.touching(lambda l: l.startswith(('arm_right_', 'gripper_right_')), target, window)]
        return [float(x) for x in c]

    def place_contacts(self, target, window=0.06):
        """lane A place sensors: padL padR inL inR, book_bin (bin sensor), robot_bin, robot_table."""
        f = lambda key: (lambda l: l == key)
        t = self.now()
        with self.lock:
            tb = self.bin_t.get(target)
        c = [self.touching(f('gripper_right_fingertip_left_link'), target, window),
             self.touching(f('gripper_right_fingertip_right_link'), target, window),
             self.touching(f('gripper_right_inner_finger_left_link'), target, window),
             self.touching(f('gripper_right_inner_finger_right_link'), target, window),
             tb is not None and t is not None and t - tb <= window,
             self.touching(lambda l: True, BIN, window),
             self.touching(lambda l: True, TABLE, window)]
        return [float(x) for x in c]

    def snapshot(self, names):
        """Poses (p, quat xyzw) of the given models, the sim time, for parity records."""
        out = {'t': self.now(), 'poses': {}}
        for n in names:
            try:
                _, p, q = self.pose(n)
                out['poses'][n] = [p.tolist(), q.tolist()]
            except KeyError:
                pass
        with self.lock:
            out['contacts'] = sorted({f'{l}|{m}' for (l, m), tc in self.contact_t.items()
                                      if tc is not None and self.sim_t is not None and self.sim_t - tc <= 0.06})
        return out

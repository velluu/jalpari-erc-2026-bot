"""Gazebo ground truth for dev-sim debugging (never used by the bot itself): model poses from gz-transport in a
background thread, so reading them never blocks the ROS spin.

  g = GzTruth(); g.wait(); g.pose('tiago_pro') -> 4x4 world pose; g.xyz('erc_collection_bin')
  g.world_T_odom(live) -> world <- odom from the robot's true pose and its odometry at the same instant
"""
import threading, time
import numpy as np
from scipy.spatial.transform import Rotation
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V

WORLD = 'erc_world'


class GzTruth:
    def __init__(self, world=WORLD):
        self.lock = threading.Lock()
        self.poses = {}
        self.node = Node()
        self.node.subscribe(Pose_V, f'/world/{world}/dynamic_pose/info', self._on)
        self.node.subscribe(Pose_V, f'/world/{world}/pose/info', self._on)

    def _on(self, msg):
        with self.lock:
            for p in msg.pose:
                o = p.orientation
                self.poses[p.name] = (np.array([p.position.x, p.position.y, p.position.z]), np.array([o.x, o.y, o.z, o.w]))

    def wait(self, names=('tiago_pro',), timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if all(n in self.poses for n in names):
                    return True
            time.sleep(0.05)
        return False

    def pose(self, name):
        with self.lock:
            xyz, q = self.poses[name]
        m = np.eye(4); m[:3, :3] = Rotation.from_quat(q).as_matrix(); m[:3, 3] = xyz
        return m

    def xyz(self, name):
        return self.pose(name)[:3, 3]

    def world_T_odom(self, live):
        return self.pose('tiago_pro') @ np.linalg.inv(live.odom_T_base())


def yaw_of(m):
    return float(np.arctan2(m[1, 0], m[0, 0]))

"""Forward kinematics from a URDF (string or path), pure numpy: any link pose in any ancestor frame, revolute,
continuous, prismatic and fixed joints, mimic joints. Used by the policy runner to build the learned policies'
observations (grasping link and finger pads in base_footprint) from /joint_states and the live /robot_description.

  fk = UrdfFK(urdf); T = fk.pose('gripper_right_grasping_link', joints)      # 4x4 in base_footprint
"""
import math
import xml.etree.ElementTree as ET
import numpy as np


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, q):
    x, y, z = axis
    c, s = math.cos(q), math.sin(q)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class UrdfFK:
    def __init__(self, urdf, base='base_footprint'):
        root = ET.fromstring(urdf) if urdf.lstrip().startswith('<') else ET.parse(urdf).getroot()
        self.base = base
        self.joints = {}
        self.parent_joint = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            T = np.eye(4); T[:3, :3] = rpy_matrix(*rpy); T[:3, 3] = xyz
            a = j.find('axis')
            axis = np.array([float(v) for v in a.get('xyz').split()]) if a is not None else np.array([1.0, 0.0, 0.0])
            axis = axis / (np.linalg.norm(axis) or 1.0)
            m = j.find('mimic')
            mimic = (m.get('joint'), float(m.get('multiplier', 1)), float(m.get('offset', 0))) if m is not None else None
            lim = j.find('limit')
            rng = (float(lim.get('lower', '-inf')), float(lim.get('upper', 'inf'))) if lim is not None else (-math.inf, math.inf)
            info = {'name': j.get('name'), 'type': j.get('type'), 'parent': j.find('parent').get('link'),
                    'child': j.find('child').get('link'), 'T': T, 'axis': axis, 'mimic': mimic, 'range': rng}
            self.joints[info['name']] = info
            self.parent_joint[info['child']] = info
        self._chains = {}

    def chain(self, tip, base=None):
        base = base or self.base
        key = (tip, base)
        if key not in self._chains:
            c, link = [], tip
            while link != base:
                if link not in self.parent_joint:
                    raise KeyError(f'{tip} is not below {base} in the URDF')
                j = self.parent_joint[link]
                c.append(j)
                link = j['parent']
            self._chains[key] = list(reversed(c))
        return self._chains[key]

    def value(self, j, q):
        if j['mimic'] is not None:
            src, mul, off = j['mimic']
            return mul * self.value(self.joints[src], q) + off
        return float(q.get(j['name'], 0.0))

    def pose(self, tip, q, base=None):
        """4x4 pose of `tip` in `base` (default base_footprint). q: {joint name: value}, missing joints read 0."""
        T = np.eye(4)
        for j in self.chain(tip, base):
            T = T @ j['T']
            t = j['type']
            if t in ('revolute', 'continuous'):
                M = np.eye(4); M[:3, :3] = axis_angle(j['axis'], self.value(j, q)); T = T @ M
            elif t == 'prismatic':
                M = np.eye(4); M[:3, 3] = j['axis'] * self.value(j, q); T = T @ M
        return T

    def range(self, name):
        return self.joints[name]['range']

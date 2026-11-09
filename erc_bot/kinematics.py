"""Arm kinematics from the robot's own URDF (the live /robot_description), joint ranges narrowed to the live limits.

From team JALPARI's Phase 1 solution (Phroggo/jalpari-erc-2026-old, erc_solution/kinematics.py): the chain parser,
FK, multi-start IK, the upright transport path and the compact postures are theirs. Changed for the live robot: the
URDF comes as a string, and every joint range is intersected with the Interface Guide limits less the checking
layer margin (the live arm joints 1 and 5 are asymmetric, the simulation URDF is not).
"""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """Build a 4x4 pose matrix from metres and roll/pitch/yaw in radians."""
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    result[:3, 3] = xyz
    return result


# The real robot's /robot_description (PAL's xacro, session-1 bag) has no gripper_<side>_grasping_link, and its TF has no
# such frame: its chain ends arm_<side>_7_link -> arm_<side>_tool_link -> gripper_<side>_base_link. The sim's fixed joint
# (both sides, same values) puts the grasping link here on gripper_<side>_base_link:
GRASP_XYZ, GRASP_RPY = (0.0, 0.0, 0.157157), (0.0, -1.57, 0.0)
GRASP_OFFSET = transform(GRASP_XYZ, GRASP_RPY)


def with_grasping_links(urdf):
    """The URDF string itself when it has both grasping links (the sim), else a copy with the sim's fixed joints added."""
    try:
        names = {l.get('name') for l in ET.fromstring(urdf).iter('link')}
        add = [s for s in ('right', 'left') if f'gripper_{s}_grasping_link' not in names and f'gripper_{s}_base_link' in names]
        if not add:
            return urdf
        xyz, rpy = ' '.join(map(str, GRASP_XYZ)), ' '.join(map(str, GRASP_RPY))
        extra = ''.join(f'<link name="gripper_{s}_grasping_link"/><joint name="gripper_{s}_grasping_frame_joint" type="fixed">'
                        f'<origin xyz="{xyz}" rpy="{rpy}"/><parent link="gripper_{s}_base_link"/>'
                        f'<child link="gripper_{s}_grasping_link"/></joint>' for s in add)
        i = urdf.rindex('</robot>')
        return urdf[:i] + extra + urdf[i:]
    except Exception:                          # never breaks the subscription callback; Kinematics then reports the URDF
        return urdf


def transport_path(position):
    """Upright carrying poses, used only after retreating clear of the shelf.

    A low tool (row 4, about (0.72, -0.24, 0.96) with the torso up) goes straight up in 2 cm steps. The Phase 1 path
    (outward to y -0.50, then in to (0.50, -0.24, 1.28)) jumped IK branches (2.52 rad on one step, offline replay on
    the sim URDF, 2026-11-05 23:30) and left the wrist flipped, so the place IK put links 6 and 7 on the bin's side
    wall (row 4 placed 0 of 5 held out). Straight up stays in one branch (largest step 0.43 rad, tool within 2.1 cm).
    A straight inward/up line still crosses a joint-limit hole, so the tool keeps its x and y.
    """
    position = np.asarray(position,dtype=float).copy()
    if position[2] < 1.10:
        top = position.copy(); top[2] = 1.28
        n = max(2, int(np.ceil((top[2] - position[2]) / 0.02)))
        return [transform(position + a*(top-position)) for a in np.linspace(0., 1., n + 1)[1:]]
    endpoints = [np.array([.50,-.24,1.28])]
    goals = []
    for endpoint in endpoints:
        goals.extend(transform(position+a*(endpoint-position))
                     for a in np.linspace(0.,1.,9)[1:])
        position = endpoint
    return goals


class Kinematics:
    def __init__(self, urdf, base='torso_lift_link', tip='gripper_right_grasping_link', limits=None, margin=0.07):
        root = ET.fromstring(urdf) if urdf.lstrip().startswith('<') else ET.parse(urdf).getroot()
        parents = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain = []
        while tip != base:
            j = parents[tip]
            chain.append(j)
            tip = j.find('parent').get('link')
        self.chain, self.names, self.lower, self.upper = [], [], [], []
        for j in reversed(chain):
            origin = j.find('origin')
            xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()] if origin is not None else [0]*3
            rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()] if origin is not None else [0]*3
            axis = j.find('axis')
            axis = np.array([float(v) for v in axis.get('xyz').split()]) if axis is not None else np.array([1., 0., 0.])
            movable = j.get('type') == 'revolute'
            if movable:
                limit = j.find('limit')
                self.names.append(j.get('name'))
                lo, hi = float(limit.get('lower')) + margin, float(limit.get('upper')) - margin   # checking-layer margin on every joint
                if limits and j.get('name') in limits:
                    lo, hi = max(lo, limits[j.get('name')][0] + margin), min(hi, limits[j.get('name')][1] - margin)
                self.lower.append(lo)
                self.upper.append(hi)
            elif j.get('type') != 'fixed':
                raise ValueError('Arm chain must contain only revolute/fixed joints')
            self.chain.append((transform(xyz, rpy), axis, movable))
        self.lower, self.upper = np.array(self.lower), np.array(self.upper)

    def fk(self, q):
        """Get the tool pose from joint angles, relative to the chain's base."""
        result, i = np.eye(4), 0
        for fixed, axis, movable in self.chain:
            result = result @ fixed
            if movable:
                turn = np.eye(4)
                turn[:3, :3] = Rotation.from_rotvec(axis*q[i]).as_matrix()
                result = result @ turn
                i += 1
        return result

    def points(self, q):
        """Return joint/link origins for rough arm-clearance checks."""
        result, i, points = np.eye(4), 0, []
        for fixed,axis,movable in self.chain:
            result = result @ fixed
            if movable:
                turn = np.eye(4)
                turn[:3,:3] = Rotation.from_rotvec(axis*q[i]).as_matrix()
                result = result @ turn
                i += 1
            points.append(result[:3,3].copy())
        return np.array(points)

    def solve(self, goal, current, clear=None):
        """Find joints that match both tool position and angle.

        The grasp frame's X axis points out of the fingers. Rotation vectors
        keep the angle error usable even when the tool is 180 degrees off.
        """
        current = np.clip(current, self.lower, self.upper)
        # clear = (z_min, indices): keep those chain points (and the midpoints between consecutive ones) above z_min in
        # the chain's base frame, e.g. the elbow and forearm above a bin rim while the gripper goes below it
        # clear = {'z': z_min, 'box': (x0, x1, y0, y1), 'outer': (x0, x1, y0, y1), 'idx': chain point indices, 'n':
        # samples per segment}: every sample along the segments between those points that lies over the bin's walls
        # (inside 'outer', the outer footprint grown by the link radius, but outside 'box', the inner footprint shrunk by
        # it) must be above z_min. z_min = rim + link radius
        # + margin, so the links are modelled at their real thickness. The tuple form is the 18:19 / 21:20 rule.
        if isinstance(clear, dict):
            idx = list(clear['idx']); box = clear['box']; ns = int(clear.get('n', 5)); x_max = None
            tt = np.linspace(0, 1, ns + 1)[:-1]
        else:
            idx = list(clear[1]) if clear else []
            x_max = clear[2] if clear and len(clear) > 2 else None
            box = None
        def below(q):
            """How far each sample of the listed chain sits under z_min (0 when above or inside the allowed box)."""
            P = self.points(q)
            if box is not None:
                A, B = P[idx[:-1]], P[idx[1:]]
                Q = np.r_[(A[:, None, :] + (B - A)[:, None, :] * tt[None, :, None]).reshape(-1, 3), P[idx[-1:]]]
                d = np.maximum(clear['z'] - Q[:, 2], 0)
                inside = (Q[:, 0] > box[0]) & (Q[:, 0] < box[1]) & (Q[:, 1] > box[2]) & (Q[:, 1] < box[3])
                d[inside] = 0.0
                ob = clear.get('outer')                # only over the walls: away from the bin nothing is in reach
                if ob is not None:
                    d[~((Q[:, 0] > ob[0]) & (Q[:, 0] < ob[1]) & (Q[:, 1] > ob[2]) & (Q[:, 1] < ob[3]))] = 0.0
                return d
            Q = np.r_[P[idx], (P[idx[:-1]] + P[idx[1:]]) / 2]
            d = np.maximum(clear[0] - Q[:, 2], 0)
            if x_max is not None:
                d[Q[:, 0] > x_max] = 0.0
            return d
        def residual(q):
            actual = self.fk(q)
            rotation = Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).as_rotvec()
            r = [4*(actual[:3, 3]-goal[:3, 3]), rotation, .005*(q-current)]
            if idx:
                r.append(3*below(q))
            return np.concatenate(r)
        best = None
        seeds = [current, np.clip([-.5, -.6, 0, 1.4, 0, -.8, 0], self.lower, self.upper)]
        rng = np.random.default_rng(2026)
        # Repeatable IK guesses only; this does not seed the simulation layout.
        seeds.extend(rng.uniform(self.lower,self.upper) for _ in range(10))
        for seed in seeds:
            sol = least_squares(residual, seed, bounds=(self.lower, self.upper), max_nfev=160)
            actual = self.fk(sol.x)
            pe = np.linalg.norm(actual[:3, 3]-goal[:3, 3])
            re = Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).magnitude()
            if idx:
                low = below(sol.x).max()
                if low > .01:
                    pe += 1.0 + low          # a solution through the rim is never accepted
            if best is None or pe+re < best[1]+best[2]:
                best = (sol.x, pe, re)
            if pe < .008 and re < .045:
                return sol.x
        if best[1] < .015 and best[2] < .07:      # close enough: the measured-pose check after the move holds 2 cm
            return best[0]
        raise RuntimeError(f'Unreachable grasp pose: position error {best[1]:.3f} m, orientation {best[2]:.3f} rad')

    def startup_waypoint(self,current,side=-1):
        goal = np.array([.25,side*.32,.10])
        seed = np.clip(current,self.lower,self.upper)
        res = lambda q:np.r_[10*(self.fk(q)[:3,3]-goal),.02*(q-seed)]
        fit = least_squares(res,seed,bounds=(self.lower,self.upper),max_nfev=200)
        if np.linalg.norm(self.fk(fit.x)[:3,3]-goal)<=.01:
            return fit.x
        # From deep in the bin (tool z 0.87, rows 1 and 4 in the offline replay) the local fit stalls: other starts, the
        # solution with the smallest joint move wins.
        rng = np.random.default_rng(7)
        best = None
        for initial in [np.clip(np.zeros(7),self.lower,self.upper)]+[rng.uniform(self.lower,self.upper) for _ in range(12)]:
            fit = least_squares(res,initial,bounds=(self.lower,self.upper),max_nfev=200)
            if np.linalg.norm(self.fk(fit.x)[:3,3]-goal)<=.01:
                step = np.abs(fit.x-seed).max()
                if best is None or step<best[0]:
                    best = (step, fit.x)
        if best is None:
            raise RuntimeError('Cannot reach raised startup waypoint')
        return best[1]

    def compact(self, current, side=-1):
        """Find a tucked pose with limited forward and sideways arm reach."""
        goal = np.array([.25,side*.32,.10])
        seed = np.clip(current,self.lower,self.upper)
        def residual(q):
            points = self.points(q)
            return np.r_[2*(points[-1]-goal),.02*(q-seed),
                         30*np.maximum(points[:,0]-.38,0),
                         10*np.maximum(abs(points[:,1])-.50,0)]
        rng = np.random.default_rng(19)
        for initial in [seed]+[rng.uniform(self.lower,self.upper) for _ in range(8)]:
            fit = least_squares(residual,initial,bounds=(self.lower,self.upper),max_nfev=200)
            points = self.points(fit.x)
            if np.linalg.norm(points[-1]-goal)<.18 and points[:,0].max()<.40 and abs(points[:,1]).max()<.53:
                return fit.x
        raise RuntimeError('Cannot find whole-arm compact startup posture')

"""Right-arm motion for the bot: IK through the live URDF, compact parking for both arms, checked tool poses.

The parking sequence and the measured-pose check follow team JALPARI's Phase 1 Robot.park and Robot.arm_pose.
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import GRASP_OFFSET, Kinematics, transform
from .live import Abort, ARM_LEFT, ARM_RIGHT, LIVE_LIMITS, MARGIN


class Arm:
    def __init__(self, live):
        self.r = live
        self.k = Kinematics(live.urdf, tip='gripper_right_grasping_link', limits=LIVE_LIMITS, margin=MARGIN)
        self.k_left = Kinematics(live.urdf, tip='gripper_left_grasping_link', limits=LIVE_LIMITS, margin=MARGIN)
        assert self.k.names == ARM_RIGHT and self.k_left.names == ARM_LEFT, (self.k.names, self.k_left.names)
        self._home = {}
        # The real robot's TF has no grasping frame (Live added the link to its URDF): the tool pose is TF to the gripper
        # base composed with the sim's fixed offset. ERC_BOT_TOOL_FALLBACK=1 forces that path in the sim, and each tool()
        # then also prints its difference from the TF frame (test only).
        self.tool_check = os.environ.get('ERC_BOT_TOOL_FALLBACK') == '1'
        self.tool_fallback = bool(getattr(live, 'urdf_injected', False)) or self.tool_check

    def q(self, names):
        return np.array([self.r.joints[n] for n in names])

    def park(self, parallel=True, canonical=True):
        """Fold both arms into a compact posture (raised waypoint first, then tucked). Both arms move at once (mirror
        images, 7 s instead of 14 s in sim); parallel=False is the old right-then-left order.
        canonical: when the tuck (fitted from the current joints) is more than 0.05 rad from the canonical tuck (the same
        fit from zero joints, the sim spawn), one more move takes the arm there, so the parked posture is the same from
        any start. From the slot sheet's present pose the local tuck was 5.1 rad away on joint 1, every later IK (seeded
        from the current joints) stayed on that branch and the wrist went into the bin (lane F 09:38, 1 of 4 placed).
        The extra move starts from the local tuck: fitted straight from the present pose, the joint line put the fingers
        on the base (offline mesh check). canonical=False: the park after the release, with the arm in the bin."""
        arms = ((self.k, -1, 'arm_right_controller'), (self.k_left, 1, 'arm_left_controller'))
        if not parallel:
            for model, side, ctrl in arms:
                self._park_one(model, side, ctrl, canonical)
            return
        plan = []
        for model, side, ctrl in arms:
            raised, tuck, home = self.park_plan(model, self.q(model.names), side, canonical)
            low = min(model.points(a + (b - a) * t)[:, 2].min() for a, b in ((raised, tuck), (tuck, home)) for t in np.linspace(0, 1, 41))
            if low < -0.70:
                raise Abort('parking path would approach the floor')
            plan.append((model, ctrl, raised, tuck, home))
        steps = (2, 3, 4) if any(p[4] is not p[3] for p in plan) else (2, 3)
        for step in steps:                                     # raised for both, tucked for both, then the canonical tuck
            (m0, c0, *q0), (m1, c1, *q1) = plan
            timeout = max(15.0, self.r.duration_for(m0.names, q0[step - 2]) + 8.0)    # the canonical move is up to 15 s
            self.r.send(c0, q0[step - 2], tolerance=0.04, wait=False)
            self.r.send(c1, q1[step - 2], tolerance=0.04)
            self.wait_joints(m0.names, q0[step - 2], timeout=timeout)

    def park_plan(self, model, cur, side, canonical=True):
        """(raised, tuck, home) from these joints: home is the canonical tuck, or the tuck itself when within 0.05 rad."""
        cur = np.clip(cur, model.lower, model.upper)
        raised = model.startup_waypoint(cur, side)
        tuck = model.compact(raised, side)
        home = self.home(model, side) if canonical else tuck
        return raised, tuck, (tuck if np.abs(home - tuck).max() <= 0.05 else home)

    def home(self, model, side):
        """The canonical tuck: the park's two fits from zero joints (the sim spawn), on this robot's URDF and limits."""
        if model not in self._home:
            self._home[model] = model.compact(model.startup_waypoint(np.clip(np.zeros(len(model.names)), model.lower, model.upper), side), side)
        return self._home[model]

    def wait_joints(self, names, target, tol=0.04, timeout=15.0):
        t0 = self.r.now()
        while self.r.now() - t0 < timeout:
            self.r.tick()
            if np.max(np.abs(self.q(names) - np.asarray(target))) < tol:
                return
        raise Abort(f'{names[0][:9]} did not reach its park target')

    def _park_one(self, model, side, ctrl, canonical=True):
        cur = np.clip(self.q(model.names), model.lower, model.upper)
        raised = model.startup_waypoint(cur, side)
        self.r.send(ctrl, raised, tolerance=0.04)
        cur = self.q(model.names)
        tuck = model.compact(cur, side)
        low = min(model.points(cur + (tuck - cur) * t)[:, 2].min() for t in np.linspace(0, 1, 41))
        if low < -0.70:
            raise Abort('parking path would approach the floor')
        self.r.send(ctrl, tuck, tolerance=0.04)
        home = self.home(model, side) if canonical else tuck
        if np.abs(home - tuck).max() > 0.05:
            if min(model.points(tuck + (home - tuck) * t)[:, 2].min() for t in np.linspace(0, 1, 41)) < -0.70:
                raise Abort('parking path would approach the floor')
            self.r.send(ctrl, home, tolerance=0.04)

    def solve(self, goal_base, seed=None, clear_z=None, clear_x=None, clear_box=None, clear_outer=None):
        """clear_z: keep the arm from the upper arm to the wrist (chain points 2 to 6 and the segments between) above
        this height in base_footprint, where they lie nearer than clear_x (the bin's near wall plus a margin), so the
        wrist and gripper may go below the rim over the bin while the forearm crosses the wall above it. Without
        clear_x only points 2 to 4 are held up (the 18:19 rule, which let arm_right_5_link brush the rim once)."""
        torso_T_base = self.r.matrix('torso_lift_link', 'base_footprint')
        if clear_z is None:
            clear = None
        elif clear_box is not None:            # (x0, x1, y0, y1) in base_footprint: the arm may go below clear_z only there
            dx, dy = torso_T_base[0, 3], torso_T_base[1, 3]
            x0, x1, y0, y1 = clear_box
            clear = {'z': clear_z + torso_T_base[2, 3], 'box': (x0 + dx, x1 + dx, y0 + dy, y1 + dy), 'idx': (1, 2, 3, 4, 5, 6, 7), 'n': 5}
            if clear_outer is not None:
                o0, o1, o2, o3 = clear_outer
                clear['outer'] = (o0 + dx, o1 + dx, o2 + dy, o3 + dy)
        elif clear_x is None:
            clear = (clear_z + torso_T_base[2, 3], (2, 3, 4))
        else:
            clear = (clear_z + torso_T_base[2, 3], (2, 3, 4, 5, 6), clear_x + torso_T_base[0, 3])
        try:
            return self.k.solve(torso_T_base @ goal_base, self.q(self.k.names) if seed is None else seed, clear)
        except RuntimeError as e:
            raise Abort(f'IK: {e} at {np.round(goal_base[:3, 3], 3).tolist()}')

    def tool(self):
        if not self.tool_fallback:
            return self.r.matrix('base_footprint', 'gripper_right_grasping_link')
        T = self.r.matrix('base_footprint', 'gripper_right_base_link') @ GRASP_OFFSET
        if self.tool_check and not self.r.urdf_injected:
            ref = self.r.matrix('base_footprint', 'gripper_right_grasping_link')
            print(f'[tool-fallback] dpos {np.linalg.norm(T[:3, 3] - ref[:3, 3]):.6f} m dang '
                  f'{Rotation.from_matrix(T[:3, :3] @ ref[:3, :3].T).magnitude():.6f} rad', flush=True)
        return T

    def pose(self, goal_base, duration=None, pos_tol=0.02, ang_tol=0.09, clear_z=None, clear_x=None, clear_box=None,
             clear_outer=None, stall=None):
        """Move the grasping link to a 4x4 pose in base_footprint and check where it really went."""
        q = self.solve(goal_base, clear_z=clear_z, clear_x=clear_x, clear_box=clear_box, clear_outer=clear_outer)
        self.r.send('arm_right_controller', q, duration=duration, tolerance=0.025, stall=stall)
        self.r.wait(0.15)
        actual = self.tool()
        err = np.linalg.norm(actual[:3, 3] - goal_base[:3, 3])
        ang = Rotation.from_matrix(goal_base[:3, :3] @ actual[:3, :3].T).magnitude()
        if err > pos_tol or ang > ang_tol:
            raise Abort(f'tool off target by {err:.3f} m / {ang:.3f} rad')
        return actual

    def path(self, goals, duration=None):
        """Check every waypoint is reachable before moving (a way back out must exist), then follow them."""
        seed = self.q(self.k.names)
        torso_T_base = self.r.matrix('torso_lift_link', 'base_footprint')
        for g in goals:
            try:
                seed = self.k.solve(torso_T_base @ g, seed)
            except RuntimeError as e:
                raise Abort(f'IK: path waypoint {np.round(g[:3, 3], 3).tolist()} unreachable: {e}')
        for g in goals:
            self.pose(g, duration)


__all__ = ['Arm', 'transform']

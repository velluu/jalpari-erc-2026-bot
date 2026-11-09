"""Learned-policy plumbing that needs no ROS: the exported policy (numpy), the observation vectors of lane A's
pi-pick and pi-place (teacher and student), and the action-to-command mapping of their step(), so the same code
runs on the laptop, in the Gazebo probes and in the MuJoCo parity check.

Source of truth: ~/projects/erc-learn (lane A), erc_learn/env_pick.py (PickEnv._obs, _student_obs, step) and
erc_learn/env_place.py (PlaceEnv._obs). Every constant below is copied from there and cited; the parity test
(tests/t_policy_offline.py) checks these builders against lane A's own _obs on replica states.

Frames. W is the replica world: shelf front plane x = 2.755 with the inward normal +x, +y toward column 1, z up from
the floor (Gazebo world = W). On the live robot W is built from the shelf line (digit plates) in odom: x along the
inward normal n, the front plane at 2.755, z = odom z. Observations only use base-relative vectors, absolute heights
and the base yaw relative to n, so the y origin of W never matters.
"""
import json, math, os
import numpy as np

# ---- lane A constants (erc_learn/scene.py, env_pick.py, env_place.py) ----
SHELF_FRONT = 2.755
BOOK_H, BOOK_W, BOOK_D = 0.25, 0.03, 0.16
BOARD_TOP = np.array([1.452, 1.122, 0.792, 0.462])     # rows 1..4
BAY_PITCH = 1.035
BAY_Y = np.array([(2 - c) * BAY_PITCH for c in range(5)])   # physical columns 1..5
SIM_OPEN = 0.069                                         # sim finger joint: 0.069 open .. 0.0 closed
REAL_CLOSED = 0.79                                       # live gripper: 0.0 open .. 0.79 closed (erc-gym shim mapping)
CAPS = np.array([0.035, 0.60, 0.60, 0.75, 0.75, 0.75, 0.75, 0.75])
JOINTS8 = ['torso_lift_joint'] + [f'arm_right_{i}_joint' for i in range(1, 8)]
FINGER = 'gripper_right_finger_joint'
REACH = {1: .55, 2: .72, 3: .75, 4: .85}
TORSO = {1: .295, 2: .20, 3: .055, 4: .055}           # lane A 18:51: erc-bot rows clamped to the live 0.055 .. 0.295 m
ARM_Y = -0.24
BIN_FLOOR = 0.75
# Joint ranges of the replica (assets/erc_pick_train.xml jnt_range, from tiago_pro_sim.urdf), torso live 0..0.35
RANGE8 = np.array([[-0.001, 0.35], [-4.712389, 0.523599], [-2.443461, 1.134464], [-2.617994, 2.617994],
                   [-2.443461, 1.134464], [-1.570796, 3.665191], [-1.884956, 3.001966], [-2.443461, 2.443461]])
# Pre-grasp postures per row (assets/pregrasp.json q_pre) and the place hold (assets/place_start.json)
Q_PRE = {1: [-3.024236, -1.780963, -0.322309, 0.627139, 0.743656, -1.814956, 0.208181],
         2: [-4.213773, -2.373461, -1.108022, -2.055219, -1.455278, 0.842742, 1.989445],
         3: [-4.236597, -2.373461, -0.900385, -2.306145, -1.039953, 0.745485, 1.594923],
         4: [-1.430193, -0.995395, 0.244388, -2.225594, -0.231877, 0.782759, -1.757668]}
Q_HOLD = [-2.332066, -1.85528, 0.82236, -2.365002, 0.361259, 0.648217, -1.009101]
Q_HOLD_1830 = list(Q_HOLD)      # place_start.json q_hold at 18:30 (torso 0.345), the start of runs trained before 19:07
TORSO_HOLD = 0.295
Q_CLOSED_ON_BOOK = 0.03
# Midpoint of the two finger-pad sites (fingertip link inertial origins, build_model.py) in the grasping link frame as
# a function of the sim finger value, x and z: quartic fit over 70 samples of the replica (max fit error 2.6e-8 m,
# 2026-11-05 on the Z8); y is 0 (symmetric). Pad sites 2.62 cm apart closed, 8.03 cm open.
PAD_MID_X = np.array([9.35963997e+00, 3.51899704e-02, -1.71497224e+00, 1.03486882e-05, -1.50462189e-02])
PAD_MID_Z = np.array([7.46343214e-03, 1.00309003e-05, -1.36566290e-03, 1.52912867e-06, -1.20393434e-05])


def real_to_sim(q):
    return float(np.clip(SIM_OPEN * (1.0 - q / REAL_CLOSED), 0.0, SIM_OPEN))


def sim_to_real(q):
    return float(np.clip(REAL_CLOSED * (1.0 - q / SIM_OPEN), 0.0, 0.803))


def pad_mid_in_grasp(fing):
    return np.array([np.polyval(PAD_MID_X, fing), 0.0, np.polyval(PAD_MID_Z, fing)])


def rot6(Rb, R):
    return (Rb.T @ R)[:, :2].reshape(-1)


def yaw_R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------- policy
class NumpyPolicy:
    """erc_learn.export.NumpyPolicy (lane A), plus the sidecar meta (task, student, control config)."""

    def __init__(self, path):
        z = np.load(path)
        self.path = path
        self.mean, self.std = z['obs_mean'], z['obs_std']
        self.W = [z[f'W{i}'] for i in range(int(z['n_layers']))]
        self.b = [z[f'b{i}'] for i in range(int(z['n_layers']))]
        self.act_size = int(z['act_size'])
        self.obs_size = int(self.mean.shape[0])
        meta = os.path.splitext(path)[0] + '.json'
        self.meta = json.load(open(meta)) if os.path.exists(meta) else {}

    def act(self, obs):
        x = (np.asarray(obs, np.float32) - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = x / (1.0 + np.exp(-x))          # swish
        return np.tanh(x[..., :self.act_size])


# ---------------------------------------------------------------- joint convention
# Lane A trains in the sim URDF's joint coordinates (tiago_pro_sim.urdf): arm_right_1 axis +z, -4.712 .. 0.524 rad,
# arm_right_5 axis +z, -1.571 .. 3.665 rad. The Interface Guide gives the real right arm 1 as -30 .. 270 deg and 5 as
# -210 .. 90 deg, the exact negation. erc-gym's tiago_pro_real.urdf (the live profile's /robot_description) matches
# the sim. Every other joint is symmetric or matches.
SIM_REF = {'arm_right_1_joint': ((0.0, 0.0, 1.0), (-4.71238898038469, 0.5235987755982988)),
           'arm_right_5_joint': ((0.0, 0.0, 1.0), (-1.5707963267948966, 3.6651914291880923))}
LIVE_MARGIN = 0.055          # checking layer 0.05 rad / 0.05 m, plus 5 mm of our own
TORSO_LIVE = (0.0 + LIVE_MARGIN, 0.35 - LIVE_MARGIN)     # 0.055 .. 0.295 m
TWO_PI = 2 * math.pi


class ConventionError(RuntimeError):
    pass


class Convention:
    """Sim (policy) joint coordinates <-> live joint coordinates for JOINTS8, keyed off the live /robot_description
    (axis xyz and limits of arm_right_1 and arm_right_5). Per joint:
      id    same axis, same limits                       q_live = q_sim
      neg   axis flipped, limits negated                 q_live = -q_sim (the same physical rotation)
      wrap  same axis, other limits (e.g. the Guide's)   q_live = q_sim + 2 pi k, k picked so q_live is inside the live
            range less the margin; a sim value with no such k is unreachable (ConventionError). The legal sim range is
            then the interval around the current posture (two intervals exist for joint 1).
    force={'arm_right_1_joint': 'neg', ...} overrides the automatic choice (e.g. 'neg' on a same-axis URDF if the
    first hardware move shows the joint counts the other way). Anything else fails loudly."""

    MODES = ('id', 'neg', 'wrap')

    def __init__(self, modes=None, live_ranges=None, axis=None, report=None):
        self.modes = list(modes or ['id'] * 8)
        self.live_ranges = live_ranges or {}           # index -> (lo, hi) without margin
        self.axis = np.ones(8) if axis is None else np.asarray(axis, float)
        self.report = report or {}

    @classmethod
    def from_fk(cls, fk, tol=1e-3, force=None):
        force = force or {}
        modes, ranges, axis, rep = ['id'] * 8, {}, np.ones(8), {}
        for name, (a_sim, (lo_s, hi_s)) in SIM_REF.items():
            i = JOINTS8.index(name)
            j = fk.joints.get(name)
            if j is None:
                raise ConventionError(f'{name} missing from the live URDF')
            a_l = np.asarray(j['axis'], float); lo_l, hi_l = j['range']
            dot = float(a_l @ np.asarray(a_sim))
            if abs(abs(dot) - 1.0) > tol:
                raise ConventionError(f'{name}: live axis {a_l.tolist()} is neither +-{list(a_sim)}')
            same = abs(lo_l - lo_s) < tol and abs(hi_l - hi_s) < tol
            neg = abs(lo_l + hi_s) < tol and abs(hi_l + lo_s) < tol
            if name in force:
                mode = force[name]
                if mode not in cls.MODES:
                    raise ConventionError(f'{name}: forced mode {mode} not in {cls.MODES}')
            elif dot > 0 and same:
                mode = 'id'
            elif dot < 0 and neg:
                mode = 'neg'
            elif dot > 0 and math.isfinite(lo_l) and math.isfinite(hi_l) and hi_l - lo_l < TWO_PI + tol:
                mode = 'wrap'                  # same physical axis: angles equal mod 2 pi, only the legal window differs
            else:
                raise ConventionError(f'{name}: live axis {a_l.tolist()} limits ({lo_l:.4f}, {hi_l:.4f}) fit no convention '
                                      f'(sim axis {list(a_sim)} limits ({lo_s:.4f}, {hi_s:.4f}))')
            modes[i], axis[i] = mode, np.sign(dot)
            ranges[i] = (lo_l, hi_l)
            rep[name] = {'axis': a_l.tolist(), 'limits': [lo_l, hi_l], 'mode': mode, 'forced': name in force}
            if mode == 'wrap':
                rep[name]['note'] = 'same axis, other limits: angles mapped mod 2 pi into the live window'
            if mode == 'neg' and dot > 0:
                rep[name]['warning'] = 'negation forced on a same-axis URDF: verify the first move on the robot'
        return cls(modes, ranges, axis, rep)

    def _live_win(self, i):
        lo, hi = self.live_ranges[i]
        return lo + LIVE_MARGIN, hi - LIVE_MARGIN

    def to_sim(self, q_live):
        q = np.array(q_live, float)
        for i, m in enumerate(self.modes):
            if m == 'neg':
                q[i] = -q[i]
            elif m == 'wrap':                  # the representative inside (or nearest to) the training range
                lo, hi = RANGE8[i]
                c = [q[i] + TWO_PI * k for k in (-2, -1, 0, 1, 2)]
                q[i] = min(c, key=lambda v: max(lo - v, v - hi, 0.0))
        return q

    def vel_to_sim(self, v_live):
        v = np.array(v_live, float)
        for i, m in enumerate(self.modes):
            if m == 'neg':
                v[i] = -v[i]
        return v

    def to_live(self, q_sim, name=''):
        q = np.array(q_sim, float)
        for i, m in enumerate(self.modes):
            if m == 'neg':
                q[i] = -q[i]
            elif m == 'wrap':
                lo, hi = self._live_win(i)
                c = [q[i] + TWO_PI * k for k in (-2, -1, 0, 1, 2) if lo <= q[i] + TWO_PI * k <= hi]
                if not c:
                    raise ConventionError(f'{name or "posture"}: {JOINTS8[i]} sim {q[i]:.3f} has no live equivalent in '
                                          f'[{lo:.3f}, {hi:.3f}] (mod 2 pi)')
                q[i] = c[0]
        return q

    def fk_q(self, q_sim):
        """Joint values to feed the LIVE URDF's FK so it reproduces the sim URDF's FK of q_sim (the pose the policy
        was trained on): the same rotation about the same physical axis."""
        q = np.array(q_sim, float)
        for i, m in enumerate(self.modes):
            if m == 'neg':
                q[i] = q[i] * self.axis[i]
        return dict(zip(JOINTS8, q))

    def live_range_sim(self, fk=None, around=None):
        """Live legal range (URDF limits less LIVE_MARGIN, torso 0.055 .. 0.295) in sim coordinates. For wrap joints
        the interval that contains `around` (the current sim posture)."""
        lo, hi = np.full(8, -np.inf), np.full(8, np.inf)
        lo[0], hi[0] = TORSO_LIVE
        for i, n in enumerate(JOINTS8[1:], 1):
            if i in self.live_ranges:
                l, h = self._live_win(i)
            elif fk is not None and n in fk.joints:
                l, h = fk.joints[n]['range']; l, h = l + LIVE_MARGIN, h - LIVE_MARGIN
            else:
                l, h = RANGE8[i]; l, h = l + LIVE_MARGIN, h - LIVE_MARGIN
            m = self.modes[i]
            if m == 'neg':
                l, h = -h, -l
            elif m == 'wrap':
                if around is None:
                    raise ConventionError(f'{n}: wrap convention needs the current posture')
                x = float(around[i])
                ivs = [(l - TWO_PI * k, h - TWO_PI * k) for k in (-2, -1, 0, 1, 2)]
                hit = [iv for iv in ivs if iv[0] <= x <= iv[1]]
                if not hit:
                    raise ConventionError(f'{n}: sim posture {x:.3f} is outside every live window')
                l, h = hit[0]
            lo[i], hi[i] = l, h
        return lo, hi


# ---------------------------------------------------------------- gripper calibration
class GripCal:
    """Live gripper reading -> the sim finger value the policy was trained on (observation), piecewise linear over
    measured points. Default: erc-gym's shim map (live 0.0 open = sim 0.069, live 0.79 = sim 0.0). The replica's
    fingers stop at sim 0.030 on a 3 cm book; the real reading on a book is measured on the robot (lane F slot script)
    and loaded with grip_cal=PATH.json: {"live": [0.0, 0.46, 0.79], "sim": [0.069, 0.030, 0.0], "source": "..."}.
    Commands keep the policy's own semantics (closed fraction on the live scale, sim_to_real) unless the file has
    "cmd": true, then commands use the inverse of the same map."""

    def __init__(self, live=(0.0, REAL_CLOSED), sim=(SIM_OPEN, 0.0), cmd=False, source='erc-gym shim linear map'):
        o = np.argsort(live)
        self.live = np.asarray(live, float)[o]; self.sim = np.asarray(sim, float)[o]
        if np.any(np.diff(self.live) <= 0) or np.any(np.diff(self.sim) >= 0):
            raise ValueError(f'gripper calibration must be strictly monotonic: live {self.live} sim {self.sim}')
        self.cmd_inverse, self.source = bool(cmd), source

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(d['live'], d['sim'], d.get('cmd', False), d.get('source', path))

    def obs(self, q_live):
        return float(np.interp(q_live, self.live, self.sim))

    def obs_vel(self, q_live, v_live):
        k = int(np.clip(np.searchsorted(self.live, q_live) - 1, 0, len(self.live) - 2))
        slope = (self.sim[k + 1] - self.sim[k]) / (self.live[k + 1] - self.live[k])
        return float(slope * v_live)

    def cmd(self, g_sim):
        if self.cmd_inverse:
            return float(np.interp(g_sim, self.sim[::-1], self.live[::-1]))
        return sim_to_real(g_sim)

    def as_dict(self):
        return {'live': self.live.tolist(), 'sim': self.sim.tolist(), 'cmd': self.cmd_inverse, 'source': self.source}


# ---------------------------------------------------------------- action -> command
class ActionMapper:
    """PickEnv.step's command logic (lane A 18:51): arm and torso target increments a * cap * dt * action_scale; every
    new target within one step of the previous commanded target (a reversal never exceeds the cap) AND of the measured
    position (the checking layer's test); if those windows do not overlap, step from the previous target toward the
    measured position; then inside the command range (lo, hi from the checkpoint meta, torso 0.055 .. 0.295) and the
    live legal range. Gripper closed fraction (a + 1) / 2, rate limited to full travel in grip_close_time; base twist
    a * base_v in base_footprint. ctrl_dt and action_scale can be overridden for the robot (cfg); cmd_scale narrows the
    previous-target window below action_scale (None = lane A's rule exactly)."""

    def __init__(self, cfg=None, live_lo=None, live_hi=None, cmd_scale=None):
        cfg = dict(cfg or {})
        self.dt = float(cfg.get('ctrl_dt', 0.1))
        self.scale = float(cfg.get('action_scale', 0.95))
        self.grip_time = float(cfg.get('grip_close_time', 0.8))
        self.base_v = np.array(cfg.get('base_v', (0.10, 0.10, 0.20)), float)
        margin = float(cfg.get('margin', 0.07))
        self.lo0 = RANGE8[:, 0] + margin; self.hi0 = RANGE8[:, 1] - margin
        self.lo0[0], self.hi0[0] = 0.055, 0.295         # lane A 18:51 (was 0.005 .. 0.345)
        if cfg.get('lo') is not None:                    # the checkpoint's own range (export meta 'env')
            self.lo0, self.hi0 = np.array(cfg['lo'], float), np.array(cfg['hi'], float)
        self.lo, self.hi = self.lo0.copy(), self.hi0.copy()
        self.step_max = CAPS * self.dt * self.scale
        self.cmd_step = self.step_max if cmd_scale is None else np.minimum(self.step_max, CAPS * self.dt * float(cmd_scale))
        if live_lo is not None:
            self.set_live_range(live_lo, live_hi)
        self.reset(np.zeros(8), SIM_OPEN)

    def set_live_range(self, live_lo, live_hi):
        self.lo = np.maximum(self.lo0, live_lo); self.hi = np.minimum(self.hi0, live_hi)
        if np.any(self.lo > self.hi):
            raise ConventionError(f'empty joint range after the live limits: lo {self.lo.round(3)} hi {self.hi.round(3)}')

    def reset(self, q8, finger_target):
        self.target = np.asarray(q8, float).copy()
        self.finger_target = float(finger_target)
        self.last_act = np.zeros(12)

    def step(self, action, q8_meas):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        q8_meas = np.asarray(q8_meas, float)
        s_, c_, prev = self.step_max, self.cmd_step, self.target
        lo_w = np.maximum(q8_meas - s_, prev - c_); hi_w = np.minimum(q8_meas + s_, prev + c_)
        tgt = np.clip(prev + a[:8] * s_, lo_w, hi_w)
        tgt = np.where(lo_w > hi_w, np.clip(q8_meas, prev - c_, prev + c_), tgt)
        tgt = np.clip(tgt, self.lo, self.hi)
        g_cmd = SIM_OPEN * (1.0 - (a[8] + 1.0) / 2.0)
        g_step = SIM_OPEN / self.grip_time * self.dt
        g_tgt = float(np.clip(g_cmd, self.finger_target - g_step, self.finger_target + g_step))
        twist = a[9:12] * self.base_v
        self.target, self.finger_target, self.last_act = tgt, g_tgt, a
        return tgt, g_tgt, twist


# ---------------------------------------------------------------- scripted expert (residual policies)
EXPERT_REACH = np.array([.55, .72, .75, .85])


def _rot_err(Rc):
    I = np.eye(3)
    return 0.5 * (np.cross(Rc[:, 0], I[:, 0]) + np.cross(Rc[:, 1], I[:, 1]) + np.cross(Rc[:, 2], I[:, 2]))


def numeric_jacobian(fk_fn, q8, h=1e-6):
    """(6, 7) Jacobian of the grasping link in base_footprint w.r.t. the 7 arm joints (SIM coordinates): fk_fn(q8) ->
    4x4 pose in base. Linear rows d(position), angular rows the rotation vector of dR R^T."""
    T0 = fk_fn(q8)
    J = np.zeros((6, 7))
    for i in range(7):
        q = np.array(q8, float); q[i + 1] += h
        T = fk_fn(q)
        J[:3, i] = (T[:3, 3] - T0[:3, 3]) / h
        dR = T[:3, :3] @ T0[:3, :3].T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * h)
    return J, T0


def expert_action(spine_b, yaw, row, g_b, Rg_b, J, step_max, phase, t, reach=EXPERT_REACH, arm_y=ARM_Y):
    """erc_learn/expert.py expert_action (lane A 20:45) in numpy. spine_b: book spine centre in base_footprint (the
    estimate through odom for 'est', the truth for 'true'); yaw: base yaw relative to the shelf normal; row 0..3;
    g_b, Rg_b: grasping link in base_footprint; J: (6, 7) arm Jacobian in base_footprint; step_max: (8,) joint step
    caps. Returns (action (12,), (phase, t) after the update)."""
    ph = int(phase)
    P_pre = spine_b + np.array([-0.09, 0.0, 0.0])
    P_ins = spine_b + np.array([0.10, 0.0, 0.0])
    P_lift = P_ins + np.array([0.0, 0.0, 0.035])
    P_back = P_ins + np.array([-0.20, 0.0, 0.05])
    target = [P_pre, P_ins, P_ins, P_lift, P_back][min(ph, 4)]
    vmax = 0.03 if ph == 0 else 0.008 if ph == 1 else 0.012 if ph == 4 else 0.01
    e = target - g_b
    n = float(np.linalg.norm(e))
    step = e * min(1.0, vmax / (n + 1e-9))
    w = np.clip(_rot_err(Rg_b), -0.1, 0.1)
    lam = 0.05
    dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), np.concatenate([step, w]))
    a_arm = np.clip(dq / np.asarray(step_max)[1:], -1.0, 1.0)
    if ph == 4 and t > 20:
        a_arm = np.zeros(7)
    eb = spine_b[:2] - np.array([reach[row], arm_y])
    v = np.clip(0.8 * eb, -0.1, 0.1) / 0.1
    wz = float(np.clip(-1.2 * yaw, -0.2, 0.2) / 0.2)
    if ph == 0:
        a_base = np.array([v[0], v[1], wz])
    elif ph == 4:
        a_base = (np.array([-1.0, 0.0, 0.0]) if t > 20 else np.zeros(3)) * float(t < 49)
    else:
        a_base = np.zeros(3)
    a_grip = 1.0 if ph >= 2 else -1.0
    act = np.concatenate([np.zeros(1), a_arm, [a_grip], a_base])
    near = n < (0.015 if ph == 0 else 0.01)
    base_ok = (np.linalg.norm(eb) < 0.02) and (abs(yaw) < 0.03)
    adv = ((near and base_ok) or t > 60) if ph == 0 else (near or t > 40) if ph == 1 else (t > 12) if ph in (2, 3) else False
    return act, ((min(ph + 1, 4), 0.0) if adv else (ph, t + 1.0))


# ---------------------------------------------------------------- observations
def _common(q8, fing, v8, fingv, mapper, Rb, grip_b, R_bg):
    return [np.asarray(q8, float), [fing], np.asarray(v8, float) * 0.5, [fingv * 10.0],
            mapper.target - q8, [mapper.finger_target * 10.0], mapper.last_act, grip_b, R_bg[:, :2].reshape(-1)]


def pick_student(q8, fing, v8, fingv, mapper, T_bg, base_R, base_p, est_w, front, plate_y, board, yaw, wall_side, step, T,
                 fing_geom=None):
    """PickEnv._student_obs. base_R, base_p: base_footprint in W; T_bg: grasping link in base; est_w: spine-face centre
    estimate in W; front: shelf front x in W; plate_y: digit-plate centre y in W (bay walls assumed at +-0.5); board:
    board top z of the row; yaw: base yaw relative to the shelf normal. fing_geom: finger value for the pad geometry
    (lane A uses the true finger, the robot only has its reading: default = fing)."""
    Rb, pb = base_R, base_p
    tob = lambda v: Rb.T @ (np.asarray(v, float) - pb)
    grip_b = T_bg[:3, :3] @ pad_mid_in_grasp(fing if fing_geom is None else fing_geom) + T_bg[:3, 3]
    pts = [[front, est_w[1], board], [front, plate_y + 0.5, board], [front, plate_y - 0.5, board]]
    sp_b = np.concatenate([tob(p) for p in pts])
    e_b = tob(est_w)
    return np.concatenate(_common(q8, fing, v8, fingv, mapper, Rb, grip_b, T_bg[:3, :3]) + [
        e_b, e_b + np.array([0.03, 0.0, 0.0]) - grip_b, sp_b, [math.sin(yaw), math.cos(yaw)], [wall_side, step / T]])


def book_tilt(R):
    return math.acos(float(np.clip(-R[2, 0], -1.0, 1.0)))


def pick_teacher(q8, fing, v8, fingv, mapper, T_bg, base_R, base_p, book_p, book_R, book_v, book_p0, nb, bay, row,
                 yaw, wall_side, contacts, step, T):
    """PickEnv._obs (privileged actor). Everything in W from ground truth. contacts: padL padR inL inR shelf other
    arm_book (0/1). nb: largest displacement of the column's other books from their start (m)."""
    Rb, pb = base_R, base_p
    tob = lambda v: Rb.T @ (np.asarray(v, float) - pb)
    grip_b = T_bg[:3, :3] @ pad_mid_in_grasp(fing) + T_bg[:3, 3]
    grip_w = Rb @ grip_b + pb
    G = book_p + book_R @ np.array([0.0, 0.0, -BOOK_D / 2 + 0.03])
    board = BOARD_TOP[row - 1]
    pts = [[SHELF_FRONT, book_p[1], board], [SHELF_FRONT, bay + 0.5, board], [SHELF_FRONT, bay - 0.5, board]]
    sp_b = np.concatenate([tob(p) for p in pts])
    return np.concatenate(_common(q8, fing, v8, fingv, mapper, Rb, grip_b, T_bg[:3, :3]) + [
        tob(book_p), rot6(Rb, book_R), Rb.T @ book_v, Rb.T @ (G - grip_w), (book_p - book_p0) * 10.0,
        sp_b, [math.sin(yaw), math.cos(yaw)], [wall_side, book_tilt(book_R), nb * 10.0, step / T], contacts])


def place_student(q8, fing, v8, fingv, mapper, T_bg, base_R, base_p, bin_w, yaw, step, T, fing_geom=None):
    """PlaceEnv._obs student (lane A 18:51): bin centre estimate (xy) at the nominal floor height, base pose from odom,
    yaw relative to facing the bin across the arena (world yaw - pi; the shelf normal is world +x)."""
    Rb, pb = base_R, base_p
    grip_b = T_bg[:3, :3] @ pad_mid_in_grasp(fing if fing_geom is None else fing_geom) + T_bg[:3, 3]
    c = np.array([bin_w[0], bin_w[1], BIN_FLOOR])
    return np.concatenate(_common(q8, fing, v8, fingv, mapper, Rb, grip_b, T_bg[:3, :3]) + [
        [step / T], Rb.T @ (c - pb), [math.sin(yaw - math.pi), math.cos(yaw - math.pi)]])


def place_teacher(q8, fing, v8, fingv, mapper, T_bg, base_R, base_p, book_p, book_R, book_v, bin_c, bin_ax, contacts, step, T):
    """PlaceEnv._obs privileged. bin_c: interior centre (W), bin_ax: long axis (W). contacts: padL padR inL inR
    book_bin robot_bin robot_table first_contact gentle."""
    Rb, pb = base_R, base_p
    tob = lambda v: Rb.T @ (np.asarray(v, float) - pb)
    grip_b = T_bg[:3, :3] @ pad_mid_in_grasp(fing) + T_bg[:3, 3]
    return np.concatenate(_common(q8, fing, v8, fingv, mapper, Rb, grip_b, T_bg[:3, :3]) + [
        [step / T], tob(book_p), rot6(Rb, book_R), Rb.T @ book_v, tob(bin_c), Rb.T @ bin_ax,
        [book_p[2] - BOOK_H / 2 - BIN_FLOOR], contacts])


# Named slices for parity reports
COMMON = [('q8', 8), ('fing', 1), ('v8', 8), ('fingv', 1), ('target-q8', 8), ('finger_target', 1), ('last_act', 12),
          ('grip_b', 3), ('rot6_grasp', 6)]
LAYOUT = {
    'pick_student': COMMON + [('est_b', 3), ('est-grip', 3), ('shelf_pts', 9), ('yaw', 2), ('wall_side', 1), ('step', 1)],
    'pick_teacher': COMMON + [('book_b', 3), ('rot6_book', 6), ('book_v', 3), ('G-grip', 3), ('book_disp', 3),
                              ('shelf_pts', 9), ('yaw', 2), ('wall_side', 1), ('tilt', 1), ('nb', 1), ('step', 1),
                              ('contacts', 7)],
    'place_student': COMMON + [('step', 1), ('bin_b', 3), ('yaw', 2)],
    'place_teacher': COMMON + [('step', 1), ('book_b', 3), ('rot6_book', 6), ('book_v', 3), ('bin_b', 3), ('bin_ax', 3),
                               ('book_h', 1), ('contacts', 9)],
}
SIZES = {k: sum(n for _, n in v) for k, v in LAYOUT.items()}     # 67, 88, 54, 77


def residual_of(policy):
    return bool(policy.meta.get('config', {}).get('residual'))


def kind_of(policy):
    """'pick_student' etc. from the sidecar meta, else from the observation size."""
    m = policy.meta
    if m.get('task') and 'student' in m:
        return f"{m['task']}_{'student' if m['student'] else 'teacher'}"
    for k, n in SIZES.items():
        if n == policy.obs_size or n + 12 == policy.obs_size:
            return k
    raise ValueError(f'unknown observation size {policy.obs_size}')


def slices(kind, residual=False):
    out, i = {}, 0
    for name, n in LAYOUT[kind] + ([('x_act', 12)] if residual else []):
        out[name] = slice(i, i + n); i += n
    return out

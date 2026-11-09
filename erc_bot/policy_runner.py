"""Learned pick and place stages (lane A's pi-pick / pi-place) on the live interface.

  python3 -m erc_bot.mission --digit 2 --label red \
      --pick erc_bot.policy_runner:LearnedPick --pick-opt checkpoint=/opt/policies/pick_s.npz \
      --place erc_bot.policy_runner:LearnedPlace --place-opt checkpoint=/opt/policies/place_s.npz

The policy is the numpy export (erc_bot/policy_export.py: weights .npz plus a .json sidecar), no JAX. Every control
period (ctrl_dt of the checkpoint, 0.1 s) the runner reads /joint_states and odometry, builds the student observation
(erc_bot/policy_obs.py, checked against lane A's own code), evaluates the policy and streams the command:
  arm_right_controller and torso_controller   one-point JointTrajectory on /erc/<controller>/joint_trajectory,
                                              time_from_start = ctrl_dt, each joint within ctrl_dt * cap * 0.95 of its
                                              measured position AND of the previous commanded target (lane A's rule,
                                              cmd_scale narrows it), inside the live range less 0.055 (torso 0.055 ..
                                              0.295), in the robot's joint convention
  gripper_right_controller                    the closed fraction on the live scale (0.0 open .. 0.79 closed);
                                              the observation reads it back through the gripper calibration
  base                                        Live.velocity (laser stop) with the policy's twist
Self-collision is modelled nowhere (not in lane A's replica, not in the erc-gym shim), so any /erc/rejections
message or any goal ending with error -1 while the policy runs stops it (PolicyFallback): the stage backs off and
runs the scripted stage instead (fallback=abort reports the failure instead). Stale joint states hold the base and
fall back after `stale_abort` s. An unreachable start posture under the robot's joint convention falls back too.

Joint convention: lane A trains in the sim URDF's coordinates. At start-up the runner reads arm_right_1 and 5 (axis
and limits) from the live /robot_description and maps every observation and command (policy_obs.Convention:
id, neg, or wrap mod 2 pi; ConventionError = no policy on this robot, scripted stage).

Stage options (--pick-opt / --place-opt key=value):
  checkpoint=PATH.npz   required
  grip_cal=PATH.json    gripper calibration, live reading -> sim finger value (policy_obs.GripCal), default the erc-gym
                        shim's linear map; on the robot load lane F's measurement
  ctrl_dt=0.1           control period override (default: the checkpoint's)
  action_scale=0.95     joint step override (default: the checkpoint's)
  cmd_scale=X           narrow the previous-target window to X * cap * dt (default: lane A's rule, action_scale)
  convention=J:MODE,..  force the joint convention (arm_right_1_joint:neg ...), default from the URDF
  fallback=scripted     or abort
  on_reject=fallback    or reanchor (simulation experiments only)
  truth=gz              SIMULATION ONLY: teacher checkpoints, privileged inputs from gz ground truth (policy_truth.py)
  mode=topic|action     command transport (default topic)
  steps=N               control periods (default: the checkpoint's episode length)
  log=PATH              per-period record (JSONL); default <mission --out>/policy_<stage>.jsonl
  actions=parity        SIMULATION/TEST: a fixed deterministic action sequence instead of the policy (parity probe)
  wall=auto|-1|0|1      pick: the book's wall side (auto: from the book estimate and the digit plate)
  clearance=0.05        laser stop distance for the policy's base twist
  expert_mode=est|true  residual checkpoints: the expert's book pose (est: the detection through odom, deployable;
                        true: gz truth, simulation only). Default: the checkpoint's
  expert_only=1         residual checkpoints: run lane A's scripted expert alone (policy output zeroed)
  tfs=0.2               time_from_start of the streamed arm and torso points (default ctrl_dt). With lead=1 the point
  lead=1                is placed (tfs / ctrl_dt) times the policy's step ahead of the measured joints, within
                        0.95 x cap x tfs (lane F's measured robot lag: 0.1 s points reach 0.58-0.60 of the speed)
"""
import json, math, os, time
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .live import Abort, ARM_RIGHT, clamp
from .stages import Stage
from . import policy_obs as P
from .policy_fk import UrdfFK


def _dur(s):
    return Duration(sec=int(s), nanosec=int(round((s % 1) * 1e9)))


def parity_actions(k):
    """Deterministic test sequence: arm and torso sinusoids (0.5 amplitude, 2 s period, staggered phases), gripper
    closes over steps 20..35 and opens from 45, base: back, left, right, turn in small pulses."""
    a = np.zeros(12)
    a[:8] = 0.5 * np.sin(2 * math.pi * k / 20.0 + np.arange(8) * 0.7)
    a[8] = 1.0 if 20 <= k < 45 else -1.0
    a[9] = -0.5 if 5 <= k < 15 else 0.0
    a[10] = 0.4 if 15 <= k < 25 else (-0.4 if 25 <= k < 35 else 0.0)
    a[11] = 0.4 if 35 <= k < 45 else 0.0
    return a


class PolicyFallback(Abort):
    """The policy stopped itself (a rejection, a failed goal, an unreachable posture): the stage hands over to the
    scripted stage."""


class PolicyRunner:
    """Policy + observation + streaming on one Live node. opts: the stage's key=value options (see module doc)."""

    def __init__(self, stage, checkpoint, opts=None, log=''):
        o = dict(opts or {})
        self.stage, self.bot, self.r = stage, stage.bot, stage.r
        self.pol = P.NumpyPolicy(checkpoint)
        self.kind = P.kind_of(self.pol)
        cfg = dict(self.pol.meta.get('config', {}))
        self.train_cfg = dict(cfg)
        self.envc = self.pol.meta.get('env', {})          # lane A's start postures and command range as trained
        if self.pol.meta.get('env_stale'):
            self.r.get_logger().warn(f'{os.path.basename(checkpoint)}: lane A changed {self.pol.meta["env_stale"]} after this run '
                                     f'started; the start posture may not be the trained one')
        if 'lo' in self.envc:
            cfg['lo'], cfg['hi'] = self.envc['lo'], self.envc['hi']
        for k in ('ctrl_dt', 'action_scale'):          # overridable for the robot (Friday): changes the policy's timing
            if k in o:
                cfg[k] = float(o[k])
        self.dt = float(cfg.get('ctrl_dt', 0.1))
        self.T = int(cfg.get('episode_length', 250 if self.kind.startswith('pick') else 150))
        self.student = self.kind.endswith('student')
        self.residual = bool(cfg.get('residual'))                 # action = expert + residual_scale x policy
        self.residual_scale = float(cfg.get('residual_scale', 0.3))
        self.expert_mode = o.get('expert_mode', cfg.get('expert_mode', 'est'))
        self.expert_only = o.get('expert_only') == '1'               # residual checkpoints: the scripted expert alone
        truth = o.get('truth', '')
        expert_alone = o.get('expert_only') == '1' and o.get('expert_mode', cfg.get('expert_mode')) == 'est'
        if not self.student and truth != 'gz' and not expert_alone:
            raise Abort(f'{os.path.basename(checkpoint)} is a teacher (privileged inputs): it runs only in simulation with truth=gz')
        self.truth = None
        if truth == 'gz':
            from .policy_truth import GzTruth
            self.truth = GzTruth()
            if not self.truth.wait():
                raise Abort('gz ground truth not available (truth=gz runs only in simulation)')
        self.fk = UrdfFK(self.r.urdf)
        force = dict(x.split(':') for x in o['convention'].split(',')) if o.get('convention') else None
        self.conv = P.Convention.from_fk(self.fk, force=force)        # ConventionError: no policy on this robot
        for n, v in self.conv.report.items():
            if 'warning' in v:
                self.r.get_logger().warn(f'joint convention {n}: {v["warning"]}')
        self.grip = P.GripCal.load(o['grip_cal']) if o.get('grip_cal') else P.GripCal()
        cmd_scale = o.get('cmd_scale')
        self.mapper = P.ActionMapper(cfg, cmd_scale=None if cmd_scale is None else float(cmd_scale))
        self.mode, self.actions = o.get('mode', 'topic'), o.get('actions', '')
        self.clearance = float(o.get('clearance', 0.05))
        self.on_reject = o.get('on_reject', 'fallback')     # fallback (any /erc/rejections message ends the policy) | reanchor
        self.stale_abort = float(o.get('stale_abort', 3.0))
        self.tfs = float(o.get('tfs', self.dt))            # time_from_start of the streamed points
        self.lead = o.get('lead') == '1'                    # target led tfs ahead of the measured joints (lane F's rate law)
        self.js = {'pos': {}, 'vel': {}, 't': -1.0}
        self.js_log = [] if (self.actions or o.get('record_js') == '1') else None
        self.r.create_subscription(JointState, '/joint_states', self._on_js, qos_profile_sensor_data)
        self.pub = {c: self.r.create_publisher(JointTrajectory, f'/erc/{c}/joint_trajectory', 10)
                    for c in ('arm_right_controller', 'torso_controller', 'gripper_right_controller')}
        self.goal_results = []
        self.log_path = log
        self.log_f = open(log, 'a') if log else None
        self.n_rej_seen = len(self.r.rejections)
        self.stats = {'periods': 0, 'overruns': 0, 'rejections': 0, 'stale_periods': 0, 'base_blocked': 0, 'max_lag': 0.0}
        self._log({'event': 'start', 'kind': self.kind, 'checkpoint': checkpoint, 'meta': self.pol.meta, 'convention': self.conv.report,
                   'grip_cal': self.grip.as_dict(), 'cfg': cfg, 'cmd_scale': cmd_scale, 'opts': o,
                   'sim_only': bool(truth or self.actions)})

    # ---------------------------------------------------------------- state
    def _on_js(self, m):
        t = self.r.now()
        for i, n in enumerate(m.name):
            self.js['pos'][n] = m.position[i]
            if i < len(m.velocity):
                self.js['vel'][n] = m.velocity[i]
        self.js['t'] = t
        if self.js_log is not None and P.FINGER in self.js['pos']:
            self.js_log.append([round(t, 4)] + [self.js['pos'].get(n, float('nan')) for n in P.JOINTS8 + [P.FINGER]])

    def measured(self):
        """Sim-convention q8, v8, finger (sim scale, through the gripper calibration) and its velocity, joint state age."""
        p, v = self.js['pos'], self.js['vel']
        if not all(n in p for n in P.JOINTS8 + [P.FINGER]):
            return None
        q8 = self.conv.to_sim([p[n] for n in P.JOINTS8])
        v8 = self.conv.vel_to_sim([v.get(n, 0.0) for n in P.JOINTS8])
        fing = self.grip.obs(p[P.FINGER])
        fingv = self.grip.obs_vel(p[P.FINGER], v.get(P.FINGER, 0.0))
        return q8, v8, fing, fingv, self.r.now() - self.js['t']

    def T_bg(self, q8_sim):
        return self.fk.pose('gripper_right_grasping_link', self.conv.fk_q(q8_sim))

    # ---------------------------------------------------------------- commands
    def publish(self, tgt_sim, g_tgt, twist, q_meas=None):
        tfs = self.tfs
        if self.lead and q_meas is not None and tfs > self.dt:
            # lane F (robot, 22:42): the arm starts 33-54 ms late and lags 45-60 ms, so a 0.1 s point reaches only 0.58-0.60
            # of the commanded speed. Point the same velocity tfs ahead of the measured joints, within 0.95 cap x tfs.
            lim = P.CAPS * tfs * 0.95
            tgt_sim = q_meas + np.clip((np.asarray(tgt_sim) - q_meas) * (tfs / self.dt), -lim, lim)
            tgt_sim = np.clip(tgt_sim, self.mapper.lo, self.mapper.hi)
        try:
            ql = self.conv.to_live(tgt_sim, 'command')
        except P.ConventionError as e:
            raise PolicyFallback(f'policy: {e}')
        cmds = {'arm_right_controller': (ARM_RIGHT, list(ql[1:])), 'torso_controller': (['torso_lift_joint'], [ql[0]]),
                'gripper_right_controller': (['gripper_right_finger_joint'], [clamp('gripper_right_finger_joint', self.grip.cmd(g_tgt))])}
        for c, (names, pos) in cmds.items():
            pt = JointTrajectoryPoint(); pt.positions = [float(x) for x in pos]; pt.time_from_start = _dur(tfs if c != 'gripper_right_controller' else self.dt)
            if self.mode == 'action':
                g = FollowJointTrajectory.Goal(); g.trajectory.joint_names = list(names); g.trajectory.points = [pt]
                fut = self.r.actions[c].send_goal_async(g)
                fut.add_done_callback(lambda f, c=c: self._goal_sent(f, c))
            else:
                msg = JointTrajectory(); msg.joint_names = list(names); msg.points = [pt]
                self.pub[c].publish(msg)
        if bool(np.any(np.abs(twist) > 1e-4)):
            if not self.r.velocity(float(twist[0]), float(twist[1]), float(twist[2]), clearance=self.clearance):
                self.stats['base_blocked'] += 1
        else:
            self.r.stop()

    def _goal_sent(self, fut, c):
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.goal_results.append((c, None, 'goal not accepted')); return
        gh.get_result_async().add_done_callback(
            lambda rf: self.goal_results.append((c, rf.result().result.error_code, rf.result().result.error_string)))

    def hold(self):
        """Stop the base and hold the arm where it is (measured, so no speed test can fail)."""
        self.r.stop()
        m = self.measured()
        if m is not None:
            try:
                self.publish(m[0], self.mapper.finger_target, np.zeros(3))
            except PolicyFallback:
                pass

    def _check_rejections(self, q8):
        new = [t for _, t in self.r.rejections[self.n_rej_seen:]]
        self.n_rej_seen = len(self.r.rejections)
        failed = [f'{c} goal error {code}: {why}' for c, code, why in self.goal_results if code is not None and code == -1]
        failed += [f'{c}: {why}' for c, code, why in self.goal_results if code is None]
        self.goal_results = [x for x in self.goal_results if x[1] not in (-1, None)]
        hit = new + failed
        if hit:
            self.stats['rejections'] += len(hit)
            if self.on_reject == 'fallback':        # self-collision is modelled nowhere: never let the policy push on
                raise PolicyFallback(f'policy stopped on a checking-layer rejection: {hit[-1][:160]}')
            for text in new:
                for c, idx in (('arm_right_controller', slice(1, 8)), ('torso_controller', slice(0, 1))):
                    if text.startswith(c):
                        self.mapper.target[idx] = q8[idx]
        return hit

    def _log(self, rec):
        if self.log_f:
            self.log_f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
            self.log_f.flush()

    # ---------------------------------------------------------------- loop
    def run(self, obs_fn, stop_fn=None, steps=None, finger_target0=P.SIM_OPEN):
        """obs_fn(k, q8, v8, fing, fingv, T_bg) -> (obs, extra dict for the log). stop_fn(k, extra) -> True ends the
        loop. Returns (periods run, last extra). Raises PolicyFallback on a rejection or an unreachable command."""
        T = int(steps or self.T)
        m = None
        w0 = time.monotonic()
        while m is None:
            self.r.tick(); m = self.measured()
            if time.monotonic() - w0 > 10:
                raise Abort('policy: no joint states')
        try:
            lo, hi = self.conv.live_range_sim(self.fk, around=m[0])
            self.mapper.set_live_range(lo, hi)
        except P.ConventionError as e:
            raise PolicyFallback(f'policy: {e}')
        self.mapper.reset(m[0], finger_target0)
        self._log({'event': 'range', 'lo': self.mapper.lo, 'hi': self.mapper.hi})
        self.n_rej_seen = len(self.r.rejections)
        t_next = self.r.now()
        stale_since, extra, k = None, {}, 0
        try:
            for k in range(T):
                while self.r.now() < t_next:
                    self.r.tick(0.005)
                lag = self.r.now() - t_next
                self.stats['max_lag'] = max(self.stats['max_lag'], lag)
                if lag > self.dt:
                    self.stats['overruns'] += 1
                    t_next = self.r.now()
                t_next += self.dt
                q8, v8, fing, fingv, age = self.measured()
                if age > 0.5:
                    self.stats['stale_periods'] += 1
                    self.r.stop()
                    stale_since = stale_since or self.r.now()
                    if self.r.now() - stale_since > self.stale_abort:
                        raise PolicyFallback(f'policy: joint states stale for {self.stale_abort:.0f} s')
                    continue
                stale_since = None
                hit = self._check_rejections(q8)
                Tb = self.T_bg(q8)
                obs, extra = obs_fn(k, q8, v8, fing, fingv, Tb)
                if self.actions == 'parity':
                    a = parity_actions(k)
                elif self.residual and self.expert_only:     # the expert alone: no policy evaluation
                    a = np.clip(np.asarray(extra['x_act']), -1.0, 1.0)
                else:
                    a = self.pol.act(obs)
                    if self.residual:
                        a = np.clip(np.asarray(extra['x_act']) + self.residual_scale * a * (0.0 if self.expert_only else 1.0), -1.0, 1.0)
                tgt, g_tgt, tw = self.mapper.step(a, q8)
                self.publish(tgt, g_tgt, tw, q8)
                self.stats['periods'] += 1
                self._log({'k': k, 't': round(self.r.now(), 4), 'obs': obs, 'act': a, 'tgt': tgt, 'g_tgt': g_tgt, 'twist': tw,
                           'q8': q8, 'v8': v8, 'fing': fing, 'fingv': fingv, 'js_age': round(age, 4), 'odom': self.r.base_in_odom(),
                           'gripper_live': self.js['pos'].get(P.FINGER), 'rej': hit, **extra})
                if stop_fn is not None and stop_fn(k, extra):
                    break
        except PolicyFallback as e:
            self._log({'event': 'fallback', 'k': k, 'why': str(e)})
            raise
        finally:
            self.hold()
            self._log({'event': 'end', 'k': k, 'stats': self.stats})
        return k + 1, extra

    def close(self):
        if self.js_log is not None and self.log_path:
            json.dump(self.js_log, open(os.path.splitext(self.log_path)[0] + '_js.json', 'w'))
        if self.log_f:
            self.log_f.close(); self.log_f = None


class _PolicyStage(Stage):
    task_kind = 'pick'
    scripted = None

    def _runner(self):
        o = self.opts
        if 'checkpoint' not in o:
            raise Abort(f'{self.name}: --{self.task_kind}-opt checkpoint=PATH.npz is required')
        log = o.get('log') or (os.path.join(self.bot.out, f'policy_{self.task_kind}.jsonl') if getattr(self.bot, 'out', '') else '')
        try:
            run = PolicyRunner(self, o['checkpoint'], o, log)
        except P.ConventionError as e:
            raise PolicyFallback(f'joint convention: {e}')
        if not run.kind.startswith(self.task_kind):
            raise Abort(f'{self.name}: {o["checkpoint"]} is a {run.kind} policy')
        self.name = f"{self.base_name}-{'expert' if (run.residual and run.expert_only) else 'student' if run.student else 'teacher-gz-SIMONLY'}"
        return run

    def run(self, task):
        """The learned stage, and on PolicyFallback the scripted stage from a safe posture (opt fallback=abort to
        report the failure instead)."""
        try:
            return self.run_policy(task)
        except PolicyFallback as e:
            b = self.bot
            b.stage('POLICY_FALLBACK', why=str(e)[:200])
            if self.opts.get('fallback', 'scripted') != 'scripted' or self.opts.get('actions'):
                raise
            self.recover(task)
            from . import stages
            st = getattr(stages, self.scripted)(b)
            b.stage('SCRIPTED_' + self.task_kind.upper())
            res = st.run(task) or {}
            res['fallback'] = str(e)[:200]
            return res


class ShelfFrame:
    """W (lane A's replica world) from the shelf line in odom: x along the inward normal with the front plane at 2.755,
    y = left of the normal (toward column 1), origin y at the digit plate, z = odom z."""

    def __init__(self, n, front_s, y0):
        self.n = np.asarray(n[:2], float) / np.linalg.norm(n[:2]); self.l = np.array([-self.n[1], self.n[0]])
        self.theta = math.atan2(self.n[1], self.n[0])
        self.front_s, self.y0 = float(front_s), float(y0)

    def p(self, p_odom):
        p = np.asarray(p_odom, float)
        return np.array([p[:2] @ self.n - self.front_s + P.SHELF_FRONT, p[:2] @ self.l - self.y0, p[2] if len(p) > 2 else 0.0])

    def base(self, base_odom):
        x, y, yaw = base_odom
        pw = self.p([x, y, 0.0]); yw = yaw - self.theta
        return P.yaw_R(yw), pw, math.atan2(math.sin(yw), math.cos(yw))


class LearnedPick(_PolicyStage):
    """Scripted approach to lane A's start (base at the pre-grasp standoff, torso at the row height, arm at q_pre[row],
    gripper open), then the policy, then the scripted retreat and carry posture. Falls back to ScriptedPick."""
    name = base_name = 'rl-pick'
    task_kind = 'pick'
    scripted = 'ScriptedPick'

    def recover(self, task):
        """After a stopped policy: open the gripper, back the base straight out 0.25 m (arm held), so the scripted pick
        starts clear of the shelf."""
        b, r, n = self.bot, self.r, np.asarray(task.n, float)
        r.stop()
        try:
            r.gripper(False)
        except Abort:
            pass
        x, y, yaw = r.base_in_odom()
        try:
            b.go(x - .25 * n[0], y - .25 * n[1], yaw, tol=0.05, ytol=0.1, vmax=0.08, clearance=0.05, timeout=20)
        except Abort:
            pass

    def run_policy(self, task):
        b, r = self.bot, self.r
        run = self._runner()
        book, row, n = np.asarray(task.book, float), int(task.row), np.asarray(task.n, float)
        yaw = math.atan2(n[1], n[0]); left = np.array([-n[1], n[0]])
        ec = run.envc
        torso = float(np.clip(float(ec.get('torso', {}).get(str(row), P.TORSO[row])), *P.TORSO_LIVE))
        q_pre = ec.get('q_pre', {}).get(str(row), P.Q_PRE[row])
        reach = float(ec.get('reach', {}).get(str(row), P.REACH[row])); arm_y = float(ec.get('arm_y', P.ARM_Y))
        # lane B's ScriptedPick.prepare: clear point 1.40 m back, torso, gripper open, frontal re-look of the book (the
        # same detection the scripted pick uses); then lane A's start torso
        from .stages import ScriptedPick
        sp = ScriptedPick(b)
        if getattr(b, 'eyes', None) is not None:
            book = np.asarray(sp.prepare(task), float)
            task.book = book
        else:
            r.torso(torso, wait=False)
            clear = book[:2] - 1.40 * n - arm_y * left
            b.go(clear[0], clear[1], yaw)
            r.gripper(False)
        r.torso(torso)
        b.stage('POLICY_DEPLOY_ARM', row=row)
        try:
            q_start = run.conv.to_live(np.r_[torso, q_pre], f'q_pre row {row}')
        except P.ConventionError as e:
            raise PolicyFallback(str(e))
        r.send('arm_right_controller', list(q_start[1:]), tolerance=0.03)
        b.stage('POLICY_APPROACH')
        app = book[:2] - reach * n - arm_y * left
        b.go(app[0], app[1], yaw, tol=0.015, ytol=0.015, vmax=0.12, clearance=0.05)
        # ---- frames ----
        plate = None if task.plate is None else np.asarray(task.plate, float)
        front_s = float(plate[:2] @ n) if plate is not None else float(book[:2] @ n) - (2.82 - P.SHELF_FRONT)
        W = ShelfFrame(n, front_s, float((plate if plate is not None else book)[:2] @ left))
        est_w = W.p(book)
        plate_y = 0.0
        o = self.opts.get('wall', 'auto')
        off = est_w[1] - plate_y
        wall_side = float(o) if o != 'auto' else (float(np.sign(off)) if abs(off) > 0.38 else 0.0)
        board = float(P.BOARD_TOP[row - 1])
        tr = run.truth
        ctx = {}
        if tr is not None:                                   # SIM ONLY: privileged inputs from gz
            wTo = tr.world_T_odom(r.odom_T_base())
            bw = (wTo @ np.r_[book, 1.0])[:3] + np.array([P.BOOK_D / 2, 0, 0])
            target = tr.nearest_book(bw)
            col = int(target.split('_')[2])
            others = [x for x in tr.books() if x.startswith(f'book_col_{col}_') and x != target]
            p0 = {x: tr.pose(x)[1] for x in [target] + others}
            _, tp, _ = tr.pose(target)
            bay = float(P.BAY_Y[int(np.argmin(np.abs(P.BAY_Y - tp[1])))])
            ctx = {'target': target, 'others': others, 'p0': p0, 'bay': bay,
                   'wall_side_t': float(np.sign(tp[1] - bay)) if abs(tp[1] - bay) > 0.4 else 0.0, 'prev': None}
        b.stage('POLICY_PICK', kind=run.kind, row=row, wall_side=wall_side, est_w=np.round(est_w, 3).tolist(),
                truth=ctx.get('target'))
        T = int(self.opts.get('steps', run.T))
        state = {'out_steps': 0, 'succ_steps': 0}

        xstate = {'phase': 0.0, 't': 0.0}

        def obs_fn(k, q8, v8, fing, fingv, Tb):
            obs, extra = obs_core(k, q8, v8, fing, fingv, Tb)
            if not run.residual:
                return obs, extra
            # lane A's scripted expert at this state (erc_learn/expert.py, numpy port checked in t_policy_offline.py)
            if run.expert_mode == 'true' and tr is not None:
                Rbx, pbx, yx = P.yaw_R(extra['truth_base_w'][2]), np.r_[extra['truth_base_w'][:2], 0.0], extra['truth_base_w'][2]
                spine = ctx['p0'][ctx['target']] - np.array([P.BOOK_D / 2, 0, 0])
            else:
                Rbx, pbx, yx = P.yaw_R(extra['base_w'][2]), np.r_[extra['base_w'][:2], 0.0], extra['base_w'][2]
                spine = est_w
            J, Tg = P.numeric_jacobian(run.T_bg, q8)
            x_act, (ph, t) = P.expert_action(Rbx.T @ (spine - pbx), yx, row - 1, Tg[:3, 3], Tg[:3, :3], J, run.mapper.step_max,
                                             xstate['phase'], xstate['t'])
            xstate['phase'], xstate['t'] = ph, t
            extra['x_act'], extra['x_phase'] = x_act, ph
            for key in ('obs_student', 'obs_teacher'):
                if key in extra:
                    extra[key] = np.concatenate([extra[key], x_act])
            return np.concatenate([obs, x_act]), extra

        def obs_core(k, q8, v8, fing, fingv, Tb):
            Rb, pb, yaw_w = W.base(r.base_in_odom())
            stud = P.pick_student(q8, fing, v8, fingv, run.mapper, Tb, Rb, pb, est_w, P.SHELF_FRONT, plate_y, board, yaw_w,
                                  wall_side, k, run.T)
            grip_w = Rb @ (Tb[:3, :3] @ P.pad_mid_in_grasp(fing) + Tb[:3, 3]) + pb
            extra = {'grip_w': grip_w, 'base_w': [pb[0], pb[1], yaw_w], 'obs_student': stud}
            if tr is None:
                return stud, extra
            Tr = tr.T(P_ROBOT); Rt, pt = Tr[:3, :3], Tr[:3, 3]
            yaw_t = math.atan2(Rt[1, 0], Rt[0, 0])
            Rbt, pbt = P.yaw_R(yaw_t), np.array([pt[0], pt[1], 0.0])
            BR, bp, _ = tr.pose(ctx['target'])
            tnow = tr.now()
            prev = ctx['prev']
            bv = (bp - prev[1]) / max(1e-3, tnow - prev[0]) if prev and tnow and tnow > prev[0] else np.zeros(3)
            ctx['prev'] = (tnow, bp)
            nb = max([float(np.linalg.norm(tr.pose(x)[1] - ctx['p0'][x])) for x in ctx['others']] or [0.0])
            cts = tr.pick_contacts(ctx['target'], set(ctx['others']))
            teach = P.pick_teacher(q8, fing, v8, fingv, run.mapper, Tb, Rbt, pbt, bp, BR, bv, ctx['p0'][ctx['target']], nb,
                                   ctx['bay'], row, yaw_t, ctx['wall_side_t'], cts, k, run.T)
            lift = bp[2] - ctx['p0'][ctx['target']][2]
            held = (cts[0] or cts[2]) and (cts[1] or cts[3])
            out = bp[0] + P.BOOK_D / 2 <= P.SHELF_FRONT
            extra.update({'obs_teacher': teach, 'truth': tr.snapshot([P_ROBOT, ctx['target']] + ctx['others']),
                          'truth_base_w': [pt[0], pt[1], yaw_t], 'contacts': cts, 'lift': lift, 'held': bool(held),
                          'out': bool(out), 'success': bool(held and lift >= 0.02 and out), 'nb': nb, 'book_v': bv})
            return (teach if not run.student else stud), extra

        def stop_fn(k, ex):
            closed = run.mapper.finger_target < 0.25 * P.SIM_OPEN
            if closed and state.get('tool_z_grasp') is None:
                Tb = run.T_bg(run.measured()[0])
                state['tool_z_grasp'] = float(Tb[2, 3])        # base_footprint z = odom z
            state['out_steps'] = state['out_steps'] + 1 if closed and ex['grip_w'][0] < P.SHELF_FRONT - 0.15 else 0
            state['succ_steps'] = state['succ_steps'] + 1 if ex.get('success') else 0
            return (self.opts.get('actions') != 'parity') and (state['out_steps'] >= 5 or state['succ_steps'] >= 5)

        try:
            steps, last = run.run(obs_fn, stop_fn, T, finger_target0=P.SIM_OPEN)
        finally:
            run.close()
        b.stage('POLICY_PICK_END', steps=steps, stats=run.stats, finger_target=round(run.mapper.finger_target, 4),
                gripper=round(float(r.joints.get('gripper_right_finger_joint', -1)), 3),
                truth_success=last.get('success'), truth_lift=None if 'lift' not in last else round(float(last['lift']), 4))
        if self.opts.get('actions') == 'parity':
            return {'held': False, 'parity': True}
        if run.mapper.finger_target > 0.5 * P.SIM_OPEN:
            raise PolicyFallback('policy pick ended with the gripper open (no grasp in the episode)')
        # ---- retreat (if the policy did not back off itself), then lane B's ScriptedPick.finish: hold check (gripper
        # position or sight), carry tail; book_below_tool from the measured grasp height ----
        x, y, _ = r.base_in_odom()
        out_by = float((book[:2] - np.array([x, y])) @ n) - reach          # how far the base already stands back
        if out_by < 0.25:
            b.go(x - (0.28 - max(0.0, out_by)) * n[0], y - (0.28 - max(0.0, out_by)) * n[1], yaw, clearance=0.05)
        try:
            res = sp.finish(task)
        except PolicyFallback:
            raise
        except Abort as e:
            # The policy closed on nothing: lane B's hold check aborts the mission here. Hand over to the scripted pick
            # instead (s4, held out and E-expert-s4b by 23:25: 20 learned picks hit this abort, 18 of them empty-handed
            # by gz truth at the policy's end, the book still in its slot). A book seen on the floor ends the run anyway.
            if 'no book in the gripper' not in str(e) or '(floor)' in str(e):
                raise
            raise PolicyFallback(f'policy pick ended without the book: {e}')
        if state.get('tool_z_grasp') is not None:
            res['book_below_tool'] = round(P.BOOK_H / 2 + float(state['tool_z_grasp'] - book[2]), 4)
        res.update(policy_steps=steps)
        return res


P_ROBOT = 'tiago_pro'


class LearnedPlace(_PolicyStage):
    """Scripted approach to lane A's place start (bin centre 0.70 m ahead at the right arm's offset, torso high, arm at
    q_hold over the bin), then the policy (lower, release), then open and lift clear. Falls back to ScriptedPlace (the
    book still held: no recovery move, the scripted place re-plans from where the arm is)."""
    name = base_name = 'rl-place'
    task_kind = 'place'
    scripted = 'ScriptedPlace'

    def recover(self, task):
        self.r.stop()

    def run_policy(self, task):
        b, r = self.bot, self.r
        run = self._runner()
        bin_p = np.asarray(task.bin, float)
        x, y, _ = r.base_in_odom()
        to = bin_p[:2] - np.array([x, y]); u = to / np.linalg.norm(to)
        if np.linalg.norm(to) > 1.5:                          # look again from 1.2 m, as ScriptedPlace
            near = bin_p[:2] - 1.2 * u
            b.go(near[0], near[1], math.atan2(u[1], u[0]), tol=0.05, ytol=0.05, clearance=0.15)
            try:
                b.look_at(b.in_base(bin_p)); r.wait(0.3)
                pts, floors, rims = b.look_bin(frames=5)
                if len(pts) >= 3:
                    bin_p = b.bin_summary(pts, floors, rims)
            except Exception as e:                            # keep the first estimate
                b.stage('BIN_RELOOK_FAILED', why=str(e)[:80])
            x, y, _ = r.base_in_odom()
            to = bin_p[:2] - np.array([x, y]); u = to / np.linalg.norm(to)
        yaw = math.atan2(u[1], u[0]); left = np.array([-u[1], u[0]])
        stand = bin_p[:2] - 0.70 * u - P.ARM_Y * left
        b.go(stand[0], stand[1], yaw, tol=0.03, ytol=0.03, vmax=0.15, clearance=0.08)
        torso_hold = float(np.clip(run.envc.get('torso_hold', P.TORSO_HOLD), *P.TORSO_LIVE))
        r.torso(torso_hold)
        b.stage('POLICY_HOLD_POSTURE')
        try:
            q_start = run.conv.to_live(np.r_[torso_hold, run.envc.get('q_hold', P.Q_HOLD)], 'q_hold')
        except P.ConventionError as e:
            raise PolicyFallback(str(e))
        r.send('arm_right_controller', list(q_start[1:]), tolerance=0.03)
        n = np.asarray(task.n if task.n is not None else -u, float)
        W = ShelfFrame(n, 0.0, 0.0)
        bin_w = W.p(bin_p)
        tr = run.truth
        ctx = {}
        if tr is not None:
            wTo = tr.world_T_odom(r.odom_T_base())
            # the held book: the book nearest the gripper
            tool_w = (wTo @ r.odom_T_base() @ b.arm.tool())[:3, 3]
            ctx = {'target': tr.nearest_book(tool_w), 'prev': None, 'first_contact': 0.0, 'gentle': 0.0, 'released': 0.0}
        b.stage('POLICY_PLACE', kind=run.kind, bin_w=np.round(bin_w, 3).tolist(), truth=ctx.get('target'))
        T = int(self.opts.get('steps', run.T))
        state = {'open_steps': 0}

        def obs_fn(k, q8, v8, fing, fingv, Tb):
            Rb, pb, yaw_w = W.base(r.base_in_odom())
            stud = P.place_student(q8, fing, v8, fingv, run.mapper, Tb, Rb, pb, bin_w, yaw_w, k, run.T)
            extra = {'base_w': [pb[0], pb[1], yaw_w], 'obs_student': stud}
            if tr is None:
                return stud, extra
            Tr = tr.T(P_ROBOT); yaw_t = math.atan2(Tr[1, 0], Tr[0, 0])
            Rbt, pbt = P.yaw_R(yaw_t), np.array([Tr[0, 3], Tr[1, 3], 0.0])
            BR, bp, _ = tr.pose(ctx['target'])
            tnow = tr.now(); prev = ctx['prev']
            bv = (bp - prev[1]) / max(1e-3, tnow - prev[0]) if prev and tnow and tnow > prev[0] else np.zeros(3)
            ctx['prev'] = (tnow, bp)
            Rbin, pbin, _ = tr.pose('erc_collection_bin')
            cbin = pbin + Rbin @ np.array([0, 0, -0.0073]); ax = -Rbin[:, 2]
            c = tr.place_contacts(ctx['target'])
            touch = any(c[:4])
            if c[4]:
                ctx['first_contact'] = 1.0
            d = bp[:2] - cbin[:2]; ax2 = ax[:2] / (np.linalg.norm(ax[:2]) + 1e-9)
            inside = abs(d @ ax2) < 0.245 - 0.03 and abs(d @ np.array([-ax2[1], ax2[0]])) < 0.145 - 0.03
            h = bp[2] - P.BOOK_H / 2 - P.BIN_FLOOR
            if ctx['released'] == 0 and not touch and fing > P.Q_CLOSED_ON_BOOK + 0.004:
                ctx['released'] = 1.0; ctx['gentle'] = float(inside and h < 0.02)
            teach = P.place_teacher(q8, fing, v8, fingv, run.mapper, Tb, Rbt, pbt, bp, BR, bv, cbin, ax,
                                    c + [ctx['first_contact'], ctx['gentle']], k, run.T)
            extra.update({'obs_teacher': teach, 'truth': tr.snapshot([P_ROBOT, ctx['target'], 'erc_collection_bin']),
                          'truth_base_w': [Tr[0, 3], Tr[1, 3], yaw_t], 'contacts': c, 'released': ctx['released'],
                          'gentle': ctx['gentle'], 'h': h, 'inside': bool(inside)})
            return (teach if not run.student else stud), extra

        def stop_fn(k, ex):
            state['open_steps'] = state['open_steps'] + 1 if run.mapper.finger_target > 0.9 * P.SIM_OPEN else 0
            return state['open_steps'] >= 15

        try:
            steps, last = run.run(obs_fn, stop_fn, T, finger_target0=0.0)
        finally:
            run.close()
        b.stage('POLICY_PLACE_END', steps=steps, stats=run.stats, released=last.get('released'), gentle=last.get('gentle'))
        r.gripper(False)
        r.wait(0.5)
        try:
            up = b.arm.tool(); up[2, 3] += 0.12
            b.arm.pose(up, duration=1.5, pos_tol=0.05, ang_tol=0.3)
        except Abort as e:
            b.stage('LIFT_CLEAR_FAILED', why=str(e)[:80])
        return {'policy_steps': steps}

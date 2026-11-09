"""Pluggable manipulation stages: the mission calls a pick stage and a place stage through this interface, so a learned
policy can replace either one and be scored by erc-gym on the same seeds as the scripted code.

Contract (see README, "Stage interface"):

  class MyPick(Stage):
      name = 'my-pick'
      def run(self, task: PickTask) -> dict:  ...   # book held and clear of the shelf on return, else raise Abort

  python3 -m erc_bot.mission --digit 2 --label red --pick my_pkg.my_module:MyPick --pick-opt checkpoint=/x.onnx

A stage gets the Bot (`self.bot`): `bot.r` (Live: sensors, TF, `send`, `gripper`, `head`, `torso`, `velocity`, `wait`),
`bot.arm` (IK, `pose`, `tool`), `bot.eyes` (detector in 3D), `bot.go` (laser-guarded holonomic go-to in odom) and
`bot.stage(name, **kw)` to log sub-stages. Every command still passes the organisers' checking layer.
"""
import importlib
import math
from dataclasses import dataclass, field
import numpy as np
from .live import Abort
from .arm import transform
from .kinematics import transport_path

ROW_Z = np.array([1.577, 1.247, .917, .587])   # spine-centre heights of rows 1-4 (Jalpari Phase 1, measured in sim)
TORSO = {1: .345, 2: .20, 3: .0, 4: .0}         # torso and reach per row: Jalpari Phase 1 grasp geometry
REACH = {1: .55, 2: .72, 3: .75, 4: .85}
ARM_Y = -0.24                                   # the right gripper works 0.24 m right of the base centre
SURE_HOLD = 0.75                                # live gripper position below this after a close: something is between the
                                                # fingers. The sim gripper closes to a 2.8 cm gap, a 3 cm book reads 0.79 like
                                                # air, so above it the hold is confirmed by sight after extraction.
BOOK_HALF_H = 0.125                             # books are 0.25 m tall
BOOK_BELOW_TOOL = 0.115                         # held upright, the book's bottom is this far under the grasping link (book 0.25 m tall,
                                                # centre 1 cm above the link, measured with gz in the dev sim)
GRASP_DZ = {1: -0.05, 2: 0.0, 3: 0.0, 4: 0.0}    # grasp this far below the spine centre: the top row is out of reach at the centre
                                                # with the live torso range (0.055 to 0.295 m as sent), 5 cm lower it is reachable
SPINE_BEHIND_PLATE = 0.065                      # spine faces 6.5 cm behind the digit-plate line along the shelf normal (gz layout:
                                                # plates flush with the shelf front at x 2.755, spines at 2.82)
INSERT_PAST = 0.07                              # grasping link this far past the spine face (opt insert_past). At 0.10 the palm
                                                # (gripper base, lumped into arm_right_7_link) hit the target book: 219 of 258
                                                # collisions in sweep bot-sweep1. Pads 3.5 to 7.7 cm behind the link.


@dataclass
class PickTask:
    """Everything the mission knows when the pick starts. All positions in odom (metres)."""
    target: str                 # detector class of the target book (book_red ... book_AGENT)
    book: np.ndarray            # spine-face centre of the target book (x, y, z)
    row: int                    # shelf row 1 (top) to 4
    n: np.ndarray               # shelf inward normal (xy, unit), from the robot side into the shelf
    t: np.ndarray               # shelf tangent (xy, unit)
    plate: np.ndarray           # digit plate of the column (x, y, z)


@dataclass
class PlaceTask:
    """The place starts with the book held, the robot facing the bin from about 2 to 3 m."""
    target: str
    bin: np.ndarray             # bin estimate (x, y, z) in odom, median of several detections
    n: np.ndarray               # shelf inward normal (the bin lies across the arena, along -n)
    extra: dict = field(default_factory=dict)


class Stage:
    """Base class. `opts` are the --pick-opt / --place-opt key=value strings."""
    name = 'stage'

    def __init__(self, bot, **opts):
        self.bot, self.r, self.opts = bot, bot.r, opts

    def run(self, task):
        raise NotImplementedError


def load(spec, bot, opts=()):
    """'pkg.module:Class' -> instance. opts: ['k=v', ...]."""
    mod, _, cls = spec.partition(':')
    kw = dict(o.split('=', 1) for o in opts)
    return getattr(importlib.import_module(mod), cls or 'Stage')(bot, **kw)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class ScriptedPick(Stage):
    """Jalpari Phase 1 grasp: clear space, deploy, approach, insert in steps, close, test lift, extract, then the torso
    starts up without waiting (opt carry=overlap, default) or the Phase 1 upright carry posture (carry=posture).

    Three overridable steps, so a variant (lane G's wall manoeuvre) replaces only the grasp:
      prepare(task) -> book      clear point 1.40 m back, torso to the row, gripper open, frontal re-look
      grasp(task, book) -> None  deploy, approach, insert, close, lift, extract, back off 0.28 m
      finish(task) -> dict       hold check (position or sight), carry tail; returns the PickTask result
    Geometry is per instance (set it after construction, never patch the module constants):
      self.torso, self.reach, self.grasp_dz  {row: value}; self.arm_y; opts insert_past, carry."""
    name = 'scripted'

    def __init__(self, bot, **opts):
        super().__init__(bot, **opts)
        self.torso, self.reach, self.grasp_dz, self.arm_y = dict(TORSO), dict(REACH), dict(GRASP_DZ), ARM_Y

    def run(self, task):
        book = self.prepare(task)
        self.grasp(task, book)
        return self.finish(task)

    def prepare(self, task):
        b, r = self.bot, self.r
        book, row, n = np.asarray(task.book, float), task.row, np.asarray(task.n, float)
        yaw = math.atan2(n[1], n[0]); left = np.array([-n[1], n[0]])
        r.torso(self.torso[row], wait=False)
        clear = book[:2] - 1.40 * n - self.arm_y * left
        b.go(clear[0], clear[1], yaw)
        r.torso(self.torso[row])
        r.gripper(False)
        return self.relook(task, book, n)

    def grasp(self, task, book):
        b, r, arm = self.bot, self.r, self.bot.arm
        row, n = task.row, np.asarray(task.n, float)
        yaw = math.atan2(n[1], n[0]); left = np.array([-n[1], n[0]])
        reach, arm_y = self.reach[row], self.arm_y
        b.stage('DEPLOY_ARM')
        book = np.r_[book[:2], book[2] + self.grasp_dz.get(row, 0.0)]
        arm.pose(transform([reach - .09, arm_y, book[2]]))
        b.stage('APPROACH_BOOK')
        app = book[:2] - reach * n - arm_y * left
        d0, p0 = (self.shelf_range_settled(n) if getattr(b, 'anchor', False) else None), r.base_in_odom()
        b.go(app[0], app[1], yaw, tol=0.015, ytol=0.015, vmax=0.12, clearance=0.05)
        book, app = self.anchor_approach(book, app, n, yaw, d0, p0)
        p = b.in_base(book)
        pre = transform(p + [-.09, 0., 0.])
        deep = 0.09 + float(self.opts.get('insert_past', INSERT_PAST))
        seed = arm.q(arm.k.names); T = r.matrix('torso_lift_link', 'base_footprint')
        for dx, dz in ((0., 0.), (deep, 0.), (deep, .025), (-.05, .025)):
            g = pre.copy(); g[0, 3] += dx; g[2, 3] += dz
            try:
                seed = arm.k.solve(T @ g, seed)           # a way back out must exist before going in
            except RuntimeError as e:
                raise Abort(f'IK: grasp path ({dx:.2f}, {dz:.3f}) unreachable: {e}')
        arm.pose(pre)
        b.stage('INSERT')
        for k, adv in enumerate(np.linspace(0, deep, 13)[1:]):
            g = pre.copy(); g[0, 3] += adv
            arm.pose(g, duration=0.7, pos_tol=0.012 if k == 11 else 0.02)
        b.stage('CLOSE')
        pos = r.gripper(True)
        b.stage('GRIPPER', position=round(pos, 3), sure=pos < SURE_HOLD)
        lifted = g.copy(); lifted[2, 3] += .025
        arm.pose(lifted, duration=1.5)
        r.wait(0.6)
        b.stage('EXTRACT')
        back = lifted.copy(); back[0, 3] -= .20
        arm.pose(back, duration=2.0)
        x, y, _ = r.base_in_odom()
        b.go(x - .28 * n[0], y - .28 * n[1], yaw, clearance=0.05)

    def finish(self, task):
        b, r, arm = self.bot, self.r, self.bot.arm
        if not (b.holding() or b.seen_in_hand()):
            # The gripper reads about 0.79 closed with or without a book, and the in-hand look misses held books on the
            # top row (held out G-hd1-default: 4 of 5 row-1 aborts had the book 0.42 m out with the gripper). Abort
            # only when the book is seen back in its slot or on the floor.
            where = self.slot_check(task)
            if where != 'unseen':
                raise Abort(f'no book in the gripper after extraction ({where})')
            b.stage('HOLD_ASSUMED')
        b.stage('HOLDING')
        if self.opts.get('carry', 'overlap') == 'posture':     # Jalpari Phase 1: torso up, then the upright transport path
            b.stage('CARRY_POSTURE')
            if arm.tool()[2, 3] < 1.20:
                r.torso(.345)
            tip = arm.tool()
            if tip[2, 3] < 1.20:
                arm.path(transport_path(tip[:3, 3]), duration=None)
        else:                                                    # the torso rises (23 s in sim) while the mission turns and
            b.stage('CARRY_OVERLAP')                             # looks for the bin and drives; the place waits for it
            r.torso(.345, wait=False)
        dzg = self.grasp_dz.get(task.row, 0.0)
        return {'row': task.row, 'held': True, 'book_below_tool': BOOK_BELOW_TOOL + dzg}

    def slot_check(self, task, frames=4):
        """Where the target is when the in-hand look failed: 'shelf' (seen within 15 cm of its slot, odom), 'floor', or
        'unseen' (not seen away from the gripper: taken as held). The held book is about 0.45 m in front of the slot."""
        b, r = self.bot, self.r
        slot = np.asarray(task.book, float)
        b.look_at(b.in_base(slot)); r.wait(0.3)
        tool = (r.odom_T_base() @ b.arm.tool())[:3, 3]
        where = 'unseen'
        for _ in range(frames):
            dets, _ = b.eyes.look([task.target], conf=0.3, kp=False)
            for d in dets or []:
                if np.linalg.norm(d['p'] - tool) < 0.20:
                    continue
                if np.linalg.norm(d['p'][:2] - slot[:2]) < 0.15 and abs(d['p'][2] - slot[2]) < 0.10:
                    where = 'shelf'
                elif d['p'][2] < 0.35 and where == 'unseen':
                    where = 'floor'
            r.wait(0.1)
        b.stage('SLOT_CHECK', where=where)
        return where

    def relook(self, task, book, n, frames=6):
        """Re-observe the target from the clear point, within about 10 deg of frontal (the camera is 0.24 m beside the
        book at 1.4 m). From 20 deg off-axis the box centre lands on the book's side face, 1.5 cm lateral and 4 to 6 cm
        too deep (lane D, unseen seeds), which pushes the insertion too deep. Depth along the normal is checked against
        the shelf face line from the plates (spines sit 6.5 cm behind the plate line): an outlier keeps the line value."""
        b, r = self.bot, self.r
        plate = self.replate(task)
        b.look_at(b.in_base(book)); r.wait(0.3)
        pts = []
        for _ in range(frames):
            dets, _ = b.eyes.look([task.target])
            pts += [d['p'] for d in dets or [] if np.linalg.norm(d['p'][:2] - book[:2]) < 0.15 and abs(d['p'][2] - book[2]) < 0.08]
            r.wait(0.1)
        new = np.median(np.array(pts), 0) if len(pts) >= 3 else np.asarray(book, float).copy()
        line = None
        if plate is not None and getattr(b, 'column_source', 'plate') == 'plate':   # an override plate's depth is the laser's
            line = float(np.asarray(plate)[:2] @ n) + SPINE_BEHIND_PLATE
            dn = float(new[:2] @ n) - line
            if 0.04 < abs(dn) < 0.10:
                new[:2] -= dn * n                           # depth outlier: keep the lateral and the height, depth from the line
        b.stage('BOOK_RELOOK', seen=len(pts), moved=np.round(new - book, 3).tolist(),
                depth_vs_line=None if line is None else round(float(new[:2] @ n) - line, 3))
        return new

    def shelf_range(self, n, half_width=0.5, near=0.2, far=3.0):
        """Distance to the shelf along its normal by laser: the densest 1 cm band of scan points ahead, within
        half_width of the base centre sideways (the back panel between the dividers, or whatever the laser sees there).
        Returns (distance, points in the band) or None."""
        pts = self.r.scan_points()
        if pts is None or not len(pts):
            return None
        yaw = self.r.base_in_odom()[2]
        nb = np.array([math.cos(yaw) * n[0] + math.sin(yaw) * n[1], -math.sin(yaw) * n[0] + math.cos(yaw) * n[1]])
        along = pts @ nb; side = pts @ np.array([-nb[1], nb[0]])
        a = along[(np.abs(side) < half_width) & (along > near) & (along < far)]
        if len(a) < 10:
            return None
        h, e = np.histogram(a, bins=np.arange(near, far + 0.01, 0.01))
        k = int(np.argmax(np.convolve(h, np.ones(3), 'same')))
        band = a[np.abs(a - (e[k] + 0.005)) < 0.015]
        return (float(np.median(band)), int(len(band))) if len(band) >= 10 else None

    def shelf_range_settled(self, n, scans=3):
        """shelf_range over a few scans (median), the base standing still."""
        got = []
        for k in range(scans):
            if k:
                self.r.wait(0.12)
            d = self.shelf_range(n)
            if d is not None:
                got.append(d)
        if not got:
            return None
        return float(np.median([g[0] for g in got])), int(min(g[1] for g in got))

    def anchor_approach(self, book, app, n, yaw, d0, p0):
        """The last leg to the book ran on odometry (lane F's 3 % proxy: 1.5 to 2.4 cm on 0.52 and 0.82 m legs, dev gz
        truth). The laser
        measures how far the base really came toward the shelf: the difference from the odometry's travel moves the
        book (and the stand-off point) along the normal in odom, then a short leg puts the base right. Skipped when the
        laser band is missing, thin, or disagrees with odometry by more than 6 cm (a different structure in the band)."""
        b, r = self.bot, self.r
        if not getattr(b, 'anchor', False) or not int(self.opts.get('anchor', 1)):
            return book, app
        r.wait(0.6)                                       # settled: the scan and the odometry describe the same pose
        d1, p1 = self.shelf_range_settled(n), r.base_in_odom()
        odo = float((p1[:2] - p0[:2]) @ n[:2])
        if d0 is None or d1 is None:
            b.stage('APPROACH_ANCHOR', applied=False, why='no laser band', d0=d0, d1=d1)
            return book, app
        las = d0[0] - d1[0]
        err = odo - las                                  # > 0: the base is short of where odometry puts it
        ok = abs(err) < 0.06
        b.stage('APPROACH_ANCHOR', applied=bool(ok and abs(err) > 0.003), odom_travel=round(odo, 4), laser_travel=round(las, 4),
                err=round(err, 4), d0=[round(d0[0], 3), d0[1]], d1=[round(d1[0], 3), d1[1]])
        if not ok or abs(err) <= 0.003:
            return book, app
        book = np.r_[book[:2] + err * n[:2], book[2]]       # in_base(book) is now right wherever the base stops
        app = app + err * n[:2]
        if abs(err) > 0.01:
            try:
                b.go(app[0], app[1], yaw, tol=0.01, ytol=0.015, vmax=0.06, timeout=15.0, clearance=0.05)
            except Abort as e:                            # the arm reaches the residual (IK is checked next)
                b.stage('APPROACH_ANCHOR_STEP', why=str(e)[:80])
        return book, app

    def replate(self, task, frames=3):
        """The target column's plate seen again from here, so the depth check of the re-look compares two sightings
        from the same pose. Odometry slips on every leg (lane F's 3 % proxy: a stale plate line overwrote a correct frontal
        depth in 4 of 8 held-out episodes). Returns the plate in odom (the earlier value, re-anchored,
        when it is not seen), or None without a plate."""
        b, r = self.bot, self.r
        if task.plate is None or not hasattr(b, 'shelf') or not getattr(b, 'anchor', False):
            return task.plate
        b.shelf.new_epoch()
        old = np.asarray(task.plate, float)
        b.look_at(b.in_base(old)); r.wait(0.3)
        b.look_plates(frames)
        fresh = b.shelf.fresh(min_obs=2)
        p = fresh.get(b.digit)
        if p is None or np.linalg.norm(p[:2] - old[:2]) > 0.25:
            b.stage('PLATE_RELOOK', seen=False)
            return old
        b.stage('PLATE_RELOOK', seen=True, moved=np.round(p - old, 3).tolist())
        return p


class ScriptedPlace(Stage):
    """Stop 1.2 m short of the bin and look again (rim centre, rim and floor height), stand so the bin is at the right
    gripper's reach, hold the book over the far half of the bin, lower it below the rim, open.

    opts: relook=1|0, stand=0.70 (m, base to bin centre), deep=0.12 (tool beyond the centre, m: the book over the far half,
    the wrist inside the near wall), release=floor (book bottom `gap` above the measured bin floor) or a tool height above
    the rim in m, gap=0.005 (a 0.5 cm fall is 0.31 m/s, the scorer's gentle limit is 0.5), finallook=0, clear=0.08 (elbow and upper arm kept this far above the rim while the gripper goes below it)."""
    name = 'scripted'

    def carry_posture(self):
        """Torso up (the pick may have started it) and the book upright above the rim height, before the final approach."""
        r, arm = self.r, self.bot.arm
        r.torso(.345)
        tip = arm.tool()
        if tip[2, 3] < 1.20:
            self.bot.stage('CARRY_POSTURE')
            arm.path(transport_path(tip[:3, 3]), duration=None)

    def run(self, task):
        b, r, arm = self.bot, self.r, self.bot.arm
        # finallook=0: from the stand the held book and arm cover the bin (it saw no bin in two of two v2 dev runs, 6 s each);
        # v4's aside + look saw no bin in 70 of 70 B-v52 episodes (BIN_FINAL seen 0) and cost about 14 s per attempt
        o = {'relook': 1, 'finallook': 0, 'stand': 0.70, 'deep': 0.12, 'release': 'floor', 'gap': 0.005, 'clear': 0.08, 'link_r': 0.06, 'measure_hold': 1, 'aside': 1, 'press': 0.04, 'stall': 0.6, 'lift': 0.005, **self.opts}
        bin_p, info, yaw = self.approach(task, o=o)
        u = np.array([math.cos(yaw), math.sin(yaw)]); left = np.array([-u[1], u[0]])
        p_base = None
        if int(o['finallook']):
            p_base = self.final_look(bin_p, info, o)
        rim = info.get('rim_z') or 0.95
        floor = info.get('floor_z')
        deep = float(o['deep'])
        bih = (getattr(b, 'pick_result', None) or {}).get('book_in_hand') or {}
        short = 0.0
        if bih.get('past_spine') is not None:              # lane E's pick: the book reaches (0.16 - past_spine) past the
            short = INSERT_PAST - float(bih['past_spine'])  # tool, 'deep' is tuned on the scripted 0.07 insert
            deep = min(deep, max(0.0, deep - short))
            b.stage('HAND_OFFSET', past_spine=round(float(bih['past_spine']), 3), deep=round(deep, 3))
        if info.get('rim_xy') is not None and len(info['rim_xy']) > 20:
            far = np.percentile((info['rim_xy'] - bin_p[:2]) @ u, 97)    # far rim along the approach, from the centre
            deep = max(0.0, min(deep, far - 0.06 - max(0.0, short) - 0.05))              # the book's far end 6 cm past the tool, 5 cm off the wall
        p = b.in_base(bin_p) if p_base is None else p_base.copy()
        near, far, lo, hi = -0.28, 0.28, -0.155, 0.155       # official bin outer rim, from its centre (u, left)
        if info.get('rim_xy') is not None and len(info['rim_xy']) > 20:
            rel = info['rim_xy'] - bin_p[:2]
            near, far = np.percentile(rel @ u, [3, 97])
            lo, hi = np.percentile(rel @ left, [3, 97])
        # The arm (links about 6 cm in radius) may go below the rim only inside the bin's inner footprint shrunk by the
        # link radius; everywhere else every sample along links 1 to 7 stays rim + clear (0.08 = radius + 2 cm) high.
        # The 21:20 rule held only joint points and midpoints up to the near wall as lines: links 5 to 7 touched the bin
        # on rows 2 and 4 in B-sweep3b, and those episodes never scored the place.
        r_link = float(o['link_r']) + 0.01                     # plus the 1 cm wall
        clear_x = None
        clear_box = (p[0] + near + r_link, p[0] + far - r_link, p[1] + lo + r_link, p[1] + hi - r_link)
        clear_outer = (p[0] + near - r_link, p[0] + far + r_link, p[1] + lo - r_link, p[1] + hi + r_link)
        p[0] += deep
        p[2] = rim + 0.18                                      # vertical book (bottom 0.115 m under the tool) clears the rim
        clear = rim + float(o['clear'])
        try:
            arm.pose(transform(p), duration=None, clear_z=clear, clear_box=clear_box, clear_outer=clear_outer)
        except Abort as e:                                     # no elbow-clear solution: hold higher, release as a drop
            b.stage('CLEAR_IK_FAILED', why=str(e)[:80])
            clear, o['release'] = None, 0.10
            p[2] = rim + 0.18
            arm.pose(transform(p), duration=None)
        if o['release'] == 'floor' and floor is not None and rim - floor > 0.08:
            below = (getattr(b, 'pick_result', None) or {}).get('book_below_tool', BOOK_BELOW_TOOL)
            seen = self.hold_offset(task) if int(o['measure_hold']) else None
            if seen is not None:                               # the book can slip in the fingers between grasp and bin
                below = BOOK_HALF_H - seen                     # (rows 2 and 4 hit the floor early, then rim or tip)
            # aim press (0.04) under the estimate: the lowering stops on the floor (LOWER_STOPPED), which B-v52 placed
            # gently in all 23 scored cases (impact 0.05 m/s); 2 of 2 releases that reached the estimate dropped (0.50, 0.82 m/s)
            z_rel = floor + float(o['gap']) + below - float(o['press'])
        else:
            z_rel = rim + (0.10 if o['release'] == 'floor' else float(o['release']))
        stalled = False
        try:
            for z in np.linspace(p[2], z_rel, max(2, int(np.ceil((p[2] - z_rel) / 0.04)) + 1))[1:]:
                q = p.copy(); q[2] = z
                arm.pose(transform(q), duration=0.8, pos_tol=0.03, clear_z=clear, clear_box=clear_box, clear_outer=clear_outer,
                         stall=float(o['stall']) or None)    # on the floor the joints stop: B-v52 waited the full 8.8 s
        except Abort as e:                                     # the book is over the bin: open anyway, a drop beats a miss
            b.stage('LOWER_STOPPED', why=str(e)[:80])
            stalled = True
            # Unload: the stalled goal keeps the arm pushing the book into the floor (0.11 rad short), and the open
            # fingers then kept the book (Z8 v5.6 seed 1001 4/yellow, crash8 2 of 8: the back step and rise carried it out)
            try:
                r.send('arm_right_controller', arm.q(arm.k.names), duration=0.3, tolerance=0.03, check=False)
            except Abort:
                pass
        if stalled and float(o['lift']) > 0:
            # Take the floor load off before opening: unloaded and opened (grip 0.0), the book still followed the back step
            # on Z8 s1001 5/yellow at v5.8. Lifted 5 mm it falls 5 mm when the fingers open (0.31 m/s, gentle below 0.5).
            try:
                tl = arm.tool()[:3, 3].copy(); tl[2] += float(o['lift'])
                arm.pose(transform(tl), duration=0.5, pos_tol=0.03)
            except Abort as e:
                b.stage('LIFT_FAILED', why=str(e)[:80])
        b.stage('RELEASE', tool_z=round(float(arm.tool()[2, 3]), 3), target_z=round(float(z_rel), 3), deep=round(deep, 3))
        r.gripper(False)
        r.wait(1.0)
        b.stage('OPENED', grip=round(float(r.joints['gripper_right_finger_joint']), 3))   # 0.0 open, about 0.79 closed
        # Back off the book before rising: the palm brushed the standing book on the way up (scored as a collision),
        # and with the open fingers still around it a straight rise lifts the book out (dev seed 1003 3/red, v5.3: book
        # carried to z 1.26). The thick-link clearance makes the back step infeasible deep in the bin (B-v52: 15 of 70
        # failed it, then crashed in park); the fallbacks drop the clearance for this short move toward the bin centre,
        # only on a solution near the current joints (no branch swing inside the bin).
        tool0 = arm.tool().copy()
        for dx, kw in ((0.06, dict(clear_z=clear, clear_box=clear_box, clear_outer=clear_outer)), (0.06, {}), (0.03, {})):
            try:
                back = tool0.copy(); back[0, 3] -= dx
                if not kw:
                    step = np.abs(arm.solve(transform(back[:3, 3])) - arm.q(arm.k.names)).max()
                    if step > 0.6:
                        raise Abort(f'back step would swing {step:.2f} rad')
                arm.pose(transform(back[:3, 3]), duration=1.0, pos_tol=0.04, **kw)
                if kw == {}:
                    b.stage('BACK_UNCLEARED', dx=dx)
                break
            except Abort as e:
                b.stage('BACK_STOPPED', dx=dx, cleared=bool(kw), why=str(e)[:80])
        try:
            up = arm.tool()[:3, 3].copy(); up[2] = p[2]
            arm.pose(transform(up), duration=1.5, pos_tol=0.04)
        except Abort as e:
            b.stage('RETRACT_STOPPED', why=str(e)[:80])
        return self.confirm(task, p_base if p_base is not None else b.in_base(bin_p), rim)

    def approach(self, task, carry=True, o=None):
        """Drive to the place stand: approach along -task.n, re-look at the bin from 1.2 m (BIN_RELOOK), then the stand
        with the bin at (stand, ARM_Y) in base_footprint, facing along -task.n. carry=False leaves the arm as it is (lane
        E's learned place). Returns (bin_p in odom, bin_info, yaw)."""
        b, r = self.bot, self.r
        o = o or {'relook': 1, 'stand': 0.70, **self.opts}
        bin_p = np.asarray(task.bin, float)
        info = getattr(b, 'bin_info', {}) or {}
        x, y, _ = r.base_in_odom()
        to = bin_p[:2] - np.array([x, y]); dist = np.linalg.norm(to); u = to / dist
        # Approach along the shelf normal (away from the shelf). The bin's table is 1.4 m wide across that line and 0.66 m
        # deep along it: from a side column the straight line from the robot ran the base into the table (10 base-table
        # collisions and 26 'base blocked' aborts in sweep bot-sweep1). The official layout puts the bin on the middle
        # column's line with its long axis along the normal.
        if task.n is not None and o.get('axis', 'shelf') == 'shelf':
            u = -np.asarray(task.n, float)[:2]; u = u / np.linalg.norm(u)
        if int(o['relook']) and dist > 1.5:
            left0 = np.array([-u[1], u[0]])
            near = bin_p[:2] - 1.2 * u - ARM_Y * left0             # on the stand's line: the last leg is straight ahead
            b.go(near[0], near[1], math.atan2(u[1], u[0]), tol=0.05, ytol=0.05, clearance=0.15)
            if carry:
                self.carry_posture()
            b.look_at(b.in_base(bin_p)); r.wait(0.3)
            pts, floors, rims = b.look_bin(frames=5)
            if len(pts) >= 3:
                old = bin_p
                bin_p = b.bin_summary(pts, floors, rims); info = b.bin_info
                b.stage('BIN_RELOOK', p=np.round(bin_p, 3).tolist(), moved=round(float(np.linalg.norm(bin_p[:2] - old[:2])), 3),
                        rim_z=round(info['rim_z'], 3), floor_z=None if info['floor_z'] is None else round(info['floor_z'], 3))
            if task.n is None or o.get('axis', 'shelf') != 'shelf':
                x, y, _ = r.base_in_odom()
                to = bin_p[:2] + ARM_Y * left0 - np.array([x, y]); u = to / np.linalg.norm(to)
        yaw = math.atan2(u[1], u[0]); left = np.array([-u[1], u[0]])
        stand = bin_p[:2] - float(o['stand']) * u - ARM_Y * left
        if carry:
            self.carry_posture()                               # book above the rim before the last leg (no-op if done)
        b.go(stand[0], stand[1], yaw, tol=0.03, ytol=0.03, vmax=0.15, clearance=0.08)
        return bin_p, info, yaw

    def final_look(self, bin_p, info, o, tries=2):
        """Bin in the base frame from the stand, with the held book moved out of the camera's view first (at the stand
        it covered the bin in every v2 run). The omni-wheel odometry slips 10 to 15 cm on the 0.5 m last leg (dev trace
        seed 1000 2/blue: drift 0.06 to 0.158 m, tool 9.5 cm off the bin centre sideways, the wrist on the side wall), so
        if the bin is more than 4 cm from where the stand puts it, the base steps by the measured error and looks again.
        Returns the bin centre in base_footprint (z = rim) or None."""
        b, r, arm = self.bot, self.r, self.bot.arm
        if int(o['aside']):
            try:
                tip = arm.tool()[:3, 3]                     # sideways only: (0.40, -0.55) swung row 1 through 3.6 rad; y -0.53 keeps the bin 0.29 m from the tool (look_bin drops < 0.25)
                arm.pose(transform([tip[0], -0.53, max(1.20, tip[2])]), duration=None, pos_tol=0.04)   # (replay <= 1.01)
            except Abort as e:
                b.stage('ASIDE_FAILED', why=str(e)[:80])
        want = np.array([float(o['stand']), ARM_Y])           # where the stand puts the bin centre in base_footprint
        rim0 = info.get('rim_z') or 0.95
        p_base = None
        for k in range(tries):
            exp = b.in_base(bin_p)
            b.look_at(exp); r.wait(0.3)
            pts, floors, rims = b.look_bin(frames=4, frame='base_footprint')
            ok = [i for i, q in enumerate(pts) if abs(rims[i] - rim0) < 0.04 and np.linalg.norm(q[:2] - exp[:2]) < 0.30]
            if len(ok) < 2:
                b.stage('BIN_FINAL', kept=0, seen=len(pts), note='odometry estimate kept')
                return p_base
            p_base = np.median(np.array([pts[i] for i in ok]), 0)
            err = p_base[:2] - want
            b.stage('BIN_FINAL', p_base=np.round(p_base, 3).tolist(), stand_err=np.round(err, 3).tolist(), kept=len(ok))
            if np.linalg.norm(err) < 0.04 or k == tries - 1:
                return p_base
            x, y, yaw = r.base_in_odom()                       # step by the measured error (short leg, little slip)
            c, s_ = math.cos(yaw), math.sin(yaw)
            b.go(x + c * err[0] - s_ * err[1], y + s_ * err[0] + c * err[1], yaw, tol=0.015, ytol=0.02, vmax=0.10, clearance=0.05)
        return p_base

    def hold_offset(self, task, frames=4):
        """Height of the held book's spine centre above the grasping link, measured by sight over the bin (base frame).
        None when not seen or implausible."""
        b, r = self.bot, self.r
        tool = b.arm.tool()[:3, 3]
        b.look_at(tool + [0.0, 0.0, 0.05]); r.wait(0.3)
        dz = []
        for _ in range(frames):
            dets, _ = b.eyes.look([task.target], frame='base_footprint', conf=0.3, kp=False)
            dz += [d['p'][2] - tool[2] for d in dets or [] if np.linalg.norm(d['p'][:2] - tool[:2]) < 0.20]
            r.wait(0.1)
        if len(dz) < 2:
            b.stage('HOLD_MEASURED', seen=len(dz))
            return None
        v = float(np.median(dz))
        b.stage('HOLD_MEASURED', dz=round(v, 3), seen=len(dz))
        return v if -0.12 < v < 0.12 else None

    def confirm(self, task, bin_base, rim, frames=5):
        """After the release, look for the book: inside the bin (centre within the rim and below it), on the rim, or on
        the floor. A sweep run reported DONE with no book in the bin, so success is never declared without this."""
        b, r = self.bot, self.r
        b.look_at(bin_base); r.wait(0.3)
        seen = []
        for _ in range(frames):
            dets, _ = b.eyes.look([task.target], frame='base_footprint', conf=0.3, kp=False)
            seen += [d['p'] for d in dets or []]
            r.wait(0.1)
        where = 'unseen'
        if seen:
            q = np.median(np.array(seen), 0)
            off = float(np.linalg.norm(q[:2] - bin_base[:2]))
            tool = b.arm.tool()[:3, 3]
            held = bool(np.linalg.norm(q[:2] - tool[:2]) < 0.15 and q[2] > rim)   # seen with the gripper: carried out
            if off < 0.30 and q[2] < rim - 0.01:
                where = 'bin'
            elif off < 0.40 and q[2] >= rim - 0.01:
                where = 'rim'
            elif q[2] < 0.4:
                where = 'floor'
            else:
                where = 'elsewhere'
            b.stage('PLACE_CHECK', where=where, p_base=np.round(q, 3).tolist(), off=round(off, 3), seen=len(seen), held=held)
        else:
            b.stage('PLACE_CHECK', where=where, seen=0)
        if where in ('rim', 'floor', 'elsewhere'):
            raise Abort(f'book on the {where} after the release, not in the bin')
        return {'confirmed': where == 'bin', 'where': where}

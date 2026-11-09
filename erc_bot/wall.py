"""Parametrised pick (lane G): lane B's ScriptedPick (erc_bot/stages.py) with its constants set from a parameter file,
plus a manoeuvre for books pressed against a column side wall (Round 2 column 4, x1.5).

  python3 -m erc_bot.mission --digit 4 --label red --params best.json            (erc_bot/params.py: file layout)
  python3 -m erc_bot.mission --digit 4 --label red --pick erc_bot.wall:ParamPick --pick-opt params=best.json

No copy of B's pick: ParamPick subclasses ScriptedPick (B's interface since 21:15: per-instance geometry and the
prepare / grasp / finish steps). The constructor sets self.torso, self.reach, self.grasp_dz (per row) from the file,
run() sets self.arm_y and B's opts insert_past and carry for the row, then calls ScriptedPick.run. With
params_default.json (B's values) it is B's pick. The literals inside B's steps (pre-grasp 9 cm, test lift 2.5 cm,
extraction 20 cm, clear point 1.40 m, back-off 0.28 m) are B's and are not knobs here.

A book counts as a wall book when its spine centre lies more than wall.detect_off (0.40 m) along the shelf from its
column's digit plate (Round 1 and columns 1, 3, 5 jitter books by at most 0.25 m, Round 2 column 4 books sit 0.485 m
off centre). Wall books take the wall manoeuvre when wall.enable is 1: book = B's prepare (clear point, torso, open,
frontal re-look), then lane G's push and yawed grasp in place of B's grasp, then B's finish (hold check, carry).
  1. optional push: closed fingertips press the spine near its open edge and drive it into the shelf. The book slides
     back to the back panel and yaws about its back corner at the wall, which opens a gap between its front and the wall.
  2. grasp: open gripper, tool yawed and offset from the spine centre, insert along the tool axis, close, lift, extract
     straight back, back off 0.28 m. Every lateral value is signed away from the wall, so one parameter set serves both
     side walls.
"""
import math
import numpy as np
from . import params as P
from .arm import transform
from .live import Abort
from .stages import ScriptedPick, PickTask, SURE_HOLD

__all__ = ['ParamPick', 'PickTask']


def _yawed(xyz, yaw):
    return transform(list(xyz), (0.0, 0.0, yaw))


class ParamPick(ScriptedPick):
    name = 'param'

    def __init__(self, bot, params=None, **opts):
        super().__init__(bot, **opts)
        self.p = P.load(params if params else {})
        self.params_path = params
        for row in (1, 2, 3, 4):
            k = P.knobs(self.p, row)
            self.torso[row], self.reach[row], self.grasp_dz[row] = float(k['torso']), float(k['reach']), float(k['dz'])

    def wall_side(self, task):
        """+1 wall on the robot's left (shelf frame), -1 on its right, 0 not a wall book."""
        if task.plate is None:
            return 0
        n = np.asarray(task.n, float); left = np.array([-n[1], n[0]])
        off = float((np.asarray(task.book[:2], float) - np.asarray(task.plate[:2], float)) @ left)
        thr = float(P.knobs(self.p, task.row, wall=True)['detect_off'])
        return int(np.sign(off)) if abs(off) > thr else 0

    def run(self, task):
        w = self.wall_side(task)
        wk = P.knobs(self.p, task.row, wall=True)
        k = P.knobs(self.p, task.row)
        self.arm_y = float(k['arm_y'])
        self.opts = dict(self.opts, insert_past=k['insert_past'], carry=k['carry'])
        wall = bool(w and int(wk['enable']))
        self.bot.stage('PICK_PARAMS', wall_side=w, wall=wall, params=self.params_path)
        if not wall:
            return ScriptedPick.run(self, task)
        book = self.prepare(task)
        self.wall_grasp(task, book, w, wk)
        return self.finish(task)

    # ------------------------------------------------------------------ wall book
    def _ik_path(self, goals, what):
        arm, r = self.bot.arm, self.r
        seed = arm.q(arm.k.names); T = r.matrix('torso_lift_link', 'base_footprint')
        for i, g in enumerate(goals):
            try:
                seed = arm.k.solve(T @ g, seed)
            except RuntimeError as e:
                raise Abort(f'IK: {what} waypoint {i} {np.round(g[:3, 3], 3).tolist()} unreachable: {e}')

    def _grip(self, cmd):
        self.r.send('gripper_right_controller', [float(cmd)], duration=1.5)
        return self.r.joints['gripper_right_finger_joint']

    def wall_grasp(self, task, book, w, k):
        """In place of ScriptedPick.grasp: optional push, then the yawed, offset grasp; ends backed off 0.28 m."""
        b, r, arm = self.bot, self.r, self.bot.arm
        row, n = task.row, np.asarray(task.n, float)
        yaw = math.atan2(n[1], n[0]); left = np.array([-n[1], n[0]])
        away = -w * left                                   # odom xy, away from the wall
        reach = self.reach[row] + float(k['reach_add'])
        base_off = float(k['base_off'])
        dz0 = self.grasp_dz.get(row, 0.0)
        push = int(k['push'])
        if push:
            r.gripper(True)
        elif float(k['g_open']) > 0:
            self._grip(k['g_open'])
        b.stage('WALL_DEPLOY', side=w)
        arm.pose(transform([reach - .09, self.arm_y, book[2] + dz0]))
        app = book[:2] - reach * n - self.arm_y * left + base_off * away
        b.go(app[0], app[1], yaw, tol=0.015, ytol=0.015, vmax=0.12, clearance=0.05)
        p = b.in_base(book)
        aw = np.array([0.0, -w, 0.0])                        # base frame, away from the wall (base y = shelf left)

        if push:
            ang = w * float(k['push_yaw']); ax = np.array([math.cos(ang), math.sin(ang), 0.0])
            c = p + aw * float(k['push_off']) + [0, 0, dz0 + float(k['push_dz'])]
            start = c - float(k['push_start']) * ax
            end = c + float(k['push_dx']) * ax
            sweep = aw * float(k['push_sweep'])
            back = end + sweep - float(k['push_back']) * ax
            self._ik_path([_yawed(start, ang), _yawed(end + sweep, ang), _yawed(back, ang)], 'push')
            arm.pose(_yawed(start, ang), pos_tol=0.03, ang_tol=0.15)
            b.stage('WALL_PUSH', yaw=round(ang, 3))
            ns = max(1, int(k['push_steps'])); blocked = None
            for i, s in enumerate(np.linspace(0, 1, ns + 1)[1:]):
                q = start + s * (end - start) + (sweep if i == ns - 1 else 0)
                try:
                    arm.pose(_yawed(q, ang), duration=float(k['push_s']), pos_tol=0.05, ang_tol=0.3)
                except Abort as e:
                    blocked = str(e)[:80]
                    break
            tip = arm.tool()[:3, 3]
            b.stage('WALL_PUSHED', reached=np.round(tip - p, 3).tolist(), blocked=blocked)
            try:
                arm.pose(_yawed(tip - float(k['push_back']) * ax, ang), duration=1.5, pos_tol=0.05, ang_tol=0.3)
            except Abort as e:
                b.stage('WALL_PUSH_BACK_SHORT', why=str(e)[:80])
            self._grip(k['g_open'])
            if int(k['redetect']):
                p = self._redetect(task, p)

        ang = w * float(k['g_yaw']); ax = np.array([math.cos(ang), math.sin(ang), 0.0])
        tgt = p + np.array([float(k['g_dx']), 0, 0]) + aw * float(k['g_off']) + [0, 0, dz0 + float(k['g_dz'])]
        pre = tgt - float(k['g_pre']) * ax
        deep = tgt + float(k['g_insert']) * ax
        lifted = deep + [0, 0, float(k['g_lift'])]
        out = lifted - [float(k['g_extract']), 0, 0]
        self._ik_path([_yawed(pre, ang), _yawed(deep, ang), _yawed(lifted, ang), _yawed(lifted - [0.05, 0, 0], ang)], 'wall grasp')
        arm.pose(_yawed(pre, ang))
        b.stage('WALL_INSERT', yaw=round(ang, 3), target=np.round(tgt - b.in_base(book), 3).tolist())
        ns = max(1, int(k['g_steps'])); g = pre
        for i, s in enumerate(np.linspace(0, 1, ns + 1)[1:]):
            g = pre + s * (deep - pre)
            arm.pose(_yawed(g, ang), duration=float(k['g_step_s']), pos_tol=0.012 if i == ns - 1 else 0.02)
        b.stage('CLOSE')
        pos = r.gripper(True)
        b.stage('GRIPPER', position=round(pos, 3), sure=pos < SURE_HOLD)
        arm.pose(_yawed(lifted, ang), duration=1.5)
        r.wait(0.6)
        b.stage('EXTRACT')
        arm.pose(_yawed(out, ang), duration=2.0, pos_tol=0.03)
        x, y, _ = r.base_in_odom()
        b.go(x - .28 * n[0], y - .28 * n[1], yaw, clearance=0.05)
        self.grasp_dz[row] = dz0 + float(k['g_dz'])       # B's finish and the place read the grasp height from here

    def _redetect(self, task, p_old):
        """Spine centre again after the push (base frame), the detection nearest the old estimate within 12 cm."""
        b = self.bot
        b.look_at(p_old); self.r.wait(0.3)
        hits = []
        for _ in range(6):
            dets, _ = b.eyes.look([task.target], frame='base_footprint', conf=0.25)
            hits += [d['p'] for d in dets or [] if np.linalg.norm(d['p'][:2] - p_old[:2]) < 0.12]
            self.r.wait(0.1)
        if len(hits) >= 2:
            p = np.median(np.array(hits), 0)
            b.stage('WALL_REDETECT', moved=np.round(p - p_old, 3).tolist(), n=len(hits))
            return np.r_[p[:2], p_old[2]]
        b.stage('WALL_REDETECT', moved=None, n=len(hits))
        return p_old

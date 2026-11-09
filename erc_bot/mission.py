"""ERC 2026 Phase 2 bot for the live TIAGo Pro: find the column, find the book, pick it, carry it, place it in the bin.

  python3 -m erc_bot.mission --digit 2 --label red  [--model det.onnx]      Round 1 (colour) or Round 2 column 2 (A-D)

Stages: start (park both arms) -> find the shelf and the column from the digit plates -> approach the column and find
the target book with the detector -> pick (the right arm, Jalpari Phase 1 grasp geometry) -> carry posture -> find
the bin -> lower the book into it and open -> park. Every frame of reference is odom plus what the camera sees: no
walls, no map, no contact sensors.
"""
import argparse, math, os, sys, time
# rclpy already imported means a launcher runs this module in-process (lane F's pert_boot.py patches the action clients
# first): a child process would lose that patch, so the mission runs here unsupervised.
if __name__ == '__main__' and not (os.environ.get('ERC_BOT_CHILD') or os.environ.get('ERC_BOT_NO_SUPERVISOR')
                                   or 'rclpy' in sys.modules):
    from .supervisor import supervise      # stdlib only: the parent stays small and survives what kills the child
    sys.exit(supervise())
import numpy as np
import rclpy
from .live import Live, Abort
from .arm import Arm
from .perception import Eyes
from .shelf import Shelf, frontal_stand, laser_shelf_line, override_plates
from . import stages
from .stages import ROW_Z, SURE_HOLD, PickTask, PlaceTask

import os as _os
WALL_PARAMS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'params_wall_g1.json')

WORDS = {'A': 'book_FUTURE', 'B': 'book_ROBOT', 'C': 'book_LIBRARY', 'D': 'book_AGENT'}


BLIND_PLATES = os.environ.get('ERC_BOT_BLIND_PLATES') == '1'     # test only


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Bot:
    def __init__(self, live, model, digit, label, pick='erc_bot.stages:ScriptedPick', place='erc_bot.stages:ScriptedPlace',
                 pick_opts=(), place_opts=(), out='', word_model=None):
        self.r, self.digit, self.label = live, int(digit), label
        self.out, self.result, self.reason = out, 'running', ''
        self.target = WORDS[label.upper()] if label.upper() in WORDS else f'book_{label.lower()}'
        self.eyes = Eyes(live, model, word_model=word_model)
        self.arm = Arm(live)
        self.shelf = Shelf()
        self.anchor = os.environ.get('ERC_BOT_ANCHOR', '1') != '0'   # lane H re-anchoring after base moves (0: v5.8 behaviour)
        col = os.environ.get('ERC_BOT_COLUMN', '').strip()    # physical column 1-5 (left to right facing the shelf) used
        self.column_override = int(col) if col in ('1', '2', '3', '4', '5') else None   # only when the plate is not read
        self.column_source, self.column_phys, self.plates_read = 'plate', None, set()
        self.t0 = live.now()
        self.events = []
        self.picker = stages.load(pick, self, pick_opts)
        self.placer = stages.load(place, self, place_opts)

    def stage(self, name, **kw):
        self.r.stop()
        if name in ('HOLDING', 'PLACE', 'HOLD_MEASURED', 'RELEASE', 'OPENED'):   # arm joints at the pick end, the place start and the release
            try:                                                 # (lanes A/E compare their learned holds' IK branch with these)
                kw['q_arm'] = [round(float(v), 3) for v in self.arm.q(self.arm.k.names)]
            except Exception:
                pass
        self.events.append({'stage': name, 't': round(self.r.now() - self.t0, 2), **kw})
        self.r.say(f'[{self.r.now() - self.t0:6.1f} s] {name} {kw if kw else ""}')
        self.save()

    def save(self):
        """mission.json in --out, rewritten at every stage so a harness that kills the bot still reads how far it got."""
        if not self.out:
            return
        import json, os
        ev = self.events
        rec = {'result': self.result, 'reason': self.reason, 'digit': self.digit, 'label': self.label, 'target': self.target,
               'pick': self.picker.name if hasattr(self, 'picker') else None, 'place': self.placer.name if hasattr(self, 'placer') else None,
               'last_stage': ev[-1]['stage'] if ev else None, 'mission_time': round(self.r.now() - self.t0, 2),
               'stage_t': {e['stage']: e['t'] for e in ev}, 'events': ev}
        os.makedirs(self.out, exist_ok=True)
        tmp = os.path.join(self.out, '.mission.json.tmp')
        json.dump(rec, open(tmp, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        os.replace(tmp, os.path.join(self.out, 'mission.json'))

    # ---------- base ----------
    def go(self, x, y, yaw, tol=0.03, ytol=0.03, vmax=0.25, timeout=90.0, clearance=0.25):
        """Holonomic point-to-pose in odom, laser stop on, pauses while blocked and gives up after 6 s blocked."""
        t0, blocked = self.r.now(), None
        try:
            while self.r.now() - t0 < timeout:
                self.r.tick()
                bx, by, byaw = self.r.base_in_odom()
                ex, ey, eyaw = x - bx, y - by, wrap(yaw - byaw)
                if math.hypot(ex, ey) < tol and abs(eyaw) < ytol:
                    return
                c, s = math.cos(byaw), math.sin(byaw)
                lx, ly = c * ex + s * ey, -s * ex + c * ey
                d = math.hypot(lx, ly)
                v = min(vmax, 0.8 * d) * (0.35 if abs(eyaw) > 0.35 else 1.0)
                vx, vy = (lx / d * v, ly / d * v) if d > 1e-3 else (0.0, 0.0)
                if d < tol:
                    vx = vy = 0.0
                wz = max(-0.45, min(0.45, 1.2 * eyaw))
                if not self.r.velocity(vx, vy, wz, clearance=clearance):
                    blocked = blocked or self.r.now()
                    if self.r.now() - blocked > 6.0:
                        raise Abort('base blocked for 6 s')
                else:
                    blocked = None
            raise Abort(f'navigation timed out, {math.hypot(ex, ey):.2f} m and {eyaw:.2f} rad left')
        finally:
            self.r.stop()

    def face(self, yaw):
        x, y, _ = self.r.base_in_odom()
        self.go(x, y, yaw, tol=1.0)

    # ---------- stages ----------
    def start(self):
        self.stage('START')
        self.start_pose = self.r.base_in_odom()
        self.arm.park()
        self.r.head(0.0, 0.12)

    def look_plates(self, frames=3):
        n = 0
        for _ in range(frames):
            dets, _ = self.eyes.look([f'marker_{i}' for i in range(1, 6)])
            if BLIND_PLATES:                     # test only: no plate is ever read (the real plate "1", lane P)
                dets = []
            for d in dets or []:
                self.shelf.add(int(d['class'][-1]), d['p']); n += 1
                self.plates_read.add(int(d['class'][-1]))
            self.r.wait(0.12)
        return n

    def find_column(self):
        """Pan the head until the target plate and two others are known; check after every head pose. If the scan at
        this heading saw some plates but not enough, face their bearing next (the shelf is there); with none seen, try
        the other headings. Wide pans first."""
        self.stage('FIND_COLUMN', digit=self.digit)
        x0, y0, yaw0 = self.start_pose
        tried, heading = [], yaw0
        fallback = [yaw0 + math.pi / 2, yaw0 - math.pi / 2, yaw0 + math.pi]
        for _ in range(4):
            if tried:
                self.face(heading)
            tried.append(heading)
            for pan in (0.0, -1.1, 1.1, -0.55, 0.55):
                for tilt in (0.12, 0.27):          # the higher tilt keeps plates whole when the robot starts closer
                    self.r.head(pan, tilt)
                    self.r.wait(0.2)
                    self.look_plates(2)
                    col = self.shelf.column(self.digit, self.r.base_in_odom()[:2])
                    if col is not None and len(col[3]['plates']) >= 3:
                        return self.refine_column()
            if self.column_override is not None and self.shelf.column(self.digit, self.r.base_in_odom()[:2]) is None:
                return self.override_column()     # the operator named the column: one heading's pans, not four (128 s)
            P = self.shelf.plates(min_obs=1)
            x, y, _ = self.r.base_in_odom()
            nxt = None
            if P:
                c = np.mean([p[:2] for p in P.values()], 0)
                b = math.atan2(c[1] - y, c[0] - x)
                if all(abs(wrap(b - h)) > 0.3 for h in tried):
                    nxt = b
            if nxt is None:
                nxt = next((h for h in fallback if all(abs(wrap(h - t)) > 0.3 for t in tried)), None)
            if nxt is None:
                break
            heading = nxt
        col = self.shelf.column(self.digit, self.r.base_in_odom()[:2])
        if col is None and self.column_override is not None:
            return self.override_column()
        if col is None:
            raise Abort(f'plate {self.digit} not found, plates seen {sorted(self.shelf.plates())}')
        return self.refine_column()

    def override_column(self):
        """ERC_BOT_COLUMN set and the target plate not read in the first heading's pans: the operator's physical column on the
        laser's shelf line (shelf.laser_shelf_line; any plate boxes seen, whatever digit they read as, are column-centre
        evidence). The panel is looked at again from 2.6 m in front of its middle (whole panel in view, both ends). The
        five column plates go into the shelf model (the target's under its digit, the others under -column), so the
        approach, the bin prediction and the re-anchoring run as usual; the book re-look keeps its own depth (the
        plate's depth here is the laser panel's less PANEL_BEHIND_PLATE, not a sighting)."""
        k = self.column_override
        line = self.shelf_line()
        if line is None:
            for h in (0.0, math.pi / 2, -math.pi / 2, math.pi):
                self.face(self.start_pose[2] + h)
                line = self.shelf_line()
                if line is not None:
                    break
        if line is None:
            raise Abort(f'plate {self.digit} not found and no shelf line by laser (column override {k})')
        for _ in range(2):                       # re-centre once more when the first frontal look saw part of it
            stand = frontal_stand(line)
            x, y, _ = self.r.base_in_odom()
            if np.linalg.norm(stand - [x, y]) < 0.3:
                break
            self.go(stand[0], stand[1], math.atan2(line['n'][1], line['n'][0]))
            line = self.shelf_line() or line
            if line['span'] >= 4.6:
                break
        cols, how = override_plates(line, k, list(self.shelf.plates(min_obs=1).values()))
        self.shelf.past = {}
        self.shelf.obs = {(self.digit if j == k else -j): [p, p] for j, p in cols.items()}
        self.column_source, self.column_phys = 'override', k
        self.stage('COLUMN_OVERRIDE', column=k, how=how, plate=np.round(cols[k], 3).tolist(), inliers=line['k'],
                   read=sorted(self.plates_read))
        print(f'[column] digit {self.digit} plate not read: physical column {k} from {how}', flush=True)
        return self.refine_column()

    def shelf_line(self, scans=3):
        """laser_shelf_line over a few scans from here (odom), the one with the longest panel."""
        best = None
        for i in range(scans):
            if i:
                self.r.wait(0.15)
            pts = self.r.scan_points()
            if pts is None or not len(pts):
                continue
            x, y, yaw = self.r.base_in_odom()
            c, s_ = math.cos(yaw), math.sin(yaw)
            got = laser_shelf_line(pts @ np.array([[c, s_], [-s_, c]]) + [x, y], (x, y))
            if got is not None and (best is None or got['span'] > best['span']):
                best = got
        return best

    def refine_column(self):
        plate, n, t, f = self.shelf.column(self.digit, self.r.base_in_odom()[:2])
        phys = self.column_phys
        if phys is None and sorted(f['plates']) == [1, 2, 3, 4, 5]:   # physical column from the plates: 1 is leftmost facing the shelf
            left = np.array([-n[1], n[0]])
            phys = sorted(f['plates'], key=lambda d: -float(f['plates'][d][:2] @ left)).index(self.digit) + 1
        self.stage('COLUMN', plate=np.round(plate, 3).tolist(), plates=sorted(f['plates']), line_resid=round(f['resid'], 3),
                   fresh=sorted(self.shelf.fresh()), source=self.column_source, column=phys, read=sorted(self.plates_read))
        return plate, n, t

    def find_book(self, plate, n, t):
        """Stand 1.15 m in front of the plate, square to the shelf, and tilt through the rows."""
        self.stage('APPROACH_COLUMN')
        yaw = math.atan2(n[1], n[0])
        stand = plate[:2] - 1.15 * n
        self.go(stand[0], stand[1], yaw)
        if self.anchor:
            self.shelf.new_epoch()               # re-anchor the plates to this pose (odometry slipped on the way here)
        for tilt in ((0.12, 0.27) if self.anchor else (0.12,)):
            self.r.head(0.0, tilt); self.r.wait(0.3); self.look_plates(4)
            if self.digit in self.shelf.fresh():
                break
        plate, n, t = self.refine_column()
        yaw = math.atan2(n[1], n[0])
        self.stage('FIND_BOOK', target=self.target)
        for tilt in (0.25, 0.0, -0.3, -0.6):
            self.r.head(0.0, tilt); self.r.wait(0.3)
            prev, stable = None, 0
            for _ in range(12):
                dets, _ = self.eyes.look([self.target])
                cand = []
                print(f'[findbook] tilt {tilt:+.2f}: ' + ('; '.join(f"along {float((d['p'] - plate)[:2] @ t):+.2f} depth {float((d['p'] - plate)[:2] @ n):+.2f} z {float(d['p'][2]):.2f} s {d['score']:.2f}" for d in dets or []) or 'none'), flush=True)
                for d in dets or []:
                    rel = d['p'] - plate
                    if abs(rel[:2] @ t) < 0.515 and -0.3 < rel[:2] @ n < 0.4:   # Round 2 column 4 books stand against the side walls, 0.485 m off centre
                        row = int(np.argmin(np.abs(ROW_Z - d['p'][2]))) + 1
                        if abs(ROW_Z[row - 1] - d['p'][2]) < 0.12:
                            cand.append((d, row))
                if cand:                       # two same-colour books in the window (real shelf 6 Nov): track one
                    d, row = (min(cand, key=lambda c: np.linalg.norm(c[0]['p'] - prev)) if prev is not None
                              else max(cand, key=lambda c: c[0]['score']))
                    stable = stable + 1 if prev is not None and np.linalg.norm(d['p'] - prev) < 0.03 else 1
                    prev = d['p']
                    if stable >= 3:
                        self.stage('BOOK', row=row, p=np.round(d['p'], 3).tolist(), score=d['score'])
                        return d['p'], row, plate, n, t
                else:
                    stable = 0
                self.r.wait(0.1)
        raise Abort(f'{self.target} not found in column {self.digit}')

    def in_base(self, p_odom):
        return (np.linalg.inv(self.r.odom_T_base()) @ np.r_[p_odom, 1.0])[:3]

    def holding(self):
        return self.r.joints['gripper_right_finger_joint'] < SURE_HOLD

    def look_at(self, p_base):
        """Point the head at a point in base_footprint (pan about the neck, tilt from the camera height)."""
        cam = self.r.matrix('base_footprint', 'head_front_camera_color_optical_frame')[:3, 3]
        d = p_base - cam
        self.r.head(math.atan2(d[1], d[0]), math.atan2(d[2], math.hypot(d[0], d[1])))   # tilt negative looks down

    def seen_in_hand(self, frames=6):
        """The target book seen within 18 cm of the gripper (base frame): the hold confirmed by sight."""
        tool = self.arm.tool()[:3, 3]
        self.look_at(tool); self.r.wait(0.3)
        hits = 0
        for _ in range(frames):
            dets, _ = self.eyes.look([self.target], frame='base_footprint', conf=0.25, kp=False)
            hits += any(np.linalg.norm(d['p'] - tool) < 0.18 for d in dets or [])   # spine face 10 cm before the link, 5 cm
                                                                                  # above it on a row-1 grasp: 0.12 missed a held book
            self.r.wait(0.1)
        return hits >= 2

    def pick(self, book, row, n, t=None, plate=None):
        """The pluggable pick stage (default: Jalpari Phase 1 grasp, then the carry posture)."""
        self.stage('PICK', row=row, stage_impl=self.picker.name)
        res = self.picker.run(PickTask(self.target, np.asarray(book), row, np.asarray(n),
                                       None if t is None else np.asarray(t), None if plate is None else np.asarray(plate)))
        if not (res or {}).get('held') and not (self.holding() or self.seen_in_hand()):
            raise Abort('pick stage returned without the book')
        self.pick_result = res or {}
        return res

    def look_bin(self, frames=4, conf=None, frame='odom'):
        """Bin detections this head pose: (rim centres in `frame`, floor heights, rim heights)."""
        pts, floors, rims = [], [], []
        self.bin_rim_xy = []
        # The held book is in view while the torso rises and a red book can read as 'bin': in bot-sweep2 six episodes
        # took the book in hand (rim 1.17 m, about 0.5 m away) for the bin and released it onto the floor. Reject
        # detections near the gripper or with a rim above 1.10 m (the bin rim is 0.95 m on its table).
        T = self.r.odom_T_base() if frame == 'odom' else np.eye(4)
        tool = (T @ self.arm.tool())[:3, 3]
        for _ in range(frames):
            dets, _ = self.eyes.look(['bin'], conf=conf, frame=frame)
            for d in dets or []:
                if 'rim_z' in d and (d['rim_z'] > 1.10 or np.linalg.norm(d['p'][:2] - tool[:2]) < 0.25):
                    continue
                if 0.5 < d['p'][2] < 1.3 and 'rim_z' in d:
                    pts.append(d['p']); rims.append(d['rim_z']); self.bin_rim_xy.append(d['rim_xy'])
                    if d.get('floor_z') is not None:
                        floors.append(d['floor_z'])
            self.r.wait(0.1)
        return pts, floors, rims

    def bin_summary(self, pts, floors, rims):
        p = np.median(np.array(pts), 0)
        self.bin_info = {'p': p, 'rim_z': float(np.median(rims)), 'floor_z': float(np.median(floors)) if floors else None,
                         'rim_xy': np.concatenate(self.bin_rim_xy) if self.bin_rim_xy else None}
        return p

    def predicted_bin(self, n):
        """Official layout: the bin 3.75 m from the shelf face on the middle column's line. The middle column is the
        median plate along the shelf when all five are known, else the plates' centre. None without a shelf fit."""
        fl = os.environ.get('ERC_BOT_BIN')       # operator: 'forward,left' metres from the START pose (tape)
        if fl:
            fwd, left = (float(v) for v in fl.split(','))
            x0, y0, yaw0 = self.start_pose
            return np.array([x0 + fwd * math.cos(yaw0) - left * math.sin(yaw0), y0 + fwd * math.sin(yaw0) + left * math.cos(yaw0)])
        f = self.shelf.fit(self.r.base_in_odom()[:2])
        if f is None:
            return None
        P = np.array([p[:2] for p in f['plates'].values()])
        along = (P - f['c']) @ f['t']
        mid = f['c'] + (np.median(along) if len(P) == 5 else 0.0) * f['t']
        return mid - 3.75 * np.asarray(n)[:2]

    def find_bin(self, n):
        """The bin stands on a table across the arena from the shelf (official layout: 3.75 m from the shelf face, on the
        middle column's line). Turn away from the shelf and look; then try either side."""
        self.stage('FIND_BIN')
        away = math.atan2(-n[1], -n[0])
        # The laser guard sees the base, not the arms: turning in place with the book held forward sweeps about 0.8 m.
        # Back straight off to 1.3 m from the shelf face (plate line) before any turn (lane F: a 180 deg turn put an elbow
        # on a table). After a row-1 pick the base is about 0.83 m from the spines.
        f = self.shelf.fit(self.r.base_in_odom()[:2])
        if f is not None:
            x, y, yaw = self.r.base_in_odom()
            d = float((f['c'] - np.array([x, y])) @ np.asarray(n)[:2])
            if d < 1.3:
                self.go(x - (1.3 - d) * n[0], y - (1.3 - d) * n[1], yaw, clearance=0.15)
        pred = self.predicted_bin(n)
        if pred is not None:                     # look straight at where the official layout puts it first
            x, y, _ = self.r.base_in_odom()
            hd = math.atan2(pred[1] - y, pred[0] - x)
            self.face(hd)
            dist = math.hypot(pred[0] - x, pred[1] - y)
            if dist > 1.8:                       # real bin (6 Nov): det-v4 sees it steadily only from close range
                k = (dist - 1.4) / dist
                self.go(x + k * (pred[0] - x), y + k * (pred[1] - y), hd, clearance=0.15)
                self.stage('BIN_CLOSER', moved=round(dist - 1.4, 2))
            for pan in (0.0, 0.3, -0.3):
                for tilt in (-0.35, -0.6):
                    self.r.head(pan, tilt); self.r.wait(0.2)
                    pts, floors, rims = self.look_bin()
                    if len(pts) >= 3:
                        p = self.bin_summary(pts, floors, rims)
                        self.stage('BIN', p=np.round(p, 3).tolist(), seen=len(pts), rim_z=round(self.bin_info['rim_z'], 3),
                                   from_prediction=round(float(np.linalg.norm(p[:2] - pred[:2])), 3))
                        return p
        for turn in (0.0, math.pi / 2, -math.pi / 2):
            self.face(away + turn)
            for pan in (0.0, 0.5, -0.5, 1.0, -1.0):
                for tilt in (-0.35, -0.6):
                    self.r.head(pan, tilt); self.r.wait(0.3)
                    pts, floors, rims = self.look_bin()
                    if len(pts) >= 3:
                        p = self.bin_summary(pts, floors, rims)
                        self.stage('BIN', p=np.round(p, 3).tolist(), seen=len(pts), rim_z=round(self.bin_info['rim_z'], 3))
                        return p
        raise Abort('bin not found')

    def laser_normal(self, n, max_dev=0.26):
        """The shelf normal again, by laser, after the turn toward the bin. Odometry slips on turns (lane F's 3 % proxy:
        about 5 deg on the 180 deg turn after the pick), so the normal from the plates, seen before the turn, no longer
        agrees with the bin seen after it, and the place approaches the bin's table off its axis ('base blocked' at the
        stand in 2 of 8 held-out episodes). The shelf's back panel is a straight 5 m line behind the robot: RANSAC for
        the longest line within max_dev of the old normal, refit, and return the normal in odom now. None when there is
        no such line (the old normal stays)."""
        pts = self.r.scan_points()
        if pts is None or len(pts) < 30:
            return None
        yaw = self.r.base_in_odom()[2]
        c, s = math.cos(yaw), math.sin(yaw)
        nb = np.array([c * n[0] + s * n[1], -s * n[0] + c * n[1]])
        along, side = pts @ nb, pts @ np.array([-nb[1], nb[0]])
        P = pts[(along > 0.4) & (along < 5.0) & (np.abs(side) < 2.6)]
        if len(P) < 30:
            return None
        rng = np.random.default_rng(0)
        best = (0, None)
        for _ in range(300):
            i, j = rng.choice(len(P), 2, replace=False)
            d = P[j] - P[i]; L = float(np.linalg.norm(d))
            if L < 0.4:
                continue
            m = np.array([-d[1], d[0]]) / L
            if abs(float(m @ nb)) < math.cos(max_dev):
                continue
            inl = np.abs((P - P[i]) @ m) < 0.02
            if inl.sum() > best[0]:
                best = (int(inl.sum()), inl)
        if best[1] is None or best[0] < 30:
            return None
        Q = P[best[1]]
        q0 = Q.mean(0)
        t = np.linalg.svd(Q - q0)[2][0]
        m = np.array([-t[1], t[0]])
        if m @ nb < 0:
            m = -m
        span = float(np.ptp((Q - q0) @ t))
        if span < 1.0:
            return None
        n_new = np.array([c * m[0] - s * m[1], s * m[0] + c * m[1]])
        dev = math.atan2(float(nb[0] * m[1] - nb[1] * m[0]), float(nb @ m))
        return n_new, dev, best[0], span, float(q0 @ m)

    def reanchor_normal(self, n):
        res = self.laser_normal(n)
        if res is None:
            self.stage('NORMAL_ANCHOR', applied=False)
            return n
        n_new, dev, k, span, dist = res
        self.stage('NORMAL_ANCHOR', applied=True, dev_deg=round(math.degrees(dev), 2), inliers=k, span=round(span, 2),
                   dist=round(dist, 3))
        return np.r_[n_new, np.asarray(n, float)[2:]] if len(n) > 2 else n_new

    def place(self, bin_p, n=None):
        """The pluggable place stage (default: stand off the bin, lower the book into it, open)."""
        self.stage('PLACE', stage_impl=self.placer.name)
        return self.placer.run(PlaceTask(self.target, np.asarray(bin_p), None if n is None else np.asarray(n)))

    def run(self):
        self.start()
        plate, n, t = self.find_column()
        book, row, plate, n, t = self.find_book(plate, n, t)
        self.pick(book, row, n, t, plate)
        bin_p = self.find_bin(n)
        if self.anchor:
            n = self.reanchor_normal(n)             # the bin was seen after the turn: the place's axis must be too
        res = self.place(bin_p, n) or {}
        self.stage('DONE', confirmed=res.get('confirmed'), where=res.get('where'))
        try:                      # the book is released: a park failure must not end the run before the scorer sees it
            self.arm.park(canonical=False)   # the arm is in the bin: no canonical move. Settle (B-v52: 7 of 64 crashed here, 'Cannot reach raised startup waypoint', place lost)
            self.stage('PARKED')
        except Exception as e:
            self.stage('PARK_FAILED', why=f'{type(e).__name__}: {e}'[:100])
        self.r.wait(3.0)          # hold still: the book settles with nothing touching it. The gym ends an episode 10 s wall
                                  # after the solution exits, about 1.4 s sim at RTF 0.14, too short for the scorer's 1 s
                                  # fingers-off plus 1 s at rest when the park was still moving the arm near the book


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--digit', type=int, required=True); ap.add_argument('--label', required=True, help='red|green|blue|yellow or A|B|C|D')
    ap.add_argument('--model', default='/opt/model/det_v4_736x1280.onnx')
    ap.add_argument('--sim-time', action='store_true', help='use /clock (erc-gym simulation)')
    ap.add_argument('--pick', default='erc_bot.stages:ScriptedPick', help='pick stage, module:Class (see erc_bot/stages.py)')
    ap.add_argument('--place', default='erc_bot.stages:ScriptedPlace', help='place stage, module:Class')
    ap.add_argument('--pick-opt', action='append', default=[], help='key=value for the pick stage, repeatable')
    ap.add_argument('--place-opt', action='append', default=[], help='key=value for the place stage, repeatable')
    ap.add_argument('--out', default='', help='directory for mission.json (stage times, result, failure reason)')
    ap.add_argument('--word-model', default='', help='box detector for the white-book words (default: --model). With a pose '
                    '--model pass /opt/model/det_v4_736x1280.onnx here')
    ap.add_argument('--ort-threads', type=int, default=None, help='detector CPU threads (same as ERC_BOT_ORT_THREADS, 0 = all cores)')
    ap.add_argument('--params', default='', help='pick/place parameter file (erc_bot/params.py, lane G); absent = unchanged')
    ap.add_argument('--wall-params', default=WALL_PARAMS, help='lane G wall-book manoeuvre for the default pick (column 4 '
                    'wall books only, every other book runs ScriptedPick unchanged); "none" = no wall manoeuvre')
    a, _ = ap.parse_known_args(argv)
    if a.params:
        from .params import stage_args
        a.pick, a.place, a.pick_opt, a.place_opt = stage_args(a.params, a.pick, a.place, a.pick_opt, a.place_opt)
    elif a.wall_params and a.wall_params != 'none' and a.pick == 'erc_bot.stages:ScriptedPick':
        # G1 held out (heldout1.csv, s4, 14af39f): column 4 wall books 4.70 per attempt vs 2.33, grasps 13 of 20 vs 0.
        # Only the pick changes (wall knobs); the place keeps its own defaults.
        a.pick = 'erc_bot.wall:ParamPick'
        a.pick_opt = [f'params={a.wall_params}'] + list(a.pick_opt)
    if a.ort_threads is not None:
        import os
        os.environ['ERC_BOT_ORT_THREADS'] = str(a.ort_threads)
    rclpy.init()
    import signal                               # after rclpy.init, which installs its own SIGTERM handler
    def _stop(*_):                              # harness stop: record, then exit. Only the first one raises: a group kill
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # plus the supervisor's forward must not hit main()'s finally
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, signal.default_int_handler)   # Ctrl-C: KeyboardInterrupt, context stays valid for the stop
    live = Live(sim_time=a.sim_time)
    bot, rc = None, 1
    result, reason = 'error', ''
    try:
        live.ready()
        bot = Bot(live, a.model, a.digit, a.label, a.pick, a.place, a.pick_opt, a.place_opt, out=a.out, word_model=a.word_model or None)
        bot.run()
        result, rc = 'done', 0
    except Abort as e:
        live.say(f'FAILED: {e}')
        result, reason = 'aborted', str(e)
    except KeyboardInterrupt:
        result, reason = 'stopped', 'stopped by the harness (episode over)'
    except Exception as e:                       # a crash is a result too: the harness tables it
        import traceback; traceback.print_exc()
        result, reason, rc = 'crashed', f'{type(e).__name__}: {e}', 2
    finally:
        try:
            live.stop()
        except Exception:
            pass
        if bot is not None:
            bot.result, bot.reason = result, reason
            bot.save()
        elif a.out:
            import json, os
            os.makedirs(a.out, exist_ok=True)
            json.dump({'result': result, 'reason': reason or 'bot never started', 'last_stage': None, 'events': []},
                      open(os.path.join(a.out, 'mission.json'), 'w'))
        live.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())

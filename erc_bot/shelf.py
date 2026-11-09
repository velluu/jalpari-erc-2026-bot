"""Shelf model in odom from the five digit plates (flush with the shelf front, top row, z about 1.93 m in Phase 2).

No arena walls are assumed (the competition area is open): the plates give the shelf line, its inward normal and the
column of every digit. Column pitch in the official model is 1.035 m.
"""
import numpy as np

PLATE_Z = 1.93
PITCH = 1.035


class Shelf:
    """Plate sightings in odom, grouped in epochs. Odometry slips about 1 to 3 % of every leg and turn (and more of the
    range for a turn), so sightings from two poses disagree by that much in odom: one median over all of them stays on
    the earliest, most numerous sightings. new_epoch() after a base move starts a fresh group; plates() then returns
    the fresh plates as seen now and the older ones shifted by the fresh-minus-old offset of the plates seen in both
    (lane H, odom 3 %: the stale plate line overwrote the frontal re-look's depth in 4 of 8 held-out episodes)."""

    def __init__(self):
        self.obs = {}            # digit -> list of xyz in odom, this epoch
        self.past = {}           # digit -> xyz from earlier epochs, re-anchored at each new_epoch()

    def add(self, digit, p):
        if abs(p[2] - PLATE_Z) < 0.2:
            self.obs.setdefault(int(digit), []).append(np.asarray(p, float))

    def new_epoch(self):
        """Call after a base move, before looking at the plates again."""
        self.past = self.plates(min_obs=1)
        self.obs = {}

    def fresh(self, min_obs=2):
        """Plates seen this epoch only: median per digit, outliers (> 8 cm from the median) dropped."""
        out = {}
        for d, v in self.obs.items():
            a = np.array(v)
            if len(a) < min_obs:
                continue
            m = np.median(a, 0)
            a = a[np.linalg.norm(a - m, axis=1) < 0.08]
            if len(a) >= min_obs:
                out[d] = a.mean(0)
        return out

    def plates(self, min_obs=2):
        """Fresh plates, plus earlier ones shifted by the median fresh-minus-past offset (xy) of the digits in both."""
        cur = self.fresh(min_obs)
        if not self.past:
            return cur
        common = [d for d in cur if d in self.past]
        shift = np.zeros(3)
        if common:
            shift[:2] = np.median(np.array([cur[d][:2] - self.past[d][:2] for d in common]), 0)
        out = {d: p + shift for d, p in self.past.items()}
        out.update(cur)
        return out

    def fit(self, robot_xy):
        """Line through the plates. Returns centre, tangent t, inward normal n (from the robot side into the shelf)."""
        P = self.plates()
        if len(P) < 2:
            return None
        pts = np.array([p[:2] for p in P.values()])
        c = pts.mean(0)
        t = np.linalg.svd(pts - c)[2][0]
        n = np.array([-t[1], t[0]])
        if np.dot(c - np.asarray(robot_xy), n) < 0:
            n = -n
        resid = np.abs((pts - c) @ n).max()
        return {'c': c, 't': t, 'n': n, 'plates': P, 'resid': float(resid)}

    def column(self, digit, robot_xy):
        """Plate xyz, inward normal and tangent for a digit. None until that plate and one other are known."""
        f = self.fit(robot_xy)
        if f is None or digit not in f['plates']:
            return None
        return f['plates'][digit], f['n'], f['t'], f


# Column override (ERC_BOT_COLUMN, lane B2): the real plate "1" is never read as marker_1 (lane P, 30 Oct robot frames). The
# operator names the target's physical column (1-5, left to right facing the shelf); the shelf comes from the laser.
PANEL_BEHIND_PLATE = 0.47    # the laser's shelf line (back panel) behind the plate line: sim APPROACH_ANCHOR d0 1.594-1.625 m
                             # at the 1.15 m stand, median 1.618 (B2-urdf-pair30, 30 episodes)
SHELF_SPAN_MIN = 2.0         # metres of straight panel needed to call a laser line the shelf
SHELF_SPAN_MAX = 5.8         # the panel is 5 columns (5.2 m official); the gym's 10 m arena walls are longer


def laser_shelf_line(pts, robot_xy, near=0.4, far=7.0, tol=0.02, iters=800, seed=0):
    """The shelf as a straight run of laser points (no 0.9 m gap, 2 to 5.8 m long) in a 2-D scan (odom points). A run
    with structure 5 to 55 cm in front of it along 3 or more 0.5 m stretches (the gym: back panel behind open dividers)
    wins, else the longest run (the real shelf: a solid face 1.31 m ahead in the team's 30 Oct bag, nothing in front). A bin table is
    about 1 m, the gym's arena walls 10 m. Returns {'c': line point nearest the robot, 'n': unit normal from the robot into the shelf, 'left': unit
    tangent pointing left when facing the shelf, 'lo', 'hi': inlier extent along 'left' from 'c', 'k': inliers} or None."""
    pts = np.asarray(pts, float)
    r = np.linalg.norm(pts - np.asarray(robot_xy, float), axis=1)
    P = pts[(r > near) & (r < far)]
    if len(P) < 30:
        return None
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(iters):
        i, j = rng.choice(len(P), 2, replace=False)
        d = P[j] - P[i]; L = float(np.linalg.norm(d))
        if L < 0.5:
            continue
        m = np.array([-d[1], d[0]]) / L
        inl = np.abs((P - P[i]) @ m) < tol
        if inl.sum() < 30:
            continue
        idx = np.flatnonzero(inl)                 # the longest run of inliers without a 0.9 m gap (stray collinear hits
        a = (P[idx] - P[i]) @ (d / L)             # on far structures do not stretch the panel)
        o = np.argsort(a); a, idx = a[o], idx[o]
        cuts = np.flatnonzero(np.diff(a) > 0.9) + 1    # a 47 cm deep divider hides up to 0.8 m of panel at 60 deg
        runs = np.split(np.arange(len(a)), cuts)
        r_ = max(runs, key=len)
        a, idx = a[r_], idx[r_]
        k = len(idx)
        span = float(a[-1] - a[0]) if k else 0.0
        if k < 30 or not SHELF_SPAN_MIN <= span <= SHELF_SPAN_MAX:
            continue
        inl = np.zeros(len(P), bool); inl[idx] = True
        # The shelf's dividers and side panels stand in front of its panel (toward the robot): laser points 5 to 55 cm
        # in front of the line, within its extent. A wall has nothing there.
        s_ = (P - P[i]) @ m
        if (np.asarray(robot_xy, float) - P[i]) @ m < 0:
            s_ = -s_
        along = (P - P[i]) @ (d / L)
        fa = along[(s_ > 0.05) & (s_ < 0.55) & (along > a.min()) & (along < a.max())]
        h = np.histogram(fa, bins=np.arange(a.min(), a.max() + 0.5, 0.5))[0] if len(fa) else np.zeros(1)
        front = int((h >= 2).sum())               # 0.5 m stretches with points in front: dividers repeat along the panel,
        cands.append((span, k, front, inl))       # a person or a cart in front of a solid face fills one or two
    if not cands:
        return None
    # The gym's shelf: a back panel (run >= 4 m) behind open dividers (points in front in 3 or more 0.5 m stretches); its walls
    # have nothing in front. The real shelf (30 Oct bag, 110 scans): a solid face, the longest run.
    shelf_like = [c for c in cands if c[2] >= 3 and c[0] >= 4.0]
    best = max(shelf_like or cands, key=lambda c: (round(c[0], 1), c[1]))     # the longest run, then the most points
    Q = P[best[3]]
    q0 = Q.mean(0)
    t = np.linalg.svd(Q - q0)[2][0]
    n = np.array([-t[1], t[0]])
    if (q0 - np.asarray(robot_xy, float)) @ n < 0:
        n = -n
    left = np.array([-n[1], n[0]])                       # facing along n, +90 deg is left
    a = (Q - q0) @ left
    lo, hi = float(a.min()), float(a.max())               # inliers lie within tol of the line: the panel's ends
    return {'c': q0, 'n': n, 'left': left, 'lo': lo, 'hi': hi, 'k': int(best[1]), 'span': hi - lo, 'front': int(best[2])}


def override_plates(line, column, seen=(), pitch=PITCH):
    """Plate positions (xyz, odom) for the five physical columns from a laser shelf line, column 1 leftmost facing the
    shelf: centred on the run's middle, the official pitch (span / 5 read 1.03 to 1.08 in the gym, whose pitch is 1.0,
    and 1.12 on the real face). The plate line is the laser line itself for a solid face (the real shelf), and
    PANEL_BEHIND_PLATE in front of it for a panel behind open dividers (the gym). Any plate boxes seen (whatever digit
    they read as) are column-centre evidence: a column whose predicted centre has a seen plate within 0.35 m moves to
    it (median lateral and depth shift of all such matches). Returns ({column: xyz}, how)."""
    c, n, left = line['c'], line['n'], line['left']
    behind = PANEL_BEHIND_PLATE if line.get('front', 0) >= 3 else 0.0
    mid = c + 0.5 * (line['lo'] + line['hi']) * left - behind * n
    cols = {k: mid + (3 - k) * pitch * left for k in range(1, 6)}
    how = (f'laser line (span {line["span"]:.2f} m, {line.get("k")} points, {line.get("front")} stretches with dividers, plate line '
           f'{behind:.2f} m before it)')
    shifts = []
    for p in seen:
        p = np.asarray(p, float)[:2]
        k = min(cols, key=lambda j: np.linalg.norm(cols[j] - p))
        if np.linalg.norm(cols[k] - p) < 0.35:
            shifts.append(p - cols[k])
    if shifts:
        sh = np.median(np.array(shifts), 0)
        cols = {k: v + sh for k, v in cols.items()}
        how += f' + {len(shifts)} plate box(es), shift {np.round(sh, 3).tolist()}'
    return {k: np.r_[v, PLATE_Z] for k, v in cols.items()}, how


def frontal_stand(line, dist=2.6):
    """Where to look at the whole panel from: dist in front of the middle of the seen panel, facing it."""
    return line['c'] + 0.5 * (line['lo'] + line['hi']) * line['left'] - dist * line['n']

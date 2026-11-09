"""Lane G trial server: many scored picks in one Gazebo instance, for black-box optimisation of the pick.

Runs as the solution of an erc-gym pool episode (Round 2, any seed; the episode's own scorer target must be a book this
server never touches, so its scorer runs to its own timeout). The server finds the column once, records the settled
pose of every book in it, then serves trials from a shared queue until the queue says stop or its wall budget ends:

  queue/pending/<id>.json   {"id", "row" or "label", "params": "<path to params json>"}  (truth_fallback 0 by default:
                            a book the detector misses scores J 0, the pick is not run)
  queue/results/<id>.json   detector estimate error, scorer stages for this trial (Phase 2 scorer, gz ground truth),
                            collisions, hold after the carry posture, pick reason, sim and wall times

Before each trial the arm is opened and parked, the base goes back to the column stand-off (find_book does that) and
every book of the column is put back at its recorded pose with the gz set_pose service. Each trial gets a fresh
instance of the gym scorer, fed from this process's own gz subscriptions, so its stages and collisions are this pick's.

  python -m erc_bot._trials --queue Q --digit 4 --model det.onnx --out OUT [--max-wall 3300]
"""
import argparse, glob, importlib.util, json, math, os, socket, sys, threading, time, traceback
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.world_stats_pb2 import WorldStatistics
from erc_bot.live import Live, Abort
from erc_bot.mission import Bot
from erc_bot.stages import PickTask
from erc_bot.wall import ParamPick
from erc_bot.arm import transform

WORLD, ROBOT = 'erc_world', 'tiago_pro'


def load_scorer():
    gym = os.environ.get('GYM', '/opt/gym')
    spec = importlib.util.spec_from_file_location('erc_scorer', f'{gym}/gym/scorer.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Truth:
    """gz ground truth: full model poses, sim time, robot contacts routed to the current trial's scorer."""
    def __init__(self, scorer_mod):
        self.S = scorer_mod
        self.lock = threading.Lock()
        self.poses, self.sim_t, self.cur = {}, None, None
        self.node = GzNode()
        self.node.subscribe(Pose_V, f'/world/{WORLD}/dynamic_pose/info', self.on_pose)
        self.node.subscribe(Pose_V, f'/world/{WORLD}/pose/info', self.on_pose)
        self.node.subscribe(WorldStatistics, f'/world/{WORLD}/stats', self.on_stats)
        topics, t0 = [], time.time()
        while time.time() - t0 < 30 and len(topics) < 30:
            topics = [t for t in self.node.topic_list() if t.startswith(f'/world/{WORLD}/model/{ROBOT}/link/') and t.endswith('_contact/contact')]
            time.sleep(0.5)
        for t in topics:
            self.node.subscribe(Contacts, t, self.on_contacts)
        self.n_contact_topics = len(topics)
        self.stop = False
        threading.Thread(target=self.stepper, daemon=True).start()

    def on_pose(self, msg):
        with self.lock:
            for p in msg.pose:
                o = p.orientation
                self.poses[p.name] = (np.array([p.position.x, p.position.y, p.position.z]), np.array([o.x, o.y, o.z, o.w]))
        s = self.cur
        if s is not None:
            s.on_pose(msg)

    def on_stats(self, msg):
        with self.lock:
            self.sim_t = msg.sim_time.sec + msg.sim_time.nsec * 1e-9
        s = self.cur
        if s is not None:
            s.on_stats(msg)

    def on_contacts(self, msg):
        s = self.cur
        if s is not None:
            s.on_robot_contacts(msg)

    def stepper(self):
        while not self.stop:
            s = self.cur
            if s is not None:
                with s.lock:
                    s.step()
            time.sleep(0.02)

    def pose(self, name):
        with self.lock:
            xyz, q = self.poses[name]
        m = np.eye(4); m[:3, :3] = Rotation.from_quat(q).as_matrix(); m[:3, 3] = xyz
        return m

    def raw(self, name):
        with self.lock:
            return tuple(np.copy(v) for v in self.poses[name])

    def set_pose(self, name, xyz, q):
        p = Pose(); p.name = name
        p.position.x, p.position.y, p.position.z = map(float, xyz)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
        ok, rep = self.node.request(f'/world/{WORLD}/set_pose', p, Pose, Boolean, 3000)
        return bool(ok and rep.data)


class Palm:
    """The palm (gripper_right_base_link) has no contact sensor, so the scorer cannot see it touch the shelf. Proxy: four
    spheres along the gripper axis (0 to 9 cm from the base link toward the fingers, radius 4 cm) checked against the
    shelf collision boxes (erc-gym collision/shelf.json, shelf model pose from gz). An event is a run of penetrating
    samples, a new one after 0.2 s clear, as the scorer counts contacts."""
    S_AX = (0.0, 0.03, 0.06, 0.09); RAD = 0.04

    def __init__(self, truth):
        gym = os.environ.get('GYM', '/opt/gym')
        boxes = json.load(open(f'{gym}/collision/shelf.json'))['boxes']
        self.c = np.array([b['center'] for b in boxes]); self.h = np.array([b['size'] for b in boxes]) / 2
        try:
            M = truth.pose('erc_shelf')
        except Exception:
            M = np.eye(4); M[:3, :3] = Rotation.from_euler('xyz', (1.5708, 0, -1.5708)).as_matrix(); M[:3, 3] = (3.0, 0.0, 1.1)
        self.Minv = np.linalg.inv(M)
        self.reset()

    def reset(self):
        self.events, self.last_hit, self.min_clear, self.n = [], -1e9, 1e9, 0

    def check(self, W_gb, t):
        pts = np.array([(W_gb @ np.r_[0, 0, s, 1.0])[:3] for s in self.S_AX])
        pm = (self.Minv @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        q = np.abs(pm[:, None, :] - self.c[None]) - self.h[None]
        d = np.linalg.norm(np.maximum(q, 0), axis=2) + np.minimum(q.max(axis=2), 0)   # signed distance, point to box
        clear = float(d.min()) - self.RAD
        self.min_clear = min(self.min_clear, clear); self.n += 1
        if clear < 0:
            if t - self.last_hit > 0.2:
                i = np.unravel_index(np.argmin(d), d.shape)
                self.events.append({'t': round(t, 2), 'depth': round(-clear, 3), 'sphere': int(i[0]), 'box': int(i[1])})
            self.last_hit = t


def scorer_for(S, layout_path, digit, label):
    a = argparse.Namespace(layout=layout_path, digit=digit, label=label, round=2, timeout=1e9, band=0.45,
                           gentle_speed=0.5, out=None, wall_timeout=1e9)
    return S.Scorer(a)


class Server:
    def __init__(self, a):
        self.a = a
        # --queue PATH[:DIGIT],PATH[:DIGIT]...: trials are claimed from the queues in this order, each queue's trials are
        # picks in its digit's column (default --digit). Status files go to the first queue.
        self.queues = []
        for item in a.queue.split(','):
            path, _, d = item.partition(':')
            self.queues.append((path, int(d) if d else a.digit))
        self.q = self.queues[0][0]
        self.tag = f'{socket.gethostname()}_{os.environ.get("SLOT", "0")}_{os.getpid()}'
        self.t_wall0 = time.time()
        for q, _ in self.queues:
            for d in ('results', 'running', 'pending'):
                os.makedirs(f'{q}/{d}', exist_ok=True)
        os.makedirs(f'{self.q}/servers', exist_ok=True)
        self.S = load_scorer()
        self.truth = Truth(self.S)
        self.layout_path = f'{a.out}/layout.json'
        self.layout = json.load(open(self.layout_path))
        self.cols = {}
        rclpy.init()
        self.r = Live('erc_trials', sim_time=True)
        self.r.ready()
        self.bot = Bot(self.r, a.model, a.digit, 'red', out=f'{a.out}/server')
        self.palm = Palm(self.truth)
        self.palm_on, self.W_O, self._tick = False, None, self.r.tick
        def tick(dt=0.02):            # palm proxy sampled on the bot's own spin, every 0.1 s sim during a pick
            self._tick(dt)
            if self.palm_on and self.W_O is not None:
                t = self.r.now()
                if t - getattr(self, '_palm_t', -1) >= 0.1:
                    self._palm_t = t
                    try:
                        self.palm.check(self.W_O @ self.r.odom_T_base() @ self.r.matrix('base_footprint', 'gripper_right_base_link'), t)
                    except Exception:
                        pass
        self.r.tick = tick
        self.status('starting', contact_topics=self.truth.n_contact_topics)

    def status(self, state, **kw):
        rec = {'tag': self.tag, 'state': state, 'wall': round(time.time() - self.t_wall0, 1), 'sim': self.truth.sim_t,
               'out': self.a.out, **kw}
        json.dump(rec, open(f'{self.q}/servers/{self.tag}.json', 'w'), default=str)

    # ---------------------------------------------------------------- setup and reset
    def setup(self):
        b = self.bot
        b.start()
        self.torso0 = self.r.joints['torso_lift_joint']
        self.q_park = self.bot.arm.q(self.bot.arm.k.names)        # parked joints, the last-resort reset target
        b.digit = self.a.digit
        plate, n, t = b.find_column()
        self.cols[self.a.digit] = (plate, n, t)
        self.r.wait(1.0)
        self.rest = {x['name']: self.truth.raw(x['name']) for x in self.layout['books']}
        self.use(self.a.digit)
        self.status('ready', plate=np.round(self.plate, 3).tolist())

    def use(self, digit):
        """Work in this digit's column: plate, normal and tangent (from the plates seen so far, else a column search)."""
        b = self.bot
        if digit not in self.cols:
            b.digit = digit
            try:
                self.cols[digit] = b.refine_column()
            except Exception:
                self.cols[digit] = b.find_column()
        self.digit = b.digit = digit
        self.plate, self.n, self.t = self.cols[digit]
        self.col = self.layout['markers'].index(digit) + 1
        self.col_books = {x['name']: x for x in self.layout['books'] if x['column'] == self.col}

    def reset(self):
        b, r, arm = self.bot, self.r, self.bot.arm
        notes = []
        try:
            r.gripper(False)
        except Abort as e:
            notes.append(f'open: {e}')
        try:
            tool = arm.tool()
            if tool[0, 3] > 0.30:
                g = tool.copy(); g[0, 3] = max(0.25, g[0, 3] - 0.20)
                arm.pose(g, duration=2.0, pos_tol=0.2, ang_tol=3.0)
        except Exception as e:
            notes.append(f'arm back: {str(e)[:60]}')
        try:
            x, y, yaw = r.base_in_odom()
            d = float((self.plate[:2] - np.array([x, y])) @ self.n)   # base to plate (shelf front) along the normal
            if d < 1.0:
                back = np.array([x, y]) - (1.0 - d) * self.n
                b.go(back[0], back[1], yaw, clearance=0.05, timeout=30)
        except Exception as e:
            notes.append(f'base back: {str(e)[:60]}')
        try:
            r.torso(self.torso0)
        except Abort as e:
            notes.append(f'torso: {e}')
        for attempt in range(4):
            if attempt == 3:        # last resort: straight to the parked joints recorded at the start
                try:
                    r.send('arm_right_controller', self.q_park, duration=6.0, tolerance=0.05); break
                except Exception as e:
                    notes.append(f'joints: {str(e)[:60]}'); continue
            try:
                arm.park(); break
            except Exception as e:
                notes.append(f'park {attempt}: {str(e)[:60]}'); r.wait(1.0)
                try:    # from an odd pose: the upright carry pose first (park works from there), via a raised tool
                    tool = arm.tool()
                    g = tool.copy(); g[2, 3] = max(g[2, 3], 1.15); g[0, 3] = min(g[0, 3], 0.55)
                    arm.pose(transform(g[:3, 3]), duration=3.0, pos_tol=0.3, ang_tol=3.0)
                    arm.pose(transform([0.50, -0.24, 1.28]), duration=3.0, pos_tol=0.3, ang_tol=3.0)
                except Exception as e2:
                    notes.append(f'carry pose: {str(e2)[:60]}')
        else:
            raise RuntimeError('park failed four times: ' + '; '.join(notes))
        try:
            r.torso(self.torso0)
        except Abort as e:
            notes.append(f'torso: {e}')
        for _ in range(3):
            for nm, (xyz, q) in self.rest.items():
                cur = self.truth.raw(nm)
                if np.linalg.norm(cur[0] - xyz) > 0.003 or abs(abs(float(cur[1] @ q)) - 1) > 1e-4:
                    self.truth.set_pose(nm, xyz, q)
            r.wait(1.0)
            off = max(np.linalg.norm(self.truth.raw(nm)[0] - xyz) for nm, (xyz, q) in self.rest.items())
            if off < 0.005:
                break
        notes.append(f'books off {off:.4f} m')
        return notes

    # ---------------------------------------------------------------- one trial
    def trial(self, spec):
        b, r = self.bot, self.r
        res0 = self.reset()
        self.use(int(spec.get('digit', self.digit)))
        if 'row' in spec:      # the colour at this row in this server's layout (layout rows are shelf rows 2-5)
            label = next(x['name'].rsplit('_', 1)[1] for x in self.col_books.values() if x['row'] == int(spec['row']) + 1)
        else:
            label = spec['label']
        res = {'id': spec['id'], 'label': label, 'params': spec['params'], 'server': self.tag}
        w0, s0 = time.time(), self.truth.sim_t
        res['reset'] = res0; res['digit'] = self.digit
        sc = scorer_for(self.S, self.layout_path, self.digit, label)
        target_name = sc.target
        res['target'] = target_name
        b.target = f'book_{label.lower()}'
        W_O = self.truth.pose(ROBOT) @ np.linalg.inv(r.odom_T_base())          # world <- odom
        O_W = np.linalg.inv(W_O)
        rest_xyz, rest_q = self.rest[target_name]
        n_w = (W_O[:3, :3] @ np.r_[self.n, 0.0])[:2]
        spine_true_w = rest_xyz.copy(); spine_true_w[:2] -= 0.08 * n_w / np.linalg.norm(n_w)
        found = True
        try:
            book, row, plate, n, t = b.find_book(self.plate, self.n, self.t)
            self.plate, self.n, self.t = plate, n, t
        except Abort as e:
            found = False; res['find'] = str(e)
            if not int(spec.get('truth_fallback', 0)):   # default 0: a pick on a truth position says nothing about the robot
                res.update(found=False, J=0.0, wall=round(time.time() - w0, 1)); return res
            book = (O_W @ np.r_[spine_true_w, 1.0])[:3]
            row = self.col_books[target_name]['row'] - 1
            plate, n, t = self.plate, self.n, self.t
        res['found'] = found
        res['row'] = int(row)
        est_w = (W_O @ np.r_[book, 1.0])[:3]
        res['est_err'] = np.round(est_w - spine_true_w, 4).tolist()
        sc.t0 = None
        self.truth.cur = sc
        self.W_O = W_O; self.palm.reset(); self.palm_on = True
        n_ev0 = len(b.events)
        pick = ParamPick(b, params=spec['params'])
        ps0 = self.truth.sim_t
        orig_stage = b.stage
        def stage(name, **kw):      # every sub-stage also logs where the target book truly is (world, from its rest pose)
            try:
                xyz, q = self.truth.raw(target_name)
                dyaw = (Rotation.from_quat(q) * Rotation.from_quat(rest_q).inv()).as_rotvec()
                kw['bk'] = np.round(np.r_[xyz - rest_xyz, dyaw], 3).tolist()
            except Exception:
                pass
            orig_stage(name, **kw)
        b.stage = stage
        try:
            pick.run(PickTask(b.target, np.asarray(book), int(row), np.asarray(n), np.asarray(t), np.asarray(plate)))
            res['pick'] = 'ok'
        except Abort as e:
            res['pick'] = f'abort: {e}'
        except Exception as e:
            res['pick'] = f'crash: {type(e).__name__}: {e}'; res['trace'] = traceback.format_exc()[-800:]
        b.stage = orig_stage
        self.palm_on = False
        res['pick_sim'] = round((self.truth.sim_t or 0) - (ps0 or 0), 1)
        r.wait(1.0)
        bk = self.truth.pose(target_name)[:3, 3]
        tool_w = (W_O @ r.odom_T_base() @ b.arm.tool())[:3, 3]
        # held: the book is at the gripper and out of the shelf (rows 1 and 2 stay at test-lift height in the carry posture,
        # so height alone does not tell)
        held = bool(np.linalg.norm(bk - tool_w) < 0.15 and np.linalg.norm(bk[:2] - rest_xyz[:2]) > 0.15)
        self.truth.cur = None
        with sc.lock:
            rs = sc.result(time.time() - w0)
        res.update(held=held, held_rule='v2', book_to_tool=round(float(np.linalg.norm(bk - tool_w)), 3), book_z_gain=round(float(bk[2] - rest_xyz[2]), 3),
                   stages=list(rs['stages']), collisions=rs['collisions'], max_lift=rs['target_max_lift_held'],
                   max_disp=rs['target_max_disp_held'], bilateral=rs['bilateral_seen'], wrong_book=rs['wrong_book_grasped'],
                   events=[{k: v for k, v in e.items() if k in ('stage', 't', 'blocked', 'reached', 'position', 'why', 'moved', 'bk')}
                           for e in b.events[n_ev0:]])
        grasp = 'grasp' in rs['stages']
        m = rs['multiplier']
        res.update(palm_events=self.palm.events, palm_min_clear=round(self.palm.min_clear, 3), palm_samples=self.palm.n)
        res['J'] = round(2 * m * grasp + 2 * m * (grasp and held) - 0.5 * len(rs['collisions']) - 0.5 * len(self.palm.events)
                         + (0.0 if grasp else 0.3 * bool(rs['bilateral_seen']) + 0.3 * min(1.0, rs['target_max_lift_held'] / 0.04)), 3)
        res['wall'] = round(time.time() - w0, 1); res['sim'] = round((self.truth.sim_t or 0) - (s0 or 0), 1)
        return res

    def claim(self):
        for q, digit in self.queues:
            spec, run = self.claim_in(q)
            if spec is not None:
                spec.setdefault('digit', digit)
                return spec, run, q
        return None, None, None

    def claim_in(self, q):
        for f in sorted(glob.glob(f'{q}/pending/*.json')):
            dst = f'{q}/running/{os.path.basename(f)[:-5]}.{self.tag}'
            try:
                os.rename(f, dst)
                os.utime(dst)          # claim time: the optimiser requeues trials whose claim is stale
            except OSError:
                continue
            try:
                return json.load(open(dst)), dst
            except Exception:
                os.remove(dst); continue
        return None, None

    def serve(self):
        n = 0
        while True:
            if any(os.path.exists(f'{q}/STOP') for q, _ in self.queues[:1]) or time.time() - self.t_wall0 > self.a.max_wall:
                self.status('stopped', trials=n); return
            spec, run, q = self.claim()
            if spec is None:
                self.status('idle', trials=n)
                t = time.time()
                while time.time() - t < 15:
                    self.r.tick(0.1)
                continue
            self.status('trial', id=spec['id'], trials=n)
            try:
                res = self.trial(spec)
            except Exception as e:
                res = {'id': spec['id'], 'label': spec.get('label'), 'params': spec.get('params'), 'server': self.tag,
                       'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1200:]}
                tmp = f'{q}/results/.{spec["id"]}.{self.tag}'
                json.dump(res, open(tmp, 'w'), indent=1, default=str)
                # a server error is not the candidate's result: requeue the trial and leave
                try:
                    os.rename(run, f'{q}/pending/{spec["id"]}.json')
                except OSError:
                    pass
                os.rename(tmp, f'{q}/results/err_{spec["id"]}_{self.tag}.json')
                self.status('broken', error=res['error'], trials=n)
                raise
            tmp = f'{q}/results/.{spec["id"]}.{self.tag}'
            json.dump(res, open(tmp, 'w'), indent=1, default=str)
            os.rename(tmp, f'{q}/results/{spec["id"]}.json')
            try:
                os.remove(run)
            except OSError:      # requeued as stale while it ran: drop the duplicate
                try:
                    os.remove(f'{q}/pending/{spec["id"]}.json')
                except OSError:
                    pass
            n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True); ap.add_argument('--digit', type=int, default=4)
    ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-wall', type=float, default=3000.0)
    a = ap.parse_args()
    s = Server(a)
    try:
        s.setup()
        s.serve()
    except Exception as e:
        traceback.print_exc()
        try:
            s.status('crashed', error=f'{type(e).__name__}: {e}')
        except Exception:
            pass
        return 2
    finally:
        s.truth.stop = True
    return 0


if __name__ == '__main__':
    sys.exit(main())

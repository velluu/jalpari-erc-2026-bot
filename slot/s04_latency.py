#!/usr/bin/env python3
"""Slot step 4: command-to-motion latency and tracking lag, action vs joint_trajectory topic, per controller.

  python3 slot/s04_latency.py [--only head,torso,grip,arm] [--arm-joints 7,4,2] [--reps 1]

Every motion is small, inside the speed caps and the joint ranges less 0.07, and returns to where it started:
  head pan 0.30 rad, torso 3 cm, right arm joint 7 0.40 rad (joints 4 and 2 0.15 / 0.12 rad only when the right
  gripper is out in front, i.e. after `s02_gripper.py present`), right gripper open-close on air.
Per joint and mode:
  action   one-point follow_joint_trajectory goal (the bot's way), ramp at <= 50 % of the cap
  topic    the same one-point trajectory on /erc/<c>/joint_trajectory
  ol10/ol20  streamed ramp at 10 / 20 Hz, each message one point `dt` ahead on the intended path, <= 50 % of the cap
  cl10/cl20  lane A's law at 10 / 20 Hz: target = measured + 0.9 * cap * dt toward the goal (erc-learn env_pick)
Measured (laptop receive clock, node clock in sim): onset latency (first /joint_states move beyond noise, minus the
time the ideal ramp needs to cover that threshold), tracking lag (time shift that best fits the measured trace to the
commanded path), RMS residual after that shift, achieved speed against commanded, final error, rejections and the
action accept / result times. Writes latency.json and latency_traces.json to $SLOT_OUT.
"""
import argparse, math
import numpy as np
from common import start, finish, save, SIM, CAP, CONTROLLERS, GUIDE
from rclpy.time import Time as RTime

AMP = {'head_1_joint': 0.30, 'torso_lift_joint': 0.03, 'arm_right_7_joint': 0.40, 'arm_right_4_joint': 0.15,
       'arm_right_2_joint': 0.12, 'gripper_right_finger_joint': 0.79}
CTRL = {'head_1_joint': 'head_controller', 'torso_lift_joint': 'torso_controller', 'gripper_right_finger_joint': 'gripper_right_controller'}
CTRL.update({f'arm_right_{j}_joint': 'arm_right_controller' for j in range(1, 8)})


def direction(j, q0, amp):
    """Move toward the middle of the live range, so the excursion stays inside the margins."""
    lo, hi = GUIDE.get(j, (-math.inf, math.inf))
    if 'gripper' in j:
        return 1.0 if q0 < 0.4 else -1.0
    room_up, room_dn = hi - 0.07 - q0, q0 - (lo + 0.07)
    if room_up >= amp:
        return 1.0
    if room_dn >= amp:
        return -1.0
    return 0.0


class Run:
    def __init__(self, n, v):
        self.n, self.v, self.rows, self.traces = n, v, [], {}

    def hold(self, ctrl, j):
        names = CONTROLLERS[ctrl]
        return names, [self.n.joints[x] for x in names], names.index(j)

    def record(self, secs):
        self.n.js_log = []
        self.n.spin_for(secs)

    def analyse(self, key, j, t0, q0, path, T, mode, extra):
        """path(t) -> intended position at time t after t0 (the ideal ramp); T its duration."""
        log = self.n.js_log; self.n.js_log = None
        pre = np.array([p[j] for t, _, p in log if t < t0 and j in p])
        tt = np.array([t - t0 for t, _, p in log if t >= t0 and j in p]); qq = np.array([p[j] for t, _, p in log if t >= t0 and j in p])
        noise = float(pre.std()) if len(pre) > 5 else 0.0
        amp = abs(path(T) - q0)
        thr = max(0.003 if 'torso' not in j else 0.0008, 6 * noise, 0.02 * amp)
        mv = np.nonzero(np.abs(qq - q0) > thr)[0]
        onset = float(tt[mv[0]]) if len(mv) else None
        v_cmd = amp / T if T > 0 else None
        onset_corr = (onset - thr / v_cmd) if onset is not None and v_cmd else onset
        # lag: shift tau that best fits q(t) ~ path(t - tau) over [0, T + 1.5]
        w = tt <= T + 1.5
        best = (None, None)
        if w.sum() > 5 and amp > 1e-4:
            for tau in np.arange(0.0, 1.0001, 0.005):
                ref = np.array([path(max(0.0, x - tau)) for x in tt[w]])
                e = float(np.sqrt(np.mean((qq[w] - ref) ** 2)))
                if best[0] is None or e < best[1]:
                    best = (float(tau), e)
        # achieved speed on the middle 60 % of the travel, against the commanded speed
        sp = None
        if amp > 1e-4 and len(qq) > 5:
            prog = np.abs(qq - q0) / amp
            m = (prog > 0.2) & (prog < 0.8)
            if m.sum() >= 3:
                sp = float(abs(np.polyfit(tt[m], qq[m], 1)[0]))
        final = float(qq[-1]) if len(qq) else None
        row = {'key': key, 'joint': j, 'mode': mode, 'amp': round(amp, 4), 'T_cmd': round(T, 3),
               'v_cmd': round(v_cmd, 4) if v_cmd else None, 'v_meas': round(sp, 4) if sp else None,
               'v_ratio': round(sp / v_cmd, 3) if sp and v_cmd else None, 'noise': noise, 'thr': thr,
               'onset_raw_s': round(onset, 4) if onset is not None else None,
               'latency_s': round(onset_corr, 4) if onset_corr is not None else None,
               'lag_s': round(best[0], 3) if best[0] is not None else None,
               'rms_after_shift': round(best[1], 5) if best[1] is not None else None,
               'final_err': round(abs(final - path(T)), 4) if final is not None else None, 'n_samples': int(len(qq)),
               'js_hz': round(len(qq) / (tt[-1] - tt[0]), 1) if len(tt) > 2 else None, **extra}
        self.rows.append(row)
        self.traces[key] = {'t': np.round(tt, 4).tolist(), 'q': np.round(qq, 5).tolist(), 'q0': q0, 'pre_noise': noise}
        return row

    def back(self, ctrl, names, q_hold):
        r = self.n.move(ctrl, q_hold, frac=0.4)
        self.n.spin_for(0.5)
        return r

    def test(self, j, mode, rep=0):
        n = self.n
        ctrl = CTRL[j]
        names, hold, i = self.hold(ctrl, j)
        lo, hi = GUIDE.get(j, (-math.inf, math.inf))
        if 'gripper' not in j and not (lo + 0.07 <= hold[i] <= hi - 0.07):
            # a joint resting inside the checking layer's margin (the torso starts at 0.0, bound 0.05): every goal that
            # keeps it there is refused, so first move it just inside the allowed range (dry run 21:40)
            tgt = hold[:]; tgt[i] = min(max(hold[i], lo + 0.08), hi - 0.08)
            r = n.move(ctrl, tgt, frac=0.5)
            n.spin_for(0.5)
            self.v.info(f'{j} started at {hold[i]:.3f}, inside the margin: moved to {tgt[i]:.3f} first (result {r.get("error_code")})')
            names, hold, i = self.hold(ctrl, j)
        q0 = hold[i]
        amp = AMP[j]
        s = direction(j, q0, amp)
        if s == 0.0:
            self.v.check(f'{j} {mode}', False, f'no room for {amp} inside the range at {q0:.3f}', 'move the joint toward mid-range first')
            return
        grip = 'gripper' in j
        cap = CAP.get(j, 1.0)
        frac = 0.5 if not grip else 1.0
        T = max(1.0, amp / (frac * cap)) if not grip else 1.5
        q1 = q0 + s * amp
        path = lambda t: q0 + (q1 - q0) * min(1.0, max(0.0, t / T))
        key = f'{j}:{mode}:{rep}'
        n_rej = len(n.rejections)
        self.record(0.6)
        extra = {}
        if mode == 'action':
            pts = hold[:]; pts[i] = q1
            r = n.send_goal(ctrl, names, [(T, pts)], timeout=T + 15)
            r.pop('handle', None)
            t0 = r['t_send']
            extra = {'accepted': r['accepted'], 'error_code': r.get('error_code'),
                     'accept_s': round(r['t_accept'] - t0, 4) if 't_accept' in r else None,
                     'result_after_T_s': round(r['t_result'] - t0 - T, 4) if 't_result' in r else None}
            n.spin_for(1.0)
        elif mode == 'topic':
            pts = hold[:]; pts[i] = q1
            t0 = n.now(); n.topic(ctrl).publish(n.traj(names, [(T, pts)]))
            n.spin_for(T + 1.5)
        else:
            hz = 10 if mode.endswith('10') else 20
            dt = 1.0 / hz
            pub = n.topic(ctrl)
            t0 = n.now(); k = 0; sent = 0
            closed = mode.startswith('cl')
            while True:
                tk = t0 + k * dt
                n.spin_until(lambda: n.now() >= tk, dt + 1.0, dt=0.002)
                el = n.now() - t0
                meas = n.joints[j]
                if closed:
                    if abs(q1 - meas) < 0.01 * amp + 1e-4 or el > 4 * T + 3:
                        break
                    step = math.copysign(min(0.9 * cap * dt, abs(q1 - meas)), q1 - meas)   # gripper: lane A's ~0.8 s full travel
                    tgt = meas + step
                else:
                    if el > T + dt:
                        break
                    tgt = path(el + dt)
                pts = hold[:]; pts[i] = tgt
                pub.publish(n.traj(names, [(dt, pts)])); sent += 1; k += 1
            if closed:      # lane A's law: the intended path is the cap-limited ramp
                T = amp / (0.9 * cap)
                path = lambda t, T=T: q0 + (q1 - q0) * min(1.0, max(0.0, t / T))
            extra = {'hz': hz, 'sent': sent, 'stream_s': round(n.now() - t0, 3)}
            n.spin_for(1.0)
        rej = [x for _, x in n.rejections[n_rej:]]
        extra['rejections'] = len(rej); extra['rejection_text'] = rej[:3]
        row = self.analyse(key, j, t0, q0, path, T, mode, extra)
        if mode in ('action', 'topic'):     # the bot's way of commanding: must move and arrive
            ok = row['onset_raw_s'] is not None and (row['final_err'] is None or row['final_err'] < max(0.02, 0.1 * amp))
        else:                               # streams are measurements: dropped messages are the finding, not a failure
            ok = row['onset_raw_s'] is not None or row['rejections'] > 0
        self.v.check(key, ok,
                     (f"MEASURED {row['rejections']} of {row.get('sent')} streamed messages dropped, " if mode not in ('action', 'topic') else '') +
                     f"latency {row['latency_s']} s, lag {row['lag_s']} s (rms {row['rms_after_shift']}), speed {row['v_meas']} of "
                     f"{row['v_cmd']} ({row['v_ratio']}), final err {row['final_err']}, rejections {row['rejections']}"
                     + (f", accept {row.get('accept_s')} s, result {row.get('result_after_T_s')} s after T" if mode == 'action' else '')
                     + (f", sent {row.get('sent')} msgs" if 'sent' in row else ''),
                     'did not move or did not arrive: read the rejection text (the head/arm returns to start by itself), carry on', **{k: row[k] for k in
                     ('latency_s', 'lag_s', 'v_ratio', 'final_err', 'rejections')})
        self.back(ctrl, names, hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='head,torso,grip,arm')
    ap.add_argument('--arm-joints', default='7,4,2')
    ap.add_argument('--modes', default='action,topic,ol10,ol20,cl10,cl20')
    ap.add_argument('--reps', type=int, default=1)
    a = ap.parse_args()
    n, v = start('s04_latency')
    from tf2_ros import Buffer, TransformListener
    import rclpy
    tf = Buffer(); TransformListener(tf, n)
    if not v.check('joints', n.wait_joints(list(AMP), 10.0), 'all test joints on /joint_states', 'rerun s00_preflight'):
        return finish(n, v)
    n.spin_for(1.0)
    modes = a.modes.split(',')
    joints = []
    only = a.only.split(',')
    if 'head' in only: joints.append('head_1_joint')
    if 'torso' in only: joints.append('torso_lift_joint')
    if 'arm' in only:
        out_front = False
        try:
            t = tf.lookup_transform('base_footprint', 'gripper_right_grasping_link', RTime()).transform.translation
            out_front = t.x > 0.45 and t.z > 0.6
            v.info(f'right gripper at ({t.x:.2f}, {t.y:.2f}, {t.z:.2f}) in base_footprint')
        except Exception as e:
            v.info(f'no TF to the right gripper ({e})')
        for k in a.arm_joints.split(','):
            if k == '7' or out_front:
                joints.append(f'arm_right_{k}_joint')
            else:
                v.info(f'arm joint {k} skipped: the gripper is not out in front (run s02_gripper.py present first)')
    if 'grip' in only: joints.append('gripper_right_finger_joint')
    R = Run(n, v)
    for j in joints:
        for m in modes:
            if 'gripper' in j and m in ('ol20', 'cl20'):
                continue
            for rep in range(a.reps):
                try:
                    R.test(j, m, rep)
                except Exception as e:
                    v.check(f'{j}:{m}', False, f'{type(e).__name__}: {e}', 'skip this joint, carry on with the next step')
    # summary table
    print(f"{'joint':28s} {'mode':6s} {'lat s':>7s} {'lag s':>6s} {'v/vcmd':>6s} {'ferr':>7s} {'rej':>4s}")
    for r in R.rows:
        print(f"{r['joint']:28s} {r['mode']:6s} {str(r['latency_s']):>7s} {str(r['lag_s']):>6s} {str(r['v_ratio']):>6s} "
              f"{str(r['final_err']):>7s} {r['rejections']:>4d}")
    save('latency.json', {'rows': R.rows, 'sim': SIM})
    save('latency_traces.json', R.traces)
    return finish(n, v)


if __name__ == '__main__':
    raise SystemExit(main())

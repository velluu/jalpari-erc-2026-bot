"""Same-state replay of the place teacher in lane A's replica (C MuJoCo, lane A's _obs and _reward verbatim, the
numpy policy, PickEnv.step's command logic and controller emulation), from the exact start states the policy met in
Gazebo (erc_bot/policy_probe.py records, OUT/policy_place.jsonl). Separates the causes of a Gazebo failure:

  V0 exact     the Gazebo start state (base, joints, book and bin poses from gz truth), fingers on the book
  V1 latency   V0 with the Gazebo-measured 75 ms command delay
  V2 hold      V0 with the book re-seated at lane A's nominal hold (pads 3 cm behind the spine, centred, upright)
  V3 openloop  V0 driven by the commands Gazebo actually streamed (targets, gripper, twist), no policy
  control      lane A's nominal place start (bin at (-1, 0), base at the standoff, q_hold, nominal hold), no DR

Also reports the Gazebo hold in lane A's hold parameters (pad depth behind the spine, height offset, tilt).

  cd ~/erc/policy/bot && JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 PYTHONPATH=~/erc/learn:. \
      ~/venvs/erclearn/bin/python tests/policy_replay_mj.py --checkpoint ../exports/place_t1_10M.npz --eps EP ... --out replay.json
"""
import argparse, json, math, os, sys, types
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--eps', nargs='+', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--steps', type=int, default=150)
    ap.add_argument('--control-n', type=int, default=8)
    ap.add_argument('--torso-hold', type=float, default=None, help='control start torso (checkpoints trained before 18:51: 0.345)')
    ap.add_argument('--torso-max', type=float, default=None, help='command range torso max (before 18:51: 0.345)')
    ap.add_argument('--q-hold-1830', action='store_true', help='control arm posture: place_start.json q_hold as of 18:30 (torso 0.345)')
    a = ap.parse_args()
    from mujoco import mjx
    mjx.put_model = lambda m, impl=None: m
    import mujoco, jax.numpy as jp
    from erc_learn import env_place
    from erc_learn.build_model import INNER_MUL
    from erc_bot import policy_obs as P
    pol = P.NumpyPolicy(a.checkpoint)
    meta_env = pol.meta.get('env', {})
    env = env_place.PlaceEnv(env_place.default_config(), config_overrides={'privileged_actor': True}, nenv_hint=1)
    m = env.mj_model
    cfg = env._config
    nsub = int(round(cfg.ctrl_dt / cfg.sim_dt))
    q_closed = float(env._q_closed)
    pad_in_grasp = np.array(env._pad_in_grasp)

    def fake(d):
        return types.SimpleNamespace(qpos=jp.array(d.qpos), qvel=jp.array(d.qvel), xpos=jp.array(d.xpos),
                                     xmat=jp.array(d.xmat.reshape(-1, 3, 3)), site_xpos=jp.array(d.site_xpos), sensordata=jp.array(d.sensordata))

    def set_finger(d, v):
        d.qpos[env._qf] = INNER_MUL * v
        for e in range(m.neq):
            j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
            d.qpos[m.jnt_qposadr[j1]] = m.eq_data[e, 1] * d.qpos[m.jnt_qposadr[j2]]

    def wxyz(q):
        return np.array([q[3], q[0], q[1], q[2]])

    def mat2quat(R):
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1)); return q

    def roty(t):
        c, s = math.cos(t), math.sin(t); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def grasp_frame(d):
        b = env._b_grasp; return d.xmat[b].reshape(3, 3).copy(), d.xpos[b].copy()

    def hold_params(d):
        """Book pose in lane A's hold parametrisation: centre = pad + Rg Rt [BOOK_D/2 - depth, 0, dz], R = Rg Rt Ry(90deg)."""
        Rg, pg = grasp_frame(d)
        pad = pg + Rg @ pad_in_grasp
        Rb = d.xmat[env._b_books[0]].reshape(3, 3)
        Rt = Rg.T @ Rb @ roty(1.5708).T
        tilt = math.atan2(Rt[0, 2], Rt[0, 0])
        rel = Rt.T @ Rg.T @ (d.xpos[env._b_books[0]] - pad)
        return {'depth': float(P.BOOK_D / 2 - rel[0]), 'dz': float(rel[2]), 'dy': float(rel[1]), 'tilt_deg': math.degrees(tilt),
                'off_axis_deg': math.degrees(math.acos(np.clip((np.trace(Rt) - 1) / 2, -1, 1)))}

    def seat_book(d, depth=0.03, dz=0.0, tilt=0.0):
        mujoco.mj_kinematics(m, d)
        Rg, pg = grasp_frame(d)
        pad = pg + Rg @ pad_in_grasp
        Rt = roty(tilt)
        ad = env._qbook[0]
        d.qpos[ad:ad + 3] = pad + Rg @ Rt @ np.array([P.BOOK_D / 2 - depth, 0.0, dz])
        d.qpos[ad + 3:ad + 7] = mat2quat(Rg @ Rt @ roty(1.5708))
        d.qvel[env._vbook[0]:env._vbook[0] + 6] = 0

    def state_from_gz(d, r):
        d.qpos[:] = m.qpos0; d.qvel[:] = 0
        d.qpos[env._qbase] = r['truth_base_w']
        d.qpos[env._q8] = r['q8']
        set_finger(d, q_closed)                          # physically on the book (Gazebo's reading is not the gap)
        poses = r['truth']['poses']
        tb = [n for n in poses if n.startswith('book_col_')][0]
        p, q = poses[tb]; ad = env._qbook[0]
        d.qpos[ad:ad + 3] = p; d.qpos[ad + 3:ad + 7] = wxyz(q)
        p, q = poses['erc_collection_bin']; d.qpos[env._qbin:env._qbin + 3] = p; d.qpos[env._qbin + 3:env._qbin + 7] = wxyz(q)
        mujoco.mj_forward(m, d)

    def state_nominal(d, rng):
        d.qpos[:] = m.qpos0; d.qvel[:] = 0
        base = np.array([env_place.BIN_XY[0] + env_place.STAND, env_place.BIN_XY[1] - 0.24])
        d.qpos[env._qbase] = [base[0] + rng.uniform(-0.05, 0.05), base[1] + rng.uniform(-0.05, 0.05), math.pi + rng.uniform(-0.05, 0.05)]
        qh = np.array(P.Q_HOLD_1830) if a.q_hold_1830 else np.array(env._q_hold)
        d.qpos[env._q8] = np.r_[float(env._torso_hold) if a.torso_hold is None else a.torso_hold, qh]
        set_finger(d, q_closed)
        seat_book(d, rng.uniform(0.015, 0.045), rng.uniform(-0.03, 0.03), math.radians(rng.uniform(-5, 5)))
        mujoco.mj_forward(m, d)

    def rollout(d, delay_sub=0, actions=None, steps=None):
        """Closed loop (actions None) or open loop on recorded commands. Returns the outcome dict."""
        steps = steps or a.steps
        hi = list(meta_env.get('hi')) if meta_env.get('hi') else None
        if hi and a.torso_max is not None:
            hi[0] = a.torso_max
        mp = P.ActionMapper({**pol.meta.get('config', {}), 'lo': meta_env.get('lo'), 'hi': hi})
        q8 = d.qpos[env._q8].copy()
        mp.reset(q8, 0.0)
        d.ctrl[:] = 0; d.ctrl[env._a8] = q8; d.ctrl[env._afing] = 0.0      # squeezing, as lane A's reset
        info = {'target': jp.array(q8), 'finger_target': jp.array(0.0), 'last_act': jp.zeros(12), 'qbias': jp.zeros(8),
                'step': jp.array(0.0), 'first_contact': jp.array(0.0), 'gentle': jp.array(0.0), 'released': jp.array(0.0),
                'grip_z_release': jp.array(0.0)}
        out = {'success': False, 'released_k': None, 'h_release': None, 'gentle': 0.0, 'fail': False, 'min_h': 9.0,
               'vz_first_bin': None, 'k_first_bin': None, 'max_vz_held': 0.0}
        for k in range(steps):
            fd = fake(d)
            if actions is None:
                info['step'] = jp.array(float(k)); info['target'] = jp.array(mp.target)
                info['finger_target'] = jp.array(mp.finger_target); info['last_act'] = jp.array(mp.last_act)
                obs = np.array(env._obs(fd, info)['state'])
                act = pol.act(obs)
                tgt, g_t, tw = mp.step(act, d.qpos[env._q8])
            else:
                if k >= len(actions):
                    break
                tgt, g_t, tw = np.array(actions[k]['tgt']), actions[k]['g_tgt'], np.array(actions[k]['twist'])
                act = np.array(actions[k]['act'])
            ref0 = d.ctrl[env._a8].copy(); g0 = float(d.ctrl[env._afing[0]] / INNER_MUL)
            yaw = d.qpos[env._qbase[2]]
            vw = [math.cos(yaw) * tw[0] - math.sin(yaw) * tw[1], math.sin(yaw) * tw[0] + math.cos(yaw) * tw[1], tw[2]]
            for i in range(nsub):
                f = min(1.0, max(0.0, (i + 1 - delay_sub) / nsub))
                d.ctrl[env._a8] = ref0 + f * (tgt - ref0)
                d.ctrl[env._afing] = INNER_MUL * (g0 + f * (g_t - g0))
                d.ctrl[env._abase] = vw
                mujoco.mj_step(m, d)
                vz = float(d.qvel[env._vbook[0] + 2])
                if out['vz_first_bin'] is None and d.sensordata[env._s['book0_bin']] > 0:   # the scorer's rule, at 500 Hz
                    out['vz_first_bin'], out['k_first_bin'] = vz, k
                if any(d.sensordata[env._s[s_]] > 0 for s_ in ('padL_book0', 'padR_book0', 'innerL_book0', 'innerR_book0')):
                    out['max_vz_held'] = max(out['max_vz_held'], -vz)
            fd = fake(d)
            _, terms, fail, success = env._reward(fd, info, jp.array(act))
            h = float(d.xpos[env._b_books[0]][2] - P.BOOK_H / 2 - env_place.BIN_FLOOR)
            out['min_h'] = min(out['min_h'], h)
            if out['released_k'] is None and float(info['released']) > 0:
                out['released_k'], out['h_release'] = k, h
                out['inside_at_release'] = bool(env._in_bin(fd, fd.xpos[env._b_books[0]], 0.03)[0])
            if bool(success):
                out['success'] = True
            if bool(fail):
                out['fail'] = True
                break
        out['gentle'] = float(info['gentle'])
        out['scorer_gentle'] = out['vz_first_bin'] is not None and abs(out['vz_first_bin']) < 0.5
        bp = d.xpos[env._b_books[0]]; bc = d.xpos[env._b_bin]
        out['book_minus_bin_xy'] = [float(bp[0] - bc[0]), float(bp[1] - bc[1])]; out['book_z'] = float(bp[2])
        return out

    res = {'checkpoint': a.checkpoint, 'meta_step': pol.meta.get('step'), 'episodes': {}, 'control': []}
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    for i in range(a.control_n):
        state_nominal(d, rng)
        hp = hold_params(d)
        o = rollout(d)
        o['hold'] = hp
        res['control'].append(o)
    nc = len(res['control'])
    print(f"control (lane A nominal start, no DR, {nc}): success {sum(o['success'] for o in res['control'])}/{nc}, "
          f"scorer-gentle {sum(o['scorer_gentle'] for o in res['control'])}/{nc}, vz@bin {[None if o['vz_first_bin'] is None else round(o['vz_first_bin'], 2) for o in res['control']]}, "
          f"released h {[None if o['h_release'] is None else round(o['h_release'], 3) for o in res['control']]}")
    for ep in a.eps:
        f = os.path.join(ep, 'policy_place.jsonl')
        if not os.path.exists(f):
            continue
        R = [json.loads(l) for l in open(f)]
        recs = [r for r in R if 'k' in r and 'truth' in r]
        if not recs:
            continue
        r0 = recs[0]
        e = {'gz_released_k': next((r['k'] for r in recs if r.get('released')), None), 'gz_h_end': recs[-1]['h']}
        state_from_gz(d, r0); e['hold_gz'] = hold_params(d)
        state_from_gz(d, r0); e['V0_exact'] = rollout(d)
        state_from_gz(d, r0); e['V1_latency75'] = rollout(d, delay_sub=int(round(0.075 / cfg.sim_dt)))
        state_from_gz(d, r0); seat_book(d); mujoco.mj_forward(m, d); e['V2_nominal_hold'] = rollout(d)
        state_from_gz(d, r0); e['V3_openloop_gz_cmds'] = rollout(d, actions=recs)
        res['episodes'][ep] = e
        hp = e['hold_gz']
        print(f"== {os.path.basename(ep)}: Gazebo hold depth {hp['depth']:.3f} m (trained 0.015..0.045), dz {hp['dz']:+.3f} (+-0.03), "
              f"dy {hp['dy']:+.3f}, tilt {hp['tilt_deg']:+.1f} deg (+-5), off-axis {hp['off_axis_deg']:.1f} deg; Gazebo released at k {e['gz_released_k']}")
        for v in ('V0_exact', 'V1_latency75', 'V2_nominal_hold', 'V3_openloop_gz_cmds'):
            o = e[v]
            print(f"   {v:20s} success {o['success']!s:5s} scorer-gentle {o['scorer_gentle']!s:5s} vz@bin {None if o['vz_first_bin'] is None else round(o['vz_first_bin'], 2)} "
                  f"max down speed held {o['max_vz_held']:.2f} released k {o['released_k']} h {None if o['h_release'] is None else round(o['h_release'], 3)} "
                  f"gentle {o['gentle']:.0f} fail {o['fail']!s:5s} book-bin {np.round(o['book_minus_bin_xy'], 3).tolist()} z {o['book_z']:.3f}")
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


if __name__ == '__main__':
    sys.exit(main())

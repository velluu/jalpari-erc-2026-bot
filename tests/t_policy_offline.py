"""Known-answer check of the policy observation builders (erc_bot/policy_obs.py + policy_fk.py) against lane A's own
PickEnv._obs / PlaceEnv._obs on replica states (C MuJoCo forward kinematics, lane A's observation code verbatim).
Runs on the Z8 in lane A's venv (CPU only, nothing trains):

  cd ~/erc/policy/bot && JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 PYTHONPATH=~/erc/learn:. \
      ~/venvs/erclearn/bin/python tests/t_policy_offline.py --urdf ~/erc/gym/urdf/tiago_pro_sim.urdf --n 300

Also checks ActionMapper against a line-by-line numpy transcription of PickEnv.step's command logic.
"""
import argparse, math, sys, types
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from mujoco import mjx
    mjx.put_model = lambda m, impl=None: m            # lane A's _obs needs only array fields of the model: use C MuJoCo
    import mujoco, jax, jax.numpy as jp
    from erc_learn import env_pick, env_place
    from erc_learn.build_model import INNER_MUL
    from erc_bot import policy_obs as P
    from erc_bot.policy_fk import UrdfFK
    fk = UrdfFK(a.urdf)
    rng = np.random.default_rng(a.seed)
    worst = {}
    ok = True

    def fake(m, d):
        return types.SimpleNamespace(qpos=jp.array(d.qpos), qvel=jp.array(d.qvel), xpos=jp.array(d.xpos),
                                     xmat=jp.array(d.xmat.reshape(-1, 3, 3)), site_xpos=jp.array(d.site_xpos),
                                     sensordata=jp.array(d.sensordata), xaxis=jp.array(d.xaxis), xanchor=jp.array(d.xanchor))

    def set_finger(env, d, v):
        m = env.mj_model
        d.qpos[env._qf] = INNER_MUL * v
        for e in range(m.neq):
            j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
            d.qpos[m.jnt_qposadr[j1]] = m.eq_data[e, 1] * d.qpos[m.jnt_qposadr[j2]]

    def report(kind, got, ref):
        nonlocal ok
        sl = P.slices(kind)
        if got.shape != ref.shape:
            print(kind, 'SIZE MISMATCH', got.shape, ref.shape); ok = False; return
        for name, s in sl.items():
            e = float(np.abs(got[s] - ref[s]).max())
            worst[(kind, name)] = max(worst.get((kind, name), 0.0), e)

    # ------------------------------------------------ pick
    for privileged in (False, True):
        env = env_pick.PickEnv(config_overrides={'privileged_actor': privileged}, nenv_hint=1)
        m = env.mj_model; d = mujoco.MjData(m)
        kind = 'pick_teacher' if privileged else 'pick_student'
        T = env._config.episode_length
        for i in range(a.n):
            d.qpos[:] = m.qpos0; d.qvel[:] = 0
            row = int(rng.integers(1, 5)); col = int(rng.integers(1, 6)); wall = rng.random() < 0.3
            col = 4 if wall else col
            bay = float(P.BAY_Y[col - 1]); plate = -1.03 if wall else (2 - (col - 1)) * 1.03
            side = [1.0, -1.0, 1.0, -1.0][row - 1] if wall else 0.0
            ys = bay + rng.uniform(-0.45, 0.45, 4)
            for k in range(4):
                ad = env._qbook[k]
                z = P.BOARD_TOP[[row - 1, (row) % 4, (row + 1) % 4, (row + 2) % 4][k]] + P.BOOK_H / 2
                yaw_b = rng.uniform(-0.3, 0.3); tilt = rng.uniform(-0.3, 0.3)
                q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([0.0, 1.5708 + tilt, yaw_b]), 'xyz')
                d.qpos[ad:ad + 3] = [2.9 + rng.uniform(-0.15, 0.05), ys[k], z + rng.uniform(-0.02, 0.06)]
                d.qpos[ad + 3:ad + 7] = q
                d.qvel[env._vbook[k]:env._vbook[k] + 6] = rng.normal(0, 0.2, 6)
            spine = np.array([2.82, ys[0]])
            d.qpos[env._qbase] = [spine[0] - P.REACH[row] + rng.uniform(-0.15, 0.15), spine[1] + 0.24 + rng.uniform(-0.15, 0.15),
                                  rng.uniform(-0.3, 0.3)]
            q8 = np.r_[np.clip(P.TORSO[row] + rng.uniform(-0.05, 0.05), 0.0, 0.35), np.array(P.Q_PRE[row]) + rng.uniform(-0.4, 0.4, 7)]
            q8 = np.clip(q8, P.RANGE8[:, 0], P.RANGE8[:, 1])
            d.qpos[env._q8] = q8
            fing = rng.uniform(0.0, 0.069)
            set_finger(env, d, fing)
            d.qvel[env._v8] = rng.normal(0, 0.5, 8); d.qvel[env._vf] = INNER_MUL * rng.normal(0, 0.05)
            mujoco.mj_forward(m, d)
            qbias = rng.uniform(-0.005, 0.005, 8)
            book_init = d.xpos[env._b_books].copy() + rng.normal(0, 0.01, (4, 3))
            mapper = P.ActionMapper()
            mapper.target = q8 + qbias + rng.uniform(-0.05, 0.05, 8); mapper.finger_target = rng.uniform(0, 0.069)
            mapper.last_act = rng.uniform(-1, 1, 12)
            step = float(rng.integers(0, T))
            est_err = rng.normal(0, 0.008, 3); shelf_err = rng.uniform(-1, 1, 2) * np.array([0.01, math.radians(1)])
            info = {'target': jp.array(mapper.target), 'finger_target': jp.array(mapper.finger_target),
                    'last_act': jp.array(mapper.last_act), 'book_init': jp.array(book_init), 'layout_row': jp.array(float(row - 1)),
                    'col': jp.array(float(col)), 'wall_side': jp.array(side), 'bay': jp.array(bay), 'wall': jp.array(float(wall)),
                    'plate': jp.array(plate), 'qbias': jp.array(qbias), 'step': jp.array(step), 'est_err': jp.array(est_err),
                    'shelf_err': jp.array(shelf_err), 'rng': jax.random.PRNGKey(i), 'q8_lag': jp.array(q8 + rng.normal(0, 0.01, 8)),
                    'fing_lag': jp.array(fing + rng.normal(0, 0.002)), 'odom_err': jp.array(rng.normal(0, 0.03, 3)),
                    'q_lag': jp.array(int(rng.integers(0, 16))), 'fing_sc': jp.array(rng.uniform(0.6, 1.4)),
                    'fing_off': jp.array(rng.uniform(-0.004, 0.004)), 'fing_cmd_mode': jp.array(float(rng.random() < 0.2))}
            fd = fake(m, d)
            ref = np.array(env._obs(fd, info)['state'])
            # ---- ours, from what the robot would measure ----
            qm = q8 + qbias
            v8 = d.qvel[env._v8].copy(); fv = d.qvel[env._vf] / INNER_MUL
            fm = d.qpos[env._qf] / INNER_MUL
            T_bg = fk.pose('gripper_right_grasping_link', dict(zip(P.JOINTS8, q8)))     # true joints: the replica's FK
            bx, by, byaw = d.qpos[env._qbase]
            Rb = P.yaw_R(byaw); pb = np.array([bx, by, 0.0])
            if privileged:
                pb4 = d.xpos[env._b_books[0]]; R4 = d.xmat[env._b_books[0]].reshape(3, 3)
                vb = d.qvel[env._vbook[0]:env._vbook[0] + 3]
                nb = float(np.linalg.norm(d.xpos[env._b_books[1:]] - book_init[1:], axis=-1).max())
                sd = d.sensordata
                c = [float(sd[env._s[k]] > 0) for k in ('padL_book0', 'padR_book0', 'innerL_book0', 'innerR_book0', 'robot_shelf')]
                c += [float(max(sd[env._s[f'robot_book{k}']] for k in (1, 2, 3)) > 0), float(sd[env._s['arm_book0']] > 0)]
                got = P.pick_teacher(qm, fm, v8, fv, mapper, T_bg, Rb, pb, pb4, R4, vb, book_init[0], nb, bay, row, byaw, side, c, step, T)
            else:
                # what the robot measures, from lane A's own reading model (lagged noisy joints, finger reading, odom belief)
                q8s, v8s, fs = [np.array(x) for x in env._proprio(fd, info)]
                Ro, po, yo = [np.array(x) for x in env._odom_frame(fd, info)]
                est = book_init[0] + np.array([-P.BOOK_D / 2, 0, 0]) + est_err
                got = P.pick_student(q8s, float(fs), v8s, fv, mapper, T_bg, Ro, po, est, P.SHELF_FRONT + shelf_err[0], plate,
                                     P.BOARD_TOP[row - 1], float(yo) - shelf_err[1], side, step, T, fing_geom=fm)
            report(kind, got, ref)

    # ------------------------------------------------ place
    for privileged in (False, True):
        env = env_place.PlaceEnv(config_overrides={'privileged_actor': privileged}, nenv_hint=1)
        m = env.mj_model; d = mujoco.MjData(m)
        kind = 'place_teacher' if privileged else 'place_student'
        T = env._config.episode_length
        for i in range(a.n):
            d.qpos[:] = m.qpos0; d.qvel[:] = 0
            bxy = np.array([-1.0, 0.0]) + rng.uniform(-0.05, 0.05, 2); byaw_bin = rng.uniform(-0.1, 0.1)
            qz = np.array([math.cos(byaw_bin / 2), 0, 0, math.sin(byaw_bin / 2)]); q0 = m.qpos0[env._qbin + 3:env._qbin + 7]
            qq = np.zeros(4); mujoco.mju_mulQuat(qq, qz, q0)
            d.qpos[env._qbin:env._qbin + 3] = [bxy[0], bxy[1], 0.845]; d.qpos[env._qbin + 3:env._qbin + 7] = qq
            d.qpos[env._qbase] = [-0.3 + rng.uniform(-0.1, 0.1), -0.24 + rng.uniform(-0.1, 0.1), math.pi + rng.uniform(-0.3, 0.3)]
            q8 = np.r_[0.345 - rng.uniform(0, 0.05), np.array(P.Q_HOLD) + rng.uniform(-0.4, 0.4, 7)]
            q8 = np.clip(q8, P.RANGE8[:, 0], P.RANGE8[:, 1]); d.qpos[env._q8] = q8
            set_finger(env, d, rng.uniform(0, 0.069))
            ad = env._qbook[0]
            qb = np.zeros(4); mujoco.mju_euler2Quat(qb, rng.uniform(-0.3, 0.3, 3) + np.array([0, 1.5708, 0]), 'xyz')
            d.qpos[ad:ad + 3] = [-1.0 + rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.2)]; d.qpos[ad + 3:ad + 7] = qb
            d.qvel[env._vbook[0]:env._vbook[0] + 6] = rng.normal(0, 0.3, 6)
            d.qvel[env._v8] = rng.normal(0, 0.5, 8)
            mujoco.mj_forward(m, d)
            qbias = rng.uniform(-0.005, 0.005, 8)
            mapper = P.ActionMapper(); mapper.target = q8 + qbias + rng.uniform(-0.05, 0.05, 8)
            mapper.finger_target = rng.uniform(0, 0.069); mapper.last_act = rng.uniform(-1, 1, 12)
            step = float(rng.integers(0, T)); err = rng.normal(0, 0.012, 2)
            fc, gentle = float(rng.integers(0, 2)), float(rng.integers(0, 2))
            e3 = np.r_[err, 0.0]
            info = {'target': jp.array(mapper.target), 'finger_target': jp.array(mapper.finger_target), 'last_act': jp.array(mapper.last_act),
                    'qbias': jp.array(qbias), 'step': jp.array(step), 'est_err': jp.array(e3), 'first_contact': jp.array(fc),
                    'gentle': jp.array(gentle), 'bin_xy': jp.array(bxy), 'rng': jax.random.PRNGKey(i),
                    'q8_lag': jp.array(q8 + rng.normal(0, 0.01, 8)), 'fing_lag': jp.array(rng.uniform(0, 0.069)),
                    'odom_err': jp.array(rng.normal(0, 0.03, 3)), 'q_lag': jp.array(int(rng.integers(0, 16))),
                    'fing_sc': jp.array(rng.uniform(0.6, 1.4)), 'fing_off': jp.array(rng.uniform(-0.004, 0.004)),
                    'fing_cmd_mode': jp.array(float(rng.random() < 0.2))}
            fd = fake(m, d)
            ref = np.array(env._obs(fd, info)['state'])
            qm = q8 + qbias; v8 = d.qvel[env._v8].copy(); fv = d.qvel[env._vf] / INNER_MUL; fm = d.qpos[env._qf] / INNER_MUL
            T_bg = fk.pose('gripper_right_grasping_link', dict(zip(P.JOINTS8, q8)))
            bx, by, byaw = d.qpos[env._qbase]; Rb = P.yaw_R(byaw); pb = np.array([bx, by, 0.0])
            Rbin = d.xmat[env._b_bin].reshape(3, 3); cbin = d.xpos[env._b_bin] + Rbin @ np.array([0, 0, -0.0073])
            if privileged:
                sd = d.sensordata
                c = [float(sd[env._s[k]] > 0) for k in ('padL_book0', 'padR_book0', 'innerL_book0', 'innerR_book0', 'book0_bin',
                                                         'robot_bin', 'robot_shelf')] + [fc, gentle]
                got = P.place_teacher(qm, fm, v8, fv, mapper, T_bg, Rb, pb, d.xpos[env._b_books[0]], d.xmat[env._b_books[0]].reshape(3, 3),
                                      d.qvel[env._vbook[0]:env._vbook[0] + 3], cbin, -Rbin[:, 2], c, step, T)
            else:
                q8s, v8s, fs = [np.array(x) for x in env._proprio(fd, info)]
                Ro, po, yo = [np.array(x) for x in env._odom_frame(fd, info)]
                got = P.place_student(q8s, float(fs), v8s, fv, mapper, T_bg, Ro, po, bxy + np.array([-err[0], err[1]]), float(yo), step, T, fing_geom=fm)
            report(kind, got, ref)

    # ------------------------------------------------ scripted expert (residual policies) vs erc_learn/expert.py
    try:
        from erc_learn.expert import expert_action as x_ref
        envx = env_pick.PickEnv(env_pick.default_config(), config_overrides={'residual': True, 'privileged_actor': False}, nenv_hint=1)
        m = envx.mj_model; d = mujoco.MjData(m)
        e_act, e_ph, n_ph = 0.0, 0, 0
        for i in range(a.n):
            d.qpos[:] = m.qpos0; d.qvel[:] = 0
            row = int(rng.integers(1, 5))
            ad = envx._qbook[0]; d.qpos[ad:ad + 3] = [2.9, rng.uniform(-2, 2), P.BOARD_TOP[row - 1] + P.BOOK_H / 2]
            d.qpos[ad + 3:ad + 7] = [math.cos(0.7854), 0, math.sin(0.7854), 0]
            d.qpos[envx._qbase] = [2.82 - P.REACH[row] + rng.uniform(-0.15, 0.15), d.qpos[ad + 1] + 0.24 + rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)]
            q8 = np.clip(np.r_[P.TORSO[row], np.array(P.Q_PRE[row]) + rng.uniform(-0.3, 0.3, 7)], P.RANGE8[:, 0], P.RANGE8[:, 1])
            d.qpos[envx._q8] = q8
            mujoco.mj_forward(m, d)
            book_init = d.xpos[envx._b_books].copy()
            ph, t = float(rng.integers(0, 5)), float(rng.integers(0, 70))
            est_err = rng.normal(0, 0.01, 3); odom_err = rng.normal(0, 0.02, 3)
            info = {'book_init': jp.array(book_init), 'est_err': jp.array(est_err), 'odom_err': jp.array(odom_err),
                    'layout_row': jp.array(float(row - 1)), 'xphase': jp.array(ph), 'xt': jp.array(t)}
            Tb = lambda qq: fk.pose('gripper_right_grasping_link', dict(zip(P.JOINTS8, qq)))
            J, Tg = P.numeric_jacobian(Tb, q8)
            bx, by, byaw = d.qpos[envx._qbase]
            for mode in ('est', 'true'):
                ref, upd = x_ref(envx, fake(m, d), info, mode)
                if mode == 'est':
                    yo = byaw + odom_err[2]; Ro = P.yaw_R(yo); po = np.array([bx + odom_err[0], by + odom_err[1], 0.0])
                    spine_b = Ro.T @ (book_init[0] + np.array([-0.08, 0, 0]) + est_err - po)
                else:
                    yo = byaw; spine_b = P.yaw_R(byaw).T @ (book_init[0] + np.array([-0.08, 0, 0]) - np.array([bx, by, 0.0]))
                got, (ph2, t2) = P.expert_action(spine_b, yo, row - 1, Tg[:3, 3], Tg[:3, :3], J, np.array(envx._step_max), ph, t)
                e_act = max(e_act, float(np.abs(got - np.array(ref)).max()))
                e_ph += int(ph2 != float(upd['xphase']) or abs(t2 - float(upd['xt'])) > 1e-6); n_ph += 1
        print(f'expert (numpy port, URDF FK and numeric Jacobian) vs erc_learn.expert_action: max action diff {e_act:.2e} over {n_ph} '
              f'cases (est and true, all phases), phase/timer mismatches {e_ph}')
        ok &= e_act < 2e-3 and e_ph == 0
    except ImportError:
        print('erc_learn.expert not present: expert check skipped')

    # ------------------------------------------------ action mapping vs PickEnv.step (transcribed)
    env = env_pick.PickEnv(nenv_hint=1); cfg = env._config
    lo, hi = np.array(env._lo), np.array(env._hi); smax = np.array(env._step_max)
    mp = P.ActionMapper({**cfg.to_dict(), 'lo': lo, 'hi': hi})     # training logic exactly
    e_map = 0.0
    for i in range(2000):
        q = rng.uniform(lo - 0.05, hi + 0.05); tgt0 = q + rng.uniform(-0.1, 0.1, 8); ft0 = rng.uniform(0, 0.069)
        act = rng.uniform(-1.3, 1.3, 12)
        mp.target, mp.finger_target = tgt0.copy(), ft0
        t1, g1, tw = mp.step(act, q)
        aa = np.clip(act, -1, 1)
        lo_w = np.maximum(q - smax, tgt0 - smax); hi_w = np.minimum(q + smax, tgt0 + smax)     # PickEnv.step 18:51
        r = np.clip(tgt0 + aa[:8] * smax, lo_w, hi_w)
        r = np.where(lo_w > hi_w, np.clip(q, tgt0 - smax, tgt0 + smax), r)
        r = np.clip(r, lo, hi)
        g = np.clip(0.069 * (1 - (aa[8] + 1) / 2), ft0 - 0.069 / cfg.grip_close_time * cfg.ctrl_dt, ft0 + 0.069 / cfg.grip_close_time * cfg.ctrl_dt)
        e_map = max(e_map, float(np.abs(t1 - r).max()), abs(g1 - g), float(np.abs(tw - aa[9:12] * np.array(cfg.base_v)).max()))
    print(f'action mapper vs PickEnv.step logic: max abs diff {e_map:.2e} over 2000 random cases; lo/hi from env {np.round(lo, 4).tolist()} {np.round(hi, 4).tolist()}')
    # live addition: no streamed target moves more than 0.95 cap dt from the previous command, even on reversals
    ml = P.ActionMapper({**cfg.to_dict(), 'lo': lo, 'hi': hi}); worst_rate = 0.0; prev = None
    q = (lo + hi) / 2; ml.reset(q, 0.069)
    for i in range(3000):
        lag = rng.uniform(-2, 2, 8) * smax                      # measured position lagging or leading the command
        prev = ml.target.copy()
        t1, _, _ = ml.step(rng.choice([-1.0, 1.0], 12) * rng.uniform(0.5, 1.0, 12), np.clip(prev + lag, lo, hi))
        worst_rate = max(worst_rate, float((np.abs(t1 - prev) / (P.CAPS * cfg.ctrl_dt)).max()))
    print(f'live rate limit: max |command - previous command| = {worst_rate:.4f} x cap x dt over 3000 random lagged steps (limit 0.95)')
    e_map = max(e_map, max(0.0, worst_rate - 0.95 - 1e-9))
    d0 = P.ActionMapper(cfg.to_dict())
    print(f'mapper default lo/hi (no meta) equal to the env now: {np.allclose(d0.lo, lo, atol=1e-6) and np.allclose(d0.hi, hi, atol=1e-6)}')
    print('max abs error per observation block (ours vs lane A _obs), n =', a.n, 'states per kind:')
    for (kind, name), e in sorted(worst.items()):
        flag = '' if e < 1e-5 else '  <-- CHECK'
        ok &= e < 1e-5
        print(f'  {kind:14s} {name:14s} {e:.2e}{flag}')
    print('PASS' if ok and e_map < 1e-6 else 'FAIL')   # env limits are float32
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

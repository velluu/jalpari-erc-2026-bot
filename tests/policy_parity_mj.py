"""Observation and action parity, Gazebo (live interface) vs lane A's MuJoCo replica, from the records the policy
probe writes in erc-gym episodes (erc_bot/policy_probe.py: OUT/policy_pick.jsonl). Runs on the Z8 in lane A's venv:

  cd ~/erc/policy/bot && JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 PYTHONPATH=~/erc/learn:. \
      ~/venvs/erclearn/bin/python tests/policy_parity_mj.py --eps EP_DIR [EP_DIR ...] --out parity.json

Observation parity: every recorded control period's state (gz ground truth: base pose, target and column books;
measured joints; gripper reading) is set in the replica, lane A's own _obs computes the teacher and the student
vectors, and they are compared block by block with what the runner built through the live interface.
The student is computed with lane A's noise switched off (odom error, detector error, finger mapping) except the
joint-reading noise its _proprio draws (0.002 rad, 0.05 rad/s), which sets the floor of q8 and v8.

Action parity (records made with actions=parity): from the first recorded state, the replica replays the same
streamed commands (targets, gripper, base twist) with lane A's controller emulation (linear reference ramp over the
control period, no latency, nominal physics, no DR) and the joint, finger and base trajectories are compared period
by period with Gazebo's.
"""
import argparse, glob, json, math, os, sys, types
import numpy as np

TOL = {   # block: max abs tolerance for PASS (units of the observation entry)
    'q8': 0.01, 'fing': 0.003, 'v8': 0.05, 'fingv': 0.05, 'target-q8': 0.01, 'finger_target': 1e-4, 'last_act': 1e-6,
    'grip_b': 0.005, 'rot6_grasp': 0.01, 'book_b': 0.005, 'rot6_book': 0.01, 'book_v': 0.05, 'G-grip': 0.005,
    'book_disp': 0.02, 'shelf_pts': 0.01, 'yaw': 0.01, 'wall_side': 0, 'tilt': 0.01, 'nb': 0.02, 'step': 1e-6,
    'est_b': 0.02, 'est-grip': 0.02, 'contacts': 0,
}


def load_records(ep):
    recs, start = [], None
    for f in sorted(glob.glob(os.path.join(ep, 'policy_pick.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            if r.get('event') == 'start':
                start, recs = r, []          # the last run in the file
            elif 'k' in r and 'obs' in r:
                recs.append(r)
    return start, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', nargs='+', required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    from mujoco import mjx
    mjx.put_model = lambda m, impl=None: m
    import mujoco, jax, jax.numpy as jp
    from erc_learn import env_pick
    from erc_learn.build_model import INNER_MUL
    from erc_bot import policy_obs as P
    # a fresh config per env: PickEnv's default config object is shared and config_overrides mutates it
    envs = {k: env_pick.PickEnv(env_pick.default_config(), config_overrides={'privileged_actor': k == 'teacher'}, nenv_hint=1)
            for k in ('teacher', 'student')}
    env = envs['teacher']; m = env.mj_model
    nsub = int(round(0.1 / m.opt.timestep))
    out = {'episodes': {}, 'tol': TOL}
    agg = {'pick_teacher': {}, 'pick_student': {}}
    contact_agree = {'n': 0, 'agree': np.zeros(7), 'gz_on': np.zeros(7), 'mj_on': np.zeros(7)}
    act_err = []

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

    def set_state(d, r, books):
        d.qpos[:] = m.qpos0; d.qvel[:] = 0
        bx, by, byaw = r['truth_base_w']
        d.qpos[env._qbase] = [bx, by, byaw]
        d.qpos[env._q8] = r['q8']
        set_finger(d, r['fing'])
        poses = r['truth']['poses']
        for k, name in enumerate(books):
            p, q = poses[name]
            ad = env._qbook[k]
            d.qpos[ad:ad + 3] = p; d.qpos[ad + 3:ad + 7] = wxyz(q)
        d.qvel[env._v8] = r['v8']; d.qvel[env._vf] = INNER_MUL * r['fingv']
        d.qvel[env._vbook[0]:env._vbook[0] + 3] = r.get('book_v', [0, 0, 0])
        mujoco.mj_forward(m, d)

    for ep in a.eps:
        start, recs = load_records(ep)
        if not recs or 'truth' not in recs[0]:
            print(f'{ep}: no truth records'); continue
        mis = json.load(open(os.path.join(ep, 'mission.json')))
        ev = {e['stage']: e for e in mis['events']}
        target = ev['POLICY_PICK']['truth']; row = int(ev['POLICY_PICK']['row'])
        others = sorted({n for n in recs[0]['truth']['poses'] if n.startswith('book_col_') and n != target},
                        key=lambda s: int(s.split('_')[4]))[:3]
        books = [target] + others
        lay = json.load(open(os.path.join(ep, 'layout.json')))
        col = int(target.split('_')[2])
        plate = float(lay['column_y'][col - 1])
        d = mujoco.MjData(m)
        set_state(d, recs[0], books)
        book_init = np.array([recs[0]['truth']['poses'][n][0] for n in books])
        bay = float(P.BAY_Y[int(np.argmin(np.abs(P.BAY_Y - book_init[0][1])))])
        epres = {'target': target, 'row': row, 'n': len(recs), 'actions': ((start or {}).get('opts') or {}).get('actions', (start or {}).get('actions', ''))}
        errs = {'pick_teacher': {}, 'pick_student': {}}
        for i in range(1, len(recs)):
            r, prev = recs[i], recs[i - 1]
            set_state(d, r, books)
            fd = fake(d)
            wall_side = float(np.array(r['obs_teacher'])[P.slices('pick_teacher')['wall_side']][0])
            base = {'target': jp.array(prev['tgt']), 'finger_target': jp.array(prev['g_tgt']), 'last_act': jp.array(prev['act']),
                    'book_init': jp.array(book_init), 'layout_row': jp.array(float(row - 1)), 'col': jp.array(float(col)),
                    'wall_side': jp.array(wall_side), 'bay': jp.array(bay), 'wall': jp.array(float(wall_side != 0)),
                    'plate': jp.array(plate), 'qbias': jp.zeros(8), 'step': jp.array(float(r['k'])),
                    'shelf_err': jp.zeros(2), 'est_err': jp.zeros(3), 'rng': jax.random.PRNGKey(i), 'q8_lag': jp.array(r['q8']),
                    'fing_lag': jp.array(r['fing']), 'odom_err': jp.zeros(3), 'q_lag': jp.array(0), 'fing_sc': jp.array(1.0),
                    'fing_off': jp.array(0.0), 'fing_cmd_mode': jp.array(0.0)}
            for kind, ek in (('pick_teacher', 'teacher'), ('pick_student', 'student')):
                ref = np.array(envs[ek]._obs(fd, base)['state'])
                assert ref.shape[0] == P.SIZES[kind], (kind, ref.shape)
                got = np.array(r['obs_teacher' if ek == 'teacher' else 'obs_student'])
                for name, sl in P.slices(kind).items():
                    errs[kind].setdefault(name, []).append(np.abs(got[sl] - ref[sl]).max() if name != 'contacts' else got[sl] - ref[sl])
            gzc = np.array(r['contacts']); sd = d.sensordata
            mjc = np.array([float(sd[env._s[k]] > 0) for k in ('padL_book0', 'padR_book0', 'innerL_book0', 'innerR_book0', 'robot_shelf')]
                           + [float(max(sd[env._s[f'robot_book{k}']] for k in (1, 2, 3)) > 0), float(sd[env._s['arm_book0']] > 0)])
            contact_agree['n'] += 1; contact_agree['agree'] += (gzc == mjc); contact_agree['gz_on'] += gzc; contact_agree['mj_on'] += mjc
        for kind in errs:
            epres[kind] = {}
            for name, v in errs[kind].items():
                if name == 'contacts':
                    v = np.abs(np.array(v)); epres[kind][name] = {'mismatch_rate': float((v > 0).mean())}
                    continue
                v = np.array(v)
                epres[kind][name] = {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(v.max())}
                agg[kind].setdefault(name, []).extend(v.tolist())
        # ---------------- action parity: replay the streamed commands in the replica
        if epres['actions'] == 'parity':
            d = mujoco.MjData(m)
            set_state(d, recs[0], books)
            d.qvel[:] = 0
            a8 = env._a8; af = env._afing; ab = env._abase
            d.ctrl[a8] = recs[0]['q8']; d.ctrl[af] = INNER_MUL * recs[0]['fing']; d.ctrl[ab] = 0
            g_prev = recs[0]['fing']
            rows = []
            for i in range(len(recs) - 1):
                r, nxt = recs[i], recs[i + 1]
                tgt = np.array(r['tgt']); g_t = r['g_tgt']; tw = np.array(r['twist'])
                ref0 = d.ctrl[a8].copy()
                yaw = d.qpos[env._qbase[2]]
                vw = [math.cos(yaw) * tw[0] - math.sin(yaw) * tw[1], math.sin(yaw) * tw[0] + math.cos(yaw) * tw[1], tw[2]]
                dt_rec = nxt['t'] - r['t']
                n = max(1, int(round(dt_rec / m.opt.timestep)))
                for j in range(n):
                    frac = min(1.0, (j + 1) / nsub)
                    d.ctrl[a8] = ref0 + frac * (tgt - ref0)
                    d.ctrl[af] = INNER_MUL * (g_prev + frac * (g_t - g_prev))
                    d.ctrl[ab] = vw
                    mujoco.mj_step(m, d)
                g_prev = g_t
                q_mj = d.qpos[env._q8].copy(); f_mj = d.qpos[env._qf] / INNER_MUL
                bx, by, byaw = d.qpos[env._qbase]
                tb = nxt['truth_base_w']
                rows.append({'k': i + 1, 'dt': dt_rec, 'q_err': (np.array(nxt['q8']) - q_mj).tolist(), 'q_gz': nxt['q8'], 'q_mj': q_mj.tolist(),
                             'tgt': tgt.tolist(), 'fing_gz': nxt['fing'], 'fing_mj': float(f_mj), 'g_tgt': g_t,
                             'base_err': [tb[0] - bx, tb[1] - by, math.atan2(math.sin(tb[2] - byaw), math.cos(tb[2] - byaw))]})
                # restart each period from Gazebo's state? no: open loop over the whole sequence (accumulated)
            qe = np.abs(np.array([x['q_err'] for x in rows]))
            fe = np.abs(np.array([x['fing_gz'] - x['fing_mj'] for x in rows]))
            be = np.abs(np.array([x['base_err'] for x in rows]))
            lag_gz = np.abs(np.array([np.array(x['tgt']) - np.array(x['q_gz']) for x in rows]))
            lag_mj = np.abs(np.array([np.array(x['tgt']) - np.array(x['q_mj']) for x in rows]))
            epres['action'] = {'periods': len(rows), 'mean_dt': float(np.mean([x['dt'] for x in rows])),
                               'joint_err_rms': np.sqrt((qe ** 2).mean(0)).tolist(), 'joint_err_max': qe.max(0).tolist(),
                               'track_lag_gz_mean': lag_gz.mean(0).tolist(), 'track_lag_mj_mean': lag_mj.mean(0).tolist(),
                               'finger_err_max': float(fe.max()), 'finger_err_rms': float(np.sqrt((fe ** 2).mean())),
                               'base_err_max': be.max(0).tolist(), 'rows': rows}
            act_err.append(epres['action'])
        out['episodes'][ep] = epres
        print(f'== {os.path.basename(ep)} target {target} row {row}: {len(recs)} periods, actions={epres["actions"] or "policy"}')
    # ---------------- summary
    names8 = ['torso', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
    allpass = True
    summ = {}
    for kind in agg:
        print(f'\n{kind}: Gazebo live-interface obs vs lane A _obs on the same state (abs error over all periods)')
        summ[kind] = {}
        for name, v in agg[kind].items():
            v = np.array(v); tol = TOL.get(name, 0.01)
            ok = v.max() <= tol + 1e-9
            # student: q8, v8, target-q8 carry lane A's own random joint-reading noise (exact in the teacher block);
            # est_b, est-grip, shelf_pts, yaw carry the real odom drift since the detection (reported, not a defect)
            if kind == 'pick_teacher' or name not in ('q8', 'v8', 'target-q8', 'est_b', 'est-grip', 'shelf_pts', 'yaw'):
                allpass &= bool(ok)
            summ[kind][name] = {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(v.max()), 'tol': tol, 'pass': bool(ok)}
            print(f'  {name:14s} median {np.median(v):.2e}  p95 {np.percentile(v, 95):.2e}  max {v.max():.2e}  tol {tol:g}  {"ok" if ok else "OVER"}')
    if contact_agree['n']:
        nm = ['padL', 'padR', 'inL', 'inR', 'shelf', 'other', 'arm_book']
        print('\ncontacts, gz sensors vs replica geometry on the same state (agreement, gz on, replica on, of', contact_agree['n'], 'periods):')
        for j, n in enumerate(nm):
            print(f'  {n:9s} agree {contact_agree["agree"][j] / contact_agree["n"]:.3f}  gz on {int(contact_agree["gz_on"][j])}  mj on {int(contact_agree["mj_on"][j])}')
        summ['contacts'] = {n: {'agree': float(contact_agree['agree'][j] / contact_agree['n']), 'gz_on': int(contact_agree['gz_on'][j]),
                                'mj_on': int(contact_agree['mj_on'][j])} for j, n in enumerate(nm)}
    for e in act_err:
        print(f'\naction parity ({e["periods"]} periods, mean period {e["mean_dt"]:.3f} s sim): open-loop replay, Gazebo minus replica')
        print('  joint       ' + ' '.join(f'{n:>7s}' for n in names8))
        print('  err rms    ' + ' '.join(f'{x:7.4f}' for x in e['joint_err_rms']))
        print('  err max    ' + ' '.join(f'{x:7.4f}' for x in e['joint_err_max']))
        print('  lag gz     ' + ' '.join(f'{x:7.4f}' for x in e['track_lag_gz_mean']) + '   |target - measured| mean')
        print('  lag mj     ' + ' '.join(f'{x:7.4f}' for x in e['track_lag_mj_mean']))
        print(f'  finger (sim scale) err max {e["finger_err_max"]:.4f} rms {e["finger_err_rms"]:.4f}; base err max x {e["base_err_max"][0]:.3f} m '
              f'y {e["base_err_max"][1]:.3f} m yaw {e["base_err_max"][2]:.3f} rad')
    out['summary'] = summ
    out['pass_interface'] = bool(allpass)
    print('\ninterface blocks (all but the odom-dependent student blocks):', 'PASS' if allpass else 'OVER TOLERANCE (see above)')
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


if __name__ == '__main__':
    sys.exit(main())

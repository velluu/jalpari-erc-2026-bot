"""Same-seed, same-state MuJoCo counterpart of Gazebo pick episodes: for every Gazebo episode the learned pick ran in
(records OUT/policy_pick.jsonl from LearnedPick, mission or probe), the pick policy is rolled out closed loop in
lane A's replica (C MuJoCo, lane A's _obs and _reward verbatim, the numpy policy, lane A's command rule and
controller ramp, no DR) from
  S  the exact state the policy started from in Gazebo (base, joints, target and column books from gz truth), and
  N  lane A's nominal start for the same book layout (base at the pre-grasp standoff, torso and q_pre of the row),
and compared with the Gazebo outcome (gz truth success: both fingers on the book, lifted >= 2 cm, out of the shelf;
and the scorer's grasp stage). Student policies see lane A's student observation with its noise model switched off
(an ideal detector at the true spine, no odom error): the Gazebo-minus-replica gap is then the interface, perception
and physics together. Run on the Z8 in lane A's venv:

  cd ~/erc/policy/bot && JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1 PYTHONPATH=~/erc/learn:. \
      ~/venvs/erclearn/bin/python tests/policy_replay_pick_mj.py --checkpoint X.npz --eps EP ... --out replay_pick.json
"""
import argparse, json, math, os, sys, types
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--eps', nargs='+', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--latency', type=float, default=0.0, help='command delay in s (Gazebo measured 0.075)')
    ap.add_argument('--expert-only', action='store_true', help='residual checkpoints: run the expert alone (policy output 0)')
    a = ap.parse_args()
    from mujoco import mjx
    mjx.put_model = lambda m, impl=None: m
    import mujoco, jax, jax.numpy as jp
    from erc_learn import env_pick
    from erc_learn.build_model import INNER_MUL
    from erc_bot import policy_obs as P
    pol = P.NumpyPolicy(a.checkpoint)
    kind = P.kind_of(pol)
    student = kind.endswith('student')
    me = pol.meta.get('env', {})
    pc = pol.meta.get('config', {})
    residual = bool(pc.get('residual'))
    over = {'privileged_actor': not student}
    if residual:
        over.update(residual=True, residual_scale=float(pc.get('residual_scale', 0.3)), expert_mode=pc.get('expert_mode', 'est'))
    env = env_pick.PickEnv(env_pick.default_config(), config_overrides=over, nenv_hint=1)
    if residual:
        from erc_learn.expert import expert_action as x_ref
    m = env.mj_model; cfg = env._config
    nsub = int(round(cfg.ctrl_dt / cfg.sim_dt)); T = a.steps or cfg.episode_length
    delay = int(round(a.latency / cfg.sim_dt))

    def fake(d):
        return types.SimpleNamespace(qpos=jp.array(d.qpos), qvel=jp.array(d.qvel), xpos=jp.array(d.xpos),
                                     xmat=jp.array(d.xmat.reshape(-1, 3, 3)), site_xpos=jp.array(d.site_xpos), sensordata=jp.array(d.sensordata),
                                     xaxis=jp.array(d.xaxis), xanchor=jp.array(d.xanchor))

    def set_finger(d, v):
        d.qpos[env._qf] = INNER_MUL * v
        for e in range(m.neq):
            j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
            d.qpos[m.jnt_qposadr[j1]] = m.eq_data[e, 1] * d.qpos[m.jnt_qposadr[j2]]

    def wxyz(q):
        return np.array([q[3], q[0], q[1], q[2]])

    def load(ep):
        f = os.path.join(ep, 'policy_pick.jsonl')
        if not os.path.exists(f):
            return None
        R = [json.loads(l) for l in open(f)]
        recs = [r for r in R if 'k' in r and 'truth' in r]
        if not recs:
            return None
        mis = json.load(open(os.path.join(ep, 'mission.json')))
        ev = {e['stage']: e for e in mis['events']}
        target = ev['POLICY_PICK']['truth']; row = int(ev['POLICY_PICK']['row'])
        others = sorted({n for n in recs[0]['truth']['poses'] if n.startswith('book_col_') and n != target}, key=lambda s: int(s.split('_')[4]))[:3]
        lay = json.load(open(os.path.join(ep, 'layout.json')))
        sc = json.load(open(os.path.join(ep, 'score.json'))) if os.path.exists(os.path.join(ep, 'score.json')) else {}
        end = ev.get('POLICY_PICK_END', {})
        return {'recs': recs, 'target': target, 'others': others, 'row': row, 'col': int(target.split('_')[2]), 'layout': lay,
                'gz_success': bool(end.get('truth_success')) or any(r.get('success') for r in recs),
                'gz_scorer_grasp': 'grasp' in (sc.get('stages') or {}), 'round': lay.get('round')}

    def set_books(d, E, r0):
        poses = r0['truth']['poses']
        for k, n in enumerate([E['target']] + E['others']):
            p, q = poses[n]; ad = env._qbook[k]
            d.qpos[ad:ad + 3] = p; d.qpos[ad + 3:ad + 7] = wxyz(q)

    def start_gz(d, E):
        r0 = E['recs'][0]
        d.qpos[:] = m.qpos0; d.qvel[:] = 0
        d.qpos[env._qbase] = r0['truth_base_w']; d.qpos[env._q8] = r0['q8']; set_finger(d, r0['fing'])
        set_books(d, E, r0); mujoco.mj_forward(m, d)

    def start_nominal(d, E):
        r0 = E['recs'][0]; row = E['row']
        d.qpos[:] = m.qpos0; d.qvel[:] = 0
        set_books(d, E, r0); mujoco.mj_forward(m, d)
        p = d.xpos[env._b_books[0]]
        reach = float(me.get('reach', {}).get(str(row), P.REACH[row])); arm_y = float(me.get('arm_y', P.ARM_Y))
        d.qpos[env._qbase] = [p[0] - P.BOOK_D / 2 - reach, p[1] - arm_y, 0.0]
        d.qpos[env._q8] = np.r_[float(me.get('torso', {}).get(str(row), P.TORSO[row])), me.get('q_pre', {}).get(str(row), P.Q_PRE[row])]
        set_finger(d, P.SIM_OPEN); mujoco.mj_forward(m, d)

    def rollout(d, E):
        row = E['row']; col = E['col']; lay = E['layout']
        p0 = np.array([d.xpos[b] for b in env._b_books])
        bay = float(P.BAY_Y[col - 1])
        side = float(np.sign(p0[0][1] - bay)) if abs(p0[0][1] - bay) > 0.4 else 0.0
        plate = float(lay['column_y'][col - 1])
        mp = P.ActionMapper({**pol.meta.get('config', {}), 'lo': me.get('lo'), 'hi': me.get('hi')})
        q8 = d.qpos[env._q8].copy(); f0 = float(d.qpos[env._qf] / INNER_MUL)
        mp.reset(q8, P.SIM_OPEN)
        d.ctrl[:] = 0; d.ctrl[env._a8] = q8; d.ctrl[env._afing] = INNER_MUL * P.SIM_OPEN
        info = {'book_init': jp.array(p0), 'layout_row': jp.array(float(row - 1)), 'col': jp.array(float(col)), 'wall_side': jp.array(side),
                'bay': jp.array(bay), 'wall': jp.array(float(side != 0)), 'plate': jp.array(plate), 'qbias': jp.zeros(8),
                'shelf_err': jp.zeros(2), 'est_err': jp.zeros(3), 'odom_err': jp.zeros(3), 'rng': jax.random.PRNGKey(0),
                'q8_lag': jp.array(q8), 'fing_lag': jp.array(f0), 'q_lag': jp.array(0), 'fing_sc': jp.array(1.0),
                'fing_off': jp.array(0.0), 'fing_cmd_mode': jp.array(0.0), 'success_steps': jp.array(0.0),
                'ever_success': jp.array(0.0), 'ever_held': jp.array(0.0), 'xphase': jp.array(0.0), 'xt': jp.array(0.0),
                'x_act': jp.zeros(12)}
        out = {'success': False, 'k_success': None, 'held': False, 'max_lift': 0.0, 'shelf_steps': 0, 'fail': False}
        for k in range(T):
            fd = fake(d)
            info.update(step=jp.array(float(k)), target=jp.array(mp.target), finger_target=jp.array(mp.finger_target),
                        last_act=jp.array(mp.last_act))
            if residual:                     # lane A's step: expert at this state, phases advance, action = expert + scale x policy
                x_act, x_upd = x_ref(env, fd, info, env._config.expert_mode)
                info['x_act'] = x_act
                act = pol.act(np.array(env._obs(fd, info)['state']))
                info.update(x_upd)
                act = np.clip(np.array(x_act) + env._config.residual_scale * act * (0.0 if a.expert_only else 1.0), -1.0, 1.0)
            else:
                act = pol.act(np.array(env._obs(fd, info)['state']))
            tgt, g_t, tw = mp.step(act, d.qpos[env._q8])
            ref0 = d.ctrl[env._a8].copy(); g0 = float(d.ctrl[env._afing[0]] / INNER_MUL)
            yaw = d.qpos[env._qbase[2]]
            vw = [math.cos(yaw) * tw[0] - math.sin(yaw) * tw[1], math.sin(yaw) * tw[0] + math.cos(yaw) * tw[1], tw[2]]
            for i in range(nsub):
                f = min(1.0, max(0.0, (i + 1 - delay) / nsub))
                d.ctrl[env._a8] = ref0 + f * (tgt - ref0); d.ctrl[env._afing] = INNER_MUL * (g0 + f * (g_t - g0)); d.ctrl[env._abase] = vw
                mujoco.mj_step(m, d)
            info['q8_lag'] = jp.array(d.qpos[env._q8]); info['fing_lag'] = jp.array(d.qpos[env._qf] / INNER_MUL)
            fd = fake(d)
            _, terms, fail, success = env._reward(fd, info, jp.array(act))
            out['held'] |= bool(terms['held'] > 0); out['max_lift'] = max(out['max_lift'], float(d.xpos[env._b_books[0]][2] - p0[0][2]))
            out['shelf_steps'] += int(terms['shelf'] > 0)
            if bool(success) and not out['success']:
                out['success'], out['k_success'] = True, k
            if bool(fail):
                out['fail'] = True; break
        return out

    res = {'checkpoint': a.checkpoint, 'kind': kind, 'step': pol.meta.get('step'), 'latency': a.latency, 'episodes': {}}
    d = mujoco.MjData(m)
    tally = {'S': 0, 'N': 0, 'gz': 0, 'n': 0}
    by = {}
    for ep in a.eps:
        E = load(ep)
        if E is None:
            continue
        start_gz(d, E); S = rollout(d, E)
        start_nominal(d, E); N = rollout(d, E)
        e = {'row': E['row'], 'col': E['col'], 'round': E['round'], 'gz_success': E['gz_success'], 'gz_scorer_grasp': E['gz_scorer_grasp'],
             'replica_same_state': S, 'replica_nominal': N}
        res['episodes'][ep] = e
        tally['n'] += 1; tally['S'] += S['success']; tally['N'] += N['success']; tally['gz'] += E['gz_success']
        for key in (f"row {E['row']}", f"col {E['col']}", f"round {E['round']}"):
            b = by.setdefault(key, {'n': 0, 'S': 0, 'N': 0, 'gz': 0})
            b['n'] += 1; b['S'] += S['success']; b['N'] += N['success']; b['gz'] += E['gz_success']
        print(f"{os.path.basename(ep)} r{E['round']} row {E['row']} col {E['col']}: Gazebo {'OK ' if E['gz_success'] else 'no '} | "
              f"replica same state {'OK ' if S['success'] else 'no '} (held {S['held']}, lift {S['max_lift']:.3f}, shelf {S['shelf_steps']}) | "
              f"replica nominal {'OK ' if N['success'] else 'no '} (held {N['held']}, lift {N['max_lift']:.3f})")
    n = max(1, tally['n'])
    print(f"\n{kind} {os.path.basename(a.checkpoint)}: n {tally['n']}  Gazebo {tally['gz'] / n:.2f}  replica same state {tally['S'] / n:.2f}  "
          f"replica nominal {tally['N'] / n:.2f}")
    for k in sorted(by):
        b = by[k]
        print(f"  {k:8s} n {b['n']:3d}  Gazebo {b['gz'] / b['n']:.2f}  same state {b['S'] / b['n']:.2f}  nominal {b['N'] / b['n']:.2f}")
    res['tally'] = tally; res['by'] = by
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))


if __name__ == '__main__':
    sys.exit(main())

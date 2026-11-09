exec(open(__file__.rsplit('/',1)[0] + '/t_replay_carry_base.py').read().split("def run(")[0])
bin_c = np.array([0.70, ARM_Y]); rl = 0.07
box = (bin_c[0] - 0.28 + rl, bin_c[0] + 0.28 - rl, bin_c[1] - 0.155 + rl, bin_c[1] + 0.155 - rl)
outer = (bin_c[0] - 0.28 - rl, bin_c[0] + 0.28 + rl, bin_c[1] - 0.155 - rl, bin_c[1] + 0.155 + rl)
def path(qs, goals, torso, **kw):
    qq = qs.copy(); mj = 0; dev = 0
    for g in goals:
        q1 = solve(transform(g), qq, torso, **kw)
        for a in (0.25, 0.5, 0.75):                      # tool deviation along the joint-space interpolation
            qm = qq + a * (q1 - qq); pm = (base_T_torso(torso) @ k.fk(qm))[:3, 3]
            dev = max(dev, np.linalg.norm(pm - g) - 0.0)
        mj = max(mj, np.abs(q1 - qq).max()); qq = q1
    return qq, mj, dev
tip = (base_T_torso(0.345) @ k.fk(qg))[:3, 3]
for name, ends in (('up128', [[tip[0], tip[1], 1.28]]), ('up128-in62', [[tip[0], tip[1], 1.28], [0.62, -0.24, 1.28]]),
                   ('up130', [[tip[0], tip[1], 1.30]])):
    goals, cur = [], tip.copy()
    for e in ends:
        e = np.asarray(e, float); n = int(np.ceil(np.linalg.norm(e - cur) / 0.02))
        goals += [cur + a * (e - cur) for a in np.linspace(0, 1, n + 1)[1:]]; cur = e
    try:
        qc, mj, dev = path(qg, goals, 0.345)
    except RuntimeError as ex:
        print(name, 'FAIL', ex); continue
    p = np.array([0.82, ARM_Y, 1.13]); zs = list(np.linspace(1.13, 0.87, 8))
    try:
        qp, mj2, dev2 = path(qc, [np.r_[0.82, ARM_Y, z] for z in zs], 0.345, clear_z=1.03, box=box, outer=outer)
        P = pts(qp, 0.345)
        print(f'{name}: carry max jump {mj:.2f} dev {dev:.3f} | place first-move jump, descent max jump {mj2:.2f} dev {dev2:.3f}; final q {np.round(qp,2)}')
        print('   points', ' '.join(f'{i}:({a:.2f},{b:.2f},{c:.2f})' for i, (a, b, c) in enumerate(P)))
    except RuntimeError as ex:
        print(name, 'place FAIL', ex)

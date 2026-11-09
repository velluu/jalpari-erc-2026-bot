exec(open(__file__.rsplit('/',1)[0] + '/t_place_replay.py').read().split("q0 = np.zeros(7)")[0])
q0 = np.zeros(7); w = k.startup_waypoint(np.clip(q0, k.lower, k.upper)); q = k.compact(w)
row = 4; t = TORSO[row]; z = ROW_Z[row - 1] + GRASP_DZ[row]; reach = REACH[row]
qq = solve(transform([reach - .09, ARM_Y, z]), q.copy(), t)
for adv in np.linspace(0, 0.16, 13)[1:]:
    qq = solve(transform([reach - .09 + adv, ARM_Y, z]), qq, t)
qq = solve(transform([reach + .07, ARM_Y, z + .025]), qq, t)
qq = solve(transform([reach + .07 - .20, ARM_Y, z + .025]), qq, t)
qg = qq.copy()
def run(name, pts_, torso):
    qq = qg.copy(); mj = 0; cur = (base_T_torso(torso) @ k.fk(qq))[:3, 3]
    goals = []
    for e in pts_:
        e = np.asarray(e, float)
        for a in np.linspace(0, 1, 9)[1:]:
            goals.append(cur + a * (e - cur))
        cur = e
    try:
        for g in goals:
            q1 = solve(transform(g), qq, torso); mj = max(mj, np.abs(q1 - qq).max()); qq = q1
    except RuntimeError as ex:
        print(name, 'FAIL', str(ex)[-40:]); return
    # place under v4
    bin_c = np.array([0.70, ARM_Y]); rl = 0.07
    box = (bin_c[0] - 0.28 + rl, bin_c[0] + 0.28 - rl, bin_c[1] - 0.155 + rl, bin_c[1] + 0.155 - rl)
    outer = (bin_c[0] - 0.28 - rl, bin_c[0] + 0.28 + rl, bin_c[1] - 0.155 - rl, bin_c[1] + 0.155 + rl)
    qp = solve(transform([0.82, ARM_Y, 1.13]), qq, 0.345, clear_z=1.03, box=box, outer=outer)
    print(f'{name}: max jump {mj:.2f} end q {np.round(qq,2)} place-high jump {np.abs(qp-qq).max():.2f}')
tip0 = (base_T_torso(t) @ k.fk(qg))[:3, 3]; print('tip at torso', t, np.round(tip0, 3))
# torso up first (0.345) then paths
tip = (base_T_torso(0.345) @ k.fk(qg))[:3, 3]
run('up-then-in', [[tip[0], tip[1], 1.28], [0.50, -0.24, 1.28]], 0.345)
run('up-only', [[tip[0], tip[1], 1.28]], 0.345)
run('in-then-up', [[0.55, -0.24, tip[2]], [0.50, -0.24, 1.28]], 0.345)
run('diag', [[0.50, -0.24, 1.28]], 0.345)
run('out-up (current)', [[tip[0], -0.50, tip[2]], [0.50, -0.24, 1.28]], 0.345)
run('out-up-high', [[tip[0], -0.50, tip[2]], [tip[0], -0.50, 1.28], [0.50, -0.24, 1.28]], 0.345)

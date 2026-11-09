exec(open(__file__.rsplit('/',1)[0] + '/t_replay_carry.py').read().split("tip = (base_T_torso(0.345)")[0])
def grasp(row):
    t = TORSO[row]; z = ROW_Z[row - 1] + GRASP_DZ[row]; reach = REACH[row]
    qq = solve(transform([reach - .09, ARM_Y, z]), q.copy(), t)
    for adv in np.linspace(0, 0.16, 13)[1:]:
        qq = solve(transform([reach - .09 + adv, ARM_Y, z]), qq, t)
    qq = solve(transform([reach + .07, ARM_Y, z + .025]), qq, t)
    return solve(transform([reach + .07 - .20, ARM_Y, z + .025]), qq, t)
for row in (1, 2, 3, 4):
    qg = grasp(row); tip = (base_T_torso(0.345) @ k.fk(qg))[:3, 3]
    qc = qg
    if tip[2] < 1.20:
        qc, mj, dev = path(qg, [g[:3, 3] for g in transport_path(tip)], 0.345)
    tipc = (base_T_torso(0.345) @ k.fk(qc))[:3, 3]
    qa = solve(transform([0.40, -0.55, max(1.20, tipc[2])]), qc, 0.345)
    ja = np.abs(qa - qc).max()
    # tool path deviation during the joint interpolation carry -> aside
    devs = [np.linalg.norm((base_T_torso(0.345) @ k.fk(qc + a * (qa - qc)))[:3, 3] - (tipc + a * (np.array([0.40, -0.55, max(1.20, tipc[2])]) - tipc))) for a in np.linspace(0, 1, 11)]
    qh = solve(transform([0.82, ARM_Y, 1.13]), qa, 0.345, clear_z=1.03, box=box, outer=outer)
    devh = [np.linalg.norm((base_T_torso(0.345) @ k.fk(qa + a * (qh - qa)))[:3, 3] - (np.array([0.40, -0.55, max(1.20, tipc[2])]) + a * (np.array([0.82, ARM_Y, 1.13]) - np.array([0.40, -0.55, max(1.20, tipc[2])])))) for a in np.linspace(0, 1, 11)]
    zmin = min((base_T_torso(0.345) @ k.fk(qa + a * (qh - qa)))[2, 3] for a in np.linspace(0, 1, 11))
    qp, mj2, _ = path(qh, [np.r_[0.82, ARM_Y, z] for z in np.linspace(1.13, 0.87, 8)[1:]], 0.345, clear_z=1.03, box=box, outer=outer)
    print(f'row {row}: carry tip {np.round(tipc,2)} | aside jump {ja:.2f} dev {max(devs):.2f} | aside->high jump {np.abs(qh-qa).max():.2f} dev {max(devh):.2f} tool zmin {zmin:.2f} | descent max jump {mj2:.2f}')

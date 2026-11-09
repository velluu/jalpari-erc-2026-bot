exec(open(__file__.rsplit('/',1)[0] + '/t_replay_aside.py').read().split("for row in (1, 2, 3, 4):")[0])
def seg(qa, qb, pa, pb):
    return max(np.linalg.norm((base_T_torso(0.345) @ k.fk(qa + a * (qb - qa)))[:3, 3] - (pa + a * (pb - pa))) for a in np.linspace(0, 1, 11))
H = np.array([0.82, ARM_Y, 1.13])
for row in (1, 2, 3, 4):
    qg = grasp(row); tip = (base_T_torso(0.345) @ k.fk(qg))[:3, 3]; qc = qg
    if tip[2] < 1.20:
        qc, _, _ = path(qg, [g[:3, 3] for g in transport_path(tip)], 0.345)
    tipc = (base_T_torso(0.345) @ k.fk(qc))[:3, 3]
    qh = solve(transform(H), qc, 0.345, clear_z=1.03, box=box, outer=outer)
    out = [f'direct jump {np.abs(qh-qc).max():.2f} dev {seg(qc, qh, tipc, H):.2f}']
    for A in ([0.40, -0.55, max(1.20, tipc[2])], [tipc[0] - 0.15, -0.55, tipc[2]], [tipc[0], -0.50, tipc[2]], [0.45, -0.45, tipc[2]]):
        A = np.array(A)
        try:
            qa = solve(transform(A), qc, 0.345)
            qh2 = solve(transform(H), qa, 0.345, clear_z=1.03, box=box, outer=outer)
            out.append(f'aside {np.round(A,2)}: {np.abs(qa-qc).max():.2f}/{seg(qc, qa, tipc, A):.2f} then {np.abs(qh2-qa).max():.2f}/{seg(qa, qh2, A, H):.2f}')
        except RuntimeError as e:
            out.append(f'aside {np.round(A,2)} FAIL')
    print(f'row {row}:', ' | '.join(out))

"""Offline, no ROS: the bot's kinematics on the real robot's /robot_description (PAL's xacro, team session-1 bag, lane F
bagq_g8ns, md5 b1b9e0e0), which has no gripper_<side>_grasping_link. Checks that Arm's Kinematics builds after
with_grasping_links, that the sim URDF passes through unchanged, and that tool FK on real + injection equals FK on the sim
URDF (two FK codes: Kinematics in torso_lift_link, UrdfFK in base_footprint with the torso), 20 random in-limit sets.
  python3 tests/t_urdf_real.py        exits 1 on any failure"""
import importlib, os, sys, types
import numpy as np
from scipy.spatial.transform import Rotation
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class _Stub(types.ModuleType):              # erc_bot.live imports rclpy and ROS messages; only its constants are used here
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return type(k, (), {'__init__': lambda s, *a, **kw: None})


for m in ('cv2', 'rclpy', 'rclpy.action', 'rclpy.node', 'rclpy.qos', 'rclpy.time', 'builtin_interfaces', 'builtin_interfaces.msg',
          'control_msgs', 'control_msgs.action', 'geometry_msgs', 'geometry_msgs.msg', 'nav_msgs', 'nav_msgs.msg', 'sensor_msgs',
          'sensor_msgs.msg', 'std_msgs', 'std_msgs.msg', 'tf2_ros', 'trajectory_msgs', 'trajectory_msgs.msg'):
    try:
        importlib.import_module(m)
    except Exception:
        sys.modules[m] = _Stub(m)
from erc_bot.live import LIVE_LIMITS, MARGIN
from erc_bot.kinematics import GRASP_OFFSET, Kinematics, with_grasping_links
from erc_bot.policy_fk import UrdfFK

real = open(os.path.join(HERE, 'data', 'tiago_pro_real.urdf')).read()
sim = open(os.path.join(HERE, 'data', 'tiago_pro_sim.urdf')).read()
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f': {detail}' if detail else ''))
    if not ok:
        fails.append(name)


def kin(urdf, side, tip='grasping'):
    return Kinematics(urdf, tip=f'gripper_{side}_{tip}_link', limits=LIVE_LIMITS, margin=MARGIN)


try:
    kin(real, 'right'); check('real URDF as-is fails like c4', False, 'built without the injection')
except KeyError as e:
    check('real URDF as-is fails like c4', True, f'KeyError {e}')
check('sim URDF passes through unchanged', with_grasping_links(sim) is sim)
fixed = with_grasping_links(real)
i = real.rindex('</robot>')
check('real URDF gets both grasping links, nothing else changed', all(f'<link name="gripper_{s}_grasping_link"/>' in fixed for s in
      ('right', 'left')) and fixed.startswith(real[:i]) and fixed.endswith(real[i:]), f'{len(fixed) - len(real)} characters added')
check('injection is idempotent', with_grasping_links(fixed) is fixed)

rng = np.random.default_rng(1106)
for side in ('right', 'left'):
    kr, ks = kin(fixed, side), kin(sim, side)
    check(f'{side}: Kinematics builds on real + injection', kr.names == ks.names, ' '.join(kr.names))
    print(f'  {side} real range ', np.round(kr.lower, 3).tolist(), np.round(kr.upper, 3).tolist())
    print(f'  {side} sim range  ', np.round(ks.lower, 3).tolist(), np.round(ks.upper, 3).tolist())
    lo, hi = np.maximum(kr.lower, ks.lower), np.minimum(kr.upper, ks.upper)
    j7 = kr.names.index(f'arm_{side}_7_joint')
    check(f'{side}: arm 7 range inside LIVE_LIMITS less the margin', abs(kr.upper[j7] - (np.radians(150) - MARGIN)) < 1e-9
          and abs(kr.lower[j7] + (np.radians(150) - MARGIN)) < 1e-9, f'real +-{kr.upper[j7]:.3f}, sim +-{ks.upper[j7]:.3f}')
    kb = kin(sim, side, 'base')                     # the tool() fallback: gripper base (TF) composed with GRASP_OFFSET
    fr, ff = UrdfFK(fixed), UrdfFK(sim)
    dp = da = dpu = dau = dpb = dab = 0.0
    for _ in range(20):
        q = rng.uniform(lo, hi)
        A, B = kr.fk(q), ks.fk(q)
        dp = max(dp, np.linalg.norm(A[:3, 3] - B[:3, 3])); da = max(da, Rotation.from_matrix(A[:3, :3] @ B[:3, :3].T).magnitude())
        C = kb.fk(q) @ GRASP_OFFSET
        dpb = max(dpb, np.linalg.norm(C[:3, 3] - B[:3, 3])); dab = max(dab, Rotation.from_matrix(C[:3, :3] @ B[:3, :3].T).magnitude())
        qd = dict(zip(kr.names, q)); qd['torso_lift_joint'] = rng.uniform(0.0, 0.35)
        U, V = fr.pose(f'gripper_{side}_grasping_link', qd), ff.pose(f'gripper_{side}_grasping_link', qd)
        dpu = max(dpu, np.linalg.norm(U[:3, 3] - V[:3, 3])); dau = max(dau, Rotation.from_matrix(U[:3, :3] @ V[:3, :3].T).magnitude())
    check(f'{side}: tool FK real + injection = sim (Kinematics, torso_lift_link), 20 sets', dp < 1e-3 and da < 1e-3,
          f'max {dp * 1e3:.1e} mm, {da * 1e3:.1e} mrad')
    check(f'{side}: tool FK real + injection = sim (UrdfFK, base_footprint, torso 0-0.35), 20 sets', dpu < 1e-3 and dau < 1e-3,
          f'max {dpu * 1e3:.1e} mm, {dau * 1e3:.1e} mrad')
    check(f'{side}: sim gripper base @ GRASP_OFFSET = sim grasping link (tool() fallback math), 20 sets', dpb < 1e-3 and dab < 1e-3,
          f'max {dpb * 1e3:.1e} mm, {dab * 1e3:.1e} mrad')
print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)

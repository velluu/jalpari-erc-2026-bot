"""Offline, no ROS: Arm.park ends on the same posture from any start pose (Arm.park_plan: c4's raised waypoint and tuck
fitted from the current joints, then one more move to the canonical tuck when the tuck is more than 0.05 rad from it).
Canonical = the same fits from zero joints (the sim spawn). Starts: zeros, the slot sheet's s02 present pose (lane F
F-k9-ctl gripper_present.json), the flipped-branch HOLD_MEASURED, the real robot's start pose (lane F, session-1 bag), 20
random in-limit poses; on the sim URDF and on the real URDF plus the grasping-link injection. Also prints c4's local fit
(canonical=False) from the same starts, and the park move's timing at the bot's and the slot sheet's speed.
  python3 tests/t_park_branch.py      exits 1 on any failure"""
import importlib, os, sys, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class _Stub(types.ModuleType):
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
from erc_bot.live import LIVE_LIMITS, MARGIN, SPEED, SPEED_USE, ARM_RIGHT
from erc_bot.kinematics import Kinematics, with_grasping_links
from erc_bot.arm import Arm

SPAWN_TUCK = np.array([0.454, 1.064, 1.533, -2.373, -1.015, -1.815, 0.000])      # lead 09:50, c4 park from the sim spawn
STARTS = {'zeros (sim spawn)': np.zeros(7),
          's02 present': np.array([-4.391847, -2.372578, -1.131927, -2.373461, -1.165121, 0.844931, 1.708958]),
          'HOLD flipped': np.array([-4.075, -1.477, -0.792, -1.865, 2.632, -1.108, -1.580]),
          'HOLD good': np.array([-4.384, -1.641, -0.767, -1.919, -0.499, 0.870, 1.737]),
          'real start': np.array([0.469, -1.0, -0.5, -1.55, -0.65, -1.5, 0.0])}
fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f': {detail}' if detail else ''))
    if not ok:
        fails.append(name)


ARM = Arm.__new__(Arm); ARM._home = {}


def park(k, cur, canonical, side=-1):
    """(raised, final posture, extra move?) as Arm.park commands them."""
    raised, tuck, home = ARM.park_plan(k, cur, side, canonical)
    return raised, home, home is not tuck


def dur(names, a, b, use):
    return max([abs(y - x) / (SPEED[n] * use) for n, x, y in zip(names, a, b)] + [0.6])


for label, f in (('sim', 'tiago_pro_sim.urdf'), ('real', 'tiago_pro_real.urdf')):
    urdf = with_grasping_links(open(os.path.join(HERE, 'data', f)).read())
    k = Kinematics(urdf, tip='gripper_right_grasping_link', limits=LIVE_LIMITS, margin=MARGIN)
    kl = Kinematics(urdf, tip='gripper_left_grasping_link', limits=LIVE_LIMITS, margin=MARGIN)
    assert k.names == ARM_RIGHT
    c_raised, c_tuck, _ = park(k, np.zeros(7), True)
    check(f'{label}: canonical tuck = c4 park from the sim spawn', np.abs(c_tuck - SPAWN_TUCK).max() < 0.002,
          np.round(c_tuck, 3).tolist())
    rng = np.random.default_rng(611)
    starts = dict(STARTS, **{f'random {i}': rng.uniform(k.lower, k.upper) for i in range(20)})
    worst = 0.0; slow = 0.0; extra = 0
    for name, q0 in starts.items():
        raised, final, moved = park(k, q0, True)
        extra += moved
        d = np.abs(final - c_tuck).max()
        worst = max(worst, d)
        _, old, _ = park(k, q0, False)
        q0c = np.clip(q0, k.lower, k.upper)
        t6, t35 = dur(k.names, old, final, SPEED_USE), dur(k.names, old, final, 0.35)
        slow = max(slow, t35)
        if not name.startswith('random') or d > 0.05:
            print(f'   {name:18s} final off canonical {d:.4f} rad, c4 tuck off {np.abs(old - c_tuck).max():.3f} rad, extra move '
                  + (f'{t6:.1f} s at {SPEED_USE} of the caps ({t35:.1f} s at 0.35)' if moved else 'none'))
        if d >= 0.05 or not name.startswith('random'):
            check(f'{label}: {name} ends on the canonical tuck', d < 0.05, f'{d:.4f} rad')
    check(f'{label}: all {len(starts)} starts end within 0.05 rad of the canonical tuck', worst < 0.05,
          f'worst {worst:.4f} rad, {extra} needed the extra move')
    check(f'{label}: the sim spawn needs no extra move (c4 commands unchanged)', not park(k, np.zeros(7), True)[2])
    # Mirror for the left arm: the canonical left posture is the same from every start too
    lz = park(kl, np.zeros(7), True, side=1)[1]
    lw = max(np.abs(park(kl, rng.uniform(kl.lower, kl.upper), True, side=1)[1] - lz).max() for _ in range(10))
    check(f'{label}: left arm, 10 random starts on its canonical tuck', lw < 0.05, f'worst {lw:.4f} rad')
    print(f'   longest extra move at 0.35 of the caps: {slow:.1f} s (the park waits that plus 8 s, at least 15 s)')
    v = max(abs(y - x) / dur(k.names, np.zeros(7), c_raised, SPEED_USE) / SPEED[n] for n, x, y in zip(k.names, np.zeros(7), c_raised))
    check(f'{label}: joint speed of the planned move <= the caps', v <= SPEED_USE + 1e-9, f'peak {v:.2f} of the cap (linear one-point goal)')
print('ALL PASS' if not fails else f'{len(fails)} FAILED: {fails}')
sys.exit(1 if fails else 0)

import rclpy, numpy as np
from erc_bot.live import Live
from erc_bot.arm import Arm
rclpy.init(); r = Live('t_park'); r.ready(); a = Arm(r)
print('right lower/upper', np.round(a.k.lower, 2), np.round(a.k.upper, 2))
t0 = r.now(); a.park(); print('parked in', round(r.now() - t0, 1), 's sim, tool', np.round(a.tool()[:3, 3], 3))
for z in (0.0, 0.3, 0.1):
    t0 = r.now()
    try:
        r.torso(z)
    except Exception as e:
        print('torso err', e)
    print('torso ->', z, 'at', round(r.joints['torso_lift_joint'], 3), 'after', round(r.now() - t0, 1), 's')
print('rejections', r.rejections[-3:])

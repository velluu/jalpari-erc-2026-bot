import rclpy
from erc_bot.live import Live
rclpy.init(); r = Live('t_torso'); r.ready()
for tgt, dur in ((0.0, 6.0), (0.3, 9.0)):
    print('from', round(r.joints['torso_lift_joint'], 4), 'to', tgt, 'commanded over', dur, 's (cap 0.035 m/s)')
    r.send('torso_controller', [tgt], duration=dur, wait=False)
    t0 = r.now(); prev = r.joints['torso_lift_joint']
    while r.now() - t0 < 40 and abs(r.joints['torso_lift_joint'] - tgt) > 0.003:
        r.wait(2.0); q = r.joints['torso_lift_joint']; print(' ', round(r.now() - t0, 1), round(q, 4), 'v', round((q - prev) / 2.0, 4)); prev = q
    print('  reached in', round(r.now() - t0, 1), 's')

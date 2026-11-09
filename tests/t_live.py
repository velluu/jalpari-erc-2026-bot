"""Live interface smoke test against the dev sim (or the robot): ready, gripper calibration, head, torso, a rejection."""
import rclpy
from erc_bot.live import Live, Abort
rclpy.init(); r = Live('t_live')
r.ready(); r.say(f'ready, {len(r.joints)} joints, urdf {len(r.urdf)} chars')
print('gripper start', round(r.joints['gripper_right_finger_joint'], 3))
print('open ->', round(r.gripper(False), 3)); print('close on air ->', round(r.gripper(True), 3)); print('open ->', round(r.gripper(False), 3))
r.head(0.4, -0.3); print('head', round(r.joints['head_1_joint'], 3), round(r.joints['head_2_joint'], 3))
r.torso(0.2); print('torso', round(r.joints['torso_lift_joint'], 3)); r.torso(0.0)
r.head(0.0, 0.0)
try:
    from erc_bot import live
    live.MARGIN = -0.2           # deliberately outside the live range: the checking layer must refuse it
    r.send('head_controller', [1.4, 0.0])
    print('REJECTION TEST FAILED: goal accepted')
except Abort as e:
    print('rejected as expected:', str(e)[:150])
cam = r.camera(); print('camera', None if cam is None else (cam[0].shape, round(float(cam[1][360, 640]), 3), cam[2]))
print('scan points', None if r.scan_points() is None else r.scan_points().shape, 'odom', r.base_in_odom().round(3))

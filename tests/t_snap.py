import sys, rclpy, cv2
from erc_bot.live import Live
rclpy.init(); r = Live('t_snap', sim_time=True); r.ready()
pan, tilt = float(sys.argv[1]), float(sys.argv[2])
r.head(pan, tilt); r.wait(1.0)
cam = r.camera(); cv2.imwrite(sys.argv[3], cam[0]); print('saved', r.joints['gripper_right_finger_joint'])

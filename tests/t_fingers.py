"""Where the right fingertips sit in the grasping-link frame, open and closed (which axis the gripper closes along)."""
import rclpy, numpy as np
from erc_bot.live import Live
rclpy.init(); r = Live('t_fingers', sim_time=True); r.ready()
def tips():
    out = {}
    for f in ('gripper_right_fingertip_left_link', 'gripper_right_fingertip_right_link', 'gripper_right_inner_finger_left_link', 'gripper_right_inner_finger_right_link', 'gripper_right_base_link'):
        try:
            out[f.replace('gripper_right_', '')] = np.round(r.matrix('gripper_right_grasping_link', f)[:3, 3], 4)
        except Exception as e:
            out[f] = type(e).__name__
    return out
for closed in (False, True, False):
    p = r.gripper(closed); r.wait(0.5)
    print('closed' if closed else 'open', 'live pos', round(p, 3), tips())

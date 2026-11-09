import sys, rclpy, numpy as np, cv2
from erc_bot.live import Live
from erc_bot.perception import Eyes
rclpy.init(); r = Live('t_look', sim_time=True); r.ready(); e = Eyes(r, '/opt/model/det_v3_736x1280.onnx')
print('odom', r.base_in_odom().round(3), 'head', round(r.joints['head_1_joint'], 3), round(r.joints['head_2_joint'], 3))
for pan, tilt in ((0.0, 0.12), (0.0, 0.25), (0.0, 0.0)):
    r.head(pan, tilt); r.wait(0.5)
    dets, pose = e.look()
    print(f'pan {pan} tilt {tilt}: head now', round(r.joints['head_1_joint'], 3), round(r.joints['head_2_joint'], 3), 'cam z', None if pose is None else round(pose[2, 3], 3))
    for d in dets or []:
        print('   ', d['class'], d['score'], [round(v) for v in d['box']], np.round(d['p'], 2), round(d['z'], 2))
cam = r.camera(); cv2.imwrite('/opt/results/dev/look.jpg', cam[0]); print('saved')

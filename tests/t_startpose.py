"""Start posture of both arms (fresh sim): joint values and the compact test the park uses."""
import numpy as np, rclpy
from erc_bot.live import Live
from erc_bot.arm import Arm
rclpy.init(); r = Live('t_startpose', sim_time=True); r.ready(); a = Arm(r)
for model, side in ((a.k, -1), (a.k_left, 1)):
    q = a.q(model.names); P = model.points(q)
    print(model.names[0], np.round(q, 3).tolist(), 'x max', round(P[:, 0].max(), 3), '|y| max', round(abs(P[:, 1]).max(), 3),
          'tip', np.round(P[-1], 3).tolist(), 'low z', round(P[:, 2].min(), 3))

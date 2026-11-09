"""Right-arm chain from the live URDF: index, joint, type, origin (index matches Kinematics.points rows)."""
import xml.etree.ElementTree as ET, rclpy
from erc_bot.live import Live
rclpy.init(); r = Live('t_chain', sim_time=True); r.ready()
root = ET.fromstring(r.urdf); parents = {j.find('child').get('link'): j for j in root.findall('joint')}
tip, ch = 'gripper_right_grasping_link', []
while tip != 'torso_lift_link':
    j = parents[tip]; ch.append((j.get('name'), j.get('type'), j.find('origin').get('xyz'))); tip = j.find('parent').get('link')
for i, c in enumerate(reversed(ch)):
    print(i, *c)

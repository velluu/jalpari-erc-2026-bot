#!/usr/bin/env python3
"""One process publishes a 1280x720 32FC1 image (3.7 MB, the sim depth frame) RELIABLE at RATE Hz, a second process
subscribes BEST_EFFORT depth 1 and counts frames for SECS. Prints the received rate."""
import os, sys, time, subprocess
role = sys.argv[1] if len(sys.argv) > 1 else 'main'
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
RATE, SECS = float(os.environ.get('RATE', '15')), float(os.environ.get('SECS', '15'))
if role == 'pub':
    rclpy.init(); n = rclpy.create_node('buf_pub')
    p = n.create_publisher(Image, '/buf/depth', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
    m = Image(); m.height, m.width, m.encoding, m.step = 720, 1280, '32FC1', 5120; m.data = bytes(720 * 5120)
    t = time.time()
    while time.time() - t < SECS + 8:
        p.publish(m); rclpy.spin_once(n, timeout_sec=0); time.sleep(1 / RATE)
elif role == 'sub':
    rclpy.init(); n = rclpy.create_node('buf_sub'); c = [0]
    n.create_subscription(Image, '/buf/depth', lambda m: c.__setitem__(0, c[0] + 1), QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    t = time.time()
    while time.time() - t < 3:
        rclpy.spin_once(n, timeout_sec=0.05)
    c[0] = 0; t = time.time()
    while time.time() - t < SECS:
        rclpy.spin_once(n, timeout_sec=0.05)
    print(f"rmem_max {open('/proc/sys/net/core/rmem_max').read().strip()} uri {os.environ.get('CYCLONEDDS_URI','').split('/')[-1]} "
          f"best-effort subscriber {c[0] / SECS:.2f} Hz of {RATE:.0f} published", flush=True)
else:
    pub = subprocess.Popen([sys.executable, __file__, 'pub'])
    time.sleep(2)
    subprocess.run([sys.executable, __file__, 'sub'])
    pub.kill()

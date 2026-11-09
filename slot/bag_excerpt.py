#!/usr/bin/env python3
"""Small excerpt of a camera bag for sending home: one colour JPEG every --every seconds (plus its aligned depth PNG),
and the bag's topic counts.   python3 slot/bag_excerpt.py BAGDIR OUTDIR [--every 2.0]"""
import argparse, json, os
import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bag'); ap.add_argument('out'); ap.add_argument('--every', type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id='sqlite3'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    counts, last, lastd, n = {}, -1e9, -1e9, 0
    while r.has_next():
        topic, data, t = r.read_next()
        counts[topic] = counts.get(topic, 0) + 1
        ts = t * 1e-9
        if topic.endswith('color/image_raw/compressed') and ts - last >= a.every:
            m = deserialize_message(data, CompressedImage)
            open(os.path.join(a.out, f'colour_{t}.jpg'), 'wb').write(bytes(m.data)); last = ts; n += 1
        elif topic.endswith('compressedDepth') and ts - lastd >= a.every:
            m = deserialize_message(data, CompressedImage); buf = bytes(m.data); k = buf.find(b'\x89PNG', 0, 64)
            if k >= 0:
                open(os.path.join(a.out, f'depth_{t}.png'), 'wb').write(buf[k:])
            lastd = ts
    json.dump({'bag': a.bag, 'counts': counts, 'types': types, 'frames_kept': n}, open(os.path.join(a.out, 'bag_counts.json'), 'w'), indent=1)
    print(f'{a.bag}: {n} frames kept, counts {counts}')


if __name__ == '__main__':
    main()

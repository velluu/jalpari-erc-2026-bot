#!/usr/bin/env python3
"""Slot step 3 (read-only): camera exposure and white balance sample, compressed stream sizes, depth coverage.

  python3 slot/s03_camera.py [--frames 30] [--tag shelf]

Point the head at the shelf first (the bot's start view). Prints per-channel means, clipping, grey-world white balance
gains, sharpness, depth valid fraction and median, and the camera driver's exposure / white balance / gain parameters
when the driver exposes them (read only). Saves 5 colour frames and one depth PNG under $SLOT_OUT/camera_<tag>/,
camera_<tag>.json. The organisers post the fixed exposure and white balance values at the session: write them in the
sheet too.
"""
import argparse, os
import numpy as np
import cv2
from common import start, finish, save, out_dir, SIM, stamp
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from rcl_interfaces.srv import ListParameters, GetParameters

C = '/head_front_camera'
KEYS = ('exposure', 'white_balance', 'gain', 'auto', 'gamma', 'brightness', 'contrast', 'saturation', 'fps', 'profile')


def driver_params(n, timeout=2.0):
    found = {}
    for nm, ns in n.get_node_names_and_namespaces():
        full = f"{ns.rstrip('/')}/{nm}"
        if not any(k in full.lower() for k in ('camera', 'realsense', 'head_front')):
            continue
        lc = n.create_client(ListParameters, f'{full}/list_parameters')
        if not lc.wait_for_service(timeout_sec=timeout):
            continue
        f = lc.call_async(ListParameters.Request(depth=10)); n.spin_until(f.done, timeout)
        if not f.done():
            continue
        names = [p for p in f.result().result.names if any(k in p.lower() for k in KEYS)]
        if not names:
            continue
        gc = n.create_client(GetParameters, f'{full}/get_parameters')
        if not gc.wait_for_service(timeout_sec=timeout):
            continue
        g = gc.call_async(GetParameters.Request(names=names[:200])); n.spin_until(g.done, timeout)
        if g.done():
            vals = {}
            for p, val in zip(names, g.result().values):
                vals[p] = next((getattr(val, a) for a in ('integer_value', 'double_value', 'bool_value', 'string_value')
                                if val.type == {'integer_value': 2, 'double_value': 3, 'bool_value': 1, 'string_value': 4}[a]), None)
            found[full] = vals
    return found


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--frames', type=int, default=30); ap.add_argument('--tag', default='shelf')
    a = ap.parse_args()
    n, v = start(f's03_camera_{a.tag}')
    col, dep = [], []
    n.create_subscription(CompressedImage, f'{C}/color/image_raw/compressed', lambda m: col.append((n.now(), m)), qos_profile_sensor_data)
    n.create_subscription(CompressedImage, f'{C}/aligned_depth_to_color/image_raw/compressedDepth', lambda m: dep.append((n.now(), m)), qos_profile_sensor_data)
    ok = n.spin_until(lambda: len(col) >= a.frames and len(dep) >= max(3, a.frames // 3), 60.0 if SIM else 15.0)
    v.check('frames', len(col) >= 3, f'{len(col)} colour, {len(dep)} depth frames',
            'no compressed colour: check ros2 topic list | grep compressed, the image_transport plugin on the robot side')
    if len(col) < 3:
        return finish(n, v)
    d = os.path.join(out_dir(), f'camera_{a.tag}'); os.makedirs(d, exist_ok=True)
    imgs = []
    for k, (_, m) in enumerate(col):
        im = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
            if k % max(1, len(col) // 5) == 0 and len(os.listdir(d)) < 6:
                cv2.imwrite(os.path.join(d, f'colour_{k:02d}.jpg'), im)
    st = np.stack(imgs).astype(np.float32)
    bgr_mean = st.mean(axis=(0, 1, 2)); bgr_std = st.std(axis=(0, 1, 2))
    lum = (0.114 * st[..., 0] + 0.587 * st[..., 1] + 0.299 * st[..., 2])
    clip_hi = float((st.max(axis=3) >= 250).mean()); clip_lo = float((st.max(axis=3) <= 5).mean())
    g = bgr_mean[1]
    wb = {'G/R': float(g / max(bgr_mean[2], 1e-3)), 'G/B': float(g / max(bgr_mean[0], 1e-3))}
    sharp = float(np.mean([cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for im in imgs[:10]]))
    flick = float(lum.mean(axis=(1, 2)).std())
    ts = [t for t, _ in col]; hz = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
    kb = float(np.mean([len(m.data) for _, m in col]) / 1000)
    v.check('exposure', 0.15 * 255 < lum.mean() < 0.85 * 255 and clip_hi < 0.05, f'mean luminance {lum.mean():.1f}, '
            f'{clip_hi * 100:.2f} % clipped white, {clip_lo * 100:.2f} % black, frame-to-frame luminance std {flick:.2f}',
            'image too dark or bright for the sim-trained detector: write down the organisers\' exposure value, record anyway '
            '(the frames are what the fine-tune needs)', lum=float(lum.mean()), clip_hi=clip_hi, clip_lo=clip_lo, flicker=flick)
    v.check('white balance', 0.6 < wb['G/R'] < 1.6 and 0.6 < wb['G/B'] < 1.6,
            f"BGR means {np.round(bgr_mean, 1).tolist()} std {np.round(bgr_std, 1).tolist()}, grey-world gains G/R {wb['G/R']:.3f} "
            f"G/B {wb['G/B']:.3f}", 'strong colour cast: colour classes (red/yellow, blue/green) may swap, check the live '
            'detector view (kit README step 3)', bgr_mean=bgr_mean, wb=wb)
    v.info(f'colour {hz:.1f} Hz received, {kb:.0f} kB per frame ({kb * hz / 1000:.2f} MB/s), sharpness (Laplacian var) {sharp:.0f}')
    dstat = {}
    if dep:
        _, m = dep[-1]; buf = bytes(m.data); k = buf.find(b'\x89PNG', 0, 64)
        dm = cv2.imdecode(np.frombuffer(buf[k:], np.uint8), cv2.IMREAD_UNCHANGED) if k >= 0 else None
        if dm is not None:
            valid = dm > 0
            dstat = {'shape': list(dm.shape), 'dtype': str(dm.dtype), 'valid': float(valid.mean()),
                     'median_m': float(np.median(dm[valid]) / 1000) if valid.any() else None,
                     'kB': float(np.mean([len(x.data) for _, x in dep]) / 1000), 'format': m.format}
            cv2.imwrite(os.path.join(d, 'depth_mm.png'), dm)
    v.check('aligned depth', bool(dstat) and dstat['valid'] > 0.3 and dstat['shape'] == [720, 1280],
            f'{dstat}', 'no or sparse aligned depth: the bot lifts detections with it. Record anyway, send the depth PNG', **dstat)
    par = driver_params(n)
    v.info(f'camera driver parameters: {par if par else "none exposed (write down the organisers posted values)"}')
    save(f'camera_{a.tag}.json', {'frames': len(imgs), 'hz': hz, 'kB': kb, 'bgr_mean': bgr_mean, 'bgr_std': bgr_std,
                                  'lum_mean': float(lum.mean()), 'clip_hi': clip_hi, 'clip_lo': clip_lo, 'wb_grey_world': wb,
                                  'sharpness': sharp, 'flicker': flick, 'depth': dstat, 'driver_params': par,
                                  'head': [n.joints.get('head_1_joint'), n.joints.get('head_2_joint')]})
    return finish(n, v)


if __name__ == '__main__':
    raise SystemExit(main())

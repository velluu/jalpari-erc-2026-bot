#!/usr/bin/env python3
"""Slot step 1 (read-only, plus a two-joint margin probe): the robot's own URDF, its joint ranges against the Guide's table and the sim URDF, the
camera intrinsics and mount, and where every joint sits now.

  python3 slot/s01_urdf.py [--no-probe]

Prints, for arm joints 1 and 5 on both arms, the joint axis and the live limits and whether they match the sim URDF
(right 1 -270..30 deg) or the Guide's table (right 1 -30..270 deg), and the torso limits. The margin probe sends torso
0.285 then 0.31 m, 0.065 then 0.04 m, and head pan 75 deg - 0.065 then - 0.04 rad (harmless if accepted) and reports
which the layer refuses (about 2 min on the real torso).

Writes robot_description.urdf, urdf_limits.json, camera_info.json, camera_tf.json, joints_start.json to $SLOT_OUT.
The bot (erc_bot/arm.py) takes arm joints 1 and 5 from /robot_description, so the convention found here is what the
bot will use; this step only records which one it is.
"""
import math, os, xml.etree.ElementTree as ET
import numpy as np
from common import start, finish, save, out_dir, GUIDE, MARGIN, D, SIM, CONTROLLERS
import rclpy
from rclpy.time import Time as RTime
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation

# sim URDF (erc_ws tiago_pro_description, as read by erc-gym's shim): joints 1 and 5 are the Guide's mirrored
SIM15 = {'arm_right_1_joint': (D(-270), D(30)), 'arm_left_1_joint': (D(-30), D(270)),
         'arm_right_5_joint': (D(-90), D(210)), 'arm_left_5_joint': (D(-210), D(90))}
GUIDE_K = (909.8, 908.6, 646.8, 378.0)


def margin_probe(n, v, lim):
    """The checking layer's margin in practice: goals just inside and just outside the bound less 0.05, run in sequence
    (inside first, then outside from there, so an accepted outside goal is a 2.5 cm / 2.5 deg step). Every target lies
    inside the joint's physical range, so an accepted probe is harmless. Torso upper and lower, head pan upper. Each
    joint goes back where it started."""
    out = []
    tl = lim.get('torso_lift_joint', {}); lo_t, hi_t = tl.get('lower', 0.0), tl.get('upper', 0.35)
    hp = lim.get('head_1_joint', {}); hi_h = hp.get('upper', D(75))
    plan = (('torso_controller', 'torso_lift_joint', [('upper', hi_t - 0.065, False), ('upper', hi_t - 0.04, True),
                                                       ('lower', lo_t + 0.065, False), ('lower', lo_t + 0.04, True)]),
            ('head_controller', 'head_1_joint', [('upper', hi_h - 0.065, False), ('upper', hi_h - 0.04, True)]))
    for ctrl, j, seq in plan:
        names = CONTROLLERS[ctrl]
        home = [n.joints[x] for x in names]
        res = []
        for side, q, expect in seq:
            tgt = [n.joints[x] for x in names]; tgt[names.index(j)] = q
            n_rej = len(n.rejections)
            r = n.move(ctrl, tgt, frac=0.5, timeout=30.0)
            n.spin_for(0.3)
            rej = [x for _, x in n.rejections[n_rej:]]
            refused = bool(rej) or r.get('accepted') is False or r.get('error_code') not in (None, 0)
            bound = hi_t if (side == 'upper' and 'torso' in j) else lo_t if 'torso' in j else hi_h
            res.append({'side': side, 'bound': bound, 'target': round(q, 4), 'inside_by': round(abs(bound - q), 4), 'expect_refused': expect,
                        'refused': refused, 'reached': abs(n.joints[j] - q) < (0.005 if 'torso' in j else 0.02),
                        'error_code': r.get('error_code'), 'error_string': r.get('error_string'), 'rejection': rej[0] if rej else None})
        n.move(ctrl, home, frac=0.5, timeout=60.0); n.spin_for(0.3)
        for side in sorted({x['side'] for x in res}):
            i, o = [x for x in res if x['side'] == side]
            v.check(f'margin {j} {side}', o['refused'] and not i['refused'],
                    f"bound {o['bound']:.4f}: {i['inside_by']:.3f} inside it {'refused' if i['refused'] else 'accepted'}, "
                    f"{o['inside_by']:.3f} inside it {'refused' if o['refused'] else 'ACCEPTED'}; reason {o['rejection'] or o['error_string']!r}",
                    'the live margin differs from the Guide 0.05: the bot keeps 0.07 (LIVE_LIMITS less MARGIN); if the 0.065 goal '
                    'was refused too, tell the lead the margin is larger than 0.065', joint=j, probes=[i, o])
        out.append({'joint': j, 'probes': res})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--no-probe', action='store_true', help='read-only, skip the margin probe')
    a = ap.parse_args()
    n, v = start('s01_urdf')
    box = {}
    n.create_subscription(String, '/robot_description', lambda m: box.setdefault('urdf', m.data),
                          QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    n.create_subscription(CameraInfo, '/head_front_camera/color/camera_info', lambda m: box.setdefault('ci', m), qos_profile_sensor_data)
    tf = Buffer(); TransformListener(tf, n)
    n.spin_until(lambda: 'urdf' in box and 'ci' in box and len(n.joints) >= 20
                 and tf.can_transform('base_footprint', 'head_front_camera_color_optical_frame', RTime()), 10.0)
    if not v.check('urdf received', 'urdf' in box, f"{len(box.get('urdf', ''))} chars", 'no /robot_description, rerun s00; ask the organisers'):
        return finish(n, v)
    open(os.path.join(out_dir(), 'robot_description.urdf'), 'w').write(box['urdf'])
    root = ET.fromstring(box['urdf'])
    lim = {}
    for j in root.iter('joint'):
        l = j.find('limit')
        if l is not None and j.get('type') in ('revolute', 'prismatic'):
            lim[j.get('name')] = {'lower': float(l.get('lower', 'nan')), 'upper': float(l.get('upper', 'nan')),
                                  'velocity': float(l.get('velocity', 'nan')), 'effort': float(l.get('effort', 'nan')),
                                  'mimic': j.find('mimic') is not None,
                                  'axis': [float(x) for x in (j.find('axis').get('xyz') if j.find('axis') is not None else '1 0 0').split()],
                                  'parent': j.find('parent').get('link') if j.find('parent') is not None else None,
                                  'child': j.find('child').get('link') if j.find('child') is not None else None}
    want = [k for k in GUIDE]
    missing = [k for k in want if k not in lim]
    v.check('joints present', not missing, f'{len(lim)} limited joints, Guide joints missing: {missing}',
            'joint names differ from the sim: send robot_description.urdf to the lead before running the bot')
    rows, diff = [], []
    for k in want:
        if k not in lim:
            continue
        lo, hi = lim[k]['lower'], lim[k]['upper']
        glo, ghi = GUIDE[k]
        d = max(abs(lo - glo), abs(hi - ghi))
        rows.append({'joint': k, 'urdf': [round(lo, 4), round(hi, 4)], 'guide': [round(glo, 4), round(ghi, 4)], 'max_diff': round(d, 4)})
        if d > 0.01:
            diff.append(k)
    conv = {}
    for k, (slo, shi) in SIM15.items():
        if k in lim:
            lo, hi = lim[k]['lower'], lim[k]['upper']
            conv[k] = ('guide' if max(abs(lo - GUIDE[k][0]), abs(hi - GUIDE[k][1])) < 0.02 else
                       'sim' if max(abs(lo - slo), abs(hi - shi)) < 0.02 else f'other [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg')
    for k, c in conv.items():
        L = lim[k]
        v.info(f"{k}: axis {L['axis']}, limits [{L['lower']:.4f}, {L['upper']:.4f}] rad = [{math.degrees(L['lower']):.1f}, "
               f"{math.degrees(L['upper']):.1f}] deg -> matches the {c.upper() if c in ('guide', 'sim') else c} "
               f"(sim URDF: right 1 -270..30 deg; Guide table: right 1 -30..270 deg)")
    t = lim.get('torso_lift_joint')
    if t:
        v.info(f"torso_lift_joint: axis {t['axis']}, limits [{t['lower']:.4f}, {t['upper']:.4f}] m, velocity {t['velocity']} "
               f"(Guide 0 .. 0.35 m, checking-layer bound 0.05 .. 0.30 m with the 0.05 m margin)")
    v.check('arm 1 and 5 convention', len(set(conv.values())) == 1 and list(conv.values())[0] in ('guide', 'sim'),
            f"{sorted(set(conv.values()))} (bot reads the live URDF, so either works)",
            'mixed or new ranges: send urdf_limits.json to the lead; the bot still clamps to the live URDF', convention=conv)
    other = [k for k in diff if k not in SIM15 and not (SIM and 'gripper' in k)]   # sim URDF grippers are on the sim scale
    for k in other:
        r = next(x for x in rows if x['joint'] == k)
        v.info(f"NOTE {k}: live URDF {r['urdf']} rad vs Guide {r['guide']} rad (the bot plans inside the intersection, send urdf_limits.json)")
    # where every joint sits now, against its live range less the 0.05 margin
    near = []
    for k, q in sorted(n.joints.items()):
        if k in lim and not lim[k]['mimic']:
            m = 0.0 if 'gripper' in k else MARGIN
            if q < lim[k]['lower'] + m or q > lim[k]['upper'] - m:
                near.append((k, round(q, 3)))
    for k, q in near:
        v.info(f"NOTE {k} starts at {q}, inside the 0.05 margin: a goal that leaves it there is refused (the bot clamps to 0.07)")
    save('joints_start.json', {'joints': n.joints, 'effort': n.effort})
    save('urdf_limits.json', {'guide_vs_urdf': rows, 'convention_1_5': conv,
                              'joints_1_5': {k: lim[k] for k in SIM15 if k in lim}, 'torso': lim.get('torso_lift_joint'), 'all': lim})
    if 'ci' in box:
        ci = box['ci']; K = (ci.k[0], ci.k[4], ci.k[2], ci.k[5])
        dk = max(abs(a - b) for a, b in zip(K, GUIDE_K))
        v.check('camera intrinsics', ci.width == 1280 and ci.height == 720 and dk < 5.0,
                f'{ci.width}x{ci.height} fx {K[0]:.1f} fy {K[1]:.1f} cx {K[2]:.1f} cy {K[3]:.1f} (Guide 909.8 908.6 646.8 378.0), '
                f'model {ci.distortion_model} D {np.round(ci.d, 4).tolist()}',
                'intrinsics differ from the Guide: the bot reads camera_info live, so it is fine; send camera_info.json to the lead',
                K=K, D=list(ci.d))
        save('camera_info.json', {'width': ci.width, 'height': ci.height, 'k': list(ci.k), 'd': list(ci.d), 'p': list(ci.p),
                                  'model': ci.distortion_model, 'frame': ci.header.frame_id})
    else:
        v.check('camera intrinsics', False, 'no camera_info', 'camera not up: check ros2 topic list | grep head_front_camera')
    cams = {}
    for f in ('head_front_camera_color_optical_frame', 'head_front_camera_depth_optical_frame'):
        try:
            t = tf.lookup_transform('base_footprint', f, RTime()).transform
            q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
            cams[f] = {'xyz': [t.translation.x, t.translation.y, t.translation.z], 'quat': q,
                       'rpy_deg': np.degrees(Rotation.from_quat(q).as_euler('xyz')).round(2).tolist()}
        except Exception as e:
            cams[f] = {'error': str(e)}
    c = cams['head_front_camera_color_optical_frame']
    v.check('camera tf', 'xyz' in c, f"colour optical in base_footprint {np.round(c.get('xyz', []), 3).tolist()} at head "
            f"{n.joints.get('head_1_joint', float('nan')):.3f}/{n.joints.get('head_2_joint', float('nan')):.3f}",
            'no TF to the camera: perception cannot lift detections to 3D, send ros2 run tf2_tools view_frames output')
    if 'xyz' in cams.get('head_front_camera_depth_optical_frame', {}) and 'xyz' in c:
        bl = np.linalg.norm(np.subtract(cams['head_front_camera_depth_optical_frame']['xyz'], c['xyz']))
        v.info(f'colour-depth optical baseline {bl * 100:.1f} cm (aligned depth is in the colour frame)')
    if not a.no_probe:
        probes = margin_probe(n, v, lim)
        save('margins.json', probes)
    save('camera_tf.json', {'frames': cams, 'head': [n.joints.get('head_1_joint'), n.joints.get('head_2_joint')],
                            'torso': n.joints.get('torso_lift_joint')})
    return finish(n, v)


if __name__ == '__main__':
    raise SystemExit(main())

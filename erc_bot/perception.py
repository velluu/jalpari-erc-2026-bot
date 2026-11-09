"""Detections in 3D: the ONNX detector on the live colour image, depth from the aligned depth image, points in odom."""
import numpy as np
import cv2
from .detector import Detector, CLASSES


class Eyes:
    WORDS = ('book_FUTURE', 'book_ROBOT', 'book_LIBRARY', 'book_AGENT')

    def __init__(self, live, model, conf=0.35, word_model=None):
        """word_model: a box detector used whenever only white-book words are asked for (det-v4 reads the words better
        at the 1.15 m stand-off than the pose models, lane D 2026-11-05). None = the main model for everything."""
        self.r = live
        import onnxruntime as ort, os
        out = ort.InferenceSession(model, providers=['CPUExecutionProvider']).get_outputs()[0].shape
        self.pose = len(out) == 3 and out[1] == 4 + len(CLASSES) + 15    # lane D's 5-keypoint export: books and plates by keypoints
        if self.pose:
            from .pose_detector import PoseDetector
            self.det = PoseDetector(model, 1280, threads=int(os.environ.get('ERC_BOT_ORT_THREADS', '0')))
            self.det.providers = self.det.s.get_providers()
        else:
            self.det = Detector(model, 1280)
        self.conf = conf
        self.word_det = Detector(word_model, 1280) if word_model and word_model != model else None
        if self.word_det is not None:
            live.say(f'word detector {word_model}')
        live.say(f'detector {model} on {self.det.providers[0]}, input {self.det.H}x{self.det.W}, '
                 f'{"keypoint lift (edge-keypoint midpoint, 15th-pct depth, 10 cm gate)" if self.pose else "box centre, median depth"}')

    def look(self, classes=None, frame='odom', conf=None, kp=True):
        """One fresh frame: [{class, score, box, p (xyz in `frame`), z (camera depth m)}], plus the camera pose.
        Points come from the median depth of the inner 40 % of each box (the spine face for a book), or with a pose
        model from the keypoint lift. kp=False keeps the box method: with the gripper in view the pose lift's depth patch
        lands on the gripper (lane D: 3.75 cm mean, 22 cm p95), so hand checks pass kp=False."""
        cam = self.r.camera()
        if cam is None:
            return None, None
        bgr, depth, (fx, fy, cx, cy), cam_in_base, t = cam
        if frame == 'base_footprint':
            pose = cam_in_base
        else:
            try:
                pose = self.r.matrix(frame, 'base_footprint') @ cam_in_base
            except Exception:
                pose = self.r.odom_T_base() @ cam_in_base
        out = []
        det = self.word_det if (self.word_det is not None and classes and all(c in self.WORDS for c in classes)) else self.det
        import os
        alias = os.environ.get('ERC_BOT_BIN_ALIAS') if classes and 'bin' in classes else None
        for d in det(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), conf or self.conf):
            if alias and d['class'] == alias and (d['box'][2] - d['box'][0]) > 0.8 * (d['box'][3] - d['box'][1]):
                d = dict(d, **{'class': 'bin', 'alias': alias})   # real bin read as a red book (6 Nov)
            if classes and d['class'] not in classes:
                continue
            x0, y0, x1, y1 = d['box']; w, h = x1 - x0, y1 - y0
            H, Wd = bgr.shape[:2]
            if x0 < 3 or y0 < 3 or x1 > Wd - 4 or y1 > H - 4:
                continue                       # cut by the image edge: the box centre is not the object's centre
            patch = depth[int(y0 + 0.3 * h):int(y1 - 0.3 * h) + 1, int(x0 + 0.3 * w):int(x1 - 0.3 * w) + 1]
            v = patch[(patch > 0.15) & (patch < 8.0)]
            if v.size < 4:
                continue
            z = float(np.median(v)); u, vv = (x0 + x1) / 2, (y0 + y1) / 2
            p = pose @ np.array([(u - cx) * z / fx, (vv - cy) * z / fy, z, 1.0])
            if kp and self.pose and 'kpts' in d and d['class'] != 'bin':
                from .pose_detector import lift
                info = {}
                q = lift(d, depth, (fx, fy, cx, cy), pose, info=info)
                if q is not None:                              # lane D: lateral 0.51 cm, depth 0.33 cm mean on unseen sets
                    p = np.r_[q, 1.0]; z = info.get('z_box') if info.get('gated') else info.get('z_kp') or z
            d = dict(d, p=p[:3], z=z, t=t)
            if d['class'] == 'bin':
                d.update(self.bin_geometry(depth, (fx, fy, cx, cy), pose, d['box'], z) or {})
            out.append(d)
        return out, pose

    @staticmethod
    def bin_geometry(depth, K, pose, box, z_med):
        """Bin centre from its rim, not from the median depth: from across the arena the inner 40 % of the box sees the
        inside of the far wall and the floor, so the median-depth point lies up to the bin's half-length beyond the centre
        along the line of sight (measured 0.24 m from 2.8 m, the book missed the bin). The rim (top 4 cm of the points)
        spans the near outer face to the far inner face, so the middle of its extent is the centre whatever the range.
        Returns p (rim centre, z = rim height), rim_z, floor_z (inside floor if seen) and the rim extent, or None."""
        fx, fy, cx, cy = K
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        ys, xs = np.mgrid[y0:y1 + 1:2, x0:x1 + 1:2]
        zz = depth[ys, xs]
        ok = (zz > 0.15) & (zz < 8.0) & (np.abs(zz - z_med) < 0.8)
        if ok.sum() < 30:
            return None
        zz, xs, ys = zz[ok], xs[ok], ys[ok]
        P = (pose @ np.stack([(xs - cx) * zz / fx, (ys - cy) * zz / fy, zz, np.ones_like(zz)]))[:3].T
        P = P[(P[:, 2] > 0.4) & (P[:, 2] < 1.6)]
        if len(P) < 30:
            return None
        top = np.percentile(P[:, 2], 98)
        rim = P[P[:, 2] > top - 0.04]
        if len(rim) < 10:
            return None
        lo, hi = np.percentile(rim[:, :2], 3, axis=0), np.percentile(rim[:, :2], 97, axis=0)
        c = (lo + hi) / 2
        inside = P[(np.abs(P[:, 0] - c[0]) < (hi[0] - lo[0]) / 2 - 0.03) & (np.abs(P[:, 1] - c[1]) < (hi[1] - lo[1]) / 2 - 0.03)]
        floor = float(np.percentile(inside[:, 2], 5)) if len(inside) > 10 else None
        return {'p': np.array([c[0], c[1], top]), 'p_median': None, 'rim_z': float(top), 'floor_z': floor,
                'extent': (hi - lo).round(3).tolist(), 'rim_xy': rim[::max(1, len(rim) // 200), :2]}


__all__ = ['Eyes', 'CLASSES']

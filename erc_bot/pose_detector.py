# Vendored from erc-gym gym/pose_detector.py (lane D). Keep in sync.
"""ONNX decoder and 3D lifting for the ERC keypoint detector (gym/train_pose.py, labels from tools/pose_labels.py).

Same call as perception_node.Detector, plus 5 keypoints per detection. The pose export's output is
(1, 4 + 14 + 15, anchors): box cx cy w h, 14 class scores, then (u, v, conf) for 5 keypoints, all in letterboxed
input pixels. The detector-only Detector would read the keypoint rows as class scores, so a pose model needs this class.

Keypoints (image pixels of the original frame): 0 face centre, 1 face top-centre, 2 face bottom-centre,
3 +y edge mid (image left when facing the shelf), 4 -y edge mid. Books: the spine face. Plates: the digit plate.

lift(det, depth, K, cam_T) gives the grasp point in the frame of cam_T (4x4, optical camera to frame), measured
offline against gz truth (tools/eval_pose3d.py, method 'kpelo'): pixel = mean of the edge keypoints (lateral) and of
top and bottom (vertical), depth = 15th percentile of a 7 x 7 px patch there (the spine face is the book's nearest
surface, the side face and the shelf back lie behind it), gated against the box's inner-40 % median depth (10 cm):
on a gate the box median depth is used with the keypoint pixel (eval method 'kpelog').

  from pose_detector import PoseDetector, lift
  det = PoseDetector('pose_s_736x1280.onnx'); dets = det(rgb)            # rgb uint8 HxWx3
  p = lift(dets[0], depth_m, (fx, fy, cx, cy), cam_T)                    # depth in metres, aligned to colour
"""
import numpy as np
import cv2

CLASSES = ['book_red', 'book_green', 'book_blue', 'book_yellow', 'book_FUTURE', 'book_ROBOT', 'book_LIBRARY',
           'book_AGENT', 'marker_1', 'marker_2', 'marker_3', 'marker_4', 'marker_5', 'bin']
NK = 5


class PoseDetector:
    def __init__(self, path, imgsz=1280, providers=None, threads=0):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        prov = providers or [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail]
        o = ort.SessionOptions()
        if threads:
            o.intra_op_num_threads = threads
        self.s = ort.InferenceSession(path, o, providers=prov)
        i0 = self.s.get_inputs()[0]; self.inp = i0.name; shp = i0.shape
        self.H = shp[2] if isinstance(shp[2], int) else imgsz
        self.W = shp[3] if isinstance(shp[3], int) else imgsz
        self.nc = len(CLASSES)

    def __call__(self, rgb, conf=0.35, iou=0.5):
        h, w = rgb.shape[:2]; SH, SW = self.H, self.W
        r = min(SW / w, SH / h); nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (SW - nw) // 2, (SH - nh) // 2
        canvas = np.full((SH, SW, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.s.run(None, {self.inp: x})[0][0].T            # (anchors, 4 + nc + 3 NK)
        assert out.shape[1] == 4 + self.nc + 3 * NK, f'not a {NK}-keypoint pose export: {out.shape}'
        scores = out[:, 4:4 + self.nc]; cls = scores.argmax(1); sc = scores[np.arange(len(cls)), cls]
        keep = sc >= conf
        if not keep.any():
            return []
        b = out[keep, :4]; cls = cls[keep]; sc = sc[keep]; kp = out[keep, 4 + self.nc:].reshape(-1, NK, 3).copy()
        x0 = (b[:, 0] - b[:, 2] / 2 - px) / r; y0 = (b[:, 1] - b[:, 3] / 2 - py) / r
        x1 = (b[:, 0] + b[:, 2] / 2 - px) / r; y1 = (b[:, 1] + b[:, 3] / 2 - py) / r
        boxes = np.stack([x0, y0, x1, y1], 1).clip([0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
        kp[:, :, 0] = (kp[:, :, 0] - px) / r; kp[:, :, 1] = (kp[:, :, 1] - py) / r
        res = []
        for c in np.unique(cls):
            m = np.where(cls == c)[0]
            bb = [[float(v[0]), float(v[1]), float(v[2] - v[0]), float(v[3] - v[1])] for v in boxes[m]]
            for k in np.array(cv2.dnn.NMSBoxes(bb, [float(s) for s in sc[m]], conf, iou)).reshape(-1):
                i = m[int(k)]
                res.append({'class': CLASSES[int(c)], 'score': round(float(sc[i]), 3),
                            'box': [round(float(v), 1) for v in boxes[i]],
                            'kpts': [[round(float(u), 2), round(float(v), 2), round(float(q), 3)] for u, v, q in kp[i]]})
        return sorted(res, key=lambda d: -d['score'])


def grasp_pixel(det):
    """Lateral from the mean of the edge keypoints, vertical from the mean of top and bottom."""
    k = det['kpts']
    return (k[3][0] + k[4][0]) / 2, (k[1][1] + k[2][1]) / 2


def box_median_depth(det, depth, zmin=0.15, zmax=8.0):
    """erc_bot/perception.py's depth: median of the inner 40 % of the box (metres), None with fewer than 4 valid pixels."""
    scale = 0.001 if depth.dtype == np.uint16 else 1.0
    x0, y0, x1, y1 = det['box']; w, h = x1 - x0, y1 - y0
    patch = depth[int(y0 + 0.3 * h):int(y1 - 0.3 * h) + 1, int(x0 + 0.3 * w):int(x1 - 0.3 * w) + 1].astype(np.float64) * scale
    v = patch[(patch > zmin) & (patch < zmax)]
    return float(np.median(v)) if v.size >= 4 else None


def lift(det, depth, K, cam_T=None, r=3, q=15, zmin=0.15, zmax=8.0, gate=0.10, info=None):
    """Grasp point of a book or plate detection as np.array xyz in the frame of cam_T (camera optical frame if None).
    Pixel: mean of the edge keypoints (lateral) and of top and bottom (vertical). Depth: 15th percentile of a 7 x 7 px
    patch there. Gate: when that depth differs from the box's inner-40 % median by more than `gate` m (the patch fell
    through a thin spine onto the shelf back, or onto an occluder), the box median is used instead (info['gated'] True).
    None for the bin (no keypoints trained) or with no valid depth."""
    if det['class'] == 'bin':
        return None
    fx, fy, cx, cy = K
    u, v = grasp_pixel(det)
    H, W = depth.shape[:2]; ui, vi = int(round(u)), int(round(v))
    scale = 0.001 if depth.dtype == np.uint16 else 1.0          # uint16 aligned depth is in millimetres
    patch = depth[max(0, vi - r):min(H, vi + r + 1), max(0, ui - r):min(W, ui + r + 1)].astype(np.float64) * scale
    z = patch[(patch > zmin) & (patch < zmax)]
    zk = float(np.percentile(z, q)) if z.size >= 4 else None
    zb = box_median_depth(det, depth, zmin, zmax) if gate else None
    gated = zk is None or (zb is not None and abs(zk - zb) > gate)
    zz = zb if gated else zk
    if info is not None:
        info.update(z_kp=zk, z_box=zb, gated=gated)
    if zz is None:
        return None
    p = np.array([(u - cx) * zz / fx, (v - cy) * zz / fy, zz, 1.0])
    return (cam_T @ p)[:3] if cam_T is not None else p[:3]

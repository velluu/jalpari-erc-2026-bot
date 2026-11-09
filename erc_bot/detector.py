"""ERC detector (erc-gym YOLO11s, ONNX): books by colour, white books by spine word, column digits, the bin.

Vendored from erc-gym gym/perception_node.py so the bot runs on the team laptop on its own (numpy, OpenCV,
onnxruntime 1.23.2). Takes RGB, returns class, score and pixel box [x0, y0, x1, y1].
"""
import numpy as np
import cv2

CLASSES = ['book_red', 'book_green', 'book_blue', 'book_yellow', 'book_FUTURE', 'book_ROBOT', 'book_LIBRARY',
           'book_AGENT', 'marker_1', 'marker_2', 'marker_3', 'marker_4', 'marker_5', 'bin']


class Detector:
    def __init__(self, path, imgsz, providers=None):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        prov = providers or [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail]
        so = ort.SessionOptions()
        import os
        so.intra_op_num_threads = int(os.environ.get('ERC_BOT_ORT_THREADS', '0'))   # 0 = all cores; fewer when sharing a box with a sim
        self.s = ort.InferenceSession(path, sess_options=so, providers=prov)
        i0 = self.s.get_inputs()[0]; self.inp = i0.name
        shp = i0.shape          # [1, 3, H, W]; a rectangular export (e.g. 736x1280) wastes no compute on padding
        self.H = shp[2] if isinstance(shp[2], int) else imgsz
        self.W = shp[3] if isinstance(shp[3], int) else imgsz
        self.providers = self.s.get_providers()

    def __call__(self, rgb, conf=0.35, iou=0.5):
        h, w = rgb.shape[:2]; SH, SW = self.H, self.W
        r = min(SW / w, SH / h); nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (SW - nw) // 2, (SH - nh) // 2
        canvas = np.full((SH, SW, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.s.run(None, {self.inp: x})[0][0]            # (4 + nc, anchors)
        out = out.T
        scores = out[:, 4:]; cls = scores.argmax(1); sc = scores[np.arange(len(cls)), cls]
        keep = sc >= conf
        if not keep.any():
            return []
        b = out[keep, :4]; cls = cls[keep]; sc = sc[keep]
        x0 = (b[:, 0] - b[:, 2] / 2 - px) / r; y0 = (b[:, 1] - b[:, 3] / 2 - py) / r
        x1 = (b[:, 0] + b[:, 2] / 2 - px) / r; y1 = (b[:, 1] + b[:, 3] / 2 - py) / r
        boxes = np.stack([x0, y0, x1, y1], 1).clip([0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
        res = []
        for c in np.unique(cls):                                # per-class NMS
            m = np.where(cls == c)[0]
            bb = [[float(v[0]), float(v[1]), float(v[2] - v[0]), float(v[3] - v[1])] for v in boxes[m]]
            for k in np.array(cv2.dnn.NMSBoxes(bb, [float(s) for s in sc[m]], conf, iou)).reshape(-1):
                i = m[int(k)]
                res.append({'class': CLASSES[int(c)] if int(c) < len(CLASSES) else str(int(c)), 'score': round(float(sc[i]), 3),
                            'box': [round(float(v), 1) for v in boxes[i]]})
        return word_nms(sorted(res, key=lambda d: -d['score']))


WORDS = ('book_FUTURE', 'book_ROBOT', 'book_LIBRARY', 'book_AGENT')
WORD_IOU = 0.5


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def word_nms(res):
    """One word per white book: per-class NMS leaves a FUTURE box and a ROBOT box on the same real book (the real spine
    reads 'Future', mixed case; lane P on the 30 Oct frames: ROBOT >= 0.35 on the FUTURE book in 33 of 54, above FUTURE
    in 8). Where word boxes of different classes overlap (IoU > WORD_IOU), keep the top-scoring one. Coloured books,
    plates and the bin are untouched. res sorted by score, highest first."""
    kept, out = [], []
    for d in res:
        if d['class'] in WORDS:
            if any(k['class'] != d['class'] and _iou(k['box'], d['box']) > WORD_IOU for k in kept):
                continue
            kept.append(d)
        out.append(d)
    return out

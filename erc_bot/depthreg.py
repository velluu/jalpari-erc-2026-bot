"""Unaligned depth registered to the colour image: a fallback for when the robot's aligned_depth_to_color stream is slow
or absent (the team's 30 Oct session-1 bag: aligned depth 58 messages in 2,765 s, unaligned depth/image_rect_raw 13.7 Hz).

register() takes the unaligned depth (metres, 0 = none) with the depth camera's intrinsics, moves every valid pixel into
the colour optical frame with the depth-to-colour extrinsic (from /tf_static), projects it with the colour intrinsics
and keeps the nearest surface per colour pixel. The depth image is coarser than colour (848x480 against 1280x720 on the
TIAGo Pro head), so each point covers a ceil(scale) x ceil(scale) block, and the holes left after that are filled from
the nearest valid neighbour for at most `fill` pixels. The output has the aligned stream's format: float32 metres at the
colour resolution, 0 = no depth.
"""
import math
import numpy as np
import cv2


def register(dep_m, Kd, T_cd, Kc, shape_c, fill=2, zmin=0.05):
    """dep_m: HdxWd float metres (0 = none). Kd, Kc: (fx, fy, cx, cy) of depth and colour. T_cd: 4x4, a point in the depth
    optical frame -> the colour optical frame. shape_c: (H, W) of colour. Returns HxW float32 metres, 0 = none."""
    Hc, Wc = shape_c
    fxd, fyd, cxd, cyd = Kd
    fxc, fyc, cxc, cyc = Kc
    v, u = np.nonzero(dep_m > zmin)
    if u.size == 0:
        return np.zeros((Hc, Wc), np.float32)
    z = dep_m[v, u].astype(np.float64)
    P = np.asarray(T_cd, float)[:3, :3] @ np.stack([(u - cxd) * z / fxd, (v - cyd) * z / fyd, z]) + np.asarray(T_cd, float)[:3, 3:4]
    zc = P[2]
    ok = zc > zmin
    P, zc = P[:, ok], zc[ok]
    # a depth pixel spans `s` colour pixels: place each point on the colour pixel grid with its block's top-left at the
    # projection of the depth pixel's top-left corner, so the blocks tile without gaps on a flat surface
    s = max(1, int(math.ceil(max(fxc / fxd, fyc / fyd) - 1e-6)))
    uc = np.floor(fxc * P[0] / zc + cxc - (s - 1) / 2.0).astype(np.int64)
    vc = np.floor(fyc * P[1] / zc + cyc - (s - 1) / 2.0).astype(np.int64)
    if s > 1:
        du, dv = np.meshgrid(np.arange(s), np.arange(s))
        uc = (uc[:, None] + du.ravel()[None]).ravel(); vc = (vc[:, None] + dv.ravel()[None]).ravel()
        zc = np.repeat(zc, s * s)
    inb = (uc >= 0) & (uc < Wc) & (vc >= 0) & (vc < Hc)
    uc, vc, zc = uc[inb], vc[inb], zc[inb]
    out = np.zeros((Hc, Wc), np.float32)
    if zc.size:
        idx = vc * Wc + uc
        order = np.lexsort((zc, idx))                      # by pixel, nearest first
        idx, zc = idx[order], zc[order]
        first = np.r_[True, idx[1:] != idx[:-1]]
        out.ravel()[idx[first]] = zc[first]
    for _ in range(fill):                                  # holes: nearest valid neighbour (3x3 min over valid pixels)
        hole = out == 0
        if not hole.any():
            break
        big = np.where(hole, np.float32(1e6), out)
        nb = cv2.erode(big, np.ones((3, 3), np.uint8))
        out = np.where(hole & (nb < 1e5), nb, out).astype(np.float32)
    return out


def decode_compressed_depth(data):
    """compressed_depth_image_transport 16UC1: a 12-byte config header, then a 16-bit PNG of millimetres. Metres or None."""
    buf = bytes(data); k = buf.find(b'\x89PNG', 0, 64)
    dep = cv2.imdecode(np.frombuffer(buf[k:], np.uint8), cv2.IMREAD_UNCHANGED) if k >= 0 else None
    return None if dep is None else dep.astype(np.float32) * 0.001

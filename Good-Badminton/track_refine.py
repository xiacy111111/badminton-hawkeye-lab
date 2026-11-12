"""
track_refine.py — 时序轨迹补全（替代/补位 TrackNet 的工程方案）
================================================================
问题：单帧检测（YOLO/帧差）在快球、遮挡、运动模糊处会漏检，
      轨迹断成碎片，导致"球飞在空中就丢了"→ 无法落点判罚。

TrackNet 用"连续 3 帧堆叠 + 热力图"从模型层面解决；但它的官方权重
（136MB，Git LFS / Google Drive）在内网环境拿不到。本模块用工程手段补这一环：

  1. 匀速/匀加速运动模型 + 自适应门限：容忍 1~6 帧漏检，不误连别的物体
  2. 断点补全：对缺口用二次拟合插值（羽毛球飞行近似抛物线）
  3. 速度连续性校验：两段轨迹若速度突变过大则不合并（防止把球员轨迹接进来）
  4. 覆盖率指标：直接量化"轨迹到底连不连续"，用于和 TrackNet 对比

用法：
    from track_refine import refine_tracks, coverage
    tracks = refine_tracks(per_frame_dets, frame_size=(w, h))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Trail:
    """一条候选轨迹。points: [(frame, x, y, filled_flag)]"""
    points: List[Tuple[int, float, float, bool]] = field(default_factory=list)
    misses: int = 0

    @property
    def last_frame(self) -> int:
        return self.points[-1][0] if self.points else -1

    def velocity(self, k: int = 3) -> Tuple[float, float]:
        """末尾 k 点的平均速度（px/帧）。"""
        if len(self.points) < 2:
            return 0.0, 0.0
        seg = self.points[-min(k, len(self.points)):]
        (f0, x0, y0, _), (f1, x1, y1, _) = seg[0], seg[-1]
        dt = max(f1 - f0, 1)
        return (x1 - x0) / dt, (y1 - y0) / dt


class _Tracker:
    def __init__(self, gate_ratio: float = 0.09, max_miss: int = 6,
                 accel_gate: float = 260.0):
        self.gate_ratio = gate_ratio     # 门限 = 画面对角线 * ratio，随漏检帧数放大
        self.max_miss = max_miss
        self.accel_gate = accel_gate     # 速度突变上限(px/帧²)，用于拒绝错误关联
        self.trails: List[Trail] = []

    def step(self, dets: List[Tuple[float, float, float]], frame_idx: int,
             diagonal: float):
        """dets: [(cx, cy, score)]，返回活跃轨迹列表。"""
        used = [False] * len(dets)

        for tr in self.trails:
            f_last, x_last, y_last, _ = tr.points[-1]
            gap = frame_idx - f_last
            if gap > self.max_miss:
                continue
            vx, vy = tr.velocity()
            px, py = x_last + vx * gap, y_last + vy * gap
            gate = diagonal * self.gate_ratio * (1.0 + 0.5 * gap)

            best, best_d = -1, gate
            for i, (dx, dy, sc) in enumerate(dets):
                if used[i]:
                    continue
                d = math.hypot(dx - px, dy - py)
                if d < best_d:
                    best, best_d = i, d

            if best >= 0:
                dx, dy, _ = dets[best]
                # 速度连续性校验：新速度与旧速度差太大则视为错误关联
                nvx, nvy = (dx - x_last) / gap, (dy - y_last) / gap
                if gap > 1 and math.hypot(nvx - vx, nvy - vy) > self.accel_gate:
                    tr.misses += 1
                    continue
                tr.points.append((frame_idx, dx, dy, False))
                tr.misses = 0
                used[best] = True
            else:
                tr.misses += 1
                # 漏检但未超限：先记预测点（filled=True），后面若接上会自然被修正
                if gap <= 2:
                    tr.points.append((frame_idx, px, py, True))

        # 新轨迹
        for i, (dx, dy, _sc) in enumerate(dets):
            if used[i]:
                continue
            self.trails.append(Trail(points=[(frame_idx, dx, dy, False)]))

        # 清理：太久没更新的轨迹
        self.trails = [t for t in self.trails
                       if frame_idx - t.last_frame <= self.max_miss]


def _fit_quadratic(t: np.ndarray, v: np.ndarray) -> Optional[np.ndarray]:
    if len(t) < 3:
        return None
    try:
        return np.polyfit(t.astype(float), v.astype(float), 2)
    except Exception:
        return None


def smooth_and_fill(points: List[Tuple[int, float, float, bool]],
                    max_fill_gap: int = 6) -> List[Tuple[int, float, float, bool]]:
    """对单条轨迹做：去除孤立跳点 + 缺口插值（二次拟合）。"""
    if len(points) < 3:
        return points
    pts = sorted(points, key=lambda p: p[0])
    frames = np.array([p[0] for p in pts], float)
    xs = np.array([p[1] for p in pts], float)
    ys = np.array([p[2] for p in pts], float)

    # 1) 剔除明显离群的跳点（相对二次拟合残差 > 30px）
    cx, cy = _fit_quadratic(frames, xs), _fit_quadratic(frames, ys)
    if cx is not None and cy is not None and len(pts) >= 6:
        rx = xs - np.polyval(cx, frames)
        ry = ys - np.polyval(cy, frames)
        res = np.hypot(rx, ry)
        bad = res > max(30.0, 3.0 * float(np.median(res)))
        if bad.sum() < len(pts) * 0.4:
            pts = [p for p, b in zip(pts, bad) if not b]
            frames = np.array([p[0] for p in pts], float)
            xs = np.array([p[1] for p in pts], float)
            ys = np.array([p[2] for p in pts], float)

    # 2) 缺口插值
    cx, cy = _fit_quadratic(frames, xs), _fit_quadratic(frames, ys)
    if cx is None:
        return pts
    out: List[Tuple[int, float, float, bool]] = []
    for i in range(len(pts)):
        out.append(pts[i])
        if i == len(pts) - 1:
            break
        gap = int(pts[i + 1][0] - pts[i][0])
        if 1 < gap <= max_fill_gap:
            for k in range(1, gap):
                f = pts[i][0] + k
                out.append((f, float(np.polyval(cx, f)), float(np.polyval(cy, f)), True))
    return out


def refine_tracks(per_frame_dets: List[List[Tuple[float, float, float]]],
                  frame_size: Tuple[int, int],
                  max_miss: int = 6,
                  min_len: int = 6) -> List[List[Tuple[int, float, float, bool]]]:
    """把逐帧检测结果整理成连续轨迹（含补全点）。"""
    w, h = frame_size[:2]
    diagonal = math.hypot(w, h)
    tracker = _Tracker(max_miss=max_miss)
    for fi, dets in enumerate(per_frame_dets):
        tracker.step(dets, fi, diagonal)

    trails = [t.points for t in tracker.trails if len(t.points) >= min_len]
    return [smooth_and_fill(t, max_fill_gap=max_miss) for t in trails]


def coverage(points: List[Tuple[int, float, float, bool]]) -> float:
    """轨迹覆盖率 = 帧数 / (首帧到尾帧跨度)。衡量"连不连续"。"""
    if len(points) < 2:
        return 0.0
    span = points[-1][0] - points[0][0] + 1
    return len(points) / max(span, 1)


def fill_ratio(points: List[Tuple[int, float, float, bool]]) -> float:
    if not points:
        return 0.0
    return sum(1 for p in points if p[3]) / len(points)

def template_recover(per_frame_dets, gray_frames, max_gap: int = 4,
                     patch: int = 7, search: int = 14,
                     thresh: float = 0.55):
    """在漏检帧用模板匹配把球"捡回来"（提升检测召回，纯 OpenCV、无权重）。

    对每帧的空缺，取前后最近一次检测处的小块作为模板，在预测位置附近搜索，
    归一化相关系数超过阈值才接受。返回补充后的 per_frame_dets（原地追加）。
    """
    import cv2

    n = len(per_frame_dets)
    added = 0
    for fi in range(n):
        if per_frame_dets[fi]:
            continue
        # 找前后最近的检测点
        prev = next_f = None
        for k in range(1, max_gap + 1):
            if fi - k >= 0 and per_frame_dets[fi - k]:
                prev = (fi - k, per_frame_dets[fi - k][0]); break
        for k in range(1, max_gap + 1):
            if fi + k < n and per_frame_dets[fi + k]:
                next_f = (fi + k, per_frame_dets[fi + k][0]); break
        base = prev or next_f
        if base is None:
            continue
        bf, (bx, by, _bs) = base
        src = gray_frames[bf]
        x0, y0 = int(bx) - patch // 2, int(by) - patch // 2
        if x0 < 0 or y0 < 0 or y0 + patch >= src.shape[0] or x0 + patch >= src.shape[1]:
            continue
        tpl = src[y0:y0 + patch, x0:x0 + patch]
        if tpl.std() < 3:          # 模板太"平"，匹配无意义
            continue

        # 预测位置：用前后两点线性外推，否则沿用基点
        if prev and next_f:
            (pf, (px, py, _)), (nf, (nx, ny, _)) = prev, next_f
            r = (fi - pf) / max(nf - pf, 1)
            cx, cy = px + (nx - px) * r, py + (ny - py) * r
        else:
            cx, cy = bx, by

        img = gray_frames[fi]
        sx0, sy0 = max(int(cx) - search, 0), max(int(cy) - search, 0)
        sx1 = min(int(cx) + search + patch, img.shape[1])
        sy1 = min(int(cy) + search + patch, img.shape[0])
        roi = img[sy0:sy1, sx0:sx1]
        if roi.shape[0] < patch or roi.shape[1] < patch:
            continue
        res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
        _min, maxv, _minl, maxl = cv2.minMaxLoc(res)
        if maxv >= thresh:
            per_frame_dets[fi].append(
                (sx0 + maxl[0] + patch / 2.0, sy0 + maxl[1] + patch / 2.0, float(maxv) * 0.5))
            added += 1
    return per_frame_dets, added

"""
calibration.py — 多视角自动球场标定
====================================
基于 Good-Badminton 自动球场线检测算法，支持：
  - 单目标定：solvePnP（16 个精化后的场地关键点）
  - 双眼标定：两侧相机独立标定，自动计算相对位姿
  - 输出可直接用于 cv2.triangulatePoints 的投影矩阵

用法：
  # 单目标定
  python calibration.py --image left.jpg

  # 双眼标定（球场两侧对拍）
  python calibration.py --left left.jpg --right right.jpg --out calib.json
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 羽毛球场标准尺寸（米）
# ═══════════════════════════════════════════════════════════════
COURT_WIDTH = 6.1
COURT_LENGTH = 13.4
SINGLES_MARGIN = 0.46
BACK_SERVICE_OFFSET = 0.76
SERVICE_LINE_FROM_NET = 1.98
NET_Y = COURT_LENGTH / 2.0

# 球场四角（3D）
COURT_CORNERS_3D = np.array(
    [[0.0, 0.0, 0.0], [COURT_WIDTH, 0.0, 0.0],
     [COURT_WIDTH, COURT_LENGTH, 0.0], [0.0, COURT_LENGTH, 0.0]],
    dtype=np.float32,
)

# 16 个场地关键点（3D）—— 所有主要线交点
COURT_KEYPOINTS_3D = np.array([
    # 外边界四角
    [0.0, 0.0, 0.0],                              #  0  左上
    [COURT_WIDTH, 0.0, 0.0],                      #  1  右上
    [COURT_WIDTH, COURT_LENGTH, 0.0],              #  2  右下
    [0.0, COURT_LENGTH, 0.0],                      #  3  左下
    # 单打边线 × 底线
    [SINGLES_MARGIN, 0.0, 0.0],                   #  4
    [COURT_WIDTH - SINGLES_MARGIN, 0.0, 0.0],     #  5
    [COURT_WIDTH - SINGLES_MARGIN, COURT_LENGTH, 0.0],  #  6
    [SINGLES_MARGIN, COURT_LENGTH, 0.0],           #  7
    # 前发球线 × 单打边线（远半区）
    [SINGLES_MARGIN, NET_Y + SERVICE_LINE_FROM_NET, 0.0],      #  8
    [COURT_WIDTH - SINGLES_MARGIN, NET_Y + SERVICE_LINE_FROM_NET, 0.0],  #  9
    # 前发球线 × 单打边线（近半区）
    [SINGLES_MARGIN, NET_Y - SERVICE_LINE_FROM_NET, 0.0],      # 10
    [COURT_WIDTH - SINGLES_MARGIN, NET_Y - SERVICE_LINE_FROM_NET, 0.0],  # 11
    # 双打后发球线 × 双打边线
    [0.0, BACK_SERVICE_OFFSET, 0.0],              # 12
    [COURT_WIDTH, BACK_SERVICE_OFFSET, 0.0],      # 13
    [0.0, COURT_LENGTH - BACK_SERVICE_OFFSET, 0.0],  # 14
    [COURT_WIDTH, COURT_LENGTH - BACK_SERVICE_OFFSET, 0.0],  # 15
], dtype=np.float32)

# 海康 MV-CU013-A0UC + 6mm 镜头的初始内参估计
# 像元 3.45µm, 焦距 6mm → f_px = 6000/3.45 ≈ 1739
SENSOR_W, SENSOR_H = 1280, 1024
PIXEL_SIZE_UM = 3.45
FOCAL_LENGTH_MM = 6.0
DEFAULT_FX = FOCAL_LENGTH_MM * 1000.0 / PIXEL_SIZE_UM  # ≈ 1739.1
DEFAULT_CX = SENSOR_W / 2.0   # 640
DEFAULT_CY = SENSOR_H / 2.0   # 512


# ═══════════════════════════════════════════════════════════════
# 标定结果数据结构
# ═══════════════════════════════════════════════════════════════
@dataclass
class CameraPose:
    """单个相机的标定结果"""
    cam_id: int = -1
    image_size: Tuple[int, int] = (0, 0)
    K: Optional[np.ndarray] = None       # 3×3 内参矩阵
    dist: Optional[np.ndarray] = None    # 5×1 畸变系数
    rvec: Optional[np.ndarray] = None    # 3×1 旋转向量
    tvec: Optional[np.ndarray] = None    # 3×1 平移向量
    P: Optional[np.ndarray] = None       # 3×4 投影矩阵 = K @ [R|t]
    corners_2d: Optional[np.ndarray] = None  # 检测到的 4 角点
    keypoints_2d: Optional[np.ndarray] = None  # 精化后的 16 关键点
    reproj_error: float = 0.0
    num_inliers: int = 0
    success: bool = False

    def to_dict(self) -> dict:
        d = {"cam_id": self.cam_id, "image_size": list(self.image_size),
             "reproj_error": self.reproj_error, "num_inliers": self.num_inliers,
             "success": self.success}
        for k in ["K", "dist", "rvec", "tvec", "P", "corners_2d", "keypoints_2d"]:
            v = getattr(self, k)
            d[k] = v.tolist() if v is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CameraPose":
        obj = cls(cam_id=d.get("cam_id", -1),
                  image_size=tuple(d.get("image_size", (0, 0))),
                  reproj_error=d.get("reproj_error", 0.0),
                  num_inliers=d.get("num_inliers", 0),
                  success=d.get("success", False))
        for k in ["K", "dist", "rvec", "tvec", "P", "corners_2d", "keypoints_2d"]:
            v = d.get(k)
            setattr(obj, k, np.array(v, dtype=np.float64) if v is not None else None)
        return obj


@dataclass
class StereoPose:
    """双目系统标定结果"""
    image_size: Tuple[int, int] = (0, 0)
    cam_left: CameraPose = field(default_factory=CameraPose)
    cam_right: CameraPose = field(default_factory=CameraPose)
    R_rel: Optional[np.ndarray] = None   # 右相机相对左相机旋转 (3×3)
    T_rel: Optional[np.ndarray] = None   # 右相机相对左相机平移 (3×1)
    baseline: float = 0.0                # 基线长度（米）
    success: bool = False

    def to_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "cam_left": self.cam_left.to_dict(),
            "cam_right": self.cam_right.to_dict(),
            "R_rel": self.R_rel.tolist() if self.R_rel is not None else None,
            "T_rel": self.T_rel.tolist() if self.T_rel is not None else None,
            "baseline": self.baseline,
            "success": self.success,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StereoPose":
        obj = cls(
            image_size=tuple(d.get("image_size", (0, 0))),
            cam_left=CameraPose.from_dict(d.get("cam_left", {})),
            cam_right=CameraPose.from_dict(d.get("cam_right", {})),
            baseline=d.get("baseline", 0.0),
            success=d.get("success", False),
        )
        for k in ["R_rel", "T_rel"]:
            v = d.get(k)
            setattr(obj, k, np.array(v, dtype=np.float64) if v is not None else None)
        return obj

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"双目标定已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "StereoPose":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def P1(self) -> Optional[np.ndarray]:
        return self.cam_left.P

    @property
    def P2(self) -> Optional[np.ndarray]:
        return self.cam_right.P


# ═══════════════════════════════════════════════════════════════
# BadmintonCourtReference — 标准球场线模型
# ═══════════════════════════════════════════════════════════════
class BadmintonCourtReference:
    """标准羽毛球场线投影与评分"""

    def __init__(self):
        w, L = COURT_WIDTH, COURT_LENGTH
        sm = SINGLES_MARGIN
        net_y = L / 2.0
        s_top = net_y - SERVICE_LINE_FROM_NET
        s_bot = net_y + SERVICE_LINE_FROM_NET
        bs_top = BACK_SERVICE_OFFSET
        bs_bot = L - BACK_SERVICE_OFFSET
        cx = w / 2.0

        self.court_corners = np.array([[0, 0], [w, 0], [w, L], [0, L]], dtype=np.float32)
        self.lines = [
            ((0, 0), (w, 0), 1.35), ((w, 0), (w, L), 1.35),
            ((w, L), (0, L), 1.35), ((0, L), (0, 0), 1.35),
            ((sm, 0), (sm, L), 0.95), ((w - sm, 0), (w - sm, L), 0.95),
            ((0, bs_top), (w, bs_top), 1.0), ((0, bs_bot), (w, bs_bot), 1.0),
            ((0, s_top), (w, s_top), 1.15), ((0, s_bot), (w, s_bot), 1.15),
            ((cx, 0), (cx, s_top), 0.8), ((cx, s_bot), (cx, L), 0.8),
            ((0, net_y), (w, net_y), 0.35),
        ]

    def prepare_line_support(self, line_mask, image_shape):
        if line_mask is None:
            return None
        h, w = image_shape[:2]
        support_mask = (line_mask > 0).astype(np.uint8) * 255
        if support_mask.shape[:2] != (h, w):
            support_mask = cv2.resize(support_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        distance_map = cv2.distanceTransform(255 - support_mask, cv2.DIST_L2, 3)
        return {"distance_map": distance_map, "tolerance": max(5.0, min(w, h) * 0.013)}

    def score_line_support(self, image_corners, line_support, image_shape):
        if line_support is None:
            return 0.0, self._empty_details()
        h, w = image_shape[:2]
        corners = np.array(image_corners, dtype=np.float32)
        if corners.shape != (4, 2):
            return 0.0, self._empty_details()
        matrix = cv2.getPerspectiveTransform(self.court_corners, corners)
        dm, tol = line_support["distance_map"], line_support["tolerance"]
        w_score, w_total = 0.0, 0.0
        coverages, supported = [], 0
        for start, end, weight in self.lines:
            si, ei = self._project_line(matrix, start, end)
            samples = self._sample_line(si, ei, w, h)
            if samples is None:
                continue
            dists = dm[samples[:, 1], samples[:, 0]]
            ls = float(np.mean(1.0 - np.clip(dists / tol, 0.0, 1.0)))
            cov = float(np.mean(dists <= tol))
            w_score += weight * (0.68 * ls + 0.32 * cov)
            w_total += weight
            coverages.append(cov)
            if cov >= 0.42:
                supported += 1
        if w_total <= 0:
            return 0.0, self._empty_details(tol)
        score = float(np.clip(w_score / w_total, 0.0, 1.0))
        details = {"reference_score": round(score, 4),
                   "reference_coverage": round(float(np.mean(coverages)) if coverages else 0.0, 4),
                   "reference_supported_lines": int(supported),
                   "reference_tolerance_px": round(float(tol), 2)}
        return score, details

    def _project_line(self, matrix, start, end):
        pts = cv2.perspectiveTransform(np.array([[start, end]], dtype=np.float32), matrix)[0]
        return pts[0], pts[1]

    def _sample_line(self, start, end, width, height):
        length = float(np.linalg.norm(end - start))
        if length < 1.0:
            return None
        n = max(18, int(length / 5.0))
        xs = np.linspace(float(start[0]), float(end[0]), n)
        ys = np.linspace(float(start[1]), float(end[1]), n)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.mean(valid) < 0.35:
            return None
        xs = np.clip(np.rint(xs[valid]).astype(np.int32), 0, width - 1)
        ys = np.clip(np.rint(ys[valid]).astype(np.int32), 0, height - 1)
        return np.column_stack([xs, ys])

    def _empty_details(self, tolerance=0.0):
        return {"reference_score": 0.0, "reference_coverage": 0.0,
                "reference_supported_lines": 0, "reference_tolerance_px": round(float(tolerance), 2)}


# ═══════════════════════════════════════════════════════════════
# 球场线检测 — 移植自 Good-Badminton detector.py
# ═══════════════════════════════════════════════════════════════
def _line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def _line_intersection(line_a, line_b):
    x1, y1, x2, y2 = line_a
    x3, y3, x4, y4 = line_b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float32)


def _inside_image(point, width, height, margin=0):
    return margin <= point[0] <= width - 1 - margin and margin <= point[1] <= height - 1 - margin


def _polygon_area(points):
    return abs(cv2.contourArea(np.array(points, dtype=np.float32)))


def _is_convex_quad(points):
    pts = np.array(points, dtype=np.float32)
    return len(cv2.convexHull(pts)) == 4 and cv2.isContourConvex(pts.astype(np.int32))


def _project_court_points(matrix, points):
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def _dedupe_lines(lines, orientation, max_count):
    selected = []
    for line in sorted(lines, key=lambda item: item["length"], reverse=True):
        duplicate = False
        for existing in selected:
            if orientation == "horizontal":
                same_band = abs(line["mid"][1] - existing["mid"][1]) < 18
                same_angle = abs(line["angle"] - existing["angle"]) < 8
            else:
                same_band = abs(line["mid"][0] - existing["mid"][0]) < 22
                same_angle = abs(line["angle"] - existing["angle"]) < 10
            if same_band and same_angle:
                duplicate = True
                break
        if not duplicate:
            selected.append(line)
        if len(selected) >= max_count:
            break
    return selected


def _build_court_line_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green_mask = (((h >= 35) & (h <= 95) & (s >= 30) & (v >= 45)).astype(np.uint8)) * 255
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(green_mask)
    if count > 1:
        img_area = image.shape[0] * image.shape[1]
        candidates = [idx for idx in range(1, count) if stats[idx, cv2.CC_STAT_AREA] > img_area * 0.08]
        if candidates:
            court_idx = max(candidates, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
            green_mask = ((labels == court_idx).astype(np.uint8)) * 255
    court_roi = cv2.dilate(green_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)), iterations=1)
    white_mask = (((s <= 95) & (v >= 135)).astype(np.uint8)) * 255
    white_mask = cv2.bitwise_and(white_mask, court_roi)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.bitwise_and(edges, court_roi)
    merged = cv2.bitwise_or(white_mask, edges)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return merged


def _build_reference_line_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green_mask = (((h >= 35) & (h <= 95) & (s >= 30) & (v >= 45)).astype(np.uint8)) * 255
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(green_mask)
    if count > 1:
        img_area = image.shape[0] * image.shape[1]
        candidates = [idx for idx in range(1, count) if stats[idx, cv2.CC_STAT_AREA] > img_area * 0.08]
        if candidates:
            court_idx = max(candidates, key=lambda idx: stats[idx, cv2.CC_STAT_AREA])
            green_mask = ((labels == court_idx).astype(np.uint8)) * 255
    court_roi = cv2.dilate(green_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=1)
    white_mask = (((s <= 90) & (v >= 145)).astype(np.uint8)) * 255
    yellow_mask = (((h >= 12) & (h <= 42) & (s >= 45) & (v >= 120)).astype(np.uint8)) * 255
    line_mask = cv2.bitwise_and(cv2.bitwise_or(white_mask, yellow_mask), court_roi)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return line_mask


def _detect_court_line_segments(image):
    h, w = image.shape[:2]
    mask = _build_court_line_mask(image)
    min_len = max(46, int(min(w, h) * 0.10))
    max_gap = max(14, int(min(w, h) * 0.045))
    raw_lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 180, threshold=60,
                                minLineLength=min_len, maxLineGap=max_gap)
    if raw_lines is None:
        return [], [], mask
    edge_margin = max(10, int(min(w, h) * 0.02))
    horizontal, side = [], []
    for rl in raw_lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in rl]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        if (min(x1, x2) <= edge_margin or max(x1, x2) >= w - 1 - edge_margin or
                min(y1, y2) <= edge_margin or max(y1, y2) >= h - 1 - edge_margin):
            continue
        angle = _line_angle(x1, y1, x2, y2)
        seg = {"points": (x1, y1, x2, y2), "length": length,
               "mid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0), "angle": angle}
        # 角度约定：0°=水平，90°=垂直，范围 [0,180)。
        # 关键修正：低机位/广角下球场边线可能只偏离水平 18°~25°
        # （如 159° 实为上升 21° 的边线），原来的固定区间会把它们判成横线。
        # 因此只把"几乎水平"(≤12°) 归为横线，其余一律进侧线桶，由四边形评分筛选。
        dev = min(angle, 180 - angle)
        if dev <= 12:
            horizontal.append(seg)
        else:
            side.append(seg)
    return _dedupe_lines(horizontal, "horizontal", 14), _dedupe_lines(side, "side", 18), mask


# ═══════════════════════════════════════════════════════════════
# 四边形评分
# ═══════════════════════════════════════════════════════════════
def _badminton_line_alignment_score(corners, line_mask):
    h, w = line_mask.shape[:2]
    img_pts = np.array(corners, dtype=np.float32)
    court_pts = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(court_pts, img_pts)
    expected = np.zeros((h, w), dtype=np.uint8)
    for y in [0, 0.76, 4.72, 6.7, 8.68, 12.64, 13.4]:
        p1, p2 = _project_court_points(matrix, [(0, y), (6.1, y)])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)
    for x in [0, 0.46, 3.05, 5.64, 6.1]:
        p1, p2 = _project_court_points(matrix, [(x, 0), (x, 13.4)])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)
    for p1c, p2c in [((3.05, 0), (3.05, 4.72)), ((3.05, 8.68), (3.05, 13.4))]:
        p1, p2 = _project_court_points(matrix, [p1c, p2c])
        cv2.line(expected, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)), 255, 5)
    expected_px = cv2.countNonZero(expected)
    if expected_px == 0:
        return 0.0
    dilated = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1)
    return cv2.countNonZero(cv2.bitwise_and(expected, dilated)) / expected_px


def _badminton_horizontal_pattern_score(corners, horizontal_lines, image_shape):
    if not horizontal_lines:
        return 0.0
    h, _w = image_shape[:2]
    img_pts = np.array(corners, dtype=np.float32)
    court_pts = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(court_pts, img_pts)
    expected_y = [0, 0.76, 4.72, 6.7, 8.68, 12.64, 13.4]
    weights = [5.0, 4.0, 1.6, 0.8, 1.6, 4.0, 5.0]
    detected_y = [float(ln["mid"][1]) for ln in horizontal_lines]
    tol = max(12.0, h * 0.022)
    w_s, w_t = 0.0, 0.0
    for cy, wt in zip(expected_y, weights):
        p1, p2 = _project_court_points(matrix, [(0, cy), (6.1, cy)])
        ey = float((p1[1] + p2[1]) / 2.0)
        w_s += max(0.0, 1.0 - min(abs(y - ey) for y in detected_y) / tol) * wt
        w_t += wt
    return w_s / w_t


def _horizontal_segment_support(seg, pL, pR, iw):
    x1, _, x2, _ = seg["points"]
    s_min, s_max = sorted((x1, x2))
    q_min, q_max = sorted((float(pL[0]), float(pR[0])))
    return max(0.0, 1.0 - (max(0, s_min - q_min) + max(0, q_max - s_max)) / max(1, iw * 0.20))


def _side_segment_support(seg, pT, pB, ih):
    _, y1, _, y2 = seg["points"]
    s_min, s_max = sorted((y1, y2))
    q_min, q_max = sorted((float(pT[1]), float(pB[1])))
    return max(0.0, 1.0 - (max(0, s_min - q_min) + max(0, q_max - s_max)) / max(1, ih * 0.16))


def _reference_segment_support(seg, line_support, image_shape):
    if line_support is None:
        return 0.0
    h, w = image_shape[:2]
    x1, y1, x2, y2 = seg["points"]
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length < 1.0:
        return 0.0
    n = max(18, int(length / 4.0))
    xs = np.linspace(float(x1), float(x2), n)
    ys = np.linspace(float(y1), float(y2), n)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if np.mean(valid) < 0.35:
        return 0.0
    xs = np.clip(np.rint(xs[valid]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(ys[valid]).astype(np.int32), 0, h - 1)
    dists = line_support["distance_map"][ys, xs]
    tol = line_support["tolerance"]
    scores = 1.0 - np.clip(dists / tol, 0.0, 1.0)
    return float(0.65 * np.mean(scores) + 0.35 * np.mean(dists <= tol))


def _promote_far_baseline(lines, horizontal_lines, image_shape):
    h, _w = image_shape[:2]
    top_line, bottom_line, left_line, right_line = lines
    top_y = float(top_line["mid"][1])
    candidates = []
    for ln in horizontal_lines:
        cy = float(ln["mid"][1])
        gap = top_y - cy
        if gap <= h * 0.04 or gap >= h * 0.09 or cy < h * 0.28:
            continue
        if ln["length"] < top_line["length"] * 0.65:
            continue
        next_gap = min([float(o["mid"][1]) - top_y
                        for o in horizontal_lines if float(o["mid"][1]) > top_y] or [h])
        if gap > next_gap * 0.9:
            continue
        candidates.append(ln)
    if not candidates:
        return lines
    return max(candidates, key=lambda x: x["length"]), bottom_line, left_line, right_line


def _score_court_quad(corners, image_shape, lines, line_mask, horizontal_lines,
                      court_reference=None, line_support=None):
    h, w = image_shape[:2]
    tl, tr, br, bl = corners
    pts = np.array(corners, dtype=np.float32)
    if not all(_inside_image(p, w, h, margin=8) for p in pts) or not _is_convex_quad(pts):
        return None
    area = _polygon_area(pts)
    if area < w * h * 0.12:
        return None
    top_w = float(np.linalg.norm(tr - tl))
    bot_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    avg_h = (left_h + right_h) / 2.0
    if min(top_w, bot_w, avg_h) <= 1 or avg_h < h * 0.34 or bot_w < top_w * 0.82:
        return None
    center = np.mean(pts, axis=0)
    top_y = (tl[1] + tr[1]) / 2.0
    bot_y = (bl[1] + br[1]) / 2.0
    w_ratio = top_w / bot_w
    # 放宽约束：允许更极端的透视变形（例如侧面拍摄）
    if not (w * 0.30 <= center[0] <= w * 0.88 and h * 0.40 <= center[1] <= h * 0.92):
        return None
    if top_y < h * 0.22 or bot_y < h * 0.60:
        return None
    if not (0.30 <= w_ratio <= 0.92):
        return None
    min_edge = min(np.min(pts[:, 0]), w - 1 - np.max(pts[:, 0]),
                   np.min(pts[:, 1]), h - 1 - np.max(pts[:, 1]))
    if min_edge < max(10, min(w, h) * 0.01):
        return None
    left_dx = bl[0] - tl[0]
    right_dx = br[0] - tr[0]
    if left_dx > w * 0.25 or right_dx < -w * 0.25:
        return None
    top_line, bottom_line, left_line, right_line = lines
    area_score = min(area / (w * h * 0.42), 1.0) * 44
    height_score = min(avg_h / (h * 0.62), 1.0) * 22
    persp_score = (1.0 - min(abs(w_ratio - 0.66) / 0.30, 1.0)) * 16
    cx_score = (1.0 - min(abs(center[0] - w * 0.50) / (w * 0.30), 1.0)) * 18
    cy_score = (1.0 - min(abs(center[1] - h * 0.70) / (h * 0.25), 1.0)) * 8
    bot_score = min(max((bot_y / h - 0.74) / 0.18, 0), 1.0) * 8
    edge_score = min(min_edge / (min(w, h) * 0.08), 1.0) * 6
    line_score = min((top_line["length"] + bottom_line["length"] +
                      left_line["length"] + right_line["length"]) / (w * 2.0 + h), 1.0) * 6
    h_sup = (_horizontal_segment_support(top_line, tl, tr, w) +
             _horizontal_segment_support(bottom_line, bl, br, w)) / 2.0
    s_sup = (_side_segment_support(left_line, tl, bl, h) +
             _side_segment_support(right_line, tr, br, h)) / 2.0
    support_score = (h_sup * 0.55 + s_sup * 0.45) * 24
    top_span = sorted((top_line["points"][0], top_line["points"][2]))
    bot_span = sorted((bottom_line["points"][0], bottom_line["points"][2]))
    top_ep = 1.0 - min((abs(tl[0] - top_span[0]) + abs(tr[0] - top_span[1])) / max(w * 0.34, 1), 1.0)
    bot_ep = 1.0 - min((abs(bl[0] - bot_span[0]) + abs(br[0] - bot_span[1])) / max(w * 0.34, 1), 1.0)
    endpoint_score = ((top_ep + bot_ep) / 2.0) * 44
    l_ref = _reference_segment_support(left_line, line_support, image_shape)
    r_ref = _reference_segment_support(right_line, line_support, image_shape)
    clean_side = min(l_ref, r_ref)
    clean_score = min(1.0, clean_side / 0.75) * 42
    align_val = _badminton_line_alignment_score(corners, line_mask)
    pattern_val = _badminton_horizontal_pattern_score(corners, horizontal_lines, image_shape)
    ref_val, ref_details = 0.0, {}
    if court_reference is not None:
        ref_val, ref_details = court_reference.score_line_support(corners, line_support, image_shape)
    total = (area_score + height_score + persp_score + cx_score + cy_score +
             bot_score + edge_score + line_score + support_score + endpoint_score +
             clean_score + align_val * 10 + pattern_val * 110 + ref_val * 18)
    return total, {"alignment_score": round(float(align_val), 4),
                   "pattern_score": round(float(pattern_val), 4),
                   "endpoint_score": round(float((top_ep + bot_ep) / 2.0), 4),
                   "clean_side_score": round(float(clean_side), 4), **ref_details}


# ═══════════════════════════════════════════════════════════════
# 自动检测入口
# ═══════════════════════════════════════════════════════════════
def auto_detect_court_corners(image):
    horizontal_lines, side_lines, mask = _detect_court_line_segments(image)
    court_ref = BadmintonCourtReference()
    ref_mask = _build_reference_line_mask(image)
    line_support = court_ref.prepare_line_support(ref_mask, image.shape)
    debug = {"horizontal": horizontal_lines, "side": side_lines, "score": None, "details": None}
    if len(horizontal_lines) < 2 or len(side_lines) < 2:
        return None, mask, debug
    h, w = image.shape[:2]
    best = None
    horizontals = sorted(horizontal_lines, key=lambda x: x["mid"][1])
    sides = sorted(side_lines, key=lambda x: x["mid"][0])
    for ti, tl in enumerate(horizontals):
        for bl_line in horizontals[ti + 1:]:
            if bl_line["mid"][1] - tl["mid"][1] < h * 0.28:
                continue
            for li, ll in enumerate(sides):
                for rl in sides[li + 1:]:
                    if rl["mid"][0] - ll["mid"][0] < w * 0.22:
                        continue
                    intersections = [
                        _line_intersection(tl["points"], ll["points"]),
                        _line_intersection(tl["points"], rl["points"]),
                        _line_intersection(bl_line["points"], rl["points"]),
                        _line_intersection(bl_line["points"], ll["points"]),
                    ]
                    if any(p is None for p in intersections):
                        continue
                    scored = _score_court_quad(intersections, image.shape,
                                               (tl, bl_line, ll, rl), mask,
                                               horizontal_lines, court_ref, line_support)
                    if scored is not None and (best is None or scored[0] > best["score"]):
                        best = {"corners": intersections, "score": scored[0],
                                "details": scored[1], "lines": (tl, bl_line, ll, rl)}
    if best is None:
        return None, mask, debug
    selected = _promote_far_baseline(best["lines"], horizontal_lines, image.shape)
    tl, bl_line, ll, rl = selected
    corners_float = [
        _line_intersection(tl["points"], ll["points"]),
        _line_intersection(tl["points"], rl["points"]),
        _line_intersection(bl_line["points"], rl["points"]),
        _line_intersection(bl_line["points"], ll["points"]),
    ]
    if any(p is None for p in corners_float):
        corners_float = best["corners"]
    final_scored = _score_court_quad(corners_float, image.shape, selected, mask,
                                     horizontal_lines, court_ref, line_support)
    debug["score"] = final_scored[0] if final_scored else best["score"]
    debug["details"] = final_scored[1] if final_scored else best["details"]
    corners = [(int(round(p[0])), int(round(p[1]))) for p in corners_float]
    return corners, mask, debug


# ═══════════════════════════════════════════════════════════════
# 关键点生成与亚像素精化
# ═══════════════════════════════════════════════════════════════
def _generate_keypoints_from_corners(corners_2d, image):
    """从 4 角点生成 16 个场地关键点的亚像素坐标"""
    src = np.array(corners_2d, dtype=np.float32)
    dst = COURT_CORNERS_3D[:, :2].copy()
    H, mask = cv2.findHomography(dst, src)
    if H is None:
        return None

    obj_xy = COURT_KEYPOINTS_3D[:, :2].reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(obj_xy, H).reshape(-1, 2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 亚像素精化：在局部窗口搜索线交叉点
    refined = []
    win_size = (9, 9)
    zero_zone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)

    for pt in img_pts:
        x, y = pt
        h_img, w_img = image.shape[:2]
        if 5 <= x < w_img - 5 and 5 <= y < h_img - 5:
            corner = np.array([[x, y]], dtype=np.float32)
            refined_corner = cv2.cornerSubPix(gray, corner, win_size, zero_zone, criteria)
            refined.append(refined_corner[0])
        else:
            refined.append(pt)

    return np.array(refined, dtype=np.float32)


def _filter_inliers_by_reprojection(obj_pts, img_pts, K, dist, rvec, tvec, threshold=5.0):
    """重投影误差过滤离群点，返回内点索引"""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    errors = np.linalg.norm(img_pts - proj, axis=1)
    return errors < threshold, errors


# ═══════════════════════════════════════════════════════════════
# MultiCameraCalibrator — 主标定类
# ═══════════════════════════════════════════════════════════════
class MultiCameraCalibrator:
    """
    多视角球场标定器。

    用法：
        calib = MultiCameraCalibrator()
        stereo = calib.calibrate_two_views(img_left, img_right)
        stereo.save("calib.json")

        # 三角测量：
        pt3d = cv2.triangulatePoints(stereo.P1, stereo.P2, ptL, ptR)
    """

    def __init__(self, debug=False):
        self.debug = debug

    # ── 单相机标定 ──────────────────────────────────────
    def calibrate_single(self, image, cam_id=0) -> CameraPose:
        """
        自动检测球场 → 生成 16 关键点 → 亚像素精化 → solvePnP RANSAC。
        """
        h, w = image.shape[:2]

        if self.debug:
            print(f"\n[标定] 相机 {cam_id}: 图像 {w}×{h}")

        # 1. 自动检测四角
        corners = self._detect_corners(image)
        if corners is None:
            if self.debug:
                print(f"[标定] 相机 {cam_id}: 自动检测失败")
            return CameraPose(cam_id=cam_id, image_size=(w, h), success=False)

        if self.debug:
            print(f"[标定] 相机 {cam_id}: 检测到 4 角点: {corners}")

        # 2. 生成 16 个精化关键点
        kps = _generate_keypoints_from_corners(corners, image)
        if kps is None:
            return CameraPose(cam_id=cam_id, image_size=(w, h),
                              corners_2d=np.array(corners, dtype=np.float32), success=False)

        # 3. 初始内参（基于相机规格）
        K = np.array([
            [DEFAULT_FX, 0, w / 2.0],
            [0, DEFAULT_FX, h / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)

        # 4. solvePnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            COURT_KEYPOINTS_3D, kps, K, dist,
            iterationsCount=200,
            reprojectionError=6.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < 6:
            if self.debug:
                print(f"[标定] 相机 {cam_id}: solvePnP 失败 (内点数不足)")
            # 降级：用全部 16 点再做一次
            success, rvec, tvec = cv2.solvePnP(COURT_KEYPOINTS_3D, kps, K, dist,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                return CameraPose(cam_id=cam_id, image_size=(w, h),
                                  corners_2d=np.array(corners, dtype=np.float32),
                                  keypoints_2d=kps, success=False)
            inliers = np.arange(len(kps)).reshape(-1, 1)

        # 5. 重投影误差
        proj, _ = cv2.projectPoints(COURT_KEYPOINTS_3D, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        errors = np.linalg.norm(kps - proj, axis=1)
        mean_error = float(np.mean(errors))

        # 6. 投影矩阵
        R, _ = cv2.Rodrigues(rvec)
        P = K @ np.hstack([R, tvec.reshape(3, 1)])

        if self.debug:
            print(f"[标定] 相机 {cam_id}: 内点数={len(inliers)}/{len(kps)}, "
                  f"重投影误差={mean_error:.2f}px")
            print(f"  tvec = ({tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f})m")

        return CameraPose(
            cam_id=cam_id, image_size=(w, h), K=K, dist=dist,
            rvec=rvec, tvec=tvec, P=P,
            corners_2d=np.array(corners, dtype=np.float32),
            keypoints_2d=kps,
            reproj_error=mean_error,
            num_inliers=len(inliers),
            success=True,
        )

    # ── 双眼标定 — 两侧对拍主入口 ────────────────────────
    def calibrate_two_views(self, img_left, img_right,
                            cam_left_id=0, cam_right_id=1) -> StereoPose:
        """
        两眼标定：两侧相机各拍一张完整球场，独立标定后计算相对位姿。

        不要求两个相机同时看到同一个标定物——球场本身就是标定物。
        """
        if self.debug:
            print("=" * 60)
            print("  双眼标定 — 球场即标定板")
            print("=" * 60)

        # 独立标定
        pose_l = self.calibrate_single(img_left, cam_left_id)
        pose_r = self.calibrate_single(img_right, cam_right_id)

        if not pose_l.success or not pose_r.success:
            if self.debug:
                print("[标定] 失败：一个或两个相机标定不成功")
            return StereoPose(image_size=pose_l.image_size,
                              cam_left=pose_l, cam_right=pose_r, success=False)

        # 计算相对位姿
        R1, _ = cv2.Rodrigues(pose_l.rvec)
        R2, _ = cv2.Rodrigues(pose_r.rvec)

        R_rel = R2 @ R1.T
        T_rel = pose_r.tvec - R_rel @ pose_l.tvec
        baseline = float(np.linalg.norm(T_rel))

        if self.debug:
            print(f"\n[标定] 双眼系统:")
            print(f"  基线长度: {baseline:.2f}m")
            print(f"  左相机位置: ({pose_l.tvec[0,0]:.2f}, {pose_l.tvec[1,0]:.2f}, {pose_l.tvec[2,0]:.2f})")
            print(f"  右相机位置: ({pose_r.tvec[0,0]:.2f}, {pose_r.tvec[1,0]:.2f}, {pose_r.tvec[2,0]:.2f})")
            print(f"  左重投影误差: {pose_l.reproj_error:.2f}px")
            print(f"  右重投影误差: {pose_r.reproj_error:.2f}px")
            print(f"  P1 @ P2 已构建，可直接用于三角测量")

        return StereoPose(
            image_size=pose_l.image_size,
            cam_left=pose_l, cam_right=pose_r,
            R_rel=R_rel, T_rel=T_rel, baseline=baseline,
            success=True,
        )

    # ── 内部方法 ──────────────────────────────────────────
    def _detect_corners(self, image):
        if image is None or image.size == 0:
            return None
        orig_h, orig_w = image.shape[:2]
        fixed = (1080, 720)
        base = cv2.resize(image, fixed)
        corners, mask, debug = auto_detect_court_corners(base)
        if corners is None:
            return None
        scale_x = orig_w / fixed[0]
        scale_y = orig_h / fixed[1]
        return [(int(x * scale_x), int(y * scale_y)) for x, y in corners]


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════
def calibrate_from_two_images(img_left, img_right, save_path=None, debug=True):
    """从两张图像做双眼标定，可选保存"""
    calib = MultiCameraCalibrator(debug=debug)
    stereo = calib.calibrate_two_views(img_left, img_right)
    if stereo.success and save_path:
        stereo.save(save_path)
    return stereo


def load_stereo_calib(path):
    return StereoPose.load(path)


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多视角球场自动标定")
    parser.add_argument("--image", help="单张球场图像（单目标定）")
    parser.add_argument("--left", help="左相机图像（双眼标定）")
    parser.add_argument("--right", help="右相机图像（双眼标定）")
    parser.add_argument("--out", default="calib_stereo.json", help="输出文件")
    args = parser.parse_args()

    if args.left and args.right:
        img_l = cv2.imread(args.left)
        img_r = cv2.imread(args.right)
        if img_l is None or img_r is None:
            print("错误: 无法读取图像文件")
            exit(1)
        stereo = calibrate_from_two_images(img_l, img_r, args.out, debug=True)

    elif args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"错误: 无法读取图像: {args.image}")
            exit(1)
        calib = MultiCameraCalibrator(debug=True)
        pose = calib.calibrate_single(img)
        if pose.success:
            print(f"\n标定成功: {pose.num_inliers} 内点, 重投影误差 {pose.reproj_error:.2f}px")

    else:
        parser.print_help()
"""
video_pipeline.py — 单视频自动识别流水线（无需模型权重，纯 OpenCV）
=====================================================================
输入：一段羽毛球视频（手机拍的也行）
输出：
  1. 球场自动检测与透视标定（复用 calibration.auto_detect_court_corners）
  2. 运动先验法羽毛球候选检测（帧差 + 连通域筛选，专治 5-15px 小目标）
  3. 轨迹串联与主轨迹筛选（最近邻 + 速度门限，取代深度模型追踪）
  4. 落点估算（地平面假设 + 单应映射 → 球场米制坐标）
  5. 界内/界外判罚（CourtJudge：单打/双打边线 + 压线裕度 mm）
  6. 标注预览图 / 标注视频 / JSON 结果

设计原则：
  · 不强依赖外部权重，装 numpy+opencv 就能跑
  · 检测不到球场不崩，降级为"仅轨迹"模式并如实标注
  · 所有结论带置信度与失败原因，不编造
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from landing_judge import CourtJudge
from track_refine import (refine_tracks, template_recover,
                          coverage as track_coverage, fill_ratio)
import court_config as CC

try:
    from calibration import auto_detect_court_corners
except Exception:  # pragma: no cover
    auto_detect_court_corners = None

FONT_CANDIDATES = ["simhei.ttf", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]


# ══════════════════════════════════════════════════════════════
# 中文绘制
# ══════════════════════════════════════════════════════════════
def _font(size: int):
    try:
        from PIL import ImageFont
        for f in FONT_CANDIDATES:
            if os.path.exists(f):
                return ImageFont.truetype(f, size)
    except Exception:
        pass
    return None


def draw_cn(img_bgr: np.ndarray, items: List[Tuple[str, Tuple[int, int]]],
            size: int = 22, color=(255, 255, 255), bg: bool = True) -> np.ndarray:
    """在 BGR 图上画中文。items = [(text, (x, y)), ...]"""
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f = _font(size)
        for text, (x, y) in items:
            if bg:
                tw = d.textlength(text, font=f) if f else size * len(text) * 0.6
                d.rectangle([x - 6, y - 4, x + tw + 6, y + size + 6], fill=(20, 30, 40))
            if f:
                d.text((x, y), text, font=f, fill=(color[2], color[1], color[0]))
            else:
                d.text((x, y), text, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        for text, (x, y) in items:
            cv2.putText(img_bgr, text, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                        size / 32.0, color, 2, cv2.LINE_AA)
        return img_bgr


# ══════════════════════════════════════════════════════════════
# 1. 球场检测
# ══════════════════════════════════════════════════════════════
def _canonical_quad(pts: np.ndarray) -> np.ndarray:
    """把任意顺序的 4 点整理成 [左上, 右上, 右下, 左下]。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_court(video_path: str, max_samples: int = 18,
                 log: Callable[[str], None] = print,
                 max_frames: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict]:
    """多帧采样找最佳球场检测结果。返回 (corners, info)。"""
    info = {"attempted": 0, "success": False, "score": 0.0, "frame": -1}
    if auto_detect_court_corners is None:
        log("  [球场] 标定模块不可用，跳过球场检测")
        return None, info

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, info
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return None, info

    limit = min(total, max_frames) if max_frames else total
    idxs = np.linspace(0, max(limit - 2, 0), num=min(max_samples, max(limit, 1)), dtype=int)
    best, best_score = None, 0.0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        info["attempted"] += 1
        # 缩小加速（检测算法对分辨率不敏感）
        h, w = frame.shape[:2]
        scale = 900.0 / max(h, w) if max(h, w) > 900 else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        try:
            corners, _mask, debug = auto_detect_court_corners(small)
        except Exception:
            continue
        if corners is None:
            continue
        c = np.asarray(corners, dtype=np.float32).reshape(-1, 2) / scale
        quad = _canonical_quad(c)
        area = cv2.contourArea(quad.astype(np.float32))
        score = area / (h * w)                      # 占画面比例
        # 几何自检：真实球场在画面里通常占 12% 以上；且透视必有"近端宽 > 远端宽"
        far_w = float(np.linalg.norm(quad[1] - quad[0]))
        near_w = float(np.linalg.norm(quad[2] - quad[3]))
        ratio = near_w / max(far_w, 1e-6)
        geom_ok = (0.15 < score < 0.98) and (1.15 <= ratio <= 5.0)
        if not geom_ok and score > 0.02:
            log(f"  [球场] 候选四边形被几何自检否决"
                f"（占比 {score*100:.0f}%，近/远端宽度比 {ratio:.2f}）")
        if geom_ok and score > best_score:
            best, best_score = quad, score
            info["frame"] = int(i)

    cap.release()
    if best is None:
        log(f"  [球场] {info['attempted']} 帧采样未检测到完整球场（降级为仅轨迹模式）")
        info["success"] = False
        return None, info

    info.update(success=True, score=round(float(best_score), 3))
    log(f"  [球场] 检测成功（第 {info['frame']} 帧，占画面 {best_score*100:.0f}%）")
    return best, info


def build_homography(corners: np.ndarray, mode: str = "doubles"):
    """图像 4 角点 → 球场米制平面。返回 (H, 场地长宽)。"""
    L, W = CC.COURT_LENGTH, CC.COURT_WIDTH
    dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return H, (W, L)


def img_to_court(H: np.ndarray, pt) -> Tuple[float, float]:
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, H)[0, 0]
    return float(q[0]), float(q[1])


# ══════════════════════════════════════════════════════════════
# 2-3. 运动法小目标检测 + 轨迹串联
# ══════════════════════════════════════════════════════════════
class MotionDetector:
    """羽毛球候选点检测：背景建模 + 帧差 双路并集（鲁棒版）

    改进点（相对原始帧差法）：
      · MOG2 背景建模：抗静态场景的光照渐变、缓慢变化
      · 帧差：补背景模型初始化期的漏检、抗运动球漏检
      · 自适应阈值：按图像噪声水平动态设门限（暗光/高噪场景不失效）
      · 尺寸门限按画面比例自动计算，排除球员/球拍等大目标
      · 内部降采样到 720p 处理，速度提升且不损失小球定位精度
    """

    def __init__(self, proc_width: int = 1280, area_ratio=(0.000004, 0.00045),
                 max_obj_px: int = 26, use_bg: bool = True, max_per_frame: int = 24):
        self.proc_width = proc_width
        self.area_ratio = area_ratio
        self.max_obj_px = max_obj_px
        self.use_bg = use_bg
        self.max_per_frame = max_per_frame
        self.last_shift = (0.0, 0.0)
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=18, detectShadows=False) if use_bg else None

    def detect(self, prev_gray, gray):
        """返回 [(cx, cy, area)]，坐标为输入图坐标系。"""
        h, w = gray.shape[:2]
        scale = 1.0
        if w > self.proc_width:
            scale = self.proc_width / float(w)
            gray_s = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            prev_s = (cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                      if prev_gray is not None else None)
        else:
            gray_s, prev_s = gray, prev_gray
        hs, ws = gray_s.shape[:2]

        masks = []
        if self.use_bg and self.bg is not None:
            masks.append(self.bg.apply(gray_s))
        if prev_s is not None:
            prev_aligned = prev_s
            # 镜头抖动补偿：相位相关估计全局位移，超过 1.2px 则对齐后再差分
            try:
                (dx, dy), _resp = cv2.phaseCorrelate(
                    np.float32(prev_s), np.float32(gray_s))
                if abs(dx) > 1.2 or abs(dy) > 1.2:
                    Mshift = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    prev_aligned = cv2.warpAffine(prev_s, Mshift, (ws, hs),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REPLICATE)
                    self.last_shift = (dx, dy)
            except Exception:
                pass
            diff = cv2.absdiff(gray_s, prev_aligned)
            diff = cv2.GaussianBlur(diff, (3, 3), 0)
            sigma = float(np.median(diff)) + 3.0 * 1.4826 * float(np.median(np.abs(diff - np.median(diff))))
            thr = max(12.0, min(60.0, sigma * 1.6))
            _, m = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
            masks.append(m)
        if not masks:
            return []
        m = masks[0] if len(masks) == 1 else cv2.bitwise_or(masks[0], masks[1])

        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)

        area_min = max(2.0, self.area_ratio[0] * hs * ws)
        area_max = max(60.0, self.area_ratio[1] * hs * ws)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_mean = float(gray_s.mean())
        cands = []
        for c in cnts:
            a = cv2.contourArea(c)
            if not (area_min <= a <= area_max):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > self.max_obj_px * ws / 1280.0 or bh > self.max_obj_px * hs / 720.0:
                continue                      # 球员/球拍等大目标
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            # 羽毛球是高亮小白点：要求候选区域均值明显亮于全图（真场景的关键过滤）
            patch = gray_s[max(int(cy) - 4, 0):int(cy) + 5, max(int(cx) - 4, 0):int(cx) + 5]
            bright = float(patch.mean()) if patch.size else 0.0
            if bright < img_mean * 1.12:
                continue
            cands.append((cx / scale, cy / scale, float(a), bright))
        # 每帧候选上限，按亮度排序取最强的
        cands.sort(key=lambda z: z[3], reverse=True)
        return [(x, y, a) for x, y, a, _b in cands[:self.max_per_frame]]


class YoloBallDetector:
    """基于 yolo11s-ball.pt 的羽毛球检测器（真实比赛视频的主力方案）。

    多人、多场地、镜头抖动的真场景里，帧差法会产生大量球员干扰点；
    专用球检测模型能把这类误检压到很低。权重文件放在 weights/yolo11s-ball.pt。
    """

    DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "weights", "yolo11s-ball.pt")

    def __init__(self, weights: Optional[str] = None, conf: float = 0.25,
                 imgsz: int = 1280, device: str = "cpu", max_per_frame: int = 8):
        from ultralytics import YOLO   # 延迟导入：没装 ultralytics 时不影响运动检测模式
        w = weights or self.DEFAULT_WEIGHTS
        if not os.path.exists(w):
            raise FileNotFoundError(f"未找到球检测权重: {w}")
        self.model = YOLO(w)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.max_per_frame = max_per_frame
        self.available = True

    @classmethod
    def is_available(cls, weights: Optional[str] = None) -> bool:
        w = weights or cls.DEFAULT_WEIGHTS
        if not os.path.exists(w):
            return False
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False

    def detect(self, frame, _gray=None):
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        out = []
        if r.boxes is None:
            return out
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            c = float(b.conf[0]) if b.conf is not None and len(b.conf) else 0.0
            out.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, float((x2 - x1) * (y2 - y1)) * 0.01 + c))
        out.sort(key=lambda z: z[2], reverse=True)
        return out[:self.max_per_frame]


def link_tracks(points_per_frame: List[List[Tuple[float, float, float]]],
                max_jump_ratio: float = 0.10, max_miss: int = 2) -> List[List[Tuple[int, float, float]]]:
    """最近邻贪心串联：返回若干条轨迹 [(frame, x, y), ...]。"""
    tracks: List[Dict] = []
    for fi, pts in enumerate(points_per_frame):
        for (x, y, a) in pts:
            best, best_d = None, 1e9
            for t in tracks:
                if fi - t["last_f"] > max_miss + 1:
                    continue
                lx, ly = t["pts"][-1][1], t["pts"][-1][2]
                d = float(np.hypot(x - lx, y - ly))
                if d < best_d:
                    best, best_d = t, d
            if best is not None and best_d < max_jump_ratio * max(1, fi + 1) * 100:
                best["pts"].append((fi, x, y))
                best["last_f"] = fi
            else:
                tracks.append({"pts": [(fi, x, y)], "last_f": fi})
    return [t["pts"] for t in tracks]


def parabola_fit_score(track, min_len: int = 8) -> float:
    """轨迹的抛物线拟合度 R²（0~1）。羽毛球飞行近似抛物线，
    球员跑动/球拍挥动等干扰轨迹拟合度低，用此打分可有效剔除。"""
    if len(track) < min_len:
        return 0.0
    xs = np.array([p[1] for p in track], float)
    ys = np.array([p[2] for p in track], float)
    if np.ptp(xs) < 12 or np.ptp(ys) < 6:
        return 0.0
    try:
        coef = np.polyfit(xs, ys, 2)
        pred = np.polyval(coef, xs)
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        return max(0.0, 1.0 - ss_res / max(ss_tot, 1e-9))
    except Exception:
        return 0.0


def pick_main_track(tracks, min_len: int = 6, max_len: int = 120,
                    W_img: float = 1280.0, H_img: float = 720.0,
                    min_disp: float = 30.0, min_r2: float = 0.35,
                    log: Optional[Callable] = None,
                    in_court_fn: Optional[Callable] = None,
                    min_in_court: float = 0.6):
    """挑主轨迹：抛物线拟合度 × 长度 × 平均速度 综合打分。

    真实场景下（多球员/观众/镜头抖动）会产生大量干扰轨迹，因此加三重硬门限：
      · 长度上限：真实球在 30fps 下通常 8~90 帧可见，几千点的必是噪声
      · 净位移下限：球必须在画面里真的移动过
      · 抛物线拟合度下限：飞行轨迹必须近似抛物线
    """
    if not tracks:
        return None, 0.0, 0.0
    scored = []
    for t in tracks:
        if not (min_len <= len(t) <= max_len):
            continue
        # 关键约束：比赛中的球绝大多数时间在球场（含小幅边距）内。
        # 不满足的轨迹多为观众/裁判/广告牌等误检，直接剔除。
        # 物理常识：羽毛球单次飞行的路径长度不可能远超场地在画面中的尺度
        diag = float(np.hypot(W_img, H_img))
        path = 0.0
        for (f0, x0, y0), (f1, x1, y1) in zip(t, t[1:]):
            path += float(np.hypot(x1 - x0, y1 - y0))
        if diag and path > 5.0 * diag:
            if log:
                log(f"  [轨迹] 剔除一条路径长度 {path:.0f}px 的轨迹（超过 5×画面对角线，"
                    "几乎必为场外移动目标）")
            continue
        if in_court_fn is not None:
            inside = sum(1 for p in t if in_court_fn(p[1], p[2]))
            ratio = inside / max(len(t), 1)
            if ratio < min_in_court:
                if log:
                    log(f"  [轨迹] 剔除一条场内占比仅 {ratio*100:.0f}% 的轨迹"
                        f"（长度 {len(t)}，多为场外误检）")
                continue
        disp = float(np.hypot(t[-1][1] - t[0][1], t[-1][2] - t[0][2]))
        if disp < min_disp:
            continue
        speed = 0.0
        for (f0, x0, y0), (f1, x1, y1) in zip(t, t[1:]):
            if f1 != f0:
                speed += np.hypot(x1 - x0, y1 - y0) / (f1 - f0)
        speed /= max(len(t) - 1, 1)
        r2 = parabola_fit_score(t)
        score = (0.25 + 0.75 * r2) * len(t) * (1.0 + min(speed, 200.0) / 80.0)
        scored.append((score, r2, t, disp))
    if not scored:
        if log:
            log(f"  [轨迹] {len(tracks)} 条候选均未通过门限（长度6~{max_len}、净位移≥{min_disp:.0f}px）")
        return None, 0.0, 0.0

    scored.sort(key=lambda z: z[0], reverse=True)
    if log and len(scored) >= 2:
        log("  [轨迹] 候选前3: " + ", ".join(
            f"R²={s[1]:.2f}/长度{len(s[2])}/位移{s[3]:.0f}px" for s in scored[:3]))
    # 先过硬门槛（抛物线拟合度），再在合格轨迹里排序取最优
    qualified = [s for s in scored if s[1] >= min_r2]
    if not qualified:
        if log:
            log(f"  [轨迹] {len(scored)} 条候选抛物线拟合度均 < {min_r2}，判为不可信（多为球员跑动干扰）")
        return None, 0.0, float(scored[0][1])
    qualified.sort(key=lambda z: (0.25 + 0.75 * z[1]) * len(z[2]) * (1.0 + min(
        float(np.hypot(z[2][-1][1] - z[2][0][1], z[2][-1][2] - z[2][0][2])) / max(len(z[2]), 1), 200.0) / 80.0),
        reverse=True)
    best = qualified[0]
    if log:
        log(f"  [轨迹] 合格轨迹 {len(qualified)} 条，选中 R²={best[1]:.2f}/长度{len(best[2])}")
    return best[2], float(best[0]), float(best[1])


def extrapolate_landing(court_pts: List[Tuple[float, float, float]],
                        frames_ahead: int = 14) -> Optional[Tuple[float, float]]:
    """在球场坐标系中二次拟合 x(t) 与 y(t)，向前外推得到估算落点。"""
    if len(court_pts) < 6:
        return None
    tail = court_pts[-max(8, int(len(court_pts) * 0.6)):]
    t = np.array([p[0] for p in tail], float)
    xx = np.array([p[1] for p in tail], float)
    yy = np.array([p[2] for p in tail], float)
    try:
        cx = np.polyfit(t, xx, 2)
        cy = np.polyfit(t, yy, 2)
    except Exception:
        return None
    t_end = float(t[-1]) + frames_ahead
    return float(np.polyval(cx, t_end)), float(np.polyval(cy, t_end))


# ══════════════════════════════════════════════════════════════
# 4-6. 主流程
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# 单机位标定复用：固定机位只需标定一次，之后直接读 JSON
# ══════════════════════════════════════════════════════════════
DEFAULT_CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "calib_single_cam.json")


def save_calib(corners, path: str = DEFAULT_CALIB, meta: Optional[Dict] = None):
    data = {"corners": [[float(x), float(y)] for x, y in corners],
            "meta": meta or {}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_calib(path: str = DEFAULT_CALIB):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        c = d.get("corners")
        if c and len(c) == 4:
            return np.array(c, dtype=np.float32)
    except Exception:
        pass
    return None


def load_ball_csv(csv_path: str, scale_x: float = 1.0, scale_y: float = 1.0,
                  min_vis: int = 1):
    """读取 TrackNetV3 输出的逐帧球坐标 CSV（Frame,Visibility,X,Y）→ 逐帧检测列表。"""
    import csv as _csv
    per = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                fi = int(float(row["Frame"]))
                vis = int(float(row.get("Visibility", 1)))
                x = float(row["X"]) * scale_x
                y = float(row["Y"]) * scale_y
            except Exception:
                continue
            if vis < min_vis:
                continue
            per.setdefault(fi, []).append((x, y, 1.0))
    if not per:
        return []
    return [per.get(i, []) for i in range(max(per) + 1)]


def fuse_detections(model_pf: List[List[Tuple[float, float, float]]],
                    csv_pf: List[List[Tuple[float, float, float]]],
                    gate_px: float = 24.0) -> List[List[Tuple[float, float, float]]]:
    """双源融合：TrackNet（时序、少误检）为主，YOLO 补孤点。

    规则（逐帧）：
      · TrackNet 有检出 → 以其为准；YOLO 的点若距最近 TrackNet 点 > gate_px 也保留
        （可能是 TrackNet 漏掉的快球），但每帧最多补 1 个，避免误检灌入；
      · TrackNet 无检出 → 完全采用 YOLO。
    """
    n = max(len(model_pf), len(csv_pf))
    out: List[List[Tuple[float, float, float]]] = []
    for i in range(n):
        m = model_pf[i] if i < len(model_pf) else []
        c = csv_pf[i] if i < len(csv_pf) else []
        if not c:
            out.append(list(m))
            continue
        merged = list(c)
        added = 0
        for (mx, my, ms) in m:
            if added >= 1:
                break
            d = min((math.hypot(mx - cx, my - cy) for cx, cy, _ in c), default=1e9)
            if d > gate_px:
                merged.append((mx, my, ms))
                added += 1
        out.append(merged)
    return out


def _ffmpeg_path() -> Optional[str]:
    for name in ("ffmpeg", "ffmpeg.exe"):
        try:
            r = subprocess.run([name, "-version"], capture_output=True, timeout=8)
            if r.returncode == 0:
                return name
        except Exception:
            continue
    return None


def ensure_decodable(path: str, log: Callable[[str], None] = print,
                     max_seconds: Optional[float] = None) -> str:
    """确保 OpenCV 能解码。H.265/HEVC、部分手机录制的 mov 无法直接解码时，
    用 ffmpeg 转 H.264 再读（真实素材最常见的坑）。"""
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    cap.release()
    if ok:
        return path

    exe = _ffmpeg_path()
    if exe is None:
        log("  [解码] 该视频 OpenCV 无法解码（HEVC/特殊封装），且未找到 ffmpeg，无法转码")
        return path

    out = os.path.join(tempfile.mkdtemp(prefix="hawkeye_conv_"),
                       Path(path).stem + "_h264.mp4")
    cmd = [exe, "-y", "-v", "error", "-i", path]
    if max_seconds:
        cmd += ["-t", str(int(max_seconds))]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an", out]
    log("  [解码] 检测到 HEVC/特殊编码，正在用 ffmpeg 转码为 H.264 …")
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800)
        if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1024:
            log(f"  [解码] 转码完成: {out}")
            return out
        log(f"  [解码] 转码失败: {r.stderr.decode('utf-8', 'ignore')[:160]}")
    except Exception as e:
        log(f"  [解码] 转码异常: {e}")
    return path


def analyze_video(video_path: str, out_dir: Optional[str] = None,
                  mode: str = "doubles",
                  log: Callable[[str], None] = print,
                  make_video: bool = True,
                  max_frames: Optional[int] = None,
                  manual_corners: Optional[List[List[float]]] = None,
                  detector: str = "auto",
                  calib_file: Optional[str] = None,
                  reuse_calib: bool = False,
                  tracker: str = "refined",
                  ball_csv: Optional[str] = None,
                  fuse_csv: bool = False,
                  skip_landing: bool = False) -> Dict:
    t0 = time.time()
    video_path = str(video_path)
    if out_dir is None:
        out_dir = str(Path(video_path).parent / "hawkeye_out" / Path(video_path).stem)
    os.makedirs(out_dir, exist_ok=True)

    result: Dict = {"video": video_path, "mode": mode, "ok": False,
                    "stages": {}, "out_dir": out_dir}

    log(f"▶ 分析: {Path(video_path).name}")
    # ---- 解码兜底（HEVC / 手机录制） ----
    vp = ensure_decodable(video_path, log=log)
    if vp != video_path:
        result["transcoded"] = vp
    # ---- 元信息 ----
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        log("✗ 无法打开视频（解码失败）")
        result["error"] = "cannot_open"
        return result
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    log(f"  分辨率 {w}×{h}, {fps:.1f} fps, {n_total} 帧"
        + (f"（只处理前 {max_frames} 帧）" if max_frames else ""))
    result.update(fps=round(fps, 2), frames=n_total, size=[w, h])

    # ---- 1. 球场（手动四点 > 已保存标定 > 自动检测）----
    saved = load_calib(calib_file or DEFAULT_CALIB)
    reused = False
    if not (manual_corners and len(manual_corners) == 4) and saved is not None:
        # 校验：分辨率不同必然不是同一机位，直接作废重检
        meta = {}
        try:
            with open(calib_file or DEFAULT_CALIB, encoding="utf-8") as _f:
                meta = json.load(_f).get("meta", {}) or {}
        except Exception:
            pass
        fs = meta.get("frame_size")
        if fs and [int(w), int(h)] != [int(fs[0]), int(fs[1])]:
            log(f"  [球场] 已保存标定来自 {fs[0]}x{fs[1]} 的画面，与当前 "
                f"{int(w)}x{int(h)} 不符 → 作废，重新检测")
            saved = None
        elif not reuse_calib:
            log("  [球场] 已保存标定存在，但本次未要求复用 → 重新检测"
                "（同机位可加 --reuse-calib 跳过检测）")
            saved = None
        else:
            log("  [球场] 复用已保存标定（假定与保存时是同一机位/同一场地，"
                "换过机位请点「清除标定」或加 --no-reuse-calib）")
    if saved is not None and not (manual_corners and len(manual_corners) == 4):
        manual_corners = saved.tolist()
        reused = True
    if manual_corners and len(manual_corners) == 4:
        corners = _canonical_quad(np.array(manual_corners, dtype=np.float32))
        court_info = {"attempted": 0, "success": True, "score": 1.0,
                      "frame": 0, "source": "reused" if reused else "manual"}
        log("  [球场] 复用标定四点（固定机位无需重复标定）" if reused
            else "  [球场] 使用手动标定的四点")
    else:
        corners, court_info = detect_court(vp, log=log, max_frames=max_frames)
        if corners is not None:
            court_info["source"] = "auto"
    H, (CW, CL) = (None, (CC.COURT_WIDTH, CC.COURT_LENGTH))
    if corners is not None:
        H, (CW, CL) = build_homography(corners, mode)
        result["stages"]["court"] = {"ok": True, **court_info,
                                     "corners": corners.astype(float).tolist()}
        # 固定机位：把这套角点存下来，后续同机位视频可跳过标定
        try:
            cp = save_calib(corners, calib_file or DEFAULT_CALIB,
                            meta={"source": court_info.get("source", "auto"),
                                  "from_video": os.path.basename(video_path),
                                  "frame_size": [int(w), int(h)]})
            log(f"  [球场] 标定已保存：{cp}（同机位下次自动复用）")
        except Exception as e:
            log(f"  [球场] 标定保存失败：{e}")
    else:
        result["stages"]["court"] = {"ok": False, **court_info}

    # 标定置信度：自动检测出的四边形若只占画面一小块，多半是"子区域"误拟合
    calib_confident = True
    if corners is not None and (court_info.get("source") == "auto"):
        area_ratio = float(court_info.get("score", 0.0))
        if area_ratio < 0.30:
            calib_confident = False
            log(f"  [球场] 标定置信度低（仅占画面 {area_ratio*100:.0f}%）："
                "本次不用于轨迹门控与落点判罚，请用「手动标定球场」点四个角")
            result["stages"]["court"]["low_confidence"] = True

    # ---- 2. 逐帧检测：TrackNet CSV 优先，否则 YOLO / 运动检测 ----
    per_frame: List[List[Tuple[float, float, float]]] = []
    keep_frames: List[np.ndarray] = []
    gray_keep: List[np.ndarray] = []
    use_yolo = False
    det_cov_before = det_cov_after = 0.0

    if ball_csv and os.path.exists(ball_csv):
        # TrackNetV3 的 predict.py 写出 CSV 时已乘过 img_scaler（w/512, h/288），
        # 即坐标已经是原视频分辨率，这里不要再缩放（曾因此把坐标放大 2 倍而全部出界）。
        sx = sy = 1.0
        per_frame = load_ball_csv(ball_csv, sx, sy)
        cap0 = cv2.VideoCapture(vp)
        while len(keep_frames) < len(per_frame):
            ok, fr = cap0.read()
            if not ok:
                break
            keep_frames.append(fr)
            gray_keep.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap0.release()
        while len(per_frame) < len(keep_frames):
            per_frame.append([])
        n = len(per_frame)
        n_cand = sum(len(p) for p in per_frame)
        det_cov_before = det_cov_after = sum(1 for p in per_frame if p) / max(n, 1)
        log(f"  [检测] 使用 TrackNetV3 CSV：{n} 帧，候选点 {n_cand} 个，"
            f"有检出帧占比 {det_cov_after*100:.0f}%（坐标已按 {sx:.2f}× 缩放）")
        if fuse_csv and YoloBallDetector.is_available():
            try:
                _yd = YoloBallDetector()
                _mpf = []
                for _fr in keep_frames:
                    _g = cv2.cvtColor(_fr, cv2.COLOR_BGR2GRAY)
                    _mpf.append(_yd.detect(_fr, _g))
                before = det_cov_after
                per_frame = fuse_detections(_mpf, per_frame)
                n_cand = sum(len(p) for p in per_frame)
                det_cov_after = sum(1 for p in per_frame if p) / max(n, 1)
                log(f"  [检测] 双源融合完成（TrackNet + YOLO）：候选点 {n_cand} 个，"
                    f"有检出帧占比 {before*100:.0f}% → {det_cov_after*100:.0f}%")
                _st = result["stages"].setdefault("detect", {})
                _st.update({"fused": True, "candidates": n_cand,
                            "frame_coverage": round(det_cov_after, 3)})
            except Exception as e:
                log(f"  [检测] 融合失败（{e}），仅用 TrackNet")
        result["stages"]["detect"] = {"ok": n > 0, "frames": n, "candidates": n_cand,
                                      "detector": "tracknet_csv",
                                      "frame_coverage": round(det_cov_after, 3)}
    else:
        cap = cv2.VideoCapture(vp)
        prev_gray = None
        if detector in ("auto", "yolo") and YoloBallDetector.is_available():
            try:
                det = YoloBallDetector()
                use_yolo = True
                log(f"  [检测] 使用 YOLO 球检测模型（imgsz={det.imgsz}, conf={det.conf}）")
            except Exception as e:
                log(f"  [检测] YOLO 初始化失败（{e}），回退运动检测")
                det = MotionDetector()
        else:
            det = MotionDetector()
            if detector == "yolo":
                log("  [检测] 未找到 weights/yolo11s-ball.pt 或未安装 ultralytics，回退运动检测")
            else:
                log("  [检测] 使用运动检测（背景建模+帧差）")
        fi = 0
        next_report = 0.1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and fi >= max_frames:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            per_frame.append(det.detect(frame, gray) if use_yolo else det.detect(prev_gray, gray))
            prev_gray = gray
            keep_frames.append(frame)
            gray_keep.append(gray)
            fi += 1
            if n_total and n_total > 60:
                prog = fi / float(min(n_total, max_frames or n_total))
                if prog >= next_report:
                    log(f"  [检测] 进度 {prog*100:.0f}%  ({fi} 帧)")
                    next_report += 0.25
        cap.release()
        n = len(per_frame)
        n_cand = sum(len(p) for p in per_frame)
        det_cov_before = sum(1 for p in per_frame if p) / max(n, 1)
        if tracker == "refined" and gray_keep:
            per_frame, n_added = template_recover(per_frame, gray_keep)
            if n_added:
                n_cand = sum(len(p) for p in per_frame)
                log(f"  [检测] 模板补检 +{n_added} 点（漏检帧找回）")
        det_cov_after = sum(1 for p in per_frame if p) / max(n, 1)
        log(f"  [检测] 处理 {n} 帧，候选点 {n_cand} 个"
            f"（{'YOLO 球检测' if use_yolo else '背景建模+帧差双路'}）"
            f"；有检出帧占比 {det_cov_before*100:.0f}% → {det_cov_after*100:.0f}%")
        result["stages"]["detect"] = {"ok": n > 0, "frames": n, "candidates": n_cand,
                                      "detector": "yolo" if use_yolo else "motion",
                                      "frame_coverage_before": round(det_cov_before, 3),
                                      "frame_coverage": round(det_cov_after, 3)}

    # ---- 3. 轨迹串联（NN 基线 vs 时序补全）+ 抛物线筛选 ----
    in_court_fn = None
    if H is not None and calib_confident:
        margin_x, margin_y = 1.0, 1.5   # 允许球出界一小段（界外球也是球）

        def in_court_fn(px, py, _H=H, _mx=margin_x, _my=margin_y):
            cx_, cy_ = img_to_court(_H, (px, py))
            return (-_mx <= cx_ <= CW + _mx) and (-_my <= cy_ <= CL + _my)

    raw_tracks = link_tracks(per_frame)
    raw_best, _s0, _r0 = pick_main_track(raw_tracks, W_img=float(w), H_img=float(h),
                                        in_court_fn=in_court_fn)
    raw_cov = 0.0
    if raw_best:
        span = raw_best[-1][0] - raw_best[0][0] + 1
        raw_cov = len({p[0] for p in raw_best}) / max(span, 1)
    result["stages"]["track_nn"] = {"len": len(raw_best) if raw_best else 0,
                                    "coverage": round(raw_cov, 3)}

    if tracker == "refined":
        ref_tracks = refine_tracks(per_frame, (int(w), int(h)))
        tracks = [[(f, x, y) for f, x, y, _filled in t] for t in ref_tracks]
        log(f"  [追踪] 时序补全：{len(ref_tracks)} 条轨迹"
            f"（NN 基线 {len(raw_tracks)} 条）")
        tracks = tracks or raw_tracks
    else:
        tracks = raw_tracks

    main_track, score, r2 = pick_main_track(tracks, log=log, W_img=float(w), H_img=float(h),
                                            in_court_fn=in_court_fn)

    # 结果驱动回退：YOLO 模式下轨迹不成形（如合成/非常规画面），补跑运动检测重试一次
    if (main_track is None or len(main_track) < 4) and use_yolo and detector == "auto":
        log("  [检测] YOLO 未形成有效轨迹，自动补跑运动检测后重试")
        det_m = MotionDetector()
        prev_gray = None
        for i, frame in enumerate(keep_frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            per_frame[i] = per_frame[i] + det_m.detect(prev_gray, gray)
            prev_gray = gray
        use_yolo = False
        tracks = link_tracks(per_frame)
        main_track, score, r2 = pick_main_track(tracks, log=log, W_img=float(w), H_img=float(h),
                                            in_court_fn=in_court_fn)
        result["stages"]["detect"]["fallback"] = "motion"

    if main_track is None or len(main_track) < 4:
        log("  [轨迹] 未形成有效轨迹（可能是镜头移动或球速过快）")
        result["stages"]["track"] = {"ok": False, "tracks": len(tracks)}
        result["error"] = "no_track"
        return _finish(result, out_dir, None, log, t0)

    log(f"  [轨迹] 候选轨迹 {len(tracks)} 条，主轨迹 {len(main_track)} 点（抛物线 R²={r2:.2f}）")
    span = main_track[-1][0] - main_track[0][0] + 1
    cov = len({p[0] for p in main_track}) / max(span, 1)
    log(f"  [追踪] 主轨迹覆盖率 {cov*100:.0f}%（{len(main_track)} 点 / 跨度 {span} 帧，"
        f"NN 基线 {raw_cov*100:.0f}%）")
    result["stages"]["track"] = {"ok": True, "tracks": len(tracks),
                                 "main_len": len(main_track),
                                 "score": round(score, 1), "parabola_r2": round(r2, 3),
                                 "coverage": round(cov, 3),
                                 "coverage_nn": round(raw_cov, 3)}

    # ---- 4. 落点：观测值 + 外推值双重估计 ----
    landing = None
    if skip_landing:
        log("  [落点] 已按要求跳过判罚（仅输出轨迹）")
        result["stages"]["landing"] = {"ok": False, "reason": "skipped_by_user"}
        landing = None
        _skip = True
    else:
        _skip = False
    last_f, lx, ly = main_track[-1]
    if H is not None and not calib_confident and not _skip:
        log("  [落点] 标定置信度不足，跳过落点判罚（请重做手动标定）")
        result["stages"]["landing"] = {"ok": False, "reason": "low_confidence_calib"}
        _skip = True

    if H is not None and not _skip:
        judge = CourtJudge(mode=mode)
        court_pts = []
        for f, x, y in main_track:
            cx_, cy_ = img_to_court(H, (x, y))
            court_pts.append((float(f), cx_, cy_))

        # 标定自洽性：若多数轨迹点映射后远超球场，说明标定本身错了 → 不判罚
        inb = sum(1 for _, cx_, cy_ in court_pts
                  if -0.5 * CW <= cx_ <= 1.5 * CW and -0.5 * CL <= cy_ <= 1.5 * CL)
        ratio_in = inb / max(len(court_pts), 1)
        if ratio_in < 0.6:
            log(f"  [落点] 轨迹只有 {ratio_in*100:.0f}% 的点映射在球场附近 → "
                "球场标定与轨迹不自洽，本次不判罚（请检查/重做标定）")
            result["stages"]["landing"] = {"ok": False, "reason": "calib_inconsistent",
                                           "in_court_ratio": round(ratio_in, 3)}
            landing = None
            return _finish(result, out_dir, None, log, t0)

        obs = court_pts[-1]
        ext = extrapolate_landing(court_pts, frames_ahead=min(8, max(4, len(main_track) // 8)))
        # 主判罚用「最后观测点」：真实视频中球在落地瞬间失联，末点即落点
        use = (obs[1], obs[2])
        plausible = (-0.3 * CW <= use[0] <= 1.3 * CW and -0.3 * CL <= use[1] <= 1.3 * CL)
        if not plausible:
            # 单应变换只对地平面(z=0)成立：球若仍在空中，映射必然离谱。
            log(f"  [落点] 映射结果 ({use[0]:.1f}, {use[1]:.1f}) m 远超场地范围 —— "
                "本段轨迹未跟到落地（球仍在空中），不给出落点判罚")
            result["stages"]["landing"] = {"ok": False, "reason": "still_in_flight",
                                           "raw_court_xy": [round(use[0], 3),
                                                            round(use[1], 3)]}
            landing = None
        else:
            j = judge.judge(use[0], use[1])
            landing = {"frame": last_f,
                       "court_xy": [round(use[0], 3), round(use[1], 3)],
                       "judge": j,
                       "method": "last_observed",
                       "image_xy": [round(lx, 1), round(ly, 1)],
                       "estimated": True}
            # 外推值仅作参考（轨迹明显未走完时更接近真实落点）
            if ext is not None and -0.3 * CW <= ext[0] <= 1.3 * CW and \
                    -0.3 * CL <= ext[1] <= 1.3 * CL:
                ej = judge.judge(ext[0], ext[1])
                landing["extrapolated_court_xy"] = [round(ext[0], 3), round(ext[1], 3)]
                landing["extrapolated_judge"] = ej["result"]
                log(f"  [落点] 参考外推: ({ext[0]:.2f}, {ext[1]:.2f}) m → "
                    f"{ej['result']}（仅供参考）")
            log(f"  [落点] 球场坐标 ({use[0]:.2f}, {use[1]:.2f}) m → {j['result']}"
                f"（压线裕度 {j['margin_mm']} mm，末观测点估算）")
    elif not _skip:
        log("  [落点] 无球场标定，跳过落点判罚（仅输出轨迹）")
        result["stages"]["landing"] = {"ok": False, "reason": "no_court_calib"}
    if landing:
        result["stages"]["landing"] = {"ok": True, **landing}

    # ---- 5. 渲染 ----
    preview = _render(keep_frames, main_track, corners, H, landing, result, CW, CL)
    prev_path = os.path.join(out_dir, "preview.png")
    cv2.imwrite(prev_path, preview)
    result["preview"] = prev_path
    log(f"  [输出] 预览图 {prev_path}")

    if make_video:
        vpath = _render_video(keep_frames, per_frame, main_track, corners, H, landing, out_dir, fps)
        if vpath:
            result["annotated_video"] = vpath
            log(f"  [输出] 标注视频 {vpath}")

    # 轨迹数据
    result["trajectory"] = [{"frame": f, "x": round(x, 1), "y": round(y, 1)}
                            for f, x, y in main_track]
    result["ok"] = True
    return _finish(result, out_dir, landing, log, t0)


def _finish(result: Dict, out_dir: str, landing, log, t0) -> Dict:
    result["elapsed_s"] = round(time.time() - t0, 2)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"✔ 完成，用时 {result['elapsed_s']}s → {out_dir}")
    return result


def _render(frames, track, corners, H, landing, result, CW, CL) -> np.ndarray:
    base = frames[min(track[-1][0], len(frames) - 1)].copy() if frames else np.zeros((720, 1280, 3), np.uint8)
    img = base.copy()

    # 球场网格（H 反向映射）
    if H is not None:
        Hinv = np.linalg.inv(H)
        def to_img(p):
            q = cv2.perspectiveTransform(np.array([[[p[0], p[1]]]], np.float32), Hinv)[0, 0]
            return int(q[0]), int(q[1])
        lines = []
        # 边线
        lines += [((0, 0), (CW, 0)), ((CW, 0), (CW, CL)), ((CW, CL), (0, CL)), ((0, CL), (0, 0))]
        # 单打边线
        m = 0.46
        lines += [((m, 0), (m, CL)), ((CW - m, 0), (CW - m, CL))]
        # 中线 & 发球线
        lines += [((0, CL / 2), (CW, CL / 2)),
                  ((0, 1.98), (CW, 1.98)), ((0, CL - 1.98), (CW, CL - 1.98)),
                  ((0, CL / 2 - 1.98), (CW, CL / 2 - 1.98)),
                  ((0, CL / 2 + 1.98), (CW, CL / 2 + 1.98))]
        for a, b in lines:
            cv2.line(img, to_img(a), to_img(b), (120, 200, 120), 1, cv2.LINE_AA)
        for c in corners:
            cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 200, 255), -1)

    # 轨迹
    pts = [(int(x), int(y)) for _, x, y in track]
    for i in range(1, len(pts)):
        alpha = i / max(len(pts) - 1, 1)
        color = (int(80 * alpha), int(120 + 100 * alpha), 255)
        cv2.line(img, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    for p in pts[:: max(len(pts) // 25, 1)]:
        cv2.circle(img, p, 2, (255, 240, 120), -1)

    # 落点
    if landing:
        x, y = int(landing["image_xy"][0]), int(landing["image_xy"][1])
        cv2.circle(img, (x, y), 14, (60, 60, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (60, 60, 255), -1)

    # 文字面板
    items = [(f"轨迹点 {len(track)}  候选轨迹 {result['stages'].get('track', {}).get('tracks', 0)}", (16, 14)),
             ("球场标定: 成功" if H is not None else "球场标定: 未检测到（仅轨迹）", (16, 46))]
    if landing:
        j = landing["judge"]
        items.append((f"落点 {j['x']:.2f}, {j['y']:.2f} m  →  {j['result']}"
                      f"（裕度 {j['margin_mm']:.0f}mm）", (16, 78)))
    return draw_cn(img, items, size=22)


def _render_video(frames, per_frame, track, corners, H, landing, out_dir, fps):
    if not frames:
        return None
    path = os.path.join(out_dir, "annotated.mp4")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, max(10, min(fps, 30)), (w, h))
    idx = {f for f, _, _ in track}
    trail = deque(maxlen=40)
    for i, frame in enumerate(frames):
        img = frame.copy()
        if H is not None:
            cv2.polylines(img, [np.array([[int(x), int(y)] for _, x, y in track], np.int32)],
                          False, (90, 160, 240), 1, cv2.LINE_AA)
        for _, x, y in [p for p in track if p[0] == i]:
            trail.append((int(x), int(y)))
        for k in range(1, len(trail)):
            cv2.line(img, trail[k - 1], trail[k], (0, 140, 255), 2, cv2.LINE_AA)
        if landing and landing["frame"] == i:
            cv2.circle(img, (int(landing["image_xy"][0]), int(landing["image_xy"][1])),
                       16, (60, 60, 255), 3, cv2.LINE_AA)
        if i % 5 == 0 or i in idx:
            img = draw_cn(img, [(f"frame {i}", (12, 10))], size=16, bg=False)
        vw.write(img)
    vw.release()
    return path


def _topdown_court(px_per_m: int = 60) -> np.ndarray:
    """俯视标准球场（含全部标线）。"""
    W, L = CC.COURT_WIDTH, CC.COURT_WIDTH  # placeholder, replaced below
    W, L = CC.COURT_WIDTH, CC.COURT_LENGTH
    w, h = int(W * px_per_m), int(L * px_per_m)
    img = np.full((h, w, 3), (60, 120, 75), np.uint8)   # 绿色场地（与真实球馆一致）
    white = (245, 245, 245)
    t = 3

    def L2(p, q):
        cv2.line(img, (int(p[0] * px_per_m), int(p[1] * px_per_m)),
                 (int(q[0] * px_per_m), int(q[1] * px_per_m)), white, t)

    m = 0.46                      # 单打边距
    sfl, sbl = 1.98, 0.76         # 前发球线 / 双打后发球线
    # 外框（双打边线 + 底线）
    L2((0, 0), (W, 0)); L2((W, 0), (W, L)); L2((W, L), (0, L)); L2((0, L), (0, 0))
    # 单打边线
    L2((m, 0), (m, L)); L2((W - m, 0), (W - m, L))
    # 中线（前后半场各一段）
    L2((W / 2, 0), (W / 2, L / 2 - sfl))
    L2((W / 2, L / 2 + sfl), (W / 2, L))
    # 前发球线（两侧）
    L2((0, L / 2 - sfl), (W, L / 2 - sfl))
    L2((0, L / 2 + sfl), (W, L / 2 + sfl))
    # 双打后发球线
    L2((0, sbl), (W, sbl))
    L2((0, L - sbl), (W, L - sbl))
    return img


def make_demo_video(path: str, n_frames: int = 90, size=(1280, 720)):
    """生成一段合成演示视频（透视球场 + 飞行小球），用于无素材时自测。"""
    w, h = size
    court_td = _topdown_court(px_per_m=60)
    ch, cw = court_td.shape[:2]
    # 透视 quadrilateral（近大远小）
    quad = np.array([[430, 140], [850, 140], [1030, 640], [250, 640]], np.float32)
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(court_td, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(38, 38, 38))
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), M, (w, h))
    court_img = np.full((h, w, 3), 38, np.uint8)
    court_img[mask > 0] = warped[mask > 0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 30, (w, h))
    p0, p1 = np.array([520.0, 320.0]), np.array([900.0, 560.0])
    for i in range(n_frames):
        img = court_img.copy()
        t = i / (n_frames - 1)
        pos = p0 + (p1 - p0) * t
        arc = 150 * np.sin(np.pi * t)
        c = (int(pos[0]), int(pos[1] - arc))
        cv2.circle(img, c, 5, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="单视频自动识别（球场/轨迹/落点判罚）")
    ap.add_argument("video", nargs="?", help="视频路径")
    ap.add_argument("--demo", action="store_true", help="生成合成演示视频并分析")
    ap.add_argument("--mode", default="doubles", choices=["doubles", "singles"])
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--calib", type=str, default=None,
                    help="标定文件路径（存在则复用；默认 calib_single_cam.json）")
    ap.add_argument("--reuse-calib", action="store_true",
                    help="复用已保存标定（同机位同场地时才用，可跳过检测）")
    ap.add_argument("--no-reuse-calib", action="store_true",
                    help="忽略已保存标定，强制重新检测/手动标定")
    ap.add_argument("--fuse-csv", action="store_true",
                    help="同时用 YOLO 补 TrackNet 的漏检（双源融合）")
    ap.add_argument("--ball-csv", default=None,
                    help="TrackNetV3 输出的逐帧球坐标 CSV（提供则跳过模型检测）")
    ap.add_argument("--tracker", default="refined", choices=["refined", "nn"],
                    help="轨迹追踪：refined=时序补全(默认)，nn=简单最近邻")
    ap.add_argument("--detector", default="auto", choices=["auto", "motion", "yolo"],
                    help="检测器：auto=有YOLO权重就用YOLO")
    ap.add_argument("--corners", type=str, default=None,
                    help='手动球场四角，格式 "x1,y1;x2,y2;x3,y3;x4,y4"（左上/右上/右下/左下）')
    args = ap.parse_args()

    vp = args.video
    if args.demo or not vp:
        vp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "demo_court.mp4")
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        make_demo_video(vp)
        print(f"[演示] 已生成合成视频: {vp}")
    mc = None
    if args.corners:
        mc = [[float(v) for v in pair.split(",")]
              for pair in args.corners.replace(" ", "").split(";") if pair]
        print(f"[手动标定] 四点: {mc}")
    analyze_video(vp, mode=args.mode, max_frames=args.max_frames,
                  manual_corners=mc, detector=args.detector,
                  calib_file=None if args.no_reuse_calib else (args.calib or None),
                  reuse_calib=bool(args.reuse_calib),
                  tracker=args.tracker, ball_csv=args.ball_csv,
                  fuse_csv=bool(args.fuse_csv))

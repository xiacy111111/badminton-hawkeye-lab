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
import os
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from landing_judge import CourtJudge
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
                 log: Callable[[str], None] = print) -> Tuple[Optional[np.ndarray], Dict]:
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

    idxs = np.linspace(0, max(total - 2, 0), num=min(max_samples, max(total, 1)), dtype=int)
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
        if 0.05 < score < 0.98 and score > best_score:
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

    def __init__(self, proc_width: int = 1280, area_ratio=(0.000004, 0.0012),
                 max_obj_px: int = 48, use_bg: bool = True):
        self.proc_width = proc_width
        self.area_ratio = area_ratio
        self.max_obj_px = max_obj_px
        self.use_bg = use_bg
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
            diff = cv2.absdiff(gray_s, prev_s)
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
        area_max = max(80.0, self.area_ratio[1] * hs * ws)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
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
            out.append((M["m10"] / M["m00"] / scale, M["m01"] / M["m00"] / scale, float(a)))
        return out


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


def pick_main_track(tracks, min_len: int = 6, log: Optional[Callable] = None):
    """挑主轨迹：抛物线拟合度 × 长度 × 平均速度 综合打分。"""
    if not tracks:
        return None, 0.0, 0.0
    scored = []
    for t in tracks:
        if len(t) < min_len:
            continue
        speed = 0.0
        for (f0, x0, y0), (f1, x1, y1) in zip(t, t[1:]):
            if f1 != f0:
                speed += np.hypot(x1 - x0, y1 - y0) / (f1 - f0)
        speed /= max(len(t) - 1, 1)
        r2 = parabola_fit_score(t)
        # 拟合度权重最高：真实球轨迹近抛物线，干扰轨迹不满足
        score = (0.35 + 0.65 * r2) * len(t) * (1.0 + min(speed, 200.0) / 80.0)
        scored.append((score, r2, t))
    if not scored:
        return None, 0.0, 0.0
    scored.sort(key=lambda z: z[0], reverse=True)
    best = scored[0]
    if log and len(scored) >= 3:
        log(f"  [轨迹] 候选打分前3: " + ", ".join(
            f"R²={s[1]:.2f}/长度{len(s[2])}" for s in scored[:3]))
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
def analyze_video(video_path: str, out_dir: Optional[str] = None,
                  mode: str = "doubles",
                  log: Callable[[str], None] = print,
                  make_video: bool = True,
                  max_frames: Optional[int] = None) -> Dict:
    t0 = time.time()
    video_path = str(video_path)
    if out_dir is None:
        out_dir = str(Path(video_path).parent / "hawkeye_out" / Path(video_path).stem)
    os.makedirs(out_dir, exist_ok=True)

    result: Dict = {"video": video_path, "mode": mode, "ok": False,
                    "stages": {}, "out_dir": out_dir}

    log(f"▶ 分析: {Path(video_path).name}")
    # ---- 元信息 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("✗ 无法打开视频")
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

    # ---- 1. 球场 ----
    corners, court_info = detect_court(video_path, log=log)
    H, (CW, CL) = (None, (CC.COURT_WIDTH, CC.COURT_LENGTH))
    if corners is not None:
        H, (CW, CL) = build_homography(corners, mode)
        result["stages"]["court"] = {"ok": True, **court_info,
                                     "corners": corners.astype(float).tolist()}
    else:
        result["stages"]["court"] = {"ok": False, **court_info}

    # ---- 2. 逐帧运动检测（背景建模 + 帧差，带进度） ----
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    per_frame: List[List[Tuple[float, float, float]]] = []
    keep_frames: List[np.ndarray] = []
    detector = MotionDetector()
    fi = 0
    next_report = 0.1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and fi >= max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        per_frame.append(detector.detect(prev_gray, gray))
        prev_gray = gray
        keep_frames.append(frame)
        fi += 1
        if n_total and n_total > 60:
            prog = fi / float(min(n_total, max_frames or n_total))
            if prog >= next_report:
                log(f"  [检测] 进度 {prog*100:.0f}%  ({fi} 帧)")
                next_report += 0.25
    cap.release()
    n = len(per_frame)
    n_cand = sum(len(p) for p in per_frame)
    log(f"  [检测] 处理 {n} 帧，候选点 {n_cand} 个（背景建模+帧差双路）")
    result["stages"]["detect"] = {"ok": n > 0, "frames": n, "candidates": n_cand}

    # ---- 3. 轨迹串联 + 抛物线拟合筛选 ----
    tracks = link_tracks(per_frame)
    main_track, score, r2 = pick_main_track(tracks, log=log)
    if main_track is None or len(main_track) < 4:
        log("  [轨迹] 未形成有效轨迹（可能是镜头移动或球速过快）")
        result["stages"]["track"] = {"ok": False, "tracks": len(tracks)}
        result["error"] = "no_track"
        return _finish(result, out_dir, None, log, t0)

    log(f"  [轨迹] 候选轨迹 {len(tracks)} 条，主轨迹 {len(main_track)} 点（抛物线 R²={r2:.2f}）")
    result["stages"]["track"] = {"ok": True, "tracks": len(tracks),
                                 "main_len": len(main_track),
                                 "score": round(score, 1), "parabola_r2": round(r2, 3)}

    # ---- 4. 落点：观测值 + 外推值双重估计 ----
    landing = None
    last_f, lx, ly = main_track[-1]
    if H is not None:
        judge = CourtJudge(mode=mode)
        court_pts = []
        for f, x, y in main_track:
            cx_, cy_ = img_to_court(H, (x, y))
            court_pts.append((float(f), cx_, cy_))

        obs = court_pts[-1]
        ext = extrapolate_landing(court_pts, frames_ahead=min(8, max(4, len(main_track) // 8)))
        # 主判罚用「最后观测点」：真实视频中球在落地瞬间失联，末点即落点
        use = (obs[1], obs[2])
        j = judge.judge(use[0], use[1])
        landing = {"frame": last_f,
                   "court_xy": [round(use[0], 3), round(use[1], 3)],
                   "judge": j,
                   "method": "last_observed",
                   "image_xy": [round(lx, 1), round(ly, 1)],
                   "estimated": True}
        # 外推值仅作参考（轨迹明显未走完时更接近真实落点）
        if ext is not None and -0.3 * CW <= ext[0] <= 1.3 * CW and -0.3 * CL <= ext[1] <= 1.3 * CL:
            ej = judge.judge(ext[0], ext[1])
            landing["extrapolated_court_xy"] = [round(ext[0], 3), round(ext[1], 3)]
            landing["extrapolated_judge"] = ej["result"]
            log(f"  [落点] 参考外推: ({ext[0]:.2f}, {ext[1]:.2f}) m → {ej['result']}（仅供参考）")
        log(f"  [落点] 球场坐标 ({use[0]:.2f}, {use[1]:.2f}) m → {j['result']}"
            f"（压线裕度 {j['margin_mm']} mm，末观测点估算）")
    else:
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
    args = ap.parse_args()

    vp = args.video
    if args.demo or not vp:
        vp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "demo_court.mp4")
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        make_demo_video(vp)
        print(f"[演示] 已生成合成视频: {vp}")
    analyze_video(vp, mode=args.mode, max_frames=args.max_frames)

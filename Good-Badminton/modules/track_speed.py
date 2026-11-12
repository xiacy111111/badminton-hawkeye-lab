"""
modules/track_speed.py — 轨迹与速度检测模块
============================================
职责：检测羽毛球轨迹 → 换算成**真实球速**（米/秒），输出速度曲线与标注视频。
不含判罚（判罚是独立模块，见 video_pipeline / tools/judge_from_upstream.py）。

速度怎么算（这是关键，不能拿像素速度糊弄）：
  1. 用球场四角单应矩阵把每帧的像素坐标映射到球场平面（米）；
  2. 相邻帧位移 / 时间间隔 = 瞬时速度；
  3. 三帧中值滤波去掉检测抖动带来的尖峰；
  4. 只统计球在场地附近（±2m）的帧，避免把场外误检算进速度。

用法：
    python modules/track_speed.py --video xx.mp4 --calib calib_single_cam.json
    python modules/track_speed.py --video xx.mp4 --corners "x1,y1;x2,y2;x3,y3;x4,y4"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import court_config as CC                      # noqa: E402
from video_pipeline import (                    # noqa: E402
    YoloBallDetector, MotionDetector, load_ball_csv,
    load_calib, ensure_decodable, DEFAULT_CALIB,
)
from track_refine import refine_tracks, coverage  # noqa: E402


def _draw_cn(img, text, org, size=22, color=(255, 255, 255)):
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f = ImageFont.truetype(str(PKG / "simhei.ttf"), size)
        tw = d.textlength(text, font=f)
        d.rectangle([org[0] - 6, org[1] - 4, org[0] + tw + 6, org[1] + size + 6],
                    fill=(18, 24, 32))
        d.text(org, text, font=f, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return img


def analyze_speed(video: str, out_dir: Optional[str] = None,
                  corners: Optional[np.ndarray] = None,
                  calib_file: Optional[str] = None,
                  ball_csv: Optional[str] = None,
                  detector: str = "auto",
                  max_frames: int = 300,
                  make_video: bool = True,
                  log=print) -> Dict:
    video = str(video)
    out_dir = out_dir or str(Path(video).parent / "speed_out" / Path(video).stem)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 标定（速度换算的前提）----
    if corners is None:
        corners = load_calib(calib_file or DEFAULT_CALIB)
    if corners is None:
        log("✗ 没有球场标定，无法把像素换算成米/秒。请先标定（GUI 手动四点或 --corners）")
        return {"ok": False, "error": "no_calibration"}

    W, L = CC.COURT_WIDTH, CC.COURT_LENGTH
    dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    # 画面每米对应多少像素（用于 sanity check）
    log(f"▶ 轨迹与速度：{Path(video).name}｜球场 {W:.2f}×{L:.2f} m")

    vp = ensure_decodable(video, log=log)
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        return {"ok": False, "error": "cannot_open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- 检测 ----
    if ball_csv and os.path.exists(ball_csv):
        per_frame = load_ball_csv(ball_csv, 1.0, 1.0)
        frames = []
        while len(frames) < len(per_frame):
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        src = "tracknet_csv"
    else:
        det = None
        if detector in ("auto", "yolo") and YoloBallDetector.is_available():
            det, src = YoloBallDetector(), "yolo"
            log(f"  检测器：YOLO（imgsz={det.imgsz}, conf={det.conf}）")
        else:
            det, src = MotionDetector(), "motion"
            log("  检测器：运动检测")
        per_frame, frames = [], []
        fi = 0
        while fi < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            fi += 1
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            per_frame.append(det.detect(fr, g) if src == "yolo" else det.detect(None, g))
            frames.append(fr)
        cap.release()

    n = min(len(per_frame), len(frames))
    per_frame, frames = per_frame[:n], frames[:n]
    points = sum(len(p) for p in per_frame)
    log(f"  {n} 帧，检测点 {points} 个（{src}）")

    # ---- 轨迹（时序补全）----
    tracks = refine_tracks(per_frame, (w, h))
    tracks = [[(f, x, y) for f, x, y, _ in t] for t in tracks]
    if not tracks:
        return {"ok": False, "error": "no_track", "out_dir": out_dir}

    # 选最"像飞行"的一条：抛物线拟合 + 场内占比
    best, best_score = None, -1
    for t in tracks:
        if len(t) < 8:
            continue
        xs = np.array([p[1] for p in t], float)
        ys = np.array([p[2] for p in t], float)
        if np.ptp(xs) < 10:
            continue
        try:
            coef = np.polyfit(xs, ys, 2)
            pred = np.polyval(coef, xs)
            r2 = max(0.0, 1 - float(((ys - pred) ** 2).sum() /
                                    max(((ys - ys.mean()) ** 2).sum(), 1e-9)))
        except Exception:
            r2 = 0.0
        inside = 0
        for _f, x, y in t:
            q = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), Hmat)[0, 0]
            if -2 <= q[0] <= W + 2 and -2 <= q[1] <= L + 2:
                inside += 1
        score = r2 * len(t) * (inside / len(t))
        if score > best_score:
            best, best_score = t, score
    if best is None:
        return {"ok": False, "error": "no_flight", "out_dir": out_dir}

    # ---- 速度换算（先平滑坐标，再差分）----
    xs = np.array([p[1] for p in best], float)
    ys = np.array([p[2] for p in best], float)
    fs = np.array([p[0] for p in best], float)
    court = cv2.perspectiveTransform(
        np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.float32), Hmat).reshape(-1, 2)
    cx, cy = court[:, 0].astype(float), court[:, 1].astype(float)

    def _med_smooth(v, k=5):
        out = v.copy()
        half = k // 2
        for i in range(len(v)):
            lo, hi = max(0, i - half), min(len(v), i + half + 1)
            out[i] = np.median(v[lo:hi])
        return out

    cx_s, cy_s = _med_smooth(cx), _med_smooth(cy)

    MAX_SPEED = 70.0          # 羽毛球物理上限（约 250 km/h），超过必为跳点
    samples, outliers = [], 0
    prev = None
    for i, f in enumerate(fs):
        v = None
        if prev is not None:
            dt = (f - prev[0]) / fps
            if dt > 0:
                raw = float(np.hypot(cx_s[i] - prev[1], cy_s[i] - prev[2]) / dt)
                if raw > MAX_SPEED:
                    outliers += 1
                else:
                    v = raw
        samples.append({"frame": int(f), "court_x": round(float(cx[i]), 2),
                        "court_y": round(float(cy[i]), 2),
                        "image_x": round(float(xs[i]), 1), "image_y": round(float(ys[i]), 1),
                        "speed_mps": round(v, 2) if v is not None else None})
        prev = (f, cx_s[i], cy_s[i])

    sp = [s0["speed_mps"] for s0 in samples]
    for i in range(1, len(sp) - 1):     # 再中值一次压残余尖峰
        win = [v for v in sp[i - 1:i + 2] if v is not None]
        if win:
            sp[i] = float(np.median(win))
    valid = [v for v in sp if v is not None]
    peak = max(valid) if valid else 0.0
    avg = float(np.mean(valid)) if valid else 0.0
    dur = (best[-1][0] - best[0][0]) / fps if len(best) > 1 else 0.0

    log(f"  主轨迹 {len(best)} 点｜覆盖 {coverage([(f, x, y, False) for f, x, y in best]) * 100:.0f}%"
        f"｜时长 {dur:.2f}s")
    if outliers:
        log(f"  剔除超物理上限（>{MAX_SPEED:.0f} m/s）跳点 {outliers} 个")
    log(f"  球速：平均 {avg:.1f} m/s（{avg * 3.6:.0f} km/h）"
        f"｜峰值 {peak:.1f} m/s（{peak * 3.6:.0f} km/h）")

    result = {"ok": True, "video": video, "fps": round(fps, 2), "frames": n,
              "detector": src, "flight_points": len(best), "duration_s": round(dur, 2),
              "outliers_removed": outliers,
              "speed_mps": {"avg": round(avg, 2), "peak": round(peak, 2),
                            "avg_kmh": round(avg * 3.6, 1), "peak_kmh": round(peak * 3.6, 1)},
              "samples": samples, "out_dir": out_dir}

    # ---- 标注视频 ----
    if make_video and frames:
        vw = cv2.VideoWriter(str(Path(out_dir) / "speed.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        idx_by_frame = {s["frame"]: i for i, s in enumerate(samples)}
        for i, fr in enumerate(frames):
            fi = i + 1
            if fi in idx_by_frame:
                k = idx_by_frame[fi]
                s = samples[k]
                cv2.circle(fr, (int(s["image_x"]), int(s["image_y"])), 8, (60, 220, 120), -1, cv2.LINE_AA)
                txt = f"{sp[k]:.1f} m/s" if sp[k] is not None else "—"
                fr = _draw_cn(fr, txt, (int(s["image_x"]) + 12, int(s["image_y"]) - 26), size=20,
                              color=(60, 220, 120))
            fr = _draw_cn(fr, f"轨迹 {len(best)} 点｜峰值 {peak:.1f} m/s", (16, 14), size=22)
            vw.write(fr)
        vw.release()
        result["annotated_video"] = str(Path(out_dir) / "speed.mp4")
        log(f"  输出：{out_dir}/speed.mp4、speed_report.json")

    with open(Path(out_dir) / "speed_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser(description="轨迹与速度检测模块")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calib", default=None, help="标定 JSON（默认 calib_single_cam.json）")
    ap.add_argument("--corners", default=None, help='四点 "x1,y1;x2,y2;x3,y3;x4,y4"')
    ap.add_argument("--ball-csv", default=None, help="TrackNetV3 逐帧 CSV（可选）")
    ap.add_argument("--detector", default="auto", choices=["auto", "yolo", "motion"])
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()
    corners = None
    if args.corners:
        corners = np.array([[float(v) for v in p.split(",")]
                            for p in args.corners.replace(" ", "").split(";")], np.float32)
    analyze_speed(args.video, args.out, corners, args.calib, args.ball_csv,
                  args.detector, args.max_frames)


if __name__ == "__main__":
    main()

"""
modules/pose.py — 运动员姿态检测模块
=====================================
职责：检测场上球员的人体关键点（骨架），并给出与羽毛球相关的运动指标。

支持的推理后端（可用 `--backend` 切换）：
  · yolo     Ultralytics YOLO11-pose，CPU 约 128 ms/帧（默认，实时性最好）
  · rtmpose  rtmlib RTMPose（-s 约 513 ms / -m 约 1090 ms），关键点精度更好，适合离线精算

输出：
  · <out>/pose.mp4        带骨架、ID、速度的标注视频
  · <out>/pose_metrics.json  逐球员指标（移动距离、平均/最高速度、击球手高度）+ 逐帧关键点

用法：
    python modules/pose.py --video xx.mp4 --out out_pose --backend yolo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

# COCO 17 关键点骨架（YOLO-pose / RTMPose body7 都是这个顺序）
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 0), (6, 0)]
KP_NAMES = ["nose", "eye_l", "eye_r", "ear_l", "ear_r", "sho_l", "sho_r",
            "elb_l", "elb_r", "wri_l", "wri_r", "hip_l", "hip_r",
            "kne_l", "kne_r", "ank_l", "ank_r"]

FONT = str(PKG / "simhei.ttf")


def draw_cn(img, text, org, size=22, color=(255, 255, 255), bg=True):
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f = ImageFont.truetype(FONT, size)
        if bg:
            tw = d.textlength(text, font=f)
            d.rectangle([org[0] - 6, org[1] - 4, org[0] + tw + 6, org[1] + size + 6],
                        fill=(18, 24, 32))
        d.text(org, text, font=f, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, (org[0], org[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 32.0, color, 2, cv2.LINE_AA)
        return img


# ══════════════════════════════════════════════════════════════
# 推理后端
# ══════════════════════════════════════════════════════════════
class YoloPoseBackend:
    """Ultralytics YOLO11-pose：CPU 约 128 ms/帧（imgsz=640, ROI 裁剪后）。"""

    name = "yolo-pose"

    def __init__(self, weights: Optional[str] = None, imgsz: int = 960,
                 conf: float = 0.22, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights or str(PKG / "weights" / "yolo11n-pose.pt"))
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def estimate(self, frame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回 [(kps(17,2), scores(17))]。"""
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                               device=self.device, verbose=False)[0]
        out = []
        if r.keypoints is None:
            return out
        data = r.keypoints.data.cpu().numpy()      # (n, 17, 3)
        for person in data:
            out.append((person[:, :2], person[:, 2]))
        return out


class RTMPoseBackend:
    """rtmlib RTMPose：精度更好但慢（-s 约 0.5s/帧），建议离线精算。"""

    name = "rtmpose"

    def __init__(self, mode: str = "lightweight", device: str = "cpu"):
        from rtmlib import Body
        self.body = Body(mode=mode, backend="onnxruntime", device=device)
        self.mode = mode

    def estimate(self, frame) -> List[Tuple[np.ndarray, np.ndarray]]:
        kps_all, scores_all = self.body(frame)
        return [(np.asarray(k, float), np.asarray(s, float))
                for k, s in zip(kps_all, scores_all)]


def make_backend(name: str, **kw):
    if name in ("yolo", "yolo-pose"):
        return YoloPoseBackend(**kw)
    if name in ("rtmpose", "rtmo"):
        return RTMPoseBackend(**{k: v for k, v in kw.items() if k in ("mode", "device")})
    raise ValueError(f"未知后端: {name}")


# ══════════════════════════════════════════════════════════════
# 简易多人跟踪（按质心最近邻 + 上一帧位置预测）
# ══════════════════════════════════════════════════════════════
class IdentityTracker:
    def __init__(self, max_dist: float = 160.0, max_miss: int = 12):
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self.max_dist = max_dist
        self.max_miss = max_miss

    def update(self, centroids: np.ndarray, frame_idx: int) -> List[int]:
        ids = [-1] * len(centroids)
        used = set()
        for tid, tr in list(self.tracks.items()):
            if frame_idx - tr["last"] > self.max_miss:
                del self.tracks[tid]

        for i, c in enumerate(centroids):
            best, best_d = None, self.max_dist
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = float(np.hypot(*(c - tr["pos"])))
                if d < best_d:
                    best, best_d = tid, d
            if best is not None:
                self.tracks[best].update(pos=c, last=frame_idx)
                used.add(best)
                ids[i] = best
            else:
                self.tracks[self.next_id] = {"pos": c, "last": frame_idx}
                ids[i] = self.next_id
                self.next_id += 1
        return ids


def kp_centroid(kps: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> Optional[np.ndarray]:
    """用髋中点作为人体中心（比全身平均更稳）。"""
    ok = scores[thr > 0] if False else scores >= thr
    if ok.sum() < 4:
        return None
    hip = [11, 12]
    if all(scores[h] >= thr for h in hip):
        return (kps[hip[0]] + kps[hip[1]]) / 2.0
    return kps[ok].mean(axis=0)


def draw_pose(frame, kps, scores, tid: int, kp_thr: float = 0.3):
    col = [(60, 220, 120), (80, 200, 255), (255, 170, 90)][tid % 3]
    ok = scores >= kp_thr
    if ok.sum() >= 4:
        xs, ys = kps[ok][:, 0], kps[ok][:, 1]
        x1, y1 = int(xs.min()) - 8, int(ys.min()) - 8
        x2, y2 = int(xs.max()) + 8, int(ys.max()) + 8
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
    for a, b in SKELETON:
        if scores[a] >= kp_thr and scores[b] >= kp_thr:
            p1, p2 = tuple(kps[a].astype(int)), tuple(kps[b].astype(int))
            cv2.line(frame, p1, p2, (0, 0, 0), 7, cv2.LINE_AA)      # 描边提升对比度
            cv2.line(frame, p1, p2, col, 4, cv2.LINE_AA)
    for i, (p, s) in enumerate(zip(kps, scores)):
        if s >= kp_thr:
            r = 4 if i >= 5 else 3
            cv2.circle(frame, tuple(p.astype(int)), r, (255, 255, 255), -1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def analyze_pose(video: str, out_dir: Optional[str] = None, backend: str = "yolo",
                 max_frames: int = 300, roi: Optional[List[int]] = None,
                 make_video: bool = True, log=print) -> Dict:
    video = str(video)
    out_dir = out_dir or str(Path(video).parent / "pose_out" / Path(video).stem)
    os.makedirs(out_dir, exist_ok=True)

    be = make_backend(backend)
    log(f"▶ 姿态检测：{Path(video).name}｜后端 {be.name}"
        + (f"｜ROI {roi}" if roi else ""))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"ok": False, "error": "cannot_open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(Path(out_dir) / "pose.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if make_video else None

    tracker = IdentityTracker()
    stats: Dict[int, Dict] = {}
    per_frame = []
    t0 = time.time()
    fi = 0
    while fi < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        sub = frame[roi[1]:roi[3], roi[0]:roi[2]] if roi else frame
        ox, oy = (roi[0], roi[1]) if roi else (0, 0)
        persons = be.estimate(sub)

        kps_all, cents = [], []
        for kps, sc in persons:
            kps = kps + np.array([ox, oy], float)
            c = kp_centroid(kps, sc)
            if c is None:
                continue
            kps_all.append((kps, sc))
            cents.append(c)

        ids = tracker.update(np.array(cents), fi) if cents else []
        for (kps, sc), tid, c in zip(kps_all, ids, cents):
            out = draw_pose(frame, kps, sc, tid)
            st = stats.setdefault(tid, {"frames": 0, "path_px": 0.0, "max_speed": 0.0,
                                        "pos": None, "wrist_h": []})
            if st["pos"] is not None:
                st["path_px"] += float(np.hypot(*(c - st["pos"])))
            st["pos"] = c
            st["frames"] += 1
            speed = 0.0
            if "prev_c" in st and "prev_t" in st:
                dt = (fi - st["prev_t"]) / fps
                if dt > 0:
                    speed = float(np.hypot(*(c - st["prev_c"])) / dt)
                    st["max_speed"] = max(st["max_speed"], speed)
            st["prev_c"], st["prev_t"] = c, fi
            # 持拍手（取置信度更高的手腕）高度
            if sc[9] > 0.3 or sc[10] > 0.3:
                wr = kps[9] if sc[9] >= sc[10] else kps[10]
                st["wrist_h"].append(float(wr[1]))
            _ly = int(kps[:, 1].min()) - 34 if np.isfinite(kps[:, 1]).any() else int(c[1]) - 40
            frame = draw_cn(frame, f"ID{tid}  {speed:.0f} px/s",
                            (int(c[0]) - 52, max(_ly, 18)), size=20,
                            color=(120, 240, 160))
            per_frame.append({"frame": fi, "id": int(tid),
                              "kps": [[round(float(x), 1), round(float(y), 1), round(float(s), 2)]
                                      for (x, y), s in zip(kps, sc)]})
        if vw is not None:
            vw.write(frame)

    cap.release()
    if vw is not None:
        vw.release()

    summary = []
    for tid, st in stats.items():
        if st["frames"] < 10:
            continue
        dur = st["frames"] / fps
        summary.append({"id": tid, "frames": st["frames"], "seconds": round(dur, 2),
                        "path_px": round(st["path_px"], 1),
                        "avg_speed_px_s": round(st["path_px"] / max(dur, 1e-3), 1),
                        "max_speed_px_s": round(st["max_speed"], 1),
                        "wrist_height_px": (round(float(np.mean(st["wrist_h"])), 1)
                                            if st["wrist_h"] else None)})
    summary.sort(key=lambda d: -d["frames"])

    result = {"ok": True, "video": video, "backend": be.name, "fps": round(fps, 2),
              "frames": fi, "elapsed_s": round(time.time() - t0, 1),
              "players": summary, "out_dir": out_dir}
    with open(Path(out_dir) / "pose_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**result, "keypoints": per_frame}, f, ensure_ascii=False, indent=2)

    log(f"  检出球员 {len(summary)} 人，处理 {fi} 帧，用时 {result['elapsed_s']}s"
        f"（{result['elapsed_s'] / max(fi, 1) * 1000:.0f} ms/帧）")
    for p in summary[:4]:
        log(f"    ID{p['id']}: {p['seconds']}s 在场｜平均 {p['avg_speed_px_s']} px/s"
            f"｜峰值 {p['max_speed_px_s']} px/s")
    log(f"  输出：{out_dir}/pose.mp4、pose_metrics.json")
    return result


def main():
    ap = argparse.ArgumentParser(description="运动员姿态检测模块")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "rtmpose"])
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--roi", default=None, help="x1,y1,x2,y2 只在该区域检测（加速+排除观众）")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    roi = [int(v) for v in args.roi.split(",")] if args.roi else None
    analyze_pose(args.video, args.out, args.backend, args.max_frames, roi,
                 make_video=not args.no_video)


if __name__ == "__main__":
    main()

"""
tools/judge_from_upstream.py — 用原版 Good-Badminton 的检测结果做落点判罚
==========================================================================
原版（badminton_analysis）擅长：姿态骨架、球员统计、球场透视网格、战术俯视图；
但**不含落点判罚**。本脚本把它产出的 detections.jsonl + detect_*.mp4 接上
本项目的判罚引擎（CourtJudge / 抛物线筛选 / 单应映射），输出：

  1) <输出目录>/judge_report.json   落点、界内/界外、压线裕度
  2) <输出目录>/detect_judged.mp4   在原版标注视频上叠加落点与判罚

用法：
    python tools/judge_from_upstream.py \
        --run-dir ../_debug_outputs/upstream_demo1 \
        --video ../badminton-videos-backup/demo1.mp4 \
        --mode doubles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # …\badminton-hawkeye-github
PKG = Path(__file__).resolve().parents[1] / "Good-Badminton"
if not (PKG / "landing_judge.py").exists():         # 脚本在 Good-Badminton/tools/ 下时
    PKG = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(PKG))

from landing_judge import CourtJudge          # noqa: E402
import court_config as CC                      # noqa: E402


FONT_PATH = str(Path(__file__).resolve().parents[1] / "simhei.ttf")


def draw_cn(frame, text, org, size=26, color=(60, 220, 120), bg=True):
    """PIL 画中文（cv2.putText 不支持中文，会变成 ???）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        font = ImageFont.truetype(FONT_PATH, size)
        x, y = org
        if bg:
            tw = d.textlength(text, font=font)
            d.rectangle([x - 8, y - 6, x + tw + 8, y + size + 8], fill=(20, 25, 35))
        d.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(frame, text, (org[0], org[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 34.0, color, 2, cv2.LINE_AA)
        return frame


def load_annotations(run_dir: str):
    """从 court_annotations.txt 读取四角（原版格式：corners=[(x,y), ...]）。"""
    p = Path(run_dir) / "court_annotations.txt"
    txt = p.read_text(encoding="utf-8")
    nums = [int(v) for v in "".join(ch if (ch.isdigit() or ch == ",") else " "
                                    for ch in txt.split("roi_corners")[0]).replace(",", " ").split()]
    if len(nums) < 8:
        raise SystemExit(f"无法解析四角: {p}")
    return np.array([[nums[0], nums[1]], [nums[2], nums[3]],
                     [nums[4], nums[5]], [nums[6], nums[7]]], dtype=np.float32)


def load_shuttle_track(jsonl: str):
    """读 detections.jsonl → [(frame, x, y)]（仅取有球的帧）。"""
    pts = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            sc = d.get("shuttlecock") or {}
            img = sc.get("image")
            if img and len(img) == 2:
                pts.append((int(d.get("frame", 0)), float(img[0]), float(img[1])))
    return pts


def img_to_court(H, x, y):
    q = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0]
    return float(q[0]), float(q[1])


def pick_flight(track, H, court_w, court_l, min_len=6):
    """挑出最像"一次飞行"的一段：抛物拟合度高、且大部分点落在场内。"""
    if len(track) < min_len:
        return []
    # 断成若干段（相邻帧间隔 > 6 视为断开）
    segs, cur = [], [track[0]]
    for prev, nxt in zip(track, track[1:]):
        if nxt[0] - prev[0] > 6:
            segs.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)
    segs.append(cur)

    best, best_score = [], -1.0
    for seg in segs:
        if len(seg) < min_len:
            continue
        xs = np.array([p[1] for p in seg], float)
        ys = np.array([p[2] for p in seg], float)
        if np.ptp(xs) < 8:
            continue
        r2 = 0.0
        try:
            coef = np.polyfit(xs, ys, 2)
            pred = np.polyval(coef, xs)
            ss_res = float(np.sum((ys - pred) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-9))
        except Exception:
            pass
        inside = 0
        for _f, x, y in seg:
            cx, cy = img_to_court(H, x, y)
            if -1.0 <= cx <= court_w + 1.0 and -1.5 <= cy <= court_l + 1.5:
                inside += 1
        ratio = inside / len(seg)
        score = r2 * len(seg) * (0.2 + 0.8 * ratio)
        if score > best_score:
            best, best_score = seg, score
    return best


def main():
    ap = argparse.ArgumentParser(description="用原版检测结果做落点判罚并叠加到原版视频")
    ap.add_argument("--run-dir", required=True, help="原版输出目录（含 detections.jsonl / detect_*.mp4）")
    ap.add_argument("--video", default=None, help="原始视频（仅用于确认帧率/尺寸，可省略）")
    ap.add_argument("--mode", default="doubles", choices=["doubles", "singles"])
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--draw", action="store_true", default=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    corners = load_annotations(str(run))
    track = load_shuttle_track(str(run / "detections.jsonl"))
    print(f"四角: {corners.astype(int).tolist()}")
    print(f"球检测帧数: {len(track)}（首帧 {track[0][0] if track else '-'}）")

    W, L = CC.COURT_WIDTH, CC.COURT_LENGTH
    dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst)

    flight = pick_flight(track, H, W, L)
    report = {"run_dir": str(run), "mode": args.mode,
              "corners": corners.astype(float).tolist(),
              "shuttle_points": len(track)}
    if len(flight) < 4:
        print("未能挑出可信飞行段，无法给出落点")
        report["ok"] = False
        (run / "judge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        return

    last_f, lx, ly = flight[-1]
    cx, cy = img_to_court(H, lx, ly)
    judge = CourtJudge(mode=args.mode)
    j = judge.judge(cx, cy)
    print(f"飞行段: {len(flight)} 点（帧 {flight[0][0]}→{last_f}）")
    print(f"落点: 图像({lx:.0f},{ly:.0f}) → 球场({cx:.2f},{cy:.2f}) m → {j['result']}"
          f"（压线裕度 {j['margin_mm']} mm）")
    report.update({"ok": True, "flight_len": len(flight),
                   "flight_frames": [flight[0][0], last_f],
                   "landing_image_xy": [round(lx, 1), round(ly, 1)],
                   "landing_court_xy": [round(cx, 3), round(cy, 3)],
                   "judge": j})

    # ---- 把判罚叠加到原版标注视频 ----
    video = args.out_video
    if video is None:
        cands = list(run.glob("detect_*.mp4"))
        if not cands:
            print("未找到原版标注视频，跳过叠加")
        else:
            video = str(cands[0])

    if video and Path(video).exists():
        out_path = run / "detect_judged.mp4"
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        text = f"落点 ({cx:.2f}, {cy:.2f})m  {j['result']}  裕度 {j['margin_mm']:.0f}mm"
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            # 判罚文字整段常驻（PIL 画中文）
            frame = draw_cn(frame, text, (18, h - 44), size=26,
                            color=(60, 220, 120) if j["inside"] else (80, 80, 255))
            # 落点前 1 秒开始出现红圈
            if idx >= max(1, last_f - 30):
                cv2.circle(frame, (int(lx), int(ly)), 18, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, (int(lx), int(ly)), 3, (0, 0, 255), -1)
            vw.write(frame)
        cap.release()
        vw.release()
        print(f"叠加完成: {out_path}")
        report["judged_video"] = str(out_path)

    (run / "judge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"报告: {run / 'judge_report.json'}")


if __name__ == "__main__":
    main()

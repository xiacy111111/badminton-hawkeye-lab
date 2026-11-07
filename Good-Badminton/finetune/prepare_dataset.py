"""
finetune/prepare_dataset.py — 从原始视频批量构建 YOLO 训练集（半自动标注）
==========================================================================
面对的真实问题：给西电羽球队采集的视频要变成训练集，纯手工标几千帧不现实。
做法：用现有球检测权重做「预标注」，人工只做修正 —— 通常能省 70~80% 工时。

流程：
    视频 → 抽帧（去重/去模糊） → YOLO 预标注 → 生成 YOLO 格式 labels
         → 划分 train/val → 写 data.yaml

用法：
    python finetune/prepare_dataset.py --videos D:/xtu_videos --out finetune/dataset
    python finetune/prepare_dataset.py --videos a.mp4 b.mp4 --frames-per-video 120 --conf 0.20

关键参数：
    --frames-per-video  每段视频抽多少帧（默认 100）
    --neg-ratio         无检测结果但保留为负样本的比例（默认 0.3，用于压制误检）
    --blur-thresh       清晰度下限（Laplacian 方差），过低丢弃
    --dedup             帧间直方图距离阈值，去重复帧（默认 0.985 相似即丢）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

BALL_CLASS_NAMES = ["ball"]


def list_videos(target: str):
    p = Path(target)
    if p.is_file():
        return [str(p)]
    exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".flv"}
    return [str(f) for f in sorted(p.rglob("*")) if f.suffix.lower() in exts]


def blur_score(gray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def hist_sig(frame):
    h = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None,
                     [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def hist_sim(a, b) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def decode_first_frame_ok(path: str) -> bool:
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return ok


def transcode_if_needed(path: str, log=print) -> str:
    """HEVC 等 OpenCV 解不了的视频，用 ffmpeg 转 H.264。"""
    if decode_first_frame_ok(path):
        return path
    from shutil import which
    exe = which("ffmpeg")
    if exe is None:
        log(f"  ! 无法解码且找不到 ffmpeg，跳过: {path}")
        return path
    out = str(Path(path).with_name(Path(path).stem + "_h264.mp4"))
    log(f"  · 转码中: {Path(path).name}")
    os.system(f'ffmpeg -y -v error -i "{path}" -c:v libx264 -preset veryfast '
              f'-crf 20 -pix_fmt yuv420p -an "{out}"')
    return out if os.path.exists(out) else path


def sample_frames(video: str, n: int, blur_thresh: float, dedup: float, log=print):
    """在整段视频上尽量均匀地抽 n 帧：候选加密采样 + 去模糊 + 与已接受帧去重。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        log(f"  ! 打不开: {video}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        return []

    # 候选点数 = 目标帧数 × 5，保证在经历去模糊/去重筛选后仍有足够样本
    cand_idx = np.unique(np.linspace(0, total - 1, num=min(max(n * 5, n), total),
                                     dtype=int))
    kept, accepted_sigs = [], []
    n_blur = n_dup = 0
    for idx in cand_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_score(gray) < blur_thresh:
            n_blur += 1
            continue
        sig = hist_sig(frame)
        # 与已接受的帧比较，过相似则丢弃（避免整段静态画面全是重复帧）
        if any(hist_sim(sig, prev) > dedup for prev in accepted_sigs[-8:]):
            n_dup += 1
            continue
        accepted_sigs.append(sig)
        kept.append((int(idx), frame))
        if len(kept) >= n:
            break
    cap.release()
    log(f"  · {Path(video).name}: {fps:.0f}fps/{total}帧 → 采得 {len(kept)} 帧"
        f"（模糊剔除 {n_blur}，重复剔除 {n_dup}）")
    return kept


def main():
    ap = argparse.ArgumentParser(description="从视频构建 YOLO 训练集（半自动预标注）")
    ap.add_argument("--videos", nargs="+", required=True, help="视频文件或目录")
    ap.add_argument("--out", default="finetune/dataset", help="输出数据集目录")
    ap.add_argument("--weights", default="weights/yolo11s-ball.pt", help="预标注用的球检测权重")
    ap.add_argument("--frames-per-video", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.20, help="预标注置信度阈值")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--neg-ratio", type=float, default=0.3,
                    help="无检测结果的帧保留为负样本的比例")
    ap.add_argument("--blur-thresh", type=float, default=25.0)
    ap.add_argument("--dedup", type=float, default=0.998)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    videos = []
    for v in args.videos:
        videos.extend(list_videos(v))
    if not videos:
        print("没有找到视频文件")
        return
    print(f"共 {len(videos)} 段视频待处理")

    # 预标注模型
    model = None
    if os.path.exists(args.weights):
        try:
            from ultralytics import YOLO
            model = YOLO(args.weights)
            print(f"预标注模型: {args.weights}")
        except Exception as e:
            print(f"预标注模型加载失败（{e}），将生成空标注供纯手工标注")
    else:
        print(f"未找到权重 {args.weights}，将生成空标注供纯手工标注")

    stats = {"frames": 0, "auto_labeled": 0, "negatives": 0, "skipped": 0,
             "per_video": {}}

    for vi, video in enumerate(videos):
        video = transcode_if_needed(video)
        frames = sample_frames(video, args.frames_per_video, args.blur_thresh,
                               args.dedup)
        stem = Path(video).stem

        for k, (fidx, frame) in enumerate(frames):
            boxes = []
            if model is not None:
                r = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                  verbose=False)[0]
                h, w = frame.shape[:2]
                for b in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    boxes.append((0, cx, cy, bw, bh))

            if not boxes:
                if random.random() > args.neg_ratio:
                    stats["skipped"] += 1
                    continue
                stats["negatives"] += 1

            is_val = random.random() < args.val_ratio
            split = "val" if is_val else "train"
            name = f"{stem}_{fidx:06d}"
            cv2.imwrite(str(out / f"images/{split}/{name}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(out / f"labels/{split}/{name}.txt", "w", encoding="utf-8") as f:
                for cls, cx, cy, bw, bh in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            stats["frames"] += 1
            if boxes:
                stats["auto_labeled"] += 1

        stats["per_video"][Path(video).name] = len(frames)

    # data.yaml
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, n in enumerate(BALL_CLASS_NAMES):
            f.write(f"  {i}: {n}\n")

    with open(out / "prepare_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    n_tr = len(list((out / "images/train").glob("*.jpg")))
    n_va = len(list((out / "images/val").glob("*.jpg")))
    print("\n" + "=" * 58)
    print(f"  完成：train {n_tr} 张 / val {n_va} 张")
    print(f"  有标注 {stats['auto_labeled']} 张，负样本 {stats['negatives']} 张，"
          f"跳过空白帧 {stats['skipped']} 张")
    print(f"  数据集: {out.resolve()}")
    print(f"  配置:   {yaml_path}")
    print("=" * 58)
    print("\n下一步：")
    print("  1) 人工检查/修正 labels/ 里的预标注（建议用 LabelImg 或 Label Studio）")
    print("  2) python finetune/train.py --data finetune/dataset/data.yaml")
    print("  3) python finetune/evaluate.py --weights runs/ball_xtu/weights/best.pt "
          "--data finetune/dataset/data.yaml")


if __name__ == "__main__":
    main()

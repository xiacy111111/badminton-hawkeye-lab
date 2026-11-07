"""
finetune/train.py — 在西电羽球队数据上微调球检测模型
====================================================
从 yolo11s-ball.pt 出发做小样本微调（领域自适应）：让模型适配西电球馆的
场地颜色、灯光、机位与球速，从而在自家素材上把漏检/误检压下去。

用法：
    python finetune/train.py --data finetune/dataset/data.yaml
    python finetune/train.py --data ... --epochs 120 --batch 8 --imgsz 1280 --device 0

提示：
    · 没有 GPU 也能跑，但 CPU 上极慢（约 <1 it/s），建议缩到 imgsz 640 + epochs 10 先验证流程
    · 数据量 300~800 张、冻结 backbone（--freeze 10）通常就够用，过拟合风险低
    · 西电素材与公开数据的差异主要在光照与场地色调，小学习率（默认 0.001）比默认值更稳
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="球检测模型微调（YOLO）")
    ap.add_argument("--data", required=True, help="data.yaml 路径")
    ap.add_argument("--weights", default="weights/yolo11s-ball.pt", help="起始权重")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto / cpu / 0")
    ap.add_argument("--lr0", type=float, default=0.001, help="初始学习率（领域自适应宜小）")
    ap.add_argument("--freeze", type=int, default=0, help="冻结前 N 层（10 表示冻结 backbone）")
    ap.add_argument("--name", default="ball_xtu")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--export-to", default="weights/yolo11s-ball-xtu.pt",
                    help="训练结束后把 best.pt 复制到这里，便于 video_pipeline 直接加载")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"找不到 {args.data}，先跑 finetune/prepare_dataset.py")

    from ultralytics import YOLO

    device = pick_device(args.device)
    print("=" * 60)
    print("  球检测模型微调")
    print(f"  数据:   {args.data}")
    print(f"  起始:   {args.weights}（{'存在' if os.path.exists(args.weights) else '缺失，将用 YOLO 官方预训练'}）")
    print(f"  设备:   {device}" + ("   ← CPU 会非常慢，建议先小规模验证" if device == "cpu" else ""))
    print(f"  轮数:   {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}")
    print("=" * 60)

    model = YOLO(args.weights if os.path.exists(args.weights) else "yolo11s.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        lr0=args.lr0,
        freeze=args.freeze or None,
        workers=args.workers,
        patience=args.patience,
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        # 小目标友好：不缩小输入、保留细粒度；羽毛球在白底/浅底上对比弱，用 HSV 增强
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        scale=0.3, translate=0.1, fliplr=0.0,   # 球轨迹有方向语义，不做水平翻转
        verbose=True,
    )

    # ultralytics 的保存目录以 trainer 为准，避免相对路径被改写
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", ""))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        Path(args.export_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.export_to)
        print(f"\n✔ 最佳权重已复制到: {args.export_to}")
        print("  之后 video_pipeline 会自动优先使用它（见 README「自定义权重」）")
    else:
        print(f"\n⚠ 未找到 {best}，请检查训练日志")

    try:
        m = results.results_dict
        print("\n训练完成指标：")
        for k in ("metrics/precision(B)", "metrics/recall(B)",
                  "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if k in m:
                print(f"  {k:24s} {m[k]:.4f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

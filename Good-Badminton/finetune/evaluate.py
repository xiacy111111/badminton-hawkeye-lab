"""
finetune/evaluate.py — 微调效果评估 + 与基线对比
================================================
用法：
    python finetune/evaluate.py --weights weights/yolo11s-ball-xtu.pt --data finetune/dataset/data.yaml
    python finetune/evaluate.py --compare weights/yolo11s-ball.pt weights/yolo11s-ball-xtu.pt \
                                --data finetune/dataset/data.yaml

输出：precision / recall / mAP@50 / mAP@50-95，并给出「微调前后提升」对比表。
这份表就是领飞科创/正大杯材料里最硬的一页数据。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


METRIC_KEYS = ["metrics/precision(B)", "metrics/recall(B)",
               "metrics/mAP50(B)", "metrics/mAP50-95(B)"]


def evaluate_one(weights: str, data: str, imgsz: int, device: str, split: str = "val"):
    from ultralytics import YOLO
    if not os.path.exists(weights):
        raise SystemExit(f"找不到权重: {weights}")
    model = YOLO(weights)
    res = model.val(data=data, imgsz=imgsz, device=device, split=split, verbose=False)
    out = {}
    for k, v in res.results_dict.items():
        if k.startswith("metrics/"):
            out[k] = float(v)
    return out


def fmt(d: dict) -> str:
    return ("  precision {:.4f} | recall {:.4f} | mAP50 {:.4f} | mAP50-95 {:.4f}"
            .format(d.get("metrics/precision(B)", float("nan")),
                    d.get("metrics/recall(B)", float("nan")),
                    d.get("metrics/mAP50(B)", float("nan")),
                    d.get("metrics/mAP50-95(B)", float("nan"))))


def main():
    ap = argparse.ArgumentParser(description="球检测模型评估/对比")
    ap.add_argument("--weights", default="weights/yolo11s-ball-xtu.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--compare", nargs="+", default=None,
                    help="给出多个权重，输出对比表")
    ap.add_argument("--out", default="finetune/eval_report.json")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    targets = args.compare if args.compare else [args.weights]
    report = {}
    print("=" * 66)
    print(f"  评估（{args.split} 集，imgsz={args.imgsz}, device={device}）")
    print("=" * 66)
    for w in targets:
        d = evaluate_one(w, args.data, args.imgsz, device, args.split)
        report[os.path.basename(w)] = d
        print(f"\n{os.path.basename(w)}")
        print(fmt(d))

    if len(targets) > 1:
        base = list(report.values())[0]
        print("\n" + "-" * 66)
        print("  提升对比（相对第一个权重）")
        print("-" * 66)
        print(f"  {'权重':32s} {'ΔP':>8s} {'ΔR':>8s} {'ΔmAP50':>9s} {'ΔmAP50-95':>11s}")
        for name, d in report.items():
            dp = d.get("metrics/precision(B)", 0) - base.get("metrics/precision(B)", 0)
            dr = d.get("metrics/recall(B)", 0) - base.get("metrics/recall(B)", 0)
            d50 = d.get("metrics/mAP50(B)", 0) - base.get("metrics/mAP50(B)", 0)
            d95 = d.get("metrics/mAP50-95(B)", 0) - base.get("metrics/mAP50-95(B)", 0)
            print(f"  {name:32s} {dp:+.4f} {dr:+.4f} {d50:+.4f} {d95:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {args.out}")


if __name__ == "__main__":
    main()

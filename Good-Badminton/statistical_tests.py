"""
statistical_tests.py — Wilcoxon 符号秩检验 + 效应量
====================================================
输入：benchmark_results.json（由 benchmark.py 生成）
输出：配对检验结果表（p 值、效应量、置信区间）

用法：
  python statistical_tests.py
  python statistical_tests.py --input benchmark_results.json
"""

import json, sys, os, argparse
import numpy as np
from scipy.stats import wilcoxon, norm


def cohens_d(paired_diff):
    """Cohen's d for paired samples: mean(diff) / std(diff)"""
    d = np.array(paired_diff, dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return float("nan"), float("nan")
    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    if std_d < 1e-10:
        return mean_d, 0.0
    return mean_d, mean_d / std_d


def wilcoxon_ci(paired_diff, alpha=0.05):
    """Hodges-Lehmann 估计量 + 置信区间（配对样本中位数差的稳健估计）"""
    d = np.array(paired_diff, dtype=np.float64)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 5:
        return float("nan"), (float("nan"), float("nan"))

    # Hodges-Lehmann estimator: median of all pairwise averages
    pairwise = []
    for i in range(n):
        for j in range(i, n):
            pairwise.append((d[i] + d[j]) / 2.0)
    pairwise = np.sort(pairwise)
    estimate = np.median(pairwise)

    # Confidence interval from Wilcoxon signed-rank critical values
    # For n >= 10, use normal approximation
    k = n * (n + 1) / 2
    w_alpha = k / 2 + norm.ppf(1 - alpha / 2) * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    w_alpha = min(w_alpha, k)

    # Approximate CI bounds via order statistics of pairwise averages
    m = len(pairwise)
    r = int(np.floor(k - w_alpha))
    if r < 0:
        r = 0
    if r >= m:
        r = m - 1

    lower = pairwise[r] if r < m else pairwise[0]
    upper = pairwise[m - 1 - r] if (m - 1 - r) >= 0 else pairwise[-1]

    return estimate, (lower, upper)


def run_tests(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = data.get("details", {})
    if not details:
        print("ERROR: No 'details' field found. Make sure benchmark_results.json is from benchmark.py v2.0")
        return

    methods = list(details.keys())
    n_scenarios = len(details[methods[0]])
    print(f"Loaded {n_scenarios} scenarios × {len(methods)} methods")

    # ── Extract paired per-scenario metrics ──
    metrics = {
        "pos_rmse_mm":        "Position RMSE (mm)",
        "landing_30pct_mm":   "Landing Error @30% (mm)",
        "landing_50pct_mm":   "Landing Error @50% (mm)",
        "landing_70pct_mm":   "Landing Error @70% (mm)",
        "landing_90pct_mm":   "Landing Error @90% (mm)",
        "landing_final_mm":   "Final Landing Error (mm)",
    }

    # Comparison pairs of interest
    pairs = [
        ("A. RK4+动态噪声", "D. 纯抛物线",   "H₁: RK4+Dynamic < Parabola"),
        ("A. RK4+动态噪声", "E. 线性外推",   "H₁: RK4+Dynamic < Linear"),
        ("A. RK4+动态噪声", "C. 纯卡尔曼",   "H₁: RK4+Dynamic < Kalman"),
        ("A. RK4+动态噪声", "B. RK4+固定噪声", "H₁: RK4+Dynamic < RK4+Fixed"),
        ("B. RK4+固定噪声", "D. 纯抛物线",   "H₁: RK4+Fixed < Parabola"),
        ("D. 纯抛物线",     "E. 线性外推",   "H₁: Parabola < Linear"),
    ]

    alpha = 0.05

    print("\n" + "=" * 105)
    print("  Wilcoxon Signed-Rank Test (paired, one-sided)")
    print("=" * 105)

    for metric_key, metric_name in metrics.items():
        print(f"\n{'─' * 90}")
        print(f"  Metric: {metric_name}")
        print(f"{'─' * 90}")
        print(f"  {'Comparison':<40s} {'Diff (mm)':>10s} {'Cohen d':>8s} {'p-value':>10s} {'Significant?':>12s}  {'95% CI':>18s}")
        print(f"  {'─' * 90}")

        for method_a, method_b, hypothesis in pairs:
            # Extract paired differences: A - B (so negative = A is better)
            diffs = []
            for i in range(n_scenarios):
                va = details[method_a][i].get(metric_key, float("nan"))
                vb = details[method_b][i].get(metric_key, float("nan"))
                if not np.isnan(va) and not np.isnan(vb):
                    diffs.append(va - vb)  # negative if A better

            diffs = np.array(diffs)
            n_valid = len(diffs)

            if n_valid < 5:
                print(f"  {method_a} vs {method_b}: insufficient data (n={n_valid})")
                continue

            # Wilcoxon signed-rank test (one-sided: diff < 0, i.e. A better than B)
            # scipy's wilcoxon with alternative='less' tests median < 0
            try:
                stat, p_val = wilcoxon(diffs, alternative='less', method='exact' if n_valid <= 25 else 'approx')
            except ValueError:
                # If all diffs are zero or equal
                stat, p_val = 0, 1.0

            mean_diff, d = cohens_d(diffs)
            hl_estimate, (ci_low, ci_high) = wilcoxon_ci(diffs)

            sig = "*** p<0.001" if p_val < 0.001 else ("** p<0.01" if p_val < 0.01 else ("* p<0.05" if p_val < 0.05 else "n.s."))
            ci_str = f"[{ci_low:.1f}, {ci_high:.1f}]" if not np.isnan(ci_low) else "N/A"

            print(f"  {method_a:<20s} vs {method_b:<20s} {mean_diff:8.1f}  {d:8.2f}  {p_val:8.4f}  {sig:>12s}  {ci_str:>18s}")

    # ── Summary table for paper ──
    print("\n\n" + "=" * 105)
    print("  TABLE X: Paired Wilcoxon Test Results (for inclusion in paper)")
    print("=" * 105)
    print()

    # Compact format
    header = f"  {'Comparison':<38s} {'Metric':<22s} {'Median Δ':>9s} {'p':>8s} {'Sig':>6s}"
    print(header)
    print("  " + "─" * 90)

    # Focus on the 4 key metrics for paper
    key_metrics = ["landing_30pct_mm", "landing_50pct_mm", "landing_final_mm", "pos_rmse_mm"]
    key_names = {k: v for k, v in metrics.items() if k in key_metrics}

    for metric_key in key_metrics:
        metric_name = key_names[metric_key]
        for method_a, method_b, _ in pairs[:4]:  # Only A vs B/C/D/E
            diffs = []
            for i in range(n_scenarios):
                va = details[method_a][i].get(metric_key, float("nan"))
                vb = details[method_b][i].get(metric_key, float("nan"))
                if not np.isnan(va) and not np.isnan(vb):
                    diffs.append(va - vb)
            diffs = np.array(diffs)
            if len(diffs) < 5:
                continue
            try:
                _, p_val = wilcoxon(diffs, alternative='less', method='exact' if n_valid <= 25 else 'approx')
            except ValueError:
                p_val = 1.0
            hl, _ = wilcoxon_ci(diffs)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            a_short = method_a.replace("A. ", "").replace("RK4+", "RK4-")
            b_short = method_b.replace("B. ", "").replace("C. ", "").replace("D. ", "").replace("E. ", "")
            comp_str = f"{a_short} vs {b_short}"
            print(f"  {comp_str:<38s} {metric_name:<22s} {hl:7.1f}mm  {p_val:6.4f}  {sig:>6s}")

    print()
    print("  Significance levels: *** p<0.001  ** p<0.01  * p<0.05  n.s. = not significant")
    print("  Median Δ = Hodges-Lehmann estimate of median pairwise difference (negative = A better)")
    print("=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank tests for benchmark results")
    parser.add_argument("--input", type=str, default="benchmark_results.json",
                        help="Path to benchmark_results.json")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run benchmark.py first.")
        sys.exit(1)

    run_tests(args.input)
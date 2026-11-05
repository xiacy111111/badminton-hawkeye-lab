"""
sensitivity_analysis.py — 三合一敏感性分析
============================================
1. 阻力系数 b 的敏感性
2. 相机基线距离对 3D 精度的影响（理论 + 数值）
3. 帧率对追踪精度的影响

用法：
  python sensitivity_analysis.py
  python sensitivity_analysis.py --scenarios 10 --seed 42

依赖：项目已有的 shuttlecock_physics.py, benchmark.py
"""

import json, os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器无头模式；本地可注释掉
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shuttlecock_physics import ShuttlecockPhysics, PhysicsParams
# 复用 benchmark.py 中的 tracker 类
from benchmark import (
    TrajectoryGenerator, DynamicNoiseModel,
    TrackerRK4Dynamic, TrackerRK4Fixed,
    TrackerKalmanOnly, TrackerParabola, TrackerLinear,
    simulate_measurements, evaluate_tracker,
)


# ═══════════════════════════════════════
# Analysis 1: Drag Coefficient Sensitivity
# ═══════════════════════════════════════

def analyze_b_sensitivity(physics, num_scenarios=10, seed=42):
    """
    对阻力系数 b 进行扫描，测量落点预测误差的变化。
    Ground truth 始终用 b=0.0014 生成；tracker 使用变化的 b_tracker。
    """
    np.random.seed(seed)
    
    b_values = [0.0001, 0.0005, 0.0010, 0.0014, 0.0020, 0.0030, 0.0050, 0.0100]
    b_labels = ['0.0001', '0.0005', '0.0010', '0.0014\n(true)', '0.0020', '0.0030', '0.0050', '0.0100']
    
    # 生成测试场景（侧重高远球——空气阻力效应最显著）
    gt_gen = TrajectoryGenerator(physics)  # b=0.0014
    
    scenarios = []
    for _ in range(num_scenarios):
        # 高远球占 60%，其余均匀
        if np.random.random() < 0.6:
            pos0 = [np.random.uniform(2.5, 3.8), np.random.uniform(0.3, 1.0), np.random.uniform(1.5, 2.5)]
            vel0 = [np.random.uniform(-2, 2), np.random.uniform(3, 7), np.random.uniform(6, 10)]
        else:
            pos0 = [np.random.uniform(2.0, 4.0), np.random.uniform(0.5, 7.0), np.random.uniform(1.5, 2.8)]
            vel0 = [np.random.uniform(-3, 3), np.random.uniform(1, 9), np.random.uniform(-6, 2)]
        scenarios.append((np.array(pos0), np.array(vel0)))
    
    noise_model = DynamicNoiseModel()
    
    # 对每个 b 值，运行完整追踪
    b_results = {b: {'l30': [], 'l50': [], 'l70': [], 'lf': [], 'pos_rmse': []} for b in b_values}
    
    for i, (pos0, vel0) in enumerate(scenarios):
        gt_traj = gt_gen.generate(pos0, vel0)
        gt_final_xy = np.array([gt_traj[-1][1], gt_traj[-1][2]])
        measurements = simulate_measurements(gt_traj, noise_model)
        total_frames = len(measurements)
        
        for b_val in b_values:
            # 创建一个使用 b_val 的临时 physics
            p_tmp = PhysicsParams()
            p_tmp.b = b_val
            phys_tmp = ShuttlecockPhysics(p_tmp)
            tracker = TrackerRK4Dynamic(phys_tmp)
            
            tracker.reset()
            lpreds = []
            pos_errs = []
            
            for fi, (t, true_3d, measured) in enumerate(measurements):
                filtered, pred_landing = tracker.update(measured, t)
                if filtered is not None:
                    pos_errs.append(float(np.linalg.norm(filtered - true_3d)))
                frac = fi / max(total_frames - 1, 1)
                if pred_landing is not None:
                    lpreds.append((frac, pred_landing[:2].copy()))
            
            if lpreds:
                for target, key in [(0.30, 'l30'), (0.50, 'l50'), (0.70, 'l70')]:
                    best = min(lpreds, key=lambda x: abs(x[0] - target))
                    if abs(best[0] - target) < 0.06:
                        b_results[b_val][key].append(
                            float(np.linalg.norm(best[1] - gt_final_xy)) * 1000)
                b_results[b_val]['lf'].append(
                    float(np.linalg.norm(lpreds[-1][1] - gt_final_xy)) * 1000)
            
            if pos_errs:
                b_results[b_val]['pos_rmse'].append(
                    np.sqrt(np.mean(np.array(pos_errs)**2)) * 1000)
        
        if (i + 1) % 3 == 0:
            print(f"  b-sensitivity: {i+1}/{num_scenarios} scenarios done")
    
    # 汇总
    summary = {}
    for b_val in b_values:
        summary[b_val] = {
            key: (np.mean(vals) if vals else float('nan'))
            for key, vals in b_results[b_val].items()
        }
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # (a) 分组柱状图
    ax = axes[0]
    bx = np.arange(len(b_values))
    w = 0.2
    stage_colors = {'l30': '#ea4335', 'l50': '#f9ab00', 'l70': '#34a853', 'lf': '#1a73e8'}
    stage_labels = {'l30': '30%', 'l50': '50%', 'l70': '70%', 'lf': 'Final'}
    
    for i, (sk, slabel) in enumerate(stage_labels.items()):
        means = [summary[b][sk] for b in b_values]
        offset = (i - 1.5) * w
        bars = ax.bar(bx + offset, means, w, label=slabel, color=stage_colors[sk],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, means):
            if not np.isnan(val) and val < 500:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=6,
                        fontweight='bold', rotation=90)
    
    ax.set_xticks(bx)
    ax.set_xticklabels(b_labels, fontsize=8.5)
    ax.set_ylabel('Landing Error (mm)', fontweight='bold')
    ax.set_title('(a) Landing Error vs Drag Coefficient b', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axvline(x=3, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # (b) 最优性检验
    ax = axes[1]
    lf_means = [summary[b]['lf'] for b in b_values]
    ax.plot(b_values, lf_means, 'o-', color='#1a73e8', linewidth=2, markersize=8,
            markerfacecolor='white', markeredgewidth=2)
    ax.axvline(x=0.0014, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Drag Coefficient b (kg/m)', fontweight='bold')
    ax.set_ylabel('Final Landing Error (mm)', fontweight='bold')
    ax.set_title('(b) Optimality — Error Minimized at True b', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标注最优点
    min_idx = np.nanargmin(lf_means)
    ax.annotate(f'b*={b_values[min_idx]}\n(err={lf_means[min_idx]:.0f}mm)',
                xy=(b_values[min_idx], lf_means[min_idx]),
                xytext=(0.003, lf_means[min_idx] + 20),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='#fce8e6', edgecolor='red', alpha=0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    fig.suptitle('Figure S1: Sensitivity to Drag Coefficient b', fontweight='bold',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_s1_b_sensitivity.png', dpi=300)
    plt.show()
    
    print("\nb-Sensitivity Summary:")
    for b_val in b_values:
        s = summary[b_val]
        print(f"  b={b_val:.4f}: pos_rmse={s['pos_rmse']:.1f}mm, "
              f"l30={s['l30']:.0f}mm, l50={s['l50']:.0f}mm, lf={s['lf']:.0f}mm")
    
    return summary


# ═══════════════════════════════════════
# Analysis 2: Baseline Distance Sensitivity
# ═══════════════════════════════════════

def analyze_baseline_sensitivity(f_px=1740.0, sigma_pixel=0.5):
    """
    理论分析：基线距离对 3D 三角测量精度的影响。
    sigma_3D = Z² / (f · B) · sigma_pixel
    """
    baselines = np.linspace(0.3, 20.0, 200)
    Z_values = [3.0, 6.7, 10.0, 13.4]
    Z_labels = ['Z=3m (near net)', 'Z=6.7m (mid-court)',
                'Z=10m (far)', 'Z=13.4m (full court)']
    colors = ['#1a73e8', '#34a853', '#f9ab00', '#ea4335']
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    for Z, label, c in zip(Z_values, Z_labels, colors):
        sigma_3d = Z**2 / (f_px * baselines) * sigma_pixel * 1000  # mm
        ax.plot(baselines, sigma_3d, linewidth=2, label=label, color=c)
        # 标注我们的方案
        idx = np.argmin(np.abs(baselines - 13.4))
        ax.plot(13.4, sigma_3d[idx], 'D', color=c, markersize=8,
                markeredgecolor='black', markeredgewidth=1)
        # 标注小基线
        idx2 = np.argmin(np.abs(baselines - 0.5))
        ax.plot(0.5, sigma_3d[idx2], 's', color=c, markersize=7, alpha=0.7)
    
    ax.axvline(x=13.4, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(13.7, 80, 'Our setup\nB=13.4m', fontsize=9, fontweight='bold')
    ax.text(0.3, 60, 'Small baseline\nB=0.5m', fontsize=8, color='gray')
    
    # 关键对比注释
    sigma_small = 6.7**2 / (f_px * 0.5) * sigma_pixel * 1000
    sigma_large = 6.7**2 / (f_px * 13.4) * sigma_pixel * 1000
    ax.annotate(
        f'At mid-court (Z=6.7m):\n'
        f'B=0.5m → σ={sigma_small:.1f}mm\n'
        f'B=13.4m → σ={sigma_large:.1f}mm\n'
        f'{sigma_small/sigma_large:.0f}× improvement',
        xy=(5, 30), xytext=(5, 250), fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='#e8f0fe', edgecolor='#1a73e8', alpha=0.95),
        arrowprops=dict(arrowstyle='->', color='#1a73e8', lw=2))
    
    ax.axvspan(0, 2, alpha=0.06, color='red')
    ax.text(1, 200, 'UNSTABLE', ha='center', fontsize=10, fontweight='bold',
            color='red', alpha=0.7)
    ax.axvspan(10, 20, alpha=0.06, color='green')
    ax.text(15, 200, 'STABLE', ha='center', fontsize=10, fontweight='bold',
            color='green', alpha=0.7)
    
    ax.set_xlabel('Baseline Distance B (m)', fontweight='bold', fontsize=12)
    ax.set_ylabel('3D Position Uncertainty σ (mm)', fontweight='bold', fontsize=12)
    ax.set_title('Figure S2: Triangulation Uncertainty vs Baseline Distance',
                 fontweight='bold', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 20)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 500)
    
    plt.tight_layout()
    plt.savefig('fig_s2_baseline.png', dpi=300)
    plt.show()
    
    # 打印汇总
    print("\nBaseline Sensitivity Summary:")
    print(f"  B=0.5m (small):  σ_3D={sigma_small:.1f}mm @mid-court")
    print(f"  B=13.4m (large): σ_3D={sigma_large:.1f}mm @mid-court")
    print(f"  Improvement factor: {sigma_small/sigma_large:.1f}x")
    for Z in Z_values:
        s_small = Z**2 / (f_px * 0.5) * sigma_pixel * 1000
        s_large = Z**2 / (f_px * 13.4) * sigma_pixel * 1000
        print(f"  Z={Z}m: {s_small:.1f}mm → {s_large:.1f}mm ({s_small/s_large:.1f}x)")


# ═══════════════════════════════════════
# Analysis 3: Frame Rate Sensitivity
# ═══════════════════════════════════════

def analyze_fps_sensitivity(physics, num_scenarios=8, seed=42):
    """
    对测量帧率进行降采样，测量追踪精度的衰减。
    """
    np.random.seed(seed)
    gt_gen = TrajectoryGenerator(physics)
    noise_model = DynamicNoiseModel()
    
    # 生成场景
    scenarios = []
    for _ in range(num_scenarios):
        if np.random.random() < 0.5:
            pos0 = [np.random.uniform(2.5, 3.8), np.random.uniform(0.3, 1.0),
                    np.random.uniform(1.5, 2.5)]
            vel0 = [np.random.uniform(-2, 2), np.random.uniform(3, 7),
                    np.random.uniform(6, 10)]
        else:
            pos0 = [np.random.uniform(2.0, 4.0), np.random.uniform(0.5, 7.0),
                    np.random.uniform(1.5, 2.8)]
            vel0 = [np.random.uniform(-3, 3), np.random.uniform(1, 9),
                    np.random.uniform(-6, 2)]
        scenarios.append((np.array(pos0), np.array(vel0)))
    
    fps_values = [10, 15, 20, 30, 60, 100]
    
    print("\nFrame Rate Sensitivity:")
    print(f"{'FPS':>8s}  {'Pos RMSE':>10s}  {'Land@50%':>10s}  {'Final Land':>10s}")
    print("-" * 50)
    
    fps_results = {'fps': [], 'pos_rmse': [], 'l50': [], 'lf': []}
    
    for fps in fps_values:
        skip = max(1, int(100 / fps))
        eff_fps = 100.0 / skip
        
        all_pos_err = []
        all_l50 = []
        all_lf = []
        
        for pos0, vel0 in scenarios:
            gt_traj = gt_gen.generate(pos0, vel0)
            gt_final_xy = np.array([gt_traj[-1][1], gt_traj[-1][2]])
            measurements_full = simulate_measurements(gt_traj, noise_model)
            measurements_ds = measurements_full[::skip]
            total_ds = len(measurements_ds)
            
            tracker = TrackerRK4Dynamic(physics)
            tracker.reset()
            lpreds = []
            
            for fi, (t, true_3d, measured) in enumerate(measurements_ds):
                filtered, pred_landing = tracker.update(measured, t)
                if filtered is not None:
                    all_pos_err.append(float(np.linalg.norm(filtered - true_3d)) * 1000)
                frac = fi / max(total_ds - 1, 1)
                if pred_landing is not None:
                    lpreds.append((frac, pred_landing[:2].copy()))
            
            if lpreds:
                best50 = min(lpreds, key=lambda x: abs(x[0] - 0.50))
                if abs(best50[0] - 0.50) < 0.10:
                    all_l50.append(float(np.linalg.norm(best50[1] - gt_final_xy)) * 1000)
                all_lf.append(float(np.linalg.norm(lpreds[-1][1] - gt_final_xy)) * 1000)
        
        pos_rmse = np.sqrt(np.mean(np.array(all_pos_err)**2)) if all_pos_err else float('nan')
        l50_mean = np.mean(all_l50) if all_l50 else float('nan')
        lf_mean = np.mean(all_lf) if all_lf else float('nan')
        
        fps_results['fps'].append(eff_fps)
        fps_results['pos_rmse'].append(pos_rmse)
        fps_results['l50'].append(l50_mean)
        fps_results['lf'].append(lf_mean)
        
        print(f"  {eff_fps:4.0f}   {pos_rmse:8.1f}mm  {l50_mean:8.0f}mm  {lf_mean:8.0f}mm")
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(fps_results['fps'], fps_results['pos_rmse'], 'o-',
            color='#1a73e8', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2)
    for f, r in zip(fps_results['fps'], fps_results['pos_rmse']):
        ax.text(f, r + 0.5, f'{r:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Effective Frame Rate (fps)', fontweight='bold')
    ax.set_ylabel('Position RMSE (mm)', fontweight='bold')
    ax.set_title('(a) Position Filtering vs Frame Rate', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax = axes[1]
    ax.plot(fps_results['fps'], fps_results['l50'], 's-',
            color='#f9ab00', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2, label='Landing@50%')
    ax.plot(fps_results['fps'], fps_results['lf'], 'D-',
            color='#34a853', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2, label='Final Landing')
    for f, l50, lf in zip(fps_results['fps'], fps_results['l50'], fps_results['lf']):
        ax.text(f, l50 + 5, f'{l50:.0f}', ha='center', fontsize=8,
                fontweight='bold', color='#f9ab00')
        ax.text(f, lf + 3, f'{lf:.0f}', ha='center', fontsize=8,
                fontweight='bold', color='#34a853')
    ax.set_xlabel('Effective Frame Rate (fps)', fontweight='bold')
    ax.set_ylabel('Landing Error (mm)', fontweight='bold')
    ax.set_title('(b) Landing Prediction vs Frame Rate', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle('Figure S3: Sensitivity to Camera Frame Rate',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('fig_s3_fps.png', dpi=300)
    plt.show()
    
    return fps_results


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensitivity Analysis")
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-b", action="store_true", help="Skip b-sensitivity (slow)")
    parser.add_argument("--skip-fps", action="store_true", help="Skip fps analysis")
    args = parser.parse_args()
    
    physics = ShuttlecockPhysics(PhysicsParams())
    
    print("=" * 60)
    print("  SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Analysis 2: Baseline (fast, pure math)
    print("\n>>> Analysis 2: Baseline Distance Sensitivity")
    analyze_baseline_sensitivity()
    
    # Analysis 1: b-sensitivity (slow)
    if not args.skip_b:
        print("\n>>> Analysis 1: Drag Coefficient Sensitivity")
        b_summary = analyze_b_sensitivity(physics, args.scenarios, args.seed)
    else:
        print("\n>>> Analysis 1: SKIPPED (use --skip-b to skip)")
    
    # Analysis 3: Frame rate (medium)
    if not args.skip_fps:
        print("\n>>> Analysis 3: Frame Rate Sensitivity")
        fps_results = analyze_fps_sensitivity(physics, args.scenarios, args.seed)
    else:
        print("\n>>> Analysis 3: SKIPPED (use --skip-fps to skip)")
    
    print("\n" + "=" * 60)
    print("  All analyses complete. Figures saved.")
    print("=" * 60)
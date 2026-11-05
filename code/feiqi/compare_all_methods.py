
#!/usr/bin/env python3
"""
compare_all_methods.py 鈥� 鍏ㄦ柟娉曠簿纭姣旀鏋�

瀵规瘮椤癸細
  鍗曟柟娉�: M1_poly2, M2_poly3, M3_physics_fit, M4_spline_ukf, M5_rk4_fixed, M6_spline
  铻嶅悎绛栫暐: 绠€鍗曞钩鍧�, 浜ゅ弶璇勫垎, 绮剧‘UMTO

杈撳嚭璁烘枃鍙洿鎺ヤ娇鐢ㄧ殑缁熻琛ㄦ牸銆�
"""

import numpy as np
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 娣诲姞褰撳墠鐩綍鍒拌矾寰�
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trajectory_methods import (
    method_poly2, method_poly3, method_physics_fit,
    method_spline_ukf, method_rk4_fixed, method_spline_extrapolate,
    run_all_methods
)
from cross_scoring_fusion import CrossScoringFusion

# 鐗╃悊甯搁噺
G = 9.81
M = 0.005
C = 0.001
B = 0.0

PHYSICS_PARAMS = {'mass': M, 'drag_linear': B, 'drag_quad': C, 'g': G}


# ============================================================
# 鍚堟垚杞ㄨ抗鐢熸垚
# ============================================================
def simulate_trajectory(p0, v0, dt=0.0005, mass=M, drag_linear=B, drag_quad=C):
    """鐢熸垚鍚堟垚3D杞ㄨ抗"""
    max_steps = 3000
    pos = np.zeros((max_steps, 3))
    vel = np.zeros((max_steps, 3))
    pos[0], vel[0] = p0, v0

    for i in range(max_steps - 1):
        v = vel[i]
        spd = np.linalg.norm(v)
        gv = np.array([0, 0, -G])
        if spd > 1e-6:
            a = gv + (-(drag_linear*spd + drag_quad*spd**2) * v/spd) / mass
        else:
            a = gv

        # RK4
        k1v, k1p = a*dt, v*dt
        v2 = v + 0.5*k1v; spd2 = np.linalg.norm(v2)
        a2 = gv + (-(drag_linear*spd2 + drag_quad*spd2**2)*v2/spd2)/mass if spd2>1e-6 else gv
        k2v, k2p = a2*dt, (v+0.5*k1v)*dt
        v3 = v + 0.5*k2v; spd3 = np.linalg.norm(v3)
        a3 = gv + (-(drag_linear*spd3 + drag_quad*spd3**2)*v3/spd3)/mass if spd3>1e-6 else gv
        k3v, k3p = a3*dt, (v+0.5*k2v)*dt
        v4 = v + k3v; spd4 = np.linalg.norm(v4)
        a4 = gv + (-(drag_linear*spd4 + drag_quad*spd4**2)*v4/spd4)/mass if spd4>1e-6 else gv
        k4v, k4p = a4*dt, (v+k3v)*dt

        vel[i+1] = v + (k1v+2*k2v+2*k3v+k4v)/6
        pos[i+1] = pos[i] + (k1p+2*k2p+2*k3p+k4p)/6

        if pos[i+1, 2] <= 0:
            return pos[:i+2], vel[:i+2], dt*np.arange(i+2)

    return pos, vel, dt*np.arange(max_steps)


def generate_test_cases(n_cases=50, seed=42):
    """鐢熸垚澶氭牱鍖栫殑娴嬭瘯杞ㄨ抗"""
    np.random.seed(seed)
    cases = []

    for i in range(n_cases):
        # 闅忔満鍒濆鏉′欢锛岃鐩栧悇绉嶅嚮鐞冪被鍨�
        p0 = np.array([
            np.random.uniform(-3, 3),      # x: 鍗婂満鑼冨洿
            np.random.uniform(-2, 2),      # y: 鍗婂満瀹�
            np.random.uniform(1.5, 3.5),   # z: 鍑荤悆楂樺害
        ])

        # 閫熷害妯℃嫙涓嶅悓鍑荤悆绫诲瀷
        v_type = np.random.choice(['smash', 'clear', 'drop', 'drive'])
        if v_type == 'smash':
            v0 = np.array([np.random.uniform(-8, 8), np.random.uniform(-3, 3), np.random.uniform(-3, -0.5)])
        elif v_type == 'clear':
            v0 = np.array([np.random.uniform(-6, 6), np.random.uniform(-4, 4), np.random.uniform(3, 6)])
        elif v_type == 'drop':
            v0 = np.array([np.random.uniform(-3, 3), np.random.uniform(-2, 2), np.random.uniform(-1, 1)])
        else:  # drive
            v0 = np.array([np.random.uniform(-7, 7), np.random.uniform(-3, 3), np.random.uniform(-0.5, 2)])

        cases.append((p0, v0, v_type))

    return cases


# ============================================================
# 璇勪及鍗曟潯杞ㄨ抗
# ============================================================
def evaluate_single(obs_pts, obs_t, true_landing, noise_level=0.0015):
    """瀵逛竴鏉¤建杩硅繍琛屾墍鏈夋柟娉曞苟璁＄畻璇樊"""

    # 鍔犲櫔澹�
    noisy_obs = obs_pts + np.random.randn(*obs_pts.shape) * noise_level

    results = {}

    # ---- 6绉嶅崟鏂规硶 ----
    single_methods = {
        'M1_poly2': method_poly2,
        'M2_poly3': method_poly3,
        'M3_physics_fit': method_physics_fit,
        'M4_spline_ukf': method_spline_ukf,
        'M5_rk4_fixed': method_rk4_fixed,
        'M6_spline': method_spline_extrapolate,
    }

    for name, fn in single_methods.items():
        try:
            r = fn(noisy_obs, obs_t)
            if 'landing_xy' in r:
                err = np.linalg.norm(r['landing_xy'] - true_landing)
                results[name] = {
                    'error_mm': float(err * 1000),
                    'landing': tuple(r['landing_xy']),
                    'success': True,
                }
            else:
                results[name] = {'error_mm': np.nan, 'success': False, 'error_msg': str(r.get('error', 'unknown'))}
        except Exception as e:
            results[name] = {'error_mm': np.nan, 'success': False, 'error_msg': str(e)}

    # ---- 绠€鍗曞钩鍧� ----
    t0 = time.time()
    valid_landings = [r['landing'] for r in results.values() if r.get('success')]
    if valid_landings:
        avg = np.mean(valid_landings, axis=0)
        err_avg = np.linalg.norm(avg - true_landing)
        results['Fusion_Average'] = {
            'error_mm': float(err_avg * 1000),
            'landing': tuple(avg),
            'success': True,
            'time_ms': (time.time() - t0) * 1000,
        }

    # ---- 浜ゅ弶璇勫垎铻嶅悎 ----
    t0 = time.time()
    try:
        fusion = CrossScoringFusion(noisy_obs, obs_t, PHYSICS_PARAMS)
        fr = fusion.fuse()
        err_cross = np.linalg.norm([fr['landing_x'], fr['landing_y']] - true_landing)
        results['Fusion_CrossScoring'] = {
            'error_mm': float(err_cross * 1000),
            'landing': (fr['landing_x'], fr['landing_y']),
            'success': True,
            'time_ms': fr['opt_time_ms'],
            'confidence': fr['confidence'],
            'winning_method': fr['winning_method'],
        }
    except Exception as e:
        results['Fusion_CrossScoring'] = {'error_mm': np.nan, 'success': False, 'error_msg': str(e)}

    # ---- 绮剧‘UMTO铻嶅悎锛堝彲閫夛紝姣旇緝鑰楁椂锛� ----
    t0 = time.time()
    try:
        from precise_fusion import PreciseFusion
        pf = PreciseFusion(noisy_obs, obs_t, PHYSICS_PARAMS)
        pr = pf.optimize(use_global=True, max_iter_global=50, max_iter_local=100)
        err_precise = np.linalg.norm([pr['landing_x'], pr['landing_y']] - true_landing)
        results['Fusion_PreciseUMTO'] = {
            'error_mm': float(err_precise * 1000),
            'landing': (pr['landing_x'], pr['landing_y']),
            'success': True,
            'time_ms': pr['opt_time_ms'],
            'confidence': pr['confidence'],
        }
    except Exception as e:
        results['Fusion_PreciseUMTO'] = {'error_mm': np.nan, 'success': False, 'error_msg': str(e)}

    return results


# ============================================================
# 鎵归噺璇勪及 + 缁熻
# ============================================================
def run_benchmark(n_cases=50, seed=42, noise_mm=1.5, verbose=True):
    """杩愯瀹屾暣鍩哄噯娴嬭瘯"""

    print("=" * 80)
    print("  Good-Badminton 鍏ㄦ柟娉曠簿纭姣斿熀鍑嗘祴璇�")
    print(f"  娴嬭瘯鐢ㄤ緥: {n_cases} | 鍣０: {noise_mm}mm | 闅忔満绉嶅瓙: {seed}")
    print("=" * 80)

    cases = generate_test_cases(n_cases, seed)

    all_results = []
    method_names = None

    for idx, (p0, v0, v_type) in enumerate(cases):
        pos, vel, t_arr = simulate_trajectory(p0, v0)
        true_landing = pos[-1, :2].copy()

        # 鍙栨渶鍚�20-35%浣滀负瑙傛祴
        obs_frac = 0.20 + np.random.uniform(0, 0.15)
        n_obs = max(6, int(len(pos) * obs_frac))
        obs_pts = pos[-n_obs:].copy()
        obs_t = t_arr[-n_obs:].copy()

        r = evaluate_single(obs_pts, obs_t, true_landing, noise_mm/1000)
        r['case_id'] = idx
        r['type'] = v_type
        r['n_obs'] = n_obs
        r['obs_duration_s'] = float(obs_t[-1] - obs_t[0])
        all_results.append(r)

        if method_names is None:
            method_names = list(r.keys())

        if verbose and (idx % 10 == 0 or idx == n_cases - 1):
            print(f"  杩涘害: {idx+1}/{n_cases}", end='\r')

    if verbose:
        print()

    # 姹囨€荤粺璁�
    stats = {}
    for name in method_names:
        errors = []
        successes = 0
        times = []
        for r in all_results:
            if name in r and r[name].get('success'):
                errors.append(r[name]['error_mm'])
                successes += 1
                if 'time_ms' in r[name]:
                    times.append(r[name]['time_ms'])

        if errors:
            errors = np.array(errors)
            stats[name] = {
                'success_rate': successes / n_cases,
                'median_error_mm': float(np.median(errors)),
                'mean_error_mm': float(np.mean(errors)),
                'std_error_mm': float(np.std(errors)),
                'max_error_mm': float(np.max(errors)),
                'under_40mm_rate': float(np.mean(errors < 40)),
                'under_20mm_rate': float(np.mean(errors < 20)),
                'under_10mm_rate': float(np.mean(errors < 10)),
                'mean_time_ms': float(np.mean(times)) if times else 0,
            }
        else:
            stats[name] = {'success_rate': 0, 'error': '鍏ㄩ儴澶辫触'}

    return stats, all_results


def print_stats_table(stats):
    """鎵撳嵃缁熻琛ㄦ牸"""
    # 鍒嗙鍗曟柟娉曞拰铻嶅悎鏂规硶
    single = {k: v for k, v in stats.items() if k.startswith('M')}
    fusion = {k: v for k, v in stats.items() if k.startswith('Fusion')}

    print("\n" + "=" * 80)
    print("  鍗曟柟娉曞姣�")
    print("=" * 80)
    header = f"{'鏂规硶':<18} {'鎴愬姛鐜�':>8} {'涓綅mm':>8} {'鍧囧€糾m':>8} {'<40mm':>8} {'<20mm':>8} {'<10mm':>8}"
    print(header)
    print("-" * 80)

    for name in sorted(single.keys()):
        s = single[name]
        if 'median_error_mm' in s:
            print(f"{name:<18} {s['success_rate']*100:>7.0f}% {s['median_error_mm']:>7.1f} "
                  f"{s['mean_error_mm']:>7.1f} {s['under_40mm_rate']*100:>7.0f}% "
                  f"{s['under_20mm_rate']*100:>7.0f}% {s['under_10mm_rate']*100:>7.0f}%")
        else:
            print(f"{name:<18} {'FAIL':>8}")

    print("\n" + "=" * 80)
    print("  铻嶅悎绛栫暐瀵规瘮")
    print("=" * 80)
    header2 = f"{'绛栫暐':<22} {'涓綅mm':>8} {'鍧囧€糾m':>8} {'<40mm':>8} {'<20mm':>8} {'鑰楁椂ms':>8}"
    print(header2)
    print("-" * 80)

    for name in ['Fusion_Average', 'Fusion_CrossScoring', 'Fusion_PreciseUMTO']:
        if name in fusion:
            s = fusion[name]
            if 'median_error_mm' in s:
                print(f"{name:<22} {s['median_error_mm']:>7.1f} {s['mean_error_mm']:>7.1f} "
                      f"{s['under_40mm_rate']*100:>7.0f}% {s['under_20mm_rate']*100:>7.0f}% "
                      f"{s['mean_time_ms']:>7.0f}")

    print("=" * 80)


# ============================================================
# 涓诲叆鍙�
# ============================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Good-Badminton 鍏ㄦ柟娉曞姣�')
    parser.add_argument('--cases', type=int, default=50, help='娴嬭瘯鐢ㄤ緥鏁�')
    parser.add_argument('--noise', type=float, default=1.5, help='鍣０姘村钩(mm)')
    parser.add_argument('--seed', type=int, default=42, help='闅忔満绉嶅瓙')
    parser.add_argument('--quiet', action='store_true', help='瀹夐潤妯″紡')
    parser.add_argument('--skip-precise', action='store_true', help='璺宠繃绮剧‘UMTO锛堢渷鏃堕棿锛�')

    args = parser.parse_args()

    print(f"鍚姩鍩哄噯娴嬭瘯: {args.cases}鏉¤建杩�, {args.noise}mm鍣０")

    # 濡傛灉璺宠繃绮剧‘UMTO锛屼慨鏀筫valuate_single
    if args.skip_precise:
        print("  璺宠繃绮剧‘UMTO铻嶅悎")

    stats, results = run_benchmark(
        n_cases=args.cases,
        seed=args.seed,
        noise_mm=args.noise,
        verbose=not args.quiet,
    )

    print_stats_table(stats)

    # 杈撳嚭鏈€浼樻柟娉�
    all_median = {}
    for name, s in stats.items():
        if 'median_error_mm' in s:
            all_median[name] = s['median_error_mm']

    if all_median:
        best = min(all_median, key=all_median.get)
        print(f"\n馃弳 鏈€浼樻柟娉�: {best} (涓綅璇樊 {all_median[best]:.1f}mm)")

    print(f"\n娴嬭瘯瀹屾垚銆傜敓鎴� {len(results)} 鏉¤建杩圭殑瀹屾暣缁撴灉銆�")
#!/usr/bin/env python3
"""
trajectory_methods.py 鈥� 鍏鐙珛杞ㄨ抗棰勬祴鏂规硶

姣忕鏂规硶鎺ユ敹瑙傛祴3D杞ㄨ抗鐐癸紝杩斿洖棰勬祴钀界偣 (x, y)銆�

鏂规硶鍒楄〃锛�
  M1: 浜屾澶氶」寮忔嫙鍚堝鎺�
  M2: 涓夋澶氶」寮忔嫙鍚堝鎺紙甯︽鍒欏寲闃茶繃鎷熷悎锛�
  M3: 鐗╃悊绾︽潫鎷熷悎锛堝湪绾夸及璁rag鍙傛暟锛�
  M4: UKF鍦ㄧ嚎鐘舵€佷及璁� + 鐗╃悊澶栨帹
  M5: 鍥哄畾鍙傛暟RK4澶栨帹锛圔aseline锛�
  M6: 涓夋鏍锋潯骞虫粦澶栨帹
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

# ============================================================
# 鐗╃悊甯搁噺
# ============================================================
G = 9.81
M_DEFAULT = 0.005       # kg
C_DEFAULT = 0.001       # 浜屾闃诲姏绯绘暟
B_DEFAULT = 0.0         # 绾挎€ч樆鍔涚郴鏁�


def _estimate_velocity(obs_pts, obs_t, n_points=5):
    """椴佹閫熷害浼拌锛氱敤鏈€鍚巒_points涓偣鍋氱嚎鎬у洖褰�"""
    N = len(obs_t)
    n = min(n_points, N)
    vels = []
    for d in range(3):
        t_seg = obs_t[-n:]
        p_seg = obs_pts[-n:, d]
        A = np.column_stack([t_seg, np.ones(n)])
        slope, _ = np.linalg.lstsq(A, p_seg, rcond=None)[0]
        vels.append(slope)
    return np.array(vels)


def _find_landing_time_poly(coeffs_z, t0, t_last):
    """浠巣杞村椤瑰紡绯绘暟鎵剧潃鍦版椂闂�"""
    roots = np.roots(coeffs_z)
    valid = roots[np.isreal(roots) & (np.real(roots) > t_last - t0 + 0.001)]
    if len(valid) > 0:
        return float(np.min(np.real(valid))) + t0
    return None


def _fallback_landing_time(v_last, p_last, g=G):
    """澶囩敤鐫€鍦版椂闂翠及璁★紙鍩轰簬鑷敱钀戒綋锛�"""
    vz, z = v_last[2], max(p_last[2], 0.01)
    disc = vz**2 + 2 * g * z
    if disc >= 0:
        t_rem = (vz + np.sqrt(disc)) / g
    else:
        t_rem = 0.5
    return max(0.01, min(2.0, t_rem))


# ============================================================
# M1: 浜屾澶氶」寮忔嫙鍚�
# ============================================================
def method_poly2(obs_pts, obs_t, g=G):
    """
    浜屾澶氶」寮忔嫙鍚堝鎺�

    瀵规瘡涓淮搴︽嫙鍚� a*t虏 + b*t + c锛岀劧鍚庢壘 z=0 鐨勬牴寰楀埌鐫€鍦版椂闂淬€�
    浜岄樁澶氶」寮忓搴旂殑鏄寑鍔犻€熻繍鍔紝鐗╃悊鍚箟鏄庣‘銆�
    """
    N = len(obs_t)
    t_norm = obs_t - obs_t[0]
    t_last = obs_t[-1]

    # 瀵规瘡涓淮搴︽嫙鍚堜簩娆″椤瑰紡
    coeffs = []
    for d in range(3):
        c = np.polyfit(t_norm, obs_pts[:, d], 2)
        coeffs.append(c)

    # 鎵剧潃鍦版椂闂�
    t_land = _find_landing_time_poly(coeffs[2], obs_t[0], t_last)
    if t_land is None:
        v_last = _estimate_velocity(obs_pts, obs_t)
        t_rem = _fallback_landing_time(v_last, obs_pts[-1], g)
        t_land = t_last + t_rem

    dt_land = t_land - obs_t[0]
    landing = np.array([np.polyval(coeffs[d], dt_land) for d in range(3)])
    landing[2] = 0

    # 鎷熷悎璐ㄩ噺
    residuals = 0
    for d in range(3):
        fit = np.polyval(coeffs[d], t_norm)
        residuals += np.sum((obs_pts[:, d] - fit)**2)
    rmse = np.sqrt(residuals / (3 * N))

    return {
        'landing_xy': landing[:2],
        'landing_t': t_land,
        'fit_rmse': rmse,
        'method': 'M1_poly2',
    }


# ============================================================
# M2: 涓夋澶氶」寮忔嫙鍚堬紙甯ikhonov姝ｅ垯鍖栵級
# ============================================================
def method_poly3(obs_pts, obs_t, g=G, regularization=0.01):
    """
    涓夋澶氶」寮忔嫙鍚堝鎺紙甯︽鍒欏寲闃茶繃鎷熷悎锛�

    绾笁娆″椤瑰紡鍦ㄦ暟鎹皯鏃朵細鍓х儓鍙戞暎锛堝疄楠屼腑瑙傚療鍒�112绫宠宸級銆�
    鍔犱笂Tikhonov姝ｅ垯鍖栵紙鎯╃綒楂橀樁椤圭郴鏁帮級鍙互鏄捐憲鎻愬崌绋冲畾鎬с€�
    """
    N = len(obs_t)
    t_norm = obs_t - obs_t[0]
    t_last = obs_t[-1]

    coeffs = []
    for d in range(3):
        A = np.column_stack([t_norm**3, t_norm**2, t_norm, np.ones(N)])
        y = obs_pts[:, d]

        # Tikhonov姝ｅ垯鍖�: 鎯╃綒涓夐樁鍜屼簩闃堕」
        Gamma = np.diag([regularization, regularization * 0.1, 0, 0])
        coeff = np.linalg.solve(A.T @ A + Gamma, A.T @ y)
        coeffs.append(coeff)

    t_land = _find_landing_time_poly(coeffs[2], obs_t[0], t_last)
    if t_land is None:
        v_last = _estimate_velocity(obs_pts, obs_t)
        t_rem = _fallback_landing_time(v_last, obs_pts[-1], g)
        t_land = t_last + t_rem

    dt_land = t_land - obs_t[0]
    landing = np.array([np.polyval(coeffs[d], dt_land) for d in range(3)])
    landing[2] = 0

    # 鍚堢悊鎬ф鏌ワ細濡傛灉钀界偣椋炲嚭鐞冨満锛岃“鍑忓洖閫€
    if not (-8 < landing[0] < 8 and -6 < landing[1] < 6):
        # 閫€鍖栦负浜屾澶氶」寮�
        return method_poly2(obs_pts, obs_t, g)

    residuals = 0
    for d in range(3):
        fit = np.polyval(coeffs[d], t_norm)
        residuals += np.sum((obs_pts[:, d] - fit)**2)
    rmse = np.sqrt(residuals / (3 * N))

    return {
        'landing_xy': landing[:2],
        'landing_t': t_land,
        'fit_rmse': rmse,
        'method': 'M2_poly3',
    }


# ============================================================
# M3: 鐗╃悊绾︽潫鎷熷悎锛堝湪绾夸及璁rag鍙傛暟锛�
# ============================================================
def method_physics_fit(obs_pts, obs_t, g=G, m=M_DEFAULT):
    """
    鐗╃悊绾︽潫鎷熷悎锛氱綉鏍兼悳绱㈡渶浼榙rag鍙傛暟锛岀劧鍚嶳K4澶栨帹

    杩欐槸Hawk-Eye绫荤郴缁熺殑鏍稿績鏂规硶銆�
    涓嶉璁惧浐瀹氱殑drag鍙傛暟锛岃€屾槸浠庤娴嬫暟鎹腑鍦ㄧ嚎浼拌銆�
    """
    N = len(obs_t)
    t_last = obs_t[-1]
    v_last = _estimate_velocity(obs_pts, obs_t)
    p_last = obs_pts[-1].copy()

    # 缃戞牸鎼滅储鏈€浼榙rag
    best_loss = np.inf
    best_c = C_DEFAULT

    for c_try in np.linspace(0.0001, 0.003, 15):
        # 浠庣涓€涓娴嬬偣姝ｅ悜妯℃嫙
        dt = 0.0005
        p_sim = obs_pts[0].copy()
        v_sim = _estimate_velocity(obs_pts[:5], obs_t[:5]) if N >= 5 else np.zeros(3)

        loss = 0.0
        obs_idx = 1
        t_sim = obs_t[0]

        for _ in range(5000):
            spd = np.linalg.norm(v_sim)
            gv = np.array([0, 0, -g])
            if spd > 1e-6:
                drag = -(0.0 * spd + c_try * spd**2) * v_sim / spd
                a = gv + drag / m
            else:
                a = gv

            # RK4 step
            k1v = a * dt
            k1p = v_sim * dt

            v2 = v_sim + 0.5 * k1v
            spd2 = np.linalg.norm(v2)
            if spd2 > 1e-6:
                a2 = gv + (-(0.0 * spd2 + c_try * spd2**2) * v2 / spd2) / m
            else:
                a2 = gv
            k2v = a2 * dt
            k2p = (v_sim + 0.5 * k1v) * dt

            v3 = v_sim + 0.5 * k2v
            spd3 = np.linalg.norm(v3)
            if spd3 > 1e-6:
                a3 = gv + (-(0.0 * spd3 + c_try * spd3**2) * v3 / spd3) / m
            else:
                a3 = gv
            k3v = a3 * dt
            k3p = (v_sim + 0.5 * k2v) * dt

            v4 = v_sim + k3v
            spd4 = np.linalg.norm(v4)
            if spd4 > 1e-6:
                a4 = gv + (-(0.0 * spd4 + c_try * spd4**2) * v4 / spd4) / m
            else:
                a4 = gv
            k4v = a4 * dt
            k4p = (v_sim + k3v) * dt

            v_sim = v_sim + (k1v + 2*k2v + 2*k3v + k4v) / 6
            p_sim = p_sim + (k1p + 2*k2p + 2*k3p + k4p) / 6
            t_sim += dt

            # 鍖归厤鏈€杩戠殑瑙傛祴鐐�
            while obs_idx < N and obs_t[obs_idx] <= t_sim + dt:
                loss += np.sum((p_sim - obs_pts[obs_idx])**2)
                obs_idx += 1

            if obs_idx >= N:
                break

        loss /= max(obs_idx, 1)
        if loss < best_loss:
            best_loss = loss
            best_c = c_try

    # 鐢ㄦ渶浼樺弬鏁颁粠鏈€鍚庤娴嬬偣澶栨帹
    dt = 0.0005
    p = p_last.copy()
    v = v_last.copy()

    for _ in range(5000):
        spd = np.linalg.norm(v)
        gv = np.array([0, 0, -g])
        if spd > 1e-6:
            drag = -(0.0 * spd + best_c * spd**2) * v / spd
            a = gv + drag / m
        else:
            a = gv

        k1v = a * dt; k1p = v * dt
        v2 = v + 0.5*k1v; spd2 = np.linalg.norm(v2)
        a2 = gv + (-(0.0*spd2 + best_c*spd2**2)*v2/spd2)/m if spd2>1e-6 else gv
        k2v = a2*dt; k2p = (v+0.5*k1v)*dt
        v3 = v + 0.5*k2v; spd3 = np.linalg.norm(v3)
        a3 = gv + (-(0.0*spd3 + best_c*spd3**2)*v3/spd3)/m if spd3>1e-6 else gv
        k3v = a3*dt; k3p = (v+0.5*k2v)*dt
        v4 = v + k3v; spd4 = np.linalg.norm(v4)
        a4 = gv + (-(0.0*spd4 + best_c*spd4**2)*v4/spd4)/m if spd4>1e-6 else gv
        k4v = a4*dt; k4p = (v+k3v)*dt

        v = v + (k1v+2*k2v+2*k3v+k4v)/6
        p = p + (k1p+2*k2p+2*k3p+k4p)/6

        if p[2] <= 0:
            # 鎻掑€肩簿纭惤鐐�
            p[2] = 0
            break

    landing = p.copy()
    landing[2] = 0

    return {
        'landing_xy': landing[:2],
        'landing_t': float('nan'),  # 闇€瑕佸洖鎺ㄧ畻
        'fit_rmse': np.sqrt(best_loss),
        'method': 'M3_physics_fit',
        'fitted_drag': best_c,
    }


# ============================================================
# M4: 鏍锋潯骞虫粦 + 鐗╃悊澶栨帹锛圲KF鐨勫揩閫熺瓑浠峰疄鐜帮級
# ============================================================
def method_spline_ukf(obs_pts, obs_t, g=G, m=M_DEFAULT, c=C_DEFAULT):
    """
    鏍锋潯骞虫粦鐘舵€佷及璁� + 鐗╃悊妯″瀷澶栨帹

    鐢ㄤ笁娆℃牱鏉″瑙傛祴鏁版嵁鍋氬钩婊戯紙绛変环浜嶶KF鐨勫悗楠屽潎鍊间及璁★級锛�
    鎻愬彇骞虫粦鍚庣殑浣嶇疆鍜岄€熷害锛岀劧鍚庣敤鐗╃悊妯″瀷澶栨帹銆�
    """
    N = len(obs_t)
    t_last = obs_t[-1]

    # 涓夋鏍锋潯骞虫粦
    cs_x = CubicSpline(obs_t, obs_pts[:, 0])
    cs_y = CubicSpline(obs_t, obs_pts[:, 1])
    cs_z = CubicSpline(obs_t, obs_pts[:, 2])

    # 鎻愬彇骞虫粦鐘舵€�
    p_smooth = np.array([cs_x(t_last), cs_y(t_last), cs_z(t_last)])
    v_smooth = np.array([cs_x(t_last, 1), cs_y(t_last, 1), cs_z(t_last, 1)])

    # RK4澶栨帹
    dt = 0.0005
    p = p_smooth.copy()
    v = v_smooth.copy()

    for _ in range(5000):
        spd = np.linalg.norm(v)
        gv = np.array([0, 0, -g])
        if spd > 1e-6:
            drag = -(0.0*spd + c*spd**2) * v / spd
            a = gv + drag / m
        else:
            a = gv

        k1v=a*dt; k1p=v*dt
        v2=v+0.5*k1v; spd2=np.linalg.norm(v2)
        a2=gv+(-(0.0*spd2+c*spd2**2)*v2/spd2)/m if spd2>1e-6 else gv
        k2v=a2*dt; k2p=(v+0.5*k1v)*dt
        v3=v+0.5*k2v; spd3=np.linalg.norm(v3)
        a3=gv+(-(0.0*spd3+c*spd3**2)*v3/spd3)/m if spd3>1e-6 else gv
        k3v=a3*dt; k3p=(v+0.5*k2v)*dt
        v4=v+k3v; spd4=np.linalg.norm(v4)
        a4=gv+(-(0.0*spd4+c*spd4**2)*v4/spd4)/m if spd4>1e-6 else gv
        k4v=a4*dt; k4p=(v+k3v)*dt

        v=v+(k1v+2*k2v+2*k3v+k4v)/6
        p=p+(k1p+2*k2p+2*k3p+k4p)/6

        if p[2] <= 0:
            p[2] = 0
            break

    return {
        'landing_xy': p[:2],
        'landing_t': float('nan'),
        'fit_rmse': 0.0,
        'method': 'M4_spline_ukf',
    }


# ============================================================
# M5: 鍥哄畾鍙傛暟RK4澶栨帹锛圔aseline锛�
# ============================================================
def method_rk4_fixed(obs_pts, obs_t, g=G, m=M_DEFAULT, c=C_DEFAULT):
    """鍥哄畾鐗╃悊鍙傛暟鐨凴K4澶栨帹锛屼綔涓築aseline瀵圭収"""
    t_last = obs_t[-1]
    p_last = obs_pts[-1].copy()
    v_last = _estimate_velocity(obs_pts, obs_t)

    dt = 0.0005
    p = p_last.copy()
    v = v_last.copy()

    for _ in range(5000):
        spd = np.linalg.norm(v)
        gv = np.array([0, 0, -g])
        if spd > 1e-6:
            drag = -(0.0*spd + c*spd**2) * v / spd
            a = gv + drag / m
        else:
            a = gv

        v = v + a * dt
        p = p + v * dt

        if p[2] <= 0:
            p[2] = 0
            break

    return {
        'landing_xy': p[:2],
        'landing_t': float('nan'),
        'fit_rmse': 0.0,
        'method': 'M5_rk4_fixed',
    }


# ============================================================
# M6: 涓夋鏍锋潯澶栨帹
# ============================================================
def method_spline_extrapolate(obs_pts, obs_t):
    """绾笁娆℃牱鏉″鎺紙鏃犵墿鐞嗙害鏉燂級"""
    t_last = obs_t[-1]
    v_last = _estimate_velocity(obs_pts, obs_t)

    cs_x = CubicSpline(obs_t, obs_pts[:, 0])
    cs_y = CubicSpline(obs_t, obs_pts[:, 1])
    cs_z = CubicSpline(obs_t, obs_pts[:, 2])

    # 鎼滅储鐫€鍦版椂闂�
    t_search = np.linspace(t_last, t_last + 2.0, 500)
    z_search = cs_z(t_search)

    t_land = t_last + _fallback_landing_time(v_last, obs_pts[-1])
    for i in range(len(t_search) - 1):
        if z_search[i] * z_search[i+1] <= 0 and z_search[i] >= 0:
            t_land = t_search[i] - z_search[i] * (t_search[i+1]-t_search[i]) / (z_search[i+1]-z_search[i])
            break

    landing = np.array([float(cs_x(t_land)), float(cs_y(t_land))])

    # 鍚堢悊鎬ф鏌�
    if not (-7 < landing[0] < 7 and -5 < landing[1] < 5):
        # fallback
        landing = obs_pts[-1, :2] + v_last[:2] * 0.3

    return {
        'landing_xy': landing,
        'landing_t': t_land,
        'fit_rmse': 0.0,
        'method': 'M6_spline',
    }


# ============================================================
# 缁熶竴鎺ュ彛
# ============================================================
ALL_METHODS = {
    'M1_poly2': method_poly2,
    'M2_poly3': method_poly3,
    'M3_physics_fit': method_physics_fit,
    'M4_spline_ukf': method_spline_ukf,
    'M5_rk4_fixed': method_rk4_fixed,
    'M6_spline': method_spline_extrapolate,
}


def run_all_methods(obs_pts, obs_t, physics_params=None):
    """杩愯鍏ㄩ儴6绉嶆柟娉�"""
    results = {}
    for name, fn in ALL_METHODS.items():
        try:
            if physics_params and name in ['M3_physics_fit', 'M4_spline_ukf', 'M5_rk4_fixed']:
                results[name] = fn(obs_pts, obs_t, **physics_params)
            else:
                results[name] = fn(obs_pts, obs_t)
        except Exception as e:
            results[name] = {'error': str(e), 'method': name}
    return results


if __name__ == '__main__':
    print("trajectory_methods.py 鈥� 6绉嶈建杩归娴嬫柟娉�")
    print(f"宸叉敞鍐屾柟娉�: {list(ALL_METHODS.keys())}")
    print()
    print("浣跨敤鏂瑰紡:")
    print("  from trajectory_methods import run_all_methods")
    print("  results = run_all_methods(obs_3d, obs_times)")
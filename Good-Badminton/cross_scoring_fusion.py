#!/usr/bin/env python3
"""
cross_scoring_fusion.py 鈥� 浜ゅ弶璇勫垎鍒惰建杩硅瀺鍚�

鏍稿績鍒涙柊锛堣鏂囪础鐚偣锛夛細
    涓嶆槸璁�6绉嶆柟娉曞悇鑷嫭绔嬮娴嬬劧鍚庡姞鏉冨钩鍧囷紙甯歌ensemble锛夛紝
    鑰屾槸璁╂瘡绉嶆柟娉曞悓鏃舵壆婕�"鎻愭鑰�"鍜�"璇勫"涓や釜瑙掕壊銆�

    1. 鎻愭闃舵锛�6绉嶆柟娉曞悇鑷敓鎴愪竴鏉″€欓€夎建杩�
    2. 璇勫垎闃舵锛�5涓瘎鍒嗙淮搴︾粰姣忔潯鍊欓€夋墦鍒�
    3. 瑁佸喅闃舵锛氭€诲垎鏈€楂樼殑鍊欓€夎建杩瑰綋閫�

涓�"骞惰璺�+鍔犳潈骞冲潎"鐨勬湰璐ㄥ尯鍒細
    鏃ф柟妗堬細缁撴灉 = (M1+M2+...+M6)/6
            鈫� 鍙鏈変竴涓柟娉曞け璐ワ紝缁撴灉灏辫姹℃煋

    鏂版柟妗堬細缁撴灉 = argmax_i 危_j w_j 脳 score_j(proposal_i)
            鈫� 澶辫触鏂规硶鑷劧寰楀垎浣庯紝涓嶅奖鍝嶆渶缁堢粨鏋�
            鈫� 鎵€鏈夋柟娉曚簰鐩稿鏌ワ紝褰㈡垚"鍚岃璇勮"
"""

import numpy as np
from scipy.interpolate import CubicSpline
import time
import warnings
warnings.filterwarnings('ignore')

# 鐗╃悊甯搁噺
G = 9.81
M_DEFAULT = 0.005
C_DEFAULT = 0.001
B_DEFAULT = 0.0


class CrossScoringFusion:
    """
    浜ゅ弶璇勫垎鍒惰建杩硅瀺鍚堝櫒

    Parameters
    ----------
    obs_pts_3d : ndarray (N, 3) 鈥� 涓夎娴嬮噺寰楀埌鐨�3D杞ㄨ抗鐐�
    obs_times : ndarray (N,) 鈥� 姣忎釜鐐圭殑鏃堕棿鎴�
    physics_params : dict 鈥� {'mass', 'drag_linear', 'drag_quad', 'g'}
    """

    def __init__(self, obs_pts_3d, obs_times, physics_params=None):
        self.obs = np.asarray(obs_pts_3d)
        self.t_obs = np.asarray(obs_times)
        self.N = len(obs_times)
        self.t0 = obs_times[0]
        self.t_last = obs_times[-1]
        self.p_last = self.obs[-1]

        p = physics_params or {}
        self.m = p.get('mass', M_DEFAULT)
        self.b = p.get('drag_linear', B_DEFAULT)
        self.c = p.get('drag_quad', C_DEFAULT)
        self.g = p.get('g', G)

        # 椴佹閫熷害浼拌
        self.v_last = self._estimate_velocity()

        # 棰勬嫙鍚堝椤瑰紡
        t_norm = self.t_obs - self.t0
        self.poly2_c = [np.polyfit(t_norm, self.obs[:, d], 2) for d in range(3)]
        # 涓夋澶氶」寮忓姞姝ｅ垯鍖�
        self.poly3_c = []
        for d in range(3):
            A = np.column_stack([t_norm**3, t_norm**2, t_norm, np.ones(self.N)])
            y = self.obs[:, d]
            Gamma = np.diag([0.01, 0.001, 0, 0])
            self.poly3_c.append(np.linalg.solve(A.T @ A + Gamma, A.T @ y))

        # 璇勫鏉冮噸
        self.scorer_weights = np.array([0.25, 0.20, 0.25, 0.10, 0.20])
        #                               poly   phys  data  smooth reason

    def _estimate_velocity(self):
        if self.N >= 5:
            dts = np.diff(self.t_obs[-5:])
            dps = np.diff(self.obs[-5:], axis=0)
            return np.mean(dps / dts[:, None], axis=0)
        elif self.N >= 2:
            return (self.obs[-1] - self.obs[-2]) / (self.t_obs[-1] - self.t_obs[-2])
        return np.zeros(3)

    def _fallback_t_landing(self):
        vz, z = self.v_last[2], max(self.p_last[2], 0.01)
        disc = vz**2 + 2 * self.g * z
        t_rem = (vz + np.sqrt(max(disc, 0))) / self.g if disc >= 0 else 0.5
        return self.t_last + max(0.01, min(1.5, t_rem))

    def _rk4_forward(self, p0, v0, m_fit, c_fit, max_steps=4000):
        dt = 0.0005
        p, v = p0.copy(), v0.copy()
        traj_t, traj_p = [self.t_last], [p.copy()]

        for _ in range(max_steps):
            spd = np.linalg.norm(v)
            gv = np.array([0, 0, -self.g])
            if spd > 1e-6:
                a = gv + (-(self.b*spd + c_fit*spd**2) * v/spd) / m_fit
            else:
                a = gv

            k1v, k1p = a*dt, v*dt
            v2 = v + 0.5*k1v; spd2 = np.linalg.norm(v2)
            a2 = gv + (-(self.b*spd2 + c_fit*spd2**2) * v2/spd2) / m_fit if spd2 > 1e-6 else gv
            k2v, k2p = a2*dt, (v+0.5*k1v)*dt
            v3 = v + 0.5*k2v; spd3 = np.linalg.norm(v3)
            a3 = gv + (-(self.b*spd3 + c_fit*spd3**2) * v3/spd3) / m_fit if spd3 > 1e-6 else gv
            k3v, k3p = a3*dt, (v+0.5*k2v)*dt
            v4 = v + k3v; spd4 = np.linalg.norm(v4)
            a4 = gv + (-(self.b*spd4 + c_fit*spd4**2) * v4/spd4) / m_fit if spd4 > 1e-6 else gv
            k4v, k4p = a4*dt, (v+k3v)*dt

            v = v + (k1v+2*k2v+2*k3v+k4v)/6
            p = p + (k1p+2*k2p+2*k3p+k4p)/6
            traj_t.append(traj_t[-1]+dt)
            traj_p.append(p.copy())

            if p[2] <= 0:
                break

        return np.array(traj_t), np.array(traj_p)

    def _find_landing(self, traj_t, traj_p):
        for i in range(len(traj_p) - 1):
            if traj_p[i+1, 2] <= 0 and traj_p[i, 2] > 0:
                alpha = -traj_p[i, 2] / (traj_p[i+1, 2] - traj_p[i, 2])
                landing = traj_p[i] + alpha * (traj_p[i+1] - traj_p[i])
                landing[2] = 0
                return landing, traj_t[i] + alpha * (traj_t[i+1] - traj_t[i])
        landing = traj_p[-1].copy()
        landing[2] = 0
        return landing, traj_t[-1]

    def _merge_traj(self, traj_t, traj_p, t_land):
        t_full = np.linspace(self.t0, t_land, 100)
        pos_full = np.zeros((100, 3))
        for i, t in enumerate(t_full):
            if t <= self.t_last:
                idx = min(np.searchsorted(self.t_obs, t), self.N - 1)
                if idx == 0:
                    pos_full[i] = self.obs[0]
                else:
                    alpha = (t - self.t_obs[idx-1]) / (self.t_obs[idx] - self.t_obs[idx-1])
                    pos_full[i] = self.obs[idx-1] + alpha * (self.obs[idx] - self.obs[idx-1])
            else:
                idx = min(np.searchsorted(traj_t, t), len(traj_t) - 1)
                if idx == 0:
                    pos_full[i] = traj_p[0]
                else:
                    alpha = (t - traj_t[idx-1]) / (traj_t[idx] - traj_t[idx-1])
                    pos_full[i] = traj_p[idx-1] + alpha * (traj_p[idx] - traj_p[idx-1])
        return t_full, pos_full

    def _mk_proposal(self, landing, t_land, t_full, pos_full, name, extra=None):
        return {'landing': landing, 't_land': t_land, 't': t_full,
                'pos': pos_full, 'name': name, 'extra': extra or {}}

    # ---- 6涓彁妗堣€� ----

    def _propose_poly2(self):
        c = self.poly2_c
        roots = np.roots(c[2])
        valid = roots[np.isreal(roots) & (np.real(roots) > self.t_last - self.t0 + 0.001)]
        t_land = float(np.min(np.real(valid))) + self.t0 if len(valid) > 0 else self._fallback_t_landing()
        landing = np.array([np.polyval(c[d], t_land - self.t0) for d in range(3)])
        landing[2] = 0
        t_full = np.linspace(self.t0, t_land, 100)
        pos_full = np.array([[np.polyval(c[d], t - self.t0) for d in range(3)] for t in t_full])
        return self._mk_proposal(landing, t_land, t_full, pos_full, 'M1_quadratic')

    def _propose_poly3(self):
        c = self.poly3_c
        roots = np.roots(c[2])
        valid = roots[np.isreal(roots) & (np.real(roots) > self.t_last - self.t0 + 0.001)]
        t_land = float(np.min(np.real(valid))) + self.t0 if len(valid) > 0 else self._fallback_t_landing()
        landing = np.array([np.polyval(c[d], t_land - self.t0) for d in range(3)])
        landing[2] = 0
        # 鍚堢悊鎬ф鏌�
        if not (-7 < landing[0] < 7 and -5 < landing[1] < 5):
            return self._propose_poly2()
        t_full = np.linspace(self.t0, t_land, 100)
        pos_full = np.array([[np.polyval(c[d], t - self.t0) for d in range(3)] for t in t_full])
        return self._mk_proposal(landing, t_land, t_full, pos_full, 'M2_cubic')

    def _propose_physics(self):
        best_loss, best_m, best_c = np.inf, self.m, self.c
        for m_try in [0.003, 0.004, 0.005, 0.006, 0.007]:
            for c_try in [0.0001, 0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.003]:
                dt = 0.0005
                p_sim = self.obs[0].copy()
                v_sim = (self.obs[1] - self.obs[0]) / (self.t_obs[1] - self.t_obs[0]) if self.N >= 2 else np.zeros(3)
                total_err, obs_idx = 0.0, 1

                for _ in range(4000):
                    spd = np.linalg.norm(v_sim)
                    gv = np.array([0, 0, -self.g])
                    if spd > 1e-6:
                        a = gv + (-(0.0*spd + c_try*spd**2) * v_sim/spd) / m_try
                    else:
                        a = gv

                    k1v, k1p = a*dt, v_sim*dt
                    v2 = v_sim + 0.5*k1v; spd2 = np.linalg.norm(v2)
                    a2 = gv + (-(0.0*spd2 + c_try*spd2**2)*v2/spd2)/m_try if spd2>1e-6 else gv
                    k2v, k2p = a2*dt, (v_sim+0.5*k1v)*dt
                    v3 = v_sim + 0.5*k2v; spd3 = np.linalg.norm(v3)
                    a3 = gv + (-(0.0*spd3 + c_try*spd3**2)*v3/spd3)/m_try if spd3>1e-6 else gv
                    k3v, k3p = a3*dt, (v_sim+0.5*k2v)*dt
                    v4 = v_sim + k3v; spd4 = np.linalg.norm(v4)
                    a4 = gv + (-(0.0*spd4 + c_try*spd4**2)*v4/spd4)/m_try if spd4>1e-6 else gv
                    k4v, k4p = a4*dt, (v_sim+k3v)*dt

                    v_sim += (k1v+2*k2v+2*k3v+k4v)/6
                    p_sim += (k1p+2*k2p+2*k3p+k4p)/6

                    while obs_idx < self.N and self.t_obs[obs_idx] <= self.t_obs[0] + obs_idx * dt:
                        total_err += np.sum((p_sim - self.obs[obs_idx])**2)
                        obs_idx += 1
                    if obs_idx >= self.N:
                        break

                loss = total_err / max(obs_idx, 1)
                if loss < best_loss:
                    best_loss, best_m, best_c = loss, m_try, c_try

        traj_t, traj_p = self._rk4_forward(self.p_last, self.v_last, best_m, best_c)
        landing, t_land = self._find_landing(traj_t, traj_p)
        t_full, pos_full = self._merge_traj(traj_t, traj_p, t_land)
        return self._mk_proposal(landing, t_land, t_full, pos_full, 'M3_physics',
                                 {'fitted_mass': best_m, 'fitted_drag': best_c})

    def _propose_ukf(self):
        cs_x = CubicSpline(self.t_obs, self.obs[:, 0])
        cs_y = CubicSpline(self.t_obs, self.obs[:, 1])
        cs_z = CubicSpline(self.t_obs, self.obs[:, 2])
        sp = np.array([cs_x(self.t_last), cs_y(self.t_last), cs_z(self.t_last)])
        sv = np.array([cs_x(self.t_last, 1), cs_y(self.t_last, 1), cs_z(self.t_last, 1)])

        traj_t, traj_p = self._rk4_forward(sp, sv, self.m, self.c)
        landing, t_land = self._find_landing(traj_t, traj_p)

        t_full = np.linspace(self.t0, t_land, 100)
        pos_full = np.zeros((100, 3))
        for i, t in enumerate(t_full):
            if t <= self.t_last:
                pos_full[i] = np.array([cs_x(t), cs_y(t), cs_z(t)])
            else:
                idx = min(np.searchsorted(traj_t, t), len(traj_t) - 1)
                if idx == 0:
                    pos_full[i] = traj_p[0]
                else:
                    alpha = (t - traj_t[idx-1]) / (traj_t[idx] - traj_t[idx-1])
                    pos_full[i] = traj_p[idx-1] + alpha * (traj_p[idx] - traj_p[idx-1])

        return self._mk_proposal(landing, t_land, t_full, pos_full, 'M4_UKF')

    def _propose_baseline(self):
        traj_t, traj_p = self._rk4_forward(self.p_last, self.v_last, self.m, self.c)
        landing, t_land = self._find_landing(traj_t, traj_p)
        t_full, pos_full = self._merge_traj(traj_t, traj_p, t_land)
        return self._mk_proposal(landing, t_land, t_full, pos_full, 'M5_baseline')

    def _propose_spline(self):
        cs_x = CubicSpline(self.t_obs, self.obs[:, 0])
        cs_y = CubicSpline(self.t_obs, self.obs[:, 1])
        cs_z = CubicSpline(self.t_obs, self.obs[:, 2])

        t_search = np.linspace(self.t_last, self.t_last + 2.0, 500)
        z_vals = cs_z(t_search)
        t_land = self._fallback_t_landing()
        for i in range(len(t_search) - 1):
            if z_vals[i] * z_vals[i+1] <= 0 and z_vals[i] >= 0:
                t_land = t_search[i] - z_vals[i] * (t_search[i+1] - t_search[i]) / (z_vals[i+1] - z_vals[i])
                break

        landing = np.array([float(cs_x(t_land)), float(cs_y(t_land)), 0.0])
        if not (-7 < landing[0] < 7 and -5 < landing[1] < 5):
            landing = np.array([float(cs_x(self.t_last) + self.v_last[0]*0.3),
                               float(cs_y(self.t_last) + self.v_last[1]*0.3), 0.0])

        t_full = np.linspace(self.t0, t_land, 100)
        pos_full = np.column_stack([cs_x(t_full), cs_y(t_full), cs_z(t_full)])
        pos_full[-1, 2] = 0

        return self._mk_proposal(landing, t_land, t_full, pos_full, 'M6_spline')

    # ---- 5涓瘎濮� ----

    def _score_polynomial_fit(self, prop):
        pos, t = prop['pos'], prop['t']
        t_norm = t - t[0]
        total_res = 0.0
        for d in range(3):
            for order in [2, 3]:
                c = np.polyfit(t_norm, pos[:, d], order)
                fit = np.polyval(c, t_norm)
                total_res += np.mean((pos[:, d] - fit)**2)
        return float(np.clip(np.exp(-total_res * 10), 0.01, 1))

    def _score_physics_consistency(self, prop):
        pos, t = prop['pos'], prop['t']
        mask = t > self.t_last
        if np.sum(mask) < 3:
            return 0.5
        tp, pp = t[mask], pos[mask]
        ve = np.diff(pp, axis=0) / np.diff(tp)[:, None]
        ac = np.diff(ve, axis=0) / np.diff(tp[1:])[:, None]
        if len(ac) == 0:
            return 0.5
        err = 0.0
        for i in range(len(ac)):
            v = ve[min(i+1, len(ve)-1)]
            spd = np.linalg.norm(v)
            gv = np.array([0, 0, -self.g])
            if spd > 1e-6:
                ea = gv + (-(self.b*spd + self.c*spd**2) * v/spd) / self.m
            else:
                ea = gv
            err += np.sum((ac[i] - ea)**2)
        return float(np.clip(np.exp(-err/len(ac) * 0.1), 0.01, 1))

    def _score_data_fidelity(self, prop):
        pos, t = prop['pos'], prop['t']
        total_err = 0.0
        for i, to in enumerate(self.t_obs):
            idx = min(np.searchsorted(t, to), len(t) - 1)
            total_err += np.sum((pos[idx] - self.obs[i])**2)
        rmse = np.sqrt(total_err / self.N)
        return float(np.clip(np.exp(-rmse / 0.002), 0.01, 1))

    def _score_smoothness(self, prop):
        pos = prop['pos']
        jerk = np.diff(np.diff(np.diff(pos, axis=0), axis=0), axis=0)
        if len(jerk) == 0:
            return 0.8
        jerk_rms = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        return float(np.clip(np.exp(-jerk_rms * 0.001), 0.01, 1))

    def _score_reasonability(self, prop):
        landing = prop['landing']
        in_x = -7.0 <= landing[0] <= 7.0
        in_y = -5.0 <= landing[1] <= 5.0
        score = 1.0 if (in_x and in_y) else 0.1

        max_dist = np.linalg.norm(self.v_last) * 1.5 + 1.0
        actual_dist = np.linalg.norm(landing[:2] - self.p_last[:2])
        if actual_dist > max_dist:
            score *= max_dist / actual_dist

        return float(np.clip(score, 0.01, 1))

    # ---- 涓诲叆鍙� ----

    def fuse(self):
        """鎵ц浜ゅ弶璇勫垎铻嶅悎锛岃繑鍥炴渶缁堝垽缃氱粨鏋�"""
        t_start = time.time()

        # Step 1: 6绉嶆柟娉曟彁妗�
        proposals = [
            self._propose_poly2(),
            self._propose_poly3(),
            self._propose_physics(),
            self._propose_ukf(),
            self._propose_baseline(),
            self._propose_spline(),
        ]

        # Step 2: 5浣嶈瘎濮旂粰6鏉℃彁妗堟墦鍒�
        scorers = [
            self._score_polynomial_fit,
            self._score_physics_consistency,
            self._score_data_fidelity,
            self._score_smoothness,
            self._score_reasonability,
        ]

        score_matrix = np.zeros((len(proposals), len(scorers)))
        for i, prop in enumerate(proposals):
            for j, sc in enumerate(scorers):
                score_matrix[i, j] = sc(prop)

        # Step 3: 鍔犳潈鎬诲垎
        total_scores = score_matrix @ self.scorer_weights

        # Step 4: 鏈€楂樺垎褰撻€�
        best_idx = int(np.argmax(total_scores))
        best = proposals[best_idx]

        # Step 5: 缃俊搴�
        sorted_scores = np.sort(total_scores)[::-1]
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        else:
            margin = 0

        all_landings = np.array([p['landing'][:2] for p in proposals])
        landing_std = np.mean(np.std(all_landings, axis=0))
        consensus = float(np.exp(-landing_std / 0.05))

        # 鏁版嵁鍏呭垎鎬�
        obs_duration = self.t_last - self.t0
        data_suff = min(1.0, self.N / 15) * min(1.0, obs_duration / 0.2)

        confidence = float(np.clip(0.35 * margin + 0.35 * consensus + 0.30 * data_suff, 0, 1))
        if landing_std < 0.01:
            confidence = min(1.0, confidence + 0.1)

        elapsed = (time.time() - t_start) * 1000

        return {
            'landing_x': float(best['landing'][0]),
            'landing_y': float(best['landing'][1]),
            'landing_t': float(best['t_land']),
            'confidence': float(np.clip(confidence, 0, 1)),
            'winning_method': best['name'],
            'all_landings': {
                p['name']: (float(p['landing'][0]), float(p['landing'][1]))
                for p in proposals
            },
            'all_scores': {
                p['name']: float(total_scores[i])
                for i, p in enumerate(proposals)
            },
            'score_matrix': score_matrix.tolist(),
            'landing_consensus_mm': float(landing_std * 1000),
            'opt_time_ms': float(elapsed),
            'n_proposals': len(proposals),
            'n_scorers': len(scorers),
        }


if __name__ == '__main__':
    print("CrossScoringFusion 妯″潡鍔犺浇鎴愬姛銆�")
    print("浣跨敤鏂规硶:")
    print("  fusion = CrossScoringFusion(obs_3d, obs_t, physics_params)")
    print("  result = fusion.fuse()")
    print("  print(f\"钀界偣: ({result['landing_x']:.3f}, {result['landing_y']:.3f})\")")

#!/usr/bin/env python3
"""
precise_fusion.py 鈥� 绮剧‘杞ㄨ抗铻嶅悎锛堝叏灞€浼樺寲鐗堬級

杩欐槸"绮惧害浼樺厛"鐨勬渶楂樼骇铻嶅悎鏂规銆�
鍒╃敤2绉掓椂闂撮绠楋紝浣跨敤鍏ㄥ眬浼樺寲(differential evolution)
+ 灞€閮ㄧ簿鍖�(L-BFGS-B)锛屾壘鍒扮湡姝ｇ殑鏈€浼樿建杩广€�

鏍稿績鎬濊矾锛�
  瀹氫箟缁煎悎鎹熷け鍑芥暟 L(trajectory) = 危 w_i * L_i(trajectory)
  鍏朵腑 L_1~L_6 鏉ヨ嚜6绉嶆柟娉曠殑璇勪环鏍囧噯銆�

  鐢ㄥ叏灞€浼樺寲鍦ㄥ弬鏁扮┖闂翠腑鎼滅储锛屾壘鍒� min L(trajectory) 鐨勮建杩广€�
  杩欐潯杞ㄨ抗鏄�"鎵€鏈夋柟娉曞叡鍚岃鍙殑鏈€浼樿В"銆�

涓庝氦鍙夎瘎鍒嗙殑鍖哄埆锛�
  浜ゅ弶璇勫垎锛氫粠6鏉″€欓€夎建杩逛腑閫夋渶浼橈紙绂绘暎閫夋嫨锛�
  绮剧‘铻嶅悎锛氬湪鏁翠釜杞ㄨ抗绌洪棿涓悳绱㈡渶浼橈紙杩炵画浼樺寲锛�

  绮剧‘铻嶅悎 = 鐪熸鐨�"鎵€鏈夋柟娉曚竴璧锋壘鍑烘嫙鍚堝害鏈€楂樼殑鏇茬嚎"
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize, Bounds
from scipy.interpolate import CubicSpline
import time
import warnings
warnings.filterwarnings('ignore')

G = 9.81
M_DEFAULT = 0.005
C_DEFAULT = 0.001


class PreciseFusion:
    """
    绮剧‘杞ㄨ抗铻嶅悎浼樺寲鍣�

    鐢ㄥ弬鏁板寲澶氶」寮� + 鍏ㄥ眬浼樺寲锛屽湪杩炵画绌洪棿涓悳绱㈡渶浼樿建杩广€�

    Parameters
    ----------
    obs_pts_3d : ndarray (N, 3)
    obs_times : ndarray (N,)
    physics_params : dict
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
        self.b = p.get('drag_linear', 0.0)
        self.c = p.get('drag_quad', C_DEFAULT)
        self.g = p.get('g', G)

        # 閫熷害/鍔犻€熷害浼拌
        if self.N >= 5:
            dts = np.diff(self.t_obs[-5:])
            dps = np.diff(self.obs[-5:], axis=0)
            self.v_last = np.mean(dps / dts[:, None], axis=0)
            vels_all = np.diff(self.obs, axis=0) / np.diff(self.t_obs)[:, None]
            accs_all = np.diff(vels_all, axis=0) / np.diff(self.t_obs[1:])[:, None]
            self.a_last = np.mean(accs_all[-3:], axis=0) if len(accs_all) >= 3 else np.zeros(3)
        elif self.N >= 2:
            self.v_last = (self.obs[-1] - self.obs[-2]) / (self.t_obs[-1] - self.t_obs[-2])
            self.a_last = np.zeros(3)
        else:
            self.v_last = np.zeros(3)
            self.a_last = np.zeros(3)

        # 棰勮绠�
        t_norm = self.t_obs - self.t0
        self.poly2_c = [np.polyfit(t_norm, self.obs[:, d], 2) for d in range(3)]
        self.poly3_c = []
        for d in range(3):
            A = np.column_stack([t_norm**3, t_norm**2, t_norm, np.ones(self.N)])
            y = self.obs[:, d]
            Gamma = np.diag([0.01, 0.001, 0, 0])
            self.poly3_c.append(np.linalg.solve(A.T @ A + Gamma, A.T @ y))

        # 鍒濆鐫€鍦版椂闂翠及璁�
        self.t_land_init = self._estimate_landing_time()

        # Baseline
        self._compute_baseline()

    def _estimate_landing_time(self):
        roots = np.roots(self.poly2_c[2])
        valid = roots[np.isreal(roots) & (np.real(roots) > self.t_last - self.t0 + 0.001)]
        if len(valid) > 0:
            return float(np.min(np.real(valid))) + self.t0
        vz, z = self.v_last[2], max(self.p_last[2], 0.01)
        disc = vz**2 + 2 * self.g * z
        t_rem = (vz + np.sqrt(max(disc, 0))) / self.g if disc >= 0 else 0.5
        return self.t_last + max(0.01, min(1.5, t_rem))

    def _compute_baseline(self):
        """鐢ㄥ浐瀹氬弬鏁癛K4绠梑aseline钀界偣"""
        dt = 0.0005
        p = self.p_last.copy()
        v = self.v_last.copy()
        for _ in range(5000):
            spd = np.linalg.norm(v)
            gv = np.array([0, 0, -self.g])
            if spd > 1e-6:
                a = gv + (-(0.0*spd + self.c*spd**2) * v/spd) / self.m
            else:
                a = gv
            v = v + a*dt
            p = p + v*dt
            if p[2] <= 0:
                p[2] = 0
                break
        self.bl_landing = p[:2].copy()

    # ================================================================
    # 杞ㄨ抗鍙傛暟鍖�
    #
    # 棰勬祴娈� r(dt) = p_last + v_last*dt + 0.5*a_last*dt虏
    #                 + 伪*dt鲁 + 尾*dt鈦� + 纬*dt鈦�
    #
    # 鑷姩婊¤冻C2杩炵画锛堜綅缃€侀€熷害銆佸姞閫熷害鍦╰_last澶勫尮閰嶏級
    # 鑷敱鍙傛暟锛毼�(3) + 尾(3) + 纬(3) + t_landing(1) = 10涓�
    # ================================================================

    def _eval_trajectory(self, dt, alpha, beta, gamma):
        """鍦� dt = t - t_last 澶勬眰鍊�"""
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5

        pos = (self.p_last + self.v_last*dt + 0.5*self.a_last*dt2
               + alpha*dt3 + beta*dt4 + gamma*dt5)

        vel = (self.v_last + self.a_last*dt
               + 3*alpha*dt2 + 4*beta*dt3 + 5*gamma*dt4)

        acc = (self.a_last + 6*alpha*dt + 12*beta*dt2 + 20*gamma*dt3)

        return pos, vel, acc

    def _loss_function(self, params):
        """
        缁煎悎鎹熷け = 鎵€鏈夋柟娉曡瘎浠锋爣鍑嗙殑鍔犳潈鍜�

        params: [伪x, 伪y, 伪z, 尾x, 尾y, 尾z, 纬x, 纬y, 纬z, t_landing]
        """
        alpha = params[:3]
        beta = params[3:6]
        gamma = params[6:9]
        t_land = params[9]

        dt_land = t_land - self.t_last
        if dt_land <= 0.001:
            return 1e10

        # 棰勬祴娈甸噰鏍�
        n_samp = 20
        dt_samples = np.linspace(0, dt_land, n_samp)
        pos_s = np.zeros((n_samp, 3))
        vel_s = np.zeros((n_samp, 3))
        acc_s = np.zeros((n_samp, 3))
        for i, dt in enumerate(dt_samples):
            pos_s[i], vel_s[i], acc_s[i] = self._eval_trajectory(dt, alpha, beta, gamma)

        loss = 0.0

        # ---- L1: 鏁版嵁淇濈湡搴︼紙鐢ㄥ椤瑰紡鍥炴帹鍒拌娴嬫楠岃瘉锛� ----
        dt_back = self.t_obs - self.t_last
        data_err = 0.0
        n_valid = 0
        for i, dt in enumerate(dt_back):
            if dt >= -0.5:  # 鍙獙璇佹渶鍚�0.5绉�
                pred_pos, _, _ = self._eval_trajectory(dt, alpha, beta, gamma)
                data_err += np.sum((pred_pos - self.obs[i])**2)
                n_valid += 1
        if n_valid > 0:
            loss += 500.0 * data_err / n_valid

        # ---- L2: 鐗╃悊涓€鑷存€э紙棰勬祴娈碉級 ----
        phys_err = 0.0
        for i in range(n_samp):
            v = vel_s[i]
            spd = np.linalg.norm(v)
            gv = np.array([0, 0, -self.g])
            if spd > 1e-6:
                drag = -(self.b*spd + self.c*spd**2) * v/spd
                expected_acc = gv + drag / self.m
            else:
                expected_acc = gv
            phys_err += np.sum((acc_s[i] - expected_acc)**2)
        loss += 2.0 * phys_err / n_samp

        # ---- L3: 澶氶」寮忓悎鐞嗘€э紙楂橀樁椤逛笉搴斿お澶э級 ----
        loss += 0.05 * (np.sum(beta**2) + 0.1 * np.sum(gamma**2))

        # ---- L4: 骞虫粦搴︼紙jerk鐨凴MS锛� ----
        if n_samp >= 3:
            jerk = np.diff(acc_s, axis=0)
            jerk_rms = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
            loss += 0.005 * jerk_rms

        # ---- L5: Baseline涓€鑷存€� ----
        landing_xy = pos_s[-1, :2]
        bl_dev = np.sum((landing_xy - self.bl_landing)**2)
        loss += 0.05 * bl_dev

        # ---- L6: 鍦伴潰绾︽潫 ----
        ground_err = pos_s[-1, 2]**2 * 10000
        loss += ground_err

        # ---- L7: 鐞冨満鑼冨洿绾︽潫 ----
        if not (-7 < landing_xy[0] < 7 and -5 < landing_xy[1] < 5):
            loss += 100.0

        return float(loss)

    def optimize(self, use_global=True, max_iter_global=100, max_iter_local=200):
        """
        鎵ц绮剧‘浼樺寲

        Parameters
        ----------
        use_global : bool 鈥� 鏄惁浣跨敤鍏ㄥ眬浼樺寲锛坉ifferential evolution锛�
        max_iter_global : int 鈥� 鍏ㄥ眬浼樺寲鏈€澶ц凯浠�
        max_iter_local : int 鈥� 灞€閮ㄤ紭鍖栨渶澶ц凯浠�
        """
        t_start = time.time()

        # 鍙傛暟: 伪(3) + 尾(3) + 纬(3) + t_landing(1) = 10
        # 鍒濆鐚滄祴锛氬叏闆讹紙閫€鍖栦负鍖€鍔犻€熻繍鍔級+ 澶氶」寮忎及璁＄殑鐫€鍦版椂闂�
        x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, self.t_land_init])

        # 杈圭晫
        lb = np.array([
            -200, -200, -200,    # 伪 (dt鲁 term)
            -50, -50, -50,       # 尾 (dt鈦� term)
            -10, -10, -10,       # 纬 (dt鈦� term)
            self.t_last + 0.005  # t_landing
        ])
        ub = np.array([
            200, 200, 200,
            50, 50, 50,
            10, 10, 10,
            self.t_last + 2.5
        ])
        bounds = Bounds(lb, ub)

        if use_global:
            # Phase 1: 鍏ㄥ眬鎼滅储 (differential evolution)
            result_de = differential_evolution(
                self._loss_function,
                bounds,
                maxiter=max_iter_global,
                tol=1e-6,
                seed=42,
                polish=False,  # 鎴戜滑鑷繁鍋氬眬閮ㄧ簿鍖�
            )
            x_best = result_de.x
        else:
            x_best = x0

        # Phase 2: 灞€閮ㄧ簿鍖� (L-BFGS-B)
        result_local = minimize(
            self._loss_function,
            x_best,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter_local, 'ftol': 1e-14, 'gtol': 1e-10}
        )

        self.opt_result = result_local
        opt_params = result_local.x

        # 鎻愬彇缁撴灉
        self.alpha_opt = opt_params[:3]
        self.beta_opt = opt_params[3:6]
        self.gamma_opt = opt_params[6:9]
        self.t_landing_opt = opt_params[9]

        # 鐫€鍦扮偣
        dt_land = self.t_landing_opt - self.t_last
        land_pos, land_vel, _ = self._eval_trajectory(dt_land, self.alpha_opt, self.beta_opt, self.gamma_opt)
        self.landing = land_pos.copy()
        self.landing[2] = 0

        # 鐢熸垚瀹屾暣杞ㄨ抗
        self._build_full_trajectory()

        # 缃俊搴�
        self._compute_confidence()

        elapsed = (time.time() - t_start) * 1000

        return {
            'landing_x': float(self.landing[0]),
            'landing_y': float(self.landing[1]),
            'landing_t': float(self.t_landing_opt),
            'confidence': float(self.final_confidence),
            'loss_final': float(result_local.fun),
            'n_iter_global': getattr(result_de, 'nit', 0) if use_global else 0,
            'n_iter_local': int(result_local.nit),
            'opt_time_ms': float(elapsed),
            'method': 'PreciseFusion_UMTO',
        }

    def _build_full_trajectory(self):
        """鏋勫缓瑙傛祴娈�+棰勬祴娈电殑瀹屾暣杞ㄨ抗"""
        t_full = np.linspace(self.t0, self.t_landing_opt, 200)
        pos_full = np.zeros((200, 3))

        for i, t in enumerate(t_full):
            if t <= self.t_last:
                idx = np.searchsorted(self.t_obs, t)
                idx = min(idx, self.N - 1)
                if idx == 0:
                    pos_full[i] = self.obs[0]
                else:
                    alpha = (t - self.t_obs[idx-1]) / (self.t_obs[idx] - self.t_obs[idx-1])
                    pos_full[i] = self.obs[idx-1] + alpha * (self.obs[idx] - self.obs[idx-1])
            else:
                dt = t - self.t_last
                pos_full[i], _, _ = self._eval_trajectory(dt, self.alpha_opt, self.beta_opt, self.gamma_opt)

        self.full_traj_t = t_full
        self.full_traj_pos = pos_full

    def _compute_confidence(self):
        # 鏁版嵁淇濈湡搴�
        dt_back = self.t_obs - self.t_last
        data_err = 0.0
        n_valid = 0
        for i, dt in enumerate(dt_back):
            if dt >= -0.5:
                pred_pos, _, _ = self._eval_trajectory(dt, self.alpha_opt, self.beta_opt, self.gamma_opt)
                data_err += np.sum((pred_pos - self.obs[i])**2)
                n_valid += 1
        rmse = np.sqrt(data_err / max(n_valid, 1))
        q_fit = np.exp(-rmse / 0.003)

        # 鏁版嵁鍏呭垎鎬�
        obs_dur = self.t_last - self.t0
        q_data = min(1.0, self.N / 15) * min(1.0, obs_dur / 0.2)

        # 澶栨帹瀹夊叏鎬�
        ext_dt = self.t_landing_opt - self.t_last
        q_ext = np.exp(-max(0, ext_dt / max(obs_dur, 0.01) - 3.0) * 0.5)

        self.final_confidence = float(np.clip(0.4*q_fit + 0.3*q_data + 0.3*q_ext, 0, 1))

    def get_full_result(self):
        """鑾峰彇瀹屾暣缁撴灉锛堝寘鍚建杩规暟鎹紝鐢ㄤ簬鍙鍖栵級"""
        return {
            'landing_x': float(self.landing[0]),
            'landing_y': float(self.landing[1]),
            'landing_t': float(self.t_landing_opt),
            'confidence': float(self.final_confidence),
            'trajectory_t': self.full_traj_t.tolist(),
            'trajectory_x': self.full_traj_pos[:, 0].tolist(),
            'trajectory_y': self.full_traj_pos[:, 1].tolist(),
            'trajectory_z': self.full_traj_pos[:, 2].tolist(),
            'observed_t': self.t_obs.tolist(),
            'observed_x': self.obs[:, 0].tolist(),
            'observed_y': self.obs[:, 1].tolist(),
            'observed_z': self.obs[:, 2].tolist(),
            'opt_time_ms': float((time.time() - self.t_last) * 1000),
            'method': 'PreciseFusion_UMTO',
        }


# ================================================================
# 渚挎嵎鍑芥暟锛氫竴閿繍琛岀簿纭瀺鍚�
# ================================================================
def precise_fusion_predict(obs_pts_3d, obs_times, physics_params=None, 
                           use_global=True):
    """
    涓€閿繍琛岀簿纭瀺鍚堥娴�

    Parameters
    ----------
    obs_pts_3d : ndarray (N, 3)
    obs_times : ndarray (N,)
    physics_params : dict
    use_global : bool 鈥� 鏄惁浣跨敤鍏ㄥ眬浼樺寲

    Returns
    -------
    dict 鈥� 棰勬祴缁撴灉
    """
    fusion = PreciseFusion(obs_pts_3d, obs_times, physics_params)
    return fusion.optimize(use_global=use_global)


if __name__ == '__main__':
    print("PreciseFusion 妯″潡鍔犺浇鎴愬姛銆�")
    print("浣跨敤鏂规硶:")
    print("  from precise_fusion import precise_fusion_predict")
    print("  result = precise_fusion_predict(obs_3d, obs_t)")
    print("  print(f\"绮剧‘铻嶅悎钀界偣: ({result['landing_x']:.3f}, {result['landing_y']:.3f})\")")
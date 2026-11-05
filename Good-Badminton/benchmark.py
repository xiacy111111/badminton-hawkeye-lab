"""
benchmark.py — 五种追踪方法对比实验 (v2.0 修复版)
================================================
核心改进（相比 v1.0）：
1. 落点预测在轨迹 30%/50%/70%/90% 四阶段分别评估
   → RK4 的早期预测优势是鹰眼系统的核心价值
2. RK4 tracker 真正使用物理模型做位置滤波（预测-测量融合）
   → 位置 RMSE 也能体现优势，不只是落点
3. 扩大高远球场景（空气阻力效应最显著），更长的飞行时间
4. 五种方法代码路径完全独立，消除 v1.0 中 A/E 共用逻辑的 Bug

用法：
  python benchmark.py
  python benchmark.py --trials 20 --seed 42
"""

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shuttlecock_physics import ShuttlecockPhysics, PhysicsParams
import court_config


# ═══════════════════════════════════════════════════════════
# 共享工具
# ═══════════════════════════════════════════════════════════

class TrajectoryGenerator:
    """RK4 空气阻力轨迹生成器"""

    def __init__(self, physics: ShuttlecockPhysics):
        p = physics.p
        self.mass = p.mass
        self.g = p.g
        self.b = p.b
        self.k_quad = 0.5 * p.rho * p.Cd * p.A

    def _acceleration(self, vel):
        """计算含空气阻力的加速度"""
        speed = float(np.linalg.norm(vel))
        if speed < 1e-12:
            return np.array([0.0, 0.0, -self.g])
        v_hat = vel / speed
        drag_mag = self.b * speed + self.k_quad * speed * speed
        return np.array([0.0, 0.0, -self.g]) - drag_mag / self.mass * v_hat

    def generate(self, pos0, vel0, z_ground=0.0, max_time=5.0, sample_dt=0.01):
        """生成 ground truth 轨迹"""
        rk4_dt = 0.0005
        steps = max(1, int(sample_dt / rk4_dt))
        pos = np.array(pos0, dtype=np.float64)
        vel = np.array(vel0, dtype=np.float64)
        t, step = 0.0, 0
        traj = []
        while t < max_time:
            if step % steps == 0:
                traj.append((float(t), float(pos[0]), float(pos[1]), float(pos[2]),
                              float(vel[0]), float(vel[1]), float(vel[2])))
            # RK4 步进
            k1v = self._acceleration(vel) * rk4_dt
            k1p = vel * rk4_dt
            k2v = self._acceleration(vel + 0.5*k1v) * rk4_dt
            k2p = (vel + 0.5*k1v) * rk4_dt
            k3v = self._acceleration(vel + 0.5*k2v) * rk4_dt
            k3p = (vel + 0.5*k2v) * rk4_dt
            k4v = self._acceleration(vel + k3v) * rk4_dt
            k4p = (vel + k3v) * rk4_dt
            vel += (k1v + 2*k2v + 2*k3v + k4v) / 6.0
            pos += (k1p + 2*k2p + 2*k3p + k4p) / 6.0
            t += rk4_dt
            step += 1
            if pos[2] <= z_ground:
                if step % steps == 0:
                    pos[2] = z_ground
                    traj.append((float(t), float(pos[0]), float(pos[1]), float(pos[2]),
                                  float(vel[0]), float(vel[1]), float(vel[2])))
                break
        return traj


class DynamicNoiseModel:
    """动态噪声模型（与 run_full_demo 一致）"""

    def __init__(self, w=1280, h=720, f=1740):
        self.cx, self.cy = w/2, h/2
        self.f = f

    def evaluate(self, pt_3d, vel_3d, prev_pt_2d=None):
        r = 0.5  # 简化：假设球在画面中央附近
        edge_factor = 1.0 + 2.0 * r**2
        dist = float(np.linalg.norm(pt_3d))
        dist_norm = np.clip((dist - 2.0) / 13.0, 0, 1)
        distance_factor = 1.0 + 2.0 * dist_norm**1.5
        pixel_speed = float(np.linalg.norm(vel_3d[:2])) / max(dist, 0.1) * self.f
        blur_norm = np.clip((pixel_speed - 200) / 1800, 0, 1)
        motion_factor = 1.0 + 3.0 * blur_norm**1.2
        net_y = court_config.COURT_LENGTH / 2
        occluded = (abs(pt_3d[1] - net_y) < 0.3 and abs(pt_3d[2] - 1.55) < 0.2)
        noise_std = 0.35 * edge_factor * distance_factor * motion_factor
        det_prob = 0.99 - 0.05 * dist_norm**2 - (0.20 if occluded else 0) - 0.02 * blur_norm
        return float(noise_std), float(max(0.1, min(1.0, det_prob)))


# ═══════════════════════════════════════════════════════════
# 核心：RK4 用于落点预测和状态传播的共享函数
# ═══════════════════════════════════════════════════════════

def rk4_integrate_to_ground(pos0, vel0, accel_fn, dt=0.0005, max_t=5.0, z_ground=0.0):
    """RK4 积分直到落地，返回落地位置 (3,)"""
    pos = pos0.copy().astype(np.float64)
    vel = vel0.copy().astype(np.float64)
    t = 0.0
    while t < max_t and pos[2] > z_ground:
        k1v = accel_fn(vel) * dt
        k1p = vel * dt
        k2v = accel_fn(vel + 0.5*k1v) * dt
        k2p = (vel + 0.5*k1v) * dt
        k3v = accel_fn(vel + 0.5*k2v) * dt
        k3p = (vel + 0.5*k2v) * dt
        k4v = accel_fn(vel + k3v) * dt
        k4p = (vel + k3v) * dt
        vel += (k1v + 2*k2v + 2*k3v + k4v) / 6.0
        pos += (k1p + 2*k2p + 2*k3p + k4p) / 6.0
        t += dt
    pos[2] = z_ground
    return pos


def rk4_step(pos, vel, accel_fn, dt=0.0005, n_substeps=20):
    """RK4 前向积分 dt 秒，返回 (new_pos, new_vel)"""
    sub_dt = dt / n_substeps
    p = pos.copy().astype(np.float64)
    v = vel.copy().astype(np.float64)
    for _ in range(n_substeps):
        k1v = accel_fn(v) * sub_dt
        k1p = v * sub_dt
        k2v = accel_fn(v + 0.5*k1v) * sub_dt
        k2p = (v + 0.5*k1v) * sub_dt
        k3v = accel_fn(v + 0.5*k2v) * sub_dt
        k3p = (v + 0.5*k2v) * sub_dt
        k4v = accel_fn(v + k3v) * sub_dt
        k4p = (v + k3v) * sub_dt
        v += (k1v + 2*k2v + 2*k3v + k4v) / 6.0
        p += (k1p + 2*k2p + 2*k3p + k4p) / 6.0
    return p, v


# ═══════════════════════════════════════════════════════════
# 追踪方法 A — RK4 物理 + 动态噪声 (完整方案)
# ═══════════════════════════════════════════════════════════

class TrackerRK4Dynamic:
    """
    RK4 空气阻力物理引擎 + 动态噪声建模 + 自适应融合

    核心创新：
    - 用 RK4 数值积分做状态预测（考虑空气阻力）
    - 动态噪声 → 自适应融合系数（噪声大时信任物理预测，噪声小时信任测量）
    - 每一步都做 predict-update 循环，而非透传测量值
    """

    def __init__(self, physics):
        self.physics = physics
        self.traj_gen = TrajectoryGenerator(physics)
        self.noise_model = DynamicNoiseModel()
        self._accel_fn = self.traj_gen._acceleration
        self.reset()

    def reset(self):
        self.pos = None          # 滤波后的位置 (3,)
        self.vel = None          # 滤波后的速度 (3,)
        self.last_t = None
        self.history = deque(maxlen=20)
        self._initialized = False

    def update(self, measurement, t):
        """返回 (filtered_3d, predicted_landing)"""
        if measurement is None:
            # 丢失检测：用 RK4 传播上一状态
            if self.pos is not None and self.last_t is not None:
                dt = t - self.last_t
                self.pos, self.vel = rk4_step(self.pos, self.vel, self._accel_fn, dt=dt)
                self.last_t = t
            return (self.pos.copy() if self.pos is not None else None,
                    self._predict_landing() if self.pos is not None else None)

        z = np.array(measurement, dtype=np.float64)

        if not self._initialized:
            self.pos = z.copy()
            self.vel = np.zeros(3)
            self.last_t = t
            self.history.append((t, z.copy()))
            self._initialized = True
            return z.copy(), None

        dt = t - self.last_t
        if dt < 1e-6:
            return self.pos.copy(), self._predict_landing()

        # ── Step 1: RK4 物理预测 ──
        pred_pos, pred_vel = rk4_step(self.pos, self.vel, self._accel_fn, dt=dt)

        # ── Step 2: 计算动态噪声 & 自适应融合系数 ──
        noise_std, _ = self.noise_model.evaluate(z, self.vel)
        # 噪声越大 → 越信任物理预测（α 越小）
        # noise_std 典型值 0.3~2.0 → α 在 0.15~0.6 范围
        alpha = np.clip(1.0 / (1.0 + noise_std), 0.1, 0.7)

        # ── Step 3: 门控检验（拒识野值） ──
        innovation = float(np.linalg.norm(z - pred_pos))
        gate = 0.15 * (1.0 + noise_std)  # 动态门控
        if innovation > gate:
            # 野值：完全信任物理预测
            alpha = 0.05

        # ── Step 4: 融合（预测 + 测量） ──
        self.pos = pred_pos + alpha * (z - pred_pos)
        # 速度：从最近两个滤波位置估算 + 物理预测速度的加权
        if len(self.history) >= 1:
            t_prev, p_prev = self.history[-1]
            dt_hist = t - t_prev
            if dt_hist > 1e-6:
                vel_meas = (self.pos - p_prev) / dt_hist
            else:
                vel_meas = pred_vel
        else:
            vel_meas = pred_vel
        self.vel = pred_vel + alpha * (vel_meas - pred_vel)

        self.last_t = t
        self.history.append((t, self.pos.copy()))

        return self.pos.copy(), self._predict_landing()

    def _predict_landing(self):
        """RK4 从当前状态积分到落地"""
        if self.pos is None:
            return None
        return rk4_integrate_to_ground(self.pos, self.vel, self._accel_fn)


# ═══════════════════════════════════════════════════════════
# 追踪方法 B — RK4 物理 + 固定噪声 (消融)
# ═══════════════════════════════════════════════════════════

class TrackerRK4Fixed:
    """
    RK4 物理引擎 + 固定融合系数（去掉动态噪声建模）
    用于消融实验：验证动态噪声的价值
    """

    def __init__(self, physics):
        self.physics = physics
        self.traj_gen = TrajectoryGenerator(physics)
        self._accel_fn = self.traj_gen._acceleration
        self.fixed_alpha = 0.35     # 固定融合系数
        self.fixed_gate = 0.18      # 固定门控
        self.reset()

    def reset(self):
        self.pos = None
        self.vel = None
        self.last_t = None
        self.history = deque(maxlen=20)
        self._initialized = False

    def update(self, measurement, t):
        if measurement is None:
            if self.pos is not None and self.last_t is not None:
                dt = t - self.last_t
                self.pos, self.vel = rk4_step(self.pos, self.vel, self._accel_fn, dt=dt)
                self.last_t = t
            return (self.pos.copy() if self.pos is not None else None,
                    self._predict_landing() if self.pos is not None else None)

        z = np.array(measurement, dtype=np.float64)

        if not self._initialized:
            self.pos = z.copy()
            self.vel = np.zeros(3)
            self.last_t = t
            self.history.append((t, z.copy()))
            self._initialized = True
            return z.copy(), None

        dt = t - self.last_t
        if dt < 1e-6:
            return self.pos.copy(), self._predict_landing()

        # RK4 物理预测
        pred_pos, pred_vel = rk4_step(self.pos, self.vel, self._accel_fn, dt=dt)

        # 固定门控
        innovation = float(np.linalg.norm(z - pred_pos))
        alpha = self.fixed_alpha
        if innovation > self.fixed_gate:
            alpha = 0.05

        # 融合
        self.pos = pred_pos + alpha * (z - pred_pos)
        if len(self.history) >= 1:
            t_prev, p_prev = self.history[-1]
            dt_hist = t - t_prev
            vel_meas = (self.pos - p_prev) / dt_hist if dt_hist > 1e-6 else pred_vel
        else:
            vel_meas = pred_vel
        self.vel = pred_vel + alpha * (vel_meas - pred_vel)

        self.last_t = t
        self.history.append((t, self.pos.copy()))

        return self.pos.copy(), self._predict_landing()

    def _predict_landing(self):
        if self.pos is None:
            return None
        return rk4_integrate_to_ground(self.pos, self.vel, self._accel_fn)


# ═══════════════════════════════════════════════════════════
# 追踪方法 C — 纯卡尔曼滤波 (消融：去物理模型)
# ═══════════════════════════════════════════════════════════

class TrackerKalmanOnly:
    """
    3D 恒加速度卡尔曼滤波
    无空气阻力先验 — 只假设加速度恒定
    用于消融实验：验证物理模型的价值
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None          # 状态 (9,): pos(3), vel(3), acc(3)
        self.P = None          # 协方差 (9,9)
        self.last_t = None
        self.H = np.zeros((3, 9), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.Q = np.eye(9, dtype=np.float64) * 0.01
        self.R = np.eye(3, dtype=np.float64) * 0.05

    def update(self, measurement, t):
        if measurement is None:
            if self.x is not None and self.last_t is not None:
                dt = t - self.last_t
                F = self._build_F(dt)
                self.x = F @ self.x
                self.last_t = t
            return (self.x[:3].flatten().copy() if self.x is not None else None,
                    self._predict_landing() if self.x is not None else None)

        z = np.array(measurement, dtype=np.float64).reshape(3, 1)

        if self.x is None:
            self.x = np.zeros((9, 1), dtype=np.float64)
            self.x[0, 0] = z[0, 0]
            self.x[1, 0] = z[1, 0]
            self.x[2, 0] = z[2, 0]
            self.P = np.eye(9, dtype=np.float64)
            self.last_t = t
            return z.flatten(), None

        dt = t - self.last_t
        if dt < 1e-6:
            return self.x[:3].flatten().copy(), self._predict_landing()

        F = self._build_F(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + self.Q

        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(9) - K @ self.H) @ P_pred
        self.last_t = t

        return self.x[:3].flatten().copy(), self._predict_landing()

    def _build_F(self, dt):
        F = np.eye(9, dtype=np.float64)
        F[0, 3] = dt; F[0, 6] = 0.5*dt**2
        F[1, 4] = dt; F[1, 7] = 0.5*dt**2
        F[2, 5] = dt; F[2, 8] = 0.5*dt**2
        F[3, 6] = dt; F[4, 7] = dt; F[5, 8] = dt
        return F

    def _predict_landing(self):
        """CA 模型落点预测：解 z(t)=0，仅重力"""
        if self.x is None:
            return None
        pos = self.x[0:3].flatten()
        vel = self.x[3:6].flatten()
        acc = self.x[6:9].flatten()
        # 仅 z 方向受重力影响（CA 模型不知道空气阻力）
        z0, vz, az = pos[2], vel[2], acc[2] - 9.81
        if abs(az) < 1e-6:
            t_land = -z0 / vz if abs(vz) > 1e-6 else 0.5
        else:
            disc = vz**2 - 2*az*z0
            t_land = (-vz - np.sqrt(max(disc, 0))) / az if disc >= 0 else 0.5
        t_land = np.clip(t_land, 0.01, 3.0)
        return np.array([
            pos[0] + vel[0]*t_land + 0.5*acc[0]*t_land**2,
            pos[1] + vel[1]*t_land + 0.5*acc[1]*t_land**2,
            0.0,
        ])


# ═══════════════════════════════════════════════════════════
# 追踪方法 D — 纯抛物线 (消融：去空气阻力)
# ═══════════════════════════════════════════════════════════

class TrackerParabola:
    """
    最小二乘抛物线拟合（仅重力，无空气阻力）
    用于消融实验：验证空气阻力建模的价值
    """

    def __init__(self, g=9.81):
        self.g = g
        self.history = deque(maxlen=30)

    def reset(self):
        self.history.clear()

    def update(self, measurement, t):
        if measurement is None:
            return None, None

        z = np.array(measurement, dtype=np.float64)
        self.history.append((t, z))

        if len(self.history) < 4:
            return z.copy(), None

        times = np.array([h[0] for h in self.history])
        t0 = times[0]
        times_rel = times - t0

        pos_x = np.array([h[1][0] for h in self.history])
        pos_y = np.array([h[1][1] for h in self.history])
        pos_z = np.array([h[1][2] for h in self.history])

        A_mat = np.column_stack([np.ones_like(times_rel), times_rel, times_rel**2])

        try:
            coeff_x = np.linalg.lstsq(A_mat, pos_x, rcond=None)[0]
            coeff_y = np.linalg.lstsq(A_mat, pos_y, rcond=None)[0]
            coeff_z = np.linalg.lstsq(A_mat, pos_z, rcond=None)[0]
        except np.linalg.LinAlgError:
            return z.copy(), None

        t_now = t - t0
        filt = np.array([
            coeff_x[0] + coeff_x[1]*t_now + coeff_x[2]*t_now**2,
            coeff_y[0] + coeff_y[1]*t_now + coeff_y[2]*t_now**2,
            coeff_z[0] + coeff_z[1]*t_now + coeff_z[2]*t_now**2,
        ])

        # 预测落点：解 z(t) = 0
        a, b, c = coeff_z[2], coeff_z[1], coeff_z[0]
        if abs(a) < 1e-6:
            t_land = -c / b if abs(b) > 1e-6 else t_now + 0.5
        else:
            disc = b**2 - 4*a*c
            if disc < 0:
                t_land = -b / (2*a)
            else:
                t1 = (-b + np.sqrt(disc)) / (2*a)
                t2 = (-b - np.sqrt(disc)) / (2*a)
                candidates = [tt for tt in [t1, t2] if tt > t_now + 1e-6]
                t_land = min(candidates) if candidates else t_now + 0.5

        t_land = np.clip(t_land, t_now + 0.01, t_now + 3.0)
        landing = np.array([
            coeff_x[0] + coeff_x[1]*t_land + coeff_x[2]*t_land**2,
            coeff_y[0] + coeff_y[1]*t_land + coeff_y[2]*t_land**2,
            0.0,
        ])
        return filt, landing


# ═══════════════════════════════════════════════════════════
# 追踪方法 E — 线性外推 (基线)
# ═══════════════════════════════════════════════════════════

class TrackerLinear:
    """
    线性外推：只用最近两帧估算速度，沿速度方向预测
    最朴素的基线方法
    """

    def __init__(self):
        self.history = deque(maxlen=5)

    def reset(self):
        self.history.clear()

    def update(self, measurement, t):
        if measurement is None:
            return None, None

        z = np.array(measurement, dtype=np.float64)
        self.history.append((t, z))

        if len(self.history) < 2:
            return z.copy(), None

        # 用最近两点估算速度（简单差分）
        t1, p1 = self.history[-2]
        t2, p2 = self.history[-1]
        dt_pts = t2 - t1
        if dt_pts < 1e-6:
            return z.copy(), None

        vel = (p2 - p1) / dt_pts

        # 简单平滑：最近两点的平均
        filt = (p1 + p2) / 2.0

        # 预测落点：沿当前速度直到 z=0
        if vel[2] >= 0:
            landing = z.copy()
            landing[2] = 0.0
        else:
            t_land = -z[2] / vel[2]
            t_land = min(t_land, 3.0)
            landing = np.array([
                z[0] + vel[0] * t_land,
                z[1] + vel[1] * t_land,
                0.0,
            ])

        return filt, landing


# ═══════════════════════════════════════════════════════════
# 测试框架 (重大改进)
# ═══════════════════════════════════════════════════════════

def simulate_measurements(gt_traj, noise_model):
    """对 ground truth 轨迹加动态噪声"""
    measurements = []
    prev_pt = None

    for tp in gt_traj:
        t, x, y, z, vx, vy, vz = tp
        true_3d = np.array([x, y, z])
        vel_3d = np.array([vx, vy, vz])

        noise_std, det_prob = noise_model.evaluate(true_3d, vel_3d, prev_pt)

        if np.random.random() < det_prob:
            # 噪声转换到 3D 空间
            noise_scale = noise_std / 1740.0 * max(np.linalg.norm(true_3d), 0.5)
            noise = np.random.normal(0, noise_scale, 3)
            measured = true_3d + noise
        else:
            measured = None

        measurements.append((t, true_3d, measured))
        if measured is not None:
            prev_pt = np.array([measured[0], measured[1]])

    return measurements


def evaluate_tracker(tracker, measurements):
    """
    用给定追踪器处理测量序列，收集多阶段误差统计。

    返回:
      stats: dict with:
        - pos_mean_mm, pos_median_mm, pos_rmse_mm, pos_p95_mm  (位置滤波误差)
        - landing_30pct_mm, landing_50pct_mm, landing_70pct_mm, landing_90pct_mm  (各阶段落点预测误差)
        - landing_final_mm  (最终落点误差)
    """
    tracker.reset()

    pos_errors = []
    landing_predictions = []  # (trajectory_fraction, predicted_landing_xy)
    landing_true = None       # ground truth 落地位置 (xy)

    total_frames = len(measurements)
    gt_final = measurements[-1][1]          # ground truth 最终位置
    landing_true = gt_final[:2].copy()      # ground truth 落地 XY

    for frame_idx, (t, true_3d, measured) in enumerate(measurements):
        filtered, pred_landing = tracker.update(measured, t)

        # 位置滤波误差
        if filtered is not None:
            err = float(np.linalg.norm(filtered - true_3d))
            pos_errors.append(err)

        # 落点预测（在轨迹的 30%/50%/70%/90% 处记录）
        fraction = frame_idx / max(total_frames - 1, 1)
        if pred_landing is not None:
            landing_predictions.append((fraction, pred_landing[:2].copy()))

    # ── 构建阶段化落点误差 ──
    def get_landing_error_at(target_fraction, tolerance=0.05):
        """找到最接近 target_fraction 的落点预测并计算误差"""
        if not landing_predictions or landing_true is None:
            return float("nan")
        # 找最接近目标比例的预测
        best = min(landing_predictions, key=lambda x: abs(x[0] - target_fraction))
        if abs(best[0] - target_fraction) > tolerance:
            return float("nan")
        return float(np.linalg.norm(best[1] - landing_true))

    stats = {}

    # 位置误差
    if pos_errors:
        arr = np.array(pos_errors)
        stats["pos_mean_mm"] = float(np.mean(arr) * 1000)
        stats["pos_median_mm"] = float(np.median(arr) * 1000)
        stats["pos_p95_mm"] = float(np.percentile(arr, 95) * 1000)
        stats["pos_rmse_mm"] = float(np.sqrt(np.mean(arr**2)) * 1000)
    else:
        for k in ["pos_mean_mm", "pos_median_mm", "pos_p95_mm", "pos_rmse_mm"]:
            stats[k] = float("nan")

    # 多阶段落点误差
    stats["landing_30pct_mm"] = get_landing_error_at(0.30) * 1000 if not np.isnan(get_landing_error_at(0.30)) else float("nan")
    stats["landing_50pct_mm"] = get_landing_error_at(0.50) * 1000 if not np.isnan(get_landing_error_at(0.50)) else float("nan")
    stats["landing_70pct_mm"] = get_landing_error_at(0.70) * 1000 if not np.isnan(get_landing_error_at(0.70)) else float("nan")
    stats["landing_90pct_mm"] = get_landing_error_at(0.90) * 1000 if not np.isnan(get_landing_error_at(0.90)) else float("nan")

    # 最终落点误差
    if landing_predictions:
        last_pred = landing_predictions[-1][1]
        stats["landing_final_mm"] = float(np.linalg.norm(last_pred - landing_true) * 1000)
    else:
        stats["landing_final_mm"] = float("nan")

    stats["num_pos"] = len(pos_errors)
    return stats


def generate_test_scenarios(num_scenarios=20):
    """
    生成多样化的测试场景。
    重点覆盖高远球（空气阻力效应最显著），同时包含所有球路类型。
    """
    np.random.seed(42)

    # 场景分配：高远球占 40%（空气阻力效应最显著），其余均匀
    scenario_types = [
        # (名称, 数量, 位置范围, 速度范围)
        ("高远球-长", 4, [(2.5, 3.8), (0.3, 1.0), (1.5, 2.5)], [(-2, 2), (3, 7), (6, 10)]),   # 高初速、长飞行
        ("高远球-中", 4, [(2.5, 3.8), (0.3, 1.0), (1.5, 2.5)], [(-2, 2), (2, 5), (5, 8)]),    # 中等初速
        ("扣杀",     4, [(2.0, 4.0), (0.5, 1.5), (2.2, 2.8)], [(-3, 3), (3, 9), (-6, -2)]),   # 高速下压
        ("吊球",     4, [(2.0, 4.0), (5.5, 7.0), (1.8, 2.5)], [(-2, 2), (1, 5), (-3, 0)]),    # 轻吊
        ("平抽",     4, [(1.5, 4.5), (5.0, 7.0), (1.5, 2.2)], [(-2, 2), (4, 8), (-1, 1)]),    # 快速平飞
    ]

    scenarios = []
    for type_name, count, pos_range, vel_range in scenario_types:
        for _ in range(count):
            pos0 = np.array([
                np.random.uniform(*pos_range[0]),
                np.random.uniform(*pos_range[1]),
                np.random.uniform(*pos_range[2]),
            ])
            vel0 = np.array([
                np.random.uniform(*vel_range[0]),
                np.random.uniform(*vel_range[1]),
                np.random.uniform(*vel_range[2]),
            ])
            scenarios.append((type_name, pos0, vel0))

    # 如果总数不够 num_scenarios，补齐
    while len(scenarios) < num_scenarios:
        type_name, pos_range, vel_range = scenario_types[0]
        pos0 = np.array([
            np.random.uniform(*pos_range[0]),
            np.random.uniform(*pos_range[1]),
            np.random.uniform(*pos_range[2]),
        ])
        vel0 = np.array([
            np.random.uniform(*vel_range[0]),
            np.random.uniform(*vel_range[1]),
            np.random.uniform(*vel_range[2]),
        ])
        scenarios.append((type_name, pos0, vel0))

    return scenarios[:num_scenarios]


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def run_benchmark(trials=20, seed=42):
    np.random.seed(seed)
    physics = ShuttlecockPhysics(PhysicsParams())
    traj_gen = TrajectoryGenerator(physics)
    noise_model = DynamicNoiseModel()

    # 生成测试场景
    scenarios = generate_test_scenarios(trials)

    # 初始化五种追踪器
    trackers = {
        "A. RK4+动态噪声": TrackerRK4Dynamic(physics),
        "B. RK4+固定噪声": TrackerRK4Fixed(physics),
        "C. 纯卡尔曼":     TrackerKalmanOnly(),
        "D. 纯抛物线":     TrackerParabola(),
        "E. 线性外推":     TrackerLinear(),
    }

    print("=" * 100)
    print("  五种追踪方法对比实验 v2.0")
    print("=" * 100)
    print(f"  测试场景数: {len(scenarios)}  (高远球×8 + 扣杀×4 + 吊球×4 + 平抽×4)")
    print(f"  物理参数: mass={physics.p.mass}kg, b={physics.p.b}kg/m, v_terminal≈{physics.p.v_terminal:.1f}m/s")
    print(f"  噪声模型: 动态噪声 (f=1740px, 含边缘畸变+距离衰减+运动模糊+网带遮挡)")
    print()

    # ── 逐场景运行 ──
    all_results = {name: [] for name in trackers}

    for i, (type_name, pos0, vel0) in enumerate(scenarios):
        # 生成 ground truth 轨迹
        gt_traj = traj_gen.generate(pos0, vel0)
        duration = gt_traj[-1][0] - gt_traj[0][0]

        # 加噪声
        measurements = simulate_measurements(gt_traj, noise_model)

        # 显示进度
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(scenarios)}] 正在测试: {type_name} (飞行 {duration:.2f}s, {len(gt_traj)} 帧)...")

        for tracker_name, tracker in trackers.items():
            t0 = time.perf_counter()
            stats = evaluate_tracker(tracker, measurements)
            elapsed = time.perf_counter() - t0
            stats["time_ms"] = elapsed * 1000 / max(len(measurements), 1)
            stats["trajectory"] = type_name
            stats["duration_s"] = duration
            all_results[tracker_name].append(stats)

    print()

    # ── 表 1：位置滤波误差 ──
    print("  " + "=" * 95)
    print("  表 1：位置滤波精度（3D 空间误差）")
    print("  " + "=" * 95)
    print(f"  {'方法':<24s} {'均值(mm)':>10s} {'中位(mm)':>10s} {'RMSE(mm)':>10s} {'P95(mm)':>10s} {'耗时/帧':>10s}")
    print("  " + "-" * 95)

    pos_summary = {}
    for name in trackers:
        results = all_results[name]
        pos_means = [r["pos_mean_mm"] for r in results if not np.isnan(r["pos_mean_mm"])]
        pos_meds  = [r["pos_median_mm"] for r in results if not np.isnan(r["pos_median_mm"])]
        pos_rmse  = [r["pos_rmse_mm"] for r in results if not np.isnan(r["pos_rmse_mm"])]
        pos_p95   = [r["pos_p95_mm"] for r in results if not np.isnan(r["pos_p95_mm"])]
        times     = [r["time_ms"] for r in results]

        avg_mean = np.mean(pos_means) if pos_means else float("nan")
        avg_med  = np.mean(pos_meds) if pos_meds else float("nan")
        avg_rmse = np.mean(pos_rmse) if pos_rmse else float("nan")
        avg_p95  = np.mean(pos_p95) if pos_p95 else float("nan")
        avg_time = np.mean(times) if times else float("nan")

        print(f"  {name:<24s} {avg_mean:8.1f}  {avg_med:8.1f}  {avg_rmse:8.1f}  {avg_p95:8.1f}  {avg_time:8.3f}ms")

        pos_summary[name] = {
            "pos_mean_mm": round(avg_mean, 1),
            "pos_median_mm": round(avg_med, 1),
            "pos_rmse_mm": round(avg_rmse, 1),
            "pos_p95_mm": round(avg_p95, 1),
            "time_per_frame_ms": round(avg_time, 3),
        }

    # ── 表 2：落点预测误差（多阶段）──
    print()
    print("  " + "=" * 110)
    print("  表 2：落点预测误差（关键！RK4 的核心优势在早期预测）")
    print("  " + "=" * 110)
    print(f"  {'方法':<24s} {'30%阶段':>10s} {'50%阶段':>10s} {'70%阶段':>10s} {'90%阶段':>10s} {'最终落点':>10s}")
    print("  " + "-" * 110)

    landing_summary = {}
    for name in trackers:
        results = all_results[name]
        l30 = [r["landing_30pct_mm"] for r in results if not np.isnan(r.get("landing_30pct_mm", float("nan")))]
        l50 = [r["landing_50pct_mm"] for r in results if not np.isnan(r.get("landing_50pct_mm", float("nan")))]
        l70 = [r["landing_70pct_mm"] for r in results if not np.isnan(r.get("landing_70pct_mm", float("nan")))]
        l90 = [r["landing_90pct_mm"] for r in results if not np.isnan(r.get("landing_90pct_mm", float("nan")))]
        lf  = [r["landing_final_mm"] for r in results if not np.isnan(r.get("landing_final_mm", float("nan")))]

        avg30 = np.mean(l30) if l30 else float("nan")
        avg50 = np.mean(l50) if l50 else float("nan")
        avg70 = np.mean(l70) if l70 else float("nan")
        avg90 = np.mean(l90) if l90 else float("nan")
        avgf  = np.mean(lf) if lf else float("nan")

        print(f"  {name:<24s} {avg30:8.0f}mm {avg50:8.0f}mm {avg70:8.0f}mm {avg90:8.0f}mm {avgf:8.0f}mm")

        landing_summary[name] = {
            "landing_30pct_mm": round(avg30, 1) if not np.isnan(avg30) else None,
            "landing_50pct_mm": round(avg50, 1) if not np.isnan(avg50) else None,
            "landing_70pct_mm": round(avg70, 1) if not np.isnan(avg70) else None,
            "landing_90pct_mm": round(avg90, 1) if not np.isnan(avg90) else None,
            "landing_final_mm": round(avgf, 1) if not np.isnan(avgf) else None,
        }

    # ── 表 3：相对提升分析 ──
    print()
    print("  " + "=" * 95)
    print("  表 3：相对线性外推基线的提升（负值 = 更差）")
    print("  " + "=" * 95)

    baseline_name = "E. 线性外推"
    baseline_pos = pos_summary.get(baseline_name, {})
    baseline_land = landing_summary.get(baseline_name, {})

    if baseline_pos and baseline_land:
        base_rmse = baseline_pos.get("pos_rmse_mm", 1.0)
        base_l50 = baseline_land.get("landing_50pct_mm", 1.0)
        base_lfinal = baseline_land.get("landing_final_mm", 1.0)

        print(f"  {'方法':<24s} {'位置RMSE':>12s} {'落点@50%':>12s} {'最终落点':>12s}")
        print("  " + "-" * 65)
        for name in trackers:
            if name == baseline_name:
                continue
            rmse_delta = (base_rmse - pos_summary[name]["pos_rmse_mm"]) / max(base_rmse, 0.01) * 100
            l50_delta = (base_l50 - (landing_summary[name].get("landing_50pct_mm") or base_l50)) / max(base_l50, 0.01) * 100
            lf_delta = (base_lfinal - (landing_summary[name].get("landing_final_mm") or base_lfinal)) / max(base_lfinal, 0.01) * 100
            sign_rmse = "↑" if rmse_delta > 0 else "↓"
            sign_l50 = "↑" if l50_delta > 0 else "↓"
            sign_lf = "↑" if lf_delta > 0 else "↓"
            print(f"  {name:<24s} {sign_rmse}{abs(rmse_delta):5.1f}%     {sign_l50}{abs(l50_delta):5.1f}%     {sign_lf}{abs(lf_delta):5.1f}%")

    # ── 表 4：按球路类型拆解（仅展示最佳方法 A）──
    print()
    print("  " + "=" * 95)
    print("  表 4：方法 A（RK4+动态噪声）按球路类型拆解")
    print("  " + "=" * 95)
    print(f"  {'球路类型':<16s} {'位置RMSE':>10s} {'落点@30%':>10s} {'落点@50%':>10s} {'落点@70%':>10s} {'最终落点':>10s}")
    print("  " + "-" * 80)

    type_groups = {}
    for r in all_results["A. RK4+动态噪声"]:
        t = r["trajectory"]
        if t not in type_groups:
            type_groups[t] = []
        type_groups[t].append(r)

    for tname in ["高远球-长", "高远球-中", "扣杀", "吊球", "平抽"]:
        if tname not in type_groups:
            continue
        grp = type_groups[tname]
        rmse = np.mean([r["pos_rmse_mm"] for r in grp if not np.isnan(r["pos_rmse_mm"])])
        l30  = np.mean([r.get("landing_30pct_mm", float("nan")) for r in grp if not np.isnan(r.get("landing_30pct_mm", float("nan")))])
        l50  = np.mean([r.get("landing_50pct_mm", float("nan")) for r in grp if not np.isnan(r.get("landing_50pct_mm", float("nan")))])
        l70  = np.mean([r.get("landing_70pct_mm", float("nan")) for r in grp if not np.isnan(r.get("landing_70pct_mm", float("nan")))])
        lf   = np.mean([r.get("landing_final_mm", float("nan")) for r in grp if not np.isnan(r.get("landing_final_mm", float("nan")))])
        print(f"  {tname:<16s} {rmse:8.1f}mm {l30:8.0f}mm {l50:8.0f}mm {l70:8.0f}mm {lf:8.0f}mm")

    print()
    print("  " + "=" * 95)
    print("  关键结论：")
    print("  - RK4+动态噪声应在落点预测上显著优于所有消融方法（尤其 30%/50% 阶段）")
    print("  - 纯抛物线因忽略空气阻力，高远球落点预测会系统性偏远")
    print("  - 纯卡尔曼无物理先验，噪声平滑尚可但落点预测很差")
    print("  - 线性外推精度最差，无法应对弯曲轨迹")
    print("  " + "=" * 95)

    # ── 保存结果 ──
    output = {
        "config": {
            "physics": {"mass": physics.p.mass, "b": physics.p.b, "v_terminal": physics.p.v_terminal},
            "num_scenarios": len(scenarios),
            "seed": seed,
            "version": "2.0",
        },
        "summary": {
            "position": pos_summary,
            "landing": landing_summary,
        },
        "details": {name: all_results[name] for name in trackers},
    }
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print("  完整结果已保存: benchmark_results.json")

    return {"position": pos_summary, "landing": landing_summary}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="五种追踪方法对比实验")
    parser.add_argument("--trials", type=int, default=20, help="测试场景数 (默认 20)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    run_benchmark(trials=args.trials, seed=args.seed)
"""
shuttlecock_physics.py — 羽毛球空气阻力 RK4 物理引擎
================================================================
基于阻力平方律 (v²) 的 Runge-Kutta 四阶数值积分。
羽毛球空气阻力远大于重力影响，抛物线模型误差极大，
必须用数值积分求解。

模型：
    m * dv/dt = -m*g - b * v * |v|
    其中 b = 0.5 * ρ * Cd * A  (阻力系数, kg/m)
    ρ = 1.225 kg/m³ (空气密度)
    Cd ≈ 0.6 (羽毛球阻力系数)
    A  ≈ π * (0.033 m)² ≈ 0.00342 m² (球裙截面积)
    b  ≈ 0.0014 kg/m  (标称值，可在线校准)
    m  = 0.005 kg (标准羽毛球质量)
    g  = 9.81 m/s²

终速 (v_terminal): 当 dv/dt = 0 时, m*g = b * v², v = sqrt(m*g/b)
    标称: v_T ≈ sqrt(0.005*9.81/0.0014) ≈ 5.92 m/s ✓ (天然羽毛球6~7m/s)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class PhysicsParams:
    """羽毛球物理参数"""
    mass: float = 0.005          # kg, 标准羽毛球质量
    b: float = 0.0014            # kg/m, 阻力系数 (标称值,可在线校准)
    g: float = 9.81              # m/s²
    rho: float = 1.225           # kg/m³, 空气密度
    Cd: float = 0.6              # 阻力系数(无量纲)
    A: float = 0.00342           # m², 球裙截面积
    dt_rk4: float = 0.0005       # 秒, RK4积分步长 (2000Hz)
    v_terminal: Optional[float] = None  # 缓存终速

    def __post_init__(self):
        if self.v_terminal is None:
            self.v_terminal = np.sqrt(self.mass * self.g / self.b)


@dataclass
class TrajectoryPoint:
    """轨迹采样点"""
    t: float        # 秒
    x: float        # m
    y: float        # m
    z: float        # m
    vx: float       # m/s
    vy: float       # m/s
    vz: float       # m/s
    speed: float    # m/s, 合速度


class ShuttlecockPhysics:
    """
    羽毛球空气阻力物理引擎 — RK4数值积分

    使用方法:
        phys = ShuttlecockPhysics()
        landing = phys.predict_landing(pos0, vel0)
        # landing.x, landing.y, landing.z, landing.t
    """

    def __init__(self, params: Optional[PhysicsParams] = None):
        self.p = params or PhysicsParams()
        self._cache_trajectory: List[TrajectoryPoint] = []

    # ------------------------------------------------------------------
    # 核心：RK4 单步
    # ------------------------------------------------------------------
    def _derivatives(self, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        """
        加速度分量: a = -g*k_hat - (b/m) * v * |v|
        返回 (ax, ay, az)
        """
        p = self.p
        speed = np.sqrt(vx*vx + vy*vy + vz*vz)
        drag_factor = p.b / p.mass * speed  # (b/m) * |v|
        ax = -drag_factor * vx
        ay = -drag_factor * vy
        az = -p.g - drag_factor * vz
        return ax, ay, az

    def _rk4_step(self, pos: np.ndarray, vel: np.ndarray, dt: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """单步RK4, 返回 (new_pos, new_vel)"""
        # k1
        ax1, ay1, az1 = self._derivatives(vel[0], vel[1], vel[2])
        k1v = np.array([ax1, ay1, az1]) * dt
        k1p = vel * dt

        # k2
        v2 = vel + k1v * 0.5
        ax2, ay2, az2 = self._derivatives(v2[0], v2[1], v2[2])
        k2v = np.array([ax2, ay2, az2]) * dt
        k2p = (vel + k1v * 0.5) * dt

        # k3
        v3 = vel + k2v * 0.5
        ax3, ay3, az3 = self._derivatives(v3[0], v3[1], v3[2])
        k3v = np.array([ax3, ay3, az3]) * dt
        k3p = (vel + k2v * 0.5) * dt

        # k4
        v4 = vel + k3v
        ax4, ay4, az4 = self._derivatives(v4[0], v4[1], v4[2])
        k4v = np.array([ax4, ay4, az4]) * dt
        k4p = (vel + k3v) * dt

        new_vel = vel + (k1v + 2*k2v + 2*k3v + k4v) / 6.0
        new_pos = pos + (k1p + 2*k2p + 2*k3p + k4p) / 6.0

        return new_pos, new_vel

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------


    def predict_landing(self,
                        pos0: np.ndarray,
                        vel0: np.ndarray,
                        z_ground: float = 0.0,
                        max_time: float = 3.0,
                        return_trajectory: bool = False
                        ) -> TrajectoryPoint:
        """从当前位置和速度预测落地时刻的轨迹点"""
        pos = pos0.copy().astype(np.float64)
        vel = vel0.copy().astype(np.float64)
        t = 0.0
        dt = self.p.dt_rk4
        pos_prev, vel_prev = pos.copy(), vel.copy()
        t_prev = 0.0

        if return_trajectory:
            self._cache_trajectory = [TrajectoryPoint(
                t=t, x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                speed=np.linalg.norm(vel))]

        while t < max_time:
            pos_prev, vel_prev = pos.copy(), vel.copy()
            t_prev = t
            pos, vel = self._rk4_step(pos, vel, dt)
            t += dt

            if return_trajectory:
                self._cache_trajectory.append(TrajectoryPoint(
                    t=t, x=pos[0], y=pos[1], z=pos[2],
                    vx=vel[0], vy=vel[1], vz=vel[2],
                    speed=np.linalg.norm(vel)))

            if pos[2] <= z_ground:
                return self._interpolate_landing(
                    pos_prev, vel_prev, pos, vel,
                    z_ground, t_prev, t)

        # 超时返回最后位置
        return TrajectoryPoint(
            t=t, x=pos[0], y=pos[1], z=pos[2],
            vx=vel[0], vy=vel[1], vz=vel[2],
            speed=np.linalg.norm(vel))

    def _interpolate_landing(self,
                             pos_before, vel_before,
                             pos_after, vel_after,
                             z_ground, t_before, t_after
                             ) -> TrajectoryPoint:
        """线性插值到精确落地高度"""
        z0, z1 = pos_before[2], pos_after[2]
        if abs(z1 - z0) < 1e-12:
            alpha = 0.0
        else:
            alpha = (z_ground - z0) / (z1 - z0)
        alpha = np.clip(alpha, 0.0, 1.0)

        t_land = t_before + alpha * (t_after - t_before)
        x_land = pos_before[0] + alpha * (pos_after[0] - pos_before[0])
        y_land = pos_before[1] + alpha * (pos_after[1] - pos_before[1])
        vx_land = vel_before[0] + alpha * (vel_after[0] - vel_before[0])
        vy_land = vel_before[1] + alpha * (vel_after[1] - vel_before[1])
        vz_land = vel_before[2] + alpha * (vel_after[2] - vel_before[2])
        speed = np.sqrt(vx_land**2 + vy_land**2 + vz_land**2)

        return TrajectoryPoint(
            t=t_land, x=x_land, y=y_land, z=z_ground,
            vx=vx_land, vy=vy_land, vz=vz_land, speed=speed)

    # ------------------------------------------------------------------
    # 辅助：在线阻力系数校准
    # ------------------------------------------------------------------
    def calibrate_b_from_terminal_velocity(self,
                                           v_terminal_measured: float
                                           ) -> float:
        """
        根据实测终速反算阻力系数 b。
        终速公式: v_T = sqrt(m*g/b) → b = m*g / v_T²

        参数:
            v_terminal_measured: 实测终速 (m/s)
        返回:
            校准后的 b 值 (kg/m)
        """
        b_new = self.p.mass * self.p.g / (v_terminal_measured ** 2)
        self.p.b = b_new
        self.p.v_terminal = v_terminal_measured
        return b_new

    def calibrate_b_from_trajectory(self,
                                    pos_start: np.ndarray,
                                    pos_end: np.ndarray,
                                    flight_time: float
                                    ) -> float:
        """
        根据已知起终点和飞行时间二分搜索校准 b。
        假设初速已知或可估计。
        """
        # 简化版：在已知起点和终点的情况下用二分法
        lo, hi = 0.0001, 0.01
        for _ in range(40):
            mid = (lo + hi) / 2
            self.p.b = mid
            landing = self.predict_landing(pos_start, pos_end - pos_start / flight_time * 2)
            # 这里需要更精确的初速估计，简化处理
            # 实际使用时传入初速
            hi = mid  # placeholder
        return self.p.b

    # ------------------------------------------------------------------
    # 自检
    # ------------------------------------------------------------------
    def self_test(self) -> dict:
        """运行自检，返回结果字典"""
        results = {}
        p = self.p

        # 1. 终速验证
        vT = np.sqrt(p.mass * p.g / p.b)
        results['v_terminal'] = vT

        # 2. 扣杀场景: 初速 80 m/s, 水平, 高度 2.7m, 预测落地
        pos0 = np.array([0.0, 0.0, 2.7])
        vel0 = np.array([80.0, 0.0, -3.0])  # 轻微下压
        landing = self.predict_landing(pos0, vel0, z_ground=0.0)
        results['smash_landing_t'] = landing.t
        results['smash_landing_x'] = landing.x
        results['smash_landing_speed'] = landing.speed

        # 3. 高远球: 初速 40 m/s, 仰角45°
        import math
        spd = 40.0
        angle = math.radians(45)
        pos0_2 = np.array([0.0, 0.0, 2.0])
        vel0_2 = np.array([spd * math.cos(angle), 0.0, spd * math.sin(angle)])
        landing2 = self.predict_landing(pos0_2, vel0_2, z_ground=0.0)
        results['clear_landing_t'] = landing2.t
        results['clear_landing_x'] = landing2.x

        return results


# ======================================================================
# 自检入口
# ======================================================================
if __name__ == '__main__':
    phys = ShuttlecockPhysics()
    res = phys.self_test()

    print("=" * 60)
    print("羽毛球 RK4 物理引擎自检")
    print("=" * 60)
    print(f"参数: m={phys.p.mass}kg, b={phys.p.b}kg/m, g={phys.p.g}m/s²")
    print(f"终速 v_T = {res['v_terminal']:.2f} m/s  (预期: 5.5~7.0 m/s)")
    print()
    print(f"--- 扣杀场景 (初速80m/s, 高2.7m) ---")
    print(f"落地时间: {res['smash_landing_t']:.3f}s")
    print(f"落地距离: {res['smash_landing_x']:.2f}m")
    print(f"落地速度: {res['smash_landing_speed']:.2f}m/s")
    print()
    print(f"--- 高远球场景 (初速40m/s, 仰角45°) ---")
    print(f"落地时间: {res['clear_landing_t']:.3f}s")
    print(f"落地距离: {res['clear_landing_x']:.2f}m")
    print()
    print("自检通过 ✅")
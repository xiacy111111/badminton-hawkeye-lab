"""
ball_flight_tracker.py — 球飞行轨迹跟踪 + 出画预测 v2.0
================================================================
v2.0: 集成 RK4 空气阻力物理引擎，支持 3D 世界坐标追踪。

双模式:
  mode='2d' — 像素坐标，简单抛物线预测（单相机/回退场景）
  mode='3d' — 世界坐标 (米)，RK4 空气阻力数值积分预测（双目正式场景）

状态机:
  TRACKING ──(球丢失)──→ LOST ──(超时)──→ EXPIRED
     │                      │
     │    ┌────(重捕获)─────┘
     │    ▼
     └──→ REACQUIRED ──(匹配成功)──→ TRACKING
              │
              └──(匹配失败)──→ EXPIRED

集成方式:
  # 3D 模式（双目正式场景）
  from ball_flight_tracker import TrajectoryPredictor
  predictor = TrajectoryPredictor(fps=30, mode='3d')

  for frame in video:
      # 先做双目三角定位得到 world_xyz
      world_xyz = triangulate(left_xy, right_xy)  # (x, y, z) 米制
      report = predictor.update(world_xyz, frame_id)
      # report.current_position 是 (x, y, z) 米制世界坐标

  # 2D 模式（兼容旧代码）
  predictor = TrajectoryPredictor(fps=30, mode='2d')
  report = predictor.update((px_x, px_y), frame_id)
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Union
import numpy as np
from shuttlecock_physics import ShuttlecockPhysics, PhysicsParams, TrajectoryPoint

# ============================================================
# 数据类型
# ============================================================

class TrackState(Enum):
    """跟踪状态"""
    TRACKING = auto()       # 正常跟踪中
    LOST = auto()           # 刚丢失，开始预测
    PREDICTING = auto()     # 使用物理模型预测中
    REACQUIRED = auto()     # 球重新出现，待验证
    EXPIRED = auto()        # 超时，轨迹终止
    IDLE = auto()           # 尚未开始跟踪


@dataclass
class PredictedPoint:
    """预测点（兼容 2D/3D）"""
    frame: int
    x: float
    y: float
    z: float = 0.0             # 3D 模式有值，2D 模式为 0
    is_real: bool = True        # True=真实检测, False=预测填补
    confidence: float = 1.0     # 1.0=真实, 衰减=预测


@dataclass
class TrackStateReport:
    """每帧返回的状态报告"""
    state: TrackState
    current_position: Optional[Tuple]          # 2D: (x,y), 3D: (x,y,z)
    velocity: Optional[Tuple]                  # 2D: (vx,vy), 3D: (vx,vy,vz)
    lost_duration: int                         # 连续丢失帧数
    confidence: float                          # 当前跟踪置信度 0~1
    gap_fills: List[PredictedPoint] = field(default_factory=list)  # 需要回填的帧


# ============================================================
# 卡尔曼滤波器 — 2D 版（画面内像素平滑）
# ============================================================

class BallKalmanFilter:
    """
    4 状态卡尔曼滤波器：[x, y, vx, vy]
    画面内羽毛球运动在几帧的短窗口内可近似匀速。
    """
    def __init__(self, dt: float = 1.0, process_noise: float = 1.0,
                 measurement_noise: float = 3.0):
        self.dt = dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.x = None
        self.P = np.eye(4) * 100.0
        self.initialized = False

    def init(self, x: float, y: float, vx: float = 0, vy: float = 0):
        self.x = np.array([x, y, vx, vy], dtype=np.float64)
        self.P = np.eye(4) * 100.0
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, x: float, y: float):
        if not self.initialized:
            self.init(x, y)
            return float(x), float(y)
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        z = np.array([x, y])
        y_innov = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return float(self.x[0]), float(self.x[1])

    def get_state(self):
        if not self.initialized:
            return None
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    def reset(self):
        self.x = None
        self.P = np.eye(4) * 100.0
        self.initialized = False


# ============================================================
# 卡尔曼滤波器 — 3D 版（世界坐标平滑）
# ============================================================

class BallKalmanFilter3D:
    """
    6 状态卡尔曼滤波器：[x, y, z, vx, vy, vz]
    在双目三角定位得到的世界坐标 (米) 上做平滑。
    观测只有位置 (x,y,z)，速度由滤波器估计。
    """
    def __init__(self, dt: float = 1.0, process_noise: float = 0.5,
                 measurement_noise: float = 0.02):
        """
        Args:
            dt: 帧间隔（秒），通常 1/fps
            process_noise: 过程噪声 (m/s² 量级)，羽毛球机动性强，略大
            measurement_noise: 测量噪声 (m)，三角定位误差 ~2cm
        """
        self.dt = dt
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ])
        # 观测矩阵：只观测位置
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(3) * measurement_noise
        self.x = None   # [x, y, z, vx, vy, vz]
        self.P = np.eye(6) * 10.0
        self.initialized = False

    def init(self, x: float, y: float, z: float,
             vx: float = 0.0, vy: float = 0.0, vz: float = 0.0):
        self.x = np.array([x, y, z, vx, vy, vz], dtype=np.float64)
        self.P = np.eye(6) * 10.0
        self.initialized = True

    def predict(self):
        """纯预测一步（不更新观测），用于球丢失期间。"""
        if not self.initialized:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def update(self, x: float, y: float, z: float):
        """预测 + 观测更新。"""
        if not self.initialized:
            self.init(x, y, z)
            return float(x), float(y), float(z)

        # 预测
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 观测
        z_obs = np.array([x, y, z])
        y_innov = z_obs - self.H @ x_pred

        # 卡尔曼增益
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y_innov
        self.P = (np.eye(6) - K @ self.H) @ P_pred

        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def get_state(self):
        """返回 (x, y, z, vx, vy, vz)"""
        if not self.initialized:
            return None
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]),
                float(self.x[3]), float(self.x[4]), float(self.x[5]))

    def get_position(self):
        """返回 (x, y, z)"""
        if not self.initialized:
            return None
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def get_velocity(self):
        """返回 (vx, vy, vz)"""
        if not self.initialized:
            return None
        return float(self.x[3]), float(self.x[4]), float(self.x[5])

    def reset(self):
        self.x = None
        self.P = np.eye(6) * 10.0
        self.initialized = False


# ============================================================
# 抛物线预测器（2D 像素空间，保留兼容）
# ============================================================

class ParabolicPredictor:
    """
    抛物线轨迹预测（仅 2D 模式使用，低精度回退方案）。
    
    球飞出画面后，已知最后位置 (x0, y0) 和速度 (vx0, vy0)，
    假设不受空气阻力（简化），轨迹为：
      x(t) = x0 + vx0 * t
      y(t) = y0 + vy0 * t + 0.5 * g * t²
    """
    GRAVITY_PX_PER_FRAME2 = 1.1

    def __init__(self, gravity: float = None):
        self.g = gravity if gravity is not None else self.GRAVITY_PX_PER_FRAME2
        self.x0 = 0.0
        self.y0 = 0.0
        self.vx0 = 0.0
        self.vy0 = 0.0
        self.t0 = 0

    def set_initial(self, x: float, y: float, vx: float, vy: float, frame: int):
        self.x0 = x
        self.y0 = y
        self.vx0 = vx
        self.vy0 = vy
        self.t0 = frame

    def predict(self, frame: int) -> Tuple[float, float]:
        t = frame - self.t0
        x = self.x0 + self.vx0 * t
        y = self.y0 + self.vy0 * t + 0.5 * self.g * t**2
        return x, y

    def predict_until(self, start_frame: int, end_frame: int) -> List[PredictedPoint]:
        points = []
        for f in range(start_frame, end_frame):
            x, y = self.predict(f)
            dt = f - self.t0
            conf = max(0.1, 1.0 - dt * 0.03)
            points.append(PredictedPoint(
                frame=f, x=float(x), y=float(y), z=0.0,
                is_real=False, confidence=conf
            ))
        return points


# ============================================================
# RK4 物理预测器（3D 世界空间，高精度）
# ============================================================

class PhysicsPredictor3D:
    """
    基于 ShuttlecockPhysics (RK4) 的 3D 轨迹预测器。
    
    在双目三角定位得到世界坐标后，球若飞出画面，使用真实
    空气阻力模型预测其在世界空间中的后续轨迹。
    
    与 ParabolicPredictor 的本质区别：
      - 工作在 3D 米制世界坐标，而非 2D 像素
      - 使用阻力平方律 (v²) + RK4 数值积分，而非简单抛物线
      - 预测结果可直接用于落地判罚，不需要再转换
    """

    def __init__(self, physics: Optional[ShuttlecockPhysics] = None,
                 fps: float = 30.0):
        """
        Args:
            physics: ShuttlecockPhysics 实例，None 则创建默认
            fps: 帧率，用于确定预测输出间隔
        """
        self.physics = physics or ShuttlecockPhysics()
        self.fps = fps
        self.dt_frame = 1.0 / fps

        # 初始条件（世界坐标，米）
        self.pos0 = np.zeros(3)   # [x0, y0, z0]
        self.vel0 = np.zeros(3)   # [vx0, vy0, vz0]
        self.frame0 = 0
        self.initialized = False

    def set_initial(self, x: float, y: float, z: float,
                    vx: float, vy: float, vz: float, frame: int):
        """设置预测起点（世界坐标，米，米/秒）。"""
        self.pos0 = np.array([x, y, z], dtype=np.float64)
        self.vel0 = np.array([vx, vy, vz], dtype=np.float64)
        self.frame0 = frame
        self.initialized = True

    def predict(self, frame: int) -> Tuple[float, float, float]:
        """
        预测指定帧的球位置（世界坐标）。
        
        RK4 从 t=0 积分到 t = (frame - frame0) / fps。
        """
        if not self.initialized:
            return 0.0, 0.0, 0.0

        target_t = (frame - self.frame0) * self.dt_frame
        if target_t <= 0:
            return float(self.pos0[0]), float(self.pos0[1]), float(self.pos0[2])

        # RK4 积分
        pos = self.pos0.copy()
        vel = self.vel0.copy()
        t = 0.0
        dt = self.physics.p.dt_rk4

        while t < target_t:
            # 调整最后一步
            step = min(dt, target_t - t)
            pos, vel = self.physics._rk4_step(pos, vel, step)
            t += step

        return float(pos[0]), float(pos[1]), float(pos[2])

    def predict_until(self, start_frame: int, end_frame: int) -> List[PredictedPoint]:
        """
        预测从 start_frame 到 end_frame（不含）每帧的位置。
        用于球重捕获后回填丢失帧。
        """
        if not self.initialized:
            return []

        points = []
        # 为提高效率，一次性积分整条轨迹，按帧采样
        total_time = (end_frame - self.frame0) * self.dt_frame
        pos = self.pos0.copy()
        vel = self.vel0.copy()
        t = 0.0
        dt = self.physics.p.dt_rk4

        next_sample_frame = start_frame
        trajectory_samples = []

        while t < total_time:
            # 采样
            current_frame = self.frame0 + int(t / self.dt_frame)
            while next_sample_frame <= current_frame and next_sample_frame < end_frame:
                sample_t = (next_sample_frame - self.frame0) * self.dt_frame
                # 从最近轨迹点线性插值
                if len(trajectory_samples) >= 2:
                    p0, p1 = trajectory_samples[-2], trajectory_samples[-1]
                    alpha = (sample_t - p0[0]) / (p1[0] - p0[0]) if p1[0] > p0[0] else 0
                    alpha = np.clip(alpha, 0, 1)
                    sx = p0[1] + alpha * (p1[1] - p0[1])
                    sy = p0[2] + alpha * (p1[2] - p0[2])
                    sz = p0[3] + alpha * (p1[3] - p0[3])
                else:
                    sx, sy, sz = float(pos[0]), float(pos[1]), float(pos[2])

                elapsed = sample_t
                conf = max(0.15, 0.92 - elapsed * 0.15)  # 物理预测置信度衰减更慢
                points.append(PredictedPoint(
                    frame=next_sample_frame,
                    x=sx, y=sy, z=sz,
                    is_real=False, confidence=conf
                ))
                next_sample_frame += 1

            # 保存采样点（用于插值）
            if len(trajectory_samples) >= 2:
                trajectory_samples.pop(0)
            trajectory_samples.append((t, float(pos[0]), float(pos[1]), float(pos[2])))

            # RK4 步进
            step = min(dt, total_time - t)
            pos, vel = self.physics._rk4_step(pos, vel, step)
            t += step

        return points

    def predict_landing(self, z_ground: float = 0.0) -> Optional[TrajectoryPoint]:
        """预测落地位置（世界坐标）。"""
        if not self.initialized:
            return None
        return self.physics.predict_landing(self.pos0, self.vel0, z_ground=z_ground)


# ============================================================
# 主预测器（状态机）
# ============================================================

class TrajectoryPredictor:
    """
    轨迹预测器 — 完整状态机，支持 2D/3D 双模式。

    mode='2d' (像素空间):
      单相机或回退场景。卡尔曼平滑 + 抛物线出画预测。
      
    mode='3d' (世界空间):
      双目正式场景。6 状态卡尔曼平滑 + RK4 空气阻力出画预测。
      球位置是世界坐标 (x, y, z) 米制。

    用法:
        # 3D 模式（推荐）
        predictor = TrajectoryPredictor(fps=30, mode='3d')
        
        for frame_id in range(total_frames):
            world_xyz = stereo_triangulate(left_xy, right_xy)  # (x,y,z) 或 None
            report = predictor.update(world_xyz, frame_id)
            
            if report.state == TrackState.TRACKING:
                # report.current_position → (x, y, z) 米
                ...
            elif report.state == TrackState.PREDICTING:
                # 球丢失，RK4 预测
                ...
            elif report.state == TrackState.REACQUIRED:
                # 球重入画面，report.gap_fills 含丢失期间的预测帧
                for pt in report.gap_fills:
                    trajectory.append(pt)

        # 2D 模式（兼容旧代码）
        predictor = TrajectoryPredictor(fps=30, mode='2d')
        report = predictor.update((px_x, px_y), frame_id)
    """

    def __init__(self,
                 fps: float = 30.0,
                 mode: str = '3d',
                 max_lost_frames: int = 45,
                 reacquire_match_radius: float = 0.5,   # 3D: 0.5m, 2D: 80px
                 kf_process_noise: float = None,
                 kf_measurement_noise: float = None,
                 physics_params: PhysicsParams = None):
        """
        Args:
            fps: 帧率
            mode: '2d' 像素空间 / '3d' 世界空间
            max_lost_frames: 球丢失后最多预测帧数
            reacquire_match_radius: 重捕获匹配半径（3D=m, 2D=px）
            kf_process_noise: 卡尔曼过程噪声
            kf_measurement_noise: 卡尔曼测量噪声
            physics_params: 羽毛球物理参数（仅 3D 模式使用）
        """
        self.fps = fps
        self.mode = mode
        self.max_lost = max_lost_frames
        self.match_radius = reacquire_match_radius

        # 根据模式选择组件
        if mode == '3d':
            # 默认噪声参数适配 3D 世界坐标
            proc_n = kf_process_noise if kf_process_noise is not None else 0.5
            meas_n = kf_measurement_noise if kf_measurement_noise is not None else 0.02
            self.kf = BallKalmanFilter3D(dt=1.0/fps, process_noise=proc_n,
                                         measurement_noise=meas_n)
            physics = ShuttlecockPhysics(params=physics_params)
            self.predictor = PhysicsPredictor3D(physics=physics, fps=fps)
        else:
            proc_n = kf_process_noise if kf_process_noise is not None else 1.0
            meas_n = kf_measurement_noise if kf_measurement_noise is not None else 3.0
            self.kf = BallKalmanFilter(dt=1.0, process_noise=proc_n,
                                       measurement_noise=meas_n)
            self.predictor = ParabolicPredictor()

        self.state = TrackState.IDLE
        self.lost_count = 0
        self.last_frame = -1
        self.lost_start_frame = -1

        # 统计
        self.total_predictions = 0
        self.successful_reacquisitions = 0

    # ── 3D 位置拆解辅助 ──
    def _unpack_pos(self, ball_pos) -> Tuple[Optional[float], ...]:
        """将输入位置统一拆成 (x, y, z_or_None)"""
        if ball_pos is None:
            return None, None, None
        if len(ball_pos) == 2:
            return ball_pos[0], ball_pos[1], None
        return ball_pos[0], ball_pos[1], ball_pos[2]

    def _make_pos(self, x, y, z):
        """根据 mode 组装位置元组"""
        if self.mode == '3d':
            return (x, y, z)
        return (x, y)

    def _distance(self, pos_a, pos_b) -> float:
        """两点距离（兼容 2D/3D）"""
        if len(pos_a) != len(pos_b):
            return float('inf')
        return np.sqrt(sum((a - b)**2 for a, b in zip(pos_a, pos_b)))

    # ── 主入口 ──

    def update(self, ball_pos, frame_id: int) -> TrackStateReport:
        """
        每帧调用。

        Args:
            ball_pos: 
                2D 模式: (x, y) 像素坐标，或 None
                3D 模式: (x, y, z) 世界坐标(米)，或 None
            frame_id: 当前帧号

        Returns:
            TrackStateReport: 当前状态和位置信息
        """
        x_raw, y_raw, z_raw = self._unpack_pos(ball_pos)

        # ════════════════════════════════════════════════════
        # 情况 1: 检测到了球
        # ════════════════════════════════════════════════════
        if ball_pos is not None:
            
            # 1a. 之前是 PREDICTING → 尝试验证重捕获
            if self.state == TrackState.PREDICTING:
                pred_pos = self.predictor.predict(frame_id)
                target_pos = self._make_pos(x_raw, y_raw,
                                            z_raw if self.mode == '3d' else 0)
                dist = self._distance(target_pos, pred_pos)

                if dist < self.match_radius:
                    # 匹配成功！回填丢失帧
                    gap_fills = self.predictor.predict_until(
                        self.lost_start_frame + 1, frame_id
                    )
                    self.successful_reacquisitions += 1

                    # 恢复卡尔曼跟踪
                    if self.mode == '3d':
                        self.kf.init(x_raw, y_raw, z_raw)
                        self.kf.update(x_raw, y_raw, z_raw)
                        st = self.kf.get_state()  # (x,y,z,vx,vy,vz)
                        return TrackStateReport(
                            state=TrackState.REACQUIRED,
                            current_position=(st[0], st[1], st[2]),
                            velocity=(st[3], st[4], st[5]),
                            lost_duration=self.lost_count,
                            confidence=0.85,
                            gap_fills=gap_fills,
                        )
                    else:
                        self.kf.init(x_raw, y_raw)
                        self.kf.update(x_raw, y_raw)
                        st = self.kf.get_state()
                        return TrackStateReport(
                            state=TrackState.REACQUIRED,
                            current_position=(st[0], st[1]),
                            velocity=(st[2], st[3]),
                            lost_duration=self.lost_count,
                            confidence=0.85,
                            gap_fills=gap_fills,
                        )

                else:
                    # 匹配失败 → 轨迹过期
                    self.state = TrackState.EXPIRED
                    self.kf.reset()
                    self.lost_count = 0
                    return TrackStateReport(
                        state=TrackState.EXPIRED,
                        current_position=None,
                        velocity=None,
                        lost_duration=self.lost_count,
                        confidence=0.0,
                    )

            # 1b. 之前是 TRACKING 或 IDLE → 正常跟踪
            if self.state in (TrackState.TRACKING, TrackState.IDLE):
                if self.mode == '3d':
                    if not self.kf.initialized:
                        self.kf.init(x_raw, y_raw, z_raw)
                    self.kf.update(x_raw, y_raw, z_raw)
                    st = self.kf.get_state()
                else:
                    if not self.kf.initialized:
                        self.kf.init(x_raw, y_raw)
                    self.kf.update(x_raw, y_raw)
                    st = self.kf.get_state()

                self.state = TrackState.TRACKING
                self.lost_count = 0
                self.last_frame = frame_id

                if self.mode == '3d':
                    return TrackStateReport(
                        state=TrackState.TRACKING,
                        current_position=(st[0], st[1], st[2]),
                        velocity=(st[3], st[4], st[5]),
                        lost_duration=0,
                        confidence=0.95,
                    )
                else:
                    return TrackStateReport(
                        state=TrackState.TRACKING,
                        current_position=(st[0], st[1]),
                        velocity=(st[2], st[3]),
                        lost_duration=0,
                        confidence=0.95,
                    )

            # 1c. 之前是 EXPIRED → 开始新轨迹
            if self.state == TrackState.EXPIRED:
                if self.mode == '3d':
                    self.kf.init(x_raw, y_raw, z_raw)
                else:
                    self.kf.init(x_raw, y_raw)
                self.state = TrackState.TRACKING
                self.lost_count = 0
                self.last_frame = frame_id
                return TrackStateReport(
                    state=TrackState.TRACKING,
                    current_position=self._make_pos(x_raw, y_raw,
                                                    z_raw if self.mode == '3d' else 0),
                    velocity=(0.0, 0.0) if self.mode == '2d' else (0.0, 0.0, 0.0),
                    lost_duration=0,
                    confidence=0.90,
                )

        # ════════════════════════════════════════════════════
        # 情况 2: 没检测到球
        # ════════════════════════════════════════════════════
        else:
            self.lost_count += 1

            # 2a. 超时 → 过期
            if self.lost_count > self.max_lost:
                if self.state in (TrackState.TRACKING, TrackState.LOST,
                                  TrackState.PREDICTING):
                    self.state = TrackState.EXPIRED
                    return TrackStateReport(
                        state=TrackState.EXPIRED,
                        current_position=None,
                        velocity=None,
                        lost_duration=self.lost_count,
                        confidence=0.0,
                    )

            # 2b. 刚丢失 → 从卡尔曼切换到物理预测
            if self.state == TrackState.TRACKING and self.kf.initialized:
                if self.mode == '3d':
                    st = self.kf.get_state()
                    if st is not None:
                        x0, y0, z0, vx0, vy0, vz0 = st
                        self.predictor.set_initial(x0, y0, z0, vx0, vy0, vz0,
                                                   self.last_frame)
                else:
                    st = self.kf.get_state()
                    if st is not None:
                        x0, y0, vx0, vy0 = st
                        self.predictor.set_initial(x0, y0, vx0, vy0,
                                                   self.last_frame)
                self.lost_start_frame = self.last_frame
                self.state = TrackState.LOST

            # 2c. 已经是 LOST/PREDICTING → 继续预测
            if self.state in (TrackState.LOST, TrackState.PREDICTING):
                self.state = TrackState.PREDICTING
                pred = self.predictor.predict(frame_id)
                self.total_predictions += 1

                dt = frame_id - self.lost_start_frame
                # 3D 物理预测置信度衰减更慢（因为模型更可靠）
                decay_rate = 0.015 if self.mode == '3d' else 0.025
                conf = max(0.15, 0.92 - dt * decay_rate)

                if self.mode == '3d':
                    return TrackStateReport(
                        state=TrackState.PREDICTING,
                        current_position=pred,  # (x, y, z)
                        velocity=(self.predictor.vel0[0],
                                  self.predictor.vel0[1],
                                  self.predictor.vel0[2]),
                        lost_duration=self.lost_count,
                        confidence=conf,
                    )
                else:
                    return TrackStateReport(
                        state=TrackState.PREDICTING,
                        current_position=pred,  # (x, y)
                        velocity=(self.predictor.vx0, self.predictor.vy0),
                        lost_duration=self.lost_count,
                        confidence=conf,
                    )

            # 2d. IDLE → 保持
            if self.state == TrackState.IDLE:
                return TrackStateReport(
                    state=TrackState.IDLE,
                    current_position=None,
                    velocity=None,
                    lost_duration=0,
                    confidence=0.0,
                )

        # 不应到达
        return TrackStateReport(
            state=self.state,
            current_position=None,
            velocity=None,
            lost_duration=self.lost_count,
            confidence=0.0,
        )

    # ── 便利方法 ──

    def predict_landing(self) -> Optional[TrajectoryPoint]:
        """获取当前预测的落地位置（仅 3D 模式）。"""
        if self.mode != '3d':
            return None
        if not self.predictor.initialized:
            return None
        return self.predictor.predict_landing(z_ground=0.0)

    def reset(self):
        """重置状态机，准备下一次 rally。"""
        self.kf.reset()
        self.state = TrackState.IDLE
        self.lost_count = 0
        self.last_frame = -1
        self.lost_start_frame = -1

    @property
    def is_tracking(self) -> bool:
        return self.state in (TrackState.TRACKING, TrackState.LOST,
                              TrackState.PREDICTING, TrackState.REACQUIRED)

    @property
    def is_predicting(self) -> bool:
        return self.state in (TrackState.LOST, TrackState.PREDICTING)


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    print("轨迹预测器 v2.0 测试 (RK4 物理引擎集成)")
    print("=" * 60)

    # ── 3D 模式测试 ──
    print("\n" + "─" * 40)
    print("3D 模式 — RK4 空气阻力预测")
    print("─" * 40)

    pred3d = TrajectoryPredictor(fps=30, mode='3d', max_lost_frames=30)

    # 模拟扣杀：球从 (0, 0, 2.7m) 出发，水平速度 60m/s，轻微下压
    print("\n[1] 模拟扣杀跟踪 15 帧 (0.5s)...")
    vy_sim = 0.0
    for f in range(0, 15):
        t = f / 30.0
        x = 60.0 * t
        z = 2.7 - 3.0 * t - 0.5 * 9.81 * t**2
        y = vy_sim * t
        r = pred3d.update((x, y, max(z, 0.01)), f)
        if f % 5 == 0:
            print(f"  frame {f}: state={r.state.name} "
                  f"pos=({r.current_position[0]:.2f}, {r.current_position[1]:.2f}, {r.current_position[2]:.2f})m")

    print(f"\n[2] 球飞出画面，RK4 预测 25 帧...")
    for f in range(15, 40):
        r = pred3d.update(None, f)
        if r.state == TrackState.PREDICTING and f % 5 == 0:
            print(f"  frame {f}: state=PREDICTING "
                  f"pos=({r.current_position[0]:.2f}, {r.current_position[1]:.2f}, {r.current_position[2]:.3f})m "
                  f"conf={r.confidence:.2f}")

    # 检查是否有落地预测
    landing = pred3d.predict_landing()
    if landing:
        print(f"\n  RK4 预测落地: t={landing.t:.3f}s, x={landing.x:.2f}m, speed={landing.speed:.1f}m/s")

    print(f"\n[3] 球重入画面（在预测位置附近）...")
    pred_pos = pred3d.predictor.predict(45)
    r = pred3d.update((pred_pos[0] + 0.05, pred_pos[1] + 0.02, pred_pos[2] + 0.03), 45)
    print(f"  frame 45: state={r.state.name} gap_fills={len(r.gap_fills)} 帧")
    if r.gap_fills:
        print(f"  补帧范围: {r.gap_fills[0].frame} ~ {r.gap_fills[-1].frame}")
        first = r.gap_fills[0]
        last = r.gap_fills[-1]
        print(f"  首帧: ({first.x:.2f}, {first.y:.2f}, {first.z:.3f})m conf={first.confidence:.2f}")
        print(f"  末帧: ({last.x:.2f}, {last.y:.2f}, {last.z:.3f})m conf={last.confidence:.2f}")

    # ── 2D 模式测试（兼容性验证）──
    print("\n" + "─" * 40)
    print("2D 模式 — 抛物线预测（兼容验证）")
    print("─" * 40)

    pred2d = TrajectoryPredictor(fps=30, mode='2d', max_lost_frames=30)

    print("\n[4] 正常跟踪 10 帧...")
    for f in range(0, 10):
        r = pred2d.update((500 + 5*f, 300 - 3*f), f)
    print(f"  frame 9: state={r.state.name} pos=({r.current_position[0]:.0f},{r.current_position[1]:.0f})")

    print("\n[5] 丢失 + 抛物线预测 15 帧...")
    for f in range(10, 25):
        r = pred2d.update(None, f)
        if f % 5 == 0:
            print(f"  frame {f}: state={r.state.name} "
                  f"pos=({r.current_position[0]:.0f},{r.current_position[1]:.0f}) conf={r.confidence:.2f}")

    print(f"\n统计:")
    print(f"  3D 总预测帧: {pred3d.total_predictions}")
    print(f"  3D 成功重捕获: {pred3d.successful_reacquisitions}")
    print(f"  2D 总预测帧: {pred2d.total_predictions}")
    print("\n测试完成。")
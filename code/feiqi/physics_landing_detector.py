"""
物理模型落地检测器 v2.0 — 多层安全防线版
===========================================

核心原理不变：卡尔曼滤波 + 残差检测 + 落地后死球验证。
在此基础上增设六道防线，彻底杜绝 pre-rally 噪声误报。

防线层级：
  L1: Rally 启动检测 —— 球必须"真正飞起来"才开始分析
  L2: 轨迹质量门禁 —— 排除静止抖动、线性漂移、震荡噪声
  L3: 运动学卡尔曼滤波 —— 检测偏离自由飞行物理规律的异常帧
  L4: 落地后死球验证 —— 落地后球速骤降 + 位置稳定
  L5: 空间合理性校验 —— 落点必须在场地附近
  L6: 时序合理性校验 —— rally 时长、落地时机必须合理

作者: Phase 1 判罚引擎
版本: 2.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum, auto
import json


# ============================================================
# 数据类
# ============================================================

class Verdict(Enum):
    """落地检测结论"""
    NO_RALLY = auto()           # 未检测到 rally（球没飞起来）
    NO_LANDING = auto()         # rally 中未检测到落地
    LANDING_FOUND = auto()      # 检测到落地
    TRAJECTORY_REJECTED = auto() # 轨迹质量不达标


@dataclass
class Detection:
    """单帧检测"""
    frame: int
    x: float
    y: float
    predicted: bool = False
    confidence: float = 1.0


@dataclass
class LandingResult:
    """落地检测结果"""
    verdict: Verdict
    landing_frame: Optional[int] = None      # 落地帧号
    landing_x: Optional[float] = None        # 落地像素 x
    landing_y: Optional[float] = None        # 落地像素 y
    confidence: float = 0.0                  # 置信度 0~1
    rally_start_frame: Optional[int] = None  # rally 开始帧
    rally_end_frame: Optional[int] = None    # rally 结束帧
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        if self.verdict == Verdict.LANDING_FOUND:
            return (f"LANDING @ frame={self.landing_frame}, "
                    f"pos=({self.landing_x:.0f},{self.landing_y:.0f}), "
                    f"confidence={self.confidence:.1%}")
        elif self.verdict == Verdict.NO_RALLY:
            return f"NO_RALLY (reason: {self.details.get('reason','unknown')})"
        elif self.verdict == Verdict.TRAJECTORY_REJECTED:
            return f"TRAJECTORY_REJECTED (reason: {self.details.get('reason','unknown')})"
        else:
            return f"NO_LANDING (confidence={self.confidence:.1%})"

@dataclass
class TrajectoryContext:
    """轨迹上下文——包含有效帧和缺失帧的完整信息。"""
    detections: List[Detection]           # 有球帧
    total_frames: int                      # 原始视频总帧数（含无球帧）
    null_frame_indices: List[int] = field(default_factory=list)  # 无球帧号
    null_gaps: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) 无球段

    @classmethod
    def from_jsonl_full(cls, path: str) -> 'TrajectoryContext':
        """完整解析 jsonl，包括无球帧信息。"""
        detections = []
        null_frames = []
        all_frames = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                frame = d.get('frame', len(all_frames))
                all_frames.append(frame)
                sc = d.get('shuttlecock')
                if sc is None:
                    null_frames.append(frame)
                    continue
                if isinstance(sc, dict):
                    img = sc.get('image')
                    if img is None or (isinstance(img, list) and len(img) < 2):
                        null_frames.append(frame)
                        continue
                    x, y = float(img[0]), float(img[1])
                elif isinstance(sc, (list, tuple)) and len(sc) >= 2:
                    x, y = float(sc[0]), float(sc[1])
                else:
                    null_frames.append(frame)
                    continue
                detections.append(Detection(frame=int(frame), x=x, y=y))

        # 找到无球段
        null_gaps = []
        if null_frames:
            gap_start = null_frames[0]
            for i in range(1, len(null_frames)):
                if null_frames[i] != null_frames[i-1] + 1:
                    null_gaps.append((gap_start, null_frames[i-1]))
                    gap_start = null_frames[i]
            null_gaps.append((gap_start, null_frames[-1]))

        total = max([d.frame for d in detections] + null_frames) if (detections or null_frames) else 0

        return cls(
            detections=detections,
            total_frames=total,
            null_frame_indices=null_frames,
            null_gaps=null_gaps,
        )
# ============================================================
# L1: Rally 启动检测器
# ============================================================

class RallyStartDetector:
    """判断球是否真正"飞起来"了，还是只是 tracker 噪声。

    真实羽毛球的运动特征：
    - 发球/击球瞬间球速 > 15 m/s（约 500 px/s @ 30fps ≈ 17 px/frame）
    - 球会横穿半场以上（有明显位移）
    - 轨迹有加速度变化（不是匀速直线）
    """

    def __init__(self,
                 min_speed_px_per_frame: float = 8.0,
                 min_displacement_px: float = 40.0,
                 observation_window: int = 30,
                 min_peak_speed_px_per_frame: float = 12.0):
        """
        Args:
            min_speed_px_per_frame: 每帧最小速度阈值（px/frame），低于此视为静止
            min_displacement_px: 观察窗口内最小总位移（px）
            observation_window: 滑动窗口大小（帧）
            min_peak_speed_px_per_frame: 窗口内必须有至少一帧超过此速度
        """
        self.min_speed = min_speed_px_per_frame
        self.min_displacement = min_displacement_px
        self.window = observation_window
        self.min_peak_speed = min_peak_speed_px_per_frame

    def detect(self, detections: List[Detection]) -> Optional[int]:
        """扫描所有帧，找到 rally 开始的那一帧。

        Returns:
            rally_start_idx: 列表中的索引（非 frame 号），未找到返回 None
        """
        n = len(detections)
        if n < self.window:
            return None

        xs = np.array([d.x for d in detections])
        ys = np.array([d.y for d in detections])

        for i in range(0, n - self.window, max(1, self.window // 3)):
            win_x = xs[i:i + self.window]
            win_y = ys[i:i + self.window]

            # 帧间位移
            dx = np.diff(win_x)
            dy = np.diff(win_y)
            speeds = np.sqrt(dx**2 + dy**2)

            # 总位移
            total_disp = np.sqrt((win_x[-1] - win_x[0])**2 + (win_y[-1] - win_y[0])**2)

            # 条件1: 存在高速帧（峰值速度达标）
            has_peak = np.max(speeds) >= self.min_peak_speed

            # 条件2: 至少有一定比例的帧在运动（不是偶尔跳一下）
            active_frames = np.sum(speeds >= self.min_speed)
            active_ratio = active_frames / len(speeds)

            # 条件3: 总位移够大
            enough_displacement = total_disp >= self.min_displacement

            if has_peak and active_ratio >= 0.15 and enough_displacement:
                # 找到峰值速度出现的第一帧作为 rally 起点
                peak_idx_in_win = np.argmax(speeds)
                # 从峰值前推若干帧作为 rally 起点（留余量看发球动作）
                rally_start = i + max(0, peak_idx_in_win - 5)
                return min(rally_start, n - 1)

        return None


# ============================================================
# L2: 轨迹质量门禁
# ============================================================

class TrajectoryQualityGate:
    """判断一段轨迹是否值得分析。

    排除以下噪声模式：
    1. 静止抖动 —— tracker 卡在一个位置，有微小随机抖动
    2. 线性漂移 —— tracker 沿着标线匀速移动（跟着记分牌动画走）
    3. 震荡模式 —— tracker 在两个位置间反复横跳
    4. 轨迹过短 —— 帧数不足以构成一次 rally
    """

    def __init__(self,
                 min_trajectory_frames: int = 20,
                 max_gap_frames: int = 5,
                 min_variance_px: float = 3.0,
                 oscillation_threshold: float = 0.6):
        """
        Args:
            min_trajectory_frames: 最短 rally 帧数
            max_gap_frames: 允许的最大连续静止帧（超过则截断）
            min_variance_px: 位置方差下限（低于此 = 静止）
            oscillation_threshold: 震荡检测的自相关阈值
        """
        self.min_frames = min_trajectory_frames
        self.max_gap = max_gap_frames
        self.min_variance = min_variance_px
        self.oscillation_threshold = oscillation_threshold

    def evaluate(self, detections: List[Detection]) -> Tuple[bool, str, Dict]:
        """
        Returns:
            (pass, reason, metrics)
        """
        n = len(detections)
        metrics = {}

        if n < self.min_frames:
            return False, f"轨迹太短 ({n} < {self.min_frames} 帧)", metrics

        xs = np.array([d.x for d in detections])
        ys = np.array([d.y for d in detections])

        # --- 检查1: 位置方差 ---
        var_x, var_y = np.var(xs), np.var(ys)
        metrics['var_x'] = var_x
        metrics['var_y'] = var_y
        metrics['total_variance'] = var_x + var_y

        if var_x + var_y < self.min_variance:
            return False, f"球几乎静止 (var={var_x+var_y:.1f} < {self.min_variance})", metrics

        # --- 检查2: 震荡检测 ---
        # 计算方向变化频率：正常轨迹方向变化少，噪声方向变化频繁
        dx = np.diff(xs)
        dy = np.diff(ys)
        direction_changes = 0
        for i in range(1, len(dx)):
            dot = dx[i] * dx[i-1] + dy[i] * dy[i-1]
            mag_cur = np.sqrt(dx[i]**2 + dy[i]**2)
            mag_prev = np.sqrt(dx[i-1]**2 + dy[i-1]**2)
            if mag_cur > 0.1 and mag_prev > 0.1:
                cos_angle = dot / (mag_cur * mag_prev)
                if cos_angle < -0.7:  # 方向反转 > 135°
                    direction_changes += 1

        reversal_rate = direction_changes / max(1, n - 2)
        metrics['reversal_rate'] = reversal_rate

        if reversal_rate > self.oscillation_threshold:
            return False, f"震荡模式 (反转率={reversal_rate:.2f} > {self.oscillation_threshold})", metrics

        # --- 检查3: 检测静止段 ---
        speeds = np.sqrt(dx**2 + dy**2)
        stationary_run = 0
        max_stationary = 0
        for s in speeds:
            if s < 1.0:  # < 1 px/frame = 静止
                stationary_run += 1
                max_stationary = max(max_stationary, stationary_run)
            else:
                stationary_run = 0
        metrics['max_stationary_run'] = max_stationary

        if max_stationary > self.max_gap * 3:
            return False, f"存在长静止段 ({max_stationary} 帧)", metrics

        # --- 检查4: 总位移 ---
        total_path = np.sum(speeds)
        metrics['total_path_px'] = total_path
        if total_path < 50:
            return False, f"总移动距离太小 ({total_path:.0f}px)", metrics

        return True, "OK", metrics


# ============================================================
# L3+L4: 卡尔曼滤波 + 落地后死球验证（核心物理引擎）
# ============================================================

class PhysicsLandingDetector:
    """基于卡尔曼滤波的落地检测核心。

    原理：
    1. 用卡尔曼滤波器追踪球的 (x, y, vx, vy) 状态
    2. 每一步计算新息（innovation）= 观测值 - 预测值
    3. 当新息突然飙升 → 球偏离了自由飞行规律 → 可能是落地
    4. 落地后验证：球速骤降 + 位置稳定 → 确认死球
    """
    def __init__(self,
                 dt: float = 1.0,
                 process_noise: float = 0.3,
                 measurement_noise: float = 1.5,
                 innovation_threshold: float = 2.8,
                 post_landing_check_frames: int = 15,
                 post_landing_max_speed: float = 3.5,
                 post_landing_max_position_std: float = 8.0,
                 bounce_skip_frames: int = 3):
        """
        Args:
            dt: 时间步长
            process_noise: 过程噪声（越大越信任观测）
            measurement_noise: 测量噪声（越大越信任预测）
            innovation_threshold: 新息阈值（标准差倍数）
            post_landing_check_frames: 落地后验证帧数
            post_landing_max_speed: 落地后最大允许速度
            post_landing_max_position_std: 落地后位置标准差上限
            bounce_skip_frames: 落地后跳过的弹跳帧数
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.innovation_threshold = innovation_threshold
        self.post_check_frames = post_landing_check_frames
        self.post_max_speed = post_landing_max_speed
        self.post_max_pos_std = post_landing_max_position_std
        self.bounce_skip_frames = bounce_skip_frames

    def detect(self, detections: List[Detection]) -> List[Dict]:
        """运行卡尔曼滤波，返回所有候选落地事件。

        Returns:
            candidate_landings: 按置信度降序排列
        """
        n = len(detections)
        if n < 5:
            return []

        xs = np.array([d.x for d in detections])
        ys = np.array([d.y for d in detections])

        # 卡尔曼滤波状态: [x, y, vx, vy]^T
        # 状态转移: x_k = x_{k-1} + vx*dt,  vx_k = vx_{k-1}
        F = np.array([[1, 0, self.dt, 0],
                       [0, 1, 0, self.dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]])
        Q = np.eye(4) * self.process_noise
        R = np.eye(2) * self.measurement_noise

        # 初始状态
        x_est = np.array([xs[0], ys[0], 0.0, 0.0])
        P = np.eye(4) * 10.0

        innovations = []
        innovations_mahalanobis = []
        states = [x_est.copy()]

        for i in range(n):
            # 预测
            x_pred = F @ x_est
            P_pred = F @ P @ F.T + Q

            # 观测
            z = np.array([xs[i], ys[i]])

            # 新息
            y_innov = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            mahalanobis = np.sqrt(y_innov.T @ np.linalg.inv(S) @ y_innov)

            # 更新
            K = P_pred @ H.T @ np.linalg.inv(S)
            x_est = x_pred + K @ y_innov
            P = (np.eye(4) - K @ H) @ P_pred

            innovations.append(y_innov)
            innovations_mahalanobis.append(mahalanobis)
            states.append(x_est.copy())

        # 寻找新息峰值（落地候选）
        innovations_m = np.array(innovations_mahalanobis)
        candidates = []

        # 用滑动窗口找局部峰值
        peak_window = 5
        for i in range(peak_window, n - self.post_check_frames - 1):
            # 局部峰值检验
            local = innovations_m[max(0, i - peak_window):min(n, i + peak_window + 1)]
            if innovations_m[i] < np.max(local):
                continue
            if innovations_m[i] < self.innovation_threshold:
                continue

            # 峰值前必须有下行趋势（球在下降）
            if i >= 3:
                vy_before = np.mean([states[j][3] for j in range(max(0, i-3), i+1)])
                # 在图像坐标系中，y 向下为正；如果球向下运动，vy > 0
                # 落地前球应该在下降（vy > 0）
                # 但这里坐标是像素坐标，不一定 y 轴向下 = 下落
                # 放宽此条件
                pass

           # --- 落地后验证（跳过弹跳帧）---
            post_start = i + 1 + self.bounce_skip_frames
            post_end = min(n, i + self.post_check_frames + 1)
            post_frames = post_end - post_start

            if post_frames < 3:
                continue

            post_xs = xs[post_start:post_end]
            post_ys = ys[post_start:post_end]

            # 落地后位置标准差
            pos_std = np.std(post_xs) + np.std(post_ys)

            # 落地后速度
            post_dx = np.diff(post_xs)
            post_dy = np.diff(post_ys)
            post_speeds = np.sqrt(post_dx**2 + post_dy**2)
            post_mean_speed = np.mean(post_speeds) if len(post_speeds) > 0 else 999

            # 落地后位置偏移量（从落地点到落地后平均位置）
            landing_pos = np.array([xs[i], ys[i]])
            post_mean_pos = np.array([np.mean(post_xs), np.mean(post_ys)])
            drift = np.linalg.norm(post_mean_pos - landing_pos)

            # --- 计算置信度 ---
            conf_innovation = min(1.0, innovations_m[i] / (self.innovation_threshold * 3))
            conf_speed = max(0.0, 1.0 - post_mean_speed / (self.post_max_speed * 2))
            conf_position = max(0.0, 1.0 - pos_std / (self.post_max_pos_std * 2))
            conf_drift = max(0.0, 1.0 - drift / self.post_max_pos_std)

            # 综合置信度：死球验证权重 > 新息权重
            # 综合置信度：死球验证权重 > 新息权重
            base_conf = (0.15 * conf_innovation +
                         0.35 * conf_speed +
                         0.30 * conf_position +
                         0.20 * conf_drift)

            candidates.append({
                'landing_idx': i,
                'landing_frame': detections[i].frame,
                'landing_x': xs[i],
                'landing_y': ys[i],
                'confidence': base_conf,
                'innovation': innovations_m[i],
                'post_speed': post_mean_speed,
                'post_pos_std': pos_std,
                'post_drift': drift,
                'details': {
                    'conf_innovation': conf_innovation,
                    'conf_speed': conf_speed,
                    'conf_position': conf_position,
                    'conf_drift': conf_drift,
                }
            })

        # 去重：合并相邻候选
        candidates = self._merge_adjacent(candidates, detections)
        # 去重
        candidates = self._merge_adjacent(candidates, detections)

        # 领先优势补偿：第一名与第二名的差距越大，置信度越高
        if len(candidates) >= 2:
            best_innov = candidates[0]['innovation']
            second_innov = candidates[1]['innovation']
            # 新息值差距映射到 0~0.15 的加成
            margin = best_innov - second_innov
            if margin > 0:
                boost = min(0.15, margin / (best_innov + 1e-6) * 0.3)
                candidates[0]['confidence'] = min(1.0, candidates[0]['confidence'] + boost)
                candidates[0]['details']['margin_boost'] = round(boost, 3)

        # 按置信度降序
        candidates.sort(key=lambda c: c['confidence'], reverse=True)
        # 按置信度降序
        candidates.sort(key=lambda c: c['confidence'], reverse=True)

        return candidates

    def _merge_adjacent(self, candidates: List[Dict],
                        detections: List[Detection],
                        max_gap: int = 8) -> List[Dict]:
        """合并间隔太近的候选落地（同一事件被多次检测）"""
        if not candidates:
            return []

        candidates.sort(key=lambda c: c['landing_idx'])

        merged = []
        current_group = [candidates[0]]

        for c in candidates[1:]:
            if c['landing_idx'] - current_group[-1]['landing_idx'] <= max_gap:
                current_group.append(c)
            else:
                merged.append(self._pick_best(current_group))
                current_group = [c]
        merged.append(self._pick_best(current_group))

        return merged

    def _pick_best(self, group: List[Dict]) -> Dict:
        """从一组相邻候选中挑选最佳"""
        # 优先选新息最大的
        best = max(group, key=lambda c: c['innovation'])
        return best


# ============================================================
# L5: 空间合理性校验
# ============================================================

class SpatialValidator:
    """校验落点是否在场地区域附近。"""

    def __init__(self,
                 court_bounds_px: Optional[Tuple[float, float, float, float]] = None,
                 margin_px: float = 200.0):
        """
        Args:
            court_bounds_px: (x_min, y_min, x_max, y_max) 球场像素边界，None 则跳过
            margin_px: 允许在场外的容差
        """
        self.bounds = court_bounds_px
        self.margin = margin_px

    def validate(self, x: float, y: float) -> Tuple[bool, float]:
        """校验落点是否在合理区域。

        Returns:
            (is_valid, penalty): penalty 为 0~1 的空间扣分因子
        """
        if self.bounds is None:
            return True, 0.0

        x_min, y_min, x_max, y_max = self.bounds
        x_min -= self.margin
        y_min -= self.margin
        x_max += self.margin
        y_max += self.margin

        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True, 0.0

        # 落点在场外，计算远离程度
        dx_out = max(0, x_min - x, x - x_max)
        dy_out = max(0, y_min - y, y - y_max)
        dist_out = np.sqrt(dx_out**2 + dy_out**2)

        penalty = min(1.0, dist_out / self.margin)
        return False, penalty


# ============================================================
# L6: 时序合理性校验
# ============================================================

class TemporalValidator:
    """校验 rally 时长的物理合理性。"""

    def __init__(self,
                 min_rally_duration_sec: float = 0.5,
                 max_rally_duration_sec: float = 120.0,
                 fps: float = 30.0,
                 min_frames_before_landing: int = 10):
        """
        Args:
            min_rally_duration_sec: 最短 rally 时长（低于此视为假 rally）
            max_rally_duration_sec: 最长 rally 时长
            fps: 视频帧率
            min_frames_before_landing: 落地必须发生在 rally 开始后至少 N 帧
        """
        self.min_frames = int(min_rally_duration_sec * fps)
        self.max_frames = int(max_rally_duration_sec * fps)
        self.fps = fps
        self.min_frames_before_landing = min_frames_before_landing

    def validate_rally(self, rally_frames: int,
                       landing_offset: int = 0) -> Tuple[bool, str, float]:
        """校验 rally 整体时序。

        Args:
            rally_frames: rally 总帧数
            landing_offset: 落地距 rally 开始的帧偏移

        Returns:
            (valid, reason, penalty)
        """
        if rally_frames < self.min_frames:
            return False, f"Rally 太短 ({rally_frames}f < {self.min_frames}f)", 1.0

        if rally_frames > self.max_frames:
            return False, f"Rally 太长 ({rally_frames}f > {self.max_frames}f)", 1.0

        if landing_offset > 0 and landing_offset < self.min_frames_before_landing:
            return False, f"落地过早 (rally 开始后仅 {landing_offset}f)", 0.8

        return True, "OK", 0.0


# ============================================================
# 主流水线：六道防线串联
# ============================================================

class SafeLandingPipeline:
    """多层安全防线流水线。

    数据流：
    detections → [L1] RallyStart → [L2] QualityGate → [L3+L4] PhysicsDetector
                                                                    ↓
                                               [L5] SpatialValidator ←
                                                                    ↓
                                               [L6] TemporalValidator
                                                                    ↓
                                                               LandingResult
    """

    def __init__(self,
                 rally_detector: Optional[RallyStartDetector] = None,
                 quality_gate: Optional[TrajectoryQualityGate] = None,
                 physics_detector: Optional[PhysicsLandingDetector] = None,
                 spatial_validator: Optional[SpatialValidator] = None,
                 temporal_validator: Optional[TemporalValidator] = None,
                 min_confidence: float = 0.40):
        """
        Args:
            min_confidence: 最低置信度阈值，低于此判 NO_LANDING
        """
        self.rally_detector = rally_detector or RallyStartDetector()
        self.quality_gate = quality_gate or TrajectoryQualityGate()
        self.physics_detector = physics_detector or PhysicsLandingDetector()
        self.spatial_validator = spatial_validator or SpatialValidator()
        self.temporal_validator = temporal_validator or TemporalValidator()
        self.min_confidence = min_confidence

    def analyze(self, detections: List[Detection],
                court_bounds_px: Optional[Tuple[float, float, float, float]] = None,
                fps: float = 30.0) -> LandingResult:
        """完整的防线分析。

        L1: Rally 启动检测
        L2: 轨迹质量门禁
        L3+L4: 卡尔曼滤波 + 死球验证
        L5: 空间合理性
        L6: 时序合理性
        L7: 数据末端警戒（新增）
        L8: 落地下行验证（新增）
        L9: 消失确认（新增）
        """
        n_total = len(detections)
        details = {'total_frames': n_total}

        # ============================================
        # L1: Rally 启动检测
        # ============================================
        rally_start_idx = self.rally_detector.detect(detections)

        if rally_start_idx is None:
            details['reason'] = 'L1: 未检测到 rally 启动（球没有飞起来）'
            return LandingResult(
                verdict=Verdict.NO_RALLY,
                confidence=0.0,
                details=details
            )

        details['rally_start_idx'] = rally_start_idx
        details['rally_start_frame'] = detections[rally_start_idx].frame

        # 截取 rally 开始后的数据
        rally_detections = detections[rally_start_idx:]
        details['rally_frames'] = len(rally_detections)

        # ============================================
        # L2: 轨迹质量门禁
        # ============================================
        qa_pass, qa_reason, qa_metrics = self.quality_gate.evaluate(rally_detections)
        details['qa_reason'] = qa_reason
        details['qa_metrics'] = qa_metrics

        if not qa_pass:
            details['reason'] = f'L2: {qa_reason}'
            return LandingResult(
                verdict=Verdict.TRAJECTORY_REJECTED,
                confidence=0.0,
                details=details
            )

        # ============================================
        # L3+L4: 物理模型落地检测
        # ============================================
        candidates = self.physics_detector.detect(rally_detections)
        details['num_candidates'] = len(candidates)

        if not candidates:
            details['reason'] = 'L3+L4: 未检测到物理落地事件'
            return LandingResult(
                verdict=Verdict.NO_LANDING,
                confidence=0.0,
                rally_start_frame=detections[rally_start_idx].frame,
                rally_end_frame=detections[-1].frame,
                details=details
            )

        # 取最佳候选
        best = candidates[0]
        details['best_candidate'] = dict(best)
        details['all_candidates'] = [
            {k: v for k, v in c.items() if k != 'details'}
            for c in candidates[:5]
        ]

        # ============================================
        # ============================================
        # L5: 空间合理性
        # ============================================
        if court_bounds_px is not None:
            sp_valid, sp_penalty = self.spatial_validator.validate(
                best['landing_x'], best['landing_y'])
            details['spatial_valid'] = sp_valid
            details['spatial_penalty'] = sp_penalty
            best['confidence'] *= (1.0 - 0.4 * sp_penalty)
        else:
            # 无标定文件时，用轨迹历史范围做自动校验
            rally_xs = [d.x for d in rally_detections]
            rally_ys = [d.y for d in rally_detections]
            x_min, x_max = min(rally_xs), max(rally_xs)
            y_min, y_max = min(rally_ys), max(rally_ys)
            margin_x = (x_max - x_min) * 0.3
            margin_y = (y_max - y_min) * 0.3
            if (x_min - margin_x <= best['landing_x'] <= x_max + margin_x and
                    y_min - margin_y <= best['landing_y'] <= y_max + margin_y):
                details['L5_note'] = '自动范围校验通过'
            else:
                best['confidence'] *= 0.7
                details['L5_auto_penalty'] = 0.3
                details['L5_note'] = '落地在轨迹范围之外，扣分'

        # ============================================
        # L6: 时序合理性
        # ============================================
        tm_valid, tm_reason, tm_penalty = self.temporal_validator.validate_rally(
            len(rally_detections),
            landing_offset=best['landing_idx']
        )
        details['temporal_valid'] = tm_valid
        details['temporal_reason'] = tm_reason
        if not tm_valid:
            best['confidence'] *= (1.0 - 0.5 * tm_penalty)

        # ============================================
        # L7: 数据末端警戒（新增）
        # 落地发生在轨迹最后 N 帧时降低置信度
        # —— 可能是 tracker 丢帧，而非真落地
        # ============================================
        landing_idx = best['landing_idx']
        rally_len = len(rally_detections)
        trailing_frames = rally_len - landing_idx - 1
        details['trailing_frames'] = trailing_frames

        if trailing_frames <= 0:
            # 落地是最后一帧：tracker 可能在球落地瞬间丢帧
            # 不直接拒绝，但大幅扣分
            edge_penalty = 0.25
            details['L7_edge_penalty'] = edge_penalty
            details['L7_note'] = '落地在轨迹最后一帧，可能是 tracker 丢失'
            best['confidence'] *= (1.0 - edge_penalty)
        elif trailing_frames <= 3:
            # 落地后仅 1-3 帧有数据，轻微扣分
            edge_penalty = 0.10
            details['L7_edge_penalty'] = edge_penalty
            details['L7_note'] = f'落地后仅 {trailing_frames} 帧数据'
            best['confidence'] *= (1.0 - edge_penalty)
        else:
            details['L7_edge_penalty'] = 0.0
            details['L7_note'] = f'落地后有 {trailing_frames} 帧数据，正常'

        # ============================================
        # L8: 落地前下降验证（新增）
        # 落地前球必须在下降（图像坐标系中 y 增大）
        # 如果球在上升或水平飞行 → 不是落地
        # ============================================
        if landing_idx >= 8:
            pre_ys = np.array([rally_detections[i].y
                               for i in range(landing_idx - 8, landing_idx + 1)])
            early_trend = pre_ys[5] - pre_ys[0]
            late_trend = pre_ys[7] - pre_ys[4]
            details['L8_early_trend'] = float(early_trend)
            details['L8_late_trend'] = float(late_trend)
            if early_trend <= 0 and late_trend <= 0:
                best['confidence'] *= 0.6
                details['L8_descend_penalty'] = 0.4
                details['L8_note'] = '落地前无下降趋势'
            else:
                details['L8_descend_penalty'] = 0.0
                details['L8_note'] = f'下降趋势: early={early_trend:.1f} late={late_trend:.1f}'
        elif landing_idx >= 4:
            pre_ys = np.array([rally_detections[i].y
                               for i in range(landing_idx - 4, landing_idx + 1)])
            pre_dy = np.diff(pre_ys)
            descending_frames = int(np.sum(np.array(pre_dy) > 0))
            details['L8_descending_frames'] = descending_frames
            if descending_frames < 2:
                best['confidence'] *= 0.6
                details['L8_descend_penalty'] = 0.4
                details['L8_note'] = f'落地前仅 {descending_frames}/4 帧在下降'
            else:
                details['L8_descend_penalty'] = 0.0
                details['L8_note'] = 'OK'
        else:
            details['L8_note'] = '轨迹太短，跳过'

        # ============================================
        # L9: 消失确认（新增）
        # 如果 detections.jsonl 原始数据中存在"落地帧之后无球数据"，
        # 这是正面信号（球确实落地了，tracker 找不到 = 球停在地上）
        # 注意：这里需要原始完整数据，传入时放在 details 里
        # ============================================
        # L9 依赖外部传入原始数据中的 null 帧信息，此处标记
        details['L9_note'] = '需要外部传入 null-frame 信息以增强验证'
        # ============================================
        # L10: 轨迹终点验证（新增）
        # 最佳候选靠近轨迹末尾 → 是真正的终点落地
        # ============================================
        best_idx = best['landing_idx']
        rally_len = len(rally_detections)
        frac = best_idx / rally_len if rally_len > 0 else 0
        details['L10_end_frac'] = round(frac, 3)
        if frac > 0.85:
            best['confidence'] = min(1.0, best['confidence'] + 0.08)
            details['L10_end_bonus'] = 0.08
            details['L10_note'] = f'候选在轨迹末端 ({frac:.1%})，+8%'
        elif frac > 0.6:
            best['confidence'] = min(1.0, best['confidence'] + 0.04)
            details['L10_end_bonus'] = 0.04
            details['L10_note'] = f'候选靠后 ({frac:.1%})，+4%'
        else:
            details['L10_note'] = f'候选在轨迹中段 ({frac:.1%})，不加分'

        # ============================================
        # L11: 下降强度加分（新增）
        # early_trend 和 late_trend 都 > 10 → 强烈下降 → 高度可信
        # ============================================
        early = details.get('L8_early_trend', 0)
        late = details.get('L8_late_trend', 0)
        if early > 20 and late > 10:
            best['confidence'] = min(1.0, best['confidence'] + 0.06)
            details['L11_descent_bonus'] = 0.06
            details['L11_note'] = f'强烈下降趋势 (early={early:.1f} late={late:.1f})，+6%'
        elif early > 10:
            best['confidence'] = min(1.0, best['confidence'] + 0.03)
            details['L11_descent_bonus'] = 0.03
            details['L11_note'] = f'下降趋势 (early={early:.1f})，+3%'
        else:
            details['L11_note'] = '下降趋势一般，不加分'
        # ============================================
        # 最终判定
        # ============================================
        final_conf = best['confidence']
        details['final_confidence_raw'] = float(final_conf)

        if final_conf < self.min_confidence:
            details['reason'] = f'置信度过低 ({final_conf:.2f} < {self.min_confidence})'
            return LandingResult(
                verdict=Verdict.NO_LANDING,
                confidence=final_conf,
                rally_start_frame=detections[rally_start_idx].frame,
                rally_end_frame=detections[-1].frame,
                details=details
            )

        return LandingResult(
            verdict=Verdict.LANDING_FOUND,
            landing_frame=best['landing_frame'],
            landing_x=best['landing_x'],
            landing_y=best['landing_y'],
            confidence=final_conf,
            rally_start_frame=detections[rally_start_idx].frame,
            rally_end_frame=detections[-1].frame,
            details=details
        )


# ============================================================
# 便捷函数：从 JSONL 加载数据
# ============================================================

def load_detections_from_jsonl(path: str) -> List[Detection]:
    """从 Good-Badminton 的 detections.jsonl 加载数据。"""
    detections = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sc = d.get('shuttlecock')
            if sc is None:
                continue
            
            if isinstance(sc, dict):
                img = sc.get('image')
                if img is None or (isinstance(img, list) and len(img) < 2):
                    continue
                x, y = float(img[0]), float(img[1])
            elif isinstance(sc, (list, tuple)) and len(sc) >= 2:
                x, y = float(sc[0]), float(sc[1])
            else:
                continue

            frame = d.get('frame', len(detections))
            detections.append(Detection(frame=int(frame), x=x, y=y))
    return detections

# ============================================================
# 自适应参数调节
# ============================================================

class AdaptiveSafeLandingPipeline(SafeLandingPipeline):
    """自适应安全流水线：根据数据统计自动调整参数。

    在初始化时分析一段数据，计算统计量，自动设置合理阈值。
    """

    @classmethod
    def from_calibration(cls, detections: List[Detection],
                         fps: float = 30.0,
                         court_bounds_px=None) -> 'AdaptiveSafeLandingPipeline':
        """先用一小段数据标定参数，再创建流水线。"""
        if len(detections) < 30:
            return cls(fps=fps)

        xs = np.array([d.x for d in detections])
        ys = np.array([d.y for d in detections])
        dx = np.diff(xs)
        dy = np.diff(ys)
        speeds = np.sqrt(dx**2 + dy**2)

        # 自动推断噪声水平
        noise_speed = np.percentile(speeds, 50)  # 中位数速度
        peak_speed = np.percentile(speeds, 95)   # 峰值速度

        # 如果中位数速度极低 → 大部分时间是静止的 → tracker 噪声严重
        # 调高启动阈值
        if noise_speed < 2.0:
            min_speed = max(8.0, peak_speed * 0.3)
            min_displacement = max(40.0, peak_speed * 5)
            min_peak = max(12.0, peak_speed * 0.5)
        else:
            min_speed = max(5.0, noise_speed * 2)
            min_displacement = max(30.0, np.std(xs) + np.std(ys))
            min_peak = max(8.0, peak_speed * 0.4)

        rally_detector = RallyStartDetector(
            min_speed_px_per_frame=float(min_speed),
            min_displacement_px=float(min_displacement),
            min_peak_speed_px_per_frame=float(min_peak),
        )

        quality_gate = TrajectoryQualityGate(
            min_variance_px=max(2.0, noise_speed * 1.5),
        )

        physics_detector = PhysicsLandingDetector(
            measurement_noise=max(1.0, noise_speed * 0.5),
        )

        return cls(
            rally_detector=rally_detector,
            quality_gate=quality_gate,
            physics_detector=physics_detector,
            spatial_validator=SpatialValidator(court_bounds_px=court_bounds_px),
        )


# ============================================================
# 命令行入口
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='物理模型落地检测器 v2.0')
    parser.add_argument('--jsonl', type=str, required=True,
                        help='detections.jsonl 路径')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='视频帧率')
    parser.add_argument('--min-confidence', type=float, default=0.40,
                        help='最低置信度阈值')
    parser.add_argument('--adaptive', action='store_true',
                        help='使用自适应参数')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='输出详细信息')

    args = parser.parse_args()

    # 加载数据
    detections = load_detections_from_jsonl(args.jsonl)
    print(f"加载 {len(detections)} 帧检测数据")

    # 创建流水线
    if args.adaptive:
        pipeline = AdaptiveSafeLandingPipeline.from_calibration(
            detections, fps=args.fps)
    else:
        pipeline = SafeLandingPipeline()

    pipeline.min_confidence = args.min_confidence

    # 分析
    result = pipeline.analyze(detections, fps=args.fps)

    # 输出
    print()
    print("=" * 60)
    print(f"判定结果: {result.verdict.name}")
    print(f"置信度:   {result.confidence:.1%}")
    print("-" * 60)

    if result.verdict == Verdict.LANDING_FOUND:
        print(f"落地帧:   {result.landing_frame}")
        print(f"落地位置: ({result.landing_x:.1f}, {result.landing_y:.1f})")
        print(f"Rally:    frame {result.rally_start_frame} → {result.rally_end_frame}")
    else:
        print(f"原因:     {result.details.get('reason', 'unknown')}")

    if args.verbose:
        print("-" * 60)
        for k, v in result.details.items():
            if k not in ('all_candidates',):
                print(f"  {k}: {v}")
        if 'all_candidates' in result.details:
            print("\n全部候选:")
            for i, c in enumerate(result.details['all_candidates']):
                print(f"  [{i}] frame={c['landing_frame']}, conf={c['confidence']:.3f}, "
                      f"innov={c['innovation']:.2f}")

    print("=" * 60)
"""
test_integration.py — 全链路模拟集成测试 (动态噪声版)
=====================================================
每帧的检测质量根据真实物理情况动态计算：
  - 球到相机距离 → 球在画面上的大小 → 检测难度
  - 球在画面上的位置 → 边缘畸变 → 定位精度下降
  - 球的 2D 速度 → 运动模糊 → 噪声加大
  - 球是否被网/人遮挡 → 漏检概率

高铁上直接跑：
  python test_integration.py
"""

import os
import sys
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shuttlecock_physics import ShuttlecockPhysics, PhysicsParams
import court_config


# ═══════════════════════════════════════════════════════════
# 动态噪声模型
# ═══════════════════════════════════════════════════════════
@dataclass
class NoiseParams:
    """每帧动态计算的噪声参数"""
    base_noise: float        # 基础噪声 (px)
    edge_factor: float       # 边缘畸变放大倍数
    motion_factor: float     # 运动模糊放大倍数
    distance_factor: float   # 距离衰减因子
    total_noise_std: float   # 最终噪声标准差 (px)
    detection_prob: float    # 本帧检测成功概率
    failure_reason: str      # 如果漏检，原因


class DynamicNoiseModel:
    """
    根据球在画面中的实时状态，动态计算噪声和检测率。

    因素：
      1. 边缘畸变 — 越靠近画面边缘，定位越不准
      2. 运动模糊 — 球在画面上的速度越快，越模糊
      3. 距离衰减 — 球越远越小，检测越困难
      4. 网带遮挡 — 球在网高附近且 Y 坐标在中线，可能被网遮挡
    """

    def __init__(self, image_width=1280, image_height=720,focal_length=1740):
        self.w = image_width
        self.h = image_height
        self.f = focal_length
        self.cx = image_width / 2
        self.cy = image_height / 2

    def evaluate(
        self,
        pt_2d: np.ndarray,       # 球在图像上的位置 (u, v)
        pt_3d: np.ndarray,       # 球在世界坐标 (x, y, z)
        vel_3d: np.ndarray,      # 球在世界速度 (vx, vy, vz)
        prev_pt_2d: np.ndarray = None,  # 上一帧的 2D 位置（用于算画面速度）
    ) -> NoiseParams:
        """根据当前帧状态，计算本帧的噪声和检测概率"""

        # ── 1. 基础噪声（完美条件下的最小噪声） ──
        base_noise = 0.35  # px

        # ── 2. 边缘畸变 ──
        # 径向距离 / 最大半径，0=中心 1=角落
        r = np.sqrt((pt_2d[0] - self.cx)**2 + (pt_2d[1] - self.cy)**2)
        r_max = np.sqrt(self.cx**2 + self.cy**2)
        r_norm = np.clip(r / r_max, 0.0, 1.0)
        # 边缘畸变：中心 1x，角落 3x
        edge_factor = 1.0 + 2.0 * r_norm**2

        # ── 3. 距离衰减 ──
        # 球距相机越远，成像越小，检测越难
        # 球场近端约 2.5m，远端约 14m
        distance = float(np.linalg.norm(pt_3d))
        # 归一化距离：2m=最好，15m=最差
        dist_norm = np.clip((distance - 2.0) / 13.0, 0.0, 1.0)
        distance_factor = 1.0 + 2.0 * dist_norm**1.5

        # ── 4. 运动模糊 ──
        # 球在画面上的速度 (px/s)
        if prev_pt_2d is not None:
            pixel_speed = float(np.linalg.norm(pt_2d - prev_pt_2d)) / 0.01  # dt=0.01s
        else:
            # 从 3D 速度估算
            # 简化：速度在 XY 平面的分量 / 距离 × 焦距
            lateral_speed = float(np.linalg.norm(vel_3d[:2]))
            pixel_speed = lateral_speed / max(distance, 0.1) * self.f

        # 模糊阈值：球速 < 200 px/s 无影响，> 2000 px/s 翻倍噪声
        blur_norm = np.clip((pixel_speed - 200) / 1800, 0.0, 1.0)
        motion_factor = 1.0 + 3.0 * blur_norm**1.2

        # ── 5. 网带遮挡判断 ──
        net_y = court_config.COURT_LENGTH / 2  # 网在 Y=6.7m
        net_height = 1.55  # 网高
        near_net = abs(pt_3d[1] - net_y) < 0.3  # 距网 ±30cm
        near_net_height = abs(pt_3d[2] - net_height) < 0.2  # 高度接近网高
        is_occluded = near_net and near_net_height

        # ── 6. 合成总噪声 ──
        total_noise = base_noise * edge_factor * distance_factor * motion_factor

        # ── 7. 检测概率 ──
        # 基础检测率受距离和遮挡影响
        p = 0.98  # 理想检测率
        p -= 0.08 * dist_norm**2  # 远距离降低
        if is_occluded:
            p -= 0.25  # 网带遮挡大幅降低
        p -= 0.03 * blur_norm  # 高速模糊降低
        detection_prob = max(0.1, min(1.0, p))

        # ── 8. 失败原因 ──
        reason = "ok"
        if detection_prob < 0.6:
            reasons = []
            if is_occluded:
                reasons.append("网带遮挡")
            if dist_norm > 0.7:
                reasons.append("距离过远")
            if blur_norm > 0.6:
                reasons.append("运动模糊")
            reason = "+".join(reasons) if reasons else "综合"

        return NoiseParams(
            base_noise=base_noise,
            edge_factor=float(edge_factor),
            motion_factor=float(motion_factor),
            distance_factor=float(distance_factor),
            total_noise_std=float(total_noise),
            detection_prob=float(detection_prob),
            failure_reason=reason,
        )


# ═══════════════════════════════════════════════════════════
# RK4 轨迹生成器（完整中间点，空气阻力）
# ═══════════════════════════════════════════════════════════
class TrajectoryGenerator:
    """
    RK4 数值积分，空气阻力模型：
      F_lin = -b * v
      F_quad = -0.5 * rho * Cd * A * |v| * v
      a = (0, 0, -g) + (F_lin + F_quad) / m
    """

    def __init__(self, physics: ShuttlecockPhysics):
        p = physics.p
        self.mass = p.mass
        self.g = p.g
        self.b = p.b
        self.k_quad = 0.5 * p.rho * p.Cd * p.A

    def _acceleration(self, vel: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(vel))
        if speed < 1e-12:
            return np.array([0.0, 0.0, -self.g])
        v_hat = vel / speed
        drag_mag = self.b * speed + self.k_quad * speed * speed
        return np.array([0.0, 0.0, -self.g]) - drag_mag / self.mass * v_hat

    def generate(self, pos0, vel0, z_ground=0.0, max_time=3.0, sample_dt=0.01):
        rk4_dt = 0.0005
        steps_per_sample = max(1, int(sample_dt / rk4_dt))

        pos = np.array(pos0, dtype=np.float64)
        vel = np.array(vel0, dtype=np.float64)
        t, step = 0.0, 0
        trajectory = []
        self._record(trajectory, t, pos, vel)

        while t < max_time:
            k1v = self._acceleration(vel) * rk4_dt
            k1p = vel * rk4_dt
            k2v = self._acceleration(vel + 0.5 * k1v) * rk4_dt
            k2p = (vel + 0.5 * k1v) * rk4_dt
            k3v = self._acceleration(vel + 0.5 * k2v) * rk4_dt
            k3p = (vel + 0.5 * k2v) * rk4_dt
            k4v = self._acceleration(vel + k3v) * rk4_dt
            k4p = (vel + k3v) * rk4_dt

            vel += (k1v + 2*k2v + 2*k3v + k4v) / 6.0
            pos += (k1p + 2*k2p + 2*k3p + k4p) / 6.0
            t += rk4_dt
            step += 1

            if step % steps_per_sample == 0:
                self._record(trajectory, t, pos, vel)

            if pos[2] <= z_ground:
                pos[2] = z_ground
                self._record(trajectory, t, pos, vel)
                break

        return trajectory

    @staticmethod
    def _record(traj, t, pos, vel):
        traj.append((float(t), float(pos[0]), float(pos[1]), float(pos[2]),
                      float(vel[0]), float(vel[1]), float(vel[2]),
                      float(np.linalg.norm(vel))))


# ═══════════════════════════════════════════════════════════
# 虚拟双目相机
# ═══════════════════════════════════════════════════════════
class VirtualStereoCamera:
    def __init__(self, width=1280, height=720,focal_length=1740, baseline=0.5,
                 cam_distance=2.5, cam_height=3.0):
        self.w, self.h = width, height
        self.f = focal_length
        self.baseline = baseline
        self.K = np.array([[focal_length, 0, width/2], [0, focal_length, height/2], [0, 0, 1]], dtype=np.float64)
        self.R = np.eye(3)
        cx = court_config.COURT_WIDTH / 2
        self.t_left  = np.array([cx, -cam_distance, cam_height], dtype=np.float64).reshape(3, 1)
        self.t_right = np.array([cx + baseline, -cam_distance, cam_height], dtype=np.float64).reshape(3, 1)
        self.P_left  = self.K @ np.hstack([self.R, self.t_left])
        self.P_right = self.K @ np.hstack([self.R, self.t_right])

    def project(self, pt_3d):
        X = np.append(np.asarray(pt_3d[:3], dtype=np.float64), 1.0).reshape(4, 1)
        pL = self.P_left @ X;  pL = (pL[:2] / pL[2]).flatten()
        pR = self.P_right @ X; pR = (pR[:2] / pR[2]).flatten()
        return pL, pR

    def triangulate(self, ptL, ptR):
        pL = np.array(ptL, dtype=np.float64).reshape(2, 1)
        pR = np.array(ptR, dtype=np.float64).reshape(2, 1)
        pt4 = cv2.triangulatePoints(self.P_left, self.P_right, pL, pR)
        return (pt4[:3] / pt4[3]).flatten()


# ═══════════════════════════════════════════════════════════
# 简易落地检测器
# ═══════════════════════════════════════════════════════════
class SimpleLandingDetector:
    def __init__(self, height_threshold=0.02, z_history_size=10):
        self.thresh = height_threshold
        self.z_hist = deque(maxlen=z_history_size)
        self.detected = False

    def update(self, pt_3d, ts=None):
        self.z_hist.append(pt_3d[2])
        if not self.detected and len(self.z_hist) >= 5:
            recent = list(self.z_hist)[-5:]
            if all(v < self.thresh for v in recent):
                self.detected = True
                return {"position": pt_3d, "timestamp": ts, "height": pt_3d[2]}
        return None

    def reset(self):
        self.z_hist.clear()
        self.detected = False


# ═══════════════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════════════
def run_full_simulation():
    print("=" * 65)
    print("  羽毛球鹰眼 — 全链路动态噪声模拟")
    print("=" * 65)

    # ── 1. 双目 + 噪声模型 ────────────────────────
    stereo = VirtualStereoCamera()
    noise_model = DynamicNoiseModel()
    print(f"\n[1/5] 虚拟双目 + 动态噪声模型")
    print(f"  双目: {stereo.w}×{stereo.h}, f={stereo.f}px, 基线={stereo.baseline}m")
    print(f"  噪声: 边缘畸变(1~3x) × 距离衰减(1~3x) × 运动模糊(1~4x)")

    # ── 2. 轨迹生成 ──────────────────────────────
    physics = ShuttlecockPhysics(PhysicsParams())
    gen = TrajectoryGenerator(physics)
    print(f"\n[2/5] RK4 空气阻力轨迹 (mass={physics.p.mass}kg, b={physics.p.b}kg/m, "
          f"v_terminal≈{physics.p.v_terminal:.1f}m/s)")

    scenarios = [
        ("扣杀",   np.array([3.05, 1.0, 2.5]),  np.array([1.5,  7.0, -4.0])),
        ("高远球", np.array([3.05, 0.5, 1.5]),  np.array([0.5,  4.0,  6.0])),
        ("吊球",   np.array([3.05, 6.5, 2.2]),  np.array([-0.3, 2.0, -1.5])),
        ("平抽",   np.array([2.00, 6.0, 1.8]),  np.array([0.8,  6.5, -1.0])),
    ]

    all_traj = []
    for name, p0, v0 in scenarios:
        traj = gen.generate(p0, v0)
        d = traj[-1][0] - traj[0][0]
        print(f"  {name}: {len(traj)}点, {d:.3f}s, 初速{np.linalg.norm(v0):.1f}m/s → "
              f"落点({traj[-1][1]:.2f},{traj[-1][2]:.2f}), 终速{traj[-1][7]:.1f}m/s")
        all_traj.append((name, traj))

    # ── 3. 逐帧处理：每帧独立噪声 ────────────────
    np.random.seed(42)
    print(f"\n[3/5] 逐帧动态噪声 + 三角测量重建")

    all_results = []
    noise_stats = []  # 收集所有帧的噪声参数

    for traj_name, traj in all_traj:
        results = []
        n_info_list = []
        prev_ptL, prev_ptR = None, None

        for i, tp in enumerate(traj):
            t, x, y, z, vx, vy, vz, speed = tp
            true_3d = np.array([x, y, z])

            # 投影（无噪声）
            ptL_true, ptR_true = stereo.project(true_3d)

            # ── 动态计算本帧噪声 ──
            vel_3d = np.array([vx, vy, vz])
            n_info = noise_model.evaluate(ptL_true, true_3d, vel_3d, prev_ptL)
            n_info_list.append(n_info)

            # 尝试检测
            recon_3d, error, status = None, None, "missed"

            if np.random.random() < n_info.detection_prob:
                # 加动态噪声
                noise_L = np.random.normal(0, n_info.total_noise_std, 2)
                noise_R = np.random.normal(0, n_info.total_noise_std, 2)
                ptL_det = ptL_true + noise_L
                ptR_det = ptR_true + noise_R

                # 三角测量
                try:
                    recon_3d = stereo.triangulate(ptL_det, ptR_det)
                    error = float(np.linalg.norm(recon_3d - true_3d))
                    status = "ok"
                except Exception:
                    status = "triang_failed"
                    recon_3d, error = None, None
            else:
                # 漏检
                status = f"missed:{n_info.failure_reason}"

            results.append({
                "t": t, "true_3d": true_3d, "recon_3d": recon_3d,
                "error": error, "status": status,
                "noise_info": n_info,
            })

            prev_ptL = ptL_true
            prev_ptR = ptR_true

        all_results.append((traj_name, results))
        noise_stats.append((traj_name, n_info_list))

        # 统计本轨迹
        ok_results = [r for r in results if r["status"] == "ok"]
        missed_results = [r for r in results if r["status"] != "ok"]
        errors = [r["error"] for r in ok_results]
        total = len(results)

        print(f"\n  {traj_name} ({total}帧):")
        print(f"    成功: {len(ok_results)} ({100*len(ok_results)/total:.1f}%)  "
              f"漏检: {len(missed_results)} ({100*len(missed_results)/total:.1f}%)")

        # 漏检原因统计
        miss_reasons = {}
        for r in missed_results:
            reason = r["status"].replace("missed:", "") if "missed:" in r["status"] else r["status"]
            miss_reasons[reason] = miss_reasons.get(reason, 0) + 1
        if miss_reasons:
            print(f"    漏检原因: {miss_reasons}")

        if errors:
            print(f"    3D误差: 均值{np.mean(errors)*1000:.1f}mm  "
                  f"中位{np.median(errors)*1000:.1f}mm  "
                  f"最大{np.max(errors)*1000:.1f}mm")

        # 噪声变化范围
        noises = [ni.total_noise_std for ni in n_info_list]
        det_probs = [ni.detection_prob for ni in n_info_list]
        print(f"    噪声范围: {min(noises):.2f}~{max(noises):.2f} px  "
              f"检测率: {min(det_probs):.2f}~{max(det_probs):.2f}")

    # ── 4. 落地检测 + 判罚 ───────────────────────
    print(f"\n[4/5] 落地检测")
    try:
        from physics_landing_detector import LandingDetector
        ld = LandingDetector()
        print("  使用: physics_landing_detector.py")
    except ImportError:
        ld = SimpleLandingDetector(height_threshold=0.02)
        print("  使用: SimpleLandingDetector (threshold=0.02m)")

    try:
        from judgment_engine import LineCallEngine
        judge = JudgmentEngine()
        print("  使用: judgment_engine.py")
    except ImportError:
        judge = None
        print("  判罚: 简易边界检查")

    judgments = []
    for traj_name, results in all_results:
        ld.reset()
        for r in results:
            if r["recon_3d"] is not None:
                landing = ld.update(r["recon_3d"], r["t"])
                if landing is not None:
                    pos = landing["position"]
                    if judge:
                        j = judge.judge(pos)
                    else:
                        in_b = (0 <= pos[0] <= court_config.COURT_WIDTH and
                                0 <= pos[1] <= court_config.COURT_LENGTH)
                        j = "界内" if in_b else "界外"
                    judgments.append((traj_name, pos, j))
                    print(f"  {traj_name}: ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.4f}) → {j}")

    # ── 5. 汇总 ──────────────────────────────────
    print(f"\n[5/5] 测试汇总")
    print("=" * 65)

    all_errs = []
    for _, results in all_results:
        for r in results:
            if r["error"] is not None:
                all_errs.append(r["error"])

    all_statuses = []
    for _, results in all_results:
        for r in results:
            all_statuses.append(r["status"])

    ok_count = sum(1 for s in all_statuses if s == "ok")
    total_count = len(all_statuses)

    print(f"  总帧数: {total_count}")
    print(f"  成功重建: {ok_count} ({100*ok_count/total_count:.1f}%)")
    print(f"  漏检/失败: {total_count - ok_count}")

    if all_errs:
        print(f"\n  3D 重建精度 (成功帧):")
        print(f"    均值:   {np.mean(all_errs)*1000:.1f} mm")
        print(f"    中位数: {np.median(all_errs)*1000:.1f} mm")
        print(f"    最大:   {np.max(all_errs)*1000:.1f} mm")
        print(f"    RMSE:   {np.sqrt(np.mean(np.array(all_errs)**2))*1000:.1f} mm")
        p95 = np.percentile(all_errs, 95) * 1000
        print(f"    P95:    {p95:.1f} mm")

    print(f"\n  落地事件: {len(judgments)}")
    for name, pos, result in judgments:
        print(f"    {name}: ({pos[0]:.2f}, {pos[1]:.2f}) → {result}")

    avg_mm = np.mean(all_errs)*1000 if all_errs else float("inf")
    if avg_mm < 20:
        grade = "优秀 — 3D重建精度满足鹰眼判罚需求"
    elif avg_mm < 50:
        grade = "良好 — 精度足够，可优化标定提升"
    else:
        grade = "需改进 — 检查标定和噪声参数"

    print(f"\n  综合: {grade}")
    print("=" * 65)
    return all_results, judgments


if __name__ == "__main__":
    results, judgments = run_full_simulation()
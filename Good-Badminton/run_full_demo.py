"""
run_full_demo.py — 双目羽毛球鹰眼判罚系统
==========================================
支持三种模式：
  python run_full_demo.py --mode simulate    模拟数据（高铁上就能跑）
  python run_full_demo.py --mode stereo      双相机实时
  python run_full_demo.py --mode video       单视频文件

流程：
  标定(一次) → 采集 → 球检测(左右) → 三角测量 → 追踪 → 落地判罚 → 3D可视化
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ─── 项目模块 ────────────────────────────────────────────
from shuttlecock_physics import ShuttlecockPhysics, PhysicsParams
import court_config

# 可选模块
try:
    from ball_flight_tracker import TrajectoryPredictor
except ImportError:
    TrajectoryPredictor = None

try:
    from physics_landing_detector import LandingDetector
except ImportError:
    LandingDetector = None

try:
    from judgment_engine import LineCallEngine as JudgeEngine
except ImportError:
    JudgeEngine = None

try:
    from court_3d import Court3D
except ImportError:
    Court3D = None


# ═══════════════════════════════════════════════════════════
# RK4 轨迹生成器（完整中间点，空气阻力）
# ═══════════════════════════════════════════════════════════
class TrajectoryGenerator:
    def __init__(self, physics: ShuttlecockPhysics):
        p = physics.p
        self.mass = p.mass
        self.g = p.g
        self.b = p.b
        self.k_quad = 0.5 * p.rho * p.Cd * p.A

    def _acceleration(self, vel):
        speed = float(np.linalg.norm(vel))
        if speed < 1e-12:
            return np.array([0.0, 0.0, -self.g])
        v_hat = vel / speed
        drag_mag = self.b * speed + self.k_quad * speed * speed
        return np.array([0.0, 0.0, -self.g]) - drag_mag / self.mass * v_hat

    def generate(self, pos0, vel0, z_ground=0.0, max_time=3.0, sample_dt=0.01):
        rk4_dt = 0.0005
        steps = max(1, int(sample_dt / rk4_dt))
        pos = np.array(pos0, dtype=np.float64)
        vel = np.array(vel0, dtype=np.float64)
        t, step = 0.0, 0
        traj = []

        traj.append((float(t), float(pos[0]), float(pos[1]), float(pos[2]),
                      float(vel[0]), float(vel[1]), float(vel[2]),
                      float(np.linalg.norm(vel))))

        while t < max_time:
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

            if step % steps == 0:
                traj.append((float(t), float(pos[0]), float(pos[1]), float(pos[2]),
                              float(vel[0]), float(vel[1]), float(vel[2]),
                              float(np.linalg.norm(vel))))

            if pos[2] <= z_ground:
                pos[2] = z_ground
                traj.append((float(t), float(pos[0]), float(pos[1]), float(pos[2]),
                              float(vel[0]), float(vel[1]), float(vel[2]),
                              float(np.linalg.norm(vel))))
                break

        return traj


# ═══════════════════════════════════════════════════════════
# 动态噪声模型
# ═══════════════════════════════════════════════════════════
class DynamicNoiseModel:
    def __init__(self, w=1280, h=720, f=1740):
        self.w, self.h, self.f = w, h, f
        self.cx, self.cy = w/2, h/2

    def evaluate(self, pt_2d, pt_3d, vel_3d, prev_pt_2d=None):
        # 边缘畸变
        r = np.sqrt((pt_2d[0]-self.cx)**2 + (pt_2d[1]-self.cy)**2)
        r_max = np.sqrt(self.cx**2 + self.cy**2)
        r_norm = np.clip(r / r_max, 0, 1)
        edge_factor = 1.0 + 2.0 * r_norm**2

        # 距离衰减
        dist = float(np.linalg.norm(pt_3d))
        dist_norm = np.clip((dist - 2.0) / 13.0, 0, 1)
        distance_factor = 1.0 + 2.0 * dist_norm**1.5

        # 运动模糊
        if prev_pt_2d is not None:
            pixel_speed = float(np.linalg.norm(pt_2d - prev_pt_2d)) / 0.01
        else:
            pixel_speed = float(np.linalg.norm(vel_3d[:2])) / max(dist, 0.1) * self.f
        blur_norm = np.clip((pixel_speed - 200) / 1800, 0, 1)
        motion_factor = 1.0 + 3.0 * blur_norm**1.2

        # 网带遮挡
        net_y = court_config.COURT_LENGTH / 2
        occluded = (abs(pt_3d[1] - net_y) < 0.3 and abs(pt_3d[2] - 1.55) < 0.2)

        # 合成
        noise_std = 0.35 * edge_factor * distance_factor * motion_factor
        det_prob = 0.99 - 0.05 * dist_norm**2 - (0.20 if occluded else 0) - 0.02 * blur_norm
        det_prob = max(0.1, min(1.0, det_prob))

        return float(noise_std), float(det_prob)


# ═══════════════════════════════════════════════════════════
# 虚拟双目相机
# ═══════════════════════════════════════════════════════════
class VirtualStereoCamera:
    def __init__(self, width=1280, height=720, focal=1740, baseline=0.5, cam_dist=2.5, cam_h=3.0):
        self.w, self.h, self.f = width, height, focal
        self.K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=np.float64)
        cx = court_config.COURT_WIDTH / 2
        self.t_left  = np.array([cx, -cam_dist, cam_h], dtype=np.float64).reshape(3, 1)
        self.t_right = np.array([cx + baseline, -cam_dist, cam_h], dtype=np.float64).reshape(3, 1)
        self.P_left  = self.K @ np.hstack([np.eye(3), self.t_left])
        self.P_right = self.K @ np.hstack([np.eye(3), self.t_right])

    def project(self, pt_3d):
        X = np.append(np.asarray(pt_3d[:3], dtype=np.float64), 1.0).reshape(4, 1)
        pL = (self.P_left @ X); pL = (pL[:2]/pL[2]).flatten()
        pR = (self.P_right @ X); pR = (pR[:2]/pR[2]).flatten()
        return pL, pR

    def triangulate(self, pL, pR):
        pL = np.array(pL, dtype=np.float64).reshape(2, 1)
        pR = np.array(pR, dtype=np.float64).reshape(2, 1)
        pt4 = cv2.triangulatePoints(self.P_left, self.P_right, pL, pR)
        return (pt4[:3] / pt4[3]).flatten()


# ═══════════════════════════════════════════════════════════
# 简易落地检测器
# ═══════════════════════════════════════════════════════════
class SimpleLandingDetector:
    def __init__(self, thresh=0.02, hist=10):
        self.thresh = thresh
        self.z_hist = deque(maxlen=hist)
        self.detected = False

    def update(self, pt, ts=None):
        self.z_hist.append(pt[2])
        if not self.detected and len(self.z_hist) >= 5:
            if all(v < self.thresh for v in list(self.z_hist)[-5:]):
                self.detected = True
                return {"position": pt, "timestamp": ts}
        return None

    def reset(self):
        self.z_hist.clear()
        self.detected = False


# ═══════════════════════════════════════════════════════════
# 鹰眼系统主类
# ═══════════════════════════════════════════════════════════
class HawkEyeSystem:
    def __init__(self):
        self.physics = ShuttlecockPhysics(PhysicsParams())
        self.traj_gen = TrajectoryGenerator(self.physics)
        self.trajectory_3d = []
        self.tracker = None
        self.landing_det = None
        self.judge = None
        self.court_3d = None

    def _init_modules(self):
        if TrajectoryPredictor is not None and self.tracker is None:
            self.tracker = TrajectoryPredictor()
            print("[初始化] 飞行追踪器已加载")

        if LandingDetector is not None and self.landing_det is None:
            self.landing_det = LandingDetector()
            print("[初始化] 落地检测器已加载")
        elif self.landing_det is None:
            self.landing_det = SimpleLandingDetector()
            print("[初始化] 简易落地检测器已加载")

        if JudgeEngine is not None and self.judge is None:
            self.judge = JudgeEngine()
            print("[初始化] 判罚引擎已加载")

        if Court3D is not None and self.court_3d is None:
            self.court_3d = Court3D()
            print("[初始化] 3D 可视化已加载")

    # ── 模拟模式 ────────────────────────────────────
    def run_simulate(self):
        self._init_modules()
        print("\n" + "=" * 60)
        print("  模拟模式 — 全链路集成测试")
        print("=" * 60)

        stereo = VirtualStereoCamera(baseline=13.4, cam_dist=2.5, cam_h=3.0)
        noise_model = DynamicNoiseModel()
        np.random.seed(42)

        print(f"  虚拟双目: {stereo.w}×{stereo.h}, f={stereo.K[0,0]:.0f}px, "
              f"基线={abs(stereo.t_right[0,0]-stereo.t_left[0,0]):.2f}m")
        print(f"  物理参数: mass={self.physics.p.mass}kg, b={self.physics.p.b}, "
              f"v_terminal≈{self.physics.p.v_terminal:.1f}m/s")

        scenarios = [
            ("扣杀",   np.array([3.05, 1.0, 2.5]),  np.array([1.5,  7.0, -4.0])),
            ("高远球", np.array([3.05, 0.5, 1.5]),  np.array([0.5,  4.0,  6.0])),
            ("吊球",   np.array([3.05, 6.5, 2.2]),  np.array([-0.3, 2.0, -1.5])),
            ("平抽",   np.array([2.00, 6.0, 1.8]),  np.array([0.8,  6.5, -1.0])),
        ]

        all_errors = []
        judgments = []

        for traj_name, pos0, vel0 in scenarios:
            print(f"\n{'─'*50}")
            print(f"  {traj_name}")

            traj = self.traj_gen.generate(pos0, vel0)
            duration = traj[-1][0] - traj[0][0]
            print(f"  轨迹: {len(traj)}点, {duration:.3f}s, "
                  f"初速{np.linalg.norm(vel0):.1f}m/s → "
                  f"落点({traj[-1][1]:.2f},{traj[-1][2]:.2f}), 终速{traj[-1][7]:.1f}m/s")

            if self.tracker:
                self.tracker.reset()
            if self.landing_det:
                self.landing_det.reset()

            traj_3d_points = []
            prev_ptL, prev_ptR = None, None
            errors_this = []

            for i, tp in enumerate(traj):
                t_val, x, y, z, vx, vy, vz, speed = tp
                true_3d = np.array([x, y, z])

                ptL, ptR = stereo.project(true_3d)
                noise_std, det_prob = noise_model.evaluate(ptL, true_3d, np.array([vx, vy, vz]), prev_ptL)

                recon_3d = None
                if np.random.random() < det_prob:
                    noise_L = np.random.normal(0, noise_std, 2)
                    noise_R = np.random.normal(0, noise_std, 2)
                    try:
                        recon_3d = stereo.triangulate(ptL + noise_L, ptR + noise_R)
                        error = float(np.linalg.norm(recon_3d - true_3d))
                        errors_this.append(error)
                    except Exception:
                        recon_3d = None

                if self.tracker and recon_3d is not None:
                    self.tracker.update(recon_3d, t_val)

                if self.landing_det and recon_3d is not None:
                    landing = self.landing_det.update(recon_3d, t_val)
                    if landing is not None:
                        pos = landing["position"]
                        if self.judge:
                            j = self.judge.judge_landing(pos[0], pos[1])
                        else:
                            in_b = (0 <= pos[0] <= court_config.COURT_WIDTH and
                                    0 <= pos[1] <= court_config.COURT_LENGTH)
                            j = "界内" if in_b else "界外"
                        judgments.append((traj_name, pos, j))
                        print(f"  落地: ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.4f}) → {j}")

                        if self.court_3d:
                            self.court_3d.mark_landing(pos)

                if recon_3d is not None:
                    traj_3d_points.append(recon_3d)

                prev_ptL, prev_ptR = ptL, ptR

            if errors_this:
                print(f"  3D误差: 均值{np.mean(errors_this)*1000:.1f}mm "
                      f"中位{np.median(errors_this)*1000:.1f}mm "
                      f"最大{np.max(errors_this)*1000:.1f}mm")
                all_errors.extend(errors_this)

            if self.court_3d and traj_3d_points:
                pts_arr = np.array(traj_3d_points)
                self.court_3d.update_trajectory(pts_arr)

        # ── 汇总 ──
        print(f"\n{'='*60}")
        print(f"  汇总")
        print(f"{'='*60}")
        if all_errors:
            print(f"  总重建点: {len(all_errors)}")
            print(f"  3D 误差: 均值{np.mean(all_errors)*1000:.1f}mm "
                  f"中位{np.median(all_errors)*1000:.1f}mm "
                  f"P95 {np.percentile(all_errors,95)*1000:.1f}mm")

        print(f"  落地事件: {len(judgments)}")
        for name, pos, j in judgments:
            print(f"    {name}: ({pos[0]:.2f},{pos[1]:.2f}) → {j}")

        if self.court_3d:
            print(f"\n  正在播放 3D 轨迹动画...")
            self.court_3d.show()
            if judgments:
                self.court_3d.zoom_to_landing()

        print("=" * 60)

    # ── 双目模式 ────────────────────────────────────
    def run_stereo(self, calib_path, cam_left=0, cam_right=1, num_frames=300):
        """双相机实时模式（相机到货后用）"""
        self._init_modules()

        if not os.path.exists(calib_path):
            print(f"[错误] 标定文件不存在: {calib_path}")
            print("  请先用 calibration.py 做双目标定")
            return

        # 加载 calibration.py 输出的 StereoPose JSON
        from calibration import StereoPose
        stereo_calib = StereoPose.load(calib_path)
        if not stereo_calib.success:
            print("[错误] 标定数据无效")
            return

        P1 = stereo_calib.P1
        P2 = stereo_calib.P2

        print(f"\n[运行] 双目模式: 相机 {cam_left} + {cam_right}")
        print(f"[运行] 标定: {calib_path} (基线 {stereo_calib.baseline:.2f}m)")
        print(f"[运行] 按 ESC 退出\n")

        capL = cv2.VideoCapture(cam_left)
        capR = cv2.VideoCapture(cam_right)
        if not capL.isOpened() or not capR.isOpened():
            print("[错误] 无法打开相机")
            return

        if self.landing_det:
            self.landing_det.reset()

        traj_3d = []
        fc = 0
        t0 = time.time()

        while fc < num_frames:
            retL, fL = capL.read()
            retR, fR = capR.read()
            if not retL or not retR:
                break

            fc += 1
            ts = time.time()
            ball_3d = None

            # TODO: 相机到货后接入 ball_detector.py
            # ballL = self.ball_detector.detect(fL)
            # ballR = self.ball_detector.detect(fR)
            # if ballL and ballR:
            #     pt4 = cv2.triangulatePoints(P1, P2,
            #         np.array(ballL[:2]).reshape(2,1),
            #         np.array(ballR[:2]).reshape(2,1))
            #     ball_3d = (pt4[:3]/pt4[3]).flatten()

            if ball_3d is not None:
                traj_3d.append(ball_3d)
                if self.tracker:
                    self.tracker.update(ball_3d, ts)
                if self.landing_det:
                    landing = self.landing_det.update(ball_3d, ts)
                    if landing and self.judge:
                        j = self.judge.judge_landing(landing["position"][0],
                                                     landing["position"][1])
                        print(f"  落地: {landing['position'][:2]} → {j}")

            cv2.imshow("Left", fL)
            cv2.imshow("Right", fR)

            if self.court_3d and traj_3d:
                self.court_3d.update_trajectory(np.array(traj_3d))

            if cv2.waitKey(1) & 0xFF == 27:
                break

        elapsed = time.time() - t0
        print(f"\n[统计] {fc}帧, {elapsed:.1f}s, {fc/elapsed:.1f} FPS")

        capL.release()
        capR.release()
        cv2.destroyAllWindows()

    # ── 视频模式 ────────────────────────────────────
    def run_video(self, video_path):
        """单视频模式（测试用）"""
        self._init_modules()

        if not os.path.exists(video_path):
            print(f"[错误] 视频不存在: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n[运行] 视频: {video_path}")
        print(f"  {total}帧, {fps:.1f} FPS\n")

        fc = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fc += 1

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if fc % 100 == 0:
                print(f"  进度: {fc}/{total}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[完成] {fc} 帧")


# ═══════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="双目羽毛球鹰眼判罚系统")
    parser.add_argument("--mode", default="simulate", choices=["simulate", "stereo", "video"])
    parser.add_argument("--calib", default="calib_stereo.json", help="双目标定 JSON")
    parser.add_argument("--cam-left", type=int, default=0)
    parser.add_argument("--cam-right", type=int, default=1)
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    system = HawkEyeSystem()

    if args.mode == "simulate":
        system.run_simulate()
    elif args.mode == "stereo":
        system.run_stereo(args.calib, args.cam_left, args.cam_right, args.frames)
    elif args.mode == "video":
        system.run_video(args.video or "videos/demo.mp4")


if __name__ == "__main__":
    main()
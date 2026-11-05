"""
双目立体视觉管线
================
- 基于场地线交点的手动标定 (无需棋盘格)
- 自动/手动对应点标注
- 立体标定 (内参 + 外参)
- 三角定位 (像素 → 真实世界 3D 坐标)
- 重投影误差验证
"""

import numpy as np
import cv2
import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger("StereoPipeline")


# ── 标准羽毛球场地关键点 (右手坐标系) ──────────────
# 原点: 球场左下角地面
# X: 球场长边方向 (13.40m)
# Y: 球场短边方向 (6.10m, 双打)
# Z: 竖直向上

COURT_3D_POINTS = np.array([
    # ── 底线/边线交点 ──
    [0.0,   0.0,   0.0],     # 0:  左下角 (双打边线)
    [13.40, 0.0,   0.0],     # 1:  右下角 (双打边线)
    [13.40, 6.10,  0.0],     # 2:  右上角 (双打边线)
    [0.0,   6.10,  0.0],     # 3:  左上角 (双打边线)
    # ── 单打边线交点 ──
    [0.0,   0.46,  0.0],     # 4:  左下 (单打)
    [13.40, 0.46,  0.0],     # 5:  右下 (单打)
    [13.40, 5.18,  0.0],     # 6:  右上 (单打)
    [0.0,   5.18,  0.0],     # 7:  左上 (单打)
    # ── 发球线交点 ──
    [1.98,  0.46,  0.0],     # 8:  前发球线 (右)
    [1.98,  5.18,  0.0],     # 9:  前发球线 (左)
    [11.42, 0.46,  0.0],     # 10: 后发球线 (右, 仅双打)
    [11.42, 5.18,  0.0],     # 11: 后发球线 (左, 仅双打)
    # ── 中线 ──
    [6.70,  0.46,  0.0],     # 12: 中线交点 (右)
    [6.70,  5.18,  0.0],     # 13: 中线交点 (左)
    # ── 网柱 (可选参考) ──
    [6.70,  0.0,   1.55],    # 14: 右网柱顶部
    [6.70,  6.10,  1.55],    # 15: 左网柱顶部
], dtype=np.float64)

COURT_POINT_NAMES = [
    "左下角(双打)", "右下角(双打)", "右上角(双打)", "左上角(双打)",
    "左下(单打)", "右下(单打)", "右上(单打)", "左上(单打)",
    "前发球线(右)", "前发球线(左)", "后发球线(右)", "后发球线(左)",
    "中线(右)", "中线(左)", "右网柱顶", "左网柱顶"
]


@dataclass
class StereoCalibration:
    """立体标定参数"""
    # ── 相机内参 ──
    K1: np.ndarray       # 相机 A 内参矩阵 (3×3)
    D1: np.ndarray       # 相机 A 畸变系数 (5,)
    K2: np.ndarray       # 相机 B 内参矩阵 (3×3)
    D2: np.ndarray       # 相机 B 畸变系数 (5,)

    # ── 立体外参 ──
    R: np.ndarray        # 旋转矩阵 (3×3), 从 A 到 B
    T: np.ndarray        # 平移向量 (3×1), 从 A 到 B (米)

    # ── 投影矩阵 ──
    P1: np.ndarray = None  # (3×4), 相机 A 投影矩阵
    P2: np.ndarray = None  # (3×4), 相机 B 投影矩阵

    # ── 基础/本质矩阵 ──
    E: np.ndarray = None   # 本质矩阵
    F: np.ndarray = None   # 基础矩阵

    # ── 标定质量 ──
    reprojection_error: float = 0.0
    n_points_used: int = 0

    def __post_init__(self):
        """自动计算派生矩阵"""
        if self.R is not None and self.T is not None:
            T_col = self.T.reshape(3, 1)
            self.P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
            self.P2 = self.K2 @ np.hstack([self.R, T_col])
            self.E = self._compute_essential()
            self.F = self._compute_fundamental()

    def _compute_essential(self) -> np.ndarray:
        """从 R, T 计算本质矩阵 E = [T]× · R"""
        T = self.T.ravel()
        T_cross = np.array([[0, -T[2], T[1]],
                            [T[2], 0, -T[0]],
                            [-T[1], T[0], 0]])
        return T_cross @ self.R

    def _compute_fundamental(self) -> np.ndarray:
        """从 E 和内参计算基础矩阵 F"""
        return np.linalg.inv(self.K2).T @ self.E @ np.linalg.inv(self.K1)

    def save(self, filepath: str):
        """保存标定到 JSON 文件"""
        data = {
            'K1': self.K1.tolist(), 'D1': self.D1.tolist(),
            'K2': self.K2.tolist(), 'D2': self.D2.tolist(),
            'R': self.R.tolist(), 'T': self.T.ravel().tolist(),
            'reprojection_error': self.reprojection_error,
            'n_points_used': self.n_points_used
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"标定已保存: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "StereoCalibration":
        """从 JSON 加载标定"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(
            K1=np.array(data['K1']), D1=np.array(data['D1']),
            K2=np.array(data['K2']), D2=np.array(data['D2']),
            R=np.array(data['R']), T=np.array(data['T']),
            reprojection_error=data.get('reprojection_error', 0.0),
            n_points_used=data.get('n_points_used', 0)
        )


class StereoTriangulator:
    """
    三角定位器
    从两个相机的像素坐标 → 真实世界 3D 坐标
    """

    def __init__(self, calib: StereoCalibration):
        self.calib = calib

    def undistort_points(self, pts_a: np.ndarray, pts_b: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        去畸变

        Args:
            pts_a: (N, 2) 或 (2,)  相机 A 像素坐标
            pts_b: (N, 2) 或 (2,)  相机 B 像素坐标

        Returns:
            (undistorted_a, undistorted_b)
        """
        pts_a = np.atleast_2d(pts_a).astype(np.float64)
        pts_b = np.atleast_2d(pts_b).astype(np.float64)

        ua = cv2.undistortPoints(pts_a, self.calib.K1, self.calib.D1,
                                 P=self.calib.K1)
        ub = cv2.undistortPoints(pts_b, self.calib.K2, self.calib.D2,
                                 P=self.calib.K2)

        # cv2.undistortPoints 返回 (N, 1, 2)
        ua = ua.reshape(-1, 2)
        ub = ub.reshape(-1, 2)
        return ua, ub

    def triangulate(self, uv_a: Tuple[float, float],
                    uv_b: Tuple[float, float]) -> np.ndarray:
        """
        单点三角定位

        Args:
            uv_a: 相机 A 中的像素坐标 (u, v)
            uv_b: 相机 B 中的像素坐标 (u, v)

        Returns:
            3D 坐标 (x, y, z), 单位: 米
        """
        pts_a = np.array([[uv_a[0], uv_a[1]]], dtype=np.float64)
        pts_b = np.array([[uv_b[0], uv_b[1]]], dtype=np.float64)

        # 去畸变
        ua, ub = self.undistort_points(pts_a, pts_b)

        # 三角定位
        points_4d = cv2.triangulatePoints(
            self.calib.P1, self.calib.P2,
            ua.T, ub.T
        )

        # 齐次 → 欧氏
        point_3d = points_4d[:3] / points_4d[3]
        return point_3d.ravel()

    def triangulate_batch(self, uvs_a: np.ndarray, uvs_b: np.ndarray
                          ) -> np.ndarray:
        """
        批量三角定位

        Args:
            uvs_a: (N, 2) 相机 A 的像素坐标
            uvs_b: (N, 2) 相机 B 的像素坐标

        Returns:
            (N, 3) 3D 坐标
        """
        if len(uvs_a) == 0:
            return np.zeros((0, 3))

        ua, ub = self.undistort_points(uvs_a, uvs_b)

        points_4d = cv2.triangulatePoints(
            self.calib.P1, self.calib.P2,
            ua.T, ub.T
        )

        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    def reprojection_error_single(self, uv_a: Tuple[float, float],
                                  uv_b: Tuple[float, float],
                                  point_3d: np.ndarray) -> float:
        """
        单点重投影误差

        将 3D 点分别投影回两个相机，
        计算与原始像素坐标的距离。
        """
        X = point_3d.reshape(4, 1) if len(point_3d) == 3 else point_3d.reshape(-1, 1)
        if X.shape[0] == 3:
            X = np.vstack([X, [1.0]])

        proj_a = self.calib.P1 @ X
        proj_b = self.calib.P2 @ X

        ua_proj = proj_a[0] / proj_a[2]
        va_proj = proj_a[1] / proj_a[2]
        ub_proj = proj_b[0] / proj_b[2]
        vb_proj = proj_b[1] / proj_b[2]

        err_a = np.sqrt((ua_proj - uv_a[0])**2 + (va_proj - uv_a[1])**2)
        err_b = np.sqrt((ub_proj - uv_b[0])**2 + (vb_proj - uv_b[1])**2)

        return float((err_a + err_b) / 2.0)


# ── 标定器 ────────────────────────────────────────

class CourtCalibrator:
    """
    基于场地线的立体相机标定

    流程:
      1. 在两个相机的画面中手动标注相同场地关键点
      2. cv2.stereoCalibrate() 求解内参 + 外参
      3. 验证重投影误差
    """

    def __init__(self, court_3d_points: np.ndarray = COURT_3D_POINTS):
        """
        Args:
            court_3d_points: 已知的场地 3D 点 (N, 3)
        """
        self.court_3d = court_3d_points
        self.n_points = len(court_3d)

    def calibrate(self,
                  image_points_a: np.ndarray,
                  image_points_b: np.ndarray,
                  image_size: Tuple[int, int] = (1280, 1024),
                  fix_aspect_ratio: bool = True,
                  zero_tangent_dist: bool = True
                  ) -> StereoCalibration:
        """
        执行立体标定

        Args:
            image_points_a: (N, 2) 或 (N, 1, 2)  相机 A 中标注的点
            image_points_b: (N, 2) 或 (N, 1, 2)  相机 B 中标注的点
            image_size:     (width, height)
            fix_aspect_ratio: 固定 fx=fy (简化模型)
            zero_tangent_dist: 零切向畸变 (羽毛球场景推荐)

        Returns:
            StereoCalibration
        """
        # 确保格式正确
        pts_a = np.array(image_points_a, dtype=np.float32).reshape(-1, 1, 2)
        pts_b = np.array(image_points_b, dtype=np.float32).reshape(-1, 1, 2)
        obj_pts = self.court_3d.reshape(-1, 1, 3).astype(np.float32)

        n = min(len(pts_a), len(pts_b), len(obj_pts))
        if n < 8:
            raise ValueError(f"至少需要 8 个对应点，当前仅有 {n} 个")

        pts_a = pts_a[:n]
        pts_b = pts_b[:n]
        obj_pts = obj_pts[:n]

        logger.info(f"使用 {n} 个对应点进行立体标定...")

        # ── 标定标志 ──
        flags = cv2.CALIB_FIX_K3
        if fix_aspect_ratio:
            flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if zero_tangent_dist:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST

        # 标定准则
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    100, 1e-8)

        # ── 立体标定 ──
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=[obj_pts],
            imagePoints1=[pts_a],
            imagePoints2=[pts_b],
            imageSize=image_size,
            criteria=criteria,
            flags=flags
        )

        # ── 计算重投影误差 ──
        reproj_err = self._compute_mean_reprojection(
            obj_pts, pts_a, pts_b, K1, D1, K2, D2, R, T
        )

        logger.info(f"标定完成: RMS 重投影误差 = {reproj_err:.4f} 像素")

        calib = StereoCalibration(
            K1=K1, D1=D1, K2=K2, D2=D2,
            R=R, T=T,
            reprojection_error=reproj_err, n_points_used=n
        )

        return calib

    def _compute_mean_reprojection(self,
                                   obj_pts, pts_a, pts_b,
                                   K1, D1, K2, D2, R, T) -> float:
        """计算平均重投影误差"""
        # 投影 3D 点到相机 A
        rvec_a = np.zeros((3, 1))
        tvec_a = np.zeros((3, 1))
        proj_a, _ = cv2.projectPoints(obj_pts, rvec_a, tvec_a, K1, D1)
        err_a = np.mean(np.sqrt(np.sum((proj_a - pts_a) ** 2, axis=2)))

        # 投影 3D 点到相机 B
        rvec_b, _ = cv2.Rodrigues(R)
        proj_b, _ = cv2.projectPoints(obj_pts, rvec_b, T, K2, D2)
        err_b = np.mean(np.sqrt(np.sum((proj_b - pts_b) ** 2, axis=2)))

        return float((err_a + err_b) / 2.0)

    # ── 交互式标注工具 ─────────────────────────────

    @staticmethod
    def annotate_points(image: np.ndarray,
                        point_names: List[str] = None,
                        window_name: str = "标注场地关键点"
                        ) -> np.ndarray:
        """
        交互式鼠标标注工具

        操作:
          - 左键点击: 标注下一个点
          - 右键点击: 撤销上一个点
          - ESC: 退出

        标注顺序: 按 COURT_POINT_NAMES 顺序

        Returns:
            (N, 2) 像素坐标
        """
        names = point_names or COURT_POINT_NAMES
        points: List[Tuple[float, float]] = []
        current_idx = 0
        display = image.copy()

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_idx, display
            if event == cv2.EVENT_LBUTTONDOWN and current_idx < len(names):
                points.append((float(x), float(y)))
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(display, f"{current_idx}: {names[current_idx]}",
                           (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (0, 255, 0), 1)
                current_idx += 1
                cv2.imshow(window_name, display)
            elif event == cv2.EVENT_RBUTTONDOWN and len(points) > 0:
                points.pop()
                current_idx -= 1
                display = image.copy()
                for i, (px, py) in enumerate(points):
                    cv2.circle(display, (int(px), int(py)), 5, (0, 255, 0), -1)
                    cv2.putText(display, f"{i}: {names[i]}",
                               (int(px) + 10, int(py)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imshow(window_name, display)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        # 显示提示
        display = image.copy()
        hint = f"左键标注点 {current_idx}/{len(names)}: {names[current_idx]} | 右键撤销 | ESC完成"
        cv2.putText(display, hint, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow(window_name, display)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if current_idx >= len(names):
                cv2.putText(display, "所有点已标注! 按 ESC 完成",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 255), 1)
                cv2.imshow(window_name, display)

        cv2.destroyWindow(window_name)
        return np.array(points, dtype=np.float32)


# ── 便利函数 ──────────────────────────────────────

def load_or_calibrate(calib_path: str,
                      img_a: np.ndarray,
                      img_b: np.ndarray) -> StereoCalibration:
    """
    如果标定文件存在则加载, 否则交互式标定
    """
    if os.path.exists(calib_path):
        logger.info(f"加载已有标定: {calib_path}")
        return StereoCalibration.load(calib_path)

    logger.info("开始交互式标定...")

    calibrator = CourtCalibrator()

    # 标注相机 A
    print("\n--- 请标注 相机 A 的场地关键点 ---")
    print("按标注顺序点击，右键撤销，ESC完成")
    pts_a = calibrator.annotate_points(img_a, window_name="相机A - 标注场地点")

    # 标注相机 B
    print("\n--- 请标注 相机 B 的场地关键点 ---")
    print("请按相同顺序标注 (点的数量必须一致)")
    pts_b = calibrator.annotate_points(img_b, window_name="相机B - 标注场地点")

    # 执行标定
    calib = calibrator.calibrate(pts_a, pts_b,
                                 image_size=(img_a.shape[1], img_a.shape[0]))

    calib.save(calib_path)
    return calib


# ── 自检 ──────────────────────────────────────────

if __name__ == "__main__":
    print("StereoPipeline 模块已就绪。")
    print(f"内置场地模型: {len(COURT_3D_POINTS)} 个关键点")
    print()
    print("标定流程：")
    print("  1. 用相机 A 和 B 同时拍摄场地照片")
    print("  2. calibrator = CourtCalibrator()")
    print("  3. pts_a = calibrator.annotate_points(img_a)")
    print("  4. pts_b = calibrator.annotate_points(img_b)")
    print("  5. calib = calibrator.calibrate(pts_a, pts_b)")
    print("  6. calib.save('calibration.json')")
    print()
    print("使用：")
    print("  calib = StereoCalibration.load('calibration.json')")
    print("  tri = StereoTriangulator(calib)")
    print("  point_3d = tri.triangulate((u1,v1), (u2,v2))")
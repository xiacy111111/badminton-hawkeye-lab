"""
court3d.py — 羽毛球场 3D 模型 + PnP 相机定位
无论摄像头怎么摆，自动定位并映射坐标
"""

import numpy as np
import cv2


# ============================================================
# 标准羽毛球场 3D 模型（世界坐标系，米）
# 原点 = 球场左下角  X→右(宽)  Y→上(长)  Z=0 为地面
# ============================================================

class Court3DModel:
    """羽毛球场完整 3D 特征点模型"""

    def __init__(self, doubles=True):
        self.W = 6.10 if doubles else 5.18  # 宽度
        self.L = 13.40                       # 长度
        self.doubles = doubles

        # 定义所有关键 3D 点: {名称: (X, Y, Z)}
        pts = {}
        W, L = self.W, self.L
        sw = (self.W - 5.18) / 2 if doubles else 0  # 单打边线偏移
        net_y = L / 2  # 网在中间

        # ---- 四个角 ----
        pts["近端底线左"] = (0, 0, 0)
        pts["近端底线右"] = (W, 0, 0)
        pts["远端底线左"] = (0, L, 0)
        pts["远端底线右"] = (W, L, 0)

        # ---- 网柱 ----
        pts["近端网柱左"] = (0, net_y, 0)
        pts["近端网柱右"] = (W, net_y, 0)
        pts["远端网柱左"] = (0, net_y, 0)  # 网柱在两侧对称
        pts["远端网柱右"] = (W, net_y, 0)

        # ---- 前发球线与边线交点 ----
        pts["近端前发球线左"] = (0, net_y + 1.98, 0)
        pts["近端前发球线右"] = (W, net_y + 1.98, 0)
        pts["远端前发球线左"] = (0, net_y - 1.98, 0)
        pts["远端前发球线右"] = (W, net_y - 1.98, 0)

        # ---- 中线端点 ----
        pts["近端中线端点"] = (W/2, 0, 0)     # 近端底线中点
        pts["远端中线端点"] = (W/2, L, 0)     # 远端底线中点
        pts["近端中线_前发球"] = (W/2, net_y + 1.98, 0)
        pts["远端中线_前发球"] = (W/2, net_y - 1.98, 0)

        # ---- T 点 ----
        pts["近端T点"] = (W/2, net_y + 1.98, 0)   # 同中线_前发球
        pts["远端T点"] = (W/2, net_y - 1.98, 0)

        # ---- 后发球线（双打） ----
        pts["近端后发球线左"] = (0, 0.76, 0)
        pts["近端后发球线右"] = (W, 0.76, 0)
        pts["远端后发球线左"] = (0, L - 0.76, 0)
        pts["远端后发球线右"] = (W, L - 0.76, 0)

        # ---- 单打边线与底线交点 ----
        if doubles:
            pts["近端单打左上"] = (sw, 0, 0)
            pts["近端单打右上"] = (W - sw, 0, 0)
            pts["远端单打左上"] = (sw, L, 0)
            pts["远端单打右上"] = (W - sw, L, 0)

        # 去重：同名不同 ID，名字唯一化
        unique = {}
        for name, coord in pts.items():
            key = f"{name}_{round(coord[0],3)}_{round(coord[1],3)}"
            unique[key] = (name, coord)
        self.points = {v[0]: v[1] for v in unique.values()}

        # 按名称排序
        self.point_names = sorted(self.points.keys())
        print(f"球场 3D 模型: {len(self.points)} 个特征点 | {W:.2f}m × {L:.2f}m")

    def get_court_3d_points(self):
        """返回所有 3D 点坐标列表"""
        return [self.points[n] for n in self.point_names]


class CourtPnPSolver:
    """
    从单帧图像 + 球场 3D 模型 → 自动定位摄像头
    不需要任何标定板，球场本身就是标定靶
    """

    def __init__(self, court_model: Court3DModel, camera_K=None):
        """
        camera_K: 3×3 相机内参矩阵（可选）
                  如果不给，会自动估算
        """
        self.model = court_model
        self.camera_K = camera_K  # 如果为 None → 自动估算
        self.rvec = None  # 相机旋转
        self.tvec = None  # 相机平移（世界坐标中摄像头位置）
        self.solved = False

    def _estimate_intrinsics(self, image_shape):
        """没有标定参数时，从图像尺寸估算"""
        h, w = image_shape[:2]
        fx = fy = max(w, h)  # 近似焦距
        cx, cy = w / 2, h / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def _detect_court_points_2d(self, image):
        """
        从图像中检测球场线的交点（2D 像素坐标）
        返回: [(px, py), ...]
        """
        h, w = image.shape[:2]

        # 提取白色/黄色线
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        white = ((s <= 90) & (v >= 140)).astype(np.uint8) * 255
        yellow = ((hsv[:,:,0] >= 12) & (hsv[:,:,0] <= 42) &
                  (hsv[:,:,1] >= 40) & (hsv[:,:,2] >= 120)).astype(np.uint8) * 255
        lines_binary = cv2.bitwise_or(white, yellow)

        # 形态学：连接断线，去除噪点
        kernel = np.ones((3, 3), np.uint8)
        lines_binary = cv2.morphologyEx(lines_binary, cv2.MORPH_CLOSE, kernel)

        # Hough 直线检测
        edges = cv2.Canny(lines_binary, 50, 150)
        raw_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                     minLineLength=min(w,h)//10, maxLineGap=20)

        if raw_lines is None:
            return []

        # 兼容不同 OpenCV 版本的返回格式
        if raw_lines.ndim == 3:
            raw_lines = raw_lines[:, 0, :]

        # 方向分组：水平 vs 垂直
        h_lines = []
        v_lines = []
        for x1, y1, x2, y2 in raw_lines:
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1))) % 180
            if angle < 20 or angle > 160:
                h_lines.append((x1, y1, x2, y2))
            elif 70 < angle < 110:
                v_lines.append((x1, y1, x2, y2))

        # 计算水平线和垂直线的交点
        intersections = []
        for hx1, hy1, hx2, hy2 in h_lines[:15]:
            for vx1, vy1, vx2, vy2 in v_lines[:15]:
                pt = self._line_intersection(hx1, hy1, hx2, hy2, vx1, vy1, vx2, vy2)
                if pt is not None:
                    px, py = pt
                    if 0 <= px < w and 0 <= py < h:
                        intersections.append((float(px), float(py)))

        # 去重（聚类相近的点）
        return self._cluster_points(intersections, threshold=30)

    def _line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
        py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
        return (px, py)

    def _cluster_points(self, points, threshold):
        if not points:
            return []
        pts = np.array(points)
        clusters = []
        used = set()
        for i in range(len(pts)):
            if i in used:
                continue
            cluster = [pts[i]]
            used.add(i)
            for j in range(i+1, len(pts)):
                if j in used:
                    continue
                if np.linalg.norm(pts[i] - pts[j]) < threshold:
                    cluster.append(pts[j])
                    used.add(j)
            clusters.append(np.mean(cluster, axis=0))
        return [tuple(c) for c in clusters]

    def solve(self, image):
        """
        从单帧图像求解相机位置
        返回: (success, camera_position_3d)
        """
        # 1. 检测 2D 点
        points_2d = self._detect_court_points_2d(image)
        print(f"  检测到 {len(points_2d)} 个球场交点")

        if len(points_2d) < 4:
            print(f"  需要至少 4 个交点，实际找到 {len(points_2d)} 个")
            return False, None

        # 2. 匹配 3D 点（简化：按位置自动匹配）
        h, w = image.shape[:2]
        court_3d = list(self.model.points.values())
        court_names = list(self.model.points.keys())

        # 自动匹配策略：按图像中点的相对位置 → 匹配到最可能的 3D 点
        matches = self._auto_match_2d_to_3d(points_2d, court_3d, (w, h))

        if len(matches) < 4:
            print(f"  仅匹配到 {len(matches)} 对")
            return False, None

        matched_2d = np.array([m[0] for m in matches], dtype=np.float64)
        matched_3d = np.array([m[1] for m in matches], dtype=np.float64)

        # 3. PnP 求解
        K = self.camera_K if self.camera_K is not None else self._estimate_intrinsics(image.shape)
        success, rvec, tvec = cv2.solvePnP(matched_3d, matched_2d, K, None)

        if success:
            self.rvec = rvec
            self.tvec = tvec
            self.camera_K = K
            self.solved = True
            cam_pos = (-np.linalg.inv(cv2.Rodrigues(rvec)[0]) @ tvec).ravel()
            print(f"  相机定位成功!")
            print(f"  相机位置: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})m")
            return True, cam_pos
        else:
            print(f"  PnP 求解失败")
            return False, None

    def _auto_match_2d_to_3d(self, points_2d, court_3d, image_size):
        """自动匹配：利用球场对称性 + 相对位置"""
        w, h = image_size
        matches = []

        # 简化匹配策略：
        # 将 2D 点和 3D 点都归一化到 [0,1] 范围
        # 用最近邻匹配
        pts2d_norm = np.array(points_2d) / [w, h]
        pts3d_norm = np.array([(p[0]/self.model.W, p[1]/self.model.L) for p in court_3d])

        # 对于每个 3D 点，找最近的 2D 点
        used_2d = set()
        for i, p3d in enumerate(pts3d_norm):
            best_j, best_dist = -1, float('inf')
            for j, p2d in enumerate(pts2d_norm):
                if j in used_2d:
                    continue
                dist = np.linalg.norm(p2d - p3d)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j >= 0 and best_dist < 0.5:  # 阈值
                matches.append((points_2d[best_j], court_3d[i]))
                used_2d.add(best_j)

        return matches

    def pixel_to_world(self, px, py):
        """
        单像素 → 世界 3D 坐标
        射线与 Z=0 平面（地面）的交点
        """
        if not self.solved:
            return None

        K = self.camera_K
        R = cv2.Rodrigues(self.rvec)[0]
        t = self.tvec

        # 反向投影 → 射线方向
        pt_cam = np.linalg.inv(K) @ np.array([px, py, 1.0])
        pt_world_dir = R.T @ pt_cam

        # 射线与 Z=0 平面交点
        cam_pos = -R.T @ t.ravel()
        if abs(pt_world_dir[2]) < 1e-9:
            return None

        lam = -cam_pos[2] / pt_world_dir[2]
        world_pt = cam_pos + lam * pt_world_dir

        return float(world_pt[0]), float(world_pt[1]), 0.0


def test_with_image(image_path):
    """用一张球场照片测试"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"图片读取失败: {image_path}")
        return None

    model = Court3DModel(doubles=True)
    solver = CourtPnPSolver(model)

    success, cam_pos = solver.solve(image)

    if success:
        # 可视化：在图像上标注检测到的交点
        display = image.copy()
        # 投射所有 3D 点回图像验证
        K = solver.camera_K
        for name, pt3d in model.points.items():
            pt3d_arr = np.array([pt3d], dtype=np.float32)
            projected, _ = cv2.projectPoints(pt3d_arr, solver.rvec, solver.tvec, K, None)
            px, py = int(projected[0,0,0]), int(projected[0,0,1])
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                cv2.circle(display, (px, py), 4, (0, 255, 0), -1)

        cv2.imwrite("court3d_projection.png", display)
        print("投影图已保存: court3d_projection.png (绿点 = 3D模型投影回图像)")
    else:
        print("定位失败，请确保球场线清晰可见")

    return solver


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_with_image(sys.argv[1])
    else:
        print("用法: python court3d.py 球场照片.jpg")
        print()
        print("拍一张球场照片（能看到至少4个线交点），自动定位摄像头")
        print("此后摄像头无论怎么移动，每帧重新 solve 即可")
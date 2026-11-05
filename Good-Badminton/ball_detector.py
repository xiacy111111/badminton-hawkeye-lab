"""
羽毛球视觉检测模块
==================
针对羽毛球极小目标 (3~15像素) 优化的检测流程:

1. 帧差法 → 运动区域
2. 自适应阈值 → 二值化
3. 运动一致性滤波 → 锁定目标
4. 灰度重心法 → 子像素坐标
5. 质量评分 → 检测置信度

设计考量:
  - MV-CU013-A0UC: 1280×1024, 201fps, 全局快门
  - 6mm 镜头 @ 10m: 球 ≈ 3~8 像素 (极难)
  - 12mm 镜头 @ 10m: 球 ≈ 7~15 像素 (可接受)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


@dataclass
class BallDetection:
    """单帧球检测结果"""
    # 像素坐标 (子像素精度)
    u: float                       # 列坐标
    v: float                       # 行坐标
    # 检测属性
    confidence: float              # 0~1 置信度
    size_px: float                 # 估计大小 (像素)
    brightness: float              # 区域平均亮度
    # 运动信息
    motion_dx: float = 0.0         # 帧间运动 (像素)
    motion_dy: float = 0.0
    speed_px_per_s: float = 0.0
    # 时间
    timestamp: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        return (self.u, self.v)


class BallDetector:
    """
    羽毛球检测器

    策略：
      不是每帧全图搜索——用上一帧位置 + 速度预测
      缩小搜索窗口到 RO，大幅降低误检。
      只在丢失时做全图搜索。
    """

    def __init__(self,
                 min_ball_px: int = 2,        # 最小球尺寸 (像素)
                 max_ball_px: int = 20,       # 最大球尺寸 (像素)
                 motion_threshold: int = 8,   # 帧差阈值
                 search_margin: int = 40,     # RO 搜索边距 (像素)
                 history_frames: int = 5,     # 帧差历史帧数
                 min_circularity: float = 0.3 # 最小圆度
                 ):
        self.min_ball_px = min_ball_px
        self.max_ball_px = max_ball_px
        self.motion_threshold = motion_threshold
        self.search_margin = search_margin
        self.history_frames = history_frames
        self.min_circularity = min_circularity

        # ── 内部状态 ──
        self.prev_frame: Optional[np.ndarray] = None   # 前帧 (用于帧差)
        self.frame_history: List[np.ndarray] = []       # 历史帧
        self.last_detection: Optional[BallDetection] = None
        self.consecutive_misses: int = 0
        self.is_initialized: bool = False

        # ── 自适应背景 ──
        self.bg_model: Optional[np.ndarray] = None
        self.bg_alpha: float = 0.02   # 背景更新速率

    # ── 主入口 ─────────────────────────────────────

    def detect(self, frame: np.ndarray, timestamp: float = None) -> Optional[BallDetection]:
        """
        主检测入口

        Args:
            frame:     当前帧 (H, W) 灰度图 或 (H, W, 3) 彩色图
            timestamp: 帧时间戳

        Returns:
            BallDetection 或 None
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        # ── 如果有上一帧检测，做局部搜索 ──
        if self.last_detection is not None and self.consecutive_misses < 10:
            detection = self._local_search(gray, timestamp)
            if detection is not None and detection.confidence > 0.5:
                self._update_state(gray, detection)
                return detection

        # ── 局部搜索失败 → 全图搜索 ──
        detection = self._global_search(gray, timestamp)
        self._update_state(gray, detection)
        return detection

    # ── 局部搜索 ──────────────────────────────────

    def _local_search(self, gray: np.ndarray, timestamp: float) -> Optional[BallDetection]:
        """在预测位置附近搜索球"""
        # 预测位置 (简单匀速模型)
        pred_u = int(self.last_detection.u + self.last_detection.motion_dx)
        pred_v = int(self.last_detection.v + self.last_detection.motion_dy)

        # RO (带边距)
        h, w = gray.shape
        margin = self.search_margin
        u1 = max(0, pred_u - margin)
        v1 = max(0, pred_v - margin)
        u2 = min(w, pred_u + margin)
        v2 = min(h, pred_v + margin)

        # 如果 RO 太小，扩展
        if (u2 - u1) < 20 or (v2 - v1) < 20:
            u1 = max(0, pred_u - 30)
            v1 = max(0, pred_v - 30)
            u2 = min(w, pred_u + 30)
            v2 = min(h, pred_v + 30)

        roi = gray[v1:v2, u1:u2]

        # 帧差 (与上一帧对应 RO)
        if self.prev_frame is not None:
            prev_roi = self.prev_frame[v1:v2, u1:u2]
            diff = cv2.absdiff(roi, prev_roi)
        else:
            diff = np.zeros_like(roi)

        # 找运动区域
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = 0.0
        best_info = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_ball_px ** 2 or area > self.max_ball_px ** 2:
                continue

            # 圆度
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < self.min_circularity:
                continue

            # 距预测位置的距离
            M = cv2.moments(cnt)
            if M['m00'] < 1e-6:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            # 转换到全图坐标
            cx_global = cx + u1
            cy_global = cy + v1

            # 评分: 圆度 + 靠近预测位置
            dist_to_pred = np.sqrt((cx_global - pred_u) ** 2 + (cy_global - pred_v) ** 2)
            position_score = max(0, 1.0 - dist_to_pred / self.search_margin)

            score = 0.4 * circularity + 0.6 * position_score

            if score > best_score:
                best_score = score
                best_contour = cnt
                best_info = (cx, cy, area, circularity)

        if best_info is None:
            return None

        cx, cy, area, circularity = best_info
        cx_global = cx + u1
        cy_global = cy + v1

        # 子像素：灰度重心法
        cx_sub, cy_sub = self._subpixel_refine(gray, cx_global, cy_global)

        # 运动矢量
        if self.last_detection is not None:
            mdx = cx_sub - self.last_detection.u
            mdy = cy_sub - self.last_detection.v
        else:
            mdx, mdy = 0.0, 0.0

        # RO 内平均亮度
        brightness = np.mean(roi[thresh > 0]) if np.any(thresh > 0) else 0.0

        return BallDetection(
            u=cx_sub, v=cy_sub,
            confidence=best_score if best_score > 0.5 else best_score * 0.7,
            size_px=np.sqrt(area),
            brightness=brightness,
            motion_dx=mdx, motion_dy=mdy,
            timestamp=timestamp
        )

    # ── 全图搜索 ──────────────────────────────────

    def _global_search(self, gray: np.ndarray, timestamp: float) -> Optional[BallDetection]:
        """全图搜索球 (丢失重捕获)"""
        candidates = []

        # 方法 1: 帧差法
        if self.prev_frame is not None:
            diff = cv2.absdiff(gray, self.prev_frame)
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_ball_px ** 2 or area > self.max_ball_px ** 2:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter < 1e-6:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                if circularity < self.min_circularity:
                    continue

                M = cv2.moments(cnt)
                if M['m00'] < 1e-6:
                    continue
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']

                # 子像素
                cx_sub, cy_sub = self._subpixel_refine(gray, cx, cy)

                candidates.append({
                    'u': cx_sub, 'v': cy_sub,
                    'area': area,
                    'circularity': circularity,
                    'score': circularity * min(1.0, area / 25.0)
                })

        # 方法 2: 如果帧差没有候选，尝试自适应阈值 + 运动历史
        # (对极小目标更敏感)
        if len(candidates) == 0 and len(self.frame_history) >= 2:
            # 多帧中值作为伪背景
            bg = np.median(np.stack(self.frame_history[-3:]), axis=0).astype(np.uint8)
            diff = cv2.absdiff(gray, bg)

            # 更低的阈值，捕捉微弱信号
            _, thresh = cv2.threshold(diff, self.motion_threshold * 0.5, 255, cv2.THRESH_BINARY)

            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_ball_px ** 2 or area > self.max_ball_px ** 2:
                    continue

                M = cv2.moments(cnt)
                if M['m00'] < 1e-6:
                    continue
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                cx_sub, cy_sub = self._subpixel_refine(gray, cx, cy)

                candidates.append({
                    'u': cx_sub, 'v': cy_sub,
                    'area': area,
                    'circularity': 0.5,  # 不依赖圆度
                    'score': min(1.0, area / 25.0) * 0.6  # 降权
                })

        # 选最高分
        if not candidates:
            return None

        best = max(candidates, key=lambda c: c['score'])

        # 额外验证: 如果上一帧有检测，检查一致性
        if self.last_detection is not None:
            dist = np.sqrt((best['u'] - self.last_detection.u) ** 2 +
                          (best['v'] - self.last_detection.v) ** 2)
            # 201fps 下球最大移动 ≈ 50 像素
            if dist > 60:
                best['score'] *= 0.3

        mdx = best['u'] - self.last_detection.u if self.last_detection else 0.0
        mdy = best['v'] - self.last_detection.v if self.last_detection else 0.0

        return BallDetection(
            u=best['u'], v=best['v'],
            confidence=min(1.0, best['score']),
            size_px=np.sqrt(best['area']),
            brightness=0.0,
            motion_dx=mdx, motion_dy=mdy,
            timestamp=timestamp
        )

    # ── 子像素精化 ────────────────────────────────

    def _subpixel_refine(self, gray: np.ndarray,
                         cx: float, cy: float,
                         window: int = 5) -> Tuple[float, float]:
        """
        灰度重心法做子像素定位

        在 (cx, cy) 周围的 window×window 窗口内，
        用灰度值加权重心替代几何中心。

        精度: ~0.2 像素 (对 6~10 像素大小的球)
        """
        h, w = gray.shape
        half = window // 2
        x0 = max(0, int(cx) - half)
        y0 = max(0, int(cy) - half)
        x1 = min(w, int(cx) + half + 1)
        y1 = min(h, int(cy) + half + 1)

        roi = gray[y0:y1, x0:x1].astype(np.float32)

        # 减去局部最小值（去除背景）
        roi_min = np.min(roi)
        roi = roi - roi_min
        roi[roi < 0] = 0

        total = np.sum(roi)
        if total < 1e-6:
            return float(cx), float(cy)

        ys, xs = np.mgrid[y0:y1, x0:x1]
        cx_refined = np.sum(xs * roi) / total
        cy_refined = np.sum(ys * roi) / total

        # 限制子像素偏移不超过 1 像素（防止异常值）
        if abs(cx_refined - cx) > 1.0:
            cx_refined = float(cx)
        if abs(cy_refined - cy) > 1.0:
            cy_refined = float(cy)

        return float(cx_refined), float(cy_refined)

    # ── 状态更新 ──────────────────────────────────

    def _update_state(self, gray: np.ndarray,
                      detection: Optional[BallDetection]):
        """更新内部状态"""
        # 帧历史
        self.frame_history.append(gray.copy())
        if len(self.frame_history) > self.history_frames + 5:
            self.frame_history = self.frame_history[-self.history_frames:]

        # 背景更新
        if self.bg_model is None:
            self.bg_model = gray.astype(np.float32)
        else:
            cv2.accumulateWeighted(gray, self.bg_model, self.bg_alpha)

        # 检测状态
        self.prev_frame = gray.copy()

        if detection is not None:
            self.last_detection = detection
            self.consecutive_misses = 0
            self.is_initialized = True
        else:
            self.consecutive_misses += 1

    @property
    def is_lost(self) -> bool:
        """球是否丢失 (>10 帧未检测)"""
        return self.consecutive_misses > 10

    @property
    def predicted_position(self) -> Optional[Tuple[float, float]]:
        """基于上一帧 + 运动矢量预测当前位置"""
        if self.last_detection is None:
            return None
        return (
            self.last_detection.u + self.last_detection.motion_dx,
            self.last_detection.v + self.last_detection.motion_dy
        )

    def reset(self):
        """重置检测器状态"""
        self.prev_frame = None
        self.frame_history.clear()
        self.bg_model = None
        self.last_detection = None
        self.consecutive_misses = 0
        self.is_initialized = False


# ── 自检 ──────────────────────────────────────────

if __name__ == "__main__":
    print("BallDetector 模块已就绪。")
    print()
    print("使用示例：")
    print("  detector = BallDetector(min_ball_px=2, max_ball_px=20)")
    print("  cap = cv2.VideoCapture(0)")
    print("  while True:")
    print("      ret, frame = cap.read()")
    print("      det = detector.detect(frame)")
    print("      if det:")
    print("          cv2.circle(frame, (int(det.u), int(det.v)), 3, (0,255,0), -1)")
    print("      cv2.imshow('Detection', frame)")
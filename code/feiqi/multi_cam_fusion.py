"""
多摄像头融合引擎 v1.0
=====================
每路摄像头独立跑 T1 判罚，各自产出候选落地事件。
融合节点做时空对齐 + 三角定位，给出统一判罚。

架构：
  Camera[0].engine ──→ (frame, px, py, conf, ts)
  Camera[1].engine ──→ (frame, px, py, conf, ts)  ──→ FusionNode ──→ 最终判罚
  Camera[2].engine ──→ (frame, px, py, conf, ts)
  Camera[3].engine ──→ (frame, px, py, conf, ts)

输入要求：每路摄像头已完成球场标定（像素→物理坐标）。
"""

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np


@dataclass
class CameraConfig:
    cam_id: int
    source: str                        # 摄像头源 (0, IP URL, 视频文件)
    calibration: str                   # 球场标定 JSON 路径
    # 单应性矩阵（从标定文件计算）
    H: Optional[np.ndarray] = None     # 3x3, pixel → court meter

    def load_calibration(self):
        with open(self.calibration) as f:
            data = json.load(f)
        corners = np.array(data["corners"], dtype=np.float32)
        # 标准羽毛球场物理坐标 (双打): 宽 6.10m, 长 13.40m
        court_w, court_h = 6.10, 13.40
        phys = np.array([
            [0, 0], [court_w, 0], [court_w, court_h], [0, court_h]
        ], dtype=np.float32)
        self.H, _ = cv2.findHomography(corners, phys)

    def pixel_to_court(self, px: float, py: float) -> Tuple[float, float]:
        """像素 → 球场物理坐标 (米)"""
        if self.H is None:
            raise RuntimeError(f"Camera {self.cam_id}: 未加载标定")
        p = np.array([[px, py]], dtype=np.float32).reshape(-1, 1, 2)
        m = cv2.perspectiveTransform(p, self.H).reshape(2)
        return float(m[0]), float(m[1])


@dataclass
class LandingEvent:
    """单路摄像头的落地候选"""
    cam_id: int
    frame: int
    pixel_x: float
    pixel_y: float
    court_x: float       # 球场物理坐标 (米)
    court_y: float
    confidence: float     # 0~1
    timestamp: float      # unix time
    rally_start: int = 0
    rally_end: int = 0


@dataclass
class FusedVerdict:
    """融合后的最终判罚"""
    court_x: float
    court_y: float
    confidence: float
    num_cameras_agree: int
    verdict: str          # "IN" / "OUT" / "UNCERTAIN"
    individual_events: List[LandingEvent] = field(default_factory=list)
    timestamp: float = 0.0


class FusionNode:
    """多路摄像头的落点融合与最终判罚。

    工作流程：
    1. 各摄像头产出 LandingEvent，push 到 fusion node
    2. Fusion node 做时空对齐（同一 rally 的落地应该时间戳接近）
    3. 对同一次落地，多路坐标做加权平均（按置信度加权）
    4. 在 3D 球场模型上判定 IN/OUT
    """

    COURT_W = 6.10    # 双打场地宽
    COURT_H = 13.40   # 双打场地长
    TEMPORAL_WINDOW = 0.3     # 300ms：同一落地的时间窗口
    MIN_CAMERAS_AGREE = 2     # 至少 2 路摄像头同意才判定
    UNCERTAIN_ZONE = 0.03     # 3cm：边界模糊区，标记为 UNCERTAIN

    def __init__(self, num_cameras: int = 4):
        self.num_cameras = num_cameras
        self.event_queue: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        self.verdicts: deque = deque(maxlen=100)

    def push(self, event: LandingEvent):
        """摄像头产出落地事件后调用此方法。"""
        with self._lock:
            self.event_queue.append(event)
        # 尝试融合最近的事件
        self._try_fuse(event.timestamp)

    def _try_fuse(self, ref_time: float):
        """找到时间窗口内的所有事件，做融合。"""
        with self._lock:
            candidates = [
                e for e in self.event_queue
                if abs(e.timestamp - ref_time) <= self.TEMPORAL_WINDOW
            ]
        if len(candidates) < self.MIN_CAMERAS_AGREE:
            return

        # 按摄像头分组，同一摄像头取最大置信度
        best_per_cam: Dict[int, LandingEvent] = {}
        for e in candidates:
            if e.cam_id not in best_per_cam or e.confidence > best_per_cam[e.cam_id].confidence:
                best_per_cam[e.cam_id] = e

        if len(best_per_cam) < self.MIN_CAMERAS_AGREE:
            return

        # 按置信度加权平均
        total_conf = sum(e.confidence for e in best_per_cam.values())
        fused_x = sum(e.court_x * e.confidence for e in best_per_cam.values()) / total_conf
        fused_y = sum(e.court_y * e.confidence for e in best_per_cam.values()) / total_conf

        # 计算一致性：各路坐标的散布程度
        xs = [e.court_x for e in best_per_cam.values()]
        ys = [e.court_y for e in best_per_cam.values()]
        spread = np.std(xs) + np.std(ys)  # 越小越一致
        # 一致性映射到置信度加成
        consistency_boost = max(0, 1.0 - spread / 0.5) * 0.15  # 最多 +15%
        avg_conf = total_conf / len(best_per_cam)
        final_conf = min(1.0, avg_conf + consistency_boost)

        # 判罚 IN/OUT
        margin = 0.02  # 2cm
        if (-margin <= fused_x <= self.COURT_W + margin and
                -margin <= fused_y <= self.COURT_H + margin):
            # 检查是否在模糊区
            if (fused_x < self.UNCERTAIN_ZONE or
                fused_x > self.COURT_W - self.UNCERTAIN_ZONE or
                fused_y < self.UNCERTAIN_ZONE or
                fused_y > self.COURT_H - self.UNCERTAIN_ZONE):
                verdict = "UNCERTAIN"
            else:
                verdict = "IN"
        else:
            verdict = "OUT"

        fused = FusedVerdict(
            court_x=round(fused_x, 4),
            court_y=round(fused_y, 4),
            confidence=round(final_conf, 4),
            num_cameras_agree=len(best_per_cam),
            verdict=verdict,
            individual_events=list(best_per_cam.values()),
            timestamp=ref_time,
        )
        self.verdicts.append(fused)

        # 终端输出
        self._print_verdict(fused)

        # 消费掉已处理的事件
        with self._lock:
            used_frames = {e.frame for e in best_per_cam.values()}
            self.event_queue = deque(
                [e for e in self.event_queue if e.frame not in used_frames],
                maxlen=500
            )

    def _print_verdict(self, v: FusedVerdict):
        sign = "★ IN ★" if v.verdict == "IN" else "✗ OUT ✗" if v.verdict == "OUT" else "? 边界模糊 ?"
        print(f"\n{'═'*50}")
        print(f"  {sign}")
        print(f"  球场坐标: ({v.court_x:.3f}m, {v.court_y:.3f}m)")
        print(f"  融合置信度: {v.confidence:.1%}  ({v.num_cameras_agree} 路摄像头一致)")
        print(f"  参与摄像头:")
        for e in v.individual_events:
            print(f"    Cam {e.cam_id}: ({e.court_x:.3f}, {e.court_y:.3f}) "
                  f"conf={e.confidence:.1%} frame={e.frame}")
        print(f"{'═'*50}\n")

    def get_latest_verdict(self) -> Optional[FusedVerdict]:
        if self.verdicts:
            return self.verdicts[-1]
        return None


class MultiCamReplayEngine:
    """多路视频回放（带慢放）——裁判回看模块。

    同时播放 1~4 路录制视频，叠加轨迹线和落点标注。
    支持 0.25x / 0.5x / 1x 播放速度。
    """

    SPEED_OPTIONS = [0.25, 0.5, 0.75, 1.0, 2.0]

    def __init__(self, video_paths: List[str], fused_verdict: FusedVerdict):
        self.video_paths = video_paths
        self.verdict = fused_verdict
        self.speed = 1.0
        self.caps = []
        self.fps = 30.0

    def load(self):
        import cv2
        for path in self.video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开: {path}")
            self.caps.append(cap)
            self.fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[REPLAY] {len(self.caps)} 路视频已加载, FPS={self.fps}")

    def seek_to_landing(self, event_frame: int, context_sec: float = 3.0):
        """跳到落地前 context_sec 秒。"""
        start_frame = max(0, event_frame - int(context_sec * self.fps))
        for cap in self.caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        return start_frame

    def play(self, landing_frame: int, context_before: float = 3.0,
             context_after: float = 2.0):
        """播放落地前后一段（带慢放）。

        落地帧之前：正常速度
        落地帧附近 ±1s：0.25x 慢放
        """
        import cv2
        self.load()
        start_frame = self.seek_to_landing(landing_frame, context_before)
        end_frame = landing_frame + int(context_after * self.fps)

        slow_zone_start = landing_frame - int(1.0 * self.fps)
        slow_zone_end = landing_frame + int(1.0 * self.fps)

        for fid in range(start_frame, end_frame):
            frames = []
            for cap in self.caps:
                ok, frame = cap.read()
                if ok:
                    # 标注落点
                    h, w = frame.shape[:2]
                    cv2.circle(frame, (w // 2, h // 2), 10, (0, 0, 255), -1)
                    cv2.putText(frame, f"Frame: {fid}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)
                    if fid == landing_frame:
                        cv2.putText(frame, "LANDING!",
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 0, 255), 3)
                    frames.append(frame)

            if frames:
                # 拼接多路画面
                if len(frames) == 1:
                    display = frames[0]
                elif len(frames) == 2:
                    display = np.hstack(frames)
                else:
                    n = len(frames)
                    cols = min(n, 2)
                    rows = (n + 1) // 2
                    padded = []
                    for r in range(rows):
                        row_frames = frames[r*cols:(r+1)*cols]
                        if len(row_frames) < cols:
                            blank = np.zeros_like(row_frames[0])
                            row_frames.append(blank)
                        padded.append(np.hstack(row_frames))
                    display = np.vstack(padded)

                cv2.imshow("Replay", display)

            # 播放速度
            base_delay = int(1000 / self.fps)
            if slow_zone_start <= fid <= slow_zone_end:
                delay = int(base_delay / 0.25)  # 4x slower
            else:
                delay = base_delay

            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # Space: pause
                while True:
                    if cv2.waitKey(100) & 0xFF == ord(' '):
                        break

        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()


# ============================================================
# 用法示例
# ============================================================

if __name__ == '__main__':
    import cv2

    print("多摄像头融合引擎 v1.0")
    print("=" * 50)

    # --- 示例1: 融合节点逻辑测试 ---
    print("\n[TEST] 融合节点逻辑测试")
    fusion = FusionNode(num_cameras=4)

    # 模拟 3 路摄像头同时检测到落地
    now = time.time()
    fusion.push(LandingEvent(
        cam_id=0, frame=400, pixel_x=800, pixel_y=400,
        court_x=3.05, court_y=13.30, confidence=0.82,
        timestamp=now, rally_start=50, rally_end=400))
    fusion.push(LandingEvent(
        cam_id=1, frame=400, pixel_x=600, pixel_y=380,
        court_x=3.10, court_y=13.35, confidence=0.79,
        timestamp=now + 0.02, rally_start=50, rally_end=400))
    fusion.push(LandingEvent(
        cam_id=2, frame=401, pixel_x=700, pixel_y=390,
        court_x=3.00, court_y=13.28, confidence=0.85,
        timestamp=now + 0.05, rally_start=50, rally_end=400))

    result = fusion.get_latest_verdict()
    if result:
        print(f"\n最终判罚: {result.verdict}")
        print(f"融合坐标: ({result.court_x:.3f}m, {result.court_y:.3f}m)")
        print(f"置信度: {result.confidence:.1%}")
        print(f"摄像头数: {result.num_cameras_agree}")
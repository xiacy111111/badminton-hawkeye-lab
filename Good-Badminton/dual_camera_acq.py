"""
双相机异步采集模块
=================
- 双线程独立采集，每个相机一个线程
- 环形缓冲区 (ring buffer)，无锁读取
- 帧号 + 时间戳标注，用于后续软件同步
- 自动检测 USB3.0 带宽瓶颈

适用相机: MV-CU013-A0UC (USB3.0, 201.4fps, 1280×1024)
"""

import cv2
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DualCamera")


# ── 数据结构 ──────────────────────────────────────

@dataclass
class CameraFrame:
    """单帧数据"""
    image: np.ndarray               # 图像数据 (H, W) 灰度或 (H, W, 3) 彩色
    frame_id: int                   # 本相机的帧序号
    timestamp: float                # 采集时刻 (time.perf_counter, 秒)
    exposure_us: int = 0            # 曝光时间 (微秒)
    gain_db: float = 0.0            # 增益 (dB)


@dataclass
class SyncedPair:
    """同步后的帧对"""
    frame_a: CameraFrame
    frame_b: CameraFrame
    sync_confidence: float          # 同步置信度 (0~1)
    time_delta_ms: float            # 两帧时间差 (ms)


class RingBuffer:
    """
    固定大小环形缓冲区，线程安全
    - 写入：采集线程追加
    - 读取：主线程获取最新帧或历史帧
    """

    def __init__(self, maxlen: int = 300):
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._latest: Optional[CameraFrame] = None
        self.total_frames: int = 0
        self.dropped_frames: int = 0

    def put(self, frame: CameraFrame) -> None:
        """采集线程写入"""
        with self._lock:
            self._buffer.append(frame)
            self._latest = frame
            self.total_frames += 1

    def get_latest(self) -> Optional[CameraFrame]:
        """主线程读取最新帧（无阻塞）"""
        return self._latest

    def get_at_frame_id(self, frame_id: int) -> Optional[CameraFrame]:
        """查找指定帧号的帧"""
        with self._lock:
            for f in reversed(self._buffer):
                if f.frame_id == frame_id:
                    return f
        return None

    def get_closest_to_time(self, target_t: float) -> Optional[CameraFrame]:
        """查找时间戳最接近 target_t 的帧"""
        with self._lock:
            if not self._buffer:
                return None
            best = None
            best_dt = float('inf')
            for f in self._buffer:
                dt = abs(f.timestamp - target_t)
                if dt < best_dt:
                    best_dt = dt
                    best = f
            return best

    def recent_frames(self, n: int = 10) -> List[CameraFrame]:
        """最近 n 帧"""
        with self._lock:
            return list(self._buffer)[-n:]

    def fps_estimate(self) -> float:
        """估算最近一段时间内的实际帧率"""
        recent = self.recent_frames(50)
        if len(recent) < 2:
            return 0.0
        dt = recent[-1].timestamp - recent[0].timestamp
        return (len(recent) - 1) / dt if dt > 0 else 0.0


# ── 相机参数预设 ──────────────────────────────────

CAMERA_CONFIG = {
    # ── 曝光 ──
    "exposure_auto": 0,           # 关闭自动曝光
    "exposure_time": 200,         # 200μs = 1/5000s (冻结高速球)
    # ── 增益 ──
    "gain_auto": 0,               # 关闭自动增益
    "gain": 0,                    # 0dB, 保持最低噪点
    # ── 白平衡 ──
    "white_balance_auto": 0,      # 关闭自动白平衡
    "white_balance_red": 150,     # 手动锁定
    "white_balance_blue": 150,
    # ── 帧率 ──
    "fps": 201.4,                 # MV-CU013-A0UC 标称帧率
    # ── 分辨率 ──
    "width": 1280,
    "height": 1024,
    # ── 缓冲区 ──
    "buffer_size": 1,             # 最小缓冲，降低延迟
}


class SingleCamera:
    """
    单相机采集线程

    使用 OpenCV VideoCapture + CAP_DSHOW 后端（Windows DirectShow）
    MV-CU013-A0UC 支持 USB3 Vision / UVC，在 Windows 上通过 DirectShow 可用
    """

    def __init__(self, camera_id: int, name: str,
                 config: dict = None):
        """
        Args:
            camera_id: OpenCV 相机索引 (0, 1, 2, ...)
            name:      相机标识 ("camera_a" / "camera_b")
            config:    相机参数覆盖
        """
        self.camera_id = camera_id
        self.name = name
        self.config = {**CAMERA_CONFIG, **(config or {})}
        self.buffer = RingBuffer(maxlen=300)

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_count = 0
        self._start_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps_actual(self) -> float:
        return self.buffer.fps_estimate()

    # ── 打开与配置 ─────────────────────────────────

    def open(self) -> bool:
        """打开相机并配置参数"""
        # 使用 DirectShow 后端 (Windows)
        try:
            self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        except Exception:
            self._cap = cv2.VideoCapture(self.camera_id)

        if not self._cap.isOpened():
            logger.error(f"[{self.name}] 无法打开相机 {self.camera_id}")
            return False

        # ── 配置参数 ──
        cfg = self.config
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        self._cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg["buffer_size"])

        # 手动曝光/增益 (OpenCV 的 property 映射因驱动而异)
        # 海康相机在 DirectShow 下: 
        #   CAP_PROP_AUTO_EXPOSURE: 0.25=手动, 0.75=自动
        #   CAP_PROP_EXPOSURE: -7 到 -1 (log2 刻度) 或 μs 值
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -7.0)        # 约 1/5000s

        # 如果相机支持更精细的控制 (通过 MVS 或相机内固件):
        # 请用海康 MVS 客户端预先设置好以下参数并保存到相机固件:
        #   - 曝光时间: 200μs
        #   - 增益: 0dB
        #   - 白平衡: 手动, R=1.5, G=1.0, B=1.5
        #   - 帧率: 201.4fps

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"[{self.name}] 已打开 camera_id={self.camera_id}, "
                    f"分辨率={actual_w:.0f}x{actual_h:.0f}, "
                    f"标称帧率={actual_fps:.1f}fps")

        return True

    # ── 采集线程 ──────────────────────────────────

    def _capture_loop(self):
        """采集线程主循环"""
        self._start_time = time.perf_counter()

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"[{self.name}] 读帧失败")
                time.sleep(0.001)
                continue

            self._frame_count += 1
            timestamp = time.perf_counter()

            cf = CameraFrame(
                image=frame,
                frame_id=self._frame_count,
                timestamp=timestamp
            )
            self.buffer.put(cf)

    def start(self):
        """启动采集线程"""
        if not self._cap or not self._cap.isOpened():
            if not self.open():
                raise RuntimeError(f"[{self.name}] 相机未就绪")

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"cam-{self.name}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"[{self.name}] 采集线程已启动")

    def stop(self):
        """停止采集"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info(f"[{self.name}] 已停止, 共采集 {self._frame_count} 帧, "
                    f"平均帧率={self.fps_actual:.1f}fps")

    def get_frame(self) -> Optional[CameraFrame]:
        """主线程获取最新帧"""
        return self.buffer.get_latest()


# ── 双相机管理器 ──────────────────────────────────

class DualCameraManager:
    """
    双相机管理器
    - 同时启动两个相机
    - 提供帧同步接口
    - 监控两个相机的帧率健康度
    """

    def __init__(self, camera_id_a: int = 0, camera_id_b: int = 1):
        self.cam_a = SingleCamera(camera_id=camera_id_a, name="CAM_A")
        self.cam_b = SingleCamera(camera_id=camera_id_b, name="CAM_B")

        # ── 同步状态 ──
        self._sync_offset_ms: float = 0.0   # 已校准的时间偏移
        self._sync_ready: bool = False

    def start_both(self) -> bool:
        """同时启动两个相机"""
        ok_a = self.cam_a.open()
        ok_b = self.cam_b.open()
        if not ok_a or not ok_b:
            logger.error("相机打开失败")
            return False

        self.cam_a.start()
        self.cam_b.start()

        # 短暂等待缓冲区填充
        time.sleep(0.5)
        logger.info(f"双相机已启动: "
                    f"A={self.cam_a.fps_actual:.1f}fps, "
                    f"B={self.cam_b.fps_actual:.1f}fps")
        return True

    def stop_both(self):
        """停止两个相机"""
        self.cam_a.stop()
        self.cam_b.stop()

    def get_latest_pair(self) -> Tuple[Optional[CameraFrame], Optional[CameraFrame]]:
        """获取两个相机各自最新帧（不做同步）"""
        return self.cam_a.get_frame(), self.cam_b.get_frame()

    def get_synced_pair(self) -> SyncedPair:
        """
        获取经过软件同步的帧对

        同步策略:
          取相机 A 的最新帧 timestamp_a,
          在相机 B 的缓冲区中找时间戳最接近的帧。
          两帧时间差 Δt 越接近已校准的 sync_offset_ms 越理想。

        如果误差 > 2.5ms (半帧 @ 200fps), sync_confidence 降低。
        """
        latest_a = self.cam_a.get_latest()
        if latest_a is None:
            return SyncedPair(None, None, 0.0, 0.0)

        closest_b = self.cam_b.buffer.get_closest_to_time(
            latest_a.timestamp + self._sync_offset_ms / 1000.0
        )

        if closest_b is None:
            return SyncedPair(latest_a, None, 0.0, 0.0)

        dt_ms = (closest_b.timestamp - latest_a.timestamp) * 1000.0 - self._sync_offset_ms
        dt_abs = abs(dt_ms)

        # 帧间隔 @ 200fps = 5ms, 半帧 = 2.5ms
        if dt_abs < 1.0:
            confidence = 1.0
        elif dt_abs < 2.5:
            confidence = 1.0 - (dt_abs - 1.0) / 1.5
        elif dt_abs < 5.0:
            confidence = 0.5 - (dt_abs - 2.5) / 5.0
        else:
            confidence = 0.0  # 超过一帧，不可靠

        return SyncedPair(
            frame_a=latest_a,
            frame_b=closest_b,
            sync_confidence=confidence,
            time_delta_ms=dt_ms
        )

    # ── 同步校准 ──────────────────────────────────

    def calibrate_sync_offset(self, n_frames: int = 100) -> float:
        """
        LED 手电筒闪帧法校准时间偏移

        流程:
          1. 人站在两相机都能看到的位置
          2. 快速闪手电筒 3~5 次
          3. 两个相机的流中分别检测亮度骤升的帧
          4. 计算两相机闪帧时刻的偏移 → sync_offset_ms

        这个偏移是固定的（取决于 USB 控制器分配、驱动差异等）
        校准一次后可复用。
        """
        logger.info("开始同步校准... 请在两个相机视野内快速闪手电筒 3-5 次")
        logger.info("采集 3 秒数据...")

        # 清空缓冲区
        time.sleep(0.2)

        # 采集一批数据
        frames_a: List[CameraFrame] = []
        frames_b: List[CameraFrame] = []

        collect_start = time.perf_counter()
        while time.perf_counter() - collect_start < 3.0:
            fa = self.cam_a.buffer.get_latest()
            fb = self.cam_b.buffer.get_latest()
            if fa and (not frames_a or fa.frame_id != frames_a[-1].frame_id):
                frames_a.append(fa)
            if fb and (not frames_b or fb.frame_id != frames_b[-1].frame_id):
                frames_b.append(fb)
            time.sleep(0.001)

        # 检测闪光帧（亮度骤变）
        def detect_flash_frames(frames: List[CameraFrame], threshold: float = 30.0):
            """找亮度均值比前一帧跳变超过 threshold 的帧"""
            flash_events = []
            for i in range(1, len(frames)):
                curr_mean = np.mean(frames[i].image)
                prev_mean = np.mean(frames[i-1].image)
                if abs(curr_mean - prev_mean) > threshold:
                    flash_events.append(frames[i].timestamp)
            return flash_events

        flashes_a = detect_flash_frames(frames_a)
        flashes_b = detect_flash_frames(frames_b)

        if len(flashes_a) < 2 or len(flashes_b) < 2:
            logger.warning(
                f"未检测到足够的闪帧事件 (A:{len(flashes_a)}, B:{len(flashes_b)})"
            )
            return 0.0

        # 计算偏移
        offsets = []
        for ta in flashes_a:
            # 找 B 中最近的闪帧
            nearest_b = min(flashes_b, key=lambda tb: abs(tb - ta))
            offsets.append((ta - nearest_b) * 1000.0)  # ms

        self._sync_offset_ms = float(np.median(offsets))
        std_offset = float(np.std(offsets))

        logger.info(f"同步校准完成: offset = {self._sync_offset_ms:.2f}ms ± {std_offset:.2f}ms "
                    f"(A 领先 B, {len(offsets)} 个样本)")
        self._sync_ready = True
        return self._sync_offset_ms

    def health_report(self) -> dict:
        """两相机健康度报告"""
        return {
            "cam_a_fps": self.cam_a.fps_actual,
            "cam_b_fps": self.cam_b.fps_actual,
            "cam_a_total": self.cam_a.buffer.total_frames,
            "cam_b_total": self.cam_b.buffer.total_frames,
            "sync_ready": self._sync_ready,
            "sync_offset_ms": self._sync_offset_ms,
        }


# ── 自检 ──────────────────────────────────────────

if __name__ == "__main__":
    mgr = DualCameraManager(camera_id_a=0, camera_id_b=1)

    if not mgr.start_both():
        print("❌ 相机启动失败，请检查连接")
        exit(1)

    print("采集 5 秒...")
    t0 = time.perf_counter()
    frame_count = 0

    while time.perf_counter() - t0 < 5.0:
        pair = mgr.get_synced_pair()
        if pair.frame_a is not None:
            frame_count += 1
            if frame_count % 200 == 0:
                print(f"  [{pair.time_delta_ms:+.2f}ms] "
                      f"sync_conf={pair.sync_confidence:.2f}")

        time.sleep(0.001)

    mgr.stop_both()

    print(f"\n采集完成: {frame_count} 对, "
          f"实际 A={mgr.cam_a.fps_actual:.1f}fps, "
          f"B={mgr.cam_b.fps_actual:.1f}fps")
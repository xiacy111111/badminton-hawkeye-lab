"""
hikvision_capture.py — 海康 MV-CU013-A0UC 双相机同步采集
==========================================================
MVS 5.0.1 Python SDK

两种模式自动降级:
  MVS 软触发同步 → 时间戳精度 <100μs
  UVC OpenCV 回退 → 开箱即用，无硬件同步

安装 MVS 5.0.1 后 Python 路径:
  C:\Program Files\Hikvision\MVS\Development\Samples\Python
"""

import numpy as np
import time
import sys
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass

# ============================================================
# MVS 5.0.1 Python SDK 路径
# ============================================================
_MVS_CANDIDATES = [
    r"C:\Program Files\Hikvision\MVS\Development\Samples\Python",
    r"C:\Program Files (x86)\Hikvision\MVS\Development\Samples\Python",
    r"D:\Hikvision\MVS\Development\Samples\Python",
]

_MVS_READY = False
for _p in _MVS_CANDIDATES:
    if os.path.isdir(_p):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        _MVS_READY = True
        break

# ============================================================
# 参数
# ============================================================
WIDTH, HEIGHT = 1280, 1024
FPS_TARGET = 200
EXPOSURE_US = 4000       # 4ms
GAIN_DB = 4.0
GRAB_TIMEOUT_MS = 200
PIXEL_FMT_RGB8 = 0x02180014


@dataclass
class FramePacket:
    image: np.ndarray        # BGR (H, W, 3)
    timestamp_ns: int        # 硬件时间戳
    frame_id: int
    camera_index: int

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9


# ============================================================
# MVS 单相机
# ============================================================
class MvsCamera:

    def __init__(self):
        self._cam = None
        self._open = False
        self._grabbing = False
        self._fid = 0

    def open(self, serial: str = None, index: int = 0):
        if not _MVS_READY:
            raise RuntimeError("MVS SDK 未安装，请先安装海康 MVS 5.x")

        from MvCameraControl_class import (
            MvCamera, MV_USB_DEVICE, MV_CC_DEVICE_INFO_LIST
        )

        devlist = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, devlist)
        if ret != 0 or devlist.nDeviceNum == 0:
            raise RuntimeError("未检测到海康 USB 相机")

        n = devlist.nDeviceNum
        # 匹配序列号
        target = index
        for i in range(n):
            info = devlist.pDeviceInfo[i]
            sn_field = getattr(info, 'chSerialNumber', None)
            if sn_field and serial:
                sn = sn_field.decode() if isinstance(sn_field, bytes) else str(sn_field)
                if sn.strip('\x00') == serial:
                    target = i
                    print(f"  [MVS] 序列号匹配: {sn} → idx={i}")
                    break

        self._cam = MvCamera()
        self._cam.MV_CC_CreateHandle(devlist.pDeviceInfo[target])
        self._cam.MV_CC_OpenDevice()
        self._cam.MV_CC_SetIntValue("Width", WIDTH)
        self._cam.MV_CC_SetIntValue("Height", HEIGHT)
        self._cam.MV_CC_SetFloatValue("ExposureTime", EXPOSURE_US)
        self._cam.MV_CC_SetFloatValue("Gain", GAIN_DB)
        self._cam.MV_CC_SetEnumValue("PixelFormat", PIXEL_FMT_RGB8)
        self._open = True
        return self

    def set_trigger_mode(self, on: bool):
        if on:
            self._cam.MV_CC_SetEnumValue("TriggerMode", 1)
            self._cam.MV_CC_SetEnumValue("TriggerSource", 7)    # Software
        else:
            self._cam.MV_CC_SetEnumValue("TriggerMode", 0)

    def start(self):
        self._cam.MV_CC_StartGrabbing()
        self._grabbing = True

    def trigger(self):
        self._cam.MV_CC_SetCommandValue("TriggerSoftware")

    def grab(self, timeout_ms: int = GRAB_TIMEOUT_MS) -> Optional[FramePacket]:
        from MvCameraControl_class import MV_FRAME_OUT_INFO_EX

        info = MV_FRAME_OUT_INFO_EX()
        ret = self._cam.MV_CC_GetImageForRGB(info, timeout_ms)
        if ret != 0 or info.nFrameLen == 0:
            return None

        buf = self._cam.MV_CC_GetImageBuffer(info)
        img = np.frombuffer(buf, dtype=np.uint8,
                            count=info.nFrameLen).reshape(
                                info.nHeight, info.nWidth, -1).copy()
        # RGB → BGR
        if img.shape[2] == 3:
            img = img[:, :, ::-1]

        self._cam.MV_CC_FreeImageBuffer(info)

        ts = (info.nDevTimeStampHigh << 32) | info.nDevTimeStampLow
        pkt = FramePacket(image=img, timestamp_ns=ts,
                          frame_id=self._fid, camera_index=0)
        self._fid += 1
        return pkt

    def close(self):
        if self._grabbing:
            self._cam.MV_CC_StopGrabbing()
        if self._open:
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
        self._open = False
        self._grabbing = False


# ============================================================
# UVC 回退
# ============================================================
class UvcCamera:

    def __init__(self):
        self._cap = None
        self._idx = 0
        self._fid = 0

    def open(self, index: int):
        import cv2
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开 UVC 相机 #{index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
        self._idx = index
        return self

    def set_trigger_mode(self, on: bool):
        pass

    def start(self):
        pass

    def trigger(self):
        pass

    def grab(self, timeout_ms: int = 200) -> Optional[FramePacket]:
        import cv2
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        ts = time.perf_counter_ns()
        pkt = FramePacket(image=frame, timestamp_ns=ts,
                          frame_id=self._fid, camera_index=self._idx)
        self._fid += 1
        return pkt

    def close(self):
        if self._cap is not None:
            self._cap.release()


# ============================================================
# 双相机同步采集器
# ============================================================
class DualHikvisionCapture:

    def __init__(self):
        self._cam0 = None
        self._cam1 = None
        self._is_mvs = False

    def open(self,
             cam0_serial: str = None, cam1_serial: str = None,
             cam0_index: int = 0, cam1_index: int = 1):
        # 尝试 MVS
        try:
            self._cam0 = MvsCamera().open(serial=cam0_serial, index=cam0_index)
            self._cam1 = MvsCamera().open(serial=cam1_serial, index=cam1_index)
            self._cam0.set_trigger_mode(True)
            self._cam1.set_trigger_mode(True)
            self._is_mvs = True
            print("[Capture] MVS 软触发同步模式 ✓")
            return
        except Exception as e:
            print(f"[Capture] MVS 不可用 ({e})，回退 UVC")

        self._cam0 = UvcCamera().open(cam0_index)
        self._cam1 = UvcCamera().open(cam1_index)
        self._is_mvs = False
        print("[Capture] UVC OpenCV 模式")

    def start(self):
        self._cam0.start()
        self._cam1.start()
        if self._is_mvs:
            # 清空缓冲区
            for _ in range(5):
                self._cam0.trigger(); self._cam1.trigger()
                self._cam0.grab(50); self._cam1.grab(50)

    def grab_synced(self) -> Optional[Tuple[FramePacket, FramePacket]]:
        if self._is_mvs:
            self._cam0.trigger()
            self._cam1.trigger()
        f0 = self._cam0.grab(GRAB_TIMEOUT_MS)
        f1 = self._cam1.grab(GRAB_TIMEOUT_MS)
        if f0 is None or f1 is None:
            return None
        return (f0, f1)

    def close(self):
        if self._cam0: self._cam0.close()
        if self._cam1: self._cam1.close()
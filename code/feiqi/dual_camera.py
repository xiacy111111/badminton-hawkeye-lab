"""
双路 UVC 相机 — 有主机位（有线）+ 从机位（无线）
两路直连笔记本 USB，跟有线完全一样用
"""

import cv2, threading, time
import numpy as np


class DualCamera:
    def __init__(self, cam_master=0, cam_slave=1,
                 width=1280, height=720, fps=250):
        # 主机位：有源USB放大延长
        self.master = cv2.VideoCapture(cam_master)
        self.master.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.master.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.master.set(cv2.CAP_PROP_FPS, fps)
        self.master.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲

        # 从机位：无线USB延长
        self.slave = cv2.VideoCapture(cam_slave)
        self.slave.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.slave.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.slave.set(cv2.CAP_PROP_FPS, fps)
        self.slave.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 异步读
        self._frame_m = None
        self._frame_s = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._loop_s, daemon=True).start()

    def _loop(self):
        while self._running:
            _, f = self.master.read()
            if f is not None:
                with self._lock:
                    self._frame_m = f

    def _loop_s(self):
        while self._running:
            _, f = self.slave.read()
            if f is not None:
                with self._lock:
                    self._frame_s = f

    def read(self):
        """返回 (master_frame, slave_frame)"""
        with self._lock:
            return (self._frame_m.copy() if self._frame_m is not None else None,
                    self._frame_s.copy() if self._frame_s is not None else None)

    def stop(self):
        self._running = False
        self.master.release()
        self.slave.release()
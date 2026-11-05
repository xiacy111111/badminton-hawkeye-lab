"""
LED 闪光帧检测 + 帧对齐
=======================
赛前 LED 闪 3 次 → 检测两路视频中的亮帧位置 → 计算帧偏移 → 对齐
"""

import cv2, numpy as np


class LEDSync:
    def __init__(self, flash_count=3, brightness_thresh=200):
        self.flash_count = flash_count
        self.thresh = brightness_thresh
        self.offset = 0  # slave 相对于 master 的帧偏移

    def detect_flash(self, frame):
        """检测帧中是否有 LED 闪光（全局亮度突然升高）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) > self.thresh

    def calibrate(self, cam_master, cam_slave, max_frames=500):
        """
        采集 max_frames 帧，检测闪光位置，计算 offset。
        调用前需先触发 LED 频闪。
        """
        flash_pos_m = []
        flash_pos_s = []

        for i in range(max_frames):
            fm, fs = cam_master.read()[1], cam_slave.read()[1]
            if fm is None or fs is None:
                continue
            if self.detect_flash(fm):
                flash_pos_m.append(i)
            if self.detect_flash(fs):
                flash_pos_s.append(i)

        if len(flash_pos_m) >= self.flash_count and len(flash_pos_s) >= self.flash_count:
            offsets = [s - m for m, s in zip(flash_pos_m[:self.flash_count],
                                              flash_pshuttlecock_physics.pyos_s[:self.flash_count])]
            self.offset = int(np.median(offsets))
            print(f"[SYNC] 帧偏移: slave 比 master 晚 {self.offset} 帧")
            return self.offset
        else:
            print(f"[SYNC] 闪光检测不足 m:{len(flash_pos_m)} s:{len(flash_pos_s)}")
            return None

    def align(self, frame_master, frame_slave, buffer_s):
        """对齐：slave 晚 offset 帧，从 buffer 取对应帧"""
        if self.offset == 0:
            return frame_master, frame_slave
        # buffer_s 保留最近 N 帧，取 [offset] 位置
        if len(buffer_s) > abs(self.offset):
            return frame_master, buffer_s[-self.offset]
        return frame_master, frame_slave
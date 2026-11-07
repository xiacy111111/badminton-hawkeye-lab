"""
landing_judge.py — 落地检测器 + 场地判罚器（修复版）
=====================================================
原 SimpleLandingDetector 的问题：
  阈值 0.02m（2cm）过严。3D 重建噪声 σ≈19mm 时，重建 z 在贴地段仍会
  在 ±40mm 抖动，"连续 5 帧 z<2cm" 几乎不可能成立 → 落地事件恒为 0。

本模块提供：
  1. LandingDetector —— 三重触发条件（低速贴地 / 落地反弹 / 序列兜底投影）
  2. CourtJudge      —— 按羽毛球场标准线判定界内界外，输出压线裕度(mm)
"""

from collections import deque
from typing import Optional, Dict

import numpy as np

# 羽毛球场标准尺寸（米）
COURT_WIDTH = 6.10
COURT_LENGTH = 13.40
SINGLES_MARGIN = 0.46      # 单打边线与双打边线的距离
COURT_LINE_WIDTH = 0.04    # 场地线宽 40mm


class LandingDetector:
    """落地检测：高度 + 垂直速度 + 趋势反转，三重触发。"""

    def __init__(self, z_thresh: float = 0.15, v_thresh: float = 2.5,
                 confirm_frames: int = 3, z_history_size: int = 12):
        self.z_thresh = z_thresh          # 贴地高度阈值（米），需大于重建噪声
        self.v_thresh = v_thresh          # 贴地时的垂直速度门限（m/s）
        self.confirm_frames = confirm_frames
        self.z_hist = deque(maxlen=z_history_size)
        self.t_hist = deque(maxlen=z_history_size)
        self.detected = False
        self._prev_z: Optional[float] = None
        self._rising = False

    # ---------- 主入口 ----------
    def update(self, pt_3d, ts=None) -> Optional[Dict]:
        z = float(pt_3d[2])
        self.z_hist.append(z)
        self.t_hist.append(ts if ts is not None else len(self.z_hist))

        if self.detected:
            return None

        # 触发条件 B：由下降转为上升 → 已触地反弹
        if self._prev_z is not None and z > self._prev_z + 0.02 and self._prev_z < self.z_thresh * 2:
            return self._fire(pt_3d, ts, reason="bounce")
        self._prev_z = z

        # 触发条件 A：连续 N 帧贴地且垂直速度小
        if len(self.z_hist) >= self.confirm_frames:
            recent_z = list(self.z_hist)[-self.confirm_frames:]
            recent_t = list(self.t_hist)[-self.confirm_frames:]
            if all(v < self.z_thresh for v in recent_z):
                dt = max(float(recent_t[-1]) - float(recent_t[0]), 1e-3)
                vz = abs(recent_z[-1] - recent_z[0]) / dt
                if vz < self.v_thresh:
                    return self._fire(pt_3d, ts, reason="contact")

        return None

    # ---------- 兜底：序列结束仍无触发 ----------
    def finalize(self, last_pt_3d=None) -> Optional[Dict]:
        """轨迹结束时调用：把最后一点投影到地面（z=0）作为预测落点。"""
        if self.detected or last_pt_3d is None:
            return None
        pt = np.array(last_pt_3d, dtype=float).copy()
        pt[2] = 0.0
        return self._fire(pt, None, reason="projected")

    def _fire(self, pt, ts, reason: str) -> Dict:
        self.detected = True
        return {"position": np.asarray(pt, dtype=float),
                "timestamp": ts,
                "height": float(pt[2]),
                "reason": reason}

    def reset(self):
        self.z_hist.clear()
        self.t_hist.clear()
        self.detected = False
        self._prev_z = None
        self._rising = False


class CourtJudge:
    """界内/界外判罚（含压线裕度的毫米级输出）。"""

    def __init__(self, mode: str = "doubles", line_half_width: float = COURT_LINE_WIDTH / 2.0):
        self.mode = mode
        self.line_half = line_half_width

    def bounds(self):
        if self.mode == "singles":
            return SINGLES_MARGIN, COURT_LENGTH, COURT_WIDTH - SINGLES_MARGIN, 0.0
        return 0.0, COURT_LENGTH, COURT_WIDTH, 0.0

    def judge_landing(self, x: float, y: float) -> str:
        return self.judge(x, y)["result"]

    def judge(self, x: float, y: float) -> Dict:
        x_min, y_max, x_max, y_min = self.bounds()
        # 到四条边线的距离（负值 = 出界深度）
        d = {
            "left": x - (x_min - self.line_half),
            "right": (x_max + self.line_half) - x,
            "back": (y_max + self.line_half) - y,
            "front": y - (y_min - self.line_half),
        }
        margin = min(d.values())
        inside = margin >= 0
        inside_mm = margin * 1000.0
        if inside:
            result = "界内"
            if inside_mm < COURT_LINE_WIDTH * 1000 * 0.5:
                result = "压线(界内)"
        else:
            result = "界外"
        return {"result": result, "inside": inside,
                "margin_mm": round(inside_mm, 1),
                "x": round(float(x), 4), "y": round(float(y), 4),
                "mode": self.mode}


# 向后兼容旧名
SimpleLandingDetector = LandingDetector

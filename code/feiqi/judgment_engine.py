"""
judgment_engine.py v4.0 — 集成 PnP 相机定位 + 畸变矫正
"""
import json, sys, os
import cv2
import numpy as np
from enum import Enum
from court3d import Court3DModel, CourtPnPSolver


class CourtType(Enum):
    SINGLES = "singles"
    DOUBLES = "doubles"


class LineCallEngine:
    def __init__(self, court_type=CourtType.DOUBLES):
        self.W = 6.10 if court_type == CourtType.DOUBLES else 5.18
        self.L = 13.40

    def judge_landing(self, x, y, tolerance_cm=1.0):
        in_court = 0 <= x <= self.W and 0 <= y <= self.L
        dists = {
            "左边线": x*100, "右边线": (self.W-x)*100,
            "底边线": y*100, "顶边线": (self.L-y)*100,
        }
        named = list(dists.items())
        if in_court:
            name, val = min(named, key=lambda t: t[1])
            abs_cm = val
        else:
            if x < 0: name, abs_cm = "左边线", dists["左边线"]
            elif x > self.W: name, abs_cm = "右边线", dists["右边线"]
            elif y < 0: name, abs_cm = "底边线", dists["底边线"]
            else: name, abs_cm = "顶边线", dists["顶边线"]
            abs_cm = abs(abs_cm)

        if abs_cm <= tolerance_cm:
            call, detail = "ON_LINE", f"压线！距{name}仅{abs_cm:.1f}cm"
        elif in_court:
            call, detail = "IN", f"界内，距{name}{abs_cm:.1f}cm"
        else:
            call, detail = "OUT", f"出界！超出{name}{abs_cm:.1f}cm"

        return {"call": call, "distance_to_line_cm": abs_cm, "nearest_line": name,
                "position": {"x_cm": round(x*100,1), "y_cm": round(y*100,1)}, "detail": detail}


class MatchAnalyzer:
    def __init__(self, jsonl_path, annotations_path=None, first_frame_path=None):
        self.engine = LineCallEngine()
        self.solver = None
        self.mode = None  # "pnp" 或 "homography"

        # 模式1: PnP 3D定位（首选）
        if first_frame_path and os.path.exists(first_frame_path):
            self.model = Court3DModel(doubles=True)
            self.solver = CourtPnPSolver(self.model)
            frame = cv2.imread(first_frame_path)
            success, _ = self.solver.solve(frame)
            if success:
                self.mode = "pnp"
                print("模式: PnP 3D 相机定位（换机位自动适配）")
            else:
                print("PnP 失败，回退到四角标定模式")

        # 模式2: 四角单应性（回退）
        if self.mode != "pnp" and annotations_path and os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                for line in f:
                    if line.startswith('corners='):
                        s, e = line.find('['), line.rfind(']')+1
                        corners = eval(line[s:e])
                        break
            ip = np.array(corners, dtype=np.float32)
            cp = np.array([[0,13.4],[6.1,13.4],[6.1,0],[0,0]], dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(ip, cp)
            self.mode = "homography"
            print("模式: 四角单应性映射")

        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(l) for l in f if l.strip()]
        print(f"加载 {len(self.data)} 帧")

    def pixel_to_court(self, px, py):
        if self.mode == "pnp" and self.solver and self.solver.solved:
            result = self.solver.pixel_to_world(px, py)
            if result: return result[0], result[1]
        elif self.mode == "homography":
            pt = np.array([[[px, py]]], dtype=np.float32)
            t = cv2.perspectiveTransform(pt, self.H)
            return float(t[0,0,0]), float(t[0,0,1])
        return 0, 0

    def analyze(self, min_frames_between=30):
        judgments = []
        history = []
        prev_cy = None
        prev_cy2 = None
        last_landing_frame = -100

        for rec in self.data:
            sc = rec.get("shuttlecock")
            if not sc or sc.get("image") is None:
                if len(history) >= 5:
                    result = self._check_trajectory_end_landing(history)
                    if result and rec["frame"] - last_landing_frame >= min_frames_between:
                        result["frame"] = rec["frame"] - 1
                        judgments.append(result)
                        last_landing_frame = rec["frame"]
                history = []
                prev_cy = None
                prev_cy2 = None
                continue

            px, py = sc["image"]
            cx, cy = self.pixel_to_court(px, py)
            frame = rec["frame"]

            history.append((cx, cy, frame))
            if len(history) > 30:
                history.pop(0)

            if prev_cy2 is not None and prev_cy is not None and len(history) >= 6:
                dropping = prev_cy2 > prev_cy and prev_cy > cy
                fast_drop = (prev_cy - cy) > 0.3
                near_ground = cy < 1.5
                was_airborne = prev_cy2 > 1.0
                dx = abs(cx - history[-5][0]) if len(history) >= 5 else 0
                dy = abs(cy - history[-5][1]) if len(history) >= 5 else 0
                steep_descent = dy > dx * 0.5

                conditions_met = sum([dropping, fast_drop, near_ground, was_airborne, steep_descent])

                if conditions_met >= 4 and frame - last_landing_frame >= min_frames_between:
                    r = self.engine.judge_landing(cx, cy)
                    r["frame"] = frame
                    r["court_coords"] = {"x": round(cx, 3), "y": round(cy, 3)}
                    r["conditions"] = {
                        "dropping": dropping,
                        "fast_drop": fast_drop,
                        "near_ground": near_ground,
                        "was_airborne": was_airborne,
                        "steep_descent": steep_descent,
                        "met": conditions_met,
                    }
                    judgments.append(r)
                    last_landing_frame = frame
                    history = []

            prev_cy2 = prev_cy
            prev_cy = cy

        # 视频结束时也检查一次
        if len(history) >= 5:
            result = self._check_trajectory_end_landing(history)
            if result:
                result["frame"] = history[-1][2]
                judgments.append(result)

        return judgments

    def _check_trajectory_end_landing(self, history):
        if len(history) < 5:
            return None
        last3 = history[-3:]
        dy1 = last3[1][1] - last3[0][1]
        dy2 = last3[2][1] - last3[1][1]
        dropping = dy1 < 0 or dy2 < 0
        last_cx, last_cy, _ = history[-1]
        near_ground = last_cy < 1.5
        was_airborne = any(h[1] > 1.5 for h in history[:-3])
        in_court = 0 <= last_cx <= 6.1 and 0 <= last_cy <= 13.4
        if len(history) >= 6:
            total_dx = abs(history[-1][0] - history[-6][0])
            total_dy = abs(history[-6][1] - history[-1][1])
            steep = total_dy > total_dx * 0.5
        else:
            steep = True
        conditions = [dropping, near_ground, was_airborne, in_court, steep]
        if sum(conditions) >= 4:
            r = self.engine.judge_landing(last_cx, last_cy)
            r["conditions"] = {
                "type": "trajectory_end",
                "dropping": dropping, "near_ground": near_ground,
                "was_airborne": was_airborne, "in_court": in_court, "steep": steep,
            }
            return r
        return None

    def print_summary(self, judgments):
        if not judgments: print("\n未检测到落点"); return
        ic=sum(1 for j in judgments if j["call"]=="IN")
        oc=sum(1 for j in judgments if j["call"]=="OUT")
        lc=sum(1 for j in judgments if j["call"]=="ON_LINE")
        print(f"\n{'='*50}")
        print(f"  落点:{len(judgments)} | IN:{ic} OUT:{oc} LINE:{lc} | 模式:{self.mode}")
        print(f"{'='*50}\n")
        for j in judgments[:20]:
            m = {"IN":"[IN]","OUT":"[OUT]","ON_LINE":"[LINE]"}[j["call"]]
            c=j["court_coords"]
            print(f"  {m} Frame{j['frame']:4d} | ({c['x']:.2f}m,{c['y']:.2f}m) | {j['detail']}")

    def run(self):
        return self.print_summary(self.analyze())


if __name__ == "__main__":
    jp = "outputs/demo/detections.jsonl"
    ap = "outputs/demo/court_annotations.txt"
    fp = None  # PnP 用: 视频第一帧截图
    if len(sys.argv) > 1: jp = sys.argv[1]
    if len(sys.argv) > 2: ap = sys.argv[2]
    if len(sys.argv) > 3: fp = sys.argv[3]
    MatchAnalyzer(jp, ap, fp).run()
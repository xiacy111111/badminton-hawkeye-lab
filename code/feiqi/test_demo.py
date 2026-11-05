"""test_demo.py — 单视频快速测试"""
import sys, os, cv2, numpy as np, time
sys.path.insert(0, '.')
from ultralytics import YOLO
from calibration import CourtCalibrator
from ball_flight_tracker import TrajectoryPredictor, TrackState
from physics_landing_detector import SafeLandingPipeline, Detection, Verdict
from court_config import HL, HW

BALL_R = 0.033

cap = cv2.VideoCapture('videos/demo.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'视频: {fps:.1f}fps')

model = YOLO('weights/yolo11s-ball.pt')
model(np.zeros((320,320,3), dtype=np.uint8), imgsz=320, verbose=False)

ok, first = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
calib = CourtCalibrator()
ci = calib.calibrate(first, 'cam0')
print('标定:', 'OK' if ci else 'FAIL')

predictor = TrajectoryPredictor(fps=fps if fps>0 else 30, mode='3d', max_lost_frames=45)
detector = SafeLandingPipeline()
detector.min_confidence = 0.40

traj, preds, lands = [], [], []
last_land_t = 0.0
fid, t0 = 0, time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    fid += 1

    r = model(frame, conf=0.12, imgsz=640, verbose=False)
    boxes = r[0].boxes
    if boxes is None or len(boxes) == 0:
        predictor.update(None, fid)
        if fid % 300 == 0:
            print(f'  帧{fid} 无球')
        continue

    confs = boxes.conf.cpu().numpy()
    bi = int(np.argmax(confs))
    xyxy = boxes.xyxy[bi].cpu().numpy()
    cx = (xyxy[0] + xyxy[2]) / 2
    cy = (xyxy[1] + xyxy[3]) / 2
    conf = float(confs[bi])

    wx, wy = calib.pixel_to_world(cx, cy)
    dist = abs(wx)
    wz = max(0.05, 3.0 * max(0, 1 - dist / 7.0) ** 2 + 0.3)

    report = predictor.update((float(wx), float(wy), float(wz)), fid)
    traj.append((fid, float(wx), float(wy), float(wz), conf))

    if report.state == TrackState.REACQUIRED and report.gap_fills:
        for pt in report.gap_fills:
            preds.append((pt.frame, pt.x, pt.y, pt.z, pt.confidence))
        print(f'  帧{fid} 重捕获! 补{len(report.gap_fills)}帧')

    now = time.perf_counter()
    if now - last_land_t > 1.0:
        recent = [Detection(frame=f, x=float(px), y=float(py))
                  for f, px, py, _, _ in traj[-30:]]
        if len(recent) > 15:
            res = detector.analyze(recent)
            if res.verdict == Verdict.LANDING_FOUND:
                in_x = abs(wx) <= HL
                in_y = abs(wy) <= HW
                dy = abs(wy) - HW
                if in_x and in_y:
                    rule = 'IN'
                else:
                    rule = f'OUT (边线+{dy*100:.1f}cm)' if in_x else 'OUT'
                lands.append({
                    'frame': fid, 'x': float(wx), 'y': float(wy),
                    'ruling': rule, 'conf': res.confidence
                })
                last_land_t = now
                print(f'  [帧{fid:5d}] {rule} '
                      f'({wx:+.3f},{wy:+.3f})m c={res.confidence:.1%}')

    if fid % 300 == 0:
        t = time.perf_counter() - t0
        print(f'  帧{fid:5d} FPS:{fid/t:.0f} 轨迹:{len(traj)} '
              f'落地:{len(lands)}', flush=True)

cap.release()
t_total = time.perf_counter() - t0
print(f'\n===== 完成 =====')
print(f'{t_total:.1f}s | {fid}帧 | FPS:{fid/t_total:.0f}')
print(f'轨迹:{len(traj)} 预测:{len(preds)} 落地事件:{len(lands)}')
for i, ev in enumerate(lands):
    print(f'  #{i+1} 帧{ev["frame"]:5d} '
          f'({ev["x"]:+.3f},{ev["y"]:+.3f})m '
          f'c={ev["conf"]:.1%} -> {ev["ruling"]}')
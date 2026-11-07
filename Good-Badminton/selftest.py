"""
selftest.py — 一键自检（不需要相机、不需要真实素材）
=====================================================
    python selftest.py

依次验证：
  1. 依赖是否齐全（numpy / opencv / PIL）
  2. 物理引擎（RK4）终端速度合理性
  3. 落地检测器（三重触发 + 兜底）
  4. 场地判罚器（界内/压线/界外/单打线）
  5. 全链路模拟（run_full_demo --mode simulate）误差与判罚产出
  6. 单视频流水线（video_pipeline 合成视频端到端）

全部通过打印 PASS，否则 FAIL 并给出原因。
"""

import os
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail or ""))
        print(f"  ✔ {name}" + (f" — {detail}" if detail else ""))
    except Exception as e:
        results.append((False, name, str(e)))
        print(f"  ✘ {name} — {e}")


def t_deps():
    import numpy, cv2
    try:
        import PIL  # noqa: F401
        pil = PIL.__version__
    except Exception:
        pil = "缺失"
    return f"numpy {numpy.__version__}, opencv {cv2.__version__}, pillow {pil}"


def t_physics():
    from shuttlecock_physics import ShuttlecockPhysics, PhysicsParams
    ph = ShuttlecockPhysics(PhysicsParams())
    vt = ph.p.v_terminal
    assert 4.0 < vt < 8.0, f"终端速度异常: {vt}"
    return f"终端速度 {vt:.2f} m/s"


def t_landing():
    from landing_judge import LandingDetector
    import numpy as np
    ld = LandingDetector()
    fired = None
    for i, z in enumerate([1.2, 0.8, 0.4, 0.18, 0.09, 0.05, 0.03, 0.02]):
        r = ld.update(np.array([3.0, 6.0, z]), ts=i * 0.02)
        if r:
            fired = r
            break
    assert fired is not None, "贴地序列未触发落地"
    ld2 = LandingDetector()
    for i, z in enumerate([1.0, 0.7, 0.5]):
        ld2.update(np.array([1.0, 2.0, z]), ts=i * 0.02)
    assert ld2.finalize(np.array([1.0, 2.0, 0.4])) is not None, "兜底投影失效"
    return f"触发方式={fired['reason']}，兜底可用"


def t_judge():
    from landing_judge import CourtJudge
    d = CourtJudge("doubles")
    assert d.judge_landing(3.0, 6.0) == "界内"
    assert d.judge(3.0, 6.0)["inside"] is True
    assert d.judge(6.4, 6.0)["inside"] is False          # 双打边线外
    near = d.judge(6.085, 6.0)
    assert near["result"].startswith("压线"), f"压线未识别: {near}"
    s = CourtJudge("singles")
    assert s.judge(0.2, 6.0)["inside"] is False          # 单打边线外
    return "双打/单打/压线 判定正确"


def t_simulate():
    p = subprocess.run([sys.executable, "run_full_demo.py", "--mode", "simulate"],
                       cwd=BASE, capture_output=True, text=True,
                       encoding="utf-8", errors="ignore", timeout=300)
    out = p.stdout
    assert "落地事件: 4" in out, "落地事件数不为 4"
    assert "基线=13.40m" in out.replace(" ", ""), "未使用 13.4m 大基线"
    mm = [ln for ln in out.splitlines() if "3D 误差:" in ln]
    assert mm, "未输出 3D 误差汇总"
    return mm[-1].strip()


def t_video_pipeline():
    from video_pipeline import analyze_video, make_demo_video
    tmpdir = tempfile.mkdtemp(prefix="hawkeye_selftest_")
    vp = os.path.join(tmpdir, "demo.mp4")
    make_demo_video(vp, n_frames=80)
    res = analyze_video(vp, out_dir=os.path.join(tmpdir, "out"), max_frames=80,
                        log=lambda m: None, make_video=False)
    st = res["stages"]
    assert st.get("court", {}).get("ok"), "球场检测失败"
    assert st.get("track", {}).get("ok"), "轨迹未形成"
    land = st.get("landing", {})
    assert land.get("ok"), f"落点判罚失败: {land}"
    assert os.path.exists(res.get("preview", "")), "预览图未生成"
    return (f"轨迹 {st['track']['main_len']} 点, R²={st['track'].get('parabola_r2')}, "
            f"落点 {land['court_xy']} → {land['judge']['result']}")


def main():
    print("=" * 62)
    print("  羽毛球鹰眼项目自检")
    print("=" * 62)
    print("[1] 依赖")
    check("依赖导入", t_deps)
    print("[2] 物理引擎")
    check("RK4 终端速度", t_physics)
    print("[3] 落地检测")
    check("三重触发 + 兜底", t_landing)
    print("[4] 场地判罚")
    check("界内/压线/界外", t_judge)
    print("[5] 全链路模拟")
    check("run_full_demo simulate", t_simulate)
    print("[6] 单视频流水线")
    check("video_pipeline demo", t_video_pipeline)

    ok = sum(1 for r in results if r[0])
    total = len(results)
    print("=" * 62)
    print(f"  结果: {ok}/{total} 通过")
    if ok != total:
        print("  失败项：")
        for good, name, detail in results:
            if not good:
                print(f"    · {name}: {detail}")
        print("  ==> FAIL")
        return 1
    print("  ==> PASS（全部通过）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

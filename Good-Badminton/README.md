# Good-Badminton: AI 羽毛球鹰眼系统 🏸

<div align="center">

[![GitHub license](https://img.shields.io/github/license/yo-WASSUP/Good-Badminton)](https://github.com/yo-WASSUP/Good-Badminton/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-开发中-yellow)]()

**从视频分析到双目实时判罚——完整的羽毛球鹰眼解决方案**

[中文](README.md)

</div>

---

## 📖 项目简介

本项目起源于 [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)（一个基于计算机视觉的单视频羽毛球比赛分析工具），并在此基础上扩展为**双目立体视觉实时鹰眼判罚系统**。

### 两个子系统

| 子系统 | 输入 | 输出 | 适用场景 |
|---|---|---|---|
| **单视频分析**（Good-Badminton 原版）| 比赛录像 MP4 | 球员轨迹、热力图、统计 | 赛后复盘、技术分析 |
| **双目鹰眼判罚**（本系统新增）| 双路工业相机实时画面 | 3D 球轨迹、落地判罚 | 实时判罚、辅助裁判 |

### 核心思路

单视频分析沿用了 Good-Badminton 的完整流程——姿态检测、球检测、球场映射、轨迹统计——没有改动。

双目鹰眼判罚是全新的：**两个相机架在球场两侧，各自独立标定（球场本身就是标定板），实时检测羽毛球在左右画面中的位置，三角测量重建出球的真实 3D 坐标，然后用 RK4 空气阻力物理引擎预测轨迹，最终判断落点是否出界。**

---

## 🎬 效果预览

*（相机到货后补充真实截图）*

```
┌─────────────────────────────────────────────────┐
│  ╔═══════════════════════════════════════════╗  │
│  ║              球网                         ║  │
│  ║                                          ║  │
│  ║     ·  ·  ·                              ║  │
│  ║        ·    ·    ← 3D 球轨迹             ║  │
│  ║           ·                              ║  │
│  ║              ·   ● ← 落点（界内/界外）   ║  │
│  ║                                          ║  │
│  ╚═══════════════════════════════════════════╝  │
│                  Hawk-Eye 风格 3D 视图           │
└─────────────────────────────────────────────────┘
```

---

## 🆕 更新日志

- **2026-07**：完成双目鹰眼判罚系统核心代码
  - 移植 Good-Badminton 自动球场检测算法 → 扩展为 16 关键点 + solvePnP RANSAC 双目独立标定
  - 实现 RK4 空气阻力物理引擎（线性 + 二次混合阻力，终端速度 5.92m/s）
  - 实现 3D 飞行追踪（卡尔曼滤波 + 出画预测 + 入画重捕）
  - 实现 Hawk-Eye 风格 3D 可视化（旋转/缩放/落点放大动画）
  - 实现全链路动态噪声模拟测试（不需要相机也能跑通）
  - 海康 MV-CU013-A0UC 工业相机驱动（MVS SDK / UVC 双模式）
- **2026-06-27**：Good-Badminton 上游——优化自动球场线检测,减少误匹配。
- **2026-06-20**：Good-Badminton 上游——正式开源。

---

## ✨ 功能

### 单视频分析（Good-Badminton）
- **球员姿态检测** — RTMPose / RTMO / YOLO Pose，识别人体关键点
- **羽毛球检测** — YOLO 实时检测球的位置和轨迹
- **球场坐标映射** — 自动 / 手动标注球场四角，建立图像到标准球场的映射
- **球员追踪与统计** — 移动距离、速度、回合数
- **可视化输出** — 骨架叠加、轨迹图、热力图、散点图
- **WebUI** — Gradio 浏览器界面，无需命令行

### 双目鹰眼判罚（新增）
- **自动双目标定** — 两个相机各拍一张球场照片，自动检测场地线 → 16 关键点 → solvePnP RANSAC → 输出投影矩阵
- **实时 3D 三角测量** — 左右画面各检测球的位置 → 重建真实 3D 坐标
- **RK4 物理引擎** — 线性 + 二次混合空气阻力，精确预测球飞行轨迹
- **落地判罚** — 高度阈值 + 多帧确认 + 界内/界外判定
- **3D 可视化** — Hawk-Eye 风格，鼠标旋转/缩放，落点自动放大动画

---

## 📋 系统要求

### 单视频分析
- Python 3.8+
- FFmpeg（加入系统 PATH）
- GPU 建议 6GB+ 显存（CPU 也能跑，但更慢）
- 16GB+ 内存

### 双目鹰眼判罚（额外需要）
- Windows 10/11
- 海康 MVS SDK 5.0.1（相机驱动）
- 2 个 USB3.0 口（或 1 个 USB3.0 + 1 个 Thunderbolt/USB-C）
- 海康 MV-CU013-A0UC 工业相机 ×2
- 6mm C 口镜头 ×2、三脚架 ×2、USB3.0 有源延长线（15m）×2

---

## 🚀 安装

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

GPU 加速（可选）：
```bash
pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu==1.20.1
```

---

## 📝 使用指南

### 一、单视频分析（立即可用）

```bash
python main.py --video-path videos/demo.mp4
```

程序会自动检测球场或弹出窗口让你手动标注 4 个角点。分析结果在 `outputs/` 目录。

### 二、全链路模拟测试（立即可用，不需要相机）

在无相机的情况下，用物理引擎生成虚拟轨迹，模拟双目投影和噪声，验证整个 pipeline：

```bash
python test_integration.py
python run_full_demo.py --mode simulate
```

这会生成扣杀、高远球、吊球、平抽四条真实空气阻力轨迹，用虚拟双目相机（焦距 1740px，基线 13.4m）投影，加入动态噪声（边缘畸变/距离衰减/运动模糊/网带遮挡），三角测量重建，输出 3D 误差统计。

### 三、双目鹰眼判罚（需要相机到货）

**第 1 步：标定（相机到货后只做一次）**

架好相机，每个相机拍一张完整球场的照片：

```bash
python calibration.py --left left.jpg --right right.jpg --out calib.json
```

这会自动检测球场四角 → 生成 16 个场地关键点 → 亚像素精化 → solvePnP RANSAC → 双眼独立标定 → 输出投影矩阵 P1/P2。

**第 2 步：实时判罚**

```bash
python run_full_demo.py --mode stereo --calib calib.json
```

左右画面实时检测球 → 三角测量 → 3D 追踪 → 落地判罚 → 3D 可视化。

---

## 🔧 硬件部署

### 相机安装

两个相机分别架在球场**两端底线后方**，各拍摄完整球场。

```
              13.4m
  ┌─────────────────────────────┐
  │                             │
  │         羽毛球场地           │
  │                             │
  │              ┃              │  ← 球网
  │                             │
  └─────────────────────────────┘
     ▲                    ▲
     │                    │
  相机1                相机2
  (底线后)             (底线后)
     │                    │
     └──── 笔记本(中段) ────┘
          USB3.0 15m 线 ×2
```

### 接线

- 相机 1 → USB3.0 15m 延长线 → 笔记本 USB-A 口
- 相机 2 → USB3.0 15m 延长线 → Type-C 转接头 → 笔记本另一侧 Thunderbolt 口

**两个相机不能插在同一侧的 USB 口上**（供电和带宽不足）。

### 硬件清单

| 设备 | 型号/规格 | 数量 | 单价 | 状态 |
|---|---|---|---|---|
| 工业相机 | 海康 MV-CU013-A0UC | ×2 | ¥400 | ✅ 已购 |
| 镜头 | 海康 6mm C 口 | ×2 | ¥200 | ✅ 已购 |
| USB3.0 延长线 | 毕亚兹 15m 带信号放大器 | ×2 | ¥139 | ⚠️ 还差 1 根 |
| 三脚架 + 云台 | 任意品牌 | ×2 | — | ⚠️ 待购 |
| Type-C 转 USB-A | 任意品牌（USB 3.0）| ×1 | ¥10 | ⚠️ 待购 |
| 电脑 | Dell Precision 3590 | ×1 | — | ✅ 已有 |

---

## 🧩 项目结构

```
Good-Badminton/
│
├── 📄 单视频分析（Good-Badminton 原版）────
├── main.py                     # CLI 入口（单视频分析）
├── badminton_analysis/
│   ├── system.py               # 视频分析主流程
│   ├── court/                  # 球场标注与坐标映射
│   │   ├── mapper.py           # 坐标转换
│   │   ├── detector.py         # ★ 自动球场线检测
│   │   └── reference.py        # 标准球场线模型
│   ├── detection/              # 球检测 + 姿态检测
│   ├── tracking/               # 球员追踪
│   ├── visualization/          # 可视化叠加层
│   └── media/                  # 视频音频处理
├── webui/                      # Gradio 浏览器界面
│
├── 📄 双目鹰眼判罚（新增）────
├── calibration.py              # ★ 多视角自动双目标定
│                               #   4 角点检测 → 16 关键点精化
│                               #   → solvePnP RANSAC → P1/P2 输出
├── shuttlecock_physics.py      # ★ RK4 空气阻力物理引擎
├── ball_flight_tracker.py      # ★ 3D 飞行追踪 + 卡尔曼滤波
├── ball_detector.py            # YOLO 球检测
├── physics_landing_detector.py # 落地检测（6 层防线）
├── judgment_engine.py          # 判罚引擎
├── court_3d.py                 # ★ Hawk-Eye 风格 3D 可视化
├── court_config.py             # 场地尺寸常量
├── hikvision_capture.py        # 海康相机驱动
├── stereo_pipeline.py          # 手动双目标定（备用）
├── run_full_demo.py            # 双目系统主入口
├── test_integration.py         # 全链路模拟测试
│
├── 📄 配置文件 ────────────
├── requirements.txt
├── requirements-webui.txt
├── .gitignore
├── weights/
│   └── yolo11s-ball.pt         # YOLO 球检测模型权重
└── README.md
```

---

## ⚙️ 技术细节

### 标定算法

1. **HSV 绿色分割** → 提取场地 ROI
2. **白线 + Canny 边缘** → 二值线 mask
3. **HoughLinesP** → 水平/垂直线段 → 去重
4. **暴力搜索**：2 水平 × 2 垂直 → 交点四边形
5. **14 维评分**：面积、高度、透视比、中心位置、边缘距离、线段长度、端点对齐……
6. **最优四边形** → 4 个角点
7. **单应性投影** → 16 个场地关键点（所有主要线交点）
8. **cornerSubPix** → 亚像素精化
9. **solvePnP RANSAC**（200 次迭代）→ 相机位姿
10. **双眼独立标定** → 相对位姿 → P1/P2

### 物理引擎

| 参数 | 值 | 含义 |
|---|---|---|
| 球质量 | 0.005 kg | 标准羽毛球 |
| 线性阻力系数 b | 0.0014 kg/m | 低速段主导 |
| 阻力系数 Cd | 0.6 | 高速段主导 |
| 截面积 A | 0.00342 m² | |
| 终端速度 | 5.92 m/s | 实测吻合 |
| 积分步长 | 0.0005 s | RK4 四阶 |

阻力模型：**F = b·v + ½ρCdA|v|v**（线性 + 二次混合）

### 3D 精度

理论精度取决于：
- 基线 b：13.4m（球场长度）
- 焦距 f：1740px（6mm 镜头 + 3.45µm 像元）
- 检测噪声 σ：0.35~2px（随距离/速度/边缘位置动态变化）

深度误差数量级：**ΔZ ≈ Z²·σ / (f·b)** ≈ 几十毫米（近端）到 ~200mm（远端）。

---

## 📊 球场参数

| 参数 | 尺寸 | 说明 |
|---|---|---|
| 全场宽 | 6.10 m | 双打边线外沿 |
| 全场长 | 13.40 m | 底线外沿 |
| 单打边线内缩 | 0.46 m | |
| 网高 | 1.55 m | 中央高度 |
| 发球线距网 | 1.98 m | |
| 双打后发球线距底线 | 0.76 m | |

---

## 📄 开发进度

### 已完成

- [x] Good-Badminton 单视频分析完整流程
- [x] Good-Badminton 自动球场线检测算法移植
- [x] 16 关键点 + 亚像素精化 + solvePnP RANSAC 标定
- [x] 双眼独立标定（球场即标定板）
- [x] RK4 空气阻力物理引擎（自检通过）
- [x] 全链路动态噪声模拟测试（通过）
- [x] 3D 飞行追踪（卡尔曼滤波 + 出画预测）
- [x] 落地判罚引擎
- [x] Hawk-Eye 风格 3D 可视化
- [x] 海康 MV-CU013-A0UC 驱动
- [x] 工业相机采购

### 待完成

- [ ] 第二根 15m USB3.0 延长线采购
- [ ] 三脚架 + 云台 ×2 采购
- [ ] Type-C 转接头采购
- [ ] 相机到货 → MVS SDK 验证出图
- [ ] 真实球场双目标定
- [ ] 真实场景全链路跑通
- [ ] 3D 可视化坐标系对齐调试
- [ ] 废弃文件清理（移入 feiqi/）

---

## 🙏 致谢

- [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton) — 本项目上游，提供了完整的单视频分析流程和自动球场检测算法
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 羽毛球检测模型
- [Tau-J/rtmlib](https://github.com/Tau-J/rtmlib) — 轻量姿态估计运行库
- [OpenMMLab / RTMPose](https://github.com/open-mmlab/mmpose) — 姿态估计算法基础
- [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) — 羽毛球数据集

## 📄 许可证

本项目代码使用 **Apache License 2.0**。

`weights/yolo11s-ball.pt` 使用 Apache License 2.0。

RTMPose / RTMO / YOLOX ONNX 权重来自 OpenMMLab / RTMPose 生态，按其上游 Apache License 2.0 授权使用，保留原始归属。

Good-Badminton 上游项目版权归 [yo-WASSUP](https://github.com/yo-WASSUP) 所有。

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yo-WASSUP/Good-Badminton&type=Date)](https://www.star-history.com/#yo-WASSUP/Good-Badminton&Date)

---

*最后更新：2026 年 7 月*

---

## 🆕 2026-11 修复与新功能

### 一、关键 Bug 修复

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| 1 | 模拟模式默认基线 **0.5m**（小基线） | 3D 重建误差 234mm，几何上本该失败 | 改为全场 **13.4m 大基线**，误差降至 **18.9mm**（P95 42mm），与误差传播理论一致 |
| 2 | 落地检测阈值 2cm 过严（重建噪声就有 ±19mm） | 落地事件恒为 0，判罚永不触发 | 新增 `landing_judge.py`：三重触发（低速贴地 / 落地反弹 / 序列兜底投影），4/4 球全部判罚成功 |
| 3 | `test_integration.py` 中 `JudgmentEngine` 未定义 | 判罚引擎静默失效 | 改用 `LineCallEngine`，并补 `CourtJudge` 兜底 |
| 4 | 判罚只有粗暴的矩形边界判断 | 无单打/双打线区分，无压线裕度 | `CourtJudge`：支持单打/双打边线 + 输出压线裕度（mm）+ 压线提示 |

### 二、新增：单视频自动识别（无需模型权重）

`video_pipeline.py` —— 纯 OpenCV 流水线：

```
视频 → 多帧采样自动球场检测 → 单应标定（米制）
     → 帧差运动检测（专治 5-15px 小目标，排除球员等大目标）
     → 最近邻轨迹串联 → 主轨迹筛选
     → 落点估算（地平面假设）→ 界内/界外判罚（含裕度 mm）
     → preview.png + annotated.mp4 + result.json
```

```bash
python video_pipeline.py 你的视频.mp4          # 分析真实视频
python video_pipeline.py --demo                # 无素材时用合成球场自测
```

实测（合成绿色球场 + 飞行球，90 帧）：球场检测成功 → 主轨迹 89 点 → 落点 (5.26, 11.44)m → 界内，用时 2.3s。

### 三、新增：桌面拖拽窗口

`app_gui.py` —— 把视频拖进窗口即自动识别：

```bash
启动.bat            # 双击启动（自动用项目 .venv）
# 或
python app_gui.py
```

- 拖拽区支持 mp4/mov/avi/mkv，也可点击选择文件
- 一键「模拟全链路演示」：无需相机与视频，直接跑物理引擎 + 双目重建 + 判罚
- 日志逐阶段输出、右侧实时预览、一键打开输出目录

依赖：`numpy opencv-python pillow windnd`

### 四、检测鲁棒性升级（2026-11-07）

| 改进 | 说明 |
|---|---|
| 双路检测 | MOG2 背景建模 ∪ 帧差，抗光照渐变与静态场景 |
| 自适应阈值 | 按图像噪声中位数动态设门限（暗光/高噪场景不失效） |
| 尺度自适应 | 尺寸门限按画面比例计算，自动排除球员/球拍 |
| 抛物线拟合筛选 | 轨迹按 y=ax²+bx+c 拟合度 R² 打分选主轨迹，剔除跑动/挥拍干扰 |
| 落点双估计 | 主判罚用末观测点，二次外推值作参考量输出（JSON 内 both） |
| 压线判定 | 距线外沿不足 40mm（一个线宽）判「压线(界内)」 |
| 处理进度 | 每 25% 输出一次进度，GUI 状态栏同步 |

### 五、一键自检

```bash
python selftest.py      # 6 项全通过 → PASS
```


### 六、真实素材实测（2026-11-07）

在 5 段真实视频上验证（录于真实球馆，HEVC/H.264，720p~1080p）：

| 素材 | 场景 | 球场自动标定 | 球检测 | 结论 |
|---|---|---|---|---|
| `demo1.mp4`（BWF 转播，单机位） | 标准转播视角 | ✅ 精准（红框与白线吻合） | YOLO 正常检出 | 链路可用，落点判罚需轨迹跟到落地 |
| `2.mp4` / `test_1min.mp4`（1080p 球馆） | 单场地广角 | ⚠️ 部分成功，鱼眼广角易选错线 | YOLO 正常（0.11~0.49 置信度） | 建议手动四点标定 |
| `1.mp4`（手持/多场地同框） | 6 名球员 + 观众 | ❌ 自动失败 | 帧差法误检爆炸（6 万点） | 自动标定不适用，须手动四点 |

关键发现与对应修复：

1. **HEVC 编码打不开** → 加 ffmpeg 自动转码兜底（`ensure_decodable`）
2. **帧差法在多球员场景误检爆炸**（300 帧 63008 点）→ 加镜头抖动补偿（相位相关）、白点亮度过滤、候选上限、尺寸按比例自适应 → 降到 6131 点
3. **专用球检测模型远优于帧差法**：`yolo11s-ball.pt` 在 1080p 转播上 240 帧仅 214 个候选（≈0.9/帧），轨迹 R²=1.00
4. **单应变换只对地平面成立**：球还在空中时把末点映射回球场会得到 -40m 之类的荒谬值 → 现在会明确报 `still_in_flight` 而不是硬给判罚
5. **标定帧必须在分析范围内采样**：原来跨全片采样，遇到镜头切换会用到别的画面的角点 → 已修

新增能力：

```bash
# 手动四点标定（自动失败时用，格式：左上;右上;右下;左下）
python video_pipeline.py 视频.mp4 --corners "780,440;1130,440;1580,890;430,780"

# 指定检测器
python video_pipeline.py 视频.mp4 --detector yolo      # 需要 weights/yolo11s-ball.pt
python video_pipeline.py 视频.mp4 --detector motion    # 无权重时的纯 OpenCV 方案

# GUI 里同样支持：检测器下拉框 + 「手动标定球场」按钮（点四角即可）
```

**权重说明**：球检测权重 `weights/yolo11s-ball.pt` 因体积原因不入库（见 .gitignore），
需自行放入 `weights/` 目录；缺失时 `--detector auto` 会自动降级为运动检测。

**已知限制（如实记录）**：
- 单目视频的落点是"落地平面单应"估算，不是双目三角测量，精度不可与双目方案比较；
- 球在画面里未跟到落地即无法给判罚（会明确标注 `still_in_flight`）；
- 多场地同框 + 手持抖动场景，自动标定不可靠，需手动四点；
- CPU 上 YOLO 约 2 秒/帧，长视频建议分段或用 GPU。


### 七、微调：用西电羽球队数据做领域自适应

预训练权重来自转播素材，直接用在自家球馆会明显掉点。`finetune/` 提供了完整闭环：

```
原始视频 → 抽帧(去重/去模糊) → YOLO 预标注 → 人工修正 → 划分 train/val
        → 微调(yolo11s-ball 起点) → 评估(与基线对比) → badcase 回灌迭代
```

```bash
# 1) 构建数据集（自动抽帧 + 预标注）
python finetune/prepare_dataset.py --videos "D:/xtu_raw" --out finetune/dataset \
       --frames-per-video 150 --conf 0.20

# 2) 人工修正 labels/（LabelImg / Label Studio；保留负样本以压制误检）

# 3) 微调（小数据集建议 --freeze 10）
python finetune/train.py --data finetune/dataset/data.yaml \
       --epochs 100 --imgsz 1280 --batch 8 --device auto --freeze 10 --lr0 0.001

# 4) 评估 + 与基线对比（这张表就是材料里的核心数据）
python finetune/evaluate.py --compare weights/yolo11s-ball.pt weights/yolo11s-ball-xtu.pt \
       --data finetune/dataset/data.yaml
```

采集规范、标注规范、工时估算、验收指标与迭代方法见 **[finetune/README.md](finetune/README.md)**。

> 预标注能省掉"从零打框"，人工只做修正，实测可省约 70~80% 标注工时——
> 这是小团队能把几千帧数据做下来的关键。

微调所需的额外依赖：`torch`（CPU 版即可跑通流程）、`ultralytics`、`matplotlib`、`pandas`、`polars`。


### 八、单机位快速上手（推荐工作流）

固定一个机位后，**标定只做一次，之后所有视频自动复用**：

```bash
# 第一次：自动检测（失败就加 --corners 手动点四角）
python video_pipeline.py 第一段视频.mp4
#   → 四角存进 calib_single_cam.json

# 之后：同机位的所有视频直接跑，自动读标定
python video_pipeline.py 第二段视频.mp4
```

窗口里同样如此：标定成功会提示「同机位下次自动复用」；换场地/换机位时点
「清除标定」，再重新自动检测或点四角。

```
[球场] 复用已保存的单机位标定（…/calib_single_cam.json）
[球场] 复用标定四点（固定机位无需重复标定）
[检测] 使用 YOLO 球检测模型（imgsz=1280, conf=0.15）
```

**单机位机位摆放三条硬性要求**（详见 [finetune/README.md](finetune/README.md) 第二节）：
双打场地四角全在画面内、球网完整可见、球在画面里不能只有 2~3 像素。


### 九、联网实测（2026-11-10）

用 `yt-dlp` 从 B 站下载真实比赛片段（BWF 迪拜总决赛、国内球馆 1080p）跑端到端验证：

| 环节 | 结果 |
|---|---|
| 下载 / 解码 | ✅ yt-dlp 抓取 + ffmpeg 转码，HEVC/H.264 都能读 |
| 判罚模式切换 | ✅ `--mode doubles/singles` |
| 球检测（YOLO） | ✅ 稳定：160 帧 61 点、120 帧 162 点、300 帧 301 点 |
| 轨迹筛选 | ✅ 抛物线 R²=0.88~1.00 的段能正确挑出 |
| 球场自动标定 | ⚠️ 部分素材成功（demo1 精准），部分生成"看似合理但错误"的小四边形 |
| 端到端判罚 | ⚠️ 仅在「标定正确 + 球轨迹连续跟到落地」时给出；否则如实拒绝 |

本轮修复的真实 bug：

1. **标定被跨视频无条件复用** → 现在需显式 `--reuse-calib`（GUI 勾选框，默认关）且校验分辨率；
2. **边线角度分类错误**：`_line_angle` 返回 [0,180)，原实现把 159°（实为上升 21° 的边线）判成横线 → 改为"偏离水平 ≤12° 才算横线"，`test_1min.mp4` 由此从失败变成功；
3. **自动标定的"假成功"**：新增两道自检——几何自检（占画面 ≥15%、近/远端宽度比 1.15~5.0）+ 标定与轨迹自洽性（轨迹点映射后须 ≥60% 落在球场附近），不自洽则**拒绝判罚并提示重做标定**。

**当前真实能力边界**：单机位 + 正确标定 + 球轨迹连续 → 能给出落点判罚；
球在回合中被漏检断开（本项目常见）→ 会如实输出 `still_in_flight` / `calib_inconsistent`，不会编造判罚。
下一步的提升方向是把检测换成时序模型（TrackNetV3）以提升轨迹连续性，或用微调后的权重提高检出率。

覆盖：依赖 / RK4 终端速度 / 落地三重触发 + 兜底 / 场地判罚 / 全链路模拟 / 单视频流水线。
当前状态：**6/6 PASS**（3D 误差 18.9mm，落点判罚正确，轨迹 R²=0.999）。


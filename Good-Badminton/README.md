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

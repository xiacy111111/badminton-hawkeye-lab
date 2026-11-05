# Badminton Hawk-Eye Lab — 低成本双目羽毛球鹰眼 & 视频分析

> 从单视频技战术分析，到双目立体视觉实时判罚——一个本科生的完整羽毛球视觉项目记录。

## 项目简介

本项目以开源项目 [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)（Apache-2.0，单视频比赛分析）为起点，扩展为**双目立体视觉实时鹰眼判罚系统**，并完整保留了理论推导、实验记录与基准数据。

### 核心技术路线

```
双路工业相机（海康 MV-CU013-A0UC，全局快门）
        │  各自独立标定（球场本身即标定板：16 关键点 + solvePnP RANSAC）
        ▼
左右画面羽毛球检测（5–15px 极小目标 + 运动拖尾）
        ▼
双目三角测量 → 真实 3D 坐标（大基线 B≈13.4m）
        ▼
RK4 空气阻力物理引擎（线性+二次混合阻力，非对称阻力建模）
        ▼
多方法轨迹预测融合（6 种基线 + 交叉评分 / 贝叶斯后验 / Bootstrap 概率化判罚）
        ▼
落点判罚（界内/界外 + 置信度）
```

与主流开源（TrackNetV3 / YOLO-Pose / CNN-LSTM 等单目深度学习统计路线）的差异：本路线面向**毫米级判罚**而非技战术统计，用物理引擎先验替代纯数据驱动外推，在观测段短、球被遮挡时更稳，并输出带置信度的概率化判罚。

## 仓库结构

```
├── Good-Badminton/        主代码库（上游单视频分析 + 双目鹰眼判罚扩展）
├── code/
│   ├── feiqi/             独立流水线原型（双目采集、LED 同步、判罚引擎等）
│   └── tools/             报告生成、球场测试脚本
├── docs/
│   ├── 理论文档/           误差传播理论、RK4 物理引擎、标定方法、市调分析
│   ├── 实验与过程记录/      实验记录、测试数据、基准结果（原始工作日志）
│   ├── new_method协作记录/  阶段性研究分工与方案
│   └── figures/           可视化结果
├── references/            参考文献清单（PDF 仅供个人学习）
└── LICENSE                Apache-2.0（继承自上游 Good-Badminton）
```

## 已有成果（节选）

- 2500 条全变异合成轨迹数据集 + 7 个消融/精进实验（非对称阻力、相位感知融合、串行梯度提升、Bootstrap、贝叶斯后验、序贯假设检验等）
- 全链路动态噪声模拟测试（无需相机即可复现：`python test_integration.py`）
- 基准结果：完整轨迹落点预测误差中位 ≈ 13–22mm，位置 RMSE ≈ 5–8mm（详见 `Good-Badminton/benchmark_results.json`）
- 双目大基线误差传播模型：σ_3D = Z²/(fB)·σ_pixel，用于指导硬件选型

## 硬件清单

| 硬件 | 型号/规格 |
|---|---|
| 工业相机 ×2 | 海康 MV-CU013-A0UC（130 万像素，全局快门）|
| 镜头 ×2 | 6mm C 口 |
| 采集 | USB3.0 有源延长线（15m）×2、三脚架 ×2 |
| 驱动 | 海康 MVS SDK 5.0.1 |

## 快速开始

```bash
cd Good-Badminton
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt

# 无相机全链路模拟测试
python test_integration.py
python run_full_demo.py --mode simulate

# 双目标定 + 实时判罚（需相机）
python calibration.py --left left.jpg --right right.jpg --out calib.json
python run_full_demo.py --mode stereo --calib calib.json
```

## 许可与致谢

- 单视频分析部分继承自 [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)（Apache-2.0），保留原许可证与致谢。
- 双目鹰眼判罚、物理引擎、融合判罚与全部理论文档为本项目原创。
- 为遵守许可证合规，本仓库不含 YOLO 预训练权重（AGPL），请自行下载。

## 说明

`docs/实验与过程记录/` 为原始工作日志（含 AI 辅助分析的对话记录），保留完整探索过程，供参考；结论请以 `docs/理论文档/` 与 benchmark 数据为准。

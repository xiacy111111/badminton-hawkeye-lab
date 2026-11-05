---
model: GPT 5.6 Sol
date: 2026-07-18T14:39:56.431Z
---


> **思考过程**
> 
> **Planning work assignment timing**

```json
{
  "toolName": "getCurrentTime",
  "state": "output-available",
  "input": {
    "timezone": "Asia/Shanghai"
  },
  "output": {
    "iso": "2026-07-18T14:40:04.634Z",
    "formatted": "2026年7月18日星期六 GMT+8 22:40:04",
    "timezone": "Asia/Shanghai"
  }
}
```

> **思考过程**
> 
> **Planning roles and schedule divisions**

> **思考过程**
> 
> **Defining roles, APIs, and initial tasks**

> **思考过程**
> 
> **Establishing detailed project milestones**

> **思考过程**
> 
> **Establishing Git workflow and ownership**

> **思考过程**
> 
> **Defining detailed RACI and schedule**

> **思考过程**
> 
> **Outlining initial two-week schedule**

> **思考过程**
> 
> **Defining success gates for roles**

> **思考过程**
> 
> **Planning paper section division and authorship**

可以。按照你们要冲击 SCI 二区的目标，最合理的分工不是“两个人都写算法”，而是建立两条并行主线：

- **你：理论模型、轨迹估计、不确定性与统计负责人**
- **队友：双目视觉、硬件实验、数据集与系统实现负责人**

这样你从今晚开始搭建理论，他明天下午开始硬件和视觉，二者通过统一数据接口汇合，避免重复劳动。

现在是北京时间 **2026 年 7 月 18 日晚**，下面按这个时间制定。

---

# 一、两个人的固定角色

## 你：A 负责人——理论、算法与统计主线

你的核心职责：

1. 统一数学模型；
2. 建立轨迹预测方法；
3. 设计不确定性传播；
4. 设计概率判罚；
5. 建立仿真和基线；
6. 制定统计检验；
7. 负责论文方法章节和最终统稿。

你拥有以下目录的主要修改权：

```text
src/
├─ physics/
├─ estimation/
├─ prediction/
├─ uncertainty/
├─ decision/
└─ statistics/
```

## 队友：B 负责人——视觉、硬件与数据主线

队友核心职责：

1. 相机驱动与同步；
2. 内参、外参、球场坐标标定；
3. 羽毛球二维检测；
4. 左右图像关联与三角化；
5. Ground truth 系统；
6. 真实数据采集与标注；
7. 系统速度和硬件稳定性测试；
8. 负责论文系统、数据和实验设备章节。

队友拥有以下目录的主要修改权：

```text
src/
├─ capture/
├─ detection/
├─ calibration/
├─ geometry/
├─ association/
└─ dataset/
```

## 共同维护

```text
configs/
docs/
tests/integration/
scripts/
paper/
```

共同模块不能随意修改，必须双方确认。

---

# 二、为什么这样分最合理

你们研究中的主要误差链是：

\[
\text{相机采集}
\rightarrow
\text{二维检测}
\rightarrow
\text{双视角关联}
\rightarrow
\text{三维状态}
\rightarrow
\text{轨迹预测}
\rightarrow
\text{落点分布}
\rightarrow
\text{判罚}
\]

队友负责前半段：

\[
\text{图像}\rightarrow\text{二维观测及协方差}
\]

你负责后半段：

\[
\text{二维观测及协方差}
\rightarrow
\text{物理约束轨迹}
\rightarrow
P(\mathrm{IN})
\]

你们只需在中间统一接口，不需要相互等待。

---

# 三、今晚你立即完成的任务

## 今晚目标

不要急着跑大规模实验。今晚最重要的是冻结理论和接口，否则队友明天写的代码可能无法接入。

建议今晚完成以下 6 项。

---

## 任务 A1：冻结坐标系

建立：

```text
docs/PROJECT_CONTRACT.md
```

统一采用：

\[
X\in[-6.70,6.70]
\]

沿球场长度方向，球网位于 \(X=0\)。

\[
Y\in[-3.05,3.05]
\]

沿球场宽度方向。

\[
Z\geq0
\]

竖直向上，地面为 \(Z=0\)。

明确：

- 单位统一为米、秒、弧度；
- 像素单位只能用于图像层；
- 世界坐标不得混用毫米；
- 落地定义为 \(Z=0\) 且 \(v_z<0\)；
- 轨迹超时不能当作落地；
- 球网平面为 \(X=0\)。

---

## 任务 A2：统一物理方程

不要再让不同脚本各写一套阻力模型。

统一为：

\[
m\dot{\mathbf v}
=
m\mathbf g
-c_1\mathbf v_r
-c_2\|\mathbf v_r\|\mathbf v_r
\]

其中：

\[
\mathbf v_r=\mathbf v-\mathbf w
\]

参数定义：

- \(m\)：羽毛球质量，kg；
- \(c_1\)：线性阻力系数，kg/s；
- \(c_2\)：二次阻力系数，kg/m；
- \(\mathbf w\)：风速，m/s；
- \(\mathbf g=[0,0,-9.81]^\top\)。

第一版可以固定 \(c_1=0\)，只使用 \(c_2\)，但接口必须保留两个参数。

建立：

```text
docs/PHYSICS_MODEL.md
src/physics/dynamics.py
```

---

## 任务 A3：定义预测模型和真值模型

为避免“自己生成数据证明自己最好”，必须有两套模型。

### 预测模型

简单、可在线运行：

- 固定或慢时变 \(c_2\)；
- 无旋转或简化旋转；
- RK4 积分。

### 仿真真值模型

比预测模型复杂：

- 时变阻力；
- 轻微各向异性；
- 风；
- 参数扰动；
- 可选随机模型失配。

例如：

\[
c_{2,k+1}=c_{2,k}+\eta_k
\]

以及：

\[
c_h\neq c_v
\]

但应称为“经验模型失配”，不能称为完整 CFD。

---

## 任务 A4：写出联合估计目标函数

建立：

```text
docs/METHOD_FORMULATION.md
```

模型核心为：

\[
\min_{\mathbf s_{1:K}}
\sum_{k,c}
\rho
\left(
\left\|
\mathbf u_k^c-\pi_c(\mathbf p_k)
\right\|_{\Sigma_{u,k}^{c^{-1}}}^{2}
\right)
+
\lambda
\sum_k
\rho
\left(
\left\|
\mathbf s_{k+1}-F_{\mathrm{RK4}}(\mathbf s_k)
\right\|_{Q^{-1}}^{2}
\right)
+
L_{\mathrm{prior}}
\]

这里暂时只需要完成理论定义，不必今晚完成全部优化器。

状态建议：

\[
\mathbf s_k=
[\mathbf p_k,\mathbf v_k,c_{1,k},c_{2,k}]
\]

可选加入风速，但第一版不要一下估计太多不可辨识参数。

---

## 任务 A5：定义概率落点判罚

写清楚：

\[
p(x_L,y_L\mid\mathcal D_{1:k})
\]

和：

\[
P(\mathrm{IN})
=
\int_{\Omega_{\mathrm{IN}}}
p(x_L,y_L)\,dx_Ldy_L
\]

先定义三级结果：

```text
IN
OUT
UNCERTAIN
```

建议初始规则：

```text
P(IN) >= 0.99       → IN
P(IN) <= 0.01       → OUT
其他                 → UNCERTAIN
```

正式阈值以后在验证集冻结。

---

## 任务 A6：冻结中间接口

建立：

```text
docs/API_CONTRACT.md
```

至少包含以下数据结构。

```python
@dataclass
class ImageObservation:
    frame_id: int
    timestamp: float

    left_uv: tuple[float, float] | None
    right_uv: tuple[float, float] | None

    left_cov: np.ndarray | None
    right_cov: np.ndarray | None

    left_confidence: float
    right_confidence: float

    sync_error_std: float
    calibration_id: str
```

三维结果：

```python
@dataclass
class StateEstimate:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray

    drag_linear: float
    drag_quadratic: float

    reprojection_error: float
    condition_number: float
    confidence: float
    failure_reason: str | None
```

落点输出：

```python
@dataclass
class LandingPrediction:
    mean_xy: np.ndarray
    covariance_xy: np.ndarray

    p_in: float
    ruling: str

    observation_fraction: float
    confidence: float
    runtime_ms: float
```

只要这三个接口冻结，队友明天就可以独立工作。

---

# 四、你明天上午继续做什么

队友明天下午才开始，因此你明天上午完成理论主线的第一版。

## 上午任务 A7：数值积分验证

完成：

```text
src/physics/integrator.py
tests/test_integrator.py
```

测试：

1. 无阻力时与解析抛物线比较；
2. 与高精度 `solve_ivp` 比较；
3. 步长 10、5、2.5、1 ms 收敛；
4. 地面事件插值；
5. timeout 不当落地；
6. 无 NaN/Inf；
7. 参数单位一致。

不要追求“结果特别漂亮”，只验证代码正确。

## 上午任务 A8：建立统计协议

建立：

```text
docs/EXPERIMENT_PROTOCOL.md
```

明确：

- 按完整轨迹拆分数据；
- 禁止按帧随机拆分；
- 测试集不能调参；
- Template 库只能来自训练集；
- 所有方法必须使用相同观测；
- 所有失败样本必须保留；
- 连续误差用配对 Bootstrap 和 Wilcoxon；
- 判罚准确率用 McNemar；
- 多方法比较使用 Holm 校正。

---

# 五、队友明天下午开始的具体任务

建议队友明天下午不要直接开始训练 YOLO，而是先把相机系统和数据质量跑通。

## 明天下午 B1：项目和硬件审计

队友先输出：

```text
docs/HARDWARE_REPORT.md
```

记录：

- 相机型号；
- 镜头型号；
- 实际分辨率；
- 实际帧率；
- 可用曝光范围；
- 增益；
- 是否全局快门；
- 两台相机序列号；
- USB 控制器分配；
- MVS SDK 版本；
- 硬件时间戳是否可读；
- 软件触发和自由采集是否可用。

必须记录实际结果，不能只抄产品规格。

---

## 明天下午 B2：双相机采集程序

负责整理：

```text
src/capture/hikvision.py
src/capture/synchronizer.py
tests/test_dual_capture.py
```

输出每帧：

```python
CameraFrame(
    image=...,
    frame_id=...,
    timestamp_device=...,
    timestamp_host=...,
    exposure_us=...,
    gain_db=...
)
```

要求：

- 两台相机独立帧号；
- 设备时间戳与主机时间戳分开；
- 记录丢帧；
- 记录实际 FPS；
- 不允许只靠 `time.time()` 假装硬同步；
- UVC 回退模式必须明确标记为低可信模式。

---

## 明天下午 B3：曝光与帧率测试

测试矩阵：

| 曝光 | 目标帧率 |
|---:|---:|
| 0.2 ms | 60/100/130 fps |
| 0.5 ms | 60/100/130 fps |
| 1.0 ms | 60/100 fps |
| 2.0 ms | 60/100 fps |

每组记录：

- 实际 FPS；
- 丢帧率；
- 平均亮度；
- 图像噪声；
- 羽毛球拖影长度；
- 是否过曝；
- USB 带宽是否稳定。

输出：

```text
outputs/hardware/fps_exposure_matrix.xlsx
```

---

## 明天下午 B4：标定方案

先不要立即拍大规模数据，建立标定流程：

```text
src/calibration/intrinsic.py
src/calibration/extrinsic.py
src/calibration/court_alignment.py
```

标定分三层：

1. 棋盘格内参和畸变；
2. 每台相机到球场坐标系的外参；
3. 双相机相对位姿与重投影检查。

不要只使用“检测四角＋solvePnP”作为最终标定。自动球场线可以提供初始值，但正式结果最好由更多球场关键点优化。

---

# 六、明天晚上两个人必须进行第一次接口会议

时间控制在 30～45 分钟。

## 你向队友交付

```text
docs/PROJECT_CONTRACT.md
docs/API_CONTRACT.md
docs/PHYSICS_MODEL.md
docs/METHOD_FORMULATION.md
docs/EXPERIMENT_PROTOCOL.md
src/physics/
tests/test_integrator.py
```

## 队友向你交付

```text
docs/HARDWARE_REPORT.md
双相机实际出图截图
FPS和曝光初步数据
设备时间戳可用性结果
相机驱动问题列表
```

## 会议必须冻结四件事

1. 世界坐标系；
2. 时间戳单位；
3. 图像观测接口；
4. 标定文件格式。

这四项冻结后，任何人都不能私自修改。

---

# 七、接下来两周详细分工

## 第 1 周：基础正确性和试运行

### 你负责

- 统一物理内核；
- RK4 数值验证；
- 固定阻力轨迹预测；
- KF/UKF 基线；
- 模型失配仿真；
- 概率落点基础；
- 统计协议；
- 测试配置。

### 队友负责

- 相机驱动；
- 双机时间戳；
- 曝光、增益与帧率；
- 内参和畸变标定；
- 球场外参；
- 静态三维网格验证；
- 二维检测初始基线。

### 一周联合产出

必须跑通一条最小链路：

```text
双相机图像
→ 左右二维坐标
→ 三维重建
→ 状态估计
→ RK4落点
→ 3D可视化
```

此时不追求高精度，只追求：

- 接口正确；
- 坐标正确；
- 时间正确；
- 可以复现；
- 失败有记录。

---

## 第 2 周：动态试运行和方法原型

### 你负责

- 滑动窗口联合估计第一版；
- 鲁棒损失；
- 固定阻力与在线阻力；
- 落点 Monte Carlo；
- \(P(\mathrm{IN})\)；
- 基线统一评估器；
- 多阶段预测指标。

### 队友负责

- 二维检测器基线比较；
- 帧差、YOLO、TrackNet 类方法测试；
- 双目极线门控；
- 错误匹配分析；
- 30～50 条动态试运行；
- Ground truth 试验；
- 数据清单和标注工具。

### 两周联合产出

完成第一版开发报告：

```text
outputs/reports/pilot_report.md
```

报告至少包含：

- 实际 FPS；
- 同步误差；
- 标定重投影误差；
- 静态三维误差；
- 二维检测召回率；
- 动态轨迹成功率；
- 初步落点误差；
- 失败类型；
- 下一阶段需要修正的问题。

---

# 八、第 3～6 周分工

## 你：方法深化

负责：

1. 二维重投影＋动力学联合优化；
2. 在线等效阻力；
3. 模型失配鲁棒性；
4. 状态协方差；
5. 概率落点；
6. 判罚校准；
7. 基线、消融和统计检验；
8. 算法运行效率优化。

## 队友：真实数据与视觉深化

负责：

1. 扩大检测数据；
2. 多光照、多球型采集；
3. 二维定位误差标注；
4. 同步与遮挡测试；
5. 真实轨迹 Ground truth；
6. near-line 数据采集；
7. 数据质量审计；
8. 系统实时性与部署。

## 共同任务

每周固定完成：

- 一次代码审查；
- 一次实验结果审查；
- 一次论文写作；
- 一次失败案例分析。

---

# 九、第 7～10 周分工

## 你负责论文部分

主要写：

- Introduction 的研究缺口；
- Physics Model；
- Joint Estimation；
- Uncertainty Propagation；
- Probabilistic Line Calling；
- Statistical Analysis；
- Results；
- 统稿。

## 队友负责论文部分

主要写：

- System Overview；
- Camera Setup；
- Calibration；
- Shuttlecock Detection；
- Stereo Association；
- Dataset；
- Ground Truth；
- Runtime Implementation；
- Experimental Setup。

## 共同写

- Related Work；
- Ablation；
- Failure Cases；
- Limitations；
- Discussion；
- Conclusion。

最终只能有一个人统稿，建议由你负责，否则符号、坐标和术语容易前后不一致。

---

# 十、代码协作规则

## Git 分支

```text
main                 稳定版本
theory/<task>        你的功能分支
vision/<task>        队友功能分支
experiment/<task>    联合实验分支
paper/<section>      论文分支
```

禁止两个人长期直接改 `main`。

## 合并规则

每次合并必须：

1. 测试通过；
2. 接口未静默修改；
3. 提交信息清楚；
4. 新增配置有说明；
5. 失败情况有日志；
6. 数据文件不直接塞进 Git。

## 每日提交

你：

```text
[theory] 完成动力学模型和数值测试
```

队友：

```text
[vision] 完成双相机采集与帧率测试
```

联合：

```text
[integration] 跑通单条双目轨迹
```

---

# 十一、两个人必须共享的文件

## 1. 项目状态

```text
docs/TASK_STATE.md
```

格式：

```markdown
## 2026-07-19

### A负责人完成
- 物理方程冻结
- RK4单元测试
- 联合目标函数初稿

### B负责人完成
- 双相机出图
- 曝光帧率测试
- 时间戳检查

### 当前阻塞
- 硬件同步尚未确认
- 标定板未到

### 下一步
- 静态三维网格测试
```

## 2. 实验日志

```text
experiments/EXPERIMENT_LOG.md
```

每次实验必须记录：

- 日期；
- 场地；
- 相机位置；
- 相机配置；
- 标定文件；
- 球型号；
- 数据编号；
- 操作者；
- 异常；
- 对应 Git commit。

## 3. 数据清单

```text
data/manifest.xlsx
```

至少包含：

```text
trajectory_id
session_id
shuttlecock_id
shuttlecock_type
wear_level
camera_config
calibration_id
fps
exposure
ground_truth_method
split
validity
failure_reason
```

---

# 十二、两个人不得重复做的事情

## 你不要做

- 不要私自重写相机驱动；
- 不要在没有实际相机数据时声称检测噪声是 0.5 px；
- 不要替队友修改标定逻辑后不通知；
- 不要用真实测试集调气动参数。

## 队友不要做

- 不要自行更换世界坐标定义；
- 不要在多个脚本复制自己的 RK4；
- 不要为了三维轨迹“更平滑”直接删除异常帧；
- 不要按帧随机拆分训练测试集；
- 不要在检测器中使用未来帧后又声称在线预测。

---

# 十三、阶段验收门槛

这些是建议的工程门槛，不是论文中预先保证的结果。

## 阶段 1：硬件可用

- 双相机连续采集稳定；
- 实际 FPS 和丢帧率可测；
- 时间戳可追踪；
- 标定可重复；
- 图像不严重过曝。

## 阶段 2：几何可用

建议目标：

- 重投影 RMS 尽量低于 0.5 px；
- 静态三维误差有完整 P50/P95；
- 全场无明显系统性坐标翻转；
- 三角化退化区域已识别；
- 理论协方差趋势与实测基本一致。

## 阶段 3：动态可用

- 轨迹成功率有明确统计；
- 失败轨迹不删除；
- 落点真值方法可重复；
- 多阶段预测可运行；
- 所有方法使用相同输入。

## 阶段 4：论文可用

- 独立锁定测试集；
- OOD 测试；
- 方法显著优于强基线；
- 消融结果解释一致；
- 概率覆盖率基本校准；
- 代码和实验可复现。

---

# 十四、你今晚最小完成清单

如果今晚时间有限，优先级按下面顺序：

### 必须完成

- [ ] `PROJECT_CONTRACT.md`
- [ ] `API_CONTRACT.md`
- [ ] 统一坐标系
- [ ] 统一阻力方程
- [ ] 定义联合估计目标函数
- [ ] 定义队友明天需要返回的图像观测格式

### 尽量完成

- [ ] `dynamics.py`
- [ ] RK4 积分器
- [ ] 无阻力解析解测试
- [ ] ground event 测试
- [ ] 统计协议初稿

### 今晚不要做

- [ ] 不跑 2500 条正式实验；
- [ ] 不写商业计划书；
- [ ] 不设计 GUI；
- [ ] 不训练新检测网络；
- [ ] 不急着画论文结果图；
- [ ] 不重新实现六套融合算法。

---

# 十五、最终分工一句话版本

你可以直接把这一段发给队友：

> 我负责羽毛球动力学、轨迹联合估计、在线阻力参数、不确定性传播、落点概率判罚、基线和统计分析；你负责双相机采集与同步、内外参及球场标定、二维羽毛球检测、双目关联和三角化、真实 Ground truth、数据采集标注以及系统实时性。我们以统一的 ImageObservation、StateEstimate 和 LandingPrediction 接口衔接，每晚同步一次代码和问题，每两天完成一次端到端集成测试，任何人不得私自修改坐标系、单位和公共接口。

按照这个分工，你们两个人可以真正并行推进：**你今晚开始搭理论，队友明天下午直接接硬件与视觉，两条线在第一周末汇合成完整系统。**

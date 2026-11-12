# TrackNetV3 权重获取说明（当前网络不可达）

## 现状

- 代码已下载：`third_party/TrackNetV3/qaz812345-TrackNetV3-*/`（MIT 许可）
- **权重未获取**：官方 ckpts 只有两个来源，本机网络均不可达
  1. Google Drive（README 第 59 行链接）：`https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view`
     → `curl` 超时，无法访问
  2. GitHub 上他人提交的 LFS 副本：`Siddharth194/BadmintonAnalyticsCV/ckpts/TrackNet_best.pt`
     → 是 136 MB 的 Git LFS 对象，`github.com`（LFS 端点）在本机被墙，只能取到 134 字节指针

## 拿到权重后怎么接（3 步）

```bash
cd third_party/TrackNetV3/qaz812345-TrackNetV3-*/
mkdir -p ckpts
# 把 TrackNet_best.pt 与 InpaintNet_best.pt 放进 ckpts/
python predict.py --video_file 你的视频.mp4 \
       --tracknet_file ckpts/TrackNet_best.pt \
       --inpaintnet_file ckpts/InpaintNet_best.pt \
       --save_dir prediction --output_video
```

输出的 `prediction/*.csv` 是逐帧球坐标（x, y, visibility），可直接喂给：
`video_pipeline.py` 的轨迹/落点环节（把 CSV 读成 `per_frame` 列表即可，
参考 `link_tracks` 的输入格式 `[(cx, cy, score), ...]`）。

## 为什么没等它也在推进

TrackNet 解决的是"球轨迹连续性"。本轮先用工程手段顶上（见 `track_refine.py`）：

| 指标（200 帧真实素材） | 改进前 | 改进后 |
|---|---|---|
| 有检出帧占比（demo1） | 86% | **97%** |
| 主轨迹长度（demo1） | 16 点 | **93 点** |
| 主轨迹长度（test_1min） | 37 点 | **102 点** |
| 轨迹覆盖率 | — | 92~100% |

手段：运动模型关联（容忍 1~6 帧漏检）+ 二次拟合插值 + 模板匹配补检 +
场内占比/路径长度物理门限。等权重到位后可直接替换/对照。

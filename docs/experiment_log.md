# 实验日志

## 2026-08-09：环境与 SFT 基线链路

### 目标

验证 π0.5 SFT checkpoint 能否在 LIBERO-Spatial 中完成完整 rollout。

### 配置

- checkpoint：`RLinf/RLinf-Pi05-LIBERO-SFT`
- suite：`libero_spatial`
- task / trial：`0 / 0`
- 最大步数：240
- 评估轨迹数：1
- renderer：OSMesa

### 结果

- `success_once=1.0`
- `success_at_end=1.0`
- `return=1.0`
- `episode_len=240`

这是**链路验证结果**，不是 10-trial 或 benchmark 汇总。

### 已解决问题

1. 服务器不能直连 Hugging Face，使用 `HF_ENDPOINT=https://hf-mirror.com`。
2. 下载中断文件即使大小正确也可能无法被 safetensors 解析；正式使用前以 `safe_open` 校验。
3. EGL 录制的视频出现绿屏。OSMesa 的原始 RGB 帧正常；浏览器端 MP4 兼容性待单独处理，不影响环境成功信号和 PPO 训练。

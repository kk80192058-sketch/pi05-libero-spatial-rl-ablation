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

## 2026-08-09：SFT 小基线（task 0，10 个初始状态）

### 目的

把单条 smoke test 扩展为可汇总的、固定协议的小基线。此处评估的不是整个 LIBERO-Spatial benchmark，而是其中的 task 0。

### 固定协议

- checkpoint：`RLinf/RLinf-Pi05-LIBERO-SFT`
- suite：`libero_spatial`
- task：`0`（通过 `+env.eval.task_id_filter=[0]` 过滤）
- 初始状态：trial `0`–`9`
- 最大步数：240
- 评估轨迹数：10
- seed：0
- renderer：OSMesa
- 视频：关闭；避免把浏览器的视频解码问题混入策略评估

### 结果

- 10 / 10 个 trial 成功
- `success_once=1.0`
- `success_at_end=1.0`
- `return=1.0`
- `episode_len=240`
- 总耗时：约 3 分 43 秒

### 解释与边界

这个 checkpoint 在该小协议下稳定完成了 task 0，故可作为后续“少数据 SFT”与“PPO 后训练”的对照项。它不说明其余 9 个 Spatial task 的表现，也不等价于官方 benchmark 汇总成绩。

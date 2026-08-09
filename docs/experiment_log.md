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

## 2026-08-09：主任务筛选与 task 6 基线

### 筛选

对 Spatial 的 10 个 task 各取 trial 0、并行评测后，公开 SFT checkpoint 的汇总成功率为 8 / 10。task 6 和 task 9 失败，其余 task 成功。

### task 6 固定基线

- task：`6`
- trial：`0`–`9`
- 最大步数：240
- 评估轨迹数：10
- seed：0
- renderer：OSMesa，视频关闭

结果为 **6 / 10 = 60%** 成功：trial `0, 1, 6, 8` 失败，trial `2, 3, 4, 5, 7, 9` 成功。

### 项目决策

task 0 已饱和（10 / 10），不适合验证 PPO 的增益；task 6 的 60% 基线提供了明确的改善空间。因此后续 PPO 实验固定使用 task 6 和上述 10-trial 协议。这个选择来自实际基线测量，而不是事后挑选单条成功视频。

## 2026-08-09：task 6 demonstration 子集

从 `physical-intelligence/libero` 的 LeRobot 数据集中仅下载了 task 6 的 10 条专家轨迹，作为数据链路验证与后续少数据扩展实验的数据子集。数据目录不提交至 Git，仅记录来源与索引。

- 任务语言：`put both moka pots on the stove`
- episode index：`10, 20, 23, 46, 51, 54, 57, 67, 70, 73`
- 每条轨迹：359–455 frames
- 本地服务器目录：`/root/autodl-tmp/vla-rl/data/libero_spatial_task6_10demos`

注：数据集的 episode 编号按任务交错，而非“前 10 条都属于 task 0”。此前下载的 `episode_000000`–`episode_000009` 是混合格式检查样本，不能作为 task 0 的十条 demonstration。

## 2026-08-09：PPO 配置 dry-run

使用 RLinf 官方 `libero_spatial_ppo_openpi_pi05` 配置，以 `--cfg job` 仅解析配置；未加载模型权重、未创建训练 worker、未发生训练。

本次验证的最小参数为：8 个训练环境、1 个 rollout epoch、micro/global batch 均为 8、1 个训练 epoch、评测视频关闭。模型 actor 和 rollout 路径均正确指向 `RLinf-Pi05-LIBERO-SFT`，且 π0.5 actor 的 value head 正确启用。

首次尝试传入重复的 `--config-path`，Hydra 将其拼成不存在的路径；切换到 `examples/embodiment` 工作目录后成功。这一问题已纳入运行脚本的目录处理。

## 2026-08-10：A800 迁移与 PPO smoke run

### A800 迁移验证

- 硬件：1 × NVIDIA A800-SXM4-80GB、18 vCPU、120GB RAM、300GB 数据盘
- 迁移目录：`/root/autodl-tmp/vla-rl`，迁移后大小约 36GB
- PyTorch：CUDA 可用；`model.safetensors` 可读取 812 个 tensor

### PPO smoke run（真实更新，不作结果比较）

- task：6；训练环境数：2；rollout epoch：1；训练 epoch：1
- 视频、checkpoint 和周期评测均关闭
- 结果：2 条轨迹均至少成功一次；rollout 约 133 秒，actor 训练约 13 秒
- 更新日志：`policy_loss=0.0091`、`approx_kl=-0.0058`、`grad_norm=57.274`、`value_loss=0.048`

该 smoke run 的作用是确认 A800 上环境采样、PPO advantage/return、actor 与 critic 反向传播、权重同步均已真实执行；由于只有 1 epoch / 2 条轨迹，不能作为策略效果结论。

### A800 训练前评估

在相同 task 6 的 10-trial 协议下，本次采样得到 `success_once=0.8`、`success_at_end=0.4`。模型策略采样本身存在随机性，且成功后继续执行可使任务状态被扰动；项目将以 `success_once` 为主指标，并在正式前后用相同次数的重复评估报告均值，避免将单次 0.6 或 0.8 误写成稳定性能。

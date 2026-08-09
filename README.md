# π0.5 LIBERO-Spatial：SFT + PPO 在线后训练消融实验

一个面向 VLA / 强化学习实习的可复现实验项目。项目使用 RLinf、OpenPI π0.5 和 LIBERO-Spatial 仿真环境，研究少量专家演示（SFT）与在线 PPO 后训练分别、以及组合后，能否提升机器人策略的任务成功率。

## 正式系统消融：task 9（进行中）

不再随意挑单一任务。先以相同协议筛查全部 10 个 LIBERO-Spatial task，再选择基础成功率最低且未饱和的 task 9：

> `pick up the book and place it in the back compartment of the caddy`

固定评测协议：每个窗口 10 条轨迹、每条最多 240 个环境步；窗口 `0–9` 与 `10–19` 互不重叠。`success_once`（轨迹中曾完成任务）为主指标，`success_at_end`（最后一步仍完成）为辅助指标。

| task id | A0 `success_once`（0–9） |
| ---: | ---: |
| 9 | **30%（3/10）** |
| 6 | 60%（6/10） |
| 4 | 70%（7/10） |
| 3 / 5 | 80%（8/10） |
| 8 | 90%（9/10） |
| 0 / 1 / 2 / 7 | 100%（10/10） |

完整筛查数据在 [results/spatial_seen_diagnostic.csv](results/spatial_seen_diagnostic.csv)。task 9 的 A0 在独立的未见窗口 `10–19` 也是 **30%（3/10）**，因此它是合理的正式训练目标。

## 正式 2×2 消融设计（task 9）

| 组别 | 初始化策略 | 专项 SFT | 在线 PPO | 目的 |
| --- | --- | --- | --- | --- |
| A0 | 公开 π0.5 LIBERO SFT | 否 | 否 | 基线 |
| A1 | A0 | 否 | 是 | 单独衡量 PPO |
| B20 | A0 | 20 条 task 9 演示 | 否 | 单独衡量少样本 SFT |
| C20 | B20 | 已完成 | 是 | 衡量 SFT+PPO 组合 |

所有组别都会在两个 reset 窗口分别评测，最终报告只填写实际日志产生的结果。当前已完成 A0；B20 的 500-step SFT 已结束，正在做 checkpoint 复测；A1、C20 将在此后依次训练。

## 环境与资源

- 4090 24GB：环境配置与 rollout 调试
- A800 80GB：task 9 的正式 SFT、PPO 与多窗口评测
- 框架：RLinf + OpenPI π0.5 + LIBERO / MuJoCo

80GB 显存提供 PPO 所需的模型、优化器和轨迹缓存余量；并行环境数同时受 18 vCPU 仿真吞吐限制，实验采用 8 个训练环境以保持稳定、可比较的训练预算。

## 可复现运行

```bash
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

### 1. 评测指定任务与 reset 窗口

`patches/rlinf_eval_reset_offset.patch` 为 RLinf 增加 `eval_reset_start_idx`，只改变评测从第几个 reset state 开始，不改变环境、奖励或模型。

```bash
RLINF_DIR=/path/to/RLinf \
PROJECT_DIR=$PWD \
MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT \
TASK_ID=9 RESET_START=0 NUM_ENVS=10 \
OUTPUT_DIR=/path/to/results/task9_a0_seen \
bash scripts/run_libero_task_window_eval.sh
```

### 2. 生成固定的 20 条演示训练集

```bash
python scripts/download_libero_spatial_task_demos.py \
  --task-id 9 --output /path/to/libero_spatial_task9_all

python scripts/build_libero_task_sft_subset.py \
  --source /path/to/libero_spatial_task9_all \
  --output /path/to/libero_spatial_task9_train20 \
  --train-count 20 --seed 42
```

`split.csv` 会记录训练 episode；数据和模型权重体积较大，不提交 Git。

### 3. B20：少样本 SFT

```bash
RLINF_DIR=/path/to/RLinf PROJECT_DIR=$PWD \
MODEL_PATH=/path/to/RLinf-Pi05-LIBERO-SFT \
DATA_PATH=/path/to/libero_spatial_task9_train20 \
OUTPUT_DIR=/path/to/task9_sft20_run1 EXPERIMENT_NAME=task9_sft20 \
bash scripts/run_libero_task_sft20.sh
```

默认训练预算：500 steps、batch size 8、学习率 `5e-6`。脚本还会把评测必需的 OpenPI normalization assets 放进每个 checkpoint，确保可直接复测和继续 PPO。

### 4. A1 或 C20：单任务 PPO

```bash
RLINF_DIR=/path/to/RLinf \
MODEL_PATH=/path/to/policy_or_sft_checkpoint_actor \
OUTPUT_DIR=/path/to/task9_ppo_run1 \
TASK_ID=9 EXPERIMENT_NAME=task9_pi05_ppo \
MAX_EPOCHS=10 TRAIN_ENVS=8 \
bash scripts/run_libero_task_ppo.sh
```

## 仓库内容

```text
configs/       SFT 配置
docs/          task 9 实验计划与最终报告
patches/       RLinf 评测 reset-window 补丁
results/       可提交的小型指标汇总 CSV
scripts/       数据准备、评测、SFT、PPO 启动脚本
```

## 结果口径与限制

- `n/10` 同时写为百分比，例如 `30%（3/10）`；不能把单条视频当 benchmark。
- 10 条轨迹的方差仍较大，最终报告会明确样本数和限制，而不把单次结果包装为通用能力结论。
- 10-task 筛查表保留每个 task 的结果，作为 task 9 被选为主实验任务的依据；它不是对其他 task 的训练结论。

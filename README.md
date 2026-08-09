# π0.5 LIBERO 单任务 SFT + PPO 后训练

这是一个面向 VLA/RL 实习的最小可复现项目：在 LIBERO-Spatial 单任务中，先建立 π0.5 的 SFT 基线，再对存在失败空间的任务进行 PPO 在线后训练，并比较训练前后的成功率。

## 项目问题

在固定的 LIBERO-Spatial task 6 和评测协议下，π0.5 SFT 策略能否通过在线 PPO 的环境成功奖励提高任务完成率？

## 当前进度

- [x] 在 RTX 4090 上完成 RLinf、OpenPI、LIBERO 环境配置。
- [x] 下载并校验 `RLinf/RLinf-Pi05-LIBERO-SFT` 权重。
- [x] 跑通 LIBERO-Spatial task 0 的完整 240-step SFT rollout。
- [x] 建立 task 0、10 个初始状态的 SFT 小基线（10/10 成功）。
- [x] 筛选出有提升空间的主任务：task 6 的 SFT 基线为 6/10 成功。
- [x] 下载 task 6 的全部 29 条 demonstration，并固定 20/9 train/holdout 划分。
- [x] 校验 20 条 task 6 训练示范可被 LeRobot/RLinf 数据加载器读取（8,082 帧）。
- [ ] 训练 task 6 的 20-demo few-shot SFT 策略。
- [ ] 对 task 6 的 π0.5 SFT 策略执行 PPO 在线后训练。
- [ ] 对 20-demo SFT 策略执行 PPO 在线后训练。
- [ ] 比较 SFT-only 与 SFT + PPO 的成功率和样本效率。

当前单条成功 rollout 仅用于验证链路，**不作为正式 benchmark 结果**。
当前的 10-trial 结果只覆盖指定 Spatial 子任务，不能表述为整个 benchmark 的成绩。

## 结构

```text
configs/       实验配置
docs/          实验日志与问题记录
results/       小型指标汇总（不含大文件）
scripts/       可复现的服务器运行命令
```

## 环境

- GPU：RTX 4090 24GB（环境、推理、调试）
- 正式训练：计划使用 A800 80GB
- 框架：RLinf + OpenPI π0.5 + LIBERO
- 仿真：LIBERO-Spatial / MuJoCo

## 复现：单条 smoke evaluation

在 RLinf 仓库中执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

bash evaluations/run_eval.sh libero_spatial_openpi_pi05_eval \
  rollout.model.model_path=/path/to/RLinf-Pi05-LIBERO-SFT \
  env.eval.total_num_envs=1 \
  env.eval.specific_reset_id=0 \
  env.eval.max_steps_per_rollout_epoch=240 \
  env.eval.max_episode_steps=240
```

## 结果口径

正式结果将固定任务、随机种子、评估 episode 数和最大步数，并报告 success rate。单个 episode 的 `success_once=1` 只表示该轨迹成功，不能代表泛化性能。

本项目当前的小基线固定为：`task_id=0`、10 个不同 reset state、`max_steps=240`、`seed=0`，因此成功率为 `successes / 10`。

## 少样本 SFT（B20）

`configs/libero_task6_sft20_pi05.yaml` 固定了 B20 组的主要超参数：从公共 π0.5 LIBERO SFT 权重开始、20 条 task 6 示范、500 个更新步、batch size 8、学习率 `5e-6`。实际运行时使用：

```bash
RLINF_DIR=/root/autodl-tmp/vla-rl/RLinf \
PROJECT_DIR=/root/autodl-tmp/vla-rl/project \
MODEL_PATH=/root/autodl-tmp/vla-rl/models/RLinf-Pi05-LIBERO-SFT \
DATA_PATH=/root/autodl-tmp/vla-rl/data/libero_spatial_task6_train20 \
OUTPUT_DIR=/root/autodl-tmp/vla-rl/results/task6_sft20_run1 \
bash scripts/run_task6_sft20.sh
```

训练完成后，会用与 A1 相同的 task 6、10 个固定 reset state、240-step 上限评估 B20；随后再以 B20 checkpoint 作为 PPO 的起点得到 C20。所有最终结论只填写真实日志中的结果。

## B20 + PPO（C20）

当 B20 在第 500 步保存 checkpoint 后，以其 `actor` 子目录作为 C20 的初始化：

```bash
RLINF_DIR=/root/autodl-tmp/vla-rl/RLinf \
PROJECT_DIR=/root/autodl-tmp/vla-rl/project \
SFT_CHECKPOINT=/root/autodl-tmp/vla-rl/results/task6_sft20_run1/task6_sft20/checkpoints/global_step_500/actor \
OUTPUT_DIR=/root/autodl-tmp/vla-rl/results/task6_sft20_ppo_run1 \
bash scripts/run_task6_sft20_ppo.sh
```

脚本会先检查 SFT checkpoint 目录存在，再复用与 A1 完全相同的 PPO 预算（10 epoch、8 个并行环境）。因此 B20→C20 的差异只来自初始策略，而不是悄悄改变了 PPO 训练条件。

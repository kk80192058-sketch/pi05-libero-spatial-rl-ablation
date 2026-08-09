# π0.5 LIBERO 单任务 SFT + PPO 后训练

这是一个面向 VLA/RL 实习的最小可复现项目：在 LIBERO-Spatial 单任务中，先建立 π0.5 的 SFT 基线，再用少量 demonstration 构造低数据基线，并比较 PPO 在线后训练前后的成功率。

## 项目问题

少量 demonstration 下，π0.5 的 SFT 策略能否通过在线 PPO 的环境成功奖励提高任务完成率？

## 当前进度

- [x] 在 RTX 4090 上完成 RLinf、OpenPI、LIBERO 环境配置。
- [x] 下载并校验 `RLinf/RLinf-Pi05-LIBERO-SFT` 权重。
- [x] 跑通 LIBERO-Spatial task 0 的完整 240-step SFT rollout。
- [ ] 建立 10-trial SFT 基线。
- [ ] 训练少数据 SFT 策略。
- [ ] 对少数据 SFT 策略执行 PPO 在线后训练。
- [ ] 比较 SFT-only 与 SFT + PPO 的成功率和样本效率。

当前单条成功 rollout 仅用于验证链路，**不作为正式 benchmark 结果**。

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

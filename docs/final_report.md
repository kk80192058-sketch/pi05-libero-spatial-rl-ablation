# π0.5 在 LIBERO-Spatial task 6 上的 PPO 在线后训练报告

> 状态：实验进行中。所有“PPO 后”字段必须由训练完成后的实际日志填充；本报告不预填结论。

## 摘要

本项目研究预训练 π0.5 SFT 策略能否通过在线 PPO 提升 LIBERO-Spatial 单任务 task 6（`put both moka pots on the stove`）的成功率。流程包括：环境与 checkpoint 复现、任务筛选、固定协议基线、A800 上的 PPO 链路验证、正式在线训练和前后对比。

## 1. 问题定义

- 输入：两路 256×256 RGB 观测、机器人 proprioception 与自然语言任务指令。
- 输出：7 维连续机器人动作，按 action chunk 执行。
- 环境：LIBERO-Spatial / MuJoCo。
- 任务：task 6，最大 240 environment steps。
- 主指标：`success_once`，即单条轨迹在任意时刻是否达到任务成功状态。

选择 `success_once` 的原因是策略成功后仍会继续执行；继续动作可能扰动已完成的场景，导致 `success_at_end` 偏低，故后者不作为主要效果指标。

## 2. 方法

### 2.1 初始策略

使用公开 checkpoint `RLinf/RLinf-Pi05-LIBERO-SFT`，在 RLinf 的 OpenPI π0.5 接口下运行。该 checkpoint 是策略初始化，而非本项目从零训练得到。

### 2.2 在线 PPO

每个 epoch 包含：

1. rollout worker 在 LIBERO 中收集策略轨迹和环境奖励；
2. 以 GAE 计算 advantage 与 return；
3. actor 使用 clipped PPO objective 更新策略；
4. value head / critic 回归 return；
5. 将更新后的权重同步回 rollout worker。

固定主参数：`gamma=0.99`、`gae_lambda=0.95`、PPO clip ratio `0.2`、actor learning rate `5e-6`、value learning rate `1e-4`。

### 2.3 正式训练配置

| 项目 | 设置 |
| --- | --- |
| GPU | 1 × A800 80GB |
| 训练任务 | LIBERO-Spatial task 6 |
| 并行环境 | 8 |
| 训练 epoch | 10 |
| 评测 / 保存频率 | epoch 5、10 |
| 视频 | 关闭（浏览器端编码兼容问题不参与策略评价） |

## 3. 实验协议

评测固定 task 6 的 trial 0–9、10 条轨迹、每条最多 240 步。训练前后必须使用同一协议，并报告 `success_once`、每 trial 成败和运行 seed。

单次 π0.5 rollout 存在采样随机性，因此最终报告应对 pre-PPO 与 post-PPO 各执行多次 10-trial 评估，并报告均值、最小/最大值；不将单次结果表述为稳定 benchmark 分数。

## 4. 已验证结果

| 阶段 | 条件 | 结果 | 解释 |
| --- | --- | --- | --- |
| 4090 task 筛选 | 10 个 Spatial task 各 trial 0 | 8/10 成功 | task 6、9 出现失败 |
| 4090 baseline | task 6, trial 0–9 | `success_once=0.6` | 选定 PPO 目标 |
| A800 pre-PPO | task 6, trial 0–9 | `success_once=0.8` | 单次随机采样；用于记录方差，不和 0.6 混为确定值 |
| A800 PPO smoke | 2 env × 1 epoch | rollout、反向传播、权重同步均成功 | 不是效果结论 |
| A800 formal PPO | 8 env × 10 epoch | 待训练完成 | 填入 epoch 5、10 checkpoint 与评测 |

## 5. 分析计划

1. 绘制每 epoch 的 rollout success、reward、policy loss、value loss、KL 和 gradient norm。
2. 比较 pre-PPO 与 epoch 5 / epoch 10 的 `success_once` 均值。
3. 对失败 trial 做定性归类：抓取失败、物体定位偏差、放置不稳定、成功后扰动。
4. 说明训练环境、GPU、随机性、任务选择和视频兼容性对结论的限制。

## 6. 复现资产

- task 6 基线：`scripts/run_task6_baseline.sh`
- PPO 启动：`scripts/run_task6_ppo.sh`
- task 6 demonstration 清单：`docs/task6_demo_manifest.csv`
- 实验日志：`docs/experiment_log.md`
- 指标汇总：`results/baseline_summary.csv`

## 7. 结论（训练结束后填写）

本项目的结论应严格限于“公开 π0.5 SFT checkpoint 在固定 LIBERO-Spatial task 6 协议下，经过指定预算的在线 PPO 后，`success_once` 从 ___ 变化到 ___”。不得外推为整个 LIBERO benchmark、π0.5 的通用性能或新算法结论。

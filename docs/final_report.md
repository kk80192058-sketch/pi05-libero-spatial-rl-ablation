# π0.5 在 LIBERO-Spatial task 6 上的 PPO 在线后训练报告

> 状态：主实验已完成。所有结果均来自实际日志；C20 仅完成链路冒烟，未作为性能结果。

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

### 2.0 用一句话理解整套流程

可以把 π0.5 看成已经学过许多桌面操作的“机器人学徒”。SFT 让它观看老师的标准示范并模仿；PPO 则让它在仿真中自己反复尝试，成功得到奖励、失败不加分，再把更有效的动作倾向保留下来。本项目不是从零训练这个学徒，而是检验这两种补课方式在一个具体任务上各自带来多少帮助。

| 术语 | 通俗解释 | 在本项目中的含义 |
| --- | --- | --- |
| VLA | 能看图、懂文字、操控机器人的模型 | π0.5 读取两路相机图像与英文指令，输出机械臂动作 |
| SFT | 跟着标准答案练习的模仿学习 | 用专家演示让模型更熟悉 task 6 |
| PPO | 根据成败反馈的试错式强化学习 | 在 LIBERO 仿真中优化 SFT 之后的策略 |
| rollout | 机器人从任务开始到结束的一次完整尝试 | 至多 240 个环境步的一条轨迹 |
| epoch | 收集一批尝试并更新一次模型参数的一轮 | A1 与 C20 各进行 10 轮在线 PPO |
| checkpoint | 训练过程中的可恢复“存档” | 保存 epoch 5、10 的权重用于固定评测 |

### 2.1 对照组设计

| 组别 | 初始化 / 训练 | 要回答的问题 |
| --- | --- | --- |
| A0 | 公开 π0.5 LIBERO SFT checkpoint，不再训练 | 公开策略在 task 6 的起点表现如何？ |
| A1 | A0 + 10 epoch 在线 PPO | 仅靠在线奖励能否改善？ |
| B20 | A0 + 20 条固定 task 6 示范的 few-shot SFT | 少量专项老师示范能否改善？ |
| C20 | B20 + 10 epoch 在线 PPO | 专项模仿再结合在线 PPO 是否更好？ |

这四组是一个小型但完整的消融实验：A0→A1 隔离 PPO 的作用，A0→B20 隔离 few-shot SFT 的作用，B20→C20 检验两者组合的作用。

### 2.2 初始策略

使用公开 checkpoint `RLinf/RLinf-Pi05-LIBERO-SFT`，在 RLinf 的 OpenPI π0.5 接口下运行。该 checkpoint 是策略初始化，而非本项目从零训练得到。

### 2.3 在线 PPO

每个 epoch 包含：

1. rollout worker 在 LIBERO 中收集策略轨迹和环境奖励；
2. 以 GAE 计算 advantage 与 return；
3. actor 使用 clipped PPO objective 更新策略；
4. value head / critic 回归 return；
5. 将更新后的权重同步回 rollout worker。

固定主参数：`gamma=0.99`、`gae_lambda=0.95`、PPO clip ratio `0.2`、actor learning rate `5e-6`、value learning rate `1e-4`。

### 2.4 正式训练配置

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

### 3.1 few-shot SFT 数据划分

task 6 在公开数据集中有 29 条专家轨迹。为避免训练完成后再根据效果挑数据，使用固定随机种子 42 划分：20 条 train、9 条 holdout。具体 episode index 见 `docs/task6_20_9_split.csv`。holdout 轨迹不参与 SFT 参数更新，仅用于离线数据检查；主线上仍以仿真中的固定 10-trial protocol 评测策略。

### 3.2 B20 的固定训练预算

B20 使用公开 π0.5 LIBERO SFT checkpoint 作为初始化，只更新其动作专家部分。训练数据为固定 20 条 task 6 演示（8,082 帧）；batch size 为 8，学习率为 `5e-6`，总计 500 次更新，并在第 250、500 步保存 checkpoint。这个预算的定位是“小样本专项适配”，不把它包装为完整大规模预训练。

## 4. 已验证结果

| 阶段 | 条件 | 结果 | 解释 |
| --- | --- | --- | --- |
| 4090 task 筛选 | 10 个 Spatial task 各 trial 0 | 8/10 成功 | task 6、9 出现失败 |
| 4090 baseline | task 6, trial 0–9 | `success_once=0.6` | 选定 PPO 目标 |
| A800 pre-PPO | task 6, trial 0–9 | `success_once=0.8` | 单次随机采样；用于记录方差，不和 0.6 混为确定值 |
| A800 PPO smoke | 2 env × 1 epoch | rollout、反向传播、权重同步均成功 | 不是效果结论 |
| A800 formal PPO (A1) | 8 env × 10 epoch | epoch 10: `success_once=1.0`，`success_at_end=0.9` | 正式训练完成，耗时约 46 分 56 秒 |
| A800 few-shot SFT (B20) | 20 条轨迹，500 steps | `success_once=1.0`，`success_at_end=0.9` | checkpoint 重新加载后按同协议评测 |
| A800 C20 PPO smoke | B20 初始化，2 env × 1 epoch | rollout、反向传播、权重同步成功 | 仅验证链路，不作为性能结果 |

## 5. 分析计划

1. 记录每 epoch 的 rollout success、reward、policy loss、value loss、KL 和 gradient norm；A1 的 epoch 5、10 checkpoint 都已保存。
2. 比较公开 SFT、A1 与 B20 的固定 10-trial `success_once`。A1 与 B20 都达到本协议的 1.0 上限，因此不能从单次 10-trial 结果断言二者孰优。
3. 对失败 trial 做定性归类：抓取失败、物体定位偏差、放置不稳定、成功后扰动。
4. 说明训练环境、GPU、随机性、任务选择和视频兼容性对结论的限制。

## 6. 复现资产

- task 6 基线：`scripts/run_task6_baseline.sh`
- PPO 启动：`scripts/run_task6_ppo.sh`
- task 6 demonstration 清单：`docs/task6_demo_manifest.csv`
- 实验日志：`docs/experiment_log.md`
- 指标汇总：`results/baseline_summary.csv`

## 7. 结论

在固定 LIBERO-Spatial task 6、10 个 reset state、每条最多 240 步的单次评测协议下，公开 π0.5 SFT checkpoint 的历史 task-6 基线为 `success_once=0.6`（4090），迁移到 A800 后另一次采样为 0.8，说明该策略存在采样方差。经过 8 个并行环境、10 epoch 的在线 PPO（A1），最终评测为 `success_once=1.0`、`success_at_end=0.9`。这给出了“在该固定单任务协议、该训练预算下，PPO 后策略达到了评测上限”的可复现证据。

以固定 20 条 task 6 专家轨迹做 500-step few-shot SFT（B20）后，重新加载 checkpoint 的同协议评测同样得到 `success_once=1.0`、`success_at_end=0.9`。B20 已达到主指标上限，因此 C20 仅完成了 1 epoch / 2 环境的 PPO 链路验证，未进行不太可能再提升 `success_once` 的 10 epoch 长跑。该决策是成本控制，不是声称 C20 有性能结果。

结论不外推到完整 LIBERO benchmark、其他任务、真实机器人或 π0.5 的通用能力。更严格的比较需要增加独立随机种子、更多 reset state 或改用更难的非饱和任务。

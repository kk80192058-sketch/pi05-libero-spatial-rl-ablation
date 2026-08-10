# task 9 正式 2×2 消融实验计划

## 选择依据

先在公开 π0.5 LIBERO SFT checkpoint 上，用相同的 10-trial / 240-step 协议评测全部 10 个 LIBERO-Spatial task。task 9 在窗口 0–9 的 `success_once=30% (3/10)`，低于其余所有任务；在完全分离的窗口 10–19 上也为 `30% (3/10)`。因此它被预先指定为主实验任务，而不是根据训练结果事后选择。

任务文本：`pick up the black bowl on the wooden cabinet and place it on the plate`。

## 因子与组别

两个因子：20 条专家演示的专项 SFT，及 10 epoch 在线 PPO。

| 组别 | SFT | PPO | 初始化 |
| --- | --- | --- | --- |
| A0 | 否 | 否 | 公开 π0.5 LIBERO SFT checkpoint |
| A1 | 否 | 是 | A0 |
| B20 | 是 | 否 | A0 |
| C20 | 是 | 是 | B20 的最终 SFT checkpoint |

## 固定项

- 20 条 train demonstration，由 task 9 的 44 条可用 demonstration 按 seed 42 固定划分；其余 24 条不参与 SFT。
- SFT：500 steps，batch size 8，learning rate `5e-6`。
- PPO：10 epochs、8 个训练环境，其他参数由同一 RLinf config 提供。
- 每个组别在 reset 0–9 和 10–19 各评测 10 条，每条最多 240 environment steps。
- 主指标 `success_once`；辅助指标 `success_at_end`、reward、episode length。

## 状态记录（截至 2026-08-10）

| 项目 | 状态 | 已知结果 |
| --- | --- | --- |
| 10 task 基线筛查 | 完成 | 见 `results/spatial_seen_diagnostic.csv` |
| A0 task 9，窗口 0–9 | 完成 | `success_once=30% (3/10)` |
| A0 task 9，窗口 10–19 | 完成 | `success_once=30% (3/10)` |
| B20 演示数据与 SFT | 完成 | 44 条可用演示中固定抽取 20 条；500-step SFT 的 seen/unseen 分别为 70% / 80% |
| C20 PPO | 完成 | 从正确 B20 checkpoint 做 10 epoch PPO；seen/unseen 均为 10%，保留为负向消融 |
| A1 PPO | 完成 | 仅 PPO，不依赖演示数据映射 |

没有从未完成的实验推断或填写性能数字。发现综合演示集（40 task）与 LIBERO-Spatial benchmark（10 task）的数值编号不一致后，先前的 B20/C20 运行被标记为无效并已删除；本报告只使用按任务文本正确匹配后重新运行的日志。

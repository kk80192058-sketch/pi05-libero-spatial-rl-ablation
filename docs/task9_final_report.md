# π0.5 在 LIBERO-Spatial task 9 上的 SFT 与 PPO 在线后训练消融

## 一句话结论

本实验在一个基础策略仅有 30% 成功率的 LIBERO-Spatial 任务上，比较原始 π0.5、仅 PPO、仅 20 条专家演示 SFT、以及 SFT 后 PPO。所有最终数字均来自两组互不重叠的 10 条仿真轨迹评测；不会用单条视频或训练过程中的采样值替代正式结果。

## 任务与选择过程

主任务为 LIBERO-Spatial `task_id=9`：

> `pick up the black bowl on the wooden cabinet and place it on the plate`

即：从木柜上抓起黑碗，放到盘子上。公开 π0.5 LIBERO SFT checkpoint 先在 10 个 Spatial task 上按相同 10-trial 协议筛查。task 9 的 `success_once=30% (3/10)`，为最低结果，因此在训练前被确定为正式消融目标，而非根据训练结果事后挑选。

评测使用两组不重叠 reset state：`0–9`（seen window）和 `10–19`（unseen window）；每条轨迹最多 240 个环境步。主指标为 `success_once`，即轨迹任意时刻是否完成任务；`success_at_end` 作为辅助指标，因为成功后继续动作可能扰动已完成状态。

## 数据映射校验

公开 `physical-intelligence/libero` 演示集包含 40 个综合任务，而 LIBERO-Spatial benchmark 仅有 10 个任务；两个数值索引不能直接对应。本项目先从运行中的 LIBERO-Spatial benchmark 读取 `task_id=9` 的语言指令，再在演示集按完全相同的文本匹配数据。正确匹配为：

- benchmark：Spatial task 9；
- public dataset task index：39；
- 可用 demonstration：44 条；
- 固定 seed 42 抽取训练轨迹：20 条（2,861 frames）；
- 余下 24 条：holdout，不参与 SFT。

这个映射由 `scripts/download_libero_spatial_task_demos.py` 固化。曾经因直接把数字 9 用于综合数据集而产生的书→caddy数据、B20/C20 checkpoint、评测和视频都已删除，不进入本报告。

## 实验设计

| 组别 | 初始化 | 专项 SFT | 在线 PPO | 用途 |
| --- | --- | --- | --- | --- |
| A0 | 公开 π0.5 LIBERO SFT | 否 | 否 | 基线 |
| A1 | A0 | 否 | 10 epoch | 单独测量 PPO |
| B20 | A0 | 20 demos，500 steps | 否 | 单独测量 SFT |
| C20 | B20 | 已完成 | 10 epoch | 测量 SFT+PPO |

固定预算：SFT batch size 8、learning rate `5e-6`、500 updates；PPO 8 个训练环境、10 epoch。A1 与 C20 使用相同 PPO 预算。

## 正式结果

| 组别 | 已见 `success_once` | 未见 `success_once` | 已见 `success_at_end` | 未见 `success_at_end` |
| --- | ---: | ---: | ---: | ---: |
| A0 | 30% (3/10) | 30% (3/10) | 20% | 30% |
| A1：仅 PPO | 50% (5/10) | 20% (2/10) | 40% | 20% |
| B20：仅 SFT | 70% (7/10) | 80% (8/10) | 50% | 80% |
| C20：SFT + PPO | 待 C20 完成 | 待 C20 完成 | 待 C20 完成 | 待 C20 完成 |

可复现的原始日志汇总由 `scripts/collect_task9_ablation.py` 生成到 `results/task9_ablation.csv`。

## 当前可得结论

在正确映射的数据和固定评测协议下，20 条严格匹配的专项专家演示带来了最清晰的提升：B20 从 A0 的 30% 提升到已见 70%、未见 80%。A1 的 PPO 则在已见窗口上升到 50%，但未见窗口下降到 20%，提示在线优化可能更贴合训练时遇到的状态，而未必自动提高泛化。C20 完成后才会判断 SFT 初始化是否改变 PPO 的这种行为。

## 限制

- 每个窗口只有 10 条轨迹，结果应写成 `n/10` 并同时给出百分比，不能包装成高置信度的通用 benchmark 结论。
- 只验证一个经过预筛选的困难 Spatial 任务；这不是全 LIBERO benchmark 平均成绩。
- 本项目的价值在于可复现的任务选择、数据映射校验、双窗口评测和消融比较，而不是声称训练了一个通用机器人策略。

# task 9 视频素材说明

任务：`pick up the black bowl on the wooden cabinet and place it on the plate`。

- `comparison_a0_vs_b20_same_reset.mp4`：最推荐观看。左右是**同一个 reset 3**；左侧为 A0 基础 π0.5，未完成任务，右侧为 B20（20 条严格匹配示范 SFT），完成任务。为便于观察，播放速度放慢为原始视频的 1/3。
- `b20_unseen_success_slow.mp4`：B20 在未参与 seen 窗口的 reset 12 成功，放慢为原速度 1/3。

仓库只保留这两段最有信息量的视频，避免把重复的原始编码文件混入面试材料。

视频用于定性展示，不替代正式指标。正式结果来自每组两个独立窗口、各 10 条轨迹的汇总：[task9_ablation.csv](../../../results/task9_ablation.csv)。每次单条 rollout 可能有策略动作随机性，因此视频文件旁记录的是该次实际 rollout 的结果，而不是批量评测的预测标签。

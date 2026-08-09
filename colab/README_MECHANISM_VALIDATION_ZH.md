# GPPO Colab 单种子机制验证

直接把 `GPPO_Mechanism_Validation_Colab.ipynb` 上传到 Google Colab，选择 GPU 后“全部运行”。

默认设置：

- T5-10-48；
- 训练 seed=1；
- 300 iterations；
- rollout/batch=512；
- 六个可学习模型；
- 固定 test100；
- 两个并行训练进程；
- 六条模型进度条、总体进度条、预计剩余时间和实时 reward/makespan/loss；
- test100、基线、通信重放、gate、故障探针与报告阶段进度条；
- 自动尝试从 Drive 中上次 `GPPO_one_click/quick_seed1_100` 续训；
- 自动生成中文报告、审计 JSON、图和 zip。

最终还会生成 `BASELINE_MECHANISM_EVIDENCE.json`，记录异构边类型、动态图状态版本、逐步合法“子任务–无人机”动作、动作掩码违规数、事件同步和通信统计。

输出目录：`/content/drive/MyDrive/GPPO_mechanism_validation/`

本 Notebook 是单种子机制验证，不是论文级统计复现。它只加入旧环境中的一次确定性领导机故障诊断，不加入 Gilbert–Elliott、随机时延、风场或任务取消。

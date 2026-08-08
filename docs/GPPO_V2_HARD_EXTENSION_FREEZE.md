# GPPO-v2 Hard Dynamic Extension 冻结说明

## 定位

`gppo-v2-hard-3` 自 2026-08-07 起仅作为 **Hard Dynamic Extension** 保存，不能再称作论文实验基线，也不能用来判断论文 Literal-AHGNN 的模块贡献。

其 40 个 checkpoint、42 组评估、16,800 条评估记录和已公布的负向消融结果均保留。冻结清单及 SHA-256 位于 `outputs/paper_aligned/gppo_v2_hard/formal/summary/frozen_manifest.json`。

## 必须保留的负向结果

| 方法 | DCR | Makespan |
|---|---:|---:|
| Adaptive GPPO-event | 0.7774 | 20.2619 |
| No-gate | 0.7764 | 20.3470 |
| Single-head | 0.7865 | 20.1115 |

这些结果说明旧实现中的 adaptive gate 没有产生可观察收益；single-head 反而在两个主指标上更好。该矛盾不得通过重命名、覆盖输出或选择性汇报消除。

## 为什么它不构成 Literal-AHGNN 消融

旧 adaptive 分支包含论文公式之外的 relation bias、priority bias、独立通信注意力、残差和 LayerNorm，并用固定斜率 LeakyReLU(0.2) 替代 RReLU。旧 single-head 分支同时更换 UAV 注意力和任务更新结构。因此 single-head 与 adaptive 的比较不是单变量消融。

后续论文忠实复现全部写入独立的 `paper_faithful_*` 模块和输出目录；不得修改本冻结目录下的历史产物。

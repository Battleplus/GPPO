# GPPO 单种子快速机制验证报告

> 范围：T5-10-48、训练 seed=1、100 iterations、固定 test100。该结果仅用于方向筛查，不能证明 GPPO 稳定优于 PPO，也不能替代五种子论文级统计复现。

## 结论

**继续完整五种子正式复现；当前快速验证方向通过**

## 六模型结果

| 方法 | Return mean | Return median | Makespan mean | Makespan median | Completion | Comm bytes |
|---|---:|---:|---:|---:|---:|---:|
| GPPO-event | -6.8085 | -6.7250 | 16.1276 | 16.0044 | 1.0000 | 2792.8 |
| PPO-none | -9.2121 | -9.1396 | 18.5312 | 18.2374 | 1.0000 | 1156.5 |
| PPO-event | -8.2679 | -8.3162 | 17.5870 | 17.4998 | 1.0000 | 2877.9 |
| GPPO-none | -7.0407 | -6.8951 | 16.3599 | 16.3561 | 1.0000 | 1015.7 |
| GPPO-NoGate-event | -8.1611 | -7.9801 | 17.4802 | 17.5366 | 1.0000 | 2872.8 |
| GPPO-SingleHead-event | -7.5470 | -7.3475 | 16.8662 | 16.8631 | 1.0000 | 2835.7 |

## 配对 test100 差值

负 makespan 差值表示左侧方法更好。95% CI 是固定模型在 100 个测试实例上的实例级区间，不是训练种子置信区间。

| 对比 | Mean Δ makespan | Median Δ | 95% CI |
|---|---:|---:|---:|
| GPPO-event_vs_PPO-none | -2.4036 | -2.3106 | [-2.6889, -2.1183] |
| GPPO-event_vs_PPO-event | -1.4594 | -1.6063 | [-1.7517, -1.1672] |
| GPPO-none_vs_PPO-none | -2.1713 | -2.0107 | [-2.4910, -1.8517] |
| GPPO-event_vs_GPPO-none | -0.2323 | -0.0916 | [-0.4140, -0.0506] |
| Adaptive_vs_NoGate | -1.3526 | -1.3575 | [-1.5852, -1.1200] |
| Adaptive_vs_SingleHead | -0.7386 | -0.7077 | [-0.9740, -0.5031] |

## Random / Greedy

- Random makespan mean：20.4073
- Greedy makespan mean：17.3752

## Event / Full 同 checkpoint 重放

- Event makespan：16.1276
- Full makespan：16.1276
- 相对质量差：0.00%
- Event bytes：2792.8
- Full bytes：4290990.2
- 通信减少率：99.93%

## 判读边界

- 这里只能判断单规模、单训练种子、短训练预算下的机制方向。
- test100 的实例级 CI 不能替代至少五个训练种子的 CI。
- 若 adaptive 未同时优于 NoGate 和 SingleHead，应暂停 adaptive 有益的论文式表述。
- 若 GPPO-event 未优于 PPO-none 或 Random，应暂停 PCRL 与世界模型接入，先修 GPPO。

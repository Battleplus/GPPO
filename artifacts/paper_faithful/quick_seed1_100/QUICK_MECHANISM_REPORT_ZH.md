# GPPO 单种子快速机制验证报告

> 范围：T5-10-48、训练 seed=1、100 iterations、固定 test100。该结果仅用于方向筛查，不能证明 GPPO 稳定优于 PPO，也不能替代五种子论文级统计复现。

## 结论

**继续 GPPO 正式复现，但停止宣称 adaptive gate 有益并优先排查该模块**

## 六模型结果

| 方法 | Return mean | Return median | Makespan mean | Makespan median | Completion | Comm bytes |
|---|---:|---:|---:|---:|---:|---:|
| GPPO-event | -6.8185 | -6.8186 | 16.1376 | 15.9449 | 1.0000 | 2788.3 |
| PPO-none | -8.9821 | -8.8825 | 18.3013 | 18.1229 | 1.0000 | 1136.6 |
| PPO-event | -8.4444 | -8.3684 | 17.7635 | 17.7132 | 1.0000 | 2891.4 |
| GPPO-none | -6.6475 | -6.5537 | 15.9666 | 15.8320 | 1.0000 | 988.8 |
| GPPO-NoGate-event | -6.7152 | -6.5263 | 16.0344 | 15.9501 | 1.0000 | 2785.8 |
| GPPO-SingleHead-event | -6.7974 | -6.8343 | 16.1166 | 15.9045 | 1.0000 | 2790.2 |

## 配对 test100 差值

负 makespan 差值表示左侧方法更好。95% CI 是固定模型在 100 个测试实例上的实例级区间，不是训练种子置信区间。

| 对比 | Mean Δ makespan | Median Δ | 95% CI |
|---|---:|---:|---:|
| GPPO-event_vs_PPO-none | -2.1637 | -2.2793 | [-2.4694, -1.8579] |
| GPPO-event_vs_PPO-event | -1.6259 | -1.8963 | [-1.9169, -1.3349] |
| GPPO-none_vs_PPO-none | -2.3346 | -2.4429 | [-2.6121, -2.0572] |
| GPPO-event_vs_GPPO-none | 0.1710 | 0.1155 | [-0.0287, 0.3707] |
| Adaptive_vs_NoGate | 0.1032 | 0.0159 | [-0.1160, 0.3225] |
| Adaptive_vs_SingleHead | 0.0210 | 0.0356 | [-0.1567, 0.1988] |

## Random / Greedy

- Random makespan mean：20.4073
- Greedy makespan mean：17.3752

## Event / Full 同 checkpoint 重放

- Event makespan：16.1376
- Full makespan：16.1376
- 相对质量差：0.00%
- Event bytes：2788.3
- Full bytes：4279957.4
- 通信减少率：99.93%

## 判读边界

- 这里只能判断单规模、单训练种子、短训练预算下的机制方向。
- test100 的实例级 CI 不能替代至少五个训练种子的 CI。
- 若 adaptive 未同时优于 NoGate 和 SingleHead，应暂停 adaptive 有益的论文式表述。
- 若 GPPO-event 未优于 PPO-none 或 Random，应暂停 PCRL 与世界模型接入，先修 GPPO。

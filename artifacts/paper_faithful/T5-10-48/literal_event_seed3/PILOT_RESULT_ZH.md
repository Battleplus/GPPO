# T5-10-48 GPPO-Literal Event 单种子结果

## 实验身份

- 方法：GPPO-Literal + event-triggered communication
- 训练种子：3
- 训练轮数：2000
- 测试集：固定 test split，100 个实例
- checkpoint 选择：validation realized makespan 最小
- 最优 validation realized makespan：15.093991

## Test100 结果

| 指标 | 结果 |
|---|---:|
| realized makespan | 15.308944 |
| 实例标准差 | 1.433970 |
| median makespan | 15.140943 |
| 近似 95% CI | [15.024, 15.594] |
| completion rate | 1.000000 |
| all tasks completed | 1.000000 |
| communication events | 3.000000 |
| communication bytes | 2738.400000 |
| heartbeat messages | 59.400000 |
| mean inference latency | 3.255099 ms |

## Adaptive gate 诊断

| 指标 | 结果 |
|---|---:|
| gate mean | 0.344217 |
| gate median | 0.037406 |
| gate p05 / p95 | 0.000452 / 0.984747 |
| fraction gate < 0.1 | 0.560619 |
| fraction gate > 0.9 | 0.269833 |
| gate gradient L2 | 0.559243 |

gate 不是常数，也没有全部饱和在 1；其梯度非零，说明该分支在当前 checkpoint 中实际参与决策。

## 解释边界

这是一个单规模、单训练种子的完整 GPPO 结果，可证明训练、推理、事件通信和 gate 诊断链路可运行。它不能单独证明 GPPO 优于 PPO，也不能证明 adaptive gate 优于 NoGate 或 SingleHead；这些方向性结论至少需要在相同训练预算下增加对应对照。

本种子曾由 legacy candidate 在第 1800 轮恢复；恢复时未保留旧 Adam、RNG 和 episode offset，相关不连续性已记录在最终 checkpoint 的 `recovery_info` 中。

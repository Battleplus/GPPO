# 候选 Checkpoint 重新评估

> 只使用 validation100-A 选择 checkpoint；validation100-B 和原 test100 仅复核。

## 每轮结果

| 模型 | 轮次 | validation-A | validation-B | test100 | completion A/B/test |
|---|---:|---:|---:|---:|---:|
| Adaptive | 50 | 18.2275 | 18.6492 | 18.4886 | 1.000 / 1.000 / 1.000 |
| Adaptive | 100 | 16.1250 | 16.1420 | 16.1276 | 1.000 / 1.000 / 1.000 |
| Adaptive | 150 | 16.1268 | 16.1872 | 16.0298 | 1.000 / 1.000 / 1.000 |
| Adaptive | 200 | 16.1057 | 16.3439 | 16.1225 | 1.000 / 1.000 / 1.000 |
| Adaptive | 250 | 16.1292 | 16.0882 | 16.1673 | 1.000 / 1.000 / 1.000 |
| Adaptive | 300 | 15.9429 | 16.0585 | 15.8673 | 1.000 / 1.000 / 1.000 |
| NoGate | 50 | 18.2699 | 18.5186 | 18.3401 | 1.000 / 1.000 / 1.000 |
| NoGate | 100 | 17.2670 | 17.7874 | 17.4802 | 1.000 / 1.000 / 1.000 |
| NoGate | 150 | 15.9540 | 16.1454 | 15.8279 | 1.000 / 1.000 / 1.000 |
| NoGate | 200 | 16.1499 | 16.0438 | 15.8190 | 1.000 / 1.000 / 1.000 |
| NoGate | 250 | 16.1075 | 16.1081 | 15.9732 | 1.000 / 1.000 / 1.000 |
| NoGate | 300 | 15.9670 | 15.9562 | 15.9012 | 1.000 / 1.000 / 1.000 |
| SingleHead | 50 | 17.9536 | 18.2606 | 18.1489 | 1.000 / 1.000 / 1.000 |
| SingleHead | 100 | 17.0265 | 17.1371 | 16.8713 | 1.000 / 1.000 / 1.000 |
| SingleHead | 150 | 16.1605 | 16.2465 | 16.0999 | 1.000 / 1.000 / 1.000 |
| SingleHead | 200 | 16.0324 | 16.1992 | 15.9389 | 1.000 / 1.000 / 1.000 |
| SingleHead | 250 | 15.9796 | 15.9817 | 15.8285 | 1.000 / 1.000 / 1.000 |
| SingleHead | 300 | 15.9798 | 16.0086 | 15.8112 | 1.000 / 1.000 / 1.000 |

## validation-A 选择结果

- Adaptive：第 300 轮；validation-A / B / test100 makespan = 15.9429 / 16.0585 / 15.8673。冻结 checkpoint 实际保存第 250 轮权重（逐张量一致：True）。
- NoGate：第 150 轮；validation-A / B / test100 makespan = 15.9540 / 16.1454 / 15.8279。冻结 checkpoint 实际保存第 250 轮权重（逐张量一致：True）。
- SingleHead：第 250 轮；validation-A / B / test100 makespan = 15.9796 / 15.9817 / 15.8285。冻结 checkpoint 实际保存第 200 轮权重（逐张量一致：True）。

## Adaptive 配对差值

正值表示 Adaptive 更差；区间为实例级 95% CI。

| 轮次 | 数据库 | Adaptive-NoGate | 95% CI | Adaptive-SingleHead | 95% CI |
|---:|---|---:|---:|---:|---:|
| 50 | validation_a | -0.0424 | [-0.3627, 0.2779] | 0.2739 | [-0.0128, 0.5606] |
| 50 | validation_b | 0.1306 | [-0.1356, 0.3968] | 0.3886 | [0.1297, 0.6475] |
| 50 | test | 0.1485 | [-0.0897, 0.3866] | 0.3396 | [0.0942, 0.5851] |
| 100 | validation_a | -1.1420 | [-1.3842, -0.8999] | -0.9015 | [-1.1205, -0.6825] |
| 100 | validation_b | -1.6453 | [-1.9062, -1.3844] | -0.9950 | [-1.2471, -0.7430] |
| 100 | test | -1.3526 | [-1.5852, -1.1201] | -0.7437 | [-0.9780, -0.5094] |
| 150 | validation_a | 0.1728 | [-0.0049, 0.3505] | -0.0337 | [-0.2139, 0.1466] |
| 150 | validation_b | 0.0418 | [-0.1428, 0.2264] | -0.0593 | [-0.2501, 0.1315] |
| 150 | test | 0.2020 | [0.0301, 0.3739] | -0.0701 | [-0.2119, 0.0718] |
| 200 | validation_a | -0.0442 | [-0.2462, 0.1577] | 0.0733 | [-0.1194, 0.2659] |
| 200 | validation_b | 0.3000 | [0.0718, 0.5283] | 0.1447 | [-0.0989, 0.3884] |
| 200 | test | 0.3035 | [0.1034, 0.5037] | 0.1836 | [-0.0401, 0.4072] |
| 250 | validation_a | 0.0217 | [-0.1831, 0.2265] | 0.1496 | [-0.0480, 0.3471] |
| 250 | validation_b | -0.0199 | [-0.2167, 0.1769] | 0.1065 | [-0.1084, 0.3214] |
| 250 | test | 0.1942 | [0.0061, 0.3823] | 0.3388 | [0.1154, 0.5622] |
| 300 | validation_a | -0.0241 | [-0.2039, 0.1557] | -0.0370 | [-0.2159, 0.1420] |
| 300 | validation_b | 0.1023 | [-0.0912, 0.2957] | 0.0499 | [-0.1568, 0.2566] |
| 300 | test | -0.0340 | [-0.2093, 0.1413] | 0.0560 | [-0.1038, 0.2159] |

## 排序一致性

| 轮次 | validation-A排序 | validation-B排序 | test100排序 | A-test Spearman | B-test Spearman |
|---:|---|---|---|---:|---:|
| 50 | SingleHead < Adaptive < NoGate | SingleHead < NoGate < Adaptive | SingleHead < NoGate < Adaptive | 0.500 | 1.000 |
| 100 | Adaptive < SingleHead < NoGate | Adaptive < SingleHead < NoGate | Adaptive < SingleHead < NoGate | 1.000 | 1.000 |
| 150 | NoGate < Adaptive < SingleHead | NoGate < Adaptive < SingleHead | NoGate < Adaptive < SingleHead | 1.000 | 1.000 |
| 200 | SingleHead < Adaptive < NoGate | NoGate < SingleHead < Adaptive | NoGate < SingleHead < Adaptive | -0.500 | 1.000 |
| 250 | SingleHead < NoGate < Adaptive | SingleHead < Adaptive < NoGate | SingleHead < NoGate < Adaptive | 1.000 | 0.500 |
| 300 | Adaptive < NoGate < SingleHead | NoGate < SingleHead < Adaptive | SingleHead < Adaptive < NoGate | -0.500 | -0.500 |

## 解释边界

- test100 未参与 checkpoint 选择。
- 当前仍是单训练种子诊断，不能用实例级区间替代训练种子级区间。
- 若 validation-A 选择与 validation-B/test100 排名反转，应视为 checkpoint 选择不稳定，而不是从 test100 重新挑选。
- 冻结的 300 轮 Adaptive 负结果保持不变；本报告只判断候选轮次和小验证集选模是否影响结论。

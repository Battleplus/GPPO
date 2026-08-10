# Gate 三训练种子筛选报告

训练代码提交：`324995843dbc7f41d49d207006dbe5a0accbb32d`。协议哈希：`0b863bf37045fa3287268c6f7695c439c529d4ca075032df3c04ce2d87f68f13`。

checkpoint 和候选结构只由 validation-A 选择；validation-B 与 test100 仅用于独立确认或否证，未用于重新挑选。

| 变体 | validation-A makespan | 验证集入选 | 完整规则通过 |
|---|---:|---:|---:|
| Adaptive-current | 15.9069 | 是 | 是 |
| Adaptive-bias2 | 15.9310 | 是 | 是 |
| Adaptive-warmup | 15.9872 | 否/不适用 | 否/不适用 |
| Adaptive-score | 15.9025 | 是 | 是 |
| Adaptive-softplus | 15.9087 | 是 | 是 |
| NoGate | 15.9805 | 否/不适用 | 否/不适用 |
| SingleHead | 15.9613 | 否/不适用 | 否/不适用 |

## 冻结判定

Adaptive 晋级候选：`Adaptive-score`。工程基线仍保留 `SingleHead` 对照。

详细种子差值、Student-t 95% CI 与排序反转审计见 `outputs/gate_screening/seed_level_comparisons.json`。

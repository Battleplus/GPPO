# GPPO-v2 正式验收标准

本文件在查看正式学习方法结果前冻结，适用于 `gppo-v2-hard-3` 协议。它定义的是
“是否可以进入 PCRL/世界模型阶段”的工程验收门槛，不声称可以得到原论文的数值级复现。

## 1. 协议完整性

- 8 个 learned method、5 个训练种子，共 40 个 checkpoint。
- 每个 checkpoint 的 `training_history` 恰好 100 行，`update=1..100`；
  `validation_history` 恰好 10 个节点，`update=10,20,...,100`。
- 40 个 learned evaluation 加 2 个固定 baseline evaluation，共 42 个
  `evaluation.json` 和 42 个非空、逐行可解析的 `event_log.jsonl`。
- 每份 evaluation 包含 4 个规模、每个规模 100 个 episode，评估种子为
  `50000..50099`，共 400 行；所有 manifest 元数据、场景/评估哈希、实现哈希、
  超参数和严格模型加载均匹配。
- 汇总中的 `protocol_validation.valid` 必须为 `true`；协议不完整时不得宣称性能结论。

## 2. 主性能指标

`makespan`（越低越好）与 `deadline_completion_rate`（越高越好）是双主指标。
`deadline_remaining_tasks` 和 `throughput` 是支持指标；`mission_success` 可能出现
地板效应，`completion_rate` 可能出现天花板效应，只作补充。通信、无效动作和重分配
指标用于解释质量-成本折衷，不能单独作为性能优越证据；`return` 仅作训练诊断。

### GPPO-event 相对 Random

- 四个规模的宏平均 DCR 差值（GPPO-event 减 Random）95% Student-t CI 下界大于 0，
  且宏平均 makespan 差值 95% CI 上界小于 0。
- 至少 3/4 个规模在两个主指标上方向一致改善；没有一个规模出现 CI 明确反向。

### GPPO-event 接近 Greedy

- 宏平均 DCR 不低于 Greedy 超过 0.03，宏平均 makespan 不高于 Greedy 超过 5%。
- 同时报告 gap-closure：DCR 使用
  `(GPPO - Random) / (Greedy - Random)`，makespan 使用相应的反向差值；目标值为至少
  `0.8`。若分母接近 0，必须报告不可判定而不是省略。

## 3. 稳定性与机制证据

- 至少 4/5 个训练种子的宏平均同时优于 Random；不得有种子相对 Random 出现
  DCR 低于 0.05 或 makespan 高于 10% 的灾难性退化。
- 报告每个种子的均值、95% CI、`best_update` 和验证曲线后段；checkpoint 是验证节点
  选择结果，不等同于 `update=100` 的 final checkpoint。
- `gppo_event` 相对 `ppo_event`（图结构）以及相对 `gppo_none`（事件同步）应至少在
  宏平均主指标上方向正确，否则只能报告“超过随机但机制证据不足”。
- `event` 的通信次数应低于 `always`，并在任务质量上接近；`periodic`/`always` 是
  通信折衷或上界参照，不作为简单胜负结论。
- `single_head` 与 `no_gate` 作为结构消融报告；single-head 的有效容量差异意味着
  不能把该比较解释为 gate 的纯因果效应。

## 4. 统计口径

Learned 方法先在每个训练种子内汇总 100 个 evaluation episode，再以 5 个训练种子为
重复单位计算 Student-t 95% CI。Random/Greedy 的 CI 反映 episode 场景波动，不包含
训练方差。预注册差值使用共同评估种子，报告 paired delta、CI、exact sign-flip p 值和
Holm 校正；`n=5` 时双侧 exact sign-flip 的最小 p 值为 0.0625，因此不得写成
`p<0.05` 的显著性结论。所有 negative/inconclusive 结果必须保留。

任一硬门槛失败时，结论为“GPPO 方法结构原型尚未达到正式基准标准”，保留全部结果并
诊断下一轮 GPPO；在此之前不进入 PCRL 或世界模型集成。

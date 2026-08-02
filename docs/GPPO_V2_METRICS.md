# GPPO-v2 困难场景与指标定义

## 冻结场景

正式配置位于 `configs/gppo_v2_hard.json`。每个 episode 的 deadline 在实例生成前固定，同一规模、同一 eval seed 对所有方法完全相同。

| 规模 | Mission deadline | 校准目的 |
| --- | ---: | --- |
| `2x12` | 14 | 小规模仍不能被随机策略稳定完成 |
| `3x16` | 13 | 增加资源竞争与前序约束影响 |
| `3x20` | 18 | 保留成功可能，同时避免完成率饱和 |
| `4x24` | 16 | 大规模高负载与动态事件场景 |

其他固定项：任务链长度 5、workload scale 1.8、time-periodic interval 2.0；weather/failure/task-change/communication-drop 的连续时间 hazard rate 分别为 0.06/0.02/0.04/0.04。环境将下一任务完成、外生事件、heartbeat、periodic sync 和 deadline 作为竞争时间边界，因此外生事件可在长任务执行中途发生。场景在查看学习方法正式结果前，仅用随机与贪心固定种子校准。

## 正式指标

| 指标 | 定义 | 方向 |
| --- | --- | --- |
| `makespan` | 当前状态下所有 active task 的最大预计完成时间 | 越低越好 |
| `deadline_completion_rate` | deadline 前完成数 / deadline 前到达任务数 | 越高越好 |
| `mission_success` | 所有 deadline-eligible 任务均在 deadline 前完成 | 越高越好 |
| `deadline_remaining_tasks` | deadline-eligible 数 - deadline 前完成数 | 越低越好 |
| `throughput` | deadline 前完成数 / deadline | 越高越好 |
| `invalid_actions` | belief 合法但 truth 非法，或本身非法的动作次数 | 越低越好 |
| `communication_events` | 实际同步次数；跨多个 periodic interval 分别计数 | 越低越好，但需与质量共同解释 |
| `heartbeat_messages` | 成功传递的固定间隔 heartbeat 消息数；独立于状态同步次数 | 越低越好，但需与失效发现能力共同解释 |
| `reallocation_success_rate` | 重分配后最终完成数 / 重分配尝试数 | 越高越好 |

同时保留 `completion_rate` 作为最终完成率。环境默认在 deadline 后继续执行到完成、失效、stalled 或 max decisions，因此它与 deadline completion rate 不混用。`stop_at_deadline=True` 仅用于需要硬截止终止的附加实验。

## 统计协议

- 六个核心学习配置均使用 seeds 1-5；single-head 补充消融同样使用五个种子。
- 每个 checkpoint 在共同的固定 eval seeds 上评估。
- 学习方法先在每个训练种子内汇总 episode，再以五个训练种子作为重复单位报告 Student-t 95% CI。
- 无训练的 random/greedy 以 episode 反映环境随机性；该 CI 不包含训练方差。
- 重分配成功率使用 `sum(successes) / sum(attempts)`，不平均无重分配 episode 的零值。
- 预注册方法对比使用 paired seed delta、95% CI、精确 sign-flip p-value，并对同一指标的多重比较执行 Holm 校正。
- 五个训练种子的双侧 exact sign-flip 检验最小原始 p 值为 `2/32 = 0.0625`，Holm 校正后只会更大；因此该检验在当前协议中是描述性方向证据，不可能产生 `p < 0.05` 的显著结论。
- 所有 `negative` 和 `inconclusive` 结果进入 `negative_results`，不得从报告中删除。

## 非饱和验收

`tests/test_gppo_v2.py::test_hard_scenario_random_policy_does_not_saturate_deadline` 固定随机策略和独立种子，要求平均 deadline completion rate `< 0.8`（测试使用更紧的 8.0 deadline 压力配置）。正式 manifest 的 smoke gate 要求每个评估规模的随机 deadline completion rate `< 0.95`。

# GPPO 单种子机制验证报告

> 范围：T5-10-48, training seed=1, 300 iterations, fixed test100。仅用于机制方向筛查，不能替代五训练种子的论文级统计复现。

## 结论

**机制验证未通过；暂停 PCRL/世界模型接入，按失败检查项修正 GPPO**

## 六模型 test100

| 方法 | Return mean | Makespan mean | Makespan median | Completion | Comm bytes |
|---|---:|---:|---:|---:|---:|
| GPPO-event | -6.8482 | 16.1673 | 16.0369 | 1.0000 | 2790.9 |
| PPO-none | -7.8528 | 17.1720 | 17.1472 | 1.0000 | 1069.4 |
| PPO-event | -7.7099 | 17.0290 | 16.7780 | 1.0000 | 2844.6 |
| GPPO-none | -6.7068 | 16.0260 | 15.9242 | 1.0000 | 992.6 |
| GPPO-NoGate-event | -6.6540 | 15.9732 | 15.8206 | 1.0000 | 2776.8 |
| GPPO-SingleHead-event | -6.6198 | 15.9389 | 15.8049 | 1.0000 | 2776.8 |

## 配对 test100 差值

负 makespan 差值表示左侧方法更好；该区间是实例级区间，不是训练种子置信区间。

| 对比 | Mean Δ | Median Δ | 95% CI |
|---|---:|---:|---:|
| GPPO-event_vs_PPO-none | -1.0046 | -0.9874 | [-1.3246, -0.6847] |
| GPPO-event_vs_PPO-event | -0.8617 | -0.7740 | [-1.1529, -0.5706] |
| GPPO-none_vs_PPO-none | -1.1460 | -1.1266 | [-1.4273, -0.8647] |
| GPPO-event_vs_GPPO-none | 0.1413 | 0.0453 | [-0.0388, 0.3215] |
| Adaptive_vs_NoGate | 0.1942 | 0.1362 | [0.0061, 0.3823] |
| Adaptive_vs_SingleHead | 0.2284 | 0.0896 | [0.0317, 0.4251] |

## Random / Greedy

- Random makespan：20.4073
- Greedy makespan：17.3752

## Event / Full 同 checkpoint 重放

- Event / Full makespan：16.1673 / 16.1673
- 质量差：0.00%
- Event / Full bytes：2790.9 / 4266481.0
- 通信减少率：99.93%

## Adaptive gate 诊断

- gate mean/std：0.351699 / 0.329883
- policy sensitivity gradient L2：0.0435161
- PPO probe gradient L2：0.10391

## 旧环境领导机故障探针

- 故障/重选/任务释放/任务恢复覆盖率：100.00% / 100.00% / 100.00% / 100.00%
- clean event / failure event / failure full makespan：21.4285 / 29.2134 / 29.2134
- failure-event completion：100.00%

## 基线系统逐步机制证据

- 异构关系累计计数：{'self': 4081, 'uav_communication': 1540, 'task_predecessor': 2926, 'uav_task_capability': 18480, 'task_successor': 2926}
- 图状态版本数：77
- 合法子任务–无人机配对：48，mask 违规：0，环境非法动作：0
- 事件决策/同步：3 / 3；非事件错误同步：0
- 全部子任务完成：True

## 验收检查

- PASS — formula_and_mechanism_tests_passed
- PASS — training_protocol_exact
- PASS — same_fixed_test100
- PASS — same_fixed_event_tapes
- PASS — gppo_event_better_than_ppo_none
- PASS — graph_benefit_under_event
- PASS — graph_benefit_under_none
- FAIL — adaptive_better_than_nogate
- FAIL — adaptive_better_than_singlehead
- PASS — gppo_better_than_random
- PASS — gppo_within_5_percent_of_greedy
- PASS — event_full_same_checkpoint
- PASS — event_full_same_tapes
- PASS — event_within_5_percent_of_full
- PASS — event_uses_fewer_bytes
- PASS — gate_not_saturated_constant
- PASS — gate_has_policy_gradient
- PASS — gate_has_ppo_probe_gradient
- PASS — baseline_heterogeneous_relations_present
- PASS — baseline_graph_state_changes_over_time
- PASS — baseline_literal_ahgnn_checkpoint_loaded
- PASS — baseline_all_selected_actions_pass_mask
- PASS — baseline_environment_received_no_invalid_actions
- PASS — baseline_stepwise_subtask_uav_pairs_selected
- PASS — baseline_all_subtasks_completed
- PASS — baseline_event_tape_was_exercised
- PASS — baseline_every_event_triggered_sync
- PASS — baseline_non_events_did_not_trigger_state_sync
- PASS — baseline_event_sync_is_sparser_than_decisions

### 故障探针

- PASS — failure_injected_all_instances
- PASS — leader_elected_all_instances
- PASS — running_task_released_all_instances
- PASS — released_task_recovered_all_instances
- PASS — failure_event_completion_at_least_95_percent

## 结论边界

- 已验证的是公式实现、机制消融和固定实例上的行为，不是原论文数值级复现。
- 单个训练 seed 无法证明 GPPO 稳定优于 PPO/Greedy；正式结论仍需至少五个训练 seed。
- 领导机故障属于旧工程环境诊断，未混入论文主 test100 排名。

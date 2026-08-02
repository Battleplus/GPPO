# GPPO-v2 难度校准记录

## 目的

在查看任何正式学习方法结果之前，只使用 Random 和 Greedy 固定困难场景，避免“所有方法最终完成全部任务”使完成率失去区分度。校准不用于选择 GPPO 超参数。

## 最终校准

评估种子为 `50000..50029`，每个规模 30 episodes。数值为 episode 均值。

| 方法 | 规模 | Deadline | Deadline completion | Mission success | Final completion |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | `2x12` | 14 | 0.703 | 0.067 | 0.870 |
| Greedy | `2x12` | 14 | 0.731 | 0.067 | 0.876 |
| Random | `3x16` | 13 | 0.721 | 0.000 | 0.984 |
| Greedy | `3x16` | 13 | 0.761 | 0.000 | 0.988 |
| Random | `3x20` | 18 | 0.771 | 0.067 | 0.962 |
| Greedy | `3x20` | 18 | 0.837 | 0.100 | 0.963 |
| Random | `4x24` | 16 | 0.724 | 0.000 | 0.953 |
| Greedy | `4x24` | 16 | 0.789 | 0.000 | 0.958 |

原始结果：

- `outputs/paper_aligned/gppo_v2_hard/calibration_eval_range/random_event/evaluation.json`
- `outputs/paper_aligned/gppo_v2_hard/calibration_eval_range/greedy_event/evaluation.json`

## 判定

- Random 的 deadline completion 为 0.703-0.771，四个规模均明显低于饱和门槛。
- Greedy 的 deadline completion 在四个规模均高于 Random，改善为 0.028-0.066，说明场景保留了可利用的调度结构。
- `3x16` 和 `4x24` 的 30 个 episode 中 Random/Greedy mission success 均为 0；该严格全任务指标可能出现地板效应，必须与连续的 deadline completion 和 remaining tasks 联合解释。
- Final completion 在 `3x16`、`3x20`、`4x24` 仍接近 1.0，但 deadline completion 仅为 0.721-0.837，证明两者不能混用。
- 本轮校准在 stale-cache 死锁、能力图全连接、heartbeat freshness、连续事件时钟和 AHGNN 通道修正后重新运行；旧 `hard-2` 数据已归档，不用于 `hard-3` 结论。

## Smoke Gate

最终 smoke 使用 `eval_seed=50000..50009`，Random 在 `2x12` 和 `4x24` 的 deadline completion 分别为 0.713 和 0.808，mission success 分别为 0.10 和 0.00，通过非饱和门槛。十个方法、8 个 learned checkpoint、10 份 evaluation 和 10 份 event log 均通过派生 smoke manifest 的严格协议验证。证据位于：

- `outputs/paper_aligned/gppo_v2_hard/smoke/random_nonsaturation.json`
- `outputs/paper_aligned/gppo_v2_hard/smoke/summary/summary.json`
- `outputs/paper_aligned/gppo_v2_hard/smoke/summary/validation_curves.png`

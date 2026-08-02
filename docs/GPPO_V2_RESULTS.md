# GPPO-v2 第一阶段正式验收报告

## 验收结论

**GPPO-v2 通过正式工程基线验收。** 在冻结的 `gppo-v2-hard-3` 协议下，
`gppo_event` 稳定优于 Random，整体接近 Greedy，并以约一半的同步通信达到接近
Always/Oracle 的任务质量。5 个训练种子均同时在 deadline completion 和 makespan 上
优于 Random，没有单种子灾难性退化。

该结论只表示当前实现可作为后续算法改造的稳定工程基线，不表示论文数值级复现，也不
证明自适应 gate 优于更简单的图注意力结构。相反，single-head 补充消融在宏平均
deadline completion 上明确更好；no-gate 与 adaptive 的主指标差异不明确。本任务止于
GPPO-v2 验收，没有开展 PCRL 或世界模型开发。

## 冻结协议与复现边界

- Manifest：`configs/gppo_v2_hard.json`
- 版本：`gppo-v2-hard-3`
- Implementation hash：
  `a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6`
- Training scenario hash：
  `707d4046a6254d7f32a051d29b4fc29b557ecd94a6467fd6f9fa19bb131ff4ec`
- Evaluation scenario hash：
  `002abcff01a57257e6e66a73d70a7bf70e06ad6cb2ea3f878efe5fe3136f4ee7`
- 训练：8 个 learned methods × 5 seeds，100 updates，18 episodes/update。
- 验证：每 10 updates、每个训练规模 20 episodes，固定 validation seed 40000。
- 评估：4 个规模，每规模 100 episodes，固定 seeds `50000..50099`。

Yu et al. 未公开训练代码、环境、任务实例生成器和完整通信仿真细节。因此本工作是依据
公开公式和正文的机制级独立复现，不是论文表格的数值级复现。原论文主要场景为
`T5-10-48`、`T10-10-53`、`T15-8-66`、`T20-10-92`，正文 generalization 段又出现
`T15-08-62`，原文本身存在编号不一致；本报告不自行假定两者等价。

### 公式—模块—代码—测试对齐结论

| 论文机制 | 当前模块 | 代码 | 验证结论 |
| --- | --- | --- | --- |
| UAV/task 异构节点、能力边、前驱/后继边 | 动态异构图 observation | `paper_env.py::_build_observation` | 稀疏能力图、padding、前序解锁测试通过 |
| Eq. (1)-(3) UAV 第一阶段更新 | adaptive/single-head UAV attention | `paper_models.py::AdaptiveUAVAttention` | self/task 独立 value、边特征、优先级、attention 域测试通过 |
| Eq. (4)-(5) task 第二阶段更新 | predecessor/successor/UAV/self 多路聚合 | `paper_models.py::AdaptiveTaskUpdate` | 关系路径和端到端输出测试通过 |
| 动态 `(subtask, UAV)` 配对与约束 | 固定容量 pair 编码、noop、动态 mask | `paper_env.py::_valid_mask_for` | 前序、能力、忙闲、stale belief mask 测试通过 |
| Eq. (7)-(16) 时间与 makespan | 飞行时间、执行时间、未分配任务估计、max completion | `paper_env.py::_execution_components`, `makespan` | 可手算时间、异步完成、未分配估计测试通过 |
| Eq. (17) 主奖励 | `R_t = M_(t-1) - M_t` | `paper_env.py::step` | 正式配置关闭工程奖励，差分奖励测试通过 |
| heartbeat、leader 接任、事件同步 | heartbeat、选举、结构化事件库 | `paper_env.py::_process_heartbeats`, `_sample_event` | freshness、timeout、失效、重分配、事件日志测试通过 |
| 弱通信 | belief cache、partial sync、stale conflict | `paper_env.py::_synchronize_belief`, `step` | 部分同步、冲突反馈和零时间死锁回归测试通过 |

完整证据边界见 `docs/GPPO_V2_ALIGNMENT.md`。需要特别限定：Eq. (3) 的
`f_{i,j,k}` 作用域和注意力归一化域在论文中并不明确；当前 task-only sigmoid gate、
self/task 联合 softmax、relation bias、输出投影、残差和 LayerNorm 都是预先冻结的兼容
实现选择。belief cache、timeout、action conflict、连续时间 hazard、天气、mission
deadline、padding/noop、`none/periodic/always` 则是工程扩展。

## 正式产物审计

| 项目 | 预期 | 实际 | 状态 |
| --- | ---: | ---: | --- |
| Learned checkpoints | 40 | 40 | 通过 |
| Training histories | 40 × 100 updates | 全部连续 `1..100` | 通过 |
| Validation histories | 40 × 10 节点 | 全部为 `10,20,...,100` | 通过 |
| Evaluation files | 42 | 42 | 通过 |
| Evaluation rows | 16800 | 16800 | 通过 |
| Event logs | 42 个非空 JSONL | 42 个，共 935138 条事件 | 通过 |
| 协议、哈希、严格模型加载 | 全部匹配 | 0 错误 | 通过 |
| `protocol_validation.valid` | `true` | `true` | 通过 |
| 训练/验证曲线 | CSV + PNG | 均已生成并人工检查 | 通过 |
| 单元测试 | 全部通过 | 92 passed | 通过 |

`validation_curves.png` 与 `training_curves.png` 已人工检查：8 个 learned methods 均出现，
坐标轴、置信带和图例可读，无裁切或空图。Anaconda 环境会自动加载 Dash/Jupyter pytest
插件并报错，正式测试使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 禁用无关第三方插件。

## 八个核心方法宏平均

以下数值先在各规模汇总，再对四个评估规模作等权宏平均。无训练基线的 CI 反映 episode
场景波动；learned 方法的正式 CI 以 5 个训练种子为重复单位，不能把 500 个 episode
视为独立训练重复。

| 方法 | DCR ↑ | Makespan ↓ | Remaining ↓ | Throughput ↑ | Invalid ↓ | Comm. ↓ | Heartbeat ↓ | Realloc success ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random-event | 0.7246 | 21.9621 | 4.995 | 0.8741 | 0.350 | 21.858 | 29.375 | 0.472 |
| Greedy-event | 0.7807 | 20.4467 | 3.938 | 0.9427 | 0.285 | 21.663 | 27.790 | 0.444 |
| PPO-none | 0.7303 | 80.9501 | 4.882 | 0.8828 | 77.065 | 0.000 | 34.371 | 0.000 |
| PPO-event | 0.7749 | 20.5344 | 4.044 | 0.9368 | 0.286 | 21.678 | 27.838 | 0.432 |
| GPPO-none | 0.7472 | 79.6047 | 4.578 | 0.9030 | 75.865 | 0.000 | 33.924 | 0.000 |
| **GPPO-event** | **0.7774** | **20.2619** | **4.010** | **0.9391** | **0.315** | **21.650** | **27.613** | **0.430** |
| GPPO-periodic | 0.7777 | 21.6121 | 3.987 | 0.9405 | 4.446 | 9.835 | 27.655 | 0.448 |
| GPPO-always | 0.7789 | 20.1806 | 3.978 | 0.9410 | 0.000 | 42.831 | 27.582 | 0.431 |

`none` 在 stale cache 下出现大量 truth-conflict 动作，因此 makespan 和 invalid actions
显著恶化；它是零同步压力基线，不是可部署策略。Periodic 大幅节省同步次数，但仍出现
更多 stale conflict；Event 以更多通信换取明显更低的 makespan。

## 四规模主结果

括号为 95% CI。Random/Greedy 的重复单位是 episode；PPO/GPPO 的重复单位是训练种子。

| 规模 | 方法 | DCR ↑ | Makespan ↓ |
| --- | --- | ---: | ---: |
| `2x12` | Random | 0.7075 [0.6673, 0.7476] | 19.077 [18.169, 19.984] |
|  | Greedy | 0.7603 [0.7218, 0.7988] | 17.576 [16.859, 18.293] |
|  | PPO-event | 0.7519 [0.7434, 0.7603] | 17.839 [17.565, 18.112] |
|  | GPPO-event | **0.7639 [0.7582, 0.7697]** | **17.359 [17.106, 17.611]** |
| `3x16` | Random | 0.6848 [0.6470, 0.7226] | 20.268 [19.260, 21.276] |
|  | Greedy | 0.7307 [0.6959, 0.7654] | 19.090 [18.270, 19.911] |
|  | PPO-event | **0.7352 [0.7166, 0.7539]** | **18.925 [18.400, 19.450]** |
|  | GPPO-event | 0.7312 [0.7205, 0.7419] | 18.936 [18.703, 19.168] |
| `3x20` | Random | 0.7611 [0.7235, 0.7987] | 24.403 [23.284, 25.522] |
|  | Greedy | **0.8206 [0.7882, 0.8530]** | 23.143 [22.090, 24.197] |
|  | PPO-event | 0.8015 [0.7856, 0.8174] | 23.448 [23.068, 23.828] |
|  | GPPO-event | 0.7989 [0.7929, 0.8049] | **23.195 [23.016, 23.375]** |
| `4x24` | Random | 0.7450 [0.7118, 0.7782] | 24.101 [23.118, 25.083] |
|  | Greedy | 0.8112 [0.7826, 0.8399] | 21.977 [21.148, 22.807] |
|  | PPO-event | 0.8110 [0.8038, 0.8182] | 21.926 [21.707, 22.145] |
|  | GPPO-event | **0.8155 [0.8090, 0.8220]** | **21.558 [21.214, 21.901]** |

## 预注册比较与验收门槛

| 门槛 | 正式结果 | 判定 |
| --- | --- | --- |
| GPPO-event 明显优于 Random | 宏平均 DCR 差 `+0.05278`，CI `[0.04795, 0.05760]`；makespan 差 `-1.7001`，CI `[-1.8141, -1.5862]`；4/4 规模双主指标均改善 | 通过 |
| 至少接近 Greedy | DCR 仅低 `0.00333`，gap-closure `0.941`；makespan 反而低 `0.1848`，gap-closure `1.122` | 通过 |
| 不依赖单个种子 | 5/5 seeds 的宏平均 DCR 与 makespan 同时优于 Random；无灾难性 seed | 通过 |
| 图结构方向正确 | 相对 PPO-event：DCR `+0.00248`，CI 跨 0；makespan `-0.2725`，CI `[-0.3830, -0.1620]` | 方向通过，增益主要体现在 makespan |
| 事件同步方向正确 | 相对 GPPO-none：DCR `+0.03015`，CI `[0.01715, 0.04316]`；makespan `-59.34`，CI `[-62.26, -56.43]` | 通过 |
| Event/Always 通信折衷 | Event 少 `21.18` 次通信，约 `49.5%`；DCR 仅差 `-0.00152`，makespan 仅高 `0.0813`，质量差异 CI 均跨 0 | 通过 |

GPPO-event 五个种子的 best validation updates 为 `[60, 60, 50, 90, 20]`。验证曲线
非单调，但没有系统性后段崩塌：update 80→100 的 DCR 平均变化为 `-0.00529`，CI
`[-0.02244, 0.01187]`；makespan 平均变化为 `+0.1090`，CI
`[-0.2386, 0.4567]`。

## 通信和结构消融

### 通信模式

| GPPO 通信 | DCR ↑ | Makespan ↓ | Invalid ↓ | Comm. ↓ | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| none | 0.7472 | 79.6047 | 75.865 | 0.000 | stale cache 下冲突严重 |
| periodic | 0.7777 | 21.6121 | 4.446 | 9.835 | 最省通信，但质量和冲突代价较高 |
| event | 0.7774 | 20.2619 | 0.315 | 21.650 | 冻结主方法，质量—通信折衷较稳健 |
| always | 0.7789 | 20.1806 | 0.000 | 42.831 | 即时全状态工程 Oracle |

Event 相对 Periodic 的 DCR 基本相同（差 `-0.00035`，CI 跨 0），makespan 低
`1.3502`，CI `[-2.2765, -0.4240]`，但多使用 `11.815` 次同步。该结果是明确的
质量—通信折衷，不能简单宣称某一种模式全面更好。

### AHGNN 补充消融

| 结构 | DCR ↑ | Makespan ↓ | Comm. ↓ | 结论 |
| --- | ---: | ---: | ---: | --- |
| adaptive GPPO-event | 0.7774 | 20.2619 | 21.650 | 冻结主方法 |
| no-gate | 0.7764 | 20.3470 | 21.644 | 与 adaptive 两个主指标均无明确差异 |
| single-head | **0.7865** | **20.1115** | 21.646 | DCR 比 adaptive 高 `0.00915`，CI `[0.00218, 0.01613]` |

所有模式的 checkpoint 容器总参数均为 155223，但 active path 分别为：none 24899、
single-head 58703、adaptive/no-gate 121419。总参数相同不等于有效容量相同；single-head
比较同时包含结构和有效路径容量变化，不能作为 gate 的纯因果实验。No-gate 与 adaptive
没有明确差异，因此本轮实验**没有证明自适应 gate 有效**。

## 负面和不确定结果

正式 `negative_results` 共 367 条，其中 46 条 negative、321 条 inconclusive。所有 Holm
校正 p 值均为 `1.0`。5 个训练种子的双侧 exact sign-flip 最小原始 p 值为 `0.0625`，
因此不得使用 `p < 0.05` 或“统计显著”措辞；本文的 positive/negative 仅表示 Student-t
CI 的方向性分离。

必须保留的关键负面证据：

- `3x20` 上 GPPO-event 的 DCR 比 Greedy 低 `0.02169`，CI
  `[-0.02770, -0.01568]`；remaining tasks 多 `0.452`、throughput 低 `0.02467`、
  invalid actions 多 `0.102`，重分配成功率低 `0.1299`。
- Single-head 宏平均 DCR 明确优于 adaptive；`3x16` makespan 和 `4x24` DCR 也出现
  对 adaptive 不利的逐规模结果。
- No-gate 与 adaptive 的两个主指标均不明确，gate 贡献尚未建立。
- Periodic 的通信次数仅为 Event 的约 45%，但 makespan 更高且 invalid actions 更多。
- GPPO-event 相对 PPO-event 的 DCR 宏平均 CI 跨 0；当前图结构收益主要体现在
  makespan，而不是所有质量指标。

## 最终判断与下一步边界

按照结果出现前冻结的 `docs/GPPO_V2_ACCEPTANCE.md`，所有硬门槛均通过。因此：

> GPPO-event 已从“方法结构原型”提升为可验收的正式工程基线：它稳定超过 Random，
> 整体接近 Greedy，并以约一半的同步通信达到接近 Always 的任务质量。

同时必须附带以下限定：

1. 这不是论文数值级复现。
2. Adaptive gate 未得到正向消融支持，single-head 反而更强。
3. Event 的优势依赖当前独立实现的 deadline、cache 和事件压力协议，不能归因于论文未
   公开的精确事件触发器或信道模型。
4. 当前 `n=5` 对 sign-flip 显著性检验功效不足，后续若需要统计显著性，应增加训练种子。
5. 本任务未进入 PCRL，也未接入世界模型；是否启动下一阶段应以本报告为基线冻结点。

## 可追溯产物与复现命令

- 正式训练：`outputs/paper_aligned/gppo_v2_hard/formal/train/`
- 正式评估：`outputs/paper_aligned/gppo_v2_hard/formal/eval/`
- 正式汇总：`outputs/paper_aligned/gppo_v2_hard/formal/summary/`
- 全量产物审计：`formal/summary/formal_artifact_audit.json`
- 冻结门槛分析：`formal/summary/acceptance_analysis.json`
- 完整负面结果：`formal/summary/summary.json` 与 `comparisons.csv`
- 曲线：`validation_curves.png`、`training_curves.png` 及对应 CSV
- 参数公平性：`parameter_audit.json`

```powershell
D:\anaconda\python.exe run_paper_gppo_v2.py --manifest configs\gppo_v2_hard.json --phase train --jobs 8 --output-root outputs\paper_aligned\gppo_v2_hard
D:\anaconda\python.exe run_paper_gppo_v2.py --manifest configs\gppo_v2_hard.json --phase evaluate --jobs 8 --output-root outputs\paper_aligned\gppo_v2_hard
D:\anaconda\python.exe run_paper_gppo_v2.py --manifest configs\gppo_v2_hard.json --phase summarize --output-root outputs\paper_aligned\gppo_v2_hard
D:\anaconda\python.exe audit_gppo_v2_formal.py --manifest configs\gppo_v2_hard.json --output-root outputs\paper_aligned\gppo_v2_hard\formal --scope all --json-output outputs\paper_aligned\gppo_v2_hard\formal\summary\formal_artifact_audit.json
D:\anaconda\python.exe analyze_gppo_v2_acceptance.py --summary outputs\paper_aligned\gppo_v2_hard\formal\summary\summary.json --per-seed outputs\paper_aligned\gppo_v2_hard\formal\summary\per_seed.csv --training-root outputs\paper_aligned\gppo_v2_hard\formal\train --artifact-audit outputs\paper_aligned\gppo_v2_hard\formal\summary\formal_artifact_audit.json --output outputs\paper_aligned\gppo_v2_hard\formal\summary\acceptance_analysis.json
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; D:\anaconda\python.exe -m pytest -q
```

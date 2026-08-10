# 第一阶段基线与后续转段存档（2026-08-10）

## 1. 存档目的

本记录冻结 2026-08-10 的阶段判断，避免后续把已完成的 100/300 轮机制验证重复执行，也避免把最终论文统计矩阵误认为进入 PCRL 或世界模型开发前必须完成的阻塞项。

本记录不覆盖、删除或改写既有 checkpoint、训练历史、负结果、筛选协议和正式矩阵产物。

## 2. 当前结论

- 第一阶段中的 **GPPO 基线机制复现已完成，可作为后续工程开发起点**。
- 第一阶段中的 **多源扰动仿真环境尚未完成**，下一步应补齐扰动机制，而不是重复六模型 100/300 轮训练。
- 既有单种子 300 轮实验是机制筛查证据，不是五训练种子、四规模的论文级统计证据。
- 三训练种子 Gate 筛选已经完成，并按 validation-A 预注册规则选出 `Adaptive-score` 作为 `GPPO-Best`。
- 四规模、五训练种子、2000 轮正式矩阵属于最终对比、消融和泛化实验。它最终仍需完成，但不阻塞 PCRL/世界模型的接口实现、数据结构开发和多源扰动环境建设。
- PCRL 的正式效果结论仍必须以冻结且通过多种子复核的 GPPO 基线为对照；在此之前只能开发，不能宣称提升成立。

## 3. 已完成的第一阶段内容

- Python 离散事件任务分配环境；
- 多规模 UAV、任务、子任务、前序约束、能力匹配和异步执行；
- 异构任务图和论文兼容 AHGNN/GPPO 实现；
- 动态图状态、动作掩码和逐步合法“子任务-UAV”配对；
- 弱通信缓存、事件触发同步和固定 event tape；
- 六种学习方法、单训练种子 300 轮、固定 test100；
- Random、Greedy、PPO、NoGate、SingleHead 对照；
- Gate 分布、策略敏感度和 PPO probe 梯度诊断；
- validation100-A/B 候选 checkpoint 重评估；
- 七种 Gate 变体、三个训练种子、每种 300 轮的预注册筛选；
- 基于 validation-A 选择 `Adaptive-score`，未使用 test100 选模。

## 4. 300 轮证据边界

300 轮固定 test100 的主要结果：

| 方法 | realized makespan mean |
|---|---:|
| GPPO-event | 16.1673 |
| PPO-none | 17.1720 |
| PPO-event | 17.0290 |
| GPPO-none | 16.0260 |
| GPPO-NoGate-event | 15.9732 |
| GPPO-SingleHead-event | 15.9389 |
| Random | 20.4073 |
| Greedy | 17.3752 |

同 checkpoint 通信重放中，Event 与 Full 的 makespan 均为 `16.1673`，Event 通信量减少约 `99.93%`。

该实验支持图结构、合法动作链路和事件通信机制可运行，并显示 GPPO 相对 PPO/Random 的正向结果。但 Adaptive 相对 NoGate 的 makespan 差值为 `+0.1942`，相对 SingleHead 为 `+0.2284`（正值表示 Adaptive 更差）。因此原 Adaptive 的独立收益未由单种子 300 轮证明。

这些实例级比较不能替代训练种子级置信区间，也不能证明四规模泛化或论文数值级复现。

## 5. 第一阶段剩余内容

下一步只补齐以下多源扰动环境和可审计测试：

1. Gilbert-Elliott 突发丢包；
2. 随机通信时延和网络分区；
3. UAV 故障、任务释放/恢复和能量衰减；
4. 任务新增、取消和优先级变化；
5. 风场导致的航时与能耗偏差；
6. 弱、中、强扰动配置、确定性事件脚本和随机种子复现；
7. 事件覆盖率、非法动作、能量边界、任务恢复和通信缓存语义测试；
8. 为世界模型提供统一 observation/action/event/target 轨迹格式。

上述工作以实现和短程 smoke 为主，不要求重新执行已有六模型 100/300 轮实验。

## 6. 正式矩阵的定位与恢复条件

正式协议仍冻结为四规模、五训练种子、每个可学习模型 2000 iterations。完整矩阵用于：

- 训练种子级 Student-t 95% CI；
- 四规模泛化；
- PPO/GPPO、Gate 和结构消融；
- Event/None/Periodic/Full 因果通信重放；
- 最终论文图表与统计结论。

该矩阵应在基线、扰动接口、PCRL 和世界模型结构冻结后分片执行，避免每次算法修改后重复高成本训练。暂停前应先完成 Drive 备份；已有 `resume_latest.pt`、candidate 和 frozen checkpoint 均应保留，未来从断点恢复。

2026-08-10 用户截图所示 Colab 状态仅作为运行观察记录：总目标 `270,000` iterations、两个并发训练进程、约 `0.510 iter/s`、T4 GPU 利用率约 `16%`，面板外推 ETA 约六天。该 ETA 不是阶段验收结果。

## 7. 转段规则

- **允许立即开展**：多源扰动环境、轨迹格式、向量指标、偏好对生成、奖励模型接口、世界模型数据管线和模型骨架。
- **允许在最小扰动 smoke 通过后开展**：Preference-GPPO 与世界模型的训练联调。
- **暂不允许宣称**：PCRL 优于 GPPO、预测式触发优于反应式触发、五种子稳定性、四规模泛化或论文数值级复现。
- **最终必须完成**：冻结算法后的正式多种子对比、消融、泛化、校准和显著性实验。

## 8. 权威证据与完整性

- `artifacts/phase1_seed1_300/raw/MECHANISM_ACCEPTANCE.json`
  - SHA256: `a703b92bebf4b7bb01d37ceb61ee13264df0d6b359323c8046423be285d408a0`
- `artifacts/phase1_seed1_300/ARTIFACT_AUDIT.json`
  - SHA256: `d70c36df950507a18af50785999715d103f96b9cd32c97aaa6c196e53c52bd7a`
- `configs/PHASE1_FROZEN_PROTOCOL.json`
- `reports/CANDIDATE_CHECKPOINT_REEVALUATION.md`
- `reports/GATE_THREE_SEED_SCREENING.md`
- `docs/PHASE1_MODEL_DEFINITIONS.md`

若本记录与原始 JSON、checkpoint 哈希或冻结协议冲突，以原始可审计产物和冻结协议为准。

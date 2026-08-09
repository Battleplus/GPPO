# 多源扰动现有能力审计

## 审计结论

现有 `paper_env.py` 中的随机 hazard 只能视为早期工程原型，不能作为 Phase 1B 的正式多源扰动环境。正式实现必须在 Phase 1A 模型协议冻结后，以独立、可关闭、可重放的 disturbance tape 层接入；不得改变无扰动基线的 observation、奖励、动作掩码或硬约束语义。

## 可复用能力

| 能力 | 当前证据 | 复用边界 |
|---|---|---|
| 事件结构化日志 | `EventRecord` 保存 decision、物理时间、来源、目标、before/after 和 details | 可作为 `DisturbanceLogger` 的底层记录格式 |
| UAV 永久故障 | 故障后设置 `alive=false`，释放运行任务并增加重分配计数 | 只能复用状态转移与释放逻辑；故障时刻和目标必须由固定 tape 驱动 |
| leader 故障与重选 | leader 失效后选择存活 UAV，并记录 `leader_failure/leader_elected` | 可复用确定性重选机制 |
| 新任务插入 | 可激活预分配的 inactive task slot | 必须补充前序关系、取消、运行任务释放和优先级独立事件 |
| 动作掩码与真实硬约束 | belief mask 和 true mask 分开检查，非法动作计数 | 所有扰动必须经过该路径，禁止绕过 |
| 异步执行 | 按下一个任务完成、心跳或外生事件推进物理时间 | 可承载延迟队列和定时扰动，但需要独立队列实现 |
| faithful 固定事件带 | `PaperFaithfulUAVEnv` 固定任务分布变化时刻、目标和强度 | 仅用于 GPPO 公平比较，不等于多源扰动 tape |

## 不满足项

| 任务书要求 | 当前实现 | 判定 |
|---|---|---|
| Gilbert–Elliott 突发丢包 | 仅按指数等待时间抽取一次 `communication_drop`，随后修改 UAV 的 communication 标量 | 未实现 Good/Bad 马尔可夫链、逐包丢失和 burst 校准 |
| 随机时延 | 无消息对象、发送时间、到达时间或延迟队列 | 未实现 |
| 网络分区 | 无连通分量、跨区可见性和恢复同步协议 | 未实现 |
| 临时 UAV 故障 | 仅永久死亡 | 未实现 |
| 能量衰减 | UAV 状态没有 battery/energy，航行和执行不扣能量 | 未实现 |
| 低能量降速/降能力 | 无能量状态 | 未实现 |
| 任务取消 | 无取消事件和运行任务释放语义 | 未实现 |
| 动态前序关系 | 新任务没有正式更新父任务/子任务 DAG | 未实现 |
| 独立优先级变化 | task state change 同时随机修改位置、工作量和优先级 | 不满足单因素可审计要求 |
| 风向与风速 | weather 只有单一 severity 标量 | 未实现矢量风场、航向相关航时和能耗 |
| 固定扰动重放 | hazard 由环境 RNG 在线抽样，没有独立序列化 tape | 未实现完整重放 |
| weak/medium/strong 校准 | 没有按扰动类型分级的冻结配置和非饱和检查 | 未实现 |

## 实施边界

1. Phase 1A 冻结前不把扰动接入训练或模型选择。
2. 新建 `DisturbanceConfig`、`DisturbanceTape`、`DisturbanceEvent`、`DisturbanceLogger`，不得继续扩展单一 `event_id` 随机分支。
3. tape 在 episode reset 前由独立 seed 完整生成；运行期只能消费，不得临时改变概率或根据策略表现重采样。
4. 丢包和时延只影响消息传递，不能直接修改真实环境状态。
5. 所有任务与 UAV 状态变化必须继续通过现有动作掩码、能力约束、前序约束和终止判定。
6. 原 hard-extension 结果和代码保留为历史负结果，不作为 Phase 1B 验收证据。

当前判定：Phase 1B 尚未实现；只有若干可复用底层状态转移和日志组件。

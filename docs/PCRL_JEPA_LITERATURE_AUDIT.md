# PCRL / 世界模型后置审计

## Preference Controllable RL 论文要点

已阅读用户提供的 `Preference Controllable Reinforcement Learning with Advanced Multi-Objective Optimization`。论文的 PCRL 不是在标量 reward 上简单乘一个偏好系数，而是：

- 使用向量奖励的 MOMDP，并把 preference vector `p` 作为条件输入给策略；
- 估计每个 objective 的 value vector；
- 用 preference similarity 梯度和 objective Jacobian 做 MOO 梯度操纵；
- PreCo 目标是同时逼近 Pareto stationary 和偏好方向，论文还讨论 calibration。

因此，当前 GPPO 不能直接声称已经完成 PCRL：现有正式协议的主奖励是 makespan 增量，尚未定义经过尺度归一化的多目标向量，也没有 preference-conditioned actor 或 PreCo/MGDA/EPO 等梯度更新。合理顺序是先完成 GPPO 的 reward、realized makespan、Graph x Sync 和通信因果审计，再冻结一组可复现的多目标定义，最后做 PCRL 消融。

## From Observations to Events / JEPA 的接口要求

事件感知世界模型的接入需要独立验证 observation-to-event 标签、预测校准、触发延迟和误触发通信成本。当前 paper-faithful 环境使用固定 event tape，目的是让 GPPO 方法比较共享外部事件序列；它不是世界模型预测器。把 tape 直接当作预测结果会造成标签泄漏，不能作为 JEPA/world-model 结果。

## 当前决策

PCRL、JEPA 和 world-model 代码接入继续暂停。只有当 `GPPO-event` 相对论文式 PPO、随机和合理贪心基线达到预注册门槛，并完成五 seed 的模块/通信统计后，才进入下一阶段。

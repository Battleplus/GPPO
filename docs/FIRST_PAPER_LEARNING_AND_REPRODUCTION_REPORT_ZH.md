# 第一篇论文学习与机制复现报告

## 1. 报告目的

本报告用于开学前系统理解论文《Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication》的完整方法、代码流程、已有 300 轮实验结果和复现边界。

当前阶段不追加长时间训练，不修改算法，不开展 Preference-GPPO、世界模型或 GRPO/RLHF 实验。报告中的结论均来自当前分支 `8.8-GPPO无偏好` 的已冻结代码和可审计产物。

## 2. 当前完成状态

截至当前提交 `f2cfc06`：

- GPPO 关键机制已完成结构级/机制级复现；
- 已完成 `T5-10-48 / training seed 1 / 300 iterations / fixed test100` 六模型验证；
- 已完成七种 Gate 变体、三个训练种子、每种 300 轮的筛选；
- 已完成固定 checkpoint 的 Event/Full 通信重放；
- 已完成 Phase 1B 多源扰动环境、确定性回放和审计；
- 测试结果为 `114 passed`；
- Phase 1B 最终审计为 `valid=true`。

这些产物已经足够用于当前的论文学习、代码理解和开会汇报，不需要重复运行 300 轮训练。

## 3. 论文要解决的问题

研究对象是在弱通信条件下，为多架异构 UAV 动态分配具有前序依赖的子任务。

核心困难包括：

1. UAV 能力不同，不是每架 UAV 都能执行所有子任务；
2. 一个任务由多个子任务构成，必须满足 DAG 前序约束；
3. UAV 异步执行，空闲时间、位置和可用状态不断变化；
4. 新任务、任务完成、UAV 故障等事件会使原分配失效；
5. 弱通信导致每架 UAV 持有的信息可能过时；
6. 动态动作空间中必须避免重复分配、能力不匹配和前序冲突；
7. 系统既要降低总完成时间，也要减少不必要通信。

论文的解决思路可以概括为：

```text
局部可观测 UAV/任务状态
        ↓
异构动态图：UAV 节点 + 子任务节点 + 多类关系
        ↓
AHGNN：自适应注意力提取节点表示
        ↓
动态动作掩码：只保留合法“子任务-UAV”配对
        ↓
PPO Actor-Critic：选择分配动作
        ↓
离散事件环境：异步执行并推进物理时间
        ↓
关键事件触发同步，否则继续使用 belief cache
```

## 4. 环境、状态、动作、奖励和转移

### 4.1 环境

环境采用离散事件推进，而不是固定长度时间步。一次动作通常表示把一个当前可执行的子任务分配给一架空闲且能力匹配的 UAV。环境随后推进到新的决策边界，例如 UAV 完成任务、出现动态事件或需要重新分配。

主要实现：

- `src/uav_assignment/paper_faithful_env.py`：论文式冻结环境；
- `src/uav_assignment/phase1b_env.py`：在冻结环境上增加通信、UAV、任务和风场扰动；
- `src/uav_assignment/disturbances/`：扰动配置、事件带、通信、能量、动态任务和轨迹记录。

### 4.2 状态与 observation

模型输入包含：

- 节点特征 `nodes`；
- 关系类型 `edge_types`；
- 边特征 `edge_features`；
- 动态合法动作 `action_mask`。

节点由 UAV 节点和子任务节点组成；边至少覆盖 self、UAV 通信、任务前序/后继、UAV-任务能力等关系。弱通信环境下，模型看到的是 belief/partial observation，审计同时保存 true state，用于判断缓存是否真实过时。

### 4.3 动作

动作是一个动态生成的“子任务-UAV”配对，外加必要的 no-op。动作只有在以下条件满足时才合法：

- 子任务处于活动状态且尚未完成；
- 所有前序子任务已经完成；
- UAV 可用、存活且未忙碌；
- UAV 能力与子任务要求匹配；
- 弱通信 belief 下该动作未被掩码。

`PaperFaithfulActorCritic.forward()` 会在输出 logits 上将非法动作填充为 `-1e9`，保证 categorical policy 不选择被掩码动作。环境仍独立统计非法动作，形成双层防线。

### 4.4 奖励

论文定义总体预计完成时间（makespan）为所有任务完成时间的最大值，并使用：

```text
R_t = M_(t-1) - M_t
```

即新分配若降低预计 makespan，则获得正向奖励。代码在 `paper_faithful_env.py::step()` 中以相同方向计算 `previous_makespan - current_makespan`。

训练时记录的 reward 用于 PPO 更新；最终算法质量主要通过 `realized_makespan`、任务完成率、非法动作和通信量验证。训练 reward 与最终 realized makespan 必须区分。

### 4.5 转移与终止

环境执行动作后更新：

- UAV 忙碌/空闲和预计可用时间；
- 子任务分配、执行和完成状态；
- DAG 后继任务的可执行性；
- UAV 和任务图特征；
- belief cache 与通信统计；
- 当前物理时间和预计/实际 makespan。

所有有效任务完成时 episode 正常终止；Phase 1B 还处理临时故障后的未来恢复、能量耗尽和动态任务取消等情况。

## 5. 异构图与 AHGNN Eq.(1)-(5)

### 5.1 图的组成

图包含两类节点：

- UAV 节点：位置、能力、可用时间、忙碌状态、存活状态等；
- 子任务节点：任务类型、前序状态、预计执行时间、活动/完成状态等。

边承担两种作用：

- 表示结构约束，如 task predecessor/successor；
- 携带关系特征，如 UAV 到任务的航行/执行代价和能力匹配。

这种表示使模型能够处理节点数量变化、任务结构变化和异构能力约束。

### 5.2 Eq.(1)：UAV 对相邻任务的注意力

对 UAV 节点 `v_k` 与相邻任务扩展特征 `μ_(i,j,k)` 计算注意力分数。代码由 `LiteralUAVAttention` 完成，分别投影 UAV、任务和边特征后计算 score。

作用：让 UAV 表示关注与自己能力、位置和当前任务结构最相关的子任务。

### 5.3 Eq.(2)：self attention

self 项表示 UAV 自身信息对更新后表示的贡献。当前实现明确使用独立的 `W^T v_k`，而不是错误复用任务边变换或 `W^U v_k`。

作用：即使邻居信息变化，UAV 自身状态仍能参与更新。

### 5.4 Eq.(3)：Adaptive Gate

任务消息先经过注意力，再由可学习 Gate 调节。论文兼容版本默认：

- `gate_scope=task_message`；
- sigmoid gate；
- self 与 task neighbors 位于同一 softmax 域；
- gate 后不进行二次归一化；
- expected-RReLU 作为正式默认。

作用：理论上根据 UAV 与任务组合动态抑制或放大任务消息。

需要注意：已有诊断证明 Gate 不是常数并且具有梯度，但单种子 300 轮结果没有证明原始 Gate 比 NoGate 或 SingleHead 更好。

### 5.5 Eq.(4)-(5)：任务节点关系聚合

任务节点分别聚合：

- 自身；
- 前序任务；
- 后继任务；
- 相邻 UAV。

四类关系经过独立 MLP，再由 fusion MLP 合并。这部分由 `LiteralTaskUpdate` 实现。

作用：使任务表示同时包含 DAG 位置、上下游约束和可执行 UAV 信息。

## 6. PPO 与 GPPO 的关系

GPPO 不是一种完全不同于 PPO 的优化器。当前项目中：

- PPO 负责 rollout、advantage、clip objective、value loss 和参数更新；
- GNN/AHGNN 负责把动态异构图编码为 actor/critic 可使用的表示；
- 动作掩码保证 policy 只在合法动作上归一化；
- 事件触发机制决定何时更新 belief cache；
- 因此 GPPO 可以理解为“图编码器 + 动态掩码 + 事件同步机制上的 PPO”。

当前训练流程：

1. 用固定配置创建环境和模型；
2. 收集指定数量 rollout steps；
3. 使用 reward 和 value 计算 GAE；
4. 标准化 advantage；
5. 计算新旧 policy probability ratio；
6. 使用 clipped surrogate objective 更新 actor；
7. 用 return 与 critic value 的 MSE 更新 critic；
8. 加入 entropy 鼓励探索；
9. 记录近似 KL、clip fraction、Gate 梯度等诊断；
10. 每隔固定 iterations 在 validation-A 上评估；
11. 按 validation realized makespan 选择 checkpoint；
12. 最终在固定 test100 上一次性报告结果。

关键训练代码：`train_paper_faithful.py`。

## 7. 事件触发通信

### 7.1 四种模式

- `none`：除基础心跳等必要流量外不主动同步；
- `event`：只有事件发生时同步相关状态；
- `periodic`：缓存年龄超过固定周期时同步；
- `always/full`：每个决策点进行完整同步。

### 7.2 belief cache

弱通信下模型不一定看到真实全局状态，而是使用最近同步得到的 belief。信息过时可能影响任务可见性和动作掩码，因此代码同时保存 partial observation、true state 和 cache age 进行审计。

### 7.3 heartbeat 与 leader

heartbeat 用于确认 UAV 是否存活及恢复过时缓存。leader 失效时系统按确定性规则重新选举，并释放、重分配原 leader 或故障 UAV 的任务。

### 7.4 Event/Full 结果的解释

同一 checkpoint、同一 test100 和同一 event tape 重放中：

- Event makespan：`16.1673`；
- Full makespan：`16.1673`；
- Event bytes：`2790.9`；
- Full bytes：`4,266,481.0`；
- 通信减少约 `99.93%`。

这说明在当前固定事件协议中，关键事件同步已经提供了影响分配所需的信息，额外的全量同步没有改善 makespan。它证明的是该 checkpoint 下的因果重放结果，不代表所有弱通信强度下 Event 都与 Full 等价。

## 8. 300 轮实验结果

实验范围：`T5-10-48 / training seed=1 / 300 iterations / fixed test100`。

| 方法 | realized makespan mean | median | completion | communication bytes |
|---|---:|---:|---:|---:|
| GPPO-event | 16.1673 | 16.0369 | 1.0000 | 2790.9 |
| PPO-none | 17.1720 | 17.1472 | 1.0000 | 1069.4 |
| PPO-event | 17.0290 | 16.7780 | 1.0000 | 2844.6 |
| GPPO-none | 16.0260 | 15.9242 | 1.0000 | 992.6 |
| GPPO-NoGate-event | 15.9732 | 15.8206 | 1.0000 | 2776.8 |
| GPPO-SingleHead-event | 15.9389 | 15.8049 | 1.0000 | 2776.8 |

非学习基线：

- Random：`20.4073`；
- Greedy：`17.3752`。

### 8.1 已支持的正结论

- GPPO-event 相对 PPO-none 的实例级配对差值为 `-1.0046`，95% CI `[-1.3246, -0.6847]`；
- GPPO-event 相对 PPO-event 的差值为 `-0.8617`，95% CI `[-1.1529, -0.5706]`；
- GPPO-event 优于 Random 和当前 Greedy；
- 图中多类异构关系被实际使用，图状态随决策变化；
- 48 次子任务-UAV 配对均通过 mask，环境非法动作为 0；
- 事件决策 3 次且 3 次均同步，非事件错误同步为 0；
- Event 与 Full 同 checkpoint 重放质量相同，通信量显著下降。

### 8.2 必须保留的负结果

- Adaptive 相对 NoGate：`+0.1942`，95% CI `[+0.0061, +0.3823]`；
- Adaptive 相对 SingleHead：`+0.2284`，95% CI `[+0.0317, +0.4251]`。

正值表示原始 Adaptive 更差。因此不能声称原始 Adaptive Gate 已被证明有效。

Gate 的分布和梯度诊断表明：

- gate mean/std：`0.3517 / 0.3299`；
- policy sensitivity gradient L2：`0.0435`；
- PPO probe gradient L2：`0.1039`。

所以负结果不能简单归因于 Gate 没有参与前向或没有梯度，更可能涉及作用位置、优化难度、任务规模、样本量或结构归纳偏置。

### 8.3 三训练种子 Gate 筛选

后续对七个变体进行三训练种子、每种 300 轮的 validation-A-only 筛选：

- Adaptive-score：`15.9025`；
- SingleHead：`15.9613`；
- NoGate：`15.9805`。

`Adaptive-score` 被选为 `GPPO-Best`。该结果只用于工程候选晋级，不替代四规模五种子的正式统计结论。

## 9. GPPO-Literal 与 GPPO-Best

### GPPO-Literal

- 目标：保留论文兼容解释；
- Gate 作用于 task message；
- 即使结果为负也必须保留；
- 用于回答“按当前理解直接实现论文机制会发生什么”。

### GPPO-Best

- 目标：验证集选择的工程基线；
- 使用 `Adaptive-score`；
- 只由 validation-A 选出；
- 用于后续工程比较，但不能冒充原始论文公式的唯一实现。

二者必须并存：Literal 保证研究可解释性，Best 提供更有竞争力的工程起点。

## 10. Phase 1B 多源扰动

Phase 1B 在冻结基线上增加：

- Gilbert-Elliott 突发丢包；
- 消息时延、TTL、网络分区和恢复；
- UAV 永久/临时故障、能量消耗和低能量退化；
- 动态任务新增、取消、优先级和 deadline 变化；
- 风场导致的航时与能耗偏差；
- 可序列化、可哈希、可重放的 disturbance tape；
- partial observation、true state、通信历史和未来事件标签。

全扰动关闭时，固定 test100 的 2,000 个配对决策在 observation、mask、reward、done、info 和指标上保持等价。

校准结果：

| 强度 | makespan | 丢包率 | 平均时延 | 最低能量 | 完成率 |
|---|---:|---:|---:|---:|---:|
| Off | 16.935 | 0% | 0 | 1.000 | 1.0 |
| Weak | 18.673 | 0.98% | 0.124 | 0.914 | 1.0 |
| Medium | 19.971 | 10.24% | 0.385 | 0.757 | 1.0 |
| Strong | 20.903 | 28.84% | 0.837 | 0.564 | 1.0 |

这些结果验证扰动强度和环境机制，不是 GPPO 在扰动下优于其他算法的证据。

## 11. 为什么当前不用再训练

现有 300 轮结果已经完成当前学习阶段需要的：

- 训练流程；
- checkpoint；
- test100；
- 图/PPO 对照；
- Gate 消融；
- 通信重放；
- 逐步机制审计；
- Phase 1B 环境验证。

重复运行同一配置只会再次采集同类证据，不能自动提升结论强度。只有在以下情况才应重新训练：

- 发现明确实现错误；
- 模型、环境、reward 或协议发生变化；
- 老师要求四规模、五训练种子的正式统计；
- 开始验证新的 GRPO/RLHF 研究假设；
- 现有 checkpoint 或日志无法通过完整性审计。

## 12. 必要知识清单

### 强化学习

- MDP：状态、动作、奖励、转移和折扣；
- return 与 value；
- advantage：动作相对状态平均水平的好坏；
- GAE：用 `γ` 和 `λ` 平衡偏差与方差；
- PPO clip：限制一次更新偏离旧策略过远；
- entropy：维持探索；
- KL 和 clip fraction：监控策略更新幅度。

### 图学习

- GNN message passing；
- GAT attention；
- 异构节点和异构关系；
- edge feature；
- 动态图；
- 动作掩码与结构约束。

### 系统机制

- 离散事件仿真；
- leader/follower；
- heartbeat；
- belief cache；
- 丢包、时延、TTL 和网络分区；
- 事件触发与周期/全量同步。

### 为后续 GRPO/RLHF 准备

- PPO 与 GRPO 的优势估计差异；
- 同一 prompt 的 group sampling；
- group-relative reward normalization；
- KL reference policy；
- RLHF 中的偏好数据、reward model 和策略优化。

## 13. PPO、GRPO、PCRL 与 RLHF 的区别

| 名称 | 主要问题 | 当前项目中的位置 |
|---|---|---|
| PPO | 用 critic/advantage 稳定更新策略 | 第一篇论文的优化器基础 |
| GPPO | 图编码、动态 mask、事件同步上的 PPO | 已完成机制级复现 |
| GRPO | 用同组样本的相对奖励形成优势，常用于大模型训练 | 未来可能研究，尚未实现 |
| RLHF | 从人类偏好数据学习奖励或直接优化策略 | 长期研究背景，尚未实现 |
| PCRL/PreCo | 显式数值偏好条件下的多目标 RL | 第二篇论文，仅轻量阅读 |

第二篇 PCRL 的 preference vector 不是 RLHF 中从人工比较学习得到的偏好模型。两者可以组合，但不能直接等同。

## 14. 复现边界

当前可以准确声明：

> 已完成第一篇论文关键结构的机制级复现，包括异构任务图、AHGNN、动态动作掩码、PPO Actor-Critic、事件触发通信、固定 test100 验证和可审计多源扰动环境。

当前不能声明：

- 原论文公开数值被精确复现；
- GPPO 在五训练种子上稳定优于 PPO/Greedy；
- Adaptive Gate 已稳定有效；
- 四规模泛化已经完成；
- 现有实例级置信区间等同于训练种子级置信区间；
- 当前扰动环境性能等同于真实 UAV 系统；
- Preference-GPPO、世界模型或 GRPO/RLHF 已经完成。

原因是原论文未公开完整环境、实例生成器、训练代码和随机细节；现有 300 轮主结果以单训练种子和单规模为主。

## 15. 开会时应能回答的问题

1. 为什么任务分配适合表示为异构图？
2. Eq.(1)-(5) 分别更新什么信息？
3. 动作掩码为什么既是效率机制也是安全约束？
4. GPPO 与普通 PPO 的真正差别在哪里？
5. Event 为什么可以大幅减少通信量？
6. Event/Full 相同 makespan 能证明什么、不能证明什么？
7. Adaptive Gate 为什么是负结果？
8. GPPO-Literal 与 GPPO-Best 为什么必须同时保留？
9. 300 轮结果为什么不能替代五训练种子？
10. 哪些机制可能启发 GRPO/RLHF，哪些只是 UAV 场景特有机制？

## 16. 权威证据索引

- 第一篇论文 PDF：用户提供的 IEEE TASE 2025 论文；
- `README.md`：当前项目总览和结果边界；
- `src/uav_assignment/paper_faithful_models.py`：AHGNN 与 Actor-Critic；
- `src/uav_assignment/paper_faithful_env.py`：论文式状态、奖励和通信模式；
- `train_paper_faithful.py`：rollout、GAE 和 PPO 更新；
- `artifacts/phase1_seed1_300/raw/MECHANISM_ACCEPTANCE.json`：300 轮权威结果；
- `artifacts/phase1_seed1_300/raw/GPPO_MECHANISM_REPORT_ZH.md`：中文机制报告；
- `reports/GATE_THREE_SEED_SCREENING.md`：三训练种子 Gate 筛选；
- `docs/PHASE1_MODEL_DEFINITIONS.md`：Literal、Best 与消融定义；
- `configs/PHASE1_FROZEN_PROTOCOL.json`：冻结实验协议；
- `deliverables/phase1b/DISTURBANCE_IMPLEMENTATION_AUDIT.json`：Phase 1B 最终审计；
- `deliverables/phase1b/ALL_OFF_EQUIVALENCE_TEST100.json`：全关闭等价证据。

## 17. 学习与自测入口

完成本报告阅读后，使用 `docs/OPENING_MEETING_STUDY_WORKBOOK_ZH.md` 沿一次 episode 对照代码，并完成 20 题口述自测。只有能够不看材料讲清环境、图编码、动作、通信、奖励、PPO 更新、结果与边界，才视为达到开学前的理解目标。

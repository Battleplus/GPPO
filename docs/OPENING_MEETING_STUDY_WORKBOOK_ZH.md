# 开学前学习导读与自测手册

## 1. 这份手册解决什么问题

现有训练、评估和对照已经足够。现在需要把已有证据转化成能够独立表达的理解。

本手册不要求重新训练，也不要求重新实现代码。完成标准是：不看材料时，能够沿一次 episode 讲清楚“环境生成—观测—图编码—动作—事件推进—通信—奖励—rollout—PPO 更新—选模—test100”。

建议分五次学习，每次只完成一个闭环：

1. 问题、环境和动作约束；
2. AHGNN Eq.(1)～(5) 与模型输出；
3. 一次 episode 与事件通信；
4. PPO 更新、checkpoint 和 test100；
5. 300 轮结果、复现边界和后续方向。

## 2. 一次 episode 的完整代码路径

### 2.1 初始化环境和实例

入口：

- `src/uav_assignment/paper_faithful_env.py:115`：`PaperFaithfulUAVEnv`；
- `src/uav_assignment/paper_faithful_env.py:133`：`reset()`；
- `src/uav_assignment/paper_faithful_env.py:141`：生成带前序依赖的任务；
- `src/uav_assignment/paper_faithful_env.py:167`：建立固定 event tape。

`reset(seed)` 做三件关键事情：

1. 根据固定实例 seed 生成 UAV 和任务；
2. 建立子任务的 predecessor 链，形成当前实现支持的 DAG 特例；
3. 初始化 true state、belief state、leader、heartbeat 时钟和通信统计。

必须注意：当前 paper-faithful adapter 的每个任务槽最多只有一个直接前驱，因此它实现的是一般 DAG 的受限形式，不应说成支持任意多前驱 DAG。

### 2.2 构造 observation

入口：

- `src/uav_assignment/paper_env.py:1093`：`_build_observation()`；
- `src/uav_assignment/paper_env.py:1231`：`observe()`；
- `src/uav_assignment/paper_env.py:1241` 附近：`true_observation()`。

observation 包含五组核心数据：

| 字段 | 作用 |
|---|---|
| `nodes` | UAV 与任务节点特征 |
| `edge_types` | self、UAV 通信、前序/后继、UAV-task 等关系类型 |
| `edge_features` | 飞行时间、执行时间、能力、通信质量、依赖是否满足等 |
| `adjacency` | 是否存在有效关系边 |
| `action_mask` | 当前 belief 下合法的 UAV-task 配对 |

UAV 节点包含 active、alive、idle、位置、health、通信、剩余时间、累计处理量、leader 标志、heartbeat freshness、能力等。

任务节点包含 active、completed、位置、priority、workload、risk、前序是否满足、是否已分配、处理时间、predecessor、任务类型和 age 等。

关键区别：

- `observe()` 使用 belief state，是策略实际看到的状态；
- `true_observation()` 使用真实状态，只用于审计；
- 两者不同，才是真正的弱通信与过时信息，而不是给模型完整状态后仅统计通信量。

### 2.3 生成动态 action mask

入口：

- `src/uav_assignment/paper_env.py:506` 附近：`_valid_mask_for()`；
- `src/uav_assignment/paper_env.py:524`：`valid_action_mask()`。

一个 UAV-task 动作只有同时满足以下条件才合法：

1. UAV 处于 idle；
2. 任务 active、未完成、未被分配；
3. predecessor 已完成；
4. UAV 对该任务类型的 capability 达到阈值；
5. 配对在 belief state 中可见。

动作编号通过 `divmod(action, max_tasks)` 还原为 UAV index 和 task index。只有不存在任何合法配对时，no-op 才合法。

因此动态 mask 不是训练技巧，而是把硬约束直接编码进策略动作空间。模型在 `paper_faithful_models.py:294` 再把非法动作 logits 置为 `-1e9`，使其采样概率近似为零。

### 2.4 AHGNN 把 observation 编成节点表示

入口：`src/uav_assignment/paper_faithful_models.py`。

| 论文结构 | 代码位置 | 可口述解释 |
|---|---|---|
| Eq.(1) UAV-task 注意力 | `LiteralUAVAttention.forward()`，约 75～129 行 | 每个 UAV 在 self 与可连接任务的共同 softmax 域中分配注意力 |
| Eq.(2) self 项 | 53、92～99 行 | self 表示使用独立变换，不能和 task-edge 变换混为一个矩阵 |
| Eq.(3) Adaptive Gate | 57～62、100～137 行 | Gate 调节任务消息进入聚合的强度；默认是 post-softmax task-message gate |
| Eq.(4) 多关系任务更新 | `LiteralTaskUpdate`，177～208 行 | 分别聚合 UAV、前驱、后继和 self 关系 |
| Eq.(5) 关系融合 | 209～210 行 | 拼接四类关系表示后通过 fusion MLP 得到任务节点新表示 |

必须能解释：Gate 有非零梯度，只能证明它参与了优化；不能据此证明它对最终 test100 必然有收益。

### 2.5 Actor-Critic 输出动作和 value

入口：

- `paper_faithful_models.py:214`：`PaperFaithfulActorCritic`；
- `paper_faithful_models.py:251`：`pair_actor`；
- `paper_faithful_models.py:253`：scalar critic；
- `paper_faithful_models.py:277`：`forward()`。

Actor 将每个 UAV embedding、task embedding 和对应 edge feature 拼接，输出该配对的 logit。所有配对 logit 再加 no-op logit，经过 action mask 后形成 categorical distribution。

Critic 对有效节点表示做池化，输出当前状态的一个标量 value。这个标量 critic 是 PPO 估计 advantage 的基线。

### 2.6 执行动作并按事件推进时间

入口：

- `src/uav_assignment/paper_env.py:845`：基础 `step()`；
- `src/uav_assignment/paper_faithful_env.py:239`：paper-faithful 包装层 `step()`。

一次 `step(action)` 的逻辑顺序是：

1. 同时检查 belief mask 和 true-state mask；
2. 合法时在 belief 和 true state 中应用任务分配；
3. 若当前不能继续分配，则跳到下一个离散事件边界；
4. 更新任务 age、UAV remaining time 和已完成任务；
5. 完成前驱任务后解锁 successor；
6. 处理 heartbeat、故障、天气或重分配事件；
7. 根据通信模式决定是否同步 belief；
8. 计算 reward、done 和 info；
9. 返回新的 `observe()`。

事件驱动意味着仿真时间不一定每步加一，而是直接跳到任务完成、外生事件、heartbeat、同步或 deadline 中最近的边界。

### 2.7 Event 通信、leader 和 belief cache

入口：

- `paper_env.py:391`：基础 `_synchronize_belief()`；
- `paper_env.py:467`：`_process_heartbeats()`；
- `paper_env.py:833` 附近：`_should_sync()`；
- `paper_faithful_env.py:180`：同步包装与 cache age；
- `phase1b_env.py:178`：带丢包/时延的 belief report。

四种模式的含义：

- `none`：不进行状态同步；
- `event`：出现被记录的关键事件才同步；
- `periodic`：到固定同步时间点才同步；
- `always`：每个决策步都同步。

heartbeat 是固定间隔的存活检查，不等于全量状态同步。leader 汇总可用信息；leader 或 UAV 故障后，系统更新 leader 状态，释放故障 UAV 占用的任务，并让后续合法动作重新分配任务。

Phase 1B 中，非 leader 的 belief report 经过弱链路发送，可能丢失或延迟；因此 `pending_beliefs`、发送时间和最后接收时间共同决定缓存是否更新。

### 2.8 reward、done 和 info

paper-faithful reward 在 `paper_faithful_env.py:274` 定义为：

```text
r_t = previous_makespan - current_makespan
```

它直接鼓励降低当前估计 makespan。基础环境还保留非法动作、通信和终止惩罚等工程项，但 paper-faithful 配置关闭工程 reward，因此报告论文机制结果时应以 `paper_reward` 为准。

`done` 可能由以下条件触发：

- 所有活动任务完成；
- 没有存活 worker；
- 系统无可推进工作而 stalled；
- deadline；
- 达到最大决策次数。

`info` 保存 makespan、当前时间、同步状态、通信量、完成任务、leader、终止原因、cache age 等诊断数据。

## 3. 从 episode 到 PPO 更新

入口：`train_paper_faithful.py`。

### 3.1 rollout

`collect()` 从约 128 行开始：

1. 按确定性的实例 seed bank 创建环境；
2. `reset()` 后循环调用 `model.act()`；
3. 保存 observation、action、old log probability、value 和 reward；
4. episode 结束后计算 GAE 与 return；
5. 收集到至少指定 rollout steps 后组成 batch。

### 3.2 GAE

`gae()` 从约 116 行开始，反向计算：

```text
δ_t = r_t + γ V(s_{t+1}) − V(s_t)
A_t = δ_t + γλ A_{t+1}
R_t = A_t + V(s_t)
```

`γ` 决定未来奖励权重，`λ` 在偏差和方差间折中。当前实现对完整 episode 计算，因此末端 `next_value=0`。

### 3.3 PPO update

训练循环从约 421 行开始。每轮：

1. 标准化 advantage；
2. 多个 update epoch、minibatch 重放同一 rollout；
3. 计算新旧 log probability 比率；
4. 使用 clipped surrogate actor loss；
5. critic 回归 return；
6. entropy 保持探索；
7. 记录 approximate KL、clip fraction 和 Gate gradient norm。

核心损失在 450～455 行。`clip fraction` 很高表示大量样本触发裁剪；approximate KL 很大表示新策略偏离旧策略较多，但二者是诊断量，不是单独的“越低越好”指标。

### 3.4 validation、checkpoint 与 test100

`validate()` 从约 211 行开始，使用 deterministic action 在固定 validation seed bank 上评估。

每到 validation interval：

1. 保存当前 candidate；
2. 比较 validation realized makespan；
3. 选择 makespan 最低且符合 warmup 条件的 `best_state`；
4. 训练结束后把 `best_state` 写入 `checkpoint.pt`。

test100 是冻结选模之后，在独立固定测试实例上的最终比较。不能反过来使用 test100 选 checkpoint，否则会产生测试集泄漏。

## 4. 300 轮结果的标准回答

### 4.1 GPPO 相对 PPO

- GPPO-event：16.1673；
- PPO-none：17.1720；
- PPO-event：17.0290；
- GPPO-event 相对 PPO-none 平均差：−1.0046，95% CI `[−1.3246, −0.6847]`；
- 相对 PPO-event：−0.8617，95% CI `[−1.1529, −0.5706]`。

在固定 seed1 checkpoint 和 test100 上，图结构版本明显优于两个 PPO baseline。这支持“图表示在当前环境与配置中产生收益”，但不是多训练种子的普遍性结论。

### 4.2 Event 相对 Full

- 两者同 checkpoint 回放 makespan 均为 16.1673；
- Event：2,790.9 bytes；
- Full：4,266,481 bytes；
- Event 减少约 99.93% 通信量。

结果相同的原因不是 Event 和 Full 理论上总等价，而是当前 event tape 中关键事件同步已经提供了会影响动作的必要信息，额外广播没有改变动作序列。

### 4.3 Adaptive Gate 的负结果

- NoGate-event：15.9732；
- SingleHead-event：15.9389；
- Adaptive 相对 NoGate：+0.1942；
- Adaptive 相对 SingleHead：+0.2284。

正差表示 Adaptive 更慢。Gate 有梯度并有非退化分布，因此不是“没工作”；更可能是额外自由度、单种子方差、门控位置或优化难度没有转化为泛化收益。必须保留这个负结果。

### 4.4 Literal 与 Best

- `GPPO-Literal`：按论文公式结构保留的机制复现版本，用于回答“论文机制是否被实现”；
- `GPPO-Best`：按预注册三种子 validation-A 筛选出的工程配置，用于后续稳定基线。

不能使用 Best 的结果替代 Literal 的论文复现结论。

## 5. 第二篇论文需要掌握到什么程度

### 5.1 MOMDP 与向量 reward/value

多目标环境的单步奖励是向量：

```text
r_t = [r_t^(1), r_t^(2), …, r_t^(K)]
```

向量 value 对每个目标分别估计未来回报。给定偏好向量 `w` 后，可以通过标量化或更复杂的多目标优化得到当前偏好下的决策目标。

### 5.2 preference-conditioned policy

策略和 critic 同时接收状态与偏好：

```text
π(a | s, w),    V(s, w) 或 V⃗(s, w)
```

同一个模型可通过改变 `w` 产生不同目标权衡，而不必为每个偏好重新训练一个策略。

### 5.3 Pareto front 与 PreCo

Pareto front 由互不支配的目标解组成。线性加权可能漏掉非凸区域，因此不能只用若干固定权重下的加权和判断覆盖质量。

PreCo 的概念重点是：寻找多个目标共同有利的更新成分，同时使用偏好相似性梯度保持策略对输入偏好的响应。当前只需理解思想、输入输出和评价指标，不需要运行作者代码。

### 5.4 与 RLHF 的区别

- PCRL 的 `w` 是显式给定的数值偏好；
- RLHF 的偏好通常来自人类或 judge 对回答的比较；
- RLHF 可能先训练 reward model，再用 PPO/GRPO 等优化策略，也可能直接做偏好优化；
- 两者可组合，但 preference vector 不能直接等同于人类偏好模型。

## 6. 不看材料自测

每题用 1～2 分钟口述，不要求背代码行号，但要讲清因果关系。

### A. 第一篇论文与环境

1. 为什么该问题适合异构图，而不是普通固定长度向量？
2. DAG 前序约束如何进入 observation、edge 和 action mask？
3. 为什么 no-op 不是任何时候都允许？
4. belief mask 与 true mask 不一致会发生什么？
5. 离散事件推进和固定时间步有什么区别？

### B. 模型与训练

6. Eq.(1)～(3) 如何更新 UAV 表示？
7. Eq.(4)～(5) 如何更新 task 表示？
8. pair actor 为什么需要 UAV、task 和 edge feature？
9. critic 输出什么，怎样参与 GAE？
10. PPO clip、entropy、KL 和 clip fraction 分别在监控什么？

### C. 通信与重分配

11. event、periodic、always、none 的区别是什么？
12. heartbeat 为什么不等于全量同步？
13. leader 或 UAV 故障后，任务如何重新进入可分配状态？
14. Event 与 Full 结果相同为什么不是理论等价证明？

### D. 结果和边界

15. GPPO 相对 PPO 的核心数值证据是什么？
16. Gate 的负结果是什么，为什么仍不能说 Gate 永远无效？
17. Literal 和 Best 分别回答什么问题？
18. 为什么当前只能声称 mechanism/structural reproduction？

### E. 后续知识

19. PPO 与 GRPO 在 advantage 来源和 critic 依赖上有什么主要区别？
20. PCRL preference vector 与 RLHF human preference 有什么区别？

## 7. 自测通过标准

现阶段完成不能只按“文档已读”判断。建议采用以下标准：

- 20 题中至少 16 题能独立回答；
- 第 2、6、7、13、14、16、18、19、20 题不能出现原则性错误；
- 能用 5～8 分钟完整讲一次 episode；
- 能在代码中定位环境、模型、训练和评估四个入口；
- 能明确说出三条正结果、两条负结果和三条复现边界；
- 面对“为什么现在不继续训练”时，能说明现有证据已经覆盖什么、什么条件下才需要补实验。

达到以上标准，才说明开学前“理解和总结已有成果”的目标真正完成。

## 8. 会议前最后清单

- [ ] 不看报告讲完一次 episode；
- [ ] 对着模型代码解释 Eq.(1)～(5)；
- [ ] 解释 300 轮六模型表格和三组置信区间；
- [ ] 解释 Event/Full 的同 checkpoint 因果重放；
- [ ] 主动报告 Gate 负结果；
- [ ] 说明当前 DAG 和复现边界；
- [ ] 区分 PPO、GPPO、GRPO、PCRL 和 RLHF；
- [ ] 准备好四个会议决策问题；
- [ ] 不在会议前启动新的长训练或第二篇复现。

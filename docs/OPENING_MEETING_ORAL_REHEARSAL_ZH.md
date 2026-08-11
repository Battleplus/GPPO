# 开学会议 5～8 分钟试讲与追问准备

## 1. 试讲目标

这不是逐字背诵稿。练习时先按本稿讲一遍，再只看标题复述，最后完全不看材料讲一次。

合格的试讲应让学长或老师在 5～8 分钟内听明白：

1. 第一篇论文解决什么问题；
2. GPPO 的完整决策链路；
3. 当前代码复现到什么程度；
4. 300 轮结果支持什么、不支持什么；
5. 为什么现在不继续训练；
6. 后续 GRPO/RLHF 研究还需先确定什么。

## 2. 5～8 分钟试讲稿

### 第 0～1 分钟：当前阶段和问题定义

我现在做的不是继续增加训练规模，而是基于已经完成的 300 轮实验，把第一篇论文的方法、代码和结果系统梳理清楚。

第一篇论文研究弱通信条件下的多无人机动态任务分配。系统中 UAV 能力不同，任务会动态出现，一个任务还可能包含具有前序依赖的子任务。因此策略需要在信息可能过时的情况下，决定哪架 UAV 执行哪个当前可行的子任务，并尽量降低整体 makespan。

这个问题适合用异构图表示：一类节点是 UAV，一类节点是子任务；边包括 UAV 间通信、任务前序与后继、UAV 与任务的能力关系以及 self edge。

### 第 1～2.5 分钟：AHGNN 与动作产生

环境首先把 belief state 转换成节点特征、关系类型、边特征和动态 action mask。

AHGNN 的 Eq.(1)～(3) 主要更新 UAV 表示。每架 UAV 在自身信息和相关任务消息之间计算注意力；Adaptive Gate 调节任务消息进入聚合的强度。Eq.(4)～(5) 更新任务表示，分别聚合 UAV、前驱、后继和 self 四类关系，再通过融合网络得到新的任务 embedding。

Actor 将 UAV embedding、task embedding 和对应边特征拼接，为每个 UAV-task 配对输出 logit。action mask 会把不合法配对的 logit 设成极小值。一个动作只有在 UAV 空闲、任务未完成且未分配、前序已完成、能力匹配时才合法。没有任何合法配对时才允许 no-op。

Critic 输出标量 value，用于 PPO 的 advantage 估计。所以 GPPO 可以理解为“图结构策略表示 + 动态动作约束 + PPO 更新”，这里的 G 是 Graph，不是 GRPO 的 Group Relative。

### 第 2.5～4 分钟：一次 episode 与弱通信

一次 episode 从 `reset()` 开始。环境根据固定 seed 生成 UAV、任务、前序关系和 event tape，然后返回 belief observation。

模型从 mask 后的 categorical distribution 中选择 UAV-task 动作。环境同时检查 belief mask 和真实状态 mask：合法时应用分配；如果当前没有可继续分配的任务，就把仿真时间直接推进到最近的任务完成、heartbeat、外生事件、同步或 deadline，而不是固定每次加一。

任务完成后，系统释放 UAV，并解锁满足前序条件的后继任务。若 UAV 故障，其占用任务重新回到未分配状态，随后通过新的 action mask 参与重分配。leader 负责当前协调信息；heartbeat 用来确认成员存活，但 heartbeat 不等于全量状态广播。

事件触发模式只有出现关键事件时才同步 belief cache。Full/always 模式则每个决策步都同步。策略真正读取的是 belief，而 true state 只用于环境执行和审计，因此这里确实存在弱通信下的信息过时。

### 第 4～5 分钟：PPO 训练、选模和测试

rollout 保存 observation、action、旧策略 log probability、value 和 reward。paper-faithful reward 使用前后 makespan 的下降量。episode 结束后用 reward 与 value 计算 GAE 和 return。

PPO 更新时计算新旧策略概率比率，使用 clipped actor loss 限制单次更新幅度；critic 用 value loss 拟合 return；entropy 保持探索。训练历史还记录 approximate KL、clip fraction 和 Gate gradient norm，用于判断更新幅度和 Gate 是否收到梯度。

checkpoint 不是直接按训练 reward 选择，而是在固定 validation 集上按 realized makespan 选出 best state。最后才在独立的固定 test100 上报告结果，不能使用 test100 反向选模型。

### 第 5～6.5 分钟：300 轮结果

在 T5-10-48、训练 seed1、300 iterations 和固定 test100 上，GPPO-event 的 makespan 是 16.1673，PPO-none 是 17.1720，PPO-event 是 17.0290。GPPO-event 分别降低 1.0046 和 0.8617，配对置信区间都没有跨过零。因此在当前固定训练条件下，图结构相对 PPO baseline 有明确收益。

同一个 GPPO checkpoint 在 Event 和 Full replay 下 makespan 都是 16.1673，但通信量从 4,266,481 bytes 降到 2,790.9 bytes，约减少 99.93%。这说明当前 event tape 中关键同步已经足以维持动作结果，不代表所有弱通信强度下 Event 与 Full 都等价。

必须保留的负结果是 Adaptive Gate 在 seed1 test100 上没有优于 NoGate 和 SingleHead。Adaptive 比 NoGate 慢 0.1942，比 SingleHead 慢 0.2284。Gate 的均值、方差和梯度都不是零，因此不能说实现失效；更准确的说法是它参与了学习，但在这次单种子训练中没有转化成更好的泛化结果。

### 第 6.5～8 分钟：复现边界与下一步

当前工作属于机制或结构复现，不是论文数值级复现，因为原论文完整环境、代码和随机细节无法全部获得。当前实现还把任务 DAG 限制为每个任务槽最多一个直接前驱。因此我可以报告机制链路、对照结果和通信收益，但不能声称完全复现原论文数值，也不能用 seed1 证明某个结构普遍有效或无效。

目前不需要再跑同类 300 轮训练。第二篇 PCRL 只做多目标、偏好条件策略、向量 reward/value、Pareto front 和 PreCo 的概念阅读。

后续是否改进 GRPO/RLHF，需要会议先确定研究对象、偏好数据来源、准备修改的算法接口和评价指标。现阶段只保留图结构奖励模型、图感知分组、事件触发式偏好查询、多目标相对奖励和偏好条件化策略等候选方向，不提前断言哪一个一定有效。

## 3. 老师可能追问的问题

### 3.1 为什么不能直接用 MLP？

不是说 MLP 理论上不能处理，而是异构图更自然地编码 UAV、任务以及不同关系，并允许动态任务数量和关系变化。当前 test100 中 GPPO 相对 PPO-MLP 有收益，支持图结构在当前设置中有用；但仍需多训练种子才能强化普遍性结论。

### 3.2 你们真的实现了 DAG 吗？

实现了前序/后继关系、前序完成解锁和 mask 约束，但当前 paper-faithful adapter 每个任务槽最多接受一个直接 predecessor，是一般 DAG 的受限形式。不能说支持任意多前驱 DAG。

### 3.3 Event 与 Full 结果完全相同，是不是通信根本没用？

不是。相同结果来自同一 checkpoint、同一 test100、同一事件 tape 的因果回放，说明 Event 已同步了当前动作需要的关键信息。若完全关闭或改变丢包、时延和事件分布，belief 可能更旧，动作就可能不同。

### 3.4 Gate 为什么更差？

当前证据只能说明单 seed1 test100 上 Adaptive 泛化不如 NoGate/SingleHead。可能原因包括额外自由度、门控位置、训练方差、优化难度或过拟合。Gate 有梯度，所以不能归因于“代码没连上”；也不能用一个 seed 宣称 Gate 普遍无效。

### 3.5 为什么三种子筛选又选择了 Adaptive-score？

因为两个实验回答的问题不同。300 轮 Literal 是论文结构复现和主机制对照；三种子 validation-A screening 是工程选模。前者不能被后者覆盖，所以项目同时保留 GPPO-Literal 和 GPPO-Best。

### 3.6 makespan reward 会不会被策略钻漏洞？

需要结合完成率、非法动作数和终止原因检查。当前六个学习模型完成率为 1，非法动作审计为 0，因此没有看到通过不完成任务来获得较好 makespan 的情况。仍不能只看单一 reward 曲线判断策略质量。

### 3.7 为什么现在不补五种子？

当前阶段目标是理解流程，现有 300 轮已经覆盖训练、选模、test100、图/PPO 对照、Gate 消融和通信回放。五种子正式矩阵属于论文级统计扩展，应在老师确认研究问题和算力投入后再做，否则会重复采集同类证据。

### 3.8 GPPO 和 GRPO 有什么关系？

名称相似但含义不同。当前 GPPO 是 graph-based PPO，仍用 critic 和 GAE。GRPO 通常针对同一输入采样一组输出，使用组内相对奖励估计 advantage，并减少对传统 critic 的依赖。第一篇论文可以提供图表示、动态 mask 和事件触发等启发，但不能直接把 GPPO 当成 GRPO 改进。

### 3.9 PCRL 的偏好和 RLHF 偏好一样吗？

不一样。PCRL 输入显式数值 preference vector，控制多个环境目标的权衡；RLHF 的偏好通常来自人工或 judge 的成对比较，用来学习 reward model 或直接优化策略。二者可能结合，但数据来源和数学对象不同。

### 3.10 如果老师要求马上选一个方向，你建议什么？

先确定实际痛点：标注昂贵就研究事件触发式偏好查询；组内 advantage 噪声大就研究图/难度感知分组；奖励缺少结构就研究图结构 reward model；目标冲突就研究多目标相对奖励。无论选择哪条路线，都先建立标准 GRPO baseline，再只加入一个明确机制和对应消融。

## 4. 容易说错的表述

| 不建议说 | 应该说 |
|---|---|
| 我们完全复现了论文 | 我们完成了关键机制的结构复现 |
| Gate 没有作用 | Gate 有梯度，但 seed1 test100 未产生泛化收益 |
| Event 和 Full 等价 | 当前固定 checkpoint 和 event tape 回放结果相同 |
| GPPO 已经证明图一定更好 | 当前固定配置与 test100 上图版本优于 PPO baseline |
| 300 轮已经足够发表 | 300 轮足够当前学习阶段，不等于正式多种子统计完成 |
| PCRL 就是 RLHF | PCRL 是显式偏好条件多目标 RL，RLHF 偏好通常来自比较数据 |
| 下一步一定做图 GRPO | 图 GRPO 只是候选，需先确认问题、数据和评价标准 |

## 5. 试讲评分表

每项 0～2 分：0 分为未提及或错误，1 分为大致正确但因果不清，2 分为准确且能说明边界。

| 项目 | 分数 |
|---|---:|
| 问题定义与异构图 | /2 |
| DAG 与动态 mask | /2 |
| AHGNN Eq.(1)～(5) | /2 |
| Actor-Critic 与 PPO | /2 |
| 一次 episode | /2 |
| Event、leader、heartbeat、belief | /2 |
| 300 轮正结果 | /2 |
| Gate 负结果 | /2 |
| 复现边界 | /2 |
| GRPO/PCRL/RLHF 区分 | /2 |

建议 16/20 以上且“DAG 与 mask”“一次 episode”“复现边界”均不低于 1 分，才算试讲通过。

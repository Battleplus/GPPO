# GPPO-v2 论文机制对齐表

## 复现边界

本实现依据 Yu et al., *Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication*（IEEE TASE, 2025）的公开正文。论文未公开代码、环境和实例生成器，因此 GPPO-v2 是公式与机制级独立复现，不是论文数值级复现。

下表使用四档证据状态：

- `公式支持`：论文公式直接规定了核心运算。
- `正文语义支持`：正文明确描述机制，但未给出完整计算细节。
- `兼容实现选择`：与论文语义一致，但论文未唯一规定该实现。
- `工程扩展`：为弱通信实验、固定张量接口或验收协议增加的设计，不能归因于原论文。

AHGNN Eq. (3) 中自适应网络 `f_ijk` 的精确输入、输出范围、排版作用域以及 `e'` 的归一化域存在歧义。正文仅明确它按任务-UAV 对动态调整注意力。本实现预先固定为每条 UAV-task 边的 sigmoid gate，仅调制 task message，不对 `alpha * gate` 二次归一化，并将 self 与 task 邻居置于同一 softmax 域；这些均是预先冻结的兼容实现选择，正式结果产生后不得调整解释。

论文对事件触发只给出高层语义，包括任务完成公告、UAV 故障或新增、任务增删以及 leader 失效后的接任。论文没有定义 timeout、action conflict、通信阈值、结构化事件 schema 或连续时间事件库；当前实现中的这些内容必须始终标为工程扩展。

第一阶段 AHGNN 还包含 relation edge bias、输出投影、残差和 LayerNorm，并用这一输出栈实现稳定的确定性网络；这些细节不是 Eq. (3) 的字面规定，尤其不能把它们等同于公式中的显式 `sigma`。它们属于机制兼容的工程网络设计。

## 公式 - 模块 - 代码 - 测试

| 机制 | 论文证据 | 当前实现 | 代码 | 主要验证 | 证据状态 |
| --- | --- | --- | --- | --- | --- |
| UAV、任务两类节点 | Sec. III-A, Fig. 2，PDF p.5 | 固定容量异构节点张量，padding 节点不参与 pooling | `paper_env.py::_build_observation` | 节点、padding、pooling 测试 | 两类节点受正文/图支持；固定容量与 padding 为工程接口 |
| UAV-task 能力关系 | Sec. III-A；Eq. (6)-(7)，PDF p.5-7 | 能力不足不建边、不进入动作 mask；每类任务至少一架 UAV 可执行，同时保证存在能力缺边 | `paper_env.py::_sample_uav`, `_build_observation`, `_valid_mask_for` | `test_capability_graph_is_sparse_but_every_task_type_is_covered` | 正文语义支持；实例分布为工程选择 |
| 任务前驱、后继关系 | Sec. III-B.2, Eq. (5)，PDF p.7 | predecessor/successor 两类有向边 | `paper_env.py::_build_observation` | 双向前序关系、解锁测试 | 公式支持 |
| UAV 第一阶段加性注意力 | Eq. (1)-(3)，PDF p.6 | `c^T[W_U u || W_T t || W_E e]`；使用确定性 `LeakyReLU(0.2)` 近似训练态随机 RReLU，并增加 relation bias、输出投影、残差和 LayerNorm | `paper_models.py::AdaptiveUAVAttention` | score、mask、有限输出测试 | 核心加性注意力受公式支持；激活与输出栈为兼容工程选择，不是 Eq. (3) 字面复现 |
| task-edge 扩展表示 | Eq. (1)-(3)，PDF p.6 | task 和 edge 独立投影后拼接；论文只定义扩展表示 `mu_ijk`，未规定该分解 | `paper_models.py::AdaptiveUAVAttention` | edge 扰动与输出变化测试 | 兼容实现选择 |
| UAV self 与 task message | Eq. (1)-(3)，PDF p.6 | self、task 使用独立 value 变换；self 与可执行任务邻居置于同一 softmax 域 | `paper_models.py::AdaptiveUAVAttention` | 独立变换、attention 域测试 | 核心消息关系受公式支持；独立 value 变换和联合归一化域为兼容实现选择 |
| 自适应权重 `f_ijk` | Eq. (3)，PDF p.6 | task-edge sigmoid gate，仅作用于 task message | `paper_models.py::AdaptiveUAVAttention.gate` | adaptive/no-gate 参数路径匹配消融 | 兼容实现选择 |
| 任务优先级参与注意力 | Sec. III-B.1，PDF p.6 | 优先级同时进入 task node embedding，并作为显式 score/gate 输入 | `paper_models.py::AdaptiveUAVAttention` | priority 干预测试 | 正文语义支持；显式加分项为实现选择 |
| UAV communication 关系 | Sec. II-A，PDF p.4；IV-B/Fig. 5，PDF p.9 | 使用独立 communication softmax、value 和 fusion 支路，不进入 Eq. (1)-(3) 的 task attention 归一化 | `paper_models.py::AdaptiveUAVAttention` | 通信边不改变 task alpha 测试 | 通信关系受正文语义支持；独立注意力与 fusion 支路为工程扩展 |
| 任务第二阶段 UAV 求和 | Eq. (4)，PDF p.7 | 相邻 UAV 更新表示求和 | `paper_models.py::AdaptiveTaskUpdate` | 关系聚合测试 | 公式支持 |
| 任务第二阶段五路 MLP | Eq. (5)，PDF p.7 | 前驱、后继、UAV、自身四路处理后 fusion；层数、宽度、残差和 LayerNorm 由本实现固定 | `paper_models.py::AdaptiveTaskUpdate` | 路径与端到端输出测试 | 核心关系为公式支持；网络细节为实现选择 |
| UAV-task 配对动作 | Sec. IV-A.2，PDF p.8-9 | 固定 `max_uavs * max_tasks + noop` 编码，mask 后语义等价于动态合法 pair 集合 | `paper_env.py::n_actions`, `_valid_mask_for` | shape、mask、noop 测试 | 合法 `(subtask, UAV)` 配对语义受正文支持；固定 padding 编码与 noop 为工程扩展 |
| 前序、能力、忙闲掩码 | Eq. (14) 及 Action 正文，PDF p.8-9 | belief 视角动态合法动作集 | `paper_env.py::_valid_mask_for` | 前序、能力、忙闲、stale mask 测试 | 正文语义支持；不能全部归因于 Eq. (14) |
| 飞行与执行时间 | Eq. (7)，PDF p.7 | 距离/速度 + workload/capability；天气乘子改变执行时间 | `paper_env.py::_execution_components` | 可手算 processing time 和异步完成测试 | `平均飞行时间 + 执行时间` 受正文支持；具体函数和天气为工程扩展 |
| 未分配任务完成时间估计 | State 正文、Eq. (8)-(9)，PDF p.7-8 | 对可执行 UAV 候选完成时间取平均，并递归考虑前驱；未知未来共享队列仍是近似 | `paper_env.py::_estimated_task_completion` | 双 UAV 可手算均值测试 | 正文只支持使用前驱完成或平均 processing time 进行估计；候选完成时间均值与递归细节为兼容实现选择 |
| makespan 定义 | Eq. (15)-(16)，PDF p.9 | 所有 active task 预计完成时间的最大值 | `paper_env.py::makespan` | 已分配/未分配任务测试 | 公式支持 |
| 主奖励 | Eq. (17)，PDF p.9 | `R_t = M_(t-1) - M_t` | `paper_env.py::step` | 奖励差分测试 | 差分公式支持；数值依赖上述自定义 M 估计器 |
| leader-follower 与重新选举 | Sec. II-A，PDF p.4；IV-B/Fig. 5，PDF p.9 | 首个存活 UAV 接任；相当于对论文“后续编号 UAV”规则增加跳过失效节点 | `paper_env.py::_first_alive_uav`, `_sample_event` | leader failure/election 测试 | 正文语义支持；跳过失效节点为推广 |
| 固定间隔 heartbeat | Sec. II-A，PDF p.4；IV-B/Fig. 5，PDF p.9 | 独立于状态同步发送；成功 heartbeat 直接刷新 belief freshness；timeout 只记录一次 | `paper_env.py::_process_heartbeats` | heartbeat freshness、timeout 测试 | 固定 heartbeat 受正文支持；阈值、计数和 timeout 语义为工程选择 |
| 外生事件中断 | 动态事件语义；Fig. 5 | completion、外生扰动、heartbeat、periodic、deadline 使用竞争时间边界；长任务可在中途被事件打断 | `paper_env.py::_next_time_boundary_delta`, `_sample_event` | `test_scheduled_exogenous_event_interrupts_long_running_task` | 工程扩展 |
| 事件触发同步 | Sec. II-A，PDF p.4；Fig. 5，PDF p.9 | completion/unlock、失效/reallocation、leader、timeout、通信、任务变化和 action conflict 形成结构化事件库 | `paper_env.py::step`, `_sample_event` | 各事件记录和同步原因测试 | 任务完成、UAV/任务变化和 leader 失效等高层触发语义受正文支持；timeout、action conflict、具体事件 schema 与时间机制均为工程扩展 |
| 弱通信 stale cache | Sec. IV-B，PDF p.9 | 仅刷新 leader 可达或事件明确报告的实体；冲突动作产生反馈并推进物理时钟，避免零时间死锁 | `paper_env.py::_synchronize_belief`, `step` | partial sync、stale conflict 回归测试 | 弱通信思想受正文支持；belief cache 为工程模型 |
| 四种通信模式 | 原论文 event/受限通信讨论 | `none`、`event`、`periodic`、`always`；`always` 为全状态 Oracle | `paper_env.py::_should_sync` | 四模式和 `sync_reason` 测试 | 消融协议扩展 |
| hard deadline 与非饱和指标 | 原论文未使用该验收指标 | 四规模独立 deadline，区分最终完成率与截止完成率 | `configs/gppo_v2_hard.json`, `paper_env.py::metrics` | nonsaturation gate | 工程验收协议 |

## 公平性与可解释性约束

- `none`、`single_head`、`adaptive_no_gate`、`adaptive` 的 checkpoint 总参数量一致，是因为模型实例化了相同模块集合；实际前向有效路径容量不同，必须同时报告 `parameter_audit.json`，不能把总参数相同表述为容量完全公平。
- `adaptive` 与 `adaptive_no_gate` 使用相同有效参数路径，只将 `f_ijk` 固定为 1，是 gate 的参数路径匹配控制。
- `single_head` 与 `adaptive` 的差异同时包含结构和有效容量变化，只作为补充结构对照，不作 gate 的纯因果解释。
- 所有学习方法共享 actor/critic 输出头、环境、动作 mask、奖励、训练预算、训练规模和训练种子。
- `staged` 仅是 v1 checkpoint 的兼容别名，不作为 v2 论文机制名称。
- 五训练种子的双侧 exact sign-flip 检验最小 p 值为 0.0625；因此它只能作为方向一致性和描述性证据，不能在 0.05 水平产生显著结论。正式报告必须保留这一功效限制。

## 独立验收协议，不等同于论文实验设置

- 原论文公开场景包括 T5-10-48、T10-10-53、T15-8-66、T20-10-92，并报告每环境 100 实例、5 次实验；Table II 的训练超参数包括 2000 最大迭代和 batch size 512。正文 generalization 段另出现 `T15-08-62`，与前述 `T15-8-66` 不一致，复现时不能自行假定二者等价。
- GPPO-v2 hard 协议使用 `2x12`、`3x16`、`3x20`、`4x24`，100 updates、18 episodes/update、5 seeds。它用于验证当前独立实现的稳定性和消融关系，不声称复现论文表格数值。
- 原论文没有公开 mission deadline、belief cache、结构化事件 schema、连续时间 hazard、固定 padding 动作编码或 Oracle 通信模式；这些均必须标为工程实现或验收扩展。
- 正式 v2 关闭非法动作、通信成本和 deadline 等工程奖励，只保留 Eq. (17) 的 makespan 差分；通信、无效动作和 deadline 仅作为评价指标。

## 参考证据

- 论文 PDF：用户提供的 `Multi-UAV_Dynamic_Task_Assignment_Based_on_Event-Triggered_Graph_Reinforcement_Learning_Under_Weak_Communication.pdf`。
- 页面渲染：`tmp/pdfs/multi_uav-05.jpg`、`multi_uav-06.jpg`、`multi_uav-07.jpg`。
- 可检索正文：`tmp/pdfs/multi_uav.txt`，重点为 Sec. III-B 至 IV-B、Eq. (1)-(17)。
- 环境与模型测试：`tests/test_gppo_v2.py`、`tests/test_gppo_v2_env_regressions.py`、`tests/test_gppo_v2_ahgnn_alignment.py`。

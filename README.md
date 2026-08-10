# 弱通信多无人机动态任务分配：GPPO 机制复现与预测偏好决策路线

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
![Status](https://img.shields.io/badge/Phase%201A-GPPO%20baseline%20frozen-2E8B57)
![Scope](https://img.shields.io/badge/reproduction-mechanism--level-orange)

本项目研究**弱通信和多源扰动条件下，异构无人机集群的动态任务分配**。当前分支首先复现论文 *Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication* 中的异构任务图、AHGNN、动作掩码、事件触发通信和 GPPO，再逐步扩展到多源扰动、偏好奖励和图世界模型。

> 原论文未公开完整环境、实例生成器、训练代码和随机细节。本仓库提供的是**可审计的机制级/结构级复现**，不宣称原论文数值级复现。

## 项目现在做到哪里

```mermaid
flowchart LR
    A["Phase 1A<br/>GPPO 基线复现"] --> B["Phase 1B<br/>多源扰动环境"]
    B --> C["Phase 2<br/>规则-导师偏好学习"]
    B --> D["Phase 3<br/>图世界模型"]
    C --> E["Phase 4<br/>风险增强图与预测触发"]
    D --> E
    E --> F["Phase 5<br/>对比、消融与泛化"]
    F --> G["Phase 6<br/>代码、数据与论文"]

    classDef done fill:#d9f2e6,stroke:#218c5a,color:#123b29,stroke-width:2px;
    classDef next fill:#fff0c7,stroke:#d89000,color:#5a3a00,stroke-width:2px;
    classDef future fill:#eef2f7,stroke:#7b8794,color:#27313a;
    class A done;
    class B next;
    class C,D,E,F,G future;
```

| 阶段 | 状态 | 当前结论 |
|---|---|---|
| Phase 1A：GPPO 基线 | **已冻结，带局限完成** | 六模型 300 轮机制验证、候选重评估、Gate 诊断和三种子筛选均已完成 |
| Phase 1B：多源扰动 | **下一优先级** | 已完成能力审计和接口预注册；正式 disturbance tape 尚未实现 |
| Phase 2：偏好强化学习 | 设计阶段 | 不在当前“无偏好”分支中；不能把标量奖励简单乘偏好权重 |
| Phase 3：图世界模型 | 设计阶段 | 固定 event tape 只用于公平比较，不能冒充 learned world model |
| Phase 4：预测触发 | 未开始 | 依赖扰动标签、模型校准与不确定度 |
| Phase 5：正式实验 | 暂缓全量执行 | 应在算法冻结后分片运行，避免反复消耗计算资源 |

完整的阶段判断见 [`docs/STAGE1_TRANSITION_ARCHIVE_2026-08-10.md`](docs/STAGE1_TRANSITION_ARCHIVE_2026-08-10.md)，机器可读决策见 [`configs/STAGE1_TRANSITION_DECISION_2026-08-10.json`](configs/STAGE1_TRANSITION_DECISION_2026-08-10.json)。

## 当前 GPPO 基线在做什么

```mermaid
flowchart LR
    O["局部可观测状态<br/>UAV / Task / Link"] --> H["异构动态图<br/>前序 / 能力 / 通信关系"]
    H --> G["Literal-AHGNN<br/>Eq. 1-5"]
    G --> P["PPO Actor-Critic"]
    H --> M["动作掩码<br/>前序与能力硬约束"]
    P --> X["Masked policy"]
    M --> X
    X --> A["选择合法<br/>子任务-UAV 配对"]
    A --> E["离散事件环境<br/>异步执行"]
    E --> O
    E --> T{"事件发生?"}
    T -- "是" --> S["同步缓存状态"]
    T -- "否" --> C["继续使用弱通信缓存"]
    S --> O
    C --> O
```

已实现的关键机制：

- 任务/子任务两层结构、DAG 前序约束、异构 UAV 能力与异步执行；
- Literal-AHGNN Eq.(1)–(5)，self 使用独立 `W^T v_k`；
- self 与 task neighbors 共享 attention softmax 域；
- expected-RReLU 正式默认；
- Adaptive、NoGate、SingleHead 和 PPO-MLP 模式；
- `none/event/periodic/always` 通信模式；
- belief mask 与 true mask 分离审计，环境非法动作计数；
- 固定 train/validation/test instance bank 与 event tape；
- 候选 checkpoint、完整续训状态、哈希和恢复审计。

模型的冻结定义见 [`docs/PHASE1_MODEL_DEFINITIONS.md`](docs/PHASE1_MODEL_DEFINITIONS.md)，正式协议见 [`configs/PHASE1_FROZEN_PROTOCOL.json`](configs/PHASE1_FROZEN_PROTOCOL.json)。

## 300 轮机制验证：已经证明什么

实验范围为 `T5-10-48 / training seed 1 / 300 iterations / fixed test100`。六种学习方法使用相同测试实例和 event tape。

| 方法 | realized makespan mean | median | completion |
|---|---:|---:|---:|
| GPPO-event | 16.1673 | 16.0369 | 1.0000 |
| PPO-none | 17.1720 | 17.1472 | 1.0000 |
| PPO-event | 17.0290 | 16.7780 | 1.0000 |
| GPPO-none | 16.0260 | 15.9242 | 1.0000 |
| GPPO-NoGate-event | 15.9732 | 15.8206 | 1.0000 |
| GPPO-SingleHead-event | 15.9389 | 15.8049 | 1.0000 |

![六模型 fixed test100 realized makespan](artifacts/phase1_seed1_300/raw/mechanism_makespan.png)

对照结果：

- Random makespan：`20.4073`；
- Greedy makespan：`17.3752`；
- GPPO-event 相对 PPO-none：`-1.0046`，实例级 95% CI `[-1.3246, -0.6847]`；
- GPPO-event 相对 PPO-event：`-0.8617`，实例级 95% CI `[-1.1529, -0.5706]`；
- 同 checkpoint 的 Event/Full makespan 均为 `16.1673`；
- Event 相对 Full 通信字节减少约 `99.93%`；
- 所有审计实例均完成任务，动作掩码违规和环境非法动作均为 `0`。

这些结果支持：**GPPO 主体机制、异构图、动作掩码和事件通信链路能够按当前协议运行**。权威结果与哈希见 [`artifacts/phase1_seed1_300`](artifacts/phase1_seed1_300)。

### 必须保留的 Adaptive 负结果

原始 task-message Adaptive 并未在单种子 300 轮中优于简单结构：

- Adaptive − NoGate makespan：`+0.1942`，95% CI `[+0.0061, +0.3823]`；
- Adaptive − SingleHead makespan：`+0.2284`，95% CI `[+0.0317, +0.4251]`。

正值表示 Adaptive 更差。Gate 分布、策略敏感度和 PPO probe 梯度表明 gate 并非常数、也不是完全无梯度，因此该负结果不能通过“gate 没有参与训练”简单解释。

进一步完成了七变体、三训练种子、每种 300 轮的预注册筛选。只使用 validation-A 选模后，`Adaptive-score` 以 `15.9025` 的平均 makespan 获得 `GPPO-Best` 资格；NoGate 和 SingleHead 分别为 `15.9805` 和 `15.9613`。这只是正式实验的晋级依据，**仍不是四规模五种子的最终统计结论**。详见 [`reports/GATE_THREE_SEED_SCREENING.md`](reports/GATE_THREE_SEED_SCREENING.md)。

## 结果边界

当前证据可以说明：

- GPPO 基线链路和论文关键机制已经实现；
- 单规模、单训练种子下 GPPO 相对 PPO/Random 呈现正向结果；
- Event 通信可在同 checkpoint 重放中保持质量并显著减少字节；
- Adaptive 的原始实现存在真实、可复核的负结果；
- `Adaptive-score` 已通过三种子 validation-only 筛选，可以进入正式验证。

当前证据不能说明：

- GPPO 已在五训练种子上稳定优于 PPO 或 Greedy；
- Adaptive 已在四种规模上稳定优于 NoGate/SingleHead；
- 多源扰动环境、Preference-GPPO 或图世界模型已经完成；
- 实现复现了原论文公开数值；
- 实例级置信区间可以替代训练种子级置信区间。

## 下一步：先补齐多源扰动环境

现有 `paper_env.py` 中的随机 hazard 只是早期工程原型。Phase 1B 必须以**独立、可关闭、可序列化、可哈希、可重放**的 disturbance tape 接入，并保证关闭全部扰动时与冻结基线行为一致。

```mermaid
flowchart TD
    P["冻结 GPPO 基线<br/>已完成"] --> I["统一 DisturbanceConfig / Tape / Event / Logger"]
    I --> L["链路扰动<br/>Gilbert-Elliott / Delay / Partition"]
    L --> U["UAV 扰动<br/>Failure / Recovery / Energy"]
    U --> K["任务扰动<br/>Arrival / Cancellation / Priority"]
    K --> W["风场扰动<br/>航时与能耗偏差"]
    W --> Q["确定性重放与不变量测试"]
    Q --> R["统一轨迹与未来 1-5 步事件标签"]
    R --> PR["Preference-GPPO"]
    R --> WM["Graph World Model"]
    PR --> RG["风险增强图与预测式触发"]
    WM --> RG
    RG --> EX["冻结算法后执行正式分片实验"]

    classDef done fill:#d9f2e6,stroke:#218c5a,color:#123b29,stroke-width:2px;
    classDef next fill:#fff0c7,stroke:#d89000,color:#5a3a00,stroke-width:2px;
    classDef future fill:#eef2f7,stroke:#7b8794,color:#27313a;
    class P done;
    class I,L,U,K,W,Q,R next;
    class PR,WM,RG,EX future;
```

建议实现顺序：

1. **统一接口**：实现 `DisturbanceConfig`、`DisturbanceTape`、不可变 `DisturbanceEvent` 和 `DisturbanceLogger`；
2. **通信扰动**：Gilbert-Elliott Good/Bad 链、逐包丢失、消息到达队列、网络分区和恢复同步；
3. **UAV 扰动**：永久/临时故障、能量扣减、低能量退化、任务释放与 leader 接替；
4. **任务扰动**：任务新增、取消、优先级和 deadline 变化，并保持 DAG 合法；
5. **风场扰动**：空间风向/风速、航向相关航时和能耗偏差；
6. **可审计测试**：固定 seed 生成字节一致的 tape，检查事件覆盖、缓存语义、动作合法性、能量边界和任务恢复；
7. **轨迹标准化**：输出 partial graph、action、mask、communication history、未来五步事件标签、objective vector 和 terminal metrics。

接口预注册见 [`configs/DISTURBANCE_INTERFACE_PROTOCOL.json`](configs/DISTURBANCE_INTERFACE_PROTOCOL.json)，现有能力与缺口见 [`docs/DISTURBANCE_EXISTING_CAPABILITY_AUDIT.md`](docs/DISTURBANCE_EXISTING_CAPABILITY_AUDIT.md)。

## 后续“预测—偏好—决策”闭环

多源扰动环境稳定后，项目将形成以下闭环：

```mermaid
flowchart LR
    O["部分可观测异构图"] --> WM["图世界模型<br/>未来五步扰动 + 不确定度"]
    WM --> R["风险增强图"]
    R --> PG["Preference-GPPO"]
    P["规则偏好 + 少量导师标注"] --> BT["Bradley-Terry<br/>奖励模型集成"]
    BT --> PG
    PG --> A["合法任务分配动作"]
    A --> ENV["弱通信多源扰动环境"]
    ENV --> O
    WM --> TR["风险 / 分歧超过阈值"]
    ENV --> TR
    TR --> COM["事件触发通信与重分配"]
    COM --> O
```

Preference-GPPO 不应实现为“标量奖励 × 偏好权重”。计划使用任务成功、截止期、makespan、能耗、通信量、重分配次数和方案稳定性等向量指标，构造规则—导师混合偏好对，再训练 Bradley-Terry 奖励模型。

世界模型必须接受独立的 observation/action/history 监督，报告 AUROC/F1、RMSE、Brier、校准误差、提前量和误触发成本。固定 event tape 只服务于公平比较。

文献与实现边界审计见 [`docs/PCRL_JEPA_LITERATURE_AUDIT.md`](docs/PCRL_JEPA_LITERATURE_AUDIT.md)。

## 运行、测试与复核

### 安装

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 测试

```bash
python -m pytest -q
```

### 单模型训练示例

```bash
python train_paper_faithful.py \
  --mode literal \
  --sync-mode event \
  --scale T5-10-48 \
  --seed 1 \
  --iterations 300 \
  --rollout-steps 512 \
  --batch-size 512 \
  --update-epochs 4 \
  --validation-interval 50 \
  --validation-instances 100 \
  --validation-split validation_a \
  --output outputs/example/literal_event_seed1
```

### 固定 test100

```bash
python evaluate_paper_faithful.py \
  --checkpoint outputs/example/literal_event_seed1/checkpoint.pt \
  --instances 100 \
  --split test \
  --trace \
  --output outputs/example/literal_event_seed1/test100.json
```

### Google Colab

- 单规模机制验证：[`colab/GPPO_Mechanism_Validation_Colab.ipynb`](colab/GPPO_Mechanism_Validation_Colab.ipynb)
- Phase-1 正式完整流水线：[`colab/GPPO_Phase1_Full_Colab.ipynb`](colab/GPPO_Phase1_Full_Colab.ipynb)
- [直接在 Colab 打开正式完整流水线](https://colab.research.google.com/github/Battleplus/GPPO/blob/8.8-GPPO%E6%97%A0%E5%81%8F%E5%A5%BD/colab/GPPO_Phase1_Full_Colab.ipynb)

> **计算成本提示：**正式完整流水线最多包含 135 个训练任务 × 2000 iterations，即 270,000 iterations，约 1.38 亿 rollout transitions，另有 validation/test。它用于最终论文统计，不是当前进入多源扰动、PCRL 或世界模型开发的前置任务。建议在算法冻结后按规模和种子拆成多个 Colab 分片执行。

## 仓库导航

| 路径 | 内容 |
|---|---|
| `src/uav_assignment/` | 环境、图编码器、策略和训练核心 |
| `tests/` | 公式、环境、掩码、恢复、CUDA和审计测试 |
| `configs/` | 冻结协议、Gate筛选和扰动接口预注册 |
| `artifacts/phase1_seed1_300/` | 六模型300轮权威机制证据 |
| `reports/` | Gate、候选checkpoint和公式审计报告 |
| `docs/` | 阶段状态、模型定义、矛盾诊断和文献审计 |
| `colab/` | Colab训练、恢复、监控和正式矩阵入口 |

## 复现原则

1. 不使用 test/test100 选择 checkpoint、结构或超参数；
2. 不删除或覆盖 Adaptive 和 Hard Dynamic Extension 的历史负结果；
3. 不把单训练种子的实例级 CI 写成训练种子稳定性；
4. 不把固定事件带写成 learned world model；
5. 不在算法仍会修改时提前消耗完整正式矩阵；
6. 所有正式结果保留配置、种子、checkpoint 哈希、实例库和事件带证据；
7. 只有冻结算法后的多种子对比、消融和泛化实验才能支持最终论文结论。

---

当前最优先任务：**补齐 Phase 1B 多源扰动环境及可审计测试，然后构建 Preference-GPPO 与图世界模型的数据接口；不重复已有六模型 100/300 轮机制实验。**

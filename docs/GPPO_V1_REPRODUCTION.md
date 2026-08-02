# GPPO-v1 论文结构对齐复现

## 结论摘要

后续 `100x16` 五种子稳定性复核已完成，最新准入结论和结果见 `docs/GPPO_V1_STABILITY_REVIEW.md`。本文件保留最初 `50x8` 结构验收结果作为实验历史。

本项目完成的是结构级复现，不是论文数值级复现。参考论文没有公开原始环境、实例生成器和训练代码，因此不能声称复现论文表格中的绝对数值。

在独立实现的弱通信异步调度环境中，事件触发 GPPO 与同环境、同预算、同五个训练种子的无图 PPO 进行比较：

- `3x12`、`4x20`、`5x28`、`6x36` 四个规模的 makespan 差异显著为负；
- `2x8` 是预先保留的负结果，GPPO makespan 反而更差；
- GPPO 完成任务数在五个规模都更高；
- 因此满足“多数规模优于公平 PPO 对照”的验收条件，但不等于超过启发式贪心，也不等于论文数值复现。

在此基线达到稳定验收前，不进入 PCRL 或世界模型集成。

## 参考论文对齐点

依据 `Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication`（Yu, Zhang, Sun, IEEE TASE, 2025）正文：

1. UAV 和任务作为异构图节点，保留 UAV-任务能力关系、任务前序关系及 UAV 通信关系。
2. UAV 节点和任务节点分阶段更新，任务节点使用前序、后继和可执行 UAV 信息。
3. 动作是“可用 UAV - 可执行子任务”配对，并使用动作掩码避免冲突。
4. 任务完成时间包含飞行时间与执行时间；所有任务的最大预计完成时间为 `M`。
5. 论文主奖励为 `R_t = M_(t-1) - M_t`。
6. 弱通信采用 leader-follower、动态同步和事件触发；leader 失效后重新选举。

论文原文位于项目外部资料目录；本项目已将可检索文本和页面渲染保存在 `tmp/pdfs/`，便于复核。

## 实现结构

| 组件 | 实现 |
| --- | --- |
| 异步环境 | `src/uav_assignment/paper_env.py` |
| 异构图策略 | `src/uav_assignment/paper_models.py` |
| 训练入口 | `train_paper_gppo.py` |
| 评测入口 | `evaluate_paper_gppo.py` |
| 统计汇总 | `summarize_paper_gppo.py` |
| 单元测试 | `tests/test_paper_env.py` |

环境包含 UAV 忙闲状态、任务开始/完成时间、处理时间、前序解锁、失效重分配、leader/heartbeat、真实状态与 stale belief 缓存。`event`、`periodic`、`always`、`none` 四种同步模式均可独立评测。

主论文奖励默认不包含非法动作、通信成本和终局惩罚。工程惩罚只在 `include_engineering_rewards=True` 时打开，避免污染论文主基线。

GPPO 使用 staged UAV-node -> task-node attention 和边特征；PPO 使用同样的网络宽度、动作空间、奖励、训练预算和实例，只关闭图消息与同步。两者参数量由测试固定相等。

## 正式实验配置

- 容量：`6 UAV / 36 tasks`；
- 训练规模轮换：`3x12`、`4x20`、`5x28`；
- 更新次数：50；每次更新 8 episodes；隐藏维度 64；
- 训练种子：`1..5`；每 10 updates 用 20 episodes 验证，验证种子起点 `40000`；
- 评测种子：`50000..50099`，每个规模 100 episodes；
- 测试规模：`2x8`、`3x12`、`4x20`、`5x28`、`6x36`；
- 正式修正版产物：`outputs/paper_aligned/formal_v2/` 和 `outputs/paper_aligned/formal_eval_v2/`；
- 统计文件：`outputs/paper_aligned/formal_summary_v2/summary.json`。

`outputs/paper_aligned/formal/` 是奖励修正前的历史产物，不用于最终结论。

## 关键结果

下表为事件 GPPO 减去无同步 PPO；makespan 越小越好，完成任务数越大越好。区间是五个训练种子的配对 95% CI。

| 规模 | makespan delta +/- CI | 完成任务 delta +/- CI | 结论 |
| --- | ---: | ---: | --- |
| `2x8` 未见 | `+0.123 +/- 0.070` | `+0.962 +/- 0.081` | GPPO 负结果 |
| `3x12` | `-0.626 +/- 0.098` | `+1.896 +/- 0.041` | GPPO 更好 |
| `4x20` | `-1.164 +/- 0.345` | `+7.396 +/- 0.260` | GPPO 更好 |
| `5x28` | `-1.248 +/- 0.390` | `+10.896 +/- 0.334` | GPPO 更好 |
| `6x36` 未见 | `-1.368 +/- 0.774` | `+16.604 +/- 0.330` | GPPO 更好 |

事件同步的通信代价显著低于周期和全同步，但无效动作更多：在 `6x36` 上通信事件均值分别为 event `5.558`、periodic `10.032`、always `41.002`；无效动作分别为 `19.148`、`0.434`、`0`。这说明事件触发确实节省通信，但当前策略仍需改进 stale belief 下的动作鲁棒性。

贪心基线仍然很强：事件 GPPO 相对贪心的 makespan 只在 `4x20` 占优，在 `2x8`、`3x12`、`5x28`、`6x36` 仍略差。因此当前结论是“相对公平 PPO 的结构增益已出现”，而不是“GPPO 已达到最终算法标准”。

## 复现实验

设置环境：

```powershell
$env:PYTHONPATH = 'src'
$python = 'D:\anaconda\python.exe'
```

训练一个 GPPO 种子示例：

```powershell
& $python train_paper_gppo.py --algorithm gppo --graph-mode staged --sync-mode event `
  --seed 1 --max-uavs 6 --max-tasks 36 --active-uavs 4 --initial-tasks 24 `
  --train-scales 3x12 4x20 5x28 --updates 50 --episodes-per-update 8 `
  --max-decisions 180 --hidden-dim 64 --validation-episodes 20 --validation-interval 10 `
  --validation-seed 40000 --output outputs/paper_aligned/reproduce/gppo/seed_1
```

评测一个 checkpoint：

```powershell
& $python evaluate_paper_gppo.py `
  --checkpoint outputs/paper_aligned/reproduce/gppo/seed_1/checkpoint.pt `
  --scales 2x8 3x12 4x20 5x28 6x36 --episodes 100 --eval-seed 50000 `
  --output outputs/paper_aligned/reproduce_eval/gppo_event/seed_1
```

随机和贪心使用相同配置 checkpoint 生成评测：

```powershell
& $python evaluate_paper_gppo.py --baseline random `
  --config-checkpoint outputs/paper_aligned/reproduce/gppo/seed_1/checkpoint.pt `
  --scales 2x8 3x12 4x20 5x28 6x36 --episodes 100 --eval-seed 50000 `
  --output outputs/paper_aligned/reproduce_eval/random_event
```

汇总：

```powershell
& $python summarize_paper_gppo.py (Get-ChildItem outputs/paper_aligned/reproduce_eval -Filter evaluation.json -Recurse).FullName `
  --output outputs/paper_aligned/reproduce_summary
```

## 验收边界

已通过：环境、图边、异步执行、前序解锁、makespan 奖励、stale cache、参数量公平和失效重分配测试；正式 GPPO/PPO 五种子和多尺度评测已完成。

尚未通过的更高标准：GPPO 尚未稳定超过合理贪心，事件模式无效动作较多，且原论文没有可用于数值复现的公开代码和环境。因此下一阶段应先改进 GPPO-v1 的策略训练和动作鲁棒性，再考虑 PCRL；不能把当前结果包装成论文数值级复现。

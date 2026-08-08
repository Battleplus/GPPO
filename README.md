# Paper-Faithful GPPO for Multi-UAV Task Assignment

本分支是一个独立、无偏好强化学习模块的 GPPO 机制级复现，目标论文为 *Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication*。

原论文未公开完整环境、实例生成器和训练代码，因此本项目不宣称论文数值级复现。当前提交提供可复核的 Paper-Faithful 环境、Literal-AHGNN、PPO 训练、事件通信、消融入口、测试和一个完整 seed 的结果。

## 当前状态（2026-08-08）

- `T5-10-48`，`GPPO-Literal + event`，seed 3 已训练 `2000` iterations；
- 固定 test split 上评估 `100` 个实例，全部任务完成；
- realized makespan：`15.308944 ± 1.433970`（实例标准差）；
- median makespan：`15.140943`；
- 每实例 event communication：`3` 次；
- 平均通信量：`2738.4 bytes`；
- 平均推理时间：`3.255 ms`；
- adaptive gate 梯度 L2：`0.559243`，gate 分支实际参与决策；
- 原工作区完整回归：`274 passed`；本独立精简分支：`32 passed`。

完整结果见 [`artifacts/paper_faithful/T5-10-48/literal_event_seed3`](artifacts/paper_faithful/T5-10-48/literal_event_seed3)。

该结果是单规模、单训练种子证据，只证明训练、推理、事件通信和 gate 诊断链路可运行；它不能单独证明 GPPO 优于 PPO，也不能证明 Adaptive 优于 NoGate 或 SingleHead。

### 100 轮六模型快速机制验证

同一 `T5-10-48`、训练 seed 1、100 iterations、固定 test100 下，六模型快速筛查得到：

| 方法 | realized makespan mean | median |
|---|---:|---:|
| GPPO-none | 15.9666 | 15.8320 |
| GPPO-NoGate-event | 16.0344 | 15.9501 |
| GPPO-SingleHead-event | 16.1166 | 15.9045 |
| GPPO-event | 16.1376 | 15.9449 |
| PPO-event | 17.7635 | 17.7132 |
| PPO-none | 18.3013 | 18.1229 |

GPPO-event 相对 PPO-none 的同实例差值为 `-2.1637`，实例级 95% CI 为 `[-2.4694, -1.8579]`；但 Adaptive 同时落后 NoGate 和 SingleHead，不能支持 adaptive gate 的独立正向贡献。相同 GPPO checkpoint 的 Event/Full 重放具有相同 makespan，Event 通信字节减少 `99.93%`。

因此当前快速判定是：**继续 GPPO 正式复现，但停止宣称 adaptive gate 有益并优先排查该模块**。该实验仍是单规模、单训练种子、短训练预算证据，不能写成“GPPO 已稳定优于 PPO”。完整报告、精简 test100 行、六个 checkpoint 和哈希清单见 [`artifacts/paper_faithful/quick_seed1_100`](artifacts/paper_faithful/quick_seed1_100)。

## 主要实现

- 任务/子任务两层结构、前序约束和异构 UAV 能力；
- 无 hard deadline 的 realized makespan 协议；
- Literal-AHGNN Eq.(1)–(5)；
- 独立 `W^T v_k` self transform；
- self 与 task neighbors 共享 attention softmax 域；
- expected-RReLU 正式默认；
- adaptive、NoGate、SingleHead 和 PPO-MLP 模式；
- `none/event/periodic/always` 通信模式；
- 固定 train/validation/test instance bank 与 event tape；
- 动作掩码、版本化 checkpoint、候选快照恢复和恢复审计。

## 安装与测试

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

## 快速单种子实验

Windows PowerShell 可直接运行六模型验证流水线（最多 4 个并行训练进程）：

```powershell
.\run_paper_faithful_quick.ps1
```

单模型入口示例：

```bash
python train_paper_faithful.py \
  --mode literal \
  --sync-mode event \
  --scale T5-10-48 \
  --seed 1 \
  --iterations 100 \
  --rollout-steps 512 \
  --batch-size 512 \
  --update-epochs 4 \
  --validation-interval 50 \
  --validation-instances 20 \
  --output outputs/pilot/literal_event_seed1
```

测试 checkpoint：

```bash
python evaluate_paper_faithful.py \
  --checkpoint outputs/pilot/literal_event_seed1/checkpoint.pt \
  --instances 100 \
  --split test \
  --trace \
  --output outputs/pilot/literal_event_seed1/test100.json
```

## 复现边界

旧 `GPPO-v2 hard` 结果被保留为 `Hard Dynamic Extension`，不再作为论文原始数值基线。seed 3 在第 1800 轮曾由 legacy candidate 恢复；该次恢复没有旧 Adam、RNG 和 episode offset，恢复信息已写入 checkpoint。后续从版本化 `resume_latest.pt` 恢复时可保存完整优化状态。

PCRL、JEPA 和世界模型不属于本分支。

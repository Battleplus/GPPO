# GPPO-v1 实验协议

## 目标

验证论文中的异构图、异步执行、弱通信缓存和事件触发同步，是否在同一环境、奖励、训练预算和随机种子下为 GPPO 带来相对普通 PPO 的可重复收益。

## 阶段 A: 环境验收

必须通过以下测试：

- UAV-任务能力边、任务前序边和边特征保留；
- 分配动作只让空闲且有能力的 UAV 开始任务；
- 任务不会在分配动作后立即完成；
- predecessor 完成前 successor 不可选；
- makespan 包含未分配 active task；
- `R_t = M_(t-1) - M_t` 在完整 transition 后成立；
- no-sync 不刷新 stale belief；
- 故障会重开运行任务，通信事件刷新缓存，重分配任务最终完成；
- GPPO 与 PPO 参数量相同。

执行：

```powershell
$env:PYTHONPATH = 'src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
D:\anaconda\python.exe -m pytest -q tests/test_paper_env.py
```

## 阶段 B: 公平训练

两种算法必须共享：

- `PaperEnvConfig` 和任务实例生成器；
- `3x12`、`4x20`、`5x28` 轮换训练规模；
- hidden dimension、learning rate、update count、episodes/update；
- seeds `1..5`；
- validation seed range；
- 主奖励和终止规则。

GPPO 使用 `graph_mode=staged`、`sync_mode=event`；公平 PPO 使用 `graph_mode=none`、`sync_mode=none`。周期和全同步只作为通信消融，不作为 PPO 公平主对照。

## 阶段 C: 固定评测

每个 checkpoint 在相同的 `eval_seed=50000..50099` 上评测五种规模：

```text
2x8, 3x12, 4x20, 5x28, 6x36
```

至少报告：

- makespan；
- completed task count 和 completion rate；
- communication events 和 heartbeat messages；
- invalid actions；
- reallocated tasks、reallocation success rate；
- return。

报告模型平均值、五训练种子的 95% CI，并用同种子配对比较 event-GPPO 和 no-sync PPO。随机与贪心没有训练种子，CI 按 100 个固定评测 episode 计算。

## 阶段 D: 同步消融

固定 GPPO checkpoint，分别用 `event`、`periodic`、`always` 评测。主要检查通信次数与任务质量的折衷，而不是只看 reward。

## 当前正式产物

```text
outputs/paper_aligned/formal_v2/
outputs/paper_aligned/formal_eval_v2/
outputs/paper_aligned/formal_summary_v2/summary.json
```

修正前 `outputs/paper_aligned/formal/` 只作为历史记录，不纳入结论。

## 通过条件与停止条件

通过条件：环境测试全通过、GPPO/PPO 各五种子、所有规模固定评测、报告 CI，且 GPPO 在多数规模的 makespan 差异方向正确并可解释。

停止条件：若 GPPO 不优于 PPO，结论写为“论文结构在当前复现环境中未显示优势”；若 GPPO 优于 PPO 但明显不如 greedy，也只能称为结构级基线，先改进 GPPO，再进入 PCRL 或世界模型。

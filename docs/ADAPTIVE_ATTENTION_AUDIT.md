# Adaptive Attention 复现审计

## 当前判断

旧版 `Hard Dynamic Extension` 的负结果不能用于证明论文中的 adaptive attention 无效：它混入了旧弱通信缓存、非论文奖励和过低任务难度。当前结果最多说明旧实现/旧训练设置下没有观察到 gate 收益。

## 公式核对

原论文第 6 页的 Eq.(1)-(3) 给出：

1. `e_{i,j,k}=RReLU(c^T[W^U v_k || W^T mu_{i,j,k}])`；
2. 自边使用独立的 `W^T v_k`；
3. 先把 `e_{k,k}` 与任务邻居系数在同一邻域归一化，再计算 `sigma(e'_{k,k}W^U v_k + sum e'_{i,j,k} W^T mu_{i,j,k} f_{i,j,k})`。

因此当前正式默认的解释是：

- RReLU 使用论文给出的随机斜率区间 `[0.125, 1/3]`；`expected` 模式使用其期望斜率，保证 PPO old/new log-prob 比较确定；`stochastic` 仅作为敏感性实验。
- self 与 task neighbor 共用一个 softmax 域。
- `f` 乘在任务消息上，`alpha * f` 后不再归一化；这正是 Eq.(3) 的括号结构，而不是把 `f` 直接加到 logits。
- 当前默认 `f` 为 sigmoid，仅允许抑制消息。这一点是论文没有规定的额外假设，不能隐藏在“严格复现”表述中。

## 新增可审计敏感性

模型现在显式支持 `gate_activation`：

- `sigmoid`：正式冻结默认，保持历史结果可比；
- `softplus`：始终为正且可大于 1，允许 adaptive gate 放大任务消息，用于检验“sigmoid 只能衰减”是否造成 single-head 优势。

评估器会保存 gate 的均值、P05/P95、接近 0/1 的比例以及大于 1 的放大比例。训练完成后应对同一 train/validation/test bank 运行 `task_message+sigmoid` 与 `task_message+softplus`，并保持 5 个训练种子。

正式评估流水线还会使用 `--trace` 保存每个决策步的 action、reward、event、同步状态，以及 projected/realized makespan；同时给出相对估计误差均值和 P95。

另外修正了 Paper-Faithful `always` 通信：此前该分支触发同步但遗漏 `full=True`，实际只发送增量 belief；现在每个 always decision 都复制完整 UAV/task/edge 状态，测试明确检查 full bytes 和全 UAV 更新列表。旧的 always 结果不可用于 Table VII，后续矩阵会重新生成。

修正后的 5-instance smoke（同一 checkpoint、同一 deterministic test bank）显示 event 平均约 2,953 bytes，而 always/full 约 4,694,294 bytes；任务质量相同是因为策略和外部 tape 相同，通信成本差异已恢复为可观测量。该 smoke 不替代正式五 seed 结果。

## 验收所需证据

Gate scope/activation 的同 checkpoint 推理敏感性可用以下脚本批量生成：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_paper_faithful_gate_sensitivity.ps1 `
  -Root outputs/paper_faithful/formal/T5-10-48_literal_event `
  -Instances 100 -Split test
```

该脚本不重新训练模型；每个 JSON 都记录 checkpoint hash、覆盖项和 `retrained=false`，所以只能作为机制敏感性证据，不能替代五 seed 的重新训练消融。

1. 五 seed 的 test100 realized makespan、completion rate、通信字节数及 paired 95% CI；
2. gate 分布和 gate 参数的真实 PPO 更新梯度（局部 policy-sensitivity 只能作为辅助）；
3. `task_message/score/aggregate` scope 消融；
4. single-head、no-gate、普通 PPO、随机、合理贪心、周期通信和 always/oracle 对照；
5. 若 adaptive 仍不优于 single-head，应报告为“论文机制未在独立机制级复现中重现收益”，而不是继续调参直到出现正结果。

截至本审计时，T5-10-48 Literal-event 五个正式训练仍在进行（约 iteration 470/2000），尚无最终 checkpoint 或 test100 结果；因此不能提前宣称 GPPO 基线验收通过。

作为机制 pilot 的可复核诊断（旧 sigmoid checkpoint），gate 均值约 0.77，P05/P95 约为 0.57/0.91，`gate<0.1` 为 0，`gate>0.9` 约 13%-15%，局部 policy-sensitivity 梯度 L2 为 0.03496；按真实 rollout batch 计算的 PPO probe gate 梯度 L2 为 0.20028。也就是说，当前负结果不能简单归因于 gate 恒等于 1 或完全没有 PPO 梯度；仍需正式五 seed 和 activation/scope 消融。

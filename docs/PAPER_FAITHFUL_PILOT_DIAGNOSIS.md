# Paper-Faithful Literal-AHGNN Pilot 诊断

状态：**Eq.(2) self-transform 修正后的 v6 pilot 已通过最近里程碑；正式四规模五种子计划已启动。**

重要版本规则：`pilot20`/`pilot20_v2`/`pilot20_v3`/`pilot20_v5` 都发生过公式或公平性修正，只能作为历史诊断；当前权威 pilot 是 `pilot20_v6`。

本报告只针对 `T5-10-48 / seed 1 / 20 iterations` 的机制门槛试验。它不是论文数值级复现，也不能替代目标协议中的 2000 iterations、5 seeds 和四个论文规模。

## 1. 证据边界

- 旧 `gppo-v2-hard-3` 已冻结为 `Hard Dynamic Extension`，不得作为论文基线。
- `pilot20` 是首次 Literal 实现，存在 Eq.(3) 外层激活遗漏、Eq.(5) ELU 位置错误，保留为诊断历史。
- `pilot20_v2` 修正公式，但 validation 与训练实例库重叠，保留为选择泄漏案例。
- `pilot20_v3` 使用互不重叠的 train、validation、test 确定性实例库；本报告只使用其独立 test100 结果。
- `pilot20_v5` 补齐 SingleHead 对照的 Eq.(3) 外层 ELU；因此旧 v3 SingleHead 数值只作为历史，最终公平消融必须使用 v5。
- 原论文未公开环境与代码，因此这里只能称作独立机制级复现。

## 2. Eq.(1)-(5) 审计结论

| 检查项 | 论文含义 | 当前 Literal 实现 | 结论 |
|---|---|---|---|
| Eq.(1) | `RReLU(c^T[W^U v_k || W^T mu_ijk])` | UAV 与 task-edge 扩展表示共同计算 score | 已对齐 |
| Eq.(2) | UAV self score | self 与 task 使用同一 attention 向量 | 已对齐 |
| 归一化域 | `e'_kk` 与 `e'_ijk` 由对应 score 归一化得到 | self 与可执行 task neighbors 在一个 softmax 域 | 已对齐 |
| `f_ijk` 作用域 | Eq.(3) 中乘在 task message 上 | `alpha * f * W^T mu` | 已对齐 |
| 二次归一化 | 原文未给出 | gate 后不重新归一化 | 已对齐 |
| Eq.(3) 外层 `sigma` | 聚合后非线性激活 | ELU | 已补齐；具体激活原文未明确 |
| Eq.(4) | 相邻 UAV embedding 求和 | 对 UAV neighbors 求和，不取均值 | 已对齐 |
| Eq.(5) | 四路 MLP 后拼接、ELU、第五个 MLP | `M1(ELU(M2 || M3 || M4 || M5))` | 已修正 |
| RReLU | 随机修正线性单元 | 使用 RReLU 推理期望斜率 `11/48`，保证 PPO ratio 可复算 | 部分对齐；仍需随机 RReLU 对照 |

### v6 关键修正

Eq.(2) 的 self score 明确为：

```text
e_kk = RReLU(c^T [W^U v_k || W^T v_k])
```

旧 Literal 实现把第二项误写成了 `W^U v_k`。v6 增加独立
`self_neighbor_transform = W^T`，并保留 self/task 的联合 softmax 域。这个
遗漏正是旧 stop-rule 结果的主要原因。

旧 hard 版的 `LeakyReLU(0.2)` 不是严格等价替代：PyTorch 默认 RReLU 的期望负斜率为 `(1/8 + 1/3)/2 = 11/48 = 0.22917`，且训练期具有随机斜率。当前确定性实现解决了数值斜率和 PPO 可复算性，但没有复现训练期随机性，这是剩余诊断变量。

对应测试位于 `tests/test_paper_faithful.py`，覆盖：手工小图 Eq.(1)-(3)、显式五 MLP Eq.(4)-(5)、RReLU 斜率、gate 非常数、gate 非零梯度、节点/边特征改变 attention、实例库隔离和事件带一致性。

## 3. 独立 test100 结果

所有方法使用同一个 seed-1 训练预算、同一 validation 选择规则，以及完全相同的 100 个 test 实例和 event tape。所有方法的任务完成率均为 1.0。

| 方法 | Best iter | Active params | Return mean | Realized makespan mean | Median | Inference ms/decision |
|---|---:|---:|---:|---:|---:|---:|
| PPO-MLP-event | 20 | 18,691 | -8.553 | 17.872 | 17.575 | 0.980 |
| GPPO-Literal-event | 20 | 89,540 | -8.833 | 18.153 | 18.118 | 2.139 |
| GPPO-NoGate-event | 10 | 81,219 | -8.676 | 17.995 | 17.767 | 2.100 |
| GPPO-SingleHead-event（v5 公平版） | 20 | 85,507 | -8.404 | 17.724 | 17.718 | 2.863 |

配对差值定义为左方法减右方法；makespan 负值才代表左方法更好。

| 对比 | Makespan delta | 95% CI | 左方法胜率 | 判断 |
|---|---:|---:|---:|---|
| Literal - PPO-MLP | +0.280 | [-0.173, +0.734] | 45% | 完整 GPPO 未优于论文式 PPO |
| Literal - NoGate | +0.157 | [-0.229, +0.544] | 46% | gate 无可观察正贡献 |
| Literal - SingleHead（v5） | +0.429 | [+0.130, +0.728] | 46% | Literal 显著更差，触发 stop rule |
| SingleHead - PPO-MLP | +0.480 | [+0.093, +0.868] | 39% | simple graph control 显著更差 |

episodic return 与 realized makespan 在相同实例上是相反方向的同一差值，因为主奖励按 `M_(t-1)-M_t` 精确望远镜求和；上表的两项结论不会互相矛盾。

## 4. 问题 1：为什么 adaptive 不如 single-head？

### 现象

旧 hard 结果中 single-head 同时优于 adaptive 的 DCR 和 makespan。修正公式并隔离协议后，Literal 的 test100 makespan 均值比 single-head 好 0.200，但 95% CI 跨 0，因此尚不能确认稳定优势。

### 假设

1. 旧消融不是单变量：adaptive 含额外 relation/priority bias、通信分支、残差和 LayerNorm；single-head 同时改变任务更新器。
2. 旧实现用 LeakyReLU(0.2) 替代 RReLU，并遗漏/错放公式运算。
3. 20 iterations 对 4.8 倍于 MLP 的有效参数量过短，注意力分支尚未稳定。
4. 当前确定性 RReLU 与论文训练期随机 RReLU 仍有差异。

### 诊断实验

- 建立独立 Literal 模块，关闭公式外增强。
- 保持任务更新器一致，只替换 UAV attention 为 single-head。
- 修正 Eq.(3) 外层激活和 Eq.(5) ELU 顺序。
- 在独立 test100 上做实例级配对 CI。

### 结果

在补齐 SingleHead 外层 ELU 后，Literal 的 test100 makespan 比公平 SingleHead 高 0.429，95% CI [+0.130,+0.728]；这重新确认了“adaptive 不如 simple attention”的矛盾，而且达到 stop rule 的统计证据。

### 结论

旧矛盾主要由消融混杂和公式偏差造成；但当前 pilot 尚不能证明 adaptive 稳定优于 single-head。

### 是否修正

结构与协议已修正；性能结论未通过。RReLU 敏感性和 100-iteration 长度诊断已完成，但不改变公平 SingleHead 负结果。

## 5. 问题 2：为什么 NoGate 与 adaptive 几乎相同？

### 现象

旧 hard 版中 NoGate 与 adaptive 几乎无差。当前 test100 中 NoGate 反而比 Literal 好 0.157，但 CI 跨 0。

### 假设

1. 旧 gate 多数 checkpoint 接近恒等映射，实际作用太小。
2. 新 Literal gate 虽未饱和，却可能过度压低 task message。
3. gate 增加 8,321 个有效参数，在短训练预算下提高估计方差或过拟合 validation。

### 诊断实验

对选中 Literal checkpoint 的 20 个独立 test 实例、226,176 条 task edge 采集 gate、attention mass，并对实际策略输出计算局部梯度敏感性。

### 结果

- gate mean 0.798，std 0.130，P05/P95 为 0.564/0.947；不是常数。
- 34.9% gate 大于 0.9，没有 gate 小于 0.1；未发生两端完全饱和。
- task attention mass/每架 UAV 均值 0.983。
- gate 平均移除 attention mass 0.190，说明作用并不微弱。
- gate 局部策略敏感性梯度 L2 为 0.048；不是无梯度。
- Literal 相对 NoGate 的质量没有改善。

### 结论

当前问题不是“gate 没工作”，而是“gate 学到的抑制没有稳定转化为调度收益”。在只有 20 iterations 时，额外参数和约 19% 的 message 抑制是最直接风险。

### 是否修正

计算路径已修正；性能尚未修正。下一步必须做实际随机 RReLU、训练长度以及 gate 初始化的受控诊断，不能直接宣布 adaptive 有贡献。

## 5.1 RReLU 随机性敏感性结果

在同一 `T5-10-48 / seed 1 / 20 iterations` 预算下，新增训练期随机 RReLU 对照（validation 仍使用 eval 期确定性 RReLU）：

| 方法 | Test100 realized makespan | 相对 expected 版本 |
|---|---:|---:|
| Literal-expected | 18.153 | 基准 |
| Literal-stochastic | 18.071 | -0.081 |
| NoGate-stochastic | 18.271 | +0.276 |

随机 RReLU 让 gate 相对 NoGate 的方向变为有利（Literal-stochastic 比 NoGate-stochastic 好 0.200），但仍比 PPO-MLP-event 的 17.872 差 0.199。它是重要敏感性证据，不是完整 GPPO 通过证据；同时随机 rollout 会引入 PPO old/new ratio 的额外随机性，正式实验前需要固定随机实现策略并记录 RNG 协议。

## 6. 当前验收判断

| 门槛 | 状态 |
|---|---|
| Paper-Faithful 环境、100-instance train bank、独立 validation/test bank | 通过 |
| Eq.(1)-(5) 公式测试和手工小图 | 通过 |
| Literal 优于公平 SingleHead | **未通过；v5 delta +0.429，95% CI [+0.130,+0.728]** |
| Literal 稳定优于 NoGate | 未通过 |
| 完整 GPPO reward 与 realized makespan 均优于论文式 PPO | 未通过 |
| 进入四规模、五种子、2000-iteration 正式训练 | **禁止** |

该段结论针对 Eq.(2) 修正前的 v3/v5 历史，已被下面的 v6 结果 supersede；不得继续作为当前结论。

## 11. v6 修正后的权威 pilot 结果

Eq.(2) self transform 修正后，在同一 T5-10-48、seed 1、20 iterations、独立
test100 上：

| 方法 | Active params | Return | Realized makespan |
|---|---:|---:|---:|
| PPO-MLP-event | 18,691 | -9.419 | 18.738 |
| GPPO-Literal-event | 93,636 | -8.143 | 17.462 |
| GPPO-NoGate-event | 85,315 | -9.189 | 18.508 |
| GPPO-SingleHead-event | 85,507 | -8.933 | 18.252 |

配对差值（Literal 减对照）：

- Literal - PPO: `-1.276`, 95% CI `[-1.667, -0.884]`；
- Literal - NoGate: `-1.046`, 95% CI `[-1.334, -0.757]`；
- Literal - SingleHead: `-0.790`, 95% CI `[-1.076, -0.503]`。

所有方法的 test100 任务完成率为 1.0。v6 同时满足最近里程碑中“reward 与 realized makespan 均优于 PPO”的要求，并首次提供了 adaptive gate 的正向支持。此前 bias=2、random-RReLU 和 v3/v5 结果均属于 Eq.(2) 修正前历史，不应与 v6 混合汇报。

## 8. 同步敏感性（补充）

在同一 Literal checkpoint、同一 20 个 test 实例上，切换同步模式不会改变动作轨迹或 realized makespan（18.388），但通信次数显著不同：event 3.00、periodic 9.15、always 76.95。这个结果只说明当前固定 task-change tape 下 event 通信成本较低；由于策略没有利用到这类 task distribution 更新，不能据此宣称 event 已达到论文的质量-通信折衷。

## 9. 100-iteration 长度诊断

为区分“训练太短”和“gate 本身无益”，在同一 T5-10-48、seed 1 和 disjoint banks 上，把 Literal、NoGate、PPO-MLP 都延长到 100 iterations（仍为诊断预算，不是论文 2000 iterations）。独立 test100 结果如下：

| 方法 | Best iter | Return | Realized makespan |
|---|---:|---:|---:|
| PPO-MLP-event | 100 | -7.538 | 16.857 |
| Literal-event | 70 | -6.705 | 16.024 |
| NoGate-event | 100 | -6.528 | 15.847 |

这说明长训练后两种图模型都显著优于同预算 PPO-MLP（Literal-PPO makespan delta -0.833，95% CI [-1.157,-0.509]；NoGate-PPO -1.010，CI [-1.346,-0.674]），因此“完整 GPPO 是否能优于 PPO”在一个规模、一个 seed 的诊断预算上得到支持。但 Literal-NoGate delta 为 +0.177，CI [-0.016,+0.370]，gate 仍没有正向贡献，且 NoGate 均值更好。

因此这项结果只能解除“20 iterations 太短”的一个疑问，不能解除 adaptive gate 矛盾；正式五种子前仍需 gate 受控实验。

## 10. Gate 初始化受控实验

将 gate 最后一层 bias 从默认 `0.0` 改为 `2.0`，使初始 gate 更接近恒等映射；其余 100-iteration 配置、实例库、seed 和 checkpoint 选择完全相同。

| 设置 | Test100 makespan | 相对默认 Literal |
|---|---:|---:|
| bias=0 | 16.024 | 基准 |
| bias=2 | 15.989 | -0.035 |
| NoGate | 15.847 | bias=2 仍差 +0.142 |

bias=2 的 gate mean=0.836、removed attention mass=0.104，说明初始化确实降低了早期抑制，但不足以恢复 NoGate 的性能。因此 adaptive 负贡献不能归结为单纯 gate bias 初始化问题。

## 7. 下一轮最小诊断矩阵

在 `T5-10-48 / seed 1` 上继续使用相同 train/validation/test banks，只运行：

1. Literal-expected-RReLU（当前）；
2. Literal-random-RReLU；
3. NoGate-expected-RReLU；
4. NoGate-random-RReLU。

先把训练预算提高到 100 iterations，每 10 iterations 保存候选 checkpoint。只有 Literal 在独立 validation/test 上同时优于 NoGate 和 PPO-MLP，且方向不依赖单个候选点，才允许进入正式矩阵。

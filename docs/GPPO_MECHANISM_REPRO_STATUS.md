# GPPO 机制级复现状态（2026-08-08）

## 结论先行

旧版 `gppo-v2-hard-3` 的矛盾是真实的工程结果，但不能直接解释为论文 AHGNN 结论失败，因为该版本同时包含弱通信缓存修正前的观测泄漏、非论文奖励、额外 relation/communication 分支，以及错误的 Eq.(2) self-score。它已冻结并标记为 `Hard Dynamic Extension`，不再作为论文数值级基线。

对照论文 PDF 第 6--7 页逐项重写后的 `Paper-Faithful Literal-AHGNN` 修正了 Eq.(2)：

```text
e_ijk = RReLU(c^T [W^U v_k || W^T μ_i,j,k])
e_kk  = RReLU(c^T [W^U v_k || W^T v_k])
```

其中 self 项使用独立的 `W^T` 变换；self 与可执行 task neighbor 在同一 softmax 域。需要特别保留一个论文歧义：PDF 的 Eq.(3) 将 `f_{i,j,k}` 排在聚合括号之后，而正文又称其为按边/任务-UAV 对调整 attention weight，索引与排版并不完全一致。当前主协议冻结为“逐 task message 调制、gate 后不二次归一化”，并把“调制 score 后再 softmax”和“聚合后调制”作为后续敏感性对照，不能事后挑选解释。Eq.(4)--(5) 的四个关系 MLP 与 fusion MLP 已按字面实现。

## 已完成的机制审计

- 论文忠实环境：四个规模 `T5-10-48`、`T10-10-53`、`T15-8-66`、`T20-10-92`；每个规模独立 train/validation/test bank，各 100 个实例。
- 固定 event tape，所有同步模式复用相同外部事件；记录 realized makespan、projected makespan、通信事件数、字节数与 heartbeat 数。
- `always` 通信已修正为真正的 full-state synchronization；此前遗漏 `full=True` 的旧 always 输出不再作为 Table VII 证据。
- 四个结构：`ppo_mlp`、`literal`、`literal_no_gate`、`literal_single_head`。
- 262 项测试通过，公式手工小图、RReLU 斜率、gate 非常数/非零梯度、节点与边特征敏感性、事件带一致性和参数审计均通过。

## v6 诊断 pilot

设置为 `T5-10-48 / seed 1 / event / 20 iterations / test100`。这是机制门槛诊断，不是论文数值级复现。

| 方法 | realized makespan |
|---|---:|
| PPO-MLP-event | 18.738 |
| Literal-AHGNN-event | **17.462** |
| NoGate-event | 18.508 |
| SingleHead-event | 18.252 |

配对差值（Literal 减对照）为：PPO `-1.276`，95% CI `[-1.667,-0.884]`；NoGate `-1.046`，95% CI `[-1.334,-0.757]`；SingleHead `-0.790`，95% CI `[-1.076,-0.503]`。四种方法 test100 完成率均为 1.0。该结果支持“修正后的 Literal 在该规模/seed 上优于 PPO 和两个结构对照”，但不能替代五 seed、2000 iteration 正式实验。

## 对“adaptive 不如 simple”的判断

旧结果中最应优先排查的项目确实是公式实现，而不是先调 gate bias。当前审计结论如下：

1. `LeakyReLU(0.2)` 不是严格的 RReLU；RReLU 期望负斜率为 `11/48≈0.22917`，训练态还带随机性。正式协议默认 deterministic expected-RReLU 以保证 PPO ratio 可复算，并保留 stochastic-RReLU 敏感性实验。
2. Eq.(2) 的 self-score 必须使用独立 `W^T v_k`。旧实现复用了 `W^U v_k`，会改变 self/task 的竞争关系，是旧 adaptive 失败的首要代码级原因。
3. `f_ijk` 作用在 task message；alpha 与 gate 相乘后不再归一化，因为论文没有给出二次归一化。
4. self 与 task neighbor 置于同一 softmax 域；拆成两个域会改变 Eq.(3) 的相对权重。
5. v6 gate 不是恒等常数，也不是无梯度：pilot 统计 gate mean 约 0.798、std 约 0.130，局部策略敏感性梯度 L2 约 0.048。因此旧矛盾不能简单归结为 gate 饱和。
6. adaptive 有效参数多于 NoGate；在短训练预算下可能增加估计方差/过拟合风险，所以必须进行长训练、五 seed 和参数量审计，不能只比较单次点估计。

## 当前正式实验状态

`T5-10-48 / Literal-event / seeds 1--5 / 2000 iterations` 已启动，当前约在 iteration 525，仍在训练。完成后将按预注册协议继续 PPO-none/event、Literal-none/event、NoGate、SingleHead、periodic、always，并计算五 seed 的均值、标准差、配对 95% CI、通信成本和泛化误差。在这些正式产物完成前，PCRL、JEPA 和世界模型接入保持暂停。

## 建议的后续改进

- 先冻结 `Paper-Faithful Protocol`、事件带和 checkpoint 选择规则，再进行任何超参数搜索。
- 以 realized makespan 改善为主奖励，保留 completion/invalid-action 作为诊断；避免 8 步内所有方法完成全部任务的饱和场景。
- 对 Literal/NoGate 使用完全相同的训练器、实例库、同步模式、seed 和有效容量审计；SingleHead 只作结构对照，不宣称是 gate 的纯因果消融。
- 至少 5 个训练 seed；统计单位先按 seed 聚合，再跨 seed 计算 CI 和显著性，不能把相关 episode 当作独立样本。
- 只有在 GPPO-event 稳定优于 PPO、明确优于随机且接近合理 greedy 后，才进入 PCRL；世界模型应先独立报告事件分类 F1、校准误差、触发延迟和通信节省，再接入调度器。

原论文未公开完整环境与代码，因此最终措辞应为“机制级/结构级复现”，不能声称论文数值级复现。

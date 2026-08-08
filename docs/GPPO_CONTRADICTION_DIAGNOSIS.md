# GPPO 矛盾诊断总表

本表把旧 `Hard Dynamic Extension` 的现象与新 `Paper-Faithful` 诊断分开。除非明确标注为 test100 或 hard-v2 summary，所有内容都是待验证假设，不是论文结论。

| 编号 | 现象 | 当前证据 | 当前判断 | 修正/下一步 |
|---:|---|---|---|---|
| 1 | Adaptive 不如 simple/single-head | 旧 hard：single-head DCR 0.7865、makespan 20.1115；v6 修正 Eq.(2) 后 Literal 相对 SingleHead makespan delta -0.790，95% CI [-1.076,-0.503] | 旧版遗漏独立 `W^T v_k` self transform，并混入额外 bias、通信分支、残差/LN、LeakyReLU 和任务更新器；公式修正后原矛盾消失 | v6 已通过最近里程碑；进入正式多规模、多种子验证 |
| 2 | NoGate 与 Adaptive 几乎相同 | 旧 hard 两主指标几乎相同；v6 Literal-NoGate delta -1.046，95% CI [-1.334,-0.757] | 旧问题的核心是 Eq.(2) self score 偏差，而不是 gate 饱和/无梯度；v6 gate 产生稳定正向贡献 | 保留随机 RReLU、bias 初始化作为敏感性分析，但不覆盖 v6 主结论 |
| 3 | `3x16` GPPO 不如 PPO | hard-v2 文档列为异常规模；PPO stale-belief invalid actions 和任务完成不足明显 | 可能是小规模下事件通信/缓存冲突成本超过图表示收益；尚无 paper-faithful 同规模多 seed 证据 | faithful 矩阵必须保留该负结果，不能删 scale |
| 4 | `3x20` GPPO 不如 Greedy | hard summary 中 Greedy 在该规模 makespan/DCR 更强 | Greedy 直接利用当前任务效用，短训练 GPPO 尚未学会前序/重分配 trade-off；属于算法成熟度而非 GPPO 已被证伪 | 增加训练曲线、action-mask conflict 和 warm-start 检查；不得用 Greedy 调参后再比较 |
| 5 | GPPO 重分配成功率低于 Random | hard summary 的 reallocation 指标受事件样本数、失效时机和 stale action 影响 | Random 的“成功”可能来自偶然选择可行 survivor，不能直接解释为策略更强；需要按尝试次数和 failure tape 配对 | faithful event tape 中固定 task distribution change；后续报告 attempts、successes、failure target 原始行 |
| 6 | 图结构收益小于同步收益 | hard formal summary：event communication 约 1.06--5.56，always 约 9.18--41.00；makespan 差距通常小于通信差异 | 当前任务变化主要由 state freshness 驱动，图编码增益被 stale cache 混淆；不能把 always 的收益归因于 AHGNN | faithful 2x2 固定 Graph/Sync，另列 event/periodic/full；质量与通信分开统计 |
| 7 | estimated 与 realized makespan 误差不明 | 旧 hard CSV 只保存最终 `makespan/current_time`，没有每步真实完成时间；新 faithful evaluator 同时保存二者，完成后差值为 0 | 旧实现无法支持误差结论；faithful 已把 realized completion_time 作为主指标，但需要保存每个 transition 的 raw trace | 增加 raw transition trace，报告 `|projected-realized|/realized` 的 mean/median/P95 |
| 8 | 同 seed 是否得到同事件序列 | 旧硬环境事件时间由物理 clock 和动作影响，seed 相同不保证 action-dependent sequence 相同 | 旧协议不满足严格 replay；faithful 用 train/validation/test disjoint banks + fixed event tape，已有一致性测试 | 正式矩阵只使用同一 tape manifest，并在输出保存 tape hash |

## 当前总体结论

旧 hard 的“GPPO 工程基线通过”只适用于工程扩展，不适用于论文模块贡献。v6 Literal 实现已通过公式级、事件重放级和最近 test100 机制门槛；正式四规模五种子仍需验证，PCRL 和世界模型继续暂停。

# GPPO 论文忠实复现目标审计

更新时间：2026-08-08。状态只能取 `通过 / 进行中 / 缺失 / 被证据否定`。本表中的“通过”必须有当前工作区中的权威产物支持。

| 目标要求 | 状态 | 权威证据 | 尚缺内容 |
|---|---|---|---|
| 冻结旧 GPPO-v2 hard 结果，不覆盖负结果 | 通过 | `outputs/paper_aligned/gppo_v2_hard/formal/summary/frozen_manifest.json`、`docs/GPPO_V2_HARD_EXTENSION_FREEZE.md` | 无 |
| 四个论文规模与每规模 100 个互斥 train/validation/test 实例 | 通过 | `src/uav_assignment/paper_faithful_env.py`、`tests/test_paper_faithful.py` | 无 |
| 固定 event tape 在方法和同步模式间回放 | 通过 | 环境实现、event-tape/hash 测试 | 正式结果仍需保存每实例 tape hash |
| realized makespan、无 deadline、所有任务最终完成 | 通过 | 环境单元测试、pilot test100 completion=1.0 | 正式四尺度结果待生成 |
| Eq.(1)-(5) Literal-AHGNN 公式实现 | 通过（含一个冻结歧义） | `src/uav_assignment/paper_faithful_models.py`、手工小图公式测试 | Eq.(3) gate scope 需报告 task-message/score/aggregate 敏感性 |
| RReLU、独立 `W^U/W^T`、self/task 联合 softmax | 通过 | v6 Eq.(2) 修正、RReLU 与归一化测试 | 随机 RReLU 仅作敏感性，不替代正式 expected 版本 |
| gate 非常数且有非零梯度 | 通过 | `test_gate_is_not_constant_and_receives_gradient`、实际 PPO probe 诊断脚本 | 正式 checkpoint 的 gate 分布需补充 |
| Literal 在一个规模/seed 上同时优于 PPO、NoGate、SingleHead | 通过（pilot） | `outputs/paper_faithful/pilot20_v6/T5-10-48` | 不是正式多 seed 证据 |
| 五 seed、2000 iteration、四论文规模 | 进行中 | `outputs/paper_faithful/formal/T5-10-48_literal_event`、`outputs/paper_faithful/formal/pipeline.log` | 当前首个 T5 Literal-event 单元仍在训练，五个 seed 已保存 iteration 1,600/2,000 候选快照（2026-08-08 08:04）；后续矩阵由后台流水线接续 |
| GPPO-event vs PPO-none 原论文整体比较 | 缺失 | 无正式 test100 汇总 | 等待后台矩阵 |
| Graph/Sync 2x2 因果隔离 | 缺失 | runner/流水线已配置 | 等待 PPO-none/event 与 Literal-none/event |
| NoGate 与 SingleHead 正式消融 | 缺失 | runner/流水线已配置 | 等待五 seed test100 |
| Event/Periodic/Always 通信质量和成本 | 缺失（smoke 已验证计量） | 通信计数和字节实现、流水线已配置；`always` 已修正为 `full=True`；同一 event-trained checkpoint 的 sync-mode 重放已接入；5-instance smoke 已观察 event/full bytes 差异 | 旧 always 结果作废，等待重新生成正式 checkpoint 和 test100 |
| Fig.8 五 seed训练曲线 | 缺失 | 绘图脚本存在 | 训练历史未齐 |
| Tables III-IV realized makespan | 缺失 | evaluator/summarizer 存在 | 正式评估未齐 |
| Table V 泛化误差与 general model | 入口已实现，结果缺失 | `train_paper_faithful_general.py`、`evaluate_paper_faithful_unknown.py`、流水线 general stage | 跨尺度 general 模型与至少 8 个未知场景尚未运行 |
| Table VI 推理时间、P50/P95、Pearson R | 入口已实现 | `evaluate_paper_faithful_inference_scaling.py` 固定 UAV、递增子任务数并计算 P50/P95/FLOPs/Pearson R | 等待 general checkpoint 执行 |
| Table VII 通信比较 | 缺失 | bytes/events/heartbeat 字段已实现；审计将检查 event/full checkpoint hash 与 event-tape 一致 | 正式 Event/Full 结果未齐 |
| 八个特殊矛盾逐项完成 paper-faithful 诊断 | 部分通过 | `docs/GPPO_CONTRADICTION_DIAGNOSIS.md` | Eq.(2) 解决 adaptive 主矛盾；其余需正式矩阵或 raw trace |
| 随机、贪心和全状态启发式基线 | 入口已实现，结果缺失 | `evaluate_paper_faithful_baselines.py`、`run_paper_faithful_baselines.ps1` | 正式四尺度 baseline JSON 尚未齐 |
| 最终复现报告和“完全/部分/未通过”分级 | 缺失 | `report_paper_faithful_formal_zh.py` 已接入流水线 | 只能在所有正式证据生成后写结论 |

## 统计与运行可靠性补充

- `evaluate_paper_faithful.py` 现在保存训练 seed、同步模式、gate scope、逐决策延迟分布和 P50/P95。
- 评估器已支持 `--trace`，保存每步 action/reward/event/sync 与 projected/realized makespan，并报告相对估计误差和 analytical forward FLOPs。
- `summarize_paper_faithful_formal.py` 使用五 seed 的 Student-t 95% 区间，并检查配对方法的 event-tape hash 一致性。
- `run_paper_faithful_formal.py` 默认跳过已有最终 checkpoint；后续 job 可用版本化 `resume_latest.pt` 精确恢复模型、优化器、RNG 和历史。
- 2026-08-08 首批 T5 Literal-event 五进程在 1799/1804 附近被外部中断，且旧进程尚未写出版本化 `resume_latest.pt`。已从 seed1/2/4/5 的 `candidate_1750.pt` 与 seed3 的 `candidate_1800.pt` 恢复；超出快照的历史行已裁剪。恢复不包含 Adam moments、RNG 与 episode offset，最终 checkpoint/audit/report 将以 `candidate-recovery-v1` 明示这一优化连续性边界。
- 同批训练在 seed1/2/4/5 的 1949 与 seed3 的 1999 附近再次发生进程中断。seed3 幸存并完成 2000；其余四个种子从 `resume_latest.pt` 的第 1900 轮精确恢复模型、Adam、RNG、episode offset 和历史，并以 `exact-resume-event-v1` 记录。重跑的 1901–1949 来自同一恢复状态，不与中断时未落盘的临时历史拼接。
- 首批训练进程启动后，脚本又增加了显式 `gate_scope/gate_activation` 元数据与恢复支持；已用 `strict=True` 将其 `candidate_1200.pt` 加载到当前 `task_message + sigmoid` Literal 模型，45 个 state-dict 键全部匹配、参数总数同为 106,564。因此首批权重结构与冻结的正式默认行为兼容；最终报告仍需把缺省元数据按 `task_message/sigmoid` 明示，而不能把该兼容性扩展为其他 gate 解释。

## 当前不能回答为“已完成”的三个核心问题

1. 完整 GPPO 是否在多数论文规模上稳定优于论文式 PPO：**正式证据缺失**。
2. 异构图和 adaptive gate 是否分别产生正向贡献：**v6 pilot 支持，但五 seed 正式证据缺失**。
3. Event 是否以明显更低通信成本达到接近 Full 的任务质量：**计量实现存在，正式比较缺失**。

因此，当前目标状态必须保持为进行中。

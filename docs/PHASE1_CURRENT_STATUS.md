# 第一阶段当前状态与冻结记录

更新时间：2026-08-10（Asia/Shanghai）

## 1. 阶段边界

第一阶段拆分为：

- **Phase 1A：GPPO 机制级基线。** 完成论文兼容 `GPPO-Literal`、Adaptive 歧义诊断、验证集预注册筛选、四规模五训练种子和通信因果审计。
- **Phase 1B：多源扰动环境。** 在冻结基线上依次加入突发丢包、时延/分区、故障/能量、任务变化和风场。

在 `PHASE1_ACCEPTANCE.json` 通过之前，不接入 PCRL、Bradley–Terry 奖励、Preference-GPPO、世界模型、JEPA、想象轨迹、风险增强图或预测式事件触发。

## 2. 仓库冻结点

- 仓库：`E:\Z博士\github_submission\GPPO-paper-faithful-clean`
- 分支：`8.8-GPPO无偏好`
- 冻结前 commit：`071bf0bce76da44f193ac534a10800561b4c20c0`
- 冻结前工作树：clean

## 3. 300 轮历史基线

原始包已解压到：

```text
artifacts/phase1_seed1_300/raw
```

该目录包含 96 个文件、28,987,490 字节；所有 raw 文件已设为只读。后续实验不得覆盖这些文件。

源文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `GPPO_mechanism_seed1_300.zip` | `c772494a2e1aa029d796e92d8b8cb4d46d59205dfe87e3dd87e846b3aaad9a62` |
| `MECHANISM_ACCEPTANCE.json` | `a703b92bebf4b7bb01d37ceb61ee13264df0d6b359323c8046423be285d408a0` |
| `GPPO_MECHANISM_REPORT_ZH.md` | `865f14768e5c8f51eecc99729f983a6c3c2fdae260e7ecfe0b6a637be4a21b17` |

六个模型均具备 checkpoint、50–300 轮六个 candidate、300 条连续训练历史、6 条 validation20 历史、resume、固定 test100、配置和 seed=1：

| 模型目录 | 最优轮次 | checkpoint SHA256 |
|---|---:|---|
| `literal_event_seed1` | 250 | `3bac3faf988e9e4d08aaa9d37bbe148d36547e8a8791ceecf56d8f5371ea3df6` |
| `literal_no_gate_event_seed1` | 250 | `8bc2d3116424e2b9399d93814efce06dd0a4e0512a0c223b6cd09e2f85ccdf0f` |
| `literal_none_seed1` | 250 | `d5427c3352d5d1c62acdb01d1a09024ed9c911de387bef58f05a84421602e598` |
| `literal_single_head_event_seed1` | 200 | `04eeca7502c632b977ff6fa7963ed385a5d6c3f6b80a319672ad49c0c30c3d4b` |
| `ppo_mlp_event_seed1` | 250 | `465283ae9e0c6e2fb18b501197f2356f6e966352089f0c1cb68050edb866c0fd` |
| `ppo_mlp_none_seed1` | 300 | `9fce1201654f3d3f6ce9da3fd34523c203281448d59e2e06223601c7c7829aaf` |

权威冻结清单与新审计：

- `artifacts/phase1_seed1_300/ARTIFACT_MANIFEST.json`
- `artifacts/phase1_seed1_300/ARTIFACT_AUDIT.json`

新审计结果为 `valid=true`、`errors=0`。六种方法共享同一 test100 实例库和 event-tape bank，evaluation 中的 checkpoint SHA256 与实际文件一致。

## 4. 必须保留的负结果与限制

- 300 轮单种子 test100 中，Adaptive 相对 NoGate 的 makespan 配对差值为 `+0.1942`，95% CI `[+0.0061, +0.3823]`。
- Adaptive 相对 SingleHead 的差值为 `+0.2284`，95% CI `[+0.0317, +0.4251]`。
- GPPO-event 相对 GPPO-none 的差值为 `+0.1413`，95% CI `[-0.0388, +0.3215]`；Event 相对 None 的决策收益尚未得到因果证明。
- 原 checkpoint 选择使用 validation20；必须在 Step 1 使用预注册的 validation100-A 重新选择，validation100-B 和 test100 只作复核。
- 当前结果只有一个训练种子，实例级 CI 不能代替训练种子级 CI。
- ZIP 内自带的 `quick_artifact_audit.json` 保存的是训练续跑前的旧 checkpoint 哈希，与 ZIP 中最终 checkpoint 不一致。该文件作为历史记录保留，但不得再作为权威哈希清单；新 `ARTIFACT_AUDIT.json` 为权威审计。

## 5. 已证明与未证明

当前单种子证据支持：

- 图结构在 event 与 none 条件下均优于相应 PPO；
- 动作掩码和环境非法动作均为 0；
- Event 与 Full 使用同一 checkpoint 重放时质量一致，Event 通信字节显著更低；
- gate 非常数且存在策略敏感度/PPO probe 梯度；
- 领导机故障、重选、任务释放和恢复探针可运行。

当前证据**不支持**：

- Adaptive 产生独立正收益；
- Event 明确优于完全不通信；
- 五训练种子稳定性；
- 四规模统计结论；
- 多源扰动环境已完成；
- 论文数值级复现。

## 6. 当前执行状态

- 已有 2000 轮 `GPPO-Literal` 正式矩阵继续作为论文兼容版本运行，不因后续 Gate 筛选而覆盖或删除。
- Step 1 将只复评现有 candidate，不重复已有 300 轮训练。
- Gate 新变体必须先写入 `configs/GATE_DIAGNOSTIC_PROTOCOL.json`，再运行 Smoke 和三种子筛选。
- `GPPO-Best` 只能由 validation 协议选出；不得使用 test100 调参。

Step 0 判定：**通过，带四项明确警告；历史负结果已冻结并可复核。**

## 7. Step 1 候选 checkpoint 重评估结果

Step 1 已完成 54 个固定评估：3 个模型 × 6 个候选轮次 ×
validation100-A、validation100-B、原 test100。三套实例库两两互斥；每套库内
所有模型和轮次使用相同实例与事件带。审计结果 `valid=true`。

只按 validation-A 选择得到：

| 模型 | validation-A选择轮次 | A | B | test100 | 原冻结checkpoint轮次 |
|---|---:|---:|---:|---:|---:|
| Adaptive | 300 | 15.9429 | 16.0585 | 15.8673 | 250 |
| NoGate | 150 | 15.9540 | 16.1454 | 15.8279 | 250 |
| SingleHead | 250 | 15.9796 | 15.9817 | 15.8285 | 200 |

逐张量核对证明，原冻结 `checkpoint.pt` 分别等于 Adaptive 第250轮、NoGate
第250轮、SingleHead第200轮候选权重。因此原 validation20 确实改变了最终
checkpoint 选择；它不是简单保存了第300轮权重。

但是 validation100 仍未证明 Adaptive 有稳定独立收益：

- 第300轮 Adaptive−NoGate：A `-0.0241`、B `+0.1023`、test `-0.0340`，三个实例级区间均跨零；
- 第300轮 Adaptive−SingleHead：A `-0.0370`、B `+0.0499`、test `+0.0560`，三个实例级区间均跨零；
- 第300轮三模型排序在 A、B、test 之间反转，A-test 与 B-test Spearman 均为 `-0.5`；
- 第100轮 Adaptive 明显领先，但第150–300轮优势不稳定，说明结果对训练轮次敏感。

因此 Step 1 的结论是：原 validation20 的候选选择存在明显方差，但把验证集
扩大到100后，Adaptive 也只能从“显著更差”修正为“没有稳定独立收益”，不能
据此宣称 Adaptive 优于 NoGate 或 SingleHead。下一步必须执行预注册的 Gate
结构诊断和三训练种子筛选，禁止用 test100 重新选轮次。

完整证据：

- `outputs/checkpoint_reevaluation/results.json`
- `outputs/checkpoint_reevaluation/learning_curve.png`
- `reports/CANDIDATE_CHECKPOINT_REEVALUATION.md`

## 8. Step 2 Gate公式审计与Smoke状态

Gate诊断已在任何新变体训练前预注册，启动时协议 SHA256 为
`0b863bf37045fa3287268c6f7695c439c529d4ca075032df3c04ce2d87f68f13`。
七个固定变体为 Adaptive-current、bias2、warmup、score、softplus、NoGate、
SingleHead；三种子均为1、2、3。

已经实现并验证：

- warmup 前期 task gate 精确为1且 gate MLP 冻结，解冻后恢复有效梯度；
- validation split 可显式指定为 validation-A；
- 训练历史新增 policy entropy、approx KL、clip fraction、actor/value loss、gate gradient；
- gate诊断新增 removed attention mass 和 UAV特征范数；
- 评估结果新增非法动作和重分配任务计数；
- 两轮最小训练验证第1轮 gate gradient=0、第2轮 gate gradient>0；
- 当前完整测试为39项通过。

正式300轮 warmup 的选模资格修正登记在
`configs/GATE_DIAGNOSTIC_PROTOCOL_AMENDMENT_001.json`，不修改 Smoke 启动协议
哈希，且未使用 test/test100 作出该修正。

公式歧义和六个核心问题的完整判断见
`reports/ADAPTIVE_GATE_FORMULA_AUDIT.md`。20轮 Smoke 正在后台执行；Smoke
仅验证训练与诊断链，不选模型，也不读取 test100。

### Step 2 Smoke完成

七变体 × 三种子 × 20轮已全部完成，`SMOKE_AUDIT.json` 为 `valid=true`。
21个checkpoint均无非法动作并完成任务；训练诊断值有限；非warmup Adaptive有训练期
gate梯度；warmup前20轮按协议保持gate=1和零gate梯度；重载后的15个Adaptive
checkpoint均显示非恒定gate和有效PPO probe梯度。全过程未读取test/test100。

Smoke只通过机制准入，不产生变体排名。完整结果见
`reports/GATE_SMOKE_REPORT.md`。下一步为预注册三种子300轮筛选。

## 9. Step 3 三训练种子 Gate 筛选结果

七个预注册变体均已完成 `T5-10-48`、训练种子1/2/3、300轮训练，随后使用
validation100-A 选择 checkpoint。validation100-B 与固定 test100 只用于独立
确认或否证，没有参与 checkpoint 或变体选择。筛选汇总审计为 `valid=true`，并且：

- 七个变体均使用相同的三套100实例库与相同 event tape；
- validation-A、validation-B、test100 两两互斥；
- 所有模型、所有分割的非法动作数均为0；
- 所有必需训练诊断均为有限值；
- warmup 第50轮按预注册修正不具备选模资格；
- `test_used_for_selection=false`。

validation-A 三种子平均 realized makespan 为：

| 变体 | makespan | 完整预注册规则 |
|---|---:|---:|
| Adaptive-current | 15.9069 | 通过 |
| Adaptive-bias2 | 15.9310 | 通过 |
| Adaptive-warmup | 15.9872 | 未通过 |
| Adaptive-score | 15.9025 | 通过 |
| Adaptive-softplus | 15.9087 | 通过 |
| NoGate | 15.9805 | 对照 |
| SingleHead | 15.9613 | 对照 |

四个 Adaptive 候选通过完整预注册规则；只按 validation-A 的预注册排序，选择
`Adaptive-score` 进入正式实验。该结果说明 Adaptive 在三种子筛选中获得了继续验证
资格，但**尚不能代替四规模、五训练种子的正式统计结论**。

权威证据：

- `outputs/gate_screening/summary.json`
- `outputs/gate_screening/seed_level_comparisons.json`
- `reports/GATE_THREE_SEED_SCREENING.md`

## 10. Step 4 正式协议冻结

正式协议已冻结为 `configs/PHASE1_FROZEN_PROTOCOL.json`，模型定义记录在
`docs/PHASE1_MODEL_DEFINITIONS.md`：

- `GPPO-Literal`：task-message sigmoid gate、bias=0、不二次归一化的论文兼容解释；
- `GPPO-Best`：validation-A 选择的 `Adaptive-score`；
- `PPO-MLP`：不使用图编码器的普通PPO；
- `NoGate`：自适应 gate 消融；
- `SingleHead`：简单注意力结构对照。

冻结后禁止依据正式 test100 修改结构或超参数。冻结协议明确使用四个论文规模、五个
独立训练种子、2000轮、rollout/batch均为512，并按训练种子计算 Student-t 95% CI。
Gate筛选及冻结产物已在提交 `d9e5643` 同步到远端分支 `8.8-GPPO无偏好`。

## 11. Step 5–7 正式矩阵执行状态

四规模五种子正式矩阵已启动。原有 `GPPO-Literal-event` 五个2000轮 checkpoint
通过 validation-A 候选重选冻结为 `checkpoint_phase1_frozen.pt`，原始
`checkpoint.pt` 保持不变；重选清单 `PHASE1_CANDIDATE_RESELECTION.json` 为
`valid=true`。

当前首批 `T5-10-48 / PPO-none` 四个并行训练进程正常运行，随后调度器会依次完成：

1. T5 的 PPO-event、Literal-none、NoGate-event、SingleHead-event 五种子；
2. 其余三个规模的完整主矩阵；
3. 四规模 `GPPO-Best (Adaptive-score)-event` 五种子；
4. 固定 Literal-event checkpoint 的 None/Event/Periodic/Full 同实例同事件带重放；
5. `PHASE1_COMMUNICATION_CAUSAL_AUDIT.json` 通信因果审计。

截至本次更新，正式训练未发现 traceback、NaN、Inf、非法动作或 checkpoint 覆盖。
最新完整回归测试为 `60 passed`。Phase 1A 尚未完成，Phase 1B 多源扰动实现不得提前
作为已完成内容申报。

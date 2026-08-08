"""Write the final Chinese GPPO mechanism-level reproduction report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PAPER_SCALES = ("T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Chinese paper-faithful GPPO report")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--generalization", type=Path)
    parser.add_argument("--inference-scaling", type=Path)
    return parser.parse_args()


def f(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def ci(metric: dict[str, object]) -> str:
    mean = f(metric.get("mean"))
    half = metric.get("ci95_half_width")
    return mean if half is None or not isinstance(half, (int, float)) or half != half else f"{mean} +/- {f(half)}"


def _paired_row(summary: dict[str, object], scale: str, left: str, right: str) -> dict[str, object] | None:
    """Return a native-scale left-minus-right comparison.

    The summarizer stores one lexicographically ordered orientation. Invert
    the reported difference when callers request the opposite orientation so
    acceptance logic does not depend on label ordering.
    """
    for row in summary.get("paired_method_differences", []):
        if row.get("training_scale") != scale or row.get("evaluation_scale") != scale:
            continue
        if row.get("left_label") == left and row.get("right_label") == right:
            return row
        if row.get("left_label") == right and row.get("right_label") == left:
            metric = dict(row.get("realized_makespan_difference", {}))
            if metric.get("mean") is not None:
                metric["mean"] = -float(metric["mean"])
            old_low = metric.get("ci95_low")
            old_high = metric.get("ci95_high")
            if old_low is not None and old_high is not None:
                metric["ci95_low"] = -float(old_high)
                metric["ci95_high"] = -float(old_low)
            return {
                **row,
                "left_label": left,
                "right_label": right,
                "realized_makespan_difference": metric,
            }
    return None


def _direction_check(summary: dict[str, object], left: str, right: str) -> dict[str, object]:
    rows = [_paired_row(summary, scale, left, right) for scale in PAPER_SCALES]
    rows = [row for row in rows if row is not None]
    means = [float(row["realized_makespan_difference"]["mean"]) for row in rows]
    ci_below_zero = [
        float(row["realized_makespan_difference"].get("ci95_high", float("nan"))) < 0.0
        for row in rows
    ]
    return {
        "left": left,
        "right": right,
        "scales_observed": len(rows),
        "negative_mean_scales": sum(value < 0.0 for value in means),
        "negative_ci_scales": sum(ci_below_zero),
        "means": means,
        "pass": len(rows) == len(PAPER_SCALES)
        and sum(value < 0.0 for value in means) >= 3
        and sum(ci_below_zero) >= 2,
    }


def _communication_check(native: list[dict[str, object]]) -> dict[str, object]:
    by_key = {(row.get("evaluation_scale"), row.get("label")): row for row in native}
    rows = []
    for scale in PAPER_SCALES:
        event = by_key.get((scale, "literal_expected_task_message_event"))
        full = by_key.get((scale, "literal_expected_task_message_always"))
        if not event or not full:
            continue
        event_metrics = event.get("metrics", {})
        full_metrics = full.get("metrics", {})
        event_m = float(event_metrics["realized_makespan"]["mean"])
        full_m = float(full_metrics["realized_makespan"]["mean"])
        event_b = float(event_metrics["communication_bytes"]["mean"])
        full_b = float(full_metrics["communication_bytes"]["mean"])
        rows.append({
            "scale": scale,
            "relative_quality_gap": (event_m - full_m) / max(abs(full_m), 1e-9),
            "event_bytes": event_b,
            "full_bytes": full_b,
        })
    return {
        "scales_observed": len(rows),
        "rows": rows,
        "pass": len(rows) == len(PAPER_SCALES)
        and all(
            item["event_bytes"] < item["full_bytes"]
            and item["relative_quality_gap"] <= 0.05
            for item in rows
        ),
    }


def _baseline_check(
    native: list[dict[str, object]], baseline: dict[str, object] | None
) -> dict[str, object]:
    learned = {
        str(row["evaluation_scale"]): float(row["metrics"]["realized_makespan"]["mean"])
        for row in native
        if row.get("label") == "literal_expected_task_message_event"
    }
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    if baseline:
        for result in baseline.get("results", []):
            grouped[(str(result["scale"]), str(result["policy"]))].append(
                float(result["summary"]["realized_makespan"]["mean"])
            )
    rows = []
    for scale in PAPER_SCALES:
        random_values = grouped.get((scale, "random"), [])
        greedy_values = grouped.get((scale, "greedy"), [])
        if scale not in learned or not random_values or not greedy_values:
            continue
        gppo = learned[scale]
        random_mean = sum(random_values) / len(random_values)
        greedy_mean = sum(greedy_values) / len(greedy_values)
        rows.append({
            "scale": scale,
            "gppo": gppo,
            "random": random_mean,
            "greedy": greedy_mean,
            "gppo_vs_random_relative": (gppo - random_mean) / max(abs(random_mean), 1e-9),
            "gppo_vs_greedy_relative": (gppo - greedy_mean) / max(abs(greedy_mean), 1e-9),
        })
    random_pass = len(rows) == len(PAPER_SCALES) and all(
        row["gppo_vs_random_relative"] < 0.0 for row in rows
    )
    greedy_pass = len(rows) == len(PAPER_SCALES) and sum(
        row["gppo_vs_greedy_relative"] <= 0.05 for row in rows
    ) >= 3
    return {
        "scales_observed": len(rows),
        "rows": rows,
        "random_pass": random_pass,
        "greedy_close_pass": greedy_pass,
        "pass": random_pass and greedy_pass,
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    baseline = json.loads(args.baselines.read_text(encoding="utf-8")) if args.baselines and args.baselines.exists() else None
    general = json.loads(args.generalization.read_text(encoding="utf-8")) if args.generalization and args.generalization.exists() else None
    scaling = json.loads(args.inference_scaling.read_text(encoding="utf-8")) if args.inference_scaling and args.inference_scaling.exists() else None

    audit_valid = bool(audit.get("valid", False))
    groups = summary.get("summary", [])
    native = [row for row in groups if row.get("training_scale") == row.get("evaluation_scale")]
    native_keys = {(row["evaluation_scale"], row["label"]) for row in native}
    required = [
        (scale, "literal_expected_task_message_event") for scale in PAPER_SCALES
    ]
    formal_complete = audit_valid and all(key in native_keys for key in required)
    recovered_checkpoint_count = int(
        audit.get("observed", {}).get("recovered_checkpoint_count", 0)
    )
    label = lambda graph, sync: f"{graph}_expected_task_message_{sync}"
    acceptance = {
        "formal_artifacts_complete": formal_complete,
        "gppo_vs_paper_ppo": _direction_check(
            summary, label("literal", "event"), label("ppo_mlp", "none")
        ),
        "graph_contribution": _direction_check(
            summary, label("literal", "event"), label("ppo_mlp", "event")
        ),
        "adaptive_gate_contribution": _direction_check(
            summary, label("literal", "event"), label("literal_no_gate", "event")
        ),
        "single_head_contrast": _direction_check(
            summary, label("literal", "event"), label("literal_single_head", "event")
        ),
        "communication_tradeoff": _communication_check(native),
        "non_learning_baselines": _baseline_check(native, baseline),
    }
    overall_pass = bool(acceptance["gppo_vs_paper_ppo"]["pass"])
    random_pass = bool(acceptance["non_learning_baselines"]["random_pass"])
    if not formal_complete or not overall_pass or not random_pass:
        classification = "未通过"
    else:
        # The simulator/code are unavailable, so even positive mechanism
        # trends cannot be called numerical reproduction of the source paper.
        classification = "部分通过"

    lines = [
        "# GPPO 机制级复现最终报告",
        "",
        "> 本报告针对论文《Multi-UAV Dynamic Task Assignment Based on Event-Triggered Graph Reinforcement Learning Under Weak Communication》给出独立的机制级/结构级复现结论。原论文没有公开完整环境和训练代码，因此不能称为数值级复现。",
        "",
        f"## 最终分类：{classification}",
        "",
        "分类依据：正式产物完整 = `{}`；GPPO 相对 PPO = `{}`；GPPO 优于 Random = `{}`；GPPO 接近 Greedy = `{}`；图结构 = `{}`；adaptive gate = `{}`；single-head 对照 = `{}`；event/full 通信折衷 = `{}`。".format(
            acceptance["formal_artifacts_complete"],
            acceptance["gppo_vs_paper_ppo"]["pass"],
            acceptance["non_learning_baselines"]["random_pass"],
            acceptance["non_learning_baselines"]["greedy_close_pass"],
            acceptance["graph_contribution"]["pass"],
            acceptance["adaptive_gate_contribution"]["pass"],
            acceptance["single_head_contrast"]["pass"],
            acceptance["communication_tradeoff"]["pass"],
        ),
        "",
        "## 1. 论文公式与实现对齐",
        "",
        "- Eq.(1) 和 Eq.(2) 使用 RReLU；正式默认采用确定性 expected-RReLU，随机 RReLU 仅作为敏感性变量。",
        "- Eq.(2) 的 self score 使用独立的 `W^T v_k`，不复用 UAV 的 `W^U v_k`。",
        "- self node 与可执行 task neighbors 进入同一个 attention softmax 域。",
        "- Eq.(3) 的 `f_{i,j,k}` 正式冻结为 task-message 调制，`alpha * gate` 后不追加未在论文中规定的二次归一化；score/aggregate 作为显式敏感性实现。",
        "- Eq.(4)-(5) 使用四种关系 MLP 和 fusion MLP；未加入额外 relation bias、通信分支、LayerNorm 或非论文残差。",
        "",
        "## 2. 正式协议",
        "",
        "- 规模：T5-10-48、T10-10-53、T15-8-66、T20-10-92。",
        "- 每个规模使用 disjoint train/validation/test bank，各 100 个实例；跨方法共享固定 event tape。",
        "- 不设置 hard deadline、weather hazard 或 action-conflict trigger；主指标为 realized makespan，所有 active task 必须完成。",
        "- PPO 参数：learning rate 2e-4、gamma 0.99、clip 0.2、entropy 0.01、value coefficient 0.5、batch 512、2000 iterations、五个训练种子。",
        "- checkpoint 只依据 validation realized makespan 选择，test bank 不参与选择。",
        f"- 训练连续性：有 `{recovered_checkpoint_count}` 个 checkpoint 经候选权重续训恢复；这些恢复保留模型权重和历史，但未恢复 Adam 状态、RNG 与 episode offset，故相关五种子结果必须标为优化过程非完全连续。",
        "",
        "## 3. 原型负结果与正式结果边界",
        "",
        "旧 `gppo-v2-hard-3` 已冻结为 `Hard Dynamic Extension`，保留其 adaptive、NoGate、SingleHead 的负结果，但不再把它当作论文数值基线。其弱通信缓存修正前的观测泄漏、非论文奖励和过低任务难度会混淆模块归因。",
        "",
        "## 4. 正式 native-scale 结果",
        "",
        "| 训练/测试规模 | 方法 | 种子数 | active 参数 | total 参数 | realized makespan | completion | 通信字节 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in native:
        metrics = row.get("metrics", {})
        lines.append(
            f"| {row['evaluation_scale']} | {row['label']} | {row['training_seed_count']} | "
            f"{','.join(str(value) for value in row.get('active_parameter_counts', [])) or 'NA'} | "
            f"{','.join(str(value) for value in row.get('total_parameter_counts', [])) or 'NA'} | "
            f"{ci(metrics.get('realized_makespan', {}))} | {ci(metrics.get('completion_rate', {}))} | "
            f"{ci(metrics.get('communication_bytes', {}))} |"
        )

    lines.extend([
        "",
        "### 4.1 projected 与最终 realized makespan 误差",
        "",
        "每个决策点的 projected makespan 与该实例最终 realized makespan 比较，而不是与当时的已完成任务时长比较。",
        "",
        "| 规模 | 方法 | 平均相对误差 | 每实例 P95 相对误差 |",
        "|---|---|---:|---:|",
    ])
    for row in native:
        metrics = row.get("metrics", {})
        if "projection_error_relative_mean" not in metrics:
            continue
        lines.append(
            f"| {row['evaluation_scale']} | {row['label']} | "
            f"{ci(metrics['projection_error_relative_mean'])} | "
            f"{ci(metrics['projection_error_relative_p95'])} |"
        )

    comparison_specs = (
        ("完整 GPPO vs 论文式 PPO", label("literal", "event"), label("ppo_mlp", "none")),
        ("Graph（event 条件）", label("literal", "event"), label("ppo_mlp", "event")),
        ("Graph（none 条件）", label("literal", "none"), label("ppo_mlp", "none")),
        ("Event 同步贡献", label("literal", "event"), label("literal", "none")),
        ("Adaptive gate", label("literal", "event"), label("literal_no_gate", "event")),
        ("Adaptive vs SingleHead", label("literal", "event"), label("literal_single_head", "event")),
    )
    lines.extend([
        "",
        "### 4.2 五 seed 配对方法差异",
        "",
        "差异定义为左方法减右方法；realized makespan 差异为负表示左方法更好。区间为训练 seed 层面的双侧 Student-t 95% CI。",
        "",
        "| 规模 | 对照 | 左方法 | 右方法 | makespan 差异 | 实例胜率 |",
        "|---|---|---|---|---:|---:|",
    ])
    for scale in PAPER_SCALES:
        for comparison_name, left_label, right_label in comparison_specs:
            row = _paired_row(summary, scale, left_label, right_label)
            if row is None:
                continue
            lines.append(
                f"| {scale} | {comparison_name} | {left_label} | {right_label} | "
                f"{ci(row['realized_makespan_difference'])} | "
                f"{f(row.get('mean_instance_win_rate'))} |"
            )

    lines.extend([
        "",
        "### 4.3 Gate 分布",
        "",
        "| 规模 | 方法 | gate 统计 seed 数 | gate mean 范围 | P05 范围 | P95 范围 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in native:
        gate_rows = [item["gate"] for item in row.get("gate_summaries", []) if item.get("gate")]
        if not gate_rows:
            continue
        means = [float(item["mean"]) for item in gate_rows]
        p05s = [float(item["p05"]) for item in gate_rows]
        p95s = [float(item["p95"]) for item in gate_rows]
        lines.append(
            f"| {row['evaluation_scale']} | {row['label']} | {len(gate_rows)} | "
            f"{min(means):.3f}--{max(means):.3f} | "
            f"{min(p05s):.3f}--{max(p05s):.3f} | "
            f"{min(p95s):.3f}--{max(p95s):.3f} |"
        )

    lines.extend([
        "",
        "## 5. 机制问题逐项结论",
        "",
        "### 5.1 Adaptive 为什么可能不如 single-head",
        "",
        "旧 hard 结果不能单独支持 gate 失败。正式实现已修正 self transform、RReLU 和联合 softmax；pilot 中 gate 分布非恒定且 PPO probe 梯度非零。因此剩余差异应由结构归纳偏置、有效参数量/估计方差和训练稳定性解释，必须以五 seed 正式 CI 为准，不能事后调 gate bias 直到出现预期排序。",
        "",
        "### 5.2 NoGate 与 Adaptive 的关系",
        "",
        "NoGate 只移除 gate 参数，保留同一图结构；若正式 paired CI 包含 0，应结论为 gate 增益未被确认，而不是宣称 gate 无效。若 CI 稳定偏向 Literal，才可称 gate 有独立正贡献。",
        "",
        "### 5.3 Graph、同步和 reward 的归因",
        "",
        "完整 GPPO-event 与 PPO-none 是论文式总对照；PPO-none/event 与 Literal-none/event 是 Graph x Sync 2x2。奖励使用 projected makespan 的增量（与论文 Eq.(17) 一致），验收指标使用 realized makespan，二者均在 trace 中保存，不能混写。",
        "",
        "### 5.4 Event 与 Full 的通信因果对照",
        "",
        "通信表固定 event-trained checkpoint，仅切换 `sync_mode` 重放；不分别训练 periodic/full 策略。若 event 的 realized makespan 接近 full 且字节数显著更低，才支持低通信成本结论；否则报告质量-成本 trade-off。",
        "",
        "| 规模 | Event 相对 Full 质量差 (%) | Event 字节 | Full 字节 | 通信减少率 (%) |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in acceptance["communication_tradeoff"]["rows"]:
        reduction = 100.0 * (1.0 - row["event_bytes"] / max(row["full_bytes"], 1e-9))
        lines.append(
            f"| {row['scale']} | {100.0 * row['relative_quality_gap']:.3f} | "
            f"{row['event_bytes']:.1f} | {row['full_bytes']:.1f} | {reduction:.2f} |"
        )
    lines.extend([
        "",
        "## 6. 非学习基线",
        "",
    ])
    if baseline:
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for result in baseline.get("results", []):
            grouped[(str(result["scale"]), str(result["policy"]))].append(result)
        lines.extend([
            "随机、belief-state earliest-finish greedy 和 true-state/full-sync earliest-finish oracle 已单独评估。oracle 是一阶启发式上界参考，不是精确组合优化最优值。",
            "",
            "| 规模 | 基线 | policy seed 数 | realized makespan（跨 seed 均值） | completion |",
            "|---|---|---:|---:|---:|",
        ])
        for (scale, policy), rows in sorted(grouped.items()):
            makespan = [float(row["summary"]["realized_makespan"]["mean"]) for row in rows]
            completion = [float(row["summary"]["completion_rate"]["mean"]) for row in rows]
            lines.append(f"| {scale} | {policy} | {len(rows)} | {f(sum(makespan) / len(makespan))} | {f(sum(completion) / len(completion))} |")
        lines.extend([
            "",
            "| 规模 | GPPO makespan | Random | Greedy | GPPO-Random (%) | GPPO-Greedy (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in acceptance["non_learning_baselines"]["rows"]:
            lines.append(
                f"| {row['scale']} | {row['gppo']:.3f} | {row['random']:.3f} | "
                f"{row['greedy']:.3f} | {100.0 * row['gppo_vs_random_relative']:.2f} | "
                f"{100.0 * row['gppo_vs_greedy_relative']:.2f} |"
            )
    else:
        lines.append("基线 JSON 尚未提供；因此不能宣称 GPPO 已经优于随机或接近合理贪心。")

    lines.extend(["", "## 7. 泛化与推理效率", ""])
    if general:
        lines.append(f"跨尺度已知规模比较：{len(general.get('known_scale_comparison', []))} 个；未知规模压力测试：{general.get('unknown_scale_count', 0)} 个，每个 50 个实例。")
    else:
        lines.append("泛化结果待正式 general checkpoint 生成。")
    if scaling:
        lines.append(f"固定 UAV 推理缩放 Pearson R：{f(scaling.get('pearson_r_subtasks_mean_inference_ms'))}。")
    else:
        lines.append("推理缩放结果待正式 general checkpoint 生成。")

    lines.extend([
        "",
        "## 8. 证据审计与限制",
        "",
        f"formal artifact audit.valid = `{audit_valid}`；checkpoint 数 = `{audit.get('observed', {}).get('checkpoint_count', 'NA')}`；evaluation 数 = `{audit.get('observed', {}).get('evaluation_count', 'NA')}`。",
        f"候选快照中断恢复 checkpoint 数 = `{recovered_checkpoint_count}`。该项作为审计警告而非文件完整性错误；若这些种子进入统计，结论不得表述为精确可续训或完全连续的五种子实验。",
        "",
        "原论文完整环境、代码和随机数细节未公开；论文正文对 f 的作用域也存在排版/文字歧义。因此本报告只对公式、数据协议、同步机制和统计流程作可复核的结构级判断，不把结果写成原论文数值复现。PCRL、JEPA/world model 接入继续后置，直到 GPPO 的正式总对照、模块消融和通信表通过审计。",
        "",
        "## 9. 后续进入 PCRL 的门槛",
        "",
        "1. Literal-event 在多数论文规模上相对 PPO-none 的 paired makespan CI 稳定为负；",
        "2. GPPO-event 明确优于 random，并至少接近 belief-state greedy；",
        "3. NoGate/SingleHead 的差异有五 seed 统计证据；",
        "4. event/full 使用固定 checkpoint 的质量-通信 trade-off 已报告；",
        "5. 以上条件未满足前，不把 PCRL 或世界模型的结果解释为 GPPO 改进。",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "classification": classification}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch


METHODS = {
    "GPPO-event": "literal_expected_task_message_event",
    "PPO-none": "ppo_mlp_expected_task_message_none",
    "PPO-event": "ppo_mlp_expected_task_message_event",
    "GPPO-none": "literal_expected_task_message_none",
    "GPPO-NoGate-event": "literal_no_gate_expected_task_message_event",
    "GPPO-SingleHead-event": "literal_single_head_expected_task_message_event",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the single-seed 100-iteration mechanism check")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def paired_difference(left: dict[str, object], right: dict[str, object], metric: str) -> dict[str, float]:
    left_rows = {int(row["instance_seed"]): float(row[metric]) for row in left["rows"]}
    right_rows = {int(row["instance_seed"]): float(row[metric]) for row in right["rows"]}
    if set(left_rows) != set(right_rows):
        raise ValueError(f"test instance mismatch for paired {metric}")
    values = [left_rows[key] - right_rows[key] for key in sorted(left_rows)]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    # n=100 in this protocol; 1.984 is the two-sided t critical value for df=99.
    half = 1.984216951 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "median": statistics.median(values),
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n_instances": len(values),
    }


def metric(payload: dict[str, object], name: str, field: str = "mean") -> float:
    return float(payload["summary"][name][field])


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    evaluations: dict[str, dict[str, object]] = {}
    checkpoints: dict[str, dict[str, object]] = {}
    checkpoint_rows: list[dict[str, object]] = []
    for checkpoint_path in sorted(args.root.glob("**/checkpoint.pt")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        label = checkpoint_path.parent.name
        checkpoints[label] = checkpoint
        checkpoint_rows.append(
            {
                "directory": label,
                "path": checkpoint_path.relative_to(args.root).as_posix(),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
        )
        training = checkpoint.get("training", {})
        expected = {
            "iterations": 100,
            "rollout_steps": 512,
            "batch_size": 512,
            "validation_interval": 50,
            "validation_instances": 20,
            "seed": 1,
        }
        for key, value in expected.items():
            if int(training.get(key, -1)) != value:
                errors.append(f"{label}: {key}={training.get(key)} expected {value}")
        if [int(row["iteration"]) for row in checkpoint.get("validation_history", [])] != [50, 100]:
            errors.append(f"{label}: validation history is not exactly [50, 100]")
        if len(checkpoint.get("history", [])) != 100:
            errors.append(f"{label}: training history length is not 100")
        evaluation_path = checkpoint_path.parent / "evaluations" / "test_native_100.json"
        if not evaluation_path.exists():
            errors.append(f"missing {evaluation_path}")
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluations[str(evaluation["label"])] = evaluation

    missing = [name for name, label in METHODS.items() if label not in evaluations]
    if missing:
        errors.append(f"missing method evaluations: {missing}")
    if errors:
        raise ValueError("; ".join(errors))

    instance_sets = {
        name: {int(row["instance_seed"]) for row in evaluations[label]["rows"]}
        for name, label in METHODS.items()
    }
    if any(len(seeds) != 100 for seeds in instance_sets.values()):
        raise ValueError("each method must contain 100 unique test instances")
    first = next(iter(instance_sets.values()))
    if any(seeds != first for seeds in instance_sets.values()):
        raise ValueError("methods do not share the same fixed test100 bank")
    tape_sets = {
        name: set(evaluations[label].get("event_tape_hashes", []))
        for name, label in METHODS.items()
    }
    first_tapes = next(iter(tape_sets.values()))
    if not first_tapes or any(tapes != first_tapes for tapes in tape_sets.values()):
        raise ValueError("methods do not share the same fixed event-tape bank")
    all_tasks_completed = all(
        metric(evaluations[label], "all_tasks_completed") == 1.0
        for label in METHODS.values()
    )
    if not all_tasks_completed:
        raise ValueError("at least one method failed to complete every test task")

    gppo = evaluations[METHODS["GPPO-event"]]
    always_path = Path(str(gppo["checkpoint"])) .parent / "evaluations" / "test_native_always_100.json"
    if not always_path.exists():
        raise FileNotFoundError(always_path)
    full = json.loads(always_path.read_text(encoding="utf-8"))
    if gppo.get("checkpoint_sha256") != full.get("checkpoint_sha256"):
        raise ValueError("Event/Full comparison does not use the same checkpoint")
    if set(gppo.get("event_tape_hashes", [])) != set(full.get("event_tape_hashes", [])):
        raise ValueError("Event/Full comparison does not use the same event tapes")

    baseline_payload = json.loads(args.baselines.read_text(encoding="utf-8"))
    baselines = {
        str(row["policy"]): row
        for row in baseline_payload.get("results", [])
        if row.get("scale") == "T5-10-48" and int(row.get("policy_seed", -1)) == 1
    }
    for name in ("random", "greedy"):
        if name not in baselines:
            raise ValueError(f"missing baseline: {name}/seed1")

    comparisons = {
        "GPPO-event_vs_PPO-none": paired_difference(gppo, evaluations[METHODS["PPO-none"]], "realized_makespan"),
        "GPPO-event_vs_PPO-event": paired_difference(gppo, evaluations[METHODS["PPO-event"]], "realized_makespan"),
        "GPPO-none_vs_PPO-none": paired_difference(evaluations[METHODS["GPPO-none"]], evaluations[METHODS["PPO-none"]], "realized_makespan"),
        "GPPO-event_vs_GPPO-none": paired_difference(gppo, evaluations[METHODS["GPPO-none"]], "realized_makespan"),
        "Adaptive_vs_NoGate": paired_difference(gppo, evaluations[METHODS["GPPO-NoGate-event"]], "realized_makespan"),
        "Adaptive_vs_SingleHead": paired_difference(gppo, evaluations[METHODS["GPPO-SingleHead-event"]], "realized_makespan"),
    }
    gppo_makespan = metric(gppo, "realized_makespan")
    random_makespan = float(baselines["random"]["summary"]["realized_makespan"]["mean"])
    greedy_makespan = float(baselines["greedy"]["summary"]["realized_makespan"]["mean"])
    full_makespan = metric(full, "realized_makespan")
    event_bytes = metric(gppo, "communication_bytes")
    full_bytes = metric(full, "communication_bytes")
    communication = {
        "event_makespan": gppo_makespan,
        "full_makespan": full_makespan,
        "quality_gap_percent": 100.0 * (gppo_makespan - full_makespan) / max(abs(full_makespan), 1e-12),
        "event_bytes": event_bytes,
        "full_bytes": full_bytes,
        "communication_reduction_percent": 100.0 * (full_bytes - event_bytes) / max(full_bytes, 1e-12),
    }
    gates_better = (
        comparisons["Adaptive_vs_NoGate"]["mean"] < 0.0
        and comparisons["Adaptive_vs_SingleHead"]["mean"] < 0.0
    )
    gppo_continue = (
        comparisons["GPPO-event_vs_PPO-none"]["mean"] < 0.0
        and gppo_makespan < random_makespan
    )
    if not gppo_continue:
        decision = "停止进入 PCRL/世界模型；先修复 GPPO 基线"
    elif not gates_better:
        decision = "继续 GPPO 正式复现，但停止宣称 adaptive gate 有益并优先排查该模块"
    else:
        decision = "继续完整五种子正式复现；当前快速验证方向通过"

    methods = []
    for name, label in METHODS.items():
        row = evaluations[label]
        methods.append({
            "method": name,
            "label": label,
            "episode_return": row["summary"]["episode_return"],
            "realized_makespan": row["summary"]["realized_makespan"],
            "completion_rate": row["summary"]["completion_rate"],
            "communication_bytes": row["summary"]["communication_bytes"],
        })
    result = {
        "version": "paper-faithful-quick-mechanism-v1",
        "scope": "single-scale, single-training-seed, 100-iteration quick mechanism validation",
        "not_claimed": "This is not a multi-seed or paper-level statistical reproduction.",
        "protocol": {"scale": "T5-10-48", "training_seed": 1, "iterations": 100, "rollout_steps": 512, "batch_size": 512, "validation_iterations": [50, 100], "validation_instances": 20, "test_instances": 100},
        "methods": methods,
        "comparisons": comparisons,
        "baselines": {name: baselines[name]["summary"] for name in ("random", "greedy")},
        "communication": communication,
        "checks": {
            "gppo_better_than_paper_ppo_on_test100": comparisons["GPPO-event_vs_PPO-none"]["mean"] < 0.0,
            "gppo_better_than_random": gppo_makespan < random_makespan,
            "gppo_within_5_percent_of_greedy": gppo_makespan <= 1.05 * greedy_makespan,
            "adaptive_better_than_nogate": comparisons["Adaptive_vs_NoGate"]["mean"] < 0.0,
            "adaptive_better_than_singlehead": comparisons["Adaptive_vs_SingleHead"]["mean"] < 0.0,
            "event_within_5_percent_of_full": abs(communication["quality_gap_percent"]) <= 5.0,
            "event_uses_fewer_bytes_than_full": event_bytes < full_bytes,
        },
        "decision": decision,
        "artifact_audit": {
            "valid": True,
            "checkpoint_count": len(checkpoint_rows),
            "native_test100_evaluation_count": len(evaluations),
            "checkpoint_rows": checkpoint_rows,
            "training_protocol_exact": True,
            "validation_iterations_exact": True,
            "unique_test_instances_per_method": 100,
            "same_fixed_test_bank": True,
            "same_fixed_event_tape_bank": True,
            "all_tasks_completed": all_tasks_completed,
            "event_full_same_checkpoint": True,
            "event_full_same_event_tapes": True,
        },
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    audit_path = args.json_output.parent / "quick_artifact_audit.json"
    audit_path.write_text(
        json.dumps(result["artifact_audit"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# GPPO 单种子快速机制验证报告",
        "",
        "> 范围：T5-10-48、训练 seed=1、100 iterations、固定 test100。该结果仅用于方向筛查，不能证明 GPPO 稳定优于 PPO，也不能替代五种子论文级统计复现。",
        "",
        "## 结论",
        "",
        f"**{decision}**",
        "",
        "## 六模型结果",
        "",
        "| 方法 | Return mean | Return median | Makespan mean | Makespan median | Completion | Comm bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in methods:
        lines.append(
            f"| {row['method']} | {row['episode_return']['mean']:.4f} | {row['episode_return']['median']:.4f} | "
            f"{row['realized_makespan']['mean']:.4f} | {row['realized_makespan']['median']:.4f} | "
            f"{row['completion_rate']['mean']:.4f} | {row['communication_bytes']['mean']:.1f} |"
        )
    lines += ["", "## 配对 test100 差值", "", "负 makespan 差值表示左侧方法更好。95% CI 是固定模型在 100 个测试实例上的实例级区间，不是训练种子置信区间。", "", "| 对比 | Mean Δ makespan | Median Δ | 95% CI |", "|---|---:|---:|---:|"]
    for name, row in comparisons.items():
        lines.append(f"| {name} | {row['mean']:.4f} | {row['median']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |")
    lines += [
        "", "## Random / Greedy", "",
        f"- Random makespan mean：{random_makespan:.4f}",
        f"- Greedy makespan mean：{greedy_makespan:.4f}",
        "", "## Event / Full 同 checkpoint 重放", "",
        f"- Event makespan：{gppo_makespan:.4f}",
        f"- Full makespan：{full_makespan:.4f}",
        f"- 相对质量差：{communication['quality_gap_percent']:.2f}%",
        f"- Event bytes：{event_bytes:.1f}",
        f"- Full bytes：{full_bytes:.1f}",
        f"- 通信减少率：{communication['communication_reduction_percent']:.2f}%",
        "", "## 判读边界", "",
        "- 这里只能判断单规模、单训练种子、短训练预算下的机制方向。",
        "- test100 的实例级 CI 不能替代至少五个训练种子的 CI。",
        "- 若 adaptive 未同时优于 NoGate 和 SingleHead，应暂停 adaptive 有益的论文式表述。",
        "- 若 GPPO-event 未优于 PPO-none 或 Random，应暂停 PCRL 与世界模型接入，先修 GPPO。",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()

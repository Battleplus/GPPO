from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


ADAPTIVE = (
    "Adaptive-current",
    "Adaptive-bias2",
    "Adaptive-warmup",
    "Adaptive-score",
    "Adaptive-softplus",
)
BASELINES = ("NoGate", "SingleHead")
SPLITS = ("validation_a", "validation_b", "test")


def slug(name: str) -> str:
    return name.lower().replace("adaptive-", "adaptive_").replace("-", "_")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError("Command failed:\n" + " ".join(command) + "\n" + "\n".join(tail))


def mean_ci95(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    # Student-t 0.975 quantiles for df 1..30. Screening always has n=3.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(
        max(1, array.size - 1), 1.96
    )
    half = float(critical * std / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {"n": int(array.size), "mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def metric(payload: dict[str, Any], name: str) -> float:
    return float(payload["summary"][name]["mean"])


def clear_opposite_sign(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_significant = first["ci95"][1] < 0 or first["ci95"][0] > 0
    second_significant = second["ci95"][1] < 0 or second["ci95"][0] > 0
    return bool(first_significant and second_significant and first["mean"] * second["mean"] < 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize preregistered Phase-1 gate screening.")
    parser.add_argument("--screening-root", type=Path, default=Path("outputs/gate_screening/screening"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gate_screening"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/GATE_DIAGNOSTIC_PROTOCOL.json"))
    parser.add_argument(
        "--runtime-cohorts",
        type=Path,
        default=Path("configs/GATE_SCREENING_RUNTIME_COHORTS.json"),
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    screening_root = args.screening_root.resolve()
    manifest_path = screening_root / "run_manifest.json"
    manifest = read_json(manifest_path)
    protocol = read_json(args.protocol)
    runtime_cohorts = read_json(args.runtime_cohorts)
    if not manifest.get("valid") or any(
        record.get("status") not in {"completed", "reused"} for record in manifest["records"]
    ):
        raise RuntimeError("Screening training manifest is not complete and valid")
    if manifest["protocol_sha256"] != sha256(args.protocol):
        raise RuntimeError("Frozen protocol hash differs from screening launch")

    records = {(record["variant"], int(record["seed"])): record for record in manifest["records"]}
    variants = [entry["name"] for entry in protocol["variants"]]
    seeds = [int(seed) for seed in protocol["screening_protocol"]["training_seeds"]]
    expected = {(variant, seed) for variant in variants for seed in seeds}
    if set(records) != expected:
        raise RuntimeError("Training records do not match frozen 7 x 3 matrix")
    declared_runs = {
        run
        for cohort in runtime_cohorts["cohorts"]
        for run in cohort["runs"]
    }
    expected_run_labels = {f"{variant}:{seed}" for variant, seed in expected}
    if declared_runs != expected_run_labels:
        raise RuntimeError("Runtime cohorts do not cover the exact 7 x 3 screening matrix")
    if runtime_cohorts["protocol_sha256"] != manifest["protocol_sha256"]:
        raise RuntimeError("Runtime cohort protocol hash differs from launch manifest")
    current_cohort = runtime_cohorts["cohorts"][-1]
    if any(sha256(Path(path)) != digest for path, digest in current_cohort["files_sha256"].items()):
        raise RuntimeError("Frozen training source changed after runtime cohort registration")

    evaluations: dict[tuple[str, int, str], Path] = {}
    commands: list[tuple[list[str], Path]] = []
    evaluator = repo / "evaluate_paper_faithful.py"
    diagnostic = repo / "diagnose_paper_faithful_gate.py"
    for variant, seed in sorted(expected):
        checkpoint = Path(records[(variant, seed)]["checkpoint"])
        if sha256(checkpoint) != records[(variant, seed)]["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {variant} seed {seed}")
        for split in SPLITS:
            output = screening_root / "evaluations" / slug(variant) / f"seed{seed}" / f"{split}.json"
            evaluations[(variant, seed, split)] = output
            if output.is_file():
                payload = read_json(output)
                if payload.get("checkpoint_sha256") == sha256(checkpoint) and payload.get("instances") == 100:
                    continue
            command = [
                sys.executable, str(evaluator), "--checkpoint", str(checkpoint),
                "--split", split, "--instances", "100", "--output", str(output),
            ]
            commands.append((command, output.with_suffix(".log")))
        if variant in ADAPTIVE:
            output = screening_root / "diagnostics" / slug(variant) / f"seed{seed}.json"
            if not output.is_file() or read_json(output).get("checkpoint_sha256") != sha256(checkpoint):
                commands.append(([
                    sys.executable, str(diagnostic), "--checkpoint", str(checkpoint),
                    "--split", "validation_a", "--instances", "100", "--output", str(output),
                ], output.with_suffix(".log")))

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        future_map = {executor.submit(run_logged, command, log): log for command, log in commands}
        for index, future in enumerate(as_completed(future_map), 1):
            future.result()
            print(f"[{index}/{len(future_map)}] {future_map[future]} completed", flush=True)

    payloads = {key: read_json(path) for key, path in evaluations.items()}
    comparisons: dict[str, Any] = {}
    for adaptive in ADAPTIVE:
        comparisons[adaptive] = {}
        for baseline in BASELINES:
            split_stats: dict[str, Any] = {}
            for split in SPLITS:
                differences = [
                    metric(payloads[(adaptive, seed, split)], "realized_makespan")
                    - metric(payloads[(baseline, seed, split)], "realized_makespan")
                    for seed in seeds
                ]
                split_stats[split] = {
                    "adaptive_minus_baseline_by_seed": dict(zip(map(str, seeds), differences)),
                    "paired_seed_statistics": mean_ci95(differences),
                    "adaptive_better_seed_count": sum(value < 0 for value in differences),
                }
            a_stats = split_stats["validation_a"]["paired_seed_statistics"]
            b_stats = split_stats["validation_b"]["paired_seed_statistics"]
            t_stats = split_stats["test"]["paired_seed_statistics"]
            split_stats["raw_rank_reversal_validation_a_vs_test"] = bool(a_stats["mean"] * t_stats["mean"] < 0)
            split_stats["clear_rank_reversal_validation_a_vs_test"] = clear_opposite_sign(a_stats, t_stats)
            split_stats["clear_rank_reversal_validation_a_vs_validation_b"] = clear_opposite_sign(a_stats, b_stats)
            comparisons[adaptive][baseline] = split_stats

    decisions: dict[str, Any] = {}
    for adaptive in ADAPTIVE:
        diagnostics = [
            read_json(screening_root / "diagnostics" / slug(adaptive) / f"seed{seed}.json")
            for seed in seeds
        ]
        eval_rows = [payloads[(adaptive, seed, split)] for seed in seeds for split in SPLITS]
        versus = comparisons[adaptive]
        validation_only = {
            "better_than_nogate_at_least_2_of_3": versus["NoGate"]["validation_a"]["adaptive_better_seed_count"] >= 2,
            "better_than_singlehead_at_least_2_of_3": versus["SingleHead"]["validation_a"]["adaptive_better_seed_count"] >= 2,
            "mean_not_worse_than_nogate": versus["NoGate"]["validation_a"]["paired_seed_statistics"]["mean"] <= 0,
            "mean_not_worse_than_singlehead": versus["SingleHead"]["validation_a"]["paired_seed_statistics"]["mean"] <= 0,
        }
        mechanics = {
            "zero_invalid_actions": all(metric(item, "invalid_actions") == 0 for item in eval_rows),
            "gate_nonconstant": all(float(item["gate"]["std"]) > 0 for item in diagnostics),
            "gate_has_gradient": all(float(item["gate_gradient_l2_actual_ppo_probe"]) > 0 for item in diagnostics),
        }
        confirmation = {
            "no_clear_validation_test_rank_reversal": not any(
                versus[baseline]["clear_rank_reversal_validation_a_vs_test"] for baseline in BASELINES
            ),
            "no_clear_validation_a_b_rank_reversal": not any(
                versus[baseline]["clear_rank_reversal_validation_a_vs_validation_b"] for baseline in BASELINES
            ),
        }
        decisions[adaptive] = {
            "validation_only_checks": validation_only,
            "mechanism_checks": mechanics,
            "confirmation_checks": confirmation,
            "selected_on_validation_a": all(validation_only.values()) and all(mechanics.values()),
            "passes_full_preregistered_rule": all(validation_only.values()) and all(mechanics.values()) and all(confirmation.values()),
        }

    eligible = [name for name in ADAPTIVE if decisions[name]["passes_full_preregistered_rule"]]
    validation_means = {
        variant: float(np.mean([metric(payloads[(variant, seed, "validation_a")], "realized_makespan") for seed in seeds]))
        for variant in variants
    }
    selected_adaptive = min(eligible, key=validation_means.get) if eligible else None
    engineering_baseline = min(BASELINES, key=validation_means.get)
    outcome = {
        "adaptive_candidates_passing": eligible,
        "selected_adaptive": selected_adaptive,
        "selection_basis": "validation_a only among preregistered candidates; confirmation can veto but never reselect",
        "engineering_baseline_if_no_adaptive_passes": engineering_baseline,
        "required_negative_statement": None if eligible else "未独立复现 Adaptive 正收益",
    }
    seed_results: dict[str, Any] = {}
    histories: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for variant in variants:
        seed_results[variant] = {}
        for seed in seeds:
            run_root = Path(records[(variant, seed)]["output"])
            history = read_json(run_root / "training_history.json")
            histories[(variant, seed)] = history
            per_split = {split: payloads[(variant, seed, split)] for split in SPLITS}
            item: dict[str, Any] = {
                "checkpoint_sha256": records[(variant, seed)]["checkpoint_sha256"],
                "best_iteration": per_split["validation_a"].get("best_iteration"),
                "final_training_metrics": history[-1],
                "evaluation": {
                    split: {
                        "realized_makespan": metric(payload, "realized_makespan"),
                        "completion_rate": metric(payload, "completion_rate"),
                        "invalid_actions": metric(payload, "invalid_actions"),
                        "inference_latency_ms": payload["inference_latency_ms"],
                        "active_parameter_count": payload["active_parameter_count"],
                        "total_parameter_count": payload["total_parameter_count"],
                    }
                    for split, payload in per_split.items()
                },
            }
            if variant in ADAPTIVE:
                diagnostic_payload = read_json(
                    screening_root / "diagnostics" / slug(variant) / f"seed{seed}.json"
                )
                item["gate_diagnostic"] = {
                    key: diagnostic_payload[key]
                    for key in (
                        "gate", "attention_on_task_edges", "task_attention_mass_per_uav",
                        "removed_attention_mass_per_uav", "uav_attention_output_l2_norm",
                        "gate_gradient_l2_policy_sensitivity",
                        "gate_gradient_l2_actual_ppo_probe",
                    )
                }
            seed_results[variant][str(seed)] = item

    event_tape_checks: dict[str, bool] = {}
    instance_bank_checks: dict[str, bool] = {}
    for split in SPLITS:
        split_payloads = [payloads[(variant, seed, split)] for variant in variants for seed in seeds]
        reference_tapes = split_payloads[0]["event_tape_hashes"]
        reference_seeds = [row["instance_seed"] for row in split_payloads[0]["rows"]]
        event_tape_checks[split] = all(item["event_tape_hashes"] == reference_tapes for item in split_payloads)
        instance_bank_checks[split] = (
            len(reference_seeds) == 100
            and len(set(reference_seeds)) == 100
            and all([row["instance_seed"] for row in item["rows"]] == reference_seeds for item in split_payloads)
        )
    validation_banks_disjoint = not (
        {row["instance_seed"] for row in payloads[(variants[0], seeds[0], "validation_a")]["rows"]}
        & {row["instance_seed"] for row in payloads[(variants[0], seeds[0], "validation_b")]["rows"]}
    )
    test_disjoint = not (
        {row["instance_seed"] for row in payloads[(variants[0], seeds[0], "test")]["rows"]}
        & {
            row["instance_seed"]
            for split in ("validation_a", "validation_b")
            for row in payloads[(variants[0], seeds[0], split)]["rows"]
        }
    )
    warmup_histories_valid = True
    for seed in seeds:
        run_root = Path(records[("Adaptive-warmup", seed)]["output"])
        validation_history = read_json(run_root / "validation_history.json")
        warmup_histories_valid &= all(
            bool(entry["selection_eligible"]) == (float(entry["iteration"]) > 50)
            for entry in validation_history
        )
        warmup_histories_valid &= int(
            payloads[("Adaptive-warmup", seed, "validation_a")]["best_iteration"]
        ) > 50
    finite_training_diagnostics = all(
        all(
            isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
            for key in (
                "policy_entropy", "ppo_approx_kl", "ppo_clip_fraction", "value_loss",
                "realized_makespan",
            )
        )
        for history in histories.values()
        for row in history
    )
    global_checks = {
        "all_splits_use_identical_event_tapes_across_models": all(event_tape_checks.values()),
        "all_splits_use_identical_100_instance_banks_across_models": all(instance_bank_checks.values()),
        "validation_a_and_b_are_disjoint": validation_banks_disjoint,
        "test_is_disjoint_from_both_validation_banks": test_disjoint,
        "warmup_iteration_50_is_never_selection_eligible": warmup_histories_valid,
        "required_training_diagnostics_are_finite": finite_training_diagnostics,
        "zero_invalid_actions_all_models_all_splits": all(
            metric(payload, "invalid_actions") == 0 for payload in payloads.values()
        ),
    }
    summary = {
        "version": "phase1-gate-three-seed-screening-v1",
        "protocol_sha256": manifest["protocol_sha256"],
        "amendment_sha256": manifest["amendment_sha256"],
        "training_code_commit": manifest["code_commit"],
        "runtime_cohorts": runtime_cohorts,
        "test_used_for_selection": False,
        "variants": variants,
        "seeds": seeds,
        "validation_a_mean_makespan": validation_means,
        "seed_results": seed_results,
        "global_checks": global_checks,
        "decisions": decisions,
        "outcome": outcome,
        "valid": all(global_checks.values()),
    }
    write_json(args.output_root / "summary.json", summary)
    write_json(args.output_root / "seed_level_comparisons.json", comparisons)

    lines = [
        "# Gate 三训练种子筛选报告", "",
        f"训练代码提交：`{manifest['code_commit']}`。协议哈希：`{manifest['protocol_sha256']}`。", "",
        "checkpoint 和候选结构只由 validation-A 选择；validation-B 与 test100 仅用于独立确认或否证，未用于重新挑选。", "",
        "| 变体 | validation-A makespan | 验证集入选 | 完整规则通过 |", "|---|---:|---:|---:|",
    ]
    for variant in variants:
        decision = decisions.get(variant)
        lines.append(
            f"| {variant} | {validation_means[variant]:.4f} | "
            f"{('是' if decision and decision['selected_on_validation_a'] else '否/不适用')} | "
            f"{('是' if decision and decision['passes_full_preregistered_rule'] else '否/不适用')} |"
        )
    lines += ["", "## 冻结判定", ""]
    if selected_adaptive:
        lines.append(f"Adaptive 晋级候选：`{selected_adaptive}`。工程基线仍保留 `{engineering_baseline}` 对照。")
    else:
        lines.append(f"未独立复现 Adaptive 正收益。工程基线冻结为 validation-A 更优的 `{engineering_baseline}`；Literal 负结果完整保留。")
    lines += ["", "详细种子差值、Student-t 95% CI 与排序反转审计见 `outputs/gate_screening/seed_level_comparisons.json`。", ""]
    report = repo / "reports" / "GATE_THREE_SEED_SCREENING.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(args.output_root / 'summary.json'), "outcome": outcome}, ensure_ascii=False))


if __name__ == "__main__":
    main()

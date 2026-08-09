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

import matplotlib.pyplot as plt
import numpy as np
import torch


MODELS = {
    "Adaptive": "literal_event_seed1",
    "NoGate": "literal_no_gate_event_seed1",
    "SingleHead": "literal_single_head_event_seed1",
}
ITERATIONS = (50, 100, 150, 200, 250, 300)
SPLITS = ("validation_a", "validation_b", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(payload: dict[str, Any], name: str) -> float:
    return float(payload["summary"][name]["mean"])


def paired_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float | int]:
    left_rows = {int(row["instance_seed"]): row for row in left["rows"]}
    right_rows = {int(row["instance_seed"]): row for row in right["rows"]}
    if left_rows.keys() != right_rows.keys():
        raise ValueError("Paired evaluation banks differ")
    values = np.asarray(
        [
            float(left_rows[seed]["realized_makespan"])
            - float(right_rows[seed]["realized_makespan"])
            for seed in sorted(left_rows)
        ],
        dtype=np.float64,
    )
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(values.mean() - 1.984 * standard_error),
        "ci95_high": float(values.mean() + 1.984 * standard_error),
    }


def rank_order(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda name: (values[name], name))


def spearman_three(left: dict[str, float], right: dict[str, float]) -> float:
    left_order = rank_order(left)
    right_order = rank_order(right)
    left_rank = {name: rank for rank, name in enumerate(left_order, 1)}
    right_rank = {name: rank for rank, name in enumerate(right_order, 1)}
    squared = sum((left_rank[name] - right_rank[name]) ** 2 for name in left)
    return float(1.0 - 6.0 * squared / (3 * (3**2 - 1)))


def evaluate_one(
    evaluator: Path,
    checkpoint: Path,
    metadata_checkpoint: Path,
    split: str,
    output: Path,
) -> tuple[str, bool]:
    expected_hash = sha256_file(checkpoint)
    if output.is_file():
        existing = load_json(output)
        if (
            existing.get("checkpoint_sha256") == expected_hash
            and existing.get("split") == split
            and int(existing.get("instances", 0)) == 100
        ):
            return str(output), True
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(evaluator),
        "--checkpoint",
        str(checkpoint),
        "--metadata-checkpoint",
        str(metadata_checkpoint),
        "--split",
        split,
        "--instances",
        "100",
        "--sync-mode",
        "event",
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for {checkpoint} / {split}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return str(output), False


def build_report(result: dict[str, Any]) -> str:
    lines = [
        "# 候选 Checkpoint 重新评估",
        "",
        "> 只使用 validation100-A 选择 checkpoint；validation100-B 和原 test100 仅复核。",
        "",
        "## 每轮结果",
        "",
        "| 模型 | 轮次 | validation-A | validation-B | test100 | completion A/B/test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for iteration in ITERATIONS:
            row = result["candidates"][model][str(iteration)]
            lines.append(
                f"| {model} | {iteration} | {row['validation_a']['makespan']:.4f} | "
                f"{row['validation_b']['makespan']:.4f} | {row['test']['makespan']:.4f} | "
                f"{row['validation_a']['completion']:.3f} / {row['validation_b']['completion']:.3f} / {row['test']['completion']:.3f} |"
            )
    lines += ["", "## validation-A 选择结果", ""]
    for model, row in result["selection"].items():
        lines.append(
            f"- {model}：第 {row['selected_iteration']} 轮；validation-A / B / test100 "
            f"makespan = {row['validation_a_makespan']:.4f} / {row['validation_b_makespan']:.4f} / {row['test_makespan']:.4f}。"
            f"冻结 checkpoint 实际保存第 {row['frozen_checkpoint_iteration']} 轮权重"
            f"（逐张量一致：{row['frozen_weights_match_candidate']}）。"
        )
    lines += [
        "",
        "## Adaptive 配对差值",
        "",
        "正值表示 Adaptive 更差；区间为实例级 95% CI。",
        "",
        "| 轮次 | 数据库 | Adaptive-NoGate | 95% CI | Adaptive-SingleHead | 95% CI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for iteration in ITERATIONS:
        for split in SPLITS:
            row = result["comparisons"][str(iteration)][split]
            ng = row["Adaptive_vs_NoGate"]
            sh = row["Adaptive_vs_SingleHead"]
            lines.append(
                f"| {iteration} | {split} | {ng['mean']:.4f} | "
                f"[{ng['ci95_low']:.4f}, {ng['ci95_high']:.4f}] | {sh['mean']:.4f} | "
                f"[{sh['ci95_low']:.4f}, {sh['ci95_high']:.4f}] |"
            )
    lines += [
        "",
        "## 排序一致性",
        "",
        "| 轮次 | validation-A排序 | validation-B排序 | test100排序 | A-test Spearman | B-test Spearman |",
        "|---:|---|---|---|---:|---:|",
    ]
    for iteration, row in result["ranking_consistency"].items():
        lines.append(
            f"| {iteration} | {' < '.join(row['validation_a_order'])} | "
            f"{' < '.join(row['validation_b_order'])} | {' < '.join(row['test_order'])} | "
            f"{row['validation_a_test_spearman']:.3f} | {row['validation_b_test_spearman']:.3f} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- test100 未参与 checkpoint 选择。",
        "- 当前仍是单训练种子诊断，不能用实例级区间替代训练种子级区间。",
        "- 若 validation-A 选择与 validation-B/test100 排名反转，应视为 checkpoint 选择不稳定，而不是从 test100 重新挑选。",
        "- 冻结的 300 轮 Adaptive 负结果保持不变；本报告只判断候选轮次和小验证集选模是否影响结论。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate frozen Phase-1 candidate checkpoints.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoint_reevaluation"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    evaluator = Path(__file__).resolve().parent / "evaluate_paper_faithful.py"
    jobs: list[tuple[Path, Path, str, Path]] = []
    for model, directory in MODELS.items():
        for iteration in ITERATIONS:
            checkpoint = raw_root / "T5-10-48" / directory / f"candidate_{iteration:04d}.pt"
            metadata_checkpoint = raw_root / "T5-10-48" / directory / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if not metadata_checkpoint.is_file():
                raise FileNotFoundError(metadata_checkpoint)
            for split in SPLITS:
                output = output_root / "evaluations" / model / f"candidate_{iteration:04d}_{split}.json"
                jobs.append((checkpoint, metadata_checkpoint, split, output))

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                evaluate_one, evaluator, checkpoint, metadata_checkpoint, split, output
            ): (checkpoint, split)
            for checkpoint, metadata_checkpoint, split, output in jobs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            checkpoint, split = futures[future]
            path, reused = future.result()
            print(f"[{completed:02d}/{len(futures)}] {checkpoint.name} {split}: {'reused' if reused else path}")

    payloads: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    candidates: dict[str, dict[str, dict[str, Any]]] = {}
    for model in MODELS:
        payloads[model] = {}
        candidates[model] = {}
        for iteration in ITERATIONS:
            payloads[model][iteration] = {}
            candidates[model][str(iteration)] = {}
            for split in SPLITS:
                path = output_root / "evaluations" / model / f"candidate_{iteration:04d}_{split}.json"
                evaluation = load_json(path)
                payloads[model][iteration][split] = evaluation
                candidates[model][str(iteration)][split] = {
                    "makespan": mean_metric(evaluation, "realized_makespan"),
                    "completion": mean_metric(evaluation, "completion_rate"),
                    "checkpoint_sha256": evaluation["checkpoint_sha256"],
                    "instance_seeds_sha256": hashlib.sha256(
                        json.dumps([row["instance_seed"] for row in evaluation["rows"]]).encode()
                    ).hexdigest(),
                    "event_tape_hashes_sha256": hashlib.sha256(
                        json.dumps(evaluation["event_tape_hashes"]).encode()
                    ).hexdigest(),
                }

    bank_checks: dict[str, bool] = {}
    for split in SPLITS:
        seeds = {
            tuple(row["instance_seed"] for row in payloads[model][iteration][split]["rows"])
            for model in MODELS
            for iteration in ITERATIONS
        }
        tapes = {
            tuple(payloads[model][iteration][split]["event_tape_hashes"])
            for model in MODELS
            for iteration in ITERATIONS
        }
        bank_checks[f"same_{split}_instances"] = len(seeds) == 1
        bank_checks[f"same_{split}_event_tapes"] = len(tapes) == 1
    representative_banks = {
        split: set(row["instance_seed"] for row in payloads["Adaptive"][50][split]["rows"])
        for split in SPLITS
    }
    bank_checks["validation_a_b_disjoint"] = representative_banks["validation_a"].isdisjoint(
        representative_banks["validation_b"]
    )
    bank_checks["validation_a_test_disjoint"] = representative_banks["validation_a"].isdisjoint(
        representative_banks["test"]
    )
    bank_checks["validation_b_test_disjoint"] = representative_banks["validation_b"].isdisjoint(
        representative_banks["test"]
    )

    selection: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        selected_iteration = min(
            ITERATIONS,
            key=lambda iteration: (
                candidates[model][str(iteration)]["validation_a"]["makespan"],
                iteration,
            ),
        )
        selected = candidates[model][str(selected_iteration)]
        metadata_path = raw_root / "T5-10-48" / MODELS[model] / "checkpoint.pt"
        frozen_checkpoint = torch.load(metadata_path, map_location="cpu", weights_only=False)
        frozen_iteration = int(frozen_checkpoint["best_iteration"])
        frozen_candidate_path = (
            raw_root / "T5-10-48" / MODELS[model] / f"candidate_{frozen_iteration:04d}.pt"
        )
        frozen_candidate = torch.load(frozen_candidate_path, map_location="cpu", weights_only=False)
        weights_match = all(
            torch.equal(frozen_checkpoint["model_state"][key], frozen_candidate["model_state"][key])
            for key in frozen_checkpoint["model_state"]
        )
        selection[model] = {
            "selection_source": "validation_a_only",
            "selected_iteration": selected_iteration,
            "validation_a_makespan": selected["validation_a"]["makespan"],
            "validation_b_makespan": selected["validation_b"]["makespan"],
            "test_makespan": selected["test"]["makespan"],
            "final_iteration": 300,
            "final_validation_a_makespan": candidates[model]["300"]["validation_a"]["makespan"],
            "final_validation_b_makespan": candidates[model]["300"]["validation_b"]["makespan"],
            "final_test_makespan": candidates[model]["300"]["test"]["makespan"],
            "frozen_checkpoint_iteration": frozen_iteration,
            "frozen_weights_match_candidate": weights_match,
            "frozen_validation_a_makespan": candidates[model][str(frozen_iteration)]["validation_a"]["makespan"],
            "frozen_validation_b_makespan": candidates[model][str(frozen_iteration)]["validation_b"]["makespan"],
            "frozen_test_makespan": candidates[model][str(frozen_iteration)]["test"]["makespan"],
            "change_100_to_300": {
                split: candidates[model]["300"][split]["makespan"]
                - candidates[model]["100"][split]["makespan"]
                for split in SPLITS
            },
        }

    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    ranking_consistency: dict[str, dict[str, Any]] = {}
    for iteration in ITERATIONS:
        comparisons[str(iteration)] = {}
        for split in SPLITS:
            comparisons[str(iteration)][split] = {
                "Adaptive_vs_NoGate": paired_difference(
                    payloads["Adaptive"][iteration][split], payloads["NoGate"][iteration][split]
                ),
                "Adaptive_vs_SingleHead": paired_difference(
                    payloads["Adaptive"][iteration][split], payloads["SingleHead"][iteration][split]
                ),
            }
        values = {
            split: {
                model: candidates[model][str(iteration)][split]["makespan"] for model in MODELS
            }
            for split in SPLITS
        }
        ranking_consistency[str(iteration)] = {
            "validation_a_order": rank_order(values["validation_a"]),
            "validation_b_order": rank_order(values["validation_b"]),
            "test_order": rank_order(values["test"]),
            "validation_a_test_spearman": spearman_three(values["validation_a"], values["test"]),
            "validation_b_test_spearman": spearman_three(values["validation_b"], values["test"]),
        }

    result = {
        "version": "phase1-candidate-checkpoint-reevaluation-v1",
        "protocol": "configs/CHECKPOINT_REEVALUATION_PROTOCOL.json",
        "selection_uses_test": False,
        "selection_uses_validation_b": False,
        "bank_checks": bank_checks,
        "valid": all(bank_checks.values()),
        "candidates": candidates,
        "selection": selection,
        "comparisons": comparisons,
        "ranking_consistency": ranking_consistency,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = build_report(result)
    report_path = Path(__file__).resolve().parent / "reports" / "CANDIDATE_CHECKPOINT_REEVALUATION.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, split in zip(axes, SPLITS):
        for model in MODELS:
            axis.plot(
                ITERATIONS,
                [candidates[model][str(iteration)][split]["makespan"] for iteration in ITERATIONS],
                marker="o",
                label=model,
            )
        axis.set_title(split.replace("_", "-"))
        axis.set_xlabel("candidate iteration")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("realized makespan (lower is better)")
    axes[-1].legend()
    figure.tight_layout()
    figure.savefig(output_root / "learning_curve.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"valid": result["valid"], "selection": selection}, indent=2, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) > 1:
        critical = T_975.get(len(array) - 1, 1.96)
        half = float(critical * np.std(array, ddof=1) / math.sqrt(len(array)))
    else:
        half = float("nan")
    return {
        "mean": mean,
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else float("nan"),
        "median": float(np.median(array)),
        "n": len(array),
        "ci_method": "two-sided Student-t, 95%",
        "ci95_half_width": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "significant_two_sided_0_05": bool(half == half and (mean - half > 0.0 or mean + half < 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize formal paper-faithful evaluations")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.root.glob("**/evaluations/test_*.json"))
    if not files:
        raise FileNotFoundError("no formal evaluation files")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    numeric = (
        "episode_return", "realized_makespan", "completion_rate",
        "all_tasks_completed", "communication_events", "communication_bytes",
        "heartbeat_messages", "mean_inference_ms",
        "inference_p50_ms", "inference_p95_ms",
        "estimated_forward_flops",
        "projection_error_relative_mean", "projection_error_relative_p95",
    )
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for payload in payloads:
        key = (
            str(payload.get("training_scale", payload["scale"])),
            str(payload["scale"]),
            str(payload["label"]),
        )
        grouped[key].append(payload)
    summary: list[dict[str, object]] = []
    for (training_scale, eval_scale, label), rows in sorted(grouped.items()):
        per_seed = []
        for payload in rows:
            seed_row = {
                metric: float(payload["summary"][metric]["mean"])
                for metric in numeric
                if metric in payload["summary"]
            }
            seed_row["training_seed"] = int(payload.get("training_seed", -1))
            latency = payload.get("inference_latency_ms", {})
            if "p50" in latency:
                seed_row["inference_p50_ms"] = float(latency["p50"])
            if "p95" in latency:
                seed_row["inference_p95_ms"] = float(latency["p95"])
            per_seed.append(seed_row)
        summary.append(
            {
                "training_scale": training_scale,
                "evaluation_scale": eval_scale,
                "label": label,
                "training_seed_count": len(rows),
                "metrics": {
                    metric: mean_ci([float(row[metric]) for row in per_seed])
                    for metric in numeric
                    if all(metric in row for row in per_seed)
                },
                "gate_summaries": [
                    {
                        "training_seed": int(payload.get("training_seed", -1)),
                        "gate": payload.get("gate_summary"),
                    }
                    for payload in rows
                    if payload.get("gate_summary") is not None
                ],
                "active_parameter_counts": sorted({
                    int(payload["active_parameter_count"])
                    for payload in rows
                    if payload.get("active_parameter_count") is not None
                }),
                "total_parameter_counts": sorted({
                    int(payload["total_parameter_count"])
                    for payload in rows
                    if payload.get("total_parameter_count") is not None
                }),
                "recovery_infos": [
                    {
                        "training_seed": int(payload.get("training_seed", -1)),
                        "recovery": payload.get("recovery_info"),
                    }
                    for payload in rows
                    if payload.get("recovery_info") is not None
                ],
                "resume_events": [
                    {
                        "training_seed": int(payload.get("training_seed", -1)),
                        "events": payload.get("resume_events", []),
                    }
                    for payload in rows
                    if payload.get("resume_events")
                ],
                "per_seed": per_seed,
            }
        )
    base_lookup = {
        (row["training_scale"], row["label"]): row
        for row in summary
        if row["training_scale"] == row["evaluation_scale"]
    }
    generalization: list[dict[str, object]] = []
    for row in summary:
        if row["training_scale"] == row["evaluation_scale"]:
            continue
        base = base_lookup.get((row["training_scale"], row["label"]))
        if base is None:
            continue
        base_metric = base["metrics"]["realized_makespan"]["mean"]
        general_metric = row["metrics"]["realized_makespan"]["mean"]
        generalization.append(
            {
                "training_scale": row["training_scale"],
                "evaluation_scale": row["evaluation_scale"],
                "label": row["label"],
                "generalization_error_percent": 100.0 * abs(general_metric - base_metric) / max(abs(base_metric), 1e-9),
            }
        )
    inference_points = []
    for row in summary:
        if row["training_scale"] == row["evaluation_scale"]:
            scale_parts = row["evaluation_scale"].removeprefix("T").split("-")
            inference_points.append(
                {
                    "scale": row["evaluation_scale"],
                    "subtasks": int(scale_parts[-1]),
                    "label": row["label"],
                    "mean_inference_ms": row["metrics"]["mean_inference_ms"]["mean"],
                }
            )
    complexity_correlations: dict[str, float] = {}
    for label in sorted({point["label"] for point in inference_points}):
        points = [point for point in inference_points if point["label"] == label]
        if len(points) >= 3:
            complexity_correlations[label] = float(
                np.corrcoef(
                    [point["subtasks"] for point in points],
                    [point["mean_inference_ms"] for point in points],
                )[0, 1]
            )
    paired_cells: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    by_seed: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for payload in payloads:
        by_seed[
            (
                str(payload.get("training_scale", payload["scale"])),
                str(payload["scale"]),
                int(payload.get("training_seed", -1)),
            )
        ].append(payload)
    for (training_scale, eval_scale, seed), seed_payloads in sorted(by_seed.items()):
        by_label = {str(payload["label"]): payload for payload in seed_payloads}
        labels = sorted(by_label)
        for left_index, left_label in enumerate(labels):
            for right_label in labels[left_index + 1:]:
                left = by_label[left_label]
                right = by_label[right_label]
                left_rows = {int(row["instance_seed"]): row for row in left["rows"]}
                right_rows = {int(row["instance_seed"]): row for row in right["rows"]}
                common = sorted(set(left_rows) & set(right_rows))
                if not common:
                    continue
                if any(
                    left_rows[item]["event_tape_hash"] != right_rows[item]["event_tape_hash"]
                    for item in common
                ):
                    raise ValueError(
                        f"event-tape mismatch for {training_scale}/{eval_scale}/seed{seed}: "
                        f"{left_label} vs {right_label}"
                    )
                differences = [
                    float(left_rows[item]["realized_makespan"])
                    - float(right_rows[item]["realized_makespan"])
                    for item in common
                ]
                paired_cells[(training_scale, eval_scale, left_label, right_label)].append(
                    {
                        "training_seed": float(seed),
                        "instance_count": float(len(common)),
                        "mean_difference": float(np.mean(differences)),
                        "win_rate": float(np.mean(np.asarray(differences) < 0.0)),
                    }
                )
    paired_method_differences = []
    for (training_scale, eval_scale, left_label, right_label), cells in sorted(paired_cells.items()):
        paired_method_differences.append(
            {
                "training_scale": training_scale,
                "evaluation_scale": eval_scale,
                "left_label": left_label,
                "right_label": right_label,
                "definition": "left minus right; negative realized-makespan difference favors left",
                "training_seed_count": len(cells),
                "realized_makespan_difference": mean_ci(
                    [float(cell["mean_difference"]) for cell in cells]
                ),
                "mean_instance_win_rate": float(
                    np.mean([float(cell["win_rate"]) for cell in cells])
                ),
                "per_seed": cells,
            }
        )
    output = {
        "version": "paper-faithful-formal-summary-v1",
        "evaluation_files": len(files),
        "summary": summary,
        "generalization_error": generalization,
        "inference_complexity": {
            "points": inference_points,
            "pearson_r_by_label": complexity_correlations,
        },
        "paired_method_differences": paired_method_differences,
        "limitations": [
            "This file summarizes completed evaluation artifacts only.",
            "Generalization error requires a native-scale base row and a cross-scale row.",
            "Communication bytes use the documented graph-state serialization accounting.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "groups": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

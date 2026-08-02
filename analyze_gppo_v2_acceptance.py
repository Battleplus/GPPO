from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = ("deadline_completion_rate", "makespan", "communication_events")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen GPPO-v2 acceptance gates")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-seed", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(array), 1.96)
    mean = float(array.mean())
    half = (
        0.0
        if len(array) <= 1
        else float(critical * array.std(ddof=1) / np.sqrt(len(array)))
    )
    return {"mean": mean, "ci95_lower": mean - half, "ci95_upper": mean + half}


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    audit = json.loads(args.artifact_audit.read_text(encoding="utf-8"))
    summary_rows = payload["summary"]
    scales = sorted({str(row["scale"]) for row in summary_rows})

    macro: dict[str, dict[str, float]] = {}
    for method_id in sorted({str(row["method_id"]) for row in summary_rows}):
        rows = [row for row in summary_rows if row["method_id"] == method_id]
        macro[method_id] = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        }

    with args.per_seed.open(encoding="utf-8-sig", newline="") as stream:
        per_seed_rows = list(csv.DictReader(stream))
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[(row["method_id"], int(row["training_seed"]))].append(row)
    seed_macro: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for (method_id, seed), rows in grouped.items():
        seed_macro[method_id][seed] = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in METRICS
        }

    def paired(left: str, right: str) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for metric in METRICS:
            paired_seeds = sorted(set(seed_macro[left]) & set(seed_macro.get(right, {})))
            if paired_seeds:
                values = [
                    seed_macro[left][seed][metric] - seed_macro[right][seed][metric]
                    for seed in paired_seeds
                ]
            else:
                values = [
                    seed_macro[left][seed][metric] - macro[right][metric]
                    for seed in sorted(seed_macro[left])
                ]
            result[metric] = mean_ci(values)
        return result

    comparisons = {
        name: paired(left, right)
        for name, left, right in (
            ("gppo_event_minus_random", "gppo_event", "random_event"),
            ("gppo_event_minus_greedy", "gppo_event", "greedy_event"),
            ("gppo_event_minus_ppo_event", "gppo_event", "ppo_event"),
            ("gppo_event_minus_gppo_none", "gppo_event", "gppo_none"),
            ("gppo_event_minus_gppo_always", "gppo_event", "gppo_always"),
            ("single_head_minus_gppo_event", "gppo_event_single_head", "gppo_event"),
            ("no_gate_minus_gppo_event", "gppo_event_no_gate", "gppo_event"),
            ("gppo_event_minus_periodic", "gppo_event", "gppo_periodic"),
        )
    }

    random_dcr_gain = macro["greedy_event"]["deadline_completion_rate"] - macro["random_event"]["deadline_completion_rate"]
    random_makespan_gain = macro["random_event"]["makespan"] - macro["greedy_event"]["makespan"]
    gap_closure = {
        "deadline_completion_rate": (
            macro["gppo_event"]["deadline_completion_rate"]
            - macro["random_event"]["deadline_completion_rate"]
        )
        / random_dcr_gain,
        "makespan": (
            macro["random_event"]["makespan"] - macro["gppo_event"]["makespan"]
        )
        / random_makespan_gain,
    }

    random_scale_wins = 0
    random_reverse_scales: list[str] = []
    for scale in scales:
        comparison = next(
            row
            for row in payload["comparisons"]
            if row["comparison"] == "gppo_event-minus-random_event"
            and row["scale"] == scale
        )
        dcr_positive = comparison["deadline_completion_rate_delta_ci95_lower"] > 0
        makespan_positive = comparison["makespan_delta_ci95_upper"] < 0
        if dcr_positive and makespan_positive:
            random_scale_wins += 1
        if (
            comparison["deadline_completion_rate_delta_ci95_upper"] < 0
            or comparison["makespan_delta_ci95_lower"] > 0
        ):
            random_reverse_scales.append(scale)

    stable_seeds = 0
    catastrophic_seeds: list[int] = []
    seed_details: list[dict[str, float | int | bool]] = []
    for seed in sorted(seed_macro["gppo_event"]):
        dcr_delta = (
            seed_macro["gppo_event"][seed]["deadline_completion_rate"]
            - macro["random_event"]["deadline_completion_rate"]
        )
        makespan_delta = (
            seed_macro["gppo_event"][seed]["makespan"]
            - macro["random_event"]["makespan"]
        )
        simultaneous_win = dcr_delta > 0 and makespan_delta < 0
        if simultaneous_win:
            stable_seeds += 1
        catastrophic = dcr_delta < -0.05 or (
            seed_macro["gppo_event"][seed]["makespan"]
            > macro["random_event"]["makespan"] * 1.10
        )
        if catastrophic:
            catastrophic_seeds.append(seed)
        seed_details.append(
            {
                "seed": seed,
                "deadline_completion_rate": seed_macro["gppo_event"][seed][
                    "deadline_completion_rate"
                ],
                "deadline_completion_delta_vs_random": dcr_delta,
                "makespan": seed_macro["gppo_event"][seed]["makespan"],
                "makespan_delta_vs_random": makespan_delta,
                "simultaneous_win": simultaneous_win,
                "catastrophic": catastrophic,
            }
        )

    best_updates: list[int] = []
    validation_dcr_drift: list[float] = []
    validation_makespan_drift: list[float] = []
    for seed in sorted(seed_macro["gppo_event"]):
        root = args.training_root / "gppo_event" / f"seed_{seed}"
        history = json.loads((root / "validation_history.json").read_text(encoding="utf-8"))
        by_update = {int(float(row["update"])): row for row in history}
        validation_dcr_drift.append(
            float(by_update[100]["deadline_completion_rate"])
            - float(by_update[80]["deadline_completion_rate"])
        )
        validation_makespan_drift.append(
            float(by_update[100]["makespan"]) - float(by_update[80]["makespan"])
        )
        import torch

        checkpoint = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=False)
        best_updates.append(int(checkpoint["best_update"]))

    negative_counts = Counter(row["effect"] for row in payload["negative_results"])
    all_holm = [float(row["p_holm"]) for row in payload["negative_results"]]

    random_gate = (
        comparisons["gppo_event_minus_random"]["deadline_completion_rate"]["ci95_lower"] > 0
        and comparisons["gppo_event_minus_random"]["makespan"]["ci95_upper"] < 0
        and random_scale_wins >= 3
        and not random_reverse_scales
    )
    greedy_gate = (
        macro["greedy_event"]["deadline_completion_rate"]
        - macro["gppo_event"]["deadline_completion_rate"]
        <= 0.03
        and macro["gppo_event"]["makespan"]
        <= macro["greedy_event"]["makespan"] * 1.05
        and gap_closure["deadline_completion_rate"] >= 0.8
        and gap_closure["makespan"] >= 0.8
    )
    stability_gate = stable_seeds >= 4 and not catastrophic_seeds
    mechanism_gate = (
        comparisons["gppo_event_minus_ppo_event"]["deadline_completion_rate"]["mean"] > 0
        and comparisons["gppo_event_minus_ppo_event"]["makespan"]["mean"] < 0
        and comparisons["gppo_event_minus_gppo_none"]["deadline_completion_rate"]["mean"] > 0
        and comparisons["gppo_event_minus_gppo_none"]["makespan"]["mean"] < 0
    )
    communication_gate = (
        macro["gppo_event"]["communication_events"]
        < macro["gppo_always"]["communication_events"]
        and abs(
            macro["gppo_event"]["deadline_completion_rate"]
            - macro["gppo_always"]["deadline_completion_rate"]
        )
        <= 0.03
        and macro["gppo_event"]["makespan"]
        <= macro["gppo_always"]["makespan"] * 1.05
    )

    gates = {
        "protocol_complete": bool(audit.get("valid")),
        "clearly_better_than_random": random_gate,
        "approaches_greedy": greedy_gate,
        "stable_across_training_seeds": stability_gate,
        "mechanism_directions_correct": mechanism_gate,
        "event_communication_tradeoff": communication_gate,
    }
    report = {
        "overall_pass": all(gates.values()),
        "gates": gates,
        "macro_metrics": macro,
        "paired_macro_comparisons": comparisons,
        "gap_closure": gap_closure,
        "random_scale_wins": random_scale_wins,
        "random_reverse_scales": random_reverse_scales,
        "stable_seed_count": stable_seeds,
        "catastrophic_seeds": catastrophic_seeds,
        "gppo_event_seed_details": seed_details,
        "gppo_event_best_updates": best_updates,
        "validation_update_80_to_100": {
            "deadline_completion_rate": mean_ci(validation_dcr_drift),
            "makespan": mean_ci(validation_makespan_drift),
        },
        "negative_results": {
            "total": len(payload["negative_results"]),
            "by_effect": dict(negative_counts),
            "all_holm_p_values_equal_one": bool(all(value == 1.0 for value in all_holm)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

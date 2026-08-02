from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


VARIANTS = {
    "A": "variant_a_raw_preference_only",
    "B": "variant_b_balanced_capability_only",
    "C": "variant_c_split_plus_balanced",
    "D": "variant_d_full_hard5",
}
CONDITIONED = "pcrl_gppo_adaptive"
NO_CONDITIONING = "pcrl_gppo_adaptive_no_conditioning"
GPPO = "gppo_event"
PRIMARY_PROFILES = (
    "balanced",
    "calibration_search_priority",
    "calibration_reconnaissance_priority",
    "calibration_strike_priority",
    "calibration_recovery_priority",
)
PRIORITY_PROFILES = PRIMARY_PROFILES[1:]
HELD_OUT_PROFILES = (
    "calibration_search_strike_interp",
    "calibration_recon_recovery_interp",
    "calibration_asymmetric_search_recon_recovery",
)
DIRECTION_INDEX = {
    "calibration_search_priority": 0,
    "calibration_reconnaissance_priority": 1,
    "calibration_strike_priority": 2,
    "calibration_recovery_priority": 3,
}
T_CRITICAL_95_DF2 = 4.302652729696142


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hard-5 calibration variants")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/pcrl_v0/hard5/calibration10"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pcrl_v0/hard5/calibration10/summary"),
    )
    return parser.parse_args()


def mean_ci(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "lower": mean, "upper": mean}
    half = T_CRITICAL_95_DF2 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def load_rows(variant_root: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    result: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    eval_root = variant_root / "eval" / "controllability_phase"
    for method in (CONDITIONED, NO_CONDITIONING, GPPO):
        for seed_dir in sorted((eval_root / method).glob("seed_*")):
            seed = int(seed_dir.name.split("_", maxsplit=1)[1])
            payload = json.loads(
                (seed_dir / "evaluation.json").read_text(encoding="utf-8")
            )
            rows = payload if isinstance(payload, list) else payload["rows"]
            if len(rows) != 320:
                raise ValueError(
                    f"{variant_root.name}/{method}/seed_{seed} has {len(rows)} rows"
                )
            result[method][seed] = rows
    return result


def selected(rows: list[dict[str, Any]], profiles: tuple[str, ...]) -> list[dict[str, Any]]:
    return [row for row in rows if row["preference_profile"] in profiles]


def metric_mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def method_means(
    rows_by_method: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, dict[str, float]]:
    fields = (
        "preference_l1",
        "deadline_completion_rate",
        "makespan",
        "minimum_task_coverage",
        "invalid_actions",
    )
    result: dict[str, dict[str, float]] = {}
    for method, by_seed in rows_by_method.items():
        rows = [row for seed_rows in by_seed.values() for row in seed_rows]
        result[method] = {field: metric_mean(rows, field) for field in fields}
    return result


def improvement_summary(
    rows_by_method: dict[str, dict[int, list[dict[str, Any]]]],
    profiles: tuple[str, ...],
) -> dict[str, Any]:
    per_seed: dict[str, float] = {}
    for seed in sorted(rows_by_method[CONDITIONED]):
        conditioned = metric_mean(
            selected(rows_by_method[CONDITIONED][seed], profiles), "preference_l1"
        )
        comparator = metric_mean(
            selected(rows_by_method[NO_CONDITIONING][seed], profiles),
            "preference_l1",
        )
        per_seed[str(seed)] = (comparator - conditioned) / comparator
    return {
        "per_seed": per_seed,
        "ci95": mean_ci(per_seed.values()),
        "positive_seeds": sum(value > 0 for value in per_seed.values()),
    }


def direction_summary(
    rows_by_method: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile, component in DIRECTION_INDEX.items():
        per_seed: dict[str, float] = {}
        for seed, rows in rows_by_method[CONDITIONED].items():
            grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
            for row in rows:
                if row["preference_profile"] not in {"balanced", profile}:
                    continue
                key = (str(row["scale"]), int(row["evaluation_seed"]))
                grouped[key][str(row["preference_profile"])] = float(
                    row["priority_weighted_assignment_mix"][component]
                )
            differences = [
                values[profile] - values["balanced"]
                for values in grouped.values()
                if profile in values and "balanced" in values
            ]
            per_seed[str(seed)] = float(np.mean(differences))
        result[profile] = {
            "component": component,
            "per_seed": per_seed,
            "ci95": mean_ci(per_seed.values()),
            "positive_seeds": sum(value > 0 for value in per_seed.values()),
        }
    return result


def efficiency_summary(
    rows_by_method: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    fields = {
        "deadline_completion_delta": "deadline_completion_rate",
        "makespan_increase_ratio": "makespan",
        "minimum_task_coverage_delta": "minimum_task_coverage",
        "invalid_actions_delta": "invalid_actions",
    }
    result: dict[str, Any] = {}
    for output_name, field in fields.items():
        values: dict[str, float] = {}
        for seed in sorted(rows_by_method[CONDITIONED]):
            candidate = metric_mean(rows_by_method[CONDITIONED][seed], field)
            reference = metric_mean(rows_by_method[GPPO][seed], field)
            value = (
                candidate / reference - 1.0
                if output_name == "makespan_increase_ratio"
                else candidate - reference
            )
            values[str(seed)] = value
        result[output_name] = {
            "per_seed": values,
            "ci95": mean_ci(values.values()),
        }
    return result


def checkpoint_summary(variant_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in (CONDITIONED, NO_CONDITIONING):
        method_rows: dict[str, Any] = {}
        for checkpoint in sorted((variant_root / "train" / method).glob("seed_*/checkpoint.pt")):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            selection = payload.get("checkpoint_selection", {})
            seed = checkpoint.parent.name.split("_", maxsplit=1)[1]
            method_rows[seed] = {
                "best_update": int(payload.get("best_update", -1)),
                "mode": selection.get("mode"),
                "selection_passed": bool(selection.get("selection_passed", False)),
                "reasons": (selection.get("selected_diagnostics") or {}).get(
                    "reasons", []
                ),
            }
        result[method] = method_rows
    return result


def analyze_variant(variant_root: Path) -> dict[str, Any]:
    rows = load_rows(variant_root)
    overall = method_means(rows)
    improvements = {
        "all_profiles": improvement_summary(
            rows, PRIMARY_PROFILES + HELD_OUT_PROFILES
        ),
        "primary_five": improvement_summary(rows, PRIMARY_PROFILES),
        "priority_four": improvement_summary(rows, PRIORITY_PROFILES),
        "held_out_three": improvement_summary(rows, HELD_OUT_PROFILES),
    }
    directions = direction_summary(rows)
    efficiency = efficiency_summary(rows)
    checkpoints = checkpoint_summary(variant_root)
    point_efficiency_pass = (
        efficiency["deadline_completion_delta"]["ci95"]["mean"] >= -0.03
        and efficiency["makespan_increase_ratio"]["ci95"]["mean"] <= 0.05
        and efficiency["minimum_task_coverage_delta"]["ci95"]["mean"] >= -0.05
    )
    return {
        "variant_root": str(variant_root.resolve()),
        "method_means": overall,
        "improvements_vs_no_conditioning": improvements,
        "directions": directions,
        "efficiency_vs_gppo": efficiency,
        "checkpoints": checkpoints,
        "screening": {
            "primary_improvement_ge_30pct": improvements["primary_five"]["ci95"][
                "mean"
            ]
            >= 0.30,
            "all_directions_positive_3_of_3": all(
                item["positive_seeds"] == 3 for item in directions.values()
            ),
            "point_efficiency_guards_pass": point_efficiency_pass,
            "all_learned_checkpoint_selections_pass": all(
                item["selection_passed"]
                for method in checkpoints.values()
                for item in method.values()
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PCRL hard-5 calibration10 result",
        "",
        "This is three-seed screening evidence, not pilot or formal evidence.",
        "",
        "| Variant | Primary-5 L1 improvement vs no-conditioning | 95% CI | Positive seeds | Directionality | Efficiency | Selection |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for name, result in report["variants"].items():
        improvement = result["improvements_vs_no_conditioning"]["primary_five"]
        ci = improvement["ci95"]
        screening = result["screening"]
        lines.append(
            f"| {name} | {100 * ci['mean']:.2f}% | "
            f"[{100 * ci['lower']:.2f}%, {100 * ci['upper']:.2f}%] | "
            f"{improvement['positive_seeds']}/3 | "
            f"{'pass' if screening['all_directions_positive_3_of_3'] else 'fail'} | "
            f"{'pass' if screening['point_efficiency_guards_pass'] else 'fail'} | "
            f"{'pass' if screening['all_learned_checkpoint_selections_pass'] else 'diagnostic only'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "The 30% gate was not changed. Pilot20 and formal100 remain locked, and no world-model or JEPA work is authorized.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    variants = {
        name: analyze_variant(args.root / directory)
        for name, directory in VARIANTS.items()
    }
    passing = [
        name
        for name, result in variants.items()
        if result["screening"]["primary_improvement_ge_30pct"]
        and result["screening"]["all_directions_positive_3_of_3"]
        and result["screening"]["point_efficiency_guards_pass"]
        and result["screening"]["all_learned_checkpoint_selections_pass"]
    ]
    decision = (
        f"Calibration winner(s): {', '.join(passing)}."
        if passing
        else "No variant passes the preregistered 30% controllability gate; do not start pilot20."
    )
    report = {
        "version": "pcrl-v0-hard-5-calibration10-analysis-v1",
        "variants": variants,
        "passing_variants": passing,
        "decision": decision,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "PCRL_HARD5_CALIBRATION.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "passing": passing}))


if __name__ == "__main__":
    main()

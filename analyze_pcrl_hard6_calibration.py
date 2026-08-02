from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


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
    parser = argparse.ArgumentParser(description="Analyze hard-6 calibration20")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/pcrl_v0/hard6/calibration20"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pcrl_v0/hard6/calibration20/summary"),
    )
    return parser.parse_args()


def mean_ci(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "lower": mean, "upper": mean}
    half = T_CRITICAL_95_DF2 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def metric_mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def selected(
    rows: list[dict[str, Any]], profiles: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [row for row in rows if row["preference_profile"] in profiles]


def load_rows(root: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    result: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for run_root in sorted(root.glob("seed_*_run")):
        for method in (CONDITIONED, NO_CONDITIONING, GPPO):
            for seed_dir in (
                run_root / "eval" / "controllability_phase" / method
            ).glob("seed_*"):
                seed = int(seed_dir.name.split("_", maxsplit=1)[1])
                rows = json.loads(
                    (seed_dir / "evaluation.json").read_text(encoding="utf-8")
                )
                if len(rows) != 320:
                    raise ValueError(
                        f"{run_root.name}/{method}/seed_{seed} has {len(rows)} rows"
                    )
                result[method][seed] = rows
    expected = {11, 12, 13}
    for method in (CONDITIONED, NO_CONDITIONING, GPPO):
        if set(result[method]) != expected:
            raise ValueError(
                f"{method} seeds are {sorted(result[method])}, expected {sorted(expected)}"
            )
    return result


def improvement(
    rows: dict[str, dict[int, list[dict[str, Any]]]],
    profiles: tuple[str, ...],
) -> dict[str, Any]:
    per_seed: dict[str, float] = {}
    for seed in sorted(rows[CONDITIONED]):
        conditioned = metric_mean(
            selected(rows[CONDITIONED][seed], profiles), "preference_l1"
        )
        comparator = metric_mean(
            selected(rows[NO_CONDITIONING][seed], profiles), "preference_l1"
        )
        per_seed[str(seed)] = (comparator - conditioned) / comparator
    return {
        "per_seed": per_seed,
        "ci95": mean_ci(per_seed.values()),
        "positive_seeds": sum(value > 0 for value in per_seed.values()),
    }


def directions(
    rows: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile, component in DIRECTION_INDEX.items():
        per_seed: dict[str, float] = {}
        for seed, seed_rows in rows[CONDITIONED].items():
            paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
            for row in seed_rows:
                name = str(row["preference_profile"])
                if name not in {"balanced", profile}:
                    continue
                key = (str(row["scale"]), int(row["evaluation_seed"]))
                paired[key][name] = float(
                    row["priority_weighted_assignment_mix"][component]
                )
            values = [
                pair[profile] - pair["balanced"]
                for pair in paired.values()
                if profile in pair and "balanced" in pair
            ]
            per_seed[str(seed)] = float(np.mean(values))
        result[profile] = {
            "component": component,
            "per_seed": per_seed,
            "ci95": mean_ci(per_seed.values()),
            "positive_seeds": sum(value > 0 for value in per_seed.values()),
        }
    return result


def efficiency(
    rows: dict[str, dict[int, list[dict[str, Any]]]]
) -> dict[str, Any]:
    mapping = {
        "deadline_completion_delta": "deadline_completion_rate",
        "makespan_increase_ratio": "makespan",
        "minimum_task_coverage_delta": "minimum_task_coverage",
        "invalid_actions_delta": "invalid_actions",
    }
    result: dict[str, Any] = {}
    for output, field in mapping.items():
        per_seed: dict[str, float] = {}
        for seed in sorted(rows[CONDITIONED]):
            candidate = metric_mean(rows[CONDITIONED][seed], field)
            reference = metric_mean(rows[GPPO][seed], field)
            per_seed[str(seed)] = (
                candidate / reference - 1.0
                if output == "makespan_increase_ratio"
                else candidate - reference
            )
        result[output] = {
            "per_seed": per_seed,
            "ci95": mean_ci(per_seed.values()),
        }
    return result


def checkpoints(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for run_root in sorted(root.glob("seed_*_run")):
        for method in (CONDITIONED, NO_CONDITIONING):
            for path in (run_root / "train" / method).glob("seed_*/checkpoint.pt"):
                payload = torch.load(path, map_location="cpu", weights_only=False)
                seed = int(path.parent.name.split("_", maxsplit=1)[1])
                candidates = sorted(
                    (path.parent / "validation_candidates").glob("update_*.pt")
                )
                selection = payload.get("checkpoint_selection", {})
                result[f"{method}/seed_{seed}"] = {
                    "selection_passed": bool(selection.get("selection_passed", False)),
                    "selected_update": int(selection.get("selected_update", -1)),
                    "reasons": (selection.get("selected_diagnostics") or {}).get(
                        "reasons", []
                    ),
                    "candidate_count": len(candidates),
                    "candidate_files": [item.name for item in candidates],
                }
    return result


def markdown(report: dict[str, Any]) -> str:
    primary = report["improvements"]["primary_five"]
    held = report["improvements"]["held_out_three"]
    lines = [
        "# PCRL hard-6 calibration20 result",
        "",
        "This is three-seed screening evidence, not pilot or formal evidence.",
        "",
        f"- Primary-five improvement: {100 * primary['ci95']['mean']:.2f}% "
        f"(95% CI [{100 * primary['ci95']['lower']:.2f}%, "
        f"{100 * primary['ci95']['upper']:.2f}%]), "
        f"{primary['positive_seeds']}/3 positive seeds.",
        f"- Held-out-three improvement: {100 * held['ci95']['mean']:.2f}% "
        f"(95% CI [{100 * held['ci95']['lower']:.2f}%, "
        f"{100 * held['ci95']['upper']:.2f}%]), "
        f"{held['positive_seeds']}/3 positive seeds.",
        "",
        "## Directionality",
        "",
    ]
    for name, item in report["directions"].items():
        lines.append(
            f"- {name}: mean {item['ci95']['mean']:+.6f}, "
            f"{item['positive_seeds']}/3 positive seeds."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            "JEPA and world-model work remain locked unless every calibration gate passes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.root)
    primary = improvement(rows, PRIMARY_PROFILES)
    held = improvement(rows, HELD_OUT_PROFILES)
    direction_result = directions(rows)
    efficiency_result = efficiency(rows)
    checkpoint_result = checkpoints(args.root)
    gates = {
        "primary_mean_ge_30pct": primary["ci95"]["mean"] >= 0.30,
        "primary_positive_3_of_3": primary["positive_seeds"] == 3,
        "held_out_mean_ge_20pct": held["ci95"]["mean"] >= 0.20,
        "held_out_positive_3_of_3": held["positive_seeds"] == 3,
        "all_directions_positive_3_of_3": all(
            item["positive_seeds"] == 3 for item in direction_result.values()
        ),
        "deadline_guard": efficiency_result["deadline_completion_delta"]["ci95"][
            "mean"
        ]
        >= -0.03,
        "makespan_guard": efficiency_result["makespan_increase_ratio"]["ci95"][
            "mean"
        ]
        <= 0.05,
        "coverage_guard": efficiency_result["minimum_task_coverage_delta"]["ci95"][
            "mean"
        ]
        >= -0.05,
        "relative_invalid_guard": efficiency_result["invalid_actions_delta"]["ci95"][
            "mean"
        ]
        <= 0.05,
        "all_selections_pass": all(
            item["selection_passed"] for item in checkpoint_result.values()
        ),
        "all_candidate_sets_complete": all(
            item["candidate_count"] == 11 for item in checkpoint_result.values()
        ),
    }
    passed = all(gates.values())
    decision = (
        "All hard-6 calibration gates pass; a fresh pilot20 may be scheduled."
        if passed
        else "Hard-6 calibration does not pass every preregistered gate; pilot20 remains locked."
    )
    report = {
        "version": "pcrl-v0-hard-6-calibration20-analysis-v1",
        "improvements": {
            "primary_five": primary,
            "held_out_three": held,
            "all_profiles": improvement(
                rows, PRIMARY_PROFILES + HELD_OUT_PROFILES
            ),
        },
        "directions": direction_result,
        "efficiency_vs_gppo": efficiency_result,
        "checkpoints": checkpoint_result,
        "gates": gates,
        "passed": passed,
        "decision": decision,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "PCRL_HARD6_CALIBRATION.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "passed": passed}))


if __name__ == "__main__":
    main()

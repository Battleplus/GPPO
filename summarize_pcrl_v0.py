from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from uav_assignment.pcrl_v0 import (
    CALIBRATION_PRIORITY_PROFILE_NAMES,
    HELD_OUT_PROFILE_NAMES,
    calibration_priority_profiles,
)


TASK_PROFILE_INDEX = {
    "search": 0,
    "reconnaissance": 1,
    "strike": 2,
    "recovery": 3,
}


PAIRWISE_METRICS = (
    "preference_l1",
    "preference_l2",
    "preference_cosine",
    "deadline_completion_rate",
    "makespan",
    "communication_events",
    "invalid_actions",
    "minimum_task_coverage",
)

LOWER_IS_BETTER = {
    "preference_l1",
    "preference_l2",
    "makespan",
    "communication_events",
    "invalid_actions",
}


# Two-sided 95% Student-t critical values for the small seed counts used here.
_T_CRITICAL_95 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.57058183564,
    6: 2.44691185114,
    7: 2.36462425101,
    8: 2.30600413503,
    9: 2.26215716285,
    10: 2.22813885196,
    11: 2.20098516008,
    12: 2.17881282967,
    13: 2.16036865646,
    14: 2.14478668792,
    15: 2.13144954556,
    16: 2.11990529922,
    17: 2.10981557783,
    18: 2.10092204024,
    19: 2.09302405441,
    20: 2.08596344727,
    21: 2.07961384473,
    22: 2.0738730679,
    23: 2.06865761042,
    24: 2.06389856163,
    25: 2.05953855275,
    26: 2.05552943864,
    27: 2.05183051648,
    28: 2.0484071418,
    29: 2.04522964213,
    30: 2.0422724563,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PCRL-v0 formal results")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--pcrl-method", default="pcrl_gppo_adaptive")
    parser.add_argument(
        "--no-conditioning-method", default="pcrl_gppo_adaptive_no_conditioning"
    )
    parser.add_argument("--gppo-method", default="gppo_event")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypervolume-samples", type=int, default=100_000)
    parser.add_argument(
        "--hard4-pilot",
        action="store_true",
        help="run strict hard4 PCRL/no-conditioning pilot acceptance",
    )
    parser.add_argument(
        "--hard4-protocol-version", default="pcrl-v0-hard-4"
    )
    parser.add_argument("--hard4-priority-share", type=float, default=0.40)
    parser.add_argument(
        "--hard4-training-seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5)
    )
    return parser.parse_args()


def evaluation_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("evaluation.json")))
        else:
            files.append(path)
    return files


def load_rows(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in evaluation_files(inputs):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"evaluation file must contain a list: {path}")
        for row in payload:
            copied = dict(row)
            copied["source_file"] = str(path.resolve())
            rows.append(copied)
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def ci95(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    estimate = float(array.mean()) if array.size else float("nan")
    if array.size <= 1:
        return {"mean": estimate, "low": estimate, "high": estimate, "n": int(array.size)}
    critical = _T_CRITICAL_95.get(array.size - 1, 1.96)
    half = critical * float(array.std(ddof=1)) / math.sqrt(array.size)
    return {
        "mean": estimate,
        "low": estimate - half,
        "high": estimate + half,
        "n": int(array.size),
    }


def quality_vector(row: dict[str, Any]) -> np.ndarray:
    coverage = np.asarray(
        row.get(
            "deadline_assignment_priority_coverage_by_type",
            row.get(
                "deadline_assignment_coverage_by_type",
                row["availability_coverage_by_type"],
            ),
        ),
        dtype=np.float64,
    )
    deadline = max(float(row.get("mission_deadline", 0.0)), 1e-6)
    makespan_quality = 1.0 / (1.0 + max(float(row["makespan"]), 0.0) / deadline)
    max_decisions = max(float(row.get("max_decisions", 200.0)), 1.0)
    communication_quality = 1.0 - float(row["communication_events"]) / max_decisions
    unresolved = max(
        0.0,
        float(row.get("reallocated_tasks", 0.0))
        - float(row.get("reallocation_successes", 0.0)),
    )
    safety_quality = 1.0 - (
        float(row["invalid_actions"]) + unresolved
    ) / max_decisions
    return np.clip(
        np.concatenate(
            (coverage, (makespan_quality, communication_quality, safety_quality))
        ),
        0.0,
        1.0,
    )


def nondominated(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        if not keep[index]:
            continue
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        if np.any(dominates):
            keep[index] = False
    return points[keep]


def monte_carlo_hypervolume(
    points: np.ndarray, *, samples: int, seed: int = 20_260_802
) -> float:
    if len(points) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    probes = rng.random((samples, points.shape[1]))
    dominated = np.zeros(samples, dtype=bool)
    for point in points:
        dominated |= np.all(probes <= point, axis=1)
    return float(dominated.mean())


def grouped_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return groups


def method_seed_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = grouped_rows(rows, ("method_id", "training_seed"))
    result: list[dict[str, Any]] = []
    for (method_id, training_seed), selected in sorted(groups.items()):
        result.append(
            {
                "method_id": method_id,
                "training_seed": int(training_seed),
                "preference_l1": mean(selected, "preference_l1"),
                "preference_l2": mean(selected, "preference_l2"),
                "preference_cosine": mean(selected, "preference_cosine"),
                "deadline_completion_rate": mean(
                    selected, "deadline_completion_rate"
                ),
                "makespan": mean(selected, "makespan"),
                "communication_events": mean(selected, "communication_events"),
                "invalid_actions": mean(selected, "invalid_actions"),
                "minimum_task_coverage": mean(selected, "minimum_task_coverage"),
            }
        )
    return result


def method_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = grouped_rows(seed_rows, ("method_id",))
    result: list[dict[str, Any]] = []
    for (method_id,), selected in sorted(groups.items()):
        row: dict[str, Any] = {
            "method_id": method_id,
            "training_seeds": len({item["training_seed"] for item in selected}),
        }
        for key in (
            "preference_l1",
            "preference_l2",
            "preference_cosine",
            "deadline_completion_rate",
            "makespan",
            "communication_events",
            "invalid_actions",
            "minimum_task_coverage",
        ):
            interval = ci95([float(item[key]) for item in selected])
            row[key] = interval["mean"]
            row[f"{key}_ci_low"] = interval["low"]
            row[f"{key}_ci_high"] = interval["high"]
        result.append(row)
    return result


def pareto_metrics(
    rows: list[dict[str, Any]], *, hypervolume_samples: int
) -> dict[str, dict[str, float]]:
    point_groups = grouped_rows(
        rows, ("method_id", "training_seed", "preference_profile", "scale")
    )
    method_points: dict[str, list[np.ndarray]] = defaultdict(list)
    for (method_id, _, _, _), selected in point_groups.items():
        method_points[str(method_id)].append(
            np.mean([quality_vector(row) for row in selected], axis=0)
        )
    all_points = np.asarray(
        [point for points in method_points.values() for point in points]
    )
    reference_front = nondominated(all_points)
    result: dict[str, dict[str, float]] = {}
    for method_id, point_list in method_points.items():
        points = np.asarray(point_list)
        distances = [
            float(np.min(np.linalg.norm(points - reference, axis=1)))
            for reference in reference_front
        ]
        result[method_id] = {
            "hypervolume": monte_carlo_hypervolume(
                nondominated(points), samples=hypervolume_samples
            ),
            "igd": float(np.mean(distances)),
            "points": float(len(points)),
        }
    return result


def preference_regret(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = grouped_rows(rows, ("method_id", "preference_profile", "scale"))
    utilities: list[dict[str, Any]] = []
    for (method_id, profile, scale), selected in groups.items():
        values = []
        for row in selected:
            preference = np.asarray(row["optimization_preference"], dtype=np.float64)
            values.append(float(np.dot(preference, quality_vector(row))))
        utilities.append(
            {
                "method_id": method_id,
                "preference_profile": profile,
                "scale": scale,
                "utility": float(np.mean(values)),
            }
        )
    best: dict[tuple[str, str], float] = defaultdict(lambda: -float("inf"))
    for row in utilities:
        key = (str(row["preference_profile"]), str(row["scale"]))
        best[key] = max(best[key], float(row["utility"]))
    for row in utilities:
        key = (str(row["preference_profile"]), str(row["scale"]))
        row["observed_preference_regret"] = best[key] - float(row["utility"])
    return utilities


def lookup_seed(
    seed_rows: list[dict[str, Any]], method: str
) -> dict[int, dict[str, Any]]:
    return {
        int(row["training_seed"]): row
        for row in seed_rows
        if row["method_id"] == method
    }


def exact_sign_flip_pvalue(values: list[float], *, max_exact_n: int = 16) -> float:
    """Return a two-sided paired randomization p-value for a mean difference."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or array.size > max_exact_n:
        return float("nan")
    observed = abs(float(array.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=int(array.size)):
        permuted = float(np.mean(array * np.asarray(signs, dtype=np.float64)))
        if abs(permuted) >= observed - 1e-15:
            extreme += 1
        total += 1
    return extreme / total


def paired_seed_comparison(
    seed_rows: list[dict[str, Any]],
    *,
    first_method: str,
    second_method: str,
    metrics: Iterable[str] = PAIRWISE_METRICS,
) -> dict[str, Any]:
    """Compare methods on matched training seeds using first minus second."""
    first = lookup_seed(seed_rows, first_method)
    second = lookup_seed(seed_rows, second_method)
    common_seeds = sorted(set(first) & set(second))
    result: dict[str, Any] = {
        "first_method": first_method,
        "second_method": second_method,
        "difference_definition": "first_method - second_method",
        "paired_training_seeds": common_seeds,
        "metrics": {},
    }
    for metric in metrics:
        differences = [
            float(first[seed][metric]) - float(second[seed][metric])
            for seed in common_seeds
        ]
        interval = ci95(differences)
        array = np.asarray(differences, dtype=np.float64)
        standard_deviation = (
            float(array.std(ddof=1)) if array.size > 1 else 0.0
        )
        standard_error = (
            standard_deviation / math.sqrt(array.size) if array.size else float("nan")
        )
        mean_difference = float(interval["mean"])
        if standard_error > 0.0:
            t_statistic = mean_difference / standard_error
        elif mean_difference != 0.0:
            t_statistic = math.copysign(float("inf"), mean_difference)
        else:
            t_statistic = 0.0
        lower_is_better = metric in LOWER_IS_BETTER
        improvement = [
            difference < 0.0 if lower_is_better else difference > 0.0
            for difference in differences
        ]
        worsening = [
            difference > 0.0 if lower_is_better else difference < 0.0
            for difference in differences
        ]
        result["metrics"][metric] = {
            "direction": "lower_is_better" if lower_is_better else "higher_is_better",
            "seed_differences": [
                {"training_seed": seed, "difference": difference}
                for seed, difference in zip(common_seeds, differences)
            ],
            "mean_difference": mean_difference,
            "ci_low": float(interval["low"]),
            "ci_high": float(interval["high"]),
            "n": int(interval["n"]),
            "standard_deviation": standard_deviation,
            "standard_error": standard_error,
            "t_statistic": t_statistic,
            "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
            "ci_excludes_zero": bool(
                float(interval["high"]) < 0.0 or float(interval["low"]) > 0.0
            ),
            "improved_seed_count": int(sum(improvement)),
            "worsened_seed_count": int(sum(worsening)),
            "tied_seed_count": int(
                len(differences) - sum(improvement) - sum(worsening)
            ),
        }
    return result


def acceptance(
    rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    *,
    pcrl_method: str,
    no_conditioning_method: str,
    gppo_method: str,
) -> dict[str, Any]:
    pcrl = lookup_seed(seed_rows, pcrl_method)
    no_conditioning = lookup_seed(seed_rows, no_conditioning_method)
    gppo = lookup_seed(seed_rows, gppo_method)
    common_no_conditioning = sorted(set(pcrl) & set(no_conditioning))
    common_gppo = sorted(set(pcrl) & set(gppo))
    pcrl_l1 = float(np.mean([pcrl[seed]["preference_l1"] for seed in pcrl])) if pcrl else float("nan")
    no_conditioning_l1 = (
        float(np.mean([no_conditioning[seed]["preference_l1"] for seed in no_conditioning]))
        if no_conditioning
        else float("nan")
    )
    error_reduction = (
        1.0 - pcrl_l1 / no_conditioning_l1
        if no_conditioning_l1 > 0
        else float("nan")
    )

    selected_pcrl = [row for row in rows if row["method_id"] == pcrl_method]
    profile_groups = grouped_rows(selected_pcrl, ("preference_profile",))
    balanced = profile_groups.get(("balanced",), [])
    monotonicity: dict[str, Any] = {}
    for profile, index in TASK_PROFILE_INDEX.items():
        priority_rows = profile_groups.get((profile,), [])
        priority_coverage = (
            float(
                np.mean(
                    [row["deadline_assignment_priority_coverage_by_type"][index] for row in priority_rows]
                )
            )
            if priority_rows
            else float("nan")
        )
        balanced_coverage = (
            float(
                np.mean([row["deadline_assignment_priority_coverage_by_type"][index] for row in balanced])
            )
            if balanced
            else float("nan")
        )
        monotonicity[profile] = {
            "priority_coverage": priority_coverage,
            "balanced_coverage": balanced_coverage,
            "delta": priority_coverage - balanced_coverage,
            "passed": bool(priority_coverage > balanced_coverage),
        }
    held_out = [
        row
        for row in selected_pcrl
        if row["preference_profile"] in HELD_OUT_PROFILE_NAMES
    ]
    held_out_l1 = mean(held_out, "preference_l1") if held_out else float("nan")

    stable_seeds: list[int] = []
    for seed in sorted(set(common_no_conditioning) & set(common_gppo)):
        preference_pass = (
            pcrl[seed]["preference_l1"]
            <= 0.70 * no_conditioning[seed]["preference_l1"]
        )
        dcr_pass = (
            pcrl[seed]["deadline_completion_rate"]
            >= gppo[seed]["deadline_completion_rate"] - 0.03
        )
        makespan_pass = pcrl[seed]["makespan"] <= 1.05 * gppo[seed]["makespan"]
        coverage_pass = (
            pcrl[seed]["minimum_task_coverage"]
            >= gppo[seed]["minimum_task_coverage"] - 0.05
        )
        if preference_pass and dcr_pass and makespan_pass and coverage_pass:
            stable_seeds.append(seed)

    gppo_dcr = float(np.mean([row["deadline_completion_rate"] for row in gppo.values()])) if gppo else float("nan")
    gppo_makespan = float(np.mean([row["makespan"] for row in gppo.values()])) if gppo else float("nan")
    pcrl_dcr = float(np.mean([row["deadline_completion_rate"] for row in pcrl.values()])) if pcrl else float("nan")
    pcrl_makespan = float(np.mean([row["makespan"] for row in pcrl.values()])) if pcrl else float("nan")
    gates = {
        "preference_error_reduction": {
            "value": error_reduction,
            "threshold": 0.30,
            "passed": bool(error_reduction >= 0.30),
        },
        "monotonicity": {
            "details": monotonicity,
            "passed": bool(monotonicity and all(item["passed"] for item in monotonicity.values())),
        },
        "held_out_interpolation": {
            "mean_l1": held_out_l1,
            "passed": bool(np.isfinite(held_out_l1) and held_out_l1 < no_conditioning_l1),
        },
        "deadline_completion": {
            "pcrl": pcrl_dcr,
            "gppo": gppo_dcr,
            "drop": gppo_dcr - pcrl_dcr,
            "threshold": 0.03,
            "passed": bool(pcrl_dcr >= gppo_dcr - 0.03),
        },
        "makespan": {
            "pcrl": pcrl_makespan,
            "gppo": gppo_makespan,
            "increase_ratio": pcrl_makespan / gppo_makespan - 1.0 if gppo_makespan > 0 else float("nan"),
            "threshold": 0.05,
            "passed": bool(pcrl_makespan <= 1.05 * gppo_makespan),
        },
        "stable_seeds": {
            "passed_seeds": stable_seeds,
            "count": len(stable_seeds),
            "threshold": 4,
            "passed": len(stable_seeds) >= 4,
        },
    }
    return {
        "methods": {
            "pcrl": pcrl_method,
            "no_conditioning": no_conditioning_method,
            "gppo": gppo_method,
        },
        "gates": gates,
        "passed": all(item["passed"] for item in gates.values()),
        "negative_or_inconclusive": [
            name for name, item in gates.items() if not item["passed"]
        ],
    }


HARD4_PRIMARY_PROFILES = CALIBRATION_PRIORITY_PROFILE_NAMES[:5]
HARD4_HELD_OUT_PROFILES = CALIBRATION_PRIORITY_PROFILE_NAMES[5:]
HARD4_DIRECTION_INDEX = {
    "calibration_search_priority": 0,
    "calibration_reconnaissance_priority": 1,
    "calibration_strike_priority": 2,
    "calibration_recovery_priority": 3,
}
HARD4_CELL_FIELDS = (
    "training_seed",
    "preference_profile",
    "scale",
    "eval_seed",
)
HARD4_SCALES = ("2x12", "3x16", "3x20", "4x24")


def _hard4_float(row: dict[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"hard4 row lacks valid {field!r}") from error
    if not np.isfinite(value):
        raise ValueError(f"hard4 field {field!r} must be finite")
    return value


def _hard4_float_matches(
    row: dict[str, Any], field: str, expected: float
) -> bool:
    return bool(
        np.isclose(
            _hard4_float(row, field),
            expected,
            rtol=0.0,
            atol=1e-12,
        )
    )


def _hard4_sha256(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field, ""))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"hard4 row lacks valid SHA-256 field {field!r}")
    return value.lower()


def _hard4_frozen_gppo_identity(
    method: str, expected_seeds: tuple[int, ...]
) -> tuple[str, dict[int, str]]:
    registry_path = WORKSPACE / "configs" / "pcrl_v0_frozen_checkpoints.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    implementation = _hard4_sha256(
        {"value": registry.get("frozen_gppo_implementation_hash")}, "value"
    )
    checkpoint_sha256: dict[int, str] = {}
    for entry in registry.get("checkpoints", []):
        if str(entry.get("method", "")) != method:
            continue
        seed = int(entry.get("seed", -1))
        digest = _hard4_sha256(
            {"value": entry.get("sha256")}, "value"
        )
        if seed in checkpoint_sha256:
            raise ValueError(
                f"frozen GPPO registry has duplicate {method!r} seed {seed}"
            )
        checkpoint_sha256[seed] = digest
    expected = set(expected_seeds)
    if set(checkpoint_sha256) != expected:
        raise ValueError(
            f"frozen GPPO registry seed set {tuple(sorted(checkpoint_sha256))!r} "
            f"does not match {expected_seeds!r} for {method!r}"
        )
    return implementation, checkpoint_sha256


def _hard4_index(
    rows: list[dict[str, Any]],
    *,
    method: str,
    expected_seeds: tuple[int, ...],
    expected_profiles: tuple[str, ...],
    expected_vectors: dict[str, np.ndarray],
    protocol_version: str,
    priority_share: float,
    conditioning: bool,
    expected_algorithm: str,
    expected_graph_mode: str,
    expected_updates: int,
    frozen_gppo_implementation_hash: str,
    source_checkpoint_sha256: dict[int, str],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    selected = [row for row in rows if str(row.get("method_id")) == method]
    if not selected:
        raise ValueError(f"hard4 input has no rows for method {method!r}")
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in selected:
        if str(row.get("protocol_version", "")) != protocol_version:
            raise ValueError(f"{method} contains a mismatched protocol_version")
        if str(row.get("artifact_group", "")) != "pilot20":
            raise ValueError(
                f"{method} contains a non-pilot or missing artifact_group"
            )
        if str(row.get("evaluation_suite", "")) != "controllability_phase":
            raise ValueError(f"{method} contains a non-primary evaluation suite")
        if str(row.get("preference_profile_family", "")) != (
            "dynamic-priority-share-v1"
        ):
            raise ValueError(f"{method} contains a mismatched profile family")
        if str(row.get("scenario_version", "")) != "gppo-v2-hard-3":
            raise ValueError(f"{method} contains a mismatched scenario_version")
        for sha_field in (
            "protocol_config_sha256",
            "scenario_config_sha256",
            "scenario_hash",
            "frozen_gppo_implementation_hash",
            "source_gppo_checkpoint_sha256",
            "pcrl_implementation_hash",
        ):
            _hard4_sha256(row, sha_field)
        if str(row.get("output_namespace", "")) != "outputs/pcrl_v0/hard4":
            raise ValueError(f"{method} contains a mismatched output_namespace")
        if str(row.get("base_protocol", "")) != "gppo-v2-hard-3":
            raise ValueError(f"{method} contains a mismatched base_protocol")
        if int(row.get("evaluation_seed", -1)) != int(row.get("eval_seed", -2)):
            raise ValueError(f"{method} evaluation_seed alias is inconsistent")
        if str(row.get("algorithm", "")) != expected_algorithm:
            raise ValueError(f"{method} contains a mismatched algorithm identity")
        if str(row.get("graph_mode", "")) != expected_graph_mode:
            raise ValueError(f"{method} contains a mismatched graph_mode identity")
        if int(row.get("updates", -1)) != expected_updates:
            raise ValueError(
                f"{method} contains a mismatched training update budget"
            )
        if int(row.get("episodes_per_update", -1)) != 18:
            raise ValueError(
                f"{method} contains a mismatched episodes-per-update budget"
            )
        if int(row.get("evaluation_episodes", -1)) != 20:
            raise ValueError(f"{method} contains a mismatched evaluation budget")
        if str(row.get("task_release_mode", "")) != "phase_staggered":
            raise ValueError(f"{method} contains a non-primary task_release_mode")
        if str(row.get("preference_target_mode", "")) != "nominal":
            raise ValueError(f"{method} contains a mismatched preference target mode")
        if not _hard4_float_matches(row, "deadline_scale", 0.70):
            raise ValueError(f"{method} contains a mismatched deadline scale")
        if not _hard4_float_matches(row, "assignment_priority_decay", 2.0):
            raise ValueError(f"{method} contains a mismatched assignment decay")
        if not _hard4_float_matches(row, "priority_share", priority_share):
            raise ValueError(f"{method} contains a mismatched calibration share")
        if not _hard4_float_matches(row, "background_share", 0.20):
            raise ValueError(f"{method} contains a mismatched background share")
        profile = str(row.get("preference_profile", ""))
        if profile not in expected_profiles:
            raise ValueError(f"{method} contains an out-of-family profile {profile!r}")
        vector = np.asarray(row.get("preference_profile_vector"), dtype=np.float64)
        if vector.shape != (4,) or not np.array_equal(
            vector, expected_vectors[profile].astype(np.float64)
        ):
            raise ValueError(
                f"{method} profile vector mismatch for {profile!r}"
            )
        seed = int(row.get("training_seed", -1))
        if seed not in expected_seeds:
            raise ValueError(f"{method} contains unexpected training seed {seed}")
        if not isinstance(row.get("preference_conditioning"), bool):
            raise ValueError(
                f"{method} requires boolean preference_conditioning metadata"
            )
        if row["preference_conditioning"] is not conditioning:
            raise ValueError(
                f"{method} preference_conditioning metadata is incorrect"
            )
        if (
            _hard4_sha256(row, "frozen_gppo_implementation_hash")
            != frozen_gppo_implementation_hash
        ):
            raise ValueError(
                f"{method} frozen GPPO implementation hash does not match registry"
            )
        if (
            _hard4_sha256(row, "source_gppo_checkpoint_sha256")
            != source_checkpoint_sha256[seed]
        ):
            raise ValueError(
                f"{method} source GPPO checkpoint hash does not match registry "
                f"for training seed {seed}"
            )
        key = tuple(row.get(field) for field in HARD4_CELL_FIELDS)
        if key in index:
            raise ValueError(f"{method} has duplicate evaluation cell {key!r}")
        for metric in (
            "preference_l1",
            "deadline_completion_rate",
            "makespan",
            "minimum_task_coverage",
        ):
            _hard4_float(row, metric)
        mix = np.asarray(row.get("priority_weighted_assignment_mix"), dtype=np.float64)
        if mix.shape != (4,) or not np.all(np.isfinite(mix)):
            raise ValueError(f"{method} has invalid priority assignment mix")
        index[key] = row

    seeds = tuple(sorted({int(key[0]) for key in index}))
    if seeds != expected_seeds:
        raise ValueError(
            f"{method} training seed set {seeds!r} does not match {expected_seeds!r}"
        )
    scales = {str(key[2]) for key in index}
    if scales != set(HARD4_SCALES):
        raise ValueError(
            f"{method} scale set {tuple(sorted(scales))!r} does not match "
            f"{HARD4_SCALES!r}"
        )
    reference_grid: set[tuple[Any, ...]] | None = None
    for seed in expected_seeds:
        grid = {key[1:] for key in index if int(key[0]) == seed}
        if {str(key[0]) for key in grid} != set(expected_profiles):
            raise ValueError(f"{method} seed {seed} lacks the complete profile family")
        if reference_grid is None:
            reference_grid = grid
        elif grid != reference_grid:
            raise ValueError(
                f"{method} does not have an identical eval cell grid for seed {seed}"
            )
    for seed in expected_seeds:
        seed_rows = [row for key, row in index.items() if int(key[0]) == seed]
        for profile in expected_profiles:
            for scale in {str(row["scale"]) for row in seed_rows}:
                eval_seeds = {
                    int(row["eval_seed"])
                    for row in seed_rows
                    if row["preference_profile"] == profile
                    and str(row["scale"]) == scale
                }
                if eval_seeds != set(range(76000, 76020)):
                    raise ValueError(
                        f"{method} seed {seed} has a non-pilot eval-seed grid"
                    )
    return index


def _hard4_metric_macro(
    index: dict[tuple[Any, ...], dict[str, Any]],
    seed: int,
    profiles: tuple[str, ...],
    metric: str,
) -> float:
    selected = [
        row
        for key, row in index.items()
        if int(key[0]) == seed and str(key[1]) in profiles
    ]
    return float(np.mean([_hard4_float(row, metric) for row in selected]))


def _hard4_relative_effect(
    first: list[float], second: list[float]
) -> list[float]:
    if any(value <= 0 for value in second):
        raise ValueError("hard4 no-conditioning L1 denominator must be positive")
    return [1.0 - left / right for left, right in zip(first, second, strict=True)]


def hard4_pilot_acceptance(
    rows: list[dict[str, Any]],
    *,
    pcrl_method: str = "pcrl_gppo_adaptive",
    no_conditioning_method: str = "pcrl_gppo_adaptive_no_conditioning",
    gppo_method: str = "gppo_event",
    protocol_version: str = "pcrl-v0-hard-4",
    priority_share: float = 0.40,
    training_seeds: Iterable[int] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    """Strict hard4 pilot analysis over paired training seeds and eval cells."""

    expected_seeds = tuple(sorted(int(seed) for seed in training_seeds))
    if len(expected_seeds) != 5 or len(set(expected_seeds)) != 5:
        raise ValueError("hard4 pilot requires exactly five distinct training seeds")
    if not no_conditioning_method.endswith("_no_conditioning"):
        raise ValueError(
            "hard4 no-conditioning denominator method_id must end with "
            "'_no_conditioning'"
        )
    if pcrl_method.endswith("_no_conditioning"):
        raise ValueError("hard4 PCRL numerator cannot be a no-conditioning method")
    family = calibration_priority_profiles(priority_share)
    expected_profiles = tuple(family)
    frozen_implementation_hash, source_checkpoint_sha256 = (
        _hard4_frozen_gppo_identity(gppo_method, expected_seeds)
    )
    pcrl = _hard4_index(
        rows,
        method=pcrl_method,
        expected_seeds=expected_seeds,
        expected_profiles=expected_profiles,
        expected_vectors=family,
        protocol_version=protocol_version,
        priority_share=priority_share,
        conditioning=True,
        expected_algorithm="preco",
        expected_graph_mode="adaptive",
        expected_updates=20,
        frozen_gppo_implementation_hash=frozen_implementation_hash,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    no_conditioning = _hard4_index(
        rows,
        method=no_conditioning_method,
        expected_seeds=expected_seeds,
        expected_profiles=expected_profiles,
        expected_vectors=family,
        protocol_version=protocol_version,
        priority_share=priority_share,
        conditioning=False,
        expected_algorithm="preco",
        expected_graph_mode="adaptive",
        expected_updates=20,
        frozen_gppo_implementation_hash=frozen_implementation_hash,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    gppo = _hard4_index(
        rows,
        method=gppo_method,
        expected_seeds=expected_seeds,
        expected_profiles=expected_profiles,
        expected_vectors=family,
        protocol_version=protocol_version,
        priority_share=priority_share,
        conditioning=False,
        expected_algorithm="gppo",
        expected_graph_mode="adaptive",
        expected_updates=100,
        frozen_gppo_implementation_hash=frozen_implementation_hash,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )
    for identity_field in (
        "protocol_config_sha256",
        "scenario_config_sha256",
        "frozen_gppo_implementation_hash",
        "pcrl_implementation_hash",
        "output_namespace",
        "base_protocol",
    ):
        values = {
            str(row[identity_field])
            for index in (pcrl, no_conditioning, gppo)
            for row in index.values()
        }
        if len(values) != 1:
            raise ValueError(
                f"hard4 methods disagree on {identity_field} identity"
            )
    if set(pcrl) != set(no_conditioning):
        raise ValueError(
            "PCRL and no-conditioning must have identical training-seed/eval cells"
        )
    if set(pcrl) != set(gppo):
        raise ValueError(
            "PCRL and GPPO efficiency comparator must have identical eval cells"
        )
    for key in pcrl:
        scenario_hashes = {
            _hard4_sha256(index[key], "scenario_hash")
            for index in (pcrl, no_conditioning, gppo)
        }
        if len(scenario_hashes) != 1:
            raise ValueError(
                f"hard4 methods disagree on scenario_hash for evaluation cell {key!r}"
            )

    primary_pcrl = [
        _hard4_metric_macro(pcrl, seed, HARD4_PRIMARY_PROFILES, "preference_l1")
        for seed in expected_seeds
    ]
    primary_no_conditioning = [
        _hard4_metric_macro(
            no_conditioning, seed, HARD4_PRIMARY_PROFILES, "preference_l1"
        )
        for seed in expected_seeds
    ]
    primary_effects = _hard4_relative_effect(
        primary_pcrl, primary_no_conditioning
    )
    primary_interval = ci95(primary_effects)
    threshold_effects = [effect - 0.30 for effect in primary_effects]

    held_pcrl = [
        _hard4_metric_macro(pcrl, seed, HARD4_HELD_OUT_PROFILES, "preference_l1")
        for seed in expected_seeds
    ]
    held_no_conditioning = [
        _hard4_metric_macro(
            no_conditioning, seed, HARD4_HELD_OUT_PROFILES, "preference_l1"
        )
        for seed in expected_seeds
    ]
    held_effects = _hard4_relative_effect(held_pcrl, held_no_conditioning)
    held_interval = ci95(held_effects)

    monotonicity: dict[str, Any] = {}
    for profile, component in HARD4_DIRECTION_INDEX.items():
        seed_deltas: list[float] = []
        for seed in expected_seeds:
            balanced_by_cell = {
                (key[2], key[3]): np.asarray(
                    row["priority_weighted_assignment_mix"], dtype=np.float64
                )
                for key, row in pcrl.items()
                if int(key[0]) == seed and key[1] == "balanced"
            }
            priority_by_cell = {
                (key[2], key[3]): np.asarray(
                    row["priority_weighted_assignment_mix"], dtype=np.float64
                )
                for key, row in pcrl.items()
                if int(key[0]) == seed and key[1] == profile
            }
            if set(balanced_by_cell) != set(priority_by_cell):
                raise ValueError(f"monotonic cells do not pair for {profile!r}")
            seed_deltas.append(
                float(
                    np.mean(
                        [
                            priority_by_cell[cell][component]
                            - balanced_by_cell[cell][component]
                            for cell in sorted(balanced_by_cell)
                        ]
                    )
                )
            )
        interval = ci95(seed_deltas)
        improved = sum(delta > 0 for delta in seed_deltas)
        monotonicity[profile] = {
            "component": component,
            "seed_deltas": seed_deltas,
            "mean_delta": interval["mean"],
            "ci_low": interval["low"],
            "ci_high": interval["high"],
            "improved_seeds": improved,
            "passed": bool(float(interval["low"]) > 0 and improved >= 4),
        }

    efficiency_seed_rows: list[dict[str, float | int]] = []
    for seed in expected_seeds:
        pcrl_dcr = _hard4_metric_macro(
            pcrl, seed, expected_profiles, "deadline_completion_rate"
        )
        gppo_dcr = _hard4_metric_macro(
            gppo, seed, expected_profiles, "deadline_completion_rate"
        )
        pcrl_makespan = _hard4_metric_macro(
            pcrl, seed, expected_profiles, "makespan"
        )
        gppo_makespan = _hard4_metric_macro(
            gppo, seed, expected_profiles, "makespan"
        )
        pcrl_coverage = _hard4_metric_macro(
            pcrl, seed, expected_profiles, "minimum_task_coverage"
        )
        gppo_coverage = _hard4_metric_macro(
            gppo, seed, expected_profiles, "minimum_task_coverage"
        )
        if gppo_makespan <= 0:
            raise ValueError("GPPO makespan denominator must be positive")
        efficiency_seed_rows.append(
            {
                "training_seed": seed,
                "deadline_completion_delta": pcrl_dcr - gppo_dcr,
                "makespan_increase_ratio": pcrl_makespan / gppo_makespan - 1.0,
                "minimum_coverage_delta": pcrl_coverage - gppo_coverage,
            }
        )
    dcr_ci = ci95(
        [float(row["deadline_completion_delta"]) for row in efficiency_seed_rows]
    )
    makespan_ci = ci95(
        [float(row["makespan_increase_ratio"]) for row in efficiency_seed_rows]
    )
    coverage_ci = ci95(
        [float(row["minimum_coverage_delta"]) for row in efficiency_seed_rows]
    )
    efficiency = {
        "comparison_role": "GPPO is used only for efficiency/coverage guards",
        "seed_metrics": efficiency_seed_rows,
        "deadline_completion": {
            "ci": dcr_ci,
            "threshold": -0.03,
            "passed": bool(float(dcr_ci["low"]) >= -0.03),
        },
        "makespan": {
            "ci": makespan_ci,
            "threshold": 0.05,
            "passed": bool(float(makespan_ci["high"]) <= 0.05),
        },
        "minimum_task_coverage": {
            "ci": coverage_ci,
            "threshold": -0.05,
            "passed": bool(float(coverage_ci["low"]) >= -0.05),
        },
    }
    preference_gate = {
        "denominator_method": no_conditioning_method,
        "profiles": list(HARD4_PRIMARY_PROFILES),
        "per_training_seed": [
            {
                "training_seed": seed,
                "pcrl_macro_l1": left,
                "no_conditioning_macro_l1": right,
                "relative_improvement": effect,
            }
            for seed, left, right, effect in zip(
                expected_seeds,
                primary_pcrl,
                primary_no_conditioning,
                primary_effects,
                strict=True,
            )
        ],
        "relative_improvement_ci": primary_interval,
        "threshold": 0.30,
        "threshold_exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(
            threshold_effects
        ),
        "zero_effect_exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(
            primary_effects
        ),
        "improved_seeds": sum(effect > 0 for effect in primary_effects),
        "formal_30_percent_point_target_met": bool(
            float(primary_interval["mean"]) >= 0.30
        ),
        "formal_30_percent_ci_target_met": bool(
            float(primary_interval["low"]) >= 0.30
        ),
        "passed": bool(
            float(primary_interval["mean"]) >= 0.30
            and float(primary_interval["low"]) > 0
            and sum(effect > 0 for effect in primary_effects) >= 4
        ),
    }
    held_out = {
        "profiles": list(HARD4_HELD_OUT_PROFILES),
        "denominator_method": no_conditioning_method,
        "relative_improvement_ci": held_interval,
        "improved_seeds": sum(effect > 0 for effect in held_effects),
        "passed": bool(
            float(held_interval["low"]) > 0
            and sum(effect > 0 for effect in held_effects) >= 4
        ),
    }
    gates = {
        "preference_positive_pilot": preference_gate["passed"],
        "monotonicity": all(item["passed"] for item in monotonicity.values()),
        "held_out_same_family": held_out["passed"],
        "gppo_deadline_completion": efficiency["deadline_completion"]["passed"],
        "gppo_makespan": efficiency["makespan"]["passed"],
        "gppo_minimum_coverage": efficiency["minimum_task_coverage"]["passed"],
    }
    return {
        "protocol_version": protocol_version,
        "priority_share": priority_share,
        "paired_training_seeds": list(expected_seeds),
        "exact_inference_limitation": (
            "With five paired training seeds, the minimum attainable two-sided "
            "exact sign-flip p-value is 0.0625; exact p<0.05 claims are impossible."
        ),
        "preference_control": preference_gate,
        "monotonicity": monotonicity,
        "held_out_same_family": held_out,
        "gppo_efficiency_guards": efficiency,
        "gppo_excluded_from_preference_denominator": True,
        "conditioning_verification": (
            "Exact checkpoint-derived method_id is required; optional "
            "preference_conditioning row metadata is cross-checked when present."
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    pareto: dict[str, dict[str, float]],
    acceptance_result: dict[str, Any],
    paired_comparison: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# PCRL-v0 results",
        "",
        f"Formal acceptance: **{'PASS' if acceptance_result['passed'] else 'NOT PASSED'}**.",
        "",
        "| method | preference L1 | deadline completion | makespan | HV | IGD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        method = str(row["method_id"])
        pareto_row = pareto.get(method, {"hypervolume": float("nan"), "igd": float("nan")})
        lines.append(
            f"| {method} | {row['preference_l1']:.4f} | "
            f"{row['deadline_completion_rate']:.4f} | {row['makespan']:.4f} | "
            f"{pareto_row['hypervolume']:.4f} | {pareto_row['igd']:.4f} |"
        )
    lines.extend(["", "## Acceptance gates", ""])
    for name, gate in acceptance_result["gates"].items():
        lines.append(f"- {name}: {'PASS' if gate['passed'] else 'FAIL/INCONCLUSIVE'}")
    if acceptance_result["negative_or_inconclusive"]:
        lines.extend(
            [
                "",
                "Negative or inconclusive outcomes are retained: "
                + ", ".join(acceptance_result["negative_or_inconclusive"])
                + ".",
            ]
        )
    if paired_comparison is not None:
        lines.extend(
            [
                "",
                "## Paired PCRL minus no-conditioning statistics",
                "",
                "Differences are paired by training seed and defined as PCRL minus "
                "no-conditioning. Confidence intervals use Student's t distribution; "
                "p-values use an exact two-sided sign-flip test.",
                "",
                "| metric | mean difference | 95% CI | improved seeds | exact p |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric, comparison in paired_comparison["metrics"].items():
            lines.append(
                f"| {metric} | {comparison['mean_difference']:.6f} | "
                f"[{comparison['ci_low']:.6f}, {comparison['ci_high']:.6f}] | "
                f"{comparison['improved_seed_count']}/{comparison['n']} | "
                f"{comparison['exact_two_sided_sign_flip_p']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hard4_pilot_markdown(path: Path, result: dict[str, Any]) -> None:
    preference = result["preference_control"]
    held_out = result["held_out_same_family"]
    efficiency = result["gppo_efficiency_guards"]
    lines = [
        "# PCRL hard4 pilot acceptance",
        "",
        f"Pilot acceptance: **{'PASS' if result['passed'] else 'NOT PASSED'}**.",
        "",
        "> The 30% preference denominator is the paired, newly trained true "
        "no-conditioning method. Frozen GPPO is excluded from that denominator "
        "and used only for efficiency and coverage guards.",
        "",
        f"- primary relative L1 improvement: "
        f"`{preference['relative_improvement_ci']['mean']:.2%}` "
        f"(95% CI `[{preference['relative_improvement_ci']['low']:.2%}, "
        f"{preference['relative_improvement_ci']['high']:.2%}]`)",
        f"- held-out same-family improvement: "
        f"`{held_out['relative_improvement_ci']['mean']:.2%}` "
        f"({held_out['improved_seeds']}/5 seeds improved)",
        f"- exact-test limitation: {result['exact_inference_limitation']}",
        "",
        "| gate | estimate/CI | pass |",
        "| --- | ---: | :---: |",
        f"| deadline completion delta vs GPPO | "
        f"[{efficiency['deadline_completion']['ci']['low']:.4f}, "
        f"{efficiency['deadline_completion']['ci']['high']:.4f}] | "
        f"{efficiency['deadline_completion']['passed']} |",
        f"| makespan increase vs GPPO | "
        f"[{efficiency['makespan']['ci']['low']:.2%}, "
        f"{efficiency['makespan']['ci']['high']:.2%}] | "
        f"{efficiency['makespan']['passed']} |",
        f"| minimum coverage delta vs GPPO | "
        f"[{efficiency['minimum_task_coverage']['ci']['low']:.4f}, "
        f"{efficiency['minimum_task_coverage']['ci']['high']:.4f}] | "
        f"{efficiency['minimum_task_coverage']['passed']} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.inputs)
    if not rows:
        raise SystemExit("no evaluation rows found")
    seed_rows = method_seed_metrics(rows)
    summary_rows = method_summary(seed_rows)
    pareto = pareto_metrics(rows, hypervolume_samples=args.hypervolume_samples)
    regret = preference_regret(rows)
    acceptance_result = acceptance(
        rows,
        seed_rows,
        pcrl_method=args.pcrl_method,
        no_conditioning_method=args.no_conditioning_method,
        gppo_method=args.gppo_method,
    )
    paired_comparison = paired_seed_comparison(
        seed_rows,
        first_method=args.pcrl_method,
        second_method=args.no_conditioning_method,
    )
    hard4_pilot = (
        hard4_pilot_acceptance(
            rows,
            pcrl_method=args.pcrl_method,
            no_conditioning_method=args.no_conditioning_method,
            gppo_method=args.gppo_method,
            protocol_version=args.hard4_protocol_version,
            priority_share=args.hard4_priority_share,
            training_seeds=args.hard4_training_seeds,
        )
        if args.hard4_pilot
        else None
    )
    result = {
        "evaluation_files": [str(path.resolve()) for path in evaluation_files(args.inputs)],
        "row_count": len(rows),
        "method_seed_metrics": seed_rows,
        "method_summary": summary_rows,
        "pareto": pareto,
        "preference_regret": regret,
        "paired_pcrl_minus_no_conditioning": paired_comparison,
        "acceptance": acceptance_result,
        "hard4_pilot_acceptance": hard4_pilot,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "method_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_markdown(
        args.output / "PCRL_V0_RESULTS.md",
        summary_rows,
        pareto,
        acceptance_result,
        paired_comparison,
    )
    if hard4_pilot is not None:
        write_hard4_pilot_markdown(
            args.output / "PCRL_HARD4_PILOT.md", hard4_pilot
        )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "methods": len(summary_rows),
                "accepted": (
                    hard4_pilot["passed"]
                    if hard4_pilot is not None
                    else acceptance_result["passed"]
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

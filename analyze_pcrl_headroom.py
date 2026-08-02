from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_VERSION = "pcrl-headroom-v1"
PAIR_FIELDS = (
    "preference_profile",
    "scale",
    "eval_seed",
    "assignment_priority_decay",
    "preference_target_mode",
)
METRIC_FIELDS = (
    "preference_l1",
    "deadline_completion_rate",
    "makespan",
    "minimum_task_coverage",
)


def _nonempty(row: Mapping[str, Any], field: str) -> bool:
    return field in row and row[field] is not None and str(row[field]).strip() != ""


def _common_identity_field(
    oracle_rows: Sequence[Mapping[str, Any]],
    gppo_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    *,
    label: str,
) -> str:
    for field in candidates:
        present = [_nonempty(row, field) for row in (*oracle_rows, *gppo_rows)]
        if all(present):
            return field
        if any(present):
            raise ValueError(
                f"{label} field {field!r} is only partially recorded; refusing "
                "to fall back to a weaker identity"
            )
    raise ValueError(
        f"both inputs must provide a common {label} field from {tuple(candidates)!r}"
    )


def _suite_fields(
    oracle_rows: Sequence[Mapping[str, Any]],
    gppo_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    for explicit in ("evaluation_suite", "suite"):
        present = [_nonempty(row, explicit) for row in (*oracle_rows, *gppo_rows)]
        if all(present):
            return (explicit,)
        if any(present):
            raise ValueError(
                f"suite field {explicit!r} is only partially recorded; refusing "
                "to infer a pooled suite"
            )
    fallback = ("task_release_mode", "deadline_scale")
    if all(all(_nonempty(row, field) for field in fallback) for row in (*oracle_rows, *gppo_rows)):
        return fallback
    raise ValueError(
        "both inputs must provide a common explicit suite field or the "
        "task_release_mode/deadline_scale suite signature"
    )


def _canonical(value: Any, *, field: str) -> str | int | float:
    if field == "eval_seed":
        if isinstance(value, bool):
            raise ValueError("eval_seed must be an integer")
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("eval_seed must be a finite integer")
        return int(numeric)
    if field in {"assignment_priority_decay", "deadline_scale"}:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field} must be finite")
        return numeric
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _metric(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row has invalid or missing metric {field!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"metric {field!r} must be finite")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty JSON row list")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path} contains a non-object row")
    return payload


def _validate_method(
    rows: Sequence[Mapping[str, Any]], expected_method: str, *, label: str
) -> None:
    methods = {
        str(row.get("method_id", "")).strip()
        for row in rows
        if str(row.get("method_id", "")).strip()
    }
    if methods != {expected_method}:
        raise ValueError(
            f"{label} input must contain only method_id={expected_method!r}; "
            f"found {sorted(methods)!r}"
        )


def _oracle_variant(
    rows: Sequence[Mapping[str, Any]], expected: str | None
) -> tuple[str, str]:
    """Resolve one Oracle mask/wait variant and reject pooled inputs."""

    explicit_present = [_nonempty(row, "oracle_variant") for row in rows]
    if all(explicit_present):
        field = "oracle_variant"
        values = {
            str(row["oracle_variant"]).strip().lower().replace("-", "_")
            for row in rows
        }
    elif any(explicit_present):
        raise ValueError(
            "oracle_variant is only partially recorded; Oracle variants cannot "
            "be pooled or inferred"
        )
    elif all("allow_strategic_wait" in row for row in rows):
        field = "allow_strategic_wait"
        raw_values = [row["allow_strategic_wait"] for row in rows]
        if not all(isinstance(value, (bool, np.bool_)) for value in raw_values):
            raise ValueError("allow_strategic_wait must be recorded as a boolean")
        values = {
            "relaxed_wait" if bool(value) else "frozen_mask"
            for value in raw_values
        }
    else:
        raise ValueError(
            "Oracle rows must all record oracle_variant or allow_strategic_wait; "
            "frozen-mask and relaxed-wait results cannot be pooled"
        )
    if len(values) != 1:
        raise ValueError(f"Oracle input pools multiple variants: {sorted(values)!r}")
    value = next(iter(values))
    if value not in {"frozen_mask", "relaxed_wait"}:
        raise ValueError(f"unsupported Oracle variant: {value!r}")
    if expected is not None:
        expected_value = expected.strip().lower().replace("-", "_")
        if value != expected_value:
            raise ValueError(
                f"Oracle variant mismatch: expected {expected_value!r}, found {value!r}"
            )
    return field, value


def _select_profiles(
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[str],
    *,
    label: str,
) -> list[Mapping[str, Any]]:
    selected_names = tuple(str(profile).strip() for profile in profiles)
    if not selected_names or any(not profile for profile in selected_names):
        raise ValueError("an explicit non-empty profile subset is required")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("profile subset contains duplicates")
    available = {
        str(row.get("preference_profile", "")).strip() for row in rows
    }
    missing = sorted(set(selected_names) - available)
    if missing:
        raise ValueError(f"{label} input lacks requested profiles: {missing!r}")
    return [
        row
        for row in rows
        if str(row.get("preference_profile", "")).strip() in selected_names
    ]


def _calibration_identity(
    oracle_rows: Sequence[Mapping[str, Any]],
    gppo_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate optional dynamic calibration identity across both inputs."""

    fields = ("calibration_priority_share", "calibration_profile_vector")
    combined = (*oracle_rows, *gppo_rows)
    if not any(field in row for row in combined for field in fields):
        return {"mode": "static_named_profiles"}
    for field in fields:
        incomplete = [index for index, row in enumerate(combined) if not _nonempty(row, field)]
        if incomplete:
            raise ValueError(
                "dynamic calibration identity must be completely recorded by "
                f"both inputs; missing {field!r} in {len(incomplete)} rows"
            )
    shares = [float(row["calibration_priority_share"]) for row in combined]
    if any(not math.isfinite(share) for share in shares):
        raise ValueError("calibration_priority_share must be finite")
    reference_share = shares[0]
    if any(share != reference_share for share in shares[1:]):
        raise ValueError(
            "calibration_priority_share mismatch across Oracle and GPPO inputs"
        )

    vectors_by_input: list[dict[str, tuple[float, ...]]] = []
    for label, rows in (("Oracle", oracle_rows), ("GPPO", gppo_rows)):
        vectors: dict[str, tuple[float, ...]] = {}
        for row in rows:
            profile = str(row["preference_profile"]).strip()
            try:
                vector_array = np.asarray(
                    row["calibration_profile_vector"], dtype=np.float64
                ).reshape(-1)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{label} calibration vector for {profile!r} is invalid"
                ) from error
            if (
                vector_array.size != 4
                or not np.all(np.isfinite(vector_array))
                or np.any(vector_array < 0)
                or float(vector_array.sum()) <= 0
            ):
                raise ValueError(
                    f"{label} calibration vector for {profile!r} must be a "
                    "finite non-negative four-way preference"
                )
            vector = tuple(float(value) for value in vector_array)
            previous = vectors.setdefault(profile, vector)
            if vector != previous:
                raise ValueError(
                    f"{label} records inconsistent calibration vectors for "
                    f"profile {profile!r}"
                )
        vectors_by_input.append(vectors)
    oracle_vectors, gppo_vectors = vectors_by_input
    if set(oracle_vectors) != set(gppo_vectors):
        raise ValueError(
            "Oracle and GPPO dynamic calibration profile-vector sets differ"
        )
    for profile in sorted(oracle_vectors):
        if oracle_vectors[profile] != gppo_vectors[profile]:
            raise ValueError(
                "calibration_profile_vector mismatch for profile "
                f"{profile!r}"
            )
    return {
        "mode": "dynamic_calibration",
        "priority_share": reference_share,
        "profile_vectors": {
            profile: list(oracle_vectors[profile])
            for profile in sorted(oracle_vectors)
        },
    }


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol_field: str,
    suite_fields: Sequence[str],
    label: str,
    extra_key_fields: Sequence[str] = (),
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    key_fields = (protocol_field, *suite_fields, *PAIR_FIELDS, *extra_key_fields)
    for row_number, row in enumerate(rows):
        missing = [field for field in key_fields if not _nonempty(row, field)]
        if missing:
            raise ValueError(
                f"{label} row {row_number} lacks pairing metadata: {missing!r}"
            )
        key = tuple(_canonical(row[field], field=field) for field in key_fields)
        if key in index:
            raise ValueError(
                f"{label} contains duplicate paired episode key {key!r}"
            )
        for metric in METRIC_FIELDS:
            _metric(row, metric)
        index[key] = row
    return index


def _collapse_gppo_training_seeds(
    index: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> tuple[dict[tuple[Any, ...], Mapping[str, Any]], tuple[int, ...]]:
    """Average frozen GPPO checkpoints within each paired episode cell."""

    grouped: dict[tuple[Any, ...], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for full_key, row in index.items():
        core_key = full_key[:-1]
        training_seed = int(full_key[-1])
        grouped[core_key].append((training_seed, row))
    expected_seeds: tuple[int, ...] | None = None
    collapsed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for core_key in sorted(grouped, key=repr):
        seed_rows = sorted(grouped[core_key], key=lambda item: item[0])
        seeds = tuple(seed for seed, _ in seed_rows)
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError(
                "every GPPO eval_seed × profile × scale cell must contain the "
                "same complete frozen training-seed grid; "
                f"cell={core_key!r} has {seeds!r}, expected {expected_seeds!r}"
            )
        representative = dict(seed_rows[0][1])
        representative["gppo_training_seeds"] = list(seeds)
        for metric in METRIC_FIELDS:
            representative[metric] = _mean(
                (row for _, row in seed_rows), metric
            )
        collapsed[core_key] = representative
    if not expected_seeds:
        raise ValueError("GPPO input has no training seeds")
    return collapsed, expected_seeds


def _describe_key_difference(
    oracle_keys: set[tuple[Any, ...]],
    gppo_keys: set[tuple[Any, ...]],
    key_fields: Sequence[str],
) -> str:
    missing_gppo = sorted(oracle_keys - gppo_keys, key=repr)
    missing_oracle = sorted(gppo_keys - oracle_keys, key=repr)
    return (
        "Oracle and GPPO paired episode keys differ across "
        f"{tuple(key_fields)!r}; missing_from_gppo={missing_gppo[:3]!r}, "
        f"missing_from_oracle={missing_oracle[:3]!r}"
    )


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [_metric(row, field) for row in rows]
    return float(np.mean(values))


def _relative_improvement(gppo_l1: np.ndarray, oracle_l1: np.ndarray) -> float:
    denominator = float(np.mean(gppo_l1))
    if denominator <= 0:
        raise ValueError("macro GPPO preference_l1 must be positive")
    return float(1.0 - float(np.mean(oracle_l1)) / denominator)


def paired_bootstrap_ci(
    gppo_l1: Sequence[float],
    oracle_l1: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260802,
) -> dict[str, float | int]:
    """Bootstrap paired eval-seed macro values, never raw episode rows."""

    gppo = np.asarray(gppo_l1, dtype=np.float64)
    oracle = np.asarray(oracle_l1, dtype=np.float64)
    if gppo.ndim != 1 or oracle.shape != gppo.shape or gppo.size == 0:
        raise ValueError("paired bootstrap inputs must be non-empty equal vectors")
    if not np.all(np.isfinite(gppo)) or not np.all(np.isfinite(oracle)):
        raise ValueError("paired bootstrap inputs must be finite")
    if np.any(gppo <= 0):
        raise ValueError("paired GPPO L1 values must be positive")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, gppo.size, size=gppo.size)
        estimates[index] = _relative_improvement(gppo[selected], oracle[selected])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, (alpha, 1.0 - alpha))
    return {
        "samples": samples,
        "confidence": confidence,
        "seed": seed,
        "low": float(low),
        "high": float(high),
    }


def paired_bootstrap_metrics(
    gppo: Mapping[str, Sequence[float]],
    oracle: Mapping[str, Sequence[float]],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260802,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap all headroom and guard statistics over paired eval seeds."""

    required = set(METRIC_FIELDS)
    if set(gppo) != required or set(oracle) != required:
        raise ValueError(f"bootstrap metric mappings must contain {sorted(required)!r}")
    gppo_arrays = {name: np.asarray(values, dtype=np.float64) for name, values in gppo.items()}
    oracle_arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in oracle.items()
    }
    sizes = {array.size for array in (*gppo_arrays.values(), *oracle_arrays.values())}
    if len(sizes) != 1 or next(iter(sizes), 0) == 0:
        raise ValueError("bootstrap metrics must be non-empty equal vectors")
    if any(array.ndim != 1 or not np.all(np.isfinite(array)) for array in (*gppo_arrays.values(), *oracle_arrays.values())):
        raise ValueError("bootstrap metrics must be finite one-dimensional vectors")
    if np.any(gppo_arrays["preference_l1"] <= 0):
        raise ValueError("paired GPPO L1 values must be positive")
    if np.any(gppo_arrays["makespan"] <= 0):
        raise ValueError("paired GPPO makespan values must be positive")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    count = next(iter(sizes))
    rng = np.random.default_rng(seed)
    estimates = {
        "preference_l1_relative_improvement": np.empty(samples, dtype=np.float64),
        "deadline_completion_delta": np.empty(samples, dtype=np.float64),
        "makespan_increase_ratio": np.empty(samples, dtype=np.float64),
        "minimum_task_coverage_delta": np.empty(samples, dtype=np.float64),
    }
    for index in range(samples):
        selected = rng.integers(0, count, size=count)
        estimates["preference_l1_relative_improvement"][index] = _relative_improvement(
            gppo_arrays["preference_l1"][selected],
            oracle_arrays["preference_l1"][selected],
        )
        estimates["deadline_completion_delta"][index] = float(
            np.mean(
                oracle_arrays["deadline_completion_rate"][selected]
                - gppo_arrays["deadline_completion_rate"][selected]
            )
        )
        estimates["makespan_increase_ratio"][index] = float(
            np.mean(oracle_arrays["makespan"][selected])
            / np.mean(gppo_arrays["makespan"][selected])
            - 1.0
        )
        estimates["minimum_task_coverage_delta"][index] = float(
            np.mean(
                oracle_arrays["minimum_task_coverage"][selected]
                - gppo_arrays["minimum_task_coverage"][selected]
            )
        )
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float | int]] = {}
    for name, values in estimates.items():
        low, high = np.quantile(values, (alpha, 1.0 - alpha))
        result[name] = {
            "samples": samples,
            "confidence": confidence,
            "seed": seed,
            "low": float(low),
            "high": float(high),
        }
    return result


def analyze_headroom(
    oracle_rows: Sequence[Mapping[str, Any]],
    gppo_rows: Sequence[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    oracle_method: str = "oracle_perfect_information",
    gppo_method: str = "gppo_event",
    oracle_variant: str | None = None,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 20260802,
    max_dcr_drop: float = 0.03,
    max_makespan_increase_ratio: float = 0.05,
    max_minimum_coverage_drop: float = 0.05,
) -> dict[str, Any]:
    if not oracle_rows or not gppo_rows:
        raise ValueError("Oracle and GPPO inputs must both be non-empty")
    selected_profiles = tuple(str(profile).strip() for profile in profiles)
    oracle_rows = _select_profiles(oracle_rows, selected_profiles, label="Oracle")
    gppo_rows = _select_profiles(gppo_rows, selected_profiles, label="GPPO")
    calibration_identity = _calibration_identity(oracle_rows, gppo_rows)
    for name, value in (
        ("max_dcr_drop", max_dcr_drop),
        ("max_makespan_increase_ratio", max_makespan_increase_ratio),
        ("max_minimum_coverage_drop", max_minimum_coverage_drop),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    _validate_method(oracle_rows, oracle_method, label="Oracle")
    _validate_method(gppo_rows, gppo_method, label="GPPO")
    oracle_variant_field, resolved_oracle_variant = _oracle_variant(
        oracle_rows, oracle_variant
    )
    protocol_field = _common_identity_field(
        oracle_rows,
        gppo_rows,
        ("protocol_version", "scenario_version"),
        label="protocol",
    )
    suite_fields = _suite_fields(oracle_rows, gppo_rows)
    key_fields = (protocol_field, *suite_fields, *PAIR_FIELDS)
    oracle_index = _index_rows(
        oracle_rows,
        protocol_field=protocol_field,
        suite_fields=suite_fields,
        label="Oracle",
    )
    gppo_full_index = _index_rows(
        gppo_rows,
        protocol_field=protocol_field,
        suite_fields=suite_fields,
        label="GPPO",
        extra_key_fields=("training_seed",),
    )
    gppo_index, gppo_training_seeds = _collapse_gppo_training_seeds(
        gppo_full_index
    )
    oracle_keys = set(oracle_index)
    gppo_keys = set(gppo_index)
    if oracle_keys != gppo_keys:
        raise ValueError(
            _describe_key_difference(oracle_keys, gppo_keys, key_fields)
        )

    protocol_values = sorted({key[0] for key in oracle_keys}, key=str)
    suite_values = sorted(
        {key[1 : 1 + len(suite_fields)] for key in oracle_keys}, key=repr
    )
    if len(protocol_values) != 1:
        raise ValueError(f"inputs mix multiple protocols: {protocol_values!r}")
    if len(suite_values) != 1:
        raise ValueError(f"inputs mix multiple suites: {suite_values!r}")

    decay_position = key_fields.index("assignment_priority_decay")
    target_position = key_fields.index("preference_target_mode")
    decay_values = sorted({key[decay_position] for key in oracle_keys})
    target_values = sorted({key[target_position] for key in oracle_keys}, key=str)
    if len(decay_values) != 1:
        raise ValueError(
            "headroom analysis requires one frozen decay; do not pool calibration "
            f"settings: {decay_values!r}"
        )
    if len(target_values) != 1:
        raise ValueError(
            "headroom analysis requires one target mode; do not pool settings: "
            f"{target_values!r}"
        )

    eval_seed_position = key_fields.index("eval_seed")
    seed_groups: dict[int, list[tuple[Any, ...]]] = defaultdict(list)
    cell_groups: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
    for key in oracle_keys:
        seed_groups[int(key[eval_seed_position])].append(key)
        cell_key = key[:eval_seed_position] + key[eval_seed_position + 1 :]
        cell_groups[cell_key].append(key)

    expected_seed_grid: set[tuple[Any, ...]] | None = None
    seed_summaries: list[dict[str, Any]] = []
    for eval_seed in sorted(seed_groups):
        episode_keys = sorted(seed_groups[eval_seed], key=repr)
        seed_grid = {
            key[:eval_seed_position] + key[eval_seed_position + 1 :]
            for key in episode_keys
        }
        if expected_seed_grid is None:
            expected_seed_grid = seed_grid
        elif seed_grid != expected_seed_grid:
            raise ValueError(
                "every eval_seed must contain the same paired profile×scale cell "
                f"grid; eval_seed={eval_seed} differs"
            )
        oracle_seed = [oracle_index[key] for key in episode_keys]
        gppo_seed = [gppo_index[key] for key in episode_keys]
        seed_summaries.append(
            {
                "eval_seed": eval_seed,
                "n_cells": len(episode_keys),
                "gppo": {field: _mean(gppo_seed, field) for field in METRIC_FIELDS},
                "oracle": {
                    field: _mean(oracle_seed, field) for field in METRIC_FIELDS
                },
            }
        )

    cells: list[dict[str, Any]] = []
    for cell_key in sorted(cell_groups, key=repr):
        episode_keys = sorted(cell_groups[cell_key], key=repr)
        oracle_cell = [oracle_index[key] for key in episode_keys]
        gppo_cell = [gppo_index[key] for key in episode_keys]
        oracle_metrics = {
            field: _mean(oracle_cell, field) for field in METRIC_FIELDS
        }
        gppo_metrics = {field: _mean(gppo_cell, field) for field in METRIC_FIELDS}
        if gppo_metrics["preference_l1"] <= 0:
            raise ValueError(f"cell {cell_key!r} has non-positive GPPO L1")
        if gppo_metrics["makespan"] <= 0:
            raise ValueError(f"cell {cell_key!r} has non-positive GPPO makespan")
        relative = 1.0 - (
            oracle_metrics["preference_l1"] / gppo_metrics["preference_l1"]
        )
        dcr_delta = (
            oracle_metrics["deadline_completion_rate"]
            - gppo_metrics["deadline_completion_rate"]
        )
        makespan_ratio = (
            oracle_metrics["makespan"] / gppo_metrics["makespan"] - 1.0
        )
        coverage_delta = (
            oracle_metrics["minimum_task_coverage"]
            - gppo_metrics["minimum_task_coverage"]
        )
        metadata = {
            field: value
            for field, value in zip(
                (field for field in key_fields if field != "eval_seed"),
                cell_key,
                strict=True,
            )
        }
        cells.append(
            {
                "cell": metadata,
                "paired_eval_seeds": [
                    int(key[eval_seed_position]) for key in episode_keys
                ],
                "n_paired_episodes": len(episode_keys),
                "gppo": gppo_metrics,
                "oracle": oracle_metrics,
                "preference_l1_absolute_improvement": (
                    gppo_metrics["preference_l1"]
                    - oracle_metrics["preference_l1"]
                ),
                "preference_l1_relative_improvement": relative,
                "guards": {
                    "deadline_completion": {
                        "delta": dcr_delta,
                        "pass": dcr_delta >= -max_dcr_drop,
                    },
                    "makespan": {
                        "increase_ratio": makespan_ratio,
                        "pass": makespan_ratio <= max_makespan_increase_ratio,
                    },
                    "minimum_task_coverage": {
                        "delta": coverage_delta,
                        "pass": coverage_delta >= -max_minimum_coverage_drop,
                    },
                },
            }
        )

    gppo_seed_metrics = {
        field: np.asarray(
            [seed["gppo"][field] for seed in seed_summaries], dtype=np.float64
        )
        for field in METRIC_FIELDS
    }
    oracle_seed_metrics = {
        field: np.asarray(
            [seed["oracle"][field] for seed in seed_summaries], dtype=np.float64
        )
        for field in METRIC_FIELDS
    }
    gppo_l1 = gppo_seed_metrics["preference_l1"]
    oracle_l1 = oracle_seed_metrics["preference_l1"]
    macro_gppo = {
        field: float(np.mean([seed["gppo"][field] for seed in seed_summaries]))
        for field in METRIC_FIELDS
    }
    macro_oracle = {
        field: float(np.mean([seed["oracle"][field] for seed in seed_summaries]))
        for field in METRIC_FIELDS
    }
    macro_dcr_delta = (
        macro_oracle["deadline_completion_rate"]
        - macro_gppo["deadline_completion_rate"]
    )
    if macro_gppo["makespan"] <= 0:
        raise ValueError("macro GPPO makespan must be positive")
    macro_makespan_ratio = (
        macro_oracle["makespan"] / macro_gppo["makespan"] - 1.0
    )
    macro_coverage_delta = (
        macro_oracle["minimum_task_coverage"]
        - macro_gppo["minimum_task_coverage"]
    )
    bootstrap = paired_bootstrap_metrics(
        gppo_seed_metrics,
        oracle_seed_metrics,
        samples=bootstrap_samples,
        confidence=confidence,
        seed=bootstrap_seed,
    )
    guards = {
        "deadline_completion": {
            "allowed_drop": max_dcr_drop,
            "delta": macro_dcr_delta,
            "paired_bootstrap_ci": bootstrap["deadline_completion_delta"],
            "point_estimate_pass": macro_dcr_delta >= -max_dcr_drop,
            "pass": bootstrap["deadline_completion_delta"]["low"] >= -max_dcr_drop,
        },
        "makespan": {
            "allowed_increase_ratio": max_makespan_increase_ratio,
            "increase_ratio": macro_makespan_ratio,
            "paired_bootstrap_ci": bootstrap["makespan_increase_ratio"],
            "point_estimate_pass": macro_makespan_ratio <= max_makespan_increase_ratio,
            "pass": bootstrap["makespan_increase_ratio"]["high"] <= max_makespan_increase_ratio,
        },
        "minimum_task_coverage": {
            "allowed_drop": max_minimum_coverage_drop,
            "delta": macro_coverage_delta,
            "paired_bootstrap_ci": bootstrap["minimum_task_coverage_delta"],
            "point_estimate_pass": macro_coverage_delta >= -max_minimum_coverage_drop,
            "pass": bootstrap["minimum_task_coverage_delta"]["low"] >= -max_minimum_coverage_drop,
        },
    }
    guard_cell_violations = {
        name: sum(not bool(cell["guards"][name]["pass"]) for cell in cells)
        for name in guards
    }

    by_profile: dict[str, dict[str, Any]] = {}
    for profile in sorted(
        {str(cell["cell"]["preference_profile"]) for cell in cells}
    ):
        selected = [
            cell for cell in cells if cell["cell"]["preference_profile"] == profile
        ]
        profile_gppo = np.asarray(
            [cell["gppo"]["preference_l1"] for cell in selected], dtype=np.float64
        )
        profile_oracle = np.asarray(
            [cell["oracle"]["preference_l1"] for cell in selected], dtype=np.float64
        )
        by_profile[profile] = {
            "n_cells": len(selected),
            "gppo_preference_l1": float(profile_gppo.mean()),
            "oracle_preference_l1": float(profile_oracle.mean()),
            "oracle_relative_improvement": _relative_improvement(
                profile_gppo, profile_oracle
            ),
        }

    relative = _relative_improvement(gppo_l1, oracle_l1)
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_role": (
            "pre-training Oracle-vs-frozen-GPPO headroom diagnostic; not the "
            "formal 30% acceptance comparison"
        ),
        "formal_acceptance_denominator": (
            "newly trained hard-3 no-conditioning checkpoint"
        ),
        "methods": {
            "oracle": oracle_method,
            "gppo": gppo_method,
            "frozen_gppo_training_seeds": list(gppo_training_seeds),
        },
        "profiles": list(selected_profiles),
        "oracle_variant": {
            "field": oracle_variant_field,
            "value": resolved_oracle_variant,
            "pooled": False,
        },
        "identity": {
            "protocol_field": protocol_field,
            "protocol": protocol_values[0],
            "suite_fields": list(suite_fields),
            "suite": list(suite_values[0]),
            "calibration": calibration_identity,
        },
        "statistical_unit": {
            "name": "paired_eval_seed_macros",
            "n_eval_seeds": len(seed_summaries),
            "cells_per_eval_seed": len(expected_seed_grid or ()),
            "n_cells": len(cells),
            "n_paired_episode_rows": len(oracle_keys),
            "n_gppo_rows_before_training_seed_average": len(gppo_full_index),
            "eval_seed_rows_are_training_seeds": False,
            "gppo_training_seeds_are_bootstrap_units": False,
            "gppo_training_seed_aggregation": (
                "mean within each eval_seed × profile × scale cell"
            ),
            "bootstrap_resamples": "paired_eval_seed_macros",
        },
        "macro": {
            "gppo": macro_gppo,
            "oracle": macro_oracle,
            "preference_l1_absolute_improvement": float(
                macro_gppo["preference_l1"] - macro_oracle["preference_l1"]
            ),
            "oracle_relative_improvement": relative,
            "oracle_relative_improvement_paired_bootstrap_ci": bootstrap[
                "preference_l1_relative_improvement"
            ],
        },
        "guards": {
            **guards,
            "all_macro_guards_pass": all(bool(row["pass"]) for row in guards.values()),
            "cell_violation_counts": guard_cell_violations,
        },
        "by_profile": by_profile,
        "by_eval_seed": seed_summaries,
        "cells": cells,
    }
    return result


def render_markdown(result: Mapping[str, Any]) -> str:
    macro = result["macro"]
    guards = result["guards"]
    ci = macro["oracle_relative_improvement_paired_bootstrap_ci"]
    lines = [
        "# PCRL Oracle versus frozen GPPO headroom",
        "",
        "> The paired bootstrap resamples eval seeds after macro-aggregating the "
        "requested profile × scale cells within each seed. Episode rows are not "
        "treated as IID or as training seeds.",
        "",
        f"- Paired eval seeds: `{result['statistical_unit']['n_eval_seeds']}`",
        f"- Cells per eval seed: `{result['statistical_unit']['cells_per_eval_seed']}`",
        f"- Paired episode rows: `{result['statistical_unit']['n_paired_episode_rows']}`",
        f"- GPPO macro L1: `{macro['gppo']['preference_l1']:.6f}`",
        f"- Oracle macro L1: `{macro['oracle']['preference_l1']:.6f}`",
        f"- Oracle relative improvement: `{macro['oracle_relative_improvement']:.2%}`",
        f"- Paired bootstrap {ci['confidence']:.0%} CI: "
        f"`[{ci['low']:.2%}, {ci['high']:.2%}]`",
        "",
        "| guard | observed | paired bootstrap CI | limit | CI pass |",
        "| --- | ---: | ---: | ---: | :---: |",
        f"| deadline completion delta | {guards['deadline_completion']['delta']:.6f} | "
        f"[{guards['deadline_completion']['paired_bootstrap_ci']['low']:.6f}, "
        f"{guards['deadline_completion']['paired_bootstrap_ci']['high']:.6f}] | "
        f">= {-guards['deadline_completion']['allowed_drop']:.6f} | "
        f"{guards['deadline_completion']['pass']} |",
        f"| makespan increase | {guards['makespan']['increase_ratio']:.2%} | "
        f"[{guards['makespan']['paired_bootstrap_ci']['low']:.2%}, "
        f"{guards['makespan']['paired_bootstrap_ci']['high']:.2%}] | "
        f"<= {guards['makespan']['allowed_increase_ratio']:.2%} | "
        f"{guards['makespan']['pass']} |",
        f"| minimum coverage delta | {guards['minimum_task_coverage']['delta']:.6f} | "
        f"[{guards['minimum_task_coverage']['paired_bootstrap_ci']['low']:.6f}, "
        f"{guards['minimum_task_coverage']['paired_bootstrap_ci']['high']:.6f}] | "
        f">= {-guards['minimum_task_coverage']['allowed_drop']:.6f} | "
        f"{guards['minimum_task_coverage']['pass']} |",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze paired Oracle headroom over frozen GPPO-event"
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument(
        "--gppo",
        type=Path,
        nargs="+",
        required=True,
        help="one or more frozen-GPPO evaluation.json files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        required=True,
        help="explicit profile subset; no profiles are pooled implicitly",
    )
    parser.add_argument("--oracle-method", default="oracle_perfect_information")
    parser.add_argument("--gppo-method", default="gppo_event")
    parser.add_argument(
        "--oracle-variant", choices=("frozen_mask", "relaxed_wait")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-dcr-drop", type=float, default=0.03)
    parser.add_argument("--max-makespan-increase-ratio", type=float, default=0.05)
    parser.add_argument("--max-minimum-coverage-drop", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gppo_rows = [row for path in args.gppo for row in _load_rows(path)]
    result = analyze_headroom(
        _load_rows(args.oracle),
        gppo_rows,
        profiles=args.profiles,
        oracle_method=args.oracle_method,
        gppo_method=args.gppo_method,
        oracle_variant=args.oracle_variant,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        bootstrap_seed=args.bootstrap_seed,
        max_dcr_drop=args.max_dcr_drop,
        max_makespan_increase_ratio=args.max_makespan_increase_ratio,
        max_minimum_coverage_drop=args.max_minimum_coverage_drop,
    )
    result["sources"] = {
        "oracle": str(args.oracle.resolve()),
        "gppo": [str(path.resolve()) for path in args.gppo],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "headroom.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "PCRL_HEADROOM.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_cells": result["statistical_unit"]["n_cells"],
                "oracle_relative_improvement": result["macro"][
                    "oracle_relative_improvement"
                ],
                "all_macro_guards_pass": result["guards"][
                    "all_macro_guards_pass"
                ],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

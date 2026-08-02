from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import numpy as np

from summarize_pcrl_v0 import (
    HARD4_DIRECTION_INDEX,
    ci95,
    exact_sign_flip_pvalue,
    hard4_pilot_acceptance,
    paired_seed_comparison,
)
from uav_assignment.pcrl_v0 import calibration_priority_profiles


FROZEN_REGISTRY = json.loads(
    Path("configs/pcrl_v0_frozen_checkpoints.json").read_text(encoding="utf-8")
)
FROZEN_GPPO_IMPLEMENTATION_HASH = FROZEN_REGISTRY[
    "frozen_gppo_implementation_hash"
]
FROZEN_GPPO_SHA_BY_SEED = {
    int(entry["seed"]): str(entry["sha256"])
    for entry in FROZEN_REGISTRY["checkpoints"]
    if entry["method"] == "gppo_event"
}


def test_ci95_uses_student_t_for_five_seeds() -> None:
    interval = ci95([1.0, 2.0, 3.0, 4.0, 5.0])

    assert interval["mean"] == pytest.approx(3.0)
    assert interval["low"] == pytest.approx(1.0367568385)
    assert interval["high"] == pytest.approx(4.9632431615)
    assert interval["n"] == 5


def test_paired_seed_comparison_reports_differences_ci_and_exact_test() -> None:
    seed_rows = []
    for seed, difference in enumerate((-1.0, -2.0, -3.0, -4.0, -5.0), start=1):
        seed_rows.extend(
            (
                {
                    "method_id": "pcrl",
                    "training_seed": seed,
                    "preference_l1": 10.0 + difference,
                    "preference_cosine": 10.0 - difference,
                },
                {
                    "method_id": "no_conditioning",
                    "training_seed": seed,
                    "preference_l1": 10.0,
                    "preference_cosine": 10.0,
                },
            )
        )

    comparison = paired_seed_comparison(
        seed_rows,
        first_method="pcrl",
        second_method="no_conditioning",
        metrics=("preference_l1", "preference_cosine"),
    )

    assert comparison["paired_training_seeds"] == [1, 2, 3, 4, 5]
    l1 = comparison["metrics"]["preference_l1"]
    assert [row["difference"] for row in l1["seed_differences"]] == pytest.approx(
        [-1.0, -2.0, -3.0, -4.0, -5.0]
    )
    assert l1["mean_difference"] == pytest.approx(-3.0)
    assert l1["ci_low"] == pytest.approx(-4.9632431615)
    assert l1["ci_high"] == pytest.approx(-1.0367568385)
    assert l1["exact_two_sided_sign_flip_p"] == pytest.approx(0.0625)
    assert l1["ci_excludes_zero"] is True
    assert l1["improved_seed_count"] == 5
    assert l1["worsened_seed_count"] == 0
    assert l1["tied_seed_count"] == 0

    cosine = comparison["metrics"]["preference_cosine"]
    assert cosine["direction"] == "higher_is_better"
    assert cosine["mean_difference"] == pytest.approx(3.0)
    assert cosine["improved_seed_count"] == 5


def test_exact_sign_flip_pvalue_is_nan_above_exact_limit() -> None:
    assert math.isnan(exact_sign_flip_pvalue([1.0] * 17))


def hard4_rows() -> list[dict[str, object]]:
    family = calibration_priority_profiles(0.40)
    rows: list[dict[str, object]] = []
    methods = (
        ("pcrl_gppo_adaptive", True),
        ("pcrl_gppo_adaptive_no_conditioning", False),
        ("gppo_event", None),
    )
    for seed in range(1, 6):
        for profile, vector in family.items():
            for scale in ("2x12", "3x16", "3x20", "4x24"):
                for eval_seed in range(76000, 76020):
                    for method, conditioning in methods:
                        primary = profile in tuple(family)[:5]
                        if method == "pcrl_gppo_adaptive":
                            l1 = 0.60 if primary else 0.84
                            dcr, makespan, coverage = 0.79, 10.2, 0.59
                        elif method == "pcrl_gppo_adaptive_no_conditioning":
                            l1 = 1.00 if primary else 1.20
                            dcr, makespan, coverage = 0.78, 10.3, 0.58
                        else:
                            l1 = 0.01
                            dcr, makespan, coverage = 0.80, 10.0, 0.60
                        mix = np.full(4, 0.25, dtype=np.float64)
                        if (
                            method == "pcrl_gppo_adaptive"
                            and profile in HARD4_DIRECTION_INDEX
                        ):
                            component = HARD4_DIRECTION_INDEX[profile]
                            mix[component] = 0.35
                            mix[(component + 1) % 4] = 0.15
                        row: dict[str, object] = {
                            "method_id": method,
                            "training_seed": seed,
                            "preference_profile": profile,
                            "scale": scale,
                            "eval_seed": eval_seed,
                            "protocol_version": "pcrl-v0-hard-4",
                            "protocol_config_sha256": "1" * 64,
                            "artifact_group": "pilot20",
                            "output_namespace": "outputs/pcrl_v0/hard4",
                            "base_protocol": "gppo-v2-hard-3",
                            "scenario_version": "gppo-v2-hard-3",
                            "scenario_config_sha256": "2" * 64,
                            "scenario_hash": "3" * 64,
                            "frozen_gppo_implementation_hash": (
                                FROZEN_GPPO_IMPLEMENTATION_HASH
                            ),
                            "source_gppo_checkpoint_sha256": (
                                FROZEN_GPPO_SHA_BY_SEED[seed]
                            ),
                            "pcrl_implementation_hash": "4" * 64,
                            "evaluation_suite": "controllability_phase",
                            "preference_profile_family": "dynamic-priority-share-v1",
                            "updates": 100 if method == "gppo_event" else 20,
                            "episodes_per_update": 18,
                            "evaluation_episodes": 20,
                            "evaluation_seed": eval_seed,
                            "algorithm": "gppo" if method == "gppo_event" else "preco",
                            "graph_mode": "adaptive",
                            "task_release_mode": "phase_staggered",
                            "preference_target_mode": "nominal",
                            "deadline_scale": 0.70,
                            "assignment_priority_decay": 2.0,
                            "priority_share": 0.40,
                            "background_share": 0.20,
                            "preference_profile_vector": vector.tolist(),
                            "preference_l1": l1,
                            "deadline_completion_rate": dcr,
                            "makespan": makespan,
                            "minimum_task_coverage": coverage,
                            "priority_weighted_assignment_mix": mix.tolist(),
                        }
                        row["preference_conditioning"] = (
                            False if conditioning is None else conditioning
                        )
                        rows.append(row)
    return rows


def test_hard4_pilot_uses_true_no_conditioning_and_exact_cells() -> None:
    result = hard4_pilot_acceptance(hard4_rows())

    preference = result["preference_control"]
    assert preference["denominator_method"] == (
        "pcrl_gppo_adaptive_no_conditioning"
    )
    assert preference["relative_improvement_ci"]["mean"] == pytest.approx(0.40)
    assert preference["relative_improvement_ci"]["low"] == pytest.approx(0.40)
    assert preference["threshold_exact_two_sided_sign_flip_p"] == pytest.approx(
        0.0625
    )
    assert result["gppo_excluded_from_preference_denominator"] is True
    assert all(item["passed"] for item in result["monotonicity"].values())
    assert result["held_out_same_family"]["relative_improvement_ci"][
        "mean"
    ] == pytest.approx(0.30)
    assert result["gppo_efficiency_guards"]["deadline_completion"]["passed"]
    assert result["gppo_efficiency_guards"]["makespan"]["passed"]
    assert result["gppo_efficiency_guards"]["minimum_task_coverage"]["passed"]
    assert "minimum attainable" in result["exact_inference_limitation"]


def test_hard4_pilot_rejects_missing_or_mismatched_eval_cells() -> None:
    rows = hard4_rows()
    rows.pop(
        next(
            index
            for index, row in enumerate(rows)
            if row["method_id"] == "pcrl_gppo_adaptive"
        )
    )
    with pytest.raises(ValueError, match="identical eval cell grid|identical training-seed"):
        hard4_pilot_acceptance(rows)


def test_hard4_pilot_rejects_false_no_conditioning_metadata() -> None:
    rows = hard4_rows()
    for row in rows:
        if row["method_id"] == "pcrl_gppo_adaptive_no_conditioning":
            row["preference_conditioning"] = True
            break
    with pytest.raises(ValueError, match="preference_conditioning metadata"):
        hard4_pilot_acceptance(rows)


@pytest.mark.parametrize("invalid", (None, 0, "false"))
def test_hard4_requires_explicit_boolean_conditioning_metadata(
    invalid: object,
) -> None:
    rows = hard4_rows()
    target = next(
        row
        for row in rows
        if row["method_id"] == "pcrl_gppo_adaptive_no_conditioning"
    )
    if invalid is None:
        target.pop("preference_conditioning")
    else:
        target["preference_conditioning"] = invalid
    with pytest.raises(ValueError, match="boolean preference_conditioning"):
        hard4_pilot_acceptance(rows)


def test_hard4_requires_gppo_conditioning_identity_to_be_false() -> None:
    rows = hard4_rows()
    next(row for row in rows if row["method_id"] == "gppo_event")[
        "preference_conditioning"
    ] = True
    with pytest.raises(ValueError, match="preference_conditioning metadata"):
        hard4_pilot_acceptance(rows)


def test_hard4_primary_effect_excludes_held_out_and_gppo_l1() -> None:
    rows = hard4_rows()
    held = set(tuple(calibration_priority_profiles(0.40))[5:])
    for row in rows:
        if (
            row["method_id"] == "pcrl_gppo_adaptive_no_conditioning"
            and row["preference_profile"] in held
        ):
            row["preference_l1"] = 100.0
        if row["method_id"] == "gppo_event":
            row["preference_l1"] = 999.0
    result = hard4_pilot_acceptance(rows)
    assert result["preference_control"]["relative_improvement_ci"][
        "mean"
    ] == pytest.approx(0.40)
    assert result["held_out_same_family"]["relative_improvement_ci"][
        "mean"
    ] > 0.99


def test_hard4_rejects_profile_vector_mismatch() -> None:
    rows = hard4_rows()
    rows[0]["preference_profile_vector"] = [0.4, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError, match="profile vector mismatch"):
        hard4_pilot_acceptance(rows)


def test_hard4_accepts_json_roundoff_in_background_share() -> None:
    rows = hard4_rows()
    for row in rows:
        row["background_share"] = 0.19999999999999998
    result = hard4_pilot_acceptance(rows)
    assert result["protocol_version"] == "pcrl-v0-hard-4"


def test_hard4_rejects_material_background_share_mismatch() -> None:
    rows = hard4_rows()
    rows[0]["background_share"] = 0.2001
    with pytest.raises(ValueError, match="background share"):
        hard4_pilot_acceptance(rows)


def test_hard4_reports_monotonic_and_gppo_guard_failures_separately() -> None:
    rows = hard4_rows()
    for row in rows:
        if row["method_id"] == "pcrl_gppo_adaptive":
            row["makespan"] = 11.0
        if (
            row["method_id"] == "pcrl_gppo_adaptive"
            and row["preference_profile"] == "calibration_search_priority"
        ):
            row["priority_weighted_assignment_mix"] = [0.20, 0.30, 0.25, 0.25]
    result = hard4_pilot_acceptance(rows)
    assert result["monotonicity"]["calibration_search_priority"]["passed"] is False
    assert result["gppo_efficiency_guards"]["makespan"]["passed"] is False
    assert result["preference_control"]["relative_improvement_ci"][
        "mean"
    ] == pytest.approx(0.40)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_group", "formal100", "artifact_group"),
        ("evaluation_suite", "chain_compatibility", "evaluation suite"),
        ("preference_profile_family", "old-family", "profile family"),
        ("scenario_version", "gppo-v2-hard-2", "scenario_version"),
        ("output_namespace", "outputs/pcrl_v0/hard3", "output_namespace"),
        ("algorithm", "ls", "algorithm identity"),
        ("graph_mode", "single_head", "graph_mode identity"),
        ("updates", 100, "update budget"),
    ),
)
def test_hard4_rejects_mixed_pilot_identity(
    field: str, value: object, message: str
) -> None:
    rows = hard4_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        hard4_pilot_acceptance(rows)


def test_hard4_requires_frozen_gppo_training_budget() -> None:
    rows = hard4_rows()
    for row in rows:
        if row["method_id"] == "gppo_event":
            row["updates"] = 20
            break
    with pytest.raises(ValueError, match="training update budget"):
        hard4_pilot_acceptance(rows)


def test_hard4_rejects_incomplete_scale_set() -> None:
    rows = [row for row in hard4_rows() if row["scale"] != "4x24"]
    with pytest.raises(ValueError, match="scale set"):
        hard4_pilot_acceptance(rows)


@pytest.mark.parametrize(
    "field",
    (
        "protocol_config_sha256",
        "scenario_config_sha256",
        "scenario_hash",
        "frozen_gppo_implementation_hash",
        "source_gppo_checkpoint_sha256",
        "pcrl_implementation_hash",
    ),
)
def test_hard4_rejects_non_sha256_identity(field: str) -> None:
    rows = hard4_rows()
    rows[0][field] = "not-a-sha256"
    with pytest.raises(ValueError, match="valid SHA-256"):
        hard4_pilot_acceptance(rows)


@pytest.mark.parametrize(
    "field",
    (
        "protocol_config_sha256",
        "scenario_config_sha256",
        "pcrl_implementation_hash",
    ),
)
def test_hard4_rejects_cross_method_shared_identity_mismatch(field: str) -> None:
    rows = hard4_rows()
    next(row for row in rows if row["method_id"] == "gppo_event")[field] = "a" * 64
    with pytest.raises(ValueError, match=field):
        hard4_pilot_acceptance(rows)


def test_hard4_rejects_scenario_hash_mismatch_in_same_cell() -> None:
    rows = hard4_rows()
    next(row for row in rows if row["method_id"] == "gppo_event")[
        "scenario_hash"
    ] = "a" * 64
    with pytest.raises(ValueError, match="scenario_hash"):
        hard4_pilot_acceptance(rows)


def test_hard4_rejects_frozen_implementation_hash_not_in_registry() -> None:
    rows = hard4_rows()
    rows[0]["frozen_gppo_implementation_hash"] = "a" * 64
    with pytest.raises(ValueError, match="implementation hash does not match registry"):
        hard4_pilot_acceptance(rows)


def test_hard4_rejects_source_checkpoint_hash_not_in_seed_registry() -> None:
    rows = hard4_rows()
    target = next(
        row
        for row in rows
        if row["method_id"] == "pcrl_gppo_adaptive"
        and row["training_seed"] == 2
    )
    target["source_gppo_checkpoint_sha256"] = FROZEN_GPPO_SHA_BY_SEED[1]
    with pytest.raises(ValueError, match="checkpoint hash does not match registry"):
        hard4_pilot_acceptance(rows)

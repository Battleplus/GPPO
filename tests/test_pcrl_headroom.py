from __future__ import annotations

from copy import deepcopy

import pytest

from analyze_pcrl_headroom import analyze_headroom, paired_bootstrap_ci


def make_rows(method: str, *, oracle: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile_index, profile in enumerate(("balanced", "search_moderate_2to1")):
        for scale_index, scale in enumerate(("2x12", "3x16")):
            for episode in range(3):
                gppo_l1 = 0.4 + 0.04 * profile_index + 0.02 * scale_index
                rows.append(
                    {
                        "method_id": method,
                        "protocol_version": "pcrl-v0-hard-3-calibration",
                        "scenario_version": "gppo-v2-hard-3",
                        "evaluation_suite": "controllability_phase",
                        "task_release_mode": "phase_staggered",
                        "deadline_scale": 0.7,
                        "preference_profile": profile,
                        "scale": scale,
                        "eval_seed": 71000 + episode,
                        "assignment_priority_decay": 2.1,
                        "preference_target_mode": "nominal",
                        "preference_l1": gppo_l1 * (0.5 if oracle else 1.0),
                        "deadline_completion_rate": 0.79 if oracle else 0.80,
                        "makespan": 10.2 if oracle else 10.0,
                        "minimum_task_coverage": 0.58 if oracle else 0.60,
                        **({"oracle_variant": "relaxed_wait"} if oracle else {}),
                        **({"training_seed": 1} if not oracle else {}),
                    }
                )
    return rows


def test_headroom_aggregates_paired_cells_not_episode_rows() -> None:
    result = analyze_headroom(
        make_rows("oracle_perfect_information", oracle=True),
        make_rows("gppo_event", oracle=False),
        profiles=("balanced", "search_moderate_2to1"),
        bootstrap_samples=500,
        bootstrap_seed=9,
    )

    assert result["statistical_unit"] == {
        "name": "paired_eval_seed_macros",
        "n_eval_seeds": 3,
        "cells_per_eval_seed": 4,
        "n_cells": 4,
        "n_paired_episode_rows": 12,
        "n_gppo_rows_before_training_seed_average": 12,
        "eval_seed_rows_are_training_seeds": False,
        "gppo_training_seeds_are_bootstrap_units": False,
        "gppo_training_seed_aggregation": (
            "mean within each eval_seed × profile × scale cell"
        ),
        "bootstrap_resamples": "paired_eval_seed_macros",
    }
    assert result["macro"]["oracle_relative_improvement"] == pytest.approx(0.5)
    assert result["methods"]["frozen_gppo_training_seeds"] == [1]
    ci = result["macro"]["oracle_relative_improvement_paired_bootstrap_ci"]
    assert ci["low"] == pytest.approx(0.5)
    assert ci["high"] == pytest.approx(0.5)
    assert result["guards"]["all_macro_guards_pass"] is True
    assert all(cell["n_paired_episodes"] == 3 for cell in result["cells"])


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("protocol_version", "other-protocol"),
        ("evaluation_suite", "chain_compatibility"),
        ("preference_profile", "strike_moderate_2to1"),
        ("scale", "4x24"),
        ("eval_seed", 99999),
        ("assignment_priority_decay", 3.0),
        ("preference_target_mode", "dependency_feasible_projection"),
    ),
)
def test_headroom_rejects_any_pairing_metadata_mismatch(
    field: str, replacement: object
) -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    gppo[0][field] = replacement

    with pytest.raises(ValueError, match="paired episode keys differ"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )


def test_headroom_rejects_duplicate_or_mixed_method_rows() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    with pytest.raises(ValueError, match="duplicate paired episode key"):
        analyze_headroom(
            oracle + [deepcopy(oracle[0])],
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )
    gppo[0]["method_id"] = "gppo_event_single_head"
    with pytest.raises(ValueError, match="only method_id"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )


def test_headroom_uses_strict_suite_signature_fallback() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    for row in oracle + gppo:
        row.pop("evaluation_suite")
        row.pop("protocol_version")
    result = analyze_headroom(
        oracle,
        gppo,
        profiles=("balanced", "search_moderate_2to1"),
        bootstrap_samples=10,
    )
    assert result["identity"]["protocol_field"] == "scenario_version"
    assert result["identity"]["suite_fields"] == [
        "task_release_mode",
        "deadline_scale",
    ]
    gppo[0]["deadline_scale"] = 1.0
    with pytest.raises(ValueError, match="paired episode keys differ"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )


def test_headroom_reports_each_failed_macro_guard() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    for row in oracle:
        row["deadline_completion_rate"] = 0.70
        row["makespan"] = 11.0
        row["minimum_task_coverage"] = 0.50
    result = analyze_headroom(
        oracle,
        gppo,
        profiles=("balanced", "search_moderate_2to1"),
        bootstrap_samples=20,
    )
    assert result["guards"]["all_macro_guards_pass"] is False
    assert result["guards"]["deadline_completion"]["pass"] is False
    assert result["guards"]["makespan"]["pass"] is False
    assert result["guards"]["minimum_task_coverage"]["pass"] is False
    assert result["guards"]["deadline_completion"]["paired_bootstrap_ci"][
        "high"
    ] == pytest.approx(-0.10)
    assert result["guards"]["makespan"]["paired_bootstrap_ci"][
        "low"
    ] == pytest.approx(0.10)
    assert result["guards"]["minimum_task_coverage"]["paired_bootstrap_ci"][
        "high"
    ] == pytest.approx(-0.10)
    assert result["guards"]["cell_violation_counts"] == {
        "deadline_completion": 4,
        "makespan": 4,
        "minimum_task_coverage": 4,
    }


def test_paired_bootstrap_is_deterministic_and_validates_inputs() -> None:
    first = paired_bootstrap_ci([0.4, 0.6], [0.3, 0.2], samples=100, seed=4)
    second = paired_bootstrap_ci([0.4, 0.6], [0.3, 0.2], samples=100, seed=4)
    assert first == second
    with pytest.raises(ValueError, match="equal vectors"):
        paired_bootstrap_ci([0.4], [0.3, 0.2])
    with pytest.raises(ValueError, match="positive"):
        paired_bootstrap_ci([0.0], [0.0])


def test_explicit_profile_subset_excludes_unrequested_interpolations() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    for source in (oracle, gppo):
        additions = []
        for row in source:
            if row["preference_profile"] != "balanced":
                continue
            copy = deepcopy(row)
            copy["preference_profile"] = "search_strike_interp"
            copy["preference_l1"] = 1.5
            additions.append(copy)
        source.extend(additions)
    result = analyze_headroom(
        oracle,
        gppo,
        profiles=("balanced",),
        bootstrap_samples=20,
    )
    assert result["profiles"] == ["balanced"]
    assert result["statistical_unit"]["cells_per_eval_seed"] == 2
    assert "search_strike_interp" not in result["by_profile"]
    with pytest.raises(ValueError, match="explicit non-empty"):
        analyze_headroom(oracle, gppo, profiles=(), bootstrap_samples=10)


def test_oracle_variants_cannot_be_pooled_and_boolean_fallback_is_explicit() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    oracle[0]["oracle_variant"] = "frozen_mask"
    with pytest.raises(ValueError, match="multiple variants"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )
    for row in oracle:
        row.pop("oracle_variant")
        row["allow_strategic_wait"] = False
    result = analyze_headroom(
        oracle,
        gppo,
        profiles=("balanced", "search_moderate_2to1"),
        oracle_variant="frozen_mask",
        bootstrap_samples=10,
    )
    assert result["oracle_variant"] == {
        "field": "allow_strategic_wait",
        "value": "frozen_mask",
        "pooled": False,
    }


def test_partial_strict_identity_metadata_cannot_fall_back() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    for row in oracle:
        row.pop("protocol_version")
    with pytest.raises(ValueError, match="partially recorded"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )


def test_multiple_gppo_training_seeds_are_averaged_with_complete_grid() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    second_seed = deepcopy(gppo)
    for row in second_seed:
        row["training_seed"] = 2
        row["preference_l1"] = float(row["preference_l1"]) * 1.2
    result = analyze_headroom(
        oracle,
        gppo + second_seed,
        profiles=("balanced", "search_moderate_2to1"),
        bootstrap_samples=10,
    )
    assert result["methods"]["frozen_gppo_training_seeds"] == [1, 2]
    assert result["macro"]["oracle_relative_improvement"] == pytest.approx(
        1.0 - 0.5 / 1.1
    )

    with pytest.raises(ValueError, match="same complete frozen training-seed grid"):
        analyze_headroom(
            oracle,
            gppo + second_seed[:-1],
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )


def add_dynamic_calibration_identity(
    rows: list[dict[str, object]], share: float = 0.4
) -> None:
    vectors = {
        "balanced": [0.25, 0.25, 0.25, 0.25],
        "search_moderate_2to1": [share, (1.0 - share) / 3.0, (1.0 - share) / 3.0, (1.0 - share) / 3.0],
    }
    for row in rows:
        row["calibration_priority_share"] = share
        row["calibration_profile_vector"] = vectors[
            str(row["preference_profile"])
        ]


def test_dynamic_calibration_identity_is_recorded_when_exactly_matched() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    add_dynamic_calibration_identity(oracle)
    add_dynamic_calibration_identity(gppo)
    result = analyze_headroom(
        oracle,
        gppo,
        profiles=("balanced", "search_moderate_2to1"),
        bootstrap_samples=10,
    )
    assert result["identity"]["calibration"]["mode"] == "dynamic_calibration"
    assert result["identity"]["calibration"]["priority_share"] == pytest.approx(0.4)


def test_dynamic_calibration_share_mismatch_is_rejected() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    add_dynamic_calibration_identity(oracle, share=0.4)
    add_dynamic_calibration_identity(gppo, share=0.425)
    with pytest.raises(ValueError, match="calibration_priority_share mismatch"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )


def test_dynamic_calibration_vector_mismatch_is_rejected() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    add_dynamic_calibration_identity(oracle)
    add_dynamic_calibration_identity(gppo)
    for row in gppo:
        if row["preference_profile"] == "balanced":
            row["calibration_profile_vector"] = [0.24, 0.26, 0.25, 0.25]
    with pytest.raises(ValueError, match="calibration_profile_vector mismatch"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )


def test_partial_dynamic_calibration_identity_is_rejected() -> None:
    oracle = make_rows("oracle_perfect_information", oracle=True)
    gppo = make_rows("gppo_event", oracle=False)
    add_dynamic_calibration_identity(oracle)
    with pytest.raises(ValueError, match="completely recorded"):
        analyze_headroom(
            oracle,
            gppo,
            profiles=("balanced", "search_moderate_2to1"),
            bootstrap_samples=10,
        )

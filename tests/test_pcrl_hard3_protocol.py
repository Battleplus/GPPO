from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

import run_pcrl_v0
from evaluate_pcrl_oracle import PROFILE_SETS, resolve_profiles
from run_pcrl_v0 import protocol_evaluation_profiles
from uav_assignment.pcrl_v0 import (
    HARD2_EVALUATION_PROFILE_NAMES,
    HARD3_CALIBRATION_PROFILE_NAMES,
    MODERATE_PRIORITY_PROFILE_NAMES,
    task_preference_profile,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict[str, object]:
    return json.loads((WORKSPACE / "configs" / name).read_text(encoding="utf-8"))


def test_hard3_is_separately_versioned_from_hard2() -> None:
    hard2 = load_config("pcrl_v0.json")
    hard3 = load_config("pcrl_v0_hard3.json")

    assert hard2["version"] == "pcrl-v0-hard-2"
    assert hard3["version"] == "pcrl-v0-hard-3"
    assert hard3["predecessor_protocol"] == hard2["version"]
    assert hard3["base_protocol"] == hard2["base_protocol"] == "gppo-v2-hard-3"

    hard2_metric = hard2["controllability"]
    hard3_metric = hard3["controllability"]
    assert isinstance(hard2_metric, dict)
    assert isinstance(hard3_metric, dict)
    assert "assignment_priority_decay" not in hard2_metric
    assert hard3_metric["hard2_reference_decay"] == pytest.approx(1.0)
    assert hard3_metric["assignment_priority_decay"] == pytest.approx(2.0)
    assert hard3_metric["target"] == "suite_specific"
    assert hard3_metric["hard2_reference_target"] == hard2_metric["target"]
    assert hard3_metric["targets_by_suite"] == {
        "controllability_phase": "nominal",
        "chain_compatibility": "dependency_feasible_projection",
    }
    assert hard3_metric["constraints"] == hard2_metric["constraints"]

    hard2_suites = hard2["evaluation_suites"]
    hard3_suites = hard3["evaluation_suites"]
    assert all("preference_target_mode" not in suite for suite in hard2_suites)
    assert [suite["preference_target_mode"] for suite in hard3_suites] == [
        "nominal",
        "dependency_feasible_projection",
    ]


def test_hard3_decay_range_contains_extreme_anchor() -> None:
    hard3 = load_config("pcrl_v0_hard3.json")
    controllability = hard3["controllability"]
    assert isinstance(controllability, dict)
    decay = float(controllability["assignment_priority_decay"])
    idealized_maximum_share = 1.0 / (1.0 + 3.0 * math.exp(-decay))

    assert idealized_maximum_share >= 0.70
    assert 1.0 / (1.0 + 3.0 * math.exp(-1.0)) < 0.70


def test_hard3_protocol_documents_exist() -> None:
    hard3 = load_config("pcrl_v0_hard3.json")
    documents = hard3["protocol_documents"]
    assert isinstance(documents, dict)
    assert set(documents) == {"baseline", "objectives", "acceptance"}
    for relative_path in documents.values():
        assert (WORKSPACE / str(relative_path)).is_file()


def test_moderate_profiles_are_separately_named_two_to_one_anchors() -> None:
    hard2_expected = {
        "search": (0.70, 0.10, 0.10, 0.10),
        "reconnaissance": (0.10, 0.70, 0.10, 0.10),
        "strike": (0.10, 0.10, 0.70, 0.10),
        "recovery": (0.10, 0.10, 0.10, 0.70),
    }
    for priority_index, (moderate_name, hard2_name) in enumerate(
        zip(MODERATE_PRIORITY_PROFILE_NAMES, hard2_expected, strict=True)
    ):
        moderate = task_preference_profile(moderate_name)
        assert moderate.sum() == pytest.approx(1.0)
        assert moderate[priority_index] == pytest.approx(0.40)
        assert all(
            value == pytest.approx(0.20)
            for index, value in enumerate(moderate)
            if index != priority_index
        )
        assert moderate[priority_index] / moderate[(priority_index + 1) % 4] == (
            pytest.approx(2.0)
        )
        assert task_preference_profile(hard2_name) == pytest.approx(
            hard2_expected[hard2_name]
        )


def test_hard3_config_and_oracle_select_moderate_calibration_set() -> None:
    hard2 = load_config("pcrl_v0.json")
    hard3 = load_config("pcrl_v0_hard3.json")

    assert tuple(
        list(hard2["training_anchor_profiles"])
        + list(hard2["held_out_profiles"])
    ) == HARD2_EVALUATION_PROFILE_NAMES
    assert hard3["calibration_profile_set"] == "hard3_moderate"
    assert tuple(hard3["calibration_profiles"]) == HARD3_CALIBRATION_PROFILE_NAMES
    assert PROFILE_SETS["hard2"] == HARD2_EVALUATION_PROFILE_NAMES
    assert resolve_profiles(None, "hard3_moderate") == (
        HARD3_CALIBRATION_PROFILE_NAMES
    )
    assert resolve_profiles(["balanced", "search"], "hard3_moderate") == (
        "balanced",
        "search",
    )
    assert protocol_evaluation_profiles(hard3) == list(
        HARD2_EVALUATION_PROFILE_NAMES
    )
    assert protocol_evaluation_profiles(hard3, use_calibration=True) == list(
        HARD3_CALIBRATION_PROFILE_NAMES
    )


def test_runner_rejects_missing_or_invalid_calibration_profiles() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        protocol_evaluation_profiles({}, use_calibration=True)
    with pytest.raises(ValueError, match="duplicate"):
        protocol_evaluation_profiles(
            {"calibration_profiles": ["balanced", "balanced"]},
            use_calibration=True,
        )
    with pytest.raises(ValueError, match="unknown task preference profile"):
        protocol_evaluation_profiles(
            {"calibration_profiles": ["not_a_profile"]},
            use_calibration=True,
        )


def test_runner_calibration_flag_routes_profiles_and_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_config("pcrl_v0_hard3.json")
    protocol["output_namespace"] = str(tmp_path / "hard3")
    protocol["methods"] = ["greedy_preference"]
    protocol["training_seeds"] = [1]
    protocol["evaluation_scales"] = ["2x12"]
    protocol["evaluation_episodes"] = 1
    protocol["evaluation_suites"] = [protocol["evaluation_suites"][0]]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    commands: list[list[str]] = []

    monkeypatch.setattr(
        run_pcrl_v0,
        "run",
        lambda command, *, environment: commands.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pcrl_v0.py",
            "--protocol",
            str(protocol_path),
            "--phase",
            "eval",
            "--calibration-profiles",
        ],
    )

    run_pcrl_v0.main()

    assert len(commands) == 1
    command = commands[0]
    start = command.index("--profiles") + 1
    end = command.index("--scales")
    assert tuple(command[start:end]) == HARD3_CALIBRATION_PROFILE_NAMES
    manifest_path = tmp_path / "hard3" / "calibration" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile_collection"] == "calibration_profiles"
    assert tuple(manifest["profiles"]) == HARD3_CALIBRATION_PROFILE_NAMES

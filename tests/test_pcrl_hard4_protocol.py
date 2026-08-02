from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uav_assignment.gppo_v2 import implementation_hash
from uav_assignment.pcrl_v0 import (
    CALIBRATION_PRIORITY_PROFILE_NAMES,
    calibration_priority_profiles,
)


WORKSPACE = Path(__file__).resolve().parents[1]
HARD2_CONFIG_SHA256 = (
    "597fcf012985e887f560f29a3e0b4fefe8a7c6381f982df60d99ab977e5cfaf7"
)
HARD3_CONFIG_SHA256 = (
    "6fada52cb3b6700a8a18d1038ac414804a4f62bd4f89fc08529e48c821ae3f2a"
)
FROZEN_GPPO_SHA256 = (
    "a62c17f721688e2ae0c36c6fe11ef1a6cced365c8468155bfacbf4f286ea01a6"
)


def load_config(name: str) -> dict[str, object]:
    return json.loads(
        (WORKSPACE / "configs" / name).read_text(encoding="utf-8")
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hard4_is_new_and_does_not_relabel_hard3() -> None:
    hard2 = load_config("pcrl_v0.json")
    hard3 = load_config("pcrl_v0_hard3.json")
    hard4 = load_config("pcrl_v0_hard4.json")

    assert hard2["version"] == "pcrl-v0-hard-2"
    assert hard3["version"] == "pcrl-v0-hard-3"
    assert hard4["version"] == "pcrl-v0-hard-4"
    assert hard4["protocol_kind"] == "separately_versioned_engineering_protocol"
    assert hard4["predecessor_protocol"] == hard3["version"]
    assert hard4["base_protocol"] == hard3["base_protocol"] == "gppo-v2-hard-3"
    assert hard4["output_namespace"] == "outputs/pcrl_v0/hard4"
    assert hard4["output_namespace"] != hard3["output_namespace"]
    assert hard4["evidence_boundary"]["hard3_artifacts_must_not_be_relabeled"] is True


def test_hard2_and_hard3_machine_readable_configs_are_unchanged() -> None:
    assert sha256(WORKSPACE / "configs" / "pcrl_v0.json") == HARD2_CONFIG_SHA256
    assert (
        sha256(WORKSPACE / "configs" / "pcrl_v0_hard3.json")
        == HARD3_CONFIG_SHA256
    )


def test_hard4_frozen_gppo_identity_is_unchanged() -> None:
    hard4 = load_config("pcrl_v0_hard4.json")
    assert hard4["frozen_gppo_implementation_hash"] == FROZEN_GPPO_SHA256
    assert implementation_hash() == FROZEN_GPPO_SHA256
    assert hard4["frozen_checkpoint_registry"] == (
        "configs/pcrl_v0_frozen_checkpoints.json"
    )


def test_hard4_profile_family_matches_dynamic_generator_exactly() -> None:
    hard4 = load_config("pcrl_v0_hard4.json")
    controllability = hard4["controllability"]
    family = hard4["preference_profile_family"]
    expected = calibration_priority_profiles(0.40)

    assert controllability["priority_share"] == pytest.approx(0.40)
    assert controllability["background_share"] == pytest.approx(0.20)
    assert controllability["assignment_priority_decay"] == pytest.approx(2.0)
    assert family["id"] == "dynamic-priority-share-v1"
    assert family["generator"] == "calibration_priority_profiles"
    assert family["priority_share"] == pytest.approx(0.40)
    assert family["background_share"] == pytest.approx(0.20)
    assert tuple(family["profile_names"]) == CALIBRATION_PRIORITY_PROFILE_NAMES
    assert tuple(family["profile_vectors"]) == CALIBRATION_PRIORITY_PROFILE_NAMES
    for name, vector in expected.items():
        assert family["profile_vectors"][name] == pytest.approx(vector)

    assert tuple(
        hard4["training_anchor_profiles"] + hard4["held_out_profiles"]
    ) == CALIBRATION_PRIORITY_PROFILE_NAMES
    assert len(hard4["training_anchor_profiles"]) == 5
    assert len(hard4["held_out_profiles"]) == 3


def test_hard4_pilot_and_formal_budgets_are_isolated_and_exact() -> None:
    hard4 = load_config("pcrl_v0_hard4.json")
    groups = hard4["artifact_groups"]
    pilot = groups["pilot20"]
    formal = groups["formal100"]

    assert pilot["output_subdirectory"] == "pilot20"
    assert pilot["training_seeds"] == [1, 2, 3, 4, 5]
    assert pilot["updates"] == 20
    assert pilot["episodes_per_update"] == 18
    assert pilot["evaluation_episodes"] == 20
    assert pilot["evaluation_seed_range"] == [76000, 76019]
    assert pilot["may_initialize_formal"] is False

    assert formal["output_subdirectory"] == "formal100"
    assert formal["training_seeds"] == [1, 2, 3, 4, 5]
    assert formal["updates"] == hard4["updates"] == 100
    assert formal["episodes_per_update"] == hard4["episodes_per_update"] == 18
    assert formal["evaluation_episodes"] == hard4["evaluation_episodes"] == 100
    assert formal["evaluation_seed_range"] == [50000, 50099]
    assert formal["requires_fresh_training"] is True
    assert formal["may_initialize_from_pilot"] is False
    assert pilot["output_subdirectory"] != formal["output_subdirectory"]


def test_hard4_run_identity_contract_is_complete() -> None:
    hard4 = load_config("pcrl_v0_hard4.json")
    required = set(hard4["run_identity"]["required_fields"])
    expected = {
        "protocol_version",
        "protocol_config_sha256",
        "artifact_group",
        "output_namespace",
        "scenario_config_sha256",
        "scenario_hash",
        "frozen_gppo_implementation_hash",
        "source_gppo_checkpoint_sha256",
        "pcrl_implementation_hash",
        "method_id",
        "graph_mode",
        "preference_conditioning",
        "training_seed",
        "evaluation_seed",
        "evaluation_suite",
        "scale",
        "task_release_mode",
        "deadline_scale",
        "preference_target_mode",
        "assignment_priority_decay",
        "preference_profile_family",
        "priority_share",
        "background_share",
        "preference_profile",
        "preference_profile_vector",
        "updates",
        "episodes_per_update",
        "evaluation_episodes",
    }
    assert expected <= required
    reject_if = set(hard4["run_identity"]["reject_if"])
    assert "hard2_or_hard3_rows_are_pooled_or_relabeled" in reject_if
    assert "pilot_and_formal_rows_are_pooled" in reject_if


def test_hard4_documents_exist_and_state_evidence_boundary() -> None:
    hard4 = load_config("pcrl_v0_hard4.json")
    documents = hard4["protocol_documents"]
    assert set(documents) == {"baseline", "objectives", "acceptance"}
    combined = ""
    for relative_path in documents.values():
        path = WORKSPACE / relative_path
        assert path.is_file()
        combined += path.read_text(encoding="utf-8").lower()
    assert "new engineering protocol" in combined
    assert "not a relabeling" in combined
    assert "training has not started" in combined

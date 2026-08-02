from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluate_pcrl_v0
import run_pcrl_v0
import train_pcrl_v0
from uav_assignment.pcrl_models import PreferenceConditionedPaperActorCritic
from uav_assignment.pcrl_training import model_conditioning_inputs
from uav_assignment.pcrl_v0 import (
    CALIBRATION_PRIORITY_PROFILE_NAMES,
    LEGACY_PREFERENCE_SAMPLER_MODE,
    MODERATE_PREFERENCE_SAMPLER_MODE,
    TASK_PREFERENCE_PROFILES,
    TRAINING_ANCHOR_NAMES,
    calibration_priority_profiles,
    map_task_preference_to_full,
    normalize_simplex,
    sample_training_task_preference,
    task_preference_profile,
)


def _historical_sample(rng: np.random.Generator) -> np.ndarray:
    continuous = rng.dirichlet(np.full(4, 0.7))
    if rng.random() < 0.30:
        anchor_name = str(rng.choice(TRAINING_ANCHOR_NAMES))
        continuous = 0.90 * TASK_PREFERENCE_PROFILES[anchor_name] + 0.10 * continuous
    return normalize_simplex(continuous)


def test_omitted_sampler_reproduces_historical_rng_sequence_exactly() -> None:
    expected_rng = np.random.default_rng(9182)
    default_rng = np.random.default_rng(9182)
    explicit_rng = np.random.default_rng(9182)
    for _ in range(64):
        expected = _historical_sample(expected_rng)
        assert sample_training_task_preference(default_rng) == pytest.approx(expected)
        assert sample_training_task_preference(
            explicit_rng, sampler_mode=LEGACY_PREFERENCE_SAMPLER_MODE
        ) == pytest.approx(expected)


def test_moderate_sampler_is_continuous_and_inside_selected_anchor_hull() -> None:
    rng = np.random.default_rng(41)
    samples = np.stack(
        [
            sample_training_task_preference(
                rng,
                sampler_mode=MODERATE_PREFERENCE_SAMPLER_MODE,
                calibration_priority_share=0.40,
            )
            for _ in range(1_000)
        ]
    )
    assert np.allclose(samples.sum(axis=1), 1.0)
    assert float(samples.min()) >= 0.20 - 1e-6
    assert float(samples.max()) <= 0.40 + 1e-6
    assert len(np.unique(np.round(samples, 8), axis=0)) == len(samples)
    old_extremes = np.stack(
        [TASK_PREFERENCE_PROFILES[name] for name in TRAINING_ANCHOR_NAMES[1:]]
    )
    assert not np.any(
        np.all(np.isclose(samples[:, None, :], old_extremes[None, :, :]), axis=2)
    )


def test_moderate_sampler_anchor_probability_creates_continuous_anchor_bias() -> None:
    anchors = np.stack(
        list(calibration_priority_profiles(0.40).values())[1:5]
    )

    def mean_nearest_distance(probability: float) -> float:
        rng = np.random.default_rng(10)
        samples = np.stack(
            [
                sample_training_task_preference(
                    rng,
                    sampler_mode=MODERATE_PREFERENCE_SAMPLER_MODE,
                    calibration_priority_share=0.40,
                    anchor_probability=probability,
                    anchor_jitter=0.10,
                )
                for _ in range(500)
            ]
        )
        assert len(np.unique(np.round(samples, 8), axis=0)) == len(samples)
        return float(
            np.linalg.norm(samples[:, None, :] - anchors[None, :, :], axis=2)
            .min(axis=1)
            .mean()
        )

    assert mean_nearest_distance(1.0) < 0.25 * mean_nearest_distance(0.0)


def test_moderate_sampler_is_seed_deterministic() -> None:
    kwargs = {
        "sampler_mode": MODERATE_PREFERENCE_SAMPLER_MODE,
        "calibration_priority_share": 0.40,
        "concentration": 1.2,
        "anchor_probability": 0.45,
        "anchor_jitter": 0.2,
    }
    left_rng = np.random.default_rng(77)
    right_rng = np.random.default_rng(77)
    other_rng = np.random.default_rng(78)
    left = np.stack([sample_training_task_preference(left_rng, **kwargs) for _ in range(20)])
    right = np.stack([sample_training_task_preference(right_rng, **kwargs) for _ in range(20)])
    other = np.stack([sample_training_task_preference(other_rng, **kwargs) for _ in range(20)])
    assert np.array_equal(left, right)
    assert not np.array_equal(left, other)


def _hard4_protocol(output: Path, methods: list[str]) -> dict[str, object]:
    return {
        "version": "pcrl-v0-hard-4",
        "output_namespace": str(output),
        "base_scenario_config": "configs/gppo_v2_hard.json",
        "training_seeds": [1],
        "methods": methods,
        "training_task_release_modes": ["phase_staggered"],
        "phase_staggered_deadline_scale": 0.7,
        "training_preference_sampler": {
            "mode": MODERATE_PREFERENCE_SAMPLER_MODE,
            "priority_share": 0.40,
            "concentration": 1.2,
            "anchor_probability": 0.45,
            "anchor_jitter": 0.20,
        },
        "evaluation_suites": [
            {
                "name": "controllability_phase",
                "task_release_mode": "phase_staggered",
                "deadline_scale": 0.7,
                "preference_target_mode": "nominal",
                "primary": True,
            }
        ],
        "controllability": {"assignment_priority_decay": 2.0},
        "training_anchor_profiles": list(CALIBRATION_PRIORITY_PROFILE_NAMES[:5]),
        "held_out_profiles": list(CALIBRATION_PRIORITY_PROFILE_NAMES[5:]),
        "evaluation_scales": ["2x12"],
        "evaluation_episodes": 1,
        "evaluation_seed": 50000,
        "artifact_groups": {
            "pilot20": {
                "output_subdirectory": "pilot20",
                "training_seeds": [1],
                "updates": 20,
                "episodes_per_update": 18,
                "evaluation_episodes": 20,
                "evaluation_seed": 76000,
            }
        },
        "acceptance": {"primary_suite": "controllability_phase"},
    }


@pytest.mark.parametrize(
    ("method", "fixed"),
    (("pcrl_gppo_adaptive", False), ("ls_gppo_adaptive_fixed_balanced", True)),
)
def test_runner_propagates_moderate_sampler_and_dynamic_validation_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    fixed: bool,
) -> None:
    protocol_path = tmp_path / f"{method}.json"
    output = tmp_path / f"out_{method}"
    protocol_path.write_text(
        json.dumps(_hard4_protocol(output, [method])), encoding="utf-8"
    )
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
            "train",
            "--artifact-group",
            "pilot20",
        ],
    )
    run_pcrl_v0.main()

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--preference-sampler-mode") + 1] == (
        MODERATE_PREFERENCE_SAMPLER_MODE
    )
    assert command[command.index("--calibration-priority-share") + 1] == "0.4"
    assert command[command.index("--artifact-group") + 1] == "pilot20"
    assert command[command.index("--updates") + 1] == "20"
    assert command[command.index("--episodes-per-update") + 1] == "18"
    validation = command[command.index("--validation-profiles") + 1 :]
    assert set(CALIBRATION_PRIORITY_PROFILE_NAMES).issubset(validation)
    assert ("--fixed-training-profile" in command) is fixed
    if fixed:
        assert command[command.index("--fixed-training-profile") + 1] == "balanced"

    manifest = json.loads(
        (output / "pilot20" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_group"] == "pilot20"
    assert manifest["updates"] == 20
    assert manifest["evaluation_episodes"] == 20
    assert manifest["evaluation_seed"] == 76000
    assert manifest["training_preference_sampler"]["mode"] == (
        MODERATE_PREFERENCE_SAMPLER_MODE
    )
    assert manifest["training_preference_sampler"]["calibration_priority_share"] == pytest.approx(0.4)
    assert tuple(manifest["training_preference_sampler_profile_vectors"]) == (
        CALIBRATION_PRIORITY_PROFILE_NAMES
    )


def test_hard4_smoke_records_effective_budget_and_isolates_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    official_namespace = tmp_path / "hard4_official"
    smoke_root = tmp_path / "hard4_smoke" / "seed_1"
    methods = [
        "pcrl_gppo_adaptive",
        "pcrl_gppo_adaptive_no_conditioning",
        "gppo_event",
    ]
    protocol_path = tmp_path / "hard4.json"
    protocol_path.write_text(
        json.dumps(_hard4_protocol(official_namespace, methods)),
        encoding="utf-8",
    )
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
            "--artifact-group",
            "pilot20",
            "--phase",
            "all",
            "--methods",
            *methods,
            "--seeds",
            "1",
            "--smoke",
            "--output-root",
            str(smoke_root),
        ],
    )

    run_pcrl_v0.main()

    manifest = json.loads(
        (smoke_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["smoke"] is True
    assert manifest["updates"] == 2
    assert manifest["episodes_per_update"] == 2
    assert manifest["evaluation_episodes"] == 2
    assert manifest["protocol_budget"] == {
        "updates": 20,
        "episodes_per_update": 18,
        "evaluation_episodes": 20,
        "evaluation_seed": 76000,
    }
    assert not (official_namespace / "pilot20").exists()
    assert not (official_namespace / "formal100").exists()

    smoke_root_resolved = smoke_root.resolve()
    train_commands = [
        command for command in commands if Path(command[1]).name == "train_pcrl_v0.py"
    ]
    evaluation_commands = [
        command
        for command in commands
        if Path(command[1]).name == "evaluate_pcrl_v0.py"
    ]
    assert len(train_commands) == 2
    assert len(evaluation_commands) == 3
    for command in train_commands:
        assert command.count("--updates") == 1
        assert command[command.index("--updates") + 1] == "2"
        assert command.count("--episodes-per-update") == 1
        assert command[command.index("--episodes-per-update") + 1] == "2"
        output = Path(command[command.index("--output") + 1]).resolve()
        assert output.is_relative_to(smoke_root_resolved)
        frozen_source = Path(
            command[command.index("--init-checkpoint") + 1]
        )
        assert frozen_source == run_pcrl_v0.checkpoint_path("gppo_event", 1)

    for command in evaluation_commands:
        assert command[command.index("--episodes") + 1] == "2"
        output = Path(command[command.index("--output") + 1]).resolve()
        assert output.is_relative_to(smoke_root_resolved)
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        if checkpoint == run_pcrl_v0.checkpoint_path("gppo_event", 1):
            continue
        assert checkpoint.resolve().is_relative_to(smoke_root_resolved)


def test_old_protocol_without_sampler_resolves_to_exact_legacy_defaults() -> None:
    sampler = run_pcrl_v0.protocol_preference_sampler({})
    assert sampler == {
        "mode": LEGACY_PREFERENCE_SAMPLER_MODE,
        "calibration_priority_share": None,
        "concentration": 0.7,
        "anchor_probability": 0.30,
        "anchor_jitter": 0.10,
    }


def test_hard4_config_selects_moderate_sampler_and_requires_artifact_group() -> None:
    protocol = json.loads(
        Path("configs/pcrl_v0_hard4.json").read_text(encoding="utf-8")
    )
    sampler = run_pcrl_v0.protocol_preference_sampler(protocol)
    assert sampler["mode"] == MODERATE_PREFERENCE_SAMPLER_MODE
    assert sampler["calibration_priority_share"] == pytest.approx(0.40)
    with pytest.raises(ValueError, match="explicit --artifact-group"):
        run_pcrl_v0.protocol_artifact_group(protocol, None)
    name, group = run_pcrl_v0.protocol_artifact_group(protocol, "pilot20")
    assert name == "pilot20"
    assert group is not None
    assert group["updates"] == 20
    assert group["evaluation_episodes"] == 20


def test_training_checkpoint_records_sampler_and_dynamic_profile_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "train"
    monkeypatch.setattr(
        PreferenceConditionedPaperActorCritic,
        "initialize_from_gppo_checkpoint",
        lambda self, path: {"path": str(path)},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_pcrl_v0.py",
            "--init-checkpoint",
            str(tmp_path / "unused.pt"),
            "--graph-mode",
            "adaptive",
            "--updates",
            "0",
            "--validation-episodes",
            "0",
            "--preference-sampler-mode",
            MODERATE_PREFERENCE_SAMPLER_MODE,
            "--calibration-priority-share",
            "0.4",
            "--validation-profiles",
            *CALIBRATION_PRIORITY_PROFILE_NAMES,
            "--output",
            str(output),
        ],
    )
    train_pcrl_v0.main()
    checkpoint = torch.load(
        output / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["preference_sampler"]["mode"] == MODERATE_PREFERENCE_SAMPLER_MODE
    assert checkpoint["preference_sampler"]["calibration_priority_share"] == pytest.approx(0.4)
    assert tuple(checkpoint["preference_sampler_profile_vectors"]) == (
        CALIBRATION_PRIORITY_PROFILE_NAMES
    )
    assert checkpoint["fixed_training_preference"] is None


def test_no_conditioning_boundary_receives_neutral_preference_and_no_deficit() -> None:
    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=2,
        max_tasks=4,
        hidden_dim=16,
        graph_mode="adaptive",
        preference_conditioning=False,
    )
    observation = {
        "nodes": torch.zeros(6, 24),
        "edge_types": torch.zeros(6, 6, dtype=torch.long),
        "edge_features": torch.zeros(6, 6, 5),
        "action_mask": torch.ones(9, dtype=torch.bool),
        "preference_deficit": torch.tensor((0.4, -0.2, -0.1, -0.1)),
    }
    actual = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("search"))
    )
    sanitized, neutral = model_conditioning_inputs(model, observation, actual)
    assert "preference_deficit" not in sanitized
    assert neutral == pytest.approx(model.default_preference)
    assert not torch.allclose(neutral, actual)


def test_runner_passes_hard4_pilot_identity_to_frozen_gppo_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "hard4"
    protocol_path = tmp_path / "hard4.json"
    protocol_path.write_text(
        json.dumps(_hard4_protocol(output, ["gppo_event"])), encoding="utf-8"
    )
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
            "--artifact-group",
            "pilot20",
        ],
    )
    run_pcrl_v0.main()
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--artifact-group") + 1] == "pilot20"
    assert command[command.index("--preference-profile-family") + 1] == (
        "dynamic-priority-share-v1"
    )
    assert command[command.index("--evaluation-suite") + 1] == (
        "controllability_phase"
    )
    assert command[command.index("--episodes") + 1] == "20"
    assert command[command.index("--eval-seed") + 1] == "76000"
    assert "--calibration-priority-share" in command


def test_evaluator_rows_include_complete_hard4_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "identity_eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pcrl_v0.py",
            "--baseline",
            "random",
            "--protocol-version",
            "pcrl-v0-hard-4",
            "--protocol-config",
            "configs/pcrl_v0_hard4.json",
            "--artifact-group",
            "pilot20",
            "--preference-profile-family",
            "dynamic-priority-share-v1",
            "--evaluation-suite",
            "controllability_phase",
            "--calibration-priority-share",
            "0.4",
            "--scales",
            "2x12",
            "--episodes",
            "1",
            "--task-release-mode",
            "phase_staggered",
            "--deadline-scale",
            "0.7",
            "--assignment-priority-decay",
            "2.0",
            "--preference-target-mode",
            "nominal",
            "--output",
            str(output),
        ],
    )
    evaluate_pcrl_v0.main()
    rows = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert len(rows) == len(CALIBRATION_PRIORITY_PROFILE_NAMES)
    for row in rows:
        assert row["artifact_group"] == "pilot20"
        assert row["preference_profile_family"] == "dynamic-priority-share-v1"
        assert row["evaluation_suite"] == "controllability_phase"
        assert row["preference_conditioning"] is False
        assert row["updates"] == -1
        assert row["episodes_per_update"] == -1
        assert row["evaluation_episodes"] == 1
        assert row["priority_share"] == pytest.approx(0.4)
        assert row["background_share"] == pytest.approx(0.2)
        assert row["preference_profile_vector"] == pytest.approx(
            row["calibration_profile_vector"]
        )
        assert row["evaluation_seed"] == row["eval_seed"]
        for field in (
            "protocol_config_sha256",
            "scenario_config_sha256",
            "frozen_gppo_implementation_hash",
            "pcrl_implementation_hash",
        ):
            assert len(row[field]) == 64
        assert row["output_namespace"] == "outputs/pcrl_v0/hard4"
        assert row["base_protocol"] == "gppo-v2-hard-3"


def test_evaluator_rejects_learned_checkpoint_artifact_group_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = json.loads(
        Path("configs/gppo_v2_hard.json").read_text(encoding="utf-8")
    )
    payload = {
        "protocol_version": "pcrl-v0-hard-4",
        "artifact_group": "formal100",
        "env_config": scenario["scenario"],
        "scenario": scenario["version"],
        "training": {"seed": 1, "updates": 100, "episodes_per_update": 18},
        "preference_conditioning": True,
    }
    monkeypatch.setattr(
        evaluate_pcrl_v0,
        "load_model",
        lambda path: (object(), payload, "pcrl"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pcrl_v0.py",
            "--checkpoint",
            str(tmp_path / "fake.pt"),
            "--protocol-version",
            "pcrl-v0-hard-4",
            "--protocol-config",
            "configs/pcrl_v0_hard4.json",
            "--artifact-group",
            "pilot20",
            "--preference-profile-family",
            "dynamic-priority-share-v1",
            "--evaluation-suite",
            "controllability_phase",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="artifact_group"):
        evaluate_pcrl_v0.main()


def test_evaluator_no_conditioning_removes_deficit_and_uses_neutral_preference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = json.loads(
        Path("configs/gppo_v2_hard.json").read_text(encoding="utf-8")
    )

    class NoConditioningPolicy:
        preference_conditioning = False
        default_preference = torch.as_tensor(
            map_task_preference_to_full(task_preference_profile("balanced"))
        )

        def act(self, observation, preference, *, deterministic=False):
            assert "preference_deficit" not in observation
            assert preference == pytest.approx(self.default_preference)
            valid = torch.nonzero(
                observation["action_mask"], as_tuple=False
            ).reshape(-1)
            action = valid[0]
            return action, torch.tensor(0.0), torch.zeros(7)

    payload = {
        "protocol_version": "pcrl-v0-hard-4",
        "artifact_group": "pilot20",
        "env_config": scenario["scenario"],
        "scenario": scenario["version"],
        "method_id": "pcrl_gppo_adaptive_no_conditioning",
        "algorithm": "preco",
        "graph_mode": "adaptive",
        "training": {"seed": 1, "updates": 20, "episodes_per_update": 18},
        "preference_conditioning": False,
        "model_config": {"preference_conditioning": False},
        "source_gppo": {"checkpoint_sha256": "5" * 64},
    }
    monkeypatch.setattr(
        evaluate_pcrl_v0,
        "load_model",
        lambda path: (NoConditioningPolicy(), payload, "pcrl"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pcrl_v0.py",
            "--checkpoint",
            str(tmp_path / "fake.pt"),
            "--protocol-config",
            "configs/pcrl_v0_hard4.json",
            "--protocol-version",
            "pcrl-v0-hard-4",
            "--artifact-group",
            "pilot20",
            "--preference-profile-family",
            "dynamic-priority-share-v1",
            "--evaluation-suite",
            "controllability_phase",
            "--profiles",
            "balanced",
            "--scales",
            "2x12",
            "--episodes",
            "1",
            "--output",
            str(tmp_path / "no_conditioning_eval"),
        ],
    )
    evaluate_pcrl_v0.main()

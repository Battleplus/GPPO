from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import run_pcrl_v0
from train_pcrl_v0 import constrained_selection_diagnostics
from uav_assignment.gppo_v2 import implementation_hash
from uav_assignment.paper_env import PaperEnvConfig
from uav_assignment.pcrl_models import (
    MASS_CONTROL_PREFERENCE_INPUT_MODE,
    PreferenceConditionedPaperActorCritic,
)
from uav_assignment.pcrl_training import collect_pcrl_rollouts
from uav_assignment.pcrl_v0 import (
    BALANCED_CAPABILITY_COVERAGE_MODE,
    map_task_preference_to_full,
)


def hard6() -> dict[str, object]:
    return json.loads(
        Path("configs/pcrl_v0_hard6.json").read_text(encoding="utf-8")
    )


def synthetic_observation() -> dict[str, torch.Tensor]:
    max_uavs, max_tasks = 2, 4
    nodes = torch.zeros(max_uavs + max_tasks, 24)
    nodes[:, 2] = 1.0
    for task_type in range(4):
        nodes[max_uavs + task_type, 16 + task_type] = 1.0
    action_mask = torch.zeros(max_uavs * max_tasks + 1, dtype=torch.bool)
    action_mask[:max_tasks] = True
    action_mask[-1] = True
    return {
        "nodes": nodes,
        "edge_types": torch.zeros(max_uavs + max_tasks, max_uavs + max_tasks, dtype=torch.long),
        "edge_features": torch.zeros(max_uavs + max_tasks, max_uavs + max_tasks, 5),
        "action_mask": action_mask,
        "preference_deficit": torch.tensor([0.4, 0.2, 0.2, 0.2]),
    }


def task_group_mass(
    probabilities: torch.Tensor, nodes: torch.Tensor, *, max_uavs: int, max_tasks: int
) -> torch.Tensor:
    probabilities = probabilities.squeeze(0)
    pair = probabilities[:-1].reshape(max_uavs, max_tasks)
    task_types = nodes[max_uavs : max_uavs + max_tasks, 16:20]
    return torch.einsum("ut,ti->i", pair, task_types)


def test_hard6_is_separate_and_keeps_world_model_locked() -> None:
    protocol = hard6()
    assert protocol["version"] == "pcrl-v0-hard-6"
    assert protocol["predecessor_protocol"] == "pcrl-v0-hard-5"
    assert protocol["output_namespace"] == "outputs/pcrl_v0/hard6"
    assert protocol["frozen_gppo_implementation_hash"] == implementation_hash()
    assert protocol["preference_input_mode"] == MASS_CONTROL_PREFERENCE_INPUT_MODE
    assert protocol["acceptance"]["preference_error_reduction_vs_no_conditioning"] == 0.30
    assert protocol["evidence_boundary"]["world_model_and_jepa_out_of_scope"] is True


def test_mass_controller_is_monotonic_and_preserves_action_mask() -> None:
    observation = synthetic_observation()
    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=2,
        max_tasks=4,
        hidden_dim=16,
        graph_mode="none",
        preference_input_mode=MASS_CONTROL_PREFERENCE_INPUT_MODE,
    )
    model.task_preference_gain.data.fill_(20.0)
    search = torch.tensor([0.4, 0.2, 0.2, 0.2])
    strike = torch.tensor([0.2, 0.2, 0.4, 0.2])
    search_dist, _ = model(
        **observation,
        preference=torch.as_tensor(map_task_preference_to_full(search)),
        task_preference=search,
    )
    strike_observation = dict(observation)
    strike_observation["preference_deficit"] = strike
    strike_dist, _ = model(
        **strike_observation,
        preference=torch.as_tensor(map_task_preference_to_full(strike)),
        task_preference=strike,
    )
    search_mass = task_group_mass(
        search_dist.probs, observation["nodes"], max_uavs=2, max_tasks=4
    )
    strike_mass = task_group_mass(
        strike_dist.probs, observation["nodes"], max_uavs=2, max_tasks=4
    )
    assert search_mass[0] > strike_mass[0]
    assert strike_mass[2] > search_mass[2]
    assert torch.all(search_dist.probs.squeeze(0)[~observation["action_mask"]] == 0)
    assert torch.all(strike_dist.probs.squeeze(0)[~observation["action_mask"]] == 0)


def test_zero_alpha_preserves_precalibration_policy_across_preferences() -> None:
    observation = synthetic_observation()
    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=2,
        max_tasks=4,
        hidden_dim=16,
        graph_mode="none",
        preference_input_mode=MASS_CONTROL_PREFERENCE_INPUT_MODE,
    )
    model.task_preference_gain.data.fill_(-20.0)
    first = torch.tensor([0.4, 0.2, 0.2, 0.2])
    second = torch.tensor([0.2, 0.2, 0.4, 0.2])
    first_dist, _ = model(
        **observation,
        preference=torch.as_tensor(map_task_preference_to_full(first)),
        task_preference=first,
    )
    changed = dict(observation)
    changed["preference_deficit"] = second
    second_dist, _ = model(
        **changed,
        preference=torch.as_tensor(map_task_preference_to_full(second)),
        task_preference=second,
    )
    assert torch.allclose(first_dist.probs, second_dist.probs, atol=1e-6)


def test_rollout_reuses_one_preference_across_complete_trajectories() -> None:
    config = PaperEnvConfig(
        max_uavs=3,
        max_tasks=8,
        active_uavs=3,
        initial_tasks=8,
        max_decisions=40,
        mission_deadline=12.0,
        task_chain_length=4,
        weather_probability=0.0,
        failure_probability=0.0,
        task_change_probability=0.0,
        communication_drop_probability=0.0,
        seed=5,
    )
    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=3,
        max_tasks=8,
        hidden_dim=16,
        graph_mode="none",
        preference_input_mode=MASS_CONTROL_PREFERENCE_INPUT_MODE,
    )
    batch = collect_pcrl_rollouts(
        model,
        config,
        episodes=4,
        seed_offset=0,
        sync_mode="event",
        gamma=0.99,
        gae_lambda=0.95,
        capability_coverage_mode=BALANCED_CAPABILITY_COVERAGE_MODE,
        episodes_per_preference=4,
    )
    assert torch.unique(batch.preference_ids).numel() == 1
    assert batch.trajectory_ids is not None
    assert torch.unique(batch.trajectory_ids).numel() == 4
    assert torch.unique(batch.task_preferences, dim=0).shape[0] == 1


def test_relative_invalid_action_guard_uses_gppo_delta() -> None:
    reference = {
        "preference_l1": 0.30,
        "deadline_completion_rate": 0.70,
        "makespan": 20.0,
        "minimum_task_coverage": 0.60,
        "invalid_actions": 0.35,
    }
    candidate = {
        "preference_l1": 0.20,
        "deadline_completion_rate": 0.70,
        "makespan": 20.0,
        "minimum_task_coverage": 0.60,
        "invalid_actions": 0.38,
    }
    diagnostics = constrained_selection_diagnostics(
        candidate,
        reference,
        max_deadline_completion_drop=0.03,
        max_makespan_increase_ratio=0.05,
        max_coverage_loss=0.05,
        max_invalid_actions=0.0,
        max_invalid_action_increase=0.05,
    )
    assert diagnostics["feasible"] is True
    assert diagnostics["invalid_action_delta"] == pytest.approx(0.03)


def test_runner_keeps_held_out_profiles_out_of_checkpoint_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def capture(command: list[str], *, environment: dict[str, str]) -> None:
        assert environment
        commands.append(command)

    monkeypatch.setattr(run_pcrl_v0, "run", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pcrl_v0.py",
            "--protocol",
            "configs/pcrl_v0_hard6.json",
            "--artifact-group",
            "calibration20",
            "--phase",
            "train",
            "--seeds",
            "11",
            "--methods",
            "pcrl_gppo_adaptive",
            "--output-root",
            str(tmp_path / "hard6"),
        ],
    )
    run_pcrl_v0.main()
    command = next(item for item in commands if "train_pcrl_v0.py" in item)
    start = command.index("--validation-profiles") + 1
    end = start
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    selected_profiles = command[start:end]
    protocol = hard6()
    assert selected_profiles == protocol["training_anchor_profiles"]
    assert not set(selected_profiles) & set(protocol["held_out_profiles"])
    assert "--save-validation-candidates" in command
    assert "--freeze-backbone" in command
    assert command[command.index("--checkpoint-selection-mode") + 1] == (
        "gppo_efficiency_relative_v2"
    )

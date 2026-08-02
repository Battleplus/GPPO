from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import run_pcrl_v0
from train_pcrl_v0 import (
    checkpoint_selection_key,
    constrained_selection_diagnostics,
)
from uav_assignment.gppo_v2 import config_hash, implementation_hash
from uav_assignment.paper_env import PaperAlignedUAVEnv, PaperEnvConfig
from uav_assignment.pcrl_models import (
    SPLIT_PREFERENCE_INPUT_MODE,
    PreferenceConditionedPaperActorCritic,
)
from uav_assignment.pcrl_v0 import (
    BALANCED_CAPABILITY_COVERAGE_MODE,
    LEGACY_CAPABILITY_COVERAGE_MODE,
    PreferencePaperEnv,
    map_task_preference_to_full,
    pcrl_scenario_hash,
    task_preference_profile,
)


def hard5() -> dict[str, object]:
    return json.loads(
        Path("configs/pcrl_v0_hard5.json").read_text(encoding="utf-8")
    )


def small_config(*, active_uavs: int = 3) -> PaperEnvConfig:
    return PaperEnvConfig(
        max_uavs=4,
        max_tasks=8,
        active_uavs=active_uavs,
        initial_tasks=8,
        max_decisions=20,
        mission_deadline=12.0,
        task_chain_length=4,
        seed=7,
    )


def tensor_observation(env: PreferencePaperEnv) -> dict[str, torch.Tensor]:
    observation = env.reset(seed=123)
    return {
        "nodes": torch.as_tensor(observation["nodes"], dtype=torch.float32),
        "edge_types": torch.as_tensor(observation["edge_types"], dtype=torch.long),
        "edge_features": torch.as_tensor(
            observation["edge_features"], dtype=torch.float32
        ),
        "action_mask": torch.as_tensor(observation["action_mask"], dtype=torch.bool),
        "preference_deficit": torch.as_tensor(
            observation["preference_deficit"], dtype=torch.float32
        ),
    }


def test_hard5_is_separate_and_keeps_frozen_gppo_identity() -> None:
    protocol = hard5()
    hard4_protocol = json.loads(
        Path("configs/pcrl_v0_hard4.json").read_text(encoding="utf-8")
    )
    assert protocol["version"] == "pcrl-v0-hard-5"
    assert protocol["predecessor_protocol"] == "pcrl-v0-hard-4"
    assert protocol["output_namespace"] == "outputs/pcrl_v0/hard5"
    assert hard4_protocol["version"] == "pcrl-v0-hard-4"
    assert protocol["frozen_gppo_implementation_hash"] == implementation_hash()
    assert protocol["acceptance"]["preference_error_reduction_vs_no_conditioning"] == 0.30
    assert protocol["evidence_boundary"]["world_model_and_jepa_out_of_scope"] is True


@pytest.mark.parametrize("active_uavs", (1, 2, 3, 4))
def test_balanced_capability_mode_has_exact_cyclic_min2_coverage(
    active_uavs: int,
) -> None:
    config = small_config(active_uavs=active_uavs)
    for seed in range(5):
        env = PreferencePaperEnv(
            config,
            capability_coverage_mode=BALANCED_CAPABILITY_COVERAGE_MODE,
        )
        env.reset(seed=seed)
        capabilities = np.stack(
            [uav.capabilities for uav in env.uavs[:active_uavs]], axis=0
        )
        capable = capabilities >= config.capability_threshold
        assert capable.sum(axis=0).tolist() == [min(2, active_uavs)] * 4
        for task_type in range(4):
            expected = {
                task_type % active_uavs,
                (task_type + 1) % active_uavs,
            }
            assert set(np.flatnonzero(capable[:, task_type])) == expected


def test_legacy_wrapper_preserves_parent_capabilities_exactly() -> None:
    config = small_config(active_uavs=3)
    parent = PaperAlignedUAVEnv(config)
    wrapper = PreferencePaperEnv(
        config, capability_coverage_mode=LEGACY_CAPABILITY_COVERAGE_MODE
    )
    parent.reset(seed=99)
    wrapper.reset(seed=99)
    assert np.array_equal(
        np.stack([uav.capabilities for uav in parent.uavs]),
        np.stack([uav.capabilities for uav in wrapper.uavs]),
    )


def test_split_preference_inputs_isolate_actor_and_critic() -> None:
    env = PreferencePaperEnv(
        small_config(active_uavs=3),
        task_release_mode="phase_staggered",
        capability_coverage_mode=BALANCED_CAPABILITY_COVERAGE_MODE,
    )
    observation = tensor_observation(env)
    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=4,
        max_tasks=8,
        hidden_dim=16,
        graph_mode="adaptive",
        preference_input_mode=SPLIT_PREFERENCE_INPUT_MODE,
    )
    model.task_preference_gain.data.fill_(2.0)
    search_task = torch.as_tensor(task_preference_profile("search"))
    strike_task = torch.as_tensor(task_preference_profile("strike"))
    fixed_objective = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("balanced"))
    )
    search_dist, search_value = model(
        **observation,
        preference=fixed_objective,
        task_preference=search_task,
    )
    strike_dist, strike_value = model(
        **observation,
        preference=fixed_objective,
        task_preference=strike_task,
    )
    assert not torch.allclose(search_dist.logits, strike_dist.logits)
    assert torch.allclose(search_value, strike_value)

    torch.nn.init.constant_(model.vector_critic_residual[-1].weight, 0.1)
    search_objective = torch.as_tensor(map_task_preference_to_full(search_task))
    strike_objective = torch.as_tensor(map_task_preference_to_full(strike_task))
    _, value_a = model(
        **observation,
        preference=search_objective,
        task_preference=search_task,
    )
    _, value_b = model(
        **observation,
        preference=strike_objective,
        task_preference=search_task,
    )
    assert not torch.allclose(value_a, value_b)


def test_constrained_selector_rejects_lower_l1_when_makespan_guard_fails() -> None:
    reference = {
        "preference_l1": 0.30,
        "deadline_completion_rate": 0.70,
        "makespan": 20.0,
        "minimum_task_coverage": 0.60,
        "invalid_actions": 0.0,
    }
    feasible = {
        "preference_l1": 0.24,
        "deadline_completion_rate": 0.69,
        "makespan": 20.5,
        "minimum_task_coverage": 0.59,
        "invalid_actions": 0.0,
    }
    lower_l1_but_infeasible = {
        "preference_l1": 0.20,
        "deadline_completion_rate": 0.69,
        "makespan": 22.0,
        "minimum_task_coverage": 0.59,
        "invalid_actions": 0.0,
    }
    kwargs = {
        "max_deadline_completion_drop": 0.03,
        "max_makespan_increase_ratio": 0.05,
        "max_coverage_loss": 0.05,
        "max_invalid_actions": 0.0,
    }
    feasible_diag = constrained_selection_diagnostics(
        feasible, reference, **kwargs
    )
    infeasible_diag = constrained_selection_diagnostics(
        lower_l1_but_infeasible, reference, **kwargs
    )
    assert feasible_diag["feasible"] is True
    assert infeasible_diag["feasible"] is False
    assert "makespan_increase" in infeasible_diag["reasons"]
    assert checkpoint_selection_key(
        feasible, feasible_diag, update=4
    ) > checkpoint_selection_key(
        lower_l1_but_infeasible, infeasible_diag, update=6
    )


def test_pcrl_scenario_hash_adds_wrapper_mode_without_changing_legacy_hash() -> None:
    config = small_config()
    assert pcrl_scenario_hash(config, LEGACY_CAPABILITY_COVERAGE_MODE) == config_hash(
        config
    )
    assert pcrl_scenario_hash(
        config, BALANCED_CAPABILITY_COVERAGE_MODE
    ) != config_hash(config)


def test_runner_materializes_preregistered_calibration_variant(
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
            "configs/pcrl_v0_hard5.json",
            "--artifact-group",
            "calibration10",
            "--calibration-variant",
            "A_raw_preference_only",
            "--phase",
            "train",
            "--seeds",
            "11",
            "--output-root",
            str(tmp_path / "variant_a"),
        ],
    )
    run_pcrl_v0.main()
    learned = [command for command in commands if "train_pcrl_v0.py" in command]
    assert len(learned) == 2
    command = learned[0]
    assert command[command.index("--init-checkpoint") + 1].endswith(
        "gppo_event\\seed_1\\checkpoint.pt"
    )
    assert command[command.index("--capability-coverage-mode") + 1] == "legacy"
    assert command[command.index("--preference-input-mode") + 1] == (
        "split_task_objective_v2"
    )
    assert command[command.index("--checkpoint-selection-mode") + 1] == (
        "legacy_l1_lexicographic"
    )
    assert command[command.index("--protocol-version") + 1].endswith("-cal-a")


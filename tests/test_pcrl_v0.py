from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluate_pcrl_v0
import run_pcrl_v0
from run_pcrl_v0 import protocol_output_root

from uav_assignment.gppo_v2 import implementation_hash
from uav_assignment.paper_env import PaperAlignedUAVEnv, PaperEnvConfig
from uav_assignment.paper_models import PaperHeteroActorCritic
from uav_assignment.pcrl_models import (
    FROZEN_GPPO_IMPLEMENTATION_HASH,
    PreferenceConditionedPaperActorCritic,
)
from uav_assignment.pcrl_training import (
    collect_pcrl_rollouts,
    grouped_policy_direction,
    preference_action_alignment_loss,
    tensor_observation,
    vector_gae,
)
from uav_assignment.pcrl_v0 import (
    CALIBRATION_PRIORITY_PROFILE_NAMES,
    DEFAULT_ASSIGNMENT_PRIORITY_DECAY,
    DEFAULT_PREFERENCE_TARGET_MODE,
    N_OBJECTIVES,
    PreferencePaperEnv,
    assignment_priority_weight,
    calibration_priority_profiles,
    map_task_preference_to_full,
    preference_allocation_target,
    project_dependency_feasible_preference,
    task_preference_profile,
)


def small_config(**overrides: object) -> PaperEnvConfig:
    values: dict[str, object] = {
        "max_uavs": 3,
        "max_tasks": 8,
        "active_uavs": 3,
        "initial_tasks": 8,
        "max_decisions": 60,
        "mission_deadline": 20.0,
        "task_chain_length": 4,
        "weather_probability": 0.0,
        "failure_probability": 0.0,
        "task_change_probability": 0.0,
        "communication_drop_probability": 0.0,
        "include_engineering_rewards": False,
        "seed": 3,
    }
    values.update(overrides)
    return PaperEnvConfig(**values)


def make_model(config: PaperEnvConfig, graph_mode: str = "adaptive"):
    return PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=config.max_uavs,
        max_tasks=config.max_tasks,
        hidden_dim=32,
        graph_mode=graph_mode,
    )


@pytest.mark.parametrize("priority_share", (0.35, 0.40, 0.50))
def test_dynamic_calibration_profiles_are_convex_and_stably_named(
    priority_share: float,
) -> None:
    profiles = calibration_priority_profiles(priority_share)
    background = (1.0 - priority_share) / 3.0

    assert tuple(profiles) == CALIBRATION_PRIORITY_PROFILE_NAMES
    assert profiles["balanced"] == pytest.approx((0.25,) * 4)
    anchor_names = CALIBRATION_PRIORITY_PROFILE_NAMES[1:5]
    for priority_index, name in enumerate(anchor_names):
        vector = profiles[name]
        assert float(vector.sum()) == pytest.approx(1.0)
        assert vector[priority_index] == pytest.approx(priority_share)
        assert all(
            value == pytest.approx(background)
            for index, value in enumerate(vector)
            if index != priority_index
        )

    search, reconnaissance, strike, recovery = (
        profiles[name] for name in anchor_names
    )
    assert profiles["calibration_search_strike_interp"] == pytest.approx(
        0.5 * search + 0.5 * strike
    )
    assert profiles["calibration_recon_recovery_interp"] == pytest.approx(
        0.5 * reconnaissance + 0.5 * recovery
    )
    assert profiles[
        "calibration_asymmetric_search_recon_recovery"
    ] == pytest.approx(
        0.5 * search + 0.3 * reconnaissance + 0.2 * recovery
    )


@pytest.mark.parametrize("priority_share", (0.25, 1.0, float("nan")))
def test_dynamic_calibration_profiles_reject_invalid_share(
    priority_share: float,
) -> None:
    with pytest.raises(ValueError, match="priority_share"):
        calibration_priority_profiles(priority_share)


def test_dynamic_calibration_cli_overrides_named_profiles_and_records_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "dynamic_eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pcrl_v0.py",
            "--baseline",
            "random",
            "--profiles",
            "search",
            "--calibration-priority-share",
            "0.4",
            "--scales",
            "2x12",
            "--episodes",
            "1",
            "--output",
            str(output),
        ],
    )

    evaluate_pcrl_v0.main()

    rows = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    expected = calibration_priority_profiles(0.4)
    assert len(rows) == len(expected)
    assert tuple(row["preference_profile"] for row in rows) == tuple(expected)
    for row in rows:
        name = row["preference_profile"]
        assert row["calibration_priority_share"] == pytest.approx(0.4)
        assert row["calibration_profile_vector"] == pytest.approx(expected[name])


def test_preference_mapping_has_fixed_constraints_and_task_floor() -> None:
    full = map_task_preference_to_full((1.0, 0.0, 0.0, 0.0))
    assert full.shape == (N_OBJECTIVES,)
    assert float(full.sum()) == pytest.approx(1.0)
    assert np.all(full[:4] > 0.0)
    assert full[:4].sum() == pytest.approx(0.70)
    assert full[4:].tolist() == pytest.approx([0.15, 0.05, 0.10])


def test_dependency_projection_enforces_lifecycle_feasibility() -> None:
    projected = project_dependency_feasible_preference(
        task_preference_profile("recovery")
    )
    assert float(projected.sum()) == pytest.approx(1.0)
    assert projected[1] <= 2.0 * projected[0] + 1e-6
    assert projected[2] <= projected[1] + 1e-6
    assert projected[3] <= projected[2] + 1e-6
    assert not np.allclose(projected, task_preference_profile("recovery"))


def test_preference_target_modes_preserve_hard2_default_and_allow_nominal() -> None:
    nominal = task_preference_profile("recovery")
    assert DEFAULT_PREFERENCE_TARGET_MODE == "dependency_feasible_projection"
    assert preference_allocation_target(nominal) == pytest.approx(
        project_dependency_feasible_preference(nominal)
    )
    assert preference_allocation_target(nominal, mode="nominal") == pytest.approx(
        nominal
    )
    default_full = map_task_preference_to_full(nominal)
    nominal_full = map_task_preference_to_full(
        nominal, preference_target_mode="nominal"
    )
    assert not np.allclose(default_full[:4], nominal_full[:4])
    with pytest.raises(ValueError, match="target mode"):
        preference_allocation_target(nominal, mode="unknown")


def test_environment_reports_selected_target_and_both_diagnostics() -> None:
    nominal = task_preference_profile("recovery")
    hard2_env = PreferencePaperEnv(small_config(), preference=nominal)
    nominal_env = PreferencePaperEnv(
        small_config(),
        preference=nominal,
        preference_target_mode="nominal",
    )
    hard2_env.reset(seed=11)
    nominal_env.reset(seed=11)
    hard2_metrics = hard2_env.metrics()
    nominal_metrics = nominal_env.metrics()
    assert hard2_metrics["preference_target_mode"] == (
        "dependency_feasible_projection"
    )
    assert hard2_metrics["allocation_target_preference"] == pytest.approx(
        project_dependency_feasible_preference(nominal)
    )
    assert nominal_metrics["preference_target_mode"] == "nominal"
    assert nominal_metrics["allocation_target_preference"] == pytest.approx(nominal)
    assert nominal_metrics["preference_l1"] == pytest.approx(
        nominal_metrics["preference_l1_nominal"]
    )
    assert "preference_l1_dependency_feasible" in nominal_metrics


def test_assignment_priority_decay_preserves_hard_2_default_and_is_configurable() -> None:
    assignment_time = 5.0
    deadline = 10.0
    expected_hard_2 = float(np.exp(-assignment_time / deadline))
    assert DEFAULT_ASSIGNMENT_PRIORITY_DECAY == 1.0
    assert assignment_priority_weight(assignment_time, deadline) == pytest.approx(
        expected_hard_2
    )
    assert assignment_priority_weight(
        assignment_time, deadline, decay=0.0
    ) == pytest.approx(1.0)
    assert assignment_priority_weight(
        assignment_time, deadline, decay=2.0
    ) == pytest.approx(expected_hard_2**2)
    with pytest.raises(ValueError, match="decay"):
        assignment_priority_weight(assignment_time, deadline, decay=-0.1)


def test_environment_records_assignment_priority_decay() -> None:
    env = PreferencePaperEnv(small_config(), assignment_priority_decay=2.5)
    env.reset(seed=17)
    assert env.metrics()["assignment_priority_decay"] == pytest.approx(2.5)
    with pytest.raises(ValueError, match="assignment_priority_decay"):
        PreferencePaperEnv(small_config(), assignment_priority_decay=-1.0)


def test_configured_decay_changes_priority_accounting_only() -> None:
    base_config = small_config(mission_deadline=20.0)
    default_env = PreferencePaperEnv(base_config, assignment_priority_decay=1.0)
    fast_decay_env = PreferencePaperEnv(base_config, assignment_priority_decay=2.0)
    for env in (default_env, fast_decay_env):
        env.reset(seed=21)
        observation = env.observe()
        done = False
        while not done:
            valid = np.flatnonzero(observation["action_mask"])
            observation, _, done, _ = env.step(int(valid[0]))
    default_priority = np.asarray(
        default_env.metrics()["deadline_assignment_priority_by_type"]
    )
    fast_priority = np.asarray(
        fast_decay_env.metrics()["deadline_assignment_priority_by_type"]
    )
    assert np.all(fast_priority <= default_priority + 1e-7)
    assert np.any(fast_priority < default_priority - 1e-7)


def test_protocol_output_root_uses_namespace_unless_explicit() -> None:
    protocol = {"output_namespace": "outputs/pcrl_v0_hard3"}
    assert protocol_output_root(protocol, None) == Path(
        "outputs/pcrl_v0_hard3/formal"
    )
    explicit = Path("outputs/custom")
    assert protocol_output_root(protocol, explicit) == explicit
    assert protocol_output_root({}, None) == Path("outputs/pcrl_v0/formal")
    with pytest.raises(ValueError, match="output_namespace"):
        protocol_output_root({"output_namespace": ""}, None)


def test_runner_propagates_protocol_version_and_namespaced_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "pcrl_v0_hard3"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "version": "pcrl-v0-hard-3",
                "output_namespace": str(namespace),
                "base_scenario_config": "configs/gppo_v2_hard.json",
                "training_seeds": [1],
                "methods": ["greedy_preference"],
                "training_task_release_modes": ["phase_staggered"],
                "phase_staggered_deadline_scale": 0.7,
                "evaluation_suites": [
                    {
                        "name": "controllability_phase",
                        "task_release_mode": "phase_staggered",
                        "deadline_scale": 0.7,
                        "preference_target_mode": "nominal",
                        "primary": True,
                    }
                ],
                "controllability": {"assignment_priority_decay": 3.0},
                "training_anchor_profiles": ["balanced"],
                "held_out_profiles": ["smooth_interp"],
                "evaluation_scales": ["2x12"],
                "evaluation_episodes": 1,
                "evaluation_seed": 50000,
                "acceptance": {"primary_suite": "controllability_phase"},
            }
        ),
        encoding="utf-8",
    )
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
            str(protocol_path),
            "--phase",
            "eval",
        ],
    )
    run_pcrl_v0.main()

    assert len(commands) == 1
    command = commands[0]
    protocol_index = command.index("--protocol-version")
    assert command[protocol_index + 1] == "pcrl-v0-hard-3"
    decay_index = command.index("--assignment-priority-decay")
    assert command[decay_index + 1] == "3.0"
    target_index = command.index("--preference-target-mode")
    assert command[target_index + 1] == "nominal"
    manifest_path = namespace / "formal" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["protocol_version"] == "pcrl-v0-hard-3"
    assert manifest["assignment_priority_decay"] == pytest.approx(3.0)
    assert (
        manifest["training_preference_target_mode"]
        == "nominal"
    )


def test_preference_environment_vector_reward_and_accounting() -> None:
    env = PreferencePaperEnv(small_config(), preference=task_preference_profile("search"))
    observation = env.reset(seed=12)
    accumulated = np.zeros(N_OBJECTIVES, dtype=np.float32)
    done = False
    saw_event_interface = False
    while not done:
        valid = np.flatnonzero(observation["action_mask"])
        action = int(valid[0])
        observation, scalar_reward, done, info = env.step(
            action,
            sync_mode="event",
            event_input={"type": "reserved_test"},
        )
        vector = np.asarray(info["vector_reward"], dtype=np.float32)
        assert vector.shape == (N_OBJECTIVES,)
        assert np.isfinite(vector).all()
        assert scalar_reward == pytest.approx(float(np.dot(env.preference, vector)))
        accumulated += vector
        saw_event_interface = info["event_input"] is not None
    metrics = env.metrics()
    assert np.allclose(accumulated, env.vector_return)
    assert len(metrics["available_tasks_by_type"]) == 4
    assert len(metrics["availability_normalized_mix"]) == 4
    assert metrics["preference_l1"] >= 0.0
    assert saw_event_interface
    assert metrics["external_event_interface_used"] == 1.0


def test_phase_staggered_mode_exposes_multiple_legal_task_types() -> None:
    env = PreferencePaperEnv(
        small_config(), task_release_mode="phase_staggered"
    )
    observation = env.reset(seed=4)
    legal_actions = np.flatnonzero(observation["action_mask"][:-1])
    legal_types = {
        env.belief_tasks[divmod(int(action), env.config.max_tasks)[1]].task_type
        for action in legal_actions
    }
    assert len(legal_types) >= 2
    assert env.metrics()["task_release_mode"] == "phase_staggered"


def test_vector_gae_is_objective_wise() -> None:
    rewards = np.zeros((3, N_OBJECTIVES), dtype=np.float32)
    rewards[:, 2] = (1.0, 2.0, 3.0)
    values = np.zeros_like(rewards)
    advantages, returns = vector_gae(rewards, values, gamma=1.0, gae_lambda=1.0)
    assert advantages[:, 2].tolist() == pytest.approx([6.0, 5.0, 3.0])
    assert np.count_nonzero(advantages[:, [0, 1, 3, 4, 5, 6]]) == 0
    assert np.allclose(advantages, returns)


def test_masked_preference_auxiliary_rewards_matching_legal_type_mass() -> None:
    nodes = torch.zeros(1, 3, 24)
    nodes[0, 1, 16] = 1.0
    nodes[0, 2, 18] = 1.0
    action_mask = torch.as_tensor([[True, True, False]])
    preference = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("search"))
    ).unsqueeze(0)
    matching = preference_action_alignment_loss(
        torch.as_tensor([[0.90, 0.10, 0.0]]),
        nodes,
        action_mask,
        preference,
        max_uavs=1,
        max_tasks=2,
    )
    mismatching = preference_action_alignment_loss(
        torch.as_tensor([[0.10, 0.90, 0.0]]),
        nodes,
        action_mask,
        preference,
        max_uavs=1,
        max_tasks=2,
    )
    assert matching < mismatching


def test_gppo_initialization_preserves_action_distribution(tmp_path: Path) -> None:
    config = small_config()
    env = PaperAlignedUAVEnv(config)
    observation = env.reset(seed=9)
    base = PaperHeteroActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=config.max_uavs,
        max_tasks=config.max_tasks,
        hidden_dim=32,
        graph_mode="adaptive",
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "version": "paper-aligned-gppo-v2",
            "model_state": base.state_dict(),
            "graph_mode": "adaptive",
            "sync_mode": "event",
            "method_id": "gppo_event",
            "implementation_hash": FROZEN_GPPO_IMPLEMENTATION_HASH,
            "scenario_hash": "test",
            "scenario": "test",
            "training": {"seed": 1},
            "best_update": 1,
        },
        checkpoint,
    )
    conditioned = make_model(config)
    conditioned.initialize_from_gppo_checkpoint(
        checkpoint, require_frozen_file=False
    )
    tensors = tensor_observation(observation)
    base_distribution, _ = base(**tensors)
    for profile in ("balanced", "search", "strike"):
        preference = torch.as_tensor(
            map_task_preference_to_full(task_preference_profile(profile))
        )
        distribution, values = conditioned(**tensors, preference=preference)
        assert torch.allclose(distribution.logits, base_distribution.logits, atol=1e-6)
        assert values.shape == (1, N_OBJECTIVES)


def test_conditioning_can_change_logits_without_changing_action_mask() -> None:
    config = small_config()
    env = PreferencePaperEnv(config)
    observation = env.reset(seed=5)
    # Build a controlled decision state with two simultaneously legal task
    # types; the real chained environment can expose this after chains desync.
    observation["nodes"][config.max_uavs + 0, 16:20] = (1.0, 0.0, 0.0, 0.0)
    observation["nodes"][config.max_uavs + 1, 16:20] = (0.0, 0.0, 1.0, 0.0)
    observation["action_mask"][:] = False
    observation["action_mask"][0] = True
    observation["action_mask"][1] = True
    model = make_model(config)
    model.task_preference_gain.data.fill_(5.0)
    tensors = tensor_observation(observation)
    search = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("search"))
    )
    strike = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("strike"))
    )
    search_distribution, search_values = model(**tensors, preference=search)
    strike_distribution, strike_values = model(**tensors, preference=strike)
    valid = tensors["action_mask"].unsqueeze(0)
    assert not torch.allclose(
        search_distribution.logits[valid], strike_distribution.logits[valid]
    )
    assert torch.all(search_distribution.logits[~valid] < -1e8)
    assert torch.all(strike_distribution.logits[~valid] < -1e8)
    assert search_values.shape == strike_values.shape == (1, N_OBJECTIVES)


def test_no_conditioning_ablation_ignores_preference_and_deficit() -> None:
    config = small_config()
    env = PreferencePaperEnv(config, preference=task_preference_profile("search"))
    observation = env.reset(seed=6)
    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=config.max_uavs,
        max_tasks=config.max_tasks,
        hidden_dim=32,
        graph_mode="adaptive",
        preference_conditioning=False,
    )
    torch.nn.init.normal_(model.pair_preference_residual[-1].weight, std=0.2)
    model.task_preference_gain.data.fill_(3.0)
    tensors = tensor_observation(observation)
    search = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("search"))
    )
    strike = torch.as_tensor(
        map_task_preference_to_full(task_preference_profile("strike"))
    )
    search_distribution, _ = model(
        **tensors, preference=search
    )
    changed_tensors = dict(tensors)
    changed_tensors["preference_deficit"] = -tensors["preference_deficit"]
    strike_distribution, _ = model(
        **changed_tensors, preference=strike
    )
    assert torch.allclose(
        search_distribution.logits, strike_distribution.logits, atol=1e-6
    )


@pytest.mark.parametrize("algorithm", ["ls", "sdmgrad", "preco"])
def test_grouped_update_supports_multiple_episode_preferences(algorithm: str) -> None:
    generator = torch.Generator().manual_seed(8)
    advantages = torch.randn(10, N_OBJECTIVES, generator=generator)
    returns = torch.randn(10, N_OBJECTIVES, generator=generator)
    preferences = torch.stack(
        (
            torch.as_tensor(map_task_preference_to_full(task_preference_profile("search"))),
            torch.as_tensor(map_task_preference_to_full(task_preference_profile("strike"))),
        )
    )
    preferences = torch.cat(
        (preferences[0].expand(4, -1), preferences[1].expand(6, -1)), dim=0
    )
    preference_ids = torch.as_tensor([11] * 4 + [12] * 6)
    direction, weights = grouped_policy_direction(
        advantages,
        returns,
        preferences,
        preference_ids,
        algorithm=algorithm,
        preco_lambda=0.25,
    )
    assert direction.shape == (10,)
    assert weights.shape == (N_OBJECTIVES,)
    assert torch.isfinite(direction).all()
    assert torch.isfinite(weights).all()


def test_collect_rollouts_samples_one_distinct_preference_per_episode() -> None:
    config = small_config(max_decisions=20)
    model = make_model(config)
    batch = collect_pcrl_rollouts(
        model,
        config,
        episodes=3,
        seed_offset=0,
        sync_mode="event",
        gamma=0.99,
        gae_lambda=0.95,
    )
    ids = torch.unique(batch.preference_ids)
    assert ids.numel() == 3
    episode_preferences = []
    for preference_id in ids:
        rows = batch.preferences[batch.preference_ids == preference_id]
        assert torch.allclose(rows, rows[0].expand_as(rows))
        episode_preferences.append(rows[0])
    assert not torch.allclose(episode_preferences[0], episode_preferences[1])
    assert batch.returns.shape[1] == N_OBJECTIVES


def test_frozen_gppo_hash_has_not_changed() -> None:
    assert implementation_hash() == FROZEN_GPPO_IMPLEMENTATION_HASH

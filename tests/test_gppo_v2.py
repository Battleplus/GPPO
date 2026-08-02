import copy

import numpy as np
import pytest
import torch

from uav_assignment.paper_env import (
    EventRecord,
    PaperAlignedUAVEnv,
    PaperEnvConfig,
    TASK_PREDECESSOR_EDGE,
    TASK_SUCCESSOR_EDGE,
)
from uav_assignment.paper_models import PaperHeteroActorCritic


def quiet_config(**overrides: object) -> PaperEnvConfig:
    values: dict[str, object] = {
        "max_uavs": 4,
        "max_tasks": 12,
        "active_uavs": 3,
        "initial_tasks": 8,
        "max_decisions": 100,
        "weather_probability": 0.0,
        "failure_probability": 0.0,
        "task_change_probability": 0.0,
        "communication_drop_probability": 0.0,
    }
    values.update(overrides)
    return PaperEnvConfig(**values)


def tensors(observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "nodes": torch.as_tensor(observation["nodes"], dtype=torch.float32),
        "edge_types": torch.as_tensor(observation["edge_types"], dtype=torch.long),
        "edge_features": torch.as_tensor(
            observation["edge_features"], dtype=torch.float32
        ),
        "action_mask": torch.as_tensor(
            observation["action_mask"], dtype=torch.bool
        ),
    }


def make_model(env: PaperAlignedUAVEnv, mode: str) -> PaperHeteroActorCritic:
    return PaperHeteroActorCritic(
        env.node_feature_dim,
        env.edge_feature_dim,
        env.config.max_uavs,
        env.config.max_tasks,
        hidden_dim=32,
        graph_mode=mode,
    )


def test_adaptive_single_head_and_no_graph_have_equal_parameter_count() -> None:
    env = PaperAlignedUAVEnv(quiet_config())
    counts = {
        mode: sum(parameter.numel() for parameter in make_model(env, mode).parameters())
        for mode in ("none", "single_head", "adaptive_no_gate", "adaptive")
    }
    assert len(set(counts.values())) == 1


def test_adaptive_and_single_head_outputs_differ_under_shared_initialization() -> None:
    torch.manual_seed(4)
    env = PaperAlignedUAVEnv(quiet_config())
    observation = tensors(env.reset(seed=20))
    adaptive = make_model(env, "adaptive")
    single = make_model(env, "single_head")
    single.load_state_dict(adaptive.state_dict())
    adaptive.eval()
    single.eval()
    adaptive_distribution, adaptive_value = adaptive(**observation)
    single_distribution, single_value = single(**observation)
    assert not torch.allclose(adaptive_distribution.logits, single_distribution.logits)
    assert not torch.allclose(adaptive_value, single_value)


def test_adaptive_gate_has_parameter_matched_no_gate_control() -> None:
    torch.manual_seed(41)
    env = PaperAlignedUAVEnv(quiet_config())
    observation = tensors(env.reset(seed=201))
    adaptive = make_model(env, "adaptive")
    no_gate = make_model(env, "adaptive_no_gate")
    no_gate.load_state_dict(adaptive.state_dict())
    active_counts = []
    logits = []
    for model in (adaptive, no_gate):
        distribution, value = model(**observation)
        logits.append(distribution.logits.detach())
        loss = distribution.logits[0, observation["action_mask"]].sum() + value.sum()
        loss.backward()
        active_counts.append(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    assert active_counts[0] == active_counts[1]
    assert not torch.allclose(logits[0], logits[1])


def test_priority_directly_changes_adaptive_attention_but_not_single_head() -> None:
    torch.manual_seed(5)
    env = PaperAlignedUAVEnv(quiet_config())
    observation = tensors(env.reset(seed=21))
    adaptive = make_model(env, "adaptive").eval()
    single = make_model(env, "single_head").eval()
    single.load_state_dict(adaptive.state_dict())
    with torch.no_grad():
        adaptive.node_encoder[0].weight[:, 7] = 0.0
        single.node_encoder[0].weight[:, 7] = 0.0
        changed = copy.deepcopy(observation)
        changed["nodes"] = observation["nodes"].clone()
        changed["nodes"][env.config.max_uavs, 7] += 0.35
        adaptive_before = adaptive._encode(
            observation["nodes"].unsqueeze(0),
            observation["edge_types"].unsqueeze(0),
            observation["edge_features"].unsqueeze(0),
        )
        adaptive_after = adaptive._encode(
            changed["nodes"].unsqueeze(0),
            changed["edge_types"].unsqueeze(0),
            changed["edge_features"].unsqueeze(0),
        )
        single_before = single._encode(
            observation["nodes"].unsqueeze(0),
            observation["edge_types"].unsqueeze(0),
            observation["edge_features"].unsqueeze(0),
        )
        single_after = single._encode(
            changed["nodes"].unsqueeze(0),
            changed["edge_types"].unsqueeze(0),
            changed["edge_features"].unsqueeze(0),
        )
    assert not torch.allclose(adaptive_before[:, : env.config.max_uavs], adaptive_after[:, : env.config.max_uavs])
    assert torch.allclose(single_before, single_after)


def test_action_mask_is_respected_by_every_graph_mode() -> None:
    env = PaperAlignedUAVEnv(quiet_config())
    observation = tensors(env.reset(seed=22))
    invalid = ~observation["action_mask"]
    for mode in ("none", "single_head", "adaptive"):
        distribution, value = make_model(env, mode)(**observation)
        assert torch.all(distribution.probs[0, invalid] == 0)
        assert torch.isfinite(distribution.probs).all()
        assert torch.isfinite(value).all()


def test_task_graph_contains_both_predecessor_and_successor_relations() -> None:
    env = PaperAlignedUAVEnv(quiet_config())
    edge_types = env.reset(seed=23)["edge_types"]
    assert np.any(edge_types == TASK_PREDECESSOR_EDGE)
    assert np.any(edge_types == TASK_SUCCESSOR_EDGE)


def test_padding_nodes_do_not_change_active_policy_or_value() -> None:
    torch.manual_seed(6)
    env = PaperAlignedUAVEnv(quiet_config())
    observation = tensors(env.reset(seed=24))
    model = make_model(env, "none").eval()
    changed = copy.deepcopy(observation)
    changed["nodes"] = observation["nodes"].clone()
    inactive = changed["nodes"][:, 2] == 0
    changed["nodes"][inactive, 5:] = 1000.0
    before_distribution, before_value = model(**observation)
    after_distribution, after_value = model(**changed)
    assert torch.allclose(before_distribution.logits, after_distribution.logits)
    assert torch.allclose(before_value, after_value)


def test_structured_completion_and_successor_unlock_events_trigger_event_sync() -> None:
    env = PaperAlignedUAVEnv(quiet_config(active_uavs=1, max_uavs=2))
    observation = env.reset(seed=25)
    action = int(np.flatnonzero(observation["action_mask"][:-1])[0])
    observation, _, done, _ = env.step(action, sync_mode="event")
    while not done and not any(row["event_type"] == "task_completed" for row in env.event_log):
        valid = np.flatnonzero(observation["action_mask"])
        observation, _, done, _ = env.step(int(valid[0]), sync_mode="event")
    event_types = {row["event_type"] for row in env.event_log}
    assert "task_completed" in event_types
    assert "successor_unlocked" in event_types
    assert all(
        row["triggered_sync"]
        for row in env.event_log
        if row["event_type"] in {"task_completed", "successor_unlocked"}
    )


def test_failure_reallocation_leader_election_and_event_records() -> None:
    env = PaperAlignedUAVEnv(quiet_config(failure_probability=1.0))
    env.reset(seed=26)
    probe = np.random.default_rng(123)
    probe.exponential(1.0)
    probe.choice(np.arange(1, 5), p=np.asarray([0.0, 1.0, 0.0, 0.0]))
    victim = int(probe.choice([0, 1, 2]))
    env.leader_id = victim
    env.rng = np.random.default_rng(123)
    env.next_exogenous_event_time = env.current_time
    env.next_exogenous_event_type = 2
    assert env._sample_event(0.0) == 2
    event_types = {record.event_type for record in env._pending_records}
    assert {"uav_failure", "leader_failure", "leader_elected"} <= event_types
    assert env.leader_id != victim
    assert env.leader_changes == 1


def test_heartbeat_timeout_is_logged_once() -> None:
    env = PaperAlignedUAVEnv(quiet_config())
    observation = env.reset(seed=27)
    follower = next(index for index in range(3) if index != env.leader_id)
    env.uavs[follower].communication = 0.0
    env.heartbeat_age[follower] = env.config.heartbeat_timeout + 0.5
    for _ in range(2):
        action = int(np.flatnonzero(observation["action_mask"])[0])
        observation, _, _, _ = env.step(action, sync_mode="none")
    timeout_rows = [
        row
        for row in env.event_log
        if row["event_type"] == "heartbeat_timeout" and row["target_id"] == follower
    ]
    assert len(timeout_rows) == 1


def test_heartbeats_run_independently_of_state_sync() -> None:
    env = PaperAlignedUAVEnv(quiet_config(heartbeat_interval=0.5))
    observation = env.reset(seed=271)
    for task in env.tasks:
        if task.active:
            task.predecessor = -1
    env._synchronize_belief(count_communication=False, full=True)
    observation = env.observe()
    while env.current_time == 0.0:
        action = int(np.flatnonzero(observation["action_mask"])[0])
        observation, _, _, _ = env.step(action, sync_mode="none")
    assert env.communication_events == 0
    assert env.heartbeat_messages > 0


def test_zero_time_decisions_do_not_sample_exogenous_events() -> None:
    env = PaperAlignedUAVEnv(quiet_config(weather_probability=1.0))
    env.reset(seed=272)
    for task in env.tasks:
        if task.active:
            task.predecessor = -1
    for uav in env.uavs:
        if uav.active:
            uav.capabilities[:] = 1.0
    env._synchronize_belief(count_communication=False, full=True)
    observation = env.observe()
    action = int(np.flatnonzero(observation["action_mask"][:-1])[0])
    _, _, _, info = env.step(action, sync_mode="event")
    assert info["time_advance"] == 0.0
    assert not any(row["event_type"] == "weather_change" for row in info["events"])


@pytest.mark.parametrize(
    ("probability", "expected"),
    (("task_change_probability", "new_task_arrival"), ("communication_drop_probability", "communication_")),
)
def test_exogenous_events_have_structured_records(
    probability: str, expected: str
) -> None:
    overrides = {probability: 1.0}
    if probability == "task_change_probability":
        overrides.update({"max_tasks": 12, "initial_tasks": 8})
    env = PaperAlignedUAVEnv(quiet_config(**overrides))
    env.reset(seed=28)
    env.current_time = env.next_exogenous_event_time
    event = env._sample_event(0.0)
    assert event in {3, 4}
    assert any(record.event_type.startswith(expected) for record in env._pending_records)


def test_partial_sync_keeps_unreachable_unreported_uav_stale() -> None:
    env = PaperAlignedUAVEnv(quiet_config())
    env.reset(seed=29)
    old_target_health = env.belief_uavs[1].health
    old_health = env.belief_uavs[2].health
    old_unreported_communication = env.belief_uavs[2].communication
    env.uavs[1].communication = 0.1
    env.uavs[2].communication = 0.1
    env.uavs[1].health = 0.2
    env.uavs[2].health = 0.3
    record = EventRecord(1, 0.0, "communication_degraded", "link", "uav", 1)
    updated = env._synchronize_belief(records=[record])
    assert 1 not in updated
    assert env.belief_uavs[1].communication == 0.1
    assert env.belief_uavs[1].health == old_target_health
    assert env.belief_uavs[2].health == old_health
    assert env.belief_uavs[2].communication == old_unreported_communication


def test_sync_modes_and_invalid_mode() -> None:
    expected = {"none": 0, "event": 1, "periodic": 1, "always": 1}
    for mode, count in expected.items():
        env = PaperAlignedUAVEnv(quiet_config(periodic_interval=1.0))
        observation = env.reset(seed=30)
        if mode == "event":
            env.next_exogenous_event_time = env.current_time
            env.next_exogenous_event_type = 1
        if mode == "periodic":
            env.current_time = 1.0
            env.belief_time = 1.0
        action = int(np.flatnonzero(observation["action_mask"])[0])
        env.step(action, sync_mode=mode)
        assert env.communication_events == count
    env = PaperAlignedUAVEnv(quiet_config())
    observation = env.reset(seed=31)
    with pytest.raises(ValueError):
        env.step(int(np.flatnonzero(observation["action_mask"])[0]), sync_mode="typo")  # type: ignore[arg-type]


def test_deadline_metrics_are_distinct_from_final_completion() -> None:
    env = PaperAlignedUAVEnv(
        quiet_config(
            max_uavs=2,
            active_uavs=1,
            max_tasks=4,
            initial_tasks=4,
            mission_deadline=0.5,
            workload_scale=2.0,
        )
    )
    observation = env.reset(seed=32)
    done = False
    while not done:
        valid = np.flatnonzero(observation["action_mask"])
        observation, _, done, _ = env.step(int(valid[0]), sync_mode="event")
    metrics = env.metrics()
    assert metrics["completion_rate"] == 1.0
    assert metrics["deadline_completion_rate"] < 1.0
    assert metrics["deadline_remaining_tasks"] > 0
    assert metrics["mission_success"] == 0.0
    assert metrics["throughput"] == pytest.approx(
        metrics["deadline_completed_total"] / env.config.mission_deadline
    )


def test_hard_scenario_random_policy_does_not_saturate_deadline() -> None:
    config = PaperEnvConfig(
        max_uavs=4,
        max_tasks=24,
        active_uavs=3,
        initial_tasks=20,
        max_decisions=200,
        mission_deadline=8.0,
        task_chain_length=5,
        workload_scale=1.8,
        weather_probability=0.06,
        failure_probability=0.02,
        task_change_probability=0.04,
        communication_drop_probability=0.04,
    )
    completion_rates = []
    for seed in range(20_000, 20_012):
        env = PaperAlignedUAVEnv(config)
        observation = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 999)
        done = False
        while not done:
            valid = np.flatnonzero(observation["action_mask"][:-1])
            action = int(rng.choice(valid)) if valid.size else env.noop_action
            observation, _, done, _ = env.step(action, sync_mode="event")
        completion_rates.append(float(env.metrics()["deadline_completion_rate"]))
    assert float(np.mean(completion_rates)) < 0.8


def test_eq4_uav_aggregation_is_a_sum() -> None:
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]])
    mask = torch.tensor([[[True, True, False]]])
    from uav_assignment.paper_models import AdaptiveTaskUpdate

    result = AdaptiveTaskUpdate._aggregate(x, mask, normalize=False)
    assert torch.equal(result, torch.tensor([[[4.0, 6.0]]]))

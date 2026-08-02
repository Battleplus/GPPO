import numpy as np
import pytest

from uav_assignment.paper_env import (
    UAV_TASK_EDGE,
    PaperAlignedUAVEnv,
    PaperEnvConfig,
)


def deterministic_config(**overrides: object) -> PaperEnvConfig:
    values: dict[str, object] = {
        "max_uavs": 3,
        "max_tasks": 8,
        "active_uavs": 3,
        "initial_tasks": 8,
        "max_decisions": 20,
        "heartbeat_interval": 0.2,
        "periodic_interval": 0.2,
        "weather_probability": 0.0,
        "failure_probability": 0.0,
        "task_change_probability": 0.0,
        "communication_drop_probability": 0.0,
    }
    values.update(overrides)
    return PaperEnvConfig(**values)


def test_capability_graph_is_sparse_but_every_task_type_is_covered() -> None:
    env = PaperAlignedUAVEnv(deterministic_config())
    observation = env.reset(seed=11)
    active = env.uavs[: env.config.active_uavs]

    for task_type in range(4):
        assert any(
            uav.capabilities[task_type] >= env.config.capability_threshold
            for uav in active
        )
    assert any(
        uav.capabilities[task_type] < env.config.capability_threshold
        for uav in active
        for task_type in range(4)
    )

    cross_edges = observation["edge_types"][
        : env.config.active_uavs,
        env.config.max_uavs : env.config.max_uavs + env.config.initial_tasks,
    ]
    density = float(np.mean(cross_edges == UAV_TASK_EDGE))
    assert 0.0 < density < 1.0


def test_unassigned_task_uses_mean_capable_uav_completion_estimate() -> None:
    env = PaperAlignedUAVEnv(
        deterministic_config(
            max_uavs=2,
            max_tasks=1,
            active_uavs=2,
            initial_tasks=1,
            task_chain_length=1,
        )
    )
    env.reset(seed=3)
    task = env.tasks[0]
    task.position = np.zeros(2, dtype=np.float32)
    task.workload = 1.0
    task.task_type = 0
    for uav in env.uavs:
        uav.position = np.zeros(2, dtype=np.float32)
        uav.speed = 1.0
        uav.remaining_time = 0.0
    env.uavs[0].capabilities[0] = 0.5
    env.uavs[1].capabilities[0] = 1.0

    assert env.makespan == pytest.approx((2.0 + 1.0) / 2.0)


@pytest.mark.parametrize("sync_mode", ["none", "event", "periodic"])
def test_stale_noop_conflict_cannot_freeze_simulation_clock(sync_mode: str) -> None:
    env = PaperAlignedUAVEnv(
        deterministic_config(
            max_uavs=2,
            max_tasks=4,
            active_uavs=2,
            initial_tasks=4,
            task_chain_length=1,
            max_decisions=5,
        )
    )
    env.reset(seed=7)
    for uav in env.uavs:
        uav.communication = 1.0
    for uav in env.belief_uavs:
        if uav.active:
            uav.busy_task = 0
            uav.remaining_time = 1.0

    _, _, done, info = env.step(env.noop_action, sync_mode=sync_mode)

    assert not done
    assert env.current_time == pytest.approx(0.2)
    assert info["time_advance"] == pytest.approx(0.2)
    assert any(
        record["event_type"] == "assignment_state_conflict"
        for record in info["events"]
    )
    if sync_mode == "event":
        assert info["synchronized"]


def test_successful_heartbeat_refreshes_policy_freshness_without_state_sync() -> None:
    env = PaperAlignedUAVEnv(
        deterministic_config(
            max_uavs=2,
            max_tasks=4,
            active_uavs=2,
            initial_tasks=4,
            task_chain_length=1,
        )
    )
    env.reset(seed=9)
    for uav in env.uavs:
        uav.communication = 1.0
    for uav in env.belief_uavs:
        uav.busy_task = 0
        uav.remaining_time = 1.0

    env.step(env.noop_action, sync_mode="none")

    assert env.heartbeat_messages == 1
    assert np.allclose(env.belief_heartbeat_age, env.heartbeat_age)


def test_scheduled_exogenous_event_interrupts_long_running_task() -> None:
    env = PaperAlignedUAVEnv(
        deterministic_config(
            max_uavs=2,
            max_tasks=1,
            active_uavs=2,
            initial_tasks=1,
            task_chain_length=1,
            heartbeat_interval=1.0,
        )
    )
    env.reset(seed=13)
    task = env.tasks[0]
    task.position = np.ones(2, dtype=np.float32)
    task.workload = 5.0
    env.uavs[0].position = np.zeros(2, dtype=np.float32)
    env.uavs[0].capabilities[task.task_type] = 1.0
    env.belief_uavs[0] = env._clone_uavs([env.uavs[0]])[0]
    env.next_exogenous_event_time = 0.25
    env.next_exogenous_event_type = 1
    action = int(np.flatnonzero(env.valid_action_mask()[:-1])[0])

    _, _, done, info = env.step(action, sync_mode="event")

    assert not done
    assert env.current_time == pytest.approx(0.25)
    assert env.tasks[0].completed is False
    assert env.uavs[0].remaining_time > 0.0
    assert info["event"] == "weather_change"
    assert any(record["event_type"] == "weather_change" for record in info["events"])


def test_periodic_sync_does_not_mark_incidental_event_as_event_triggered() -> None:
    env = PaperAlignedUAVEnv(
        deterministic_config(
            max_uavs=2,
            max_tasks=4,
            active_uavs=2,
            initial_tasks=4,
            task_chain_length=1,
        )
    )
    env.reset(seed=15)
    for uav in env.belief_uavs:
        uav.busy_task = 0
        uav.remaining_time = 1.0

    _, _, _, info = env.step(env.noop_action, sync_mode="periodic")

    assert info["sync_reason"] == "periodic_boundary"
    assert all(not record["triggered_sync"] for record in info["events"])

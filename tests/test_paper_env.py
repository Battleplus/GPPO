import numpy as np
import torch

from uav_assignment.paper_env import (
    PaperAlignedUAVEnv,
    PaperEnvConfig,
    TASK_PREDECESSOR_EDGE,
    UAV_TASK_EDGE,
)
from uav_assignment.paper_models import PaperHeteroActorCritic


def make_env() -> PaperAlignedUAVEnv:
    return PaperAlignedUAVEnv(
        PaperEnvConfig(
            max_uavs=4,
            max_tasks=12,
            active_uavs=3,
            initial_tasks=8,
            max_decisions=80,
            weather_probability=0.0,
            failure_probability=0.0,
            task_change_probability=0.0,
            communication_drop_probability=0.0,
        )
    )


def test_async_assignment_does_not_complete_immediately() -> None:
    env = make_env()
    observation = env.reset(seed=9)
    action = int(np.flatnonzero(observation["action_mask"][:-1])[0])
    _, _, _, info = env.step(action, sync_mode="none")
    task_index = action % env.config.max_tasks
    uav_index = action // env.config.max_tasks
    assert not env.tasks[task_index].completed
    assert env.tasks[task_index].start_time == 0.0
    assert env.uavs[uav_index].busy_task == task_index
    assert info["completed_tasks"] == []


def test_predecessor_unlocks_only_after_completion() -> None:
    env = make_env()
    observation = env.reset(seed=10)
    first = next(
        index
        for index, task in enumerate(env.tasks)
        if task.active and task.predecessor < 0
    )
    successor = next(
        index for index, task in enumerate(env.tasks) if task.predecessor == first
    )
    first_action = next(
        int(uav * env.config.max_tasks + first)
        for uav in range(env.config.max_uavs)
        if observation["action_mask"][uav * env.config.max_tasks + first]
    )
    _, _, _, _ = env.step(first_action, sync_mode="none")
    assert not env.observe()["action_mask"].any() or not any(
        env.observe()["action_mask"][uav * env.config.max_tasks + successor]
        for uav in range(env.config.max_uavs)
    )
    for _ in range(20):
        mask = env.observe()["action_mask"]
        if env.tasks[first].completed:
            break
        _, _, _, _ = env.step(int(np.flatnonzero(mask)[0]), sync_mode="none")
    assert env.tasks[first].completed
    assert any(
        env.observe()["action_mask"][uav * env.config.max_tasks + successor]
        for uav in range(env.config.max_uavs)
    )


def test_edge_types_and_edge_features_keep_cross_type_edges() -> None:
    env = make_env()
    observation = env.reset(seed=11)
    edge_types = observation["edge_types"]
    edge_features = observation["edge_features"]
    assert np.any(edge_types == UAV_TASK_EDGE)
    assert np.any(edge_types == TASK_PREDECESSOR_EDGE)
    cross = edge_types[: env.config.max_uavs, env.config.max_uavs :]
    assert np.any(cross == UAV_TASK_EDGE)
    assert np.isfinite(edge_features).all()
    assert edge_features.shape[-1] == 5


def test_staged_paper_model_and_no_graph_control_have_same_action_shape() -> None:
    env = make_env()
    observation = env.reset(seed=12)
    tensors = {
        key: torch.as_tensor(value)
        for key, value in observation.items()
        if key in {"nodes", "edge_types", "edge_features", "action_mask"}
    }
    for mode in ("none", "staged"):
        model = PaperHeteroActorCritic(
            env.node_feature_dim,
            env.edge_feature_dim,
            env.config.max_uavs,
            env.config.max_tasks,
            graph_mode=mode,
            hidden_dim=32,
        )
        distribution, value = model(**tensors)
        assert distribution.logits.shape == (1, env.n_actions)
        assert torch.isfinite(distribution.probs).all()
        assert value.shape == (1, 1)


def test_paper_reward_is_exact_makespan_difference() -> None:
    env = PaperAlignedUAVEnv(
        PaperEnvConfig(
            max_uavs=2,
            max_tasks=4,
            active_uavs=1,
            initial_tasks=4,
            max_decisions=20,
            weather_probability=0.0,
            failure_probability=0.0,
            task_change_probability=0.0,
            communication_drop_probability=0.0,
            include_engineering_rewards=False,
        )
    )
    observation = env.reset(seed=13)
    previous = env.makespan
    action = int(np.flatnonzero(observation["action_mask"][:-1])[0])
    _, reward, _, info = env.step(action, sync_mode="none")
    assert np.isclose(reward, previous - env.makespan)
    assert np.isclose(reward, info["paper_reward"])
    assert info["invalid_penalty"] == 0.0


def test_unassigned_active_tasks_remain_in_makespan_estimate() -> None:
    env = make_env()
    env.reset(seed=16)
    before = env.makespan
    assert before > 0.0
    assert any(task.active and not task.completed for task in env.tasks)


def test_uav_failure_reopens_running_task_and_event_sync_refreshes_belief() -> None:
    env = PaperAlignedUAVEnv(
        PaperEnvConfig(
            max_uavs=3,
            max_tasks=8,
            active_uavs=3,
            initial_tasks=8,
            max_decisions=40,
            weather_probability=0.0,
            failure_probability=0.0,
            task_change_probability=0.0,
            communication_drop_probability=0.0,
        )
    )
    observation = env.reset(seed=14)
    # Remove precedence/capability bottlenecks for this focused failure test so
    # all live UAVs can be kept busy before the injected event.
    for task in env.tasks:
        if task.active:
            task.predecessor = -1
    for uav in env.uavs:
        if uav.active:
            uav.capabilities.fill(1.0)
    env._synchronize_belief(count_communication=False)
    observation = env.observe()
    assigned_uavs: set[int] = set()
    for _ in range(env.config.active_uavs):
        valid = np.flatnonzero(observation["action_mask"][:-1])
        action = next(
            int(candidate)
            for candidate in valid
            if int(candidate) // env.config.max_tasks not in assigned_uavs
        )
        assigned_uavs.add(action // env.config.max_tasks)
        if len(assigned_uavs) < env.config.active_uavs:
            observation, _, done, _ = env.step(action, sync_mode="none")
            assert not done
        else:
            # A public step would immediately advance to the first completion
            # once all UAVs are busy. Inject the final assignment directly so
            # the subsequent failure is guaranteed to hit a running task.
            assert env._apply_assignment(
                env.uavs, env.tasks, action, env.current_time, env.weather_severity
            )
            assert env._apply_assignment(
                env.belief_uavs,
                env.belief_tasks,
                action,
                env.belief_time,
                env.belief_weather,
            )
    assert all(env.uavs[index].busy_task >= 0 for index in assigned_uavs)

    # Force failure while every live UAV has a running task, so a task must be reopened.
    env.config.failure_probability = 1.0
    env.next_exogenous_event_time = env.current_time
    env.next_exogenous_event_type = 2
    assert env._sample_event(0.0) == 2
    reopened = [
        index
        for index, task in enumerate(env.tasks)
        if task.reallocation_attempts == 1 and task.assigned_uav < 0
    ]
    assert len(reopened) == 1
    reopened_index = reopened[0]
    assert env.reallocated_tasks == 1
    assert not env.tasks[reopened_index].completed
    env.config.failure_probability = 0.0
    env._schedule_next_exogenous_event()
    # The structured failure/reallocation records themselves trigger event-mode
    # synchronization. Continue physical time until a survivor becomes idle.
    observation, _, done, info = env.step(env.noop_action, sync_mode="event")
    assert not done
    assert info["synchronized"]
    valid_reallocation: list[int] = []
    for _ in range(env.config.max_decisions):
        valid_reallocation = [
            int(index)
            for index in np.flatnonzero(observation["action_mask"][:-1])
            if int(index) % env.config.max_tasks == reopened_index
        ]
        if valid_reallocation or done:
            break
        valid = np.flatnonzero(observation["action_mask"])
        observation, _, done, _ = env.step(int(valid[0]), sync_mode="event")
    assert valid_reallocation
    observation, _, done, info = env.step(valid_reallocation[0], sync_mode="event")
    assert info["event"] == "none"
    assert all(
        belief.alive == truth.alive
        for belief, truth in zip(env.belief_uavs, env.uavs)
    )
    for _ in range(env.config.max_decisions):
        if env.tasks[reopened_index].completed or done:
            break
        valid = np.flatnonzero(observation["action_mask"])
        action = int(valid[0]) if valid.size else env.noop_action
        observation, _, done, _ = env.step(action, sync_mode="event")
    assert env.tasks[reopened_index].completed
    assert env.reallocation_successes == 1


def test_no_sync_does_not_reveal_communication_drop() -> None:
    env = PaperAlignedUAVEnv(
        PaperEnvConfig(
            max_uavs=3,
            max_tasks=8,
            active_uavs=3,
            initial_tasks=8,
            max_decisions=40,
            weather_probability=0.0,
            failure_probability=0.0,
            task_change_probability=0.0,
            communication_drop_probability=1.0,
        )
    )
    observation = env.reset(seed=15)
    before = [uav.communication for uav in env.belief_uavs]
    env.next_exogenous_event_time = env.current_time
    env.next_exogenous_event_type = 4
    assert env._sample_event(0.0) == 4
    env.config.communication_drop_probability = 0.0
    env._schedule_next_exogenous_event()
    action = int(np.flatnonzero(observation["action_mask"][:-1])[0])
    _, _, _, info = env.step(action, sync_mode="none")
    assert any(
        row["event_type"].startswith("communication_") for row in info["events"]
    )
    after_belief = [uav.communication for uav in env.belief_uavs]
    after_true = [uav.communication for uav in env.uavs]
    assert after_belief == before
    assert after_true != after_belief


def test_graph_and_no_graph_controls_have_equal_parameter_count() -> None:
    kwargs = dict(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=4,
        max_tasks=12,
        hidden_dim=32,
    )
    graph = PaperHeteroActorCritic(**kwargs, graph_mode="staged")
    plain = PaperHeteroActorCritic(**kwargs, graph_mode="none")
    assert sum(parameter.numel() for parameter in graph.parameters()) == sum(
        parameter.numel() for parameter in plain.parameters()
    )

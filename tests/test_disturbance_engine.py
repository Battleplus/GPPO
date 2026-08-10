from __future__ import annotations

from typing import Any

import numpy as np

from uav_assignment.disturbances import (
    DisturbanceConfig,
    DisturbanceEngine,
    SourceConfig,
    TaskRuntimeState,
)
from uav_assignment.paper_faithful_env import PaperFaithfulConfig, PaperFaithfulUAVEnv


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert left.dtype == right.dtype
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_combined_tape_is_deterministic_and_globally_ordered() -> None:
    cfg = DisturbanceConfig(
        instance_seed=7,
        disturbance_seed=8,
        gilbert_elliott_packet_loss=SourceConfig(
            True, {"tick": 1.0, "p_gb": 0.2, "p_bg": 0.3}
        ),
        uav_failure=SourceConfig(
            True, {"events": [{"time": 2.0, "uav_id": "u0", "permanent": True}]}
        ),
        task_priority_change=SourceConfig(
            True, {"events": [{"time": 3.0, "task_id": "t0", "priority": 2.0}]}
        ),
        wind_field=SourceConfig(True, {"vector": [0.1, 0.0]}),
    )
    kwargs = {
        "horizon": 5.0,
        "uav_ids": ("u0", "u1"),
        "tasks": (TaskRuntimeState("t0"),),
    }
    left = DisturbanceEngine(cfg, **kwargs)
    right = DisturbanceEngine(cfg, **kwargs)
    assert left.tape.sha256 == right.tape.sha256
    assert [event.sort_key for event in left.tape.events] == sorted(
        event.sort_key for event in left.tape.events
    )
    assert {event.source for event in left.tape.events} >= {
        "gilbert_elliott_packet_loss", "uav_failure", "task_priority_change", "wind_field"
    }


def test_disabled_engine_is_stepwise_equivalent_to_frozen_baseline() -> None:
    baseline = PaperFaithfulUAVEnv(PaperFaithfulConfig(instance_seed=123))
    candidate = PaperFaithfulUAVEnv(PaperFaithfulConfig(instance_seed=123))
    left = baseline.reset(seed=123)
    right = candidate.reset(seed=123)
    engine = DisturbanceEngine(
        DisturbanceConfig(instance_seed=123, disturbance_seed=456),
        horizon=100.0,
        uav_ids=tuple(f"u{index}" for index in range(candidate.faithful_config.scale.uavs)),
        tasks=tuple(
            TaskRuntimeState(f"t{index}")
            for index in range(candidate.faithful_config.scale.subtasks)
        ),
    )
    assert engine.config.all_disabled
    assert engine.tape.events == ()
    assert_nested_equal(left, right)
    for _ in range(30):
        legal = np.flatnonzero(left["action_mask"])
        action = int(legal[0])
        left, left_reward, left_done, left_info = baseline.step(action, sync_mode="event")
        right, right_reward, right_done, right_info = candidate.step(action, sync_mode="event")
        engine.advance(candidate.current_time)
        assert_nested_equal(left, right)
        assert left_reward == right_reward
        assert left_done == right_done
        assert_nested_equal(left_info, right_info)
        assert_nested_equal(baseline.true_observation(), candidate.true_observation())
        assert baseline.metrics() == candidate.metrics()
        if left_done:
            break


def test_engine_advances_all_layers_and_emits_future_one_to_five_labels() -> None:
    cfg = DisturbanceConfig(
        uav_failure=SourceConfig(
            True, {"events": [{"time": 2.0, "uav_id": "u0", "permanent": True}]}
        ),
        task_priority_change=SourceConfig(
            True, {"events": [{"time": 4.0, "task_id": "t0", "priority": 5.0}]}
        ),
    )
    engine = DisturbanceEngine(
        cfg, horizon=10.0, uav_ids=("u0", "u1"), tasks=(TaskRuntimeState("t0"),)
    )
    step = engine.advance(2.0)
    assert [event.event_type for event in step.current_events] == ["uav_failure"]
    assert not engine.uav.states["u0"].alive
    assert len(engine.logger.records) == 1
    assert engine.logger.records[0].observed_time == 2.0
    assert engine.logger.records[0].ground_truth == step.current_events[0].ground_truth
    assert engine.logger.records[0].effect["applied"] is True
    labels = engine.future_event_targets(
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0), current_decision_index=0
    )
    assert len(labels) == 5
    assert labels[1]["events"][0]["event_type"] == "uav_failure"
    assert labels[3]["events"][0]["event_type"] == "task_priority_change"
    assert engine.snapshot()["tape_sha256"] == engine.tape.sha256


def test_future_label_availability_is_explicit_near_episode_end() -> None:
    engine = DisturbanceEngine(
        DisturbanceConfig(), horizon=2.0, uav_ids=("u0",), tasks=(TaskRuntimeState("t0"),)
    )
    labels = engine.future_event_targets((0.0, 1.0, 2.0), current_decision_index=1)
    assert labels[0]["available"] is True
    assert all(item["available"] is False for item in labels[1:])

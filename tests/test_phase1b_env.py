from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from uav_assignment.disturbances import DisturbanceConfig, SourceConfig
from uav_assignment.paper_faithful_env import PaperFaithfulConfig, PaperFaithfulUAVEnv
from uav_assignment.phase1b_env import Phase1BPaperFaithfulUAVEnv


def nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            nested_equal(a, b)
    else:
        assert left == right


def test_all_disabled_adapter_is_exact_frozen_baseline_for_multiple_seeds() -> None:
    for seed in (1, 2, 3):
        baseline = PaperFaithfulUAVEnv(PaperFaithfulConfig(instance_seed=seed))
        adapted = Phase1BPaperFaithfulUAVEnv(
            PaperFaithfulConfig(instance_seed=seed), DisturbanceConfig(disturbance_seed=99)
        )
        left = baseline.reset(seed=seed)
        right = adapted.reset(seed=seed)
        nested_equal(left, right)
        for _ in range(20):
            action = int(np.flatnonzero(left["action_mask"])[0])
            left, lr, ld, li = baseline.step(action, sync_mode="event")
            right, rr, rd, ri = adapted.step(action, sync_mode="event")
            nested_equal(left, right)
            assert (lr, ld) == (rr, rd)
            nested_equal(li, ri)
            nested_equal(baseline.true_observation(), adapted.true_observation())
            assert baseline.metrics() == adapted.metrics()
            if ld:
                break


def test_failure_at_reset_removes_uav_from_true_and_belief_action_masks() -> None:
    disturbance = DisturbanceConfig(
        uav_failure=SourceConfig(
            True, {"events": [{"time": 0.0, "uav_id": "u0", "permanent": True}]}
        )
    )
    env = Phase1BPaperFaithfulUAVEnv(disturbance_config=disturbance)
    observation = env.reset(seed=7)
    assert not env.uavs[0].alive
    assert observation["nodes"][0, 3] == 0.0
    assert not observation["action_mask"][: env.config.max_tasks].any()
    assert env.leader_id == 1


def test_dynamic_task_arrival_updates_true_registry_and_action_mask() -> None:
    disturbance = DisturbanceConfig(
        task_arrival=SourceConfig(
            True,
            {"events": [{"time": 0.0, "task_id": "new", "predecessors": [], "priority": 2.0}]},
        )
    )
    env = Phase1BPaperFaithfulUAVEnv(disturbance_config=disturbance)
    env.reset(seed=8)
    slot = env.task_id_to_slot["new"]
    assert env.tasks[slot].active
    assert env.tasks[slot].priority == 2.0
    true_mask = env.true_observation()["action_mask"]
    assert any(true_mask[uav * env.config.max_tasks + slot] for uav in range(env.faithful_config.scale.uavs))


def test_temporary_failure_has_explicit_recovery_boundary() -> None:
    disturbance = DisturbanceConfig(
        uav_failure=SourceConfig(
            True,
            {"events": [{"time": 0.0, "uav_id": "u0", "permanent": False, "duration": 2.0}]},
        )
    )
    env = Phase1BPaperFaithfulUAVEnv(disturbance_config=disturbance)
    env.reset(seed=9)
    assert not env.uavs[0].alive
    assert any(event.event_type == "uav_recovery" for event in env.disturbance_engine.tape.events)
    env.current_time = 2.0
    env.advance_disturbances(2.0)
    assert env.uavs[0].alive


def test_wind_changes_flight_time_but_keeps_values_finite() -> None:
    disturbance = DisturbanceConfig(wind_field=SourceConfig(True, {"vector": [0.2, 0.0]}))
    env = Phase1BPaperFaithfulUAVEnv(disturbance_config=disturbance)
    env.reset(seed=10)
    uav, task = env.uavs[0], env.tasks[0]
    disturbed = env._execution_components(uav, task, 0.0)
    baseline = PaperFaithfulUAVEnv(PaperFaithfulConfig(instance_seed=10))
    baseline.reset(seed=10)
    plain = baseline._execution_components(baseline.uavs[0], baseline.tasks[0], 0.0)
    assert all(np.isfinite(value) and value >= 0 for value in disturbed)
    # Random geometry may make this wind cross-track, but the adapter must apply a physical change.
    assert disturbed[0] != pytest.approx(plain[0])


def test_delayed_belief_report_is_invisible_until_arrival() -> None:
    disturbance = DisturbanceConfig(
        message_delay=SourceConfig(True, {"minimum": 2.0, "maximum": 2.0, "ttl": 5.0})
    )
    env = Phase1BPaperFaithfulUAVEnv(disturbance_config=disturbance)
    env.reset(seed=11)
    reporter = 1
    old_health = float(env.belief_uavs[reporter].health)
    env.uavs[reporter].health = old_health - 0.3
    updated = env._synchronize_belief(records=[])
    assert reporter not in updated
    assert env.belief_uavs[reporter].health == old_health
    assert env._deliver_pending_beliefs(1.999) == []
    assert env.belief_uavs[reporter].health == old_health
    assert reporter in env._deliver_pending_beliefs(2.0)
    assert env.belief_uavs[reporter].health == pytest.approx(old_health - 0.3)


def test_dropped_belief_report_never_mutates_true_state_or_cache() -> None:
    disturbance = DisturbanceConfig(
        gilbert_elliott_packet_loss=SourceConfig(
            True,
            {"tick": 1.0, "p_gb": 0.0, "p_bg": 0.0, "loss_good": 1.0, "loss_bad": 1.0},
        )
    )
    env = Phase1BPaperFaithfulUAVEnv(disturbance_config=disturbance)
    env.reset(seed=12)
    reporter = 1
    old_belief = float(env.belief_uavs[reporter].communication)
    env.uavs[reporter].communication = old_belief - 0.2
    true_value = float(env.uavs[reporter].communication)
    env._synchronize_belief(records=[])
    assert env.belief_uavs[reporter].communication == old_belief
    assert env.uavs[reporter].communication == true_value
    assert env.disturbance_engine.communication.audit.messages_dropped > 0


def test_partitioned_report_arrives_at_recovery_boundary() -> None:
    disturbance = DisturbanceConfig(
        message_delay=SourceConfig(True, {"minimum": 0.0, "maximum": 0.0, "ttl": 10.0}),
        network_partition=SourceConfig(
            True,
            {"intervals": [{"start": 0.0, "end": 3.0, "groups": [["u0"], ["u1", "u2", "u3", "u4"]]}]},
        ),
    )
    env = Phase1BPaperFaithfulUAVEnv(disturbance_config=disturbance)
    env.reset(seed=13)
    reporter = 1
    old_health = float(env.belief_uavs[reporter].health)
    env.uavs[reporter].health -= 0.25
    env._synchronize_belief(records=[])
    assert env._deliver_pending_beliefs(2.999) == []
    assert env.belief_uavs[reporter].health == old_health
    assert reporter in env._deliver_pending_beliefs(3.0)
    assert env.belief_uavs[reporter].health == pytest.approx(old_health - 0.25)

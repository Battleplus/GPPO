from __future__ import annotations

import pytest

from uav_assignment.disturbances import (
    DisturbanceConfig,
    SourceConfig,
    UAVDisturbanceLayer,
    generate_uav_events,
)


def test_permanent_leader_failure_releases_task_and_elects_successor() -> None:
    cfg = DisturbanceConfig(
        uav_failure=SourceConfig(
            True,
            {"events": [{"time": 2.0, "uav_id": "u0", "permanent": True}]},
        )
    )
    events = generate_uav_events(cfg, horizon=10.0, uav_ids=("u0", "u1", "u2"))
    layer = UAVDisturbanceLayer(("u0", "u1", "u2"), events, initial_leader="u0")
    layer.assign_task("u0", "task-7")
    layer.advance(2.0)
    assert not layer.states["u0"].alive
    assert not layer.available("u0")
    assert layer.states["u0"].current_task is None
    assert [(item.task_id, item.reason) for item in layer.released_tasks] == [
        ("task-7", "permanent_failure")
    ]
    assert layer.leader_id == "u1"
    with pytest.raises(ValueError, match="unavailable"):
        layer.assign_task("u0", "task-8")


def test_temporary_failure_recovers_and_never_releases_twice() -> None:
    cfg = DisturbanceConfig(
        uav_failure=SourceConfig(
            True,
            {"events": [{"time": 1.0, "uav_id": "u1", "permanent": False, "duration": 3.0}]},
        )
    )
    layer = UAVDisturbanceLayer(
        ("u0", "u1"), generate_uav_events(cfg, horizon=8.0, uav_ids=("u0", "u1"))
    )
    layer.assign_task("u1", "task-a")
    layer.advance(1.0)
    assert not layer.available("u1")
    assert len(layer.released_tasks) == 1
    layer.advance(3.99)
    assert not layer.available("u1")
    layer.advance(4.0)
    assert layer.available("u1")
    assert len(layer.released_tasks) == 1
    layer.assign_task("u1", "task-b")
    assert layer.states["u1"].current_task == "task-b"


def test_energy_is_clamped_and_low_energy_degrades_speed_and_capability() -> None:
    cfg = DisturbanceConfig(
        energy_depletion=SourceConfig(
            True,
            {
                "travel_rate": 0.1,
                "execution_rate": 0.2,
                "standby_rate": 0.0,
                "communication_rate": 0.0,
                "low_threshold": 0.5,
                "minimum_speed_factor": 0.4,
                "minimum_capability_factor": 0.6,
            },
        )
    )
    layer = UAVDisturbanceLayer(("u0",), generate_uav_events(cfg, horizon=10.0, uav_ids=("u0",)))
    layer.advance(0.0)
    layer.consume_energy("u0", travel=6.0)
    state = layer.states["u0"]
    assert state.energy == pytest.approx(0.4)
    assert 0.4 < state.speed_factor < 1.0
    assert 0.6 < state.capability_factor < 1.0
    layer.assign_task("u0", "task-a")
    layer.consume_energy("u0", execution=100.0)
    assert state.energy == 0.0
    assert state.speed_factor == 0.0
    assert state.capability_factor == 0.0
    assert not layer.available("u0")
    assert layer.released_tasks[-1].reason == "energy_exhausted"


def test_failure_tape_is_deterministic_for_same_seed() -> None:
    cfg = DisturbanceConfig(
        instance_seed=12,
        disturbance_seed=34,
        uav_failure=SourceConfig(
            True,
            {"count": 4, "permanent_probability": 0.5, "minimum_time": 1.0, "maximum_time": 9.0},
        ),
    )
    left = generate_uav_events(cfg, horizon=10.0, uav_ids=("u0", "u1", "u2"))
    right = generate_uav_events(cfg, horizon=10.0, uav_ids=("u0", "u1", "u2"))
    assert left == right
    assert len(left) == 4


def test_time_cannot_reverse_and_activity_cannot_be_negative() -> None:
    layer = UAVDisturbanceLayer(("u0",), ())
    layer.advance(2.0)
    with pytest.raises(ValueError, match="backwards"):
        layer.advance(1.0)
    with pytest.raises(ValueError, match="negative"):
        layer.consume_energy("u0", travel=-1.0)

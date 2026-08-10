from __future__ import annotations

import pytest

from uav_assignment.disturbances import (
    DisturbanceConfig,
    SourceConfig,
    TaskDisturbanceLayer,
    TaskRuntimeState,
    generate_task_events,
)


def base_tasks() -> tuple[TaskRuntimeState, ...]:
    return (
        TaskRuntimeState("a"),
        TaskRuntimeState("b", predecessors={"a"}),
    )


def test_dynamic_arrival_enters_action_space_only_after_predecessor() -> None:
    cfg = DisturbanceConfig(
        task_arrival=SourceConfig(
            True,
            {"events": [{"time": 2.0, "task_id": "c", "predecessors": ["b"], "priority": 3.0}]},
        )
    )
    layer = TaskDisturbanceLayer(base_tasks(), generate_task_events(cfg, horizon=10.0))
    assert "c" not in layer.action_mask()
    layer.advance(2.0)
    assert layer.action_mask() == {"a": True, "b": False, "c": False}
    layer.start("a", "u0")
    layer.complete("a")
    assert layer.action_mask()["b"]
    layer.start("b", "u0")
    layer.complete("b")
    assert layer.action_mask()["c"]


def test_running_task_cancellation_releases_once_and_updates_successors() -> None:
    cfg = DisturbanceConfig(
        task_cancellation=SourceConfig(
            True,
            {"events": [{"time": 3.0, "task_id": "a"}, {"time": 4.0, "task_id": "a"}]},
        )
    )
    layer = TaskDisturbanceLayer(base_tasks(), generate_task_events(cfg, horizon=10.0))
    layer.start("a", "u0")
    layer.advance(3.0)
    assert layer.tasks["a"].status == "cancelled"
    assert layer.tasks["a"].assigned_uav is None
    assert layer.action_mask()["b"]
    assert len(layer.releases) == 1
    layer.advance(4.0)
    assert len(layer.releases) == 1


def test_priority_and_deadline_changes_are_applied_at_physical_time() -> None:
    cfg = DisturbanceConfig(
        task_priority_change=SourceConfig(
            True, {"events": [{"time": 1.0, "task_id": "a", "priority": 4.5}]}
        ),
        task_deadline_change=SourceConfig(
            True, {"events": [{"time": 2.0, "task_id": "a", "deadline": 8.0}]}
        ),
    )
    layer = TaskDisturbanceLayer(base_tasks(), generate_task_events(cfg, horizon=10.0))
    layer.advance(1.0)
    assert layer.tasks["a"].priority == 4.5
    assert layer.tasks["a"].deadline is None
    layer.advance(2.0)
    assert layer.tasks["a"].deadline == 8.0


def test_dag_cycle_and_unknown_predecessor_are_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        TaskDisturbanceLayer(
            (TaskRuntimeState("a", {"b"}), TaskRuntimeState("b", {"a"}))
        )
    cfg = DisturbanceConfig(
        task_arrival=SourceConfig(
            True, {"events": [{"time": 1.0, "task_id": "c", "predecessors": ["missing"]}]}
        )
    )
    layer = TaskDisturbanceLayer(base_tasks(), generate_task_events(cfg, horizon=5.0))
    with pytest.raises(ValueError, match="unknown predecessors"):
        layer.advance(1.0)
    assert "c" not in layer.tasks


def test_uav_failure_release_returns_task_to_pending_without_duplication() -> None:
    layer = TaskDisturbanceLayer(base_tasks())
    layer.start("a", "u0")
    assert layer.release_from_uav("u0", "uav_failed") == ("a",)
    assert layer.tasks["a"].status == "pending"
    assert layer.tasks["a"].assigned_uav is None
    assert layer.release_from_uav("u0", "uav_failed") == ()
    assert len(layer.releases) == 1


def test_cancelled_task_is_never_selectable_and_time_cannot_reverse() -> None:
    cfg = DisturbanceConfig(
        task_cancellation=SourceConfig(True, {"events": [{"time": 1.0, "task_id": "a"}]})
    )
    layer = TaskDisturbanceLayer(base_tasks(), generate_task_events(cfg, horizon=5.0))
    layer.advance(1.0)
    assert not layer.action_mask()["a"]
    with pytest.raises(ValueError, match="backwards"):
        layer.advance(0.5)

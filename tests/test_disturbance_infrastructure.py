from __future__ import annotations

import json

import pytest

from uav_assignment.disturbances import (
    DisturbanceConfig,
    DisturbanceEvent,
    DisturbanceLogger,
    DisturbanceTape,
    SourceConfig,
)


def event(index: int, time: float, *, priority: int = 100, kind: str = "wind") -> DisturbanceEvent:
    return DisturbanceEvent(
        event_id=f"event-{index}",
        event_type=kind,
        physical_time=time,
        source=kind,
        target="uav:0",
        severity=0.5,
        payload={"vector": [1.0, 0.0]},
        ground_truth={"active": True},
        observed_time=None,
        generation_index=index,
        source_priority=priority,
    )


def test_config_keeps_three_seed_domains_separate_and_hashes_canonically() -> None:
    config = DisturbanceConfig(
        severity="weak",
        instance_seed=11,
        disturbance_seed=22,
        training_seed=33,
        wind_field=SourceConfig(True, {"speed": 0.2, "cells": 4}),
    )
    restored = DisturbanceConfig.from_json(config.canonical_json())
    assert restored == config
    assert restored.sha256 == config.sha256
    assert len({config.source_seed(name) for name in ("wind_field", "message_delay")}) == 2
    assert config.source_seed("wind_field") == restored.source_seed("wind_field")


def test_disabled_config_produces_stable_empty_tape() -> None:
    config = DisturbanceConfig(instance_seed=7, disturbance_seed=9)
    assert config.all_disabled
    left = DisturbanceTape.empty(config)
    right = DisturbanceTape.empty(DisturbanceConfig.from_dict(config.to_dict()))
    assert left.events == ()
    assert left.canonical_json() == right.canonical_json()
    assert left.sha256 == right.sha256


def test_tape_has_deterministic_tie_break_and_round_trip_hash() -> None:
    config = DisturbanceConfig(instance_seed=1, disturbance_seed=2)
    events = (
        event(3, 4.0, priority=20, kind="wind"),
        event(1, 4.0, priority=10, kind="failure"),
        event(2, 2.0, priority=99, kind="delay"),
    )
    tape = DisturbanceTape.build(config, events)
    assert [item.event_id for item in tape.events] == ["event-2", "event-1", "event-3"]
    restored = DisturbanceTape.from_json(tape.canonical_json())
    assert restored == tape
    assert restored.sha256 == tape.sha256
    assert json.loads(tape.canonical_json())["config_sha256"] == config.sha256


def test_cursor_never_exposes_future_events_or_allows_time_reversal() -> None:
    config = DisturbanceConfig()
    cursor = DisturbanceTape.build(config, (event(0, 1.0), event(1, 3.0))).cursor()
    assert cursor.consume_until(0.99) == ()
    assert [item.event_id for item in cursor.consume_until(1.0)] == ["event-0"]
    assert cursor.consume_until(2.99) == ()
    assert [item.event_id for item in cursor.consume_until(3.0)] == ["event-1"]
    assert cursor.exhausted
    with pytest.raises(ValueError, match="backwards"):
        cursor.consume_until(2.0)


def test_logger_separates_physical_and_observed_time_and_is_append_only() -> None:
    source = event(0, 2.0)
    logger = DisturbanceLogger()
    applied = logger.record(
        source,
        observed_time=3.5,
        before_state={"energy": 1.0},
        after_state={"energy": 0.8},
        effect={"energy_delta": -0.2},
    )
    assert source.observed_time is None
    assert applied.physical_time == 2.0
    assert applied.observed_time == 3.5
    assert applied.before_state == {"energy": 1.0}
    assert applied.after_state == {"energy": 0.8}
    assert len(logger.sha256) == 64
    with pytest.raises(ValueError, match="already recorded"):
        logger.record(
            source,
            observed_time=4.0,
            before_state={},
            after_state={},
            effect={},
        )


def test_event_validation_rejects_observation_before_ground_truth() -> None:
    with pytest.raises(ValueError, match="precede"):
        DisturbanceEvent(
            event_id="bad",
            event_type="delay",
            physical_time=2.0,
            source="link",
            target="message:1",
            severity=1.0,
            payload={},
            ground_truth={},
            observed_time=1.0,
            generation_index=0,
        )

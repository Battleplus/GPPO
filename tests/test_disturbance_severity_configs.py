from __future__ import annotations

import math
from pathlib import Path

from uav_assignment.disturbances import (
    SOURCE_NAMES,
    DisturbanceConfig,
    DisturbanceEngine,
    TaskRuntimeState,
)


ROOT = Path(__file__).resolve().parents[1]


def configs() -> list[DisturbanceConfig]:
    return [
        DisturbanceConfig.from_json(
            (ROOT / f"configs/disturbance_{severity}.json").read_text(encoding="utf-8")
        )
        for severity in ("weak", "medium", "strong")
    ]


def test_frozen_severity_configs_enable_every_source_and_keep_seed_domains_explicit() -> None:
    values = configs()
    assert [item.severity for item in values] == ["weak", "medium", "strong"]
    assert all(not item.all_disabled for item in values)
    assert all(item.instance_seed == 0 and item.disturbance_seed == 0 for item in values)
    assert all(item.training_seed is None for item in values)
    assert len({item.sha256 for item in values}) == 3


def test_physical_parameters_are_monotonic_from_weak_to_strong() -> None:
    weak, medium, strong = configs()
    ordered = (weak, medium, strong)
    def parameter(source: str, name: str) -> list[float]:
        return [float(getattr(item, source).parameters[name]) for item in ordered]
    assert parameter("gilbert_elliott_packet_loss", "p_gb") == sorted(parameter("gilbert_elliott_packet_loss", "p_gb"))
    assert parameter("gilbert_elliott_packet_loss", "p_bg") == sorted(parameter("gilbert_elliott_packet_loss", "p_bg"), reverse=True)
    assert parameter("gilbert_elliott_packet_loss", "loss_bad") == sorted(parameter("gilbert_elliott_packet_loss", "loss_bad"))
    assert parameter("message_delay", "maximum") == sorted(parameter("message_delay", "maximum"))
    assert parameter("energy_depletion", "travel_rate") == sorted(parameter("energy_depletion", "travel_rate"))
    assert parameter("energy_depletion", "execution_rate") == sorted(parameter("energy_depletion", "execution_rate"))
    assert parameter("energy_depletion", "low_threshold") == sorted(parameter("energy_depletion", "low_threshold"))
    assert parameter("wind_field", "speed") == sorted(parameter("wind_field", "speed"))
    partition_durations = [
        item.network_partition.parameters["intervals"][0]["end"]
        - item.network_partition.parameters["intervals"][0]["start"]
        for item in ordered
    ]
    assert partition_durations == sorted(partition_durations)
    assert [len(item.uav_failure.parameters["events"]) for item in ordered] == [1, 2, 3]
    assert [len(item.task_arrival.parameters["events"]) for item in ordered] == [1, 2, 3]


def test_each_frozen_config_generates_a_deterministic_finite_combined_tape() -> None:
    tasks = tuple(TaskRuntimeState(f"t{index}") for index in range(5))
    for config in configs():
        kwargs = {
            "horizon": 100.0,
            "uav_ids": tuple(f"u{index}" for index in range(5)),
            "tasks": tasks,
        }
        left = DisturbanceEngine(config, **kwargs)
        right = DisturbanceEngine(config, **kwargs)
        assert left.tape.sha256 == right.tape.sha256
        assert len(left.tape.events) > 0
        assert all(math.isfinite(event.physical_time) for event in left.tape.events)
        assert all(event.physical_time <= 100.0 for event in left.tape.events)


def test_medium_single_category_configs_match_combined_physics() -> None:
    combined = configs()[1]
    expected = {
        "communication": {
            "gilbert_elliott_packet_loss", "message_delay", "network_partition"
        },
        "uav": {"uav_failure", "energy_depletion"},
        "task": {
            "task_arrival", "task_cancellation", "task_priority_change",
            "task_deadline_change",
        },
        "wind": {"wind_field"},
    }
    for category, enabled in expected.items():
        config = DisturbanceConfig.from_json(
            (ROOT / f"configs/disturbance_{category}_medium.json").read_text(
                encoding="utf-8"
            )
        )
        actual = {name for name in SOURCE_NAMES if getattr(config, name).enabled}
        assert actual == enabled
        for name in enabled:
            assert getattr(config, name).parameters == getattr(combined, name).parameters

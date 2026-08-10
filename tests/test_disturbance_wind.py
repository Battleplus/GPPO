from __future__ import annotations

import math

import pytest

from uav_assignment.disturbances import (
    DisturbanceConfig,
    SourceConfig,
    WindFieldLayer,
    generate_wind_events,
)


def layer(vector: tuple[float, float]) -> WindFieldLayer:
    cfg = DisturbanceConfig(
        instance_seed=10,
        disturbance_seed=20,
        wind_field=SourceConfig(True, {"vector": vector}),
    )
    return WindFieldLayer(cfg, generate_wind_events(cfg, horizon=10.0))


def test_tailwind_reduces_travel_time_and_energy_and_headwind_increases_them() -> None:
    tail = layer((0.5, 0.0)).adjust_travel(
        (0.0, 0.0), (1.0, 0.0), airspeed=1.0, base_energy_per_distance=1.0
    )
    head = layer((-0.5, 0.0)).adjust_travel(
        (0.0, 0.0), (1.0, 0.0), airspeed=1.0, base_energy_per_distance=1.0
    )
    assert tail.disturbed_travel_time < tail.base_travel_time
    assert tail.disturbed_energy < tail.base_energy
    assert head.disturbed_travel_time > head.base_travel_time
    assert head.disturbed_energy > head.base_energy


def test_crosswind_has_zero_along_track_component() -> None:
    result = layer((0.0, 0.8)).adjust_travel(
        (0.0, 0.0), (1.0, 0.0), airspeed=1.0, base_energy_per_distance=2.0
    )
    assert result.along_track_wind == pytest.approx(0.0)
    assert result.disturbed_travel_time == pytest.approx(result.base_travel_time)
    assert result.disturbed_energy == pytest.approx(result.base_energy)


def test_extreme_headwind_remains_finite_and_non_negative() -> None:
    result = layer((-100.0, 0.0)).adjust_travel(
        (0.0, 0.0), (1.0, 0.0), airspeed=1.0, base_energy_per_distance=1.0
    )
    assert math.isfinite(result.disturbed_travel_time)
    assert math.isfinite(result.disturbed_energy)
    assert result.disturbed_travel_time > 0
    assert result.disturbed_energy >= 0


def test_true_and_observed_wind_are_separate_and_deterministic() -> None:
    wind = layer((0.2, -0.1))
    first = wind.observe(
        (0.5, 0.5), physical_time=2.0, observed_time=2.5,
        noise_std=0.3, observation_id="obs-7",
    )
    second = wind.observe(
        (0.5, 0.5), physical_time=2.0, observed_time=2.5,
        noise_std=0.3, observation_id="obs-7",
    )
    assert first == second
    assert first.true_vector == pytest.approx((0.2, -0.1))
    assert first.observed_vector != first.true_vector
    assert wind.true_vector_at((0.5, 0.5), 2.0) == pytest.approx((0.2, -0.1))


def test_spatial_cells_and_time_segments_are_respected() -> None:
    cfg = DisturbanceConfig(
        wind_field=SourceConfig(
            True,
            {"segments": [
                {"start": 0.0, "end": 5.0, "vector": [1.0, 0.0], "bounds": [0, 0, 0.5, 1]},
                {"start": 5.0, "end": 10.0, "vector": [0.0, 2.0], "bounds": [0, 0, 1, 1]},
            ]},
        )
    )
    wind = WindFieldLayer(cfg, generate_wind_events(cfg, horizon=10.0))
    assert wind.true_vector_at((0.25, 0.5), 2.0) == pytest.approx((1.0, 0.0))
    assert wind.true_vector_at((0.75, 0.5), 2.0) == pytest.approx((0.0, 0.0))
    assert wind.true_vector_at((0.75, 0.5), 6.0) == pytest.approx((0.0, 2.0))


def test_wind_time_and_observation_validation() -> None:
    wind = layer((0.0, 0.0))
    wind.advance(3.0)
    with pytest.raises(ValueError, match="backwards"):
        wind.advance(2.0)
    with pytest.raises(ValueError, match="precede"):
        wind.observe((0.0, 0.0), physical_time=3.0, observed_time=2.0)

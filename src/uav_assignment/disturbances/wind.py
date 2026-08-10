from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .config import DisturbanceConfig
from .events import DisturbanceEvent


@dataclass(frozen=True, slots=True)
class WindAdjustment:
    distance: float
    base_travel_time: float
    disturbed_travel_time: float
    base_energy: float
    disturbed_energy: float
    along_track_wind: float
    true_wind_vector: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WindObservation:
    physical_time: float
    observed_time: float
    position: tuple[float, float]
    true_vector: tuple[float, float]
    observed_vector: tuple[float, float]


def _vector(specification: Mapping[str, Any]) -> tuple[float, float]:
    if "vector" in specification:
        x, y = specification["vector"]
        return float(x), float(y)
    speed = float(specification.get("speed", 0.0))
    direction = float(specification.get("direction_radians", 0.0))
    if speed < 0:
        raise ValueError("wind speed cannot be negative")
    return speed * math.cos(direction), speed * math.sin(direction)


def generate_wind_events(
    config: DisturbanceConfig,
    *,
    horizon: float,
) -> tuple[DisturbanceEvent, ...]:
    if horizon < 0:
        raise ValueError("horizon cannot be negative")
    if not config.wind_field.enabled:
        return ()
    params = dict(config.wind_field.parameters)
    segments = list(params.get("segments", []))
    if not segments:
        segments = [{
            "start": 0.0,
            "end": horizon,
            "vector": list(_vector(params)),
            "bounds": [0.0, 0.0, 1.0, 1.0],
        }]
    events: list[DisturbanceEvent] = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        vector = _vector(segment)
        bounds = tuple(float(value) for value in segment.get("bounds", (0.0, 0.0, 1.0, 1.0)))
        if len(bounds) != 4 or bounds[0] > bounds[2] or bounds[1] > bounds[3]:
            raise ValueError("wind bounds must be xmin,ymin,xmax,ymax")
        if start < 0 or end <= start or end > horizon:
            raise ValueError("invalid wind segment interval")
        speed = math.hypot(*vector)
        events.append(
            DisturbanceEvent(
                event_id=f"wind:{index}",
                event_type="wind_field",
                physical_time=start,
                source="wind_field",
                target=f"cell:{index}",
                severity=speed,
                payload={"start": start, "end": end, "vector": vector, "bounds": bounds},
                ground_truth={"wind_vector": vector},
                observed_time=None,
                generation_index=index,
                source_priority=60,
            )
        )
    return tuple(sorted(events, key=lambda item: item.sort_key))


class WindFieldLayer:
    """Ground-truth wind physics plus a separate, optional noisy observation channel."""

    def __init__(
        self,
        config: DisturbanceConfig,
        events: Iterable[DisturbanceEvent],
        *,
        minimum_ground_speed: float = 0.05,
        energy_sensitivity: float = 0.5,
        minimum_energy_factor: float = 0.25,
        maximum_energy_factor: float = 4.0,
    ):
        if minimum_ground_speed <= 0 or energy_sensitivity < 0:
            raise ValueError("invalid wind physics parameters")
        if not 0 < minimum_energy_factor <= maximum_energy_factor:
            raise ValueError("invalid energy factor bounds")
        self.config = config
        self.events = tuple(sorted(events, key=lambda item: item.sort_key))
        self.minimum_ground_speed = float(minimum_ground_speed)
        self.energy_sensitivity = float(energy_sensitivity)
        self.minimum_energy_factor = float(minimum_energy_factor)
        self.maximum_energy_factor = float(maximum_energy_factor)
        self.physical_time = 0.0

    def advance(self, physical_time: float) -> None:
        physical_time = float(physical_time)
        if physical_time < self.physical_time:
            raise ValueError("wind field time cannot move backwards")
        self.physical_time = physical_time

    def true_vector_at(
        self, position: tuple[float, float], physical_time: float | None = None
    ) -> tuple[float, float]:
        time = self.physical_time if physical_time is None else float(physical_time)
        x, y = map(float, position)
        total = np.zeros(2, dtype=np.float64)
        for event in self.events:
            start = float(event.payload["start"])
            end = float(event.payload["end"])
            xmin, ymin, xmax, ymax = map(float, event.payload["bounds"])
            if start <= time < end and xmin <= x <= xmax and ymin <= y <= ymax:
                total += np.asarray(event.payload["vector"], dtype=np.float64)
        return float(total[0]), float(total[1])

    def adjust_travel(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        airspeed: float,
        base_energy_per_distance: float,
        physical_time: float | None = None,
    ) -> WindAdjustment:
        if airspeed <= 0 or base_energy_per_distance < 0:
            raise ValueError("airspeed must be positive and energy rate non-negative")
        displacement = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        distance = float(np.linalg.norm(displacement))
        wind = self.true_vector_at(
            tuple(((np.asarray(start) + np.asarray(end)) / 2).tolist()), physical_time
        )
        if distance == 0:
            return WindAdjustment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, wind)
        heading = displacement / distance
        along_track = float(np.dot(np.asarray(wind), heading))
        ground_speed = max(self.minimum_ground_speed, airspeed + along_track)
        base_time = distance / airspeed
        disturbed_time = distance / ground_speed
        base_energy = distance * base_energy_per_distance
        factor = 1.0 - self.energy_sensitivity * along_track / airspeed
        factor = min(self.maximum_energy_factor, max(self.minimum_energy_factor, factor))
        disturbed_energy = base_energy * factor
        values = (base_time, disturbed_time, base_energy, disturbed_energy)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise FloatingPointError("wind adjustment produced a non-finite physical value")
        return WindAdjustment(
            distance,
            base_time,
            disturbed_time,
            base_energy,
            disturbed_energy,
            along_track,
            wind,
        )

    def observe(
        self,
        position: tuple[float, float],
        *,
        observed_time: float,
        physical_time: float | None = None,
        noise_std: float = 0.0,
        observation_id: str = "0",
    ) -> WindObservation:
        truth_time = self.physical_time if physical_time is None else float(physical_time)
        if observed_time < truth_time or noise_std < 0:
            raise ValueError("wind observation cannot precede truth or use negative noise")
        truth = self.true_vector_at(position, truth_time)
        seed_material = (
            f"{self.config.source_seed('wind_field')}|{observation_id}|{truth_time}|{position}"
        )
        seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
        noise = np.random.default_rng(seed).normal(0.0, noise_std, size=2)
        observed = tuple(float(value) for value in np.asarray(truth) + noise)
        return WindObservation(
            physical_time=truth_time,
            observed_time=float(observed_time),
            position=tuple(map(float, position)),
            true_vector=truth,
            observed_vector=observed,
        )

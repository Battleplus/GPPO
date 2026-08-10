from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .config import DisturbanceConfig
from .events import DisturbanceEvent


@dataclass(frozen=True, slots=True)
class DisturbanceTape:
    version: str
    config_sha256: str
    instance_seed: int
    disturbance_seed: int
    events: tuple[DisturbanceEvent, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.events, key=lambda event: event.sort_key))
        if ordered != self.events:
            raise ValueError("events must already follow the deterministic tape ordering")
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("event_id values must be unique within a tape")

    @classmethod
    def build(
        cls,
        config: DisturbanceConfig,
        events: Iterable[DisturbanceEvent],
    ) -> "DisturbanceTape":
        ordered = tuple(sorted(events, key=lambda event: event.sort_key))
        return cls(
            version="phase1b-disturbance-tape-v1",
            config_sha256=config.sha256,
            instance_seed=config.instance_seed,
            disturbance_seed=config.disturbance_seed,
            events=ordered,
        )

    @classmethod
    def empty(cls, config: DisturbanceConfig) -> "DisturbanceTape":
        return cls.build(config, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config_sha256": self.config_sha256,
            "instance_seed": self.instance_seed,
            "disturbance_seed": self.disturbance_seed,
            "events": [event.to_dict() for event in self.events],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DisturbanceTape":
        return cls(
            version=str(payload["version"]),
            config_sha256=str(payload["config_sha256"]),
            instance_seed=int(payload["instance_seed"]),
            disturbance_seed=int(payload["disturbance_seed"]),
            events=tuple(DisturbanceEvent.from_dict(item) for item in payload["events"]),
        )

    @classmethod
    def from_json(cls, text: str) -> "DisturbanceTape":
        return cls.from_dict(json.loads(text))

    def cursor(self) -> "DisturbanceTapeCursor":
        return DisturbanceTapeCursor(self)


class DisturbanceTapeCursor:
    """Monotonic consumer: future events can never become visible early."""

    def __init__(self, tape: DisturbanceTape):
        self.tape = tape
        self.index = 0
        self.physical_time = 0.0

    def consume_until(self, physical_time: float) -> tuple[DisturbanceEvent, ...]:
        physical_time = float(physical_time)
        if physical_time < self.physical_time:
            raise ValueError("disturbance tape time cannot move backwards")
        start = self.index
        while self.index < len(self.tape.events):
            if self.tape.events[self.index].physical_time > physical_time:
                break
            self.index += 1
        self.physical_time = physical_time
        return self.tape.events[start : self.index]

    @property
    def exhausted(self) -> bool:
        return self.index == len(self.tape.events)

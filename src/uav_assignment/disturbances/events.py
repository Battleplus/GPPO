from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _json_value(value: Any) -> JsonValue:
    """Return a detached, canonical-JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(encoded)


@dataclass(frozen=True, slots=True)
class DisturbanceEvent:
    """An immutable ground-truth event generated before an episode starts."""

    event_id: str
    event_type: str
    physical_time: float
    source: str
    target: str
    severity: float
    payload: Mapping[str, Any]
    ground_truth: Mapping[str, Any]
    observed_time: float | None
    generation_index: int
    source_priority: int = 100
    before_state: Mapping[str, Any] | None = None
    after_state: Mapping[str, Any] | None = None
    effect: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not self.event_type or not self.source or not self.target:
            raise ValueError("event_type, source and target must be non-empty")
        if self.physical_time < 0:
            raise ValueError("physical_time cannot be negative")
        if self.observed_time is not None and self.observed_time < self.physical_time:
            raise ValueError("observed_time cannot precede physical_time")
        if self.generation_index < 0:
            raise ValueError("generation_index cannot be negative")
        if self.source_priority < 0:
            raise ValueError("source_priority cannot be negative")
        if self.severity < 0:
            raise ValueError("severity cannot be negative")
        object.__setattr__(self, "payload", _json_value(dict(self.payload)))
        object.__setattr__(self, "ground_truth", _json_value(dict(self.ground_truth)))
        if self.before_state is not None:
            object.__setattr__(self, "before_state", _json_value(dict(self.before_state)))
        if self.after_state is not None:
            object.__setattr__(self, "after_state", _json_value(dict(self.after_state)))
        if self.effect is not None:
            object.__setattr__(self, "effect", _json_value(dict(self.effect)))

    @property
    def sort_key(self) -> tuple[float, int, str, str, int, str]:
        return (
            float(self.physical_time),
            int(self.source_priority),
            self.event_type,
            self.target,
            int(self.generation_index),
            self.event_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DisturbanceEvent":
        return cls(**dict(payload))

    def applied(
        self,
        *,
        observed_time: float,
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
        effect: Mapping[str, Any],
    ) -> "DisturbanceEvent":
        return replace(
            self,
            observed_time=float(observed_time),
            before_state=before_state,
            after_state=after_state,
            effect=effect,
        )

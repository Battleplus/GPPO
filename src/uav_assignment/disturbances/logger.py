from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .events import DisturbanceEvent


class DisturbanceLogger:
    """Append-only audit log of event application and observation delay."""

    def __init__(self) -> None:
        self._records: list[DisturbanceEvent] = []
        self._event_ids: set[str] = set()
        self._last_observed_time = 0.0

    @property
    def records(self) -> tuple[DisturbanceEvent, ...]:
        return tuple(self._records)

    def record(
        self,
        event: DisturbanceEvent,
        *,
        observed_time: float,
        before_state: Mapping[str, Any],
        after_state: Mapping[str, Any],
        effect: Mapping[str, Any],
    ) -> DisturbanceEvent:
        if event.event_id in self._event_ids:
            raise ValueError(f"event already recorded: {event.event_id}")
        if observed_time < self._last_observed_time:
            raise ValueError("observed event time cannot move backwards")
        applied = event.applied(
            observed_time=observed_time,
            before_state=before_state,
            after_state=after_state,
            effect=effect,
        )
        self._records.append(applied)
        self._event_ids.add(event.event_id)
        self._last_observed_time = float(observed_time)
        return applied

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "phase1b-disturbance-log-v1",
            "records": [event.to_dict() for event in self._records],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

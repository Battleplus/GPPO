from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .communication import CommunicationDisturbanceLayer, generate_communication_events
from .config import DisturbanceConfig
from .events import DisturbanceEvent
from .logger import DisturbanceLogger
from .tape import DisturbanceTape
from .task import TaskDisturbanceLayer, TaskRuntimeState, generate_task_events
from .uav import UAVDisturbanceLayer, generate_uav_events
from .wind import WindFieldLayer, generate_wind_events


@dataclass(frozen=True, slots=True)
class DisturbanceStep:
    physical_time: float
    current_events: tuple[DisturbanceEvent, ...]
    tape_index: int
    tape_exhausted: bool


class DisturbanceEngine:
    """Episode-scoped coordinator for a fully pre-generated disturbance tape."""

    def __init__(
        self,
        config: DisturbanceConfig,
        *,
        horizon: float,
        uav_ids: Iterable[str],
        tasks: Iterable[TaskRuntimeState],
        link_ids: Iterable[str] | None = None,
        initial_leader: str | None = None,
    ):
        if horizon < 0:
            raise ValueError("horizon cannot be negative")
        self.config = config
        self.horizon = float(horizon)
        self.uav_ids = tuple(sorted(set(str(value) for value in uav_ids)))
        if not self.uav_ids:
            raise ValueError("disturbance engine requires at least one UAV")
        self.link_ids = tuple(
            sorted(set(link_ids or self._complete_link_ids(self.uav_ids)))
        )
        task_list = list(tasks)
        events = (
            generate_communication_events(
                config, horizon=self.horizon, link_ids=self.link_ids
            )
            + generate_uav_events(
                config, horizon=self.horizon, uav_ids=self.uav_ids
            )
            + generate_task_events(config, horizon=self.horizon)
            + generate_wind_events(config, horizon=self.horizon)
        )
        self.tape = DisturbanceTape.build(config, events)
        self.cursor = self.tape.cursor()
        self.logger = DisturbanceLogger()
        communication_events = tuple(
            event for event in self.tape.events
            if event.event_type in {"link_state", "delay_profile", "network_partition"}
        )
        uav_events = tuple(
            event for event in self.tape.events
            if event.event_type in {"energy_profile", "uav_failure", "uav_recovery"}
        )
        task_events = tuple(
            event for event in self.tape.events if event.event_type.startswith("task_")
        )
        wind_events = tuple(
            event for event in self.tape.events if event.event_type == "wind_field"
        )
        self.communication = CommunicationDisturbanceLayer(config, communication_events)
        self.uav = UAVDisturbanceLayer(
            self.uav_ids, uav_events, initial_leader=initial_leader
        )
        self.task = TaskDisturbanceLayer(task_list, task_events)
        self.wind = WindFieldLayer(config, wind_events)

    @staticmethod
    def _complete_link_ids(uav_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            f"{source}->{target}"
            for source in uav_ids
            for target in uav_ids
            if source != target
        )

    def advance(self, physical_time: float) -> DisturbanceStep:
        current = self.cursor.consume_until(physical_time)
        self.communication.advance(physical_time)
        self.uav.advance(physical_time)
        self.task.advance(physical_time)
        self.wind.advance(physical_time)
        return DisturbanceStep(
            physical_time=float(physical_time),
            current_events=current,
            tape_index=self.cursor.index,
            tape_exhausted=self.cursor.exhausted,
        )

    def future_event_targets(
        self,
        decision_times: Iterable[float],
        *,
        current_decision_index: int,
        horizons: int = 5,
    ) -> tuple[dict[str, Any], ...]:
        """Ground-truth events in each of the next 1..N decision windows."""

        times = tuple(float(value) for value in decision_times)
        if horizons < 1:
            raise ValueError("horizons must be positive")
        if any(right < left for left, right in zip(times, times[1:])):
            raise ValueError("decision times cannot move backwards")
        if not 0 <= current_decision_index < len(times):
            raise IndexError("current_decision_index is outside decision_times")
        targets: list[dict[str, Any]] = []
        for offset in range(1, horizons + 1):
            target_index = current_decision_index + offset
            if target_index >= len(times):
                targets.append(
                    {"horizon": offset, "available": False, "events": []}
                )
                continue
            start = times[target_index - 1]
            end = times[target_index]
            events = [
                event.to_dict()
                for event in self.tape.events
                if start < event.physical_time <= end
            ]
            targets.append(
                {
                    "horizon": offset,
                    "available": True,
                    "decision_time": end,
                    "events": events,
                }
            )
        return tuple(targets)

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": "phase1b-disturbance-engine-v1",
            "physical_time": self.cursor.physical_time,
            "config_sha256": self.config.sha256,
            "tape_sha256": self.tape.sha256,
            "tape_event_count": len(self.tape.events),
            "tape_index": self.cursor.index,
            "uav": self.uav.snapshot(),
            "task": self.task.snapshot(),
            "communication": self.communication.audit.to_dict(),
        }

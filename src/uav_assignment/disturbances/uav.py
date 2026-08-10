from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from .config import DisturbanceConfig
from .events import DisturbanceEvent


@dataclass(slots=True)
class UAVRuntimeState:
    uav_id: str
    energy_capacity: float = 1.0
    energy: float = 1.0
    alive: bool = True
    unavailable_until: float = 0.0
    speed_factor: float = 1.0
    capability_factor: float = 1.0
    current_task: str | None = None
    is_leader: bool = False

    @property
    def temporarily_unavailable(self) -> bool:
        return self.unavailable_until > 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleasedTask:
    task_id: str
    uav_id: str
    physical_time: float
    reason: str


def generate_uav_events(
    config: DisturbanceConfig,
    *,
    horizon: float,
    uav_ids: Iterable[str],
) -> tuple[DisturbanceEvent, ...]:
    if horizon < 0:
        raise ValueError("horizon cannot be negative")
    ids = tuple(sorted(set(str(value) for value in uav_ids)))
    if not ids:
        raise ValueError("at least one UAV is required")
    events: list[DisturbanceEvent] = []
    generation_index = 0
    failure = config.uav_failure
    if failure.enabled:
        params = dict(failure.parameters)
        specifications = list(params.get("events", []))
        if not specifications:
            count = int(params.get("count", 1))
            permanent_probability = float(params.get("permanent_probability", 0.5))
            minimum_time = float(params.get("minimum_time", 0.1 * horizon))
            maximum_time = float(params.get("maximum_time", 0.8 * horizon))
            duration_min = float(params.get("duration_min", 1.0))
            duration_max = float(params.get("duration_max", max(duration_min, 3.0)))
            if count < 0 or not 0 <= permanent_probability <= 1:
                raise ValueError("invalid failure generation parameters")
            if minimum_time < 0 or maximum_time < minimum_time or maximum_time > horizon:
                raise ValueError("invalid failure time range")
            if duration_min <= 0 or duration_max < duration_min:
                raise ValueError("invalid temporary failure duration")
            rng = np.random.default_rng(config.source_seed("uav_failure"))
            for _ in range(count):
                permanent = bool(rng.random() < permanent_probability)
                specifications.append(
                    {
                        "time": float(rng.uniform(minimum_time, maximum_time)),
                        "uav_id": ids[int(rng.integers(0, len(ids)))],
                        "permanent": permanent,
                        "duration": None if permanent else float(rng.uniform(duration_min, duration_max)),
                    }
                )
        for index, specification in enumerate(specifications):
            physical_time = float(specification["time"])
            uav_id = str(specification["uav_id"])
            permanent = bool(specification.get("permanent", True))
            duration = None if permanent else float(specification["duration"])
            if not 0 <= physical_time <= horizon or uav_id not in ids:
                raise ValueError("failure event is outside the episode or targets an unknown UAV")
            if duration is not None and duration <= 0:
                raise ValueError("temporary failure duration must be positive")
            events.append(
                DisturbanceEvent(
                    event_id=f"uav-failure:{index}:{uav_id}",
                    event_type="uav_failure",
                    physical_time=physical_time,
                    source="uav_failure",
                    target=uav_id,
                    severity=1.0 if permanent else min(1.0, duration / max(horizon, 1.0)),
                    payload={"permanent": permanent, "duration": duration},
                    ground_truth={
                        "alive_after": not permanent,
                        "recovery_time": None if permanent else physical_time + float(duration),
                    },
                    observed_time=None,
                    generation_index=generation_index,
                    source_priority=40,
                )
            )
            generation_index += 1
            if duration is not None:
                recovery_time = physical_time + duration
                if recovery_time <= horizon:
                    events.append(
                        DisturbanceEvent(
                            event_id=f"uav-recovery:{index}:{uav_id}",
                            event_type="uav_recovery",
                            physical_time=recovery_time,
                            source="uav_failure",
                            target=uav_id,
                            severity=0.0,
                            payload={"failure_event_id": f"uav-failure:{index}:{uav_id}"},
                            ground_truth={"available_after": True},
                            observed_time=None,
                            generation_index=generation_index,
                            source_priority=41,
                        )
                    )
                    generation_index += 1
    energy = config.energy_depletion
    if energy.enabled:
        params = {
            "travel_rate": 0.01,
            "execution_rate": 0.02,
            "standby_rate": 0.001,
            "communication_rate": 0.0001,
            "low_threshold": 0.25,
            "minimum_speed_factor": 0.4,
            "minimum_capability_factor": 0.5,
            **dict(energy.parameters),
        }
        numeric = [float(value) for value in params.values()]
        if any(value < 0 for value in numeric) or not 0 < float(params["low_threshold"]) <= 1:
            raise ValueError("invalid energy profile")
        events.append(
            DisturbanceEvent(
                event_id="energy:profile",
                event_type="energy_profile",
                physical_time=0.0,
                source="energy_depletion",
                target="fleet",
                severity=float(params["low_threshold"]),
                payload=params,
                ground_truth={"energy_unit": "normalized_capacity"},
                observed_time=None,
                generation_index=generation_index,
                source_priority=35,
            )
        )
    return tuple(sorted(events, key=lambda item: item.sort_key))


class UAVDisturbanceLayer:
    """Owns disturbance-only UAV state without mutating task or graph semantics directly."""

    DEFAULT_PROFILE = {
        "travel_rate": 0.0,
        "execution_rate": 0.0,
        "standby_rate": 0.0,
        "communication_rate": 0.0,
        "low_threshold": 0.25,
        "minimum_speed_factor": 0.4,
        "minimum_capability_factor": 0.5,
    }

    def __init__(
        self,
        uav_ids: Iterable[str],
        events: Iterable[DisturbanceEvent],
        *,
        energy_capacity: float = 1.0,
        initial_leader: str | None = None,
    ):
        ids = tuple(sorted(set(str(value) for value in uav_ids)))
        if not ids or energy_capacity <= 0:
            raise ValueError("a non-empty fleet and positive energy capacity are required")
        self.states = {
            uav_id: UAVRuntimeState(uav_id, energy_capacity, energy_capacity)
            for uav_id in ids
        }
        leader = initial_leader or ids[0]
        if leader not in self.states:
            raise ValueError("initial leader is not in the fleet")
        self.states[leader].is_leader = True
        self.events = tuple(sorted(events, key=lambda item: item.sort_key))
        self._event_index = 0
        self.physical_time = 0.0
        self.energy_profile = dict(self.DEFAULT_PROFILE)
        self.released_tasks: list[ReleasedTask] = []
        self.applied_event_ids: list[str] = []

    @property
    def leader_id(self) -> str | None:
        return next((key for key, value in self.states.items() if value.is_leader), None)

    def available(self, uav_id: str, physical_time: float | None = None) -> bool:
        state = self.states[uav_id]
        now = self.physical_time if physical_time is None else float(physical_time)
        return state.alive and state.energy > 0 and now >= state.unavailable_until

    def assign_task(self, uav_id: str, task_id: str) -> None:
        if not self.available(uav_id):
            raise ValueError("unavailable UAV cannot receive a task")
        state = self.states[uav_id]
        if state.current_task is not None:
            raise ValueError("UAV already owns a task")
        state.current_task = str(task_id)

    def _release_task(self, state: UAVRuntimeState, reason: str) -> None:
        if state.current_task is None:
            return
        self.released_tasks.append(
            ReleasedTask(state.current_task, state.uav_id, self.physical_time, reason)
        )
        state.current_task = None

    def _elect_leader(self) -> None:
        eligible = [key for key in sorted(self.states) if self.available(key)]
        for state in self.states.values():
            state.is_leader = False
        if eligible:
            self.states[eligible[0]].is_leader = True

    def _apply_failure(self, event: DisturbanceEvent) -> None:
        state = self.states[event.target]
        permanent = bool(event.payload["permanent"])
        self._release_task(state, "permanent_failure" if permanent else "temporary_failure")
        if permanent:
            state.alive = False
            state.unavailable_until = 0.0
        else:
            state.unavailable_until = max(
                state.unavailable_until,
                event.physical_time + float(event.payload["duration"]),
            )
        if state.is_leader:
            self._elect_leader()

    def advance(self, physical_time: float) -> tuple[DisturbanceEvent, ...]:
        physical_time = float(physical_time)
        if physical_time < self.physical_time:
            raise ValueError("UAV disturbance time cannot move backwards")
        self.physical_time = physical_time
        applied: list[DisturbanceEvent] = []
        while self._event_index < len(self.events):
            event = self.events[self._event_index]
            if event.physical_time > physical_time:
                break
            if event.event_type == "energy_profile":
                self.energy_profile = {key: float(value) for key, value in event.payload.items()}
            elif event.event_type == "uav_failure":
                if event.target not in self.states:
                    raise ValueError(f"failure targets unknown UAV: {event.target}")
                self._apply_failure(event)
            elif event.event_type == "uav_recovery":
                state = self.states[event.target]
                if state.alive and state.energy > 0:
                    state.unavailable_until = 0.0
            self.applied_event_ids.append(event.event_id)
            applied.append(event)
            self._event_index += 1
        for state in self.states.values():
            if state.unavailable_until and physical_time >= state.unavailable_until:
                state.unavailable_until = 0.0
        if self.leader_id is None or not self.available(self.leader_id):
            self._elect_leader()
        return tuple(applied)

    def consume_energy(
        self,
        uav_id: str,
        *,
        travel: float = 0.0,
        execution: float = 0.0,
        standby: float = 0.0,
        communication_bytes: int = 0,
    ) -> float:
        state = self.states[uav_id]
        if min(travel, execution, standby, communication_bytes) < 0:
            raise ValueError("energy activity cannot be negative")
        consumed = (
            travel * self.energy_profile["travel_rate"]
            + execution * self.energy_profile["execution_rate"]
            + standby * self.energy_profile["standby_rate"]
            + communication_bytes * self.energy_profile["communication_rate"]
        )
        state.energy = max(0.0, min(state.energy_capacity, state.energy - consumed))
        threshold = state.energy_capacity * self.energy_profile["low_threshold"]
        if state.energy <= 0:
            self._release_task(state, "energy_exhausted")
            state.speed_factor = 0.0
            state.capability_factor = 0.0
            if state.is_leader:
                self._elect_leader()
        elif state.energy < threshold:
            ratio = state.energy / max(threshold, 1e-12)
            state.speed_factor = self.energy_profile["minimum_speed_factor"] + (
                1.0 - self.energy_profile["minimum_speed_factor"]
            ) * ratio
            state.capability_factor = self.energy_profile["minimum_capability_factor"] + (
                1.0 - self.energy_profile["minimum_capability_factor"]
            ) * ratio
        else:
            state.speed_factor = 1.0
            state.capability_factor = 1.0
        return float(consumed)

    def snapshot(self) -> dict[str, Any]:
        return {
            "physical_time": self.physical_time,
            "leader_id": self.leader_id,
            "uavs": {key: value.to_dict() for key, value in sorted(self.states.items())},
            "released_tasks": [asdict(item) for item in self.released_tasks],
        }

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .disturbances import (
    DisturbanceConfig,
    DisturbanceEngine,
    DisturbanceEvent,
    TaskRuntimeState,
)
from .paper_env import EventRecord, PaperTaskState, PaperUAVState
from .paper_faithful_env import (
    FaithfulSyncMode,
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
)


class Phase1BPaperFaithfulUAVEnv(PaperFaithfulUAVEnv):
    """Frozen paper-faithful environment with an optional Phase-1B physical layer.

    The all-disabled branch delegates byte-for-byte to the frozen baseline.  Active
    sources are consumed from a complete, pre-generated tape at physical boundaries.
    """

    def __init__(
        self,
        config: PaperFaithfulConfig | None = None,
        disturbance_config: DisturbanceConfig | None = None,
        *,
        disturbance_horizon: float = 200.0,
    ):
        self.disturbance_template = disturbance_config or DisturbanceConfig()
        self.disturbance_horizon = float(disturbance_horizon)
        if self.disturbance_horizon <= 0:
            raise ValueError("disturbance_horizon must be positive")
        self.disturbance_engine: DisturbanceEngine | None = None
        self.task_id_to_slot: dict[str, int] = {}
        self.slot_to_task_id: dict[int, str] = {}
        self.task_deadlines: dict[str, float | None] = {}
        self.uav_energy: dict[str, float] = {}
        self._nominal_speed: list[float] = []
        self._nominal_capabilities: list[np.ndarray] = []
        self._last_disturbance_events: tuple[DisturbanceEvent, ...] = ()
        super().__init__(config)

    @property
    def disturbances_disabled(self) -> bool:
        return self.disturbance_template.all_disabled

    def _runtime_disturbance_config(self, instance_seed: int) -> DisturbanceConfig:
        return replace(self.disturbance_template, instance_seed=int(instance_seed))

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        instance_seed = self.faithful_config.instance_seed if seed is None else int(seed)
        observation = super().reset(seed=instance_seed)
        self.task_id_to_slot = {
            f"t{index}": index
            for index, task in enumerate(self.tasks)
            if task.active
        }
        self.slot_to_task_id = {slot: task_id for task_id, slot in self.task_id_to_slot.items()}
        self.task_deadlines = {task_id: None for task_id in self.task_id_to_slot}
        self._nominal_speed = [float(uav.speed) for uav in self.uavs]
        self._nominal_capabilities = [uav.capabilities.copy() for uav in self.uavs]
        runtime_config = self._runtime_disturbance_config(instance_seed)
        task_states = tuple(
            TaskRuntimeState(
                task_id=task_id,
                predecessors=(
                    set() if self.tasks[slot].predecessor < 0
                    else {self.slot_to_task_id[self.tasks[slot].predecessor]}
                ),
                priority=float(self.tasks[slot].priority),
            )
            for task_id, slot in sorted(self.task_id_to_slot.items(), key=lambda item: item[1])
        )
        self.disturbance_engine = DisturbanceEngine(
            runtime_config,
            horizon=self.disturbance_horizon,
            uav_ids=tuple(f"u{index}" for index in range(self.faithful_config.scale.uavs)),
            tasks=task_states,
            initial_leader=(None if self.leader_id < 0 else f"u{self.leader_id}"),
        )
        self.uav_energy = {
            key: state.energy for key, state in self.disturbance_engine.uav.states.items()
        }
        self._last_disturbance_events = ()
        if self.disturbances_disabled:
            return observation
        initial_events = self.advance_disturbances(0.0)
        communication_active = any(
            getattr(runtime_config, name).enabled
            for name in (
                "gilbert_elliott_packet_loss", "message_delay", "network_partition"
            )
        )
        if initial_events and not communication_active:
            self._synchronize_belief(count_communication=False, full=True)
        return self.observe()

    def _next_time_boundary_delta(self, sync_mode: FaithfulSyncMode) -> float:
        baseline = super()._next_time_boundary_delta(sync_mode)
        engine = self.disturbance_engine
        if engine is None or self.disturbances_disabled or engine.cursor.exhausted:
            return baseline
        next_time = engine.tape.events[engine.cursor.index].physical_time
        disturbance_delta = max(0.0, float(next_time - self.current_time))
        positive = [value for value in (baseline, disturbance_delta) if value > 1e-9]
        return float(min(positive, default=0.0))

    def _sync_runtime_assignments(self) -> None:
        assert self.disturbance_engine is not None
        for index, uav in enumerate(self.uavs[: self.faithful_config.scale.uavs]):
            state = self.disturbance_engine.uav.states[f"u{index}"]
            state.current_task = self.slot_to_task_id.get(uav.busy_task)
        for task_id, slot in self.task_id_to_slot.items():
            task = self.tasks[slot]
            runtime = self.disturbance_engine.task.tasks[task_id]
            if task.completed:
                runtime.status = "completed"
                runtime.assigned_uav = None
            elif not task.active:
                runtime.status = "cancelled"
                runtime.assigned_uav = None
            elif task.assigned_uav >= 0:
                runtime.status = "running"
                runtime.assigned_uav = f"u{task.assigned_uav}"
            elif runtime.status != "cancelled":
                runtime.status = "pending"
                runtime.assigned_uav = None

    def _release_base_uav_task(self, uav_index: int, reason: str) -> None:
        uav = self.uavs[uav_index]
        task_index = uav.busy_task
        if task_index < 0:
            return
        task = self.tasks[task_index]
        task.assigned_uav = -1
        task.start_time = -1.0
        task.expected_completion = -1.0
        task.processing_time = 0.0
        task.reallocation_attempts += 1
        uav.busy_task = -1
        uav.remaining_time = 0.0
        self.reallocated_tasks += 1

    def _apply_uav_event(self, event: DisturbanceEvent) -> EventRecord:
        index = int(event.target.removeprefix("u"))
        uav = self.uavs[index]
        before = {"alive": bool(uav.alive), "leader": index == self.leader_id}
        if event.event_type == "uav_failure":
            self._release_base_uav_task(index, "uav_failure")
            uav.alive = False
            if bool(event.payload["permanent"]):
                uav.health = 0.0
        elif event.event_type == "uav_recovery":
            state = self.disturbance_engine.uav.states[event.target]  # type: ignore[union-attr]
            if state.alive and state.energy > 0:
                uav.alive = True
                uav.health = max(uav.health, 0.5)
        leader = self.disturbance_engine.uav.leader_id  # type: ignore[union-attr]
        self.leader_id = -1 if leader is None else int(leader.removeprefix("u"))
        return self._event_record(
            event.event_type,
            "phase1b_disturbance_tape",
            "uav",
            index,
            before=before,
            after={"alive": bool(uav.alive), "leader": index == self.leader_id},
            disturbance_event_id=event.event_id,
        )

    def _new_dynamic_task(self, event: DisturbanceEvent) -> int:
        inactive = [index for index, task in enumerate(self.tasks) if not task.active]
        if not inactive:
            raise RuntimeError("no inactive task slot remains for dynamic arrival")
        predecessors = list(event.payload.get("predecessors", []))
        if len(predecessors) > 1:
            raise ValueError("paper-faithful adapter supports one predecessor per task slot")
        predecessor = -1 if not predecessors else self.task_id_to_slot[str(predecessors[0])]
        slot = inactive[0]
        task = self._new_task(
            task_type=int(event.payload.get("task_type", slot % 4)),
            predecessor=predecessor,
            active=True,
            arrival_time=event.physical_time,
        )
        task.priority = float(event.payload.get("priority", task.priority))
        self.tasks[slot] = task
        self.task_id_to_slot[event.target] = slot
        self.slot_to_task_id[slot] = event.target
        self.task_deadlines[event.target] = (
            None if event.payload.get("deadline") is None
            else float(event.payload["deadline"])
        )
        if slot < len(self.parent_task_ids):
            self.parent_task_ids[slot] = -1
        return slot

    def _apply_task_event(self, event: DisturbanceEvent) -> EventRecord:
        if event.event_type == "task_arrival":
            slot = self._new_dynamic_task(event)
            before: Any = {"active": False}
            after: Any = {"active": True}
        else:
            slot = self.task_id_to_slot[event.target]
            task = self.tasks[slot]
            before = {
                "active": bool(task.active),
                "priority": float(task.priority),
                "deadline": self.task_deadlines.get(event.target),
            }
            if event.event_type == "task_cancellation":
                if task.assigned_uav >= 0:
                    uav = self.uavs[task.assigned_uav]
                    uav.busy_task = -1
                    uav.remaining_time = 0.0
                    task.assigned_uav = -1
                task.active = False
            elif event.event_type == "task_priority_change":
                task.priority = float(event.payload["priority"])
            elif event.event_type == "task_deadline_change":
                self.task_deadlines[event.target] = (
                    None if event.payload.get("deadline") is None
                    else float(event.payload["deadline"])
                )
            after = {
                "active": bool(task.active),
                "priority": float(task.priority),
                "deadline": self.task_deadlines.get(event.target),
            }
        return self._event_record(
            event.event_type,
            "phase1b_disturbance_tape",
            "task",
            slot,
            before=before,
            after=after,
            disturbance_event_id=event.event_id,
            task_id=event.target,
        )

    def advance_disturbances(self, physical_time: float) -> tuple[DisturbanceEvent, ...]:
        if self.disturbance_engine is None:
            raise RuntimeError("reset must be called before advancing disturbances")
        self._sync_runtime_assignments()
        step = self.disturbance_engine.advance(physical_time)
        records: list[EventRecord] = []
        for event in step.current_events:
            if event.event_type in {"uav_failure", "uav_recovery"}:
                records.append(self._apply_uav_event(event))
            elif event.event_type.startswith("task_"):
                records.append(self._apply_task_event(event))
        if records:
            self.event_log.extend(record.to_dict() for record in records)
        self._last_disturbance_events = step.current_events
        return step.current_events

    def _apply_energy(self, elapsed: float, busy_before: tuple[bool, ...]) -> None:
        assert self.disturbance_engine is not None
        if not self.disturbance_engine.config.energy_depletion.enabled or elapsed <= 0:
            return
        for index, was_busy in enumerate(busy_before):
            key = f"u{index}"
            self.disturbance_engine.uav.consume_energy(
                key,
                execution=elapsed if was_busy else 0.0,
                standby=0.0 if was_busy else elapsed,
            )
            state = self.disturbance_engine.uav.states[key]
            uav = self.uavs[index]
            uav.speed = self._nominal_speed[index] * state.speed_factor
            uav.capabilities = self._nominal_capabilities[index] * state.capability_factor
            self.uav_energy[key] = state.energy
            if state.energy <= 0 and uav.alive:
                self._release_base_uav_task(index, "energy_exhausted")
                uav.alive = False
        leader = self.disturbance_engine.uav.leader_id
        self.leader_id = -1 if leader is None else int(leader.removeprefix("u"))

    def _execution_components(
        self, uav: PaperUAVState, task: PaperTaskState, weather: float
    ) -> tuple[float, float]:
        flight_time, execution_time = super()._execution_components(uav, task, weather)
        engine = self.disturbance_engine
        if engine is None or not engine.config.wind_field.enabled:
            return flight_time, execution_time
        adjustment = engine.wind.adjust_travel(
            tuple(map(float, uav.position)),
            tuple(map(float, task.position)),
            airspeed=max(0.05, float(uav.speed)),
            base_energy_per_distance=1.0,
            physical_time=self.current_time,
        )
        return float(adjustment.disturbed_travel_time), execution_time

    def step(
        self, action: int, sync_mode: FaithfulSyncMode = "event"
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self.disturbances_disabled:
            return super().step(action, sync_mode=sync_mode)
        self.advance_disturbances(self.current_time)
        before_time = float(self.current_time)
        busy_before = tuple(
            uav.busy_task >= 0
            for uav in self.uavs[: self.faithful_config.scale.uavs]
        )
        _, reward, done, info = super().step(action, sync_mode=sync_mode)
        elapsed = max(0.0, self.current_time - before_time)
        self._apply_energy(elapsed, busy_before)
        disturbance_events = self.advance_disturbances(self.current_time)
        if disturbance_events and sync_mode == "event":
            records = [
                self._event_record(
                    event.event_type,
                    "phase1b_disturbance_tape",
                    "uav" if event.target.startswith("u") else "task",
                    (
                        int(event.target.removeprefix("u"))
                        if event.target.startswith("u")
                        else self.task_id_to_slot.get(event.target, -1)
                    ),
                    disturbance_event_id=event.event_id,
                )
                for event in disturbance_events
                if event.event_type.startswith("task_") or event.event_type.startswith("uav_")
            ]
            if records:
                self._synchronize_belief(records=records)
        active = [task for task in self.tasks if task.active]
        all_completed = bool(active) and all(task.completed for task in active)
        future_recovery = any(
            event.event_type == "uav_recovery" and event.physical_time > self.current_time
            for event in self.disturbance_engine.tape.events  # type: ignore[union-attr]
        )
        no_worker = not any(uav.active and uav.alive for uav in self.uavs)
        if all_completed:
            done = True
            self.termination_reason = "all_completed"
        elif no_worker and future_recovery:
            done = False
            self.termination_reason = "running"
        info.update(
            {
                "disturbance_events": [event.to_dict() for event in disturbance_events],
                "disturbance_config_sha256": self.disturbance_engine.config.sha256,  # type: ignore[union-attr]
                "disturbance_tape_sha256": self.disturbance_engine.tape.sha256,  # type: ignore[union-attr]
                "uav_energy": dict(self.uav_energy),
                "task_deadlines": dict(self.task_deadlines),
            }
        )
        return self.observe(), float(reward), bool(done), info

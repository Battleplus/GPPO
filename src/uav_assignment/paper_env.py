from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


TASK_NAMES = ("search", "reconnaissance", "strike", "recovery")
EVENT_NAMES = (
    "none",
    "weather_change",
    "uav_failure",
    "task_change",
    "communication_drop",
)

SELF_EDGE = 1
UAV_COMM_EDGE = 2
TASK_PREDECESSOR_EDGE = 3
UAV_TASK_EDGE = 4
TASK_SUCCESSOR_EDGE = 5
NODE_FEATURE_DIM = 24
EDGE_FEATURE_DIM = 5
SyncMode = Literal["event", "periodic", "always", "none"]


@dataclass(slots=True)
class PaperEnvConfig:
    max_uavs: int = 6
    max_tasks: int = 40
    active_uavs: int = 4
    initial_tasks: int = 24
    max_decisions: int = 160
    capability_threshold: float = 0.30
    heartbeat_timeout: float = 5.0
    heartbeat_interval: float = 1.0
    periodic_interval: float = 4.0
    weather_probability: float = 0.035
    failure_probability: float = 0.012
    task_change_probability: float = 0.025
    communication_drop_probability: float = 0.025
    invalid_action_penalty: float = 0.25
    communication_cost: float = 0.01
    terminal_penalty: float = 0.50
    include_engineering_rewards: bool = False
    mission_deadline: float = 0.0
    stop_at_deadline: bool = False
    task_chain_length: int = 4
    workload_scale: float = 1.0
    seed: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.active_uavs <= self.max_uavs:
            raise ValueError("active_uavs must be within max_uavs")
        if not 1 <= self.initial_tasks <= self.max_tasks:
            raise ValueError("initial_tasks must be within max_tasks")
        if self.mission_deadline < 0:
            raise ValueError("mission_deadline must be non-negative")
        if self.periodic_interval <= 0:
            raise ValueError("periodic_interval must be positive")
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if self.task_chain_length < 1:
            raise ValueError("task_chain_length must be positive")
        if self.workload_scale <= 0:
            raise ValueError("workload_scale must be positive")
        if not 0.0 < self.capability_threshold < 1.0:
            raise ValueError("capability_threshold must be between zero and one")
        probabilities = (
            self.weather_probability,
            self.failure_probability,
            self.task_change_probability,
            self.communication_drop_probability,
        )
        if any(probability < 0 for probability in probabilities):
            raise ValueError("event probabilities must be non-negative")
        if sum(probabilities) > 1.0:
            raise ValueError("event probabilities must sum to at most one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PaperUAVState:
    position: np.ndarray
    capabilities: np.ndarray
    speed: float
    health: float
    communication: float
    active: bool
    alive: bool
    busy_task: int = -1
    remaining_time: float = 0.0
    cumulative_processing: float = 0.0

    @property
    def idle(self) -> bool:
        return self.active and self.alive and self.busy_task < 0


@dataclass(slots=True)
class PaperTaskState:
    position: np.ndarray
    task_type: int
    priority: float
    workload: float
    risk: float
    predecessor: int
    active: bool
    completed: bool = False
    assigned_uav: int = -1
    start_time: float = -1.0
    completion_time: float = -1.0
    expected_completion: float = -1.0
    processing_time: float = 0.0
    age: float = 0.0
    reallocation_attempts: int = 0
    recovered_after_reallocation: bool = False
    arrival_time: float = 0.0


@dataclass(slots=True)
class EventRecord:
    decision: int
    time: float
    event_type: str
    source: str
    target_kind: str = ""
    target_id: int = -1
    before: Any = None
    after: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    triggered_sync: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PaperAlignedUAVEnv:
    """Event-driven UAV scheduler with a stale synchronized policy belief.

    Assignment starts work but never completes a task immediately. Time advances
    to the next completion, exogenous event, heartbeat, periodic synchronization,
    or deadline boundary when no further true assignment is possible. A rejected
    stale-belief action also advances the physical clock so cache disagreement
    cannot deadlock the event-driven simulation.
    """

    node_feature_dim = NODE_FEATURE_DIM
    edge_feature_dim = EDGE_FEATURE_DIM
    n_events = len(EVENT_NAMES)

    def __init__(self, config: PaperEnvConfig | None = None):
        self.config = config or PaperEnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.uavs: list[PaperUAVState] = []
        self.tasks: list[PaperTaskState] = []
        self.belief_uavs: list[PaperUAVState] = []
        self.belief_tasks: list[PaperTaskState] = []
        self.current_time = 0.0
        self.belief_time = 0.0
        self.weather_severity = 0.0
        self.belief_weather = 0.0
        self.decision_count = 0
        self.pending_event = 0
        self._pending_records: list[EventRecord] = []
        self.event_log: list[dict[str, Any]] = []
        self.leader_id = -1
        self.belief_leader_id = -1
        self.heartbeat_age = np.zeros(self.config.max_uavs, dtype=np.float32)
        self.belief_heartbeat_age = np.zeros(self.config.max_uavs, dtype=np.float32)
        self.heartbeat_timed_out = np.zeros(self.config.max_uavs, dtype=np.bool_)
        self.communication_events = 0
        self.sync_attempts = 0
        self.sync_successes = 0
        self.heartbeat_messages = 0
        self.leader_changes = 0
        self.invalid_actions = 0
        self.reallocated_tasks = 0
        self.reallocation_successes = 0
        self.completed_total = 0
        self.initial_active_tasks = self.config.initial_tasks
        self.next_periodic_sync_time = self.config.periodic_interval
        self.next_heartbeat_time = self.config.heartbeat_interval
        self.next_exogenous_event_time = float("inf")
        self.next_exogenous_event_type = 0
        self.deadline_reached = False
        self.termination_reason = "running"

    @property
    def n_nodes(self) -> int:
        return self.config.max_uavs + self.config.max_tasks

    @property
    def n_actions(self) -> int:
        return self.config.max_uavs * self.config.max_tasks + 1

    @property
    def noop_action(self) -> int:
        return self.n_actions - 1

    @property
    def makespan(self) -> float:
        memo: dict[int, float] = {}
        return float(
            max(
                (
                    self._estimated_task_completion(index, memo)
                    for index, task in enumerate(self.tasks)
                    if task.active
                ),
                default=self.current_time,
            )
        )

    def _estimated_task_completion(
        self, task_index: int, memo: dict[int, float]
    ) -> float:
        if task_index in memo:
            return memo[task_index]
        task = self.tasks[task_index]
        if not task.active:
            memo[task_index] = self.current_time
            return memo[task_index]
        if task.completed:
            memo[task_index] = max(self.current_time, task.completion_time)
            return memo[task_index]
        if task.assigned_uav >= 0:
            memo[task_index] = max(self.current_time, task.expected_completion)
            return memo[task_index]
        predecessor_time = self.current_time
        if task.predecessor >= 0:
            predecessor_time = self._estimated_task_completion(task.predecessor, memo)
        candidates: list[float] = []
        for uav in self.uavs:
            if not uav.active or not uav.alive:
                continue
            if uav.capabilities[task.task_type] < self.config.capability_threshold:
                continue
            flight_time, execution_time = self._execution_components(
                uav, task, self.weather_severity
            )
            available_time = self.current_time + uav.remaining_time
            candidates.append(max(predecessor_time, available_time) + flight_time + execution_time)
        memo[task_index] = (
            float(np.mean(candidates))
            if candidates
            else predecessor_time + max(1.0, task.workload)
        )
        return memo[task_index]

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_time = 0.0
        self.belief_time = 0.0
        self.weather_severity = 0.0
        self.belief_weather = 0.0
        self.decision_count = 0
        self.pending_event = 0
        self._pending_records = []
        self.event_log = []
        self.communication_events = 0
        self.sync_attempts = 0
        self.sync_successes = 0
        self.heartbeat_messages = 0
        self.leader_changes = 0
        self.invalid_actions = 0
        self.reallocated_tasks = 0
        self.reallocation_successes = 0
        self.completed_total = 0
        self.uavs = [self._sample_uav(index) for index in range(self.config.max_uavs)]
        self.tasks = self._sample_tasks()
        self.leader_id = self._first_alive_uav(self.uavs)
        self.heartbeat_age = np.zeros(self.config.max_uavs, dtype=np.float32)
        self.heartbeat_timed_out = np.zeros(self.config.max_uavs, dtype=np.bool_)
        self.next_periodic_sync_time = self.config.periodic_interval
        self.next_heartbeat_time = self.config.heartbeat_interval
        self._schedule_next_exogenous_event()
        self.deadline_reached = False
        self.termination_reason = "running"
        self._synchronize_belief(count_communication=False, full=True)
        self.initial_active_tasks = sum(task.active for task in self.tasks)
        return self.observe()

    def _sample_uav(self, index: int) -> PaperUAVState:
        active = index < self.config.active_uavs
        capabilities = self.rng.uniform(0.05, 1.0, size=4).astype(np.float32)
        if active:
            specialist_floor = max(0.82, self.config.capability_threshold + 0.05)
            for task_type in range(4):
                if task_type % self.config.active_uavs == index:
                    capabilities[task_type] = self.rng.uniform(specialist_floor, 1.0)
            if self.config.active_uavs > 1:
                blocked_type = next(
                    task_type
                    for task_type in range(4)
                    if task_type % self.config.active_uavs != index
                )
                capabilities[blocked_type] = 0.5 * self.config.capability_threshold
        return PaperUAVState(
            position=self.rng.uniform(0.0, 1.0, size=2).astype(np.float32),
            capabilities=capabilities,
            speed=float(self.rng.uniform(0.75, 1.25)),
            health=float(self.rng.uniform(0.80, 1.0)) if active else 0.0,
            communication=float(self.rng.uniform(0.70, 1.0)) if active else 0.0,
            active=active,
            alive=active,
        )

    def _new_task(
        self,
        task_type: int,
        predecessor: int = -1,
        active: bool = True,
        center: np.ndarray | None = None,
        arrival_time: float = 0.0,
    ) -> PaperTaskState:
        position = self.rng.uniform(0.0, 1.0, size=2)
        if center is not None:
            position = np.clip(center + self.rng.normal(0.0, 0.08, size=2), 0.0, 1.0)
        return PaperTaskState(
            position=position.astype(np.float32),
            task_type=task_type,
            priority=float(self.rng.uniform(0.55, 1.0)),
            workload=float(self.rng.uniform(0.35, 1.15) * self.config.workload_scale),
            risk=float(self.rng.uniform(0.05, 0.65)),
            predecessor=predecessor,
            active=active,
            arrival_time=arrival_time if active else -1.0,
        )

    def _sample_tasks(self) -> list[PaperTaskState]:
        tasks: list[PaperTaskState] = []
        active_target = self.config.initial_tasks
        chain_length = self.config.task_chain_length
        full_chains = active_target // chain_length
        for _ in range(full_chains):
            predecessor = -1
            center = self.rng.uniform(0.1, 0.9, size=2)
            for step in range(chain_length):
                task = self._new_task(step % 4, predecessor, center=center)
                tasks.append(task)
                predecessor = len(tasks) - 1
        while len(tasks) < active_target:
            tasks.append(self._new_task(len(tasks) % 4))
        while len(tasks) < self.config.max_tasks:
            tasks.append(self._new_task(len(tasks) % 4, active=False))
        return tasks

    @staticmethod
    def _first_alive_uav(uavs: list[PaperUAVState]) -> int:
        return next((index for index, uav in enumerate(uavs) if uav.active and uav.alive), -1)

    @staticmethod
    def _clone_uavs(uavs: list[PaperUAVState]) -> list[PaperUAVState]:
        cloned = copy.deepcopy(uavs)
        for uav in cloned:
            uav.position = uav.position.copy()
            uav.capabilities = uav.capabilities.copy()
        return cloned

    @staticmethod
    def _clone_tasks(tasks: list[PaperTaskState]) -> list[PaperTaskState]:
        cloned = copy.deepcopy(tasks)
        for task in cloned:
            task.position = task.position.copy()
        return cloned

    def _reachable_uavs(self) -> set[int]:
        if self.leader_id < 0 or not self.uavs[self.leader_id].alive:
            return set()
        leader = self.uavs[self.leader_id]
        return {
            index
            for index, uav in enumerate(self.uavs)
            if uav.active
            and uav.alive
            and (
                index == self.leader_id
                or (
                    min(uav.communication, leader.communication) >= 0.25
                    and not self.heartbeat_timed_out[index]
                )
            )
        }

    def _synchronize_belief(
        self,
        count_communication: bool = True,
        full: bool = False,
        records: list[EventRecord] | None = None,
    ) -> list[int]:
        if count_communication:
            self.sync_attempts += 1
        if full:
            reachable = {
                index for index, uav in enumerate(self.uavs) if uav.active
            }
        else:
            reachable = self._reachable_uavs()
        forced_uavs = {
            record.target_id
            for record in records or []
            if record.target_kind == "uav"
            and record.target_id >= 0
            and record.event_type
            in {"uav_failure", "leader_failure", "leader_elected"}
        }
        communication_reports = {
            record.target_id
            for record in records or []
            if record.target_kind == "uav"
            and record.target_id >= 0
            and record.event_type
            in {"communication_degraded", "communication_interrupted"}
        }
        heartbeat_reports = {
            record.target_id
            for record in records or []
            if record.target_kind == "uav"
            and record.target_id >= 0
            and record.event_type == "heartbeat_timeout"
        }
        forced_tasks = {
            record.target_id
            for record in records or []
            if record.target_kind == "task" and record.target_id >= 0
        }
        updated_uavs = reachable | forced_uavs

        if full or not self.belief_uavs:
            self.belief_uavs = self._clone_uavs(self.uavs)
            self.belief_tasks = self._clone_tasks(self.tasks)
            updated_uavs = set(range(self.config.max_uavs))
        else:
            for index in updated_uavs:
                self.belief_uavs[index] = self._clone_uavs([self.uavs[index]])[0]
            for index in communication_reports - updated_uavs:
                self.belief_uavs[index].communication = self.uavs[index].communication
            for index, task in enumerate(self.tasks):
                if (
                    index in forced_tasks
                    or task.assigned_uav < 0
                    or task.assigned_uav in updated_uavs
                ):
                    self.belief_tasks[index] = self._clone_tasks([task])[0]

        self.belief_time = self.current_time
        if full or any(record.event_type == "weather_change" for record in records or []):
            self.belief_weather = self.weather_severity
        if full or self.leader_id in updated_uavs:
            self.belief_leader_id = self.leader_id
        for index in updated_uavs:
            self.belief_heartbeat_age[index] = self.heartbeat_age[index]
        for index in heartbeat_reports:
            self.belief_heartbeat_age[index] = self.heartbeat_age[index]
        if count_communication:
            self.communication_events += 1
            if updated_uavs:
                self.sync_successes += 1
        return sorted(updated_uavs)

    def _process_heartbeats(self) -> None:
        """Send fixed-interval heartbeats independently of state synchronization."""

        while self.next_heartbeat_time <= self.current_time + 1e-9:
            scheduled_time = self.next_heartbeat_time
            if self.leader_id >= 0 and self.uavs[self.leader_id].alive:
                leader = self.uavs[self.leader_id]
                self.heartbeat_age[self.leader_id] = 0.0
                self.belief_heartbeat_age[self.leader_id] = 0.0
                for index, uav in enumerate(self.uavs):
                    if index == self.leader_id or not uav.active or not uav.alive:
                        continue
                    if min(uav.communication, leader.communication) >= 0.25:
                        self.heartbeat_messages += 1
                        self.heartbeat_age[index] = max(
                            0.0, self.current_time - scheduled_time
                        )
                        self.belief_heartbeat_age[index] = self.heartbeat_age[index]
                        self.heartbeat_timed_out[index] = False
            self.next_heartbeat_time += self.config.heartbeat_interval

    def _dependency_satisfied(
        self, task: PaperTaskState, tasks: list[PaperTaskState]
    ) -> bool:
        return task.predecessor < 0 or tasks[task.predecessor].completed

    def _execution_components(
        self, uav: PaperUAVState, task: PaperTaskState, weather: float
    ) -> tuple[float, float]:
        flight_time = float(np.linalg.norm(uav.position - task.position)) / max(
            0.1, uav.speed
        )
        capability = max(0.1, float(uav.capabilities[task.task_type]))
        execution_time = (1.0 + 0.8 * weather) * task.workload / capability
        return flight_time, execution_time

    def _valid_mask_for(
        self, uavs: list[PaperUAVState], tasks: list[PaperTaskState]
    ) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.bool_)
        for uav_index, uav in enumerate(uavs):
            if not uav.idle:
                continue
            for task_index, task in enumerate(tasks):
                if (
                    not task.active
                    or task.completed
                    or task.assigned_uav >= 0
                    or not self._dependency_satisfied(task, tasks)
                ):
                    continue
                if uav.capabilities[task.task_type] < self.config.capability_threshold:
                    continue
                mask[uav_index * self.config.max_tasks + task_index] = True
        mask[self.noop_action] = not bool(mask[:-1].any())
        return mask

    def valid_action_mask(self) -> np.ndarray:
        return self._valid_mask_for(self.belief_uavs, self.belief_tasks)

    def _apply_assignment(
        self,
        uavs: list[PaperUAVState],
        tasks: list[PaperTaskState],
        action: int,
        current_time: float,
        weather: float,
    ) -> bool:
        mask = self._valid_mask_for(uavs, tasks)
        if not 0 <= int(action) < self.n_actions or not bool(mask[int(action)]):
            return False
        if action == self.noop_action:
            return True
        uav_index, task_index = divmod(int(action), self.config.max_tasks)
        uav, task = uavs[uav_index], tasks[task_index]
        flight_time, execution_time = self._execution_components(uav, task, weather)
        processing_time = flight_time + execution_time
        uav.busy_task = task_index
        uav.remaining_time = processing_time
        task.assigned_uav = uav_index
        task.start_time = current_time
        task.processing_time = processing_time
        task.expected_completion = current_time + processing_time
        return True

    def _has_true_assignment(self) -> bool:
        return bool(self._valid_mask_for(self.uavs, self.tasks)[:-1].any())

    @staticmethod
    def _next_completion_delta(uavs: list[PaperUAVState]) -> float:
        remaining = [uav.remaining_time for uav in uavs if uav.busy_task >= 0]
        return float(min(remaining, default=0.0))

    def _next_time_boundary_delta(self, sync_mode: SyncMode) -> float:
        """Return the next physical boundary used by the event-driven clock."""

        if (
            sync_mode == "periodic"
            and self.current_time + 1e-9 >= self.next_periodic_sync_time
        ):
            return 0.0
        candidates: list[float] = []
        completion_delta = self._next_completion_delta(self.uavs)
        if completion_delta > 1e-9:
            candidates.append(completion_delta)
        if np.isfinite(self.next_exogenous_event_time):
            candidates.append(max(0.0, self.next_exogenous_event_time - self.current_time))
        candidates.append(max(0.0, self.next_heartbeat_time - self.current_time))
        if sync_mode == "periodic":
            candidates.append(max(0.0, self.next_periodic_sync_time - self.current_time))
        if (
            self.config.stop_at_deadline
            and self.config.mission_deadline > self.current_time + 1e-9
        ):
            candidates.append(self.config.mission_deadline - self.current_time)
        positive = [candidate for candidate in candidates if candidate > 1e-9]
        return float(min(positive, default=0.0))

    def _advance_state(
        self,
        uavs: list[PaperUAVState],
        tasks: list[PaperTaskState],
        delta: float,
        completion_time: float,
        count_metrics: bool,
    ) -> list[int]:
        if delta <= 0:
            return []
        completed: list[int] = []
        for task in tasks:
            if task.active and not task.completed:
                task.age += delta
        for uav_index, uav in enumerate(uavs):
            if uav.busy_task < 0:
                continue
            uav.remaining_time = max(0.0, uav.remaining_time - delta)
            if uav.remaining_time > 1e-7:
                continue
            task_index = uav.busy_task
            task = tasks[task_index]
            uav.position = task.position.copy()
            uav.cumulative_processing += task.processing_time
            uav.health = max(0.0, uav.health - 0.01 * task.risk * task.processing_time)
            uav.busy_task = -1
            uav.remaining_time = 0.0
            task.completed = True
            task.completion_time = completion_time
            task.expected_completion = completion_time
            completed.append(task_index)
            if count_metrics:
                self.completed_total += 1
                if task.reallocation_attempts > 0 and not task.recovered_after_reallocation:
                    task.recovered_after_reallocation = True
                    self.reallocation_successes += 1
        return completed

    def _event_record(
        self,
        event_type: str,
        source: str,
        target_kind: str = "",
        target_id: int = -1,
        before: Any = None,
        after: Any = None,
        **details: Any,
    ) -> EventRecord:
        return EventRecord(
            decision=self.decision_count + 1,
            time=float(self.current_time),
            event_type=event_type,
            source=source,
            target_kind=target_kind,
            target_id=target_id,
            before=before,
            after=after,
            details=details,
        )

    def _schedule_next_exogenous_event(self) -> None:
        probabilities = np.asarray(
            [
                self.config.weather_probability,
                self.config.failure_probability,
                self.config.task_change_probability,
                self.config.communication_drop_probability,
            ],
            dtype=np.float64,
        )
        total = float(probabilities.sum())
        if total <= 0.0:
            self.next_exogenous_event_time = float("inf")
            self.next_exogenous_event_type = 0
            return
        self.next_exogenous_event_time = self.current_time + float(
            self.rng.exponential(1.0 / total)
        )
        self.next_exogenous_event_type = int(
            self.rng.choice(np.arange(1, 5), p=probabilities / total)
        )

    def _sample_event(self, elapsed_time: float = 1.0) -> int:
        """Apply a scheduled exogenous hazard at its physical event time."""

        records: list[EventRecord] = []
        elapsed_time = max(0.0, float(elapsed_time))
        self.weather_severity *= 0.96**elapsed_time
        if self.current_time + 1e-9 < self.next_exogenous_event_time:
            self._pending_records.extend(records)
            return 0
        event = self.next_exogenous_event_type
        self._schedule_next_exogenous_event()
        if event == 1:
            before = self.weather_severity
            self.weather_severity = float(self.rng.uniform(0.2, 1.0))
            records.append(
                self._event_record(
                    "weather_change",
                    "environment",
                    before=before,
                    after=self.weather_severity,
                )
            )
        elif event == 2:
            alive = [index for index, uav in enumerate(self.uavs) if uav.active and uav.alive]
            if len(alive) <= 1:
                self._pending_records.extend(records)
                return 0
            victim = int(self.rng.choice(alive))
            was_leader = victim == self.leader_id
            self.uavs[victim].alive = False
            self.uavs[victim].health = 0.0
            records.append(
                self._event_record(
                    "uav_failure",
                    "uav",
                    "uav",
                    victim,
                    before={"alive": True, "leader": was_leader},
                    after={"alive": False, "leader": False},
                )
            )
            if was_leader:
                records.append(
                    self._event_record(
                        "leader_failure", "uav", "uav", victim, True, False
                    )
                )
                self.leader_id = -1
            task_index = self.uavs[victim].busy_task
            if task_index >= 0:
                task = self.tasks[task_index]
                task.assigned_uav = -1
                task.start_time = -1.0
                task.expected_completion = -1.0
                task.processing_time = 0.0
                task.reallocation_attempts += 1
                self.uavs[victim].busy_task = -1
                self.uavs[victim].remaining_time = 0.0
                self.reallocated_tasks += 1
                records.append(
                    self._event_record(
                        "task_reallocated",
                        "uav_failure",
                        "task",
                        task_index,
                        before={"assigned_uav": victim},
                        after={"assigned_uav": -1},
                    )
                )
            if was_leader:
                elected = self._first_alive_uav(self.uavs)
                self.leader_id = elected
                if elected >= 0:
                    self.leader_changes += 1
                    records.append(
                        self._event_record(
                            "leader_elected",
                            "leader_failure",
                            "uav",
                            elected,
                            before=-1,
                            after=elected,
                        )
                    )
        elif event == 3:
            inactive = [index for index, task in enumerate(self.tasks) if not task.active]
            if inactive:
                index = int(self.rng.choice(inactive))
                self.tasks[index] = self._new_task(
                    int(self.rng.integers(4)),
                    active=True,
                    arrival_time=self.current_time,
                )
                records.append(
                    self._event_record(
                        "new_task_arrival",
                        "environment",
                        "task",
                        index,
                        before={"active": False},
                        after={"active": True},
                    )
                )
            else:
                candidates = [
                    index
                    for index, task in enumerate(self.tasks)
                    if task.active and not task.completed and task.assigned_uav < 0
                ]
                if not candidates:
                    self._pending_records.extend(records)
                    return 0
                index = candidates[int(self.rng.integers(len(candidates)))]
                task = self.tasks[index]
                before = {
                    "position": task.position.tolist(),
                    "workload": task.workload,
                    "priority": task.priority,
                }
                task.position = self.rng.uniform(0.0, 1.0, size=2).astype(np.float32)
                task.workload = float(
                    self.rng.uniform(0.35, 1.15) * self.config.workload_scale
                )
                task.priority = float(self.rng.uniform(0.55, 1.0))
                records.append(
                    self._event_record(
                        "task_state_change",
                        "environment",
                        "task",
                        index,
                        before=before,
                        after={
                            "position": task.position.tolist(),
                            "workload": task.workload,
                            "priority": task.priority,
                        },
                    )
                )
        elif event == 4:
            alive = [
                index for index, uav in enumerate(self.uavs) if uav.active and uav.alive
            ]
            if not alive:
                self._pending_records.extend(records)
                return 0
            index = alive[int(self.rng.integers(len(alive)))]
            target = self.uavs[index]
            before = target.communication
            target.communication = float(self.rng.uniform(0.05, 0.22))
            event_type = (
                "communication_interrupted"
                if target.communication < 0.10
                else "communication_degraded"
            )
            records.append(
                self._event_record(
                    event_type,
                    "communication_link",
                    "uav",
                    index,
                    before=before,
                    after=target.communication,
                )
            )
        self._pending_records.extend(records)
        return event

    def _should_sync(self, mode: SyncMode, records: list[EventRecord]) -> bool:
        if mode not in {"none", "event", "periodic", "always"}:
            raise ValueError(f"unsupported sync mode: {mode}")
        if mode == "always":
            return True
        if mode == "periodic":
            return self.current_time >= self.next_periodic_sync_time
        if mode == "event":
            return bool(records)
        return False

    def step(
        self, action: int, sync_mode: SyncMode = "event"
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if sync_mode not in {"none", "event", "periodic", "always"}:
            raise ValueError(f"unsupported sync mode: {sync_mode}")
        previous_makespan = self.makespan
        previous_time = self.current_time
        belief_mask = self.valid_action_mask()
        belief_valid = 0 <= int(action) < self.n_actions and bool(belief_mask[int(action)])
        true_mask = self._valid_mask_for(self.uavs, self.tasks)
        true_valid = 0 <= int(action) < self.n_actions and bool(true_mask[int(action)])
        records: list[EventRecord] = []
        if not belief_valid or not true_valid:
            self.invalid_actions += 1
        if belief_valid != true_valid:
            records.append(
                self._event_record(
                    "assignment_state_conflict",
                    "action_feedback",
                    before={"belief_valid": belief_valid, "true_valid": true_valid},
                    after={"accepted": true_valid},
                    action=int(action),
                )
            )
        if belief_valid and true_valid:
            self._apply_assignment(
                self.belief_uavs,
                self.belief_tasks,
                int(action),
                self.belief_time,
                self.belief_weather,
            )
        if true_valid:
            self._apply_assignment(
                self.uavs,
                self.tasks,
                int(action),
                self.current_time,
                self.weather_severity,
            )
        paper_reward = 0.0

        delta = 0.0
        completed_true: list[int] = []
        completed_belief: list[int] = []
        rejected_with_available_work = not true_valid and self._has_true_assignment()
        if not self._has_true_assignment() or rejected_with_available_work:
            delta = self._next_time_boundary_delta(sync_mode)
            if delta > 0:
                self.current_time += delta
                self.belief_time += delta
                completed_true = self._advance_state(
                    self.uavs,
                    self.tasks,
                    delta,
                    self.current_time,
                    count_metrics=True,
                )
                completed_belief = self._advance_state(
                    self.belief_uavs,
                    self.belief_tasks,
                    delta,
                    self.belief_time,
                    count_metrics=False,
                )
                self.heartbeat_age += delta
                self.belief_heartbeat_age += delta
                self._process_heartbeats()

        for task_index in completed_true:
            records.append(
                self._event_record(
                    "task_completed",
                    "uav",
                    "task",
                    task_index,
                    before={"completed": False},
                    after={"completed": True},
                )
            )
            for successor_index, successor in enumerate(self.tasks):
                if (
                    successor.active
                    and not successor.completed
                    and successor.predecessor == task_index
                ):
                    records.append(
                        self._event_record(
                            "successor_unlocked",
                            "task_completion",
                            "task",
                            successor_index,
                            before={"dependency_satisfied": False},
                            after={"dependency_satisfied": True},
                            predecessor=task_index,
                        )
                    )

        for index, uav in enumerate(self.uavs):
            if (
                uav.active
                and uav.alive
                and index != self.leader_id
                and self.heartbeat_age[index] > self.config.heartbeat_timeout
                and not self.heartbeat_timed_out[index]
            ):
                self.heartbeat_timed_out[index] = True
                records.append(
                    self._event_record(
                        "heartbeat_timeout",
                        "leader",
                        "uav",
                        index,
                        before={"timed_out": False},
                        after={"timed_out": True},
                        heartbeat_age=float(self.heartbeat_age[index]),
                    )
                )

        if (
            self.config.mission_deadline > 0
            and not self.deadline_reached
            and previous_time < self.config.mission_deadline <= self.current_time
        ):
            self.deadline_reached = True
            records.append(
                self._event_record(
                    "mission_deadline_reached",
                    "clock",
                    before=previous_time,
                    after=self.current_time,
                    deadline=self.config.mission_deadline,
                )
            )

        event = self._sample_event(delta)
        records.extend(self._pending_records)
        self._pending_records = []
        self.pending_event = event
        self.decision_count += 1
        synchronized = self._should_sync(sync_mode, records)
        updated_uavs: list[int] = []
        if synchronized:
            repetitions = 1
            if sync_mode == "periodic":
                repetitions = 1 + int(
                    (self.current_time - self.next_periodic_sync_time)
                    // self.config.periodic_interval
                )
            updated: set[int] = set()
            for _ in range(repetitions):
                updated.update(
                    self._synchronize_belief(
                        count_communication=True,
                        full=sync_mode == "always",
                        records=records,
                    )
                )
                if sync_mode == "periodic":
                    self.next_periodic_sync_time += self.config.periodic_interval
            updated_uavs = sorted(updated)
        sync_reason = (
            "event_records"
            if synchronized and sync_mode == "event"
            else "periodic_boundary"
            if synchronized and sync_mode == "periodic"
            else "always"
            if synchronized and sync_mode == "always"
            else "none"
        )
        for record in records:
            record.triggered_sync = synchronized and sync_mode == "event"
            record.details.setdefault("sync_mode", sync_mode)
            record.details.setdefault("sync_reason", sync_reason)
            self.event_log.append(record.to_dict())

        paper_reward = previous_makespan - self.makespan

        active_tasks = [task for task in self.tasks if task.active]
        all_completed = bool(active_tasks) and all(task.completed for task in active_tasks)
        no_surviving_worker = not any(
            uav.active and uav.alive for uav in self.uavs
        )
        stalled = (
            not self._has_true_assignment()
            and self._next_completion_delta(self.uavs) <= 0
            and not all_completed
        )
        done = (
            all_completed
            or no_surviving_worker
            or stalled
            or self.decision_count >= self.config.max_decisions
            or (
                self.config.stop_at_deadline
                and self.config.mission_deadline > 0
                and self.current_time >= self.config.mission_deadline
            )
        )
        if done:
            if all_completed:
                self.termination_reason = "all_completed"
            elif no_surviving_worker:
                self.termination_reason = "no_surviving_worker"
            elif stalled:
                self.termination_reason = "stalled"
            elif self.config.stop_at_deadline and self.deadline_reached:
                self.termination_reason = "deadline"
            else:
                self.termination_reason = "max_decisions"
        invalid_penalty = self.config.invalid_action_penalty if not true_valid else 0.0
        communication_penalty = self.config.communication_cost if synchronized else 0.0
        remaining = sum(task.active and not task.completed for task in self.tasks)
        terminal_penalty = (
            self.config.terminal_penalty * remaining / max(1, len(active_tasks))
            if done and remaining
            else 0.0
        )
        reward = paper_reward
        if self.config.include_engineering_rewards:
            reward -= invalid_penalty + communication_penalty + terminal_penalty
        info: dict[str, Any] = {
            "event": EVENT_NAMES[event],
            "event_id": event,
            "events": [record.to_dict() for record in records],
            "synchronized": synchronized,
            "sync_mode": sync_mode,
            "sync_reason": sync_reason,
            "sync_attempted": synchronized,
            "sync_succeeded": bool(updated_uavs),
            "updated_uav_ids": updated_uavs,
            "valid_belief_action": belief_valid,
            "valid_true_action": true_valid,
            "paper_reward": paper_reward,
            "invalid_penalty": invalid_penalty,
            "communication_penalty": communication_penalty,
            "terminal_penalty": terminal_penalty,
            "time_advance": delta,
            "completed_tasks": completed_true,
            "belief_completed_tasks": completed_belief,
            "makespan": self.makespan,
            "current_time": self.current_time,
            "leader_id": self.leader_id,
            "deadline_reached": self.deadline_reached,
            "termination_reason": self.termination_reason,
        }
        return self.observe(), float(reward), done, info

    def _build_observation(
        self,
        uavs: list[PaperUAVState],
        tasks: list[PaperTaskState],
        current_time: float,
        weather: float,
        leader_id: int,
        heartbeat_age: np.ndarray,
    ) -> dict[str, np.ndarray]:
        nodes = np.zeros((self.n_nodes, NODE_FEATURE_DIM), dtype=np.float32)
        edge_types = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int64)
        edge_features = np.zeros(
            (self.n_nodes, self.n_nodes, EDGE_FEATURE_DIM), dtype=np.float32
        )
        time_scale = max(1.0, self.config.initial_tasks)
        for index, uav in enumerate(uavs):
            row = nodes[index]
            row[0] = 1.0
            row[2] = float(uav.active)
            row[3] = float(uav.alive)
            row[4] = float(uav.idle)
            row[5:7] = uav.position
            row[7] = uav.health
            row[8] = uav.communication
            row[9] = np.tanh(uav.remaining_time / 4.0)
            row[10] = np.tanh(uav.cumulative_processing / time_scale)
            row[11] = float(index == leader_id)
            row[12] = np.exp(-heartbeat_age[index] / max(0.1, self.config.heartbeat_timeout))
            row[13] = np.tanh(current_time / time_scale)
            row[14] = weather
            row[16:20] = uav.capabilities
            row[20] = float(uav.busy_task >= 0)
            row[22] = index / max(1, self.config.max_uavs - 1)
            if self.config.mission_deadline > 0:
                row[23] = np.clip(
                    (self.config.mission_deadline - current_time)
                    / self.config.mission_deadline,
                    0.0,
                    1.0,
                )
            if uav.active:
                edge_types[index, index] = SELF_EDGE

        offset = self.config.max_uavs
        for index, task in enumerate(tasks):
            row = nodes[offset + index]
            row[1] = 1.0
            row[2] = float(task.active)
            row[3] = float(task.active and not task.completed)
            row[4] = float(task.completed)
            row[5:7] = task.position
            row[7] = task.priority
            row[8] = task.workload
            row[9] = task.risk
            row[10] = float(self._dependency_satisfied(task, tasks))
            row[11] = float(task.assigned_uav >= 0)
            row[12] = np.tanh(max(0.0, task.start_time) / time_scale)
            row[13] = np.tanh(max(0.0, task.completion_time) / time_scale)
            row[14] = np.tanh(task.processing_time / 4.0)
            row[15] = (
                task.predecessor / max(1, self.config.max_tasks - 1)
                if task.predecessor >= 0
                else -1.0
            )
            row[16 + task.task_type] = 1.0
            row[21] = np.tanh(task.age / time_scale)
            row[22] = index / max(1, self.config.max_tasks - 1)
            if self.config.mission_deadline > 0:
                row[23] = np.clip(
                    (self.config.mission_deadline - current_time)
                    / self.config.mission_deadline,
                    0.0,
                    1.0,
                )
            if task.active:
                edge_types[offset + index, offset + index] = SELF_EDGE

        for target, target_uav in enumerate(uavs):
            if not target_uav.active:
                continue
            for source, source_uav in enumerate(uavs):
                if not source_uav.active or target == source:
                    continue
                freshness = max(heartbeat_age[target], heartbeat_age[source])
                if (
                    target_uav.alive
                    and source_uav.alive
                    and freshness <= self.config.heartbeat_timeout
                ):
                    edge_types[target, source] = UAV_COMM_EDGE
                    edge_features[target, source, 3] = min(
                        target_uav.communication, source_uav.communication
                    )
                    edge_features[target, source, 4] = np.exp(
                        -freshness / max(0.1, self.config.heartbeat_timeout)
                    )

        for task_index, task in enumerate(tasks):
            if not task.active:
                continue
            task_node = offset + task_index
            if task.predecessor >= 0 and tasks[task.predecessor].active:
                predecessor_node = offset + task.predecessor
                edge_types[task_node, predecessor_node] = TASK_PREDECESSOR_EDGE
                edge_features[task_node, predecessor_node, 4] = 1.0
                edge_types[predecessor_node, task_node] = TASK_SUCCESSOR_EDGE
                edge_features[predecessor_node, task_node, 4] = 1.0
            for uav_index, uav in enumerate(uavs):
                if not uav.active or not uav.alive:
                    continue
                capability = float(uav.capabilities[task.task_type])
                if capability < self.config.capability_threshold:
                    continue
                flight_time, execution_time = self._execution_components(
                    uav, task, weather
                )
                features = np.asarray(
                    [
                        np.tanh(flight_time / 2.0),
                        np.tanh(execution_time / 2.0),
                        capability,
                        uav.communication,
                        float(self._dependency_satisfied(task, tasks)),
                    ],
                    dtype=np.float32,
                )
                edge_types[task_node, uav_index] = UAV_TASK_EDGE
                edge_types[uav_index, task_node] = UAV_TASK_EDGE
                edge_features[task_node, uav_index] = features
                edge_features[uav_index, task_node] = features
        return {
            "nodes": nodes,
            "edge_types": edge_types,
            "edge_features": edge_features,
            "adjacency": edge_types > 0,
            "action_mask": self._valid_mask_for(uavs, tasks),
        }

    def observe(self) -> dict[str, np.ndarray]:
        return self._build_observation(
            self.belief_uavs,
            self.belief_tasks,
            self.belief_time,
            self.belief_weather,
            self.belief_leader_id,
            self.belief_heartbeat_age,
        )

    def true_observation(self) -> dict[str, np.ndarray]:
        return self._build_observation(
            self.uavs,
            self.tasks,
            self.current_time,
            self.weather_severity,
            self.leader_id,
            self.heartbeat_age,
        )

    def metrics(self) -> dict[str, Any]:
        active = [task for task in self.tasks if task.active]
        remaining = sum(not task.completed for task in active)
        deadline = self.config.mission_deadline
        deadline_tasks = [
            task
            for task in active
            if deadline <= 0 or task.arrival_time <= deadline
        ]
        completed_at_deadline = sum(
            task.completed
            and (deadline <= 0 or 0 <= task.completion_time <= deadline + 1e-7)
            for task in deadline_tasks
        )
        deadline_remaining = len(deadline_tasks) - completed_at_deadline
        deadline_completion_rate = completed_at_deadline / max(1, len(deadline_tasks))
        mission_success = float(bool(deadline_tasks) and deadline_remaining == 0)
        throughput_denominator = deadline if deadline > 0 else max(self.current_time, 1e-7)
        throughput = completed_at_deadline / max(throughput_denominator, 1e-7)
        return {
            "completed_total": float(self.completed_total),
            "completion_rate": self.completed_total / max(1, len(active)),
            "makespan": self.makespan,
            "projected_makespan": self.makespan,
            "deadline_completed_total": float(completed_at_deadline),
            "deadline_completion_rate": float(deadline_completion_rate),
            "mission_success": mission_success,
            "deadline_remaining_tasks": float(deadline_remaining),
            "throughput": float(throughput),
            "communication_events": float(self.communication_events),
            "sync_attempts": float(self.sync_attempts),
            "sync_successes": float(self.sync_successes),
            "heartbeat_messages": float(self.heartbeat_messages),
            "invalid_actions": float(self.invalid_actions),
            "reallocated_tasks": float(self.reallocated_tasks),
            "reallocation_successes": float(self.reallocation_successes),
            "reallocation_success_rate": self.reallocation_successes
            / max(1, self.reallocated_tasks),
            "remaining_tasks": float(remaining),
            "leader_changes": float(self.leader_changes),
            "current_time": float(self.current_time),
            "mission_deadline": float(deadline),
            "deadline_reached": float(self.deadline_reached),
            "event_count": float(len(self.event_log)),
            "termination_reason": self.termination_reason,
        }

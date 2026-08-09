from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .paper_env import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    EventRecord,
    PaperAlignedUAVEnv,
    PaperEnvConfig,
    PaperTaskState,
)


FaithfulSyncMode = Literal["none", "event", "periodic", "always"]


@dataclass(frozen=True, slots=True)
class PaperScale:
    """A paper scene: UAV count, parent-task count and subtask count."""

    uavs: int
    parent_tasks: int
    subtasks: int

    @property
    def name(self) -> str:
        return f"T{self.uavs}-{self.parent_tasks}-{self.subtasks}"


PAPER_SCALES = (
    PaperScale(5, 10, 48),
    PaperScale(10, 10, 53),
    PaperScale(15, 8, 66),
    PaperScale(20, 10, 92),
)


@dataclass(slots=True)
class PaperFaithfulConfig:
    scale: PaperScale = PAPER_SCALES[0]
    max_uavs: int = 20
    max_subtasks: int = 92
    instance_seed: int = 0
    event_tape_seed: int = 10_000
    reassignment_steps: tuple[int, ...] = (12, 24, 36)
    periodic_decisions: int = 8
    capability_threshold: float = 0.30
    workload_scale: float = 1.0
    max_decisions: int = 2_000

    def __post_init__(self) -> None:
        if self.scale.uavs > self.max_uavs or self.scale.subtasks > self.max_subtasks:
            raise ValueError("paper scale exceeds fixed model capacity")
        if self.scale.parent_tasks > self.scale.subtasks:
            raise ValueError("each parent task needs at least one subtask")
        if self.periodic_decisions < 1:
            raise ValueError("periodic_decisions must be positive")
        if any(step < 1 for step in self.reassignment_steps):
            raise ValueError("reassignment steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scale"]["name"] = self.scale.name
        return payload

    def base_config(self) -> PaperEnvConfig:
        return PaperEnvConfig(
            max_uavs=self.max_uavs,
            max_tasks=self.max_subtasks,
            active_uavs=self.scale.uavs,
            initial_tasks=self.scale.subtasks,
            max_decisions=self.max_decisions,
            capability_threshold=self.capability_threshold,
            weather_probability=0.0,
            failure_probability=0.0,
            task_change_probability=0.0,
            communication_drop_probability=0.0,
            include_engineering_rewards=False,
            mission_deadline=0.0,
            stop_at_deadline=False,
            workload_scale=self.workload_scale,
            seed=self.instance_seed,
        )


def deterministic_instance_seeds(
    scale: PaperScale,
    count: int = 100,
    split: Literal[
        "train", "validation", "test", "validation_a", "validation_b"
    ] = "train",
) -> tuple[int, ...]:
    """Stable disjoint instance banks, independent of algorithm/training seed."""

    split_offset = {
        "train": 0,
        "validation": 10_000_000,
        "test": 20_000_000,
        "validation_a": 30_000_000,
        "validation_b": 40_000_000,
    }[split]
    base = (
        split_offset
        + scale.uavs * 1_000_000
        + scale.parent_tasks * 10_000
        + scale.subtasks * 100
    )
    return tuple(base + index for index in range(count))


class PaperFaithfulUAVEnv(PaperAlignedUAVEnv):
    """Independent paper-faithful protocol, isolated from the hard extension.

    It retains asynchronous execution and heterogeneous capability constraints,
    but removes deadlines and stochastic hazard/conflict triggers.  The only
    event-mode communication trigger is a deterministic task-distribution tape.
    """

    def __init__(self, config: PaperFaithfulConfig | None = None):
        self.faithful_config = config or PaperFaithfulConfig()
        self.parent_task_ids = np.full(
            self.faithful_config.max_subtasks, -1, dtype=np.int64
        )
        self._event_tape: dict[int, dict[str, Any]] = {}
        self.communication_bytes = 0
        self.last_belief_sync_time = 0.0
        super().__init__(self.faithful_config.base_config())

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        instance_seed = self.faithful_config.instance_seed if seed is None else int(seed)
        observation = super().reset(seed=instance_seed)
        self.communication_bytes = 0
        self.last_belief_sync_time = 0.0
        self._event_tape = self._make_event_tape(instance_seed)
        return self.observe()

    def _sample_tasks(self) -> list[PaperTaskState]:
        scale = self.faithful_config.scale
        quotient, remainder = divmod(scale.subtasks, scale.parent_tasks)
        chain_sizes = [quotient + (index < remainder) for index in range(scale.parent_tasks)]
        tasks: list[PaperTaskState] = []
        self.parent_task_ids.fill(-1)
        for parent_id, chain_size in enumerate(chain_sizes):
            predecessor = -1
            center = self.rng.uniform(0.1, 0.9, size=2)
            start = len(tasks)
            phase = parent_id % 4
            for local_index in range(chain_size):
                task = self._new_task(
                    task_type=(phase + local_index) % 4,
                    predecessor=predecessor,
                    center=center,
                )
                tasks.append(task)
                task_index = len(tasks) - 1
                self.parent_task_ids[task_index] = parent_id
                predecessor = task_index
            assert len(tasks) - start == chain_size
        while len(tasks) < self.config.max_tasks:
            tasks.append(self._new_task(len(tasks) % 4, active=False))
        return tasks

    def _make_event_tape(self, instance_seed: int) -> dict[int, dict[str, Any]]:
        rng = np.random.default_rng(self.faithful_config.event_tape_seed + instance_seed)
        tape: dict[int, dict[str, Any]] = {}
        for step in self.faithful_config.reassignment_steps:
            task_id = int(rng.integers(0, self.faithful_config.scale.subtasks))
            tape[int(step)] = {
                "task_id": task_id,
                "position": rng.uniform(0.0, 1.0, size=2).astype(np.float32),
                "priority": float(rng.uniform(0.55, 1.0)),
                "workload_factor": float(rng.uniform(0.85, 1.15)),
            }
        return tape

    def _synchronize_belief(
        self,
        count_communication: bool = True,
        full: bool = False,
        records: list[EventRecord] | None = None,
    ) -> list[int]:
        updated = super()._synchronize_belief(
            count_communication=count_communication,
            full=full,
            records=records,
        )
        if full or updated:
            self.last_belief_sync_time = float(self.current_time)
        if count_communication:
            records = records or []
            forced_tasks = {
                record.target_id
                for record in records
                if record.target_kind == "task" and record.target_id >= 0
            }
            if full:
                synchronized_bytes = (
                    self.n_nodes * NODE_FEATURE_DIM
                    + self.n_nodes * self.n_nodes * EDGE_FEATURE_DIM
                ) * 4
            else:
                synchronized_bytes = (
                    (len(updated) + len(forced_tasks)) * NODE_FEATURE_DIM
                    + len(records) * EDGE_FEATURE_DIM
                ) * 4
            self.communication_bytes += int(synchronized_bytes)
        return updated

    def _apply_tape_event(self, event: dict[str, Any]) -> EventRecord:
        task_id = int(event["task_id"])
        task = self.tasks[task_id]
        before = {
            "position": task.position.tolist(),
            "priority": task.priority,
            "workload": task.workload,
        }
        task.position = np.asarray(event["position"], dtype=np.float32).copy()
        task.priority = float(event["priority"])
        task.workload *= float(event["workload_factor"])
        after = {
            "position": task.position.tolist(),
            "priority": task.priority,
            "workload": task.workload,
        }
        return self._event_record(
            "task_distribution_changed",
            "fixed_event_tape",
            "task",
            task_id,
            before=before,
            after=after,
            tape_step=self.decision_count,
        )

    def step(
        self, action: int, sync_mode: FaithfulSyncMode = "event"
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if sync_mode not in {"none", "event", "periodic", "always"}:
            raise ValueError(f"unsupported sync mode: {sync_mode}")
        previous_makespan = self.makespan
        cache_age_before = max(0.0, float(self.current_time - self.last_belief_sync_time))
        previous_heartbeats = self.heartbeat_messages
        _, _, done, info = super().step(action, sync_mode="none")
        self.communication_bytes += 16 * (
            self.heartbeat_messages - previous_heartbeats
        )
        records: list[EventRecord] = []
        tape_event = self._event_tape.get(self.decision_count)
        if tape_event is not None:
            records.append(self._apply_tape_event(tape_event))

        should_sync = (
            sync_mode == "always"
            or (sync_mode == "event" and bool(records))
            or (
                sync_mode == "periodic"
                and self.decision_count % self.faithful_config.periodic_decisions == 0
            )
        )
        updated: list[int] = []
        if should_sync:
            updated = self._synchronize_belief(
                records=records,
                full=sync_mode == "always",
            )
            for record in records:
                record.triggered_sync = True
        serialized = [record.to_dict() for record in records]
        self.event_log.extend(serialized)
        reward = previous_makespan - self.makespan
        info.update(
            {
                "paper_reward": reward,
                "makespan": self.makespan,
                "events": serialized,
                "event": "task_change" if records else "none",
                "synchronized": should_sync,
                "synchronized_uavs": updated,
                "protocol": "paper-faithful",
                "communication_bytes": self.communication_bytes,
                "cache_age_before": cache_age_before,
                "cache_age_after": max(
                    0.0, float(self.current_time - self.last_belief_sync_time)
                ),
            }
        )
        return self.observe(), float(reward), done, info

    def _build_observation(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        observation = super()._build_observation(*args, **kwargs)
        nodes = observation["nodes"]
        offset = self.config.max_uavs
        denominator = max(1, self.faithful_config.scale.parent_tasks - 1)
        for task_id in range(self.faithful_config.scale.subtasks):
            nodes[offset + task_id, 23] = self.parent_task_ids[task_id] / denominator
        return observation

    @property
    def realized_makespan(self) -> float:
        completed = [
            task.completion_time
            for task in self.tasks
            if task.active and task.completed and task.completion_time >= 0
        ]
        return float(max(completed, default=0.0))

    def metrics(self) -> dict[str, Any]:
        metrics = super().metrics()
        metrics["realized_makespan"] = self.realized_makespan
        metrics["all_tasks_completed"] = float(
            all(task.completed for task in self.tasks if task.active)
        )
        metrics["communication_bytes"] = float(self.communication_bytes)
        metrics["paper_scale"] = self.faithful_config.scale.name
        metrics["protocol"] = "paper-faithful"
        return metrics

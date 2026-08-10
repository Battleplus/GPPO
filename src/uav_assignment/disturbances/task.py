from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping

from .config import DisturbanceConfig
from .events import DisturbanceEvent


TaskStatus = Literal["pending", "running", "completed", "cancelled"]


@dataclass(slots=True)
class TaskRuntimeState:
    task_id: str
    predecessors: set[str] = field(default_factory=set)
    priority: float = 1.0
    deadline: float | None = None
    status: TaskStatus = "pending"
    assigned_uav: str | None = None
    arrival_time: float = 0.0
    completion_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["predecessors"] = sorted(self.predecessors)
        return payload


@dataclass(frozen=True, slots=True)
class TaskRelease:
    task_id: str
    uav_id: str
    physical_time: float
    reason: str


def generate_task_events(
    config: DisturbanceConfig,
    *,
    horizon: float,
) -> tuple[DisturbanceEvent, ...]:
    if horizon < 0:
        raise ValueError("horizon cannot be negative")
    source_specs = (
        ("task_arrival", "task_arrival", 50),
        ("task_cancellation", "task_cancellation", 51),
        ("task_priority_change", "task_priority_change", 52),
        ("task_deadline_change", "task_deadline_change", 53),
    )
    events: list[DisturbanceEvent] = []
    generation_index = 0
    for source, event_type, priority in source_specs:
        source_config = getattr(config, source)
        if not source_config.enabled:
            continue
        for index, specification in enumerate(source_config.parameters.get("events", [])):
            physical_time = float(specification["time"])
            task_id = str(specification["task_id"])
            if not 0 <= physical_time <= horizon:
                raise ValueError("task event lies outside the episode horizon")
            payload = {key: value for key, value in specification.items() if key not in {"time", "task_id"}}
            events.append(
                DisturbanceEvent(
                    event_id=f"{event_type}:{index}:{task_id}",
                    event_type=event_type,
                    physical_time=physical_time,
                    source=source,
                    target=task_id,
                    severity=float(specification.get("severity", 1.0)),
                    payload=payload,
                    ground_truth={"task_id": task_id, "change": event_type},
                    observed_time=None,
                    generation_index=generation_index,
                    source_priority=priority,
                )
            )
            generation_index += 1
    return tuple(sorted(events, key=lambda item: item.sort_key))


class TaskDisturbanceLayer:
    """Dynamic task registry with strict DAG and lifecycle invariants."""

    def __init__(
        self,
        tasks: Iterable[TaskRuntimeState],
        events: Iterable[DisturbanceEvent] = (),
    ):
        task_list = list(tasks)
        self.tasks = {task.task_id: task for task in task_list}
        if len(self.tasks) != len(task_list):
            raise ValueError("task_id values must be unique")
        self.events = tuple(sorted(events, key=lambda item: item.sort_key))
        self._event_index = 0
        self.physical_time = 0.0
        self.releases: list[TaskRelease] = []
        self.applied_event_ids: list[str] = []
        self._validate_dag()

    def _validate_dag(self) -> None:
        for task in self.tasks.values():
            unknown = task.predecessors - self.tasks.keys()
            if unknown:
                raise ValueError(f"unknown predecessors for {task.task_id}: {sorted(unknown)}")
            if task.task_id in task.predecessors:
                raise ValueError("a task cannot depend on itself")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task predecessor graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for predecessor in self.tasks[task_id].predecessors:
                visit(predecessor)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in sorted(self.tasks):
            visit(task_id)

    def eligible(self, task_id: str) -> bool:
        task = self.tasks[task_id]
        return task.status == "pending" and all(
            self.tasks[predecessor].status == "completed"
            for predecessor in task.predecessors
        )

    def action_mask(self, task_ids: Iterable[str] | None = None) -> dict[str, bool]:
        ids = tuple(task_ids) if task_ids is not None else tuple(sorted(self.tasks))
        return {task_id: self.eligible(task_id) for task_id in ids}

    def start(self, task_id: str, uav_id: str) -> None:
        if not self.eligible(task_id):
            raise ValueError("task is not currently eligible")
        task = self.tasks[task_id]
        task.status = "running"
        task.assigned_uav = str(uav_id)

    def complete(self, task_id: str) -> None:
        task = self.tasks[task_id]
        if task.status != "running":
            raise ValueError("only a running task can complete")
        task.status = "completed"
        task.assigned_uav = None
        task.completion_time = self.physical_time

    def _release(self, task: TaskRuntimeState, reason: str) -> None:
        if task.assigned_uav is None:
            return
        self.releases.append(
            TaskRelease(task.task_id, task.assigned_uav, self.physical_time, reason)
        )
        task.assigned_uav = None

    def release_from_uav(self, uav_id: str, reason: str) -> tuple[str, ...]:
        released: list[str] = []
        for task in self.tasks.values():
            if task.status == "running" and task.assigned_uav == uav_id:
                self._release(task, reason)
                task.status = "pending"
                released.append(task.task_id)
        return tuple(sorted(released))

    def _arrival(self, event: DisturbanceEvent) -> None:
        if event.target in self.tasks:
            raise ValueError(f"task already exists: {event.target}")
        predecessors = {str(value) for value in event.payload.get("predecessors", [])}
        task = TaskRuntimeState(
            task_id=event.target,
            predecessors=predecessors,
            priority=float(event.payload.get("priority", 1.0)),
            deadline=(
                None if event.payload.get("deadline") is None
                else float(event.payload["deadline"])
            ),
            arrival_time=event.physical_time,
        )
        self.tasks[event.target] = task
        try:
            self._validate_dag()
        except Exception:
            del self.tasks[event.target]
            raise

    def _cancel(self, event: DisturbanceEvent) -> None:
        task = self.tasks[event.target]
        if task.status == "cancelled":
            return
        if task.status == "completed":
            # A pre-generated cancellation may arrive after a fast policy has
            # already completed the task.  It is a valid no-op, not a reason to
            # resample the tape or fail the episode.
            return
        self._release(task, "task_cancelled")
        task.status = "cancelled"
        for successor in self.tasks.values():
            successor.predecessors.discard(task.task_id)
        self._validate_dag()

    def _apply(self, event: DisturbanceEvent) -> None:
        if event.event_type == "task_arrival":
            self._arrival(event)
            return
        if event.target not in self.tasks:
            raise ValueError(f"task event targets unknown task: {event.target}")
        task = self.tasks[event.target]
        if event.event_type == "task_cancellation":
            self._cancel(event)
        elif event.event_type == "task_priority_change":
            priority = float(event.payload["priority"])
            if priority < 0:
                raise ValueError("task priority cannot be negative")
            task.priority = priority
        elif event.event_type == "task_deadline_change":
            deadline = event.payload.get("deadline")
            task.deadline = None if deadline is None else float(deadline)
            if task.deadline is not None and task.deadline < self.physical_time:
                raise ValueError("new task deadline cannot be in the past")

    def advance(self, physical_time: float) -> tuple[DisturbanceEvent, ...]:
        physical_time = float(physical_time)
        if physical_time < self.physical_time:
            raise ValueError("task disturbance time cannot move backwards")
        self.physical_time = physical_time
        applied: list[DisturbanceEvent] = []
        while self._event_index < len(self.events):
            event = self.events[self._event_index]
            if event.physical_time > physical_time:
                break
            self._apply(event)
            self.applied_event_ids.append(event.event_id)
            applied.append(event)
            self._event_index += 1
        return tuple(applied)

    def snapshot(self) -> dict[str, Any]:
        return {
            "physical_time": self.physical_time,
            "tasks": {key: task.to_dict() for key, task in sorted(self.tasks.items())},
            "action_mask": self.action_mask(),
            "releases": [asdict(item) for item in self.releases],
        }

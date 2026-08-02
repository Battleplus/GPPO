"""Perfect-information preference oracle for PCRL headroom audits.

The oracle deliberately lives outside the learned-policy and frozen-GPPO
modules.  It uses the true physical state to rank only actions that are legal
in the true environment, then replans after every event-driven decision.  It
is therefore an upper-bound diagnostic, not a weak-communication policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .pcrl_v0 import preference_alignment, preference_allocation_target
from .paper_env import PaperTaskState


@dataclass(frozen=True, slots=True)
class OracleConfig:
    """Scoring knobs for the closed-loop full-state oracle.

    ``coverage_floor`` is expressed as a fraction of the currently observed
    available tasks of each type.  A zero floor leaves the target objective to
    determine the tradeoff; a positive floor reserves capacity for required
    task types and prevents an apparent preference win from dropping them.
    """

    decay: float = 1.0
    coverage_floor: float = 0.0
    target_mode: str = "nominal"
    deficit_gain: float = 4.0
    floor_gain: float = 2.5
    urgency_gain: float = 0.20
    downstream_gain: float = 0.35
    prerequisite_gain: float = 0.55
    rollout_candidates: int = 0
    allow_strategic_wait: bool = False
    rollout_wait_margin: float = 0.0
    deadline_completion_floor: float = 0.0
    minimum_type_coverage_floor: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.decay) or self.decay < 0:
            raise ValueError("decay must be finite and non-negative")
        if not 0.0 <= self.coverage_floor <= 1.0:
            raise ValueError("coverage_floor must lie in [0, 1]")
        if self.target_mode not in {"nominal", "dependency_feasible_projection"}:
            raise ValueError("unsupported oracle target mode")
        if self.rollout_candidates < 0:
            raise ValueError("rollout_candidates must be non-negative")
        for name in (
            "deficit_gain",
            "floor_gain",
            "urgency_gain",
            "downstream_gain",
            "prerequisite_gain",
            "rollout_wait_margin",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.deadline_completion_floor <= 1.0:
            raise ValueError("deadline_completion_floor must lie in [0, 1]")
        if not 0.0 <= self.minimum_type_coverage_floor <= 1.0:
            raise ValueError("minimum_type_coverage_floor must lie in [0, 1]")


def _safe_mix(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    return values / total if total > 1e-12 else np.zeros_like(values)


def _task_successor_types(env: object, task_index: int) -> list[int]:
    """Return active direct successor types using only public state shape.

    Keeping this helper tolerant of a minimal fake environment makes the
    action scorer unit-testable without constructing a full event simulator.
    """

    tasks = getattr(env, "tasks")
    return [
        int(successor.task_type)
        for successor in tasks
        if successor.active and not successor.completed
        and successor.predecessor == task_index
    ]


def _descendant_value(
    env: object,
    task_index: int,
    target: np.ndarray,
) -> tuple[float, int]:
    """Return discounted target mass and depth below a prerequisite task.

    Direct-successor scoring undervalues search/reconnaissance tasks whose
    important strike or recovery descendants are two or three edges away.
    The graph is a forest in the current simulator, but the visited set keeps
    this safe for future DAG task generators as well.
    """

    tasks = getattr(env, "tasks")
    frontier = [(int(task_index), 0)]
    visited = {int(task_index)}
    value = 0.0
    maximum_depth = 0
    while frontier:
        parent, depth = frontier.pop(0)
        for successor_index, successor in enumerate(tasks):
            if (
                successor_index in visited
                or not successor.active
                or successor.completed
                or successor.predecessor != parent
            ):
                continue
            visited.add(successor_index)
            successor_depth = depth + 1
            maximum_depth = max(maximum_depth, successor_depth)
            value += float(target[int(successor.task_type)]) / successor_depth
            frontier.append((successor_index, successor_depth))
    return value, maximum_depth


def _assignment_weight(env: object, *, decay: float, assignment_time: float) -> float:
    deadline = float(getattr(env.config, "mission_deadline"))
    return float(np.exp(-decay * assignment_time / max(1.0, deadline)))


def _candidate_score(
    env: object,
    action: int,
    target: np.ndarray,
    *,
    config: OracleConfig,
) -> tuple[float, tuple[float, ...]]:
    """Score one true-legal assignment from the current closed-loop state."""

    max_tasks = int(env.config.max_tasks)
    uav_index, task_index = divmod(int(action), max_tasks)
    task: PaperTaskState = env.tasks[task_index]
    task_type = int(task.task_type)

    # The environment records the first unique assignment contribution.  The
    # oracle mirrors that accounting exactly rather than using completion
    # counts, so its headroom is comparable to preference_l1.
    priority_coverage = np.asarray(
        env._deadline_assignment_priority_coverage(), dtype=np.float64
    ).copy()
    available = np.asarray(env._available_by_type, dtype=np.float64)
    if task_index not in env._deadline_assigned_tasks:
        weight = _assignment_weight(
            env,
            decay=config.decay,
            assignment_time=float(env.current_time),
        )
        if available[task_type] > 0:
            priority_coverage[task_type] += weight / available[task_type]
    current_mix = _safe_mix(
        np.asarray(env._deadline_assignment_priority_coverage(), dtype=np.float64)
    )
    projected_mix = _safe_mix(priority_coverage)
    current_l1 = float(preference_alignment(current_mix, target)["l1"])
    projected_l1 = float(preference_alignment(projected_mix, target)["l1"])
    l1_progress = current_l1 - projected_l1

    # Reserve a floor for all types that are currently observable.  This is an
    # anti-gaming guard only; it has no effect once every type is above floor.
    current_coverage = np.asarray(
        env._deadline_assignment_priority_coverage(), dtype=np.float64
    )
    floor = config.coverage_floor * np.where(available > 0, 1.0, 0.0)
    floor_deficit = max(0.0, float(floor[task_type] - current_coverage[task_type]))
    floor_progress = (
        min(1.0, floor_deficit + 1.0 / max(1.0, available[task_type]))
        if floor_deficit > 0
        else 0.0
    )

    uav = env.uavs[uav_index]
    weather = float(getattr(env, "weather_severity", 0.0))
    flight_time, execution_time = env._execution_components(uav, task, weather)
    duration = max(1e-6, float(flight_time + execution_time))
    deadline = max(1.0, float(env.config.mission_deadline))
    # Urgent jobs receive a bounded boost as the deadline approaches.  The
    # boost is deliberately small so preference deficit remains primary.
    expected_finish = float(env.current_time) + duration
    urgency = max(0.0, expected_finish - deadline) / deadline
    urgency_bonus = -config.urgency_gain * urgency

    downstream = sum(
        float(target[next_type])
        for next_type in _task_successor_types(env, task_index)
    )
    descendant_value, critical_depth = _descendant_value(
        env, task_index, target
    )
    downstream_bonus = config.downstream_gain * downstream / duration
    prerequisite_bonus = (
        config.prerequisite_gain
        * (descendant_value + 0.15 * critical_depth)
        / duration
    )
    priority_bonus = 0.05 * float(task.priority) / duration
    score = (
        config.deficit_gain * l1_progress
        + config.floor_gain * floor_progress
        + urgency_bonus
        + downstream_bonus
        + prerequisite_bonus
        + priority_bonus
    )
    # Tie-breaking is deterministic and favors shorter assignments, then
    # higher task priority, then stable task/UAV ids.
    key = (
        float(score),
        float(l1_progress),
        -float(duration),
        float(task.priority),
        -float(task_index),
        -float(uav_index),
    )
    return score, key


def _ranked_actions(
    env: object,
    *,
    config: OracleConfig,
) -> list[int]:
    target = preference_allocation_target(
        env.user_task_preference,
        mode=config.target_mode,
    )
    true_mask = np.asarray(env._valid_mask_for(env.uavs, env.tasks), dtype=np.bool_)
    valid = np.flatnonzero(true_mask[:-1])
    if valid.size == 0:
        return [int(env.noop_action)]
    ranked: list[tuple[tuple[float, ...], int]] = []
    for action in valid:
        _, key = _candidate_score(
            env,
            int(action),
            target,
            config=config,
        )
        ranked.append((key, int(action)))
    ranked.sort(reverse=True)
    return [action for _, action in ranked]


def _myopic_action(env: object, config: OracleConfig) -> int:
    return _ranked_actions(env, config=config)[0]


def _rollout_action(env: object, config: OracleConfig) -> int:
    """Deterministic rollout action with one-boundary delay lookahead."""

    ranked = _ranked_actions(env, config=config)
    best_action = ranked[0]
    if not _strategic_wait_is_eligible(env, config):
        return best_action
    _, current_key = _candidate_score(
        env,
        best_action,
        preference_allocation_target(
            env.user_task_preference, mode=config.target_mode
        ),
        config=config,
    )
    waited = copy.deepcopy(env)
    _, _, done, _ = step_oracle_action(
        waited, int(waited.noop_action), sync_mode="event"
    )
    if done:
        return best_action
    future_ranked = _ranked_actions(waited, config=config)
    if future_ranked == [int(waited.noop_action)]:
        return best_action
    _, future_key = _candidate_score(
        waited,
        future_ranked[0],
        preference_allocation_target(
            waited.user_task_preference, mode=config.target_mode
        ),
        config=config,
    )
    if future_key[0] > current_key[0] + config.rollout_wait_margin:
        return int(env.noop_action)
    return best_action


def _terminal_rollout_key(env: object, first_action: int, config: OracleConfig) -> tuple[float, ...]:
    """Evaluate one first action with a deterministic full-state rollout."""

    simulated = copy.deepcopy(env)
    _, _, done, _ = step_oracle_action(
        simulated, int(first_action), sync_mode="event"
    )
    while not done:
        action = _rollout_action(simulated, config)
        _, _, done, _ = step_oracle_action(
            simulated, action, sync_mode="event"
        )
    metrics = simulated.metrics()
    coverage = np.asarray(metrics["availability_coverage_by_type"], dtype=np.float64)
    floor_shortfall = float(
        np.maximum(config.coverage_floor - coverage, 0.0).sum()
    )
    deadline = max(1.0, float(simulated.config.mission_deadline))
    preference_l1 = float(metrics["preference_l1"])
    deadline_completion = float(metrics["deadline_completion_rate"])
    minimum_coverage = float(metrics["minimum_task_coverage"])
    makespan_ratio = float(metrics["makespan"]) / deadline
    invalid_ratio = float(metrics["invalid_actions"]) / max(
        1, int(simulated.config.max_decisions)
    )
    deadline_shortfall = max(
        0.0, config.deadline_completion_floor - deadline_completion
    )
    type_coverage_shortfall = max(
        0.0, config.minimum_type_coverage_floor - minimum_coverage
    )
    # Configured completion and coverage constraints are hard gates. Within
    # the feasible set, preference alignment is primary and efficiency terms
    # distinguish equally aligned schedules.
    return (
        -(deadline_shortfall + type_coverage_shortfall),
        -deadline_shortfall,
        -type_coverage_shortfall,
        -floor_shortfall,
        -preference_l1,
        deadline_completion,
        minimum_coverage,
        -makespan_ratio,
        -invalid_ratio,
        -float(first_action),
    )


def _strategic_wait_is_eligible(env: object, config: OracleConfig) -> bool:
    """Whether waiting one physical boundary can reveal useful work.

    Waiting is considered only while at least one assignment is available and
    another UAV is already processing a task.  In particular, a completion on
    a prerequisite chain can unlock a high-deficit successor; we never wait in
    a fully idle system or past the mission deadline.
    """

    if not config.allow_strategic_wait:
        return False
    true_mask = np.asarray(
        env._valid_mask_for(env.uavs, env.tasks), dtype=np.bool_
    )
    if not bool(true_mask[:-1].any()):
        return False
    completion_delta = float(env._next_completion_delta(env.uavs))
    if completion_delta <= 1e-9:
        return False
    deadline = float(getattr(env.config, "mission_deadline", 0.0))
    if deadline > 0 and float(env.current_time) + completion_delta > deadline + 1e-9:
        return False
    target = preference_allocation_target(
        env.user_task_preference, mode=config.target_mode
    )
    coverage = np.asarray(
        env._deadline_assignment_priority_coverage(), dtype=np.float64
    )
    deficit = target - _safe_mix(coverage)
    for uav in env.uavs:
        if uav.busy_task < 0 or uav.remaining_time > completion_delta + 1e-7:
            continue
        task_index = int(uav.busy_task)
        for successor_type in _task_successor_types(env, task_index):
            if deficit[successor_type] > 1e-9:
                return True
        descendant_value, _ = _descendant_value(env, task_index, target)
        if descendant_value > 1e-9:
            return True
    return False


def step_oracle_action(
    env: object,
    action: int,
    *,
    sync_mode: str = "event",
):
    """Apply an Oracle action, including a metric-neutral strategic wait.

    The frozen environment intentionally masks noop whenever an assignment is
    feasible.  For this perfect-information diagnostic only, an intentional
    noop uses the environment's rejected-action clock path, then removes the
    artificial invalid/safety accounting.  Physical time, completions,
    exogenous events, synchronization and termination remain exactly those of
    the frozen event simulator. Learned policies never call this helper.
    """

    action = int(action)
    true_mask = np.asarray(
        env._valid_mask_for(env.uavs, env.tasks), dtype=np.bool_
    )
    strategic_wait = action == int(env.noop_action) and bool(true_mask[:-1].any())
    invalid_before = int(env.invalid_actions)
    safety_return_before = (
        float(env.vector_return[-1]) if hasattr(env, "vector_return") else None
    )
    observation, reward, done, info = env.step(action, sync_mode=sync_mode)
    if not strategic_wait:
        return observation, reward, done, info

    invalid_increment = int(env.invalid_actions) - invalid_before
    if invalid_increment <= 0:
        return observation, reward, done, info
    env.invalid_actions -= invalid_increment
    info = dict(info)
    info["oracle_strategic_wait"] = True
    info["valid_belief_action"] = True
    info["valid_true_action"] = True
    info["invalid_penalty"] = 0.0
    if hasattr(env, "vector_return") and "vector_reward" in info:
        correction = float(safety_return_before) - float(env.vector_return[-1])
        vector_reward = np.asarray(info["vector_reward"], dtype=np.float32).copy()
        vector_reward[-1] += correction
        env.vector_return[-1] = float(safety_return_before)
        reward = float(np.dot(env.preference, vector_reward))
        info["vector_reward"] = vector_reward
        info["scalarized_preference_reward"] = reward
        observation = env.observe()
    return observation, reward, done, info


def choose_oracle_action(
    env: object,
    *,
    config: OracleConfig | None = None,
) -> int:
    """Choose a true-state legal action and replan at the next decision.

    The action mask intentionally comes from ``env.uavs``/``env.tasks`` rather
    than the stale policy belief.  Calling code should pass ``sync_mode='event'``
    to preserve the real event tape and should label resulting rows as
    ``oracle_perfect_information``.  When ``rollout_candidates`` is positive,
    the oracle performs deterministic terminal rollouts for the strongest
    current candidates and uses the full-state result only for the next action.
    Strategic noop is disabled by default because it is outside the frozen
    action mask; enabling it creates a separate relaxed-action upper bound.
    """

    oracle_config = config or OracleConfig()
    ranked = _ranked_actions(env, config=oracle_config)
    if oracle_config.rollout_candidates <= 0 or ranked == [int(env.noop_action)]:
        return ranked[0]
    candidates = ranked[: oracle_config.rollout_candidates]
    if _strategic_wait_is_eligible(env, oracle_config):
        candidates.append(int(env.noop_action))
    return max(
        candidates,
        key=lambda action: _terminal_rollout_key(env, action, oracle_config),
    )


def oracle_target(
    preference: Sequence[float] | np.ndarray,
    *,
    target_mode: str = "nominal",
) -> np.ndarray:
    """Public helper used by evaluation/reporting code."""

    return preference_allocation_target(preference, mode=target_mode)

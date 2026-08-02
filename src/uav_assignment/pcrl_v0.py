from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .paper_env import TASK_NAMES, PaperAlignedUAVEnv, PaperEnvConfig, PaperUAVState, SyncMode


PCRL_VERSION = "pcrl-v0-1"
N_TASK_OBJECTIVES = 4
N_OBJECTIVES = 7
OBJECTIVE_NAMES = (
    "search_coverage",
    "reconnaissance_coverage",
    "strike_coverage",
    "recovery_coverage",
    "makespan_efficiency",
    "communication_cost",
    "safety_cost",
)

DEFAULT_TASK_MASS = 0.70
DEFAULT_AUXILIARY_WEIGHTS = np.asarray((0.15, 0.05, 0.10), dtype=np.float32)
DEFAULT_TASK_FLOOR = 0.05
DEFAULT_ASSIGNMENT_PRIORITY_DECAY = 1.0
DEFAULT_PREFERENCE_TARGET_MODE = "dependency_feasible_projection"
LEGACY_PREFERENCE_SAMPLER_MODE = "legacy_extreme_dirichlet"
MODERATE_PREFERENCE_SAMPLER_MODE = "moderate_anchor_hull"
PREFERENCE_SAMPLER_MODES = (
    LEGACY_PREFERENCE_SAMPLER_MODE,
    MODERATE_PREFERENCE_SAMPLER_MODE,
)
PREFERENCE_TARGET_MODES = (
    DEFAULT_PREFERENCE_TARGET_MODE,
    "nominal",
)
LEGACY_CAPABILITY_COVERAGE_MODE = "legacy"
BALANCED_CAPABILITY_COVERAGE_MODE = "balanced_cyclic_min2_v1"
CAPABILITY_COVERAGE_MODES = (
    LEGACY_CAPABILITY_COVERAGE_MODE,
    BALANCED_CAPABILITY_COVERAGE_MODE,
)

TASK_PREFERENCE_PROFILES: dict[str, np.ndarray] = {
    "balanced": np.asarray((0.25, 0.25, 0.25, 0.25), dtype=np.float32),
    "search": np.asarray((0.70, 0.10, 0.10, 0.10), dtype=np.float32),
    "reconnaissance": np.asarray((0.10, 0.70, 0.10, 0.10), dtype=np.float32),
    "strike": np.asarray((0.10, 0.10, 0.70, 0.10), dtype=np.float32),
    "recovery": np.asarray((0.10, 0.10, 0.10, 0.70), dtype=np.float32),
    # Held out from the continuous training sampler and used for interpolation tests.
    "search_strike_interp": np.asarray((0.40, 0.10, 0.40, 0.10), dtype=np.float32),
    "recon_recovery_interp": np.asarray((0.10, 0.40, 0.10, 0.40), dtype=np.float32),
    "smooth_interp": np.asarray((0.35, 0.25, 0.25, 0.15), dtype=np.float32),
    # Hard-3 calibration profiles.  These are intentionally separate from the
    # frozen hard-2 0.70/0.10 anchors above: one priority receives twice the
    # mass of each non-priority task (0.40 versus 0.20).
    "search_moderate_2to1": np.asarray(
        (0.40, 0.20, 0.20, 0.20), dtype=np.float32
    ),
    "reconnaissance_moderate_2to1": np.asarray(
        (0.20, 0.40, 0.20, 0.20), dtype=np.float32
    ),
    "strike_moderate_2to1": np.asarray(
        (0.20, 0.20, 0.40, 0.20), dtype=np.float32
    ),
    "recovery_moderate_2to1": np.asarray(
        (0.20, 0.20, 0.20, 0.40), dtype=np.float32
    ),
}

TRAINING_ANCHOR_NAMES = (
    "balanced",
    "search",
    "reconnaissance",
    "strike",
    "recovery",
)
HELD_OUT_PROFILE_NAMES = (
    "search_strike_interp",
    "recon_recovery_interp",
    "smooth_interp",
)

# Preserve the exact historical hard-2 evaluation set while allowing hard-3
# metric calibration to select moderate anchors without overloading old names.
HARD2_EVALUATION_PROFILE_NAMES = TRAINING_ANCHOR_NAMES + HELD_OUT_PROFILE_NAMES
MODERATE_PRIORITY_PROFILE_NAMES = (
    "search_moderate_2to1",
    "reconnaissance_moderate_2to1",
    "strike_moderate_2to1",
    "recovery_moderate_2to1",
)
HARD3_CALIBRATION_PROFILE_NAMES = (
    "balanced",
    *MODERATE_PRIORITY_PROFILE_NAMES,
    *HELD_OUT_PROFILE_NAMES,
)

# Stable names for an opt-in, dynamically calibrated anchor family.  These
# names deliberately do not replace the historical hard-2/hard-3 constants
# above: callers must explicitly request a priority share to use them.
CALIBRATION_PRIORITY_PROFILE_NAMES = (
    "balanced",
    "calibration_search_priority",
    "calibration_reconnaissance_priority",
    "calibration_strike_priority",
    "calibration_recovery_priority",
    "calibration_search_strike_interp",
    "calibration_recon_recovery_interp",
    "calibration_asymmetric_search_recon_recovery",
)


def calibration_priority_profiles(
    priority_share: float,
) -> dict[str, np.ndarray]:
    """Build the pre-registered eight-profile calibration family.

    ``priority_share`` is the mass assigned to one prioritized task type and
    every other type receives ``(1 - priority_share) / 3``.  The three
    held-out profiles are convex combinations of the four single-priority
    anchors, so they remain genuine interpolations for every valid share.

    Historical named profiles are not modified by this helper.
    """

    priority_share = float(priority_share)
    if not np.isfinite(priority_share) or not 0.25 < priority_share < 1.0:
        raise ValueError("priority_share must be finite and lie in (0.25, 1)")
    background_share = (1.0 - priority_share) / 3.0

    anchors: list[np.ndarray] = []
    for priority_index in range(N_TASK_OBJECTIVES):
        vector = np.full(
            N_TASK_OBJECTIVES, background_share, dtype=np.float32
        )
        vector[priority_index] = priority_share
        anchors.append(normalize_simplex(vector))

    search, reconnaissance, strike, recovery = anchors
    profiles = {
        "balanced": np.full(N_TASK_OBJECTIVES, 0.25, dtype=np.float32),
        "calibration_search_priority": search,
        "calibration_reconnaissance_priority": reconnaissance,
        "calibration_strike_priority": strike,
        "calibration_recovery_priority": recovery,
        "calibration_search_strike_interp": normalize_simplex(
            0.5 * search + 0.5 * strike
        ),
        "calibration_recon_recovery_interp": normalize_simplex(
            0.5 * reconnaissance + 0.5 * recovery
        ),
        "calibration_asymmetric_search_recon_recovery": normalize_simplex(
            0.5 * search + 0.3 * reconnaissance + 0.2 * recovery
        ),
    }
    if tuple(profiles) != CALIBRATION_PRIORITY_PROFILE_NAMES:
        raise AssertionError("dynamic calibration profile order changed")
    return {name: vector.copy() for name, vector in profiles.items()}


def normalize_simplex(values: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("preference must contain finite values")
    if np.any(vector < 0):
        raise ValueError("preference weights must be non-negative")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("preference weights must have a positive sum")
    return (vector / total).astype(np.float32)


def _project_simplex_numpy(vector: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, vector.size + 1)
    valid = sorted_values - cumulative / indices > 0
    rho = indices[valid][-1]
    theta = cumulative[valid][-1] / rho
    return np.maximum(vector - theta, 0.0)


def project_dependency_feasible_preference(
    task_preference: Sequence[float] | np.ndarray,
    *,
    iterations: int = 200,
) -> np.ndarray:
    """Euclidean projection onto the chain-feasible task-priority simplex.

    For the five-stage S-R-A-H-S lifecycle, normalized coverage must satisfy
    recon <= 2*search, strike <= recon, and recovery <= strike. Dykstra's
    algorithm projects onto the simplex and these three halfspaces.
    """

    x = normalize_simplex(task_preference).astype(np.float64)
    halfspaces = (
        np.asarray((-2.0, 1.0, 0.0, 0.0)),
        np.asarray((0.0, -1.0, 1.0, 0.0)),
        np.asarray((0.0, 0.0, -1.0, 1.0)),
    )
    corrections = [np.zeros_like(x) for _ in range(1 + len(halfspaces))]
    for _ in range(iterations):
        previous = x.copy()
        candidate = x + corrections[0]
        projected = _project_simplex_numpy(candidate)
        corrections[0] = candidate - projected
        x = projected
        for index, normal in enumerate(halfspaces, start=1):
            candidate = x + corrections[index]
            violation = float(np.dot(normal, candidate))
            projected = (
                candidate
                - max(0.0, violation)
                / float(np.dot(normal, normal))
                * normal
            )
            corrections[index] = candidate - projected
            x = projected
        if np.max(np.abs(x - previous)) < 1e-10:
            break
    x = np.clip(x, 0.0, None)
    return normalize_simplex(x)


def task_preference_profile(name: str) -> np.ndarray:
    try:
        return TASK_PREFERENCE_PROFILES[name].copy()
    except KeyError as error:
        raise ValueError(f"unknown task preference profile: {name}") from error


def preference_allocation_target(
    task_preference: Sequence[float] | np.ndarray,
    *,
    mode: str = DEFAULT_PREFERENCE_TARGET_MODE,
) -> np.ndarray:
    """Resolve the policy/metric target without changing hard-2 defaults.

    The original chain suite uses the dependency-feasible projection.  A
    phase-staggered controllability suite may instead use the nominal user
    request because its rotated chains do not impose the same aggregate type
    count inequalities.  The explicit mode keeps this protocol decision out
    of the task generator and makes historical hard-2 calls reproduce exactly.
    """

    user = normalize_simplex(task_preference)
    if mode == DEFAULT_PREFERENCE_TARGET_MODE:
        return project_dependency_feasible_preference(user)
    if mode == "nominal":
        return user
    raise ValueError(
        "preference target mode must be one of "
        f"{PREFERENCE_TARGET_MODES!r}, got {mode!r}"
    )


def map_task_preference_to_full(
    task_preference: Sequence[float] | np.ndarray,
    *,
    task_mass: float = DEFAULT_TASK_MASS,
    task_floor: float = DEFAULT_TASK_FLOOR,
    auxiliary_weights: Sequence[float] | np.ndarray = DEFAULT_AUXILIARY_WEIGHTS,
    preference_target_mode: str = DEFAULT_PREFERENCE_TARGET_MODE,
) -> np.ndarray:
    """Map a four-way user task preference to the seven optimization weights.

    The floor is applied inside the four-dimensional task simplex.  It keeps
    every necessary task type represented without changing the user's ordering.
    """

    if not 0.0 < task_mass < 1.0:
        raise ValueError("task_mass must lie strictly between zero and one")
    if not 0.0 <= task_floor < 1.0 / N_TASK_OBJECTIVES:
        raise ValueError("task_floor must be in [0, 0.25)")
    task = preference_allocation_target(
        task_preference, mode=preference_target_mode
    )
    floored = task_floor + (1.0 - N_TASK_OBJECTIVES * task_floor) * task
    auxiliary = normalize_simplex(auxiliary_weights) * (1.0 - task_mass)
    full = np.concatenate((task_mass * floored, auxiliary)).astype(np.float32)
    return full / full.sum()


def split_preference(
    preference: Sequence[float] | np.ndarray,
    *,
    preference_target_mode: str = DEFAULT_PREFERENCE_TARGET_MODE,
) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(preference, dtype=np.float32).reshape(-1)
    if vector.size == N_TASK_OBJECTIVES:
        user = normalize_simplex(vector)
        return user, map_task_preference_to_full(
            user, preference_target_mode=preference_target_mode
        )
    if vector.size == N_OBJECTIVES:
        full = normalize_simplex(vector)
        user = normalize_simplex(full[:N_TASK_OBJECTIVES])
        return user, full
    raise ValueError("preference must contain either four or seven weights")


def sample_training_task_preference(
    rng: np.random.Generator,
    *,
    concentration: float = 0.7,
    anchor_probability: float = 0.30,
    anchor_jitter: float = 0.10,
    sampler_mode: str = LEGACY_PREFERENCE_SAMPLER_MODE,
    calibration_priority_share: float | None = None,
) -> np.ndarray:
    """Sample a protocol-selected continuous per-episode preference.

    The legacy branch is the exact hard-2/hard-3 implementation and retains its
    historical RNG call order. The moderate branch samples barycentric weights
    over calibrated priority anchors, so a share of 0.40 confines every task
    coordinate to the convex-hull interval [0.20, 0.40] and never reads the old
    0.70/0.10 anchors.
    """

    if sampler_mode not in PREFERENCE_SAMPLER_MODES:
        raise ValueError(
            "preference sampler mode must be one of "
            f"{PREFERENCE_SAMPLER_MODES!r}, got {sampler_mode!r}"
        )
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be positive")
    if not 0.0 <= anchor_probability <= 1.0:
        raise ValueError("anchor_probability must be in [0, 1]")
    if not 0.0 < anchor_jitter <= 1.0:
        raise ValueError("anchor_jitter must be in (0, 1]")
    if sampler_mode == LEGACY_PREFERENCE_SAMPLER_MODE:
        if calibration_priority_share is not None:
            raise ValueError(
                "legacy_extreme_dirichlet cannot use calibration_priority_share"
            )
        # Exact historical branch: do not reorder these RNG calls.
        continuous = rng.dirichlet(
            np.full(N_TASK_OBJECTIVES, concentration)
        )
        if rng.random() < anchor_probability:
            anchor_name = str(rng.choice(TRAINING_ANCHOR_NAMES))
            anchor = TASK_PREFERENCE_PROFILES[anchor_name]
            continuous = (
                (1.0 - anchor_jitter) * anchor
                + anchor_jitter * continuous
            )
        return normalize_simplex(continuous)

    if calibration_priority_share is None:
        raise ValueError(
            "moderate_anchor_hull requires calibration_priority_share"
        )
    profiles = calibration_priority_profiles(calibration_priority_share)
    anchors = np.stack(
        [
            profiles[name]
            for name in CALIBRATION_PRIORITY_PROFILE_NAMES[1:5]
        ],
        axis=0,
    ).astype(np.float64)
    barycentric = rng.dirichlet(
        np.full(N_TASK_OBJECTIVES, concentration)
    )
    if rng.random() < anchor_probability:
        anchor_index = int(rng.integers(N_TASK_OBJECTIVES))
        one_hot = np.zeros(N_TASK_OBJECTIVES, dtype=np.float64)
        one_hot[anchor_index] = 1.0
        barycentric = (
            (1.0 - anchor_jitter) * one_hot
            + anchor_jitter * barycentric
        )
    return normalize_simplex(barycentric @ anchors)


def preference_sampler_config(
    *,
    sampler_mode: str = LEGACY_PREFERENCE_SAMPLER_MODE,
    calibration_priority_share: float | None = None,
    concentration: float = 0.7,
    anchor_probability: float = 0.30,
    anchor_jitter: float = 0.10,
) -> dict[str, float | str | None]:
    """Validate and serialize training sampler parameters."""

    # Use a private generator so validation never consumes training RNG state.
    sample_training_task_preference(
        np.random.default_rng(0),
        sampler_mode=sampler_mode,
        calibration_priority_share=calibration_priority_share,
        concentration=concentration,
        anchor_probability=anchor_probability,
        anchor_jitter=anchor_jitter,
    )
    return {
        "mode": sampler_mode,
        "calibration_priority_share": (
            float(calibration_priority_share)
            if calibration_priority_share is not None
            else None
        ),
        "concentration": float(concentration),
        "anchor_probability": float(anchor_probability),
        "anchor_jitter": float(anchor_jitter),
    }


def preference_alignment(
    actual_mix: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    actual = np.asarray(actual_mix, dtype=np.float64).reshape(-1)
    target_array = normalize_simplex(target).astype(np.float64)
    if actual.shape != target_array.shape:
        raise ValueError("actual and target preferences must have equal shape")
    actual = np.clip(actual, 0.0, None)
    if float(actual.sum()) > 0:
        actual = actual / actual.sum()
    difference = actual - target_array
    actual_norm = float(np.linalg.norm(actual))
    target_norm = float(np.linalg.norm(target_array))
    cosine = (
        float(np.dot(actual, target_array) / (actual_norm * target_norm))
        if actual_norm > 0 and target_norm > 0
        else 0.0
    )
    return {
        "l1": float(np.abs(difference).sum()),
        "l2": float(np.linalg.norm(difference)),
        "cosine": cosine,
    }


def assignment_priority_weight(
    assignment_time: float,
    mission_deadline: float,
    *,
    decay: float = DEFAULT_ASSIGNMENT_PRIORITY_DECAY,
) -> float:
    """Return the deadline-normalized temporal priority of one assignment.

    ``decay=1`` is the frozen PCRL-v0-hard-2 definition.  A value of zero
    intentionally removes temporal decay while retaining unique-assignment
    accounting; larger values place progressively more weight on early
    assignments.
    """

    assignment_time = float(assignment_time)
    mission_deadline = float(mission_deadline)
    decay = float(decay)
    if not np.isfinite(assignment_time) or assignment_time < 0:
        raise ValueError("assignment_time must be finite and non-negative")
    if not np.isfinite(mission_deadline):
        raise ValueError("mission_deadline must be finite")
    if not np.isfinite(decay) or decay < 0:
        raise ValueError("assignment priority decay must be finite and non-negative")
    return float(
        np.exp(-decay * assignment_time / max(1.0, mission_deadline))
    )


class PreferencePaperEnv(PaperAlignedUAVEnv):
    """Non-mutating PCRL wrapper around the frozen paper-aligned environment.

    It exposes a seven-dimensional vector reward while retaining the original
    graph observation, action mask, event synchronization and stale belief.
    ``event_input`` is an inert interface reserved for a future world model; it
    is logged but cannot alter the physical environment in PCRL-v0.
    """

    n_objectives = N_OBJECTIVES
    objective_names = OBJECTIVE_NAMES

    def __init__(
        self,
        config: PaperEnvConfig | None = None,
        preference: Sequence[float] | np.ndarray | None = None,
        *,
        task_release_mode: str = "chain",
        assignment_priority_decay: float = DEFAULT_ASSIGNMENT_PRIORITY_DECAY,
        preference_target_mode: str = DEFAULT_PREFERENCE_TARGET_MODE,
        capability_coverage_mode: str = LEGACY_CAPABILITY_COVERAGE_MODE,
    ):
        if task_release_mode not in {"chain", "phase_staggered"}:
            raise ValueError("task_release_mode must be 'chain' or 'phase_staggered'")
        if (
            not np.isfinite(assignment_priority_decay)
            or assignment_priority_decay < 0
        ):
            raise ValueError(
                "assignment_priority_decay must be finite and non-negative"
            )
        self.task_release_mode = task_release_mode
        self.assignment_priority_decay = float(assignment_priority_decay)
        if capability_coverage_mode not in CAPABILITY_COVERAGE_MODES:
            raise ValueError(
                "capability_coverage_mode must be one of "
                f"{CAPABILITY_COVERAGE_MODES!r}"
            )
        self.capability_coverage_mode = capability_coverage_mode
        # Validate before the frozen parent environment allocates any state.
        preference_allocation_target(
            task_preference_profile("balanced"), mode=preference_target_mode
        )
        self.preference_target_mode = preference_target_mode
        super().__init__(config)
        self.user_task_preference = task_preference_profile("balanced")
        self.task_allocation_target = preference_allocation_target(
            self.user_task_preference, mode=self.preference_target_mode
        )
        self.preference = map_task_preference_to_full(
            self.user_task_preference,
            preference_target_mode=self.preference_target_mode,
        )
        self.vector_return = np.zeros(N_OBJECTIVES, dtype=np.float32)
        self._ever_available: set[int] = set()
        self._ever_reachable: set[int] = set()
        self._available_by_type = np.zeros(N_TASK_OBJECTIVES, dtype=np.int64)
        self._reachable_by_type = np.zeros(N_TASK_OBJECTIVES, dtype=np.int64)
        self._deadline_assigned_tasks: set[int] = set()
        self._deadline_assigned_by_type = np.zeros(
            N_TASK_OBJECTIVES, dtype=np.float32
        )
        self._deadline_assignment_effort_by_type = np.zeros(
            N_TASK_OBJECTIVES, dtype=np.float32
        )
        self._deadline_assignment_priority_by_type = np.zeros(
            N_TASK_OBJECTIVES, dtype=np.float32
        )
        self._failure_events = 0
        self.last_event_input: dict[str, Any] | None = None
        if preference is not None:
            self.set_preference(preference)

    def _sample_uav(self, index: int) -> PaperUAVState:
        """Preserve the frozen parent sampler unless hard-5 opts in.

        The balanced mode post-processes the already sampled capabilities and
        consumes no additional random numbers.  At reset, task type ``t`` is
        executable by exactly ``min(2, active_uavs)`` active UAVs using cyclic
        primary/secondary specialists.  Later UAV failures are intentionally
        allowed to reduce that coverage.
        """

        uav = super()._sample_uav(index)
        if (
            self.capability_coverage_mode == BALANCED_CAPABILITY_COVERAGE_MODE
            and uav.active
        ):
            active_uavs = self.config.active_uavs
            specialist_floor = max(
                0.82, self.config.capability_threshold + 0.05
            )
            blocked_value = 0.5 * self.config.capability_threshold
            capabilities = np.full(
                N_TASK_OBJECTIVES, blocked_value, dtype=np.float32
            )
            for task_type in range(N_TASK_OBJECTIVES):
                specialists = {
                    task_type % active_uavs,
                    (task_type + 1) % active_uavs,
                }
                if index in specialists:
                    capabilities[task_type] = max(
                        specialist_floor, float(uav.capabilities[task_type])
                    )
            uav.capabilities = capabilities
        return uav

    def _sample_tasks(self):
        if self.task_release_mode == "chain":
            return super()._sample_tasks()
        tasks = []
        active_target = self.config.initial_tasks
        chain_length = self.config.task_chain_length
        full_chains = active_target // chain_length
        for chain_index in range(full_chains):
            predecessor = -1
            center = self.rng.uniform(0.1, 0.9, size=2)
            phase = chain_index % N_TASK_OBJECTIVES
            for step in range(chain_length):
                task = self._new_task(
                    (phase + step) % N_TASK_OBJECTIVES,
                    predecessor,
                    center=center,
                )
                tasks.append(task)
                predecessor = len(tasks) - 1
        while len(tasks) < active_target:
            tasks.append(
                self._new_task(len(tasks) % N_TASK_OBJECTIVES, predecessor=-1)
            )
        while len(tasks) < self.config.max_tasks:
            tasks.append(
                self._new_task(
                    len(tasks) % N_TASK_OBJECTIVES,
                    predecessor=-1,
                    active=False,
                )
            )
        return tasks

    def set_preference(self, preference: Sequence[float] | np.ndarray) -> None:
        user, full = split_preference(
            preference, preference_target_mode=self.preference_target_mode
        )
        self.user_task_preference = user
        self.task_allocation_target = preference_allocation_target(
            user, mode=self.preference_target_mode
        )
        self.preference = full

    def reset(
        self,
        seed: int | None = None,
        preference: Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if preference is not None:
            self.set_preference(preference)
        observation = super().reset(seed=seed)
        self.vector_return = np.zeros(N_OBJECTIVES, dtype=np.float32)
        self._ever_available = set()
        self._ever_reachable = set()
        self._available_by_type = np.zeros(N_TASK_OBJECTIVES, dtype=np.int64)
        self._reachable_by_type = np.zeros(N_TASK_OBJECTIVES, dtype=np.int64)
        self._deadline_assigned_tasks = set()
        self._deadline_assigned_by_type = np.zeros(
            N_TASK_OBJECTIVES, dtype=np.float32
        )
        self._deadline_assignment_effort_by_type = np.zeros(
            N_TASK_OBJECTIVES, dtype=np.float32
        )
        self._deadline_assignment_priority_by_type = np.zeros(
            N_TASK_OBJECTIVES, dtype=np.float32
        )
        self._failure_events = 0
        self.last_event_input = None
        self._refresh_task_denominators()
        return self.observe()

    def _task_is_reachable(self, task_index: int) -> bool:
        task = self.tasks[task_index]
        return any(
            uav.active
            and uav.alive
            and uav.capabilities[task.task_type] >= self.config.capability_threshold
            for uav in self.uavs
        )

    def _refresh_task_denominators(self) -> None:
        for index, task in enumerate(self.tasks):
            if not task.active:
                continue
            if index not in self._ever_available:
                self._ever_available.add(index)
                self._available_by_type[task.task_type] += 1
            if index not in self._ever_reachable and self._task_is_reachable(index):
                self._ever_reachable.add(index)
                self._reachable_by_type[task.task_type] += 1

    def _completed_by_type(self, *, deadline_only: bool) -> np.ndarray:
        counts = np.zeros(N_TASK_OBJECTIVES, dtype=np.float32)
        deadline = self.config.mission_deadline
        for task in self.tasks:
            if not task.active or not task.completed:
                continue
            if deadline_only and deadline > 0:
                if task.arrival_time > deadline or task.completion_time > deadline + 1e-7:
                    continue
            counts[task.task_type] += 1.0
        return counts

    @staticmethod
    def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(numerator, dtype=np.float32)
        np.divide(
            numerator,
            denominator,
            out=result,
            where=np.asarray(denominator) > 0,
        )
        return result

    def _deadline_coverage(self) -> np.ndarray:
        completed = self._completed_by_type(deadline_only=True)
        return self._safe_ratio(completed, self._available_by_type)

    def _deadline_assignment_coverage(self) -> np.ndarray:
        return self._safe_ratio(
            self._deadline_assigned_by_type, self._available_by_type
        )

    def _deadline_assignment_priority_coverage(self) -> np.ndarray:
        return self._safe_ratio(
            self._deadline_assignment_priority_by_type, self._available_by_type
        )

    def _preference_deficit(self) -> np.ndarray:
        coverage = self._deadline_assignment_priority_coverage()
        mix = self._safe_ratio(coverage, np.asarray(coverage.sum()))
        return (self.task_allocation_target - mix).astype(np.float32)

    def observe(self) -> dict[str, np.ndarray]:
        observation = super().observe()
        observation["preference_deficit"] = self._preference_deficit()
        return observation

    def true_observation(self) -> dict[str, np.ndarray]:
        observation = super().true_observation()
        observation["preference_deficit"] = self._preference_deficit()
        return observation

    def _safety_cost(self) -> float:
        unresolved_reallocations = max(
            0, self.reallocated_tasks - self.reallocation_successes
        )
        return float(self.invalid_actions + self._failure_events + unresolved_reallocations)

    def step(
        self,
        action: int,
        sync_mode: SyncMode = "event",
        *,
        event_input: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        previous_assignment_priority = self._deadline_assignment_priority_coverage()
        previous_makespan = self.makespan
        assignment_time = self.current_time
        true_mask = self._valid_mask_for(self.uavs, self.tasks)
        true_pair_assignment = (
            0 <= int(action) < self.noop_action and bool(true_mask[int(action)])
        )
        assigned_task_index = (
            divmod(int(action), self.config.max_tasks)[1]
            if true_pair_assignment
            else -1
        )
        previous_communication = self.communication_events
        previous_safety = self._safety_cost()
        if event_input is not None:
            self.last_event_input = dict(event_input)

        observation, base_reward, done, info = super().step(
            action, sync_mode=sync_mode
        )
        failure_increment = sum(
            str(record.get("event_type")) == "uav_failure"
            for record in info.get("events", [])
        )
        self._failure_events += int(failure_increment)
        self._refresh_task_denominators()
        deadline = self.config.mission_deadline
        if (
            true_pair_assignment
            and assigned_task_index not in self._deadline_assigned_tasks
            and (deadline <= 0 or assignment_time <= deadline + 1e-7)
        ):
            task = self.tasks[assigned_task_index]
            self._deadline_assigned_tasks.add(assigned_task_index)
            self._deadline_assigned_by_type[task.task_type] += 1.0
            self._deadline_assignment_effort_by_type[task.task_type] += max(
                0.0, float(task.processing_time)
            )
            self._deadline_assignment_priority_by_type[task.task_type] += float(
                assignment_priority_weight(
                    assignment_time,
                    deadline,
                    decay=self.assignment_priority_decay,
                )
            )
        current_assignment_priority = self._deadline_assignment_priority_coverage()

        task_reward = current_assignment_priority - previous_assignment_priority
        efficiency_scale = max(
            1.0,
            self.config.mission_deadline,
            previous_makespan,
        )
        efficiency_reward = float(info["paper_reward"]) / efficiency_scale
        communication_reward = -float(
            self.communication_events - previous_communication
        ) / max(1, self.config.max_decisions)
        safety_reward = -float(self._safety_cost() - previous_safety) / max(
            1, self.config.max_tasks + self.config.max_uavs
        )
        vector_reward = np.concatenate(
            (
                task_reward.astype(np.float32),
                np.asarray(
                    (efficiency_reward, communication_reward, safety_reward),
                    dtype=np.float32,
                ),
            )
        )
        self.vector_return += vector_reward
        scalarized_reward = float(np.dot(self.preference, vector_reward))
        info = dict(info)
        info.update(
            {
                "base_scalar_reward": float(base_reward),
                "vector_reward": vector_reward.copy(),
                "scalarized_preference_reward": scalarized_reward,
                "task_preference": self.user_task_preference.copy(),
                "optimization_preference": self.preference.copy(),
                "event_input": self.last_event_input,
            }
        )
        # ``super().step`` builds its observation before the PCRL assignment
        # accounting above. Refresh so the next action sees the new deficit.
        observation = self.observe()
        return observation, scalarized_reward, done, info

    def preference_metrics(self) -> dict[str, Any]:
        completed = self._completed_by_type(deadline_only=False)
        deadline_completed = self._completed_by_type(deadline_only=True)
        raw_mix = self._safe_ratio(completed, np.asarray(completed.sum()))
        deadline_raw_mix = self._safe_ratio(
            deadline_completed, np.asarray(deadline_completed.sum())
        )
        availability_coverage = self._safe_ratio(
            deadline_completed, self._available_by_type
        )
        reachable_coverage = self._safe_ratio(
            deadline_completed, self._reachable_by_type
        )
        availability_mix = self._safe_ratio(
            availability_coverage, np.asarray(availability_coverage.sum())
        )
        reachable_mix = self._safe_ratio(
            reachable_coverage, np.asarray(reachable_coverage.sum())
        )
        assignment_coverage = self._deadline_assignment_coverage()
        assignment_mix = self._safe_ratio(
            assignment_coverage, np.asarray(assignment_coverage.sum())
        )
        raw_assignment_mix = self._safe_ratio(
            self._deadline_assigned_by_type,
            np.asarray(self._deadline_assigned_by_type.sum()),
        )
        effort_mix = self._safe_ratio(
            self._deadline_assignment_effort_by_type,
            np.asarray(self._deadline_assignment_effort_by_type.sum()),
        )
        assignment_priority_coverage = (
            self._deadline_assignment_priority_coverage()
        )
        assignment_priority_mix = self._safe_ratio(
            assignment_priority_coverage,
            np.asarray(assignment_priority_coverage.sum()),
        )
        raw_alignment = preference_alignment(raw_mix, self.task_allocation_target)
        normalized_alignment = preference_alignment(
            availability_mix, self.task_allocation_target
        )
        reachable_alignment = preference_alignment(
            reachable_mix, self.task_allocation_target
        )
        assignment_alignment = preference_alignment(
            assignment_priority_mix, self.task_allocation_target
        )
        effort_alignment = preference_alignment(
            effort_mix, self.task_allocation_target
        )
        nominal_assignment_alignment = preference_alignment(
            assignment_priority_mix, self.user_task_preference
        )
        dependency_feasible_assignment_alignment = preference_alignment(
            assignment_priority_mix,
            project_dependency_feasible_preference(self.user_task_preference),
        )
        return {
            "objective_names": list(OBJECTIVE_NAMES),
            "task_preference": self.user_task_preference.tolist(),
            "allocation_target_preference": self.task_allocation_target.tolist(),
            "preference_target_mode": self.preference_target_mode,
            "optimization_preference": self.preference.tolist(),
            "available_tasks_by_type": self._available_by_type.astype(float).tolist(),
            "reachable_tasks_by_type": self._reachable_by_type.astype(float).tolist(),
            "completed_by_type": completed.tolist(),
            "deadline_completed_by_type": deadline_completed.tolist(),
            "raw_completion_mix": raw_mix.tolist(),
            "deadline_raw_completion_mix": deadline_raw_mix.tolist(),
            "availability_coverage_by_type": availability_coverage.tolist(),
            "reachable_coverage_by_type": reachable_coverage.tolist(),
            "availability_normalized_mix": availability_mix.tolist(),
            "reachable_normalized_mix": reachable_mix.tolist(),
            "deadline_assigned_by_type": self._deadline_assigned_by_type.tolist(),
            "deadline_assignment_effort_by_type": self._deadline_assignment_effort_by_type.tolist(),
            "deadline_assignment_priority_by_type": self._deadline_assignment_priority_by_type.tolist(),
            "deadline_assignment_coverage_by_type": assignment_coverage.tolist(),
            "deadline_assignment_priority_coverage_by_type": assignment_priority_coverage.tolist(),
            "raw_assignment_mix": raw_assignment_mix.tolist(),
            "availability_normalized_assignment_mix": assignment_mix.tolist(),
            "priority_weighted_assignment_mix": assignment_priority_mix.tolist(),
            "assignment_effort_mix": effort_mix.tolist(),
            "preference_l1_raw": raw_alignment["l1"],
            "preference_l2_raw": raw_alignment["l2"],
            "preference_cosine_raw": raw_alignment["cosine"],
            "preference_l1": assignment_alignment["l1"],
            "preference_l2": assignment_alignment["l2"],
            "preference_cosine": assignment_alignment["cosine"],
            "preference_l1_nominal": nominal_assignment_alignment["l1"],
            "preference_l2_nominal": nominal_assignment_alignment["l2"],
            "preference_cosine_nominal": nominal_assignment_alignment["cosine"],
            "preference_l1_dependency_feasible": dependency_feasible_assignment_alignment["l1"],
            "preference_l2_dependency_feasible": dependency_feasible_assignment_alignment["l2"],
            "preference_cosine_dependency_feasible": dependency_feasible_assignment_alignment["cosine"],
            "preference_l1_completion": normalized_alignment["l1"],
            "preference_l2_completion": normalized_alignment["l2"],
            "preference_cosine_completion": normalized_alignment["cosine"],
            "preference_l1_effort": effort_alignment["l1"],
            "preference_l2_effort": effort_alignment["l2"],
            "preference_cosine_effort": effort_alignment["cosine"],
            "preference_l1_reachable": reachable_alignment["l1"],
            "preference_l2_reachable": reachable_alignment["l2"],
            "preference_cosine_reachable": reachable_alignment["cosine"],
            "minimum_task_coverage": float(availability_coverage.min()),
            "vector_return": self.vector_return.tolist(),
            "external_event_interface_used": float(self.last_event_input is not None),
            "task_release_mode": self.task_release_mode,
            "assignment_priority_decay": self.assignment_priority_decay,
            "capability_coverage_mode": self.capability_coverage_mode,
        }

    def metrics(self) -> dict[str, Any]:
        return {**super().metrics(), **self.preference_metrics()}


def pcrl_implementation_hash(workspace: Path | None = None) -> str:
    root = workspace or Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/uav_assignment/pcrl_v0.py",
        "src/uav_assignment/pcrl_models.py",
        "src/uav_assignment/pcrl_training.py",
        "train_pcrl_v0.py",
        "evaluate_pcrl_v0.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        if not path.exists():
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(
        json.dumps(
            {
                "version": PCRL_VERSION,
                "objective_names": OBJECTIVE_NAMES,
                "task_mass": DEFAULT_TASK_MASS,
                "auxiliary_weights": DEFAULT_AUXILIARY_WEIGHTS.tolist(),
                "task_floor": DEFAULT_TASK_FLOOR,
                "assignment_priority_decay": DEFAULT_ASSIGNMENT_PRIORITY_DECAY,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def pcrl_scenario_hash(
    config: PaperEnvConfig | dict[str, Any],
    capability_coverage_mode: str = LEGACY_CAPABILITY_COVERAGE_MODE,
) -> str:
    """Hash the PCRL environment extension without changing frozen GPPO files."""

    if capability_coverage_mode not in CAPABILITY_COVERAGE_MODES:
        raise ValueError(
            "capability_coverage_mode must be one of "
            f"{CAPABILITY_COVERAGE_MODES!r}"
        )
    from .gppo_v2 import config_hash

    base_hash = config_hash(config)
    if capability_coverage_mode == LEGACY_CAPABILITY_COVERAGE_MODE:
        return base_hash
    payload = json.dumps(
        {
            "base_scenario_hash": base_hash,
            "capability_coverage_mode": capability_coverage_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from uav_assignment.gppo_v2 import config_hash, implementation_hash
from uav_assignment.paper_env import PaperEnvConfig
from uav_assignment.pcrl_models import (
    LEGACY_PREFERENCE_INPUT_MODE,
    MASS_CONTROL_PREFERENCE_INPUT_MODE,
    PREFERENCE_INPUT_MODES,
    PreferenceConditionedPaperActorCritic,
)
from uav_assignment.pcrl_training import (
    PCRLTrainingConfig,
    collect_pcrl_rollouts,
    model_conditioning_inputs,
    pcrl_ppo_update,
    tensor_observation,
)
from uav_assignment.pcrl_v0 import (
    CALIBRATION_PRIORITY_PROFILE_NAMES,
    CAPABILITY_COVERAGE_MODES,
    DEFAULT_PREFERENCE_TARGET_MODE,
    LEGACY_CAPABILITY_COVERAGE_MODE,
    LEGACY_PREFERENCE_SAMPLER_MODE,
    MODERATE_PREFERENCE_SAMPLER_MODE,
    PREFERENCE_TARGET_MODES,
    PREFERENCE_SAMPLER_MODES,
    PCRL_VERSION,
    PreferencePaperEnv,
    calibration_priority_profiles,
    pcrl_implementation_hash,
    pcrl_scenario_hash,
    preference_sampler_config,
    task_preference_profile,
)


def parse_scale(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", maxsplit=1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-aligned PCRL-v0")
    parser.add_argument("--scenario-config", type=Path, default=Path("configs/gppo_v2_hard.json"))
    parser.add_argument("--protocol-config", type=Path)
    parser.add_argument("--protocol-version")
    parser.add_argument("--protocol-variant", default="")
    parser.add_argument("--artifact-group", default="formal")
    parser.add_argument("--protocol-config-sha256")
    parser.add_argument("--output-namespace")
    parser.add_argument("--base-protocol")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--graph-mode", choices=("none", "adaptive", "single_head"), required=True)
    parser.add_argument("--algorithm", choices=("preco", "ls", "sdmgrad"), default="preco")
    parser.add_argument("--no-conditioning", action="store_true")
    parser.add_argument(
        "--preference-input-mode",
        choices=PREFERENCE_INPUT_MODES,
        default=LEGACY_PREFERENCE_INPUT_MODE,
    )
    parser.add_argument(
        "--capability-coverage-mode",
        choices=CAPABILITY_COVERAGE_MODES,
        default=LEGACY_CAPABILITY_COVERAGE_MODE,
    )
    parser.add_argument(
        "--preference-sampler-mode",
        choices=PREFERENCE_SAMPLER_MODES,
        default=LEGACY_PREFERENCE_SAMPLER_MODE,
    )
    parser.add_argument("--calibration-priority-share", type=float)
    parser.add_argument("--preference-sampler-concentration", type=float, default=0.7)
    parser.add_argument("--preference-sampler-anchor-probability", type=float, default=0.30)
    parser.add_argument("--preference-sampler-anchor-jitter", type=float, default=0.10)
    parser.add_argument(
        "--fixed-training-profile",
        help="LS control: train every episode with one fixed task preference",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--episodes-per-update", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--preco-lambda", type=float, default=0.25)
    parser.add_argument("--preference-aux-coefficient", type=float, default=0.50)
    parser.add_argument("--initial-task-preference-gain", type=float)
    parser.add_argument("--preference-group-episodes", type=int)
    parser.add_argument(
        "--advantage-normalization-mode",
        choices=("per_objective", "shared_scale", "none"),
    )
    parser.add_argument(
        "--preco-value-transform",
        choices=("legacy_min_shift", "softplus"),
    )
    parser.add_argument("--residual-warmup-updates", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--profile-paired-validation", action="store_true")
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=(
            "legacy_l1_lexicographic",
            "gppo_efficiency_constrained_v1",
            "gppo_efficiency_relative_v2",
        ),
        default="legacy_l1_lexicographic",
    )
    parser.add_argument("--max-deadline-completion-drop", type=float, default=0.03)
    parser.add_argument("--max-makespan-increase-ratio", type=float, default=0.05)
    parser.add_argument("--max-coverage-loss", type=float, default=0.05)
    parser.add_argument("--max-invalid-actions", type=float, default=0.0)
    parser.add_argument("--max-invalid-action-increase", type=float, default=0.0)
    parser.add_argument("--save-validation-candidates", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--validation-profiles",
        nargs="+",
        default=("balanced", "search", "strike"),
    )
    parser.add_argument(
        "--task-release-modes",
        nargs="+",
        choices=("chain", "phase_staggered"),
        default=("phase_staggered",),
    )
    parser.add_argument("--phase-staggered-deadline-scale", type=float, default=0.70)
    parser.add_argument("--assignment-priority-decay", type=float, default=1.0)
    parser.add_argument(
        "--preference-target-mode",
        choices=PREFERENCE_TARGET_MODES,
        default=None,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _protocol_value(args: argparse.Namespace, protocol: dict[str, object], name: str):
    value = getattr(args, name)
    return protocol[name] if value is None else value


@torch.no_grad()
def evaluate_validation(
    model: PreferenceConditionedPaperActorCritic,
    config: PaperEnvConfig,
    *,
    scales: tuple[tuple[int, int], ...],
    scale_deadlines: dict[tuple[int, int], float],
    profiles: tuple[str, ...],
    task_release_modes: tuple[str, ...],
    phase_staggered_deadline_scale: float,
    assignment_priority_decay: float,
    preference_target_mode: str,
    dynamic_profiles: dict[str, np.ndarray] | None,
    capability_coverage_mode: str,
    profile_paired: bool,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, object]] = []
    for scale_index, (active_uavs, initial_tasks) in enumerate(scales):
        scale_config = replace(
            config,
            active_uavs=active_uavs,
            initial_tasks=initial_tasks,
            mission_deadline=scale_deadlines.get(
                (active_uavs, initial_tasks), config.mission_deadline
            ),
        )
        for mode_index, task_release_mode in enumerate(task_release_modes):
            for profile_index, profile_name in enumerate(profiles):
                if dynamic_profiles is None:
                    task_preference = task_preference_profile(profile_name)
                else:
                    try:
                        task_preference = dynamic_profiles[profile_name]
                    except KeyError as error:
                        raise ValueError(
                            "moderate sampler validation profile must use the "
                            f"dynamic calibration family: {profile_name!r}"
                        ) from error
                for episode in range(episodes):
                    episode_seed = (
                        seed
                        + 100_000 * mode_index
                        + 10_000 * scale_index
                        + (0 if profile_paired else 1_000 * profile_index)
                        + episode
                    )
                    env = PreferencePaperEnv(
                        replace(
                            scale_config,
                            mission_deadline=(
                                scale_config.mission_deadline
                                * phase_staggered_deadline_scale
                                if task_release_mode == "phase_staggered"
                                else scale_config.mission_deadline
                            ),
                        ),
                        preference=task_preference,
                        task_release_mode=task_release_mode,
                        assignment_priority_decay=assignment_priority_decay,
                        preference_target_mode=preference_target_mode,
                        capability_coverage_mode=capability_coverage_mode,
                    )
                    observation = env.reset(seed=episode_seed)
                    done = False
                    while not done:
                        (
                            policy_observation,
                            policy_task_preference,
                            policy_objective_preference,
                        ) = model_conditioning_inputs(
                            model,
                            tensor_observation(observation),
                            torch.as_tensor(
                                env.user_task_preference, dtype=torch.float32
                            ),
                            torch.as_tensor(env.preference, dtype=torch.float32),
                        )
                        action, _, _ = model.act(
                            policy_observation,
                            policy_objective_preference,
                            task_preference=policy_task_preference,
                            deterministic=True,
                        )
                        observation, _, done, _ = env.step(
                            int(action.item()), sync_mode="event"
                        )
                    metrics = env.metrics()
                    metrics["profile"] = profile_name
                    rows.append(metrics)
    scalar_fields = (
        "preference_l1",
        "preference_l2",
        "preference_cosine",
        "deadline_completion_rate",
        "makespan",
        "communication_events",
        "invalid_actions",
        "minimum_task_coverage",
    )
    result = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in scalar_fields
    }
    for profile in profiles:
        selected = [row for row in rows if row["profile"] == profile]
        result[f"{profile}_preference_l1"] = float(
            np.mean([float(row["preference_l1"]) for row in selected])
        )
        result[f"{profile}_deadline_completion_rate"] = float(
            np.mean([float(row["deadline_completion_rate"]) for row in selected])
        )
    return result


def constrained_selection_diagnostics(
    candidate: dict[str, float],
    reference: dict[str, float],
    *,
    max_deadline_completion_drop: float,
    max_makespan_increase_ratio: float,
    max_coverage_loss: float,
    max_invalid_actions: float,
    max_invalid_action_increase: float | None = None,
) -> dict[str, object]:
    fields = (
        "preference_l1",
        "deadline_completion_rate",
        "makespan",
        "minimum_task_coverage",
        "invalid_actions",
    )
    for source_name, source in (("candidate", candidate), ("reference", reference)):
        for field in fields:
            value = float(source[field])
            if not np.isfinite(value):
                raise ValueError(f"{source_name} {field} must be finite")
    reference_makespan = float(reference["makespan"])
    if reference_makespan <= 0:
        raise ValueError("reference makespan must be positive")
    tolerances = (
        max_deadline_completion_drop,
        max_makespan_increase_ratio,
        max_coverage_loss,
    )
    if any(not np.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("checkpoint selection tolerances must be finite and non-negative")
    if not np.isfinite(max_invalid_actions) or max_invalid_actions < 0:
        raise ValueError("max_invalid_actions must be finite and non-negative")
    if max_invalid_action_increase is not None and (
        not np.isfinite(max_invalid_action_increase)
        or max_invalid_action_increase < 0
    ):
        raise ValueError(
            "max_invalid_action_increase must be finite and non-negative"
        )

    deadline_delta = float(candidate["deadline_completion_rate"]) - float(
        reference["deadline_completion_rate"]
    )
    makespan_increase_ratio = (
        float(candidate["makespan"]) / reference_makespan - 1.0
    )
    coverage_delta = float(candidate["minimum_task_coverage"]) - float(
        reference["minimum_task_coverage"]
    )
    invalid_actions = float(candidate["invalid_actions"])
    invalid_action_delta = invalid_actions - float(reference["invalid_actions"])
    atol = 1e-12
    reasons: list[str] = []
    if deadline_delta < -max_deadline_completion_drop - atol:
        reasons.append("deadline_completion_drop")
    if makespan_increase_ratio > max_makespan_increase_ratio + atol:
        reasons.append("makespan_increase")
    if coverage_delta < -max_coverage_loss - atol:
        reasons.append("coverage_loss")
    if max_invalid_action_increase is None:
        if invalid_actions > max_invalid_actions + atol:
            reasons.append("invalid_actions")
        invalid_margin = max_invalid_actions - invalid_actions
    else:
        if invalid_action_delta > max_invalid_action_increase + atol:
            reasons.append("invalid_action_increase")
        invalid_margin = max_invalid_action_increase - invalid_action_delta

    def normalized_margin(value: float, tolerance: float, *, lower: bool) -> float:
        scale = max(tolerance, 1e-12)
        return (value + tolerance) / scale if lower else (tolerance - value) / scale

    margins = {
        "deadline_completion": normalized_margin(
            deadline_delta, max_deadline_completion_drop, lower=True
        ),
        "makespan": normalized_margin(
            makespan_increase_ratio, max_makespan_increase_ratio, lower=False
        ),
        "coverage": normalized_margin(
            coverage_delta, max_coverage_loss, lower=True
        ),
        "invalid_actions": invalid_margin,
    }
    return {
        "feasible": not reasons,
        "reasons": reasons,
        "deadline_completion_delta": deadline_delta,
        "makespan_increase_ratio": makespan_increase_ratio,
        "minimum_task_coverage_delta": coverage_delta,
        "invalid_actions": invalid_actions,
        "invalid_action_delta": invalid_action_delta,
        "normalized_margins": margins,
        "minimum_normalized_margin": min(margins.values()),
        "sum_normalized_margin": sum(margins.values()),
    }


def checkpoint_selection_key(
    validation: dict[str, float],
    diagnostics: dict[str, object],
    *,
    update: int,
) -> tuple[float, ...]:
    if bool(diagnostics["feasible"]):
        return (
            1.0,
            -float(validation["preference_l1"]),
            float(validation["deadline_completion_rate"]),
            -float(validation["makespan"]),
            float(validation["minimum_task_coverage"]),
            -float(update),
        )
    return (
        0.0,
        float(diagnostics["minimum_normalized_margin"]),
        float(diagnostics["sum_normalized_margin"]),
        -float(validation["preference_l1"]),
        -float(update),
    )


def main() -> None:
    args = parse_args()
    if args.fixed_training_profile is not None and args.algorithm != "ls":
        raise ValueError("--fixed-training-profile is only valid for the LS baseline")
    if args.phase_staggered_deadline_scale <= 0:
        raise ValueError("--phase-staggered-deadline-scale must be positive")
    if not np.isfinite(args.assignment_priority_decay) or args.assignment_priority_decay < 0:
        raise ValueError("--assignment-priority-decay must be finite and non-negative")
    sampler = preference_sampler_config(
        sampler_mode=args.preference_sampler_mode,
        calibration_priority_share=args.calibration_priority_share,
        concentration=args.preference_sampler_concentration,
        anchor_probability=args.preference_sampler_anchor_probability,
        anchor_jitter=args.preference_sampler_anchor_jitter,
    )
    dynamic_profiles = (
        calibration_priority_profiles(args.calibration_priority_share)
        if args.preference_sampler_mode == MODERATE_PREFERENCE_SAMPLER_MODE
        else None
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    protocol_path = args.protocol_config or args.scenario_config
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_version = args.protocol_version or PCRL_VERSION
    preference_target_mode = (
        args.preference_target_mode or DEFAULT_PREFERENCE_TARGET_MODE
    )
    config = replace(PaperEnvConfig(**protocol["scenario"]), seed=args.seed)
    train_scales = tuple(parse_scale(value) for value in protocol["train_scales"])
    scale_deadlines = {
        parse_scale(scale): float(deadline)
        for scale, deadline in protocol["scale_deadlines"].items()
    }
    updates = int(_protocol_value(args, protocol, "updates"))
    episodes_per_update = int(
        _protocol_value(args, protocol, "episodes_per_update")
    )
    learning_rate = float(
        _protocol_value(args, protocol, "learning_rate")
    )
    hidden_dim = int(_protocol_value(args, protocol, "hidden_dim"))
    validation_episodes = int(
        _protocol_value(args, protocol, "validation_episodes")
    )
    validation_interval = int(
        _protocol_value(args, protocol, "validation_interval")
    )
    validation_seed = int(
        _protocol_value(args, protocol, "validation_seed")
    )
    training_config = PCRLTrainingConfig(
        gamma=float(_protocol_value(args, protocol, "gamma")),
        gae_lambda=float(_protocol_value(args, protocol, "gae_lambda")),
        update_epochs=int(_protocol_value(args, protocol, "update_epochs")),
        minibatch_size=int(_protocol_value(args, protocol, "minibatch_size")),
        entropy_coefficient=float(
            _protocol_value(args, protocol, "entropy_coefficient")
        ),
        preco_lambda=args.preco_lambda,
        preference_aux_coefficient=args.preference_aux_coefficient,
        preference_group_episodes=int(
            args.preference_group_episodes
            if args.preference_group_episodes is not None
            else protocol.get("preference_group_episodes", 1)
        ),
        advantage_normalization_mode=str(
            args.advantage_normalization_mode
            or protocol.get("advantage_normalization_mode", "per_objective")
        ),
        preco_value_transform=str(
            args.preco_value_transform
            or protocol.get("preco_value_transform", "legacy_min_shift")
        ),
        seed=args.seed,
    )
    if training_config.preference_group_episodes <= 0:
        raise ValueError("preference_group_episodes must be positive")
    model = PreferenceConditionedPaperActorCritic(
        node_feature_dim=24,
        edge_feature_dim=5,
        max_uavs=config.max_uavs,
        max_tasks=config.max_tasks,
        hidden_dim=hidden_dim,
        graph_mode=args.graph_mode,
        preference_conditioning=not args.no_conditioning,
        preference_input_mode=args.preference_input_mode,
    )
    source_metadata = model.initialize_from_gppo_checkpoint(args.init_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    candidate_index: list[dict[str, object]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, ...] = (-float("inf"),)
    best_update = updates
    best_selection_diagnostics: dict[str, object] | None = None
    profiles = tuple(args.validation_profiles)
    if dynamic_profiles is not None:
        unknown_profiles = sorted(set(profiles) - set(dynamic_profiles))
        if unknown_profiles:
            raise ValueError(
                "moderate sampler validation profiles are outside its dynamic "
                f"family: {unknown_profiles!r}"
            )
    fixed_task_preferences = None
    if args.fixed_training_profile is not None:
        if dynamic_profiles is None:
            fixed_preference = task_preference_profile(
                args.fixed_training_profile
            )
        else:
            try:
                fixed_preference = dynamic_profiles[args.fixed_training_profile]
            except KeyError as error:
                raise ValueError(
                    "moderate sampler fixed LS profile must use the dynamic "
                    f"calibration family: {args.fixed_training_profile!r}"
                ) from error
        fixed_task_preferences = (fixed_preference,)
    reference_validation: dict[str, float] | None = None
    validation_tape_spec = {
        "seed": validation_seed,
        "episodes": validation_episodes,
        "scales": [f"{left}x{right}" for left, right in train_scales],
        "profiles": list(profiles),
        "task_release_modes": list(args.task_release_modes),
        "profile_paired": bool(args.profile_paired_validation),
        "capability_coverage_mode": args.capability_coverage_mode,
        "preference_target_mode": preference_target_mode,
    }
    validation_tape_sha256 = hashlib.sha256(
        json.dumps(
            validation_tape_spec, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    constrained_selection = args.checkpoint_selection_mode in {
        "gppo_efficiency_constrained_v1",
        "gppo_efficiency_relative_v2",
    }
    if constrained_selection:
        original_conditioning = model.preference_conditioning
        model.preference_conditioning = False
        try:
            reference_validation = evaluate_validation(
                model,
                config,
                scales=train_scales,
                scale_deadlines=scale_deadlines,
                profiles=profiles,
                task_release_modes=tuple(args.task_release_modes),
                phase_staggered_deadline_scale=args.phase_staggered_deadline_scale,
                assignment_priority_decay=args.assignment_priority_decay,
                preference_target_mode=preference_target_mode,
                dynamic_profiles=dynamic_profiles,
                capability_coverage_mode=args.capability_coverage_mode,
                profile_paired=args.profile_paired_validation,
                episodes=validation_episodes,
                seed=validation_seed,
            )
        finally:
            model.preference_conditioning = original_conditioning
    if args.save_validation_candidates:
        candidate_directory = args.output / "validation_candidates"
        candidate_directory.mkdir(parents=True, exist_ok=True)
        initial_state = copy.deepcopy(model.state_dict())
        if args.preference_input_mode == MASS_CONTROL_PREFERENCE_INPUT_MODE:
            initial_state["task_preference_gain"] = torch.tensor(-20.0)
        update_zero_path = candidate_directory / "update_0000.pt"
        torch.save(
            {
                "protocol_version": protocol_version,
                "protocol_variant": args.protocol_variant,
                "artifact_group": args.artifact_group,
                "update": 0,
                "model_state": initial_state,
                "validation": copy.deepcopy(reference_validation),
                "selection_diagnostics": {
                    "feasible": True,
                    "reasons": [],
                    "reference_only": True,
                },
            },
            update_zero_path,
        )
        update_zero_sha256 = hashlib.sha256(update_zero_path.read_bytes()).hexdigest()
        candidate_index.append(
            {
                "update": 0,
                "path": str(update_zero_path),
                "sha256": update_zero_sha256,
                "reference_only": True,
            }
        )
        (candidate_directory / "candidate_index.json").write_text(
            json.dumps(candidate_index, indent=2), encoding="utf-8"
        )
    initial_task_preference_gain = float(
        args.initial_task_preference_gain
        if args.initial_task_preference_gain is not None
        else protocol.get("initial_task_preference_gain", 1.0)
    )
    if not args.no_conditioning:
        if args.preference_input_mode == MASS_CONTROL_PREFERENCE_INPUT_MODE:
            if not 0.0 < initial_task_preference_gain < 1.0:
                raise ValueError(
                    "task-mass v3 initial gain is an alpha and must be in (0, 1)"
                )
            model.task_preference_gain.data.fill_(
                float(np.log(initial_task_preference_gain / (1.0 - initial_task_preference_gain)))
            )
        else:
            model.task_preference_gain.data.fill_(initial_task_preference_gain)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for update in range(updates):
        backbone_trainable = (
            not args.freeze_backbone
            and update >= args.residual_warmup_updates
        )
        model.set_backbone_trainable(backbone_trainable)
        batch = collect_pcrl_rollouts(
            model,
            config,
            episodes=episodes_per_update,
            seed_offset=update * episodes_per_update,
            sync_mode="event",
            gamma=training_config.gamma,
            gae_lambda=training_config.gae_lambda,
            scale_choices=train_scales,
            scale_deadlines=scale_deadlines,
            fixed_task_preferences=fixed_task_preferences,
            task_release_modes=tuple(args.task_release_modes),
            phase_staggered_deadline_scale=args.phase_staggered_deadline_scale,
            assignment_priority_decay=args.assignment_priority_decay,
            preference_target_mode=preference_target_mode,
            preference_sampler_mode=str(sampler["mode"]),
            calibration_priority_share=sampler["calibration_priority_share"],
            preference_sampler_concentration=float(sampler["concentration"]),
            preference_sampler_anchor_probability=float(sampler["anchor_probability"]),
            preference_sampler_anchor_jitter=float(sampler["anchor_jitter"]),
            capability_coverage_mode=args.capability_coverage_mode,
            episodes_per_preference=training_config.preference_group_episodes,
        )
        update_metrics = pcrl_ppo_update(
            model,
            optimizer,
            batch,
            training_config,
            algorithm=args.algorithm,
            update_index=update,
        )
        rows = batch.episode_metrics
        record = {
            "update": float(update + 1),
            "preference_l1": float(
                np.mean([float(row["preference_l1"]) for row in rows])
            ),
            "preference_cosine": float(
                np.mean([float(row["preference_cosine"]) for row in rows])
            ),
            "deadline_completion_rate": float(
                np.mean([float(row["deadline_completion_rate"]) for row in rows])
            ),
            "makespan": float(np.mean([float(row["makespan"]) for row in rows])),
            "communication_events": float(
                np.mean([float(row["communication_events"]) for row in rows])
            ),
            "minimum_task_coverage": float(
                np.mean([float(row["minimum_task_coverage"]) for row in rows])
            ),
            "backbone_trainable": float(backbone_trainable),
            **update_metrics,
        }
        history.append(record)
        (args.output / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if validation_episodes and (
            (update + 1) % validation_interval == 0 or update + 1 == updates
        ):
            validation = evaluate_validation(
                model,
                config,
                scales=train_scales,
                scale_deadlines=scale_deadlines,
                profiles=profiles,
                task_release_modes=tuple(args.task_release_modes),
                phase_staggered_deadline_scale=args.phase_staggered_deadline_scale,
                assignment_priority_decay=args.assignment_priority_decay,
                preference_target_mode=preference_target_mode,
                dynamic_profiles=dynamic_profiles,
                capability_coverage_mode=args.capability_coverage_mode,
                profile_paired=args.profile_paired_validation,
                episodes=validation_episodes,
                seed=validation_seed,
            )
            validation["update"] = float(update + 1)
            if constrained_selection:
                if reference_validation is None:
                    raise AssertionError("constrained selection requires GPPO reference")
                diagnostics = constrained_selection_diagnostics(
                    validation,
                    reference_validation,
                    max_deadline_completion_drop=args.max_deadline_completion_drop,
                    max_makespan_increase_ratio=args.max_makespan_increase_ratio,
                    max_coverage_loss=args.max_coverage_loss,
                    max_invalid_actions=args.max_invalid_actions,
                    max_invalid_action_increase=(
                        args.max_invalid_action_increase
                        if args.checkpoint_selection_mode
                        == "gppo_efficiency_relative_v2"
                        else None
                    ),
                )
                validation["selection_feasible"] = float(
                    bool(diagnostics["feasible"])
                )
                validation["selection_diagnostics"] = diagnostics
                score = checkpoint_selection_key(
                    validation, diagnostics, update=update + 1
                )
            else:
                diagnostics = {"feasible": True, "reasons": []}
                score = (
                    -validation["preference_l1"],
                    validation["deadline_completion_rate"],
                    -validation["makespan"],
                )
            if args.save_validation_candidates:
                candidate_directory = args.output / "validation_candidates"
                candidate_directory.mkdir(parents=True, exist_ok=True)
                candidate_path = candidate_directory / f"update_{update + 1:04d}.pt"
                torch.save(
                    {
                        "protocol_version": protocol_version,
                        "protocol_variant": args.protocol_variant,
                        "artifact_group": args.artifact_group,
                        "update": update + 1,
                        "model_state": copy.deepcopy(model.state_dict()),
                        "validation": copy.deepcopy(validation),
                        "selection_diagnostics": copy.deepcopy(diagnostics),
                        "selection_key": list(score),
                    },
                    candidate_path,
                )
                candidate_sha256 = hashlib.sha256(
                    candidate_path.read_bytes()
                ).hexdigest()
                validation["candidate_checkpoint"] = str(candidate_path)
                validation["candidate_checkpoint_sha256"] = candidate_sha256
                candidate_index.append(
                    {
                        "update": update + 1,
                        "path": str(candidate_path),
                        "sha256": candidate_sha256,
                        "selection_key": list(score),
                        "selection_feasible": bool(diagnostics["feasible"]),
                    }
                )
                (candidate_directory / "candidate_index.json").write_text(
                    json.dumps(candidate_index, indent=2), encoding="utf-8"
                )
            validation_history.append(validation)
            (args.output / "validation_history.json").write_text(
                json.dumps(validation_history, indent=2), encoding="utf-8"
            )
            if score > best_score:
                best_score = score
                best_update = update + 1
                best_state = copy.deepcopy(model.state_dict())
                best_selection_diagnostics = copy.deepcopy(diagnostics)
    if best_state is not None:
        model.load_state_dict(best_state)
    method_prefix = "pcrl" if args.algorithm == "preco" else args.algorithm
    backbone_name = "ppo_none" if args.graph_mode == "none" else f"gppo_{args.graph_mode}"
    method_id = f"{method_prefix}_{backbone_name}"
    if args.fixed_training_profile is not None:
        method_id += f"_fixed_{args.fixed_training_profile}"
    if args.no_conditioning:
        method_id += "_no_conditioning"
    checkpoint = {
        "version": PCRL_VERSION,
        "protocol_version": protocol_version,
        "artifact_group": args.artifact_group,
        "protocol_variant": args.protocol_variant,
        "protocol_config_sha256": args.protocol_config_sha256,
        "output_namespace": args.output_namespace,
        "base_protocol": args.base_protocol,
        "method_id": method_id,
        "model_state": model.state_dict(),
        "algorithm": args.algorithm,
        "preference_conditioning": not args.no_conditioning,
        "preference_input_mode": args.preference_input_mode,
        "capability_coverage_mode": args.capability_coverage_mode,
        "preference_sampler": sampler,
        "preference_sampler_profile_vectors": (
            {
                name: vector.tolist()
                for name, vector in dynamic_profiles.items()
            }
            if dynamic_profiles is not None
            else None
        ),
        "fixed_training_preference": (
            fixed_task_preferences[0].tolist()
            if fixed_task_preferences is not None
            else None
        ),
        "graph_mode": args.graph_mode,
        "sync_mode": "event",
        "scenario": protocol["version"],
        "scenario_hash": pcrl_scenario_hash(
            config, args.capability_coverage_mode
        ),
        "env_config": config.to_dict(),
        "gppo_implementation_hash": implementation_hash(),
        "pcrl_implementation_hash": pcrl_implementation_hash(),
        "source_gppo": source_metadata,
        "model_config": {
            "node_feature_dim": 24,
            "edge_feature_dim": 5,
            "max_uavs": config.max_uavs,
            "max_tasks": config.max_tasks,
            "hidden_dim": hidden_dim,
            "graph_mode": args.graph_mode,
            "preference_conditioning": not args.no_conditioning,
            "preference_input_mode": args.preference_input_mode,
        },
        "training": {
            **vars(args),
            "scenario_config": str(args.scenario_config),
            "protocol_config": (
                str(args.protocol_config) if args.protocol_config is not None else None
            ),
            "init_checkpoint": str(args.init_checkpoint),
            "output": str(args.output),
            "updates": updates,
            "episodes_per_update": episodes_per_update,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "validation_episodes": validation_episodes,
            "validation_interval": validation_interval,
            "validation_seed": validation_seed,
            "pcrl_training_config": asdict(training_config),
        },
        "train_scales": list(protocol["train_scales"]),
        "scale_deadlines": protocol["scale_deadlines"],
        "validation_profiles": list(profiles),
        "task_release_modes": list(args.task_release_modes),
        "assignment_priority_decay": args.assignment_priority_decay,
        "preference_target_mode": preference_target_mode,
        "capability_coverage_mode": args.capability_coverage_mode,
        "preference_input_mode": args.preference_input_mode,
        "checkpoint_selection": {
            "mode": args.checkpoint_selection_mode,
            "selection_passed": bool(
                best_selection_diagnostics is None
                or best_selection_diagnostics.get("feasible", False)
            ),
            "selected_update": best_update,
            "selected_diagnostics": best_selection_diagnostics,
            "reference_metrics": reference_validation,
            "reference_checkpoint_sha256": source_metadata.get(
                "checkpoint_sha256", "unverified"
            ),
            "source_gppo_checkpoint_seed": source_metadata.get(
                "training_seed", -1
            ),
            "validation_tape_spec": validation_tape_spec,
            "validation_tape_sha256": validation_tape_sha256,
            "thresholds": {
                "max_deadline_completion_drop": args.max_deadline_completion_drop,
                "max_makespan_increase_ratio": args.max_makespan_increase_ratio,
                "max_coverage_loss": args.max_coverage_loss,
                "max_invalid_actions": args.max_invalid_actions,
                "max_invalid_action_increase": args.max_invalid_action_increase,
            },
        },
        "history": history,
        "validation_history": validation_history,
        "validation_candidates": candidate_index,
        "best_update": best_update,
    }
    torch.save(checkpoint, args.output / "checkpoint.pt")
    print(
        json.dumps(
            {
                "saved": str(args.output / "checkpoint.pt"),
                "best_update": best_update,
                "source_gppo": source_metadata,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

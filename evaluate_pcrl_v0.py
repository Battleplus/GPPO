from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from uav_assignment.gppo_v2 import config_hash, implementation_hash
from uav_assignment.paper_env import PaperAlignedUAVEnv, PaperEnvConfig
from uav_assignment.paper_models import PaperHeteroActorCritic
from uav_assignment.pcrl_models import (
    LEGACY_PREFERENCE_INPUT_MODE,
    MASS_CONTROL_PREFERENCE_INPUT_MODE,
    PREFERENCE_INPUT_MODES,
    SPLIT_PREFERENCE_INPUT_MODE,
    PreferenceConditionedPaperActorCritic,
    verify_frozen_checkpoint_file,
)
from uav_assignment.pcrl_training import (
    model_conditioning_inputs,
    tensor_observation,
)
from uav_assignment.pcrl_v0 import (
    CAPABILITY_COVERAGE_MODES,
    DEFAULT_PREFERENCE_TARGET_MODE,
    LEGACY_CAPABILITY_COVERAGE_MODE,
    PREFERENCE_TARGET_MODES,
    PCRL_VERSION,
    PreferencePaperEnv,
    calibration_priority_profiles,
    map_task_preference_to_full,
    pcrl_implementation_hash,
    pcrl_scenario_hash,
    preference_alignment,
    task_preference_profile,
)


def parse_scale(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", maxsplit=1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PCRL-v0 preferences")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument(
        "--baseline", choices=("random", "greedy_preference", "greedy")
    )
    parser.add_argument("--scenario-config", type=Path, default=Path("configs/gppo_v2_hard.json"))
    parser.add_argument("--protocol-version")
    parser.add_argument("--protocol-variant", default="")
    parser.add_argument("--artifact-group")
    parser.add_argument("--preference-profile-family")
    parser.add_argument("--evaluation-suite")
    parser.add_argument("--protocol-config-sha256")
    parser.add_argument("--protocol-config", type=Path)
    parser.add_argument("--output-namespace")
    parser.add_argument("--base-protocol")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=(
            "balanced",
            "search",
            "reconnaissance",
            "strike",
            "recovery",
            "search_strike_interp",
            "recon_recovery_interp",
            "smooth_interp",
        ),
    )
    parser.add_argument(
        "--calibration-priority-share",
        type=float,
        help=(
            "opt-in dynamic eight-profile calibration family; overrides "
            "--profiles without changing historical named profiles"
        ),
    )
    parser.add_argument("--scales", nargs="+", default=("2x12", "3x16", "3x20", "4x24"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-event-log", action="store_true")
    parser.add_argument("--switch-from")
    parser.add_argument("--switch-to")
    parser.add_argument("--switch-after-fraction", type=float, default=0.5)
    parser.add_argument("--switch-decision", type=int)
    parser.add_argument("--task-preference-gain-override", type=float)
    parser.add_argument("--preference-input-mode", choices=PREFERENCE_INPUT_MODES)
    parser.add_argument(
        "--capability-coverage-mode", choices=CAPABILITY_COVERAGE_MODES
    )
    parser.add_argument("--allow-failed-selection-diagnostic", action="store_true")
    parser.add_argument(
        "--task-release-mode",
        choices=("chain", "phase_staggered"),
        default="chain",
    )
    parser.add_argument("--deadline-scale", type=float, default=1.0)
    parser.add_argument("--assignment-priority-decay", type=float, default=1.0)
    parser.add_argument(
        "--preference-target-mode",
        choices=PREFERENCE_TARGET_MODES,
        default=None,
    )
    return parser.parse_args()


def choose_preference_greedy(
    observation: dict[str, np.ndarray],
    env: PreferencePaperEnv,
    preference: np.ndarray,
    rng: np.random.Generator,
    *,
    random_mode: bool = False,
) -> int:
    valid = np.flatnonzero(observation["action_mask"][:-1])
    if valid.size == 0:
        return env.noop_action
    if random_mode:
        return int(rng.choice(valid))
    best_action = int(valid[0])
    best_key = (-float("inf"), float("inf"), float("inf"))
    for action in valid:
        uav_index, task_index = divmod(int(action), env.config.max_tasks)
        task = env.belief_tasks[task_index]
        edge = observation["edge_features"][uav_index, env.config.max_uavs + task_index]
        duration = float(
            np.arctanh(np.clip(edge[0], -0.999, 0.999)) * 2.0
            + np.arctanh(np.clip(edge[1], -0.999, 0.999)) * 2.0
        )
        score = float(preference[task.task_type]) * float(task.priority) / max(
            duration, 1e-3
        )
        key = (score, -duration, -float(task_index))
        if key > best_key:
            best_key = key
            best_action = int(action)
    return best_action


def load_model(
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    version = str(payload.get("version", ""))
    model_config = payload["model_config"]
    if version == PCRL_VERSION:
        model = PreferenceConditionedPaperActorCritic(**model_config)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        return model, payload, "pcrl"
    if version == "paper-aligned-gppo-v2":
        model = PaperHeteroActorCritic(**model_config)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        if str(payload.get("implementation_hash")) != implementation_hash():
            raise ValueError("GPPO checkpoint implementation hash does not match current frozen files")
        return model, payload, "gppo"
    raise ValueError(f"unsupported checkpoint version: {version!r}")


def _metrics_with_profile(env: PreferencePaperEnv, profile: str) -> dict[str, Any]:
    metrics = env.metrics()
    metrics["preference_profile"] = profile
    return metrics


def main() -> None:
    args = parse_args()
    identity_protocol: dict[str, Any] = {}
    identity_protocol_sha256 = ""
    if args.protocol_config is not None:
        identity_protocol = json.loads(
            args.protocol_config.read_text(encoding="utf-8")
        )
        identity_protocol_sha256 = hashlib.sha256(
            args.protocol_config.read_bytes()
        ).hexdigest()
        if (
            args.protocol_config_sha256
            and args.protocol_config_sha256 != identity_protocol_sha256
        ):
            raise ValueError("protocol_config_sha256 does not match protocol file")
    scenario_payload = json.loads(args.scenario_config.read_text(encoding="utf-8"))
    scenario_config_sha256 = hashlib.sha256(
        args.scenario_config.read_bytes()
    ).hexdigest()
    protocol_config = PaperEnvConfig(**scenario_payload["scenario"])
    scale_deadlines = {
        parse_scale(scale): float(deadline)
        for scale, deadline in scenario_payload["scale_deadlines"].items()
    }
    model: torch.nn.Module | None = None
    payload: dict[str, Any] | None = None
    model_kind = "baseline"
    if args.checkpoint is not None:
        model, payload, model_kind = load_model(args.checkpoint)
        if args.task_preference_gain_override is not None:
            if model_kind != "pcrl":
                raise ValueError("task preference gain override requires a PCRL checkpoint")
            override = float(args.task_preference_gain_override)
            if model.preference_input_mode == MASS_CONTROL_PREFERENCE_INPUT_MODE:  # type: ignore[union-attr]
                if not 0.0 < override < 1.0:
                    raise ValueError("task-mass v3 override is an alpha in (0, 1)")
                override = float(np.log(override / (1.0 - override)))
            model.task_preference_gain.data.fill_(override)  # type: ignore[union-attr]
        base_config = PaperEnvConfig(**payload["env_config"])
        scenario_version = str(payload.get("scenario", scenario_payload["version"]))
    else:
        base_config = protocol_config
        scenario_version = str(scenario_payload["version"])
    protocol_version = str(
        args.protocol_version
        or identity_protocol.get("version")
        or (payload or {}).get("protocol_version")
        or PCRL_VERSION
    )
    artifact_group = str(
        args.artifact_group
        or (payload or {}).get("artifact_group")
        or (payload or {}).get("training", {}).get("artifact_group")
        or ""
    )
    preference_profile_family = str(
        args.preference_profile_family
        or (
            identity_protocol.get("preference_profile_family", {}).get("id")
            if isinstance(identity_protocol.get("preference_profile_family"), dict)
            else identity_protocol.get("preference_profile_family")
        )
        or "historical_named_profiles"
    )
    evaluation_suite = str(args.evaluation_suite or args.task_release_mode)
    protocol_config_sha256 = str(
        identity_protocol_sha256
        or args.protocol_config_sha256
        or (payload or {}).get("protocol_config_sha256")
        or ""
    )
    output_namespace = str(
        args.output_namespace
        or identity_protocol.get("output_namespace")
        or (payload or {}).get("output_namespace")
        or ""
    )
    base_protocol = str(
        args.base_protocol
        or identity_protocol.get("base_protocol")
        or (payload or {}).get("base_protocol")
        or ""
    )
    if identity_protocol:
        if args.protocol_version:
            configured_version = str(identity_protocol.get("version"))
            if not (
                protocol_version == configured_version
                or (
                    args.protocol_variant
                    and protocol_version.startswith(f"{configured_version}-cal-")
                )
            ):
                raise ValueError("protocol_version does not match protocol config")
        if (
            args.output_namespace
            and args.output_namespace
            != str(identity_protocol.get("output_namespace", ""))
        ):
            raise ValueError("output_namespace does not match protocol config")
        if (
            args.base_protocol
            and args.base_protocol
            != str(identity_protocol.get("base_protocol", ""))
        ):
            raise ValueError("base_protocol does not match protocol config")
        family_payload = identity_protocol.get("preference_profile_family")
        expected_family = (
            str(family_payload.get("id", ""))
            if isinstance(family_payload, dict)
            else str(family_payload or "")
        )
        if (
            args.preference_profile_family
            and expected_family
            and args.preference_profile_family != expected_family
        ):
            raise ValueError(
                "preference_profile_family does not match protocol config"
            )
        groups = identity_protocol.get("artifact_groups")
        if isinstance(groups, dict) and artifact_group not in groups:
            raise ValueError("artifact_group is not registered by protocol config")
        expected_frozen_hash = str(
            identity_protocol.get("frozen_gppo_implementation_hash", "")
        )
        if expected_frozen_hash and expected_frozen_hash != implementation_hash():
            raise ValueError("frozen GPPO implementation hash mismatch")
    if model_kind == "pcrl" and payload is not None:
        checkpoint_protocol = str(payload.get("protocol_version", ""))
        if args.protocol_version and checkpoint_protocol != args.protocol_version:
            raise ValueError(
                "PCRL checkpoint protocol_version does not match evaluation CLI"
            )
        checkpoint_group = str(
            payload.get("artifact_group")
            or payload.get("training", {}).get("artifact_group")
            or ""
        )
        if args.artifact_group and checkpoint_group != args.artifact_group:
            raise ValueError(
                "PCRL checkpoint artifact_group does not match evaluation CLI"
            )
        checkpoint_protocol_sha = str(
            payload.get("protocol_config_sha256") or ""
        )
        if (
            args.protocol_config_sha256
            and checkpoint_protocol_sha
            and checkpoint_protocol_sha != args.protocol_config_sha256
        ):
            raise ValueError(
                "PCRL checkpoint protocol_config_sha256 does not match evaluation CLI"
            )
    capability_coverage_mode = str(
        args.capability_coverage_mode
        or (payload or {}).get("capability_coverage_mode")
        or (
            identity_protocol.get("controllability", {}).get(
                "capability_coverage_mode"
            )
            if isinstance(identity_protocol.get("controllability"), dict)
            else None
        )
        or LEGACY_CAPABILITY_COVERAGE_MODE
    )
    if capability_coverage_mode not in CAPABILITY_COVERAGE_MODES:
        raise ValueError("invalid capability_coverage_mode")
    preference_input_mode = str(
        args.preference_input_mode
        or (payload or {}).get("preference_input_mode")
        or (payload or {}).get("model_config", {}).get(
            "preference_input_mode"
        )
        or identity_protocol.get("preference_input_mode")
        or LEGACY_PREFERENCE_INPUT_MODE
    )
    if preference_input_mode not in PREFERENCE_INPUT_MODES:
        raise ValueError("invalid preference_input_mode")
    if model_kind == "pcrl" and payload is not None:
        checkpoint_mode = str(
            payload.get("model_config", {}).get(
                "preference_input_mode", LEGACY_PREFERENCE_INPUT_MODE
            )
        )
        if checkpoint_mode != preference_input_mode:
            raise ValueError(
                "PCRL checkpoint preference_input_mode does not match evaluation"
            )
        if protocol_version in {"pcrl-v0-hard-5", "pcrl-v0-hard-6"} and not bool(
            payload.get("checkpoint_selection", {}).get(
                "selection_passed", False
            )
        ):
            if not (
                args.allow_failed_selection_diagnostic
                and artifact_group.startswith("calibration")
            ):
                raise ValueError(
                    "evaluation rejects a checkpoint that failed constrained selection"
                )
    if protocol_version in {"pcrl-v0-hard-4", "pcrl-v0-hard-5", "pcrl-v0-hard-6"}:
        missing_identity = [
            name
            for name, value in (
                ("artifact_group", artifact_group),
                ("preference_profile_family", preference_profile_family),
                ("evaluation_suite", evaluation_suite),
                ("protocol_config_sha256", protocol_config_sha256),
                ("output_namespace", output_namespace),
                ("base_protocol", base_protocol),
            )
            if not value
        ]
        if missing_identity:
            raise ValueError(
                "hard4 evaluation requires identity fields: "
                f"{missing_identity!r}"
            )
        if model_kind == "gppo" and args.checkpoint is not None:
            verify_frozen_checkpoint_file(args.checkpoint)
    if protocol_version == "pcrl-v0-hard-5":
        if capability_coverage_mode == LEGACY_CAPABILITY_COVERAGE_MODE:
            raise ValueError("hard5 requires balanced capability coverage")
        if (
            model_kind == "pcrl"
            and preference_input_mode == LEGACY_PREFERENCE_INPUT_MODE
        ):
            raise ValueError(
                "hard5 PCRL requires split task/objective preference inputs"
            )
    if protocol_version == "pcrl-v0-hard-6":
        if capability_coverage_mode == LEGACY_CAPABILITY_COVERAGE_MODE:
            raise ValueError("hard6 requires balanced capability coverage")
        if (
            model_kind == "pcrl"
            and preference_input_mode != MASS_CONTROL_PREFERENCE_INPUT_MODE
        ):
            raise ValueError("hard6 PCRL requires explicit task-mass control")
    preference_target_mode = (
        args.preference_target_mode
        or (payload or {}).get("preference_target_mode")
        or (payload or {}).get("training", {}).get("preference_target_mode")
        or DEFAULT_PREFERENCE_TARGET_MODE
    )
    if preference_target_mode not in PREFERENCE_TARGET_MODES:
        raise ValueError(
            "checkpoint/CLI preference target mode is unsupported: "
            f"{preference_target_mode!r}"
        )
    method_id = (
        str(payload.get("method_id", "gppo_event"))
        if payload is not None
        else {
            "random": "random_event",
            "greedy": "greedy_event",
            "greedy_preference": "greedy_preference",
        }[str(args.baseline)]
    )
    preference_conditioning = (
        bool(payload.get("preference_conditioning", True))
        if model_kind == "pcrl" and payload is not None
        else False
    )
    if model_kind == "pcrl" and payload is not None:
        model_conditioning = payload.get("model_config", {}).get(
            "preference_conditioning"
        )
        if (
            model_conditioning is not None
            and bool(model_conditioning) is not preference_conditioning
        ):
            raise ValueError(
                "PCRL checkpoint preference_conditioning metadata disagrees"
            )
        suffix_conditioning = not method_id.endswith("_no_conditioning")
        if suffix_conditioning is not preference_conditioning:
            raise ValueError(
                "PCRL method_id suffix disagrees with preference_conditioning"
            )

    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    switch_requested = bool(args.switch_from or args.switch_to)
    if switch_requested and not (args.switch_from and args.switch_to):
        raise ValueError("--switch-from and --switch-to must be supplied together")
    if switch_requested and not 0.0 < args.switch_after_fraction < 1.0:
        raise ValueError("--switch-after-fraction must lie in (0, 1)")
    if args.deadline_scale <= 0:
        raise ValueError("--deadline-scale must be positive")
    if not np.isfinite(args.assignment_priority_decay) or args.assignment_priority_decay < 0:
        raise ValueError("--assignment-priority-decay must be finite and non-negative")

    if args.calibration_priority_share is not None and switch_requested:
        raise ValueError(
            "--calibration-priority-share cannot be combined with runtime "
            "preference switching"
        )
    if args.calibration_priority_share is not None:
        profile_vectors = calibration_priority_profiles(
            args.calibration_priority_share
        )
    else:
        profile_names = (
            (args.switch_from,) if switch_requested else tuple(args.profiles)
        )
        profile_vectors = {
            profile_name: task_preference_profile(profile_name)
            for profile_name in profile_names
        }
    for scale_text in args.scales:
        active_uavs, initial_tasks = parse_scale(scale_text)
        config = replace(
            base_config,
            active_uavs=active_uavs,
            initial_tasks=initial_tasks,
            mission_deadline=scale_deadlines.get(
                (active_uavs, initial_tasks), base_config.mission_deadline
            )
            * args.deadline_scale,
        )
        for profile_name, initial_task_preference in profile_vectors.items():
            for episode in range(args.episodes):
                episode_seed = args.eval_seed + episode
                env = PreferencePaperEnv(
                    config,
                    preference=initial_task_preference,
                    task_release_mode=args.task_release_mode,
                    assignment_priority_decay=args.assignment_priority_decay,
                    preference_target_mode=preference_target_mode,
                    capability_coverage_mode=capability_coverage_mode,
                )
                observation = env.reset(seed=episode_seed)
                rng = np.random.default_rng(episode_seed + 1_000_000)
                done = False
                switch_applied = False
                switch_snapshot: dict[str, Any] | None = None
                switch_target = (
                    task_preference_profile(args.switch_to)
                    if switch_requested
                    else None
                )
                switch_allocation_target: np.ndarray | None = None
                while not done:
                    if switch_requested and not switch_applied:
                        threshold = max(
                            1,
                            args.switch_decision
                            if args.switch_decision is not None
                            else int(
                                args.switch_after_fraction
                                * max(1, config.initial_tasks)
                            ),
                        )
                        if env.decision_count >= threshold:
                            env.set_preference(switch_target)  # type: ignore[arg-type]
                            switch_allocation_target = (
                                env.task_allocation_target.copy()
                            )
                            switch_snapshot = env.preference_metrics()
                            switch_applied = True
                            observation = env.observe()
                    if args.baseline is not None:
                        baseline_preference = (
                            np.full(4, 0.25, dtype=np.float32)
                            if args.baseline == "greedy"
                            else env.user_task_preference
                        )
                        action = choose_preference_greedy(
                            observation,
                            env,
                            baseline_preference,
                            rng,
                            random_mode=args.baseline == "random",
                        )
                    elif model_kind == "pcrl":
                        (
                            policy_observation,
                            policy_task_preference,
                            policy_objective_preference,
                        ) = model_conditioning_inputs(
                            model,  # type: ignore[arg-type]
                            tensor_observation(observation),
                            torch.as_tensor(
                                env.user_task_preference, dtype=torch.float32
                            ),
                            torch.as_tensor(env.preference, dtype=torch.float32),
                        )
                        if preference_input_mode != LEGACY_PREFERENCE_INPUT_MODE:
                            selected, _, _ = model.act(  # type: ignore[union-attr]
                                policy_observation,
                                policy_objective_preference,
                                task_preference=policy_task_preference,
                                deterministic=True,
                            )
                        else:
                            selected, _, _ = model.act(  # type: ignore[union-attr]
                                policy_observation,
                                policy_objective_preference,
                                deterministic=True,
                            )
                        action = int(selected.item())
                    else:
                        selected, _, _ = model.act(  # type: ignore[union-attr]
                            tensor_observation(observation), deterministic=True
                        )
                        action = int(selected.item())
                    observation, _, done, _ = env.step(action, sync_mode="event")
                metrics = _metrics_with_profile(env, profile_name)
                metrics.update(
                    {
                        "method_id": method_id,
                        "model_kind": model_kind,
                        "algorithm": (
                            str(payload.get("algorithm", ""))
                            if payload is not None
                            else str(args.baseline)
                        ),
                        "graph_mode": (
                            str(payload.get("graph_mode", ""))
                            if payload is not None
                            else "none"
                        ),
                        "training_seed": (
                            int(payload.get("training", {}).get("seed", -1))
                            if payload is not None
                            else -1
                        ),
                        "scale": scale_text,
                        "active_uavs": active_uavs,
                        "initial_tasks": initial_tasks,
                        "max_decisions": config.max_decisions,
                        "eval_seed": episode_seed,
                        "evaluation_seed": episode_seed,
                        "scenario_version": scenario_version,
                        "protocol_version": protocol_version,
                        "protocol_variant": args.protocol_variant,
                        "protocol_config_sha256": protocol_config_sha256,
                        "artifact_group": artifact_group,
                        "output_namespace": output_namespace,
                        "base_protocol": base_protocol,
                        "scenario_config_sha256": scenario_config_sha256,
                        "frozen_gppo_implementation_hash": implementation_hash(),
                        "pcrl_implementation_hash": pcrl_implementation_hash(),
                        "source_gppo_checkpoint_sha256": (
                            str(
                                (payload.get("source_gppo") or {}).get(
                                    "checkpoint_sha256", ""
                                )
                            )
                            if model_kind == "pcrl" and payload is not None
                            else hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                            if args.checkpoint is not None
                            else ""
                        ),
                        "preference_profile_family": preference_profile_family,
                        "evaluation_suite": evaluation_suite,
                        "preference_conditioning": preference_conditioning,
                        "preference_input_mode": preference_input_mode,
                        "capability_coverage_mode": capability_coverage_mode,
                        "checkpoint_selection_passed": (
                            bool(
                                payload.get("checkpoint_selection", {}).get(
                                    "selection_passed", False
                                )
                            )
                            if model_kind == "pcrl" and payload is not None
                            else None
                        ),
                        "updates": (
                            int(payload.get("training", {}).get("updates", -1))
                            if payload is not None
                            else -1
                        ),
                        "episodes_per_update": (
                            int(
                                payload.get("training", {}).get(
                                    "episodes_per_update", -1
                                )
                            )
                            if payload is not None
                            else -1
                        ),
                        "evaluation_episodes": int(args.episodes),
                        "scenario_hash": pcrl_scenario_hash(
                            config, capability_coverage_mode
                        ),
                        "checkpoint": str(args.checkpoint.resolve())
                        if args.checkpoint is not None
                        else None,
                        "switch_requested": switch_requested,
                        "switch_applied": switch_applied,
                        "switch_from": args.switch_from,
                        "switch_to": args.switch_to,
                        "task_preference_gain_override": args.task_preference_gain_override,
                        "checkpoint_task_preference_gain_raw": (
                            float(
                                getattr(model, "task_preference_gain")
                                .detach()
                                .cpu()
                            )
                            if model_kind == "pcrl"
                            and hasattr(model, "task_preference_gain")
                            else None
                        ),
                        "checkpoint_task_mass_base_alpha": (
                            float(
                                torch.sigmoid(
                                    model.task_preference_gain.detach().cpu()
                                )
                            )
                            if model_kind == "pcrl"
                            and hasattr(model, "task_preference_gain")
                            and preference_input_mode
                            == MASS_CONTROL_PREFERENCE_INPUT_MODE
                            else None
                        ),
                        "deadline_scale": args.deadline_scale,
                        "assignment_priority_decay": args.assignment_priority_decay,
                        "preference_target_mode": preference_target_mode,
                        "task_release_mode": args.task_release_mode,
                    }
                )
                if args.calibration_priority_share is not None:
                    metrics.update(
                        {
                            "calibration_priority_share": float(
                                args.calibration_priority_share
                            ),
                            "calibration_profile_vector": (
                                initial_task_preference.tolist()
                            ),
                            "priority_share": float(
                                args.calibration_priority_share
                            ),
                            "background_share": float(
                                (1.0 - args.calibration_priority_share) / 3.0
                            ),
                            "preference_profile_vector": (
                                initial_task_preference.tolist()
                            ),
                        }
                    )
                if (
                    switch_snapshot is not None
                    and switch_allocation_target is not None
                ):
                    before = np.asarray(
                        switch_snapshot["deadline_completed_by_type"], dtype=np.float32
                    )
                    after = np.asarray(
                        metrics["deadline_completed_by_type"], dtype=np.float32
                    )
                    post_switch = np.maximum(after - before, 0.0)
                    post_mix = post_switch / max(float(post_switch.sum()), 1e-8)
                    metrics["post_switch_mix"] = post_mix.tolist()
                    metrics["post_switch_preference_l1"] = preference_alignment(
                        post_mix, switch_allocation_target
                    )["l1"]
                    metrics["post_switch_preference_cosine"] = preference_alignment(
                        post_mix, switch_allocation_target
                    )["cosine"]
                rows.append(metrics)
                if args.save_event_log:
                    event_rows.extend(
                        {
                            "method_id": metrics["method_id"],
                            "scale": scale_text,
                            "preference_profile": profile_name,
                            "calibration_priority_share": (
                                float(args.calibration_priority_share)
                                if args.calibration_priority_share is not None
                                else None
                            ),
                            "calibration_profile_vector": (
                                initial_task_preference.tolist()
                                if args.calibration_priority_share is not None
                                else None
                            ),
                        "eval_seed": episode_seed,
                        "evaluation_seed": episode_seed,
                            **event,
                        }
                        for event in env.event_log
                    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if rows:
        with (args.output / "evaluation.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.save_event_log:
        with (args.output / "event_log.jsonl").open("w", encoding="utf-8") as stream:
            for event in event_rows:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps({"episodes": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

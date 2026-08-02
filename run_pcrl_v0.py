from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from uav_assignment.gppo_v2 import implementation_hash
from uav_assignment.pcrl_v0 import (
    BALANCED_CAPABILITY_COVERAGE_MODE,
    CALIBRATION_PRIORITY_PROFILE_NAMES,
    DEFAULT_PREFERENCE_TARGET_MODE,
    LEGACY_PREFERENCE_SAMPLER_MODE,
    MODERATE_PREFERENCE_SAMPLER_MODE,
    PREFERENCE_TARGET_MODES,
    PCRL_VERSION,
    calibration_priority_profiles,
    pcrl_implementation_hash,
    preference_sampler_config,
    task_preference_profile,
)
from uav_assignment.pcrl_models import (
    LEGACY_PREFERENCE_INPUT_MODE,
    PREFERENCE_INPUT_MODES,
)


LEARNED_METHODS = {
    "pcrl_gppo_adaptive": {
        "source": "gppo_event",
        "graph_mode": "adaptive",
        "algorithm": "preco",
    },
    "pcrl_gppo_single_head": {
        "source": "gppo_event_single_head",
        "graph_mode": "single_head",
        "algorithm": "preco",
    },
    "pcrl_gppo_adaptive_no_conditioning": {
        "source": "gppo_event",
        "graph_mode": "adaptive",
        "algorithm": "preco",
        "no_conditioning": True,
    },
    "ls_ppo_none_fixed_balanced": {
        "source": "ppo_event",
        "graph_mode": "none",
        "algorithm": "ls",
        "fixed_profile": "balanced",
    },
    "ls_gppo_adaptive_fixed_balanced": {
        "source": "gppo_event",
        "graph_mode": "adaptive",
        "algorithm": "ls",
        "fixed_profile": "balanced",
    },
    "sdmgrad_gppo_adaptive": {
        "source": "gppo_event",
        "graph_mode": "adaptive",
        "algorithm": "sdmgrad",
    },
}

FROZEN_METHODS = {
    "gppo_event": "gppo_event",
    "gppo_event_single_head": "gppo_event_single_head",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCRL-v0 train/evaluation protocol")
    parser.add_argument("--protocol", type=Path, default=Path("configs/pcrl_v0.json"))
    parser.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--artifact-group",
        help="named protocol artifact group, e.g. pilot20 or formal100",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-sdmgrad", action="store_true")
    parser.add_argument("--calibration-variant")
    parser.add_argument(
        "--calibration-profiles",
        action="store_true",
        help=(
            "evaluate the protocol's independent calibration_profiles instead "
            "of its frozen training/held-out evaluation profiles"
        ),
    )
    return parser.parse_args()


def run(command: list[str], *, environment: dict[str, str]) -> None:
    print(json.dumps({"command": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, check=True, env=environment)


def checkpoint_path(source: str, seed: int) -> Path:
    return Path(
        f"outputs/paper_aligned/gppo_v2_hard/formal/train/{source}/seed_{seed}/checkpoint.pt"
    )


def protocol_output_root(
    protocol: dict[str, object],
    explicit_output_root: Path | None,
    *,
    artifact_group: str = "formal",
) -> Path:
    """Resolve the artifact root without colliding across protocol versions."""

    if explicit_output_root is not None:
        return explicit_output_root
    namespace = protocol.get("output_namespace", "outputs/pcrl_v0")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("output_namespace must be a non-empty path string")
    namespace_path = Path(namespace)
    if namespace_path.name in {".", ".."}:
        raise ValueError("output_namespace must identify a concrete directory")
    if not artifact_group or Path(artifact_group).name in {".", ".."}:
        raise ValueError("artifact_group must identify a concrete directory")
    return namespace_path / artifact_group


def protocol_preference_sampler(
    protocol: dict[str, object],
) -> dict[str, float | str | None]:
    """Normalize sampler configuration while preserving old protocols."""

    raw = protocol.get("training_preference_sampler")
    if raw is None:
        return preference_sampler_config()
    if not isinstance(raw, dict):
        raise ValueError("training_preference_sampler must be an object")
    unknown = sorted(
        set(raw)
        - {
            "mode",
            "calibration_priority_share",
            "priority_share",
            "concentration",
            "anchor_probability",
            "anchor_jitter",
        }
    )
    if unknown:
        raise ValueError(
            f"unknown training_preference_sampler keys: {unknown!r}"
        )
    if "priority_share" in raw and "calibration_priority_share" in raw:
        raise ValueError(
            "training_preference_sampler cannot define both priority_share "
            "and calibration_priority_share"
        )
    share = raw.get(
        "calibration_priority_share", raw.get("priority_share")
    )
    return preference_sampler_config(
        sampler_mode=str(raw.get("mode", LEGACY_PREFERENCE_SAMPLER_MODE)),
        calibration_priority_share=(
            float(share) if share is not None else None
        ),
        concentration=float(raw.get("concentration", 0.7)),
        anchor_probability=float(raw.get("anchor_probability", 0.30)),
        anchor_jitter=float(raw.get("anchor_jitter", 0.10)),
    )


def protocol_artifact_group(
    protocol: dict[str, object], requested: str | None
) -> tuple[str, dict[str, object] | None]:
    """Resolve an isolated budget/output group without silent hard4 formal use."""

    raw_groups = protocol.get("artifact_groups")
    if raw_groups is None:
        return (requested or "formal"), None
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError("artifact_groups must be a non-empty object")
    if requested is None:
        raise ValueError(
            "this protocol requires explicit --artifact-group selection; "
            f"choose one of {sorted(raw_groups)!r}"
        )
    try:
        raw_group = raw_groups[requested]
    except KeyError as error:
        raise ValueError(
            f"unknown artifact group {requested!r}; choose one of "
            f"{sorted(raw_groups)!r}"
        ) from error
    if not isinstance(raw_group, dict):
        raise ValueError(f"artifact group {requested!r} must be an object")
    group = dict(raw_group)
    output_subdirectory = str(group.get("output_subdirectory", "")).strip()
    if not output_subdirectory or Path(output_subdirectory).name in {".", ".."}:
        raise ValueError(
            f"artifact group {requested!r} has invalid output_subdirectory"
        )
    for key in (
        "updates",
        "episodes_per_update",
        "evaluation_episodes",
        "evaluation_seed",
    ):
        if int(group.get(key, 0)) <= 0:
            raise ValueError(
                f"artifact group {requested!r} requires positive {key}"
            )
    seeds = group.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or any(int(seed) <= 0 for seed in seeds):
        raise ValueError(
            f"artifact group {requested!r} requires positive training_seeds"
        )
    return requested, group


def protocol_evaluation_profiles(
    protocol: dict[str, object], *, use_calibration: bool = False
) -> list[str]:
    """Resolve and validate a protocol's formal or calibration profile set.

    Calibration profiles are deliberately independent from the frozen hard-2
    training anchors and held-out profiles.  The caller must opt into them.
    """

    if use_calibration:
        raw_profiles = protocol.get("calibration_profiles")
        source = "calibration_profiles"
        if not isinstance(raw_profiles, list) or not raw_profiles:
            raise ValueError(
                "--calibration-profiles requires a non-empty "
                "protocol calibration_profiles list"
            )
    else:
        anchors = protocol.get("training_anchor_profiles")
        held_out = protocol.get("held_out_profiles")
        if not isinstance(anchors, list) or not isinstance(held_out, list):
            raise ValueError(
                "protocol training_anchor_profiles and held_out_profiles "
                "must be lists"
            )
        raw_profiles = anchors + held_out
        source = "training_anchor_profiles + held_out_profiles"
    profiles = [str(profile).strip() for profile in raw_profiles]
    if any(not profile for profile in profiles):
        raise ValueError(f"{source} contains an empty profile name")
    if len(set(profiles)) != len(profiles):
        raise ValueError(f"{source} contains duplicate profile names")
    sampler = protocol_preference_sampler(protocol)
    if sampler["mode"] == MODERATE_PREFERENCE_SAMPLER_MODE:
        dynamic = calibration_priority_profiles(
            float(sampler["calibration_priority_share"])
        )
        unknown = sorted(set(profiles) - set(dynamic))
        if unknown:
            raise ValueError(
                f"{source} contains profiles outside the dynamic sampler "
                f"family: {unknown!r}"
            )
    else:
        for profile in profiles:
            task_preference_profile(profile)
    return profiles


def evaluation_suites(protocol: dict[str, object]) -> list[dict[str, object]]:
    """Validate and normalize the protocol's independent evaluation suites."""
    raw_suites = protocol.get("evaluation_suites")
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ValueError("PCRL protocol must define a non-empty evaluation_suites list")
    suites: list[dict[str, object]] = []
    names: set[str] = set()
    for raw in raw_suites:
        if not isinstance(raw, dict):
            raise ValueError("each evaluation suite must be an object")
        name = str(raw.get("name", "")).strip()
        task_release_mode = str(raw.get("task_release_mode", "")).strip()
        try:
            deadline_scale = float(raw["deadline_scale"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid deadline_scale for evaluation suite {name!r}") from error
        if not name or name in names:
            raise ValueError(
                f"evaluation suite names must be non-empty and unique: {name!r}"
            )
        if task_release_mode not in {"chain", "phase_staggered"}:
            raise ValueError(
                f"unsupported task release mode for suite {name!r}: "
                f"{task_release_mode!r}"
            )
        if deadline_scale <= 0:
            raise ValueError(f"deadline_scale must be positive for suite {name!r}")
        preference_target_mode = str(
            raw.get("preference_target_mode", DEFAULT_PREFERENCE_TARGET_MODE)
        )
        if preference_target_mode not in PREFERENCE_TARGET_MODES:
            raise ValueError(
                f"unsupported preference target mode for suite {name!r}: "
                f"{preference_target_mode!r}"
            )
        names.add(name)
        suites.append(
            {
                "name": name,
                "task_release_mode": task_release_mode,
                "deadline_scale": deadline_scale,
                "preference_target_mode": preference_target_mode,
                "primary": bool(raw.get("primary", False)),
            }
        )
    acceptance = protocol.get("acceptance")
    primary_suite = (
        acceptance.get("primary_suite") if isinstance(acceptance, dict) else None
    )
    if primary_suite is not None and str(primary_suite) not in names:
        raise ValueError(
            "acceptance.primary_suite is not present in evaluation_suites: "
            f"{primary_suite!r}"
        )
    marked_primary = [str(suite["name"]) for suite in suites if suite["primary"]]
    if len(marked_primary) != 1:
        raise ValueError(
            "evaluation_suites must mark exactly one suite as primary; "
            f"found {marked_primary!r}"
        )
    if primary_suite is not None and marked_primary[0] != str(primary_suite):
        raise ValueError(
            "acceptance.primary_suite does not match the suite marked primary: "
            f"{primary_suite!r} != {marked_primary[0]!r}"
        )
    return suites


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_config_sha256 = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    calibration_variant: dict[str, object] | None = None
    if args.calibration_variant is not None:
        variants = protocol.get("calibration_variants")
        if not isinstance(variants, dict) or args.calibration_variant not in variants:
            raise ValueError("unknown calibration variant")
        calibration_variant = dict(variants[args.calibration_variant])
    if protocol.get("artifact_groups") is not None:
        artifact_group, artifact_group_config = protocol_artifact_group(
            protocol, args.artifact_group
        )
        output_subdirectory = str(
            artifact_group_config["output_subdirectory"]
        )
    else:
        artifact_group, artifact_group_config = protocol_artifact_group(
            protocol,
            args.artifact_group
            or ("calibration" if args.calibration_profiles else "formal"),
        )
        output_subdirectory = artifact_group
    if args.calibration_variant is not None:
        safe_variant = "".join(
            character.lower() if character.isalnum() else "_"
            for character in args.calibration_variant
        ).strip("_")
        output_subdirectory = str(Path(output_subdirectory) / f"variant_{safe_variant}")
    args.output_root = protocol_output_root(
        protocol, args.output_root, artifact_group=output_subdirectory
    )
    protocol_version = str(protocol["version"])
    effective_protocol_version = protocol_version
    if (
        args.calibration_variant is not None
        and args.calibration_variant != "D_full_hard5"
    ):
        effective_protocol_version = (
            f"{protocol_version}-cal-{args.calibration_variant[0].lower()}"
        )
    group_seeds = (
        artifact_group_config["training_seeds"]
        if artifact_group_config is not None
        else protocol["training_seeds"]
    )
    seeds = tuple(args.seeds or group_seeds)
    group_updates = (
        int(artifact_group_config["updates"])
        if artifact_group_config is not None
        else None
    )
    group_episodes_per_update = (
        int(artifact_group_config["episodes_per_update"])
        if artifact_group_config is not None
        else None
    )
    evaluation_episodes = (
        int(artifact_group_config["evaluation_episodes"])
        if artifact_group_config is not None
        else int(protocol["evaluation_episodes"])
    )
    evaluation_seed = (
        int(artifact_group_config["evaluation_seed"])
        if artifact_group_config is not None
        else int(protocol["evaluation_seed"])
    )
    protocol_updates = (
        group_updates
        if group_updates is not None
        else int(protocol["updates"])
        if "updates" in protocol
        else None
    )
    protocol_episodes_per_update = (
        group_episodes_per_update
        if group_episodes_per_update is not None
        else int(protocol["episodes_per_update"])
        if "episodes_per_update" in protocol
        else None
    )
    effective_updates = 2 if args.smoke else protocol_updates
    effective_episodes_per_update = (
        2 if args.smoke else protocol_episodes_per_update
    )
    effective_evaluation_episodes = 2 if args.smoke else evaluation_episodes
    group_methods = (
        artifact_group_config.get("methods")
        if artifact_group_config is not None
        else None
    )
    methods = list(args.methods or group_methods or protocol["methods"])
    if args.include_sdmgrad and "sdmgrad_gppo_adaptive" not in methods:
        methods.append("sdmgrad_gppo_adaptive")
    known = set(LEARNED_METHODS) | set(FROZEN_METHODS) | {"greedy_preference"}
    unknown = sorted(set(methods) - known)
    if unknown:
        raise ValueError(f"unknown PCRL methods: {unknown}")
    suites = evaluation_suites(protocol)
    if artifact_group_config is not None and artifact_group_config.get(
        "evaluation_suites"
    ):
        requested_suites = set(artifact_group_config["evaluation_suites"])
        suites = [suite for suite in suites if suite["name"] in requested_suites]
        if len(suites) != len(requested_suites):
            raise ValueError("artifact group references an unknown evaluation suite")
    raw_training_modes = protocol.get(
        "training_task_release_modes", ["phase_staggered"]
    )
    if not isinstance(raw_training_modes, list) or not raw_training_modes:
        raise ValueError("training_task_release_modes must be a non-empty list")
    training_modes = [str(mode) for mode in raw_training_modes]
    if any(mode not in {"chain", "phase_staggered"} for mode in training_modes):
        raise ValueError(f"unsupported training task release mode: {training_modes!r}")
    phase_deadline_scale = float(
        protocol.get("phase_staggered_deadline_scale", 0.70)
    )
    if phase_deadline_scale <= 0:
        raise ValueError("phase_staggered_deadline_scale must be positive")
    controllability = protocol.get("controllability", {})
    if not isinstance(controllability, dict):
        raise ValueError("controllability must be an object")
    assignment_priority_decay = float(
        controllability.get("assignment_priority_decay", 1.0)
    )
    if assignment_priority_decay < 0 or not math.isfinite(assignment_priority_decay):
        raise ValueError("assignment_priority_decay must be finite and non-negative")
    capability_coverage_mode = str(
        (
            calibration_variant.get("capability_coverage_mode")
            if calibration_variant is not None
            else None
        )
        or controllability.get("capability_coverage_mode", "legacy")
    )
    preference_input_mode = str(
        (
            calibration_variant.get("preference_input_mode")
            if calibration_variant is not None
            else None
        )
        or protocol.get("preference_input_mode", LEGACY_PREFERENCE_INPUT_MODE)
    )
    if preference_input_mode not in PREFERENCE_INPUT_MODES:
        raise ValueError("unsupported preference_input_mode")
    checkpoint_selection = protocol.get("checkpoint_selection", {})
    if not isinstance(checkpoint_selection, dict):
        raise ValueError("checkpoint_selection must be an object")
    checkpoint_selection_mode = str(
        (
            calibration_variant.get("checkpoint_selection_mode")
            if calibration_variant is not None
            else None
        )
        or checkpoint_selection.get("mode", "legacy_l1_lexicographic")
    )
    profile_paired_validation = bool(
        checkpoint_selection.get("profile_paired_validation", False)
    )
    acceptance = protocol.get("acceptance", {})
    if not isinstance(acceptance, dict):
        raise ValueError("acceptance must be an object")
    source_seed_map = {
        int(key): int(value)
        for key, value in (
            artifact_group_config.get("source_checkpoint_seed_map", {})
            if artifact_group_config is not None
            else {}
        ).items()
    }
    effective_validation_episodes = int(
        artifact_group_config.get(
            "validation_episodes", protocol.get("validation_episodes", 0)
        )
        if artifact_group_config is not None
        else protocol.get("validation_episodes", 0)
    )
    effective_validation_interval = int(
        artifact_group_config.get(
            "validation_interval", protocol.get("validation_interval", 10)
        )
        if artifact_group_config is not None
        else protocol.get("validation_interval", 10)
    )
    effective_validation_seed = int(
        artifact_group_config.get(
            "validation_seed", protocol.get("validation_seed", 40_000)
        )
        if artifact_group_config is not None
        else protocol.get("validation_seed", 40_000)
    )
    primary_suite = next(suite for suite in suites if bool(suite["primary"]))
    training_preference_target_mode = str(
        protocol.get(
            "training_preference_target_mode",
            primary_suite["preference_target_mode"],
        )
    )

    sampler = protocol_preference_sampler(protocol)
    sampler_profile_vectors = None
    if sampler["mode"] == MODERATE_PREFERENCE_SAMPLER_MODE:
        sampler_profile_vectors = {
            name: vector.tolist()
            for name, vector in calibration_priority_profiles(
                float(sampler["calibration_priority_share"])
            ).items()
        }
    raw_profile_family = protocol.get("preference_profile_family")
    if isinstance(raw_profile_family, dict):
        preference_profile_family = str(raw_profile_family.get("id", "")).strip()
    elif raw_profile_family is not None:
        preference_profile_family = str(raw_profile_family).strip()
    else:
        preference_profile_family = (
            "dynamic-priority-share-v1"
            if sampler["mode"] == MODERATE_PREFERENCE_SAMPLER_MODE
            else "historical_named_profiles"
        )
    if not preference_profile_family:
        raise ValueError("preference_profile_family id must be non-empty")
    if training_preference_target_mode not in PREFERENCE_TARGET_MODES:
        raise ValueError(
            "training_preference_target_mode must be one of "
            f"{PREFERENCE_TARGET_MODES!r}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    source_root = str((Path.cwd() / "src").resolve())
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    manifest: dict[str, object] = {
        "version": PCRL_VERSION,
        "protocol": str(args.protocol.resolve()),
        "protocol_version": effective_protocol_version,
        "base_protocol_version": protocol_version,
        "calibration_variant": args.calibration_variant,
        "gppo_implementation_hash": implementation_hash(),
        "pcrl_implementation_hash": pcrl_implementation_hash(),
        "phase": args.phase,
        "artifact_group": artifact_group,
        "artifact_group_config": artifact_group_config,
        "output_subdirectory": output_subdirectory,
        "updates": effective_updates,
        "episodes_per_update": effective_episodes_per_update,
        "evaluation_episodes": effective_evaluation_episodes,
        "evaluation_seed": evaluation_seed,
        "protocol_budget": {
            "updates": protocol_updates,
            "episodes_per_update": protocol_episodes_per_update,
            "evaluation_episodes": evaluation_episodes,
            "evaluation_seed": evaluation_seed,
        },
        "smoke": args.smoke,
        "profile_collection": (
            "calibration_profiles"
            if args.calibration_profiles
            else "formal_evaluation_profiles"
        ),
        "methods": methods,
        "seeds": list(seeds),
        "training_task_release_modes": training_modes,
        "phase_staggered_deadline_scale": phase_deadline_scale,
        "assignment_priority_decay": assignment_priority_decay,
        "capability_coverage_mode": capability_coverage_mode,
        "preference_input_mode": preference_input_mode,
        "checkpoint_selection": {
            **checkpoint_selection,
            "mode": checkpoint_selection_mode,
        },
        "source_checkpoint_seed_map": source_seed_map,
        "training_preference_target_mode": training_preference_target_mode,
        "training_preference_sampler": sampler,
        "training_preference_sampler_profile_vectors": sampler_profile_vectors,
        "preference_profile_family": preference_profile_family,
        "evaluation_suites": suites,
        "artifacts": [],
    }
    artifacts: list[dict[str, object]] = []
    scenario_config = str(Path(protocol["base_scenario_config"]))
    scenario_config_sha256 = hashlib.sha256(
        Path(scenario_config).read_bytes()
    ).hexdigest()
    output_namespace = str(protocol.get("output_namespace", "outputs/pcrl_v0"))
    base_protocol = str(protocol.get("base_protocol", ""))
    manifest["protocol_config_sha256"] = protocol_config_sha256
    manifest["scenario_config_sha256"] = scenario_config_sha256
    manifest["output_namespace"] = output_namespace
    manifest["base_protocol"] = base_protocol
    profiles = protocol_evaluation_profiles(
        protocol, use_calibration=args.calibration_profiles
    )
    raw_validation_profiles = checkpoint_selection.get("validation_profiles")
    if raw_validation_profiles is None:
        validation_profiles = list(profiles)
    else:
        if not isinstance(raw_validation_profiles, list) or not raw_validation_profiles:
            raise ValueError(
                "checkpoint_selection.validation_profiles must be a non-empty list"
            )
        validation_profiles = [str(name) for name in raw_validation_profiles]
        unknown_validation_profiles = sorted(
            set(validation_profiles) - set(profiles)
        )
        if unknown_validation_profiles:
            raise ValueError(
                "checkpoint selection profiles are outside the evaluation family: "
                f"{unknown_validation_profiles!r}"
            )
    manifest["checkpoint_selection_validation_profiles"] = validation_profiles
    manifest["profiles"] = profiles
    scales = list(protocol["evaluation_scales"])
    if args.smoke:
        if sampler["mode"] == MODERATE_PREFERENCE_SAMPLER_MODE:
            profiles = [
                "balanced",
                "calibration_search_priority",
                "calibration_strike_priority",
                "calibration_search_strike_interp",
            ]
        elif args.calibration_profiles:
            profiles = [
                "balanced",
                "search_moderate_2to1",
                "strike_moderate_2to1",
                str(protocol["held_out_profiles"][0]),
            ]
        else:
            profiles = [
                "balanced",
                "search",
                "strike",
                str(protocol["held_out_profiles"][0]),
            ]
        scales = ["3x16"]

    if args.phase in {"train", "all"}:
        for method in methods:
            if method not in LEARNED_METHODS:
                continue
            spec = LEARNED_METHODS[method]
            for seed in seeds:
                source_seed = source_seed_map.get(seed, seed)
                output = args.output_root / "train" / method / f"seed_{seed}"
                checkpoint = output / "checkpoint.pt"
                if checkpoint.exists():
                    artifacts.append({"kind": "train", "method": method, "seed": seed, "source_checkpoint_seed": source_seed, "path": str(checkpoint), "status": "existing"})
                    continue
                command = [
                    sys.executable,
                    "train_pcrl_v0.py",
                    "--scenario-config",
                    scenario_config,
                    "--protocol-config",
                    str(args.protocol),
                    "--protocol-version",
                    effective_protocol_version,
                    "--artifact-group",
                    artifact_group,
                    "--protocol-config-sha256",
                    protocol_config_sha256,
                    "--output-namespace",
                    output_namespace,
                    "--base-protocol",
                    base_protocol,
                    "--init-checkpoint",
                    str(checkpoint_path(str(spec["source"]), source_seed)),
                    "--graph-mode",
                    str(spec["graph_mode"]),
                    "--algorithm",
                    str(spec["algorithm"]),
                    "--seed",
                    str(seed),
                    "--preference-input-mode",
                    preference_input_mode,
                    "--capability-coverage-mode",
                    capability_coverage_mode,
                    "--checkpoint-selection-mode",
                    checkpoint_selection_mode,
                    "--validation-episodes",
                    str(effective_validation_episodes),
                    "--validation-interval",
                    str(effective_validation_interval),
                    "--validation-seed",
                    str(effective_validation_seed),
                    "--max-deadline-completion-drop",
                    str(acceptance.get("max_deadline_completion_drop_vs_gppo_event", 0.03)),
                    "--max-makespan-increase-ratio",
                    str(acceptance.get("max_makespan_increase_ratio_vs_gppo_event", 0.05)),
                    "--max-coverage-loss",
                    str(acceptance.get("max_required_task_coverage_loss", 0.05)),
                    "--max-invalid-actions",
                    str(checkpoint_selection.get("max_invalid_actions", 0.0)),
                    "--max-invalid-action-increase",
                    str(
                        checkpoint_selection.get(
                            "max_invalid_action_increase_vs_gppo", 0.0
                        )
                    ),
                    "--output",
                    str(output),
                ]
                if bool(checkpoint_selection.get("save_all_candidates", False)):
                    command.append("--save-validation-candidates")
                if bool(protocol.get("freeze_backbone", False)):
                    command.append("--freeze-backbone")
                for option, key in (
                    ("--preference-group-episodes", "preference_group_episodes"),
                    ("--advantage-normalization-mode", "advantage_normalization_mode"),
                    ("--preco-value-transform", "preco_value_transform"),
                    ("--initial-task-preference-gain", "initial_task_preference_gain"),
                ):
                    if key in protocol:
                        command.extend((option, str(protocol[key])))
                if args.calibration_variant is not None:
                    command.extend(
                        ("--protocol-variant", args.calibration_variant)
                    )
                if profile_paired_validation:
                    command.append("--profile-paired-validation")
                if spec.get("no_conditioning"):
                    command.append("--no-conditioning")
                if spec.get("fixed_profile"):
                    command.extend(("--fixed-training-profile", str(spec["fixed_profile"])))
                command.extend(("--task-release-modes", *training_modes))
                command.extend(
                    (
                        "--phase-staggered-deadline-scale",
                        str(phase_deadline_scale),
                    )
                )
                command.extend(
                    ("--assignment-priority-decay", str(assignment_priority_decay))
                )
                command.extend(
                    ("--preference-target-mode", training_preference_target_mode)
                )
                command.extend(
                    (
                        "--preference-sampler-mode",
                        str(sampler["mode"]),
                        "--preference-sampler-concentration",
                        str(sampler["concentration"]),
                        "--preference-sampler-anchor-probability",
                        str(sampler["anchor_probability"]),
                        "--preference-sampler-anchor-jitter",
                        str(sampler["anchor_jitter"]),
                    )
                )
                if sampler["calibration_priority_share"] is not None:
                    command.extend(
                        (
                            "--calibration-priority-share",
                            str(sampler["calibration_priority_share"]),
                        )
                    )
                    command.extend(("--validation-profiles", *validation_profiles))
                if effective_updates is not None:
                    command.extend(("--updates", str(effective_updates)))
                if effective_episodes_per_update is not None:
                    command.extend(
                        (
                            "--episodes-per-update",
                            str(effective_episodes_per_update),
                        )
                    )
                if args.smoke:
                    command.extend(
                        (
                            "--validation-episodes", "1",
                            "--validation-interval", "1",
                            "--residual-warmup-updates", "0",
                        )
                    )
                run(command, environment=environment)
                artifacts.append({"kind": "train", "method": method, "seed": seed, "source_checkpoint_seed": source_seed, "path": str(checkpoint), "status": "created"})

    if args.phase in {"eval", "all"}:
        for suite in suites:
            suite_name = str(suite["name"])
            task_release_mode = str(suite["task_release_mode"])
            deadline_scale = str(suite["deadline_scale"])
            preference_target_mode = str(suite["preference_target_mode"])
            for method in methods:
                evaluation_seeds = seeds if method != "greedy_preference" else (-1,)
                for seed in evaluation_seeds:
                    source_seed = source_seed_map.get(seed, seed)
                    output = args.output_root / "eval" / suite_name / method / f"seed_{seed}"
                    evaluation = output / "evaluation.json"
                    artifact_base = {
                        "kind": "eval",
                        "suite": suite_name,
                        "task_release_mode": task_release_mode,
                        "deadline_scale": float(deadline_scale),
                        "preference_target_mode": preference_target_mode,
                        "method": method,
                        "seed": seed,
                        "source_checkpoint_seed": source_seed,
                        "path": str(evaluation),
                    }
                    if evaluation.exists():
                        artifacts.append({**artifact_base, "status": "existing"})
                        continue
                    command = [
                        sys.executable,
                        "evaluate_pcrl_v0.py",
                        "--scenario-config",
                        scenario_config,
                        "--protocol-version",
                        effective_protocol_version,
                        "--artifact-group",
                        artifact_group,
                        "--preference-profile-family",
                        preference_profile_family,
                        "--evaluation-suite",
                        suite_name,
                        "--protocol-config-sha256",
                        protocol_config_sha256,
                        "--protocol-config",
                        str(args.protocol),
                        "--output-namespace",
                        output_namespace,
                        "--base-protocol",
                        base_protocol,
                        "--profiles",
                        *profiles,
                        "--scales",
                        *scales,
                        "--episodes",
                        str(effective_evaluation_episodes),
                        "--eval-seed",
                        str(evaluation_seed),
                        "--task-release-mode",
                        task_release_mode,
                        "--deadline-scale",
                        deadline_scale,
                        "--assignment-priority-decay",
                        str(assignment_priority_decay),
                        "--preference-target-mode",
                        preference_target_mode,
                        "--preference-input-mode",
                        preference_input_mode,
                        "--capability-coverage-mode",
                        capability_coverage_mode,
                        "--output",
                        str(output),
                    ]
                    if args.calibration_variant is not None:
                        command.extend(
                            ("--protocol-variant", args.calibration_variant)
                        )
                    if method in LEARNED_METHODS:
                        command.extend(
                            (
                                "--checkpoint",
                                str(args.output_root / "train" / method / f"seed_{seed}" / "checkpoint.pt"),
                            )
                        )
                    elif method in FROZEN_METHODS:
                        command.extend(
                            ("--checkpoint", str(checkpoint_path(FROZEN_METHODS[method], source_seed)))
                        )
                    else:
                        command.extend(("--baseline", "greedy_preference"))
                    if sampler["calibration_priority_share"] is not None:
                        command.extend(
                            (
                                "--calibration-priority-share",
                                str(sampler["calibration_priority_share"]),
                            )
                        )
                    if (
                        artifact_group_config is not None
                        and artifact_group_config.get(
                            "allow_failed_selection_diagnostic", False
                        )
                    ):
                        command.append("--allow-failed-selection-diagnostic")
                    run(command, environment=environment)
                    artifacts.append({**artifact_base, "status": "created"})

    manifest["artifacts"] = artifacts
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"artifacts": len(artifacts), "manifest": str(args.output_root / "run_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from uav_assignment.paper_env import PaperEnvConfig
from uav_assignment.pcrl_oracle import (
    OracleConfig,
    choose_oracle_action,
    step_oracle_action,
)
from uav_assignment.pcrl_v0 import (
    HARD2_EVALUATION_PROFILE_NAMES,
    HARD3_CALIBRATION_PROFILE_NAMES,
    PREFERENCE_TARGET_MODES,
    PreferencePaperEnv,
    calibration_priority_profiles,
    task_preference_profile,
)


PROFILE_SETS = {
    "hard2": HARD2_EVALUATION_PROFILE_NAMES,
    "hard3_moderate": HARD3_CALIBRATION_PROFILE_NAMES,
}
DEFAULT_PROFILES = HARD2_EVALUATION_PROFILE_NAMES


def parse_scale(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", maxsplit=1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the perfect-information PCRL preference oracle"
    )
    parser.add_argument(
        "--scenario-config", type=Path, default=Path("configs/gppo_v2_hard.json")
    )
    parser.add_argument(
        "--protocol-version",
        help="explicit protocol identity recorded in every evaluation row",
    )
    parser.add_argument(
        "--profile-set",
        choices=tuple(PROFILE_SETS),
        default="hard2",
        help=(
            "named profile collection; hard3_moderate uses separate 2:1 "
            "priority anchors"
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="explicit profile names (overrides --profile-set)",
    )
    parser.add_argument(
        "--calibration-priority-share",
        type=float,
        help=(
            "opt-in dynamic eight-profile calibration family; overrides "
            "--profiles and --profile-set"
        ),
    )
    parser.add_argument(
        "--scales", nargs="+", default=("2x12", "3x16", "3x20", "4x24")
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=70_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-release-mode",
        choices=("chain", "phase_staggered"),
        default="phase_staggered",
    )
    parser.add_argument("--deadline-scale", type=float, default=0.70)
    parser.add_argument("--decays", nargs="+", type=float, default=(1.0, 2.0, 3.0))
    parser.add_argument("--coverage-floor", type=float, default=0.0)
    parser.add_argument("--rollout-candidates", type=int, default=12)
    parser.add_argument("--deadline-completion-floor", type=float, default=0.60)
    parser.add_argument("--minimum-type-coverage-floor", type=float, default=0.40)
    parser.add_argument("--allow-strategic-wait", action="store_true")
    parser.add_argument("--rollout-wait-margin", type=float, default=0.05)
    parser.add_argument("--prerequisite-gain", type=float, default=0.55)
    parser.add_argument(
        "--preference-target-mode",
        choices=PREFERENCE_TARGET_MODES,
        default="nominal",
    )
    parser.add_argument("--save-event-log", action="store_true")
    return parser.parse_args()


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()) if array.size else float("nan")
    if array.size < 2:
        half_width = 0.0 if array.size == 1 else float("nan")
    else:
        half_width = float(1.96 * array.std(ddof=1) / math.sqrt(array.size))
    return {
        "n": int(array.size),
        "mean": mean,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _eval_seed_clustered_ci(
    rows: list[dict[str, Any]], key: str
) -> dict[str, float | int | str]:
    """CI over evaluation seeds, not correlated profile/scale rows."""

    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["eval_seed"]), []).append(float(row[key]))
    result = _mean_ci(
        [float(np.mean(values)) for _, values in sorted(by_seed.items())]
    )
    result["sampling_unit"] = "eval_seed_macro"
    return result


def resolve_profiles(
    profiles: list[str] | tuple[str, ...] | None,
    profile_set: str,
) -> tuple[str, ...]:
    """Resolve explicit profiles or a versioned calibration collection."""

    if profiles:
        return tuple(profiles)
    try:
        return tuple(PROFILE_SETS[profile_set])
    except KeyError as error:
        raise ValueError(f"unknown profile set: {profile_set}") from error


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_decay: dict[str, Any] = {}
    for decay in sorted({float(row["assignment_priority_decay"]) for row in rows}):
        selected = [
            row for row in rows if float(row["assignment_priority_decay"]) == decay
        ]
        by_decay[str(decay)] = {
            "preference_l1": _eval_seed_clustered_ci(selected, "preference_l1"),
            "deadline_completion_rate": _eval_seed_clustered_ci(
                selected, "deadline_completion_rate"
            ),
            "makespan": _eval_seed_clustered_ci(selected, "makespan"),
            "minimum_task_coverage": _eval_seed_clustered_ci(
                selected, "minimum_task_coverage"
            ),
            "invalid_actions": _eval_seed_clustered_ci(
                selected, "invalid_actions"
            ),
            "strategic_waits": _eval_seed_clustered_ci(
                selected, "strategic_waits"
            ),
            "by_profile": {
                profile: {
                    "preference_l1": _eval_seed_clustered_ci(
                        [
                            row
                            for row in selected
                            if row["preference_profile"] == profile
                        ],
                        "preference_l1",
                    ),
                    "priority_weighted_assignment_mix": np.mean(
                        [
                            np.asarray(
                                row["priority_weighted_assignment_mix"],
                                dtype=np.float64,
                            )
                            for row in selected
                            if row["preference_profile"] == profile
                        ],
                        axis=0,
                    ).tolist(),
                }
                for profile in sorted(
                    {str(row["preference_profile"]) for row in selected}
                )
            },
        }
    best_decay = min(
        by_decay,
        key=lambda decay: float(by_decay[decay]["preference_l1"]["mean"]),
    )
    relaxed = bool(rows[0].get("allow_strategic_wait", False))
    return {
        "oracle_status": (
            "relaxed-action perfect-information upper bound; strategic noop is outside the frozen action mask"
            if relaxed
            else "mask-faithful perfect-information headroom; not a weak-communication baseline"
        ),
        "action_space_status": (
            "relaxed_strategic_wait" if relaxed else "frozen_mask_faithful"
        ),
        "best_decay": float(best_decay),
        "by_decay": by_decay,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PCRL perfect-information Oracle headroom audit",
        "",
        f"> {summary['oracle_status']}.",
        "> Confidence intervals use eval-seed macro averages; profile and scale "
        "rows sharing a seed are not treated as independent samples.",
        "",
        "| decay | preference L1 | deadline completion | makespan | min type coverage | invalid actions | waits |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for decay, row in summary["by_decay"].items():
        lines.append(
            f"| {decay} | {row['preference_l1']['mean']:.4f} | "
            f"{row['deadline_completion_rate']['mean']:.4f} | "
            f"{row['makespan']['mean']:.4f} | "
            f"{row['minimum_task_coverage']['mean']:.4f} | "
            f"{row['invalid_actions']['mean']:.3f} | "
            f"{row['strategic_waits']['mean']:.3f} |"
        )
    lines.extend(("", f"Best calibrated decay: `{summary['best_decay']}`.", ""))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    protocol_version = (
        str(args.protocol_version).strip()
        if args.protocol_version is not None
        else None
    )
    if args.protocol_version is not None and not protocol_version:
        raise ValueError("--protocol-version must be non-empty when supplied")
    if args.calibration_priority_share is not None:
        profile_vectors = calibration_priority_profiles(
            args.calibration_priority_share
        )
    else:
        profile_names = resolve_profiles(args.profiles, args.profile_set)
        profile_vectors = {
            profile_name: task_preference_profile(profile_name)
            for profile_name in profile_names
        }
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.deadline_scale <= 0:
        raise ValueError("--deadline-scale must be positive")
    if not 0.0 <= args.coverage_floor <= 1.0:
        raise ValueError("--coverage-floor must lie in [0, 1]")
    if args.rollout_candidates < 0:
        raise ValueError("--rollout-candidates must be non-negative")
    if not 0.0 <= args.deadline_completion_floor <= 1.0:
        raise ValueError("--deadline-completion-floor must lie in [0, 1]")
    if not 0.0 <= args.minimum_type_coverage_floor <= 1.0:
        raise ValueError("--minimum-type-coverage-floor must lie in [0, 1]")
    if not np.isfinite(args.rollout_wait_margin) or args.rollout_wait_margin < 0:
        raise ValueError("--rollout-wait-margin must be finite and non-negative")
    if not np.isfinite(args.prerequisite_gain) or args.prerequisite_gain < 0:
        raise ValueError("--prerequisite-gain must be finite and non-negative")
    if any(not np.isfinite(decay) or decay < 0 for decay in args.decays):
        raise ValueError("all decays must be finite and non-negative")

    payload = json.loads(args.scenario_config.read_text(encoding="utf-8"))
    base_config = PaperEnvConfig(**payload["scenario"])
    scale_deadlines = {
        parse_scale(scale): float(deadline)
        for scale, deadline in payload["scale_deadlines"].items()
    }
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for decay in args.decays:
        oracle_config = OracleConfig(
            decay=float(decay),
            coverage_floor=args.coverage_floor,
            target_mode=args.preference_target_mode,
            rollout_candidates=args.rollout_candidates,
            allow_strategic_wait=args.allow_strategic_wait,
            rollout_wait_margin=args.rollout_wait_margin,
            prerequisite_gain=args.prerequisite_gain,
            deadline_completion_floor=args.deadline_completion_floor,
            minimum_type_coverage_floor=args.minimum_type_coverage_floor,
        )
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
            for profile_name, preference in profile_vectors.items():
                for episode in range(args.episodes):
                    episode_seed = args.eval_seed + episode
                    env = PreferencePaperEnv(
                        config,
                        preference=preference,
                        task_release_mode=args.task_release_mode,
                        assignment_priority_decay=float(decay),
                        preference_target_mode=args.preference_target_mode,
                    )
                    env.reset(seed=episode_seed)
                    done = False
                    strategic_waits = 0
                    method_id = (
                        "oracle_perfect_information_relaxed_wait"
                        if args.allow_strategic_wait
                        else "oracle_perfect_information_mask_faithful"
                    )
                    while not done:
                        action = choose_oracle_action(env, config=oracle_config)
                        _, _, done, info = step_oracle_action(
                            env, action, sync_mode="event"
                        )
                        strategic_waits += int(
                            bool(info.get("oracle_strategic_wait", False))
                        )
                    metrics = env.metrics()
                    metrics.update(
                        {
                            "method_id": method_id,
                            "model_kind": "oracle",
                            "preference_profile": profile_name,
                            "scale": scale_text,
                            "eval_seed": episode_seed,
                            "scenario_version": str(payload.get("version", "")),
                            "task_release_mode": args.task_release_mode,
                            "deadline_scale": args.deadline_scale,
                            "assignment_priority_decay": float(decay),
                            "preference_target_mode": args.preference_target_mode,
                            "coverage_floor": args.coverage_floor,
                            "rollout_candidates": args.rollout_candidates,
                            "strategic_waits": strategic_waits,
                            "allow_strategic_wait": args.allow_strategic_wait,
                            "rollout_wait_margin": args.rollout_wait_margin,
                            "prerequisite_gain": args.prerequisite_gain,
                            "deadline_completion_floor": args.deadline_completion_floor,
                            "minimum_type_coverage_floor": args.minimum_type_coverage_floor,
                            "uses_true_state": True,
                            "protocol_compliant_weak_communication": False,
                        }
                    )
                    if protocol_version is not None:
                        metrics["protocol_version"] = protocol_version
                    if args.calibration_priority_share is not None:
                        metrics.update(
                            {
                                "calibration_priority_share": float(
                                    args.calibration_priority_share
                                ),
                                "calibration_profile_vector": preference.tolist(),
                            }
                        )
                    rows.append(metrics)
                    if args.save_event_log:
                        event_rows.extend(
                            {
                                "method_id": method_id,
                                "protocol_version": protocol_version,
                                "scale": scale_text,
                                "preference_profile": profile_name,
                                "calibration_priority_share": (
                                    float(args.calibration_priority_share)
                                    if args.calibration_priority_share is not None
                                    else None
                                ),
                                "calibration_profile_vector": (
                                    preference.tolist()
                                    if args.calibration_priority_share is not None
                                    else None
                                ),
                                "assignment_priority_decay": float(decay),
                                "eval_seed": episode_seed,
                                **event,
                            }
                            for event in env.event_log
                        )

    summary = summarize(rows)
    summary.update(
        {
            "task_release_mode": args.task_release_mode,
            "protocol_version": protocol_version,
            "deadline_scale": args.deadline_scale,
            "preference_target_mode": args.preference_target_mode,
            "coverage_floor": args.coverage_floor,
            "rollout_candidates": args.rollout_candidates,
            "allow_strategic_wait": args.allow_strategic_wait,
            "rollout_wait_margin": args.rollout_wait_margin,
            "prerequisite_gain": args.prerequisite_gain,
            "deadline_completion_floor": args.deadline_completion_floor,
            "minimum_type_coverage_floor": args.minimum_type_coverage_floor,
            "profile_set": (
                "dynamic_calibration_priority_share"
                if args.calibration_priority_share is not None
                else args.profile_set if args.profiles is None else "explicit"
            ),
            "profiles": list(profile_vectors),
            "calibration_priority_share": (
                float(args.calibration_priority_share)
                if args.calibration_priority_share is not None
                else None
            ),
            "calibration_profile_vectors": (
                {
                    name: vector.tolist()
                    for name, vector in profile_vectors.items()
                }
                if args.calibration_priority_share is not None
                else None
            ),
            "scales": list(args.scales),
            "episodes_per_cell": args.episodes,
            "eval_seed": args.eval_seed,
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "evaluation.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "PCRL_ORACLE_HEADROOM.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    if args.save_event_log:
        with (args.output / "event_log.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for event in event_rows:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "best_decay": summary["best_decay"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

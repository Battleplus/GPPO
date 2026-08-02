from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from uav_assignment.gppo_v2 import METHOD_SPECS, config_hash, implementation_hash
from uav_assignment.paper_env import PaperEnvConfig
from uav_assignment.paper_models import PaperHeteroActorCritic


CHECKPOINT_VERSION = "paper-aligned-gppo-v2"
TRAINING_HYPERPARAMETER_FIELDS = (
    "learning_rate",
    "entropy_coefficient",
    "gamma",
    "gae_lambda",
    "update_epochs",
    "minibatch_size",
)
REQUIRED_EVALUATION_METRICS = (
    "makespan",
    "completion_rate",
    "deadline_completion_rate",
    "mission_success",
    "deadline_remaining_tasks",
    "throughput",
    "invalid_actions",
    "communication_events",
    "heartbeat_messages",
    "reallocated_tasks",
    "reallocation_successes",
    "reallocation_success_rate",
    "return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manifest-locked GPPO-v2 study")
    parser.add_argument("--manifest", type=Path, default=Path("configs/gppo_v2_hard.json"))
    parser.add_argument(
        "--phase", choices=("smoke", "train", "evaluate", "summarize", "all"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/paper_aligned/gppo_v2_hard"))
    parser.add_argument("--jobs", type=int, default=1)
    supplementary = parser.add_mutually_exclusive_group()
    supplementary.add_argument(
        "--include-supplementary",
        dest="include_supplementary",
        action="store_true",
        help=(
            "include manifest supplementary methods (the frozen formal-protocol "
            "default; this flag is retained for explicitness)"
        ),
    )
    supplementary.add_argument(
        "--exclude-supplementary",
        dest="include_supplementary",
        action="store_false",
        help="run an exploratory core-only subset instead of the frozen formal protocol",
    )
    parser.set_defaults(include_supplementary=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(cwd / "src")
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def run_many(commands: list[list[str]], cwd: Path, jobs: int) -> None:
    if jobs <= 1:
        for command in commands:
            run_command(command, cwd)
        return
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(run_command, command, cwd): command for command in commands}
        for future in as_completed(futures):
            future.result()


def methods(manifest: dict[str, object], include_supplementary: bool) -> list[str]:
    result = list(manifest["methods"])
    if include_supplementary:
        result.extend(manifest.get("supplementary_methods", []))
    return [str(value) for value in result]


def _normalized_deadlines(
    manifest: dict[str, object], scales: list[str] | tuple[str, ...]
) -> dict[str, float]:
    selected = set(scales)
    return {
        str(scale): float(deadline)
        for scale, deadline in dict(manifest.get("scale_deadlines", {})).items()
        if str(scale) in selected
    }


def expected_scenario_hash(manifest: dict[str, object]) -> str:
    return config_hash(PaperEnvConfig(**dict(manifest["scenario"])))


def expected_evaluation_hash(
    manifest: dict[str, object], scales: list[str] | tuple[str, ...]
) -> str:
    base_config = PaperEnvConfig(**dict(manifest["scenario"]))
    return config_hash(
        {
            "environment": base_config.to_dict(),
            "scale_deadlines": _normalized_deadlines(manifest, scales),
        }
    )


def checkpoint_protocol_errors(
    payload: object,
    method_id: str,
    manifest: dict[str, object],
    seed: int,
    *,
    updates: int | None = None,
    episodes_per_update: int | None = None,
    validation_episodes: int | None = None,
    validation_interval: int | None = None,
    validation_seed: int | None = None,
    hidden_dim: int | None = None,
    train_scales: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return every reason a learned checkpoint is unsafe for this protocol."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["checkpoint payload is not a dictionary"]
    if method_id not in METHOD_SPECS:
        return [f"unknown method: {method_id}"]
    spec = METHOD_SPECS[method_id]
    if not spec.learned:
        errors.append(f"{method_id} is not a learned method")

    expected_updates = int(manifest["updates"] if updates is None else updates)
    expected_episodes = int(
        manifest["episodes_per_update"]
        if episodes_per_update is None
        else episodes_per_update
    )
    expected_validation_episodes = int(
        manifest["validation_episodes"]
        if validation_episodes is None
        else validation_episodes
    )
    expected_validation_interval = int(
        manifest["validation_interval"]
        if validation_interval is None
        else validation_interval
    )
    expected_validation_seed = int(
        manifest["validation_seed"] if validation_seed is None else validation_seed
    )
    expected_hidden_dim = int(
        manifest["hidden_dim"] if hidden_dim is None else hidden_dim
    )
    expected_scales = [
        str(scale)
        for scale in (manifest["train_scales"] if train_scales is None else train_scales)
    ]
    expected_deadlines = _normalized_deadlines(manifest, expected_scales)
    scenario_hash = expected_scenario_hash(manifest)
    expected_config_hash = str(manifest.get("config_hash", scenario_hash))

    exact_fields = {
        "version": CHECKPOINT_VERSION,
        "method_id": method_id,
        "scenario": str(manifest["version"]),
        "scenario_hash": scenario_hash,
        "implementation_hash": implementation_hash(),
        "algorithm": spec.algorithm,
        "graph_mode": spec.graph_mode,
        "sync_mode": spec.sync_mode,
    }
    for field, expected in exact_fields.items():
        if payload.get(field) != expected:
            errors.append(
                f"checkpoint {field} is {payload.get(field)!r}, expected {expected!r}"
            )
    if "config_hash" in payload and payload.get("config_hash") != expected_config_hash:
        errors.append(
            f"checkpoint config_hash is {payload.get('config_hash')!r}, "
            f"expected {expected_config_hash!r}"
        )

    training = payload.get("training")
    if not isinstance(training, dict):
        errors.append("checkpoint training metadata is missing")
        training = {}
    integer_training_fields = {
        "seed": seed,
        "updates": expected_updates,
        "episodes_per_update": expected_episodes,
        "validation_episodes": expected_validation_episodes,
        "validation_interval": expected_validation_interval,
        "validation_seed": expected_validation_seed,
        "hidden_dim": expected_hidden_dim,
    }
    for field, expected in integer_training_fields.items():
        try:
            actual = int(training[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"checkpoint training.{field} is missing or invalid")
            continue
        if actual != expected:
            errors.append(
                f"checkpoint training.{field} is {actual}, expected {expected}"
            )
    for field in TRAINING_HYPERPARAMETER_FIELDS:
        expected = manifest[field]
        try:
            actual = float(training[field])
        except (KeyError, TypeError, ValueError):
            errors.append(f"checkpoint training.{field} is missing or invalid")
            continue
        if actual != float(expected):
            errors.append(
                f"checkpoint training.{field} is {actual}, expected {expected}"
            )

    actual_scales = payload.get("train_scales")
    if not isinstance(actual_scales, (list, tuple)) or [
        str(scale) for scale in actual_scales
    ] != expected_scales:
        errors.append(
            f"checkpoint train_scales are {actual_scales!r}, expected {expected_scales!r}"
        )
    training_scales = training.get("train_scales")
    if not isinstance(training_scales, (list, tuple)) or (
        [str(scale) for scale in training_scales] != expected_scales
    ):
        errors.append(
            f"checkpoint training.train_scales are {training_scales!r}, "
            f"expected {expected_scales!r}"
        )
    try:
        actual_deadlines = {
            str(scale): float(deadline)
            for scale, deadline in dict(payload.get("scale_deadlines", {})).items()
        }
    except (TypeError, ValueError):
        actual_deadlines = {}
    if actual_deadlines != expected_deadlines:
        errors.append(
            f"checkpoint scale_deadlines are {actual_deadlines!r}, "
            f"expected {expected_deadlines!r}"
        )

    env_config = payload.get("env_config")
    if not isinstance(env_config, dict):
        errors.append("checkpoint env_config is missing")
    else:
        try:
            actual_env = PaperEnvConfig(**env_config)
        except (TypeError, ValueError) as exc:
            errors.append(f"checkpoint env_config is invalid: {exc}")
        else:
            if config_hash(actual_env) != scenario_hash:
                errors.append("checkpoint env_config does not match scenario_hash")
            try:
                actual_env_seed = int(env_config.get("seed", -1))
            except (TypeError, ValueError):
                errors.append("checkpoint env_config seed is missing or invalid")
            else:
                if actual_env_seed != seed:
                    errors.append(
                        f"checkpoint env_config seed is {env_config.get('seed')!r}, "
                        f"expected {seed}"
                    )

    model_config = payload.get("model_config")
    if not isinstance(model_config, dict):
        errors.append("checkpoint model_config is missing")
    else:
        if model_config.get("graph_mode") != spec.graph_mode:
            errors.append(
                f"checkpoint model_config.graph_mode is "
                f"{model_config.get('graph_mode')!r}, expected {spec.graph_mode!r}"
            )
        for field, expected in (
            ("hidden_dim", expected_hidden_dim),
            ("max_uavs", int(dict(manifest["scenario"])["max_uavs"])),
            ("max_tasks", int(dict(manifest["scenario"])["max_tasks"])),
        ):
            try:
                actual = int(model_config[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"checkpoint model_config.{field} is missing or invalid")
                continue
            if actual != expected:
                errors.append(
                    f"checkpoint model_config.{field} is {actual}, expected {expected}"
                )
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        errors.append("checkpoint model_state is missing or invalid")
    elif isinstance(model_config, dict):
        try:
            verification_model = PaperHeteroActorCritic(**model_config)
            verification_model.load_state_dict(model_state, strict=True)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"checkpoint model_state cannot be loaded strictly: {exc}")
    return errors


def checkpoint_matches(
    path: Path,
    method_id: str,
    manifest: dict[str, object],
    seed: int,
    updates: int,
    episodes: int,
    validation_episodes: int | None = None,
    validation_interval: int | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return not checkpoint_protocol_errors(
        payload,
        method_id,
        manifest,
        seed,
        updates=updates,
        episodes_per_update=episodes,
        validation_episodes=validation_episodes,
        validation_interval=validation_interval,
    )


def evaluation_protocol_errors(
    rows: object,
    method_id: str,
    seed: int,
    manifest: dict[str, object],
    scales: list[str] | tuple[str, ...],
    episodes: int,
    *,
    training_updates: int | None = None,
    training_episodes_per_update: int | None = None,
    validation_episodes: int | None = None,
    validation_interval: int | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return ["evaluation payload must be a list of row dictionaries"]
    expected_scales = [str(scale) for scale in scales]
    expected_seed_range = list(
        range(int(manifest["evaluation_seed"]), int(manifest["evaluation_seed"]) + episodes)
    )
    expected_hash = expected_evaluation_hash(manifest, expected_scales)
    expected_impl = implementation_hash()
    spec = METHOD_SPECS.get(method_id)
    if spec is None:
        return [f"unknown method: {method_id}"]
    if len(rows) != len(expected_scales) * episodes:
        errors.append(
            f"evaluation has {len(rows)} rows, expected {len(expected_scales) * episodes}"
        )
    observed_scales = {str(row.get("scale")) for row in rows}
    if observed_scales != set(expected_scales):
        errors.append(
            f"evaluation scales are {sorted(observed_scales)}, expected {expected_scales}"
        )
    expected_checkpoint_version = CHECKPOINT_VERSION if spec.learned else "scenario-config"
    expected_fields = {
        "method_id": method_id,
        "training_seed": seed,
        "algorithm": spec.algorithm,
        "graph_mode": spec.graph_mode,
        "sync_mode": spec.sync_mode,
        "scenario_hash": expected_hash,
        "scenario_version": str(manifest["version"]),
        "implementation_hash": expected_impl,
        "checkpoint_version": expected_checkpoint_version,
    }
    if spec.learned:
        expected_fields.update(
            {
                "checkpoint_scenario_hash": expected_scenario_hash(manifest),
                "train_scales": [str(scale) for scale in manifest["train_scales"]],
                "training_updates": int(
                    manifest["updates"]
                    if training_updates is None
                    else training_updates
                ),
                "training_episodes_per_update": int(
                    manifest["episodes_per_update"]
                    if training_episodes_per_update is None
                    else training_episodes_per_update
                ),
                "validation_episodes": int(
                    manifest["validation_episodes"]
                    if validation_episodes is None
                    else validation_episodes
                ),
                "validation_interval": int(
                    manifest["validation_interval"]
                    if validation_interval is None
                    else validation_interval
                ),
                "validation_seed": int(manifest["validation_seed"]),
                "hidden_dim": int(manifest["hidden_dim"]),
                **{
                    f"training_{field}": manifest[field]
                    for field in TRAINING_HYPERPARAMETER_FIELDS
                },
            }
        )
    for index, row in enumerate(rows):
        for field, expected in expected_fields.items():
            actual = row.get(field)
            if field == "training_seed":
                try:
                    actual = int(actual)
                except (TypeError, ValueError):
                    pass
            if actual != expected:
                errors.append(
                    f"row {index} {field} is {actual!r}, expected {expected!r}"
                )
                break
        if spec.learned and row.get("checkpoint_config_hash") != expected_scenario_hash(
            manifest
        ):
            errors.append(
                f"row {index} checkpoint_config_hash is "
                f"{row.get('checkpoint_config_hash')!r}, expected "
                f"{expected_scenario_hash(manifest)!r}"
            )
    for scale in expected_scales:
        try:
            expected_uavs, expected_tasks = (
                int(part) for part in scale.lower().split("x", maxsplit=1)
            )
        except (TypeError, ValueError):
            errors.append(f"manifest evaluation scale {scale!r} is invalid")
            continue
        expected_deadline = float(
            dict(manifest.get("scale_deadlines", {})).get(
                scale, dict(manifest["scenario"])["mission_deadline"]
            )
        )
        scale_rows = [row for row in rows if str(row.get("scale")) == scale]
        if len(scale_rows) != episodes:
            errors.append(
                f"{method_id}/{scale}/seed-{seed} has {len(scale_rows)} episodes, "
                f"expected {episodes}"
            )
            continue
        try:
            eval_seeds = sorted(int(row["eval_seed"]) for row in scale_rows)
        except (KeyError, TypeError, ValueError):
            errors.append(f"{method_id}/{scale}/seed-{seed} has invalid eval seeds")
            continue
        if eval_seeds != expected_seed_range:
            errors.append(
                f"{method_id}/{scale}/seed-{seed} eval seeds are not the contiguous "
                f"range {expected_seed_range[0]}..{expected_seed_range[-1]}"
            )
        for index, row in enumerate(scale_rows):
            identity_fields = {
                "active_uavs": expected_uavs,
                "initial_tasks": expected_tasks,
                "mission_deadline": expected_deadline,
            }
            for field, expected in identity_fields.items():
                try:
                    actual = float(row[field])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"{method_id}/{scale}/seed-{seed} row {index} "
                        f"has invalid {field}"
                    )
                    continue
                if actual != float(expected):
                    errors.append(
                        f"{method_id}/{scale}/seed-{seed} row {index} {field} "
                        f"is {actual}, expected {expected}"
                    )
            for metric in REQUIRED_EVALUATION_METRICS:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"{method_id}/{scale}/seed-{seed} row {index} "
                        f"has invalid metric {metric}"
                    )
                    continue
                if not np.isfinite(value):
                    errors.append(
                        f"{method_id}/{scale}/seed-{seed} row {index} "
                        f"metric {metric} is not finite"
                    )
    return errors


def evaluation_matches(
    path: Path,
    method_id: str,
    seed: int,
    manifest: dict[str, object],
    scales: list[str] | tuple[str, ...],
    episodes: int,
    *,
    training_updates: int | None = None,
    training_episodes_per_update: int | None = None,
    validation_episodes: int | None = None,
    validation_interval: int | None = None,
) -> bool:
    if not path.exists():
        return False
    if not (path.parent / "event_log.jsonl").exists():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if len(rows) != len(scales) * episodes:
        return False
    return not evaluation_protocol_errors(
        rows,
        method_id,
        seed,
        manifest,
        scales,
        episodes,
        training_updates=training_updates,
        training_episodes_per_update=training_episodes_per_update,
        validation_episodes=validation_episodes,
        validation_interval=validation_interval,
    )


def train_commands(
    args: argparse.Namespace,
    manifest: dict[str, object],
    root: Path,
    smoke: bool,
) -> list[list[str]]:
    selected = [
        method_id
        for method_id in methods(manifest, args.include_supplementary)
        if METHOD_SPECS[method_id].learned
    ]
    seeds = [1] if smoke else [int(seed) for seed in manifest["learned_seeds"]]
    updates = 1 if smoke else int(manifest["updates"])
    episodes = 2 if smoke else int(manifest["episodes_per_update"])
    validation_episodes = 1 if smoke else int(manifest["validation_episodes"])
    train_scales = [str(value) for value in manifest["train_scales"]]
    scale_deadlines = [
        f"{scale}={deadline}"
        for scale, deadline in dict(manifest.get("scale_deadlines", {})).items()
        if scale in train_scales
    ]
    commands: list[list[str]] = []
    for method_id in selected:
        for seed in seeds:
            output = root / "train" / method_id / f"seed_{seed}"
            if not args.force and checkpoint_matches(
                output / "checkpoint.pt",
                method_id,
                manifest,
                seed,
                updates,
                episodes,
                validation_episodes,
                1 if smoke else int(manifest["validation_interval"]),
            ):
                continue
            commands.append(
                [
                    sys.executable,
                    "train_paper_gppo.py",
                    "--method-id",
                    method_id,
                    "--scenario-config",
                    str(args.manifest),
                    "--seed",
                    str(seed),
                    "--train-scales",
                    *train_scales,
                    "--scale-deadlines",
                    *scale_deadlines,
                    "--updates",
                    str(updates),
                    "--episodes-per-update",
                    str(episodes),
                    "--hidden-dim",
                    str(manifest["hidden_dim"]),
                    "--learning-rate",
                    str(manifest["learning_rate"]),
                    "--entropy-coefficient",
                    str(manifest["entropy_coefficient"]),
                    "--gamma",
                    str(manifest["gamma"]),
                    "--gae-lambda",
                    str(manifest["gae_lambda"]),
                    "--update-epochs",
                    str(manifest["update_epochs"]),
                    "--minibatch-size",
                    str(manifest["minibatch_size"]),
                    "--validation-episodes",
                    str(validation_episodes),
                    "--validation-interval",
                    "1" if smoke else str(manifest["validation_interval"]),
                    "--validation-seed",
                    str(manifest["validation_seed"]),
                    "--output",
                    str(output),
                ]
            )
    return commands


def evaluation_commands(
    args: argparse.Namespace,
    manifest: dict[str, object],
    root: Path,
    smoke: bool,
) -> list[list[str]]:
    selected = methods(manifest, args.include_supplementary)
    seeds = [1] if smoke else [int(seed) for seed in manifest["learned_seeds"]]
    episodes = 10 if smoke else int(manifest["evaluation_episodes"])
    scales = [str(value) for value in manifest["evaluation_scales"]]
    if smoke:
        scales = (scales[0], scales[-1])
    scale_deadlines = [
        f"{scale}={deadline}"
        for scale, deadline in dict(manifest.get("scale_deadlines", {})).items()
        if scale in scales
    ]
    commands: list[list[str]] = []
    for method_id in selected:
        spec = METHOD_SPECS[method_id]
        method_seeds = seeds if spec.learned else [-1]
        for seed in method_seeds:
            output = root / "eval" / method_id / (f"seed_{seed}" if spec.learned else "baseline")
            if not args.force and evaluation_matches(
                output / "evaluation.json",
                method_id,
                seed,
                manifest,
                scales,
                episodes,
                training_updates=(1 if smoke else None),
                training_episodes_per_update=(2 if smoke else None),
                validation_episodes=(1 if smoke else None),
                validation_interval=(1 if smoke else None),
            ):
                continue
            command = [
                sys.executable,
                "evaluate_paper_gppo.py",
                "--method-id",
                method_id,
                "--scales",
                *scales,
                "--episodes",
                str(episodes),
                "--scale-deadlines",
                *scale_deadlines,
                "--eval-seed",
                str(manifest["evaluation_seed"]),
                "--output",
                str(output),
                "--save-event-log",
            ]
            if spec.learned:
                expected_updates = 1 if smoke else int(manifest["updates"])
                expected_training_episodes = (
                    2 if smoke else int(manifest["episodes_per_update"])
                )
                expected_validation_episodes = (
                    1 if smoke else int(manifest["validation_episodes"])
                )
                expected_validation_interval = (
                    1 if smoke else int(manifest["validation_interval"])
                )
                command.extend(
                    [
                        "--checkpoint",
                        str(root / "train" / method_id / f"seed_{seed}" / "checkpoint.pt"),
                        "--scenario-config",
                        str(args.manifest),
                        "--expected-training-seed",
                        str(seed),
                        "--expected-updates",
                        str(expected_updates),
                        "--expected-episodes-per-update",
                        str(expected_training_episodes),
                        "--expected-validation-episodes",
                        str(expected_validation_episodes),
                        "--expected-validation-interval",
                        str(expected_validation_interval),
                        "--expected-validation-seed",
                        str(manifest["validation_seed"]),
                        "--expected-hidden-dim",
                        str(manifest["hidden_dim"]),
                    ]
                )
            else:
                command.extend(
                    [
                        "--baseline",
                        spec.algorithm,
                        "--scenario-config",
                        str(args.manifest),
                        "--sync-mode",
                        spec.sync_mode,
                    ]
                )
            commands.append(command)
    return commands


def summary_evaluation_paths(
    root: Path, manifest: dict[str, object], include_supplementary: bool
) -> list[Path]:
    evaluation_root = root / "eval"
    allowed_methods = set(methods(manifest, include_supplementary))
    return sorted(
        path
        for path in evaluation_root.glob("**/evaluation.json")
        if path.relative_to(evaluation_root).parts[0] in allowed_methods
    )


def summarize(args: argparse.Namespace, root: Path, smoke: bool) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evaluations = summary_evaluation_paths(
        root, manifest, args.include_supplementary
    )
    if not evaluations:
        raise RuntimeError(f"no evaluations under {root / 'eval'}")
    summary_manifest = args.manifest
    if smoke:
        smoke_manifest = dict(manifest)
        smoke_manifest.update(
            {
                "learned_seeds": [1],
                "updates": 1,
                "episodes_per_update": 2,
                "validation_episodes": 1,
                "validation_interval": 1,
                "evaluation_episodes": 10,
                "evaluation_scales": [
                    str(manifest["evaluation_scales"][0]),
                    str(manifest["evaluation_scales"][-1]),
                ],
            }
        )
        summary_manifest = root / "smoke_manifest.json"
        summary_manifest.write_text(
            json.dumps(smoke_manifest, indent=2), encoding="utf-8"
        )
    command = [
        sys.executable,
        "summarize_gppo_v2.py",
        *(str(path) for path in evaluations),
        "--output",
        str(root / "summary"),
        "--manifest",
        str(summary_manifest),
    ]
    if not args.include_supplementary:
        command.append("--exclude-supplementary")
    run_command(command, Path.cwd())
    run_command(
        [
            sys.executable,
            "analyze_gppo_v2.py",
            "--training-root",
            str(root / "train"),
            "--output",
            str(root / "summary"),
        ],
        Path.cwd(),
    )
    run_command(
        [
            sys.executable,
            "audit_gppo_v2_parameters.py",
            "--output",
            str(root / "summary"),
        ],
        Path.cwd(),
    )


def verify_random_smoke(root: Path) -> None:
    path = root / "eval" / "random_event" / "baseline" / "evaluation.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rate = float(np.mean([float(row["deadline_completion_rate"]) for row in rows]))
    per_scale = {
        scale: float(
            np.mean(
                [
                    float(row["deadline_completion_rate"])
                    for row in rows
                    if row["scale"] == scale
                ]
            )
        )
        for scale in sorted({str(row["scale"]) for row in rows})
    }
    per_scale_success = {
        scale: float(
            np.mean(
                [float(row["mission_success"]) for row in rows if row["scale"] == scale]
            )
        )
        for scale in per_scale
    }
    if any(value >= 0.95 for value in per_scale.values()) or any(
        value >= 0.8 for value in per_scale_success.values()
    ):
        raise RuntimeError(
            f"hard-scenario random deadline completion saturated: {per_scale}"
        )
    (root / "random_nonsaturation.json").write_text(
        json.dumps(
            {
                "mean_deadline_completion_rate": rate,
                "per_scale": per_scale,
                "per_scale_mission_success": per_scale_success,
                "threshold": 0.95,
                "mission_success_threshold": 0.8,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    workspace = Path(__file__).resolve().parent
    base = args.output_root / ("smoke" if args.phase == "smoke" else "formal")
    if args.phase in {"smoke", "all"}:
        smoke_root = args.output_root / "smoke"
        run_many(train_commands(args, manifest, smoke_root, True), workspace, args.jobs)
        run_many(evaluation_commands(args, manifest, smoke_root, True), workspace, args.jobs)
        summarize(args, smoke_root, True)
        verify_random_smoke(smoke_root)
        if args.phase == "smoke":
            return
    if args.phase in {"train", "all"}:
        run_many(train_commands(args, manifest, base, False), workspace, args.jobs)
    if args.phase in {"evaluate", "all"}:
        run_many(evaluation_commands(args, manifest, base, False), workspace, args.jobs)
    if args.phase in {"summarize", "all"}:
        summarize(args, base, False)


if __name__ == "__main__":
    main()

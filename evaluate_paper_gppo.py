from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from train_paper_gppo import parse_scale_deadlines, tensor_observation
from run_paper_gppo_v2 import (
    CHECKPOINT_VERSION,
    TRAINING_HYPERPARAMETER_FIELDS,
    checkpoint_protocol_errors,
)
from uav_assignment.gppo_v2 import METHOD_SPECS, config_hash, implementation_hash
from uav_assignment.paper_env import PaperAlignedUAVEnv, PaperEnvConfig
from uav_assignment.paper_models import PaperHeteroActorCritic


def parse_scale(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", maxsplit=1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper-aligned GPPO/PPO")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--baseline", choices=("random", "greedy"))
    parser.add_argument("--config-checkpoint", type=Path)
    parser.add_argument("--scenario-config", type=Path)
    parser.add_argument("--scales", nargs="+", default=("3x12", "4x20", "5x28", "6x36"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=50_000)
    parser.add_argument("--sync-mode", choices=("event", "periodic", "always", "none"))
    parser.add_argument("--mission-deadline", type=float)
    parser.add_argument("--scale-deadlines", nargs="*", default=())
    parser.add_argument("--method-id", choices=tuple(METHOD_SPECS))
    parser.add_argument("--expected-training-seed", type=int)
    parser.add_argument("--expected-updates", type=int)
    parser.add_argument("--expected-episodes-per-update", type=int)
    parser.add_argument("--expected-validation-episodes", type=int)
    parser.add_argument("--expected-validation-interval", type=int)
    parser.add_argument("--expected-validation-seed", type=int)
    parser.add_argument("--expected-hidden-dim", type=int)
    parser.add_argument("--save-event-log", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_baseline(observation: dict[str, np.ndarray], env: PaperAlignedUAVEnv, mode: str, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(observation["action_mask"][:-1])
    if valid.size == 0:
        return env.noop_action
    if mode == "random":
        return int(rng.choice(valid))
    best = int(valid[0])
    best_key = (float("inf"), float("inf"))
    for action in valid:
        uav_index, task_index = divmod(int(action), env.config.max_tasks)
        edge = observation["edge_features"][uav_index, env.config.max_uavs + task_index]
        duration = float(np.arctanh(np.clip(edge[0], -0.999, 0.999)) * 2.0 + np.arctanh(np.clip(edge[1], -0.999, 0.999)) * 2.0)
        key = (env.belief_time + duration, duration)
        if key < best_key:
            best_key = key
            best = int(action)
    return best


def main() -> None:
    args = parse_args()
    scenario_payload = (
        json.loads(args.scenario_config.read_text(encoding="utf-8"))
        if args.scenario_config is not None
        else None
    )
    source = args.checkpoint or args.config_checkpoint
    payload = (
        torch.load(source, map_location="cpu", weights_only=False)
        if source is not None
        else None
    )
    verified_checkpoint_metadata: dict[str, object] | None = None
    if args.checkpoint is not None:
        assert payload is not None
        if payload.get("version") == CHECKPOINT_VERSION:
            missing_context = []
            if scenario_payload is None:
                missing_context.append("--scenario-config")
            if args.method_id is None:
                missing_context.append("--method-id")
            if args.expected_training_seed is None:
                missing_context.append("--expected-training-seed")
            if missing_context:
                raise ValueError(
                    "strict GPPO-v2 checkpoint evaluation requires "
                    + ", ".join(missing_context)
                )
            assert scenario_payload is not None
            assert args.method_id is not None
            assert args.expected_training_seed is not None
            errors = checkpoint_protocol_errors(
                payload,
                args.method_id,
                scenario_payload,
                args.expected_training_seed,
                updates=args.expected_updates,
                episodes_per_update=args.expected_episodes_per_update,
                validation_episodes=args.expected_validation_episodes,
                validation_interval=args.expected_validation_interval,
                validation_seed=args.expected_validation_seed,
                hidden_dim=args.expected_hidden_dim,
            )
            if errors:
                raise ValueError("checkpoint protocol mismatch: " + "; ".join(errors))
            verified_checkpoint_metadata = {
                "checkpoint_version": str(payload["version"]),
                "checkpoint_scenario_hash": str(payload["scenario_hash"]),
                "checkpoint_config_hash": str(
                    payload.get(
                        "config_hash",
                        config_hash(PaperEnvConfig(**payload["env_config"])),
                    )
                ),
                "implementation_hash": str(payload["implementation_hash"]),
                "scenario_version": str(payload["scenario"]),
                "training_seed": int(payload["training"]["seed"]),
                "train_scales": [str(value) for value in payload["train_scales"]],
                "training_updates": int(payload["training"]["updates"]),
                "training_episodes_per_update": int(
                    payload["training"]["episodes_per_update"]
                ),
                "validation_episodes": int(payload["training"]["validation_episodes"]),
                "validation_interval": int(payload["training"]["validation_interval"]),
                "validation_seed": int(payload["training"]["validation_seed"]),
                "hidden_dim": int(payload["training"]["hidden_dim"]),
            }
            verified_checkpoint_metadata.update(
                {
                    f"training_{field}": payload["training"][field]
                    for field in TRAINING_HYPERPARAMETER_FIELDS
                }
            )
        else:
            raise ValueError(
                f"checkpoint version {payload.get('version')!r} is not a verified "
                f"{CHECKPOINT_VERSION!r} artifact"
            )

    if payload is not None:
        base_config = PaperEnvConfig(**payload["env_config"])
        scenario_version = str(payload.get("scenario", "checkpoint-config"))
    elif scenario_payload is not None:
        base_config = PaperEnvConfig(**scenario_payload["scenario"])
        scenario_version = str(scenario_payload.get("version", "scenario-config"))
    else:
        raise SystemExit("baselines require --scenario-config or --config-checkpoint")
    if args.mission_deadline is not None:
        base_config = replace(base_config, mission_deadline=args.mission_deadline)
    model = None
    algorithm = str(args.baseline)
    graph_mode = "none"
    training_seed = -1
    default_sync = "event"
    method_id = args.method_id or f"{args.baseline}_event"
    if args.checkpoint is not None:
        assert payload is not None
        algorithm = str(payload["algorithm"])
        graph_mode = str(payload["graph_mode"])
        training_seed = int(payload["training"]["seed"])
        default_sync = str(payload["sync_mode"])
        method_id = args.method_id or str(
            payload.get("method_id", f"{algorithm}_{default_sync}")
        )
        if args.method_id is not None:
            expected = METHOD_SPECS[args.method_id]
            if (
                payload.get("method_id") != args.method_id
                or algorithm != expected.algorithm
                or graph_mode != expected.graph_mode
                or default_sync != expected.sync_mode
            ):
                raise ValueError("checkpoint metadata does not match --method-id")
        model_config = payload.get(
            "model_config",
            {
                "node_feature_dim": 24,
                "edge_feature_dim": 5,
                "max_uavs": base_config.max_uavs,
                "max_tasks": base_config.max_tasks,
                "hidden_dim": int(payload["training"]["hidden_dim"]),
                "graph_mode": graph_mode,
            },
        )
        model = PaperHeteroActorCritic(
            **model_config,
        )
        state = payload["model_state"]
        model.load_state_dict(state, strict=True)
        model.eval()
    sync_mode = args.sync_mode or default_sync
    scale_deadlines = parse_scale_deadlines(args.scale_deadlines)
    protocol_hash = config_hash(
        {
            "environment": base_config.to_dict(),
            "scale_deadlines": {
                f"{uavs}x{tasks}": deadline
                for (uavs, tasks), deadline in scale_deadlines.items()
            },
        }
    )
    result_implementation_hash = (
        str(verified_checkpoint_metadata["implementation_hash"])
        if verified_checkpoint_metadata is not None
        else implementation_hash()
    )
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for scale_text in args.scales:
        active_uavs, initial_tasks = parse_scale(scale_text)
        if active_uavs > base_config.max_uavs or initial_tasks > base_config.max_tasks:
            raise ValueError(f"scale {scale_text} exceeds checkpoint capacity")
        config = replace(
            base_config,
            active_uavs=active_uavs,
            initial_tasks=initial_tasks,
            mission_deadline=scale_deadlines.get(
                (active_uavs, initial_tasks), base_config.mission_deadline
            ),
        )
        for episode in range(args.episodes):
            episode_seed = args.eval_seed + episode
            env = PaperAlignedUAVEnv(config)
            observation = env.reset(seed=episode_seed)
            rng = np.random.default_rng(episode_seed + 1_000_000)
            done = False
            return_sum = 0.0
            while not done:
                if model is None:
                    action = choose_baseline(observation, env, str(args.baseline), rng)
                else:
                    selected, _, _ = model.act(tensor_observation(observation), deterministic=True)
                    action = int(selected.item())
                observation, reward, done, _ = env.step(action, sync_mode=sync_mode)
                return_sum += reward
            rows.append({
                "method_id": method_id,
                "algorithm": algorithm,
                "graph_mode": graph_mode,
                "training_seed": training_seed,
                "scale": scale_text,
                "active_uavs": active_uavs,
                "initial_tasks": initial_tasks,
                "eval_seed": episode_seed,
                "sync_mode": sync_mode,
                "scenario_hash": protocol_hash,
                "scenario_version": scenario_version,
                "implementation_hash": result_implementation_hash,
                "checkpoint_version": (
                    str(payload.get("version", "unknown"))
                    if payload is not None
                    else "scenario-config"
                ),
                "checkpoint_scenario_hash": (
                    verified_checkpoint_metadata["checkpoint_scenario_hash"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "checkpoint_config_hash": (
                    verified_checkpoint_metadata["checkpoint_config_hash"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "train_scales": (
                    verified_checkpoint_metadata["train_scales"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "training_updates": (
                    verified_checkpoint_metadata["training_updates"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "training_episodes_per_update": (
                    verified_checkpoint_metadata["training_episodes_per_update"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "validation_episodes": (
                    verified_checkpoint_metadata["validation_episodes"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "validation_interval": (
                    verified_checkpoint_metadata["validation_interval"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "validation_seed": (
                    verified_checkpoint_metadata["validation_seed"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                "hidden_dim": (
                    verified_checkpoint_metadata["hidden_dim"]
                    if verified_checkpoint_metadata is not None
                    else None
                ),
                **{
                    f"training_{field}": (
                        verified_checkpoint_metadata[f"training_{field}"]
                        if verified_checkpoint_metadata is not None
                        else None
                    )
                    for field in TRAINING_HYPERPARAMETER_FIELDS
                },
                "return": return_sum,
                **env.metrics(),
            })
            if args.save_event_log:
                event_rows.extend(
                    {
                        "method_id": method_id,
                        "scale": scale_text,
                        "eval_seed": episode_seed,
                        **event,
                    }
                    for event in env.event_log
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (args.output / "evaluation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
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

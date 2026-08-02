from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from uav_assignment.paper_env import PaperAlignedUAVEnv, PaperEnvConfig
from uav_assignment.gppo_v2 import (
    METHOD_SPECS,
    config_hash,
    hard_v2_config,
    implementation_hash,
)
from uav_assignment.paper_models import PaperHeteroActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-aligned GPPO/PPO")
    parser.add_argument("--method-id", choices=tuple(METHOD_SPECS), default=None)
    parser.add_argument("--algorithm", choices=("gppo", "ppo"), default=None)
    parser.add_argument(
        "--graph-mode",
        choices=("none", "staged", "single_head", "adaptive_no_gate", "adaptive"),
        default=None,
    )
    parser.add_argument("--sync-mode", choices=("event", "periodic", "always", "none"), default=None)
    parser.add_argument("--scenario", choices=("legacy", "hard-v2"), default="legacy")
    parser.add_argument("--scenario-config", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-uavs", type=int, default=6)
    parser.add_argument("--active-uavs", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=40)
    parser.add_argument("--initial-tasks", type=int, default=24)
    parser.add_argument(
        "--train-scales",
        nargs="*",
        default=(),
        help="optional rotating active scales such as 3x12 4x20 5x28",
    )
    parser.add_argument(
        "--scale-deadlines",
        nargs="*",
        default=(),
        help="per-scale deadlines such as 3x16=14 3x20=20",
    )
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=160)
    parser.add_argument("--mission-deadline", type=float, default=None)
    parser.add_argument("--task-chain-length", type=int, default=None)
    parser.add_argument("--workload-scale", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--validation-episodes", type=int, default=40)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-seed", type=int, default=40_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_observation(observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "nodes": torch.as_tensor(observation["nodes"], dtype=torch.float32),
        "edge_types": torch.as_tensor(observation["edge_types"], dtype=torch.long),
        "edge_features": torch.as_tensor(observation["edge_features"], dtype=torch.float32),
        "action_mask": torch.as_tensor(observation["action_mask"], dtype=torch.bool),
    }


def parse_scale(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", maxsplit=1)
    return int(left), int(right)


def parse_scale_deadlines(values: tuple[str, ...] | list[str]) -> dict[tuple[int, int], float]:
    result: dict[tuple[int, int], float] = {}
    for value in values:
        scale, deadline = value.split("=", maxsplit=1)
        result[parse_scale(scale)] = float(deadline)
    return result


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
    bootstrap_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    next_value = bootstrap_value
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def collect_rollouts(
    model: PaperHeteroActorCritic,
    config: PaperEnvConfig,
    episodes: int,
    seed_offset: int,
    sync_mode: str,
    gamma: float,
    gae_lambda: float,
    scale_choices: tuple[tuple[int, int], ...],
    scale_deadlines: dict[tuple[int, int], float] | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    scale_deadlines = scale_deadlines or {}
    storage: dict[str, list[np.ndarray | int | float]] = {
        "nodes": [], "edge_types": [], "edge_features": [], "action_mask": [],
        "actions": [], "old_log_probs": [], "returns": [], "advantages": [],
    }
    metrics: list[dict[str, object]] = []
    model.eval()
    for episode in range(episodes):
        episode_config = config
        if scale_choices:
            active_uavs, initial_tasks = scale_choices[(seed_offset + episode) % len(scale_choices)]
            episode_config = replace(
                config,
                active_uavs=active_uavs,
                initial_tasks=initial_tasks,
                mission_deadline=scale_deadlines.get(
                    (active_uavs, initial_tasks), config.mission_deadline
                ),
            )
        env = PaperAlignedUAVEnv(episode_config)
        observation = env.reset(seed=100_000 * config.seed + seed_offset + episode)
        done = False
        final_info: dict[str, object] = {}
        states: list[dict[str, np.ndarray]] = []
        actions: list[int] = []
        log_probs: list[float] = []
        rewards: list[float] = []
        values: list[float] = []
        while not done:
            state = {key: observation[key] for key in ("nodes", "edge_types", "edge_features", "action_mask")}
            with torch.no_grad():
                action, log_prob, value = model.act(tensor_observation(observation))
            next_observation, reward, done, info = env.step(
                int(action.item()), sync_mode=sync_mode
            )
            final_info = info
            states.append(state)
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            rewards.append(float(reward))
            observation = next_observation
        bootstrap_value = 0.0
        if final_info.get("termination_reason") == "max_decisions":
            with torch.no_grad():
                _, next_value = model(
                    **tensor_observation(next_observation)
                )
            bootstrap_value = float(next_value.item())
        advantages, returns = gae(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            gamma,
            gae_lambda,
            bootstrap_value,
        )
        for index, state in enumerate(states):
            for key in ("nodes", "edge_types", "edge_features", "action_mask"):
                storage[key].append(state[key])
            storage["actions"].append(actions[index])
            storage["old_log_probs"].append(log_probs[index])
            storage["advantages"].append(float(advantages[index]))
            storage["returns"].append(float(returns[index]))
        metrics.append(env.metrics())
    batch = {
        "nodes": torch.as_tensor(np.asarray(storage["nodes"]), dtype=torch.float32),
        "edge_types": torch.as_tensor(np.asarray(storage["edge_types"]), dtype=torch.long),
        "edge_features": torch.as_tensor(np.asarray(storage["edge_features"]), dtype=torch.float32),
        "action_mask": torch.as_tensor(np.asarray(storage["action_mask"]), dtype=torch.bool),
        "actions": torch.as_tensor(np.asarray(storage["actions"]), dtype=torch.long),
        "old_log_probs": torch.as_tensor(np.asarray(storage["old_log_probs"]), dtype=torch.float32),
        "advantages": torch.as_tensor(np.asarray(storage["advantages"]), dtype=torch.float32),
        "returns": torch.as_tensor(np.asarray(storage["returns"]), dtype=torch.float32),
    }
    return batch, metrics


@torch.no_grad()
def evaluate_validation(
    model: PaperHeteroActorCritic,
    config: PaperEnvConfig,
    episodes: int,
    seed: int,
    sync_mode: str,
    scale_choices: tuple[tuple[int, int], ...],
    scale_deadlines: dict[tuple[int, int], float] | None = None,
) -> dict[str, float]:
    model.eval()
    scale_deadlines = scale_deadlines or {}
    metrics: list[dict[str, object]] = []
    validation_scales = scale_choices or ((config.active_uavs, config.initial_tasks),)
    for active_uavs, initial_tasks in validation_scales:
        scale_config = replace(
            config,
            active_uavs=active_uavs,
            initial_tasks=initial_tasks,
            mission_deadline=scale_deadlines.get(
                (active_uavs, initial_tasks), config.mission_deadline
            ),
        )
        for episode in range(episodes):
            env = PaperAlignedUAVEnv(scale_config)
            observation = env.reset(seed=seed + episode)
            done = False
            while not done:
                action, _, _ = model.act(tensor_observation(observation), deterministic=True)
                observation, _, done, _ = env.step(int(action.item()), sync_mode=sync_mode)
            metrics.append(env.metrics())
    metric_names = (
        "completed_total",
        "completion_rate",
        "makespan",
        "deadline_completed_total",
        "deadline_completion_rate",
        "mission_success",
        "deadline_remaining_tasks",
        "throughput",
        "communication_events",
        "heartbeat_messages",
        "invalid_actions",
        "reallocation_success_rate",
    )
    return {
        key: float(np.mean([float(row[key]) for row in metrics]))
        for key in metric_names
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    method = METHOD_SPECS.get(args.method_id) if args.method_id else None
    algorithm = args.algorithm or (method.algorithm if method else "gppo")
    graph_mode = args.graph_mode or (
        method.graph_mode if method else ("adaptive" if algorithm == "gppo" else "none")
    )
    sync_mode = args.sync_mode or (
        method.sync_mode if method else ("event" if algorithm == "gppo" else "none")
    )
    if method and (
        algorithm != method.algorithm
        or graph_mode != method.graph_mode
        or sync_mode != method.sync_mode
    ):
        raise ValueError(f"method configuration does not match {method.method_id}")
    if algorithm == "ppo" and graph_mode != "none":
        raise ValueError("PPO controls must use graph_mode=none")
    train_scales = tuple(parse_scale(value) for value in args.train_scales)
    scale_deadlines = parse_scale_deadlines(args.scale_deadlines)
    config_values = {
        "max_uavs": args.max_uavs,
        "max_tasks": args.max_tasks,
        "active_uavs": args.active_uavs,
        "initial_tasks": args.initial_tasks,
        "max_decisions": args.max_decisions,
        "seed": args.seed,
    }
    scenario_name = args.scenario
    if args.scenario_config is not None:
        scenario_payload = json.loads(args.scenario_config.read_text(encoding="utf-8"))
        scenario_name = str(scenario_payload.get("version", "scenario-config"))
        config = PaperEnvConfig(**scenario_payload["scenario"])
        config = replace(config, seed=args.seed)
    elif args.scenario == "hard-v2":
        config = hard_v2_config(**config_values)
    else:
        config = PaperEnvConfig(**config_values, include_engineering_rewards=False)
    if args.mission_deadline is not None:
        config = replace(config, mission_deadline=args.mission_deadline)
    if args.task_chain_length is not None:
        config = replace(config, task_chain_length=args.task_chain_length)
    if args.workload_scale is not None:
        config = replace(config, workload_scale=args.workload_scale)
    for active_uavs, initial_tasks in train_scales:
        if active_uavs > config.max_uavs or initial_tasks > config.max_tasks:
            raise ValueError("train scale exceeds fixed model capacity")
    env = PaperAlignedUAVEnv(config)
    env.reset()
    model = PaperHeteroActorCritic(
        env.node_feature_dim,
        env.edge_feature_dim,
        config.max_uavs,
        config.max_tasks,
        hidden_dim=args.hidden_dim,
        graph_mode=graph_mode,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    best_update = args.updates
    for update in range(args.updates):
        batch, metrics = collect_rollouts(
            model, config, args.episodes_per_update, update * args.episodes_per_update,
            sync_mode, args.gamma, args.gae_lambda, train_scales, scale_deadlines,
        )
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        batch["advantages"] = advantages
        count = len(advantages)
        losses: list[float] = []
        model.train()
        for _ in range(args.update_epochs):
            permutation = torch.randperm(count)
            for start in range(0, count, args.minibatch_size):
                indices = permutation[start : start + args.minibatch_size]
                distribution, values = model(
                    batch["nodes"][indices], batch["edge_types"][indices],
                    batch["edge_features"][indices], batch["action_mask"][indices],
                )
                log_probs = distribution.log_prob(batch["actions"][indices])
                ratios = torch.exp(log_probs - batch["old_log_probs"][indices])
                unclipped = ratios * batch["advantages"][indices]
                clipped = torch.clamp(ratios, 0.8, 1.2) * batch["advantages"][indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values.squeeze(-1), batch["returns"][indices])
                entropy = distribution.entropy().mean()
                loss = actor_loss + 0.5 * value_loss - args.entropy_coefficient * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(float(actor_loss.detach()))
        record = {
            "update": float(update + 1),
            "completed_total": float(np.mean([float(row["completed_total"]) for row in metrics])),
            "completion_rate": float(np.mean([float(row["completion_rate"]) for row in metrics])),
            "makespan": float(np.mean([float(row["makespan"]) for row in metrics])),
            "deadline_completion_rate": float(np.mean([float(row["deadline_completion_rate"]) for row in metrics])),
            "mission_success": float(np.mean([float(row["mission_success"]) for row in metrics])),
            "deadline_remaining_tasks": float(np.mean([float(row["deadline_remaining_tasks"]) for row in metrics])),
            "throughput": float(np.mean([float(row["throughput"]) for row in metrics])),
            "communication_events": float(np.mean([float(row["communication_events"]) for row in metrics])),
            "actor_loss": float(np.mean(losses)),
        }
        history.append(record)
        (args.output / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if args.validation_episodes and ((update + 1) % args.validation_interval == 0 or update + 1 == args.updates):
            validation = evaluate_validation(
                model,
                config,
                args.validation_episodes,
                args.validation_seed,
                sync_mode,
                train_scales,
                scale_deadlines,
            )
            validation["update"] = float(update + 1)
            validation_history.append(validation)
            (args.output / "validation_history.json").write_text(
                json.dumps(validation_history, indent=2), encoding="utf-8"
            )
            score = (
                validation["mission_success"],
                validation["deadline_completion_rate"],
                -validation["makespan"],
            )
            if score > best_score:
                best_score = score
                best_update = update + 1
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(), "algorithm": algorithm,
        "method_id": args.method_id or f"{algorithm}_{sync_mode}",
        "graph_mode": graph_mode, "sync_mode": sync_mode,
        "version": "paper-aligned-gppo-v2", "env_config": config.to_dict(),
        "scenario": scenario_name,
        "scenario_hash": config_hash(config),
        "config_hash": config_hash(config),
        "implementation_hash": implementation_hash(),
        "model_config": {
            "node_feature_dim": env.node_feature_dim,
            "edge_feature_dim": env.edge_feature_dim,
            "max_uavs": config.max_uavs,
            "max_tasks": config.max_tasks,
            "hidden_dim": args.hidden_dim,
            "graph_mode": graph_mode,
        },
        "training": vars(args), "history": history,
        "validation_history": validation_history, "best_update": best_update,
        "train_scales": list(args.train_scales),
        "scale_deadlines": {
            f"{uavs}x{tasks}": deadline
            for (uavs, tasks), deadline in scale_deadlines.items()
        },
    }, args.output / "checkpoint.pt")
    (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output / "validation_history.json").write_text(json.dumps(validation_history, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output / "checkpoint.pt"), "best_update": best_update}, ensure_ascii=False))


if __name__ == "__main__":
    main()

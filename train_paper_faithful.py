from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from torch import nn

from uav_assignment.paper_faithful_env import (
    PAPER_SCALES,
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
    PaperScale,
    deterministic_instance_seeds,
)
from uav_assignment.paper_faithful_models import PaperFaithfulActorCritic


def parse_scale(value: str) -> PaperScale:
    normalized = value.upper().removeprefix("T")
    uavs, parents, subtasks = (int(part) for part in normalized.split("-"))
    scale = PaperScale(uavs, parents, subtasks)
    if scale not in PAPER_SCALES:
        raise argparse.ArgumentTypeError(f"not a paper scale: {value}")
    return scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Literal-AHGNN under the paper-faithful protocol")
    parser.add_argument("--mode", choices=sorted(PaperFaithfulActorCritic.MODES), required=True)
    parser.add_argument("--sync-mode", choices=("none", "event", "periodic", "always"), default="event")
    parser.add_argument("--scale", type=parse_scale, default=PAPER_SCALES[0])
    parser.add_argument("--model-max-uavs", type=int)
    parser.add_argument("--model-max-subtasks", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rrelu-mode", choices=("expected", "stochastic"), default="expected")
    parser.add_argument("--gate-bias-init", type=float, default=0.0)
    parser.add_argument(
        "--gate-activation",
        choices=("sigmoid", "softplus"),
        default="sigmoid",
        help="Output nonlinearity for f_{i,j,k}; sigmoid is the frozen formal default.",
    )
    parser.add_argument(
        "--gate-scope",
        choices=("task_message", "score", "aggregate"),
        default="task_message",
        help="Interpretation of Eq.(3) adaptive f; task_message is the frozen formal default.",
    )
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-instances", type=int, default=20)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensors(observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "nodes": torch.as_tensor(observation["nodes"], dtype=torch.float32),
        "edge_types": torch.as_tensor(observation["edge_types"], dtype=torch.long),
        "edge_features": torch.as_tensor(observation["edge_features"], dtype=torch.float32),
        "action_mask": torch.as_tensor(observation["action_mask"], dtype=torch.bool),
    }


def gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def collect(
    model: PaperFaithfulActorCritic,
    config: PaperFaithfulConfig,
    sync_mode: str,
    minimum_steps: int,
    episode_offset: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], int]:
    keys = ("nodes", "edge_types", "edge_features", "action_mask")
    storage: dict[str, list[object]] = {key: [] for key in keys}
    storage.update({"actions": [], "old_log_probs": [], "advantages": [], "returns": []})
    metrics: list[dict[str, object]] = []
    instance_seeds = deterministic_instance_seeds(config.scale)
    # The stochastic variant deliberately keeps RReLU in training mode during
    # rollout; this is a diagnostic of the paper's training-time randomness.
    # Expected-RReLU remains deterministic for valid PPO old/new ratios.
    model.eval()
    if getattr(model, "rrelu_mode", "expected") == "stochastic":
        model.train()
    episode = 0
    while len(storage["actions"]) < minimum_steps:
        instance_seed = instance_seeds[(episode_offset + episode) % len(instance_seeds)]
        env = PaperFaithfulUAVEnv(config)
        observation = env.reset(seed=instance_seed)
        states: list[dict[str, np.ndarray]] = []
        actions: list[int] = []
        log_probs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        done = False
        while not done:
            state = {key: observation[key] for key in keys}
            with torch.no_grad():
                action, log_prob, value = model.act(tensors(observation))
            observation, reward, done, _ = env.step(int(action.item()), sync_mode=sync_mode)
            states.append(state)
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            rewards.append(float(reward))
        advantage, returns = gae(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            gamma,
            gae_lambda,
        )
        for index, state in enumerate(states):
            for key in keys:
                storage[key].append(state[key])
            storage["actions"].append(actions[index])
            storage["old_log_probs"].append(log_probs[index])
            storage["advantages"].append(float(advantage[index]))
            storage["returns"].append(float(returns[index]))
        episode_metrics = env.metrics()
        episode_metrics["episode_return"] = float(np.sum(rewards))
        metrics.append(episode_metrics)
        episode += 1
    batch = {
        "nodes": torch.as_tensor(np.asarray(storage["nodes"]), dtype=torch.float32),
        "edge_types": torch.as_tensor(np.asarray(storage["edge_types"]), dtype=torch.long),
        "edge_features": torch.as_tensor(np.asarray(storage["edge_features"]), dtype=torch.float32),
        "action_mask": torch.as_tensor(np.asarray(storage["action_mask"]), dtype=torch.bool),
        "actions": torch.as_tensor(storage["actions"], dtype=torch.long),
        "old_log_probs": torch.as_tensor(storage["old_log_probs"], dtype=torch.float32),
        "advantages": torch.as_tensor(storage["advantages"], dtype=torch.float32),
        "returns": torch.as_tensor(storage["returns"], dtype=torch.float32),
    }
    return batch, metrics, episode


@torch.no_grad()
def validate(model: PaperFaithfulActorCritic, config: PaperFaithfulConfig, sync_mode: str, count: int) -> dict[str, float]:
    model.eval()
    bank = deterministic_instance_seeds(config.scale, split="validation")
    rows: list[dict[str, object]] = []
    for instance_seed in bank[-count:]:
        env = PaperFaithfulUAVEnv(config)
        observation = env.reset(seed=instance_seed)
        done = False
        episode_return = 0.0
        while not done:
            action, _, _ = model.act(tensors(observation), deterministic=True)
            observation, reward, done, _ = env.step(int(action.item()), sync_mode=sync_mode)
            episode_return += float(reward)
        row = env.metrics()
        row["episode_return"] = episode_return
        rows.append(row)
    return {
        "reward": float(np.mean([float(row["episode_return"]) for row in rows])),
        "realized_makespan": float(np.mean([float(row["realized_makespan"]) for row in rows])),
        "completion_rate": float(np.mean([float(row["completion_rate"]) for row in rows])),
        "communication_events": float(np.mean([float(row["communication_events"]) for row in rows])),
    }


def _history_through(path: Path, iteration: int) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in rows if int(float(row["iteration"])) <= iteration]


def recover_candidate(
    candidate_path: Path,
    output: Path,
    model: PaperFaithfulActorCritic,
) -> dict[str, object]:
    """Recover a legacy candidate that predates versioned optimizer/RNG resumes.

    This is deliberately marked as a discontinuous recovery.  It preserves the
    trained weights and completed histories, but cannot reconstruct Adam moments,
    rollout episode offsets, or random-number generator states.
    """
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if candidate.get("version") is not None or not {
        "iteration", "model_state", "validation"
    }.issubset(candidate):
        raise ValueError(f"unsupported candidate checkpoint: {candidate_path}")
    source_iteration = int(candidate["iteration"])
    model.load_state_dict(candidate["model_state"], strict=True)
    history = _history_through(output / "training_history.json", source_iteration)
    validation_history = _history_through(
        output / "validation_history.json", source_iteration
    )

    eligible: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    for path in output.glob("candidate_*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        iteration = int(payload.get("iteration", -1))
        validation = payload.get("validation", {})
        if iteration <= source_iteration and "realized_makespan" in validation:
            eligible.append(
                (
                    float(validation["realized_makespan"]),
                    iteration,
                    payload["model_state"],
                )
            )
    if not eligible:
        raise ValueError(f"no valid candidate checkpoints through iteration {source_iteration}")
    best_makespan, best_iteration, best_state = min(
        eligible, key=lambda item: (item[0], item[1])
    )
    return {
        "iteration": source_iteration,
        "history": history,
        "validation_history": validation_history,
        "best_state": copy.deepcopy(best_state),
        "best_makespan": best_makespan,
        "best_iteration": best_iteration,
        "episode_offset": 0,
        "recovery_info": {
            "version": "candidate-recovery-v1",
            "source": str(candidate_path.resolve()),
            "source_iteration": source_iteration,
            "optimizer_state_recovered": False,
            "rng_state_recovered": False,
            "episode_offset_recovered": False,
            "history_trimmed_to_candidate": True,
            "discarded_training_rows": max(0, len(json.loads((output / "training_history.json").read_text(encoding="utf-8"))) - len(history)) if (output / "training_history.json").exists() else 0,
            "reason": "legacy formal process stopped before writing a versioned resume checkpoint",
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(os.environ.get("PAPER_TORCH_THREADS", "1"))))
    # The paper trains each scale independently; using exact capacity avoids
    # padded nodes/actions changing the optimization problem across scenes.
    model_max_uavs = args.model_max_uavs or args.scale.uavs
    model_max_subtasks = args.model_max_subtasks or args.scale.subtasks
    config = PaperFaithfulConfig(
        scale=args.scale,
        max_uavs=model_max_uavs,
        max_subtasks=model_max_subtasks,
    )
    probe = PaperFaithfulUAVEnv(config)
    observation = probe.reset(seed=deterministic_instance_seeds(args.scale)[0])
    model = PaperFaithfulActorCritic(
        observation["nodes"].shape[-1],
        observation["edge_features"].shape[-1],
        config.max_uavs,
        config.max_subtasks,
        hidden_dim=args.hidden_dim,
        graph_mode=args.mode,
        rrelu_mode=args.rrelu_mode,
        gate_bias_init=args.gate_bias_init,
        gate_scope=args.gate_scope,
        gate_activation=args.gate_activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    validation_history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_makespan = float("inf")
    best_iteration = 0
    episode_offset = 0
    start_iteration = 1
    recovery_info: dict[str, object] | None = None
    resume_events: list[dict[str, object]] = []
    if args.resume_from:
        resume = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if resume.get("version") == "paper-faithful-resume-v1":
            model.load_state_dict(resume["model_state"])
            optimizer.load_state_dict(resume["optimizer_state"])
            history = list(resume["history"])
            validation_history = list(resume["validation_history"])
            best_state = resume["best_state"]
            best_makespan = float(resume["best_makespan"])
            best_iteration = int(resume["best_iteration"])
            episode_offset = int(resume["episode_offset"])
            random.setstate(resume["python_random_state"])
            np.random.set_state(resume["numpy_random_state"])
            torch.set_rng_state(resume["torch_random_state"])
            recovery_info = resume.get("recovery_info")
            resume_events = list(resume.get("resume_events", []))
            resume_events.append(
                {
                    "version": "exact-resume-event-v1",
                    "source": str(args.resume_from.resolve()),
                    "source_iteration": int(resume["iteration"]),
                    "optimizer_state_recovered": True,
                    "rng_state_recovered": True,
                    "episode_offset_recovered": True,
                }
            )
            start_iteration = int(resume["iteration"]) + 1
        else:
            recovered = recover_candidate(args.resume_from, args.output, model)
            history = list(recovered["history"])
            validation_history = list(recovered["validation_history"])
            best_state = recovered["best_state"]
            best_makespan = float(recovered["best_makespan"])
            best_iteration = int(recovered["best_iteration"])
            episode_offset = int(recovered["episode_offset"])
            recovery_info = recovered["recovery_info"]
            start_iteration = int(recovered["iteration"]) + 1
            (args.output / "training_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            (args.output / "validation_history.json").write_text(
                json.dumps(validation_history, indent=2), encoding="utf-8"
            )
    for iteration in range(start_iteration, args.iterations + 1):
        batch, rows, episodes = collect(
            model, config, args.sync_mode, args.rollout_steps, episode_offset,
            args.gamma, args.gae_lambda,
        )
        episode_offset += episodes
        batch["advantages"] = (
            batch["advantages"] - batch["advantages"].mean()
        ) / batch["advantages"].std().clamp_min(1e-6)
        model.train()
        losses: list[float] = []
        for _ in range(args.update_epochs):
            permutation = torch.randperm(len(batch["actions"]))
            for start in range(0, len(batch["actions"]), args.batch_size):
                indices = permutation[start:start + args.batch_size]
                distribution, value = model(
                    batch["nodes"][indices], batch["edge_types"][indices],
                    batch["edge_features"][indices], batch["action_mask"][indices],
                )
                log_prob = distribution.log_prob(batch["actions"][indices])
                ratio = torch.exp(log_prob - batch["old_log_probs"][indices])
                unclipped = ratio * batch["advantages"][indices]
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * batch["advantages"][indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(value.squeeze(-1), batch["returns"][indices])
                loss = actor_loss + args.value_coefficient * value_loss - args.entropy_coefficient * distribution.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
        record = {
            "iteration": float(iteration),
            "reward": float(np.mean([float(row["episode_return"]) for row in rows])),
            "realized_makespan": float(np.mean([float(row["realized_makespan"]) for row in rows])),
            "loss": float(np.mean(losses)),
        }
        history.append(record)
        if iteration % args.validation_interval == 0 or iteration == args.iterations:
            validation = validate(model, config, args.sync_mode, args.validation_instances)
            validation["iteration"] = float(iteration)
            validation_history.append(validation)
            torch.save(
                {
                    "iteration": iteration,
                    "model_state": model.state_dict(),
                    "validation": validation,
                },
                args.output / f"candidate_{iteration:04d}.pt",
            )
            if validation["realized_makespan"] < best_makespan:
                best_makespan = validation["realized_makespan"]
                best_iteration = iteration
                best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "version": "paper-faithful-resume-v1",
                    "iteration": iteration,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "validation_history": validation_history,
                    "best_state": best_state,
                    "best_makespan": best_makespan,
                    "best_iteration": best_iteration,
                    "episode_offset": episode_offset,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "recovery_info": recovery_info,
                    "resume_events": resume_events,
                },
                args.output / "resume_latest.pt",
            )
        (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (args.output / "validation_history.json").write_text(json.dumps(validation_history, indent=2), encoding="utf-8")
    model.load_state_dict(best_state)
    checkpoint = {
        "version": "paper-faithful-literal-v1",
        "model_state": model.state_dict(),
        "model_config": {
            "node_feature_dim": observation["nodes"].shape[-1],
            "edge_feature_dim": observation["edge_features"].shape[-1],
            "max_uavs": config.max_uavs,
            "max_tasks": config.max_subtasks,
            "hidden_dim": args.hidden_dim,
            "graph_mode": args.mode,
            "rrelu_mode": args.rrelu_mode,
            "gate_bias_init": args.gate_bias_init,
            "gate_scope": args.gate_scope,
            "gate_activation": args.gate_activation,
        },
        "environment": config.to_dict(),
        "training": {key: (str(value) if isinstance(value, Path) else value.name if isinstance(value, PaperScale) else value) for key, value in vars(args).items()},
        "active_parameter_count": model.active_parameter_count(),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_iteration": best_iteration,
        "history": history,
        "validation_history": validation_history,
        "recovery_info": recovery_info,
        "resume_events": resume_events,
    }
    torch.save(checkpoint, args.output / "checkpoint.pt")
    print(json.dumps({"saved": str(args.output / "checkpoint.pt"), "best_iteration": best_iteration, "best_makespan": best_makespan}, ensure_ascii=False))


if __name__ == "__main__":
    main()

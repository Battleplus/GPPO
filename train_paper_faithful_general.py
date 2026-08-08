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

from train_paper_faithful import collect, tensors, validate
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
    values = tuple(int(part) for part in normalized.split("-"))
    return PaperScale(*values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one fixed-capacity cross-scale paper-faithful model")
    parser.add_argument("--mode", choices=sorted(PaperFaithfulActorCritic.MODES), default="literal")
    parser.add_argument("--sync-mode", choices=("none", "event", "periodic", "always"), default="event")
    parser.add_argument("--train-scale", type=parse_scale, action="append")
    parser.add_argument("--model-max-uavs", type=int, default=20)
    parser.add_argument("--model-max-subtasks", type=int, default=92)
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
    parser.add_argument("--gate-activation", choices=("sigmoid", "softplus"), default="sigmoid")
    parser.add_argument("--gate-scope", choices=("task_message", "score", "aggregate"), default="task_message")
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-instances", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_scales = list(args.train_scale or PAPER_SCALES)
    for scale in training_scales:
        if scale.uavs > args.model_max_uavs or scale.subtasks > args.model_max_subtasks:
            raise ValueError(f"training scale exceeds fixed capacity: {scale.name}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, int(os.environ.get("PAPER_TORCH_THREADS", "1"))))
    configs = [
        PaperFaithfulConfig(
            scale=scale,
            max_uavs=args.model_max_uavs,
            max_subtasks=args.model_max_subtasks,
        )
        for scale in training_scales
    ]
    probe = PaperFaithfulUAVEnv(configs[0])
    observation = probe.reset(seed=deterministic_instance_seeds(configs[0].scale)[0])
    model = PaperFaithfulActorCritic(
        observation["nodes"].shape[-1],
        observation["edge_features"].shape[-1],
        args.model_max_uavs,
        args.model_max_subtasks,
        hidden_dim=args.hidden_dim,
        graph_mode=args.mode,
        rrelu_mode=args.rrelu_mode,
        gate_bias_init=args.gate_bias_init,
        gate_scope=args.gate_scope,
        gate_activation=args.gate_activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | str]] = []
    validation_history: list[dict[str, float | str]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_makespan = float("inf")
    best_iteration = 0
    episode_offset = 0
    for iteration in range(1, args.iterations + 1):
        config = configs[(iteration - 1) % len(configs)]
        batch, rows, episodes = collect(
            model, config, args.sync_mode, args.rollout_steps, episode_offset,
            args.gamma, args.gae_lambda,
        )
        episode_offset += episodes
        batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / batch["advantages"].std().clamp_min(1e-6)
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
                ratio = torch.exp(distribution.log_prob(batch["actions"][indices]) - batch["old_log_probs"][indices])
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
        history.append({
            "iteration": float(iteration),
            "training_scale": config.scale.name,
            "reward": float(np.mean([float(row["episode_return"]) for row in rows])),
            "realized_makespan": float(np.mean([float(row["realized_makespan"]) for row in rows])),
            "loss": float(np.mean(losses)),
        })
        if iteration % args.validation_interval == 0 or iteration == args.iterations:
            validation_rows = [
                validate(model, cfg, args.sync_mode, args.validation_instances)
                for cfg in configs
            ]
            validation = {
                "iteration": float(iteration),
                "realized_makespan": float(np.mean([row["realized_makespan"] for row in validation_rows])),
                "reward": float(np.mean([row["reward"] for row in validation_rows])),
                "completion_rate": float(np.mean([row["completion_rate"] for row in validation_rows])),
                "communication_events": float(np.mean([row["communication_events"] for row in validation_rows])),
            }
            validation_history.append(validation)
            torch.save({"iteration": iteration, "model_state": model.state_dict(), "validation": validation}, args.output / f"candidate_{iteration:04d}.pt")
            if validation["realized_makespan"] < best_makespan:
                best_makespan = validation["realized_makespan"]
                best_iteration = iteration
                best_state = copy.deepcopy(model.state_dict())
        (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (args.output / "validation_history.json").write_text(json.dumps(validation_history, indent=2), encoding="utf-8")
    model.load_state_dict(best_state)
    checkpoint = {
        "version": "paper-faithful-general-v1",
        "model_state": model.state_dict(),
        "model_config": {
            "node_feature_dim": observation["nodes"].shape[-1],
            "edge_feature_dim": observation["edge_features"].shape[-1],
            "max_uavs": args.model_max_uavs,
            "max_tasks": args.model_max_subtasks,
            "hidden_dim": args.hidden_dim,
            "graph_mode": args.mode,
            "rrelu_mode": args.rrelu_mode,
            "gate_bias_init": args.gate_bias_init,
            "gate_scope": args.gate_scope,
            "gate_activation": args.gate_activation,
        },
        "environment": configs[0].to_dict(),
        "training": {key: (str(value) if isinstance(value, Path) else value.name if isinstance(value, PaperScale) else value) for key, value in vars(args).items()},
        "training_scales": [scale.name for scale in training_scales],
        "active_parameter_count": model.active_parameter_count(),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_iteration": best_iteration,
        "history": history,
        "validation_history": validation_history,
    }
    torch.save(checkpoint, args.output / "checkpoint.pt")
    print(json.dumps({"saved": str(args.output / "checkpoint.pt"), "best_iteration": best_iteration, "best_makespan": best_makespan}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_paper_faithful import collect, tensors
from uav_assignment.paper_env import UAV_TASK_EDGE
from uav_assignment.paper_faithful_env import (
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
    PaperScale,
    deterministic_instance_seeds,
)
from uav_assignment.paper_faithful_models import PaperFaithfulActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Literal-AHGNN gate behavior")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "validation_a", "validation_b", "test"),
        default="test",
    )
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "fraction_gt_0_9": float(np.mean(array > 0.9)),
        "fraction_lt_0_1": float(np.mean(array < 0.1)),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["model_config"]["graph_mode"] != "literal":
        raise ValueError("gate diagnostics require a literal checkpoint")
    scale_payload = checkpoint["environment"]["scale"]
    scale = PaperScale(
        int(scale_payload["uavs"]),
        int(scale_payload["parent_tasks"]),
        int(scale_payload["subtasks"]),
    )
    config = PaperFaithfulConfig(
        scale=scale,
        max_uavs=int(checkpoint["environment"]["max_uavs"]),
        max_subtasks=int(checkpoint["environment"]["max_subtasks"]),
    )
    model = PaperFaithfulActorCritic(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    gates: list[float] = []
    attention: list[float] = []
    removed_attention_mass: list[float] = []
    task_attention_mass: list[float] = []
    uav_feature_l2_norms: list[float] = []
    first_observation: dict[str, np.ndarray] | None = None
    sync_mode = str(checkpoint["training"]["sync_mode"])
    for instance_seed in deterministic_instance_seeds(
        scale, args.instances, split=args.split
    ):
        env = PaperFaithfulUAVEnv(config)
        observation = env.reset(seed=instance_seed)
        if first_observation is None:
            first_observation = observation
        done = False
        while not done:
            observation_tensors = tensors(observation)
            with torch.no_grad():
                distribution, _ = model(**observation_tensors)
                action = torch.argmax(distribution.logits, dim=-1)
            learned_gates = model.literal_attention.last_gates[0]
            alpha = model.literal_attention.last_attention[0]
            edge_types = observation_tensors["edge_types"][: config.max_uavs]
            task_mask = edge_types == UAV_TASK_EDGE
            task_mask[:, : config.max_uavs] = False
            gates.extend(learned_gates[task_mask].tolist())
            attention.extend(alpha[task_mask].tolist())
            removed_attention_mass.extend(
                torch.sum(alpha * task_mask * (1.0 - learned_gates), dim=-1).tolist()
            )
            task_attention_mass.extend(
                torch.sum(alpha * task_mask, dim=-1).tolist()
            )
            uav_feature_l2_norms.extend(
                torch.linalg.vector_norm(
                    model.literal_attention.last_uav_features[0], dim=-1
                ).tolist()
            )
            observation, _, done, _ = env.step(int(action.item()), sync_mode=sync_mode)

    assert first_observation is not None
    model.train()
    observation_tensors = tensors(first_observation)
    distribution, _ = model(**observation_tensors)
    selected = torch.argmax(distribution.logits, dim=-1)
    sensitivity_loss = -distribution.log_prob(selected).mean()
    model.zero_grad(set_to_none=True)
    sensitivity_loss.backward()
    gradient_norms = {
        name: float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
        if name.startswith("literal_attention.gate.") and parameter.grad is not None
    }
    gate_gradient_l2 = math.sqrt(sum(value * value for value in gradient_norms.values()))
    # A second probe follows the actual PPO loss path on one fresh rollout
    # batch.  This is stronger than the local policy-sensitivity gradient above
    # while remaining deliberately small and reproducible.
    model.zero_grad(set_to_none=True)
    config_probe = PaperFaithfulConfig(
        scale=scale,
        max_uavs=int(checkpoint["environment"]["max_uavs"]),
        max_subtasks=int(checkpoint["environment"]["max_subtasks"]),
    )
    training = checkpoint.get("training", {})
    batch, _, _ = collect(
        model,
        config_probe,
        sync_mode,
        minimum_steps=512,
        episode_offset=0,
        gamma=float(training.get("gamma", 0.99)),
        gae_lambda=float(training.get("gae_lambda", 0.95)),
        device=torch.device("cpu"),
    )
    batch["advantages"] = (
        batch["advantages"] - batch["advantages"].mean()
    ) / batch["advantages"].std().clamp_min(1e-6)
    model.train()
    distribution, value = model(
        batch["nodes"], batch["edge_types"], batch["edge_features"], batch["action_mask"]
    )
    ratio = torch.exp(distribution.log_prob(batch["actions"]) - batch["old_log_probs"])
    unclipped = ratio * batch["advantages"]
    clipped = torch.clamp(ratio, 0.8, 1.2) * batch["advantages"]
    ppo_loss = (
        -torch.minimum(unclipped, clipped).mean()
        + 0.5 * torch.nn.functional.mse_loss(value.squeeze(-1), batch["returns"])
        - 0.01 * distribution.entropy().mean()
    )
    model.zero_grad(set_to_none=True)
    ppo_loss.backward()
    ppo_gradient_norms = {
        name: float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
        if name.startswith("literal_attention.gate.") and parameter.grad is not None
    }
    ppo_gate_gradient_l2 = math.sqrt(sum(value * value for value in ppo_gradient_norms.values()))
    payload = {
        "version": "paper-faithful-gate-diagnostic-v1",
        "checkpoint": str(args.checkpoint),
        "scale": scale.name,
        "split": args.split,
        "instances": args.instances,
        "best_iteration": checkpoint.get("best_iteration"),
        "gate": stats(gates),
        "attention_on_task_edges": stats(attention),
        "task_attention_mass_per_uav": stats(task_attention_mass),
        "removed_attention_mass_per_uav": stats(removed_attention_mass),
        "uav_attention_output_l2_norm": stats(uav_feature_l2_norms),
        "gate_gradient_l2_policy_sensitivity": gate_gradient_l2,
        "gate_gradient_norms": gradient_norms,
        "gate_gradient_l2_actual_ppo_probe": ppo_gate_gradient_l2,
        "gate_gradient_norms_actual_ppo_probe": ppo_gradient_norms,
        "gate_activation": str(checkpoint["model_config"].get("gate_activation", "sigmoid")),
        "interpretation_limits": (
            "The gradient is a local policy-sensitivity diagnostic, not a PPO update gradient."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "gate": payload["gate"], "gate_gradient_l2": gate_gradient_l2}, ensure_ascii=False))


if __name__ == "__main__":
    main()

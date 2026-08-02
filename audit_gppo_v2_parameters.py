from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_paper_gppo import tensor_observation
from uav_assignment.gppo_v2 import hard_v2_config
from uav_assignment.paper_env import PaperAlignedUAVEnv
from uav_assignment.paper_models import PaperHeteroActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit GPPO-v2 active parameter paths")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(123)
    env = PaperAlignedUAVEnv(hard_v2_config())
    observation = tensor_observation(env.reset(seed=123))
    rows = []
    for mode in ("none", "single_head", "adaptive_no_gate", "adaptive"):
        model = PaperHeteroActorCritic(
            env.node_feature_dim,
            env.edge_feature_dim,
            env.config.max_uavs,
            env.config.max_tasks,
            hidden_dim=64,
            graph_mode=mode,
        )
        distribution, value = model(**observation)
        valid_action = torch.argmax(distribution.logits, dim=-1)
        loss = -(distribution.log_prob(valid_action).mean() + value.mean())
        loss.backward()
        rows.append(
            {
                "graph_mode": mode,
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "active_path_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ),
                "nonzero_gradient_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                    and bool(torch.count_nonzero(parameter.grad).item())
                ),
            }
        )
    payload = {
        "rows": rows,
        "interpretation": (
            "All modes share an equal-size checkpoint container, but their active "
            "forward-path capacity differs. Performance comparisons must not attribute "
            "all gains solely to the adaptive gate."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameter_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

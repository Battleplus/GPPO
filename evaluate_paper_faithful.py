from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from torch import nn

from train_paper_faithful import tensors
from uav_assignment.paper_faithful_env import (
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
    PaperScale,
    deterministic_instance_seeds,
)
from uav_assignment.paper_faithful_models import PaperFaithfulActorCritic
from uav_assignment.paper_env import UAV_TASK_EDGE


def estimate_forward_flops(model: PaperFaithfulActorCritic, observation: dict[str, np.ndarray]) -> int:
    """Count multiply-adds of the active linear path for one observation.

    This is a transparent analytical estimate (2 FLOPs per multiply-add), not
    a hardware profiler measurement.  It is reported alongside latency so that
    Table VI does not conflate implementation timing with model complexity.
    """
    total = 0
    hooks = []

    def hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if not inputs:
            return
        values = inputs[0]
        elements = int(values.numel() // values.shape[-1])
        total += 2 * elements * module.in_features * module.out_features
        if module.bias is not None:
            total += elements * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(**tensors(observation))
    for handle in hooks:
        handle.remove()
    return int(total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a paper-faithful checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--metadata-checkpoint",
        type=Path,
        help=(
            "Full checkpoint supplying frozen environment/model/training metadata when "
            "--checkpoint is a lightweight candidate containing only iteration/model_state."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "validation_a", "validation_b"),
        default="test",
    )
    parser.add_argument("--eval-scale", type=str)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--sync-mode", choices=("none", "event", "periodic", "always"))
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Save per-decision projected/realized makespan traces for audit tables.",
    )
    parser.add_argument(
        "--gate-scope",
        choices=("task_message", "score", "aggregate"),
        help=(
            "Override the checkpoint gate scope for an inference-only sensitivity replay. "
            "The weights are not retrained."
        ),
    )
    parser.add_argument(
        "--gate-activation",
        choices=("sigmoid", "softplus"),
        help=(
            "Override the checkpoint gate activation for an inference-only sensitivity replay. "
            "The weights are not retrained."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tape_hash(env: PaperFaithfulUAVEnv) -> str:
    serializable = {}
    for decision, event in sorted(env._event_tape.items()):
        serializable[str(decision)] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in event.items()
        }
    payload = json.dumps(serializable, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def observation_hash(observation: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("nodes", "edge_types", "edge_features", "action_mask"):
        array = np.ascontiguousarray(observation[key])
        digest.update(key.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def embedding_signature(encoded: torch.Tensor, bins: int = 8) -> list[float]:
    """Compact deterministic signature of the mean node embedding for causal replay."""
    pooled = encoded.detach().mean(dim=-2).reshape(-1)
    chunks = torch.tensor_split(pooled, bins)
    return [float(chunk.mean()) if chunk.numel() else 0.0 for chunk in chunks]


def final_projection_errors(
    transition_trace: list[dict[str, object]], final_realized_makespan: float
) -> np.ndarray:
    """Relative projected-makespan error against the episode's final outcome."""
    denominator = max(float(final_realized_makespan), 1e-9)
    return np.asarray(
        [
            abs(float(item["projected_makespan"]) - float(final_realized_makespan))
            / denominator
            for item in transition_trace
        ],
        dtype=np.float64,
    )


def evaluation_label(model_config: dict[str, object], sync_mode: str) -> str:
    graph_mode = str(model_config["graph_mode"])
    rrelu_mode = str(model_config.get("rrelu_mode", "expected"))
    gate_scope = str(model_config.get("gate_scope", "task_message"))
    gate_activation = str(model_config.get("gate_activation", "sigmoid"))
    activation_suffix = "" if gate_activation == "sigmoid" else f"_{gate_activation}"
    return f"{graph_mode}_{rrelu_mode}_{gate_scope}{activation_suffix}_{sync_mode}"


def apply_inference_overrides(
    checkpoint_config: dict[str, object],
    gate_scope: str | None = None,
    gate_activation: str | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    """Return an effective config and explicit override metadata.

    The returned configuration is only for an inference replay.  No optimizer
    state or training history is changed, and callers must preserve the
    ``retrained`` distinction in their output artifact.
    """
    model_config = dict(checkpoint_config)
    overrides: dict[str, str] = {}
    if gate_scope is not None:
        model_config["gate_scope"] = gate_scope
        overrides["gate_scope"] = gate_scope
    if gate_activation is not None:
        model_config["gate_activation"] = gate_activation
        overrides["gate_activation"] = gate_activation
    return model_config, overrides


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    weights_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata_checkpoint_sha256: str | None = None
    candidate_iteration: int | None = None
    if "environment" not in weights_checkpoint or "model_config" not in weights_checkpoint:
        if args.metadata_checkpoint is None:
            raise ValueError(
                "Lightweight candidate checkpoint requires --metadata-checkpoint with full config"
            )
        checkpoint = torch.load(args.metadata_checkpoint, map_location="cpu", weights_only=False)
        checkpoint = dict(checkpoint)
        checkpoint["model_state"] = weights_checkpoint["model_state"]
        candidate_iteration = int(weights_checkpoint["iteration"])
        checkpoint["best_iteration"] = candidate_iteration
        metadata_checkpoint_sha256 = file_hash(args.metadata_checkpoint)
    else:
        checkpoint = weights_checkpoint
    scale_payload = checkpoint["environment"]["scale"]
    checkpoint_scale = PaperScale(
        int(scale_payload["uavs"]),
        int(scale_payload["parent_tasks"]),
        int(scale_payload["subtasks"]),
    )
    scale = checkpoint_scale
    if args.eval_scale:
        normalized = args.eval_scale.upper().removeprefix("T")
        values = tuple(int(part) for part in normalized.split("-"))
        scale = PaperScale(*values)
    environment_payload = dict(checkpoint["environment"])
    environment_payload["scale"] = scale
    valid_fields = {field.name for field in fields(PaperFaithfulConfig)}
    config = PaperFaithfulConfig(
        **{
            key: value
            for key, value in environment_payload.items()
            if key in valid_fields and key != "scale"
        },
        scale=scale,
    )
    model_config, requested_overrides = apply_inference_overrides(
        checkpoint["model_config"], args.gate_scope, args.gate_activation
    )
    model = PaperFaithfulActorCritic(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sync_mode = args.sync_mode or str(checkpoint["training"]["sync_mode"])
    rows: list[dict[str, object]] = []
    event_tape_hashes: set[str] = set()
    decision_latencies_ms: list[float] = []
    gate_values: list[float] = []
    estimated_forward_flops: int | None = None
    for instance_id, instance_seed in enumerate(
        deterministic_instance_seeds(scale, args.instances, split=args.split)
    ):
        env = PaperFaithfulUAVEnv(config)
        observation = env.reset(seed=instance_seed)
        if estimated_forward_flops is None:
            estimated_forward_flops = estimate_forward_flops(model, observation)
        current_tape_hash = tape_hash(env)
        event_tape_hashes.add(current_tape_hash)
        done = False
        episode_return = 0.0
        inference_seconds = 0.0
        decisions = 0
        transition_trace: list[dict[str, object]] = []
        cache_ages_before: list[float] = []
        cache_ages_after: list[float] = []
        embedding_deltas: list[float] = []
        synchronized_flags: list[bool] = []
        while not done:
            before_hash = observation_hash(observation) if args.trace else None
            before_tensors = tensors(observation)
            start = time.perf_counter()
            with torch.no_grad():
                action, _, _ = model.act(before_tensors, deterministic=True)
            if str(checkpoint["model_config"]["graph_mode"]) == "literal":
                attention_module = model.literal_attention
                if attention_module.last_gates is not None:
                    edge_types_t = torch.as_tensor(observation["edge_types"], dtype=torch.long)
                    task_mask = edge_types_t[: config.max_uavs] == UAV_TASK_EDGE
                    task_mask[:, : config.max_uavs] = False
                    gate_values.extend(attention_module.last_gates[0][task_mask].tolist())
            elapsed = time.perf_counter() - start
            encoded_before = None
            if args.trace:
                with torch.no_grad():
                    encoded_before = model._encode(
                        before_tensors["nodes"], before_tensors["edge_types"], before_tensors["edge_features"]
                    ).detach()
            inference_seconds += elapsed
            decision_latencies_ms.append(1_000.0 * elapsed)
            action_id = int(action.item())
            observation, reward, done, info = env.step(action_id, sync_mode=sync_mode)
            after_hash = observation_hash(observation) if args.trace else None
            encoded_after = None
            embedding_delta = None
            if args.trace:
                after_tensors = tensors(observation)
                with torch.no_grad():
                    encoded_after = model._encode(
                        after_tensors["nodes"], after_tensors["edge_types"], after_tensors["edge_features"]
                    )
                assert encoded_before is not None
                embedding_delta = float(torch.linalg.vector_norm(encoded_after - encoded_before))
            cache_age_before = float(info.get("cache_age_before", 0.0))
            cache_age_after = float(info.get("cache_age_after", 0.0))
            cache_ages_before.append(cache_age_before)
            cache_ages_after.append(cache_age_after)
            if embedding_delta is not None:
                embedding_deltas.append(embedding_delta)
            synchronized_flags.append(bool(info.get("synchronized", False)))
            episode_return += float(reward)
            decisions += 1
            if args.trace:
                assert before_hash is not None and after_hash is not None
                assert embedding_delta is not None and encoded_after is not None
                transition_trace.append(
                    {
                        "decision": decisions,
                        "action": action_id,
                        "reward": float(reward),
                        "projected_makespan": float(env.makespan),
                        "realized_makespan": float(env.realized_makespan),
                        "event": str(info.get("event", "none")),
                        "synchronized": bool(info.get("synchronized", False)),
                        "cache_age_before": cache_age_before,
                        "cache_age_after": cache_age_after,
                        "observation_hash_before": before_hash,
                        "observation_hash_after": after_hash,
                        "observation_changed": before_hash != after_hash,
                        "embedding_l2_delta": embedding_delta,
                        "embedding_signature_after": embedding_signature(encoded_after),
                    }
                )
        metrics = env.metrics()
        projection_error_mean = None
        projection_error_p95 = None
        if transition_trace:
            final_realized_makespan = float(metrics["realized_makespan"])
            errors = final_projection_errors(transition_trace, final_realized_makespan)
            projection_error_mean = float(np.mean(errors))
            projection_error_p95 = float(np.quantile(errors, 0.95))
        rows.append(
            {
                "instance_id": instance_id,
                "instance_seed": instance_seed,
                "event_tape_hash": current_tape_hash,
                "episode_return": episode_return,
                "realized_makespan": float(metrics["realized_makespan"]),
                "projected_makespan": float(metrics["projected_makespan"]),
                "completion_rate": float(metrics["completion_rate"]),
                "all_tasks_completed": float(metrics["all_tasks_completed"]),
                "communication_events": float(metrics["communication_events"]),
                "communication_bytes": float(metrics["communication_bytes"]),
                "heartbeat_messages": float(metrics["heartbeat_messages"]),
                "invalid_actions": float(metrics["invalid_actions"]),
                "reallocated_tasks": float(metrics["reallocated_tasks"]),
                "decisions": decisions,
                "inference_seconds": inference_seconds,
                "mean_inference_ms": 1_000.0 * inference_seconds / max(1, decisions),
                "estimated_forward_flops": estimated_forward_flops,
                "cache_age_mean": float(np.mean(cache_ages_before)),
                "cache_age_max": float(max(cache_ages_before, default=0.0)),
                "cache_age_after_sync_mean": float(
                    np.mean([
                        age for age, synchronized in zip(cache_ages_after, synchronized_flags)
                        if synchronized
                    ])
                ) if any(synchronized_flags) else 0.0,
                "embedding_l2_delta_mean": (
                    float(np.mean(embedding_deltas)) if embedding_deltas else None
                ),
                "projection_error_relative_mean": projection_error_mean,
                "projection_error_relative_p95": projection_error_p95,
                **({"transition_trace": transition_trace} if args.trace else {}),
            }
        )
    numeric = (
        "episode_return", "realized_makespan", "projected_makespan",
        "completion_rate", "all_tasks_completed", "communication_events",
        "communication_bytes", "heartbeat_messages", "decisions", "mean_inference_ms",
        "invalid_actions", "reallocated_tasks",
        "projection_error_relative_mean", "projection_error_relative_p95",
        "estimated_forward_flops",
        "cache_age_mean", "cache_age_max", "cache_age_after_sync_mean",
        "embedding_l2_delta_mean",
    )
    summary = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows if row[key] is not None])),
            "std": float(np.std([float(row[key]) for row in rows if row[key] is not None], ddof=1)),
            "median": float(np.median([float(row[key]) for row in rows if row[key] is not None])),
        }
        for key in numeric
        if any(row[key] is not None for row in rows)
    }
    inference_latency_ms = {
        "mean": float(np.mean(decision_latencies_ms)),
        "std": float(np.std(decision_latencies_ms, ddof=1)),
        "p50": float(np.percentile(decision_latencies_ms, 50)),
        "p95": float(np.percentile(decision_latencies_ms, 95)),
        "samples": len(decision_latencies_ms),
    }
    gate_summary = None
    if gate_values:
        gate_array = np.asarray(gate_values, dtype=np.float64)
        gate_summary = {
            "count": int(gate_array.size),
            "mean": float(np.mean(gate_array)),
            "std": float(np.std(gate_array, ddof=1)) if gate_array.size > 1 else 0.0,
            "median": float(np.median(gate_array)),
            "p05": float(np.quantile(gate_array, 0.05)),
            "p95": float(np.quantile(gate_array, 0.95)),
            "fraction_gt_0_9": float(np.mean(gate_array > 0.9)),
            "fraction_lt_0_1": float(np.mean(gate_array < 0.1)),
            "positive_amplification_fraction": float(np.mean(gate_array > 1.0)),
        }
    graph_mode = str(model_config["graph_mode"])
    rrelu_mode = str(model_config.get("rrelu_mode", "expected"))
    gate_scope = str(model_config.get("gate_scope", "task_message"))
    gate_activation = str(model_config.get("gate_activation", "sigmoid"))
    # The first long-running formal batch predates the checkpoint field that
    # records total parameters.  Its state dict is architecture-compatible
    # with the frozen task_message+sigmoid model, so derive missing counts from
    # the loaded model instead of emitting NA in the final audit/report.
    active_parameter_count = int(
        checkpoint.get("active_parameter_count", model.active_parameter_count())
    )
    total_parameter_count = int(
        checkpoint.get(
            "total_parameter_count",
            sum(parameter.numel() for parameter in model.parameters()),
        )
    )
    payload = {
        "version": "paper-faithful-evaluation-v1",
        "communication_trace_version": "cache-observation-embedding-v1" if args.trace else None,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "checkpoint_kind": "candidate_overlay" if candidate_iteration is not None else "full",
        "candidate_iteration": candidate_iteration,
        "metadata_checkpoint": str(args.metadata_checkpoint) if args.metadata_checkpoint else None,
        "metadata_checkpoint_sha256": metadata_checkpoint_sha256,
        "model_config": model_config,
        "inference_overrides": requested_overrides,
        "retrained": not bool(requested_overrides),
        "mode": graph_mode,
        "rrelu_mode": rrelu_mode,
        "gate_scope": gate_scope,
        "gate_activation": gate_activation,
        "label": evaluation_label(model_config, sync_mode),
        "sync_mode": sync_mode,
        "training_seed": int(checkpoint["training"]["seed"]),
        "training_scale": checkpoint_scale.name,
        "scale": scale.name,
        "split": args.split,
        "instances": args.instances,
        "event_tape_hash_count": len(event_tape_hashes),
        "event_tape_hashes": sorted(event_tape_hashes),
        "best_iteration": checkpoint.get("best_iteration"),
        "recovery_info": checkpoint.get("recovery_info"),
        "resume_events": checkpoint.get("resume_events", []),
        "active_parameter_count": active_parameter_count,
        "total_parameter_count": total_parameter_count,
        "summary": summary,
        "inference_latency_ms": inference_latency_ms,
        "estimated_forward_flops": estimated_forward_flops,
        "gate_summary": gate_summary,
        "projection_error_definition": (
            "At each decision, abs(projected_makespan - final_realized_makespan) "
            "/ final_realized_makespan; trace realized_makespan remains the "
            "contemporaneous completed-task value."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()

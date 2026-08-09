from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_paper_faithful_formal import evaluation_directory


MODES = ("none", "event", "periodic", "always")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def action_divergence(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    denominator = max(len(left), len(right), 1)
    mismatches = sum(a["action"] != b["action"] for a, b in zip(left, right))
    mismatches += abs(len(left) - len(right))
    return float(mismatches / denominator)


def hash_divergence(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    denominator = max(len(left), len(right), 1)
    mismatches = sum(
        a["observation_hash_after"] != b["observation_hash_after"]
        for a, b in zip(left, right)
    ) + abs(len(left) - len(right))
    return float(mismatches / denominator)


def signature_distance(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    distances = []
    for a, b in zip(left, right):
        first = np.asarray(a["embedding_signature_after"], dtype=np.float64)
        second = np.asarray(b["embedding_signature_after"], dtype=np.float64)
        distances.append(float(np.linalg.norm(first - second)))
    return float(np.mean(distances)) if distances else 0.0


def mean_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(max(1, array.size - 1), 1.96)
    half = critical * std / math.sqrt(array.size) if array.size > 1 else 0.0
    return {"n": int(array.size), "mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit causal communication replay for Phase 1.")
    parser.add_argument("--matrix-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    matrix = read_json(args.matrix_manifest)
    if matrix.get("status") not in {"communication_replay_completed", "completed"}:
        raise RuntimeError("Formal matrix and communication replay are not complete")
    checkpoints = [Path(path) for path in matrix["communication_replay_checkpoints"]]
    if len(checkpoints) != 20:
        raise RuntimeError(f"Expected 20 communication checkpoints, found {len(checkpoints)}")

    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        payloads = {
            mode: read_json(evaluation_directory(checkpoint) / f"test_native_{mode}_100.json")
            for mode in MODES
        }
        hashes = {payload["checkpoint_sha256"] for payload in payloads.values()}
        instance_banks = {
            tuple(int(row["instance_seed"]) for row in payload["rows"])
            for payload in payloads.values()
        }
        tape_banks = {tuple(payload["event_tape_hashes"]) for payload in payloads.values()}
        if len(hashes) != 1 or len(instance_banks) != 1 or len(tape_banks) != 1:
            raise RuntimeError(f"Replay does not hold checkpoint/instances/tapes fixed: {checkpoint}")
        by_mode = {
            mode: {int(row["instance_seed"]): row for row in payload["rows"]}
            for mode, payload in payloads.items()
        }
        seeds = sorted(by_mode["event"])
        rows = []
        for instance_seed in seeds:
            event = by_mode["event"][instance_seed]
            none = by_mode["none"][instance_seed]
            full = by_mode["always"][instance_seed]
            periodic = by_mode["periodic"][instance_seed]
            event_trace = event["transition_trace"]
            none_trace = none["transition_trace"]
            full_trace = full["transition_trace"]
            periodic_trace = periodic["transition_trace"]
            sync_steps = [item for item in event_trace if item["synchronized"]]
            rows.append({
                "instance_seed": instance_seed,
                "event_full_relative_makespan_gap": (
                    float(event["realized_makespan"]) - float(full["realized_makespan"])
                ) / max(float(full["realized_makespan"]), 1e-9),
                "event_full_byte_reduction": 1.0 - float(event["communication_bytes"]) / max(float(full["communication_bytes"]), 1.0),
                "event_none_action_divergence": action_divergence(event_trace, none_trace),
                "event_full_action_divergence": action_divergence(event_trace, full_trace),
                "event_periodic_action_divergence": action_divergence(event_trace, periodic_trace),
                "event_none_graph_hash_divergence": hash_divergence(event_trace, none_trace),
                "event_none_embedding_signature_l2": signature_distance(event_trace, none_trace),
                "event_sync_count": len(sync_steps),
                "event_sync_cache_reset": all(float(item["cache_age_after"]) == 0.0 for item in sync_steps),
                "event_sync_only_on_events": all(
                    bool(item["synchronized"]) == (item["event"] != "none") for item in event_trace
                ),
                "event_cache_age_mean": float(event["cache_age_mean"]),
                "event_cache_age_max": float(event["cache_age_max"]),
                "none_cache_age_mean": float(none["cache_age_mean"]),
                "full_cache_age_mean": float(full["cache_age_mean"]),
            })
        record = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashes.pop(),
            "scale": payloads["event"]["scale"],
            "training_seed": payloads["event"]["training_seed"],
            "same_checkpoint_instances_and_event_tapes": True,
            "rows": rows,
            "means": {
                key: float(np.mean([float(row[key]) for row in rows]))
                for key in (
                    "event_full_relative_makespan_gap", "event_full_byte_reduction",
                    "event_none_action_divergence", "event_full_action_divergence",
                    "event_periodic_action_divergence", "event_none_graph_hash_divergence",
                    "event_none_embedding_signature_l2", "event_cache_age_mean",
                    "event_cache_age_max", "none_cache_age_mean", "full_cache_age_mean",
                )
            },
            "event_sync_cache_reset_all": all(row["event_sync_cache_reset"] for row in rows),
            "event_sync_only_on_events_all": all(row["event_sync_only_on_events"] for row in rows),
        }
        records.append(record)

    per_scale: dict[str, Any] = {}
    for scale in sorted({record["scale"] for record in records}):
        scale_records = [record for record in records if record["scale"] == scale]
        per_scale[scale] = {
            key: mean_ci([record["means"][key] for record in scale_records])
            for key in (
                "event_full_relative_makespan_gap", "event_full_byte_reduction",
                "event_none_action_divergence", "event_none_graph_hash_divergence",
                "event_none_embedding_signature_l2",
            )
        }
    quality_within_5_percent = all(
        abs(values["event_full_relative_makespan_gap"]["mean"]) <= 0.05
        for values in per_scale.values()
    )
    lower_communication = all(
        values["event_full_byte_reduction"]["mean"] > 0 for values in per_scale.values()
    )
    actual_graph_effect = any(
        record["means"]["event_none_graph_hash_divergence"] > 0
        and record["means"]["event_none_embedding_signature_l2"] > 0
        for record in records
    )
    action_effect = any(record["means"]["event_none_action_divergence"] > 0 for record in records)
    no_false_sync = all(record["event_sync_only_on_events_all"] for record in records)
    cache_reset = all(record["event_sync_cache_reset_all"] for record in records)
    conclusion = (
        "sufficient_causal_support" if quality_within_5_percent and lower_communication and actual_graph_effect and action_effect and no_false_sync and cache_reset
        else "insufficient_action_effect" if not action_effect
        else "communication_mechanism_failed_acceptance"
    )
    audit = {
        "version": "phase1-communication-causal-audit-v1",
        "same_policy_instance_and_event_tape_replay": True,
        "records": records,
        "per_scale_training_seed_statistics": per_scale,
        "checks": {
            "event_full_quality_gap_within_5_percent": quality_within_5_percent,
            "event_communication_lower_than_full": lower_communication,
            "event_changes_graph_observation_and_embedding": actual_graph_effect,
            "event_changes_actions_relative_to_none": action_effect,
            "event_sync_only_on_true_events": no_false_sync,
            "event_sync_resets_cache_age": cache_reset,
        },
        "conclusion": conclusion,
        "valid": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Event 通信因果审计", "",
        "所有 None/Event/Periodic/Full 重放均固定同一 checkpoint、test100 实例库和事件带。", "",
        "| 规模 | Event-Full makespan 相对差 | Event 字节减少率 | Event-None 动作分歧率 | 图哈希分歧率 |", "|---|---:|---:|---:|---:|",
    ]
    for scale, values in per_scale.items():
        lines.append(
            f"| {scale} | {values['event_full_relative_makespan_gap']['mean']:.4f} | "
            f"{values['event_full_byte_reduction']['mean']:.4f} | "
            f"{values['event_none_action_divergence']['mean']:.4f} | "
            f"{values['event_none_graph_hash_divergence']['mean']:.4f} |"
        )
    lines += ["", f"结论：`{conclusion}`。若 Event 与 None 动作完全一致，则按预注册规则判为通信贡献不充分。", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "conclusion": conclusion}, ensure_ascii=False))


if __name__ == "__main__":
    main()

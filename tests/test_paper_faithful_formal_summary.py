from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_paper_faithful import (
    apply_inference_overrides,
    evaluation_label,
    final_projection_errors,
    embedding_signature,
    observation_hash,
)
from summarize_paper_faithful_formal import main as summarize_main
from summarize_paper_faithful_formal import mean_ci
from report_paper_faithful_formal_zh import _baseline_check, _direction_check, _paired_row


def test_formal_ci_uses_student_t_for_five_seeds() -> None:
    result = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["ci_method"] == "two-sided Student-t, 95%"
    assert result["n"] == 5
    assert math.isclose(result["ci95_half_width"], 2.776 * 1.5811388300841898 / math.sqrt(5), rel_tol=1e-6)


def test_evaluation_label_keeps_sync_and_gate_scope_separate() -> None:
    config = {"graph_mode": "literal", "rrelu_mode": "expected", "gate_scope": "task_message"}
    assert evaluation_label(config, "none") != evaluation_label(config, "event")
    assert evaluation_label(config, "event") != evaluation_label(
        {**config, "gate_scope": "score"}, "event"
    )


def test_observation_hash_is_deterministic_and_content_sensitive() -> None:
    observation = {
        "nodes": np.zeros((2, 3), dtype=np.float32),
        "edge_types": np.zeros((2, 2), dtype=np.int64),
        "edge_features": np.zeros((2, 2, 1), dtype=np.float32),
        "action_mask": np.asarray([True, False]),
    }
    before = observation_hash(observation)
    assert observation_hash(observation) == before
    observation["nodes"][0, 0] = 1.0
    assert observation_hash(observation) != before


def test_embedding_signature_is_compact_and_deterministic() -> None:
    encoded = torch.arange(2 * 4 * 16, dtype=torch.float32).reshape(2, 4, 16)
    signature = embedding_signature(encoded)
    assert len(signature) == 8
    assert signature == embedding_signature(encoded.clone())


def test_inference_overrides_are_explicit_and_do_not_mutate_checkpoint_config() -> None:
    original = {
        "graph_mode": "literal",
        "gate_scope": "task_message",
        "gate_activation": "sigmoid",
    }
    effective, overrides = apply_inference_overrides(original, "score", "softplus")
    assert original["gate_scope"] == "task_message"
    assert original["gate_activation"] == "sigmoid"
    assert effective["gate_scope"] == "score"
    assert effective["gate_activation"] == "softplus"
    assert overrides == {"gate_scope": "score", "gate_activation": "softplus"}


def test_formal_summary_builds_paired_seed_difference(tmp_path: Path, monkeypatch) -> None:
    numeric = (
        "episode_return", "realized_makespan", "completion_rate",
        "all_tasks_completed", "communication_events", "communication_bytes",
        "heartbeat_messages", "mean_inference_ms",
    )
    for seed, left_value, right_value in ((1, 10.0, 11.0), (2, 12.0, 13.0)):
        for label, value in (("literal_expected_task_message_event", left_value), ("ppo_mlp_expected_task_message_event", right_value)):
            summary = {key: {"mean": value if key == "realized_makespan" else 1.0} for key in numeric}
            payload = {
                "training_scale": "T5-10-48",
                "scale": "T5-10-48",
                "label": label,
                "training_seed": seed,
                "summary": summary,
                "inference_latency_ms": {"p50": 1.0, "p95": 2.0},
                "rows": [
                    {"instance_seed": 100 + seed, "event_tape_hash": "same", "realized_makespan": value},
                ],
            }
            path = tmp_path / label / f"seed{seed}" / "evaluations" / "test_native_100.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", ["summarize", "--root", str(tmp_path), "--output", str(output)])
    summarize_main()
    result = json.loads(output.read_text(encoding="utf-8"))
    paired = result["paired_method_differences"]
    assert len(paired) == 1
    assert paired[0]["realized_makespan_difference"]["mean"] == -1.0
    assert paired[0]["realized_makespan_difference"]["ci_method"] == "two-sided Student-t, 95%"


def test_report_direction_check_requires_all_scales_and_stable_direction() -> None:
    rows = []
    for scale in ("T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92"):
        rows.append({
            "training_scale": scale,
            "evaluation_scale": scale,
            "left_label": "literal_expected_task_message_event",
            "right_label": "ppo_mlp_expected_task_message_none",
            "realized_makespan_difference": {"mean": -1.0, "ci95_high": -0.1},
        })
    result = _direction_check(
        {"paired_method_differences": rows},
        "literal_expected_task_message_event",
        "ppo_mlp_expected_task_message_none",
    )
    assert result["scales_observed"] == 4
    assert result["negative_ci_scales"] == 4
    assert result["pass"] is True


def test_report_direction_check_rejects_missing_scale() -> None:
    result = _direction_check(
        {
            "paired_method_differences": [{
                "training_scale": "T5-10-48",
                "evaluation_scale": "T5-10-48",
                "left_label": "literal_expected_task_message_event",
                "right_label": "ppo_mlp_expected_task_message_none",
                "realized_makespan_difference": {"mean": -1.0, "ci95_high": -0.1},
            }]
        },
        "literal_expected_task_message_event",
        "ppo_mlp_expected_task_message_none",
    )
    assert result["pass"] is False


def test_projection_error_targets_final_realized_makespan() -> None:
    trace = [
        {"projected_makespan": 8.0, "realized_makespan": 2.0},
        {"projected_makespan": 10.0, "realized_makespan": 10.0},
    ]
    values = final_projection_errors(trace, final_realized_makespan=10.0)
    assert values.tolist() == [0.2, 0.0]


def test_paired_row_inversion_swaps_confidence_interval_bounds() -> None:
    payload = {
        "paired_method_differences": [{
            "training_scale": "T5-10-48",
            "evaluation_scale": "T5-10-48",
            "left_label": "b",
            "right_label": "a",
            "realized_makespan_difference": {
                "mean": -1.5,
                "ci95_low": -2.0,
                "ci95_high": -1.0,
            },
        }]
    }
    row = _paired_row(payload, "T5-10-48", "a", "b")
    assert row is not None
    assert row["realized_makespan_difference"]["mean"] == 1.5
    assert row["realized_makespan_difference"]["ci95_low"] == 1.0
    assert row["realized_makespan_difference"]["ci95_high"] == 2.0


def test_baseline_check_requires_gppo_to_beat_random_on_every_scale() -> None:
    scales = ("T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92")
    native = [
        {
            "evaluation_scale": scale,
            "label": "literal_expected_task_message_event",
            "metrics": {"realized_makespan": {"mean": 10.0}},
        }
        for scale in scales
    ]
    results = []
    for scale in scales:
        for policy, value in (("random", 12.0), ("greedy", 10.2)):
            results.append({
                "scale": scale,
                "policy": policy,
                "summary": {"realized_makespan": {"mean": value}},
            })
    passed = _baseline_check(native, {"results": results})
    assert passed["random_pass"] is True
    assert passed["greedy_close_pass"] is True
    results[0]["summary"]["realized_makespan"]["mean"] = 9.0
    failed = _baseline_check(native, {"results": results})
    assert failed["random_pass"] is False

from __future__ import annotations

from audit_phase1_communication import action_divergence, hash_divergence, signature_distance


def trace(action: int, graph_hash: str, signature: list[float]) -> dict:
    return {
        "action": action,
        "observation_hash_after": graph_hash,
        "embedding_signature_after": signature,
    }


def test_communication_pair_metrics_detect_causal_differences() -> None:
    event = [trace(1, "a", [0.0, 0.0]), trace(2, "b", [1.0, 1.0])]
    none = [trace(1, "a", [0.0, 0.0]), trace(3, "c", [2.0, 1.0])]
    assert action_divergence(event, none) == 0.5
    assert hash_divergence(event, none) == 0.5
    assert signature_distance(event, none) > 0.0


def test_action_divergence_counts_unequal_trace_lengths() -> None:
    assert action_divergence([trace(1, "a", [0.0])], []) == 1.0

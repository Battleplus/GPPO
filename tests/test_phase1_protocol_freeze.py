from __future__ import annotations

import pytest

from freeze_phase1_protocol import build_frozen_protocol


def protocol() -> dict:
    variants = []
    for name, mode in (("NoGate", "literal_no_gate"), ("SingleHead", "literal_single_head")):
        variants.append({"name": name, "graph_mode": mode, "gate_scope": "task_message", "gate_activation": "sigmoid", "gate_bias_init": 0.0, "gate_warmup_iterations": 0})
    return {
        "frozen_literal_definition": {"name": "Literal", "graph_mode": "literal", "gate_scope": "task_message", "gate_activation": "sigmoid", "gate_bias_init": 0.0, "gate_warmup_iterations": 0, "post_gate_renormalization": False, "rrelu_mode": "expected", "definition_must_not_change": True},
        "variants": variants,
    }


def summary() -> dict:
    return {"valid": True, "test_used_for_selection": False, "protocol_sha256": "abc", "amendment_sha256": "def", "training_code_commit": "123", "outcome": {"selected_adaptive": None, "engineering_baseline_if_no_adaptive_passes": "NoGate"}}


def test_freeze_falls_back_without_hiding_literal_negative_result() -> None:
    result = build_frozen_protocol(summary(), protocol(), "abc")
    assert result["models"]["GPPO-Best"]["selected_variant"] == "NoGate"
    assert result["models"]["GPPO-Literal"]["graph_mode"] == "literal"
    assert result["test_used_for_selection"] is False


def test_freeze_rejects_test_selected_summary() -> None:
    payload = summary()
    payload["test_used_for_selection"] = True
    with pytest.raises(ValueError, match="test data"):
        build_frozen_protocol(payload, protocol(), "abc")

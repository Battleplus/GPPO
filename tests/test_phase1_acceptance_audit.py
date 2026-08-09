from __future__ import annotations

import json

from audit_phase1_acceptance import file_evidence, json_evidence


def test_missing_evidence_never_passes(tmp_path) -> None:
    assert json_evidence(tmp_path / "missing.json", lambda _: True)["valid"] is False
    assert file_evidence(tmp_path / "missing.md")["valid"] is False


def test_negative_scientific_result_can_be_valid_evidence(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"valid": True, "conclusion": "adaptive benefit not reproduced"}), encoding="utf-8")
    evidence = json_evidence(path, lambda payload: payload.get("valid") is True)
    assert evidence["valid"] is True
    assert "sha256" in evidence

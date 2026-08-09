from __future__ import annotations

import json
import zipfile

from audit_phase1_acceptance import file_evidence, json_evidence, zip_evidence


def test_missing_evidence_never_passes(tmp_path) -> None:
    assert json_evidence(tmp_path / "missing.json", lambda _: True)["valid"] is False
    assert file_evidence(tmp_path / "missing.md")["valid"] is False


def test_negative_scientific_result_can_be_valid_evidence(tmp_path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"valid": True, "conclusion": "adaptive benefit not reproduced"}), encoding="utf-8")
    evidence = json_evidence(path, lambda payload: payload.get("valid") is True)
    assert evidence["valid"] is True
    assert "sha256" in evidence


def test_zip_evidence_requires_readable_unique_required_members(tmp_path) -> None:
    path = tmp_path / "results.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("reports/final.md", "ok")
        archive.writestr("audits/final.json", "{}")
    assert zip_evidence(path, {"final.md", "final.json"})["valid"] is True
    assert zip_evidence(path, {"missing.json"})["valid"] is False


def test_plain_nonempty_file_cannot_pass_as_zip(tmp_path) -> None:
    path = tmp_path / "fake.zip"
    path.write_bytes(b"not a zip")
    assert zip_evidence(path, set())["valid"] is False

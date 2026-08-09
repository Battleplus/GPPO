from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_evidence(path: Path, predicate) -> dict[str, Any]:
    evidence: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "valid": False}
    if not path.is_file():
        evidence["reason"] = "missing"
        return evidence
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence["sha256"] = sha256(path)
        evidence["valid"] = bool(predicate(payload))
        if not evidence["valid"]:
            evidence["reason"] = "schema or acceptance predicate failed"
    except Exception as error:  # the audit must record malformed evidence, not crash silently
        evidence["reason"] = f"unreadable JSON: {error}"
    return evidence


def file_evidence(path: Path, *, nonempty: bool = True) -> dict[str, Any]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    valid = exists and (size > 0 or not nonempty)
    result: dict[str, Any] = {"path": str(path), "exists": exists, "size": size, "valid": valid}
    if valid:
        result["sha256"] = sha256(path)
    else:
        result["reason"] = "missing or empty"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict requirement-by-requirement Phase-1 acceptance audit.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/paper_faithful/formal"))
    parser.add_argument("--final-root", type=Path, default=Path("outputs/phase1_final"))
    parser.add_argument("--output", type=Path, default=Path("PHASE1_ACCEPTANCE.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    formal = args.formal_root.resolve()
    final = args.final_root.resolve()

    requirements = {
        "step0_historical_artifact_freeze": json_evidence(
            repo / "artifacts/phase1_seed1_300/ARTIFACT_AUDIT.json",
            lambda p: p.get("valid") is True and not p.get("errors"),
        ),
        "step1_candidate_checkpoint_reevaluation": json_evidence(
            repo / "outputs/checkpoint_reevaluation/results.json",
            lambda p: p.get("valid") is True
            and p.get("selection_uses_test") is False
            and p.get("selection_uses_validation_b") is False,
        ),
        "step2_gate_smoke": json_evidence(
            repo / "artifacts/phase1_gate_smoke/SMOKE_AUDIT.json",
            lambda p: p.get("valid") is True and p.get("global_checks", {}).get("no_test_artifacts") is True,
        ),
        "step3_three_seed_gate_screening": json_evidence(
            repo / "outputs/gate_screening/summary.json",
            lambda p: p.get("valid") is True and p.get("test_used_for_selection") is False,
        ),
        "step4_frozen_protocol": json_evidence(
            repo / "configs/PHASE1_FROZEN_PROTOCOL.json",
            lambda p: p.get("frozen_after_gate_screening") is True and p.get("test_used_for_selection") is False,
        ),
        "steps5_6_formal_matrix": json_evidence(
            formal / "PHASE1_FORMAL_MATRIX_RUN.json",
            lambda p: p.get("valid") is True and p.get("status") == "completed" and p.get("validation_split") == "validation_a",
        ),
        "step7_communication_causal_audit": json_evidence(
            formal / "PHASE1_COMMUNICATION_CAUSAL_AUDIT.json",
            lambda p: p.get("valid") is True and p.get("same_policy_instance_and_event_tape_replay") is True,
        ),
        "steps8_9_disturbance_implementation": json_evidence(
            final / "DISTURBANCE_IMPLEMENTATION_AUDIT.json",
            lambda p: p.get("valid") is True and p.get("all_sources_replayable") is True,
        ),
        "step10_disturbance_calibration": json_evidence(
            final / "DISTURBANCE_CALIBRATION_AUDIT.json",
            lambda p: p.get("valid") is True and p.get("non_saturation_checks_passed") is True,
        ),
        "step11_final_experiment": json_evidence(
            final / "PHASE1_FINAL_EXPERIMENT_AUDIT.json",
            lambda p: p.get("valid") is True and p.get("four_scales_five_training_seeds") is True,
        ),
    }
    deliverables: dict[str, dict[str, Any]] = {
        name: file_evidence(repo / name)
        for name in (
            "PHASE1_FINAL_REPORT_ZH.md",
            "PHASE1_REPRODUCTION_GUIDE.md",
            "PHASE1_RESULTS.zip",
        )
    }
    deliverables["PHASE1_ARTIFACT_MANIFEST.json"] = json_evidence(
        repo / "PHASE1_ARTIFACT_MANIFEST.json",
        lambda p: p.get("valid") is True and p.get("all_hashes_verified") is True,
    )
    deliverables["PHASE1_CONFIG_AUDIT.json"] = json_evidence(
        repo / "PHASE1_CONFIG_AUDIT.json",
        lambda p: p.get("valid") is True and p.get("test_leakage_detected") is False,
    )
    all_requirements = all(item["valid"] for item in requirements.values())
    all_deliverables = all(item["valid"] for item in deliverables.values())
    payload = {
        "version": "phase1-acceptance-audit-v1",
        "policy": "Negative scientific conclusions may pass; missing, malformed, partial or unaudited evidence may not.",
        "requirements": requirements,
        "deliverables": deliverables,
        "all_requirements_valid": all_requirements,
        "all_deliverables_valid": all_deliverables,
        "valid": bool(all_requirements and all_deliverables),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "valid": payload["valid"]}, ensure_ascii=False))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

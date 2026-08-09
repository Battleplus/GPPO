from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uav_assignment.paper_faithful_env import PAPER_SCALES, deterministic_instance_seeds


EXPECTED_SCALES = ["T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92"]
EXPECTED_SEEDS = [1, 2, 3, 4, 5]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def bank_audit() -> dict[str, Any]:
    scales: dict[str, Any] = {}
    for scale in PAPER_SCALES:
        banks = {
            split: deterministic_instance_seeds(scale, 100, split=split)
            for split in ("validation_a", "validation_b", "test")
        }
        unique = all(len(values) == len(set(values)) == 100 for values in banks.values())
        disjoint = not (
            set(banks["validation_a"]) & set(banks["validation_b"])
            or set(banks["validation_a"]) & set(banks["test"])
            or set(banks["validation_b"]) & set(banks["test"])
        )
        scales[scale.name] = {
            "unique_100_each": unique,
            "pairwise_disjoint": disjoint,
            "first_last": {key: [values[0], values[-1]] for key, values in banks.items()},
        }
    return {"scales": scales, "valid": all(item["unique_100_each"] and item["pairwise_disjoint"] for item in scales.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit every frozen Phase-1 configuration and leakage boundary.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/paper_faithful/formal"))
    parser.add_argument("--output", type=Path, default=Path("PHASE1_CONFIG_AUDIT.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    formal = args.formal_root.resolve()

    gate_protocol_path = repo / "configs/GATE_DIAGNOSTIC_PROTOCOL.json"
    cohorts_path = repo / "configs/GATE_SCREENING_RUNTIME_COHORTS.json"
    frozen_path = repo / "configs/PHASE1_FROZEN_PROTOCOL.json"
    matrix_path = formal / "PHASE1_FORMAL_MATRIX_RUN.json"
    gate_summary_path = repo / "outputs/gate_screening/summary.json"
    gate = load(gate_protocol_path)
    cohorts = load(cohorts_path)
    frozen = load(frozen_path)
    matrix = load(matrix_path)
    gate_summary = load(gate_summary_path)

    current_hashes_valid = False
    if cohorts:
        current_hashes_valid = all(
            (repo / relative).is_file() and sha256(repo / relative) == expected
            for relative, expected in cohorts["cohorts"][-1]["files_sha256"].items()
        )
    gate_integrity = bool(
        gate
        and cohorts
        and cohorts.get("protocol_sha256") == sha256(gate_protocol_path)
        and current_hashes_valid
        and cohorts.get("test_or_test100_used") is False
    )
    gate_selection_valid = bool(
        gate_summary
        and gate_summary.get("valid") is True
        and gate_summary.get("test_used_for_selection") is False
    )
    frozen_training = (frozen or {}).get("formal_training", {})
    frozen_valid = bool(
        frozen
        and frozen.get("test_used_for_selection") is False
        and frozen_training.get("scales") == EXPECTED_SCALES
        and frozen_training.get("training_seeds") == EXPECTED_SEEDS
        and frozen_training.get("iterations") == 2000
        and frozen_training.get("rollout_steps") == 512
        and frozen_training.get("batch_size") == 512
        and frozen_training.get("update_epochs") == 4
        and frozen_training.get("checkpoint_selection_split") == "validation_a"
        and frozen_training.get("confirmation_split") == "validation_b"
        and frozen_training.get("test_instances") == 100
    )
    matrix_valid = bool(
        matrix
        and matrix.get("valid") is True
        and matrix.get("status") == "completed"
        and matrix.get("validation_split") == "validation_a"
        and matrix.get("test_used_for_training_or_selection") is False
        and matrix.get("frozen_protocol_sha256") == (sha256(frozen_path) if frozen_path.is_file() else None)
    )
    disturbance_protocol_path = repo / "configs/DISTURBANCE_INTERFACE_PROTOCOL.json"
    disturbance_protocol = load(disturbance_protocol_path)
    disturbance_configs = {
        severity: repo / f"configs/disturbance_{severity}.json"
        for severity in ("weak", "medium", "strong")
    }
    disturbance_valid = bool(
        disturbance_protocol
        and disturbance_protocol.get("test_or_formal_test_used") is False
        and all(path.is_file() for path in disturbance_configs.values())
    )
    prohibited = ("pcrl", "preference-gppo", "bradley-terry", "jepa", "world_model")
    configuration_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").lower()
        for path in repo.joinpath("configs").glob("*.json")
    )
    prohibited_integration_detected = any(token in configuration_text for token in prohibited)
    banks = bank_audit()
    checks = {
        "gate_protocol_and_runtime_cohorts_valid": gate_integrity,
        "gate_selection_validation_only": gate_selection_valid,
        "frozen_protocol_exact_formal_constants": frozen_valid,
        "formal_matrix_uses_frozen_validation_only_protocol": matrix_valid,
        "validation_a_b_test_banks_unique_and_disjoint": banks["valid"],
        "disturbance_protocol_and_three_severities_present": disturbance_valid,
        "prohibited_preference_or_world_model_integration_absent": not prohibited_integration_detected,
    }
    test_leakage_detected = bool(
        (gate_summary and gate_summary.get("test_used_for_selection") is not False)
        or (frozen and frozen.get("test_used_for_selection") is not False)
        or (matrix and matrix.get("test_used_for_training_or_selection") is not False)
    )
    missing_evidence = [
        name
        for name, value in {
            "gate_screening_summary": gate_summary,
            "frozen_protocol": frozen,
            "formal_matrix": matrix,
            **{f"disturbance_{severity}": load(path) for severity, path in disturbance_configs.items()},
        }.items()
        if value is None
    ]
    payload = {
        "version": "phase1-config-audit-v1",
        "checks": checks,
        "instance_banks": banks,
        "config_hashes": {
            str(path.relative_to(repo)): sha256(path)
            for path in sorted(repo.joinpath("configs").glob("*.json"))
        },
        "test_leakage_detected": test_leakage_detected,
        "missing_evidence": missing_evidence,
        "prohibited_integration_detected": prohibited_integration_detected,
        "valid": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "valid": payload["valid"]}, ensure_ascii=False))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

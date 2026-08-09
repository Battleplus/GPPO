from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


METHODS = {
    "literal_event_seed1": ("literal", "event"),
    "literal_no_gate_event_seed1": ("literal_no_gate", "event"),
    "literal_none_seed1": ("literal", "none"),
    "literal_single_head_event_seed1": ("literal_single_head", "event"),
    "ppo_mlp_event_seed1": ("ppo_mlp", "event"),
    "ppo_mlp_none_seed1": ("ppo_mlp", "none"),
}
EXPECTED_CANDIDATES = tuple(range(50, 301, 50))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze and audit the Phase-1 seed1/300 historical baseline")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--enforce-read-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_value(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def file_record(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(base).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256(path),
        "read_only": not bool(path.stat().st_mode & stat.S_IWRITE),
    }


def main() -> None:
    args = parse_args()
    raw = args.raw.resolve()
    repo = Path(__file__).resolve().parent
    scale_root = raw / "T5-10-48"
    errors: list[str] = []
    warnings: list[str] = []
    method_rows: list[dict[str, object]] = []
    test_seed_sets: list[set[int]] = []
    tape_sets: list[set[str]] = []
    actual_checkpoint_hashes: dict[str, str] = {}

    for name, (expected_mode, expected_sync) in METHODS.items():
        directory = scale_root / name
        required = {
            "checkpoint": directory / "checkpoint.pt",
            "resume": directory / "resume_latest.pt",
            "training_history": directory / "training_history.json",
            "validation_history": directory / "validation_history.json",
            "test100": directory / "evaluations" / "test_native_100.json",
        }
        missing = [label for label, path in required.items() if not path.exists()]
        candidates = [directory / f"candidate_{iteration:04d}.pt" for iteration in EXPECTED_CANDIDATES]
        missing_candidates = [path.name for path in candidates if not path.exists()]
        if missing or missing_candidates:
            errors.append(f"{name}: missing required={missing}, candidates={missing_candidates}")
            continue

        checkpoint = torch.load(required["checkpoint"], map_location="cpu", weights_only=False)
        training = checkpoint.get("training", {})
        checkpoint_hash = sha256(required["checkpoint"])
        actual_checkpoint_hashes[name] = checkpoint_hash
        if training.get("mode") != expected_mode or training.get("sync_mode") != expected_sync:
            errors.append(f"{name}: mode/sync mismatch: {training.get('mode')}/{training.get('sync_mode')}")
        expected_training = {
            "seed": 1, "iterations": 300, "rollout_steps": 512, "batch_size": 512,
            "update_epochs": 4, "validation_interval": 50, "validation_instances": 20,
        }
        mismatches = {
            key: {"observed": training.get(key), "expected": value}
            for key, value in expected_training.items() if training.get(key) != value
        }
        if mismatches:
            errors.append(f"{name}: training configuration mismatch: {mismatches}")

        history = json_load(required["training_history"])
        history_iterations = [int(float(row["iteration"])) for row in history]
        if history_iterations != list(range(1, 301)):
            errors.append(f"{name}: training history is not exactly iterations 1..300")
        validation = json_load(required["validation_history"])
        validation_iterations = [int(float(row["iteration"])) for row in validation]
        if validation_iterations != list(EXPECTED_CANDIDATES):
            errors.append(f"{name}: validation iterations are {validation_iterations}")

        candidate_rows = []
        for expected_iteration, candidate_path in zip(EXPECTED_CANDIDATES, candidates):
            candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
            observed_iteration = int(candidate.get("iteration", -1))
            if observed_iteration != expected_iteration:
                errors.append(f"{name}: {candidate_path.name} stores iteration {observed_iteration}")
            candidate_rows.append({
                "iteration": observed_iteration,
                "path": candidate_path.relative_to(raw).as_posix(),
                "sha256": sha256(candidate_path),
            })

        evaluation = json_load(required["test100"])
        rows = evaluation.get("rows", [])
        seeds = [int(row.get("instance_seed", -1)) for row in rows]
        tapes = {str(row.get("event_tape_hash")) for row in rows}
        if len(rows) != 100 or len(set(seeds)) != 100:
            errors.append(f"{name}: test100 has {len(rows)} rows and {len(set(seeds))} unique seeds")
        if evaluation.get("checkpoint_sha256") != checkpoint_hash:
            errors.append(f"{name}: evaluation/checkpoint SHA256 mismatch")
        if float(evaluation.get("summary", {}).get("all_tasks_completed", {}).get("mean", 0.0)) < 1.0:
            errors.append(f"{name}: not all test tasks completed")
        test_seed_sets.append(set(seeds))
        tape_sets.append(tapes)
        method_rows.append({
            "method": name,
            "mode": expected_mode,
            "sync_mode": expected_sync,
            "seed": int(training["seed"]),
            "iterations": int(training["iterations"]),
            "validation_instances": int(training["validation_instances"]),
            "best_iteration": int(checkpoint.get("best_iteration", -1)),
            "checkpoint_sha256": checkpoint_hash,
            "candidate_checkpoints": candidate_rows,
            "training_history_rows": len(history),
            "validation_history_rows": len(validation),
            "test_rows": len(rows),
            "unique_test_seeds": len(set(seeds)),
            "event_tape_hash_count": len(tapes),
        })

    same_test_bank = bool(test_seed_sets) and all(item == test_seed_sets[0] for item in test_seed_sets[1:])
    same_tape_bank = bool(tape_sets) and all(item == tape_sets[0] for item in tape_sets[1:])
    if not same_test_bank:
        errors.append("test100 instance banks differ across methods")
    if not same_tape_bank:
        errors.append("event tape banks differ across methods")

    acceptance_path = raw / "MECHANISM_ACCEPTANCE.json"
    acceptance = json_load(acceptance_path) if acceptance_path.exists() else {}
    checks = acceptance.get("checks", {})
    negative_preserved = (
        checks.get("adaptive_better_than_nogate") is False
        and checks.get("adaptive_better_than_singlehead") is False
    )
    if not negative_preserved:
        errors.append("Adaptive negative results are missing or not explicitly false")
    warnings.extend([
        "Historical checkpoint selection used validation20; Step 1 must re-evaluate candidates on validation100-A/B.",
        "This is one training seed and cannot support training-seed confidence intervals.",
        "Adaptive is significantly worse than NoGate and SingleHead in the frozen test100 result.",
    ])

    stale_audit_rows: dict[str, str] = {}
    old_audit_path = raw / "quick_artifact_audit.json"
    if old_audit_path.exists():
        old_audit = json_load(old_audit_path)
        stale_audit_rows = {
            str(row.get("directory")): str(row.get("sha256"))
            for row in old_audit.get("checkpoint_rows", [])
        }
        stale = [name for name, digest in actual_checkpoint_hashes.items() if stale_audit_rows.get(name) != digest]
        if stale:
            warnings.append(
                "Bundled quick_artifact_audit.json contains stale checkpoint hashes for: " + ", ".join(stale)
            )

    raw_files = sorted(path for path in raw.rglob("*") if path.is_file())
    if args.enforce_read_only:
        for path in raw_files:
            os.chmod(path, stat.S_IREAD)
    source_rows = []
    for source in args.source:
        source = source.resolve()
        if not source.exists():
            errors.append(f"missing source artifact: {source}")
            continue
        source_rows.append({"path": str(source), "size": source.stat().st_size, "sha256": sha256(source)})
    manifest = {
        "version": "phase1-seed1-300-artifact-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": True,
        "scope": "T5-10-48, training seed=1, 300 iterations, six learned methods",
        "repository": {
            "path": str(repo),
            "branch": git_value(repo, "branch", "--show-current"),
            "commit": git_value(repo, "rev-parse", "HEAD"),
        },
        "source_artifacts": source_rows,
        "raw_root": str(raw),
        "file_count": len(raw_files),
        "total_bytes": sum(path.stat().st_size for path in raw_files),
        "files": [file_record(path, raw) for path in raw_files],
    }
    audit = {
        "version": "phase1-seed1-300-artifact-audit-v1",
        "scope": manifest["scope"],
        "expected_methods": list(METHODS),
        "method_rows": method_rows,
        "checks": {
            "six_methods_complete": len(method_rows) == len(METHODS),
            "candidate_iterations_50_to_300": not any("candidate" in error for error in errors),
            "training_histories_1_to_300": not any("training history" in error for error in errors),
            "validation_histories_50_to_300": not any("validation iterations" in error for error in errors),
            "test100_unique": all(row["unique_test_seeds"] == 100 for row in method_rows),
            "same_fixed_test_bank": same_test_bank,
            "same_fixed_event_tape_bank": same_tape_bank,
            "evaluation_checkpoint_hashes_match": not any("evaluation/checkpoint" in error for error in errors),
            "adaptive_negative_result_preserved": negative_preserved,
            "source_hashes_recorded": len(source_rows) == len(args.source),
        },
        "known_negative_results": {
            "adaptive_better_than_nogate": checks.get("adaptive_better_than_nogate"),
            "adaptive_better_than_singlehead": checks.get("adaptive_better_than_singlehead"),
            "decision": acceptance.get("decision"),
        },
        "bundled_audit_checkpoint_hashes": stale_audit_rows,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "manifest": str(args.manifest), "audit": str(args.audit),
        "valid": not errors, "errors": len(errors), "warnings": len(warnings),
    }, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

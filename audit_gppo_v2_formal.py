from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_paper_gppo_v2 import (
    checkpoint_protocol_errors,
    evaluation_protocol_errors,
    methods,
)
from uav_assignment.gppo_v2 import METHOD_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit complete GPPO-v2 formal artifacts")
    parser.add_argument("--manifest", type=Path, default=Path("configs/gppo_v2_hard.json"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/paper_aligned/gppo_v2_hard/formal"),
    )
    parser.add_argument("--scope", choices=("train", "all"), default="all")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def read_json(path: Path, errors: list[str]) -> object | None:
    if not path.exists():
        errors.append(f"missing file: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return None


def audit_history(
    path: Path,
    expected_updates: list[int],
    label: str,
    errors: list[str],
) -> list[dict[str, object]] | None:
    payload = read_json(path, errors)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        errors.append(f"{label} must be a list of row dictionaries: {path}")
        return None
    observed: list[int] = []
    for index, row in enumerate(payload):
        try:
            observed.append(int(float(row["update"])))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{label} row {index} has invalid update: {path}")
    if observed != expected_updates:
        errors.append(
            f"{label} updates are {observed}, expected {expected_updates}: {path}"
        )
    return payload


def audit_event_log(
    path: Path,
    method_id: str,
    scales: set[str],
    eval_seeds: set[int],
    errors: list[str],
) -> int:
    if not path.exists():
        errors.append(f"missing event log: {path}")
        return 0
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        errors.append(f"cannot read event log {path}: {exc}")
        return 0
    if not lines:
        errors.append(f"empty event log: {path}")
        return 0
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSONL {path}:{line_number}: {exc}")
            continue
        if not isinstance(event, dict):
            errors.append(f"event is not an object {path}:{line_number}")
            continue
        if event.get("method_id") != method_id:
            errors.append(
                f"event method_id is {event.get('method_id')!r}, expected {method_id!r}: "
                f"{path}:{line_number}"
            )
        if str(event.get("scale")) not in scales:
            errors.append(f"event has invalid scale: {path}:{line_number}")
        try:
            seed = int(event["eval_seed"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"event has invalid eval_seed: {path}:{line_number}")
        else:
            if seed not in eval_seeds:
                errors.append(f"event eval_seed is out of range: {path}:{line_number}")
    return len(lines)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_methods = methods(manifest, include_supplementary=True)
    learned_methods = [method for method in selected_methods if METHOD_SPECS[method].learned]
    learned_seeds = [int(seed) for seed in manifest["learned_seeds"]]
    updates = int(manifest["updates"])
    validation_interval = int(manifest["validation_interval"])
    training_updates = list(range(1, updates + 1))
    validation_updates = list(range(validation_interval, updates + 1, validation_interval))
    scales = [str(scale) for scale in manifest["evaluation_scales"]]
    episodes = int(manifest["evaluation_episodes"])
    first_eval_seed = int(manifest["evaluation_seed"])
    eval_seeds = set(range(first_eval_seed, first_eval_seed + episodes))

    errors: list[str] = []
    checkpoint_count = 0
    training_history_count = 0
    validation_history_count = 0
    evaluation_count = 0
    event_log_count = 0
    event_count = 0
    evaluation_rows = 0

    for method_id in learned_methods:
        for seed in learned_seeds:
            run_root = args.output_root / "train" / method_id / f"seed_{seed}"
            checkpoint_path = run_root / "checkpoint.pt"
            training_history = audit_history(
                run_root / "training_history.json", training_updates, "training_history", errors
            )
            validation_history = audit_history(
                run_root / "validation_history.json",
                validation_updates,
                "validation_history",
                errors,
            )
            if training_history is not None:
                training_history_count += 1
            if validation_history is not None:
                validation_history_count += 1
            if not checkpoint_path.exists():
                errors.append(f"missing checkpoint: {checkpoint_path}")
                continue
            checkpoint_count += 1
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                errors.append(f"cannot load checkpoint {checkpoint_path}: {exc}")
                continue
            errors.extend(
                f"{checkpoint_path}: {error}"
                for error in checkpoint_protocol_errors(checkpoint, method_id, manifest, seed)
            )
            if isinstance(checkpoint, dict):
                if training_history is not None and checkpoint.get("history") != training_history:
                    errors.append(f"checkpoint history differs from disk: {checkpoint_path}")
                if validation_history is not None and checkpoint.get("validation_history") != validation_history:
                    errors.append(f"checkpoint validation history differs from disk: {checkpoint_path}")
                try:
                    best_update = int(checkpoint["best_update"])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"checkpoint best_update is missing or invalid: {checkpoint_path}")
                else:
                    if best_update not in validation_updates:
                        errors.append(f"checkpoint best_update is not a validation node: {checkpoint_path}")

    expected_checkpoints = len(learned_methods) * len(learned_seeds)
    training_expected = {
        "checkpoints": expected_checkpoints,
        "training_histories": expected_checkpoints,
        "validation_histories": expected_checkpoints,
    }
    training_observed = {
        "checkpoints": checkpoint_count,
        "training_histories": training_history_count,
        "validation_histories": validation_history_count,
    }
    for field, expected in training_expected.items():
        if training_observed[field] != expected:
            errors.append(
                f"observed {field}={training_observed[field]}, expected {expected}"
            )
    if args.scope == "train":
        report = {
            "valid": not errors,
            "scope": "train",
            "manifest_version": manifest["version"],
            "expected": training_expected,
            "observed": training_observed,
            "errors": errors,
        }
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        print(rendered)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(rendered + "\n", encoding="utf-8")
        if errors:
            raise SystemExit(1)
        return

    for method_id in selected_methods:
        spec = METHOD_SPECS[method_id]
        method_seeds = learned_seeds if spec.learned else [-1]
        for seed in method_seeds:
            run_root = args.output_root / "eval" / method_id / (
                f"seed_{seed}" if spec.learned else "baseline"
            )
            rows = read_json(run_root / "evaluation.json", errors)
            if isinstance(rows, list):
                evaluation_count += 1
                evaluation_rows += len(rows)
                errors.extend(
                    f"{run_root / 'evaluation.json'}: {error}"
                    for error in evaluation_protocol_errors(
                        rows, method_id, seed, manifest, scales, episodes
                    )
                )
            count = audit_event_log(
                run_root / "event_log.jsonl",
                method_id,
                set(scales),
                eval_seeds,
                errors,
            )
            if count:
                event_log_count += 1
                event_count += count

    summary = read_json(args.output_root / "summary" / "summary.json", errors)
    protocol_valid = bool(
        isinstance(summary, dict)
        and isinstance(summary.get("protocol_validation"), dict)
        and summary["protocol_validation"].get("valid") is True
    )
    if not protocol_valid:
        errors.append("summary protocol_validation.valid is not true")

    required_summary_artifacts = (
        "summary.json",
        "summary.csv",
        "per_seed.csv",
        "comparisons.csv",
        "parameter_audit.json",
        "curve_manifest.json",
        "validation_curves.csv",
        "validation_curves.png",
        "training_curves.csv",
        "training_curves.png",
        "acceptance_analysis.json",
    )
    summary_artifact_count = 0
    for name in required_summary_artifacts:
        path = args.output_root / "summary" / name
        if not path.exists():
            errors.append(f"missing summary artifact: {path}")
        elif path.stat().st_size <= 0:
            errors.append(f"empty summary artifact: {path}")
        else:
            summary_artifact_count += 1

    expected_evaluations = expected_checkpoints + sum(
        1 for method_id in selected_methods if not METHOD_SPECS[method_id].learned
    )
    expected_rows = expected_evaluations * len(scales) * episodes
    expected_counts = {
        "checkpoints": expected_checkpoints,
        "training_histories": expected_checkpoints,
        "validation_histories": expected_checkpoints,
        "evaluations": expected_evaluations,
        "event_logs": expected_evaluations,
        "evaluation_rows": expected_rows,
        "summary_artifacts": len(required_summary_artifacts),
    }
    observed_counts = {
        "checkpoints": checkpoint_count,
        "training_histories": training_history_count,
        "validation_histories": validation_history_count,
        "evaluations": evaluation_count,
        "event_logs": event_log_count,
        "evaluation_rows": evaluation_rows,
        "event_rows": event_count,
        "summary_artifacts": summary_artifact_count,
    }
    for field, expected in expected_counts.items():
        if field in training_expected:
            continue
        if observed_counts[field] != expected:
            errors.append(
                f"observed {field}={observed_counts[field]}, expected {expected}"
            )

    report = {
        "valid": not errors,
        "manifest_version": manifest["version"],
        "expected": expected_counts,
        "observed": observed_counts,
        "protocol_validation_valid": protocol_valid,
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

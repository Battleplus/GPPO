from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


PAPER_SCALES = ("T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92")
REQUIRED_NATIVE_LABELS = (
    "ppo_mlp_expected_task_message_none",
    "ppo_mlp_expected_task_message_event",
    "literal_expected_task_message_none",
    "literal_expected_task_message_event",
    "literal_no_gate_expected_task_message_event",
    "literal_single_head_expected_task_message_event",
    "literal_expected_task_message_periodic",
    "literal_expected_task_message_always",
)
EXPECTED_TRAINING = {
    "iterations": 2000,
    "rollout_steps": 512,
    "batch_size": 512,
    "update_epochs": 4,
    "validation_interval": 50,
    "validation_instances": 100,
    "learning_rate": 0.0002,
    "gamma": 0.99,
    "clip": 0.2,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit completed paper-faithful formal artifacts")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Also require post-matrix reports, baselines, generalization, plots, and gate sensitivity.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.root.glob("**/checkpoint.pt"))
    evaluations = sorted(args.root.glob("**/evaluations/test_native_*.json"))
    errors: list[str] = []
    warnings: list[str] = []
    if not checkpoints:
        errors.append("no checkpoint.pt files")
    if not evaluations:
        errors.append("no test_native evaluation files")

    checkpoint_rows = []
    checkpoint_hashes: set[str] = set()
    for checkpoint in checkpoints:
        try:
            payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_hash = sha256(checkpoint)
            checkpoint_hashes.add(checkpoint_hash)
            version = payload.get("version")
            training = payload.get("training", {})
            recovery_info = payload.get("recovery_info")
            if recovery_info:
                warnings.append(
                    f"discontinuous candidate recovery in {checkpoint}: "
                    f"iteration={recovery_info.get('source_iteration')}, "
                    "optimizer/RNG/episode offset were not recovered"
                )
            if version == "paper-faithful-literal-v1":
                for key, expected in EXPECTED_TRAINING.items():
                    observed = training.get(key)
                    if observed is None or not math.isclose(
                        float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12
                    ):
                        errors.append(
                            f"training config mismatch in {checkpoint}: {key}={observed}, expected {expected}"
                        )
                history = payload.get("history", [])
                iterations = [int(row.get("iteration", -1)) for row in history]
                if iterations != list(range(1, EXPECTED_TRAINING["iterations"] + 1)):
                    errors.append(f"incomplete/non-contiguous training history in {checkpoint}")
                validation_history = payload.get("validation_history", [])
                expected_validation = list(
                    range(
                        EXPECTED_TRAINING["validation_interval"],
                        EXPECTED_TRAINING["iterations"] + 1,
                        EXPECTED_TRAINING["validation_interval"],
                    )
                )
                observed_validation = [
                    int(row.get("iteration", -1)) for row in validation_history
                ]
                if observed_validation != expected_validation:
                    errors.append(f"incomplete/non-contiguous validation history in {checkpoint}")
                model_config = payload.get("model_config", {})
                if (
                    model_config.get("graph_mode") == "literal"
                    and str(training.get("sync_mode")) == "event"
                ):
                    diagnostic_path = checkpoint.parent / "gate_diagnostic.json"
                    if not diagnostic_path.exists():
                        errors.append(f"missing gate diagnostic for {checkpoint}")
                    else:
                        try:
                            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                            if int(diagnostic.get("instances", -1)) != args.instances:
                                errors.append(f"wrong gate diagnostic instance count: {diagnostic_path}")
                        except Exception as exc:
                            errors.append(f"gate diagnostic unreadable: {diagnostic_path}: {exc}")
            checkpoint_rows.append(
                {
                    "path": str(checkpoint),
                    "sha256": checkpoint_hash,
                    "version": version,
                    "training_scale": payload.get("environment", {}).get("scale", {}).get("name"),
                    "mode": payload.get("model_config", {}).get("graph_mode"),
                    "seed": training.get("seed"),
                    "active_parameter_count": payload.get("active_parameter_count"),
                    "recovery_info": recovery_info,
                    "resume_events": payload.get("resume_events", []),
                }
            )
        except Exception as exc:  # pragma: no cover - corrupt artifact path
            errors.append(f"checkpoint unreadable: {checkpoint}: {exc}")

    eval_payloads = []
    for path in evaluations:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            eval_payloads.append(payload)
            if int(payload.get("instances", -1)) != args.instances:
                errors.append(f"wrong instance count in {path}")
            rows = payload.get("rows", [])
            if len(rows) != args.instances:
                errors.append(f"wrong row count in {path}: {len(rows)}")
            instance_seeds = [int(row.get("instance_seed", -1)) for row in rows]
            if len(set(instance_seeds)) != args.instances:
                errors.append(f"duplicate/missing instance seeds in {path}")
            if payload.get("checkpoint_sha256") not in checkpoint_hashes:
                errors.append(f"evaluation checkpoint hash not found in formal root: {path}")
            if float(payload["summary"]["all_tasks_completed"]["mean"]) < 1.0 - 1e-12:
                errors.append(f"not all tasks complete in {path}")
            if not rows or not all("transition_trace" in row for row in rows):
                errors.append(f"missing trace rows in {path}")
            if int(payload.get("event_tape_hash_count", 0)) < 1:
                errors.append(f"missing event-tape hash in {path}")
        except Exception as exc:  # pragma: no cover - corrupt artifact path
            errors.append(f"evaluation unreadable: {path}: {exc}")

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for payload in eval_payloads:
        groups[(str(payload.get("training_scale")), str(payload.get("label")))].append(payload)
    group_rows = []
    for (scale, label), rows in sorted(groups.items()):
        seeds = sorted(int(row.get("training_seed", -1)) for row in rows)
        tape_sets = [set(row.get("event_tape_hashes", [])) for row in rows]
        same_tape_bank = bool(tape_sets) and all(item == tape_sets[0] for item in tape_sets[1:])
        if len(rows) != 5 or seeds != [1, 2, 3, 4, 5]:
            errors.append(f"expected five seeds for {scale}/{label}, observed {seeds}")
        if not same_tape_bank:
            errors.append(f"event-tape bank mismatch for {scale}/{label}")
        group_rows.append(
            {
                "training_scale": scale,
                "label": label,
                "seed_count": len(rows),
                "seeds": seeds,
                "same_event_tape_bank": same_tape_bank,
            }
        )

    observed_group_keys = set(groups)
    for scale in PAPER_SCALES:
        for label in REQUIRED_NATIVE_LABELS:
            if (scale, label) not in observed_group_keys:
                errors.append(f"missing required formal group: {scale}/{label}")

    # Communication comparisons must replay one fixed checkpoint.  A small
    # mismatch here would turn a communication ablation into a training
    # ablation, so compare hashes per scale/seed rather than only aggregate
    # metrics.
    by_key = {
        (
            str(payload.get("training_scale")),
            int(payload.get("training_seed", -1)),
            str(payload.get("label")),
        ): payload
        for payload in eval_payloads
    }
    for (scale, _seed, label), event_payload in sorted(by_key.items()):
        if not label.endswith("_event"):
            continue
        full_label = label[:-len("_event")] + "_always"
        full_payload = by_key.get((scale, _seed, full_label))
        if full_payload is None:
            continue
        if event_payload.get("checkpoint_sha256") != full_payload.get("checkpoint_sha256"):
            errors.append(f"event/full checkpoint mismatch for {scale}/seed{_seed}/{label}")
        if set(event_payload.get("event_tape_hashes", [])) != set(full_payload.get("event_tape_hashes", [])):
            errors.append(f"event/full tape mismatch for {scale}/seed{_seed}/{label}")

    if args.require_final:
        final_files = (
            "formal_summary_all.json",
            "baselines_test100.json",
            "formal_report.md",
            "GPPO_REPRODUCTION_REPORT_ZH.md",
            "fig8_reward.png",
            "fig8_realized_makespan.png",
            "fig8_validation_makespan.png",
            "cross_scale_general/unknown_generalization.json",
            "cross_scale_general/inference_scaling.json",
        )
        for relative in final_files:
            if not (args.root / relative).exists():
                errors.append(f"missing final artifact: {relative}")
        sensitivity_files = sorted(
            args.root.glob(
                "T5-10-48_literal_event/T5-10-48/literal_event_seed*/gate_sensitivity/test_*.json"
            )
        )
        if len(sensitivity_files) < 20:
            errors.append(
                f"expected at least 20 gate sensitivity files, observed {len(sensitivity_files)}"
            )
        for path in sensitivity_files:
            try:
                sensitivity = json.loads(path.read_text(encoding="utf-8"))
                if sensitivity.get("retrained") is not False:
                    errors.append(f"gate sensitivity is not marked inference-only: {path}")
            except Exception as exc:
                errors.append(f"gate sensitivity unreadable: {path}: {exc}")

    payload = {
        "version": "paper-faithful-formal-artifact-audit-v1",
        "root": str(args.root),
        "expected": {
            "paper_scales": list(PAPER_SCALES),
            "training_seeds": [1, 2, 3, 4, 5],
            "instances_per_test_evaluation": args.instances,
            "required_native_labels": list(REQUIRED_NATIVE_LABELS),
            "require_final": bool(args.require_final),
        },
        "observed": {
            "checkpoint_count": len(checkpoints),
            "evaluation_count": len(evaluations),
            "recovered_checkpoint_count": sum(
                1 for row in checkpoint_rows if row.get("recovery_info")
            ),
            "groups": group_rows,
        },
        "checkpoint_rows": checkpoint_rows,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "valid": not errors, "errors": len(errors), "warnings": len(warnings)}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

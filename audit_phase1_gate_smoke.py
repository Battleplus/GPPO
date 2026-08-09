from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

from run_phase1_gate_screening import completed_checkpoint, sha256_file, slug


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pre-registered 20-iteration gate Smoke jobs.")
    parser.add_argument("--root", type=Path, default=Path("outputs/gate_screening/smoke"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/GATE_DIAGNOSTIC_PROTOCOL.json"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--evaluation-instances", type=int, default=5)
    parser.add_argument("--diagnostic-instances", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    protocol = load_json(args.protocol)
    launch_manifest = load_json(root / "run_manifest.json")
    stage = protocol["smoke_protocol"]
    evaluator = Path(__file__).resolve().parent / "evaluate_paper_faithful.py"
    diagnostic = Path(__file__).resolve().parent / "diagnose_paper_faithful_gate.py"
    expected: list[tuple[dict[str, Any], int, Path]] = []
    missing: list[str] = []
    for variant in protocol["variants"]:
        for seed in stage["training_seeds"]:
            output_dir = root / slug(variant["name"]) / f"seed{seed}"
            checkpoint = output_dir / "checkpoint.pt"
            expected.append((variant, seed, output_dir))
            if not completed_checkpoint(checkpoint, int(stage["iterations"]), variant):
                missing.append(str(checkpoint))

    audit_path = args.output or (root / "SMOKE_AUDIT.json")
    if missing:
        audit_path.write_text(
            json.dumps(
                {
                    "version": "phase1-gate-smoke-audit-v1",
                    "valid": False,
                    "status": "incomplete",
                    "missing_or_mismatched": missing,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"valid": False, "incomplete": len(missing)}, ensure_ascii=False))
        raise SystemExit(2)

    commands: list[tuple[list[str], Path]] = []
    for variant, seed, output_dir in expected:
        checkpoint = output_dir / "checkpoint.pt"
        evaluation = output_dir / "smoke_validation_a.json"
        if not evaluation.is_file():
            commands.append(
                (
                    [
                        sys.executable,
                        str(evaluator),
                        "--checkpoint",
                        str(checkpoint),
                        "--split",
                        "validation_a",
                        "--instances",
                        str(args.evaluation_instances),
                        "--sync-mode",
                        "event",
                        "--output",
                        str(evaluation),
                    ],
                    output_dir / "smoke_validation_a.log",
                )
            )
        if variant["graph_mode"] == "literal":
            gate_output = output_dir / "smoke_gate_diagnostic.json"
            if not gate_output.is_file():
                commands.append(
                    (
                        [
                            sys.executable,
                            str(diagnostic),
                            "--checkpoint",
                            str(checkpoint),
                            "--split",
                            "validation_a",
                            "--instances",
                            str(args.diagnostic_instances),
                            "--output",
                            str(gate_output),
                        ],
                        output_dir / "smoke_gate_diagnostic.log",
                    )
                )
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(run_command, command, log_path) for command, log_path in commands]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"[{index}/{len(futures)}] smoke post-check completed")

    metric_fields = {
        "actor_loss",
        "value_loss",
        "policy_entropy",
        "ppo_approx_kl",
        "ppo_clip_fraction",
        "gate_gradient_l2",
        "realized_makespan",
    }
    records: list[dict[str, Any]] = []
    global_checks = {
        "all_21_jobs_present": len(expected) == 21,
        "launch_protocol_hash_matches": launch_manifest.get("protocol_sha256")
        == sha256_file(args.protocol),
        "no_test_artifacts": not any(root.rglob("*test*.json")),
    }
    for variant, seed, output_dir in expected:
        checkpoint_path = output_dir / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        history = load_json(output_dir / "training_history.json")
        validation = load_json(output_dir / "validation_history.json")
        evaluation = load_json(output_dir / "smoke_validation_a.json")
        values_finite = all(
            math.isfinite(float(row[field]))
            for row in history
            for field in metric_fields
        )
        gate_gradients = [float(row["gate_gradient_l2"]) for row in history]
        if variant["name"] == "Adaptive-warmup":
            gradient_semantics = all(value == 0.0 for value in gate_gradients)
            warmup_semantics = all(not row["gate_learning_enabled"] for row in history)
        elif variant["graph_mode"] == "literal":
            gradient_semantics = any(value > 0.0 for value in gate_gradients)
            warmup_semantics = all(row["gate_learning_enabled"] for row in history)
        else:
            gradient_semantics = all(value == 0.0 for value in gate_gradients)
            warmup_semantics = True
        checks = {
            "history_reaches_20": len(history) == 20 and int(history[-1]["iteration"]) == 20,
            "diagnostic_metrics_finite": values_finite,
            "validation_a_used": checkpoint["training"].get("validation_split") == "validation_a",
            "validation100_recorded": int(checkpoint["training"].get("validation_instances", 0)) == 100,
            "one_smoke_validation_at_20": len(validation) == 1
            and int(validation[0]["iteration"]) == 20,
            "gate_gradient_semantics": gradient_semantics,
            "warmup_semantics": warmup_semantics,
            "zero_invalid_actions": float(evaluation["summary"]["invalid_actions"]["mean"]) == 0.0,
            "all_evaluation_tasks_completed": float(
                evaluation["summary"]["all_tasks_completed"]["mean"]
            )
            == 1.0,
        }
        gate_summary = None
        if variant["graph_mode"] == "literal":
            gate_payload = load_json(output_dir / "smoke_gate_diagnostic.json")
            gate_summary = {
                "gate": gate_payload["gate"],
                "removed_attention_mass_per_uav": gate_payload[
                    "removed_attention_mass_per_uav"
                ],
                "uav_attention_output_l2_norm": gate_payload[
                    "uav_attention_output_l2_norm"
                ],
                "gate_gradient_l2_actual_ppo_probe": gate_payload[
                    "gate_gradient_l2_actual_ppo_probe"
                ],
            }
            checks["gate_nonconstant"] = float(gate_payload["gate"]["std"]) > 1e-6
            checks["gate_ppo_probe_gradient"] = (
                float(gate_payload["gate_gradient_l2_actual_ppo_probe"]) > 0.0
            )
        records.append(
            {
                "variant": variant["name"],
                "seed": seed,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checks": checks,
                "valid": all(checks.values()),
                "final_training_metrics": history[-1],
                "validation_a_summary": evaluation["summary"],
                "gate_diagnostic": gate_summary,
            }
        )

    payload = {
        "version": "phase1-gate-smoke-audit-v1",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "scope": "7 variants x 3 seeds x 20 iterations; validation-A only; no test",
        "global_checks": global_checks,
        "records": records,
        "valid": all(global_checks.values()) and all(row["valid"] for row in records),
    }
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit": str(audit_path), "valid": payload["valid"]}, ensure_ascii=False))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

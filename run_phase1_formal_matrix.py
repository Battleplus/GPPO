from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluate_paper_faithful_formal import discover_checkpoints, evaluation_directory


SCALES = ("T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92")
SEEDS = (1, 2, 3, 4, 5)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True, env=env)
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
        raise RuntimeError("Command failed:\n" + " ".join(command) + "\n" + "\n".join(tail))


def runner_command(
    python: str, repo: Path, output: Path, scales: tuple[str, ...], methods: tuple[str, ...],
    jobs: int, gate: dict[str, Any] | None = None,
) -> list[str]:
    gate = gate or {}
    command = [python, str(repo / "run_paper_faithful_formal.py")]
    for scale in scales:
        command += ["--scale", scale]
    command += ["--methods", *methods]
    for seed in SEEDS:
        command += ["--seed", str(seed)]
    command += [
        "--iterations", "2000", "--rollout-steps", "512", "--batch-size", "512",
        "--update-epochs", "4", "--validation-interval", "50",
        "--validation-instances", "100", "--validation-split", "validation_a",
        "--device", "cpu", "--rrelu-mode", "expected",
        "--gate-bias-init", str(gate.get("gate_bias_init", 0.0)),
        "--gate-warmup-iterations", str(gate.get("gate_warmup_iterations", 0)),
        "--gate-activation", str(gate.get("gate_activation", "sigmoid")),
        "--gate-scope", str(gate.get("gate_scope", "task_message")),
        "--jobs", str(jobs), "--output-root", str(output),
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Phase-1 four-scale/five-seed matrix.")
    parser.add_argument("--frozen-protocol", type=Path, default=Path("configs/PHASE1_FROZEN_PROTOCOL.json"))
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/paper_faithful/formal"))
    parser.add_argument("--legacy-literal-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    formal_root = args.formal_root.resolve()
    legacy_root = args.legacy_literal_root.resolve()
    protocol = read_json(args.frozen_protocol)
    if protocol.get("test_used_for_selection") is not False:
        raise RuntimeError("Frozen protocol does not prove validation-only selection")
    reselection = read_json(legacy_root / "PHASE1_CANDIDATE_RESELECTION.json")
    if reselection.get("valid") is not True or reselection.get("test_used_for_selection") is not False:
        raise RuntimeError("Legacy Literal candidates have not passed validation-A reselection")
    frozen_checkpoints = discover_checkpoints(legacy_root)
    if len(frozen_checkpoints) != 5 or any(path.name != "checkpoint_phase1_frozen.pt" for path in frozen_checkpoints):
        raise RuntimeError("Five validation-A frozen legacy Literal checkpoints are required")

    manifest_path = formal_root / "PHASE1_FORMAL_MATRIX_RUN.json"
    manifest: dict[str, Any] = {
        "version": "phase1-formal-matrix-run-v1",
        "frozen_protocol": str(args.frozen_protocol.resolve()),
        "frozen_protocol_sha256": sha256(args.frozen_protocol),
        "legacy_reselection_sha256": sha256(legacy_root / "PHASE1_CANDIDATE_RESELECTION.json"),
        "python": args.python,
        "training_code_root": str(repo),
        "validation_split": "validation_a",
        "test_used_for_training_or_selection": False,
        "batches": [],
        "status": "running",
    }
    write_manifest(manifest_path, manifest)
    env = dict(os.environ)
    env["PAPER_TORCH_THREADS"] = "1"
    log = formal_root / "phase1_formal_matrix.log"

    batches = [
        ("T5_primary", ("T5-10-48",), (
            "ppo_mlp:none", "ppo_mlp:event", "literal:none",
            "literal_no_gate:event", "literal_single_head:event",
        ), None),
        ("remaining_scales_primary", SCALES[1:], (
            "ppo_mlp:none", "ppo_mlp:event", "literal:none", "literal:event",
            "literal_no_gate:event", "literal_single_head:event",
        ), None),
    ]
    best = protocol["models"]["GPPO-Best"]
    best_variant = best["selected_variant"]
    existing_variants = {
        "Adaptive-current": "literal",
        "NoGate": "literal_no_gate",
        "SingleHead": "literal_single_head",
    }
    if best_variant not in existing_variants:
        batches.append((f"gppo_best_{best_variant.lower().replace('-', '_')}", SCALES, ("literal:event",), best))

    for name, scales, methods, gate in batches:
        output = formal_root / name
        command = runner_command(args.python, repo, output, scales, methods, args.jobs, gate)
        record = {"name": name, "scales": scales, "methods": methods, "output": str(output), "command": command, "status": "running"}
        manifest["batches"].append(record)
        write_manifest(manifest_path, manifest)
        run(command, log, env)
        record["status"] = "completed"
        write_manifest(manifest_path, manifest)
        run([
            args.python, str(repo / "evaluate_paper_faithful_formal.py"),
            "--root", str(output), "--instances", "100", "--split", "test",
            "--jobs", str(args.jobs), "--trace",
        ], log, env)
        record["test100_completed"] = True
        write_manifest(manifest_path, manifest)

    # Evaluate the reselected legacy T5 Literal models; the evaluator explicitly
    # prefers checkpoint_phase1_frozen.pt and leaves checkpoint.pt untouched.
    run([
        args.python, str(repo / "evaluate_paper_faithful_formal.py"),
        "--root", str(legacy_root), "--instances", "100", "--split", "test",
        "--jobs", str(args.jobs), "--trace",
    ], log, env)

    # Causal communication replay: identical checkpoint, instance bank and event tape.
    literal_event_checkpoints = frozen_checkpoints + [
        path for path in discover_checkpoints(formal_root / "remaining_scales_primary")
        if path.parent.name.startswith("literal_event_seed")
    ]
    if len(literal_event_checkpoints) != 20:
        raise RuntimeError(f"Expected 20 Literal-event checkpoints for communication replay, found {len(literal_event_checkpoints)}")
    for checkpoint in literal_event_checkpoints:
        evaluation_root = evaluation_directory(checkpoint)
        for sync_mode in ("none", "event", "periodic", "always"):
            output = evaluation_root / f"test_native_{sync_mode}_100.json"
            if output.is_file():
                prior = read_json(output)
                if prior.get("checkpoint_sha256") == sha256(checkpoint) and prior.get("instances") == 100:
                    continue
            run([
                args.python, str(repo / "evaluate_paper_faithful.py"), "--checkpoint", str(checkpoint),
                "--instances", "100", "--split", "test", "--trace",
                "--sync-mode", sync_mode, "--output", str(output),
            ], log, env)

    manifest["gppo_best_variant"] = best_variant
    manifest["gppo_best_source"] = (
        existing_variants.get(best_variant) or manifest["batches"][-1]["output"]
    )
    manifest["communication_replay_checkpoints"] = [str(path) for path in literal_event_checkpoints]
    manifest["status"] = "completed"
    manifest["valid"] = True
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "valid": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()

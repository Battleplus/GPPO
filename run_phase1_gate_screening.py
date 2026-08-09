from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(name: str) -> str:
    return name.lower().replace("adaptive-", "adaptive_").replace("-", "_")


def git_snapshot() -> tuple[str, bool]:
    repo = Path(__file__).resolve().parent
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    return commit, bool(status.strip())


def completed_checkpoint(path: Path, iterations: int, variant: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    training = checkpoint.get("training", {})
    model = checkpoint.get("model_config", {})
    return (
        int(training.get("iterations", -1)) == iterations
        and model.get("graph_mode") == variant["graph_mode"]
        and model.get("gate_scope") == variant["gate_scope"]
        and model.get("gate_activation") == variant["gate_activation"]
        and float(model.get("gate_bias_init", 0.0)) == float(variant["gate_bias_init"])
        and int(training.get("gate_warmup_iterations", 0))
        == int(variant["gate_warmup_iterations"])
        and training.get("validation_split") == "validation_a"
    )


def train_one(command: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(f"Training failed ({completed.returncode}):\n" + "\n".join(tail))
    return {"returncode": completed.returncode, "log": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-registered Phase-1 gate smoke/screening jobs.")
    parser.add_argument("--protocol", type=Path, default=Path("configs/GATE_DIAGNOSTIC_PROTOCOL.json"))
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path("configs/GATE_DIAGNOSTIC_PROTOCOL_AMENDMENT_001.json"),
    )
    parser.add_argument("--stage", choices=("smoke", "screening"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/gate_screening"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(args.protocol)
    if args.stage == "screening":
        amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
        if amendment.get("base_protocol_sha256") != protocol_hash:
            raise ValueError("Gate screening amendment does not match the frozen base protocol")
    stage = protocol["smoke_protocol" if args.stage == "smoke" else "screening_protocol"]
    iterations = int(stage["iterations"])
    validation_interval = iterations if args.stage == "smoke" else int(stage["validation_interval"])
    stage_root = args.output_root.resolve() / args.stage
    trainer = Path(__file__).resolve().parent / "train_paper_faithful.py"
    records: list[dict[str, Any]] = []
    pending: list[tuple[list[str], Path, dict[str, Any]]] = []

    for variant in protocol["variants"]:
        for seed in stage["training_seeds"]:
            output = stage_root / slug(variant["name"]) / f"seed{seed}"
            checkpoint = output / "checkpoint.pt"
            record = {
                "variant": variant["name"],
                "seed": seed,
                "output": str(output),
                "checkpoint": str(checkpoint),
            }
            if completed_checkpoint(checkpoint, iterations, variant):
                record["status"] = "reused"
                record["checkpoint_sha256"] = sha256_file(checkpoint)
                records.append(record)
                continue
            command = [
                sys.executable,
                str(trainer),
                "--mode",
                variant["graph_mode"],
                "--sync-mode",
                "event",
                "--scale",
                "T5-10-48",
                "--seed",
                str(seed),
                "--iterations",
                str(iterations),
                "--rollout-steps",
                str(stage["rollout_steps"]),
                "--batch-size",
                str(stage["batch_size"]),
                "--update-epochs",
                str(stage["update_epochs"]),
                "--validation-interval",
                str(validation_interval),
                "--validation-instances",
                str(stage["validation_instances"]),
                "--validation-split",
                "validation_a",
                "--rrelu-mode",
                "expected",
                "--gate-bias-init",
                str(variant["gate_bias_init"]),
                "--gate-scope",
                variant["gate_scope"],
                "--gate-activation",
                variant["gate_activation"],
                "--gate-warmup-iterations",
                str(variant["gate_warmup_iterations"]),
                "--device",
                args.device,
                "--output",
                str(output),
            ]
            resume = output / "resume_latest.pt"
            if resume.is_file():
                command.extend(("--resume-from", str(resume)))
            record["command"] = command
            record["status"] = "dry_run" if args.dry_run else "pending"
            records.append(record)
            if not args.dry_run:
                pending.append((command, output / "runner.log", record))

    stage_root.mkdir(parents=True, exist_ok=True)
    manifest_path = stage_root / "run_manifest.json"
    code_commit, worktree_dirty = git_snapshot()
    manifest = {
        "version": "phase1-gate-screening-run-v1",
        "code_commit": code_commit,
        "worktree_dirty_at_launch": worktree_dirty,
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_hash,
        "amendment": str(args.amendment) if args.stage == "screening" else None,
        "amendment_sha256": (
            sha256_file(args.amendment) if args.stage == "screening" else None
        ),
        "stage": args.stage,
        "test_used": False if args.stage == "smoke" else "reporting_only_after_selection",
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"manifest": str(manifest_path), "jobs": len(records)}, ensure_ascii=False))
        return

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(train_one, command, log_path): record
            for command, log_path, record in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            future.result()
            checkpoint = Path(record["checkpoint"])
            record["status"] = "completed"
            record["checkpoint_sha256"] = sha256_file(checkpoint)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"[{index}/{len(futures)}] {record['variant']} seed={record['seed']} completed")

    manifest["valid"] = all(record["status"] in {"completed", "reused"} for record in records)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "valid": manifest["valid"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

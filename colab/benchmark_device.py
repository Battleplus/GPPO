from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end GPPO training speed on CPU and CUDA."
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(device: str, args: argparse.Namespace, root: Path) -> dict[str, object]:
    output = root / device
    command = [
        sys.executable,
        "train_paper_faithful.py",
        "--mode",
        "literal",
        "--sync-mode",
        "event",
        "--scale",
        "T5-10-48",
        "--seed",
        "31415",
        "--iterations",
        str(args.iterations),
        "--rollout-steps",
        str(args.rollout_steps),
        "--batch-size",
        str(args.rollout_steps),
        "--update-epochs",
        "4",
        "--validation-interval",
        str(args.iterations),
        "--validation-instances",
        "5",
        "--device",
        device,
        "--output",
        str(output),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{device} benchmark failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return {
        "device": device,
        "seconds": elapsed,
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "seconds_per_iteration": elapsed / args.iterations,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gppo_device_benchmark_") as temp:
        root = Path(temp)
        rows = [run("cpu", args, root)]
        if torch.cuda.is_available():
            rows.append(run("cuda", args, root))
    by_device = {str(row["device"]): row for row in rows}
    speedup = None
    recommendation = "cpu"
    if "cuda" in by_device:
        speedup = float(by_device["cpu"]["seconds"]) / float(
            by_device["cuda"]["seconds"]
        )
        # A small apparent win is not enough to justify scarce Colab GPU quota.
        recommendation = "cuda" if speedup >= 1.15 else "cpu"
    payload = {
        "version": "paper-faithful-device-benchmark-v1",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runs": rows,
        "cuda_speedup_over_cpu": speedup,
        "recommended_device": recommendation,
        "decision_rule": "Use CUDA only when end-to-end speedup is at least 1.15x.",
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

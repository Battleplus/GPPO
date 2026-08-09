from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def discover_checkpoints(root: Path) -> list[Path]:
    """Prefer a validation-A-reselected Phase-1 checkpoint without deleting history."""
    frozen = {path.parent: path for path in root.glob("**/checkpoint_phase1_frozen.pt")}
    ordinary = {path.parent: path for path in root.glob("**/checkpoint.pt")}
    return [frozen.get(parent, checkpoint) for parent, checkpoint in sorted(ordinary.items())]


def evaluation_directory(checkpoint: Path) -> Path:
    name = "evaluations_phase1_frozen" if checkpoint.name == "checkpoint_phase1_frozen.pt" else "evaluations"
    return checkpoint.parent / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all completed paper-faithful formal checkpoints")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--eval-scale", action="append")
    parser.add_argument(
        "--sync-mode",
        action="append",
        choices=("none", "event", "periodic", "always"),
        help="Override communication mode during evaluation; repeat for multiple modes.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--trace", action="store_true", help="save per-transition audit traces")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    checkpoints = discover_checkpoints(root)
    if not checkpoints:
        raise FileNotFoundError(f"no completed checkpoints under {root}")
    sync_modes: list[str | None] = list(args.sync_mode or [None])
    pending: list[tuple[Path, str | None, str | None]] = [
        (checkpoint, scale, sync_mode)
        for checkpoint in checkpoints
        for scale in (args.eval_scale or [None])
        for sync_mode in sync_modes
    ]
    active: list[tuple[subprocess.Popen[str], Path, str | None, str | None]] = []
    while pending or active:
        while pending and len(active) < max(1, args.jobs):
            checkpoint, eval_scale, sync_mode = pending.pop(0)
            output_dir = evaluation_directory(checkpoint)
            output_dir.mkdir(parents=True, exist_ok=True)
            label = eval_scale or "native"
            if sync_mode is not None:
                label = f"{label}_{sync_mode}"
            output = output_dir / f"{args.split}_{label}_{args.instances}.json"
            command = [
                sys.executable,
                "evaluate_paper_faithful.py",
                "--checkpoint", str(checkpoint),
                "--split", args.split,
                "--instances", str(args.instances),
                "--output", str(output),
            ]
            if eval_scale:
                command.extend(("--eval-scale", eval_scale))
            if sync_mode:
                command.extend(("--sync-mode", sync_mode))
            if args.trace:
                command.append("--trace")
            log = (output_dir / f"{args.split}_{label}_{args.instances}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((process, checkpoint, eval_scale, sync_mode))
        remaining: list[tuple[subprocess.Popen[str], Path, str | None, str | None]] = []
        for process, checkpoint, eval_scale, sync_mode in active:
            code = process.poll()
            if code is None:
                remaining.append((process, checkpoint, eval_scale, sync_mode))
            elif code != 0:
                raise RuntimeError(
                    f"evaluation failed: {checkpoint} ({eval_scale}, {sync_mode}), code={code}"
                )
            else:
                mode_label = eval_scale or "native"
                if sync_mode is not None:
                    mode_label = f"{mode_label}/{sync_mode}"
                print(f"completed {checkpoint.parent} ({mode_label})")
        active = remaining
        if active:
            time.sleep(2.0)


if __name__ == "__main__":
    main()

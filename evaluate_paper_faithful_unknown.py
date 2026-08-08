from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


UNSEEN_SCALES = (
    "T6-9-44", "T7-11-57", "T8-12-62", "T9-9-50",
    "T11-9-58", "T12-11-64", "T14-9-74", "T18-9-84",
)

KNOWN_SCALES = ("T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed-capacity model on unseen paper-faithful scales")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--known-instances", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", default=list(UNSEEN_SCALES))
    parser.add_argument(
        "--base-root",
        type=Path,
        help="Formal matrix root containing literal_event_seed1 base checkpoints.",
    )
    parser.add_argument("--known-scales", nargs="+", default=list(KNOWN_SCALES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.checkpoint.parent / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    def evaluate(checkpoint: Path, scale: str, label: str, instances: int) -> dict[str, object]:
        output = root / f"{label}_{scale}_{instances}.json"
        command = [
            python, "evaluate_paper_faithful.py", "--checkpoint", str(checkpoint),
            "--split", "test", "--instances", str(instances),
            "--eval-scale", scale, "--output", str(output),
        ]
        subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
        return json.loads(output.read_text(encoding="utf-8"))

    def find_bases(scale: str) -> list[Path]:
        if args.base_root is None:
            raise FileNotFoundError("--base-root is required for known-scale base comparisons")
        matches = sorted(
            args.base_root.glob(f"**/{scale}/literal_event_seed*/checkpoint.pt")
        )
        if not matches:
            raise FileNotFoundError(f"no literal_event base checkpoints for {scale} under {args.base_root}")
        return matches

    known_rows: list[dict[str, object]] = []
    if args.base_root is not None:
        for scale in args.known_scales:
            base_checkpoints = find_bases(scale)
            general = evaluate(args.checkpoint, scale, "general_known", args.known_instances)
            base_payloads = []
            for base_checkpoint in base_checkpoints:
                native = base_checkpoint.parent / "evaluations" / f"test_native_{args.known_instances}.json"
                if native.exists():
                    base_payloads.append(json.loads(native.read_text(encoding="utf-8")))
                else:
                    base_payloads.append(
                        evaluate(base_checkpoint, scale, f"base_known_seed{base_checkpoint.parent.name.removeprefix('literal_event_seed')}", args.known_instances)
                    )
            general_makespan = float(general["summary"]["realized_makespan"]["mean"])
            base_values = [float(item["summary"]["realized_makespan"]["mean"]) for item in base_payloads]
            base_makespan = float(np.mean(base_values))
            known_rows.append(
                {
                    "scale": scale,
                    "instances": args.known_instances,
                    "base_checkpoint_count": len(base_checkpoints),
                    "base_checkpoints": [str(path) for path in base_checkpoints],
                    "base_checkpoint_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in base_checkpoints],
                    "base_per_seed_makespan": base_values,
                    "general_realized_makespan": general_makespan,
                    "base_realized_makespan": base_makespan,
                    "generalization_error_percent": 100.0 * abs(general_makespan - base_makespan) / max(abs(base_makespan), 1e-9),
                    "completion_rate": float(general["summary"]["completion_rate"]["mean"]),
                }
            )
    rows = []
    for scale in args.scales:
        payload = evaluate(args.checkpoint, scale, "unknown", args.instances)
        makespan = float(payload["summary"]["realized_makespan"]["mean"])
        rows.append({
            "scale": scale,
            "instances": args.instances,
            "realized_makespan": makespan,
            "completion_rate": float(payload["summary"]["completion_rate"]["mean"]),
            "communication_bytes": float(payload["summary"]["communication_bytes"]["mean"]),
        })
    result = {
        "version": "paper-faithful-unknown-generalization-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "known_scale_comparison": known_rows,
        "unknown_scale_count": len(rows),
        "rows": rows,
        "protocol_note": "Known-scale generalization errors compare the cross-scale model with a dedicated literal-event base checkpoint on the same test bank. Unknown scales are fixed before evaluation and are reported as out-of-register stress tests; no unknown result is used for checkpoint selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "unknown_scale_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

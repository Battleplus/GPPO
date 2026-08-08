from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_SCALES = ("T10-10-24", "T10-10-40", "T10-10-56", "T10-10-72", "T10-10-88")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure inference scaling at fixed UAV count")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--instances", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.checkpoint.parent / "inference_scaling"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scale in args.scales:
        output = root / f"{scale}_{args.instances}.json"
        subprocess.run(
            [
                sys.executable,
                "evaluate_paper_faithful.py",
                "--checkpoint",
                str(args.checkpoint),
                "--split",
                "test",
                "--instances",
                str(args.instances),
                "--eval-scale",
                scale,
                "--output",
                str(output),
            ],
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        normalized = scale.upper().removeprefix("T").split("-")
        uavs, parents, subtasks = (int(value) for value in normalized)
        rows.append(
            {
                "scale": scale,
                "uavs": uavs,
                "parent_tasks": parents,
                "subtasks": subtasks,
                "instances": args.instances,
                "mean_inference_ms": float(payload["summary"]["mean_inference_ms"]["mean"]),
                "p50_inference_ms": float(payload["inference_latency_ms"]["p50"]),
                "p95_inference_ms": float(payload["inference_latency_ms"]["p95"]),
                "estimated_forward_flops": int(payload["estimated_forward_flops"]),
                "completion_rate": float(payload["summary"]["completion_rate"]["mean"]),
            }
        )
    x = np.asarray([row["subtasks"] for row in rows], dtype=np.float64)
    y = np.asarray([row["mean_inference_ms"] for row in rows], dtype=np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(rows) >= 2 else float("nan")
    result = {
        "version": "paper-faithful-inference-scaling-v1",
        "checkpoint": str(args.checkpoint),
        "fixed_uavs": sorted({int(row["uavs"]) for row in rows}),
        "rows": rows,
        "pearson_r_subtasks_mean_inference_ms": pearson,
        "protocol_note": "Scales are fixed before evaluation, use the same general checkpoint, and are not used for checkpoint selection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "rows": len(rows), "pearson_r": pearson}, ensure_ascii=False))


if __name__ == "__main__":
    main()

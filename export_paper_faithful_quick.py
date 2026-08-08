from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


METHOD_DIRS = (
    "literal_event_seed1",
    "ppo_mlp_none_seed1",
    "ppo_mlp_event_seed1",
    "literal_none_seed1",
    "literal_no_gate_event_seed1",
    "literal_single_head_event_seed1",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_compact_evaluation(source: Path, target: Path, checkpoint_path: str) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["checkpoint"] = checkpoint_path
    for row in payload.get("rows", []):
        row.pop("transition_trace", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the lightweight quick-validation GitHub artifact")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads((args.source / "quick_mechanism_result.json").read_text(encoding="utf-8"))
    audit = json.loads((args.source / "quick_artifact_audit.json").read_text(encoding="utf-8"))
    if result.get("scope") != "single-scale, single-training-seed, 100-iteration quick mechanism validation":
        raise ValueError("unexpected quick-validation scope")
    if audit.get("valid") is not True:
        raise ValueError("source quick artifact audit is not valid")

    args.output.mkdir(parents=True, exist_ok=True)
    for name in (
        "QUICK_MECHANISM_REPORT_ZH.md",
        "quick_mechanism_result.json",
        "quick_artifact_audit.json",
        "quick_summary.json",
        "baselines_test100.json",
    ):
        shutil.copy2(args.source / name, args.output / name)

    scale_source = args.source / "T5-10-48"
    scale_output = args.output / "T5-10-48"
    for method in METHOD_DIRS:
        source_dir = scale_source / method
        output_dir = scale_output / method
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoint.pt", "training_history.json", "validation_history.json"):
            shutil.copy2(source_dir / name, output_dir / name)
        checkpoint_relative = f"T5-10-48/{method}/checkpoint.pt"
        write_compact_evaluation(
            source_dir / "evaluations" / "test_native_100.json",
            output_dir / "test_native_100_compact.json",
            checkpoint_relative,
        )
        if method == "literal_event_seed1":
            write_compact_evaluation(
                source_dir / "evaluations" / "test_native_always_100.json",
                output_dir / "test_native_always_100_compact.json",
                checkpoint_relative,
            )

    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    manifest = {
        "version": "paper-faithful-quick-github-export-v1",
        "scope": result["scope"],
        "excluded": [
            "optimizer resume checkpoints",
            "candidate checkpoints",
            "runner logs",
            "per-transition traces (compact evaluations retain per-instance metrics)",
        ],
        "files": [
            {
                "path": path.relative_to(args.output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    (args.output / "EXPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"saved": str(args.output), "files": len(files) + 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()

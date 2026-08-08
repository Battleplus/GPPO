from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "outputs" / "paper_faithful" / "pilot20_v3" / "T5-10-48",
    )
    args = parser.parse_args()
    pilot = args.root.resolve()
    expected_methods = tuple(
        path.name
        for path in sorted(pilot.iterdir())
        if path.is_dir() and (path / "checkpoint.pt").is_file()
    )
    checkpoints = [pilot / method / "checkpoint.pt" for method in expected_methods]
    candidates = sorted(pilot.glob("*/candidate_*.pt"))
    evaluations = sorted((pilot / "evaluation").glob("*.json"))
    required = checkpoints + candidates + evaluations
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    summaries = json.loads((pilot / "evaluation" / "summary.json").read_text(encoding="utf-8"))
    import torch
    first_checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    training_payload = first_checkpoint.get("training", {})
    payload = {
        "manifest_version": f"paper-faithful-{pilot.parent.name}-manifest",
        "classification": "diagnostic-only",
        "scale": pilot.name,
        "training_seed": training_payload.get("seed"),
        "iterations": training_payload.get("iterations"),
        "train_instances": 100,
        "validation_instances": 100,
        "test_instances": 100,
        "disjoint_banks": True,
        "methods": list(expected_methods),
        "candidate_checkpoint_count": len(candidates),
        "evaluation_artifact_count": len(evaluations),
        "summary": summaries,
        "files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in required
        ],
    }
    output = pilot / "pilot_manifest.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(output), "files": len(required)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

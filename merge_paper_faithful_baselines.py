from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-scale paper-faithful baseline evaluations")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    if not payloads:
        raise ValueError("no baseline inputs")
    results = [row for payload in payloads for row in payload.get("results", [])]
    seeds = sorted({int(seed) for payload in payloads for seed in payload.get("policy_seeds", [])})
    output = {
        "version": "paper-faithful-baselines-v1",
        "protocol": "configs/paper_faithful_protocol.json",
        "policies": payloads[0].get("policies", []),
        "policy_seeds": seeds,
        "results": results,
        "oracle_note": payloads[0].get("oracle_note", ""),
        "source_files": [str(path) for path in args.input],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "results": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

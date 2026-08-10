"""Verify all-off Phase-1B equivalence on the frozen T5 test100 bank."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uav_assignment.disturbances import DisturbanceConfig  # noqa: E402
from uav_assignment.paper_faithful_env import (  # noqa: E402
    PAPER_SCALES,
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
    deterministic_instance_seeds,
)
from uav_assignment.phase1b_env import Phase1BPaperFaithfulUAVEnv  # noqa: E402


def equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def verify_seed(seed: int, max_decisions: int) -> dict[str, Any]:
    config = PaperFaithfulConfig(scale=PAPER_SCALES[0], instance_seed=seed)
    baseline = PaperFaithfulUAVEnv(config)
    candidate = Phase1BPaperFaithfulUAVEnv(config, DisturbanceConfig())
    left, right = baseline.reset(seed=seed), candidate.reset(seed=seed)
    checks = {
        "reset_observation": equal(left, right),
        "base_event_tape": equal(baseline._event_tape, candidate._event_tape),
        "disturbance_tape_empty": candidate.disturbance_engine is not None and not candidate.disturbance_engine.tape.events,
    }
    decisions = 0
    while decisions < max_decisions and checks["reset_observation"]:
        legal = np.flatnonzero(left["action_mask"])
        action = int(legal[(seed + decisions) % len(legal)])
        left, lr, ld, li = baseline.step(action, sync_mode="event")
        right, rr, rd, ri = candidate.step(action, sync_mode="event")
        decisions += 1
        checks[f"decision_{decisions}"] = (
            equal(left, right)
            and lr == rr
            and ld == rd
            and equal(li, ri)
            and equal(baseline.true_observation(), candidate.true_observation())
            and baseline.metrics() == candidate.metrics()
        )
        if ld:
            break
    return {
        "instance_seed": seed,
        "decisions": decisions,
        "valid": all(checks.values()),
        "checks": checks,
        "final_metrics": baseline.metrics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--max-decisions", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("deliverables/phase1b/ALL_OFF_EQUIVALENCE_TEST100.json"))
    args = parser.parse_args()
    seeds = deterministic_instance_seeds(PAPER_SCALES[0], args.instances, split="test")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(verify_seed, seeds, [args.max_decisions] * len(seeds)))
    payload = {
        "schema_version": "phase1b-all-off-equivalence-v1",
        "scale": PAPER_SCALES[0].name,
        "split": "frozen-test100-equivalence-only; not used for calibration",
        "instances": args.instances,
        "max_decisions": args.max_decisions,
        "rows": rows,
        "valid": len(rows) == args.instances and all(row["valid"] for row in rows),
        "inference_implication": (
            "Exact node/action-mask observations imply identical deterministic checkpoint "
            "inference for every verified decision."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"valid": payload["valid"], "instances": len(rows), "decisions": sum(row["decisions"] for row in rows)}, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

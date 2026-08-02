from __future__ import annotations

import argparse
import json
import math


def priority_share_upper_bound(
    *, decay: float, other_type_coverage_floor: float, objective_count: int = 4
) -> float:
    """Optimistic maximum share of one deadline-weighted task objective.

    The preferred type is assigned at normalized time zero with coverage one.
    Every other required type keeps ``other_type_coverage_floor`` and is
    assigned at the deadline, where its temporal weight is ``exp(-decay)``.
    """
    if decay <= 0:
        raise ValueError("decay must be positive")
    if not 0 <= other_type_coverage_floor <= 1:
        raise ValueError("other_type_coverage_floor must be in [0, 1]")
    if objective_count < 2:
        raise ValueError("objective_count must be at least two")
    competing_mass = (
        (objective_count - 1)
        * other_type_coverage_floor
        * math.exp(-decay)
    )
    return 1.0 / (1.0 + competing_mass)


def minimum_decay_for_target(
    *, target_share: float, other_type_coverage_floor: float, objective_count: int = 4
) -> float:
    """Minimum temporal decay whose optimistic bound reaches target_share."""
    if not 0 < target_share < 1:
        raise ValueError("target_share must be in (0, 1)")
    if not 0 < other_type_coverage_floor <= 1:
        raise ValueError("other_type_coverage_floor must be in (0, 1]")
    if objective_count < 2:
        raise ValueError("objective_count must be at least two")
    ratio = (1.0 / target_share - 1.0) / (
        (objective_count - 1) * other_type_coverage_floor
    )
    return max(0.0, -math.log(ratio))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit attainability of the PCRL assignment-priority mix"
    )
    parser.add_argument("--target-share", type=float, default=0.70)
    parser.add_argument("--coverage-floor", type=float, default=0.442)
    parser.add_argument("--objective-count", type=int, default=4)
    parser.add_argument("--decays", nargs="+", type=float, default=(1.0, 2.0, 3.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    minimum_decay = minimum_decay_for_target(
        target_share=args.target_share,
        other_type_coverage_floor=args.coverage_floor,
        objective_count=args.objective_count,
    )
    result = {
        "target_share": args.target_share,
        "other_type_coverage_floor": args.coverage_floor,
        "objective_count": args.objective_count,
        "minimum_decay_for_optimistic_attainability": minimum_decay,
        "bounds": [
            {
                "decay": decay,
                "upper_bound": priority_share_upper_bound(
                    decay=decay,
                    other_type_coverage_floor=args.coverage_floor,
                    objective_count=args.objective_count,
                ),
            }
            for decay in args.decays
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

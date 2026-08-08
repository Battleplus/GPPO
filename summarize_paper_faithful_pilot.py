from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired paper-faithful pilot evaluations")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", default="ppo_mlp")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def mean_ci(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    half = float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))
    return {"mean": mean, "ci95_low": mean - half, "ci95_high": mean + half, "ci95_half_width": half}


def main() -> None:
    args = parse_args()
    input_paths: list[Path] = []
    for path in args.inputs:
        matches = sorted(path.parent.glob(path.name)) if any(char in path.name for char in "*?[") else [path]
        input_paths.extend(matches)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    by_mode = {
        str(payload.get("label", payload["mode"])): payload for payload in payloads
    }
    reference = by_mode[args.reference]
    reference_rows = {int(row["instance_seed"]): row for row in reference["rows"]}
    comparisons: dict[str, object] = {}

    def method_key(prefix: str) -> str:
        if prefix in by_mode:
            return prefix
        candidates = sorted(
            key
            for key in by_mode
            if key.startswith(prefix + "_")
            and not any(
                key.startswith(other + "_")
                for other in ("literal_no_gate", "literal_single_head")
                if prefix == "literal"
            )
        )
        if len(candidates) != 1:
            raise ValueError(f"cannot resolve method prefix {prefix}: {candidates}")
        return candidates[0]

    def paired(left: str, right: str) -> dict[str, object]:
        left_rows = {
            int(row["instance_seed"]): row for row in by_mode[left]["rows"]
        }
        right_rows = {
            int(row["instance_seed"]): row for row in by_mode[right]["rows"]
        }
        if set(left_rows) != set(right_rows):
            raise ValueError(f"instance mismatch for {left} vs {right}")
        makespan_delta = np.asarray([
            float(left_rows[seed]["realized_makespan"])
            - float(right_rows[seed]["realized_makespan"])
            for seed in sorted(left_rows)
        ])
        reward_delta = np.asarray([
            float(left_rows[seed]["episode_return"])
            - float(right_rows[seed]["episode_return"])
            for seed in sorted(left_rows)
        ])
        return {
            "left": left,
            "right": right,
            "realized_makespan_delta": mean_ci(makespan_delta),
            "episode_return_delta": mean_ci(reward_delta),
            "left_win_rate_makespan": float(np.mean(makespan_delta < 0.0)),
        }

    for mode, payload in by_mode.items():
        rows = {int(row["instance_seed"]): row for row in payload["rows"]}
        if set(rows) != set(reference_rows):
            raise ValueError(f"instance mismatch for {mode}")
        for seed in sorted(rows):
            left_hash = rows[seed].get("event_tape_hash")
            right_hash = reference_rows[seed].get("event_tape_hash")
            if left_hash is not None and right_hash is not None and left_hash != right_hash:
                raise ValueError(f"event tape mismatch for {mode}, instance {seed}")
        makespan_delta = np.asarray([
            float(rows[seed]["realized_makespan"]) - float(reference_rows[seed]["realized_makespan"])
            for seed in sorted(rows)
        ])
        reward_delta = np.asarray([
            float(rows[seed]["episode_return"]) - float(reference_rows[seed]["episode_return"])
            for seed in sorted(rows)
        ])
        comparisons[mode] = {
            "realized_makespan_delta": mean_ci(makespan_delta),
            "episode_return_delta": mean_ci(reward_delta),
            "win_rate_makespan": float(np.mean(makespan_delta < 0.0)),
        }
    output = {
        "version": "paper-faithful-pilot-summary-v1",
        "reference": args.reference,
        "scale": reference["scale"],
        "split": reference["split"],
        "instances": reference["instances"],
        "methods": {
            mode: {
                "summary": payload["summary"],
                "active_parameter_count": payload["active_parameter_count"],
                "best_iteration": payload["best_iteration"],
            }
            for mode, payload in by_mode.items()
        },
        "paired_comparisons": comparisons,
        "ablation_comparisons": {
            **(
                {
                    "literal_minus_no_gate": paired(
                        method_key("literal"), method_key("literal_no_gate")
                    )
                }
                if any(key.startswith("literal_no_gate") for key in by_mode)
                else {}
            ),
            **(
                {
                    "literal_minus_single_head": paired(
                        method_key("literal"), method_key("literal_single_head")
                    )
                }
                if any(key.startswith("literal_single_head") for key in by_mode)
                else {}
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "paired_comparisons": comparisons}, ensure_ascii=False))


if __name__ == "__main__":
    main()

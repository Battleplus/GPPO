from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_paper_gppo_v2 import (
    CHECKPOINT_VERSION,
    TRAINING_HYPERPARAMETER_FIELDS,
    expected_evaluation_hash,
    expected_scenario_hash,
    evaluation_protocol_errors,
    methods as selected_methods,
)
from uav_assignment.gppo_v2 import CORE_METHOD_IDS, METHOD_SPECS, implementation_hash


METRICS = (
    "makespan",
    "completion_rate",
    "deadline_completion_rate",
    "mission_success",
    "deadline_remaining_tasks",
    "throughput",
    "invalid_actions",
    "communication_events",
    "heartbeat_messages",
    "reallocated_tasks",
    "reallocation_successes",
    "reallocation_success_rate",
    "return",
)
LOWER_IS_BETTER = {
    "makespan",
    "deadline_remaining_tasks",
    "invalid_actions",
    "communication_events",
    "heartbeat_messages",
}
COMPARISONS = (
    ("ppo_event", "ppo_none"),
    ("gppo_event", "ppo_event"),
    ("gppo_none", "ppo_none"),
    ("gppo_event", "gppo_none"),
    ("gppo_periodic", "gppo_event"),
    ("gppo_always", "gppo_event"),
    ("gppo_event", "gppo_event_no_gate"),
    ("gppo_event", "gppo_event_single_head"),
    ("gppo_event", "random_event"),
    ("gppo_event", "greedy_event"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GPPO-v2 ablations")
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    supplementary = parser.add_mutually_exclusive_group()
    supplementary.add_argument(
        "--include-supplementary",
        dest="include_supplementary",
        action="store_true",
        help="require manifest supplementary methods (the frozen formal default)",
    )
    supplementary.add_argument(
        "--exclude-supplementary",
        dest="include_supplementary",
        action="store_false",
        help="validate an explicitly exploratory core-only result set",
    )
    parser.set_defaults(include_supplementary=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def t_critical_95(n: int) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    return table.get(n, 1.96)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    half = (
        0.0
        if len(values) <= 1
        else float(t_critical_95(len(values)) * values.std(ddof=1) / np.sqrt(len(values)))
    )
    return mean, mean - half, mean + half


def aggregate(rows: list[dict[str, object]], metric: str) -> float:
    if metric == "reallocation_success_rate":
        attempts = sum(float(row["reallocated_tasks"]) for row in rows)
        successes = sum(float(row["reallocation_successes"]) for row in rows)
        return successes / attempts if attempts else 0.0
    return float(np.mean([float(row[metric]) for row in rows]))


def wilson_interval(successes: float, attempts: float) -> tuple[float, float]:
    if attempts <= 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (proportion + z * z / (2.0 * attempts)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / attempts
        + z * z / (4.0 * attempts * attempts)
    ) / denominator
    return max(0.0, float(center - half)), min(1.0, float(center + half))


def sign_flip_pvalue(values: np.ndarray) -> float:
    if len(values) == 0:
        return 1.0
    observed = abs(float(values.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        if abs(float(np.mean(values * np.asarray(signs)))) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.ones(len(pvalues), dtype=np.float64)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * pvalues[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def effect_label(metric: str, lower: float, upper: float) -> str:
    if lower <= 0 <= upper:
        return "inconclusive"
    beneficial = upper < 0 if metric in LOWER_IS_BETTER else lower > 0
    return "positive" if beneficial else "negative"


def negative_result_records(
    comparisons: list[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for comparison in comparisons:
        for metric in METRICS:
            effect = comparison.get(f"{metric}_effect")
            if effect not in {"negative", "inconclusive"}:
                continue
            records.append(
                {
                    "comparison": comparison["comparison"],
                    "left": comparison["left"],
                    "right": comparison["right"],
                    "scale": comparison["scale"],
                    "paired_training_seeds": comparison["paired_training_seeds"],
                    "metric": metric,
                    "effect": effect,
                    "delta": comparison[f"{metric}_delta"],
                    "delta_ci95_lower": comparison[f"{metric}_delta_ci95_lower"],
                    "delta_ci95_upper": comparison[f"{metric}_delta_ci95_upper"],
                    "p_raw": comparison[f"{metric}_p_raw"],
                    "p_holm": comparison[f"{metric}_p_holm"],
                }
            )
    return records


def validate_protocol(
    raw: list[dict[str, object]],
    allow_incomplete: bool,
    manifest: dict[str, object] | None = None,
    include_supplementary: bool = True,
) -> dict[str, object]:
    methods = {str(row["method_id"]) for row in raw}
    expected_methods = (
        selected_methods(manifest, include_supplementary)
        if manifest is not None
        else list(CORE_METHOD_IDS)
    )
    missing = sorted(set(expected_methods) - methods)
    unexpected = sorted(methods - set(expected_methods))
    hashes = {str(row["scenario_hash"]) for row in raw}
    implementation_hashes = {str(row["implementation_hash"]) for row in raw}
    scenario_versions = {str(row["scenario_version"]) for row in raw}
    errors: list[str] = []
    if missing:
        errors.append(f"missing methods: {missing}")
    if unexpected:
        errors.append(f"unexpected methods: {unexpected}")
    if len(hashes) != 1:
        errors.append(f"scenario hashes differ: {sorted(hashes)}")
    if len(implementation_hashes) != 1:
        errors.append(
            f"implementation hashes differ: {sorted(implementation_hashes)}"
        )
    if len(scenario_versions) != 1:
        errors.append(f"scenario versions differ: {sorted(scenario_versions)}")
    observed_scales = {str(row["scale"]) for row in raw}
    scales = (
        [str(scale) for scale in manifest["evaluation_scales"]]
        if manifest is not None
        else sorted(observed_scales)
    )
    if manifest is not None:
        if observed_scales != set(scales):
            errors.append(
                f"evaluation scales are {sorted(observed_scales)}, expected {scales}"
            )
        expected_hash = expected_evaluation_hash(manifest, scales)
        if hashes != {expected_hash}:
            errors.append(
                f"scenario hash is {sorted(hashes)}, expected {[expected_hash]}"
            )
        expected_impl = implementation_hash()
        if implementation_hashes != {expected_impl}:
            errors.append(
                f"implementation hash is {sorted(implementation_hashes)}, "
                f"expected {[expected_impl]}"
            )
        expected_version = str(manifest["version"])
        if scenario_versions != {expected_version}:
            errors.append(
                f"scenario version is {sorted(scenario_versions)}, "
                f"expected {[expected_version]}"
            )
        expected_learned_seeds = sorted(int(seed) for seed in manifest["learned_seeds"])
        expected_eval_seeds = list(
            range(
                int(manifest["evaluation_seed"]),
                int(manifest["evaluation_seed"])
                + int(manifest["evaluation_episodes"]),
            )
        )
        expected_episodes = int(manifest["evaluation_episodes"])
        checkpoint_scenario_hash = expected_scenario_hash(manifest)
    else:
        expected_learned_seeds = [1, 2, 3, 4, 5]
        expected_eval_seeds = []
        expected_episodes = 0
        checkpoint_scenario_hash = ""
    for method_id in sorted(methods):
        spec = METHOD_SPECS.get(method_id)
        if spec is None:
            errors.append(f"unknown method: {method_id}")
            continue
        method_rows = [row for row in raw if row["method_id"] == method_id]
        seeds = sorted({int(row["training_seed"]) for row in method_rows})
        if spec.learned and seeds != expected_learned_seeds:
            errors.append(
                f"{method_id} training seeds are {seeds}, "
                f"expected {expected_learned_seeds}"
            )
        if not spec.learned and seeds != [-1]:
            errors.append(f"{method_id} baseline seed must be -1")
        if manifest is not None and not allow_incomplete:
            for seed in seeds:
                seed_rows = [
                    row
                    for row in method_rows
                    if int(row["training_seed"]) == seed
                ]
                errors.extend(
                    f"{method_id}/seed-{seed}: {error}"
                    for error in evaluation_protocol_errors(
                        seed_rows,
                        method_id,
                        seed,
                        manifest,
                        scales,
                        int(manifest["evaluation_episodes"]),
                    )
                )
        expected_checkpoint_version = (
            CHECKPOINT_VERSION if spec.learned else "scenario-config"
        )
        if manifest is not None:
            semantic_fields = {
                "algorithm": spec.algorithm,
                "graph_mode": spec.graph_mode,
                "sync_mode": spec.sync_mode,
                "checkpoint_version": expected_checkpoint_version,
            }
            for field, expected in semantic_fields.items():
                mismatch = next(
                    (
                        (index, row.get(field))
                        for index, row in enumerate(method_rows)
                        if row.get(field) != expected
                    ),
                    None,
                )
                if mismatch is not None:
                    errors.append(
                        f"{method_id} row {mismatch[0]} {field} is "
                        f"{mismatch[1]!r}, expected {expected!r}"
                    )
        if manifest is not None and spec.learned:
            learned_fields = {
                "checkpoint_scenario_hash": checkpoint_scenario_hash,
                "training_updates": int(manifest["updates"]),
                "training_episodes_per_update": int(manifest["episodes_per_update"]),
                "validation_episodes": int(manifest["validation_episodes"]),
                "validation_interval": int(manifest["validation_interval"]),
                "validation_seed": int(manifest["validation_seed"]),
                "hidden_dim": int(manifest["hidden_dim"]),
                "train_scales": [str(scale) for scale in manifest["train_scales"]],
                **{
                    f"training_{field}": manifest[field]
                    for field in TRAINING_HYPERPARAMETER_FIELDS
                },
            }
            for field, expected in learned_fields.items():
                mismatch = next(
                    (
                        (index, row.get(field))
                        for index, row in enumerate(method_rows)
                        if row.get(field) != expected
                    ),
                    None,
                )
                if mismatch is not None:
                    errors.append(
                        f"{method_id} row {mismatch[0]} {field} is "
                        f"{mismatch[1]!r}, expected {expected!r}"
                    )
            config_mismatch = next(
                (
                    (index, row.get("checkpoint_config_hash"))
                    for index, row in enumerate(method_rows)
                    if row.get("checkpoint_config_hash") != checkpoint_scenario_hash
                ),
                None,
            )
            if config_mismatch is not None:
                errors.append(
                    f"{method_id} row {config_mismatch[0]} checkpoint_config_hash is "
                    f"{config_mismatch[1]!r}, expected {checkpoint_scenario_hash!r}"
                )
        for scale in scales:
            scale_rows = [row for row in method_rows if row["scale"] == scale]
            if not scale_rows:
                errors.append(f"{method_id} missing scale {scale}")
                continue
            per_seed: dict[int, list[int]] = defaultdict(list)
            for row in scale_rows:
                per_seed[int(row["training_seed"])].append(int(row["eval_seed"]))
            for seed, eval_seeds in per_seed.items():
                if len(eval_seeds) != len(set(eval_seeds)):
                    errors.append(f"{method_id}/{scale}/seed-{seed} has duplicate eval seeds")
                if manifest is not None:
                    if len(eval_seeds) != expected_episodes:
                        errors.append(
                            f"{method_id}/{scale}/seed-{seed} has {len(eval_seeds)} "
                            f"episodes, expected {expected_episodes}"
                        )
                    if sorted(eval_seeds) != expected_eval_seeds:
                        errors.append(
                            f"{method_id}/{scale}/seed-{seed} eval seeds are not "
                            f"the contiguous manifest range"
                        )
            expected_seed_set = (
                set(expected_learned_seeds) if spec.learned else {-1}
            )
            if set(per_seed) != expected_seed_set:
                errors.append(
                    f"{method_id}/{scale} seed set is {sorted(per_seed)}, "
                    f"expected {sorted(expected_seed_set)}"
                )
    if errors and not allow_incomplete:
        raise ValueError("; ".join(errors))
    return {
        "valid": not errors,
        "errors": errors,
        "methods": sorted(methods),
        "expected_methods": expected_methods,
        "scales": scales,
        "scenario_hashes": sorted(hashes),
        "implementation_hashes": sorted(implementation_hashes),
        "scenario_versions": sorted(scenario_versions),
    }


def main() -> None:
    args = parse_args()
    if args.manifest is None and not args.allow_incomplete:
        raise ValueError("strict protocol validation requires --manifest")
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.manifest is not None
        else None
    )
    raw: list[dict[str, object]] = []
    for path in args.evaluations:
        raw.extend(json.loads(path.read_text(encoding="utf-8")))
    protocol = validate_protocol(
        raw,
        args.allow_incomplete,
        manifest=manifest,
        include_supplementary=args.include_supplementary,
    )

    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    episode_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw:
        key = (str(row["method_id"]), str(row["scale"]), int(row["training_seed"]))
        grouped[key].append(row)
        episode_grouped[(key[0], key[1])].append(row)

    per_seed: list[dict[str, object]] = []
    for (method_id, scale, seed), rows in sorted(grouped.items()):
        result: dict[str, object] = {
            "method_id": method_id,
            "scale": scale,
            "training_seed": seed,
            "evaluation_episodes": len(rows),
        }
        for metric in METRICS:
            result[metric] = aggregate(rows, metric)
        per_seed.append(result)

    summary: list[dict[str, object]] = []
    for method_id, scale in sorted(episode_grouped):
        spec = METHOD_SPECS[method_id]
        rows = episode_grouped[(method_id, scale)]
        replicate_rows = [
            row for row in per_seed if row["method_id"] == method_id and row["scale"] == scale
        ]
        result: dict[str, object] = {
            "method_id": method_id,
            "scale": scale,
            "training_seeds": len(replicate_rows) if spec.learned else 0,
            "evaluation_episodes": len(rows),
            "replicate_unit": "training_seed" if spec.learned else "episode",
        }
        for metric in METRICS:
            source = replicate_rows if spec.learned else rows
            values = np.asarray([float(row[metric]) for row in source])
            mean, lower, upper = mean_ci(values)
            if metric == "reallocation_success_rate" and not spec.learned:
                mean = aggregate(rows, metric)
                attempts = sum(float(row["reallocated_tasks"]) for row in rows)
                successes = sum(float(row["reallocation_successes"]) for row in rows)
                lower, upper = wilson_interval(successes, attempts)
            result[metric] = mean
            result[f"{metric}_ci95_lower"] = lower
            result[f"{metric}_ci95_upper"] = upper
        summary.append(result)

    comparisons: list[dict[str, object]] = []
    for left, right in COMPARISONS:
        for scale in protocol["scales"]:
            left_rows = [
                row for row in per_seed if row["method_id"] == left and row["scale"] == scale
            ]
            right_rows = [
                row for row in per_seed if row["method_id"] == right and row["scale"] == scale
            ]
            if not left_rows or not right_rows:
                continue
            left_by_seed = {int(row["training_seed"]): row for row in left_rows}
            right_by_seed = {int(row["training_seed"]): row for row in right_rows}
            if METHOD_SPECS[right].learned:
                seeds = sorted(set(left_by_seed) & set(right_by_seed))
                pairs = [(left_by_seed[seed], right_by_seed[seed]) for seed in seeds]
            else:
                baseline = right_rows[0]
                pairs = [(row, baseline) for row in left_rows]
                seeds = sorted(left_by_seed)
            result: dict[str, object] = {
                "comparison": f"{left}-minus-{right}",
                "left": left,
                "right": right,
                "scale": scale,
                "paired_training_seeds": seeds,
            }
            for metric in METRICS:
                delta = np.asarray(
                    [float(left_row[metric]) - float(right_row[metric]) for left_row, right_row in pairs]
                )
                mean, lower, upper = mean_ci(delta)
                result[f"{metric}_delta"] = mean
                result[f"{metric}_delta_ci95_lower"] = lower
                result[f"{metric}_delta_ci95_upper"] = upper
                result[f"{metric}_effect"] = effect_label(metric, lower, upper)
                result[f"{metric}_p_raw"] = sign_flip_pvalue(delta)
            comparisons.append(result)

    for metric in METRICS:
        adjusted = holm_adjust(
            [float(row[f"{metric}_p_raw"]) for row in comparisons]
        )
        for row, value in zip(comparisons, adjusted):
            row[f"{metric}_p_holm"] = value

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_validation": protocol,
        "summary": summary,
        "comparisons": comparisons,
        "negative_results": negative_result_records(comparisons),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    for name, rows in (
        ("summary.csv", summary),
        ("per_seed.csv", per_seed),
        ("comparisons.csv", comparisons),
    ):
        if not rows:
            continue
        with (args.output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == "__main__":
    main()

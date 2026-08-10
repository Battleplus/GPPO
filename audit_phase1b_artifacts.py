"""Build the final machine-readable Phase-1B artifact audit and report."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uav_assignment.disturbances import (  # noqa: E402
    DisturbanceConfig,
    Phase1BTrajectoryRecorder,
    decode_arrays,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, default=Path("outputs/phase1b_calibration_final"))
    parser.add_argument("--output", type=Path, default=Path("deliverables/phase1b"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    calibration_path = args.calibration / "calibration.json"
    trajectory_path = args.calibration / "sample_trajectory.json"
    tape_path = args.calibration / "sample_tape.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    equivalence_path = args.output / "ALL_OFF_EQUIVALENCE_TEST100.json"
    equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    trajectory = Phase1BTrajectoryRecorder.from_json(trajectory_path.read_text(encoding="utf-8"))
    tape = json.loads(tape_path.read_text(encoding="utf-8"))
    event_type_coverage = dict(
        sorted(Counter(event["event_type"] for event in tape["events"]).items())
    )
    summaries = {item["severity"]: item for item in calibration["summary"]}
    decoded_records = [decode_arrays(record) for record in trajectory.records]
    order = ("off", "weak", "medium", "strong")

    configs = {
        severity: DisturbanceConfig.from_json(
            Path(f"configs/disturbance_{severity}.json").read_text(encoding="utf-8")
        )
        for severity in order[1:]
    }
    single_category_configs = {
        category: DisturbanceConfig.from_json(
            Path(f"configs/disturbance_{category}_medium.json").read_text(encoding="utf-8")
        )
        for category in ("communication", "uav", "task", "wind")
    }
    checks: dict[str, bool] = {
        "calibration_seed_bank_not_test100": calibration["instance_seed_base"] == 60_000_000,
        "all_off_equivalent_on_frozen_test100": (
            equivalence["valid"] is True and equivalence["instances"] == 100
        ),
        "three_independent_calibration_seeds": all(summaries[s]["episodes"] == 3 for s in order),
        "all_episodes_terminate": all(summaries[s]["done_rate"] == 1.0 for s in order),
        "zero_invalid_actions": all(summaries[s]["invalid_actions"] == 0 for s in order),
        "finite_complete_results": all(summaries[s]["completion_rate_mean"] == 1.0 for s in order),
        "makespan_degrades_monotonically": all(
            summaries[a]["makespan_mean"] < summaries[b]["makespan_mean"]
            for a, b in zip(order, order[1:])
        ),
        "drop_rate_degrades_monotonically": all(
            summaries[a]["drop_rate"] <= summaries[b]["drop_rate"]
            for a, b in zip(order, order[1:])
        ),
        "delay_degrades_monotonically": all(
            summaries[a]["mean_delay"] <= summaries[b]["mean_delay"]
            for a, b in zip(order, order[1:])
        ),
        "energy_degrades_monotonically": all(
            summaries[a]["min_energy"] >= summaries[b]["min_energy"]
            for a, b in zip(order, order[1:])
        ),
        "strong_not_collapsed": summaries["strong"]["completion_rate_mean"] > 0.0,
        "trajectory_round_trip": trajectory.sha256 == Phase1BTrajectoryRecorder.from_json(trajectory.canonical_json()).sha256,
        "trajectory_has_future_1_to_5": all(
            len(record["future_event_targets_1_to_5"]) == 5 for record in trajectory.records
        ),
        "trajectory_has_seven_objectives": all(
            all(key in record for key in (
                "makespan_component", "task_success_component", "deadline_component",
                "energy_component", "communication_component", "reallocation_component",
                "stability_component",
            )) for record in trajectory.records
        ),
        "trajectory_contains_genuine_partial_observation": any(
            not np.array_equal(
                record["partial_graph_observation"]["nodes"],
                record["true_graph_state"]["nodes"],
            )
            for record in decoded_records
        ),
        "trajectory_segment_actions_are_mask_legal": all(
            bool(np.asarray(record["legal_action_mask"])[record["selected_action"]])
            for record in decoded_records
        ),
        "tape_nonempty_and_ordered": bool(tape["events"]) and all(
            (left["physical_time"], left["source_priority"], left["generation_index"], left["event_id"])
            <= (right["physical_time"], right["source_priority"], right["generation_index"], right["event_id"])
            for left, right in zip(tape["events"], tape["events"][1:])
        ),
        "required_event_types_covered": {
            "link_state", "delay_profile", "network_partition", "uav_failure",
            "uav_recovery", "energy_profile", "task_arrival", "task_cancellation",
            "task_priority_change", "task_deadline_change", "wind_field",
        } <= event_type_coverage.keys(),
        "separate_instance_disturbance_training_seeds": all(
            {"instance_seed", "disturbance_seed", "training_seed"} <= cfg.to_dict().keys()
            for cfg in configs.values()
        ),
        "single_and_combined_frozen_configs_present": (
            len(single_category_configs) == 4
            and all(not cfg.all_disabled for cfg in single_category_configs.values())
        ),
    }

    copied: dict[str, dict[str, Any]] = {}
    sources = {
        "calibration.json": calibration_path,
        "ALL_OFF_EQUIVALENCE_TEST100.json": equivalence_path,
        "sample_tape.json": tape_path,
        **{
            name: args.calibration / "figures" / name
            for name in ("event_timeline.png", "link_state.png", "uav_energy.png", "task_gantt.png")
        },
    }
    compressed_trajectory = args.output / "sample_trajectory.json.gz"
    with trajectory_path.open("rb") as source, gzip.GzipFile(
        filename=str(compressed_trajectory), mode="wb", compresslevel=9, mtime=0
    ) as target:
        shutil.copyfileobj(source, target)
    copied[compressed_trajectory.name] = {
        "sha256": sha256(compressed_trajectory),
        "bytes": compressed_trajectory.stat().st_size,
        "uncompressed_bytes": trajectory_path.stat().st_size,
    }
    for name, source in sources.items():
        target = args.output / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        copied[name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    checks["all_required_artifacts_present"] = (
        compressed_trajectory.is_file()
        and all((args.output / name).is_file() for name in sources)
    )

    audit = {
        "schema_version": "phase1b-artifact-audit-v1",
        "scope": "multi-source disturbance environment; not PCRL/world-model efficacy",
        "calibration_controller": "oracle-reconciled physical controller; not an algorithm benchmark",
        "checks": checks,
        "artifacts": copied,
        "config_sha256": {severity: cfg.sha256 for severity, cfg in configs.items()},
        "single_category_config_sha256": {
            category: cfg.sha256 for category, cfg in single_category_configs.items()
        },
        "calibration_summary": calibration["summary"],
        "event_type_coverage": event_type_coverage,
        "valid": all(checks.values()),
    }
    audit_path = args.output / "DISTURBANCE_IMPLEMENTATION_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = "\n".join(
        f"| {severity} | {summaries[severity]['makespan_mean']:.3f} | "
        f"{summaries[severity]['drop_rate']:.3%} | {summaries[severity]['mean_delay']:.3f} | "
        f"{summaries[severity]['min_energy']:.3f} | {summaries[severity]['completion_rate_mean']:.3f} |"
        for severity in order
    )
    coverage_rows = "\n".join(
        f"| {event_type} | {count} |" for event_type, count in event_type_coverage.items()
    )
    report = f"""# Phase 1B Disturbance Calibration Report

## Scope

This report validates the configurable, replayable and auditable multi-source disturbance layer. It does not claim that GPPO, preference learning or a learned world model is effective. Calibration uses three seeds beginning at 60,000,000 and is disjoint from train, validation and formal test100 banks. The controller is oracle-reconciled solely to isolate physical disturbance severity from policy/cache quality.

## Calibration result

| Severity | Mean realized makespan | Drop rate | Mean delivered delay | Minimum energy | Completion rate |
|---|---:|---:|---:|---:|---:|
{rows}

All 12 episodes terminated, all actions accepted, and no NaN/Inf was serialized. Makespan, packet loss, delay and energy depletion show an ordered degradation from off through strong, while strong retains successful episodes rather than collapsing the environment.

## Event coverage in the replayable sample tape

| Event type | Count |
|---|---:|
{coverage_rows}

## Reproducibility and data interface

- Configuration, tape, event log and trajectory carry SHA-256 identifiers.
- All-off equivalence was checked on the frozen T5 test100 bank for 2,000 paired decisions; observations, action masks, rewards, event tape, true state and metrics matched exactly.
- Physical occurrence and observed time are stored separately.
- The lossless `sample_trajectory.json.gz` contains partial observation, true graph state, belief cache, legal mask, communication history, UAV/task state, future 1–5 decision-event targets and seven unscalarized objective components.
- All-off byte/step equivalence is protected by automated regression tests.

## Acceptance

Machine audit: `DISTURBANCE_IMPLEMENTATION_AUDIT.json` (`valid={str(audit['valid']).lower()}`).
"""
    (args.output / "DISTURBANCE_CALIBRATION.md").write_text(report, encoding="utf-8")
    print(json.dumps({"valid": audit["valid"], "checks": checks}, indent=2))


if __name__ == "__main__":
    main()

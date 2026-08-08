from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a formal paper-faithful report from audited summaries")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generalization", type=Path)
    parser.add_argument("--inference-scaling", type=Path)
    return parser.parse_args()


def fmt(metric: dict[str, float]) -> str:
    half = metric.get("ci95_half_width")
    if half is None or half != half:
        return f"{metric['mean']:.3f}"
    return f"{metric['mean']:.3f} +/- {half:.3f}"


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    recovered = sum(
        len(row.get("recovery_infos", [])) for row in payload.get("summary", [])
        if row.get("training_scale") == row.get("evaluation_scale")
    )
    lines = [
        "# Paper-Faithful GPPO Formal Report",
        "",
        "This report is generated only from completed, hashed evaluation artifacts.",
        "It is a structural reproduction; the original simulator and code are unavailable.",
        "Communication rows replay the same event-trained checkpoint under each sync mode; they are not separately trained communication policies.",
        f"Recovery disclosure: {recovered} native evaluation rows use candidate-recovered checkpoints; Adam/RNG/episode-offset continuity was unavailable for those runs.",
        "",
        "## Native-scale results",
        "",
        "| Training scale | Method | Seeds | Active params | Total params | Return | Realized makespan | Completion | Comm bytes | Inference mean | P50 | P95 | FLOPs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        if row["training_scale"] != row["evaluation_scale"]:
            continue
        metrics = row["metrics"]
        lines.append(
            "| {training_scale} | {label} | {seed_count} | {active} | {total} | {return_} | {makespan} | {completion} | {bytes_} | {inference} | {p50} | {p95} | {flops} |".format(
                training_scale=row["training_scale"],
                label=row["label"],
                seed_count=row["training_seed_count"],
                active=",".join(str(value) for value in row.get("active_parameter_counts", [])) or "NA",
                total=",".join(str(value) for value in row.get("total_parameter_counts", [])) or "NA",
                return_=fmt(metrics["episode_return"]),
                makespan=fmt(metrics["realized_makespan"]),
                completion=fmt(metrics["completion_rate"]),
                bytes_=fmt(metrics["communication_bytes"]),
                inference=fmt(metrics["mean_inference_ms"]),
                p50=fmt(metrics["inference_p50_ms"]),
                p95=fmt(metrics["inference_p95_ms"]),
                flops=fmt(metrics.get("estimated_forward_flops", {"mean": float("nan")})),
            )
        )
    lines.extend([
        "",
        "## Projected versus final realized makespan",
        "",
        "Errors compare each decision's projected makespan with that episode's final realized makespan.",
        "",
        "| Scale | Method | Mean relative error | Per-episode P95 relative error |",
        "|---|---|---:|---:|",
    ])
    for row in payload["summary"]:
        if row["training_scale"] != row["evaluation_scale"]:
            continue
        metrics = row["metrics"]
        if "projection_error_relative_mean" not in metrics:
            continue
        lines.append(
            f"| {row['evaluation_scale']} | {row['label']} | "
            f"{fmt(metrics['projection_error_relative_mean'])} | "
            f"{fmt(metrics['projection_error_relative_p95'])} |"
        )
    lines.extend([
        "",
        "## Paired method differences",
        "",
        "Negative realized-makespan differences favor the left method. Intervals are two-sided 95% Student-t intervals across training seeds.",
        "",
        "| Scale | Left | Right | Seeds | Difference | Win rate | Significant? |",
        "|---|---|---|---:|---:|---:|---|",
    ])
    for row in payload.get("paired_method_differences", []):
        lines.append(
            f"| {row['evaluation_scale']} | {row['left_label']} | {row['right_label']} | "
            f"{row['training_seed_count']} | {fmt(row['realized_makespan_difference'])} | "
            f"{row['mean_instance_win_rate']:.3f} | "
            f"{row['realized_makespan_difference'].get('significant_two_sided_0_05', False)} |"
        )
    lines.extend(["", "## Adaptive gate audit", "", "| Scale | Method | Seeds with gate statistics | Gate mean range | PPO probe gradient evidence |", "|---|---|---:|---|---|"])
    for row in payload["summary"]:
        gate_rows = row.get("gate_summaries", [])
        if not gate_rows:
            continue
        means = [float(item["gate"]["mean"]) for item in gate_rows if item.get("gate")]
        lines.append(
            f"| {row['evaluation_scale']} | {row['label']} | {len(gate_rows)} | "
            f"{min(means):.3f}--{max(means):.3f} | see per-checkpoint gate diagnostic |"
        )
    lines.extend(["", "## Generalization error", "", "| Training scale | Evaluation scale | Method | Error (%) |", "|---|---|---|---:|"])
    for row in payload.get("generalization_error", []):
        lines.append(f"| {row['training_scale']} | {row['evaluation_scale']} | {row['label']} | {row['generalization_error_percent']:.2f} |")
    if args.generalization and args.generalization.exists():
        generalization = json.loads(args.generalization.read_text(encoding="utf-8"))
        lines.extend(["", "### Cross-scale general model and unknown scenarios", ""])
        lines.append("| Known scale | General makespan | Base makespan | Error (%) | Completion |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in generalization.get("known_scale_comparison", []):
            lines.append(
                f"| {row['scale']} | {row['general_realized_makespan']:.3f} | "
                f"{row['base_realized_makespan']:.3f} | "
                f"{row['generalization_error_percent']:.2f} | {row['completion_rate']:.3f} |"
            )
        lines.append("")
        lines.append("Unknown-scale stress tests (50 instances each):")
        lines.append("")
        lines.append("| Scale | Realized makespan | Completion | Comm bytes |")
        lines.append("|---|---:|---:|---:|")
        for row in generalization.get("rows", []):
            lines.append(
                f"| {row['scale']} | {row['realized_makespan']:.3f} | "
                f"{row['completion_rate']:.3f} | {row['communication_bytes']:.1f} |"
            )
    if args.inference_scaling and args.inference_scaling.exists():
        scaling = json.loads(args.inference_scaling.read_text(encoding="utf-8"))
        lines.extend(["", "## Fixed-UAV inference scaling", ""])
        lines.append(
            f"Fixed UAV counts: `{scaling.get('fixed_uavs')}`; Pearson R (subtasks vs mean latency): "
            f"`{scaling.get('pearson_r_subtasks_mean_inference_ms')}`."
        )
        lines.append("")
        lines.append("| Scale | Subtasks | Mean ms | P50 ms | P95 ms | FLOPs | Completion |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in scaling.get("rows", []):
            lines.append(
                f"| {row['scale']} | {row['subtasks']} | {row['mean_inference_ms']:.3f} | "
                f"{row['p50_inference_ms']:.3f} | {row['p95_inference_ms']:.3f} | "
                f"{row['estimated_forward_flops']} | {row['completion_rate']:.3f} |"
            )

    native_rows = {
        (row["evaluation_scale"], row["label"]): row
        for row in payload["summary"]
        if row["training_scale"] == row["evaluation_scale"]
    }
    communication_rows: list[tuple[str, dict[str, object], dict[str, object]]] = []
    for (scale, label), event_row in native_rows.items():
        if not label.endswith("_event"):
            continue
        full_label = label[:-len("_event")] + "_always"
        full_row = native_rows.get((scale, full_label))
        if full_row is not None:
            communication_rows.append((scale, event_row, full_row))
    lines.extend(["", "## Event-triggered versus full communication", "", "| Scale | Event method | Full method | Quality gap (%) | Communication reduction (%) |", "|---|---|---|---:|---:|"])
    for scale, event_row, full_row in communication_rows:
        event_metrics = event_row["metrics"]
        full_metrics = full_row["metrics"]
        event_makespan = float(event_metrics["realized_makespan"]["mean"])
        full_makespan = float(full_metrics["realized_makespan"]["mean"])
        event_bytes = float(event_metrics["communication_bytes"]["mean"])
        full_bytes = float(full_metrics["communication_bytes"]["mean"])
        lines.append(
            f"| {scale} | {event_row['label']} | {full_row['label']} | "
            f"{100.0 * (event_makespan - full_makespan) / max(abs(full_makespan), 1e-9):.2f} | "
            f"{100.0 * (1.0 - event_bytes / max(full_bytes, 1e-9)):.2f} |"
        )
    lines.extend(["", "## Inference complexity", "", "Pearson correlations are computed from native-scale subtasks and mean inference time.", ""])
    lines.append("```json")
    lines.append(json.dumps(payload.get("inference_complexity", {}), indent=2))
    lines.append("```")
    lines.extend(["", "## Audit limitations", ""])
    for limitation in payload.get("limitations", []):
        lines.append(f"- {limitation}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

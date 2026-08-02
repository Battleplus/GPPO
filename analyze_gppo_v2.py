from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "tmp" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "deadline_completion_rate",
    "mission_success",
    "makespan",
    "throughput",
    "communication_events",
    "heartbeat_messages",
    "invalid_actions",
    "reallocation_success_rate",
)

TRAINING_METRICS = (
    "deadline_completion_rate",
    "mission_success",
    "makespan",
    "throughput",
    "communication_events",
    "deadline_remaining_tasks",
    "completion_rate",
    "actor_loss",
)


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    mean = float(values.mean())
    half = (
        0.0
        if len(values) <= 1
        else float(critical * values.std(ddof=1) / np.sqrt(len(values)))
    )
    return mean, mean - half, mean + half


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GPPO-v2 validation curves")
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    histories: dict[tuple[str, int], list[dict[str, float]]] = {}
    for path in sorted(args.training_root.glob("*/seed_*/validation_history.json")):
        method_id = path.parent.parent.name
        seed = int(path.parent.name.split("_")[-1])
        histories[(method_id, seed)] = json.loads(path.read_text(encoding="utf-8"))
    if not histories:
        raise RuntimeError(f"no validation histories under {args.training_root}")

    grouped: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    for (method_id, _seed), rows in histories.items():
        for row in rows:
            grouped[(method_id, float(row["update"]))].append(row)

    curve_rows: list[dict[str, object]] = []
    for (method_id, update), rows in sorted(grouped.items()):
        result: dict[str, object] = {
            "method_id": method_id,
            "update": update,
            "training_seeds": len(rows),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows])
            mean, lower, upper = mean_ci(values)
            result[metric] = mean
            result[f"{metric}_ci95_lower"] = lower
            result[f"{metric}_ci95_upper"] = upper
        curve_rows.append(result)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "validation_curves.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    methods = sorted({str(row["method_id"]) for row in curve_rows})
    figure, axes = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method_id in methods:
            rows = [row for row in curve_rows if row["method_id"] == method_id]
            updates = np.asarray([float(row["update"]) for row in rows])
            means = np.asarray([float(row[metric]) for row in rows])
            lower = np.asarray([float(row[f"{metric}_ci95_lower"]) for row in rows])
            upper = np.asarray([float(row[f"{metric}_ci95_upper"]) for row in rows])
            axis.plot(
                updates,
                means,
                label=method_id,
                linewidth=1.7,
                marker="o",
                markersize=3.0,
            )
            axis.fill_between(updates, lower, upper, alpha=0.12)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("update")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=4,
        fontsize=8,
    )
    figure.savefig(
        args.output / "validation_curves.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(figure)

    training_histories: dict[tuple[str, int], list[dict[str, float]]] = {}
    for path in sorted(args.training_root.glob("*/seed_*/training_history.json")):
        method_id = path.parent.parent.name
        seed = int(path.parent.name.split("_")[-1])
        training_histories[(method_id, seed)] = json.loads(
            path.read_text(encoding="utf-8")
        )
    training_grouped: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    for (method_id, _seed), rows in training_histories.items():
        for row in rows:
            training_grouped[(method_id, float(row["update"]))].append(row)

    training_curve_rows: list[dict[str, object]] = []
    for (method_id, update), rows in sorted(training_grouped.items()):
        result: dict[str, object] = {
            "method_id": method_id,
            "update": update,
            "training_seeds": len(rows),
        }
        for metric in TRAINING_METRICS:
            values = np.asarray([float(row[metric]) for row in rows])
            mean, lower, upper = mean_ci(values)
            result[metric] = mean
            result[f"{metric}_ci95_lower"] = lower
            result[f"{metric}_ci95_upper"] = upper
        training_curve_rows.append(result)

    with (args.output / "training_curves.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training_curve_rows[0]))
        writer.writeheader()
        writer.writerows(training_curve_rows)

    training_figure, training_axes = plt.subplots(
        2, 4, figsize=(18, 8.5), constrained_layout=True
    )
    for axis, metric in zip(training_axes.flat, TRAINING_METRICS):
        for method_id in methods:
            rows = [
                row for row in training_curve_rows if row["method_id"] == method_id
            ]
            updates = np.asarray([float(row["update"]) for row in rows])
            means = np.asarray([float(row[metric]) for row in rows])
            lower = np.asarray(
                [float(row[f"{metric}_ci95_lower"]) for row in rows]
            )
            upper = np.asarray(
                [float(row[f"{metric}_ci95_upper"]) for row in rows]
            )
            axis.plot(updates, means, label=method_id, linewidth=1.5)
            axis.fill_between(updates, lower, upper, alpha=0.09)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("update")
        axis.grid(alpha=0.25)
    training_handles, training_labels = training_axes.flat[0].get_legend_handles_labels()
    training_figure.legend(
        training_handles,
        training_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=4,
        fontsize=8,
    )
    training_figure.savefig(
        args.output / "training_curves.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(training_figure)

    payload = {
        "methods": methods,
        "training_seeds": {
            method_id: sorted(seed for candidate, seed in histories if candidate == method_id)
            for method_id in methods
        },
        "validation_metrics": list(METRICS),
        "training_metrics": list(TRAINING_METRICS),
    }
    (args.output / "curve_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PAPER_SCALES = ("T5-10-48", "T10-10-53", "T15-8-66", "T20-10-92")
T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def normalize_run_label(name: str) -> str:
    """Collapse per-seed output directories into one method curve."""
    return re.sub(r"_seed[1-9][0-9]*$", "", name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig.8-style formal training curves")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("reward", "realized_makespan", "validation_realized_makespan"),
        default="reward",
    )
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt

    args = parse_args()
    validation_metric = args.metric == "validation_realized_makespan"
    history_name = "validation_history.json" if validation_metric else "training_history.json"
    histories = sorted(args.root.glob(f"**/{history_name}"))
    grouped: dict[tuple[str, str], list[list[dict[str, float]]]] = defaultdict(list)
    for path in histories:
        checkpoint = path.parent / "checkpoint.pt"
        if not checkpoint.exists():
            continue
        # Directory convention: <scale>/<label>/training_history.json.
        scale = path.parent.parent.name
        if scale not in PAPER_SCALES:
            continue
        label = normalize_run_label(path.parent.name)
        grouped[(scale, label)].append(json.loads(path.read_text(encoding="utf-8")))
    if not grouped:
        raise FileNotFoundError("no completed training histories")
    scales = sorted({key[0] for key in grouped})
    fig, axes = plt.subplots(
        max(1, (len(scales) + 1) // 2), 2,
        figsize=(13, 4.5 * max(1, (len(scales) + 1) // 2)),
        squeeze=False,
    )
    for index, scale in enumerate(scales):
        axis = axes[index // 2][index % 2]
        for (group_scale, label), runs in sorted(grouped.items()):
            if group_scale != scale:
                continue
            common = sorted(set.intersection(*(set(row["iteration"] for row in run) for run in runs)))
            if not common:
                continue
            values = np.asarray([
                [
                    next(
                        row["realized_makespan"] if validation_metric else row[args.metric]
                        for row in run
                        if row["iteration"] == iteration
                    )
                    for iteration in common
                ]
                for run in runs
            ])
            mean = values.mean(axis=0)
            if values.shape[0] > 1:
                critical = T_975.get(values.shape[0] - 1, 1.96)
                half = critical * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            else:
                half = np.zeros_like(mean)
            axis.plot(common, mean, label=label)
            axis.fill_between(common, mean - half, mean + half, alpha=0.15)
        axis.set_title(scale)
        axis.set_xlabel("Iteration")
        axis.set_ylabel(args.metric)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for index in range(len(scales), axes.size):
        axes[index // 2][index % 2].set_visible(False)
    fig.suptitle(f"Paper-faithful Fig.8-style {args.metric} curves")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(json.dumps({"saved": str(args.output), "groups": len(grouped)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

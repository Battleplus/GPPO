"""Generate the README-ready Phase-1 result gallery from tracked artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deliverables" / "stage1_gallery"
COLORS = {
    "GPPO-event": "#2563eb",
    "PPO-none": "#94a3b8",
    "PPO-event": "#64748b",
    "GPPO-none": "#60a5fa",
    "GPPO-NoGate-event": "#f59e0b",
    "GPPO-SingleHead-event": "#10b981",
}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def moving_average(values: list[float], width: int = 15) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < width:
        return array
    padded = np.pad(array, (width - 1, 0), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def plot_method_comparison(acceptance: dict) -> None:
    methods = list(acceptance["methods"])
    means = [acceptance["methods"][name]["makespan"]["mean"] for name in methods]
    stds = [acceptance["methods"][name]["makespan"]["std"] for name in methods]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(methods, means, yerr=stds, capsize=4, color=[COLORS[name] for name in methods])
    ax.set_ylabel("Realized makespan (mean ± instance std)")
    ax.set_title("Phase 1A — six learned mechanisms on fixed test100")
    ax.set_ylim(min(means) - 1.0, max(means) + 2.0)
    ax.tick_params(axis="x", rotation=22)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.16, f"{value:.3f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "phase1a_method_comparison.png")


def plot_pairwise_effects(acceptance: dict) -> None:
    selected = [
        "GPPO-event_vs_PPO-none", "GPPO-event_vs_PPO-event",
        "GPPO-none_vs_PPO-none", "GPPO-event_vs_GPPO-none",
        "Adaptive_vs_NoGate", "Adaptive_vs_SingleHead",
    ]
    labels = [name.replace("_vs_", " − ") for name in selected]
    means = np.asarray([acceptance["comparisons"][name]["mean"] for name in selected])
    lows = np.asarray([acceptance["comparisons"][name]["ci95_low"] for name in selected])
    highs = np.asarray([acceptance["comparisons"][name]["ci95_high"] for name in selected])
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#16a34a" if value < 0 else "#dc2626" for value in means]
    ax.errorbar(means, y, xerr=[means - lows, highs - means], fmt="none", ecolor="#64748b", capsize=5, lw=2)
    ax.scatter(means, y, c=colors, s=65, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired makespan difference (left − right); lower is better")
    ax.set_title("Phase 1A — paired fixed-test100 effects with instance-level 95% CI")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "phase1a_pairwise_effects.png")


def plot_training_curves() -> None:
    label_map = {
        "literal_event_seed1": "GPPO-event",
        "ppo_mlp_none_seed1": "PPO-none",
        "ppo_mlp_event_seed1": "PPO-event",
        "literal_none_seed1": "GPPO-none",
        "literal_no_gate_event_seed1": "GPPO-NoGate-event",
        "literal_single_head_event_seed1": "GPPO-SingleHead-event",
    }
    base = ROOT / "artifacts" / "phase1_seed1_300" / "raw" / "T5-10-48"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for folder, label in label_map.items():
        history = json.loads((base / folder / "training_history.json").read_text(encoding="utf-8"))
        x = [int(row["iteration"]) for row in history]
        axes[0].plot(x, moving_average([row["realized_makespan"] for row in history]), label=label, color=COLORS[label])
        axes[1].plot(x, moving_average([row["reward"] for row in history]), label=label, color=COLORS[label])
    axes[0].set(title="Realized makespan", xlabel="Training iteration", ylabel="15-iteration moving average")
    axes[1].set(title="Episode reward", xlabel="Training iteration", ylabel="15-iteration moving average")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Phase 1A — 300-iteration training dynamics (single training seed)")
    save(fig, "phase1a_training_curves.png")


def plot_communication_tradeoff(acceptance: dict) -> None:
    methods = list(acceptance["methods"])
    x = [acceptance["methods"][name]["communication_bytes"]["mean"] for name in methods]
    y = [acceptance["methods"][name]["makespan"]["mean"] for name in methods]
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, bytes_, makespan in zip(methods, x, y):
        ax.scatter(bytes_, makespan, s=90, color=COLORS[name])
        ax.annotate(name, (bytes_, makespan), xytext=(5, 5), textcoords="offset points", fontsize=8)
    communication = acceptance["communication"]
    ax.scatter(communication["full_bytes"], communication["full_makespan"], marker="*", s=170, color="#7c3aed")
    ax.annotate("GPPO-full replay", (communication["full_bytes"], communication["full_makespan"]), xytext=(6, -14), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set(xlabel="Mean communication bytes (log scale)", ylabel="Mean realized makespan", title="Phase 1A — task quality vs communication cost")
    ax.grid(alpha=0.25, which="both")
    save(fig, "phase1a_communication_tradeoff.png")


def plot_gate_screening() -> None:
    text = (ROOT / "reports" / "GATE_THREE_SEED_SCREENING.md").read_text(encoding="utf-8")
    entries = []
    for line in text.splitlines():
        match = re.match(r"\|\s*([^|]+?)\s*\|\s*(\d+\.\d+)\s*\|", line)
        if match and match.group(1).strip() not in {"候选", "---"}:
            entries.append((match.group(1).strip(), float(match.group(2))))
    names, values = zip(*entries)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#2563eb" if name == "Adaptive-score" else "#94a3b8" for name in names]
    bars = ax.barh(names, values, color=colors)
    ax.set_xlim(min(values) - 0.12, max(values) + 0.08)
    ax.set_xlabel("Mean validation-A realized makespan across 3 training seeds")
    ax.set_title("Phase 1A — preregistered gate screening (lower is better)")
    for bar, value in zip(bars, values):
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "phase1a_gate_screening.png")


def plot_disturbance_calibration(calibration: dict) -> None:
    rows = calibration["summary"]
    severity = [row["severity"].title() for row in rows]
    metrics = [
        ("makespan_mean", "Mean realized makespan", "#2563eb"),
        ("drop_rate", "Packet drop rate", "#dc2626"),
        ("mean_delay", "Mean delivered delay", "#f59e0b"),
        ("min_energy", "Minimum UAV energy", "#16a34a"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title, color) in zip(axes.flat, metrics):
        values = [row[key] for row in rows]
        ax.plot(severity, values, marker="o", lw=2.5, color=color)
        for index, value in enumerate(values):
            label = f"{value:.1%}" if key == "drop_rate" else f"{value:.3f}"
            ax.annotate(label, (index, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Phase 1B — three-seed disturbance severity calibration")
    save(fig, "phase1b_severity_calibration.png")


def plot_stage1_dashboard(acceptance: dict, calibration: dict, audit: dict, equivalence: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    methods = list(acceptance["methods"])
    axes[0, 0].barh(methods, [acceptance["methods"][name]["makespan"]["mean"] for name in methods], color=[COLORS[name] for name in methods])
    axes[0, 0].invert_yaxis(); axes[0, 0].set_title("1A: learned-method makespan")
    severity = [row["severity"].title() for row in calibration["summary"]]
    axes[0, 1].plot(severity, [row["makespan_mean"] for row in calibration["summary"]], marker="o", color="#2563eb")
    axes[0, 1].set_title("1B: disturbance makespan degradation")
    axes[1, 0].bar(["Test100 instances", "Paired decisions", "Unit tests"], [equivalence["instances"], sum(row["decisions"] for row in equivalence["rows"]), 114], color=["#16a34a", "#0ea5e9", "#7c3aed"])
    axes[1, 0].set_title("Verification coverage")
    checks = audit["checks"]
    axes[1, 1].bar(["Passed", "Failed"], [sum(bool(v) for v in checks.values()), sum(not bool(v) for v in checks.values())], color=["#16a34a", "#dc2626"])
    axes[1, 1].set_title("Phase 1B machine-audit checks")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Phase 1 result dashboard — mechanism baseline + auditable disturbance environment")
    save(fig, "phase1_overview_dashboard.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acceptance = load("artifacts/phase1_seed1_300/raw/MECHANISM_ACCEPTANCE.json")
    calibration = load("deliverables/phase1b/calibration.json")
    audit = load("deliverables/phase1b/DISTURBANCE_IMPLEMENTATION_AUDIT.json")
    equivalence = load("deliverables/phase1b/ALL_OFF_EQUIVALENCE_TEST100.json")
    plot_method_comparison(acceptance)
    plot_pairwise_effects(acceptance)
    plot_training_curves()
    plot_communication_tradeoff(acceptance)
    plot_gate_screening()
    plot_disturbance_calibration(calibration)
    plot_stage1_dashboard(acceptance, calibration, audit, equivalence)
    manifest = {
        "schema_version": "stage1-gallery-v1",
        "source_files": [
            "artifacts/phase1_seed1_300/raw/MECHANISM_ACCEPTANCE.json",
            "artifacts/phase1_seed1_300/raw/T5-10-48/*/training_history.json",
            "reports/GATE_THREE_SEED_SCREENING.md",
            "deliverables/phase1b/calibration.json",
            "deliverables/phase1b/DISTURBANCE_IMPLEMENTATION_AUDIT.json",
            "deliverables/phase1b/ALL_OFF_EQUIVALENCE_TEST100.json",
        ],
        "figures": sorted(path.name for path in OUT.glob("*.png")),
        "notes": [
            "Phase 1A learned-method results are single-training-seed fixed-test100 mechanism evidence.",
            "Phase 1B calibration uses three independent calibration seeds and is not an algorithm benchmark.",
        ],
    }
    (OUT / "GALLERY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

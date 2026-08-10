"""Calibrate and audit Phase-1B disturbances on a non-test seed bank.

This is an environment calibration utility, not an algorithm benchmark.  Its
safe policies prefer the intersection of belief-legal and physically legal
assignments and explicitly report any residual cache-conflict actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uav_assignment.disturbances import (  # noqa: E402
    DisturbanceConfig,
    Phase1BTrajectoryRecorder,
)
from uav_assignment.paper_faithful_env import PAPER_SCALES, PaperFaithfulConfig  # noqa: E402
from uav_assignment.phase1b_env import Phase1BPaperFaithfulUAVEnv  # noqa: E402


CALIBRATION_INSTANCE_BASE = 60_000_000
CALIBRATION_DISTURBANCE_BASE = 70_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=400)
    parser.add_argument(
        "--policy", action="append", choices=("safe_greedy", "safe_random")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1b_calibration"))
    return parser.parse_args()


def load_severity(name: str) -> DisturbanceConfig:
    if name == "off":
        return DisturbanceConfig(severity="off")
    return DisturbanceConfig.from_json(
        Path(f"configs/disturbance_{name}.json").read_text(encoding="utf-8")
    )


def safe_actions(env: Phase1BPaperFaithfulUAVEnv) -> np.ndarray:
    belief = env.valid_action_mask().astype(bool)
    physical = env.true_observation()["action_mask"].astype(bool)
    legal = np.flatnonzero(belief & physical)
    non_noop = legal[legal != env.noop_action]
    return non_noop if len(non_noop) else np.asarray([env.noop_action], dtype=np.int64)


def choose_safe_action(
    env: Phase1BPaperFaithfulUAVEnv, policy: str, rng: np.random.Generator
) -> int:
    legal = safe_actions(env)
    if policy == "safe_random":
        return int(rng.choice(legal))
    candidates: list[tuple[float, int]] = []
    for action in legal:
        if action == env.noop_action:
            return int(action)
        uav_index, task_index = divmod(int(action), env.config.max_tasks)
        uav, task = env.belief_uavs[uav_index], env.belief_tasks[task_index]
        flight, execution = env._execution_components(uav, task, env.belief_weather)
        candidates.append((env.belief_time + uav.remaining_time + flight + execution, int(action)))
    return min(candidates)[1]


def network_components(env: Phase1BPaperFaithfulUAVEnv) -> list[list[str]]:
    engine = env.disturbance_engine
    assert engine is not None
    for event in engine.communication.partitions:
        if float(event.payload["start"]) <= env.current_time < float(event.payload["end"]):
            return [list(group) for group in event.payload["groups"]]
    return [list(engine.uav_ids)]


def episode(
    severity: str,
    policy: str,
    index: int,
    max_decisions: int,
    *,
    record_trajectory: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    instance_seed = CALIBRATION_INSTANCE_BASE + index
    disturbance_seed = CALIBRATION_DISTURBANCE_BASE + index
    disturbance = replace(
        load_severity(severity),
        instance_seed=instance_seed,
        disturbance_seed=disturbance_seed,
    )
    env = Phase1BPaperFaithfulUAVEnv(
        PaperFaithfulConfig(scale=PAPER_SCALES[0], instance_seed=instance_seed),
        disturbance,
    )
    observation = env.reset(seed=instance_seed)
    assert env.disturbance_engine is not None
    recorder = (
        Phase1BTrajectoryRecorder.for_engine(
            env.disturbance_engine, episode_id=f"cal-{severity}-{policy}-{index}"
        )
        if record_trajectory else None
    )
    rng = np.random.default_rng(80_000_000 + index + (0 if policy == "safe_greedy" else 10_000))
    energy_trace: list[dict[str, Any]] = []
    task_trace: list[dict[str, Any]] = []
    done = False
    episode_return = 0.0
    previous_time = 0.0
    previous_completed = 0
    for decision in range(max_decisions):
        action = choose_safe_action(env, policy, rng)
        true_state = env.true_observation()
        before_audit = env.disturbance_engine.communication.audit.to_dict()
        observation, reward, done, info = env.step(action, sync_mode="event")
        episode_return += float(reward)
        completed = sum(task.active and task.completed for task in env.tasks)
        audit = env.disturbance_engine.communication.audit.to_dict()
        energy_trace.append({"time": env.current_time, **env.uav_energy})
        for event in env._last_disturbance_events:
            if event.event_type.startswith("task_"):
                task_trace.append(
                    {"time": event.physical_time, "task": event.target, "event": event.event_type}
                )
        if recorder is not None:
            current = env._last_disturbance_events
            recorder.record_decision(
                decision_index=decision,
                physical_time=env.current_time,
                partial_graph_observation=observation,
                true_graph_state=true_state,
                belief_cache={"nodes": observation["nodes"]},
                legal_action_mask=observation["action_mask"],
                selected_action=action,
                communication_history=[audit],
                messages_sent=[{"count": audit["messages_sent"] - before_audit["messages_sent"]}],
                messages_delivered=[{"count": audit["messages_delivered"] - before_audit["messages_delivered"]}],
                messages_dropped=[{"count": audit["messages_dropped"] - before_audit["messages_dropped"]}],
                message_delays=[float(audit["mean_delivered_delay"])],
                network_components=network_components(env),
                uav_energy=env.uav_energy,
                uav_alive={f"u{i}": bool(u.alive) for i, u in enumerate(env.uavs[:5])},
                task_status={
                    key: state.status for key, state in env.disturbance_engine.task.tasks.items()
                },
                task_priority={
                    key: state.priority for key, state in env.disturbance_engine.task.tasks.items()
                },
                task_deadline=env.task_deadlines,
                current_events=current,
                event_observed_delay={
                    event.event_id: max(0.0, env.current_time - event.physical_time)
                    for event in current
                },
                objective_components={
                    "makespan_component": -(env.current_time - previous_time),
                    "task_success_component": float(completed - previous_completed),
                    "deadline_component": 0.0,
                    "energy_component": -float(sum(1.0 - value for value in env.uav_energy.values())),
                    "communication_component": -float(audit["bytes_sent"] - before_audit["bytes_sent"]),
                    "reallocation_component": -float(env.reallocated_tasks),
                    "stability_component": -float(len(current)),
                },
            )
        previous_time, previous_completed = env.current_time, completed
        if done:
            break
    if recorder is not None:
        recorder.finalize(env.disturbance_engine)
    metrics = env.metrics()
    audit = env.disturbance_engine.communication.audit.to_dict()
    row = {
        "severity": severity,
        "policy": policy,
        "calibration_index": index,
        "instance_seed": instance_seed,
        "disturbance_seed": disturbance_seed,
        "config_sha256": env.disturbance_engine.config.sha256,
        "tape_sha256": env.disturbance_engine.tape.sha256,
        "log_sha256": env.disturbance_engine.logger.sha256,
        "done": bool(done),
        "decisions": decision + 1,
        "episode_return": episode_return,
        "completion_rate": float(metrics["completion_rate"]),
        "realized_makespan": float(metrics["realized_makespan"]),
        "invalid_actions": int(metrics["invalid_actions"]),
        "min_energy": min(env.uav_energy.values(), default=1.0),
        "event_count_applied": len(env.disturbance_engine.logger.records),
        **audit,
    }
    traces = {"energy": energy_trace, "tasks": task_trace}
    return row, recorder.to_dict() if recorder else None, traces


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for severity in ("off", "weak", "medium", "strong"):
        for policy in sorted({row["policy"] for row in rows}):
            group = [row for row in rows if row["severity"] == severity and row["policy"] == policy]
            output.append({
                "severity": severity,
                "policy": policy,
                "episodes": len(group),
                "completion_rate_mean": float(np.mean([x["completion_rate"] for x in group])),
                "makespan_mean": float(np.mean([x["realized_makespan"] for x in group])),
                "drop_rate": float(sum(x["messages_dropped"] for x in group) / max(1, sum(x["messages_sent"] for x in group))),
                "mean_delay": float(np.mean([x["mean_delivered_delay"] for x in group])),
                "min_energy": float(min(x["min_energy"] for x in group)),
                "invalid_actions": int(sum(x["invalid_actions"] for x in group)),
                "done_rate": float(np.mean([x["done"] for x in group])),
            })
    return output


def plot_artifacts(output: Path, rows: list[dict[str, Any]], traces: dict[str, Any], tape: dict[str, Any]) -> None:
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    events = tape["events"]
    fig, ax = plt.subplots(figsize=(10, 5))
    kinds = sorted({event["event_type"] for event in events}); y = {kind: i for i, kind in enumerate(kinds)}
    ax.scatter([e["physical_time"] for e in events], [y[e["event_type"]] for e in events], s=8)
    ax.set_yticks(list(y.values()), list(y)); ax.set_xlabel("Physical time"); ax.set_title("Disturbance event timeline")
    fig.tight_layout(); fig.savefig(figures / "event_timeline.png", dpi=160); plt.close(fig)

    link_events = [e for e in events if e["event_type"] == "link_state" and e["target"] == "u0->u1"]
    fig, ax = plt.subplots(); ax.step([e["physical_time"] for e in link_events], [e["payload"]["loss_probability"] for e in link_events], where="post")
    ax.set(xlabel="Physical time", ylabel="Loss probability", title="Link u0→u1 state"); fig.tight_layout(); fig.savefig(figures / "link_state.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots()
    for uav in sorted(k for k in traces["energy"][0] if k != "time"):
        ax.plot([x["time"] for x in traces["energy"]], [x[uav] for x in traces["energy"]], label=uav)
    ax.set(xlabel="Physical time", ylabel="Energy", title="UAV energy"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "uav_energy.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4)); task_events = traces["tasks"]
    for i, event in enumerate(task_events): ax.barh(i, 0.4, left=event["time"]); ax.text(event["time"], i, f"{event['task']} {event['event']}", va="center")
    ax.set(xlabel="Physical time", title="Dynamic task event Gantt"); fig.tight_layout(); fig.savefig(figures / "task_gantt.png", dpi=160); plt.close(fig)


def main() -> None:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    policies = tuple(args.policy or ("safe_greedy", "safe_random"))
    rows: list[dict[str, Any]] = []; sample_trajectory = None; sample_traces = None; sample_tape = None
    for severity in ("off", "weak", "medium", "strong"):
        for policy in policies:
            for index in range(args.instances):
                row, trajectory, traces = episode(severity, policy, index, args.max_decisions, record_trajectory=(severity == "strong" and policy == "safe_greedy" and index == 0))
                rows.append(row)
                (args.output / "rows.partial.json").write_text(
                    json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(
                    f"completed {severity}/{policy}/{index}: "
                    f"done={row['done']} completion={row['completion_rate']:.3f} "
                    f"invalid={row['invalid_actions']}",
                    flush=True,
                )
                if trajectory is not None:
                    sample_trajectory, sample_traces = trajectory, traces
                    cfg = replace(load_severity("strong"), instance_seed=CALIBRATION_INSTANCE_BASE, disturbance_seed=CALIBRATION_DISTURBANCE_BASE)
                    env = Phase1BPaperFaithfulUAVEnv(PaperFaithfulConfig(scale=PAPER_SCALES[0], instance_seed=CALIBRATION_INSTANCE_BASE), cfg); env.reset(seed=CALIBRATION_INSTANCE_BASE)
                    sample_tape = env.disturbance_engine.tape.to_dict()
    summary = summarize(rows)
    payload = {
        "schema_version": "phase1b-calibration-v1",
        "seed_bank": "calibration-only; disjoint from train/validation/test100",
        "instance_seed_base": CALIBRATION_INSTANCE_BASE,
        "disturbance_seed_base": CALIBRATION_DISTURBANCE_BASE,
        "policies": "safe calibration controls; not algorithm benchmarks",
        "rows": rows,
        "summary": summary,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    (args.output / "calibration.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    assert sample_trajectory is not None and sample_tape is not None and sample_traces is not None
    (args.output / "sample_trajectory.json").write_text(json.dumps(sample_trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "sample_tape.json").write_text(json.dumps(sample_tape, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_artifacts(args.output, rows, sample_traces, sample_tape)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

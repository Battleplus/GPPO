"""Evaluate non-learning controls on the paper-faithful test banks.

The learned-policy matrix is intentionally kept separate from these controls.
Random is a stochastic legal-action policy, greedy is a belief-state
earliest-finish heuristic, and oracle is the same heuristic with perfect
current-state access and full synchronization.  The oracle is a practical
upper-reference, not an exact combinatorial optimum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from evaluate_paper_faithful import tape_hash
from uav_assignment.paper_faithful_env import (
    PAPER_SCALES,
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
    PaperScale,
    deterministic_instance_seeds,
)


POLICIES = ("random", "greedy", "oracle_earliest_finish")


def parse_scale(value: str) -> PaperScale:
    normalized = value.upper().removeprefix("T")
    values = tuple(int(part) for part in normalized.split("-"))
    for scale in PAPER_SCALES:
        if (scale.uavs, scale.parent_tasks, scale.subtasks) == values:
            return scale
    raise argparse.ArgumentTypeError(f"unknown paper scale: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper-faithful random/greedy/oracle controls")
    parser.add_argument("--scale", type=parse_scale, action="append")
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--policy-seed", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _earliest_finish_action(
    env: PaperFaithfulUAVEnv,
    *,
    true_state: bool,
) -> int:
    uavs = env.uavs if true_state else env.belief_uavs
    tasks = env.tasks if true_state else env.belief_tasks
    now = env.current_time if true_state else env.belief_time
    weather = env.weather_severity if true_state else env.belief_weather
    mask = env._valid_mask_for(uavs, tasks)
    candidates: list[tuple[float, int, int, int]] = []
    for action in np.flatnonzero(mask[:-1]):
        uav_index, task_index = divmod(int(action), env.config.max_tasks)
        uav = uavs[uav_index]
        task = tasks[task_index]
        flight, execution = env._execution_components(uav, task, weather)
        finish = max(now, now + float(uav.remaining_time)) + flight + execution
        # Stable tie-breaking prefers higher task priority, then lower action id.
        candidates.append((finish, -int(round(task.priority * 1_000_000)), int(action), task_index))
    if not candidates:
        return int(env.noop_action)
    candidates.sort()
    return candidates[0][2]


def choose_action(env: PaperFaithfulUAVEnv, policy: str, rng: np.random.Generator) -> tuple[int, str]:
    if policy == "random":
        mask = env.valid_action_mask()
        actions = np.flatnonzero(mask)
        return int(rng.choice(actions)), "belief"
    if policy == "greedy":
        return _earliest_finish_action(env, true_state=False), "belief"
    if policy == "oracle_earliest_finish":
        return _earliest_finish_action(env, true_state=True), "true"
    raise ValueError(f"unsupported policy: {policy}")


def evaluate_policy(scale: PaperScale, policy: str, policy_seed: int, instances: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for instance_seed in deterministic_instance_seeds(scale, instances, split="test"):
        config = PaperFaithfulConfig(
            scale=scale,
            max_uavs=scale.uavs,
            max_subtasks=scale.subtasks,
            instance_seed=instance_seed,
        )
        env = PaperFaithfulUAVEnv(config)
        env.reset(seed=instance_seed)
        rng = np.random.default_rng(policy_seed * 1_000_000_007 + instance_seed)
        sync_mode = "always" if policy == "oracle_earliest_finish" else "event"
        episode_return = 0.0
        decisions = 0
        done = False
        while not done:
            action, state_source = choose_action(env, policy, rng)
            _, reward, done, _ = env.step(action, sync_mode=sync_mode)
            episode_return += float(reward)
            decisions += 1
        metrics = env.metrics()
        rows.append(
            {
                "instance_seed": int(instance_seed),
                "event_tape_hash": tape_hash(env),
                "episode_return": float(episode_return),
                "realized_makespan": float(metrics["realized_makespan"]),
                "projected_makespan": float(metrics["projected_makespan"]),
                "completion_rate": float(metrics["completion_rate"]),
                "all_tasks_completed": float(metrics["all_tasks_completed"]),
                "communication_events": float(metrics["communication_events"]),
                "communication_bytes": float(metrics["communication_bytes"]),
                "heartbeat_messages": float(metrics["heartbeat_messages"]),
                "decisions": decisions,
                "state_source": state_source,
            }
        )
    numeric = (
        "episode_return", "realized_makespan", "projected_makespan", "completion_rate",
        "all_tasks_completed", "communication_events", "communication_bytes",
        "heartbeat_messages", "decisions",
    )
    summary = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
            "median": float(np.median([row[key] for row in rows])),
        }
        for key in numeric
    }
    return {
        "policy": policy,
        "policy_seed": policy_seed,
        "scale": scale.name,
        "split": "test",
        "instances": instances,
        "sync_mode": "always" if policy == "oracle_earliest_finish" else "event",
        "event_tape_hashes": sorted({str(row["event_tape_hash"]) for row in rows}),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    scales = tuple(args.scale or PAPER_SCALES)
    policy_seeds = tuple(args.policy_seed or (1, 2, 3, 4, 5))
    results = [
        evaluate_policy(scale, policy, seed, args.instances)
        for scale in scales
        for policy in POLICIES
        for seed in policy_seeds
    ]
    payload = {
        "version": "paper-faithful-baselines-v1",
        "protocol": "configs/paper_faithful_protocol.json",
        "policies": list(POLICIES),
        "policy_seeds": list(policy_seeds),
        "results": results,
        "oracle_note": "oracle_earliest_finish uses true UAV/task state and full synchronization but is a one-step earliest-finish heuristic, not an exact optimum.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "results": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""CPU-only Phase-1B smoke suitable for a cloned repository in Colab."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uav_assignment.disturbances import DisturbanceConfig  # noqa: E402
from uav_assignment.paper_faithful_env import PaperFaithfulConfig, PaperFaithfulUAVEnv  # noqa: E402
from uav_assignment.phase1b_env import Phase1BPaperFaithfulUAVEnv  # noqa: E402


def nested_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
    return left == right


def main() -> None:
    seed = 91_000_001
    config = PaperFaithfulConfig(instance_seed=seed)
    baseline = PaperFaithfulUAVEnv(config)
    adapted = Phase1BPaperFaithfulUAVEnv(config, DisturbanceConfig())
    left, right = baseline.reset(seed=seed), adapted.reset(seed=seed)
    assert nested_equal(left, right)
    decisions = 0
    while decisions < 20:
        action = int(np.flatnonzero(left["action_mask"])[0])
        left, lr, ld, li = baseline.step(action, sync_mode="event")
        right, rr, rd, ri = adapted.step(action, sync_mode="event")
        assert nested_equal(left, right) and lr == rr and ld == rd and nested_equal(li, ri)
        decisions += 1
        if ld:
            break

    weak = DisturbanceConfig.from_json(Path("configs/disturbance_weak.json").read_text(encoding="utf-8"))
    env = Phase1BPaperFaithfulUAVEnv(config, weak)
    observation = env.reset(seed=seed)
    for _ in range(30):
        legal = np.flatnonzero(observation["action_mask"])
        observation, reward, done, _ = env.step(int(legal[0]), sync_mode="event")
        assert np.isfinite(reward)
        if done:
            break
    assert env.disturbance_engine is not None
    result = {
        "valid": True,
        "all_off_equivalent_decisions": decisions,
        "weak_tape_events": len(env.disturbance_engine.tape.events),
        "weak_tape_sha256": env.disturbance_engine.tape.sha256,
        "weak_log_sha256": env.disturbance_engine.logger.sha256,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

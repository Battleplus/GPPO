from __future__ import annotations

import numpy as np

from evaluate_paper_faithful_baselines import (
    _earliest_finish_action,
    evaluate_policy,
)
from uav_assignment.paper_faithful_env import (
    PAPER_SCALES,
    PaperFaithfulConfig,
    PaperFaithfulUAVEnv,
    deterministic_instance_seeds,
)


def make_env() -> PaperFaithfulUAVEnv:
    scale = PAPER_SCALES[0]
    config = PaperFaithfulConfig(
        scale=scale,
        max_uavs=scale.uavs,
        max_subtasks=scale.subtasks,
    )
    env = PaperFaithfulUAVEnv(config)
    env.reset(seed=deterministic_instance_seeds(scale, 1, split="test")[0])
    return env


def test_earliest_finish_actions_are_legal_in_selected_state() -> None:
    env = make_env()
    belief_action = _earliest_finish_action(env, true_state=False)
    true_action = _earliest_finish_action(env, true_state=True)
    assert env.valid_action_mask()[belief_action]
    assert env._valid_mask_for(env.uavs, env.tasks)[true_action]


def test_baseline_evaluator_completes_and_preserves_tape_identity() -> None:
    scale = PAPER_SCALES[0]
    greedy = evaluate_policy(scale, "greedy", policy_seed=1, instances=1)
    oracle = evaluate_policy(scale, "oracle_earliest_finish", policy_seed=1, instances=1)
    assert greedy["summary"]["all_tasks_completed"]["mean"] == 1.0
    assert oracle["summary"]["all_tasks_completed"]["mean"] == 1.0
    assert greedy["event_tape_hashes"] == oracle["event_tape_hashes"]
    assert oracle["summary"]["communication_bytes"]["mean"] > greedy["summary"]["communication_bytes"]["mean"]


def test_random_policy_seed_is_reproducible() -> None:
    scale = PAPER_SCALES[0]
    first = evaluate_policy(scale, "random", policy_seed=3, instances=1)
    second = evaluate_policy(scale, "random", policy_seed=3, instances=1)
    assert np.isclose(
        first["summary"]["realized_makespan"]["mean"],
        second["summary"]["realized_makespan"]["mean"],
    )

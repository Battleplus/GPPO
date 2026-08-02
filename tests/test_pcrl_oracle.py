from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import evaluate_pcrl_oracle
from uav_assignment.paper_env import PaperEnvConfig
from uav_assignment.pcrl_oracle import (
    OracleConfig,
    choose_oracle_action,
    step_oracle_action,
)
from uav_assignment.pcrl_v0 import (
    PreferencePaperEnv,
    calibration_priority_profiles,
    task_preference_profile,
)


def oracle_env(preference: str = "balanced") -> PreferencePaperEnv:
    config = PaperEnvConfig(
        max_uavs=4,
        max_tasks=16,
        active_uavs=4,
        initial_tasks=16,
        max_decisions=80,
        mission_deadline=12.0,
        task_chain_length=4,
        workload_scale=1.0,
        weather_probability=0.0,
        failure_probability=0.0,
        task_change_probability=0.0,
        communication_drop_probability=0.0,
        include_engineering_rewards=False,
        seed=3,
    )
    env = PreferencePaperEnv(
        config,
        preference=task_preference_profile(preference),
        task_release_mode="phase_staggered",
        assignment_priority_decay=2.0,
        preference_target_mode="nominal",
    )
    env.reset(seed=13)
    return env


def test_oracle_always_returns_a_true_legal_action() -> None:
    env = oracle_env()
    config = OracleConfig(decay=2.0, target_mode="nominal")
    done = False
    while not done:
        true_mask = env._valid_mask_for(env.uavs, env.tasks)
        action = choose_oracle_action(env, config=config)
        assert bool(true_mask[action])
        _, _, done, _ = env.step(action, sync_mode="event")
    assert env.metrics()["completed_total"] > 0


def test_oracle_reads_true_state_instead_of_stale_belief_mask() -> None:
    env = oracle_env("search")
    for task in env.belief_tasks:
        task.completed = True
    belief_mask = env.valid_action_mask()
    action = choose_oracle_action(
        env, config=OracleConfig(decay=2.0, target_mode="nominal")
    )
    true_mask = env._valid_mask_for(env.uavs, env.tasks)
    assert action != env.noop_action
    assert bool(true_mask[action])
    assert not bool(belief_mask[action])


@pytest.mark.parametrize(
    ("profile", "expected_type"),
    (("search", 0), ("reconnaissance", 1), ("strike", 2), ("recovery", 3)),
)
def test_extreme_preference_controls_first_assignment_type(
    profile: str, expected_type: int
) -> None:
    env = oracle_env(profile)
    action = choose_oracle_action(
        env, config=OracleConfig(decay=2.0, target_mode="nominal")
    )
    _, task_index = divmod(action, env.config.max_tasks)
    assert env.tasks[task_index].task_type == expected_type


def test_oracle_config_rejects_invalid_constraints() -> None:
    assert OracleConfig().allow_strategic_wait is False
    with pytest.raises(ValueError, match="coverage_floor"):
        OracleConfig(coverage_floor=1.1)
    with pytest.raises(ValueError, match="decay"):
        OracleConfig(decay=-1.0)
    with pytest.raises(ValueError, match="rollout_candidates"):
        OracleConfig(rollout_candidates=-1)
    with pytest.raises(ValueError, match="deadline_completion_floor"):
        OracleConfig(deadline_completion_floor=1.1)
    with pytest.raises(ValueError, match="minimum_type_coverage_floor"):
        OracleConfig(minimum_type_coverage_floor=-0.1)
    with pytest.raises(ValueError, match="prerequisite_gain"):
        OracleConfig(prerequisite_gain=-1.0)


def test_mask_faithful_rollout_never_selects_masked_noop() -> None:
    env = oracle_env("smooth_interp")
    config = OracleConfig(
        decay=2.0,
        target_mode="nominal",
        rollout_candidates=4,
    )
    for _ in range(6):
        true_mask = env._valid_mask_for(env.uavs, env.tasks)
        action = choose_oracle_action(env, config=config)
        assert bool(true_mask[action])
        _, _, done, _ = env.step(action, sync_mode="event")
        if done:
            break


def test_strategic_wait_advances_clock_without_invalid_or_safety_cost() -> None:
    env = oracle_env("recovery")
    first_action = choose_oracle_action(
        env,
        config=OracleConfig(
            decay=2.0,
            target_mode="nominal",
            allow_strategic_wait=False,
        ),
    )
    env.step(first_action, sync_mode="event")
    assert env._valid_mask_for(env.uavs, env.tasks)[:-1].any()
    assert env._next_completion_delta(env.uavs) > 0

    time_before = env.current_time
    invalid_before = env.invalid_actions
    safety_before = float(env.vector_return[-1])
    _, _, _, info = step_oracle_action(
        env, env.noop_action, sync_mode="event"
    )

    assert info["oracle_strategic_wait"] is True
    assert env.current_time > time_before
    assert env.invalid_actions == invalid_before
    assert float(env.vector_return[-1]) == pytest.approx(safety_before)


def test_oracle_dynamic_calibration_cli_records_share_and_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "dynamic_oracle"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_pcrl_oracle.py",
            "--profiles",
            "search",
            "--calibration-priority-share",
            "0.4",
            "--protocol-version",
            "pcrl-v0-hard-3-anchor-grid-coarse",
            "--scales",
            "2x12",
            "--episodes",
            "1",
            "--decays",
            "2.0",
            "--rollout-candidates",
            "0",
            "--output",
            str(output),
        ],
    )

    evaluate_pcrl_oracle.main()

    rows = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    expected = calibration_priority_profiles(0.4)
    assert len(rows) == len(expected)
    assert tuple(row["preference_profile"] for row in rows) == tuple(expected)
    for row in rows:
        name = row["preference_profile"]
        assert row["protocol_version"] == (
            "pcrl-v0-hard-3-anchor-grid-coarse"
        )
        assert row["calibration_priority_share"] == pytest.approx(0.4)
        assert row["calibration_profile_vector"] == pytest.approx(expected[name])
    assert summary["profile_set"] == "dynamic_calibration_priority_share"
    assert summary["protocol_version"] == (
        "pcrl-v0-hard-3-anchor-grid-coarse"
    )
    assert summary["calibration_priority_share"] == pytest.approx(0.4)
    assert summary["calibration_profile_vectors"] == pytest.approx(
        {name: vector.tolist() for name, vector in expected.items()}
    )

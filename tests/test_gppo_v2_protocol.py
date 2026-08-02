import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from evaluate_paper_gppo import main as evaluate_main
from run_paper_gppo_v2 import (
    REQUIRED_EVALUATION_METRICS,
    TRAINING_HYPERPARAMETER_FIELDS,
    checkpoint_matches,
    checkpoint_protocol_errors,
    evaluation_protocol_errors,
    expected_evaluation_hash,
    expected_scenario_hash,
    methods,
    parse_args as parse_runner_args,
    summary_evaluation_paths,
    train_commands,
)
from summarize_gppo_v2 import (
    LOWER_IS_BETTER,
    METRICS,
    mean_ci,
    negative_result_records,
    validate_protocol,
)
from train_paper_gppo import gae
from uav_assignment.gppo_v2 import (
    CORE_METHOD_IDS,
    METHOD_SPECS,
    config_hash,
    implementation_hash,
)
from uav_assignment.paper_env import PaperEnvConfig
from uav_assignment.paper_models import PaperHeteroActorCritic


WORKSPACE = Path(__file__).resolve().parents[1]


def manifest_payload() -> dict[str, object]:
    return json.loads(
        (WORKSPACE / "configs" / "gppo_v2_hard.json").read_text(encoding="utf-8")
    )


def test_manifest_contains_core_and_required_architecture_controls() -> None:
    manifest = manifest_payload()
    assert tuple(manifest["methods"]) == CORE_METHOD_IDS
    assert set(manifest["supplementary_methods"]) == {
        "gppo_event_single_head",
        "gppo_event_no_gate",
    }
    assert manifest["learned_seeds"] == [1, 2, 3, 4, 5]
    assert set(manifest["scale_deadlines"]) == set(manifest["evaluation_scales"])


def test_runner_defaults_to_frozen_supplementary_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = manifest_payload()
    monkeypatch.setattr(sys, "argv", ["run_paper_gppo_v2.py", "--phase", "train"])
    assert parse_runner_args().include_supplementary is True
    assert set(methods(manifest, True)) == set(manifest["methods"]) | set(
        manifest["supplementary_methods"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_paper_gppo_v2.py", "--phase", "train", "--exclude-supplementary"],
    )
    assert parse_runner_args().include_supplementary is False
    assert methods(manifest, False) == manifest["methods"]


def test_scenario_hash_ignores_only_training_seed() -> None:
    first = PaperEnvConfig(seed=1)
    second = PaperEnvConfig(seed=5)
    assert config_hash(first) == config_hash(second)
    assert config_hash(first) != config_hash(PaperEnvConfig(seed=1, workload_scale=1.1))


def test_five_seed_ci_uses_student_t() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lower, upper = mean_ci(values)
    expected_half = 2.776 * values.std(ddof=1) / np.sqrt(5)
    assert mean == 3.0
    assert lower == pytest.approx(3.0 - expected_half)
    assert upper == pytest.approx(3.0 + expected_half)


def test_gae_bootstraps_time_limit_truncation() -> None:
    advantages, returns = gae(
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
        gamma=0.99,
        lam=0.95,
        bootstrap_value=2.0,
    )
    assert advantages[0] == pytest.approx(1.98)
    assert returns[0] == pytest.approx(1.98)


def protocol_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_id in CORE_METHOD_IDS:
        spec = METHOD_SPECS[method_id]
        seeds = range(1, 6) if spec.learned else (-1,)
        for seed in seeds:
            for eval_seed in (50_000, 50_001):
                rows.append(
                    {
                        "method_id": method_id,
                        "training_seed": seed,
                        "eval_seed": eval_seed,
                        "scale": "3x16",
                        "scenario_hash": "locked",
                        "implementation_hash": "implementation",
                        "scenario_version": "gppo-v2-hard-test",
                    }
                )
    return rows


def test_protocol_validator_requires_all_methods_and_five_seeds() -> None:
    result = validate_protocol(protocol_rows(), allow_incomplete=False)
    assert result["valid"]
    incomplete = [row for row in protocol_rows() if row["method_id"] != "ppo_event"]
    with pytest.raises(ValueError, match="missing methods"):
        validate_protocol(incomplete, allow_incomplete=False)


def test_protocol_validator_rejects_scenario_hash_mismatch() -> None:
    rows = protocol_rows()
    rows[0]["scenario_hash"] = "different"
    with pytest.raises(ValueError, match="scenario hashes differ"):
        validate_protocol(rows, allow_incomplete=False)


def valid_checkpoint_payload(
    manifest: dict[str, object], method_id: str = "gppo_event", seed: int = 1
) -> dict[str, object]:
    spec = METHOD_SPECS[method_id]
    env_config = PaperEnvConfig(**manifest["scenario"]).to_dict()
    env_config["seed"] = seed
    scenario_hash = expected_scenario_hash(manifest)
    train_scales = [str(scale) for scale in manifest["train_scales"]]
    model_config = {
        "node_feature_dim": 24,
        "edge_feature_dim": 5,
        "max_uavs": manifest["scenario"]["max_uavs"],
        "max_tasks": manifest["scenario"]["max_tasks"],
        "hidden_dim": manifest["hidden_dim"],
        "graph_mode": spec.graph_mode,
    }
    training = {
        "seed": seed,
        "updates": manifest["updates"],
        "episodes_per_update": manifest["episodes_per_update"],
        "validation_episodes": manifest["validation_episodes"],
        "validation_interval": manifest["validation_interval"],
        "validation_seed": manifest["validation_seed"],
        "hidden_dim": manifest["hidden_dim"],
        "train_scales": train_scales,
        **{field: manifest[field] for field in TRAINING_HYPERPARAMETER_FIELDS},
    }
    return {
        "version": "paper-aligned-gppo-v2",
        "method_id": method_id,
        "scenario": manifest["version"],
        "scenario_hash": scenario_hash,
        "config_hash": scenario_hash,
        "implementation_hash": implementation_hash(),
        "algorithm": spec.algorithm,
        "graph_mode": spec.graph_mode,
        "sync_mode": spec.sync_mode,
        "env_config": env_config,
        "model_config": model_config,
        "model_state": PaperHeteroActorCritic(**model_config).state_dict(),
        "training": training,
        "train_scales": train_scales,
        "scale_deadlines": {
            scale: float(manifest["scale_deadlines"][scale]) for scale in train_scales
        },
    }


def set_nested(payload: dict[str, object], path: str, value: object) -> None:
    target = payload
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        ("method_id", "gppo_none", "method_id"),
        ("scenario", "wrong-version", "scenario"),
        ("scenario_hash", "stale", "scenario_hash"),
        ("config_hash", "stale", "config_hash"),
        ("implementation_hash", "stale", "implementation_hash"),
        ("algorithm", "ppo", "algorithm"),
        ("graph_mode", "single_head", "graph_mode"),
        ("sync_mode", "always", "sync_mode"),
        ("training.seed", 2, "training.seed"),
        ("training.updates", 99, "training.updates"),
        ("training.episodes_per_update", 17, "training.episodes_per_update"),
        ("training.validation_episodes", 19, "training.validation_episodes"),
        ("training.validation_interval", 9, "training.validation_interval"),
        ("training.validation_seed", 39999, "training.validation_seed"),
        ("training.hidden_dim", 32, "training.hidden_dim"),
        ("training.gamma", 0.9, "training.gamma"),
        ("training.learning_rate", 0.001, "training.learning_rate"),
        ("training.train_scales", None, "training.train_scales"),
        ("model_config.hidden_dim", 32, "model_config.hidden_dim"),
        ("model_config.graph_mode", "none", "model_config.graph_mode"),
        ("model_state", None, "model_state"),
        ("train_scales", ["3x16"], "train_scales"),
    ),
)
def test_checkpoint_protocol_rejects_every_frozen_field(
    path: str, value: object, message: str
) -> None:
    manifest = manifest_payload()
    payload = valid_checkpoint_payload(manifest)
    set_nested(payload, path, value)
    errors = checkpoint_protocol_errors(payload, "gppo_event", manifest, 1)
    assert any(message in error for error in errors)


def test_checkpoint_skip_gate_accepts_only_exact_protocol(tmp_path: Path) -> None:
    manifest = manifest_payload()
    payload = valid_checkpoint_payload(manifest)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    assert checkpoint_matches(
        checkpoint,
        "gppo_event",
        manifest,
        1,
        int(manifest["updates"]),
        int(manifest["episodes_per_update"]),
        int(manifest["validation_episodes"]),
        int(manifest["validation_interval"]),
    )
    payload["training"]["validation_seed"] += 1
    torch.save(payload, checkpoint)
    assert not checkpoint_matches(
        checkpoint,
        "gppo_event",
        manifest,
        1,
        int(manifest["updates"]),
        int(manifest["episodes_per_update"]),
        int(manifest["validation_episodes"]),
        int(manifest["validation_interval"]),
    )


def test_evaluator_rejects_stale_v2_checkpoint_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = manifest_payload()
    payload = valid_checkpoint_payload(manifest)
    payload["implementation_hash"] = "stale"
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_paper_gppo.py",
            "--checkpoint",
            str(checkpoint),
            "--scenario-config",
            str(WORKSPACE / "configs" / "gppo_v2_hard.json"),
            "--method-id",
            "gppo_event",
            "--expected-training-seed",
            "1",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="implementation_hash"):
        evaluate_main()
    assert not output.exists()


def test_evaluator_rejects_legacy_checkpoint_without_current_hash_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"version": "legacy"}, checkpoint)
    output = tmp_path / "legacy-evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_paper_gppo.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="not a verified"):
        evaluate_main()
    assert not output.exists()


def miniature_manifest() -> dict[str, object]:
    manifest = copy.deepcopy(manifest_payload())
    manifest["evaluation_scales"] = ["2x12", "3x16"]
    manifest["evaluation_episodes"] = 2
    manifest["learned_seeds"] = [1, 2]
    return manifest


def evaluation_rows_for(
    manifest: dict[str, object], method_id: str, training_seed: int
) -> list[dict[str, object]]:
    spec = METHOD_SPECS[method_id]
    rows: list[dict[str, object]] = []
    for scale in manifest["evaluation_scales"]:
        active_uavs, initial_tasks = (int(part) for part in scale.split("x"))
        mission_deadline = float(manifest["scale_deadlines"][scale])
        for offset in range(int(manifest["evaluation_episodes"])):
            row = {
                    "method_id": method_id,
                    "algorithm": spec.algorithm,
                    "graph_mode": spec.graph_mode,
                    "sync_mode": spec.sync_mode,
                    "training_seed": training_seed,
                    "scale": scale,
                    "active_uavs": active_uavs,
                    "initial_tasks": initial_tasks,
                    "mission_deadline": mission_deadline,
                    "eval_seed": int(manifest["evaluation_seed"]) + offset,
                    "scenario_hash": expected_evaluation_hash(
                        manifest, manifest["evaluation_scales"]
                    ),
                    "scenario_version": manifest["version"],
                    "implementation_hash": implementation_hash(),
                    "checkpoint_version": (
                        "paper-aligned-gppo-v2" if spec.learned else "scenario-config"
                    ),
                    "checkpoint_scenario_hash": (
                        expected_scenario_hash(manifest) if spec.learned else None
                    ),
                    "checkpoint_config_hash": (
                        expected_scenario_hash(manifest) if spec.learned else None
                    ),
                    "train_scales": (
                        list(manifest["train_scales"]) if spec.learned else None
                    ),
                    "training_updates": manifest["updates"] if spec.learned else None,
                    "training_episodes_per_update": (
                        manifest["episodes_per_update"] if spec.learned else None
                    ),
                    "validation_episodes": (
                        manifest["validation_episodes"] if spec.learned else None
                    ),
                    "validation_interval": (
                        manifest["validation_interval"] if spec.learned else None
                    ),
                    "validation_seed": (
                        manifest["validation_seed"] if spec.learned else None
                    ),
                    "hidden_dim": manifest["hidden_dim"] if spec.learned else None,
                    **{
                        f"training_{field}": (
                            manifest[field] if spec.learned else None
                        )
                        for field in TRAINING_HYPERPARAMETER_FIELDS
                    },
                }
            row.update({metric: 0.0 for metric in REQUIRED_EVALUATION_METRICS})
            rows.append(row)
    return rows


def full_protocol_rows(manifest: dict[str, object], include_supplementary: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_id in methods(manifest, include_supplementary):
        spec = METHOD_SPECS[method_id]
        seeds = manifest["learned_seeds"] if spec.learned else [-1]
        for seed in seeds:
            rows.extend(evaluation_rows_for(manifest, method_id, int(seed)))
    return rows


def test_evaluation_reuse_gate_requires_exact_scales_counts_and_seed_range() -> None:
    manifest = miniature_manifest()
    rows = evaluation_rows_for(manifest, "gppo_event", 1)
    assert not evaluation_protocol_errors(
        rows,
        "gppo_event",
        1,
        manifest,
        manifest["evaluation_scales"],
        int(manifest["evaluation_episodes"]),
    )
    rows[0]["eval_seed"] += 10
    errors = evaluation_protocol_errors(
        rows,
        "gppo_event",
        1,
        manifest,
        manifest["evaluation_scales"],
        int(manifest["evaluation_episodes"]),
    )
    assert any("contiguous" in error for error in errors)


def test_evaluation_reuse_gate_requires_scale_identity_metrics_and_config_hash() -> None:
    manifest = miniature_manifest()
    rows = evaluation_rows_for(manifest, "gppo_event", 1)
    rows[0]["active_uavs"] = 99
    rows[1]["heartbeat_messages"] = float("nan")
    rows[2]["checkpoint_config_hash"] = None
    errors = evaluation_protocol_errors(
        rows,
        "gppo_event",
        1,
        manifest,
        manifest["evaluation_scales"],
        int(manifest["evaluation_episodes"]),
    )
    assert any("active_uavs" in error for error in errors)
    assert any("heartbeat_messages is not finite" in error for error in errors)
    assert any("checkpoint_config_hash" in error for error in errors)


def test_summary_protocol_requires_manifest_methods_scales_seeds_and_counts() -> None:
    manifest = miniature_manifest()
    rows = full_protocol_rows(manifest, include_supplementary=True)
    result = validate_protocol(rows, False, manifest, include_supplementary=True)
    assert result["valid"]
    assert set(result["methods"]) == set(methods(manifest, True))

    missing_supplementary = [
        row
        for row in rows
        if row["method_id"] not in set(manifest["supplementary_methods"])
    ]
    with pytest.raises(ValueError, match="missing methods"):
        validate_protocol(
            missing_supplementary, False, manifest, include_supplementary=True
        )
    core_only = validate_protocol(
        missing_supplementary, False, manifest, include_supplementary=False
    )
    assert core_only["valid"]

    duplicate = copy.deepcopy(rows)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="episodes|duplicate"):
        validate_protocol(duplicate, False, manifest, include_supplementary=True)


def test_heartbeat_metric_and_negative_result_reporting() -> None:
    assert "heartbeat_messages" in METRICS
    assert "heartbeat_messages" in LOWER_IS_BETTER
    comparison = {
        "comparison": "left-minus-right",
        "left": "left",
        "right": "right",
        "scale": "2x12",
        "paired_training_seeds": [1, 2],
    }
    for metric in METRICS:
        comparison.update(
            {
                f"{metric}_effect": "positive",
                f"{metric}_delta": 1.0,
                f"{metric}_delta_ci95_lower": 0.5,
                f"{metric}_delta_ci95_upper": 1.5,
                f"{metric}_p_raw": 0.5,
                f"{metric}_p_holm": 1.0,
            }
        )
    assert not negative_result_records([comparison])
    comparison["heartbeat_messages_effect"] = "inconclusive"
    records = negative_result_records([comparison])
    assert records == [
        {
            "comparison": "left-minus-right",
            "left": "left",
            "right": "right",
            "scale": "2x12",
            "paired_training_seeds": [1, 2],
            "metric": "heartbeat_messages",
            "effect": "inconclusive",
            "delta": 1.0,
            "delta_ci95_lower": 0.5,
            "delta_ci95_upper": 1.5,
            "p_raw": 0.5,
            "p_holm": 1.0,
        }
    ]


def test_train_commands_freeze_manifest_hyperparameters(tmp_path: Path) -> None:
    manifest = manifest_payload()
    args = type(
        "Args",
        (),
        {
            "include_supplementary": True,
            "force": True,
            "manifest": WORKSPACE / "configs" / "gppo_v2_hard.json",
        },
    )()
    command = train_commands(args, manifest, tmp_path, smoke=False)[0]
    for field, flag in (
        ("hidden_dim", "--hidden-dim"),
        ("learning_rate", "--learning-rate"),
        ("entropy_coefficient", "--entropy-coefficient"),
        ("gamma", "--gamma"),
        ("gae_lambda", "--gae-lambda"),
        ("update_epochs", "--update-epochs"),
        ("minibatch_size", "--minibatch-size"),
    ):
        assert command[command.index(flag) + 1] == str(manifest[field])


def test_core_only_summary_path_selection_excludes_existing_supplementary(
    tmp_path: Path,
) -> None:
    manifest = manifest_payload()
    for method_id in ("gppo_event", "gppo_event_single_head"):
        path = tmp_path / "eval" / method_id / "seed_1" / "evaluation.json"
        path.parent.mkdir(parents=True)
        path.write_text("[]", encoding="utf-8")
    selected = summary_evaluation_paths(
        tmp_path, manifest, include_supplementary=False
    )
    assert [path.parts[-3] for path in selected] == ["gppo_event"]

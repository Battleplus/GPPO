from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paper_env import PaperEnvConfig


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    algorithm: str
    graph_mode: str
    sync_mode: str
    learned: bool


METHOD_SPECS = {
    spec.method_id: spec
    for spec in (
        MethodSpec("random_event", "random", "none", "event", False),
        MethodSpec("greedy_event", "greedy", "none", "event", False),
        MethodSpec("ppo_none", "ppo", "none", "none", True),
        MethodSpec("ppo_event", "ppo", "none", "event", True),
        MethodSpec("gppo_none", "gppo", "adaptive", "none", True),
        MethodSpec("gppo_event", "gppo", "adaptive", "event", True),
        MethodSpec("gppo_periodic", "gppo", "adaptive", "periodic", True),
        MethodSpec("gppo_always", "gppo", "adaptive", "always", True),
        MethodSpec(
            "gppo_event_single_head", "gppo", "single_head", "event", True
        ),
        MethodSpec(
            "gppo_event_no_gate", "gppo", "adaptive_no_gate", "event", True
        ),
    )
}

CORE_METHOD_IDS = (
    "random_event",
    "greedy_event",
    "ppo_none",
    "ppo_event",
    "gppo_none",
    "gppo_event",
    "gppo_periodic",
    "gppo_always",
)


def hard_v2_config(**overrides: Any) -> PaperEnvConfig:
    values: dict[str, Any] = {
        "max_uavs": 4,
        "max_tasks": 24,
        "active_uavs": 3,
        "initial_tasks": 20,
        "max_decisions": 200,
        "mission_deadline": 16.0,
        "stop_at_deadline": False,
        "task_chain_length": 5,
        "workload_scale": 1.8,
        "periodic_interval": 2.0,
        "weather_probability": 0.06,
        "failure_probability": 0.02,
        "task_change_probability": 0.04,
        "communication_drop_probability": 0.04,
        "include_engineering_rewards": False,
    }
    values.update(overrides)
    return PaperEnvConfig(**values)


def config_hash(config: PaperEnvConfig | dict[str, Any]) -> str:
    payload = dict(config.to_dict() if isinstance(config, PaperEnvConfig) else config)

    def strip_seed(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_seed(item)
                for key, item in value.items()
                if key != "seed"
            }
        if isinstance(value, list):
            return [strip_seed(item) for item in value]
        return value

    payload = strip_seed(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def method_manifest() -> list[dict[str, Any]]:
    return [asdict(METHOD_SPECS[method_id]) for method_id in CORE_METHOD_IDS]


def implementation_hash() -> str:
    workspace = Path(__file__).resolve().parents[2]
    paths = (
        workspace / "src" / "uav_assignment" / "paper_env.py",
        workspace / "src" / "uav_assignment" / "paper_models.py",
        workspace / "src" / "uav_assignment" / "gppo_v2.py",
        workspace / "train_paper_gppo.py",
        workspace / "evaluate_paper_gppo.py",
        workspace / "summarize_gppo_v2.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(workspace).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()

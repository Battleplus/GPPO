from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .engine import DisturbanceEngine
from .events import DisturbanceEvent


OBJECTIVE_COMPONENTS = (
    "makespan_component",
    "task_success_component",
    "deadline_component",
    "energy_component",
    "communication_component",
    "reallocation_component",
    "stability_component",
)


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, DisturbanceEvent):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported trajectory value: {type(value).__name__}")


def decode_arrays(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"]))
    if isinstance(value, dict):
        return {key: decode_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_arrays(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TrajectoryMetadata:
    episode_id: str
    instance_seed: int
    disturbance_seed: int
    training_seed: int | None
    config_sha256: str
    tape_sha256: str
    schema_version: str = "phase1b-standard-trajectory-v1"


class Phase1BTrajectoryRecorder:
    def __init__(self, metadata: TrajectoryMetadata):
        if not metadata.episode_id:
            raise ValueError("episode_id must be non-empty")
        self.metadata = metadata
        self.records: list[dict[str, Any]] = []
        self._finalized = False

    @classmethod
    def for_engine(
        cls, engine: DisturbanceEngine, *, episode_id: str
    ) -> "Phase1BTrajectoryRecorder":
        return cls(
            TrajectoryMetadata(
                episode_id=episode_id,
                instance_seed=engine.config.instance_seed,
                disturbance_seed=engine.config.disturbance_seed,
                training_seed=engine.config.training_seed,
                config_sha256=engine.config.sha256,
                tape_sha256=engine.tape.sha256,
            )
        )

    def record_decision(
        self,
        *,
        decision_index: int,
        physical_time: float,
        partial_graph_observation: Mapping[str, Any],
        true_graph_state: Mapping[str, Any],
        belief_cache: Mapping[str, Any],
        legal_action_mask: Any,
        selected_action: int,
        communication_history: Iterable[Mapping[str, Any]],
        messages_sent: Iterable[Mapping[str, Any]],
        messages_delivered: Iterable[Mapping[str, Any]],
        messages_dropped: Iterable[Mapping[str, Any]],
        message_delays: Iterable[float],
        network_components: Iterable[Iterable[str]],
        uav_energy: Mapping[str, float],
        uav_alive: Mapping[str, bool],
        task_status: Mapping[str, str],
        task_priority: Mapping[str, float],
        task_deadline: Mapping[str, float | None],
        current_events: Iterable[DisturbanceEvent | Mapping[str, Any]],
        event_observed_delay: Mapping[str, float],
        objective_components: Mapping[str, float],
    ) -> None:
        if self._finalized:
            raise ValueError("cannot append to a finalized trajectory")
        if decision_index != len(self.records):
            raise ValueError("decision_index must be zero-based and contiguous")
        if physical_time < 0 or (
            self.records and physical_time < self.records[-1]["physical_time"]
        ):
            raise ValueError("trajectory physical time cannot move backwards")
        missing = set(OBJECTIVE_COMPONENTS) - objective_components.keys()
        extra = objective_components.keys() - set(OBJECTIVE_COMPONENTS)
        if missing or extra:
            raise ValueError(
                f"objective vector mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if any(not np.isfinite(float(value)) for value in objective_components.values()):
            raise ValueError("objective vector must contain finite values")
        event_payloads = [
            event.to_dict() if isinstance(event, DisturbanceEvent) else dict(event)
            for event in current_events
        ]
        record = {
            "episode_id": self.metadata.episode_id,
            "decision_index": int(decision_index),
            "physical_time": float(physical_time),
            "partial_graph_observation": _encode(partial_graph_observation),
            "true_graph_state": _encode(true_graph_state),
            "belief_cache": _encode(belief_cache),
            "legal_action_mask": _encode(legal_action_mask),
            "selected_action": int(selected_action),
            "communication_history": _encode(list(communication_history)),
            "messages_sent": _encode(list(messages_sent)),
            "messages_delivered": _encode(list(messages_delivered)),
            "messages_dropped": _encode(list(messages_dropped)),
            "message_delays": _encode(list(message_delays)),
            "network_components": _encode([list(group) for group in network_components]),
            "uav_energy": _encode(uav_energy),
            "uav_alive": _encode(uav_alive),
            "task_status": _encode(task_status),
            "task_priority": _encode(task_priority),
            "task_deadline": _encode(task_deadline),
            "current_events": _encode(event_payloads),
            "future_event_targets_1_to_5": None,
            "event_observed_delay": _encode(event_observed_delay),
            **{key: float(objective_components[key]) for key in OBJECTIVE_COMPONENTS},
        }
        self.records.append(record)

    def finalize(self, engine: DisturbanceEngine) -> None:
        if self.metadata.config_sha256 != engine.config.sha256:
            raise ValueError("trajectory config hash does not match engine")
        if self.metadata.tape_sha256 != engine.tape.sha256:
            raise ValueError("trajectory tape hash does not match engine")
        times = tuple(float(record["physical_time"]) for record in self.records)
        for index, record in enumerate(self.records):
            record["future_event_targets_1_to_5"] = _encode(
                engine.future_event_targets(
                    times, current_decision_index=index, horizons=5
                )
            )
        self._finalized = True

    def to_dict(self) -> dict[str, Any]:
        if not self._finalized:
            raise ValueError("trajectory must be finalized before serialization")
        return {
            "metadata": _encode(self.metadata.__dict__ if hasattr(self.metadata, "__dict__") else {
                "episode_id": self.metadata.episode_id,
                "instance_seed": self.metadata.instance_seed,
                "disturbance_seed": self.metadata.disturbance_seed,
                "training_seed": self.metadata.training_seed,
                "config_sha256": self.metadata.config_sha256,
                "tape_sha256": self.metadata.tape_sha256,
                "schema_version": self.metadata.schema_version,
            }),
            "records": self.records,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "Phase1BTrajectoryRecorder":
        payload = json.loads(text)
        recorder = cls(TrajectoryMetadata(**payload["metadata"]))
        recorder.records = list(payload["records"])
        recorder._finalized = True
        # Round-trip through canonical JSON also validates NaN and schema serializability.
        recorder.canonical_json()
        return recorder

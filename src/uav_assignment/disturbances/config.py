from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


SeverityName = Literal["off", "weak", "medium", "strong", "combined"]


@dataclass(frozen=True, slots=True)
class SourceConfig:
    enabled: bool = False
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical = json.loads(
            json.dumps(dict(self.parameters), sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        object.__setattr__(self, "parameters", canonical)


SOURCE_NAMES = (
    "gilbert_elliott_packet_loss",
    "message_delay",
    "network_partition",
    "uav_failure",
    "energy_depletion",
    "task_arrival",
    "task_cancellation",
    "task_priority_change",
    "task_deadline_change",
    "wind_field",
)


@dataclass(frozen=True, slots=True)
class DisturbanceConfig:
    """Frozen episode configuration; training, instance and disturbance seeds stay separate."""

    version: str = "phase1b-disturbance-config-v1"
    severity: SeverityName = "off"
    instance_seed: int = 0
    disturbance_seed: int = 0
    training_seed: int | None = None
    gilbert_elliott_packet_loss: SourceConfig = field(default_factory=SourceConfig)
    message_delay: SourceConfig = field(default_factory=SourceConfig)
    network_partition: SourceConfig = field(default_factory=SourceConfig)
    uav_failure: SourceConfig = field(default_factory=SourceConfig)
    energy_depletion: SourceConfig = field(default_factory=SourceConfig)
    task_arrival: SourceConfig = field(default_factory=SourceConfig)
    task_cancellation: SourceConfig = field(default_factory=SourceConfig)
    task_priority_change: SourceConfig = field(default_factory=SourceConfig)
    task_deadline_change: SourceConfig = field(default_factory=SourceConfig)
    wind_field: SourceConfig = field(default_factory=SourceConfig)

    def __post_init__(self) -> None:
        if self.severity not in {"off", "weak", "medium", "strong", "combined"}:
            raise ValueError(f"unsupported severity: {self.severity}")
        for name in ("instance_seed", "disturbance_seed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.training_seed is not None and self.training_seed < 0:
            raise ValueError("training_seed cannot be negative")

    @property
    def all_disabled(self) -> bool:
        return not any(getattr(self, name).enabled for name in SOURCE_NAMES)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def source_seed(self, source: str) -> int:
        if source not in SOURCE_NAMES:
            raise KeyError(source)
        material = f"{self.version}|{self.instance_seed}|{self.disturbance_seed}|{source}"
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DisturbanceConfig":
        values = dict(payload)
        for name in SOURCE_NAMES:
            if name in values and not isinstance(values[name], SourceConfig):
                values[name] = SourceConfig(**values[name])
        return cls(**values)

    @classmethod
    def from_json(cls, text: str) -> "DisturbanceConfig":
        return cls.from_dict(json.loads(text))

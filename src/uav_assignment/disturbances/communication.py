from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .config import DisturbanceConfig
from .events import DisturbanceEvent


def _uniform(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _parameters(config: DisturbanceConfig, source: str) -> dict[str, Any]:
    return dict(getattr(config, source).parameters)


def generate_communication_events(
    config: DisturbanceConfig,
    *,
    horizon: float,
    link_ids: Iterable[str],
) -> tuple[DisturbanceEvent, ...]:
    """Generate all channel-state and partition ground truth before an episode."""

    if horizon < 0:
        raise ValueError("horizon cannot be negative")
    events: list[DisturbanceEvent] = []
    generation_index = 0
    ge = config.gilbert_elliott_packet_loss
    if ge.enabled:
        params = _parameters(config, "gilbert_elliott_packet_loss")
        tick = float(params.get("tick", 1.0))
        p_gb = float(params.get("p_gb", 0.05))
        p_bg = float(params.get("p_bg", 0.35))
        loss_good = float(params.get("loss_good", 0.01))
        loss_bad = float(params.get("loss_bad", 0.8))
        initial_state = str(params.get("initial_state", "good"))
        if tick <= 0 or not all(0 <= value <= 1 for value in (p_gb, p_bg, loss_good, loss_bad)):
            raise ValueError("invalid Gilbert-Elliott parameters")
        if initial_state not in {"good", "bad"}:
            raise ValueError("initial_state must be good or bad")
        for link_id in sorted(set(link_ids)):
            rng = np.random.default_rng(config.source_seed("gilbert_elliott_packet_loss") ^ _stable_int(link_id))
            state = initial_state
            step = 0
            while step * tick <= horizon:
                physical_time = step * tick
                loss_probability = loss_good if state == "good" else loss_bad
                events.append(
                    DisturbanceEvent(
                        event_id=f"ge:{link_id}:{step}",
                        event_type="link_state",
                        physical_time=physical_time,
                        source="gilbert_elliott_packet_loss",
                        target=link_id,
                        severity=loss_probability,
                        payload={"state": state, "loss_probability": loss_probability, "tick": tick},
                        ground_truth={"state": state, "loss_probability": loss_probability},
                        observed_time=None,
                        generation_index=generation_index,
                        source_priority=10,
                    )
                )
                generation_index += 1
                draw = float(rng.random())
                if state == "good" and draw < p_gb:
                    state = "bad"
                elif state == "bad" and draw < p_bg:
                    state = "good"
                step += 1
    delay = config.message_delay
    if delay.enabled:
        params = _parameters(config, "message_delay")
        minimum = float(params.get("minimum", 0.0))
        maximum = float(params.get("maximum", minimum))
        ttl = float(params.get("ttl", max(1.0, maximum * 4)))
        if minimum < 0 or maximum < minimum or ttl <= 0:
            raise ValueError("invalid message delay parameters")
        events.append(
            DisturbanceEvent(
                event_id="delay:profile",
                event_type="delay_profile",
                physical_time=0.0,
                source="message_delay",
                target="network",
                severity=maximum,
                payload={"minimum": minimum, "maximum": maximum, "ttl": ttl},
                ground_truth={"distribution": "deterministic-keyed-uniform"},
                observed_time=None,
                generation_index=generation_index,
                source_priority=20,
            )
        )
        generation_index += 1
    partition = config.network_partition
    if partition.enabled:
        intervals = _parameters(config, "network_partition").get("intervals", [])
        for index, interval in enumerate(intervals):
            start = float(interval["start"])
            end = float(interval["end"])
            groups = [sorted(str(node) for node in group) for group in interval["groups"]]
            if start < 0 or end <= start or end > horizon:
                raise ValueError("invalid partition interval")
            flattened = [node for group in groups for node in group]
            if len(flattened) != len(set(flattened)):
                raise ValueError("a node cannot appear in multiple partition groups")
            events.append(
                DisturbanceEvent(
                    event_id=f"partition:{index}",
                    event_type="network_partition",
                    physical_time=start,
                    source="network_partition",
                    target="network",
                    severity=float(interval.get("severity", 1.0)),
                    payload={"start": start, "end": end, "groups": groups},
                    ground_truth={"components": groups},
                    observed_time=None,
                    generation_index=generation_index,
                    source_priority=30,
                )
            )
            generation_index += 1
    return tuple(sorted(events, key=lambda item: item.sort_key))


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class MessageEnvelope:
    message_id: str
    source: str
    target: str
    link_id: str
    sent_time: float
    arrival_time: float
    expiry_time: float
    byte_count: int
    payload: Mapping[str, Any]
    dropped: bool = False
    drop_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.message_id:
            raise ValueError("message_id must be non-empty")
        if self.sent_time < 0 or self.arrival_time < self.sent_time:
            raise ValueError("message arrival cannot precede send time")
        if self.expiry_time < self.sent_time:
            raise ValueError("message expiry cannot precede send time")
        if self.byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        canonical = json.loads(json.dumps(dict(self.payload), sort_keys=True, allow_nan=False))
        object.__setattr__(self, "payload", canonical)


@dataclass(slots=True)
class CommunicationAudit:
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_expired: int = 0
    bytes_sent: int = 0
    bytes_delivered: int = 0
    total_delay: float = 0.0
    drop_reasons: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped,
            "messages_expired": self.messages_expired,
            "bytes_sent": self.bytes_sent,
            "bytes_delivered": self.bytes_delivered,
            "mean_delivered_delay": self.total_delay / max(self.messages_delivered, 1),
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
        }


class CommunicationDisturbanceLayer:
    """Message-only weak-communication layer driven by a pre-generated tape."""

    def __init__(self, config: DisturbanceConfig, events: Iterable[DisturbanceEvent]):
        self.config = config
        self.events = tuple(sorted(events, key=lambda item: item.sort_key))
        self.link_state: dict[str, tuple[str, float]] = {}
        self.delay_profile = {"minimum": 0.0, "maximum": 0.0, "ttl": float("inf")}
        self.partitions = tuple(
            event for event in self.events if event.event_type == "network_partition"
        )
        self._event_index = 0
        self._time = 0.0
        self._queue: list[tuple[float, int, MessageEnvelope]] = []
        self._queue_index = 0
        self._message_ids: set[str] = set()
        self.audit = CommunicationAudit()

    def advance(self, physical_time: float) -> None:
        physical_time = float(physical_time)
        if physical_time < self._time:
            raise ValueError("communication time cannot move backwards")
        while self._event_index < len(self.events):
            event = self.events[self._event_index]
            if event.physical_time > physical_time:
                break
            if event.event_type == "link_state":
                self.link_state[event.target] = (
                    str(event.payload["state"]), float(event.payload["loss_probability"])
                )
            elif event.event_type == "delay_profile":
                self.delay_profile = {
                    "minimum": float(event.payload["minimum"]),
                    "maximum": float(event.payload["maximum"]),
                    "ttl": float(event.payload["ttl"]),
                }
            self._event_index += 1
        self._time = physical_time

    def _partition_end(self, source: str, target: str, sent_time: float) -> float | None:
        for event in self.partitions:
            start = float(event.payload["start"])
            end = float(event.payload["end"])
            if not start <= sent_time < end:
                continue
            groups = [set(group) for group in event.payload["groups"]]
            source_group = next((index for index, group in enumerate(groups) if source in group), None)
            target_group = next((index for index, group in enumerate(groups) if target in group), None)
            if source_group is not None and target_group is not None and source_group != target_group:
                return end
        return None

    def send(
        self,
        *,
        message_id: str,
        source: str,
        target: str,
        link_id: str,
        sent_time: float,
        byte_count: int,
        payload: Mapping[str, Any],
    ) -> MessageEnvelope:
        if message_id in self._message_ids:
            raise ValueError(f"duplicate message_id: {message_id}")
        self.advance(sent_time)
        self._message_ids.add(message_id)
        self.audit.messages_sent += 1
        self.audit.bytes_sent += int(byte_count)
        _, loss_probability = self.link_state.get(link_id, ("good", 0.0))
        dropped = _uniform(
            self.config.source_seed("gilbert_elliott_packet_loss"), f"drop|{message_id}|{link_id}"
        ) < loss_probability
        minimum = self.delay_profile["minimum"]
        maximum = self.delay_profile["maximum"]
        delay_draw = _uniform(self.config.source_seed("message_delay"), f"delay|{message_id}|{link_id}")
        delay = minimum + (maximum - minimum) * delay_draw
        arrival_time = float(sent_time + delay)
        partition_end = self._partition_end(source, target, sent_time)
        if partition_end is not None:
            arrival_time = max(arrival_time, partition_end)
        expiry_time = float(sent_time + self.delay_profile["ttl"])
        envelope = MessageEnvelope(
            message_id=message_id,
            source=source,
            target=target,
            link_id=link_id,
            sent_time=float(sent_time),
            arrival_time=arrival_time,
            expiry_time=expiry_time,
            byte_count=int(byte_count),
            payload=payload,
            dropped=dropped,
            drop_reason="packet_loss" if dropped else None,
        )
        if dropped:
            self.audit.messages_dropped += 1
            self.audit.drop_reasons["packet_loss"] = self.audit.drop_reasons.get("packet_loss", 0) + 1
        else:
            heapq.heappush(self._queue, (arrival_time, self._queue_index, envelope))
            self._queue_index += 1
        return envelope

    def deliver_until(self, physical_time: float) -> tuple[MessageEnvelope, ...]:
        self.advance(physical_time)
        delivered: list[MessageEnvelope] = []
        while self._queue and self._queue[0][0] <= physical_time:
            _, _, envelope = heapq.heappop(self._queue)
            if envelope.arrival_time > envelope.expiry_time:
                self.audit.messages_expired += 1
                self.audit.drop_reasons["expired"] = self.audit.drop_reasons.get("expired", 0) + 1
                continue
            delivered.append(envelope)
            self.audit.messages_delivered += 1
            self.audit.bytes_delivered += envelope.byte_count
            self.audit.total_delay += envelope.arrival_time - envelope.sent_time
        return tuple(delivered)

    @property
    def pending_messages(self) -> int:
        return len(self._queue)

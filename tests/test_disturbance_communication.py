from __future__ import annotations

from uav_assignment.disturbances import (
    CommunicationDisturbanceLayer,
    DisturbanceConfig,
    DisturbanceTape,
    SourceConfig,
    generate_communication_events,
)


def config(**overrides) -> DisturbanceConfig:
    values = {
        "instance_seed": 101,
        "disturbance_seed": 202,
        "gilbert_elliott_packet_loss": SourceConfig(
            True,
            {"tick": 1.0, "p_gb": 1.0, "p_bg": 0.0, "loss_good": 0.0, "loss_bad": 1.0},
        ),
        "message_delay": SourceConfig(True, {"minimum": 2.0, "maximum": 2.0, "ttl": 20.0}),
    }
    values.update(overrides)
    return DisturbanceConfig(**values)


def test_communication_tape_is_deterministic_and_bad_state_is_bursty() -> None:
    cfg = config()
    left = generate_communication_events(cfg, horizon=4.0, link_ids=("a-b",))
    right = generate_communication_events(cfg, horizon=4.0, link_ids=("a-b",))
    assert DisturbanceTape.build(cfg, left).sha256 == DisturbanceTape.build(cfg, right).sha256
    states = [item.payload["state"] for item in left if item.event_type == "link_state"]
    assert states == ["good", "bad", "bad", "bad", "bad"]


def test_delay_queue_never_delivers_a_message_early() -> None:
    cfg = config(
        gilbert_elliott_packet_loss=SourceConfig(False),
        message_delay=SourceConfig(True, {"minimum": 2.0, "maximum": 2.0, "ttl": 5.0}),
    )
    events = generate_communication_events(cfg, horizon=10.0, link_ids=("a-b",))
    layer = CommunicationDisturbanceLayer(cfg, events)
    message = layer.send(
        message_id="m1", source="a", target="b", link_id="a-b",
        sent_time=1.0, byte_count=32, payload={"belief": 1},
    )
    assert message.arrival_time == 3.0
    assert layer.deliver_until(2.999) == ()
    assert [item.message_id for item in layer.deliver_until(3.0)] == ["m1"]
    assert layer.audit.to_dict()["bytes_delivered"] == 32


def test_packet_loss_changes_delivery_only_and_is_audited() -> None:
    cfg = config()
    events = generate_communication_events(cfg, horizon=5.0, link_ids=("a-b",))
    layer = CommunicationDisturbanceLayer(cfg, events)
    true_state = {"uav_position": [0.1, 0.2]}
    envelope = layer.send(
        message_id="lost", source="a", target="b", link_id="a-b",
        sent_time=1.5, byte_count=64, payload={"copy": dict(true_state)},
    )
    assert envelope.dropped
    assert layer.deliver_until(5.0) == ()
    assert true_state == {"uav_position": [0.1, 0.2]}
    assert layer.audit.messages_dropped == 1
    assert layer.audit.drop_reasons == {"packet_loss": 1}


def test_partition_blocks_cross_component_visibility_until_recovery() -> None:
    cfg = config(
        gilbert_elliott_packet_loss=SourceConfig(False),
        message_delay=SourceConfig(True, {"minimum": 0.5, "maximum": 0.5, "ttl": 20.0}),
        network_partition=SourceConfig(
            True,
            {"intervals": [{"start": 2.0, "end": 6.0, "groups": [["a"], ["b"]]}]},
        ),
    )
    events = generate_communication_events(cfg, horizon=10.0, link_ids=("a-b",))
    layer = CommunicationDisturbanceLayer(cfg, events)
    message = layer.send(
        message_id="partitioned", source="a", target="b", link_id="a-b",
        sent_time=3.0, byte_count=8, payload={},
    )
    assert message.arrival_time == 6.0
    assert layer.deliver_until(5.999) == ()
    assert [item.message_id for item in layer.deliver_until(6.0)] == ["partitioned"]


def test_partition_delayed_message_expires_without_becoming_visible() -> None:
    cfg = config(
        gilbert_elliott_packet_loss=SourceConfig(False),
        message_delay=SourceConfig(True, {"minimum": 0.0, "maximum": 0.0, "ttl": 1.0}),
        network_partition=SourceConfig(
            True,
            {"intervals": [{"start": 1.0, "end": 5.0, "groups": [["a"], ["b"]]}]},
        ),
    )
    layer = CommunicationDisturbanceLayer(
        cfg, generate_communication_events(cfg, horizon=6.0, link_ids=("a-b",))
    )
    layer.send(
        message_id="expired", source="a", target="b", link_id="a-b",
        sent_time=2.0, byte_count=8, payload={},
    )
    assert layer.deliver_until(5.0) == ()
    assert layer.audit.messages_expired == 1
    assert layer.pending_messages == 0

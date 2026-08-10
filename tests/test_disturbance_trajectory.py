from __future__ import annotations

import numpy as np
import pytest

from uav_assignment.disturbances import (
    OBJECTIVE_COMPONENTS,
    DisturbanceConfig,
    DisturbanceEngine,
    Phase1BTrajectoryRecorder,
    SourceConfig,
    TaskRuntimeState,
    decode_arrays,
)


def engine() -> DisturbanceEngine:
    config = DisturbanceConfig(
        instance_seed=1,
        disturbance_seed=2,
        training_seed=3,
        task_priority_change=SourceConfig(
            True, {"events": [{"time": 2.0, "task_id": "t0", "priority": 4.0}]}
        ),
    )
    return DisturbanceEngine(
        config, horizon=10.0, uav_ids=("u0",), tasks=(TaskRuntimeState("t0"),)
    )


def objective(value: float = 0.0) -> dict[str, float]:
    return {name: value + index for index, name in enumerate(OBJECTIVE_COMPONENTS)}


def append(recorder: Phase1BTrajectoryRecorder, index: int, time: float) -> None:
    recorder.record_decision(
        decision_index=index,
        physical_time=time,
        partial_graph_observation={"nodes": np.asarray([[index]], dtype=np.float32)},
        true_graph_state={"nodes": np.asarray([[index + 10]], dtype=np.float32)},
        belief_cache={"age": np.asarray([time], dtype=np.float64)},
        legal_action_mask=np.asarray([True, False]),
        selected_action=0,
        communication_history=[{"mode": "event"}],
        messages_sent=[{"id": f"sent-{index}"}],
        messages_delivered=[],
        messages_dropped=[],
        message_delays=[],
        network_components=[["u0"]],
        uav_energy={"u0": 1.0},
        uav_alive={"u0": True},
        task_status={"t0": "pending"},
        task_priority={"t0": 1.0},
        task_deadline={"t0": None},
        current_events=[],
        event_observed_delay={},
        objective_components=objective(float(index)),
    )


def test_standard_trajectory_round_trip_preserves_arrays_hashes_and_seed_domains() -> None:
    runtime = engine()
    recorder = Phase1BTrajectoryRecorder.for_engine(runtime, episode_id="episode-1")
    append(recorder, 0, 0.0)
    append(recorder, 1, 2.0)
    append(recorder, 2, 3.0)
    recorder.finalize(runtime)
    restored = Phase1BTrajectoryRecorder.from_json(recorder.canonical_json())
    assert restored.sha256 == recorder.sha256
    assert restored.metadata.instance_seed == 1
    assert restored.metadata.disturbance_seed == 2
    assert restored.metadata.training_seed == 3
    arrays = decode_arrays(restored.records[0]["partial_graph_observation"])
    assert arrays["nodes"].dtype == np.float32
    assert np.array_equal(arrays["nodes"], np.asarray([[0]], dtype=np.float32))
    assert restored.records[0]["partial_graph_observation"] != restored.records[0]["true_graph_state"]


def test_future_one_to_five_labels_are_finalized_per_decision_window() -> None:
    runtime = engine()
    recorder = Phase1BTrajectoryRecorder.for_engine(runtime, episode_id="episode-labels")
    for index, time in enumerate((0.0, 1.0, 2.0, 3.0)):
        append(recorder, index, time)
    recorder.finalize(runtime)
    first = recorder.records[0]["future_event_targets_1_to_5"]
    assert first[0]["events"] == []
    assert first[1]["events"][0]["event_type"] == "task_priority_change"
    assert first[3]["available"] is False


def test_objective_vector_is_not_silently_scalarized_or_malformed() -> None:
    runtime = engine()
    recorder = Phase1BTrajectoryRecorder.for_engine(runtime, episode_id="episode-objective")
    bad = objective()
    del bad[OBJECTIVE_COMPONENTS[-1]]
    with pytest.raises(ValueError, match="objective vector mismatch"):
        recorder.record_decision(
            decision_index=0, physical_time=0.0,
            partial_graph_observation={}, true_graph_state={}, belief_cache={},
            legal_action_mask=[True], selected_action=0,
            communication_history=[], messages_sent=[], messages_delivered=[],
            messages_dropped=[], message_delays=[], network_components=[],
            uav_energy={}, uav_alive={}, task_status={}, task_priority={}, task_deadline={},
            current_events=[], event_observed_delay={}, objective_components=bad,
        )


def test_decision_indices_and_time_must_be_contiguous_and_monotonic() -> None:
    runtime = engine()
    recorder = Phase1BTrajectoryRecorder.for_engine(runtime, episode_id="episode-order")
    append(recorder, 0, 2.0)
    with pytest.raises(ValueError, match="contiguous"):
        append(recorder, 2, 3.0)
    with pytest.raises(ValueError, match="backwards"):
        append(recorder, 1, 1.0)


def test_unfinalized_trajectory_cannot_be_serialized() -> None:
    recorder = Phase1BTrajectoryRecorder.for_engine(engine(), episode_id="episode-open")
    append(recorder, 0, 0.0)
    with pytest.raises(ValueError, match="finalized"):
        recorder.canonical_json()

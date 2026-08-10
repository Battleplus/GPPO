from __future__ import annotations

from evaluate_paper_faithful_formal import discover_checkpoints, evaluation_directory


def test_discovery_prefers_phase1_frozen_without_deleting_legacy(tmp_path) -> None:
    first = tmp_path / "seed1"
    second = tmp_path / "seed2"
    first.mkdir()
    second.mkdir()
    (first / "checkpoint.pt").write_bytes(b"legacy")
    (first / "checkpoint_phase1_frozen.pt").write_bytes(b"selected")
    (second / "checkpoint.pt").write_bytes(b"ordinary")
    assert discover_checkpoints(tmp_path) == [
        first / "checkpoint_phase1_frozen.pt",
        second / "checkpoint.pt",
    ]
    assert evaluation_directory(first / "checkpoint_phase1_frozen.pt") == first / "evaluations_phase1_frozen"
    assert evaluation_directory(second / "checkpoint.pt") == second / "evaluations"


def test_discovery_accepts_frozen_only_migration(tmp_path) -> None:
    run = tmp_path / "T5-10-48_literal_event" / "literal_event_seed1"
    run.mkdir(parents=True)
    frozen = run / "checkpoint_phase1_frozen.pt"
    frozen.write_bytes(b"selected")

    assert discover_checkpoints(tmp_path) == [frozen]

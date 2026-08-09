from __future__ import annotations

from evaluate_paper_faithful_formal import discover_checkpoints


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

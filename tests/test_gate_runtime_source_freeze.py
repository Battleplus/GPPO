from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_lazy_gate_runtime_training_sources_remain_frozen() -> None:
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (repo / "configs/GATE_SCREENING_RUNTIME_COHORTS.json").read_text(encoding="utf-8")
    )
    current_cohort = protocol["cohorts"][-1]
    mismatches = {
        relative: {
            "expected": expected,
            "actual": hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
        }
        for relative, expected in current_cohort["files_sha256"].items()
        if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected
    }
    assert not mismatches, f"Gate runtime source freeze violated: {mismatches}"


def test_runtime_cohorts_cover_each_run_once() -> None:
    repo = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (repo / "configs/GATE_SCREENING_RUNTIME_COHORTS.json").read_text(encoding="utf-8")
    )
    runs = [run for cohort in protocol["cohorts"] for run in cohort["runs"]]
    assert len(runs) == 21
    assert len(set(runs)) == 21

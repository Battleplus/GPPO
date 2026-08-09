from __future__ import annotations

from finalize_phase1_gate_screening import clear_opposite_sign, mean_ci95, slug


def test_gate_screening_slug_is_runner_compatible() -> None:
    assert slug("Adaptive-current") == "adaptive_current"
    assert slug("NoGate") == "nogate"
    assert slug("SingleHead") == "singlehead"


def test_paired_seed_interval_and_clear_reversal() -> None:
    negative = mean_ci95([-2.0, -2.0, -2.0])
    positive = mean_ci95([2.0, 2.0, 2.0])
    uncertain = mean_ci95([-2.0, 0.0, 2.0])
    assert negative["ci95"][1] < 0
    assert clear_opposite_sign(negative, positive)
    assert not clear_opposite_sign(negative, uncertain)

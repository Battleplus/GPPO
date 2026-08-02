from __future__ import annotations

import math

import pytest

from analyze_pcrl_attainability import (
    minimum_decay_for_target,
    priority_share_upper_bound,
)


def test_decay_one_full_coverage_cannot_reach_search_target() -> None:
    bound = priority_share_upper_bound(
        decay=1.0, other_type_coverage_floor=1.0
    )
    assert bound == pytest.approx(1.0 / (1.0 + 3.0 * math.exp(-1.0)))
    assert bound < 0.70


def test_decay_two_makes_search_target_attainable_under_full_coverage() -> None:
    bound = priority_share_upper_bound(
        decay=2.0, other_type_coverage_floor=1.0
    )
    assert bound > 0.70


def test_minimum_decay_inverts_upper_bound() -> None:
    decay = minimum_decay_for_target(
        target_share=0.70, other_type_coverage_floor=1.0
    )
    assert decay == pytest.approx(math.log(7.0))
    assert priority_share_upper_bound(
        decay=decay, other_type_coverage_floor=1.0
    ) == pytest.approx(0.70)

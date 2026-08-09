from __future__ import annotations

from reselect_phase1_formal_candidates import expected_candidate_iterations


def test_formal_reselection_requires_complete_50_iteration_grid() -> None:
    iterations = expected_candidate_iterations()
    assert len(iterations) == 40
    assert iterations[0] == 50
    assert iterations[-1] == 2000
    assert all(right - left == 50 for left, right in zip(iterations, iterations[1:]))

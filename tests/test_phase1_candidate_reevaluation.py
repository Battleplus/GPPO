import pytest

from reevaluate_phase1_candidates import paired_difference, rank_order, spearman_three


def evaluation(values):
    return {
        "rows": [
            {"instance_seed": index, "realized_makespan": value}
            for index, value in enumerate(values)
        ]
    }


def test_paired_difference_uses_matching_instance_seed() -> None:
    result = paired_difference(evaluation([1.0, 3.0, 5.0]), evaluation([2.0, 2.0, 5.0]))
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(0.0)
    assert result["median"] == pytest.approx(0.0)


def test_paired_difference_rejects_different_banks() -> None:
    left = evaluation([1.0, 2.0])
    right = evaluation([1.0])
    with pytest.raises(ValueError, match="banks differ"):
        paired_difference(left, right)


def test_ranking_consistency_for_three_models() -> None:
    ascending = {"Adaptive": 1.0, "NoGate": 2.0, "SingleHead": 3.0}
    reversed_order = {"Adaptive": 3.0, "NoGate": 2.0, "SingleHead": 1.0}
    assert rank_order(ascending) == ["Adaptive", "NoGate", "SingleHead"]
    assert spearman_three(ascending, ascending) == pytest.approx(1.0)
    assert spearman_three(ascending, reversed_order) == pytest.approx(-1.0)

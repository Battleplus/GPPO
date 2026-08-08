from __future__ import annotations

from report_paper_faithful_quick import paired_difference


def test_paired_difference_uses_shared_instance_ids() -> None:
    left = {"rows": [{"instance_seed": 2, "realized_makespan": 8.0}, {"instance_seed": 1, "realized_makespan": 10.0}]}
    right = {"rows": [{"instance_seed": 1, "realized_makespan": 11.0}, {"instance_seed": 2, "realized_makespan": 10.0}]}
    result = paired_difference(left, right, "realized_makespan")
    assert result["mean"] == -1.5
    assert result["median"] == -1.5
    assert result["n_instances"] == 2

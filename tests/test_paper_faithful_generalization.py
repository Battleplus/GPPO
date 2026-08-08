from __future__ import annotations

from evaluate_paper_faithful_unknown import UNSEEN_SCALES


def test_unknown_generalization_protocol_has_eight_fixed_scales() -> None:
    assert len(UNSEEN_SCALES) == 8
    assert len(set(UNSEEN_SCALES)) == 8
    for scale in UNSEEN_SCALES:
        uavs, parents, subtasks = (int(part) for part in scale.removeprefix("T").split("-"))
        assert 0 < uavs <= 20
        assert 0 < parents <= subtasks <= 92

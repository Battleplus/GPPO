from __future__ import annotations

from audit_phase1_config import EXPECTED_SCALES, EXPECTED_SEEDS, bank_audit


def test_phase1_instance_banks_are_unique_and_pairwise_disjoint() -> None:
    audit = bank_audit()
    assert audit["valid"] is True
    assert list(audit["scales"]) == EXPECTED_SCALES


def test_formal_constants_cover_exactly_five_training_seeds() -> None:
    assert EXPECTED_SEEDS == [1, 2, 3, 4, 5]

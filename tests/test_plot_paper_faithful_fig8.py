from __future__ import annotations

from plot_paper_faithful_fig8 import normalize_run_label


def test_normalize_run_label_collapses_training_seeds() -> None:
    assert normalize_run_label("literal_event_seed1") == "literal_event"
    assert normalize_run_label("literal_event_seed12") == "literal_event"
    assert normalize_run_label("literal_event") == "literal_event"

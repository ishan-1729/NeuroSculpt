import pytest

from neurosculpt.metrics import absolute_gain_pp, gap_closed


def test_gap_closed_is_unclipped_and_flags_small_denominators():
    assert gap_closed(0.80, 0.60, 0.90)[0] == pytest.approx(2 / 3)
    assert gap_closed(0.95, 0.60, 0.90) == (pytest.approx(35 / 30), False)
    r, flagged = gap_closed(0.62, 0.60, 0.63)
    assert r == pytest.approx(2 / 3) and flagged
    assert gap_closed(0.70, 0.60, 0.60) == (0.0, True)


def test_absolute_gain_pp():
    assert absolute_gain_pp(0.91, 0.90) == pytest.approx(1.0)


@pytest.mark.skip(reason="P1: objectives.select_candidate not implemented")
def test_selection_returns_no_improving_edit_when_nothing_beats_zero():
    """If no gated candidate beats Delta = 0 by the margin on selection, keep theta_0 (NO_IMPROVING_EDIT)."""

import pytest

from neurosculpt.metrics import edit_gain_pp, preservation_drop_pp, repair_fraction


def test_pp_metrics():
    assert edit_gain_pp(0.91, 0.90) == pytest.approx(1.0)
    assert preservation_drop_pp(0.98, 0.975) == pytest.approx(0.5)


def test_repair_fraction():
    assert repair_fraction(2.0, 1.0, 0.0) == pytest.approx(0.5)
    assert repair_fraction(1.0, 0.5, 1.0) is None


@pytest.mark.skip(reason="P2: objectives.choose_safe_candidate not implemented")
def test_selection_returns_no_safe_edit_when_gate_fails():
    """If no candidate passes the preservation gate on selection, keep theta_0 and flag NO_SAFE_EDIT."""

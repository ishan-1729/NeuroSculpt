import pytest

from neurosculpt.experiment import QueryCounter


def test_counter_charges_fbe_and_enforces_cap():
    qc = QueryCounter(cap=64)
    qc.charge()
    qc.charge(63)
    assert qc.fbe == 64
    with pytest.raises(RuntimeError):
        qc.charge()


def test_selection_ledger_is_independent_of_search_budget():
    search, selection = QueryCounter(cap=0), QueryCounter(cap=3)
    selection.charge(3)
    assert (search.fbe, selection.fbe) == (0, 3)

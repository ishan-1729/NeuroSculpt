import pytest

from neurosculpt.experiment import QueryCounter


def test_counter_accumulates_and_enforces_cap():
    qc = QueryCounter(cap=100)
    qc.charge(60)
    qc.charge(40)
    assert qc.forwards == 100
    with pytest.raises(RuntimeError):
        qc.charge(1)

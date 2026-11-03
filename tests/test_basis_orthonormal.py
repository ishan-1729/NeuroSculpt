import pytest
import torch

from neurosculpt.edits import EditSpec


def test_fixture_basis_validates(spec):
    spec.validate()


def test_validate_rejects_entries_outside_support(spec):
    bad = spec.basis.clone()
    bad[0, 0, 0] = 1.0  # column 0 is not in the support
    with pytest.raises(ValueError):
        EditSpec(spec.parameter_name, spec.support_columns, bad, spec.radius).validate()


@pytest.mark.skip(reason="P2/P3: directions.construct_basis not implemented")
def test_constructed_bases_are_orthonormal_and_supported():
    """Every method (random, spectral, behavior) must pass EditSpec.validate()."""

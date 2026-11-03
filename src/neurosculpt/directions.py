"""Edit bases: random orthonormal B0, spectral B1, behaviour-informed B2 (spec §6)."""

from __future__ import annotations


def construct_basis(model, calibration_pairs, support, method: str, d: int):
    """Return a [d, m, n] Frobenius-orthonormal basis that is zero outside ``support``."""
    raise NotImplementedError("P2/P3")

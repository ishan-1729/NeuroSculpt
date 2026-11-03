"""Editable-support selection: random S0, activation-contrast S1, causal restoration S2 (spec §5)."""

from __future__ import annotations


def localize_support(model, calibration_pairs, method: str, k: int, budget):
    """Return (support_columns, diagnostics); every forward used is charged to ``budget``."""
    raise NotImplementedError("P3")

"""Activation-contrast (S1) and causal-restoration (S2) localization for the labeled CMA-S1/S2 arms (spec §6.2)."""

from __future__ import annotations


def localize_support(model, calibration_pairs, method: str, k: int, budget):
    """Return (support_columns, diagnostics); every forward used is charged to ``budget``."""
    raise NotImplementedError("P3")

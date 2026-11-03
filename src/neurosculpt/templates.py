"""Disclosed template library, TEMPLATE-SEARCH and the deterministic template extractor (spec §6.2)."""

from __future__ import annotations

# Index-only operators on W2, applied by the harness; they match the disclosed defect families.
LIBRARY = ("column_set_permutation", "rowset_x_colset_sign_flip", "column_set_zero")


def template_search(api, budget_fbe: int, seed: int):
    """Try LIBRARY instances in a preregistered random order (1 FBE each), compose improving ones
    greedily, then spend the remaining budget on CMA-FULL refinement."""
    raise NotImplementedError("P1")


def eligible_enum(api, eligible_sets, budget_fbe: int, seed: int):
    """Grey-box ELIGIBLE-ENUM: try the balance-matched F4 sets through the meter in random order."""
    raise NotImplementedError("P1")


def extract_template(patch: dict, residual_threshold: float):
    """Least-squares fit of a patch against LIBRARY; returns the template or ``"dense"``."""
    raise NotImplementedError("P0b")

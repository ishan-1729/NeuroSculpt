"""Gradient-free optimizers: random search O0, coordinate search O1, CMA-ES O2 (spec §7.2)."""

from __future__ import annotations


def optimize_edit(candidate_fn, algorithm: str, query_cap: int, seed: int):
    """Minimize ``candidate_fn(z)`` within ``query_cap`` calls; return the full candidate history."""
    raise NotImplementedError("P2")

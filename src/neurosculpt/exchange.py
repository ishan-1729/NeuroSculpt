"""Exchange rate Q*(b) between white-box access and black-box queries (spec §4.5)."""

from __future__ import annotations


def q_star(llm_gain: float, search_budgets, search_gains, censor_at: int = 4096) -> float:
    """Per-cell Q*: smallest q where the envelope arm's restart-mean sealed R, PAVA-isotonized on
    the log2 grid and interpolated linearly in log2 q, reaches ``llm_gain``; censored at 4096."""
    raise NotImplementedError("P3")


def q_star_km_median(cells, budget: int, n_boot: int, seed: int):
    """Pooled Kaplan-Meier median over cells with a seed-cluster bootstrap CI and H2 p-value."""
    raise NotImplementedError("P3")

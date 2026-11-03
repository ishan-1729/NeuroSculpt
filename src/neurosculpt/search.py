"""Black-box and grey-box search arms: CMA-FULL/SUB/NORM/PCA/S1/S2, RANDOM, LOGIT-BIAS, hybrids (spec §6.2)."""

from __future__ import annotations


def cma_popsize(q: int) -> int | None:
    """max(4, floor(q / 4)) for q >= 16; None means the budget runs as RANDOM."""
    return max(4, q // 4) if q >= 16 else None


def run_search(arm: str, api, budget_fbe: int, seed: int):
    """Run one search arm through the meter; return its top-3 candidates and full history."""
    raise NotImplementedError("P1")

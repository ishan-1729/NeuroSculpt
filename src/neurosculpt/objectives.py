"""Target/preservation losses, KL, the scalar objective J(z) and safe selection (spec §4.3)."""

from __future__ import annotations


def objective(metrics: dict, lambda_ce: float, lambda_kl: float, mu: float) -> float:
    raise NotImplementedError("P1")


def choose_safe_candidate(history, selection_batches, preserve_gate):
    """Lowest target CE among gate-passing candidates on selection; else ``NO_SAFE_EDIT``."""
    raise NotImplementedError("P2")

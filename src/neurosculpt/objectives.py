"""Search objective J and candidate selection with the gate (spec §4.2, §4.6)."""

from __future__ import annotations


def objective(metrics: dict, lambda_kl: float) -> float:
    """J = target CE + lambda * preservation KL, from one ``evaluate`` result."""
    return metrics["target_ce"] + lambda_kl * metrics["preserve_kl"]


def select_candidate(candidates, selection_ledger, gate, improve_margin_pp: float):
    """Highest raw selection accuracy on in-pool affected classes among gate-passing candidates.

    Ties break on smallest ||Delta||_F. Returns ``"NO_IMPROVING_EDIT"`` (keep theta_0) if none
    beats Delta = 0 by ``improve_margin_pp``. Never uses theta_clean.
    """
    raise NotImplementedError("P1")

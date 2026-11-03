"""Run orchestration with fixed budgets, cost ledgers and full provenance (spec §4.6, §8, §9.3, §12)."""

from __future__ import annotations


class QueryCounter:
    """Ledger in FBE: one ``evaluate`` call (or one patch in ``evaluate_batch``) costs 1 FBE.

    The search budget b and the selection ledger are separate instances; sandbox CPU time for
    synthetic compute is tracked elsewhere and is not charged here.
    """

    def __init__(self, cap: int | None = None):
        self.cap = cap
        self.fbe = 0

    def charge(self, n_calls: int = 1) -> None:
        if self.cap is not None and self.fbe + n_calls > self.cap:
            raise RuntimeError(f"query budget exceeded: {self.fbe + n_calls} > {self.cap} FBE")
        self.fbe += n_calls


def dry_run_cost(config: dict) -> dict:
    raise NotImplementedError("P0a")


def run_arm(config: dict, arm: str, cell: dict, budget_fbe: int, sample: int):
    raise NotImplementedError("P1")

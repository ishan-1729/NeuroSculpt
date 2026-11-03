"""Run orchestration with fixed budgets, query counting and full provenance (spec §11.3, §12)."""

from __future__ import annotations


class QueryCounter:
    """Counts model-example forwards, the authoritative cost unit."""

    def __init__(self, cap: int | None = None):
        self.cap = cap
        self.forwards = 0

    def charge(self, n_examples: int) -> None:
        if self.cap is not None and self.forwards + n_examples > self.cap:
            raise RuntimeError(f"query budget exceeded: {self.forwards + n_examples} > {self.cap}")
        self.forwards += n_examples


def dry_run_cost(config: dict) -> dict:
    raise NotImplementedError("P2")


def run_method(config: dict, method: dict, seed: int):
    raise NotImplementedError("P2")

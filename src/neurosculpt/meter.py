"""Metered evaluate(patch) API: the only data channel for every arm (spec §4.2)."""

from __future__ import annotations

PATCH_LANGUAGE = ("layers.1.weight", "layers.1.bias", "layers.2.weight", "layers.2.bias")


class MeteredAPI:
    """Scores a patch on the fixed adaptation batch; each call costs 1 FBE and is ledgered."""

    def __init__(self, model, damaged_params, adaptation_batch, budget_fbe: int):
        raise NotImplementedError("P0")

    def evaluate(self, patch: dict) -> dict:
        """Return target_ce, target_acc, per_class_acc and preserve_kl; reject out-of-language patches."""
        raise NotImplementedError("P0")

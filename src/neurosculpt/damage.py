"""Planted-damage generators for Task A with exact repair Delta* = W2_clean - W2_damaged (spec §5.1).

Sites are sampled uniformly without regard to contiguity; F4 permutes only balance-matched units.
Ground truth returned here is for the evaluation harness only and must never reach an arm.
"""

from __future__ import annotations

FAMILIES = (
    "F1_sign_flip_rowset_x_colset",
    "F2_column_noise_norm_matched",
    "F3_fullrank_8x8_rowset_x_colset",
    "F4_stealth_balance_matched",
)


def plant_damage(params, family: str, severity: float, seed: int):
    """Return (damaged_params, ground_truth): D, pi (F4), Delta*, affected and held-out classes."""
    raise NotImplementedError("P0")

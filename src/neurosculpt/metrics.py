"""Performance, structural and uncertainty metrics (spec §10)."""

from __future__ import annotations


def edit_gain_pp(acc_edit: float, acc_base: float) -> float:
    """G_T = 100 (A_T(edit) - A_T(base)), with accuracies as fractions."""
    return 100.0 * (acc_edit - acc_base)


def preservation_drop_pp(acc_base: float, acc_edit: float) -> float:
    """D_P = 100 (A_P(base) - A_P(edit)), with accuracies as fractions."""
    return 100.0 * (acc_base - acc_edit)


def repair_fraction(loss_damaged: float, loss_repaired: float, loss_pristine: float) -> float | None:
    """R_loss for Task A; None when the denominator is not meaningfully positive."""
    denom = loss_damaged - loss_pristine
    return None if denom <= 1e-12 else (loss_damaged - loss_repaired) / denom


def paired_bootstrap(correct_a, correct_b, n_boot: int, seed: int):
    raise NotImplementedError("P4")

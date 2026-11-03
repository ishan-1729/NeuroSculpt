"""Recoverable gap R, canonicalized repair metrics and uncertainty (spec §4.1, §7.1)."""

from __future__ import annotations

DENOMINATOR_FLOOR_PP = 5.0


def gap_closed(acc_edit: float, acc_damaged: float, acc_reference: float) -> tuple[float, bool]:
    """R = (A_edit - A_damaged) / (A_reference - A_damaged) on in-pool affected classes, unclipped.

    ``acc_reference`` is the clean model (Task A) or the ADAM-PATCH ceiling (Task B). Returns
    (R, below_floor); cells below the 5 pp floor stay in the primary analysis but are flagged.
    A zero denominator gives R = 0 and is always flagged.
    """
    denom = acc_reference - acc_damaged
    below_floor = 100.0 * denom < DENOMINATOR_FLOOR_PP
    if denom == 0:
        return 0.0, True
    return (acc_edit - acc_damaged) / denom, below_floor


def absolute_gain_pp(acc_edit: float, acc_damaged: float) -> float:
    """Absolute gain in percentage points, reported alongside R."""
    return 100.0 * (acc_edit - acc_damaged)


def canonical_repair_metrics(theta_edit, theta_clean, delta_star, calibration_inputs):
    """Canonical cosine and distance after h2-orbit matching and flat-direction projection."""
    raise NotImplementedError("P4")


def paired_bootstrap(correct_a, correct_b, n_boot: int, seed: int):
    raise NotImplementedError("P3")

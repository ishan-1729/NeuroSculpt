"""Exact function-preserving reparameterizations of an MLP hidden layer (spec §4.3).

Once its permutation or scale vector is fixed, each transform is linear in the parameters and
preserves the network function for *any* parameters, so a patch proposed in transformed
coordinates maps back exactly: f(T(theta) + delta') == f(theta + T.invert(delta')).
Build each transform once from theta_0, store it with the session, and reuse that object to
invert patches; never recompute ``canonicalize`` from edited weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

MIN_ROW_NORM = 1e-8


def _keys(layer: int) -> tuple[str, str, str]:
    return f"layers.{layer}.weight", f"layers.{layer}.bias", f"layers.{layer + 1}.weight"


@dataclass(frozen=True)
class Permutation:
    """New hidden unit i is old unit perm[i] of the units produced by ``layers.{layer}``."""

    layer: int
    perm: Tensor

    def apply(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        return _permute(params, self.layer, self.perm)

    def invert(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        return _permute(params, self.layer, self.perm.argsort())


@dataclass(frozen=True)
class Rescale:
    """Scale incoming weights and bias of each hidden unit by c > 0 and outgoing weights by 1/c."""

    layer: int
    c: Tensor

    def __post_init__(self):
        if not bool((torch.isfinite(self.c) & (self.c > 0)).all()):
            raise ValueError("rescaling factors must be finite and positive")

    def apply(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        return _scale(params, self.layer, self.c)

    def invert(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        return _scale(params, self.layer, 1.0 / self.c)


def canonicalize(params: dict[str, Tensor], layer: int) -> Rescale:
    """Rescale giving every incoming row of ``layers.{layer}``, bias included, unit norm.

    Rows with norm below MIN_ROW_NORM are left unscaled. Because the result does not depend on
    any earlier positive rescaling, this is a canonical form, not a randomized presentation.
    """
    w_in, b_in, _ = _keys(layer)
    norms = torch.cat([params[w_in], params[b_in][:, None]], dim=1).norm(dim=1)
    c = torch.where(norms < MIN_ROW_NORM, torch.ones_like(norms), 1.0 / norms)
    return Rescale(layer, c)


def _permute(params, layer, idx):
    w_in, b_in, w_out = _keys(layer)
    out = dict(params)
    if w_in in out:
        out[w_in] = out[w_in][idx, :]
    if b_in in out:
        out[b_in] = out[b_in][idx]
    if w_out in out:
        out[w_out] = out[w_out][:, idx]
    return out


def _scale(params, layer, c):
    w_in, b_in, w_out = _keys(layer)
    out = dict(params)
    if w_in in out:
        out[w_in] = out[w_in] * c[:, None]
    if b_in in out:
        out[b_in] = out[b_in] * c
    if w_out in out:
        out[w_out] = out[w_out] / c[None, :]
    return out

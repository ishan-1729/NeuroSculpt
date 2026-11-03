"""Support masks, basis matrices and the functional weight patch (spec §4.2, §7.1, §11.2).

Nothing in this module mutates the base parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call


@dataclass(frozen=True)
class EditSpec:
    parameter_name: str
    support_columns: tuple[int, ...]
    basis: Tensor  # [d, m, n], zero outside support, normalized
    radius: float

    @property
    def d(self) -> int:
        return self.basis.shape[0]

    def support_mask(self) -> Tensor:
        """Boolean [m, n] mask that is True on the supported columns."""
        mask = torch.zeros(self.basis.shape[1:], dtype=torch.bool)
        mask[:, list(self.support_columns)] = True
        return mask

    def validate(self, atol: float = 1e-6) -> None:
        if self.basis[:, ~self.support_mask()].abs().max().item() > 0:
            raise ValueError("basis has nonzero entries outside the support")
        flat = self.basis.flatten(1)
        gram = flat @ flat.T
        if not torch.allclose(gram, torch.eye(self.d, dtype=gram.dtype), atol=atol):
            raise ValueError("basis is not orthonormal in the Frobenius inner product")


def delta_w(spec: EditSpec, z: Tensor) -> Tensor:
    """Delta W(z) = sum_j z_j B_j."""
    return torch.einsum("d,dmn->mn", z.to(spec.basis.dtype), spec.basis)


def make_parameter_patch(
    base_params: dict[str, Tensor], spec: EditSpec, z: Tensor
) -> dict[str, Tensor]:
    """Return a new parameter dict with W0 + Delta W(z); other entries alias the frozen base."""
    patched = dict(base_params)
    patched[spec.parameter_name] = base_params[spec.parameter_name] + delta_w(spec, z)
    return patched


@torch.no_grad()
def forward_with_params(model: nn.Module, params: dict[str, Tensor], x: Tensor) -> Tensor:
    return functional_call(model, params, (x,))


def evaluate_edit(model, base_params, spec, z, batches, objective):
    raise NotImplementedError("P1")

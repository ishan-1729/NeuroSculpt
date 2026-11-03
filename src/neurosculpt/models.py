"""Small model and checkpoint utilities (spec §1.4, §9)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

W2_NAME = "layers.1.weight"
READABLE_SIZES = (49, 24, 16, 10)  # primary target: weights fit in an LLM context as text
SECONDARY_SIZES = (784, 128, 64, 10)  # UNTESTED extension; needs its own preregistration (spec §10)


class MLP(nn.Module):
    """ReLU MLP with no dropout or batch norm; ``layers.1.weight`` is W2."""

    def __init__(self, sizes: tuple[int, ...] = READABLE_SIZES):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))

    def forward(self, x: Tensor) -> Tensor:
        h = x.flatten(1)
        for layer in self.layers[:-1]:
            h = torch.relu(layer(h))
        return self.layers[-1](h)


def train_base_model(train_loader, seed: int, epochs: int, device: str = "cpu") -> MLP:
    raise NotImplementedError("P0")


def save_checkpoint(model: nn.Module, path: Path) -> str:
    """Save the state dict and return its SHA-256 hash."""
    raise NotImplementedError("P0")


def load_checkpoint(path: Path, expected_hash: str | None = None) -> MLP:
    raise NotImplementedError("P0")

"""Dataset indices, paired clean/corrupt views and leakage checks (spec §8)."""

from __future__ import annotations

SPLITS = ("calibration", "edit_search", "edit_selection", "sealed_test")


def download_mnist(root):
    raise NotImplementedError("P0")


def download_mnist_c(root, corruption: str):
    raise NotImplementedError("P0")


def make_splits(n_train: int, sizes: dict[str, int], seed: int) -> dict[str, list[int]]:
    """Disjoint splits over original training image IDs, so paired views never straddle splits."""
    raise NotImplementedError("P0")


def split_hashes(splits: dict[str, list[int]]) -> dict[str, str]:
    raise NotImplementedError("P0")


def assert_no_leakage(splits: dict[str, list[int]]) -> None:
    raise NotImplementedError("P0")

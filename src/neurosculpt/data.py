"""MNIST/MNIST-C loading, 7x7 area-average downsampling, splits and leakage checks (spec §1.4, §5.3)."""

from __future__ import annotations

SPLITS = ("base_train", "calibration", "adaptation", "selection", "sealed_test")
DOWNSAMPLE_KERNEL = 4  # avg_pool2d on [0, 1] 28x28 images -> 7x7


def download_mnist(root):
    raise NotImplementedError("P0")


def download_mnist_c(root, corruption: str, sha256: str):
    """Download the pinned Zenodo zip; clean views come from its ``identity/`` folder."""
    raise NotImplementedError("P0")


def downsample(images):
    raise NotImplementedError("P0")


def make_splits(seed: int, sizes: dict[str, int]) -> dict[str, list[int]]:
    """Disjoint splits by original image ID; ``sealed_test`` is the official MNIST test split."""
    raise NotImplementedError("P0")


def split_hashes(splits: dict[str, list[int]]) -> dict[str, str]:
    raise NotImplementedError("P0")


def assert_no_leakage(splits: dict[str, list[int]]) -> None:
    raise NotImplementedError("P0")

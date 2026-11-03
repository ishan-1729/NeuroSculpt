import pytest


@pytest.mark.skip(reason="P0: data.make_splits not implemented")
def test_splits_are_disjoint_by_original_image_id():
    """No original image ID may appear in more than one split, across clean and corrupted views."""


@pytest.mark.skip(reason="P0: data.split_hashes not implemented")
def test_split_hashes_are_deterministic():
    """Same seed and sizes give identical index hashes."""

import pytest
import torch

from neurosculpt.edits import EditSpec
from neurosculpt.models import MLP, W2_NAME


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MLP().eval()


@pytest.fixture
def spec():
    """Random Frobenius-orthonormal basis on k=8 columns of W2, d=8 (test fixture only)."""
    g = torch.Generator().manual_seed(1)
    support = (3, 17, 40, 41, 64, 90, 101, 127)
    q, _ = torch.linalg.qr(torch.randn(64 * len(support), 8, generator=g))
    basis = torch.zeros(8, 64, 128)
    basis[:, :, list(support)] = q.T.reshape(8, 64, len(support))
    return EditSpec(parameter_name=W2_NAME, support_columns=support, basis=basis, radius=1.0)


@pytest.fixture
def x():
    return torch.randn(32, 1, 28, 28, generator=torch.Generator().manual_seed(2))

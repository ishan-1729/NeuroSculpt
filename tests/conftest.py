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
    """Random Frobenius-orthonormal basis on 6 columns of the readable 16 x 24 W2, d=6 (test fixture only)."""
    g = torch.Generator().manual_seed(1)
    support = (1, 5, 9, 14, 20, 23)
    q, _ = torch.linalg.qr(torch.randn(16 * len(support), 6, generator=g))
    basis = torch.zeros(6, 16, 24)
    basis[:, :, list(support)] = q.T.reshape(6, 16, len(support))
    return EditSpec(parameter_name=W2_NAME, support_columns=support, basis=basis, radius=1.0)


@pytest.fixture
def x():
    return torch.rand(32, 1, 7, 7, generator=torch.Generator().manual_seed(2))

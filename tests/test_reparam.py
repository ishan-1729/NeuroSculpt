import pytest
import torch

from neurosculpt.edits import forward_with_params
from neurosculpt.models import MLP, READABLE_SIZES
from neurosculpt.reparam import Permutation, Rescale, canonicalize

PATCH_KEYS = ["layers.1.weight", "layers.1.bias", "layers.2.weight", "layers.2.bias"]
WIDTHS = {0: 24, 1: 16}


def _model():
    torch.manual_seed(0)
    return MLP(READABLE_SIZES).double().eval()


def _params(model):
    return {k: v.detach() for k, v in model.named_parameters()}


def _x(seed=2):
    return torch.rand(64, 49, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _exact(a, b):
    assert torch.allclose(a, b, rtol=0, atol=1e-10)
    assert torch.equal(a.argmax(1), b.argmax(1))


def _transforms(params, layer):
    g = torch.Generator().manual_seed(10 + layer)
    n = WIDTHS[layer]
    log_c = torch.empty(n, dtype=torch.float64).uniform_(-1.386, 1.386, generator=g)
    return [
        Permutation(layer, torch.randperm(n, generator=g)),
        Rescale(layer, torch.exp(log_c)),
        canonicalize(params, layer),
    ]


@pytest.mark.parametrize("layer", [0, 1])
def test_transforms_preserve_function(layer):
    model, x = _model(), _x()
    base = _params(model)
    expected = forward_with_params(model, base, x)
    for t in _transforms(base, layer):
        _exact(forward_with_params(model, t.apply(base), x), expected)


@pytest.mark.parametrize("layer", [0, 1])
def test_patch_inverse_is_exact(layer):
    model, x = _model(), _x()
    base = _params(model)
    g = torch.Generator().manual_seed(4)
    for t in _transforms(base, layer):
        delta_t = {k: 0.1 * torch.randn(base[k].shape, generator=g, dtype=torch.float64) for k in PATCH_KEYS}
        transformed = t.apply(base)
        lhs = forward_with_params(model, {k: transformed[k] + delta_t.get(k, 0) for k in base}, x)
        delta = t.invert(delta_t)
        rhs = forward_with_params(model, {k: base[k] + delta.get(k, 0) for k in base}, x)
        _exact(lhs, rhs)


def test_canonical_form_ignores_prior_rescaling():
    base = _params(_model())
    c = torch.exp(torch.linspace(-1.0, 1.0, 24, dtype=torch.float64))
    rescaled = Rescale(0, c).apply(base)
    a = canonicalize(base, 0).apply(base)
    b = canonicalize(rescaled, 0).apply(rescaled)
    for k in base:
        assert torch.allclose(a[k], b[k], rtol=0, atol=1e-12), k


def test_canonicalize_leaves_zero_rows_unscaled():
    base = _params(_model())
    base["layers.0.weight"] = base["layers.0.weight"].clone()
    base["layers.0.bias"] = base["layers.0.bias"].clone()
    base["layers.0.weight"][5] = 0.0
    base["layers.0.bias"][5] = 0.0
    t = canonicalize(base, 0)
    assert t.c[5] == 1.0
    assert all(torch.isfinite(v).all() for v in t.apply(base).values())


def test_rescale_rejects_nonfinite_factors():
    with pytest.raises(ValueError):
        Rescale(0, torch.tensor([1.0, float("inf")], dtype=torch.float64))


@pytest.mark.parametrize("cycle", [[3, 11], [3, 11, 17]])
def test_stealth_column_permutation_equals_incoming_row_permutation(cycle):
    """Spec §4.4: W2[:, pi] computes what W1[pi^-1, :], b1[pi^-1] would with the clean W2."""
    model, x = _model(), _x()
    base = _params(model)
    perm = torch.arange(24)
    perm[cycle] = perm[cycle[1:] + cycle[:1]]
    inv = perm.argsort()
    stealth = dict(base, **{"layers.1.weight": base["layers.1.weight"][:, perm]})
    rows = dict(base, **{"layers.0.weight": base["layers.0.weight"][inv, :],
                         "layers.0.bias": base["layers.0.bias"][inv]})
    _exact(forward_with_params(model, stealth, x), forward_with_params(model, rows, x))

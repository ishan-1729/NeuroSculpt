import torch

from neurosculpt.edits import forward_with_params, make_parameter_patch


def test_zero_edit_reproduces_base_output_exactly(model, spec, x):
    base = dict(model.named_parameters())
    patched = make_parameter_patch(base, spec, torch.zeros(spec.d))
    with torch.no_grad():
        expected = model(x)
    assert torch.equal(forward_with_params(model, patched, x), expected)

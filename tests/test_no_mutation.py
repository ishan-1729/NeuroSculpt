import torch

from neurosculpt.edits import forward_with_params, make_parameter_patch


def test_patch_and_forward_leave_base_checkpoint_unchanged(model, spec, x):
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base = dict(model.named_parameters())
    patched = make_parameter_patch(base, spec, torch.ones(spec.d))
    forward_with_params(model, patched, x)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k

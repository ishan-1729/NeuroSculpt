import torch

from neurosculpt.edits import delta_w, make_parameter_patch


def test_edit_touches_only_supported_columns(model, spec):
    base = dict(model.named_parameters())
    patched = make_parameter_patch(base, spec, torch.randn(spec.d))
    changed = patched[spec.parameter_name] != base[spec.parameter_name]
    assert not changed[:, ~spec.support_mask()[0]].any()
    for name, p in base.items():
        if name != spec.parameter_name:
            assert patched[name] is p


def test_orthonormal_basis_maps_z_norm_to_frobenius_norm(spec):
    z = torch.randn(spec.d)
    assert torch.allclose(torch.linalg.norm(delta_w(spec, z)), torch.linalg.norm(z), atol=1e-5)

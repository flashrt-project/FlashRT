"""Fractal-NZ SigLIP MLP weights on a tile-aligned width.

The SigLIP MLP is 4304 wide, which is 16.81 fractal tiles of 256 and tiles
badly on the cube pipeline. ``make_vision_mlp_nz_weights`` rounds it to 17
whole tiles with zero channels and moves both MLP weights into fractal-NZ
layout. These tests pin the two properties the change rests on: the padded
channels stay exactly zero through the activation, and the NZ binding agrees
with the ND projection it replaces.
"""

import pytest


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        import torch
        return bool(torch.npu.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _npu_available(), reason="no usable Ascend NPU on this machine")


def test_padded_channels_are_exactly_zero_through_gelu():
    """gelu(0) is exactly 0, so the padded lane contributes nothing downstream."""
    import torch
    import torch.nn.functional as F
    from flash_rt.npu.core.linear import NzBf16Weight
    from flash_rt.npu.models.pi05.fast import VIS_H_PADDED
    from flash_rt.npu.models.pi05.pipeline import VIS_D, VIS_H, GELU_TANH_APPROX

    extra = VIS_H_PADDED - VIS_H
    weight = torch.randn(VIS_H, VIS_D, device="npu", dtype=torch.bfloat16)
    bias = torch.zeros(VIS_H_PADDED, device="npu", dtype=torch.bfloat16)
    fc1 = NzBf16Weight.bind(weight, pad_out=extra)
    x = torch.randn(64, VIS_D, device="npu", dtype=torch.bfloat16)

    h = F.gelu(fc1(x, bias), approximate=GELU_TANH_APPROX)
    assert h.shape == (64, VIS_H_PADDED)
    assert torch.count_nonzero(h[:, VIS_H:]) == 0


def test_nz_binding_matches_the_nd_projection_it_replaces():
    import torch
    import torch_npu
    from flash_rt.npu.core.linear import NzBf16Weight, linear

    weight = torch.randn(1152, 1280, device="npu", dtype=torch.bfloat16)
    bias = torch.randn(1152, device="npu", dtype=torch.bfloat16)
    x = torch.randn(2, 64, 1280, device="npu", dtype=torch.bfloat16)

    reference = torch_npu.npu_linear(
        x.reshape(-1, 1280), weight, bias.float()).reshape(2, 64, 1152)
    produced = linear(x, NzBf16Weight.bind(weight), bias)

    assert produced.shape == reference.shape
    assert produced.dtype == reference.dtype
    cosine = torch.nn.functional.cosine_similarity(
        reference.float().flatten(), produced.float().flatten(), dim=0)
    assert float(cosine) > 0.9999


def test_binding_rejects_a_non_serving_dtype():
    import torch
    from flash_rt.npu.core.linear import NzBf16Weight

    with pytest.raises(ValueError):
        NzBf16Weight.bind(torch.randn(32, 32, device="npu", dtype=torch.float32))


def test_overlay_covers_every_vision_layer():
    import torch
    from flash_rt.npu.core.linear import NzBf16Weight
    from flash_rt.npu.models.pi05 import fast as npu_fast
    from flash_rt.npu.models.pi05.pipeline import VIS_D, VIS_H, VIS_L, _VP

    wb = {}
    for layer in range(VIS_L):
        prefix = f"{_VP}.encoder.layers.{layer}.mlp"
        wb[f"{prefix}.fc1.weight"] = torch.randn(VIS_H, VIS_D, device="npu", dtype=torch.bfloat16)
        wb[f"{prefix}.fc1.bias"] = torch.randn(VIS_H, device="npu", dtype=torch.float32)
        wb[f"{prefix}.fc2.weight"] = torch.randn(VIS_D, VIS_H, device="npu", dtype=torch.bfloat16)
        wb[f"{prefix}.fc2.bias"] = torch.randn(VIS_D, device="npu", dtype=torch.float32)

    overlay = npu_fast.make_vision_mlp_nz_weights(wb)
    for layer in range(VIS_L):
        prefix = f"{_VP}.encoder.layers.{layer}.mlp"
        assert isinstance(overlay[f"{prefix}.fc1.weight"], NzBf16Weight)
        assert isinstance(overlay[f"{prefix}.fc2.weight"], NzBf16Weight)
        assert overlay[f"{prefix}.fc1.bias"].shape == (npu_fast.VIS_H_PADDED,)
        # Biases move to BF16 so addmm does not promote the activation chain.
        assert overlay[f"{prefix}.fc1.bias"].dtype == torch.bfloat16
        assert overlay[f"{prefix}.fc2.bias"].dtype == torch.bfloat16
        assert torch.count_nonzero(overlay[f"{prefix}.fc1.bias"][VIS_H:]) == 0


def test_padded_width_never_discards_channels():
    from flash_rt.npu.models.pi05 import fast as npu_fast
    from flash_rt.npu.models.pi05.pipeline import VIS_H

    assert npu_fast.VIS_H_PADDED >= VIS_H
    assert npu_fast.VIS_H_PADDED % 256 == 0
    with pytest.raises(ValueError):
        npu_fast.make_vision_mlp_nz_weights({}, padded_hidden=VIS_H - 1)

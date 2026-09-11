"""Fractal-NZ operand selection.

NZ pays on wide-K reductions and loses on narrow ones, so the backend opts in
per site rather than globally. These tests pin which sites opted in and that the
layout change is value-preserving.
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


def _bind(weight, nz):
    import numpy as np
    from flash_rt.npu.core.linear import StaticRowInt8Weight
    rows = 8
    amax = np.full(rows + 1, 4.0, dtype=np.float32)
    return StaticRowInt8Weight.bind(weight, amax, rows, None, nz=nz)


def test_nz_static_row_weight_matches_the_nd_binding():
    import torch
    import torch_npu

    torch.manual_seed(0)
    weight = torch.randn(2048, 16384, device="npu", dtype=torch.bfloat16)
    nd, nz = _bind(weight, False), _bind(weight, True)
    assert torch_npu.get_npu_format(nz.tensor) != torch_npu.get_npu_format(nd.tensor)

    x = torch.randn(9, 16384, device="npu", dtype=torch.bfloat16)
    assert torch.equal(nd(x), nz(x))


def test_the_encoder_opts_in_only_on_the_down_projection():
    """The 2048- and 256-wide projections measure slower in NZ and stay ND."""
    import inspect
    from flash_rt.npu.models.pi05 import quantization

    source = inspect.getsource(quantization.calibrate_encoder)
    assert 'nz=key.endswith(".mlp.down_proj.weight")' in source


def test_vision_attention_overlay_is_value_preserving():
    import torch
    import torch_npu
    from flash_rt.npu.core.linear import NzBf16Weight, linear

    torch.manual_seed(1)
    weight = torch.randn(1280, 1152, device="npu", dtype=torch.bfloat16)
    bias = torch.randn(1280, device="npu", dtype=torch.bfloat16)
    x = torch.randn(2, 128, 1152, device="npu", dtype=torch.bfloat16)

    reference = torch_npu.npu_linear(
        x.reshape(-1, 1152), weight, bias.float()).reshape(2, 128, 1280)
    assert torch.equal(reference, linear(x, NzBf16Weight.bind(weight), bias))

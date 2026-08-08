from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_rt.structures.catalog.patch_projection.reference import (
    patch_projection_ref,
)
from flash_rt.structures.discover import discover, seam_weights
from flash_rt.structures.autobuild import auto_swaps
from flash_rt.structures.impls.patch_projection.bf16_flat import (
    FlatPatchProjection,
    bind_flat_patch_projection,
)
from flash_rt.structures.swap import attach


class FullPatchHost(nn.Module):
    def __init__(self, *, bias=True):
        super().__init__()
        self.temporal_patch_size = 2
        self.patch_size = 16
        self.in_channels = 3
        self.embed_dim = 1152
        kernel = (2, 16, 16)
        self.proj = nn.Conv3d(
            3, 1152, kernel_size=kernel, stride=kernel, bias=bias
        )

    def forward(self, x):
        x = x.view(-1, 3, 2, 16, 16).to(self.proj.weight.dtype)
        return self.proj(x).view(-1, self.embed_dim)


class LinearPatchHost(nn.Module):
    def __init__(self, weight_nk, bias):
        super().__init__()
        self.register_buffer("weight_nk", weight_nk)
        self.register_buffer("bias", bias)

    def forward(self, x):
        return F.linear(x.to(torch.bfloat16), self.weight_nk, self.bias)


class FakeKernel:
    @staticmethod
    def bf16_linear_bf16(x, w, out=None):
        result = F.linear(x, w.t())
        if out is None:
            return result
        out.copy_(result)
        return out

    @staticmethod
    def bf16_linear_bias_bf16(x, w, bias, out=None):
        result = F.linear(x, w.t(), bias)
        if out is None:
            return result
        out.copy_(result)
        return out


def test_reference_is_flat_projection():
    torch.manual_seed(7)
    x = torch.randn(5, 24)
    w = torch.randn(9, 24)
    b = torch.randn(9)
    assert torch.equal(patch_projection_ref(x, w, b), F.linear(x, w, b))


def test_discovers_only_exact_preflattened_full_patch_contract():
    host = nn.Module()
    host.patch_embed = FullPatchHost()
    seams = discover(host, structures=("patch_projection",))
    assert len(seams) == 1
    seam = seams[0]
    assert seam.path == "patch_embed"
    assert seam.dims == {"K": 1536, "N": 1152}
    weights = seam_weights(host, seam)
    assert weights["w"].shape == (1152, 1536)
    assert weights["b"].shape == (1152,)

    original_stride = host.patch_embed.proj.stride
    host.patch_embed.proj.stride = (1, 16, 16)
    assert discover(host, structures=("patch_projection",)) == []
    host.patch_embed.proj.stride = original_stride

    host.patch_embed.proj.padding = (1, 0, 0)
    assert discover(host, structures=("patch_projection",)) == []


def test_generic_conv3d_is_not_a_patch_projection():
    host = nn.Sequential(nn.Conv3d(3, 1152, kernel_size=(2, 16, 16),
                                   stride=(2, 16, 16)))
    assert discover(host, structures=("patch_projection",)) == []


def test_calibration_refuses_an_ordinary_volume_input():
    host = nn.Module()
    host.patch_embed = FullPatchHost()
    host.eval()
    volume = torch.randn(1, 3, 2, 16, 16)
    plan = auto_swaps(
        host,
        lambda: host.patch_embed(volume),
        structures=("patch_projection",),
        scheme="none",
    )
    assert not plan.swaps
    assert any(
        "not preflattened full-patch rows" in reason
        for _, reason in plan.notes["refused"]
    )


def test_guard_fallback_and_attach_detach_are_reversible():
    torch.manual_seed(11)
    weight_nk = torch.randn(4, 8, dtype=torch.bfloat16)
    bias = torch.randn(4, dtype=torch.bfloat16)
    original = LinearPatchHost(weight_nk, bias)
    bound = FlatPatchProjection(
        weight_nk.t().contiguous(),
        bias,
        row_capacity=3,
        host_dtypes=(torch.float32,),
        original=original,
        kernel=FakeKernel(),
    )
    model = nn.Module()
    model.patch = original
    model.eval()
    handle = attach(model, {"patch": bound})

    x = torch.randn(3, 8)
    expected = original(x)
    assert torch.equal(model.patch(x), expected)
    assert handle.summary()["fallbacks"] == 0

    # An unseen larger patch count cannot use the fixed capture buffer and
    # must run the retained host, visibly in the ledger.
    larger = torch.randn(4, 8)
    assert torch.equal(model.patch(larger), original(larger))
    assert handle.summary()["fallbacks"] == 1

    attached_state = model.state_dict()
    handle.detach()
    assert model.patch is original
    assert torch.equal(model.patch(x), expected)
    assert attached_state.keys() == model.state_dict().keys()


def test_missing_hub_entry_refuses_cleanly():
    weight_nk = torch.randn(8, 4, dtype=torch.bfloat16)
    bias = torch.randn(4, dtype=torch.bfloat16)
    original = LinearPatchHost(weight_nk.t().contiguous(), bias)
    try:
        FlatPatchProjection(
            weight_nk,
            bias,
            row_capacity=2,
            host_dtypes=(torch.bfloat16,),
            original=original,
            kernel=object(),
        )
    except ValueError as exc:
        assert "lacks bf16_linear_bias_bf16" in str(exc)
    else:
        raise AssertionError("missing Hub entry was accepted")


def test_cuda_binder_refuses_cpu_weights_before_loading_hub():
    original = LinearPatchHost(
        torch.randn(4, 8, dtype=torch.bfloat16),
        torch.randn(4, dtype=torch.bfloat16),
    )
    try:
        bind_flat_patch_projection(
            {"w": original.weight_nk, "b": original.bias},
            row_profile=(3,),
            host_dtypes=(torch.bfloat16,),
            original=original,
        )
    except ValueError as exc:
        assert "CUDA BF16" in str(exc)
    else:
        raise AssertionError("CPU weights were accepted")

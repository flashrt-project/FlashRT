from types import SimpleNamespace

import torch
import torch.nn.functional as F

from flash_rt.structures.impls.norm_fused import bf16


def test_norm_fused_uses_native_residual_norm_package(monkeypatch):
    calls = []

    def layer_norm_bf16(x, weight, bias, eps):
        calls.append((x.dtype, float(eps)))
        return F.layer_norm(
            x.float(), (x.shape[-1],), weight.float(), bias.float(), eps
        ).to(torch.bfloat16)

    def load(repo, version):
        assert repo == "flashrt/flashrt-residual-norm-quant"
        assert version == ">=1"
        return SimpleNamespace(layer_norm_bf16=layer_norm_bf16)

    monkeypatch.setattr(bf16, "hub_kernel", load)
    module = bf16.FusedNorm(torch.nn.LayerNorm(16, eps=1e-5))
    x = torch.randn(3, 16)
    out = module(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert calls == [(torch.bfloat16, 1e-5)]


def test_norm_fused_flattens_host_rank_for_the_2d_kernel(monkeypatch):
    # the native entry's contract is 2D [rows, width]; hosts hand the
    # norm [B, S, D] — the module must flatten and restore, or the
    # failure only shows on the first real host forward
    seen = []

    def layer_norm_bf16(x, weight, bias, eps):
        assert x.dim() == 2, "kernel contract is 2D"
        seen.append(tuple(x.shape))
        return F.layer_norm(
            x.float(), (x.shape[-1],), weight.float(), bias.float(), eps
        ).to(torch.bfloat16)

    monkeypatch.setattr(
        bf16, "hub_kernel",
        lambda repo, version: SimpleNamespace(
            layer_norm_bf16=layer_norm_bf16))
    module = bf16.FusedNorm(torch.nn.LayerNorm(16, eps=1e-6))
    x = torch.randn(2, 5, 16)
    out = module(x)

    assert out.shape == x.shape
    assert seen == [(10, 16)]
    ref = F.layer_norm(x.float(), (16,), module.host_norm.weight.float(),
                       module.host_norm.bias.float(), 1e-6)
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=2e-2)

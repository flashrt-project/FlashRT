"""The single-stream packed QK-norm/RoPE composition, on live kernels.

A host in the plain diffusers processor form — one QKV group, per-head
RMSNorm, partial rotary over the leading channels, dispatched
self-attention — composed through the packed seats and the per-head
kernel. The partial rotary is absorbed by the channel permutation, so
the checks that matter are: the routed output tracks the eager host,
the norm/rope stage tracks the torch reference on identical packed
input, same-shape layers share one pooled workspace lane, and revert
restores the host bit-exact (the permutation is undone row-for-row).
"""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="needs CUDA")

HEADS, HD, R = 2, 128, 96
DIM = HEADS * HD


def dispatch_attention_fn(query, key, value, attn_mask=None,
                          dropout_p=0.0, is_causal=False, backend=None,
                          parallel_config=None):
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2),
        value.transpose(1, 2), attn_mask=attn_mask,
        is_causal=is_causal)
    return out.transpose(1, 2)


def _rope(x, cos, sin):
    rot, keep = x[..., :R], x[..., R:]
    cos = cos.to(x.dtype)[None, :, None, :]
    sin = sin.to(x.dtype)[None, :, None, :]
    x1, x2 = rot.chunk(2, dim=-1)
    rot = rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat((rot, keep), dim=-1)


class _Proc:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb=None,
                 attention_mask=None):
        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        q = attn.norm_q(q)
        k = attn.norm_k(k)
        if rotary_emb is not None:
            q = _rope(q, *rotary_emb)
            k = _rope(k, *rotary_emb)
        h = dispatch_attention_fn(q, k, v, attn_mask=attention_mask)
        h = h.flatten(2, 3).type_as(q)
        for layer in attn.to_out:
            h = layer(h)
        return h


class StreamAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = HEADS
        self.head_dim = HD
        self.to_q = nn.Linear(DIM, DIM, bias=False)
        self.to_k = nn.Linear(DIM, DIM, bias=False)
        self.to_v = nn.Linear(DIM, DIM, bias=False)
        self.norm_q = nn.RMSNorm(HD, eps=1e-5)
        self.norm_k = nn.RMSNorm(HD, eps=1e-5)
        self.to_out = nn.ModuleList(
            [nn.Linear(DIM, DIM, bias=False), nn.Dropout(0.0)])
        self.processor = _Proc()

    def forward(self, hidden_states, rotary_emb=None,
                attention_mask=None):
        return self.processor(self, hidden_states, rotary_emb,
                              attention_mask)


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn0 = StreamAttention()
        self.attn1 = StreamAttention()

    def forward(self, x, rope):
        return self.attn1(self.attn0(x, rope), rope)


def _tables(seq, device):
    freq = torch.arange(R // 2, device=device) / (R // 2)
    ang = torch.arange(seq, device=device)[:, None] * (
        0.01 * (1.0 + freq)[None, :])
    ang = torch.cat((ang, ang), dim=-1)
    return ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)


def _build(rows=32):
    from flash_rt.structures.impls.qkv_pack import fp8_static
    torch.manual_seed(0)
    host = Host().eval().cuda().to(torch.bfloat16)
    plan = types.SimpleNamespace(swaps={})
    for name in ("attn0", "attn1"):
        attn = getattr(host, name)
        with torch.no_grad():
            parts = fp8_static.bind_qkv_pack(
                (attn.to_q, attn.to_k, attn.to_v),
                torch.ones(1, device="cuda"),
                rows=rows, in_dtype="bf16_fused_quant")
        for attr, part in zip(("to_q", "to_k", "to_v"), parts):
            plan.swaps[f"{name}.{attr}"] = part
    return host, plan


@cuda
def test_routed_stream_tracks_the_host():
    pytest.importorskip("safetensors")
    from flash_rt.structures import workspace
    from flash_rt.structures.adapters.packed_stream_qk_norm_rope import (
        PackedStreamQkNormRopeAdapter)
    from flash_rt.structures.swap import attach

    workspace.clear()
    host, plan = _build()
    seq = 16
    x = torch.randn(1, seq, DIM, device="cuda", dtype=torch.bfloat16)
    rope = _tables(seq, "cuda")
    with torch.inference_mode():
        ref = host(x, rope).float()

    result = PackedStreamQkNormRopeAdapter()(host, plan)
    assert result is not None and not result.get("refused"), result
    assert len(result["observed"]) == 2

    handle = attach(host, plan.swaps, observe=result["observed"],
                    revert=result["revert"])
    try:
        with torch.inference_mode():
            got = host(x, rope).float()
        cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
        assert float(cos) > 0.99, float(cos)

        # same-shape layers share the pooled lane
        b0 = result["observed"]["attn0::per_head_qk_norm_rope"]
        b1 = result["observed"]["attn1::per_head_qk_norm_rope"]
        assert b0.q_out.data_ptr() == b1.q_out.data_ptr()
        assert b0.v_out.data_ptr() == b1.v_out.data_ptr()
    finally:
        handle.detach()
    with torch.inference_mode():
        back = host(x, rope).float()
    assert torch.equal(back, ref)


@cuda
def test_norm_rope_stage_matches_torch_on_identical_input():
    from flash_rt.structures import workspace
    from flash_rt.structures.adapters.packed_stream_qk_norm_rope import (
        PackedStreamQkNormRopeAdapter)
    from flash_rt.structures.swap import attach

    workspace.clear()
    host, plan = _build()
    seq = 16
    rope = _tables(seq, "cuda")
    result = PackedStreamQkNormRopeAdapter()(host, plan)
    handle = attach(host, plan.swaps, observe=result["observed"],
                    revert=result["revert"])
    try:
        x = torch.randn(1, seq, DIM, device="cuda",
                        dtype=torch.bfloat16)
        with torch.inference_mode():
            host(x, rope)   # first call applies the lazy permutation

            attn = host.attn0
            pack = plan.swaps["attn0.to_q"]
            packed = pack.joint(x).reshape(1, seq, -1)
            bound = result["observed"]["attn0::per_head_qk_norm_rope"]
            from flash_rt.structures.adapters import (
                packed_stream_qk_norm_rope as mod)
            # tables through the adapter's own remap: rebuild inline
            cos, sin = rope
            half = R // 2
            c = cos.new_ones(seq, 128)
            s_ = sin.new_zeros(seq, 128)
            c[:, :half] = cos[:, :half]
            c[:, 64:64 + half] = cos[:, half:R]
            s_[:, :half] = sin[:, :half]
            s_[:, 64:64 + half] = sin[:, half:R]
            q, k, v = bound(packed.contiguous(),
                            c.unsqueeze(0).contiguous(),
                            s_.unsqueeze(0).contiguous())

            # torch reference: the pack's rows are already permuted, so
            # un-permute back to host channel order, apply the host
            # math, then re-permute with the adapter's channel map
            qp, kp, vp = packed[0].split([DIM, DIM, DIM], dim=-1)
            perm = torch.empty(128, dtype=torch.long)
            slot = {}
            for i in range(half):
                slot[i] = i
                slot[64 + i] = half + i
            spare = [s2 for s2 in range(128) if s2 not in slot]
            for s2, ch in zip(spare, range(R, 128)):
                slot[s2] = ch
            for s2 in range(128):
                perm[s2] = slot[s2]
            perm = perm.cuda()
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(128, device="cuda")
            qh = attn.norm_q(
                qp.unflatten(-1, (HEADS, HD))[None][..., inv])
            kh = attn.norm_k(
                kp.unflatten(-1, (HEADS, HD))[None][..., inv])
            qh = _rope(qh, *rope)
            kh = _rope(kh, *rope)
            torch.testing.assert_close(
                q.float(), qh[..., perm].float(), atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(
                k.float(), kh[..., perm].float(), atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(
                v.float(), vp.unflatten(-1, (HEADS, HD))[None].float(),
                atol=1e-3, rtol=1e-3)
    finally:
        handle.detach()


@cuda
def test_each_route_permutes_its_own_pack():
    """The late-binding lesson: loop-scope closures resolve free
    variables to the LAST iteration — the first route's warmup then
    permuted a stranger's weights (proven empirically) while every
    single-site check stayed above a loose floor and fifty stacked
    layers compounded to 0.71. Ownership is the invariant to pin."""
    from flash_rt.structures import workspace
    from flash_rt.structures.adapters.packed_stream_qk_norm_rope import (
        PackedStreamQkNormRopeAdapter)
    from flash_rt.structures.swap import attach

    workspace.clear()
    host, plan = _build()
    seq = 16
    x = torch.randn(1, seq, DIM, device="cuda", dtype=torch.bfloat16)
    rope = _tables(seq, "cuda")
    with torch.inference_mode():
        ref = host(x, rope).float()
    res = PackedStreamQkNormRopeAdapter()(host, plan)
    handle = attach(host, plan.swaps, observe=res["observed"],
                    revert=res["revert"])
    try:
        w0 = plan.swaps["attn0.to_q"].w8.clone()
        w1 = plan.swaps["attn1.to_q"].w8.clone()
        with torch.inference_mode():
            host.attn0(x, rope)
        assert not torch.equal(w0, plan.swaps["attn0.to_q"].w8), \
            "attn0's warmup did not permute its own pack"
        assert torch.equal(w1, plan.swaps["attn1.to_q"].w8), \
            "attn0's warmup touched attn1's pack"
        with torch.inference_mode():
            got = host(x, rope).float()
        cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
        assert float(cos) > 0.998, float(cos)
    finally:
        handle.detach()

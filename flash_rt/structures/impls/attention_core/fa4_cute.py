"""attention_core — the FlashAttention-4 (CuTe DSL) dense form.

The community ``kernels-community/flash-attn4`` package, reused as-is:
the measured Thor comparison put it at operational parity with the
native FA4 backend across the GROOT profiles (D48 / D72 / causal GQA
D128), so no FlashRT twin is published. Numerics-preserving BF16 —
in the variant family's precision ordering it sits directly after FA2,
ahead of the FP8 forms.

Same stateless dense seam as the FA2 dense form: complete Q/K/V per
call, host SDPA layout, allowed-ranges packing. GQA is native
(``pack_gqa``); custom softmax scales pass straight through. The DSL
JIT compiles per shape at first call, which the bind smoke absorbs at
bind time — a device or dependency the DSL cannot serve surfaces as a
bind refusal there, not inside the host's forward.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardedSeam
from .fa2_seqused import _allowed_ranges

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "kernels-community/flash-attn4",
    "version": ">=0",
}


class DenseAttentionFa4Cute(GuardedSeam, torch.nn.Module):
    """FA4 (CuTe DSL) replacement for an ordinary dense SDPA call.

    Inputs and outputs use the host SDPA layout ``[B, H, S, D]``; the
    kernel consumes ``[B, S, H, D]``.
    """

    def __init__(self, q_shape, kv_shape, dtype: torch.dtype, device,
                 allowed_ranges=None):
        super().__init__()
        b, heads, seq_q, head_dim = q_shape
        kb, kv_heads, seq_kv, kv_dim = kv_shape
        if kb != b or kv_dim != head_dim:
            raise ValueError(
                "attention_core fa4_cute: Q and KV batch/head "
                "dimensions differ")
        if heads % kv_heads:
            raise ValueError(
                "attention_core fa4_cute: query heads must be "
                "divisible by KV heads")
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(kv_shape)
        self.allowed_ranges = tuple(allowed_ranges or ())
        kern = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
        self._fn = kern.flash_attn_func
        if self.allowed_ranges:
            packed_seq = sum(hi - lo for lo, hi in self.allowed_ranges)
            self.register_buffer("packed_k", torch.empty(
                b, packed_seq, kv_heads, head_dim, device=device,
                dtype=dtype))
            self.register_buffer("packed_v",
                                 torch.empty_like(self.packed_k))
        self._frt_arm(
            dtypes=(dtype,), device=torch.device(device),
            k=int(head_dim), rows=int(b * heads * seq_q))

    def forward(self, query, key, value, *, scale=None):
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        if tuple(query.shape) != self.q_shape:
            raise ValueError(
                "attention_core fa4_cute: query shape moved from "
                f"{self.q_shape} to {tuple(query.shape)}")
        if tuple(key.shape) != self.kv_shape or value.shape != key.shape:
            raise ValueError(
                "attention_core fa4_cute: key/value shape moved from "
                f"{self.kv_shape} to {tuple(key.shape)}/"
                f"{tuple(value.shape)}")
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        if not q.is_contiguous():
            q = q.contiguous()
        if self.allowed_ranges:
            offset = 0
            for lo, hi in self.allowed_ranges:
                length = hi - lo
                self.packed_k[:, offset:offset + length].copy_(k[:, lo:hi])
                self.packed_v[:, offset:offset + length].copy_(v[:, lo:hi])
                offset += length
            k, v = self.packed_k, self.packed_v
        else:
            if not k.is_contiguous():
                k = k.contiguous()
            if not v.is_contiguous():
                v = v.contiguous()
        out = self._fn(q, k, v, softmax_scale=scale, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out.transpose(1, 2).to(query.dtype)


def bind_dense_attention(captures):
    """Bind one stateless dense FA4-CuTe core from repeated captures.

    Same qualification walk as the family's other dense binders. The
    DSL owns its own shape envelope, so there is no head-dim table
    here: the bind smoke below runs the real entry at the captured
    shape, and what the DSL cannot compile is a bind refusal.
    """
    if not captures:
        raise ValueError("attention_core fa4_cute: no captures")
    first = captures[0]
    query, key, value = first["q"], first["key"], first["value"]
    allowed_ranges = _allowed_ranges(first.get("mask"))
    if allowed_ranges is None:
        return None
    expected = (tuple(query.shape), tuple(key.shape), tuple(value.shape),
                query.dtype, key.dtype, value.dtype)
    for capture in captures[1:]:
        got = (
            tuple(capture["q"].shape),
            tuple(capture["key"].shape),
            tuple(capture["value"].shape),
            capture["q"].dtype,
            capture["key"].dtype,
            capture["value"].dtype,
        )
        if got != expected:
            raise ValueError(
                "attention_core fa4_cute: shape, dtype, or mask moved "
                f"within one calibration call: {expected} -> {got}")
        if _allowed_ranges(capture.get("mask")) != allowed_ranges:
            raise ValueError(
                "attention_core fa4_cute: mask pattern moved within "
                "one calibration call")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("attention_core fa4_cute: Q/K/V dtypes differ")
    bound = DenseAttentionFa4Cute(
        query.shape, key.shape, query.dtype, query.device,
        allowed_ranges=allowed_ranges)
    with torch.no_grad():
        probe = bound(torch.zeros_like(query),
                      torch.zeros_like(key),
                      torch.zeros_like(value))
    if probe.shape != query.shape or not torch.isfinite(probe).all():
        raise ValueError(
            "attention_core fa4_cute: bind smoke produced shape "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound

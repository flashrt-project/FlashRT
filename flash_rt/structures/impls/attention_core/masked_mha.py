"""attention_core — the allocation-free masked-MHA dense form.

The ``flashrt/masked-mha-runtime`` package: a padded-length MHA whose
softmax reads and writes only the valid key columns, with every buffer
caller-owned — the form that removed the per-call logits pre-fill
sweep and, with it, the graph-replay nondeterminism of reading
uninitialized padding (replays are bitwise). BF16/FP16 at the host's
own precision; in the family's ordering it follows the FA4 forms and
precedes the FP8 one.

The kernel's layout is batch-free ``(S, H, D)`` per tensor, so this
form serves the batch-of-one sites the packed dense seam sees
everywhere in the VLA hosts; a batched site stays with the other
variants. Masks reduce to contiguous allowed ranges exactly as in the
FA2 dense form — packed KV rows are the valid length, which is the
whole masking contract.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardedSeam
from .fa2_seqused import _allowed_ranges

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/masked-mha-runtime",
    "version": ">=1",
}


class DenseAttentionMaskedMha(GuardedSeam, torch.nn.Module):
    """Masked-MHA replacement for a batch-of-one dense SDPA call.

    Host layout ``[1, H, S, D]`` in and out; the kernel consumes
    ``(S, H, D)`` with a caller-owned padded logits scratch and output,
    both allocated once here — the hot path allocates nothing.
    """

    def __init__(self, q_shape, kv_shape, dtype: torch.dtype, device,
                 allowed_ranges=None):
        super().__init__()
        b, heads, seq_q, head_dim = q_shape
        kb, kv_heads, seq_kv, kv_dim = kv_shape
        if b != 1 or kb != 1:
            raise ValueError(
                "attention_core masked_mha: the kernel layout is "
                "batch-free; only batch-of-one sites qualify")
        if kv_dim != head_dim:
            raise ValueError(
                "attention_core masked_mha: Q and KV head dims differ")
        if heads != kv_heads:
            raise ValueError(
                "attention_core masked_mha: MHA form; GQA sites take "
                "the FA4 variants")
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(kv_shape)
        self.allowed_ranges = tuple(allowed_ranges or ())
        kern = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
        self._fn = kern.forward_static
        packed_seq = (sum(hi - lo for lo, hi in self.allowed_ranges)
                      if self.allowed_ranges else seq_kv)
        self._packed_seq = packed_seq
        if self.allowed_ranges:
            self.register_buffer("packed_k", torch.empty(
                packed_seq, kv_heads, head_dim, device=device,
                dtype=dtype))
            self.register_buffer("packed_v",
                                 torch.empty_like(self.packed_k))
        stride = (packed_seq + 7) // 8 * 8
        self.register_buffer("_logits", torch.empty(
            heads, seq_q, stride, device=device, dtype=dtype))
        self.register_buffer("_out", torch.empty(
            seq_q, heads, head_dim, device=device, dtype=dtype))
        self._frt_arm(
            dtypes=(dtype,), device=torch.device(device),
            k=int(head_dim), rows=int(heads * seq_q))

    def forward(self, query, key, value, *, scale=None):
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        if tuple(query.shape) != self.q_shape:
            raise ValueError(
                "attention_core masked_mha: query shape moved from "
                f"{self.q_shape} to {tuple(query.shape)}")
        if tuple(key.shape) != self.kv_shape or value.shape != key.shape:
            raise ValueError(
                "attention_core masked_mha: key/value shape moved from "
                f"{self.kv_shape} to {tuple(key.shape)}/"
                f"{tuple(value.shape)}")
        q = query[0].transpose(0, 1).contiguous()      # (S_q, H, D)
        k = key[0].transpose(0, 1)
        v = value[0].transpose(0, 1)
        if self.allowed_ranges:
            offset = 0
            for lo, hi in self.allowed_ranges:
                length = hi - lo
                self.packed_k[offset:offset + length].copy_(k[lo:hi])
                self.packed_v[offset:offset + length].copy_(v[lo:hi])
                offset += length
            k, v = self.packed_k, self.packed_v
        else:
            k = k.contiguous()
            v = v.contiguous()
        out = self._fn(q, k, v, logits=self._logits, out=self._out,
                       scale=scale)
        return out.transpose(0, 1).unsqueeze(0).to(query.dtype)


def bind_dense_attention(captures):
    """Bind one masked-MHA core from repeated host captures."""
    if not captures:
        raise ValueError("attention_core masked_mha: no captures")
    first = captures[0]
    query, key, value = first["q"], first["key"], first["value"]
    if query.shape[0] != 1 or query.shape[1] != key.shape[1]:
        return None
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
                "attention_core masked_mha: shape, dtype, or mask "
                f"moved within one calibration call: {expected} -> {got}")
        if _allowed_ranges(capture.get("mask")) != allowed_ranges:
            raise ValueError(
                "attention_core masked_mha: mask pattern moved within "
                "one calibration call")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError(
            "attention_core masked_mha: Q/K/V dtypes differ")
    bound = DenseAttentionMaskedMha(
        query.shape, key.shape, query.dtype, query.device,
        allowed_ranges=allowed_ranges)
    with torch.no_grad():
        probe = bound(torch.zeros_like(query),
                      torch.zeros_like(key),
                      torch.zeros_like(value))
    if probe.shape != query.shape or not torch.isfinite(probe).all():
        raise ValueError(
            "attention_core masked_mha: bind smoke produced shape "
            f"{tuple(probe.shape)}, "
            f"finite={bool(torch.isfinite(probe).all())}")
    return bound

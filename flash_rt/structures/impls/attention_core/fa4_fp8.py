"""attention_core — the FA4 (Blackwell FP8) dense form.

The SM100-family twin of :class:`.fa2_seqused.DenseAttention`: the same
stateless dense seam — complete Q/K/V every call, host SDPA layout,
allowed-ranges packing — executed by the
``flashrt/fp8-cross-attention-blackwell`` kernel: non-causal FP8 GQA
attention at head_dim 128, BF16 out. The two variants split the
hardware between them through their packages' own arch declarations
(this one ships ``10.0a/11.0a`` builds, the FA2 runtime ships none for
those majors), so selection is the ordinary refusal machinery rather
than a second table: the family binder tries FA2 first and falls to
this form where FA2's kernel refuses the device. On the devices this
form serves, FP8 attention is the production hot path, and the parity
gates downstream judge its quantization like any other impl's.

Activation scales are per-tensor static, calibrated from the same real
captures the qualification reads (amax over every capture, house
formula). The kernel bakes the ``1/sqrt(head_dim)`` softmax convention;
a host that calls with any other scale is refused at that call.
"""

from __future__ import annotations

import torch

from .. import hub_kernel
from ...guard import PROCEED, GuardedSeam
from .fa2_seqused import _allowed_ranges

KERNEL_DEP = {
    "provider": "huggingface_kernels",
    "repo": "flashrt/fp8-cross-attention-blackwell",
    "version": ">=1",
}

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_HEAD_DIM = 128  # the kernel's contract, exact


class DenseAttentionFa4(GuardedSeam, torch.nn.Module):
    """FA4 replacement for an ordinary dense SDPA call.

    Inputs and outputs use the host SDPA layout ``[B, H, S, D]``; the
    kernel consumes ``[B, S, H, D]``. Q/K/V are quantized per call with
    the calibrated static scales — elementwise work a compiled or
    packaged graph fuses into its neighbours.
    """

    def __init__(self, q_shape, kv_shape, dtype: torch.dtype, device,
                 scales: tuple[float, float, float],
                 allowed_ranges=None):
        super().__init__()
        b, heads, seq_q, head_dim = q_shape
        kb, kv_heads, seq_kv, kv_dim = kv_shape
        if kb != b or kv_dim != head_dim:
            raise ValueError(
                "attention_core fa4: Q and KV batch/head dimensions differ")
        if heads % kv_heads:
            raise ValueError(
                "attention_core fa4: query heads must be divisible by "
                "KV heads")
        if head_dim != _HEAD_DIM:
            raise ValueError(
                f"attention_core fa4: head_dim {head_dim} outside the "
                f"kernel contract ({_HEAD_DIM})")
        self.q_shape = tuple(q_shape)
        self.kv_shape = tuple(kv_shape)
        self.allowed_ranges = tuple(allowed_ranges or ())
        self._kfa = hub_kernel(KERNEL_DEP["repo"], KERNEL_DEP["version"])
        self._fn = self._kfa.fp8_gqa_cross_attention_bf16
        self._qs, self._ks, self._vs = (float(s) for s in scales)
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

    def _quant(self, t: torch.Tensor, scale: float) -> torch.Tensor:
        return (t.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)

    def forward(self, query, key, value, *, scale=None):
        admitted = self._frt_admit(query)
        if admitted is not PROCEED:
            return admitted
        if tuple(query.shape) != self.q_shape:
            raise ValueError(
                "attention_core fa4: query shape moved from "
                f"{self.q_shape} to {tuple(query.shape)}")
        if tuple(key.shape) != self.kv_shape or value.shape != key.shape:
            raise ValueError(
                "attention_core fa4: key/value shape moved from "
                f"{self.kv_shape} to {tuple(key.shape)}/"
                f"{tuple(value.shape)}")
        if scale is not None and abs(
                float(scale) - self.q_shape[-1] ** -0.5) > 1e-9:
            raise ValueError(
                "attention_core fa4: the kernel bakes the default "
                "softmax scale; this host passes another")
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        if self.allowed_ranges:
            offset = 0
            for lo, hi in self.allowed_ranges:
                length = hi - lo
                self.packed_k[:, offset:offset + length].copy_(k[:, lo:hi])
                self.packed_v[:, offset:offset + length].copy_(v[:, lo:hi])
                offset += length
            k, v = self.packed_k, self.packed_v
        out = self._fn(
            self._quant(q, self._qs).contiguous(),
            self._quant(k, self._ks).contiguous(),
            self._quant(v, self._vs).contiguous(),
            query_scale=self._qs, key_scale=self._ks,
            value_scale=self._vs)
        return out.transpose(1, 2).to(query.dtype)


def _amax_over(captures, key) -> float:
    amax = 0.0
    for cap in captures:
        amax = max(amax, float(cap[key].detach().float().abs().max()))
    return max(amax / _FP8_MAX, 1e-6)


def bind_dense_attention(captures):
    """Bind one stateless dense FA4 core from repeated host captures.

    Same qualification walk as the FA2 dense binder — stable shapes and
    dtypes across captures, a mask expressible as contiguous allowed
    ranges — plus the kernel's own contract (head_dim 128) and the
    per-tensor scale calibration this form adds. Returns ``None`` when
    the shape qualification fails so the caller can keep its path;
    raises when the kernel package refuses the device, so the family
    binder can record it and move on.
    """
    if not captures:
        raise ValueError("attention_core fa4: no captures")
    first = captures[0]
    query, key, value = first["q"], first["key"], first["value"]
    if query.shape[-1] != _HEAD_DIM:
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
                "attention_core fa4: shape, dtype, or mask moved within "
                f"one calibration call: {expected} -> {got}")
        if _allowed_ranges(capture.get("mask")) != allowed_ranges:
            raise ValueError(
                "attention_core fa4: mask pattern moved within one "
                "calibration call")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("attention_core fa4: Q/K/V dtypes differ")
    scales = (_amax_over(captures, "q"),
              _amax_over(captures, "key"),
              _amax_over(captures, "value"))
    return DenseAttentionFa4(
        query.shape, key.shape, query.dtype, query.device,
        scales, allowed_ranges=allowed_ranges)

"""FlashRT SageAttention2 Blackwell prefill kernels."""

from __future__ import annotations

import math
from typing import Optional

import torch

from ._ops import add_op_namespace_prefix, ops


HEAD_DIM = 128


def padded_k64(seqlen_k: int) -> int:
    seqlen_k = int(seqlen_k)
    if seqlen_k <= 0:
        raise ValueError("seqlen_k must be positive")
    return ((seqlen_k + 63) // 64) * 64


def q_scale_elems(batch: int, seqlen_q: int, q_heads: int) -> int:
    batch = int(batch)
    seqlen_q = int(seqlen_q)
    q_heads = int(q_heads)
    if batch <= 0 or seqlen_q <= 0 or q_heads <= 0:
        raise ValueError("shape values must be positive")
    return batch * q_heads * ((seqlen_q + 31) // 32)


def k_scale_elems(batch: int, seqlen_k: int, kv_heads: int) -> int:
    batch = int(batch)
    seqlen_k = int(seqlen_k)
    kv_heads = int(kv_heads)
    if batch <= 0 or seqlen_k <= 0 or kv_heads <= 0:
        raise ValueError("shape values must be positive")
    return batch * kv_heads * ((seqlen_k + 63) // 64)


def v_scale_elems(batch: int, kv_heads: int) -> int:
    batch = int(batch)
    kv_heads = int(kv_heads)
    if batch <= 0 or kv_heads <= 0:
        raise ValueError("shape values must be positive")
    return batch * kv_heads * HEAD_DIM


def _check_bhd128(x: torch.Tensor, name: str) -> None:
    if x.dim() != 4 or x.shape[-1] != HEAD_DIM:
        raise RuntimeError(f"{name} must have shape (batch, seqlen, heads, 128)")


def _empty_i8_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.int8)


def _empty_q_scale(q: torch.Tensor) -> torch.Tensor:
    return torch.empty((q_scale_elems(q.shape[0], q.shape[1], q.shape[2]),), device=q.device, dtype=torch.float32)


def _empty_k_scale(k: torch.Tensor) -> torch.Tensor:
    return torch.empty((k_scale_elems(k.shape[0], k.shape[1], k.shape[2]),), device=k.device, dtype=torch.float32)


def _empty_v_scale(v: torch.Tensor) -> torch.Tensor:
    return torch.empty((v_scale_elems(v.shape[0], v.shape[2]),), device=v.device, dtype=torch.float32)


def _empty_v_fp8_tpp(v: torch.Tensor) -> torch.Tensor:
    return torch.empty((v.shape[0], HEAD_DIM, v.shape[2], padded_k64(v.shape[1])), device=v.device, dtype=torch.int8)


@torch.library.register_fake(add_op_namespace_prefix("quantize_q_bf16_d128"))
def _quantize_q_fake(q: torch.Tensor, q_i8: torch.Tensor, q_scale: torch.Tensor) -> None:
    _check_bhd128(q, "q")
    if q_i8.shape != q.shape:
        raise RuntimeError("q_i8 shape must match q")
    if q_scale.numel() < q_scale_elems(q.shape[0], q.shape[1], q.shape[2]):
        raise RuntimeError("q_scale is too small")
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_k_bf16_d128"))
def _quantize_k_fake(k: torch.Tensor, k_i8: torch.Tensor, k_scale: torch.Tensor) -> None:
    _check_bhd128(k, "k")
    if k_i8.shape != k.shape:
        raise RuntimeError("k_i8 shape must match k")
    if k_scale.numel() < k_scale_elems(k.shape[0], k.shape[1], k.shape[2]):
        raise RuntimeError("k_scale is too small")
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_v_fp16_bf16_d128"))
def _quantize_v_fp16_fake(v: torch.Tensor, v_half: torch.Tensor) -> None:
    _check_bhd128(v, "v")
    if v_half.shape != v.shape:
        raise RuntimeError("v_half shape must match v")
    return None


@torch.library.register_fake(add_op_namespace_prefix("quantize_v_fp8_bf16_d128"))
def _quantize_v_fp8_fake(v: torch.Tensor, v_fp8_tpp: torch.Tensor, v_scale: torch.Tensor) -> None:
    _check_bhd128(v, "v")
    expected = (v.shape[0], HEAD_DIM, v.shape[2], padded_k64(v.shape[1]))
    if tuple(v_fp8_tpp.shape) != expected:
        raise RuntimeError("v_fp8_tpp shape must be (batch,128,kv_heads,padded_seqlen)")
    if v_scale.numel() < v_scale_elems(v.shape[0], v.shape[2]):
        raise RuntimeError("v_scale is too small")
    return None


@torch.library.register_fake(add_op_namespace_prefix("sage2_qk_int8_sv_f16_bf16_d128"))
def _sage2_f16_fake(
    q_i8: torch.Tensor,
    k_i8: torch.Tensor,
    v_half: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    out: torch.Tensor,
) -> None:
    _check_bhd128(q_i8, "q_i8")
    _check_bhd128(k_i8, "k_i8")
    _check_bhd128(v_half, "v_half")
    if out.shape != q_i8.shape:
        raise RuntimeError("out shape must match q_i8")
    return None


@torch.library.register_fake(add_op_namespace_prefix("sage2_qk_int8_sv_f8_bf16_d128"))
def _sage2_f8_fake(
    q_i8: torch.Tensor,
    k_i8: torch.Tensor,
    v_fp8_tpp: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    out: torch.Tensor,
) -> None:
    _check_bhd128(q_i8, "q_i8")
    _check_bhd128(k_i8, "k_i8")
    if out.shape != q_i8.shape:
        raise RuntimeError("out shape must match q_i8")
    return None


def quantize_q_bf16_d128(q: torch.Tensor, q_i8: Optional[torch.Tensor] = None, q_scale: Optional[torch.Tensor] = None):
    if q_i8 is None:
        q_i8 = _empty_i8_like(q)
    if q_scale is None:
        q_scale = _empty_q_scale(q)
    ops.quantize_q_bf16_d128(q, q_i8, q_scale)
    return q_i8, q_scale


def quantize_k_bf16_d128(k: torch.Tensor, k_i8: Optional[torch.Tensor] = None, k_scale: Optional[torch.Tensor] = None):
    if k_i8 is None:
        k_i8 = _empty_i8_like(k)
    if k_scale is None:
        k_scale = _empty_k_scale(k)
    ops.quantize_k_bf16_d128(k, k_i8, k_scale)
    return k_i8, k_scale


def quantize_v_fp16_bf16_d128(v: torch.Tensor, v_half: Optional[torch.Tensor] = None):
    if v_half is None:
        v_half = torch.empty_like(v, dtype=torch.float16)
    ops.quantize_v_fp16_bf16_d128(v, v_half)
    return v_half


def quantize_v_fp8_bf16_d128(
    v: torch.Tensor,
    v_fp8_tpp: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
):
    if v_fp8_tpp is None:
        v_fp8_tpp = _empty_v_fp8_tpp(v)
    if v_scale is None:
        v_scale = _empty_v_scale(v)
    ops.quantize_v_fp8_bf16_d128(v, v_fp8_tpp, v_scale)
    return v_fp8_tpp, v_scale


def sage2_qk_int8_sv_f16_bf16_d128(
    q_i8: torch.Tensor,
    k_i8: torch.Tensor,
    v_half: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    if out is None:
        out = torch.empty_like(q_i8, dtype=torch.bfloat16)
    ops.sage2_qk_int8_sv_f16_bf16_d128(q_i8, k_i8, v_half, q_scale, k_scale, float(softmax_scale), bool(causal), out)
    return out


def sage2_qk_int8_sv_f8_bf16_d128(
    q_i8: torch.Tensor,
    k_i8: torch.Tensor,
    v_fp8_tpp: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    if out is None:
        out = torch.empty_like(q_i8, dtype=torch.bfloat16)
    ops.sage2_qk_int8_sv_f8_bf16_d128(
        q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale, float(softmax_scale), bool(causal), out
    )
    return out


def sage2_prefill_f16_bf16_d128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_i8, q_scale = quantize_q_bf16_d128(q)
    k_i8, k_scale = quantize_k_bf16_d128(k)
    v_half = quantize_v_fp16_bf16_d128(v)
    return sage2_qk_int8_sv_f16_bf16_d128(
        q_i8, k_i8, v_half, q_scale, k_scale, softmax_scale=softmax_scale, causal=causal, out=out
    )


def sage2_prefill_fp8v_bf16_d128(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_i8, q_scale = quantize_q_bf16_d128(q)
    k_i8, k_scale = quantize_k_bf16_d128(k)
    v_fp8_tpp, v_scale = quantize_v_fp8_bf16_d128(v)
    return sage2_qk_int8_sv_f8_bf16_d128(
        q_i8, k_i8, v_fp8_tpp, q_scale, k_scale, v_scale,
        softmax_scale=softmax_scale, causal=causal, out=out
    )


__all__ = [
    "HEAD_DIM",
    "padded_k64",
    "q_scale_elems",
    "k_scale_elems",
    "v_scale_elems",
    "quantize_q_bf16_d128",
    "quantize_k_bf16_d128",
    "quantize_v_fp16_bf16_d128",
    "quantize_v_fp8_bf16_d128",
    "sage2_qk_int8_sv_f16_bf16_d128",
    "sage2_qk_int8_sv_f8_bf16_d128",
    "sage2_prefill_f16_bf16_d128",
    "sage2_prefill_fp8v_bf16_d128",
]

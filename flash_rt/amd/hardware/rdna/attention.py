"""BF16 multi-head/GQA attention for RDNA wave32 targets."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q, K, V, O,
               QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
               KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
               VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
               SQ: tl.constexpr, SK: tl.constexpr, HQ: tl.constexpr,
               HK: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    head = tl.program_id(1) % HQ
    batch = tl.program_id(1) // HQ
    kv_head = head // (HQ // HK)
    dim = tl.arange(0, BD)
    col = tl.arange(0, BN)
    q = tl.load(Q + batch * QS0 + row[:, None] * QS1 + head * QS2 + dim[None, :],
                (row[:, None] < SQ) & (dim[None, :] < D), 0)
    running_max = tl.full((BM,), -float("inf"), tl.float32)
    running_sum = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, BD), 0.0, tl.float32)
    end = SK
    if CAUSAL:
        end = tl.minimum(SK, (tl.program_id(0) + 1) * BM)
    for start in range(0, end, BN):
        keys = start + col
        k = tl.load(K + batch * KS0 + keys[None, :] * KS1 + kv_head * KS2 + dim[:, None],
                    (keys[None, :] < SK) & (dim[:, None] < D), 0)
        scores = tl.dot(q, k).to(tl.float32) * (D ** -0.5 * 1.4426950408889634)
        valid = keys[None, :] < SK
        if CAUSAL:
            valid = valid & (keys[None, :] <= row[:, None])
        scores = tl.where(valid, scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, 1))
        p = tl.exp2(scores - new_max[:, None])
        correction = tl.exp2(running_max - new_max)
        running_sum = running_sum * correction + tl.sum(p, 1)
        acc = acc * correction[:, None]
        v = tl.load(V + batch * VS0 + keys[:, None] * VS1 + kv_head * VS2 + dim[None, :],
                    (keys[:, None] < SK) & (dim[None, :] < D), 0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        running_max = new_max
    out = acc / running_sum[:, None]
    offset = ((batch * SQ + row[:, None]) * HQ + head) * D + dim[None, :]
    tl.store(O + offset, out, (row[:, None] < SQ) & (dim[None, :] < D))


def attention_bf16(q, k, v, *, causal=False):
    """Attention on BSHD tensors without expanding grouped K/V heads.

    Supports head widths 48, 64 and 128; causal mode is square prefill.
    Q/K/V may be strided fused-projection views with contiguous head channels.
    No score matrix or temporary workspace is materialized.
    """
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("RDNA attention requires BSHD tensors")
    b, sq, hq, d = q.shape
    bk, sk, hk, dk = k.shape
    if (k.shape != v.shape or b != bk or d != dk or min(b, sq, sk, hk) < 1
            or hq < 1 or hq % hk or d not in (48, 64, 128)):
        raise ValueError(f"unsupported RDNA attention shapes {q.shape}, {k.shape}, {v.shape}")
    if causal and sq != sk:
        raise ValueError("RDNA causal attention requires equal query/key lengths")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("RDNA attention requires BF16 tensors")
    if (not torch.version.hip or q.device.type != "cuda"
            or any(t.device != q.device or t.stride(-1) != 1 for t in (q, k, v))):
        raise ValueError("RDNA attention requires one ROCm device and contiguous head channels")
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    _attention[(triton.cdiv(sq, 32), b * hq)](
        q, k, v, out, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
        sq, sk, hq, hk, d, causal, 32, 64, triton.next_power_of_2(d),
        num_warps=4, num_stages=1)
    return out

"""BF16 multi-head/GQA attention for RDNA wave32 targets."""

from __future__ import annotations

import torch
from ._native import kernels


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
    with torch.cuda.device(q.device):
        kernels().attention_bshd_bf16_rdna(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
            b, sq, sk, hq, hk, d, *q.stride()[:3], *k.stride()[:3],
            *v.stride()[:3], causal, torch.cuda.current_stream(q.device).cuda_stream)
    return out

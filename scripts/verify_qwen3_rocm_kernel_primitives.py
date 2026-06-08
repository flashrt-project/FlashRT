#!/usr/bin/env python3
"""Verify Qwen3-specific ROCm kernel primitives against Torch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import flash_rt.flash_rt_rocm_kernels as kernels


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _rms_plain_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * inv * weight.float()).to(torch.bfloat16)


def main() -> None:
    torch.manual_seed(7)
    device = "cuda"
    eps = 1.0e-6

    for rows, hidden in [(1, 4096), (7, 4096), (3, 128)]:
        x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        out = torch.empty_like(x)
        kernels.rms_norm_bf16_plain_ptr(
            x.data_ptr(), w.data_ptr(), out.data_ptr(), rows, hidden, eps
        )
        ref = _rms_plain_ref(x, w, eps)
        torch.cuda.synchronize()
        print(
            "RMS_PLAIN",
            rows,
            hidden,
            "cos",
            _cos(out, ref),
            "max_abs",
            _max_abs(out, ref),
        )
        assert _cos(out, ref) > 0.9999
        assert _max_abs(out, ref) <= 0.03125

    residual = torch.randn(2, 4096, device=device, dtype=torch.bfloat16)
    x = torch.randn(2, 4096, device=device, dtype=torch.bfloat16)
    w = torch.randn(4096, device=device, dtype=torch.bfloat16)
    residual_ref = residual.clone()
    out = torch.empty_like(residual)
    kernels.residual_add_rms_norm_bf16_plain_ptr(
        residual.data_ptr(), x.data_ptr(), w.data_ptr(), out.data_ptr(), 2, 4096, eps
    )
    residual_ref = (residual_ref.float() + x.float()).to(torch.bfloat16)
    ref = _rms_plain_ref(residual_ref, w, eps)
    torch.cuda.synchronize()
    print("RESIDUAL_RMS_PLAIN", "cos", _cos(out, ref), "max_abs", _max_abs(out, ref))
    assert _cos(out, ref) > 0.9999
    assert _max_abs(out, ref) <= 0.03125

    gate_up = torch.randn(5, 2 * 12288, device=device, dtype=torch.bfloat16)
    out = torch.empty(5, 12288, device=device, dtype=torch.bfloat16)
    kernels.silu_mul_merged_bf16_ptr(
        gate_up.data_ptr(), out.data_ptr(), gate_up.shape[0], 12288
    )
    ref = (F.silu(gate_up[:, :12288].float()) * gate_up[:, 12288:].float()).to(
        torch.bfloat16
    )
    torch.cuda.synchronize()
    print("SILU_MUL_MERGED", "cos", _cos(out, ref), "max_abs", _max_abs(out, ref))
    assert _cos(out, ref) > 0.9999
    assert _max_abs(out, ref) <= 0.03125

    print("OK qwen3_rocm_kernel_primitives")


if __name__ == "__main__":
    main()

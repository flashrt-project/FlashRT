"""Dense projections at prefill M as E4M3 x E4M3 on vLLM's CUTLASS scaled GEMM (sm120, ~450 TOPS vs 250 for the
W4A8 int8 kernel): the NVFP4 weight is folded to E4M3 per output row on every call (ModelOpt layout: byte = elements
2i | 2i+1 << 4, E4M3 block scale per 16, fp32 global scale), the activation is quantized per token."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1_value(x):
    e = (x >> 1) & 3
    mm = (x & 1).to(tl.float32)
    mag = tl.where(e == 0, mm * 0.5, (1.0 + mm * 0.5) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((x >> 3) & 1 == 1, -mag, mag)


@triton.jit
def _deq_chunk(codes_ptr, scales_ptr, row, c0, gs, CH: tl.constexpr):
    # dequantized elements [c0, c0 + CH) of one NVFP4 row (ModelOpt layout)
    j = tl.arange(0, CH // 2)
    b = tl.load(codes_ptr + row * 0 + (c0 // 2) + j)
    sc = (
        tl.load(scales_ptr + (c0 // 16) + tl.arange(0, CH // 16))
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    lo = _e2m1_value(b & 15)
    hi = _e2m1_value(b >> 4)
    v = tl.reshape(tl.join(lo, hi), (CH,))
    scf = tl.reshape(
        tl.broadcast_to(tl.reshape(sc, (CH // 16, 1)), (CH // 16, 16)), (CH,)
    )
    return v * scf * gs


@triton.jit
def _fold_kernel(
    codes_ptr, scales_ptr, gs, w8_ptr, ws_ptr, K: tl.constexpr, CH: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cp = codes_ptr + row * (K // 2)
    sp = scales_ptr + row * (K // 16)
    amax = 0.0
    for c0 in tl.static_range(0, K, CH):
        v = _deq_chunk(cp, sp, row, c0, gs, CH)
        amax = tl.maximum(amax, tl.max(tl.abs(v)))
    s = tl.maximum(amax / 448.0, 1e-12)
    for c0 in tl.static_range(0, K, CH):
        v = _deq_chunk(cp, sp, row, c0, gs, CH)
        tl.store(w8_ptr + row * K + c0 + tl.arange(0, CH), (v / s).to(tl.float8e4nv))
    tl.store(ws_ptr + row, s)


@triton.jit
def _quant_rows_kernel(x_ptr, x8_ptr, xs_ptr, K: tl.constexpr, CH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    amax = 0.0
    for c0 in tl.static_range(0, K, CH):
        v = tl.load(x_ptr + row * K + c0 + tl.arange(0, CH)).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v)))
    s = tl.maximum(amax / 448.0, 1e-12)
    for c0 in tl.static_range(0, K, CH):
        v = tl.load(x_ptr + row * K + c0 + tl.arange(0, CH)).to(tl.float32)
        tl.store(x8_ptr + row * K + c0 + tl.arange(0, CH), (v / s).to(tl.float8e4nv))
    tl.store(xs_ptr + row, s)


def fold_nvfp4_to_e4m3(codes: torch.Tensor, scales: torch.Tensor, gs: float):
    N, K2 = codes.shape
    K = K2 * 2
    w8 = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=codes.device)
    ws = torch.empty((N, 1), dtype=torch.float32, device=codes.device)
    _fold_kernel[(N,)](codes, scales, float(gs), w8, ws, K=K, CH=1024, num_warps=4)
    return w8, ws


def quant_rows_e4m3(x: torch.Tensor):
    x = x.contiguous()
    M, K = x.shape
    x8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    _quant_rows_kernel[(M,)](x, x8, xs, K=K, CH=1024, num_warps=4)
    return x8, xs


def dense_fp8_gemm(
    x: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    gs: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    x8, xs = quant_rows_e4m3(x)
    w8, ws = fold_nvfp4_to_e4m3(codes, scales, gs)
    if (
        out is None
        or not out.is_contiguous()
        or out.dtype != torch.bfloat16
        or out.shape != (x.shape[0], codes.shape[0])
    ):
        out = torch.empty(
            (x.shape[0], codes.shape[0]), dtype=torch.bfloat16, device=x.device
        )
    torch.ops._C.cutlass_scaled_mm(
        out, x8, w8.t(), xs, ws, None
    )  # [M, N] bf16, written in place
    return out

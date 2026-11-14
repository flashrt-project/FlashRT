"""BF16 and OCP E4M3 W8A8 operations with caller-owned outputs.

The target selects native FP8 activation math before execution. The software
byte conversion preserves OCP E4M3 semantics without using CDNA3 FNUZ.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_e4m3(bits):
    bits = bits.to(tl.int32)
    magnitude = bits & 127
    normal = ((magnitude << 20) + 0x3C000000).to(tl.float32, bitcast=True)
    value = tl.where(magnitude < 8, magnitude.to(tl.float32) * (1.0 / 512.0), normal)
    signed = value.to(tl.int32, bitcast=True) | ((bits & 128) << 24)
    return tl.where(magnitude == 127, float("nan"), signed.to(tl.float32, bitcast=True))


@triton.jit
def _encode_e4m3(value):
    absolute = tl.minimum(tl.abs(value), 448.0)
    raw = absolute.to(tl.int32, bitcast=True)
    # Round-to-nearest, ties-to-even, including the subnormal boundary.
    normal = ((raw + 0x7FFFF + ((raw >> 20) & 1)) >> 20) - 960
    scaled = absolute * 512.0
    lower = tl.floor(scaled).to(tl.int32)
    fraction = scaled - lower.to(tl.float32)
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower & 1) != 0))
    subnormal = lower + increment.to(tl.int32)
    packed = tl.where(absolute < 0.015625, subnormal, normal)
    sign = (value.to(tl.int32, bitcast=True) >> 24) & 128
    return (packed | sign).to(tl.uint8)


@triton.jit
def _pack_rows(X, Q, SCALE, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1e-12)
    tl.store(SCALE + row, scale)
    tl.store(Q + row * K + col, _encode_e4m3(x / scale), col < K)


@triton.jit
def _gemv(X, W, SCALE, AS, O, N: tl.constexpr, K: tl.constexpr,
          FP8: tl.constexpr, A_FP8: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    acc = tl.full((ROWS, BLOCK), 0.0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK)):
        k = start * BLOCK + col
        x = tl.load(X + k, k < K, 0)
        x = _decode_e4m3(x) if A_FP8 else x.to(tl.float32)
        weight = tl.load(W + row[:, None] * K + k[None, :],
                         (row[:, None] < N) & (k[None, :] < K), 0)
        if FP8:
            w = _decode_e4m3(weight)
        else:
            w = weight.to(tl.float32)
        acc = tl.fma(w, x[None, :], acc)
    value = tl.sum(acc, 1)
    if FP8:
        scale = tl.load(SCALE + row, row < N, 0)
        if A_FP8:
            scale *= tl.load(AS)
        value *= scale
    tl.store(O + row, value, row < N)


@triton.jit(do_not_specialize=["M"])
def _fp8_gemm(A, B, AS, BS, C, M, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(A + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), 0.0)
        b = tl.load(B + n[None, :] * K + kk[:, None],
                    (n[None, :] < N) & (kk[:, None] < K), 0.0)
        acc = tl.dot(a, b, acc)
    acc *= tl.load(AS + m, m < M, 0)[:, None] * tl.load(BS + n, n < N, 0)[None, :]
    tl.store(C + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@dataclass(frozen=True)
class LinearWeight:
    values: torch.Tensor
    scale: torch.Tensor | None = None


def pack_weight(weight, *, fp8):
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("RDNA linear weights must be contiguous BF16 [N,K]")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("RDNA linear weights must be finite")
    if not fp8:
        return LinearWeight(weight)
    packed = torch.empty_like(weight, dtype=torch.uint8)
    scales = torch.empty(weight.shape[0], device=weight.device, dtype=torch.float32)
    _pack_rows[(weight.shape[0],)](weight, packed, scales, weight.shape[1],
                                  triton.next_power_of_2(weight.shape[1]))
    return LinearWeight(packed, scales)


class RdnaLinear:
    """The target binds BF16 or native W8A8 math before execution."""

    def __init__(self, *, fp8_activations=False, fp8_long_k_rows=1):
        self.fp8_activations = fp8_activations
        self.fp8_long_k_rows = fp8_long_k_rows

    def __call__(self, x, weight, out, *, quant_scratch=None):
        if x.ndim != 2 or weight.values.ndim != 2 or out.ndim != 2:
            raise ValueError("RDNA linear requires two-dimensional buffers")
        m, k = x.shape
        n = weight.values.shape[0]
        if (min(m, n, k) < 1 or weight.values.shape[1] != k or out.shape != (m, n)
                or x.device.type != "cuda"
                or x.dtype != torch.bfloat16 or out.dtype not in (torch.bfloat16, torch.float32)
                or x.device != out.device or x.device != weight.values.device
                or not x.is_contiguous() or not out.is_contiguous() or not weight.values.is_contiguous()):
            raise ValueError("RDNA linear requires contiguous BF16 inputs and BF16/FP32 output buffers")
        if weight.scale is None:
            if weight.values.dtype != torch.bfloat16:
                raise TypeError("unscaled RDNA weights must be BF16")
        elif (weight.values.dtype != torch.uint8 or weight.scale.dtype != torch.float32
              or weight.scale.shape != (n,) or weight.scale.device != x.device
              or not weight.scale.is_contiguous()):
            raise ValueError("FP8 RDNA weights require OCP bytes and one FP32 scale per output row")
        quantize = weight.scale is not None
        if quantize and not self.fp8_activations:
            raise ValueError("FP8 weights require W8A8 activation quantization")
        if quantize:
            if quant_scratch is None:
                raise ValueError("FP8 activations require caller-owned quantization scratch")
            packed, scales = quant_scratch
            if (packed.shape != x.shape or packed.dtype != torch.uint8 or packed.device != x.device
                    or scales.shape != (m,) or scales.dtype != torch.float32
                    or scales.device != x.device or not packed.is_contiguous() or not scales.is_contiguous()):
                raise ValueError("native FP8 scratch must match the activation shape and device")
            _pack_rows[(m,)](x, packed, scales, k, triton.next_power_of_2(k))
        if m == 1:
            # One output row per group avoids the long-K reduction bottleneck;
            # compressed wide projections amortize conversion across more rows.
            rows, block = (8, 512) if weight.scale is not None and n >= 4096 else (1, 512)
            if weight.scale is not None and n < 4096:
                block = 1024
                if k >= 8192:
                    rows = self.fp8_long_k_rows
            _gemv[(triton.cdiv(n, rows),)](
                packed if quantize else x, weight.values, weight.scale,
                scales if quantize else None, out, n, k,
                weight.scale is not None, quantize, rows, block, num_warps=4)
        elif weight.scale is None:
            if out.dtype == torch.float32:
                torch.mm(x, weight.values.t(), out_dtype=torch.float32, out=out)
            else:
                torch.mm(x, weight.values.t(), out=out)
        else:
            bm = 32 if m >= 128 else 16
            _fp8_gemm[(triton.cdiv(m, bm), triton.cdiv(n, 128))](
                packed.view(torch.float8_e4m3fn), weight.values.view(torch.float8_e4m3fn),
                scales, weight.scale, out, m, n, k, bm, 128, 128, num_warps=4, num_stages=1)

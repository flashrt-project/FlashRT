"""Wave32 BF16 normalization and rotate-half primitives.

Intermediate BF16 casts preserve the materialized transformer's rounding,
including RMSNorm's weight multiplication and AdaLayerNorm's affine steps.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, W, B, R, O, SUM, WIDTH: tl.constexpr, EPS: tl.constexpr,
          CENTER: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
          WEIGHT_OFFSET: tl.constexpr, ROUND_AFFINE: tl.constexpr,
          RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    row = tl.program_id(0)
    x = tl.load(X + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
    if RESIDUAL:
        r = tl.load(R + row * WIDTH + col, col < WIDTH, 0).to(tl.float32)
        x = (x + r).to(tl.bfloat16).to(tl.float32)
        tl.store(SUM + row * WIDTH + col, x, col < WIDTH)
    if CENTER:
        mean = tl.sum(x, 0) / WIDTH
        x = tl.where(col < WIDTH, x - mean, 0.0)
    variance = tl.sum(x * x, 0) / WIDTH
    y = x * tl.rsqrt(variance + EPS)
    if ROUND_AFFINE:
        y = y.to(tl.bfloat16).to(tl.float32)
    if HAS_WEIGHT:
        scale = tl.load(W + col, col < WIDTH, 0).to(tl.float32) + WEIGHT_OFFSET
        if ROUND_AFFINE:
            scale = scale.to(tl.bfloat16).to(tl.float32)
        y = y * scale
        if ROUND_AFFINE:
            y = y.to(tl.bfloat16).to(tl.float32)
    if HAS_BIAS:
        y += tl.load(B + col, col < WIDTH, 0).to(tl.float32)
    tl.store(O + row * WIDTH + col, y, col < WIDTH)


def _check_output(out, x):
    if (out.shape != x.shape or out.dtype != x.dtype or out.device != x.device
            or not out.is_contiguous()):
        raise ValueError("output must be contiguous and match the input shape, dtype and device")


def normalize(x, weight=None, shift=None, *, residual=None, kind="rms", eps=1e-5,
              weight_offset=0.0, round_affine=False, out=None, out_sum=None):
    """RMS/centered normalization with vector affine and optional residual sum.

    Reduction is FP32. ``round_affine`` materializes BF16 boundaries before
    affine, after adding ``weight_offset``, and after multiplication. Supplying
    outputs supports callers that own persistent graph buffers.
    """
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise TypeError("RDNA normalization requires a BF16 device tensor")
    if x.ndim < 2 or x.numel() == 0 or kind not in ("rms", "layer") or eps <= 0:
        raise ValueError("unsupported RDNA normalization shape, kind or epsilon")
    width = x.shape[-1]
    for value in (weight, shift):
        if value is None:
            continue
        if (value.shape != (width,) or value.dtype != x.dtype
                or value.device != x.device or not value.is_contiguous()):
            raise ValueError("normalization parameters must be contiguous BF16 vectors on the input device")
    if residual is not None:
        if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
            raise ValueError("normalization residual must match the input shape, dtype and device")
        residual = residual.contiguous()
    x = x.contiguous()
    out = torch.empty_like(x) if out is None else out
    _check_output(out, x)
    if out_sum is not None and residual is None:
        raise ValueError("out_sum requires a residual")
    summed = (torch.empty_like(x) if out_sum is None else out_sum) if residual is not None else out
    _check_output(summed, x)
    if residual is not None and out.data_ptr() == summed.data_ptr():
        raise ValueError("normalized output and residual sum must use distinct buffers")
    _norm[(x.numel() // width,)](
        x, weight, shift, residual, out, summed, width, eps, kind == "layer",
        weight is not None, shift is not None, weight_offset, round_affine,
        residual is not None, triton.next_power_of_2(width),
        num_warps=4, enable_fp_fusion=False)
    return out, summed


@triton.jit
def _rope(X, C, S, O, SIZE: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
          XS: tl.constexpr, XH: tl.constexpr, CS: tl.constexpr, SS: tl.constexpr,
          BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = index % DIM
    head = index // DIM % HEADS
    row = index // (DIM * HEADS)
    offset = row * XS + head * XH
    value = tl.load(X + offset + channel, index < SIZE, 0).to(tl.float32)
    other = tl.load(X + offset + (channel + DIM // 2) % DIM, index < SIZE, 0).to(tl.float32)
    other = tl.where(channel < DIM // 2, -other, other)
    cos = tl.load(C + row * CS + channel, index < SIZE, 0).to(tl.float32)
    sin = tl.load(S + row * SS + channel, index < SIZE, 0).to(tl.float32)
    tl.store(O + index, value * cos + other * sin, index < SIZE)


def rope(x, cos, sin, *, out=None):
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise TypeError("RDNA RoPE requires a BF16 device tensor")
    if (x.ndim != 3 or x.numel() == 0 or x.shape[-1] % 2 or x.stride(-1) != 1
            or any(t.shape != (x.shape[0], x.shape[-1]) or t.device != x.device
                   or t.dtype not in (torch.bfloat16, torch.float32) or t.stride(-1) != 1
                   for t in (cos, sin))):
        raise ValueError("RDNA RoPE requires SHD input and matching SD cosine/sine tables")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device) if out is None else out
    _check_output(out, x)
    # The kernel reads both halves; an overlapping write could race another CTA.
    if out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr():
        raise ValueError("rotate-half output must not alias the input")
    _rope[(triton.cdiv(x.numel(), 256),)](
        x, cos, sin, out, x.numel(), x.shape[1], x.shape[2],
        x.stride(0), x.stride(1), cos.stride(0), sin.stride(0), 256,
        num_warps=4, enable_fp_fusion=False)
    return out

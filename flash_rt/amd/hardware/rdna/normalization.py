"""Wave32 BF16 normalization and rotate-half primitives.

Intermediate BF16 casts preserve the materialized transformer's rounding,
including RMSNorm's weight multiplication and AdaLayerNorm's affine steps.
"""

import torch
from ._native import kernels


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
    ptr = lambda value: 0 if value is None else value.data_ptr()
    with torch.cuda.device(x.device):
        kernels().normalize_bf16_rdna(
            x.data_ptr(), ptr(weight), ptr(shift), ptr(residual), out.data_ptr(),
            summed.data_ptr(), x.numel() // width, width, eps, kind == "layer",
            weight_offset, round_affine, torch.cuda.current_stream(x.device).cuda_stream)
    return out, summed


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
    with torch.cuda.device(x.device):
        kernels().rope_bf16_rdna(
            x.data_ptr(), cos.data_ptr(), sin.data_ptr(), out.data_ptr(),
            x.numel(), x.shape[1], x.shape[2], x.stride(0), x.stride(1),
            cos.stride(0), sin.stride(0), cos.dtype == torch.float32,
            sin.dtype == torch.float32, torch.cuda.current_stream(x.device).cuda_stream)
    return out

"""NVFP4 utilities for Pi0.5 FP4 frontend (Phase 4.3, additive).

Thin Python wrapper over flash_rt.flash_rt_fp4 kernels that handles:
  1. Offline weight quantization (fp16 → packed int4 + tile-interleaved SFB)
  2. Runtime activation quantization scratch allocation
  3. FP4 GEMM invocation with per-shape variant selection

Does NOT modify any existing module. Requires flash_rt_fp4.so built with SM100
CUTLASS support (see docs/v2/fp4_kernel_impl_progress.md §3.5-3.6).

Usage:
    w_quant = quant_weight_nvfp4(gate_w_fp16)          # offline, once
    scratch = FP4ActScratch(max_M, K, device)          # preallocated
    quant_act_nvfp4(x_fp16, scratch, M, stream)        # runtime
    fp4_gemm(scratch, w_quant, out_fp16, M, N, K, variant_idx, stream)
"""
from __future__ import annotations

import torch

import flash_rt.flash_rt_fp4 as fvk_fp4


# ────────────────────────────────────────────────────────────────────
# Variant selection — picked from docs/v2/fp4_kernel_impl_progress.md §4.3
# ────────────────────────────────────────────────────────────────────
# Encoder shapes (M=968):
#   Gate+Up [N=16384, K=2048] → V6 (wide-N, cluster1x1x1)
#   Down    [N=2048,  K=8192] → V7 (wide-K, cluster1x1x1)
#   QKV     [N=2560,  K=2048] → V4
#   O       [N=2048,  K=2048] → V4
#
# Decoder shapes (M=10, compute-bound — FP4 on par with FP8, kept for completeness):
#   Same variants work; V4 default for small-N, V6 for Gate+Up.
#
# Step B meta-test MUST re-verify the pick at the exact M=968 shape before
# Step C commits to the choice.
VARIANT_V1_C2X1X1    = 1  # tile128x256x128 cluster2x1x1
VARIANT_V4_SMALL_N   = 4  # tile128x128x128 cluster1x1x1
VARIANT_V6_WIDE_N    = 6  # tile128x256x128 cluster1x1x1
VARIANT_V7_WIDE_K    = 7  # tile128x128x256 cluster1x1x1
VARIANT_V8_WIDE_NK   = 8  # tile128x256x256 cluster1x1x1 (wide N+K)


def pick_variant(N: int, K: int) -> int:
    """Shape-based default variant. Calibrated via tests/profile_fp4_vs_fp8_layer.py
    full 10-variant sweep at Se=968 in CUDA Graph:
      Gate+Up (N=16384, K=2048): V8 311.9μs best (vs V7 326μs, V6 353μs)
      Down    (N=2048,  K=8192): V1 86.6μs best (vs V6 92μs, V7 104μs)
    """
    if N >= 16384:
        return VARIANT_V8_WIDE_NK
    if K >= 8192:
        return VARIANT_V1_C2X1X1
    return VARIANT_V6_WIDE_N


# ────────────────────────────────────────────────────────────────────
# Offline weight quantization
# ────────────────────────────────────────────────────────────────────
def quant_weight_nvfp4(w: torch.Tensor) -> dict:
    """Quantize fp16 weight [N, K] → NVFP4 (packed int4 + tile-interleaved SFB).

    Uses fused F1 kernel (single launch, no linear-scales intermediate).

    Args:
        w: fp16 tensor, shape [N, K], CUDA. K must be divisible by 16.

    Returns:
        dict with:
            'packed' : uint8 tensor [N, K/2]   (e2m1 NVFP4 elements)
            'sfb'    : uint8 tensor [sfb_bytes] (CUTLASS tile-interleaved UE4M3 scales)
            'N', 'K' : ints
    """
    assert w.dtype == torch.float16, f"expected fp16 weight, got {w.dtype}"
    assert w.device.type == 'cuda', "weight must be on CUDA"
    assert w.is_contiguous(), "weight must be contiguous"
    N, K = w.shape
    assert K % 16 == 0, f"K={K} must be divisible by 16 for NVFP4 block=16"

    device = w.device
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=device)
    sfb_bytes = fvk_fp4.sfa_size_bytes(N, K, True)
    # Scale-factor buffers must be zero-initialized: the tile-interleaved
    # layout rounds K up to 64-element atoms, and the quantize kernels only
    # write entries for real (row, block) coordinates. When K is not a
    # multiple of 64 the padding entries are never written; allocation
    # garbage there can decode as UE4M3 NaN and poison the block-scaled
    # GEMM accumulator. Zero scales keep the padding mathematically inert.
    sfb = torch.zeros(sfb_bytes, dtype=torch.uint8, device=device)

    rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        w.data_ptr(), packed.data_ptr(), sfb.data_ptr(), N, K, True, 0
    )
    if rc != 0:
        raise RuntimeError(f"quantize_fp4_dynamic_sfa_fp16 (weight) failed rc={rc}")

    torch.cuda.synchronize(device)
    return {'packed': packed, 'sfb': sfb, 'N': N, 'K': K}


def quant_weight_nvfp4_inplace(w: torch.Tensor, w_quant: dict) -> None:
    """In-place version of quant_weight_nvfp4: writes into existing buffers.

    Used by AWQ requant to keep packed/sfb pointer addresses stable so the
    captured CUDA Graph can be replayed without recapture.

    Args:
        w: fp16 tensor [N, K] to quantize, CUDA, contiguous.
        w_quant: dict produced by quant_weight_nvfp4 (must match shape).
    """
    assert w.dtype == torch.float16
    assert w.device.type == 'cuda'
    assert w.is_contiguous()
    N, K = w.shape
    assert w_quant['N'] == N and w_quant['K'] == K, \
        f"shape mismatch: w=[{N},{K}] vs w_quant=[{w_quant['N']},{w_quant['K']}]"
    rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        w.data_ptr(), w_quant['packed'].data_ptr(), w_quant['sfb'].data_ptr(),
        N, K, True, 0
    )
    if rc != 0:
        raise RuntimeError(f"quantize_fp4_dynamic_sfa_fp16 (inplace) failed rc={rc}")
    torch.cuda.synchronize(w.device)


# ────────────────────────────────────────────────────────────────────
# Runtime activation scratch
# ────────────────────────────────────────────────────────────────────
class FP4ActScratch:
    """Preallocated buffers for per-call activation quantization.

    With F1 fusion, no linear_scales intermediate is needed — quant writes
    SFA tile-layout directly.
    """
    def __init__(self, max_M: int, K: int, device='cuda'):
        assert K % 16 == 0
        self.max_M = max_M
        self.K = K
        self.packed = torch.empty(max_M, K // 2, dtype=torch.uint8, device=device)
        sfa_bytes = fvk_fp4.sfa_size_bytes(max_M, K, False)
        # Zero-init: layout padding entries are never written by the
        # quantize kernels and must stay inert (see quant_weight_nvfp4).
        self.sfa = torch.zeros(sfa_bytes, dtype=torch.uint8, device=device)


def quant_act_nvfp4(x: torch.Tensor, scratch: FP4ActScratch,
                     M: int, stream: int = 0) -> None:
    """Quantize fp16 activation [M, K] → scratch.packed + scratch.sfa.

    Single-kernel fused path (F1). K is implicit (scratch.K).
    """
    assert x.dtype == torch.float16
    assert x.device.type == 'cuda'
    K = scratch.K
    assert M <= scratch.max_M, f"M={M} exceeds scratch.max_M={scratch.max_M}"

    rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        x.data_ptr(), scratch.packed.data_ptr(), scratch.sfa.data_ptr(),
        M, K, False, stream
    )
    if rc != 0:
        raise RuntimeError(f"quantize_fp4_dynamic_sfa_fp16 (act) failed rc={rc}")


# ────────────────────────────────────────────────────────────────────
# FP4 GEMM wrapper
# ────────────────────────────────────────────────────────────────────
class FP4Buffer:
    """Preallocated FP4 output buffer (packed [M, N/2] + SFA tile-interleaved).

    Used by the P1 split-GU FFN path for the gate_proj / up_proj fp4out
    GEMM outputs and the geglu_two_fp4_to_fp4 combiner output.
    """
    def __init__(self, M: int, N: int, device='cuda'):
        assert N % 16 == 0
        self.M = M
        self.N = N
        self.packed = torch.empty(M, N // 2, dtype=torch.uint8, device=device)
        sfa_bytes = fvk_fp4.sfa_size_bytes(M, N, False)
        # Zero-init: layout padding entries are never written by the fp4out
        # GEMM epilogue and must stay inert (see quant_weight_nvfp4).
        self.sfa = torch.zeros(sfa_bytes, dtype=torch.uint8, device=device)


def fp4out_gemm(scratch: FP4ActScratch, w_quant: dict,
                out_packed: int, out_sfd: int,
                M: int, N: int, K: int, stream: int = 0) -> None:
    """P1 NVFP4 GEMM with FP4-packed output + SFA.

    out_packed and out_sfd are device pointers (e.g. FP4Buffer.packed.data_ptr(),
    .sfa.data_ptr()). Used for gate_proj / up_proj in the split-GU FFN path.
    """
    assert w_quant['N'] == N and w_quant['K'] == K
    rc = fvk_fp4.cutlass_fp4_gemm_fp4out(
        scratch.packed.data_ptr(), scratch.sfa.data_ptr(),
        w_quant['packed'].data_ptr(), w_quant['sfb'].data_ptr(),
        out_packed, out_sfd,
        M, N, K, stream)
    if rc != 0:
        raise RuntimeError(f"cutlass_fp4_gemm_fp4out failed rc=0x{rc:x}")


def geglu_two_fp4_to_fp4(
        gate_packed: int, gate_sfa: int,
        up_packed: int,   up_sfa: int,
        out_packed: int,  out_sfa: int,
        S: int, H: int, stream: int = 0) -> None:
    """P1 combiner: GELU-tanh(gate) * up over two FP4 inputs → FP4 + SFA."""
    fvk_fp4.geglu_two_fp4_to_fp4(
        gate_packed, gate_sfa, up_packed, up_sfa, out_packed, out_sfa,
        S, H, stream)


def fp4_gemm(scratch: FP4ActScratch, w_quant: dict, out: torch.Tensor,
             M: int, N: int, K: int, variant_idx: int = -1,
             alpha: float = 1.0, beta: float = 0.0,
             stream: int = 0) -> None:
    """Run NVFP4 GEMM: out[M,N] (fp16) = A[M,K] (fp4) @ B[N,K]^T (fp4).

    A  = scratch.packed + scratch.sfa   (runtime-quantized activation)
    B  = w_quant['packed'] + w_quant['sfb']  (offline-quantized weight)
    """
    assert out.dtype == torch.float16
    assert out.device.type == 'cuda'
    assert out.numel() >= M * N, f"out too small ({out.numel()} < {M*N})"
    assert w_quant['N'] == N and w_quant['K'] == K, \
        f"weight shape mismatch: expected [{N},{K}] got [{w_quant['N']},{w_quant['K']}]"

    if variant_idx < 0:
        variant_idx = pick_variant(N, K)

    rc = fvk_fp4.cutlass_fp4_gemm_variant(
        variant_idx,
        scratch.packed.data_ptr(), scratch.sfa.data_ptr(),
        w_quant['packed'].data_ptr(), w_quant['sfb'].data_ptr(),
        out.data_ptr(),
        M, N, K, alpha, beta, stream
    )
    if rc != 0:
        raise RuntimeError(f"cutlass_fp4_gemm_variant(idx={variant_idx}) failed rc={rc}")

"""LingBot NVFP4 helper — thin wrapper over the pi05 FP4 kernels
(``flash_rt.flash_rt_fp4``, sm_110a block-scaled NVFP4 GEMM).

Mirrors ``flash_rt/executors/fp4_utils.py`` but binds the module name present in
this build (``flash_rt.flash_rt_fp4``). FP4 = W4A8-style NVFP4 (e2m1 4-bit
weights + acts, per-16-element UE4M3 block scales). fp16 I/O.

pi05's proven recipe: apply FP4 to the FFN (gate/up/down) only — those large-N
GEMMs get 1.6–1.9× (M≥768) / 1.34× (M=64 gate_up); QKV/attn/O stay FP8. Measured
FP4 GEMM cos ≈ 0.991/GEMM, so validate e2e cos≥0.99 before widening.
"""
from __future__ import annotations
import torch

try:
    import flash_rt.flash_rt_fp4 as _fp4
    HAS_FP4 = bool(_fp4.has_nvfp4())
except Exception:
    _fp4 = None
    HAS_FP4 = False


def pick_variant(N: int, K: int) -> int:
    """Shape→tile-variant default (calibrated by tests/fp4_vs_fp8_gemm.py at the
    LingBot shapes). The benchmark there sweeps all 10 variants per shape."""
    if N >= 8192:          # wide-N FFN gate/up (VLM 11008)
        return 4
    if K >= 8192:          # wide-K FFN down (VLM 11008)
        return 6
    if N >= 4096:          # mid FFN
        return 5
    return 1


def quant_weight(w_fp16: torch.Tensor) -> dict:
    """Offline: fp16 weight [N,K] → NVFP4 packed int4 + tile-interleaved SFB.
    K must be divisible by 16."""
    assert _fp4 is not None
    assert w_fp16.dtype == torch.float16 and w_fp16.is_contiguous()
    N, K = w_fp16.shape
    assert K % 16 == 0, f"K={K} must be %16 for NVFP4"
    packed = torch.empty(N, K // 2, dtype=torch.uint8, device=w_fp16.device)
    sfb = torch.empty(_fp4.sfa_size_bytes(N, K, True), dtype=torch.uint8, device=w_fp16.device)
    rc = _fp4.quantize_fp4_dynamic_sfa_fp16(
        w_fp16.data_ptr(), packed.data_ptr(), sfb.data_ptr(), N, K, True, 0)
    if rc != 0:
        raise RuntimeError(f"quant_weight fp4 rc={rc}")
    torch.cuda.synchronize(w_fp16.device)
    return {"packed": packed, "sfb": sfb, "N": N, "K": K}


class FP4ActScratch:
    """Preallocated per-call activation FP4 buffers (packed + SFA)."""
    def __init__(self, max_M: int, K: int, device="cuda"):
        assert K % 16 == 0
        self.max_M, self.K = max_M, K
        self.packed = torch.empty(max_M, K // 2, dtype=torch.uint8, device=device)
        self.sfa = torch.empty(_fp4.sfa_size_bytes(max_M, K, False), dtype=torch.uint8, device=device)


def quant_act(x_fp16: torch.Tensor, scratch: "FP4ActScratch", M: int, stream: int = 0) -> None:
    """Runtime: fp16 act [M,K] → scratch.packed + scratch.sfa (fused quant+layout)."""
    rc = _fp4.quantize_fp4_dynamic_sfa_fp16(
        x_fp16.data_ptr(), scratch.packed.data_ptr(), scratch.sfa.data_ptr(),
        M, scratch.K, False, stream)
    if rc != 0:
        raise RuntimeError(f"quant_act fp4 rc={rc}")


def fp4_gemm(scratch: "FP4ActScratch", w_quant: dict, out_fp16: torch.Tensor,
             M: int, N: int, K: int, variant: int = -1, stream: int = 0) -> None:
    """out[M,N] fp16 = A[M,K] (fp4) @ B[N,K]^T (fp4). out must be fp16."""
    if variant < 0:
        variant = pick_variant(N, K)
    rc = _fp4.cutlass_fp4_gemm_variant(
        variant, scratch.packed.data_ptr(), scratch.sfa.data_ptr(),
        w_quant["packed"].data_ptr(), w_quant["sfb"].data_ptr(),
        out_fp16.data_ptr(), M, N, K, 1.0, 0.0, stream)
    if rc != 0:
        raise RuntimeError(f"fp4_gemm rc=0x{rc:x}")


def rms_norm_to_fp4(x_fp16: torch.Tensor, scratch: "FP4ActScratch",
                    seq_len: int, stream: int = 0) -> None:
    """Fused (noweight) RMSNorm of fp16 [seq,dim] → FP4 packed+SFA in scratch.
    The RMS weight is folded into the downstream GEMM weight (see
    prepare_vlm_fp4_weights), so this norm is weightless."""
    _fp4.rms_norm_fp4_sfa_fp16(
        x_fp16.data_ptr(), scratch.packed.data_ptr(), scratch.sfa.data_ptr(),
        seq_len, scratch.K, stream)

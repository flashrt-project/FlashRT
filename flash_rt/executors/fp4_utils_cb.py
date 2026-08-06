"""NVFP4 utilities over CudaBuffer (torch-free variant of fp4_utils.py).

Mirror of the torch-based helpers in ``fp4_utils.py``, but every allocation
and pointer is expressed through :class:`flash_rt.core.cuda_buffer.CudaBuffer`
so the JAX Pi0.5 FP4 frontend can consume the same ``flash_rt_fp4`` kernels
without pulling torch into the runtime path.

Symmetry with the torch side:
    torch path                           CudaBuffer path
    -------------------------------      -------------------------------
    quant_weight_nvfp4(torch.Tensor)  →  quant_weight_nvfp4_from_cb(CudaBuffer, N, K)
    quant_weight_nvfp4_inplace        →  quant_weight_nvfp4_inplace_from_cb
    FP4ActScratch(max_M, K, device)   →  FP4ActScratchCB(max_M, K)
    FP4Buffer(M, N, device)           →  FP4BufferCB(M, N)
    pick_variant(N, K)                →  pick_variant(N, K)    (pure Python, duplicated)

The duplicated ``pick_variant`` is an intentional choice to keep this module
torch-free. Any future consolidation would live in a separate torch-neutral
``fp4_variants.py`` module.
"""
from __future__ import annotations

import numpy as np

import flash_rt.flash_rt_fp4 as fvk_fp4
from flash_rt.core.cuda_buffer import CudaBuffer, sync


# ────────────────────────────────────────────────────────────────────
# Variant selection — duplicated from fp4_utils.py to keep this module
# torch-free. Keep in sync with fp4_utils.pick_variant.
# ────────────────────────────────────────────────────────────────────
VARIANT_V1_C2X1X1    = 1
VARIANT_V4_SMALL_N   = 4
VARIANT_V6_WIDE_N    = 6
VARIANT_V7_WIDE_K    = 7
VARIANT_V8_WIDE_NK   = 8


def pick_variant(N: int, K: int) -> int:
    """Shape-based NVFP4 GEMM variant. Bit-identical to
    ``flash_rt.executors.fp4_utils.pick_variant``.
    """
    if N >= 16384:
        return VARIANT_V8_WIDE_NK
    if K >= 8192:
        return VARIANT_V1_C2X1X1
    return VARIANT_V6_WIDE_N


# ────────────────────────────────────────────────────────────────────
# Pointer shim — exposes ``.data_ptr()`` over a CudaBuffer so the
# shared_primitives_fp4 encoder function (written for torch tensors that
# natively carry ``.data_ptr()``) works unchanged.
# ────────────────────────────────────────────────────────────────────
class _PtrShim:
    """Duck-type adapter: ``.data_ptr()`` over a :class:`CudaBuffer`.

    Retains a reference to the underlying CudaBuffer so it stays alive
    for the shim's lifetime. ``data_ptr()`` returns a cached int.
    """
    __slots__ = ("_cb", "_ptr")

    def __init__(self, cb: CudaBuffer):
        self._cb = cb
        self._ptr = int(cb.ptr.value)

    def data_ptr(self) -> int:
        return self._ptr


# ────────────────────────────────────────────────────────────────────
# Offline weight quantization (CudaBuffer → CudaBuffer)
# ────────────────────────────────────────────────────────────────────
def quant_weight_nvfp4_from_cb(w_cb: CudaBuffer, N: int, K: int) -> dict:
    """Quantize fp16 weight [N, K] held in ``w_cb`` → NVFP4 (packed int4 + SFB).

    Uses the fused F1 kernel (single launch, no linear-scales intermediate).
    Synchronizes after launch so callers can release ``w_cb`` immediately.

    Returns
    -------
    dict with keys:
        'packed_cb' : CudaBuffer uint8 [N * K/2]    (NVFP4 e2m1 elements)
        'sfb_cb'    : CudaBuffer uint8 [sfb_bytes]  (CUTLASS tile-interleaved UE4M3 scales)
        'packed_ptr': int                           cached int(packed_cb.ptr.value)
        'sfb_ptr'   : int                           cached int(sfb_cb.ptr.value)
        'N', 'K'    : ints
    """
    assert K % 16 == 0, f"K={K} must be divisible by 16 for NVFP4 block=16"

    packed_cb = CudaBuffer.device_empty(N * (K // 2), np.uint8)
    sfb_bytes = fvk_fp4.sfa_size_bytes(N, K, True)
    # Zero-init: SF layout padding entries are never written by the
    # quantize kernels and must stay inert (see fp4_utils.quant_weight_nvfp4).
    sfb_cb    = CudaBuffer.device_zeros(sfb_bytes, np.uint8)

    rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        int(w_cb.ptr.value),
        int(packed_cb.ptr.value),
        int(sfb_cb.ptr.value),
        N, K, True, 0,
    )
    if rc != 0:
        raise RuntimeError(f"quantize_fp4_dynamic_sfa_fp16 (weight) failed rc={rc}")

    sync()
    return {
        'packed_cb':  packed_cb,
        'sfb_cb':     sfb_cb,
        'packed_ptr': int(packed_cb.ptr.value),
        'sfb_ptr':    int(sfb_cb.ptr.value),
        'N': N, 'K': K,
    }


def quant_weight_nvfp4_inplace_from_cb(w_cb: CudaBuffer, w_quant_cb: dict) -> None:
    """In-place re-quantize: rewrites ``w_quant_cb['packed_cb']`` and
    ``w_quant_cb['sfb_cb']`` from fresh fp16 data in ``w_cb``. Pointer
    addresses stay stable so a captured CUDA Graph that references this
    ``w_quant_cb`` replays correctly.

    Used by the AWQ refit path.
    """
    N = int(w_quant_cb['N'])
    K = int(w_quant_cb['K'])
    rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        int(w_cb.ptr.value),
        int(w_quant_cb['packed_cb'].ptr.value),
        int(w_quant_cb['sfb_cb'].ptr.value),
        N, K, True, 0,
    )
    if rc != 0:
        raise RuntimeError(f"quantize_fp4_dynamic_sfa_fp16 (inplace) failed rc={rc}")
    sync()


# ────────────────────────────────────────────────────────────────────
# Runtime activation scratch (CudaBuffer-backed FP4ActScratch analogue)
# ────────────────────────────────────────────────────────────────────
class FP4ActScratchCB:
    """CudaBuffer analogue of ``fp4_utils.FP4ActScratch``.

    Attributes shaped to match the torch :class:`FP4ActScratch` API so
    :func:`shared_primitives_fp4.encoder_forward_with_fp4_subset` works
    unchanged: ``.packed`` and ``.sfa`` expose ``.data_ptr()`` via
    :class:`_PtrShim`. ``.packed_cb`` / ``.sfa_cb`` retain direct
    :class:`CudaBuffer` access for refits or debugging.
    """

    __slots__ = ("max_M", "K", "packed_cb", "sfa_cb",
                 "packed", "sfa", "packed_ptr", "sfa_ptr")

    def __init__(self, max_M: int, K: int):
        assert K % 16 == 0, f"K={K} must be divisible by 16"
        self.max_M = int(max_M)
        self.K = int(K)
        self.packed_cb = CudaBuffer.device_empty(max_M * (K // 2), np.uint8)
        sfa_bytes = fvk_fp4.sfa_size_bytes(max_M, K, False)
        # Zero-init: SF layout padding must stay inert (see fp4_utils).
        self.sfa_cb = CudaBuffer.device_zeros(sfa_bytes, np.uint8)
        self.packed = _PtrShim(self.packed_cb)
        self.sfa    = _PtrShim(self.sfa_cb)
        self.packed_ptr = int(self.packed_cb.ptr.value)
        self.sfa_ptr    = int(self.sfa_cb.ptr.value)


def quant_act_nvfp4_cb(x_ptr: int, scratch: FP4ActScratchCB,
                       M: int, stream: int = 0) -> None:
    """Quantize fp16 activation [M, K] at device pointer ``x_ptr`` into
    ``scratch`` buffers. F1 fused path.
    """
    assert M <= scratch.max_M, f"M={M} exceeds scratch.max_M={scratch.max_M}"
    rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16(
        int(x_ptr),
        scratch.packed_ptr,
        scratch.sfa_ptr,
        int(M), scratch.K, False, int(stream),
    )
    if rc != 0:
        raise RuntimeError(f"quantize_fp4_dynamic_sfa_fp16 (act) failed rc={rc}")


# ────────────────────────────────────────────────────────────────────
# FP4Buffer (P1 split-GU fp4out intermediate; CudaBuffer-backed)
# ────────────────────────────────────────────────────────────────────
class FP4BufferCB:
    """CudaBuffer analogue of ``fp4_utils.FP4Buffer`` — packed FP4 output +
    SFA for the P1 split-GU gate_proj / up_proj fp4out GEMMs. Shape
    mirrors :class:`FP4ActScratchCB`.
    """

    __slots__ = ("M", "N", "packed_cb", "sfa_cb",
                 "packed", "sfa", "packed_ptr", "sfa_ptr")

    def __init__(self, M: int, N: int):
        assert N % 16 == 0, f"N={N} must be divisible by 16"
        self.M = int(M)
        self.N = int(N)
        self.packed_cb = CudaBuffer.device_empty(M * (N // 2), np.uint8)
        sfa_bytes = fvk_fp4.sfa_size_bytes(M, N, False)
        # Zero-init: SF layout padding must stay inert (see fp4_utils).
        self.sfa_cb = CudaBuffer.device_zeros(sfa_bytes, np.uint8)
        self.packed = _PtrShim(self.packed_cb)
        self.sfa    = _PtrShim(self.sfa_cb)
        self.packed_ptr = int(self.packed_cb.ptr.value)
        self.sfa_ptr    = int(self.sfa_cb.ptr.value)

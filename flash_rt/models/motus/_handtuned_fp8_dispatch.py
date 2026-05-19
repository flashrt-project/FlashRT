"""Hand-tuned FP8 GEMM dispatcher for motus action/und shapes.

Routes calls to ``gemm.fp8_nn_dev`` to per-shape hand-tuned ``ht_*`` bindings
(`fp8_smallM_handtuned*`, `fp8_smallM_handtuned_ldmatrix*`,
`fp8_smallM_handtuned_splitk*` kernels) when (N, K) matches a known motus
shape, falling back to cuBLASLt for unknown shapes.

Gated by env ``MOTUS_HANDTUNED_FP8=1``. Must be installed AFTER FP8
calibration so per-site act_scale / w_scale are finalized.

Saves ~9.00 ms / inference on motus Stage3 baseline (per-shape bench).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)


# (N, K) → dispatch entry.
# Entry format: ('simple', ht_name) or ('splitk', ht_name, k_split).
#
# CRITICAL: only shapes VERIFIED via real-shape bench (M=21/M=188, under
# CUDA graph, 3-run median, cos > 0.999) to BEAT the alternative (tinyfp8
# or cuBLASLt). Adding a shape that doesn't actually win is a regression.
#
# Verified 2026-05-16 (real-shape bench, motus_dev/megakernel_spike/
# bench_winner_vs_cublaslt.py):
#   action_o (M=21, N=1024, K=3072): ht_splitk_32x64x128_w4 = 4.10us
#                                    vs tinyfp8 = 8.19us  → save 4.09us/call
#   und_o    (M=188, N=512, K=3072): ht_ld_16x64x256_w4_s3 = 6.15us
#                                    vs tinyfp8 = 8.19us  → save 2.04us/call
# Not added (ht_ loses):
#   action_qkv (9216, 1024): tinyfp8 4.10us wins (ht best 6.15us)
#   und_qkv    (9216,  512): cuBLASLt 8.53us wins (ht best 12.30us)
# Removed (these previous entries were WRONG — bench at fake M=8/M=138):
#   (9216, 1024), (9216, 512), (4096, 1024), (1024, 4096),
#   (2048, 512), (512, 2048)
SHAPE_TO_KERNEL = {
    (1024, 3072): ('splitk', 'ht_splitk_fp8_gemm_32x64x128_w4', 8),  # action_o
    ( 512, 3072): ('simple', 'ht_ld_fp8_gemm_16x64x256_w4_s3'),      # und_o
}


def _collect_sites(model):
    """Walk modules to gather FP8 sites with (N, K, act_scale, w_scale).

    Sources:
      - ``_g724_state``  on action/und transformer blocks (QKV joint)
      - ``_awq_fp8_site`` on wan_action_o / wan_und_o Linears (O-proj)
    Old attribute names (``_fp8_site`` / ``_fp8_up_site`` / ``_fp8_down_site``)
    never existed in the runtime model — they were a guess from an earlier
    iteration and caused this dispatcher to silently no-op for months.
    """
    sites = []
    for _, module in model.named_modules():
        for attr in ('_g724_state', '_awq_fp8_site'):
            s = getattr(module, attr, None)
            if s is None:
                continue
            # Only ready sites carry concrete scales we can read.
            mode = getattr(s, 'mode', None)
            if mode is not None and mode != 'fp8_ready':
                continue
            if getattr(s, 'act_scale', None) is None or getattr(s, 'w_scale', None) is None:
                continue
            sites.append(s)
    return sites


# Cache alpha by (act_scale_ptr, w_scale_ptr) so each per-layer scale pair
# is resolved once via a D2H readback. Without this we'd either burn a
# D2H per call (slow) or use a stale snapshot from install-time (wrong:
# 30 layers each have their OWN act_scale/w_scale tensor with different
# values — using layer-0's alpha for all 30 destroys cos to ~0.69).
_ALPHA_CACHE_PTR: dict[tuple[int, int], float] = {}
_CUDART = None


def _read_fp32_dev(ptr: int) -> float:
    import ctypes
    global _CUDART
    if _CUDART is None:
        _CUDART = ctypes.CDLL('libcudart.so')
        _CUDART.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _CUDART.cudaMemcpy.restype = ctypes.c_int
    buf = (ctypes.c_float * 1)()
    _CUDART.cudaMemcpy(buf, ctypes.c_void_p(ptr), 4, 2)  # D2H = 2
    return float(buf[0])


def _alpha_from_ptrs(a_ptr: int, w_ptr: int) -> float:
    key = (a_ptr, w_ptr)
    cached = _ALPHA_CACHE_PTR.get(key)
    if cached is not None:
        return cached
    a = _read_fp32_dev(a_ptr)
    w = _read_fp32_dev(w_ptr)
    alpha = a * w
    _ALPHA_CACHE_PTR[key] = alpha
    return alpha


def install_handtuned_fp8_dispatch(
    model,
    gemm,
    env_var: str = 'MOTUS_HANDTUNED_FP8',
) -> dict:
    """Wrap ``gemm.fp8_nn_dev`` to dispatch motus shapes to hand-tuned kernels.

    Returns a stats dict. No-op if env disabled or no FP8 sites are present.
    Must be called AFTER calibration so per-site scales are final.
    """
    if os.environ.get(env_var, '0') not in ('1', 'true', 'on', 'yes'):
        logger.info(
            f"[handtuned-fp8] disabled (set {env_var}=1 to enable)")
        return {'enabled': False, 'reason': 'disabled_by_env'}

    sites = _collect_sites(model)
    if not sites:
        return {'enabled': False, 'reason': 'no_sites'}

    # We need to know that AT LEAST ONE site exists per shape so we know
    # the shape is live in this model. Per-call alpha is read live from
    # (a_ptr, w_ptr) at dispatch time (cached) — 30 layers each have
    # their own act_scale/w_scale, snapshot-at-install is WRONG.
    shape_seen: set[tuple[int, int]] = set()
    for s in sites:
        key = (int(s.N), int(s.K))
        if key in SHAPE_TO_KERNEL:
            shape_seen.add(key)

    # Pre-build w_NK transpose for every site whose (N, K) is in our
    # SHAPE_TO_KERNEL table. motus stores w_fp8 as (K, N) KN-major; ht_*
    # kernels expect (N, K) NK-major. We cache transpose by orig KN ptr
    # so the runtime dispatch is a free dict lookup.
    nk_cache: dict[int, torch.Tensor] = {}
    for s in sites:
        key = (int(s.N), int(s.K))
        if key not in SHAPE_TO_KERNEL:
            continue
        w_kn = getattr(s, 'w_fp8', None)
        if w_kn is None:
            continue
        orig_ptr = int(w_kn.data_ptr())
        if orig_ptr in nk_cache:
            continue
        nk_cache[orig_ptr] = w_kn.t().contiguous()

    resolved: dict[tuple[int, int], tuple] = {}
    scratch_buffers: list[torch.Tensor] = []

    M_MAX = 256  # safe upper bound: max(action M=21, und M=188, headroom)

    for key, entry in SHAPE_TO_KERNEL.items():
        if key not in shape_seen:
            continue  # shape not present in this particular model
        kind = entry[0]
        ht_name = entry[1]
        fn = getattr(fvk, ht_name, None)
        if fn is None:
            logger.warning(
                f"[handtuned-fp8] binding missing: {ht_name}; "
                f"shape (N={key[0]}, K={key[1]}) falls back to cuBLASLt")
            continue
        if kind == 'simple':
            resolved[key] = ('simple', fn)
        elif kind == 'splitk':
            k_split = int(entry[2])
            N = key[0]
            n_scratch_fp32 = M_MAX * N * k_split
            scratch = torch.empty(
                n_scratch_fp32, dtype=torch.float32, device='cuda')
            scratch_buffers.append(scratch)
            resolved[key] = ('splitk', fn, k_split, scratch)

    if not resolved:
        return {'enabled': False, 'reason': 'no_kernels_resolved'}

    if getattr(fvk.GemmRunner, '_handtuned_class_patched', False):
        logger.info("[handtuned-fp8] already installed; skip")
        return {'enabled': True, 'reason': 'already_installed',
                'resolved_shapes': list(resolved.keys())}

    # pybind11 GemmRunner forbids instance-attr assignment. Patch at the
    # CLASS level (same mechanism tinyfp8 uses). When tinyfp8 has already
    # installed, its router becomes our previous_fn and serves as the
    # fallback for shapes we don't override.
    previous_fn = fvk.GemmRunner.fp8_nn_dev  # tinyfp8 router (or original)

    def dispatched(self, A_ptr, B_ptr, D_ptr, M, N, K, a_s_ptr, w_s_ptr, stream):
        entry = resolved.get((int(N), int(K)))
        if entry is None:
            return previous_fn(
                self, A_ptr, B_ptr, D_ptr, M, N, K, a_s_ptr, w_s_ptr, stream)
        # Per-call alpha lookup — per-layer scale tensors differ.
        alpha = _alpha_from_ptrs(int(a_s_ptr), int(w_s_ptr))
        # Translate KN weight ptr → NK weight ptr (transpose cache).
        b_nk = nk_cache.get(int(B_ptr))
        if b_nk is None:
            # New weight ptr we didn't pre-register → fall through to
            # previous_fn (safer than passing wrong-layout bytes).
            return previous_fn(
                self, A_ptr, B_ptr, D_ptr, M, N, K, a_s_ptr, w_s_ptr, stream)
        b_nk_ptr = int(b_nk.data_ptr())
        kind = entry[0]
        if kind == 'simple':
            _, fn = entry
            return fn(A_ptr, b_nk_ptr, D_ptr, M, N, K, alpha, stream)
        # splitk
        _, fn, k_split, scratch = entry
        return fn(A_ptr, b_nk_ptr, D_ptr, M, N, K, k_split, alpha,
                  int(scratch.data_ptr()), stream)

    fvk.GemmRunner.fp8_nn_dev = dispatched
    fvk.GemmRunner._handtuned_class_patched = True
    fvk.GemmRunner._handtuned_previous_fn = previous_fn
    fvk.GemmRunner._handtuned_scratch_keepalive = scratch_buffers
    fvk.GemmRunner._handtuned_nk_cache_keepalive = nk_cache

    logger.info(
        f"[handtuned-fp8] class-patched, dispatched {len(resolved)} shape(s); "
        f"shapes={sorted(resolved.keys())}, nk_cache={len(nk_cache)} weights")
    return {
        'enabled': True,
        'resolved_shapes': sorted(resolved.keys()),
        'nk_cache_count': len(nk_cache),
    }

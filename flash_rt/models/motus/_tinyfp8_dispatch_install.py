"""Install tiny_fp8 small-shape FP8 GEMM dispatch.

Replaces ``fvk.GemmRunner.fp8_nn_dev`` with a router that picks one of
five 2-stage hand-tuned tile variants (built into flash_rt_kernels via
csrc/kernels/megakernel/tinyfp8_kernels_sm120.cu) for the action /
und FFN / projection shapes where cuBLASLt heuristic underperforms.
Any shape not in the dispatch table falls back to the original
``fp8_nn_dev``.

Custom kernels expect B in (N, K) row-major; motus stores w_fp8 in
(K, N), so weights are pre-transposed at install time and cached by
their original ``data_ptr``.

Env-disable: ``FLASH_RT_MOTUS_USE_TINYFP8_DISPATCH=0``.
"""
from __future__ import annotations

import ctypes
import os
from typing import Any

import torch


_CUDART = None


def _read_fp32_dev(ptr: int) -> float:
    global _CUDART
    if _CUDART is None:
        _CUDART = ctypes.CDLL('libcudart.so')
        _CUDART.cudaMemcpy.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _CUDART.cudaMemcpy.restype = ctypes.c_int
    buf = (ctypes.c_float * 1)()
    _CUDART.cudaMemcpy(buf, ctypes.c_void_p(ptr), 4, 2)  # D2H = 2
    return float(buf[0])


# motus call signature: fp8_nn_dev(A, B, D, M, N, K, ...). Table key
# = (M, N, K) as passed by motus.
_TABLE: dict[tuple[int, int, int], str] = {
    (8,   4096, 1024): 'tinyfp8_gemm_M8_N32_K512_sm120',   # action FFN_up
    (8,   1024, 4096): 'tinyfp8_gemm_M8_N32_K256_sm120',   # action FFN_dn
    (8,   9216, 1024): 'tinyfp8_gemm_M8_N32_K128_sm120',   # action QKV joint
    (8,   1024, 3072): 'tinyfp8_gemm_M8_N32_K512_sm120',   # action wan_o
    (21,  9216, 1024): 'tinyfp8_gemm_M32_N32_K128_sm120',  # stage3 action QKV
    (21,  1024, 3072): 'tinyfp8_gemm_M32_N32_K512_sm120',  # stage3 action wan_o
    (188, 512,  3072): 'tinyfp8_gemm3_M16_N64_K128_sm120', # stage3 und O
    (138, 2048, 512):  'tinyfp8_gemm_M16_N32_K64_sm120',   # und FFN_up
    (138, 512,  2048): 'tinyfp8_gemm_M16_N64_K64_sm120',   # und FFN_dn
    (138, 512,  3072): 'tinyfp8_gemm_M16_N32_K64_sm120',   # und O
}

_NK_TABLE: set[tuple[int, int]] = {(N, K) for _M, N, K in _TABLE}

_FNS: dict[tuple[int, int, int], Any] = {}
_B_NK_CACHE: dict[int, tuple[torch.Tensor, int, int]] = {}
_ALPHA_CACHE: dict[tuple[int, int], float] = {}

_ORIG_FP8_NN_DEV = None
_INSTALLED = False
_STATS = {'custom': 0, 'fallback': 0, 'miss_cache': 0}


def _get_alpha(act_scale_ptr: int, w_scale_ptr: int) -> float:
    key = (act_scale_ptr, w_scale_ptr)
    cached = _ALPHA_CACHE.get(key)
    if cached is not None:
        return cached
    a = _read_fp32_dev(act_scale_ptr)
    w = _read_fp32_dev(w_scale_ptr)
    alpha = a * w
    _ALPHA_CACHE[key] = alpha
    return alpha


def _register_weight(w_fp8_KN: torch.Tensor) -> None:
    orig_ptr = int(w_fp8_KN.data_ptr())
    if orig_ptr in _B_NK_CACHE:
        return
    w_NK = w_fp8_KN.t().contiguous()
    K, N = w_fp8_KN.shape
    _B_NK_CACHE[orig_ptr] = (w_NK, K, N)


def _register_model(model: Any) -> int:
    n = 0
    seen: set[int] = set()
    for _name, mod in model.named_modules():
        for attr_name in dir(mod):
            try:
                site = getattr(mod, attr_name, None)
            except Exception:
                continue
            if site is None or not hasattr(site, 'w_fp8'):
                continue
            w = getattr(site, 'w_fp8', None)
            if not torch.is_tensor(w) or w.numel() == 0:
                continue
            if w.ndim != 2:
                continue
            K, N = int(w.shape[0]), int(w.shape[1])
            if (N, K) not in _NK_TABLE:
                continue
            ptr = int(w.data_ptr())
            if ptr in seen:
                continue
            seen.add(ptr)
            _register_weight(w)
            n += 1
    return n


def install(model: Any = None) -> int:
    """Monkey-patch fvk.GemmRunner.fp8_nn_dev with the tinyfp8 router.

    Returns number of weights pre-registered.
    """
    if os.environ.get('FLASH_RT_MOTUS_USE_TINYFP8_DISPATCH', '1') == '0':
        return 0
    try:
        import flash_rt.flash_rt_kernels as fvk
    except Exception:
        return 0
    # Resolve symbol handles
    for shape, sym in _TABLE.items():
        fn = getattr(fvk, sym, None)
        if fn is None:
            # Required symbol missing — skip install entirely (kernel not built)
            return 0
        _FNS[shape] = fn

    global _ORIG_FP8_NN_DEV, _INSTALLED
    n_reg = 0
    if model is not None:
        n_reg = _register_model(model)

    if not _INSTALLED:
        _ORIG_FP8_NN_DEV = fvk.GemmRunner.fp8_nn_dev
        fvk.GemmRunner.fp8_nn_dev = _custom_fp8_nn_dev
        _INSTALLED = True
    return n_reg


def _custom_fp8_nn_dev(self, A, B, D, M, N, K, act_scale, w_scale, stream):
    """Drop-in for fvk.GemmRunner.fp8_nn_dev. B is the (K, N) ptr from
    motus; we look up its pre-transposed (N, K) replacement.
    """
    fn = _FNS.get((M, N, K))
    if fn is None:
        _STATS['fallback'] += 1
        _ORIG_FP8_NN_DEV(self, A, B, D, M, N, K, act_scale, w_scale, stream)
        return
    entry = _B_NK_CACHE.get(B)
    if entry is None:
        _STATS['miss_cache'] += 1
        _ORIG_FP8_NN_DEV(self, A, B, D, M, N, K, act_scale, w_scale, stream)
        return
    w_NK, _K, _N = entry
    alpha = _get_alpha(act_scale, w_scale)
    rc = fn(A, int(w_NK.data_ptr()), D, M, N, K, alpha, stream)
    if rc != 0:
        raise RuntimeError(
            f'tinyfp8 (M={M} N={N} K={K}) rc={rc}; fallback impossible')
    _STATS['custom'] += 1


def get_stats() -> dict[str, int]:
    return dict(_STATS)

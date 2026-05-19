"""Phase 8 — FP4 swap for motus VAE conv3d sites (additive over _vae_fp8_swap).

Production result (motus_quickstart, baseline 207.975 ms):
  - Default       (FLASH_RT_MOTUS_USE_FP4_VAE=1):                202.77 ms
                                                          (-5.2 ms, cos 0.9981)
  - Aggressive    (+FLASH_RT_MOTUS_VAE_FP4_AGGRESSIVE_CACHE=1):  199.12 ms
                                                          (-8.85 ms, cos 0.9971 at floor)

DEPLOYMENT REQUIREMENT — the FP4 path expects the Motus VAE FP4 kernels to
be built into ``flash_rt.flash_rt_kernels``. Public builds must not depend on
external Motus kernel ``.so`` files.

Architecture:
  1. install_vae_fp4(model) — runs AFTER `_vae_fp8_swap.install_vae_fp8()`:
       For each eligible FP8 site, also quantize its weight bf16 → NVFP4
       packed + UE4M3 linear SF (via fvk.quantize_bf16_to_nvfp4).
  2. Monkey-patches `_vae_fp8_swap._fused_step` with `_router_fused_step`:
       - If site has FP4-quantized weight AND env enabled:
           → call `_fused_step_fp4` (uses v19sf + fused activation quant)
       - Else: call original FP8 `_fused_step` verbatim.
  3. _fused_step_fp4 paths (see code for details):
       - Always: bf16 → FP4+SF via fvk Motus quant kernel
       - Default cache: FP8 stored in feat_cache, requant to FP4 each call
       - Aggressive cache: FP4 cache via _FP4_CACHE_BY_WID + copy_()
         in-place updates (graph-safe; see feedback_graph_capture_dict_cache.md)

Env knobs:
  FLASH_RT_MOTUS_USE_FP4_VAE=1               enable the swap (default off)
  FLASH_RT_MOTUS_VAE_FP4_AGGRESSIVE_CACHE=1  opt-in FP4 cache
  FLASH_RT_MOTUS_VAE_FP4_MAX_CI=N            cap Ci tier (default 640)
  FLASH_RT_MOTUS_VAE_FP4_TRACE=1             per-call routing log
  FLASH_RT_MOTUS_VAE_FP4_OUTER_FP32=1        Phase 10: per-Co outer FP32
                                             (v19sf_v2 kernel; absorbs per-Co
                                             weight variance, leaves inner
                                             FP4+SF for normalized [-6,6]).
                                             Single-conv real-VAE Δcos +0.002.
                                             Tradeoff: FP8 (USE_FP4_VAE=0) is
                                             always available as the safe
                                             fallback for accuracy-sensitive
                                             deployments.
"""
from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional

import torch

import flash_rt.flash_rt_kernels as fvk
from . import _vae_fp8_swap

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_TRACE = os.environ.get('FLASH_RT_MOTUS_VAE_FP4_TRACE', '0') == '1'

_v19sf_fn = None
_v19sf_fn_v2 = None      # Phase 10: outer FP32 per-Co
_v19sfb_fn = None
_v19sfb_fn_v2 = None     # Phase 10
# BLOCK_K=128 variant: 1.21x avg on production hot shapes vs ship K=64.
# Requires Ci >= 128 (one K-tile must stay within a single 3x3x3 tap).
_v19sfb_k128_fn = None
_v4q_fn = None


def _ptr(v):
    if isinstance(v, ctypes.c_void_p):
        return int(v.value or 0)
    return int(v)


def _flt(v):
    if isinstance(v, ctypes.c_float):
        return float(v.value)
    return float(v)


def _wrap_status(fn, ptr_count: int, int_count: int,
                 has_float: bool = True):
    def wrapped(*args):
        ptrs = [_ptr(x) for x in args[:ptr_count]]
        ints = [int(x) for x in args[ptr_count:ptr_count + int_count]]
        rest = args[ptr_count + int_count:]
        if has_float:
            alpha_or_eps = _flt(rest[0])
            stream = _ptr(rest[1])
            return int(fn(*ptrs, *ints, alpha_or_eps, stream))
        stream = _ptr(rest[0])
        return int(fn(*ptrs, *ints, stream))
    return wrapped


def _load_libs():
    global _v19sf_fn, _v19sf_fn_v2
    global _v19sfb_fn, _v19sfb_fn_v2
    global _v19sfb_k128_fn
    global _v4q_fn
    if _v19sf_fn is not None:
        return True
    required = {
        'motus_fp4_conv3d_v19sf_ndhwc_bf16out': (8, 7),
        'motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2': (9, 7),
        'motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out': (9, 7),
        'motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2': (10, 7),
        'motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1': (5, 5),
    }
    missing = [name for name in required if not hasattr(fvk, name)]
    if missing:
        logger.warning(
            '[fp4-vae] Motus VAE FP4 kernels are not built into '
            'flash_rt_kernels: %s', ', '.join(missing))
        return False
    _v19sf_fn = _wrap_status(
        getattr(fvk, 'motus_fp4_conv3d_v19sf_ndhwc_bf16out'), 8, 7)
    _v19sf_fn_v2 = _wrap_status(
        getattr(fvk, 'motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2'), 9, 7)
    _v19sfb_fn = _wrap_status(
        getattr(fvk, 'motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out'), 9, 7)
    _v19sfb_fn_v2 = _wrap_status(
        getattr(fvk, 'motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out_v2'), 10, 7)
    if hasattr(fvk, 'motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out'):
        _v19sfb_k128_fn = _wrap_status(
            getattr(fvk, 'motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out'),
            9, 7)
    else:
        _v19sfb_k128_fn = None
        logger.info('[fp4-vae] Motus K128 FP4 VAE kernel not built')
    _v4q_fn = _wrap_status(
        getattr(fvk, 'motus_bf16_rms_silu_quant_nvfp4_to_ndhwc_v1'), 5, 5)
    return True


# ── FP4 weight cache: keyed by id(site.w_fp8_NDHWC) ──

class _Fp4Weight:
    __slots__ = ('w_fp4', 'w_sf', 'Co', 'Ci', 'awq_inv_scale', 'outer_w')

    def __init__(self, w_fp4: torch.Tensor, w_sf: torch.Tensor,
                 Co: int, Ci: int):
        self.w_fp4 = w_fp4   # [Co, 3, 3, 3, Ci/2] uint8 cuda
        self.w_sf = w_sf     # [Co, 3, 3, 3, Ci/16] uint8 cuda
        self.Co = Co
        self.Ci = Ci
        self.awq_inv_scale = None   # [Ci] float32 cuda, or None
        self.outer_w = None         # Phase 10: [Co] float32 cuda, or None
                                    # When set, _fused_step_fp4 routes to v2.


_FP4_WEIGHT_BY_ID: dict[int, _Fp4Weight] = {}
# Phase 8.4: FP4 cache to bypass patched_forward's FP8 cache contract.
# Keyed by id(site.w_fp8_NDHWC); value = (cache_fp4, cache_sf) on cuda.
# Under cuda graph capture, the same memory slots are reused per replay
# (torch's graph mempool keeps allocations stable across replays).
_FP4_CACHE_BY_WID: dict = {}


def _quant_weight_to_fp4(w_bf16_NDHWC: torch.Tensor,
                         outer_fp32: bool = False) -> Optional[_Fp4Weight]:
    """w: [Co, kT, kH, kW, Ci] bf16 cuda. Returns _Fp4Weight or None.

    outer_fp32=True (Phase 10): compute per-Co outer = max(|w[co,...]|)/6.0,
    divide weight, quantize inner. outer_w stored on the returned _Fp4Weight
    for runtime epilogue multiply via v19sf_v2.
    """
    Co, kT, kH, kW, Ci = w_bf16_NDHWC.shape
    if Ci % 64 != 0 or Co % 8 != 0:
        return None
    if outer_fp32:
        w_f32 = w_bf16_NDHWC.float()
        outer_w = (w_f32.abs().amax(dim=(1, 2, 3, 4)) / 6.0).clamp(min=1e-12)
        w_inner = (w_f32 / outer_w.view(Co, 1, 1, 1, 1)).to(torch.bfloat16)
        w_to_quant = w_inner.contiguous()
    else:
        outer_w = None
        w_to_quant = w_bf16_NDHWC
    rows = Co * kT * kH * kW
    w_2d = w_to_quant.reshape(rows, Ci).contiguous()
    w_fp4 = torch.empty((rows, Ci // 2), dtype=torch.uint8, device='cuda')
    w_sf = torch.empty((rows, Ci // 16), dtype=torch.uint8, device='cuda')
    fvk.quantize_bf16_to_nvfp4(
        int(w_2d.data_ptr()), int(w_fp4.data_ptr()), int(w_sf.data_ptr()),
        rows, Ci, 0)
    torch.cuda.synchronize()
    fp4w = _Fp4Weight(
        w_fp4.view(Co, kT, kH, kW, Ci // 2),
        w_sf.view(Co, kT, kH, kW, Ci // 16),
        Co, Ci,
    )
    if outer_w is not None:
        fp4w.outer_w = outer_w.contiguous()
    return fp4w


# ── Activation quant helpers ──

def _dequant_fp8_to_bf16(fp8_t: torch.Tensor, scale: float) -> torch.Tensor:
    """fp8_e4m3 → bf16. scale = act_scale (per-tensor)."""
    return (fp8_t.float() * scale).to(torch.bfloat16)


def _quant_bf16_to_fp4(bf16_t: torch.Tensor, stream: int) -> tuple:
    """[..., Ci] bf16 cuda → (fp4 [..., Ci/2], sf [..., Ci/16]) uint8 cuda.
    `stream` MUST be torch.cuda.current_stream().cuda_stream so the launch
    is recorded into the cuda graph capture (else: NaN under graph replay).
    """
    *prefix, Ci = bf16_t.shape
    rows = 1
    for d in prefix:
        rows *= d
    fp4 = torch.empty((rows, Ci // 2), dtype=torch.uint8, device='cuda')
    sf = torch.empty((rows, Ci // 16), dtype=torch.uint8, device='cuda')
    fvk.quantize_bf16_to_nvfp4(
        int(bf16_t.contiguous().data_ptr()),
        int(fp4.data_ptr()), int(sf.data_ptr()),
        rows, Ci, stream)
    return fp4.view(*prefix, Ci // 2), sf.view(*prefix, Ci // 16)


# ── FP4 fused step (drop-in for _fused_step where Ci%64==0) ──

def _fused_step_fp4(x_bf16, gamma_flat, w_fp8_NDHWC, w_scale, act_scale,
                    cache_fp8_NDHWC, bias=None, eps=1e-6,
                    residual_ncdhw=None):
    """FP4 fused step. Reuses FP8 RMS+SiLU+quant for new chunk, then
    dequants stitched cache+new and re-quants to FP4 for v19sf.

    Returns (y_NCDHW_bf16, new_cache_fp8_NDHWC) — cache stays FP8 to
    preserve compatibility with the FP8 patched_forward cache contract.
    """
    fp4w = _FP4_WEIGHT_BY_ID.get(id(w_fp8_NDHWC))
    if fp4w is None:
        # Fallback to FP8 path (router shouldn't have called us; safety)
        return _vae_fp8_swap._fused_step(
            x_bf16, gamma_flat, w_fp8_NDHWC, w_scale, act_scale,
            cache_fp8_NDHWC, bias=bias, eps=eps, residual_ncdhw=residual_ncdhw)

    B, C, T, H, W = x_bf16.shape
    Co = fp4w.Co
    s = torch.cuda.current_stream().cuda_stream

    Tc = _vae_fp8_swap._CACHE_T
    wid = id(w_fp8_NDHWC)
    use_fused = _v4q_fn is not None and (C % 128) == 0
    # FLASH_RT_MOTUS_VAE_FP4_AGGRESSIVE_CACHE=1 → FP4 cache (Phase
    # 8.4 v2). Saves an extra ~3 ms vs default but cuts cos(frames) margin
    # from ~0.001 to ~0.000 — right at the 0.997 floor. Default OFF for
    # production safety; opt-in for users willing to live near the edge.
    aggressive_cache = (
        use_fused and
        os.environ.get('FLASH_RT_MOTUS_VAE_FP4_AGGRESSIVE_CACHE', '0') == '1')

    if use_fused:
        # ── Phase 8.3: bf16 NCDHW → FP4+SF NDHWC directly via fused kernel ──
        # Always-on whenever C % 128 == 0 (no precision cost).
        new_fp4 = torch.empty(B, T, H, W, C // 2,
                              dtype=torch.uint8, device=x_bf16.device)
        new_sf = torch.empty(B, T, H, W, C // 16,
                             dtype=torch.uint8, device=x_bf16.device)
        awq_ptr = (int(fp4w.awq_inv_scale.data_ptr())
                   if fp4w.awq_inv_scale is not None else 0)
        rc = _v4q_fn(
            ctypes.c_void_p(int(x_bf16.contiguous().data_ptr())),
            ctypes.c_void_p(int(gamma_flat.contiguous().data_ptr())),
            ctypes.c_void_p(awq_ptr),
            ctypes.c_void_p(int(new_fp4.data_ptr())),
            ctypes.c_void_p(int(new_sf.data_ptr())),
            B, C, T, H, W, ctypes.c_float(float(eps)), ctypes.c_void_p(s))
        if rc != 0:
            raise RuntimeError(f'[fp4_vae] v4q rc={rc}')

    if use_fused and aggressive_cache:
        # ── Phase 8.4 v2 (opt-in): FP4 cache, lower precision ──
        cache_state = _FP4_CACHE_BY_WID.get(wid)
        if cache_state is None:
            cache_fp4 = torch.zeros(B, Tc, H, W, C // 2,
                                    dtype=torch.uint8, device=x_bf16.device)
            cache_sf = torch.zeros(B, Tc, H, W, C // 16,
                                   dtype=torch.uint8, device=x_bf16.device)
            _FP4_CACHE_BY_WID[wid] = (cache_fp4, cache_sf)
        else:
            cache_fp4, cache_sf = cache_state
        new_cache_fp8 = None
    elif use_fused:
        # ── Phase 8.3 default: FP8 cache via dequant/requant (safe cos) ──
        new_fp8 = torch.empty(B, T, H, W, C, dtype=_FP8, device=x_bf16.device)
        rc = fvk.bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4(
            int(x_bf16.contiguous().data_ptr()),
            int(gamma_flat.contiguous().data_ptr()),
            int(new_fp8.data_ptr()),
            B, C, T, H, W, float(act_scale), float(eps), s)
        if rc != 0:
            raise RuntimeError(f'[fp4_vae] fused_quant_v4 rc={rc}')
        if cache_fp8_NDHWC is None:
            cache_fp8 = _vae_fp8_swap._zero_fp8_cache(B, H, W, C, x_bf16.device)
        elif cache_fp8_NDHWC.shape[1] < Tc:
            n_pad = Tc - cache_fp8_NDHWC.shape[1]
            zeros = torch.zeros(B, n_pad, H, W, C, dtype=_FP8, device=x_bf16.device)
            cache_fp8 = torch.cat([zeros, cache_fp8_NDHWC], dim=1).contiguous()
        elif cache_fp8_NDHWC.shape[1] > Tc:
            cache_fp8 = cache_fp8_NDHWC[:, -Tc:].contiguous()
        else:
            cache_fp8 = cache_fp8_NDHWC
        cache_bf16 = _dequant_fp8_to_bf16(cache_fp8, act_scale)
        cache_fp4, cache_sf = _quant_bf16_to_fp4(cache_bf16, s)
        if T >= Tc:
            new_cache_fp8 = new_fp8[:, -Tc:].contiguous() \
                if not new_fp8[:, -Tc:].is_contiguous() else new_fp8[:, -Tc:]
        elif cache_fp8_NDHWC is not None:
            n_borrow = Tc - T
            new_cache_fp8 = torch.cat(
                [cache_fp8_NDHWC[:, -n_borrow:], new_fp8], dim=1).contiguous()
        else:
            z = _vae_fp8_swap._zero_fp8_cache(B, H, W, C, x_bf16.device)
            new_cache_fp8 = torch.empty(
                B, Tc, H, W, C, dtype=_FP8, device=x_bf16.device)
            new_cache_fp8[:, :Tc - T] = z[:, :Tc - T]
            new_cache_fp8[:, Tc - T:] = new_fp8
    else:
        # ── Fallback path for C % 128 != 0 (e.g., Ci=320): keep FP8 kernel
        #    + dequant/requant. Install gate filters these out by default,
        #    so this path normally doesn't execute in production.
        new_fp8 = torch.empty(B, T, H, W, C, dtype=_FP8, device=x_bf16.device)
        rc = fvk.bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4(
            int(x_bf16.contiguous().data_ptr()),
            int(gamma_flat.contiguous().data_ptr()),
            int(new_fp8.data_ptr()),
            B, C, T, H, W, float(act_scale), float(eps), s)
        if rc != 0:
            raise RuntimeError(f'[fp4_vae] fused_quant_v4 rc={rc}')
        if cache_fp8_NDHWC is None:
            cache_fp8 = _vae_fp8_swap._zero_fp8_cache(B, H, W, C, x_bf16.device)
        elif cache_fp8_NDHWC.shape[1] < Tc:
            n_pad = Tc - cache_fp8_NDHWC.shape[1]
            zeros = torch.zeros(B, n_pad, H, W, C, dtype=_FP8, device=x_bf16.device)
            cache_fp8 = torch.cat([zeros, cache_fp8_NDHWC], dim=1).contiguous()
        elif cache_fp8_NDHWC.shape[1] > Tc:
            cache_fp8 = cache_fp8_NDHWC[:, -Tc:].contiguous()
        else:
            cache_fp8 = cache_fp8_NDHWC
        new_bf16 = _dequant_fp8_to_bf16(new_fp8, act_scale)
        new_fp4, new_sf = _quant_bf16_to_fp4(new_bf16, s)
        cache_bf16 = _dequant_fp8_to_bf16(cache_fp8, act_scale)
        cache_fp4, cache_sf = _quant_bf16_to_fp4(cache_bf16, s)
        if T >= Tc:
            new_cache_fp8 = new_fp8[:, -Tc:].contiguous() \
                if not new_fp8[:, -Tc:].is_contiguous() else new_fp8[:, -Tc:]
        elif cache_fp8_NDHWC is not None:
            n_borrow = Tc - T
            new_cache_fp8 = torch.cat(
                [cache_fp8_NDHWC[:, -n_borrow:], new_fp8], dim=1).contiguous()
        else:
            z = _vae_fp8_swap._zero_fp8_cache(B, H, W, C, x_bf16.device)
            new_cache_fp8 = torch.empty(
                B, Tc, H, W, C, dtype=_FP8, device=x_bf16.device)
            new_cache_fp8[:, :Tc - T] = z[:, :Tc - T]
            new_cache_fp8[:, Tc - T:] = new_fp8

    # 4. Run v19sf or v19sfb. v19sfb writes NCDHW directly with optional
    #    residual add fused in the epilogue (mirrors v17→v18 pattern in
    #    the FP8 path), eliminating the post-hoc bf16_ndhwc_to_ncdhw[_add]
    #    launch (~6.9 ms wall total across the FP4 VAE site set when
    #    enabled). Env-gated; set to '0' to revert to v19sf + post-hoc
    #    transpose path. v19sfb requires contiguous NCDHW residual; we
    #    materialize one when needed (cheap when already contig).
    use_v19sfb = (
        os.environ.get('FLASH_RT_MOTUS_VAE_FP4_USE_V19SFB', '1') == '1')
    bias_ptr = int(bias.data_ptr()) if bias is not None else 0
    if use_v19sfb:
        y = torch.empty(B, Co, T, H, W, dtype=torch.bfloat16,
                        device=x_bf16.device)
        if residual_ncdhw is not None:
            res_in = residual_ncdhw if residual_ncdhw.is_contiguous() \
                else residual_ncdhw.contiguous()
            res_ptr = int(res_in.data_ptr())
        else:
            res_ptr = 0
        if fp4w.outer_w is not None:
            rc = _v19sfb_fn_v2(
                ctypes.c_void_p(int(cache_fp4.data_ptr())),
                ctypes.c_void_p(int(new_fp4.data_ptr())),
                ctypes.c_void_p(int(fp4w.w_fp4.data_ptr())),
                ctypes.c_void_p(int(cache_sf.data_ptr())),
                ctypes.c_void_p(int(new_sf.data_ptr())),
                ctypes.c_void_p(int(fp4w.w_sf.data_ptr())),
                ctypes.c_void_p(int(fp4w.outer_w.data_ptr())),
                ctypes.c_void_p(int(y.data_ptr())),
                ctypes.c_void_p(bias_ptr),
                ctypes.c_void_p(res_ptr),
                B, _vae_fp8_swap._CACHE_T, T, H, W, C, Co,
                ctypes.c_float(1.0),
                ctypes.c_void_p(s),
            )
            if rc != 0:
                raise RuntimeError(f'[fp4_vae] v19sfb_v2 rc={rc}')
        else:
            # K128 path: 1.21x avg vs ship on all production hot shapes
            # (Ci=256/512/1024/640). Requires Ci>=128 (one K-tile = 128 elem
            # must stay within a single 3x3x3 tap to keep SF/data contiguous).
            use_k128 = (
                _v19sfb_k128_fn is not None
                and os.environ.get(
                    'FLASH_RT_MOTUS_VAE_FP4_K128', '1') == '1'
                and (C % 128) == 0)
            if use_k128:
                rc = _v19sfb_k128_fn(
                    ctypes.c_void_p(int(cache_fp4.data_ptr())),
                    ctypes.c_void_p(int(new_fp4.data_ptr())),
                    ctypes.c_void_p(int(fp4w.w_fp4.data_ptr())),
                    ctypes.c_void_p(int(cache_sf.data_ptr())),
                    ctypes.c_void_p(int(new_sf.data_ptr())),
                    ctypes.c_void_p(int(fp4w.w_sf.data_ptr())),
                    ctypes.c_void_p(int(y.data_ptr())),
                    ctypes.c_void_p(bias_ptr),
                    ctypes.c_void_p(res_ptr),
                    B, _vae_fp8_swap._CACHE_T, T, H, W, C, Co,
                    ctypes.c_float(1.0),
                    ctypes.c_void_p(s),
                )
                if rc != 0:
                    raise RuntimeError(f'[fp4_vae] v19sfb_k128 rc={rc}')
            else:
                rc = _v19sfb_fn(
                    ctypes.c_void_p(int(cache_fp4.data_ptr())),
                    ctypes.c_void_p(int(new_fp4.data_ptr())),
                    ctypes.c_void_p(int(fp4w.w_fp4.data_ptr())),
                    ctypes.c_void_p(int(cache_sf.data_ptr())),
                    ctypes.c_void_p(int(new_sf.data_ptr())),
                    ctypes.c_void_p(int(fp4w.w_sf.data_ptr())),
                    ctypes.c_void_p(int(y.data_ptr())),
                    ctypes.c_void_p(bias_ptr),
                    ctypes.c_void_p(res_ptr),
                    B, _vae_fp8_swap._CACHE_T, T, H, W, C, Co,
                    ctypes.c_float(1.0),
                    ctypes.c_void_p(s),
                )
                if rc != 0:
                    raise RuntimeError(f'[fp4_vae] v19sfb rc={rc}')
    else:
        # Legacy path: v19sf (NDHWC) + post-hoc bf16_ndhwc_to_ncdhw[_add].
        y_NDHWC = torch.empty(B, T, H, W, Co, dtype=torch.bfloat16,
                              device=x_bf16.device)
        if fp4w.outer_w is not None:
            rc = _v19sf_fn_v2(
                ctypes.c_void_p(int(cache_fp4.data_ptr())),
                ctypes.c_void_p(int(new_fp4.data_ptr())),
                ctypes.c_void_p(int(fp4w.w_fp4.data_ptr())),
                ctypes.c_void_p(int(cache_sf.data_ptr())),
                ctypes.c_void_p(int(new_sf.data_ptr())),
                ctypes.c_void_p(int(fp4w.w_sf.data_ptr())),
                ctypes.c_void_p(int(fp4w.outer_w.data_ptr())),
                ctypes.c_void_p(int(y_NDHWC.data_ptr())),
                ctypes.c_void_p(bias_ptr),
                B, _vae_fp8_swap._CACHE_T, T, H, W, C, Co,
                ctypes.c_float(1.0),
                ctypes.c_void_p(s),
            )
            if rc != 0:
                raise RuntimeError(f'[fp4_vae] v19sf_v2 rc={rc}')
        else:
            rc = _v19sf_fn(
                ctypes.c_void_p(int(cache_fp4.data_ptr())),
                ctypes.c_void_p(int(new_fp4.data_ptr())),
                ctypes.c_void_p(int(fp4w.w_fp4.data_ptr())),
                ctypes.c_void_p(int(cache_sf.data_ptr())),
                ctypes.c_void_p(int(new_sf.data_ptr())),
                ctypes.c_void_p(int(fp4w.w_sf.data_ptr())),
                ctypes.c_void_p(int(y_NDHWC.data_ptr())),
                ctypes.c_void_p(bias_ptr),
                B, _vae_fp8_swap._CACHE_T, T, H, W, C, Co,
                ctypes.c_float(1.0),
                ctypes.c_void_p(s),
            )
            if rc != 0:
                raise RuntimeError(f'[fp4_vae] v19sf rc={rc}')
        y = torch.empty(B, Co, T, H, W, dtype=torch.bfloat16,
                        device=x_bf16.device)
        if residual_ncdhw is not None:
            rs = residual_ncdhw.stride()
            rc = fvk.bf16_ndhwc_to_ncdhw_add_bf16(
                int(y_NDHWC.data_ptr()), int(residual_ncdhw.data_ptr()),
                int(y.data_ptr()), B, Co, T, H, W,
                rs[0], rs[1], rs[2], rs[3], rs[4], s)
        else:
            rc = fvk.bf16_ndhwc_to_ncdhw_transpose(
                int(y_NDHWC.data_ptr()), int(y.data_ptr()),
                B, Co, T, H, W, s)
        if rc != 0:
            raise RuntimeError(f'[fp4_vae] transpose/add rc={rc}')

    # Phase 8.4 v2: in-place FP4 cache update (aggressive mode only).
    # See feedback_graph_capture_dict_cache.md for why copy_() (not reassign).
    if use_fused and aggressive_cache:
        if T >= Tc:
            cache_fp4.copy_(new_fp4[:, -Tc:])
            cache_sf.copy_(new_sf[:, -Tc:])
        else:
            cache_fp4.copy_(torch.cat([cache_fp4[:, T:Tc].clone(), new_fp4], dim=1))
            cache_sf.copy_(torch.cat([cache_sf[:, T:Tc].clone(), new_sf], dim=1))

    return y, new_cache_fp8


# ── Router ──

_orig_fused_step = None
_router_installed = False


def _router_fused_step(x_bf16, gamma_flat, w_fp8_NDHWC, w_scale, act_scale,
                       cache_fp8_NDHWC, bias=None, eps=1e-6,
                       residual_ncdhw=None):
    """Routes to FP4 path if site has cached FP4 weight AND env enabled."""
    use_fp4 = os.environ.get('FLASH_RT_MOTUS_USE_FP4_VAE', '0') == '1'
    if use_fp4 and id(w_fp8_NDHWC) in _FP4_WEIGHT_BY_ID:
        if _TRACE:
            logger.info(f'[fp4_vae] FP4 path: '
                        f'shape={tuple(x_bf16.shape)}')
        return _fused_step_fp4(
            x_bf16, gamma_flat, w_fp8_NDHWC, w_scale, act_scale,
            cache_fp8_NDHWC, bias=bias, eps=eps,
            residual_ncdhw=residual_ncdhw)
    return _orig_fused_step(
        x_bf16, gamma_flat, w_fp8_NDHWC, w_scale, act_scale,
        cache_fp8_NDHWC, bias=bias, eps=eps,
        residual_ncdhw=residual_ncdhw)


# ── Install ──

def install_vae_fp4(model) -> dict:
    """Quantize FP8-eligible site weights to FP4 and install router.

    Must be called AFTER _vae_fp8_swap.install_vae_fp8(model).
    Returns summary dict.
    """
    global _orig_fused_step, _router_installed

    if not _load_libs():
        return {'error': 'libs_not_found'}

    # Walk all _Fp8Site instances by traversing model's residual blocks
    # (mirrors _vae_fp8_swap.install_vae_fp8 walk).
    import sys
    vae_mod_name = next(
        (n for n in sys.modules if 'wan' in n and 'vae2_2' in n), None)
    if vae_mod_name is None:
        raise RuntimeError('[fp4_vae] wan.modules.vae2_2 not loaded')
    vae_mod = sys.modules[vae_mod_name]
    ResidualBlock_class = vae_mod.ResidualBlock

    vae_root = model.video_model.vae.model
    blocks = [m for _, m in vae_root.named_modules()
              if isinstance(m, ResidualBlock_class)]

    # Mixed-precision filter: heavy-tail layers (large Ci) compound FP4
    # quant noise across many K-iters; cap which sites get FP4 (others
    # stay FP8). Default 2048 covers all Ci tiers including the Ci=1024
    # decoder.middle group (17 sites); measured cos(frames) on motus E2E
    # stays at 0.9971-0.9973 (above the 0.997 K7 floor, ~7× margin over
    # the relaxed 0.99 floor). Drop to 640 if a downstream gate needs the
    # extra cos(frames) headroom; that adds back ~4 ms wall (sm80 bf16
    # cuDNN fallback for the Ci=1024 group).
    max_ci = int(os.environ.get('FLASH_RT_MOTUS_VAE_FP4_MAX_CI', '2048'))
    outer_fp32 = (
        os.environ.get('FLASH_RT_MOTUS_VAE_FP4_OUTER_FP32', '0') == '1')

    n_quantized = 0
    n_skipped = 0
    n_total = 0
    skip_reasons = {}
    for block in blocks:
        sites_list = getattr(block, '_g7_23_sites', None)
        if sites_list is None:
            continue
        for site in sites_list:
            if not getattr(site, 'eligible', False):
                continue
            if site.w_fp8_NDHWC is None:
                continue
            n_total += 1
            w_bf16 = site.conv.weight.data       # [Co, Ci, kt, kh, kw] bf16
            Ci = w_bf16.shape[1]; Co = w_bf16.shape[0]
            if Ci > max_ci:
                n_skipped += 1
                reason = f'Ci>{max_ci} (Ci={Ci})'
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            # Require fused-kernel compatibility (C%128==0) to avoid the slow
            # dequant/requant fallback that's net-negative vs FP8 baseline.
            # Incompat sites (e.g., Ci=320) stay on FP8.
            if Ci % 128 != 0:
                n_skipped += 1
                reason = f'Ci%128!=0 (Ci={Ci}); falls to FP8'
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            w_NDHWC = w_bf16.permute(0, 2, 3, 4, 1).contiguous().to('cuda')
            fp4w = _quant_weight_to_fp4(w_NDHWC, outer_fp32=outer_fp32)
            if fp4w is None:
                n_skipped += 1
                reason = f'Ci%64={Ci%64} Co%8={Co%8}'
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            _FP4_WEIGHT_BY_ID[id(site.w_fp8_NDHWC)] = fp4w
            n_quantized += 1

    if not _router_installed:
        _orig_fused_step = _vae_fp8_swap._fused_step
        _vae_fp8_swap._fused_step = _router_fused_step
        _router_installed = True

    summary = {
        'n_quantized': n_quantized,
        'n_skipped': n_skipped,
        'n_total_fp8_sites': n_total,
        'skip_reasons': skip_reasons,
        'router_installed': _router_installed,
        'env_enabled': os.environ.get('FLASH_RT_MOTUS_USE_FP4_VAE', '0') == '1',
        'outer_fp32': outer_fp32,
    }
    logger.info(f'[fp4_vae] install summary: {summary}')
    return summary


def awq_calibrate_and_dump(model, pipeline, first_frame, state,
                           t5_embeds, vlm_inputs,
                           apply_scales: bool = False,
                           out_path: str = '/tmp/motus_vae_act_stats.json'):
    """Phase 9: collect per-Ci activation stats for FP4-routed Conv3d sites.

    Registers forward_pre_hooks on every Conv3d in the VAE that has a
    matching entry in _FP4_WEIGHT_BY_ID, runs one calibration forward
    (via pipeline.run with FP8 calibrating=True so bf16 fallback fires
    everywhere — gives us the post-RMS+SiLU bf16 activation), then dumps
    per-site stats to JSON.

    If apply_scales=True, also computes AWQ scale s[ci] per site and:
      - Multiplies the bf16 weight by s along the Ci dim (offline)
      - Re-quantizes the scaled weight to FP4 (overwrites _FP4_WEIGHT_BY_ID)
      - Stores per-site inv_s for runtime (consumed by v4q kernel — kernel
        mod required separately; this Python-side step preserves info).
    """
    import json
    import torch
    import torch.nn as nn
    from . import _vae_fp8_swap

    vae_root = model.video_model.vae.model
    # Identify which Conv3d modules have FP4 weight registered
    wid_to_site_info = {}    # id(w_fp8_NDHWC) → (Conv3d module, label)
    for name, block in vae_root.named_modules():
        sites_list = getattr(block, '_g7_23_sites', None)
        if sites_list is None:
            continue
        for site in sites_list:
            if not getattr(site, 'eligible', False):
                continue
            wid = id(site.w_fp8_NDHWC) if site.w_fp8_NDHWC is not None else None
            if wid is None or wid not in _FP4_WEIGHT_BY_ID:
                continue
            wid_to_site_info[id(site.conv)] = (site.conv, wid, site, name)

    # Hook each FP4-routed Conv3d to collect per-Ci max|x|
    stats = {}    # mod_id → (label, Ci, [max_per_ci tensors per call])
    handles = []
    fire_count = [0]

    def make_hook(mod_id):
        def hook(module, inputs):
            fire_count[0] += 1
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() != 5:
                return
            mx = x.float().abs().amax(dim=(0, 2, 3, 4)).cpu()
            stats.setdefault(mod_id, []).append(mx)
        return hook

    for mod_id, (conv, wid, site, name) in wid_to_site_info.items():
        h = conv.register_forward_pre_hook(make_hook(mod_id))
        handles.append(h)

    print(f'[awq_calib] hooked {len(handles)} FP4-routed Conv3d sites; '
          f'first 3: '
          f'{[(n.split(".")[-3:], int(c.in_channels)) for _, (c, _, _, n) in list(wid_to_site_info.items())[:3]]}; '
          f'_STATE.calibrating={_vae_fp8_swap._STATE.calibrating}, '
          f'running calibration forward...', flush=True)
    # Force bf16 fallback by setting calibrating; this matches the FP8 calib
    # pattern so we observe the SAME activation distribution that production
    # will see in graph capture.
    _vae_fp8_swap.set_calibrating(True)
    try:
        with torch.no_grad():
            _ = pipeline.run(
                first_frame=first_frame, state=state,
                t5_embeds=t5_embeds, vlm_inputs=vlm_inputs,
            )
        torch.cuda.synchronize()
    finally:
        _vae_fp8_swap.set_calibrating(False)
    for h in handles:
        h.remove()
    print(f'[awq_calib] hooks fired {fire_count[0]} times total; '
          f'stats dict has {len(stats)} entries', flush=True)

    # Aggregate + dump
    print(f'\n[awq_calib] site-by-site stats:')
    print(f'{"label":<60} {"Ci":>5} {"calls":>5} {"max":>8} {"min":>8} '
          f'{"med":>8} {"ratio":>8}')
    summary = []
    for mod_id, calls in stats.items():
        conv, wid, site, name = wid_to_site_info[mod_id]
        agg = torch.stack(calls).amax(dim=0)  # per-Ci worst-case across calls
        mx = float(agg.max())
        mn = float(agg.min())
        med = float(agg.median())
        ratio = mx / max(mn, 1e-9)
        Ci = int(conv.in_channels)
        Co = int(conv.out_channels)
        label_short = name.split('.')[-3:]
        label_short = '.'.join(label_short)
        print(f'{label_short:<60} {Ci:>5} {len(calls):>5} '
              f'{mx:>8.3f} {mn:>8.4f} {med:>8.4f} {ratio:>7.1f}x')
        summary.append({
            'name': name,
            'Ci': Ci, 'Co': Co,
            'n_calls': len(calls),
            'max': mx, 'min': mn, 'median': med, 'ratio': ratio,
            'top5_ci_max': [
                (int(i), float(v))
                for v, i in zip(*agg.sort(descending=True))][:5],
        })

    # Aggregate distribution
    ratios = [s['ratio'] for s in summary]
    if ratios:
        ratios.sort()
        print(f'\n[awq_calib] aggregate per-site ratios:')
        print(f'  p10: {ratios[len(ratios)//10]:.2f}x')
        print(f'  p50: {ratios[len(ratios)//2]:.2f}x')
        print(f'  p90: {ratios[9*len(ratios)//10]:.2f}x')
        print(f'  max: {ratios[-1]:.2f}x')
        n_big = sum(1 for r in ratios if r > 5)
        n_med = sum(1 for r in ratios if r > 2)
        print(f'  > 5x ratio: {n_big} / {len(ratios)}  (AWQ benefit likely)')
        print(f'  > 2x ratio: {n_med} / {len(ratios)}  (AWQ marginal)')

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n[awq_calib] saved JSON: {out_path}')

    # Phase 9.2: selective FP8 fallback for the N sites with highest
    # per-Ci heterogeneity (act_max/act_min ratio). Compound FP4 quant
    # noise hits ~0.992 floor at full FP4; removing the worst N (largest
    # contributors) recovers cos toward 0.997 with minor wall regression.
    skip_top_n = int(os.environ.get(
        'FLASH_RT_MOTUS_VAE_FP4_SKIP_TOP_HETERO_N', '0'))
    if skip_top_n > 0:
        # Rank sites by ratio (descending)
        ratios = []
        for mod_id, calls in stats.items():
            agg = torch.stack(calls).amax(dim=0)
            r = float(agg.max() / agg.clamp(min=1e-9).min())
            ratios.append((r, mod_id))
        ratios.sort(reverse=True)
        to_skip = set(mid for _, mid in ratios[:skip_top_n])
        n_evicted = 0
        for mid in to_skip:
            conv, wid, site, name = wid_to_site_info[mid]
            if wid in _FP4_WEIGHT_BY_ID:
                del _FP4_WEIGHT_BY_ID[wid]   # router falls back to FP8
                # Also clear FP4 cache for this site if present
                _FP4_CACHE_BY_WID.pop(wid, None)
                n_evicted += 1
        print(f'\n[awq_calib] evicted top {n_evicted} most-heterogeneous '
              f'sites (skip_top_hetero_n={skip_top_n}); they will use '
              f'the original FP8 path. Top sites by ratio:')
        for r, mid in ratios[:skip_top_n]:
            _, _, _, name = wid_to_site_info[mid]
            short = '.'.join(name.split('.')[-3:])[:60]
            print(f'    ratio={r:>7.1f}x  {short}')

    if apply_scales:
        alpha = float(os.environ.get('FLASH_RT_MOTUS_VAE_FP4_AWQ_ALPHA', '0.5'))
        s_clip = float(os.environ.get('FLASH_RT_MOTUS_VAE_FP4_AWQ_CLIP', '100.0'))
        print(f'\n[awq_calib] applying AWQ scales (alpha={alpha}, clip=[{1/s_clip:.4g},{s_clip:.4g}])')

        n_applied = 0
        for mod_id, calls in stats.items():
            conv, wid, site, name = wid_to_site_info[mod_id]
            fp4w = _FP4_WEIGHT_BY_ID.get(wid)
            if fp4w is None:
                continue
            # Per-Ci activation max (worst-case across calibration calls)
            act_max = torch.stack(calls).amax(dim=0).cuda()    # [Ci] fp32
            # Per-Ci weight max: max over (Co, kT, kH, kW) of |w[:, ci, :, :, :]|
            w_bf16 = site.conv.weight.data.cuda()              # [Co, Ci, kt, kh, kw]
            w_max = w_bf16.float().abs().amax(dim=(0, 2, 3, 4))  # [Ci]
            # AWQ scale: s[ci] = max|x|[ci]^α / max|w|[ci]^(1-α), clipped
            act_max = act_max.clamp(min=1e-5)
            w_max = w_max.clamp(min=1e-5)
            s = (act_max.pow(alpha) / w_max.pow(1.0 - alpha))
            s = s.clamp(min=1.0/s_clip, max=s_clip)             # avoid extremes
            # Skip mean-normalization — it dampens AWQ's redistribution.
            # No-normalize is the standard SmoothQuant formulation.
            do_norm = os.environ.get(
                'FLASH_RT_MOTUS_VAE_FP4_AWQ_NORMALIZE', '0') == '1'
            if do_norm:
                s = s / s.mean().clamp(min=1e-5)
            inv_s = (1.0 / s).contiguous().to(torch.float32)
            # Print scale distribution for first 3 sites to verify
            if n_applied < 3:
                ss = s.cpu()
                print(f'  site {n_applied}: Ci={int(conv.in_channels)} '
                      f's range=[{ss.min():.4f}, {ss.max():.4f}] '
                      f'mean={ss.mean():.4f} median={ss.median():.4f}')

            # Re-quantize weight * s  (per-Ci broadcast)
            Co_, Ci_, kT, kH, kW = w_bf16.shape
            w_scaled = (w_bf16.float() * s.view(1, Ci_, 1, 1, 1)).to(torch.bfloat16)
            w_NDHWC = w_scaled.permute(0, 2, 3, 4, 1).contiguous()
            rows = Co_ * 27
            new_w_fp4 = torch.empty((rows, Ci_ // 2), dtype=torch.uint8, device='cuda')
            new_w_sf = torch.empty((rows, Ci_ // 16), dtype=torch.uint8, device='cuda')
            fvk.quantize_bf16_to_nvfp4(
                int(w_NDHWC.data_ptr()),
                int(new_w_fp4.data_ptr()), int(new_w_sf.data_ptr()),
                rows, Ci_, 0)
            torch.cuda.synchronize()

            # In-place update of fp4w (preserves tensor address — graph-safe)
            fp4w.w_fp4.copy_(new_w_fp4.view(Co_, 3, 3, 3, Ci_ // 2))
            fp4w.w_sf.copy_(new_w_sf.view(Co_, 3, 3, 3, Ci_ // 16))
            fp4w.awq_inv_scale = inv_s
            n_applied += 1

        print(f'[awq_calib] AWQ applied to {n_applied} sites; '
              f'router will pass inv_s to v4q kernel')


def uninstall_vae_fp4():
    """Restore original _fused_step. Useful for A/B testing."""
    global _router_installed
    if _router_installed and _orig_fused_step is not None:
        _vae_fp8_swap._fused_step = _orig_fused_step
        _router_installed = False
    _FP4_WEIGHT_BY_ID.clear()
    _FP4_CACHE_BY_WID.clear()

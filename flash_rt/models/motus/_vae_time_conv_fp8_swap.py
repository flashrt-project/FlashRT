"""VAE temporal (3,1,1) CausalConv3d → FP8 GEMM swap.

Stage4 (2026-05-17) extension to _vae_fp8_swap. The (3,1,1) time_conv
modules in Resample blocks fall back to cuDNN BF16 (sm80_xmma) because
the FP8 site installer only handles (3,3,3) and (1,1,1) kernels.

Math: out[b, t, h, w, co] = sum_{kt in 0..2, ci} x[b, t+kt, h, w, ci] *
                                                  w[co, ci, kt, 0, 0]

Implementation: pad + permute NCDHW→NDHWC + per-tensor FP8 quant, then
im2col along T (cat 3 slices on channel axis), cuBLASLt FP8 NN GEMM,
NDHWC→NCDHW transpose, bias add.

Gated by FLASH_RT_MOTUS_USE_VAE_TIME_CONV_FP8 env (default OFF until cos
verified). Calibration uses act_amax tracker similar to _Fp8Site.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk
from flash_rt.models.motus._stream import cs

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_GEMM_RUNNER = None


def _get_gemm_runner():
    global _GEMM_RUNNER
    if _GEMM_RUNNER is None:
        _GEMM_RUNNER = fvk.GemmRunner()
    return _GEMM_RUNNER


class _TimeConvFp8Site:
    """Per-(3,1,1)-conv FP8 state.

    Weight layout: original conv.weight is [Co, Ci, 3, 1, 1] bf16. The
    flattened-K-axis (3*Ci) needs to match the im2col order used at
    forward (cat[x[:,0:T_out], x[:,1:T_out+1], x[:,2:T_out+2]] in NDHWC
    on the last axis → channel order is kt-outer, ci-inner). We pre-flat
    w to [3*Ci, Co] KN-major to match cuBLASLt fp8_nn_dev expectation.
    """
    __slots__ = (
        'conv', 'name',
        'Ci', 'Co',
        'w_fp8_KN', 'w_scale', 'w_scale_dev',
        'act_amax_max', 'act_scale', 'act_scale_host', 'act_scale_dev',
        'has_bias', 'bias',
        'orig_forward',
        'mode',  # 'calib' | 'fp8'
        'padding',  # CausalConv3d._padding tuple (replicated)
    )

    def __init__(self, conv: nn.Conv3d, name: str):
        self.conv = conv
        self.name = name
        self.Ci = int(conv.in_channels)
        self.Co = int(conv.out_channels)
        self.has_bias = conv.bias is not None
        self.bias = conv.bias
        # CausalConv3d sets self.padding=(0,0,0) and stores _padding. We
        # need _padding to replicate forward.
        self.padding = tuple(conv._padding)
        self.orig_forward = conv.forward
        self.act_amax_max = 0.0
        self.act_scale = 1.0
        self.act_scale_host = 1.0
        self.act_scale_dev = None
        self.w_fp8_KN = None
        self.w_scale = 1.0
        self.w_scale_dev = None
        self.mode = 'calib'
        self._quantize_weight()

    def _quantize_weight(self):
        # conv.weight: [Co, Ci, 3, 1, 1] bf16
        w = self.conv.weight.data  # [Co, Ci, 3, 1, 1]
        w_sq = w.squeeze(-1).squeeze(-1)  # [Co, Ci, 3]
        # Im2col channel order at forward will be (kt, ci) flattened with
        # kt outer (cat order). So K-axis goes [kt=0 ci=0..Ci-1, kt=1
        # ci=0..Ci-1, kt=2 ci=0..Ci-1]. Permute weight to [Co, kt, Ci] →
        # flatten last 2 dims to K = 3*Ci.
        w_perm = w_sq.permute(0, 2, 1).contiguous()  # [Co, 3, Ci]
        w_flat = w_perm.reshape(self.Co, 3 * self.Ci)  # [Co, K]
        # cuBLASLt fp8_nn_dev needs B-matrix as (K, N=Co) row-major (KN).
        w_KN = w_flat.t().contiguous()  # [K, Co]
        max_abs = float(w_KN.abs().max().item())
        scale = max(max_abs / 448.0, 1e-12)
        self.w_fp8_KN = ((w_KN.float() / scale).clamp(-448.0, 448.0)
                         .to(_FP8).contiguous())
        dev = self.w_fp8_KN.device
        self.w_scale = scale
        self.w_scale_dev = torch.tensor([scale], dtype=torch.float32,
                                        device=dev)
        # act_scale starts at 1.0 (placeholder); finalized in finalize().
        self.act_scale_dev = torch.tensor([1.0], dtype=torch.float32,
                                          device=dev)

    def finalize_act_scale(self):
        """Convert observed act_amax_max → act_scale."""
        self.act_scale = max(self.act_amax_max / 448.0, 1e-12)
        self.act_scale_host = float(self.act_scale)
        self.act_scale_dev.fill_(float(self.act_scale))
        # alpha = act_scale * w_scale (so dequant of output uses single product).
        # For cuBLASLt: we pass act_scale_dev and w_scale_dev as separate
        # ptrs; runtime multiplies internally.
        self.mode = 'fp8'

    def update_amax(self, x_bf16: torch.Tensor):
        v = float(x_bf16.float().abs().amax().item())
        if v > self.act_amax_max:
            self.act_amax_max = v


_SITES: List[_TimeConvFp8Site] = []


def _make_calib_forward(site: _TimeConvFp8Site):
    """During calibration: run BF16 path (original) but also record amax."""
    orig = site.orig_forward

    def forward(self, x, cache_x=None):
        site.update_amax(x)
        if cache_x is not None and site.padding[4] > 0:
            site.update_amax(cache_x.to(x.device))
        return orig(x, cache_x) if cache_x is not None else orig(x)
    return forward


def _make_fp8_forward(site: _TimeConvFp8Site):
    """After calibration: run FP8 path."""
    Ci = site.Ci
    Co = site.Co
    bias_ptr = (int(site.bias.data_ptr()) if site.has_bias else 0)

    def forward(self, x, cache_x=None):
        # Replicate CausalConv3d.forward padding (no super().forward).
        padding = list(site.padding)
        if cache_x is not None and site.padding[4] > 0:
            if cache_x.device != x.device:
                cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        # x: [B, Ci, T_total, H, W] bf16
        x_c = x if x.is_contiguous() else x.contiguous()
        B, _Ci, T_total, H, W = x_c.shape
        T_out = T_total - 2  # 3-tap kernel, no extra padding
        if T_out <= 0:
            # Fallback if not enough frames
            return site.orig_forward(x, cache_x) if cache_x is None else site.orig_forward(x, cache_x)
        stream = cs()
        # 1) Quant + permute NCDHW → NDHWC FP8: [B, T_total, H, W, Ci]
        x_fp8 = torch.empty(B, T_total, H, W, Ci, dtype=_FP8,
                            device=x_c.device)
        rcq = fvk.bf16_quant_fp8_ncdhw_to_ndhwc(
            int(x_c.data_ptr()), int(x_fp8.data_ptr()),
            B, Ci, T_total, H, W, site.act_scale_host, stream)
        if rcq != 0:
            raise RuntimeError(f'[time_conv_fp8] {site.name} quant rc={rcq}')
        # 2) im2col along T: cat 3 slices on last axis → [B, T_out, H, W, 3*Ci]
        # x_fp8[:, 0:T_out], x_fp8[:, 1:T_out+1], x_fp8[:, 2:T_out+2]
        x_stacked = torch.cat([
            x_fp8[:, 0:T_out],
            x_fp8[:, 1:T_out + 1],
            x_fp8[:, 2:T_out + 2],
        ], dim=-1)  # [B, T_out, H, W, 3*Ci] FP8 contiguous
        M = B * T_out * H * W
        K = 3 * Ci
        # 3) cuBLASLt FP8 GEMM: (M, K) × (K, Co) → (M, Co) bf16
        out_flat = torch.empty(M, Co, dtype=torch.bfloat16,
                               device=x_c.device)
        gemm = _get_gemm_runner()
        gemm.fp8_nn_dev(
            int(x_stacked.data_ptr()), int(site.w_fp8_KN.data_ptr()),
            int(out_flat.data_ptr()),
            M, Co, K,
            int(site.act_scale_dev.data_ptr()),
            int(site.w_scale_dev.data_ptr()),
            stream)
        # 4) NDHWC → NCDHW transpose
        out_NCDHW = torch.empty(B, Co, T_out, H, W,
                                dtype=torch.bfloat16, device=x_c.device)
        if site.has_bias and bias_ptr and hasattr(
                fvk, 'bf16_ndhwc_to_ncdhw_bias_bf16'):
            rct = fvk.bf16_ndhwc_to_ncdhw_bias_bf16(
                int(out_flat.data_ptr()), bias_ptr, int(out_NCDHW.data_ptr()),
                B, Co, T_out, H, W, stream)
        else:
            rct = fvk.bf16_ndhwc_to_ncdhw_transpose(
                int(out_flat.data_ptr()), int(out_NCDHW.data_ptr()),
                B, Co, T_out, H, W, stream)
        if rct != 0:
            raise RuntimeError(
                f'[time_conv_fp8] {site.name} ndhwc->ncdhw rc={rct}')
        # 5) Bias add
        if (site.has_bias and bias_ptr and not hasattr(
                fvk, 'bf16_ndhwc_to_ncdhw_bias_bf16')):
            fvk.add_bias_ncdhw_bf16(
                int(out_NCDHW.data_ptr()), bias_ptr,
                B, Co, T_out, H, W, stream)
        return out_NCDHW
    return forward


def install_time_conv_fp8_swap(model,
                                env_var: str = 'FLASH_RT_MOTUS_USE_VAE_TIME_CONV_FP8') -> dict:
    """Walk VAE model, find Resample.time_conv modules (CausalConv3d (3,1,1)),
    attach _TimeConvFp8Site and patch forward into calibration mode.

    Returns stats dict. Sites finalize on first call to finalize_all().
    """
    if os.environ.get(env_var, '0') not in ('1', 'true', 'on', 'yes'):
        return {'enabled': False, 'reason': 'disabled_by_env'}
    _SITES.clear()
    n_found = 0
    n_eligible = 0
    n_skipped = 0
    skip_reasons = {}
    try:
        vae_root = model.video_model.vae.model
    except AttributeError:
        return {'enabled': False, 'reason': 'no_vae'}
    for name, mod in vae_root.named_modules():
        if not name.endswith('time_conv'):
            continue
        n_found += 1
        if not isinstance(mod, nn.Conv3d):
            continue
        # Must be (3,1,1) kernel
        if tuple(mod.kernel_size) != (3, 1, 1):
            skip_reasons['kernel'] = skip_reasons.get('kernel', 0) + 1
            n_skipped += 1
            continue
        # Must have _padding attr (CausalConv3d)
        if not hasattr(mod, '_padding'):
            skip_reasons['no_padding'] = skip_reasons.get('no_padding', 0) + 1
            n_skipped += 1
            continue
        Ci = int(mod.in_channels)
        Co = int(mod.out_channels)
        if Ci % 16 != 0 or Co % 8 != 0:
            skip_reasons['align'] = skip_reasons.get('align', 0) + 1
            n_skipped += 1
            continue
        site = _TimeConvFp8Site(mod, name)
        _SITES.append(site)
        mod.forward = _make_calib_forward(site).__get__(mod, type(mod))
        n_eligible += 1
    logger.info(
        f'[time_conv_fp8] installed: n_found={n_found} '
        f'n_eligible={n_eligible} n_skipped={n_skipped} '
        f'skip_reasons={skip_reasons}')
    return {
        'enabled': True,
        'n_found': n_found,
        'n_eligible': n_eligible,
        'n_skipped': n_skipped,
        'skip_reasons': skip_reasons,
    }


def finalize_time_conv_fp8():
    """After calibration: finalize act_scale per site and swap to FP8 forward.
    Sites with amax=0 (never fired during calib) are restored to original
    BF16 forward to avoid garbage outputs.
    """
    if not _SITES:
        return {'finalized': 0}
    finalized = 0
    skipped_no_calib = []
    for site in _SITES:
        if site.act_amax_max <= 0.0:
            site.conv.forward = site.orig_forward
            skipped_no_calib.append(site.name)
            continue
        site.finalize_act_scale()
        site.conv.forward = _make_fp8_forward(site).__get__(
            site.conv, type(site.conv))
        finalized += 1
    return {'finalized': finalized,
            'skipped_no_calib': skipped_no_calib,
            'amaxes': [(s.name, s.act_amax_max) for s in _SITES]}

"""G7.23 — FP8 swap for motus VAE ResidualBlock CausalConv3d sites.

Replaces the chain
    RMS_norm -> SiLU -> CausalConv3d (kernel=3x3x3, padding=1)
within each ResidualBlock.residual sequence with the FlashVLA fused path:
    fused_quant_v4   (BF16 NCDHW -> FP8 NDHWC, fused RMS+SiLU+permute)
        cat(prev_chunk_FP8_cache | zero_pad)
        v11 fp8_conv3d (symmetric pad=1)
        slice [1 : T+1]                (causal window via time-shift trick)
        permute -> NCDHW

Eligibility per site:
  - kernel == (3, 3, 3)
  - out_channels % 8 == 0   (v11 constraint)
  - in_channels  % 4 == 0   (fused_quant_v4 constraint)
Sites failing eligibility fall through to the upstream BF16 path verbatim.

Cache contract:
  feat_cache[idx] for SWAPPED sites stores the last 2 frames of the
  post-fused-quant FP8 NDHWC tensor. Unswapped sites continue using
  the upstream bf16 NCDHW snapshot. Slots are typed consistently
  across chunks because each block's swap status is fixed at install.

Calibration:
  set_calibrating(True); pipe.infer(...); set_calibrating(False)
    During calibration, swapped sites still run the upstream bf16 path
    (so the same cache state is built). Side effect: each site records
    max(|silu(rms(x))|) seen during the call. After calibration,
    commit_calibration() converts max -> act_scale (max / 448).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_TRACE = os.environ.get('FLASH_RT_MOTUS_VAE_FP8_TRACE', '0') == '1'
_CACHE_T = 2
_ZERO_FP8_CACHE: dict[tuple[int, int, int, int, int], torch.Tensor] = {}
_T1_CONV2D_WEIGHT_CACHE: dict[int, torch.Tensor] = {}


class _State:
    calibrating = False


_STATE = _State()


def set_calibrating(on: bool) -> None:
    _STATE.calibrating = bool(on)
    logger.info(f'[g7.23] vae_fp8 calibrating={_STATE.calibrating}')


class _Fp8ShortcutSite:
    """Per-site state for a swapped (1,1,1) ResidualBlock.shortcut conv.

    The shortcut takes the BLOCK INPUT directly (no RMS/SiLU before it),
    so the FP8 path is: bare quant → fp8_nn_dev (GEMM, since 1×1×1 conv
    equals matmul over the channel axis) → NDHWC→NCDHW transpose → bias.
    """
    __slots__ = (
        'conv', 'label', 'eligible', 'reason_skipped',
        'w_fp8',          # [Ci, Co] fp8 (transposed for fp8_nn_dev nn = K×N)
        'w_scale_dev',    # fp32 [1] device scalar (for fp8_nn_dev)
        'act_scale_dev',  # fp32 [1] device scalar (for fp8_nn_dev)
        'act_scale_host', # host float (for bare_quant kernel)
        'act_scale_max',  # host float, tracked during calibration
        'has_bias', 'bias',
    )

    def __init__(self, conv_module, label):
        self.conv = conv_module
        self.label = label
        self.act_scale_max = 0.0
        self.act_scale_host = 1.0
        self.has_bias = conv_module.bias is not None
        self.bias = conv_module.bias
        self.eligible, self.reason_skipped = self._check(conv_module)
        if self.eligible:
            self._quantize_weight()
        else:
            self.w_fp8 = None
            self.w_scale_dev = None
            self.act_scale_dev = None

    @staticmethod
    def _check(conv):
        if tuple(conv.kernel_size) != (1, 1, 1):
            return False, f'kernel={tuple(conv.kernel_size)}'
        if tuple(conv.stride) != (1, 1, 1):
            return False, f'stride={tuple(conv.stride)}'
        Co = int(conv.out_channels)
        Ci = int(conv.in_channels)
        # fp8_nn_dev tile constraints (mostly 8 / 16 alignment).
        if Co % 8 != 0 or Ci % 4 != 0:
            return False, f'Co%8 or Ci%4 (Ci={Ci}, Co={Co})'
        return True, ''

    def _quantize_weight(self):
        # conv.weight: [Co, Ci, 1, 1, 1] bf16
        w = self.conv.weight.data.squeeze(-1).squeeze(-1).squeeze(-1)
        # → [Co, Ci]. fp8_nn_dev needs B-matrix as (K=Ci, N=Co) row-major.
        w_KN = w.t().contiguous()
        max_abs = float(w_KN.abs().max().item())
        scale = max(max_abs / 448.0, 1e-12)
        self.w_fp8 = ((w_KN.float() / scale).clamp(-448.0, 448.0)
                      .to(_FP8).contiguous())
        dev = self.w_fp8.device
        self.w_scale_dev = torch.tensor([scale],
                                        dtype=torch.float32, device=dev)
        self.act_scale_dev = torch.tensor([1.0],
                                          dtype=torch.float32, device=dev)


class _Fp8Site:
    """Per-site state for one swapped (RMS, SiLU, CausalConv3d) triplet."""
    __slots__ = (
        'rms', 'conv', 'label', 'site_idx',
        'eligible', 'reason_skipped',
        'w_fp8_NDHWC', 'w_scale',
        'w2d_cl',
        'act_scale_max', 'act_scale',
        'has_bias', 'bias',
    )

    def __init__(self, rms_module, conv_module, label, site_idx: int):
        self.rms = rms_module
        self.conv = conv_module
        self.label = label
        self.site_idx = int(site_idx)
        self.act_scale_max = 0.0
        self.act_scale = 1.0
        self.w2d_cl = None
        self.has_bias = conv_module.bias is not None
        self.bias = conv_module.bias
        self.eligible, self.reason_skipped = self._check_eligible(conv_module)
        if self.eligible:
            self._quantize_weight()
            self._maybe_prepare_t1_conv2d_weight()
        else:
            self.w_fp8_NDHWC = None
            self.w_scale = 1.0

    @staticmethod
    def _check_eligible(conv) -> tuple:
        if tuple(conv.kernel_size) != (3, 3, 3):
            return False, f'kernel={tuple(conv.kernel_size)}'
        Co = int(conv.out_channels)
        Ci = int(conv.in_channels)
        if Co % 8 != 0:
            return False, f'Co%8 (Co={Co})'
        if Ci % 4 != 0:
            return False, f'Ci%4 (Ci={Ci})'
        if tuple(conv.stride) != (1, 1, 1):
            return False, f'stride={tuple(conv.stride)}'
        return True, ''

    def _quantize_weight(self):
        w = self.conv.weight.data  # [Co, Ci, kt, kh, kw] bf16
        w_NDHWC = w.permute(0, 2, 3, 4, 1).contiguous()
        max_abs = float(w_NDHWC.abs().max().item())
        scale = max(max_abs / 448.0, 1e-12)
        w_q = (w_NDHWC.float() / scale).clamp(-448, 448).to(_FP8).contiguous()
        self.w_fp8_NDHWC = w_q
        self.w_scale = scale

    def _maybe_prepare_t1_conv2d_weight(self):
        if os.environ.get('FLASH_RT_MOTUS_VAE_T1_CONV2D', '0') != '1':
            return
        if self.site_idx not in range(20, 30):
            return
        conv = self.conv
        if (int(conv.in_channels) != 1024 or int(conv.out_channels) != 1024
                or tuple(conv.kernel_size) != (3, 3, 3)):
            return
        w = conv.weight.detach().contiguous()
        self.w2d_cl = (w.permute(0, 2, 1, 3, 4)
                         .reshape(1024, 3 * 1024, 3, 3)
                         .contiguous(memory_format=torch.channels_last))


class _HeadFp8Site:
    """FP8 state for decoder.head.2 CausalConv3d (Co=12 any-Co path)."""
    __slots__ = (
        'conv', 'w_fp8_NDHWC', 'w_scale', 'act_scale_max', 'act_scale',
        'has_bias', 'bias', 'orig_forward')

    def __init__(self, conv_module):
        self.conv = conv_module
        self.orig_forward = conv_module.forward
        self.act_scale_max = 0.0
        self.act_scale = 1.0
        self.has_bias = conv_module.bias is not None
        self.bias = conv_module.bias
        w = conv_module.weight.data
        w_NDHWC = w.permute(0, 2, 3, 4, 1).contiguous()
        max_abs = float(w_NDHWC.abs().max().item())
        self.w_scale = max(max_abs / 448.0, 1e-12)
        self.w_fp8_NDHWC = (
            w_NDHWC.float() / self.w_scale).clamp(-448, 448).to(
                _FP8).contiguous()


_GEMM_RUNNER = None


def _get_gemm_runner():
    global _GEMM_RUNNER
    if _GEMM_RUNNER is None:
        _GEMM_RUNNER = fvk.GemmRunner()
    return _GEMM_RUNNER


def _zero_fp8_cache(B: int, H: int, W: int, C: int,
                    device: torch.device) -> torch.Tensor:
    dev_idx = device.index
    if dev_idx is None:
        dev_idx = torch.cuda.current_device()
    key = (int(dev_idx), int(B), int(H), int(W), int(C))
    z = _ZERO_FP8_CACHE.get(key)
    if z is None or z.device != device:
        z = torch.zeros(B, _CACHE_T, H, W, C, dtype=_FP8, device=device)
        _ZERO_FP8_CACHE[key] = z
    return z


def _shortcut_fp8(x_bf16_NCDHW, site):
    """FP8 path for ResidualBlock.shortcut (kernel=1×1×1 CausalConv3d).

    Chain:
      1. bare quant + permute  : bf16 NCDHW -> fp8 NDHWC, flat (M=B·T·H·W, K=Ci)
      2. fp8_nn_dev GEMM       : (M,K) × (K,N=Co) -> (M,N) bf16
      3. NDHWC -> NCDHW transpose (custom kernel)
      4. bias add (in-place + view)
    """
    B, Ci, T, H, W = x_bf16_NCDHW.shape
    Co = site.w_fp8.shape[1]
    M = B * T * H * W
    s = torch.cuda.current_stream().cuda_stream

    # 1. bare quant + permute (per-tensor static scale via host float).
    x_fp8 = torch.empty(B, T, H, W, Ci,
                        dtype=_FP8, device=x_bf16_NCDHW.device)
    rc = fvk.bf16_quant_fp8_ncdhw_to_ndhwc(
        int(x_bf16_NCDHW.contiguous().data_ptr()),
        int(x_fp8.data_ptr()),
        B, Ci, T, H, W, site.act_scale_host, s)
    if rc != 0:
        raise RuntimeError(f'[g7.23] bare_quant rc={rc}')

    # 2. FP8 GEMM: (M, K=Ci) × (K, N=Co) → (M, N=Co) bf16
    out_flat = torch.empty(M, Co,
                           dtype=torch.bfloat16, device=x_bf16_NCDHW.device)
    gemm = _get_gemm_runner()
    gemm.fp8_nn_dev(
        int(x_fp8.data_ptr()), int(site.w_fp8.data_ptr()),
        int(out_flat.data_ptr()),
        M, Co, Ci,
        int(site.act_scale_dev.data_ptr()),
        int(site.w_scale_dev.data_ptr()),
        s)

    # 3. NDHWC → NCDHW
    out_NCDHW = torch.empty(B, Co, T, H, W,
                            dtype=torch.bfloat16, device=x_bf16_NCDHW.device)
    if site.has_bias and hasattr(fvk, 'bf16_ndhwc_to_ncdhw_bias_bf16'):
        rc = fvk.bf16_ndhwc_to_ncdhw_bias_bf16(
            int(out_flat.data_ptr()), int(site.bias.data_ptr()),
            int(out_NCDHW.data_ptr()), B, Co, T, H, W, s)
    else:
        rc = fvk.bf16_ndhwc_to_ncdhw_transpose(
            int(out_flat.data_ptr()), int(out_NCDHW.data_ptr()),
            B, Co, T, H, W, s)
    if rc != 0:
        raise RuntimeError(f'[g7.23] shortcut transpose rc={rc}')

    # 4. bias add
    if site.has_bias and not hasattr(fvk, 'bf16_ndhwc_to_ncdhw_bias_bf16'):
        fvk.add_bias_ncdhw_bf16(
            int(out_NCDHW.data_ptr()), int(site.bias.data_ptr()),
            B, Co, T, H, W, s)
    return out_NCDHW


def _fused_step(x_bf16, gamma_flat, w_fp8_NDHWC, w_scale, act_scale,
                cache_fp8_NDHWC, bias=None, eps=1e-6,
                residual_ncdhw: Optional[torch.Tensor] = None):
    """One fused FP8 causal-conv step.

    Cache contract:
      - cache_fp8_NDHWC has shape [B, _CACHE_T, H, W, C] (or None).
      - Returned new_cache_fp8 has the SAME shape, maintained by reaching
        back into the input cache when the current chunk has T < _CACHE_T
        (mirrors the bf16 CausalConv3d cache-stitch logic).

    Stream: kernels MUST run on torch.cuda.current_stream() so they
    are recorded into the active torch.cuda.graph(...) capture. Passing
    stream=0 (NULL/default) makes the launch race the capture stream
    and the kernel never makes it into the graph, so replay leaves
    output buffers untouched (root cause of the G7.23 NaN-under-graph
    discovered via tests/probe_g7_23_graph_repro.py).

    Returns (y_NCDHW_bf16, new_cache_fp8_NDHWC).
    """
    B, C, T, H, W = x_bf16.shape
    Co = w_fp8_NDHWC.shape[0]
    s = torch.cuda.current_stream().cuda_stream

    # 1. fused {RMS+SiLU+quant+permute}: BF16 NCDHW -> FP8 NDHWC
    new_fp8 = torch.empty(B, T, H, W, C,
                          dtype=_FP8, device=x_bf16.device)
    rc = fvk.bf16_rms_silu_quant_fp8_ncdhw_to_ndhwc_v4(
        int(x_bf16.contiguous().data_ptr()),
        int(gamma_flat.contiguous().data_ptr()),
        int(new_fp8.data_ptr()),
        B, C, T, H, W, float(act_scale), float(eps), s)
    if rc != 0:
        raise RuntimeError(f'[g7.23] fused_quant_v4 rc={rc}')

    # 2. Normalize cache shape to exactly [B, _CACHE_T, H, W, C]. v17
    #    takes cache + new as separate pointers, but it still assumes
    #    the cache slot has exactly T_cache=_CACHE_T (=2) frames.
    if cache_fp8_NDHWC is None:
        cache = _zero_fp8_cache(B, H, W, C, x_bf16.device)
    elif cache_fp8_NDHWC.shape[1] < _CACHE_T:
        n_pad = _CACHE_T - cache_fp8_NDHWC.shape[1]
        zeros = torch.zeros(B, n_pad, H, W, C,
                            dtype=_FP8, device=x_bf16.device)
        cache = torch.cat([zeros, cache_fp8_NDHWC], dim=1).contiguous()
    elif cache_fp8_NDHWC.shape[1] > _CACHE_T:
        cache = cache_fp8_NDHWC[:, -_CACHE_T:].contiguous()
    else:
        cache = cache_fp8_NDHWC

    # 3. v17/v18 conv: virtual cache cat + direct causal output + bias-fused
    #    epilogue. Eliminates the explicit torch.cat (cache + new) AND
    #    the post-conv [1:T+1] slice (kernel iterates only the causal
    #    output range). v18 additionally writes NCDHW directly and can
    #    fuse the ResidualBlock shortcut add in the final conv.
    alpha = float(act_scale * w_scale)
    bias_ptr = int(bias.data_ptr()) if bias is not None else 0
    use_v18 = (
        os.environ.get('FLASH_RT_MOTUS_NO_VAE_FP8_V18', '0') != '1'
        and os.environ.get('FLASH_RT_MOTUS_VAE_FP8_V18', '1') == '1'
        and residual_ncdhw is not None)
    if use_v18:
        y = torch.empty(B, Co, T, H, W,
                        dtype=torch.bfloat16, device=x_bf16.device)
        res_ptr = int(residual_ncdhw.data_ptr()) if residual_ncdhw is not None else 0
        rc = fvk.fp8_conv3d_v18_ncdhw_res_bf16out(
            int(cache.data_ptr()), int(new_fp8.data_ptr()),
            int(w_fp8_NDHWC.contiguous().data_ptr()),
            int(y.data_ptr()),
            bias_ptr, res_ptr,
            B, _CACHE_T, T, H, W, C, Co, alpha, s)
        if rc != 0:
            raise RuntimeError(f'[g7.24] v18 rc={rc}')
    else:
        y_NDHWC = torch.empty(B, T, H, W, Co,
                              dtype=torch.bfloat16, device=x_bf16.device)
        rc = fvk.fp8_conv3d_v17_ndhwc_bf16out(
            int(cache.data_ptr()), int(new_fp8.data_ptr()),
            int(w_fp8_NDHWC.contiguous().data_ptr()),
            int(y_NDHWC.data_ptr()),
            bias_ptr,
            B, _CACHE_T, T, H, W, C, Co, alpha, s)
        if rc != 0:
            raise RuntimeError(f'[g7.23] v17 rc={rc}')

        # 4. NDHWC → NCDHW transpose.
        y = torch.empty(B, Co, T, H, W,
                        dtype=torch.bfloat16, device=x_bf16.device)
        use_fused_res = (
            os.environ.get('FLASH_RT_MOTUS_VAE_FP8_FUSE_RES_TRANSPOSE', '1') == '1'
            and residual_ncdhw is not None)
        if use_fused_res:
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
            raise RuntimeError(f'[g7.23] bf16_ndhwc_to_ncdhw rc={rc}')

    # 6. build new cache (always T=_CACHE_T). For T >= _CACHE_T return
    #    a VIEW into new_fp8 — no copy; the graph mempool keeps new_fp8
    #    alive as long as feat_cache references the view, which is
    #    safe because the next chunk's captured v17 only reads from it
    #    (no in-place mutation of cache contents). Saves the
    #    .clone().contiguous() copy on the cache update path.
    if T >= _CACHE_T:
        new_cache_fp8 = new_fp8[:, -_CACHE_T:].contiguous() \
            if not new_fp8[:, -_CACHE_T:].is_contiguous() \
            else new_fp8[:, -_CACHE_T:]
    else:
        if cache_fp8_NDHWC is not None:
            n_borrow = _CACHE_T - T
            new_cache_fp8 = torch.cat(
                [cache_fp8_NDHWC[:, -n_borrow:], new_fp8], dim=1).contiguous()
        else:
            # Initial chunk with T<_CACHE_T: zero-pad the missing frames.
            z = _zero_fp8_cache(B, H, W, C, x_bf16.device)
            new_cache_fp8 = torch.empty(
                B, _CACHE_T, H, W, C, dtype=_FP8, device=x_bf16.device)
            new_cache_fp8[:, :_CACHE_T - T] = z[:, :_CACHE_T - T]
            new_cache_fp8[:, _CACHE_T - T:] = new_fp8

    return y, new_cache_fp8


def _head_conv_fp8(x_bf16, cache_bf16, site: _HeadFp8Site):
    B, C, T, H, W = x_bf16.shape
    Co = site.w_fp8_NDHWC.shape[0]
    s = torch.cuda.current_stream().cuda_stream
    x_c = x_bf16 if x_bf16.is_contiguous() else x_bf16.contiguous()
    new_fp8 = torch.empty(B, T, H, W, C, dtype=_FP8, device=x_bf16.device)
    rc = fvk.bf16_quant_fp8_ncdhw_to_ndhwc(
        int(x_c.data_ptr()), int(new_fp8.data_ptr()),
        B, C, T, H, W, float(site.act_scale), s)
    if rc != 0:
        raise RuntimeError(f'[g7.27] head new quant rc={rc}')
    if cache_bf16 is None:
        cache_fp8 = _zero_fp8_cache(B, H, W, C, x_bf16.device)
    else:
        cache_c = cache_bf16 if cache_bf16.is_contiguous() else cache_bf16.contiguous()
        Tc = int(cache_c.shape[2])
        if Tc >= _CACHE_T:
            cache_src = cache_c[:, :, -_CACHE_T:, :, :]
        else:
            cache_src = cache_c
        cache_src = cache_src if cache_src.is_contiguous() else cache_src.contiguous()
        if cache_src.shape[2] == _CACHE_T:
            cache_fp8 = torch.empty(B, _CACHE_T, H, W, C,
                                    dtype=_FP8, device=x_bf16.device)
            rc = fvk.bf16_quant_fp8_ncdhw_to_ndhwc(
                int(cache_src.data_ptr()), int(cache_fp8.data_ptr()),
                B, C, _CACHE_T, H, W, float(site.act_scale), s)
            if rc != 0:
                raise RuntimeError(f'[g7.27] head cache quant rc={rc}')
        else:
            cache_fp8 = _zero_fp8_cache(B, H, W, C, x_bf16.device)
            tail_fp8 = torch.empty(B, 1, H, W, C,
                                   dtype=_FP8, device=x_bf16.device)
            rc = fvk.bf16_quant_fp8_ncdhw_to_ndhwc(
                int(cache_src.data_ptr()), int(tail_fp8.data_ptr()),
                B, C, 1, H, W, float(site.act_scale), s)
            if rc != 0:
                raise RuntimeError(f'[g7.27] head tail quant rc={rc}')
            cache_fp8[:, -1:] = tail_fp8
    y_NDHWC = torch.empty(B, T, H, W, Co,
                          dtype=torch.bfloat16, device=x_bf16.device)
    bias_ptr = int(site.bias.data_ptr()) if site.has_bias else 0
    rc = fvk.fp8_conv3d_v17_anyco_ndhwc_bf16out(
        int(cache_fp8.data_ptr()), int(new_fp8.data_ptr()),
        int(site.w_fp8_NDHWC.data_ptr()), int(y_NDHWC.data_ptr()),
        bias_ptr, B, _CACHE_T, T, H, W, C, Co,
        float(site.act_scale * site.w_scale), s)
    if rc != 0:
        raise RuntimeError(f'[g7.27] head v17_anyco rc={rc}')
    y = torch.empty(B, Co, T, H, W, dtype=torch.bfloat16, device=x_bf16.device)
    rc = fvk.bf16_ndhwc_to_ncdhw_transpose(
        int(y_NDHWC.data_ptr()), int(y.data_ptr()), B, Co, T, H, W, s)
    if rc != 0:
        raise RuntimeError(f'[g7.27] head transpose rc={rc}')
    return y


def _rms_silu_bf16_ncdhw(x_bf16, gamma_flat, eps=1e-6,
                         cache_prev: Optional[torch.Tensor] = None,
                         cache_out: Optional[torch.Tensor] = None):
    """Fused BF16 RMS_norm + SiLU for BF16 fallback sites.

    Used mainly for T=1 VAE sites where the full FP8 conv chain is slower
    than cuDNN BF16, but separate PyTorch RMS_norm and SiLU launches still
    add avoidable overhead.
    """
    B, C, T, H, W = x_bf16.shape
    s = torch.cuda.current_stream().cuda_stream
    x_c = x_bf16 if x_bf16.is_contiguous() else x_bf16.contiguous()
    gamma_c = gamma_flat if gamma_flat.is_contiguous() else gamma_flat.contiguous()
    out = torch.empty_like(x_c)
    prev_ptr = int(cache_prev.data_ptr()) if cache_prev is not None else 0
    cache_ptr = int(cache_out.data_ptr()) if cache_out is not None else 0
    rc = fvk.bf16_rms_silu_ncdhw(
        int(x_c.data_ptr()), int(gamma_c.data_ptr()), int(out.data_ptr()),
        prev_ptr, cache_ptr,
        B, C, T, H, W, float(eps), s)
    if rc != 0:
        raise RuntimeError(f'[g7.24] bf16_rms_silu_ncdhw rc={rc}')
    return out


def _rms_norm_bf16_ncdhw(x_bf16, gamma_flat, bias_flat=None, eps=1e-12):
    """VAE standalone RMS_norm for channel-first BF16 4D/5D tensors."""
    orig_shape = x_bf16.shape
    if x_bf16.dim() == 4:
        B, C, H, W = x_bf16.shape
        T = 1
        x_view = x_bf16.reshape(B, C, T, H, W)
    elif x_bf16.dim() == 5:
        B, C, T, H, W = x_bf16.shape
        x_view = x_bf16
    else:
        raise RuntimeError(f'[g7.52] expected 4D/5D, got {tuple(orig_shape)}')
    s = torch.cuda.current_stream().cuda_stream
    x_c = x_view if x_view.is_contiguous() else x_view.contiguous()
    gamma_c = gamma_flat if gamma_flat.is_contiguous() else gamma_flat.contiguous()
    bias_ptr = 0
    if isinstance(bias_flat, torch.Tensor):
        bias_c = bias_flat if bias_flat.is_contiguous() else bias_flat.contiguous()
        bias_ptr = int(bias_c.data_ptr())
    out = torch.empty_like(x_c)
    rc = fvk.bf16_rms_norm_ncdhw(
        int(x_c.data_ptr()), int(gamma_c.data_ptr()), bias_ptr,
        int(out.data_ptr()), B, C, T, H, W, float(eps), s)
    if rc != 0:
        raise RuntimeError(f'[g7.52] bf16_rms_norm_ncdhw rc={rc}')
    return out.reshape(orig_shape)


def _t1_conv3d_as_conv2d(s: torch.Tensor,
                         prev: torch.Tensor,
                         site: _Fp8Site) -> torch.Tensor:
    """Run T=1 causal 3D conv as a 2D conv over temporal channels.

    This is only for the Stage3 decoder 1024->1024 fallback sites where
    cuDNN Conv2d NHWC is faster than cuDNN Conv3d for the same math.
    """
    if (s.shape[0] != 1 or s.shape[1] != 1024 or s.shape[2] != 1
            or s.shape[3] != 24 or s.shape[4] != 20):
        return site.conv(s, prev)
    if not (isinstance(prev, torch.Tensor) and prev.dim() == 5
            and prev.shape[1] == 1024 and prev.shape[2] >= 2):
        return site.conv(s, prev)

    w2 = site.w2d_cl
    if w2 is None:
        key = id(site.conv)
        w2 = _T1_CONV2D_WEIGHT_CACHE.get(key)
        if w2 is None or w2.device != s.device:
            w = site.conv.weight.detach().contiguous()
            w2 = (w.permute(0, 2, 1, 3, 4)
                    .reshape(1024, 3 * 1024, 3, 3)
                    .contiguous(memory_format=torch.channels_last))
            _T1_CONV2D_WEIGHT_CACHE[key] = w2

    x2 = torch.empty((1, 3 * 1024, 24, 20),
                     dtype=s.dtype, device=s.device,
                     memory_format=torch.channels_last)
    rc = fvk.bf16_pack_t1_cache3_nchw_channels_last(
        int(prev.data_ptr()), int(s.data_ptr()), int(x2.data_ptr()),
        1024, 24, 20, torch.cuda.current_stream().cuda_stream)
    if rc != 0:
        raise RuntimeError(f'[g7.24] t1 conv2d pack rc={rc}')
    y2 = F.conv2d(x2, w2, site.conv.bias, padding=1)
    return y2.unsqueeze(2).contiguous()


def _add_bf16_out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_c = a if a.is_contiguous() else a.contiguous()
    b_c = b if b.is_contiguous() else b.contiguous()
    out = torch.empty_like(a_c)
    fvk.add_bf16_out(
        int(a_c.data_ptr()), int(b_c.data_ptr()), int(out.data_ptr()),
        out.numel(), torch.cuda.current_stream().cuda_stream)
    return out


def _update_cache2_bf16_ncdhw(cur: torch.Tensor,
                              prev: Optional[torch.Tensor]) -> torch.Tensor:
    B, C, T, H, W = cur.shape
    out = torch.empty(B, C, _CACHE_T, H, W,
                      dtype=cur.dtype, device=cur.device)
    prev_ptr = 0
    if (prev is not None and isinstance(prev, torch.Tensor)
            and prev.dtype == cur.dtype and prev.dim() == 5):
        prev_ptr = int(prev.data_ptr())
    fvk.update_cache2_ncdhw_bf16(
        int(cur.data_ptr()), prev_ptr, int(out.data_ptr()),
        B, C, T, H, W, torch.cuda.current_stream().cuda_stream)
    return out


def _find_triplets(layers, RMS_norm_class, CausalConv3d_class):
    """Scan a residual Sequential for (RMS_norm, SiLU, CausalConv3d) groups.

    Tolerates nn.Dropout layers inserted between SiLU and the conv (which
    is the canonical motus VAE ResidualBlock layout — the second conv has
    a Dropout sitting between SiLU and Conv).

    Returns list of (rms_idx, silu_idx, conv_idx, conv_end_idx). The
    conv_end_idx is the LAST layer the triplet 'consumes' so the patched
    forward can advance past any intervening Dropout layers.
    """
    triplets = []
    i = 0
    n = len(layers)
    while i < n:
        if not isinstance(layers[i], RMS_norm_class):
            i += 1
            continue
        rms_idx = i
        # Find SiLU (skipping Dropouts)
        j = i + 1
        while j < n and isinstance(layers[j], nn.Dropout):
            j += 1
        if j >= n or not isinstance(layers[j], nn.SiLU):
            i += 1
            continue
        silu_idx = j
        # Find Conv (skipping Dropouts)
        k = j + 1
        while k < n and isinstance(layers[k], nn.Dropout):
            k += 1
        if k >= n or not isinstance(layers[k], CausalConv3d_class):
            i += 1
            continue
        conv_idx = k
        triplets.append((rms_idx, silu_idx, conv_idx, k))
        i = k + 1
    return triplets


def _build_up_residual_forward():
    def patched_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is None:
            return x_main
        x_shortcut = self.avg_shortcut(x, first_chunk)
        if (x_main.dtype == torch.bfloat16
                and x_shortcut.dtype == torch.bfloat16
                and x_main.shape == x_shortcut.shape):
            return _add_bf16_out(x_main, x_shortcut)
        return x_main + x_shortcut

    return patched_forward


def _build_dup_up3d_forward():
    def patched_forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        if (x.dtype != torch.bfloat16
                or not hasattr(fvk, 'dup_up3d_bf16')
                or os.environ.get('FLASH_RT_MOTUS_NO_G7_36_DUPUP_KERNEL',
                                  '0') == '1'):
            x = x.repeat_interleave(self.repeats, dim=1)
            x = x.view(
                x.size(0), self.out_channels, self.factor_t, self.factor_s,
                self.factor_s, x.size(2), x.size(3), x.size(4))
            x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
            x = x.view(
                x.size(0), self.out_channels,
                x.size(2) * self.factor_t,
                x.size(4) * self.factor_s,
                x.size(6) * self.factor_s)
            if first_chunk:
                x = x[:, :, self.factor_t - 1:, :, :]
            return x

        x_c = x if x.is_contiguous() else x.contiguous()
        B, Cin, T, H, W = x_c.shape
        out_T = T * int(self.factor_t)
        if first_chunk:
            out_T -= int(self.factor_t) - 1
        out = torch.empty(
            B, int(self.out_channels), out_T, H * int(self.factor_s),
            W * int(self.factor_s), dtype=x_c.dtype, device=x_c.device)
        fvk.dup_up3d_bf16(
            int(x_c.data_ptr()), int(out.data_ptr()),
            int(B), int(Cin), int(self.out_channels), int(T), int(H), int(W),
            int(self.factor_t), int(self.factor_s), int(self.repeats),
            1 if first_chunk else 0, torch.cuda.current_stream().cuda_stream)
        return out

    return patched_forward


def _build_patched_forward(block, sites: List[_Fp8Site],
                           RMS_norm_class, CausalConv3d_class):
    """Build a closure replacing ResidualBlock.forward."""
    # Discover triplet starting indices in block.residual
    layers0 = list(block.residual)
    found = _find_triplets(layers0, RMS_norm_class, CausalConv3d_class)
    triplet_to_site = {}
    triplet_conv_end = {}
    for ((rms_idx, silu_idx, conv_idx, end_idx), site) in zip(found, sites):
        triplet_to_site[rms_idx] = site
        triplet_conv_end[rms_idx] = end_idx
    t1_sites_env = os.environ.get('FLASH_RT_MOTUS_VAE_FP8_T1_SITES', '')
    if t1_sites_env:
        t1_sites = set(int(x) for x in t1_sites_env.split(',') if x.strip())
    elif os.environ.get('FLASH_RT_MOTUS_VAE_FP8_T1_AUTO', '1') != '0':
        # Measurement-gated on RTX 5090: enabling T=1 FP8 globally is a
        # regression, but decoder sites 36..47 reduce graph P50 by ~4-5 ms
        # while preserving the G7.23 cosine gates.
        t1_sites = set(range(36, 48))
    else:
        t1_sites = set()

    def patched_forward(self, x, feat_cache=None, feat_idx=[0]):
        # Shortcut path: FP8 (1,1,1) GEMM if eligible, else original.
        sc_site = getattr(self, '_g7_23_shortcut_site', None)
        if sc_site is not None and sc_site.eligible:
            if _STATE.calibrating:
                m = float(x.abs().max().item())
                if m > sc_site.act_scale_max:
                    sc_site.act_scale_max = m
                h = self.shortcut(x)
            else:
                h = _shortcut_fp8(x, sc_site)
        else:
            h = self.shortcut(x)
        layers = list(self.residual)
        n = len(layers)
        i = 0
        residual_fused = False
        while i < n:
            layer = layers[i]
            site = triplet_to_site.get(i)
            if site is not None:
                # T<_CACHE_T sites stay on bf16 fallback. The math
                # works under FP8 (validated by test_g7_23_fp8_causal_
                # conv3d_unit.py with cache=None which exercises T=1),
                # but launch overhead of the 7-op FP8 chain
                # (fused_quant + cat + v11 + slice + permute +
                # contiguous + bias add) exceeds the per-conv FLOPS
                # savings when input is tiny. Empirically confirmed:
                # forcing T=1 sites onto FP8 took motus P50 from
                # 324.8 ms → 358.6 ms (+34 ms regression). cuDNN
                # BF16 on T=1 small-spatial convs is already near-
                # optimal under CUDA graph replay.
                allow_t1_fp8 = (
                    os.environ.get('FLASH_RT_MOTUS_VAE_FP8_T1', '0') == '1'
                    or site.site_idx in t1_sites)
                use_bf16_fallback = (
                    _STATE.calibrating
                    or not site.eligible
                    or (x.shape[2] < _CACHE_T and not allow_t1_fp8))
                if use_bf16_fallback:
                    cache_x = None
                    if _STATE.calibrating or not site.eligible:
                        n_out = site.rms(x)
                        s = F.silu(n_out)
                    else:
                        gamma_flat = site.rms.gamma.data.view(-1).contiguous()
                        # Wan RMS_norm uses F.normalize(...)*sqrt(C), whose
                        # effective epsilon is far smaller than the FP8
                        # calibration path's 1e-6. Keep BF16 fallback aligned
                        # with the original module.
                        s = _rms_silu_bf16_ncdhw(x, gamma_flat, eps=1e-12)
                    if _STATE.calibrating and site.eligible:
                        m = float(s.abs().max().item())
                        if m > site.act_scale_max:
                            site.act_scale_max = m
                    if feat_cache is not None:
                        idx = feat_idx[0]
                        prev = feat_cache[idx]
                        if cache_x is None:
                            cache_x = _update_cache2_bf16_ncdhw(
                                s, prev if (isinstance(prev, torch.Tensor)
                                            and prev.dtype != _FP8) else None)
                        # During calibration the slot may hold an FP8 tensor
                        # from a prior steady-state run — pass None to
                        # CausalConv3d which would not understand FP8.
                        if isinstance(prev, torch.Tensor) and prev.dtype == _FP8:
                            prev = None
                        use_t1_conv2d = (
                            os.environ.get('FLASH_RT_MOTUS_VAE_T1_CONV2D',
                                           '0') == '1'
                            and site.site_idx in set(range(20, 30))
                            and isinstance(prev, torch.Tensor)
                            and x.shape[2] == 1)
                        if use_t1_conv2d:
                            x = _t1_conv3d_as_conv2d(s, prev, site)
                        else:
                            x = site.conv(s, prev)
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1
                    else:
                        x = site.conv(s)
                    i = triplet_conv_end[i] + 1
                    continue

                # Steady state FP8 fused path.
                gamma_flat = site.rms.gamma.data.view(-1).contiguous()
                cache_fp8 = None
                idx = -1
                is_final_conv = triplet_conv_end[i] == n - 1
                # DIAG knob: force zero-pad cache (loses chunk-context but
                # isolates whether NaN comes from cache flow or single conv).
                _force_zero_cache = (
                    os.environ.get('FLASH_RT_MOTUS_VAE_FP8_NOCACHE',
                                   '0') == '1')
                if feat_cache is not None and not _force_zero_cache:
                    idx = feat_idx[0]
                    prev = feat_cache[idx]
                    if isinstance(prev, torch.Tensor):
                        if prev.dtype == _FP8 and prev.dim() == 5:
                            cache_fp8 = prev
                        elif prev.dtype != _FP8 and prev.dim() == 5:
                            # Transition from bf16 cache (set by a prior
                            # bf16-fallback chunk at this site). Reuse it
                            # by quantizing post-RMS+SiLU snapshot to FP8
                            # NDHWC on the fly using this site's act_scale.
                            # Shape: (B, C, t<=2, H, W) NCDHW bf16.
                            p = prev if prev.is_contiguous() else prev.contiguous()
                            Bp, Cp, Tp, Hp, Wp = p.shape
                            cache_fp8 = torch.empty(
                                Bp, Tp, Hp, Wp, Cp,
                                dtype=_FP8, device=p.device)
                            rc = fvk.bf16_quant_fp8_ncdhw_to_ndhwc(
                                int(p.data_ptr()), int(cache_fp8.data_ptr()),
                                Bp, Cp, Tp, Hp, Wp, float(site.act_scale),
                                torch.cuda.current_stream().cuda_stream)
                            if rc != 0:
                                raise RuntimeError(
                                    f'[g7.53] cache bf16->fp8 rc={rc}')
                y_out, new_cache = _fused_step(
                    x, gamma_flat, site.w_fp8_NDHWC, site.w_scale,
                    site.act_scale, cache_fp8,
                    bias=(site.bias if site.has_bias else None),
                    residual_ncdhw=(h if is_final_conv else None))
                if (is_final_conv and
                        ((os.environ.get('FLASH_RT_MOTUS_NO_VAE_FP8_V18', '0') != '1'
                             and os.environ.get('FLASH_RT_MOTUS_VAE_FP8_V18', '1') == '1')
                         or os.environ.get('FLASH_RT_MOTUS_VAE_FP8_FUSE_RES_TRANSPOSE', '1') == '1')):
                    residual_fused = True
                if feat_cache is not None:
                    feat_cache[idx] = new_cache
                    feat_idx[0] += 1
                x = y_out
                i = triplet_conv_end[i] + 1
                continue

            # Non-triplet layer: replicate upstream cache management for
            # CausalConv3d sites we did NOT swap.
            if (isinstance(layer, CausalConv3d_class)
                    and feat_cache is not None):
                idx = feat_idx[0]
                prev = feat_cache[idx]
                cache_x = _update_cache2_bf16_ncdhw(
                    x, prev if (isinstance(prev, torch.Tensor)
                                and prev.dtype != _FP8) else None)
                if isinstance(prev, torch.Tensor) and prev.dtype == _FP8:
                    prev = None
                x = layer(x, prev)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
            i += 1
        return x if residual_fused else _add_bf16_out(x, h)

    return patched_forward


def install_vae_fp8(model) -> dict:
    """Find ResidualBlocks in the VAE; swap eligible (RMS,SiLU,3x3x3) sites."""
    if os.environ.get('FLASH_RT_MOTUS_NO_G7_23', '0') == '1':
        logger.info('[g7.23] FLASH_RT_MOTUS_NO_G7_23=1, skipping vae_fp8 install')
        return {'skipped': True}

    # Bisection helpers:
    #   FLASH_RT_MOTUS_VAE_FP8_MAX_SITES=N  → only swap first N eligible sites
    #     (in module-tree traversal order). Sites beyond N stay on bf16.
    #   FLASH_RT_MOTUS_VAE_FP8_ONLY_SITES=i,j,k  → swap ONLY these site indices
    #   FLASH_RT_MOTUS_VAE_FP8_SKIP_SITES=i,j,k  → skip (bf16) these indices
    max_sites_env = os.environ.get('FLASH_RT_MOTUS_VAE_FP8_MAX_SITES', '')
    only_sites_env = os.environ.get('FLASH_RT_MOTUS_VAE_FP8_ONLY_SITES', '')
    skip_sites_env = os.environ.get('FLASH_RT_MOTUS_VAE_FP8_SKIP_SITES', '')
    max_sites = int(max_sites_env) if max_sites_env else None
    only_sites = (set(int(x) for x in only_sites_env.split(','))
                  if only_sites_env else None)
    skip_sites = (set(int(x) for x in skip_sites_env.split(','))
                  if skip_sites_env else set())
    vae_mod_name = next(
        (n for n in sys.modules if 'wan' in n and 'vae2_2' in n), None)
    if vae_mod_name is None:
        raise RuntimeError('[g7.23] wan.modules.vae2_2 not loaded')
    vae_mod = sys.modules[vae_mod_name]
    RMS_norm_class = vae_mod.RMS_norm
    CausalConv3d_class = vae_mod.CausalConv3d
    ResidualBlock_class = vae_mod.ResidualBlock
    Up_ResidualBlock_class = getattr(vae_mod, 'Up_ResidualBlock', None)
    DupUp3D_class = getattr(vae_mod, 'DupUp3D', None)

    vae_root = model.video_model.vae.model
    if (os.environ.get('FLASH_RT_MOTUS_NO_G7_52_VAE_RMS', '0') != '1'
            and hasattr(fvk, 'bf16_rms_norm_ncdhw')
            and not getattr(RMS_norm_class, '_g7_52_patched', False)):
        orig_rms_forward = RMS_norm_class.forward

        def rms_forward_fvk(self, x):
            if (getattr(self, 'channel_first', True)
                    and x.is_cuda
                    and x.dtype == torch.bfloat16
                    and x.dim() in (4, 5)):
                gamma = self.gamma.data.view(-1)
                bias = self.bias
                bias_flat = (bias.data.view(-1)
                             if isinstance(bias, torch.Tensor) else None)
                return _rms_norm_bf16_ncdhw(
                    x, gamma, bias_flat, eps=1e-12)
            return orig_rms_forward(self, x)

        RMS_norm_class.forward = rms_forward_fvk
        RMS_norm_class._g7_52_patched = True

    if (os.environ.get('FLASH_RT_MOTUS_VAE_FP8_HEAD', '0') == '1'
            and hasattr(fvk, 'fp8_conv3d_v17_anyco_ndhwc_bf16out')):
        head_conv = getattr(getattr(vae_root.decoder, 'head', None), '2', None)
        if head_conv is None and hasattr(vae_root.decoder, 'head'):
            try:
                head_conv = vae_root.decoder.head[2]
            except Exception:
                head_conv = None
        if head_conv is not None:
            head_site = _HeadFp8Site(head_conv)
            head_conv._g7_27_head_fp8_site = head_site

            def head_forward(x, cache_x=None):
                if _STATE.calibrating:
                    m = float(x.abs().max().item())
                    if m > head_site.act_scale_max:
                        head_site.act_scale_max = m
                    return head_site.orig_forward(x, cache_x)
                return _head_conv_fp8(x, cache_x, head_site)

            head_conv.forward = head_forward

    blocks = [(n, m) for n, m in vae_root.named_modules()
              if isinstance(m, ResidualBlock_class)]

    n_eligible = 0
    n_skipped = 0
    n_blocks_swapped = 0
    skip_reasons = {}
    global_site_idx = 0
    site_inventory = []

    for name, block in blocks:
        layers = list(block.residual)
        sites: List[_Fp8Site] = []
        triplets_in_block = _find_triplets(
            layers, RMS_norm_class, CausalConv3d_class)
        for (rms_idx, silu_idx, conv_idx, _end_idx) in triplets_in_block:
            label = (f'#{global_site_idx:02d} {name}.residual'
                     f'[{rms_idx},{silu_idx},{conv_idx}]')
            site = _Fp8Site(
                layers[rms_idx], layers[conv_idx], label, global_site_idx)
            # Apply bisection mask: keep eligible only if site idx
            # passes the env-var filter.
            if site.eligible:
                keep = True
                if only_sites is not None and global_site_idx not in only_sites:
                    keep = False
                if global_site_idx in skip_sites:
                    keep = False
                if max_sites is not None and global_site_idx >= max_sites:
                    keep = False
                if not keep:
                    site.eligible = False
                    site.reason_skipped = 'env_filter'
            sites.append(site)
            if site.eligible:
                n_eligible += 1
            else:
                n_skipped += 1
                skip_reasons[site.reason_skipped] = (
                    skip_reasons.get(site.reason_skipped, 0) + 1)
            site_inventory.append(
                f'{label}  Ci={layers[conv_idx].in_channels} '
                f'Co={layers[conv_idx].out_channels}  '
                f'eligible={site.eligible}')
            global_site_idx += 1

        # Detect (1,1,1) shortcut conv on this block.
        sc = getattr(block, 'shortcut', None)
        if (sc is not None and isinstance(sc, CausalConv3d_class)):
            sc_label = f'#sc {name}.shortcut'
            sc_site = _Fp8ShortcutSite(sc, sc_label)
            block._g7_23_shortcut_site = sc_site
            site_inventory.append(
                f'{sc_label}  Ci={sc.in_channels} Co={sc.out_channels}  '
                f'eligible={sc_site.eligible} (kernel='
                f'{tuple(sc.kernel_size)})')

        if not sites:
            continue
        # Always patch (even if all sites bf16) so cache contract is
        # uniform; bf16-fallback is a no-op semantic-wise.
        n_blocks_swapped += 1
        block._g7_23_sites = sites
        patched = _build_patched_forward(
            block, sites, RMS_norm_class, CausalConv3d_class)
        block.forward = patched.__get__(block, type(block))

    n_up_blocks_patched = 0
    if (Up_ResidualBlock_class is not None
            and os.environ.get('FLASH_RT_MOTUS_NO_G7_35_VAE_UP_FUSE',
                               '0') != '1'):
        for _name, up_block in vae_root.named_modules():
            if isinstance(up_block, Up_ResidualBlock_class):
                up_block.forward = _build_up_residual_forward().__get__(
                    up_block, type(up_block))
                n_up_blocks_patched += 1
    n_dupup_patched = 0
    if DupUp3D_class is not None:
        for _name, dup in vae_root.named_modules():
            if isinstance(dup, DupUp3D_class):
                dup.forward = _build_dup_up3d_forward().__get__(
                    dup, type(dup))
                n_dupup_patched += 1

    if os.environ.get('FLASH_RT_MOTUS_VAE_FP8_LIST_SITES', '0') == '1':
        print('[g7.23] === site inventory ===', flush=True)
        for s in site_inventory:
            print(f'[g7.23.site] {s}', flush=True)
        print(f'[g7.23] === total {len(site_inventory)} sites; '
              f'{n_eligible} eligible, {n_skipped} skipped ===',
              flush=True)

    stats = dict(
        n_blocks_total=len(blocks),
        n_blocks_swapped=n_blocks_swapped,
        n_eligible_sites=n_eligible,
        n_skipped_sites=n_skipped,
        n_up_blocks_patched=n_up_blocks_patched,
        n_dupup_patched=n_dupup_patched,
        skip_reasons=skip_reasons,
    )
    logger.info(f'[g7.23] vae_fp8_swap install: {stats}')
    return stats


def commit_calibration(model) -> dict:
    """After 1 calibration inference, freeze act_scale per site."""
    n = 0
    samples = []
    for name, mod in model.video_model.vae.model.named_modules():
        sites = getattr(mod, '_g7_23_sites', None)
        if sites:
            for site in sites:
                if site.eligible and site.act_scale_max > 0:
                    site.act_scale = max(
                        site.act_scale_max / 448.0, 1e-12)
                    samples.append((site.label, site.act_scale_max,
                                    site.act_scale))
                    n += 1
        sc_site = getattr(mod, '_g7_23_shortcut_site', None)
        if sc_site is not None and sc_site.eligible:
            if sc_site.act_scale_max > 0:
                sc_site.act_scale_host = max(
                    sc_site.act_scale_max / 448.0, 1e-12)
                sc_site.act_scale_dev.fill_(sc_site.act_scale_host)
                samples.append((sc_site.label, sc_site.act_scale_max,
                                sc_site.act_scale_host))
                n += 1
        head_site = getattr(mod, '_g7_27_head_fp8_site', None)
        if head_site is not None and head_site.act_scale_max > 0:
            head_site.act_scale = max(head_site.act_scale_max / 448.0, 1e-12)
            samples.append(('decoder.head.2', head_site.act_scale_max,
                            head_site.act_scale))
            n += 1
    logger.info(f'[g7.23] vae_fp8 calibrated {n} sites')
    return dict(n=n, samples=samples)

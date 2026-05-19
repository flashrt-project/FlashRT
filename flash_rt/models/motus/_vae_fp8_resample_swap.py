"""FP8 Conv2d 3x3 swap for motus VAE Resample.1 sites.

Per-site profiling found 3 BF16 cudnn Conv2d 3x3 sites in the decoder
upsample chain consuming 13.5 ms / inference total:
  decoder.upsamples.0.upsamples.3.resample.1  Conv2d 3x3 1024→1024 @ 48×40
  decoder.upsamples.1.upsamples.3.resample.1  Conv2d 3x3 1024→1024 @ 96×80
  decoder.upsamples.2.upsamples.3.resample.1  Conv2d 3x3  512→512  @ 192×160

These are NOT in G7.23's scope (G7.23 only walks ResidualBlock
internal residual triplets). They are nn.Conv2d modules inside Resample.

This swap:
  1) walks `vae.decoder.upsamples[*].upsamples[3].resample[1]`
  2) calibrates input |amax| in calibration mode
  3) commits to FP8: pre-transposes weight to (Co, kR, kS, Ci) NHWC
     layout, FP8 quantizes per-tensor with w_scale; bias kept BF16
  4) replaces the Conv2d forward with: NCHW→NHWC permute, FP8 quant
     input, fvk.fp8_conv2d_3x3_v1_nhwc_bf16out, permute back to NCHW

Numerics: cos floor 0.998 standalone; e2e cos floor matches K7
(cos(a)≥0.995, cos(f)≥0.997).

Toggle: enabled by default; ``FLASH_RT_MOTUS_NO_FP8_RESAMPLE=1`` disables.
"""
from __future__ import annotations

import logging
import os
from typing import List, Tuple

import torch
import torch.nn as nn

from flash_rt.models.motus._stream import cs
import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0


class _Conv2dFp8Site:
    """Per-site FP8 state for a Conv2d 3x3 stride=1 padding=1 BF16 module."""
    __slots__ = (
        'name', 'mod', 'Ci', 'Co', 'has_bias', 'orig_forward',
        'w_fp8', 'w_scale', 'act_scale', 'bias_bf16',
        'act_amax_bf16',     # calibration accumulator (max(|input|))
        'committed', 'eligible', 'reason_skipped',
        'act_scale_scalar',
        'alpha_scalar',      # Python float = act_scale * w_scale (graph-safe)
    )

    def __init__(self, name: str, mod: nn.Conv2d):
        self.name = name
        self.mod = mod
        self.Ci = int(mod.in_channels)
        self.Co = int(mod.out_channels)
        self.has_bias = mod.bias is not None
        self.orig_forward = mod.forward
        # Eligibility: kernel must be (3,3), stride=1, padding=1, Ci%32==0, Co%8==0
        ks = tuple(mod.kernel_size)
        st = tuple(mod.stride)
        pd = tuple(mod.padding)
        ok = (ks == (3, 3) and st == (1, 1) and pd == (1, 1)
              and self.Ci % 32 == 0 and self.Co % 8 == 0)
        self.eligible = ok
        self.reason_skipped = (None if ok else
            f'kernel={ks} stride={st} pad={pd} Ci={self.Ci} Co={self.Co}')
        self.committed = False
        self.w_fp8 = None
        self.w_scale = None
        self.act_scale = None
        self.act_scale_scalar = 1.0
        self.bias_bf16 = None
        self.act_amax_bf16 = torch.zeros(1, dtype=torch.bfloat16,
                                          device=mod.weight.device)


# Module-level state holders for calibrate / commit sequencing.
_SITES: List[_Conv2dFp8Site] = []
_CALIBRATING = False
_CACHE_T = 2
_CUDNN_FP8_CONV2D_BAD_SHAPES: set[tuple[int, int, int, int, int]] = set()


def _fp8_conv2d_3x3_nhwc_bf16out(
    x_fp8: int,
    w_fp8: int,
    y_bf16: int,
    bias_bf16: int,
    N: int,
    H: int,
    W: int,
    Ci: int,
    Co: int,
    alpha: float,
    stream: int,
) -> int:
    shape = (int(N), int(H), int(W), int(Ci), int(Co))
    if (os.environ.get('FLASH_RT_MOTUS_USE_FP8_CONV2D_V2', '0') == '1'
            and hasattr(fvk, 'fp8_conv2d_3x3_v2_nhwc_bf16out')):
        return fvk.fp8_conv2d_3x3_v2_nhwc_bf16out(
            x_fp8=x_fp8,
            w_fp8=w_fp8,
            y_bf16=y_bf16,
            bias_bf16=bias_bf16,
            N=N, H=H, W=W, Ci=Ci, Co=Co,
            alpha=alpha, stream=stream)
    if (os.environ.get('FLASH_RT_MOTUS_NO_CUDNN_FP8_CONV2D', '0') != '1'
            and shape not in _CUDNN_FP8_CONV2D_BAD_SHAPES
            and hasattr(fvk, 'cudnn_fp8_conv2d_3x3_nhwc_bf16out')):
        rc = fvk.cudnn_fp8_conv2d_3x3_nhwc_bf16out(
            x_fp8=x_fp8,
            w_fp8=w_fp8,
            y_bf16=y_bf16,
            bias_bf16=bias_bf16,
            N=N, H=H, W=W, Ci=Ci, Co=Co,
            alpha=alpha, stream=stream)
        if rc == 0:
            return 0
        _CUDNN_FP8_CONV2D_BAD_SHAPES.add(shape)
    return fvk.fp8_conv2d_3x3_v1_nhwc_bf16out(
        x_fp8=x_fp8,
        w_fp8=w_fp8,
        y_bf16=y_bf16,
        bias_bf16=bias_bf16,
        N=N, H=H, W=W, Ci=Ci, Co=Co,
        alpha=alpha, stream=stream)


def _fp8_conv2d_3x3_nhwc_ncdhw_bf16out(
    x_fp8: int,
    w_fp8: int,
    y_bf16: int,
    bias_bf16: int,
    B: int,
    T: int,
    H: int,
    W: int,
    Ci: int,
    Co: int,
    alpha: float,
    stream: int,
) -> int:
    if (os.environ.get('FLASH_RT_MOTUS_USE_FP8_CONV2D_V2_NCDHWOUT', '0') == '1'
            and os.environ.get('FLASH_RT_MOTUS_USE_FP8_CONV2D_V2', '0') == '1'
            and hasattr(fvk, 'fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out')):
        return fvk.fp8_conv2d_3x3_v2_nhwc_ncdhw_bf16out(
            x_fp8=x_fp8,
            w_fp8=w_fp8,
            y_bf16=y_bf16,
            bias_bf16=bias_bf16,
            B=B, T=T, H=H, W=W, Ci=Ci, Co=Co,
            alpha=alpha, stream=stream)
    return -100


def set_calibrating(flag: bool):
    global _CALIBRATING
    _CALIBRATING = flag


def _update_cache2_bf16_ncdhw(cur: torch.Tensor, prev) -> torch.Tensor:
    B, C, T, H, W = cur.shape
    out = torch.empty(B, C, _CACHE_T, H, W,
                      dtype=cur.dtype, device=cur.device)
    prev_ptr = 0
    if (isinstance(prev, torch.Tensor)
            and prev.dtype == cur.dtype
            and prev.dim() == 5):
        prev_ptr = int(prev.data_ptr())
    fvk.update_cache2_ncdhw_bf16(
        int(cur.data_ptr()), prev_ptr, int(out.data_ptr()),
        int(B), int(C), int(T), int(H), int(W), cs())
    return out


def _time_unshuffle2_bf16(x: torch.Tensor, b: int, c: int,
                          t: int, h: int, w: int) -> torch.Tensor:
    if (x.dtype == torch.bfloat16
            and hasattr(fvk, 'time_unshuffle2_bf16')
            and os.environ.get('FLASH_RT_MOTUS_USE_G7_43_TIME_UNSHUFFLE',
                               '0') == '1'):
        x_c = x if x.is_contiguous() else x.contiguous()
        out = torch.empty(b, c, t * 2, h, w,
                          dtype=torch.bfloat16, device=x_c.device)
        fvk.time_unshuffle2_bf16(
            int(x_c.data_ptr()), int(out.data_ptr()),
            int(b), int(c), int(t), int(h), int(w), cs())
        return out

    x = x.reshape(b, 2, c, t, h, w)
    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
    return x.reshape(b, c, t * 2, h, w)


def _make_calib_forward(site: _Conv2dFp8Site):
    """Forward that records input |amax| during calibration; passes
    through to the original cudnn Conv2d."""
    orig = site.orig_forward

    def calib_forward(x: torch.Tensor) -> torch.Tensor:
        if _CALIBRATING:
            m = x.abs().amax().to(torch.bfloat16)
            cur = site.act_amax_bf16
            cur.copy_(torch.maximum(cur, m))
        return orig(x)
    return calib_forward


def _make_fp8_forward(site: _Conv2dFp8Site):
    """Forward that runs the FP8 conv2d 3x3 path. Graph-capture safe:
    no .item() / no host syncs. act_scale is read device-side via
    fvk.quantize_fp8_static; alpha is a Python scalar fixed at commit."""
    Ci = site.Ci
    Co = site.Co
    w_ptr = int(site.w_fp8.data_ptr())
    bias_ptr = (int(site.bias_bf16.data_ptr())
                 if site.bias_bf16 is not None else 0)
    act_scale_ptr = int(site.act_scale.data_ptr())
    act_scale_scalar = float(site.act_scale.item())
    alpha_scalar = site.alpha_scalar   # Python float, fixed

    def fp8_forward(x: torch.Tensor) -> torch.Tensor:
        # Input x is (B, Ci, H, W) BF16 NCHW.
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()
        B, _Ci_x, H, W = x.shape
        if (hasattr(fvk, 'bf16_quant_fp8_ncdhw_to_ndhwc')
                and hasattr(fvk, 'bf16_ndhwc_to_ncdhw_transpose')
                and os.environ.get('FLASH_RT_MOTUS_NO_G7_30_RESAMPLE_LAYOUT',
                                   '0') != '1'):
            x_fp8 = torch.empty(B, 1, H, W, Ci, dtype=_FP8, device=x.device)
            rcq = fvk.bf16_quant_fp8_ncdhw_to_ndhwc(
                int(x.data_ptr()), int(x_fp8.data_ptr()),
                B, Ci, 1, H, W, act_scale_scalar, cs())
            if rcq != 0:
                raise RuntimeError(
                    f'fp8_resample {site.name}: ncdhw->ndhwc quant rc={rcq}')
            y_nchw = torch.empty(B, Co, 1, H, W, dtype=torch.bfloat16,
                                  device=x.device)
            rc_direct = _fp8_conv2d_3x3_nhwc_ncdhw_bf16out(
                x_fp8=int(x_fp8.data_ptr()),
                w_fp8=w_ptr,
                y_bf16=int(y_nchw.data_ptr()),
                bias_bf16=bias_ptr,
                B=B, T=1, H=H, W=W, Ci=Ci, Co=Co,
                alpha=alpha_scalar, stream=cs())
            if rc_direct == 0:
                return y_nchw.view(B, Co, H, W)
            if rc_direct != -100:
                raise RuntimeError(
                    f'fp8_conv2d_3x3 site {site.name}: ncdhwout rc={rc_direct}')
            y_nhwc = torch.empty(B, 1, H, W, Co, dtype=torch.bfloat16,
                                  device=x.device)
        else:
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
            n = B * H * W * Ci
            x_fp8 = torch.empty(B, H, W, Ci, dtype=_FP8, device=x.device)
            fvk.quantize_fp8_static(
                int(x_nhwc.data_ptr()), int(x_fp8.data_ptr()),
                act_scale_ptr, n, cs())
            y_nhwc = torch.empty(B, H, W, Co, dtype=torch.bfloat16,
                                  device=x.device)
            y_nchw = None
        rc = _fp8_conv2d_3x3_nhwc_bf16out(
            x_fp8=int(x_fp8.data_ptr()),
            w_fp8=w_ptr,
            y_bf16=int(y_nhwc.data_ptr()),
            bias_bf16=bias_ptr,
            N=B, H=H, W=W, Ci=Ci, Co=Co,
            alpha=alpha_scalar, stream=cs())
        if rc != 0:
            raise RuntimeError(
                f'fp8_conv2d_3x3 site {site.name}: rc={rc}')
        if y_nchw is not None:
            rct = fvk.bf16_ndhwc_to_ncdhw_transpose(
                int(y_nhwc.data_ptr()), int(y_nchw.data_ptr()),
                B, Co, 1, H, W, cs())
            if rct != 0:
                raise RuntimeError(
                    f'fp8_resample {site.name}: ndhwc->ncdhw rc={rct}')
            return y_nchw.view(B, Co, H, W)
        return y_nhwc.permute(0, 3, 1, 2).contiguous()

    return fp8_forward


def _fp8_upsample2x_conv2d_forward(
    site: _Conv2dFp8Site,
    x: torch.Tensor,
    B5: int,
    T5: int,
) -> torch.Tensor:
    """Fused nearest-2x spatial upsample + FP8 quant + Conv2d.

    Input x is flattened NCHW with N=B*T. Output is restored to
    NCDHW [B, Co, T, 2H, 2W] using the existing NDHWC→NCDHW kernel.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x_c = x if x.is_contiguous() else x.contiguous()
    N, Ci, H, W = x_c.shape
    Co = site.Co
    H2 = H << 1
    W2 = W << 1
    x_fp8 = torch.empty(N, H2, W2, Ci, dtype=_FP8, device=x_c.device)
    rcq = fvk.bf16_upsample2x_quant_fp8_nchw_to_nhwc(
        int(x_c.data_ptr()), int(x_fp8.data_ptr()),
        int(N), int(Ci), int(H), int(W), float(site.act_scale_scalar), cs())
    if rcq != 0:
        raise RuntimeError(
            f'fp8_resample {site.name}: upsample2x_quant rc={rcq}')
    y_ncdhw = torch.empty(B5, Co, T5, H2, W2, dtype=torch.bfloat16,
                          device=x_c.device)
    bias_ptr = (int(site.bias_bf16.data_ptr())
                if site.bias_bf16 is not None else 0)
    rc = _fp8_conv2d_3x3_nhwc_ncdhw_bf16out(
        x_fp8=int(x_fp8.data_ptr()),
        w_fp8=int(site.w_fp8.data_ptr()),
        y_bf16=int(y_ncdhw.data_ptr()),
        bias_bf16=bias_ptr,
        B=int(B5), T=int(T5), H=int(H2), W=int(W2), Ci=int(Ci), Co=int(Co),
        alpha=float(site.alpha_scalar), stream=cs())
    if rc == 0:
        return y_ncdhw
    if rc != -100:
        raise RuntimeError(
            f'fp8_conv2d_3x3 site {site.name}: fused-up ncdhwout rc={rc}')
    y_nhwc = torch.empty(N, H2, W2, Co, dtype=torch.bfloat16,
                         device=x_c.device)
    rc = _fp8_conv2d_3x3_nhwc_bf16out(
        x_fp8=int(x_fp8.data_ptr()),
        w_fp8=int(site.w_fp8.data_ptr()),
        y_bf16=int(y_nhwc.data_ptr()),
        bias_bf16=bias_ptr,
        N=int(N), H=int(H2), W=int(W2), Ci=int(Ci), Co=int(Co),
        alpha=float(site.alpha_scalar), stream=cs())
    if rc != 0:
        raise RuntimeError(
            f'fp8_conv2d_3x3 site {site.name}: fused-up rc={rc}')
    rct = fvk.bf16_ndhwc_to_ncdhw_transpose(
        int(y_nhwc.data_ptr()), int(y_ncdhw.data_ptr()),
        int(B5), int(Co), int(T5), int(H2), int(W2), cs())
    if rct != 0:
        raise RuntimeError(
            f'fp8_resample {site.name}: fused-up transpose rc={rct}')
    return y_ncdhw


def _make_resample_forward(mod):
    orig = mod.forward
    conv = None
    try:
        conv = mod.resample[1]
    except Exception:
        conv = None
    site = getattr(conv, '_fp8_resample_site', None)
    if site is None:
        return orig

    def fp8_resample_forward(self, x, feat_cache=None, feat_idx=[0]):
        if (not site.committed
                or os.environ.get('FLASH_RT_MOTUS_NO_G7_37_UP2X_QUANT',
                                  '0') == '1'
                or mod.mode not in ('upsample2d', 'upsample3d')
                or not hasattr(fvk, 'bf16_upsample2x_quant_fp8_nchw_to_nhwc')):
            return orig(x, feat_cache, feat_idx)

        b, c, t, h, w = x.size()
        if mod.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    prev = feat_cache[idx]
                    if os.environ.get('FLASH_RT_MOTUS_USE_G7_43_RESAMPLE_CACHE',
                                      '0') == '1':
                        cache_x = _update_cache2_bf16_ncdhw(
                            x, None if prev == 'Rep' else prev)
                    else:
                        cache_x = x[:, :, -_CACHE_T:, :, :].clone()
                        if (cache_x.shape[2] < 2
                                and prev is not None
                                and prev != 'Rep'):
                            cache_x = torch.cat(
                                [prev[:, :, -1:, :, :].to(cache_x.device),
                                 cache_x], dim=2)
                        if (cache_x.shape[2] < 2
                                and prev is not None
                                and prev == 'Rep'):
                            cache_x = torch.cat(
                                [torch.zeros_like(cache_x).to(cache_x.device),
                                 cache_x], dim=2)
                    if prev == 'Rep':
                        x = mod.time_conv(x)
                    else:
                        x = mod.time_conv(x, prev)
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = _time_unshuffle2_bf16(x, b, c, t, h, w)

        t2 = x.shape[2]
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * t2, c, h, w)
        return _fp8_upsample2x_conv2d_forward(site, x2d, b, t2)

    return fp8_resample_forward


def install_vae_fp8_resample(model) -> dict:
    """Install FP8 Conv2d 3x3 calibration hooks on resample.1 sites.

    Must be called BEFORE the calibration forward. After install, run
    one forward with set_calibrating(True), then call
    commit_calibration_resample() to switch to FP8 forward.
    """
    counts = {'installed': 0, 'skipped': 0, 'reasons': {}}
    if (os.environ.get('FLASH_RT_MOTUS_NO_FP8_RESAMPLE', '0') == '1'
            or os.environ.get('FLASH_RT_MOTUS_USE_FP8_RESAMPLE', '1') == '0'):
        logger.info('[fp8_resample] disabled by env')
        return counts

    vae_root = model.video_model.vae.model

    # Walk and find all `resample.1` Conv2d modules under
    # decoder.upsamples.* (and optionally encoder.downsamples.* later).
    targets: List[Tuple[str, nn.Conv2d]] = []
    for name, mod in vae_root.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        if 'decoder.upsamples' not in name:
            continue
        if not name.endswith('.resample.1'):
            continue
        targets.append((name, mod))

    if not targets:
        logger.info('[fp8_resample] no targets found')
        return counts

    for name, mod in targets:
        site = _Conv2dFp8Site(name, mod)
        if not site.eligible:
            counts['skipped'] += 1
            counts['reasons'][site.reason_skipped] = (
                counts['reasons'].get(site.reason_skipped, 0) + 1)
            logger.info(
                f'[fp8_resample] SKIP {name}: {site.reason_skipped}')
            continue
        # Patch forward to calibration mode for now.
        mod.forward = _make_calib_forward(site)
        mod._fp8_resample_site = site
        _SITES.append(site)
        counts['installed'] += 1

    logger.info(
        f'[fp8_resample] installed calibration hooks on '
        f'{counts["installed"]} sites (skipped {counts["skipped"]})')
    return counts


def commit_calibration_resample(model) -> dict:
    """After calibration forward, finalize FP8 conversion and patch
    forward to the FP8 path.

    Reads each site's act_amax_bf16, computes act_scale = max/_FP8_MAX,
    pre-transposes + FP8-quantizes weight (Co, 3, 3, Ci) NHWC layout
    with w_scale = max(|w|)/FP8_MAX, kept bias as BF16.
    """
    n_committed = 0
    for site in _SITES:
        if site.committed:
            continue
        m = site.mod
        # Compute act_scale.
        amax = float(site.act_amax_bf16.item())
        if amax <= 0:
            logger.warning(
                f'[fp8_resample] {site.name} amax==0 — leaving BF16')
            continue
        s_act = max(amax / _FP8_MAX, 1e-6)
        site.act_scale = torch.tensor([s_act], dtype=torch.float32,
                                       device=m.weight.device)
        site.act_scale_scalar = float(s_act)
        # Quantize weight to FP8 with per-tensor scale.
        w = m.weight.data        # (Co, Ci, 3, 3) BF16
        if w.dtype != torch.bfloat16:
            w = w.to(torch.bfloat16)
        # Permute (Co, Ci, kR, kS) → (Co, kR, kS, Ci) for kernel layout.
        w_okkc = w.permute(0, 2, 3, 1).contiguous()
        w_max = float(w_okkc.abs().amax().item())
        s_w = max(w_max / _FP8_MAX, 1e-6)
        w_fp8 = (w_okkc.float() / s_w).clamp(-_FP8_MAX, _FP8_MAX).to(
            _FP8).contiguous()
        site.w_fp8 = w_fp8
        site.w_scale = torch.tensor([s_w], dtype=torch.float32,
                                     device=m.weight.device)
        # Pre-compute alpha as Python scalar (graph-capture safe — no
        # device→host sync needed inside captured forward).
        site.alpha_scalar = float(s_act) * float(s_w)
        # Bias BF16.
        if site.has_bias:
            b = m.bias.data
            if b.dtype != torch.bfloat16:
                b = b.to(torch.bfloat16)
            site.bias_bf16 = b.contiguous()
        # Swap forward to FP8 path.
        m.forward = _make_fp8_forward(site)
        site.committed = True
        n_committed += 1
        logger.info(
            f'[fp8_resample] committed {site.name}  Ci={site.Ci} '
            f'Co={site.Co}  act_scale={s_act:.5f}  w_scale={s_w:.5f}')

    n_fused_resample = 0
    vae_root = model.video_model.vae.model
    for _name, mod in vae_root.named_modules():
        if not hasattr(mod, 'mode') or not hasattr(mod, 'resample'):
            continue
        if getattr(mod, 'mode', None) not in ('upsample2d', 'upsample3d'):
            continue
        try:
            site = getattr(mod.resample[1], '_fp8_resample_site', None)
        except Exception:
            site = None
        if site is not None and site.committed:
            mod.forward = _make_resample_forward(mod).__get__(mod, type(mod))
            n_fused_resample += 1
    return {'committed': n_committed, 'fused_resample': n_fused_resample}

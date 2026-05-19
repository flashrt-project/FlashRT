"""NVFP4 W4A16 swap for motus video FFN sites (Wan FFN up + down).

Per-shape perf probe (probe_nvfp4_vs_fp8_motus_shapes.py) found:
  video_ffn_up   M=360 N=14336 K=3072    fp8=67us → nvfp4=32us  (2.12x)
  video_ffn_down M=360 N= 3072 K=14336   fp8=83us → nvfp4=47us  (1.76x)

× 300 layer-step / inference: -10.5ms (up) + -10.8ms (down) = ~-21ms
expected save before NVFP4 input-quant overhead (~6ms total ~6us × 600).
Net target: -15 ms graph-mode wall on top of G7.23 baseline 286ms.

Approach: replace each Wan video block's ffn.forward (currently G3d's
_make_fp8_ffn_forward) with an NVFP4 forward path:
  BF16 input → NVFP4 quant (packed + SF swz) → cutlass NVFP4 GEMM
  → BF16 up_out → bias_gelu_fused (existing fvk) → NVFP4 quant
  → cutlass NVFP4 GEMM → BF16 dn_out → add_bias.

Install requirements:
  - G3d FP8 swap must already be installed (we read site.w_fp8 + w_scale
    to reconstruct BF16 weights via dequant).
  - K dimension multiple of 16 (NVFP4 group size). All motus video FFN
    K sizes (3072, 14336) satisfy.

Toggle: ``FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO=1``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from flash_rt.models.motus._stream import cs
import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)


def _selected_layers() -> Optional[set[int]]:
    spec = os.environ.get('FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_LAYERS', '').strip()
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _layer_set_from_env(name: str) -> set[int]:
    spec = os.environ.get(name, '').strip()
    if not spec:
        return set()
    out: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _int_set_from_env(name: str) -> set[int]:
    spec = os.environ.get(name, '').strip()
    if not spec:
        return set()
    out: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _swizzled_sf_bytes(rows: int, cols: int) -> int:
    """Return SF buffer byte size for (rows, cols) NVFP4 swizzled layout.
    cols must be multiple of 16. Layout is super-atoms of 128 rows × 4
    sf-cols (i.e., 4 × 16 element-cols), each super-atom = 128 × 64 bytes.
    """
    assert cols % 16 == 0, f'cols={cols} must be %16==0'
    n_blocks = cols // 16
    n_row_super = (rows + 127) // 128
    n_col_super = (n_blocks + 3) // 4
    return n_row_super * n_col_super * 128 * 64


def quantize_weight_bf16_to_nvfp4_swz(w_bf16_NK: torch.Tensor):
    """BF16 weight (N, K) → persistent NVFP4 packed + swizzled SFB."""
    return _quantize_bf16_to_nvfp4_swz(w_bf16_NK)


def _quantize_bf16_to_nvfp4_swz(x_bf16: torch.Tensor):
    """BF16 (rows, cols) → (packed_u8, sf_swz_u8)."""
    rows, cols = x_bf16.shape
    dev = x_bf16.device
    packed = torch.empty(rows, cols // 2, dtype=torch.uint8, device=dev)
    sf = torch.zeros(_swizzled_sf_bytes(rows, cols), dtype=torch.uint8,
                      device=dev)
    fvk.quantize_bf16_to_nvfp4_swizzled(
        int(x_bf16.data_ptr()), int(packed.data_ptr()),
        int(sf.data_ptr()), rows, cols, cs())
    return packed, sf


class _Nvfp4VideoFfnState:
    __slots__ = (
        'up_w_packed', 'up_w_sf', 'dn_w_packed', 'dn_w_sf',
        'up_inv_s', 'dn_inv_s', 'up_clip_group_amax', 'dn_clip_group_amax',
        'up_alpha', 'dn_alpha',
        'up_bias', 'dn_bias', 'dn_site', 'K', 'F', 'M_capacity',
        'up_in_packed', 'up_in_sf', 'up_out',
        'dn_in_packed', 'dn_in_sf', 'dn_out')

    def __init__(self, up_w_packed, up_w_sf, dn_w_packed, dn_w_sf,
                 up_inv_s, dn_inv_s, up_clip_group_amax, dn_clip_group_amax,
                 up_alpha, dn_alpha,
                 up_bias, dn_bias, dn_site, K: int, F: int,
                 M_capacity: int, dev: torch.device):
        self.up_w_packed = up_w_packed
        self.up_w_sf = up_w_sf
        self.dn_w_packed = dn_w_packed
        self.dn_w_sf = dn_w_sf
        self.up_inv_s = up_inv_s
        self.dn_inv_s = dn_inv_s
        self.up_clip_group_amax = up_clip_group_amax
        self.dn_clip_group_amax = dn_clip_group_amax
        self.up_alpha = float(up_alpha)
        self.dn_alpha = float(dn_alpha)
        self.up_bias = up_bias
        self.dn_bias = dn_bias
        self.dn_site = dn_site
        self.K = K
        self.F = F
        self.M_capacity = int(M_capacity)
        M = self.M_capacity
        self.up_in_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=dev)
        self.up_in_sf = torch.zeros(_swizzled_sf_bytes(M, K),
                                    dtype=torch.uint8, device=dev)
        self.up_out = torch.empty(M, F, dtype=torch.bfloat16, device=dev)
        self.dn_in_packed = torch.empty(M, F // 2, dtype=torch.uint8, device=dev)
        self.dn_in_sf = torch.zeros(_swizzled_sf_bytes(M, F),
                                    dtype=torch.uint8, device=dev)
        self.dn_out = torch.empty(M, K, dtype=torch.bfloat16, device=dev)


def _smoothquant_nvfp4_from_fp8_site(site, alpha: float):
    """Build AWQ/SmoothQuant-folded NVFP4 weight from an already-calibrated
    FP8 site. Returns (packed, sf, inv_s) or (None, None, None) if no
    activation stats were collected.
    """
    act = getattr(site, 'nvfp4_awq_act_amax_K', None)
    if act is None or float(act.max().item()) <= 0.0:
        return None, None, None
    native_w = getattr(site, 'nvfp4_w_bf16_cpu', None)
    if native_w is not None:
        w = native_w.to(device=site.w_fp8.device, dtype=torch.bfloat16,
                        non_blocking=True)
    else:
        w = site.w_fp8.to(torch.bfloat16) * float(site.w_scale.item())  # (K, N)
    eps = 1e-5
    a = act.float().clamp(min=eps)
    style = os.environ.get(
        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_STYLE',
        'pi05').strip().lower()
    if style == 'pi05':
        # Pi0.5/Thor production FP4 path: activation-aware per-input
        # balancing only, clipped to avoid over-aggressive channel scaling.
        # W' = W * s, x' = x / s, so math is equivalent before FP4 quant.
        s = (a / a.mean().clamp(min=eps)).pow(alpha)
        clamp_min = float(os.environ.get(
            'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_CLAMP_MIN', '0.25'))
        clamp_max = float(os.environ.get(
            'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_CLAMP_MAX', '4.0'))
        s = s.clamp(min=clamp_min, max=clamp_max)
    else:
        # Legacy Motus experiment: activation/weight SmoothQuant ratio.
        w_amax_K = w.abs().amax(dim=1).float().clamp(min=eps)
        s = (a.pow(alpha) / w_amax_K.pow(1.0 - alpha)).clamp(min=eps)
        s = s / s.log().mean().exp()
    inv_s = (1.0 / s).to(torch.bfloat16).contiguous()
    w_scaled_NK = (w.float() * s.unsqueeze(1)).to(torch.bfloat16).t().contiguous()
    packed, sf = quantize_weight_bf16_to_nvfp4_swz(w_scaled_NK)
    return packed, sf, inv_s


def _quantize_act_to_nvfp4(flat: torch.Tensor, inv_s: Optional[torch.Tensor],
                           packed: torch.Tensor, sf: torch.Tensor,
                           rows: int, cols: int,
                           clip_group_amax: Optional[torch.Tensor] = None) -> None:
    if inv_s is not None:
        fvk.awq_quant_bf16_to_nvfp4_swizzled(
            int(flat.data_ptr()), int(inv_s.data_ptr()),
            int(packed.data_ptr()), int(sf.data_ptr()),
            rows, cols, cs())
    elif (os.environ.get(
            'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_SECONDMAX_Q', '0') == '1'
            and hasattr(fvk, 'quantize_bf16_to_nvfp4_swizzled_secondmax')):
        fvk.quantize_bf16_to_nvfp4_swizzled_secondmax(
            int(flat.data_ptr()), int(packed.data_ptr()), int(sf.data_ptr()),
            rows, cols,
            float(os.environ.get(
                'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_SECONDMAX_SCALE', '1.0')),
            cs())
    elif (os.environ.get(
            'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_MSE_Q', '0') == '1'
            and hasattr(fvk, 'quantize_bf16_to_nvfp4_swizzled_mse')):
        fvk.quantize_bf16_to_nvfp4_swizzled_mse(
            int(flat.data_ptr()), int(packed.data_ptr()), int(sf.data_ptr()),
            rows, cols, cs())
    elif (clip_group_amax is not None
            and os.environ.get(
                'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_CLIP_STATIC', '0') == '1'
            and hasattr(fvk, 'quantize_bf16_to_nvfp4_swizzled_static_groups')):
        fvk.quantize_bf16_to_nvfp4_swizzled_static_groups(
            int(flat.data_ptr()), int(clip_group_amax.data_ptr()),
            int(packed.data_ptr()), int(sf.data_ptr()),
            rows, cols, cs())
    elif clip_group_amax is not None and hasattr(
            fvk, 'quantize_bf16_to_nvfp4_swizzled_clipped'):
        fvk.quantize_bf16_to_nvfp4_swizzled_clipped(
            int(flat.data_ptr()), int(clip_group_amax.data_ptr()),
            int(packed.data_ptr()), int(sf.data_ptr()),
            rows, cols, cs())
    else:
        if (cols == 14336
                and os.environ.get(
                    'FLASH_RT_MOTUS_USE_FAST_K14336_NVFP4_Q', '1') == '1'
                and hasattr(fvk, 'quantize_bf16_to_nvfp4_swizzled_k14336')):
            rc = fvk.quantize_bf16_to_nvfp4_swizzled_k14336(
                int(flat.data_ptr()), int(packed.data_ptr()),
                int(sf.data_ptr()), rows, cols, cs())
            if rc == 0:
                return
        fvk.quantize_bf16_to_nvfp4_swizzled(
            int(flat.data_ptr()), int(packed.data_ptr()),
            int(sf.data_ptr()), rows, cols, cs())


def _make_nvfp4_ffn_forward(state: _Nvfp4VideoFfnState):
    """Closure: BF16 input (B, L, K) → BF16 output (B, L, K) via NVFP4
    GEMMs + bias_gelu fused.

    K = up.K = dn.N (= dim, e.g., 3072)
    F = up.N = dn.K (= ffn dim, e.g., 14336)
    """
    K = state.K
    F = state.F
    up_w_p_ptr = int(state.up_w_packed.data_ptr())
    up_w_sf_ptr = int(state.up_w_sf.data_ptr())
    dn_w_p_ptr = int(state.dn_w_packed.data_ptr())
    dn_w_sf_ptr = int(state.dn_w_sf.data_ptr())
    up_inv_s = state.up_inv_s
    dn_inv_s = state.dn_inv_s
    up_clip_group_amax = state.up_clip_group_amax
    dn_clip_group_amax = state.dn_clip_group_amax
    up_alpha = float(state.up_alpha)
    dn_alpha = float(state.dn_alpha)
    up_bias_ptr = int(state.up_bias.data_ptr()) if state.up_bias is not None else 0
    dn_bias_ptr = int(state.dn_bias.data_ptr()) if state.dn_bias is not None else 0

    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()
        in_shape = x.shape
        flat = x.reshape(-1, K)            # (M, K)
        M = flat.shape[0]
        if M > state.M_capacity:
            raise RuntimeError(
                f'NVFP4 FFN M={M} exceeds preallocated capacity '
                f'{state.M_capacity}')

        # ── Up GEMM (M, K) × (F, K)^T = (M, F) ──
        # Quantize input to NVFP4.
        up_in_packed = state.up_in_packed[:M]
        up_in_sf = state.up_in_sf
        _quantize_act_to_nvfp4(
            flat, up_inv_s, up_in_packed, up_in_sf, M, K,
            up_clip_group_amax)
        up_out = state.up_out[:M]
        dn_in_packed = state.dn_in_packed[:M]
        dn_in_sf = state.dn_in_sf

        # Recipe C Task A+B: cutlass NVFP4 GEMM_up + bias + GELU + FP4
        # quant epilogue (Task A) chained with cutlass StreamK GEMM_dn +
        # bias epilogue (Task B). Replaces the 5-launch chain
        #   GEMM_up bf16 + bias_gelu + quant_bf16_to_nvfp4 + GEMM_dn +
        #   add_bias
        # with 2 cutlass-fork kernels, saving ~17 µs/call standalone.
        # AWQ (dn_inv_s != None) takes the legacy path which applies the
        # AWQ scale during intermediate quant.
        _use_task_ab = (
            up_bias_ptr != 0
            and dn_bias_ptr != 0
            and dn_inv_s is None
            and os.environ.get(
                'FLASH_RT_MOTUS_USE_WFFN_TASK_AB', '1') == '1'
            and hasattr(fvk, 'fp4_w4a16_gemm_bias_gelu_fp4out_sm120')
            and hasattr(fvk, 'fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120'))
        if _use_task_ab:
            # Task A: fused GEMM_up + bias + GELU + FP4 quant → FP4 + SF
            # directly into dn_in_packed/dn_in_sf (no bf16 intermediate).
            fvk.fp4_w4a16_gemm_bias_gelu_fp4out_sm120(
                int(up_in_packed.data_ptr()), up_w_p_ptr,
                int(up_in_sf.data_ptr()), up_w_sf_ptr,
                up_bias_ptr,
                int(dn_in_packed.data_ptr()), int(dn_in_sf.data_ptr()),
                M, F, K, up_alpha, cs())
            # Task B: StreamK GEMM_dn + bias → bf16 out (no add_bias call).
            dn_out = state.dn_out[:M]
            fvk.fp4_w4a16_gemm_dn_streamk_bias_bf16out_sm120(
                int(dn_in_packed.data_ptr()), dn_w_p_ptr,
                int(dn_in_sf.data_ptr()), dn_w_sf_ptr,
                dn_bias_ptr, int(dn_out.data_ptr()),
                M, K, F, dn_alpha, cs())
            return dn_out.view(*in_shape[:-1], K)

        # Recipe C step 1: cutlass NVFP4 W4A16 GEMM_up with fused per-col
        # bias + GELU(tanh) epilogue, bf16 out. Replaces 2-launch
        # (GEMM_up + bias_gelu_inplace) with 1, saving ~7 µs/call. Default
        # on when symbol present; env disables for rollback.
        _use_cutlass_step1 = (
            up_bias_ptr != 0
            and os.environ.get(
                'FLASH_RT_MOTUS_USE_WFFN_CUTLASS_FUSED', '1') == '1'
            and hasattr(fvk, 'fp4_w4a16_gemm_bias_gelu_bf16out_sm120'))
        if _use_cutlass_step1:
            fvk.fp4_w4a16_gemm_bias_gelu_bf16out_sm120(
                int(up_in_packed.data_ptr()), up_w_p_ptr,
                int(up_in_sf.data_ptr()), up_w_sf_ptr,
                up_bias_ptr, int(up_out.data_ptr()),
                M, F, K, up_alpha, cs())
            _quantize_act_to_nvfp4(up_out, dn_inv_s, dn_in_packed,
                                   dn_in_sf, M, F, dn_clip_group_amax)
        else:
            fvk.fp4_w4a16_gemm_sm120_bf16out(
                int(up_in_packed.data_ptr()), up_w_p_ptr,
                int(up_out.data_ptr()),
                M, F, K,
                int(up_in_sf.data_ptr()), up_w_sf_ptr,
                up_alpha, cs())
            if not up_bias_ptr:
                # No bias: still need GELU. Use bias_gelu_fused with a zero
                # bias buffer? For simplicity here, error out — motus FFN up
                # has bias; this branch shouldn't trigger.
                raise RuntimeError('NVFP4 FFN: up bias is None — unexpected '
                                    'on motus video FFN')

            # ── Down GEMM (M, F) × (K, F)^T = (M, K) ──
            if (os.environ.get('FLASH_RT_MOTUS_NVFP4_FFN_FUSED_BIAS_GELU_Q',
                               '1') == '1'
                    and hasattr(fvk, 'bias_gelu_quant_bf16_to_nvfp4_swizzled')):
                if dn_inv_s is not None:
                    fvk.awq_bias_gelu_quant_bf16_to_nvfp4_swizzled(
                        int(up_out.data_ptr()), up_bias_ptr,
                        int(dn_inv_s.data_ptr()), int(dn_in_packed.data_ptr()),
                        int(dn_in_sf.data_ptr()), M, F, cs())
                else:
                    fvk.bias_gelu_quant_bf16_to_nvfp4_swizzled(
                        int(up_out.data_ptr()), up_bias_ptr,
                        int(dn_in_packed.data_ptr()), int(dn_in_sf.data_ptr()),
                        M, F, cs())
            else:
                fvk.bias_gelu_inplace_bf16(
                    int(up_out.data_ptr()), up_bias_ptr, M, F, cs())
                _quantize_act_to_nvfp4(up_out, dn_inv_s, dn_in_packed,
                                       dn_in_sf, M, F, dn_clip_group_amax)
        dn_out = state.dn_out[:M]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            int(dn_in_packed.data_ptr()), dn_w_p_ptr, int(dn_out.data_ptr()),
            M, K, F,
            int(dn_in_sf.data_ptr()), dn_w_sf_ptr,
            dn_alpha, cs())
        if dn_bias_ptr and not bool(getattr(state.dn_site, 'bias_skip', False)):
            fvk.add_bias_bf16(
                int(dn_out.data_ptr()), dn_bias_ptr, M, K, cs())

        return dn_out.view(*in_shape[:-1], K)

    return forward


def _make_nvfp4_ffn_down_forward(
    state: _Nvfp4VideoFfnState,
    up_site,
    down_site,
    gemm: fvk.GemmRunner,
):
    """Hybrid FFN: keep tuned FP8 up GEMM, use NVFP4 only for down GEMM."""
    K = state.K
    F = state.F
    up_w_ptr = int(up_site.w_fp8.data_ptr())
    up_w_scale = int(up_site.w_scale.data_ptr())
    up_act_scale = int(up_site.act_scale.data_ptr())
    up_bias_ptr = int(state.up_bias.data_ptr()) if state.up_bias is not None else 0
    down_act_scale = int(down_site.act_scale.data_ptr())
    dn_w_p_ptr = int(state.dn_w_packed.data_ptr())
    dn_w_sf_ptr = int(state.dn_w_sf.data_ptr())
    dn_inv_s = state.dn_inv_s
    dn_alpha = float(state.dn_alpha)
    dn_bias_ptr = int(state.dn_bias.data_ptr()) if state.dn_bias is not None else 0

    def forward(x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()
        in_shape = x.shape
        flat = x.reshape(-1, K)
        M = flat.shape[0]
        if M > state.M_capacity:
            raise RuntimeError(
                f'NVFP4 FFN down-only M={M} exceeds preallocated capacity '
                f'{state.M_capacity}')

        x_fp8 = up_site.ensure_x_fp8(M, flat.device)
        prefilled = bool(getattr(up_site, '_x_fp8_prefilled', False))
        if prefilled:
            up_site._x_fp8_prefilled = False
        else:
            fvk.quantize_fp8_static(
                int(flat.data_ptr()), int(x_fp8.data_ptr()),
                up_act_scale, M * K, cs())

        if not up_bias_ptr:
            raise RuntimeError('NVFP4 FFN down-only: up bias is None')
        dn_in_packed = state.dn_in_packed[:M]
        dn_in_sf = state.dn_in_sf
        up_out = state.up_out[:M]
        gemm.fp8_nn_dev(
            int(x_fp8.data_ptr()), up_w_ptr, int(up_out.data_ptr()),
            M, F, K, up_act_scale, up_w_scale, cs())
        # Sprint 6 T1: fuse bias+GELU+nvfp4 quant into 1 launch (was 2).
        # Eliminates the HBM round-trip of up_out (M*F=10.3MB) per call.
        # Env-gated; fallback to 2-call path if disabled or unsupported.
        _t1 = (os.environ.get('FLASH_RT_MOTUS_FFN_DOWN_FUSED_BGQ', '1') == '1')
        if _t1 and dn_inv_s is not None and hasattr(
                fvk, 'awq_bias_gelu_quant_bf16_to_nvfp4_swizzled'):
            fvk.awq_bias_gelu_quant_bf16_to_nvfp4_swizzled(
                int(up_out.data_ptr()), up_bias_ptr,
                int(dn_inv_s.data_ptr()), int(dn_in_packed.data_ptr()),
                int(dn_in_sf.data_ptr()), M, F, cs())
        elif _t1 and dn_inv_s is None and hasattr(
                fvk, 'bias_gelu_quant_bf16_to_nvfp4_swizzled'):
            fvk.bias_gelu_quant_bf16_to_nvfp4_swizzled(
                int(up_out.data_ptr()), up_bias_ptr,
                int(dn_in_packed.data_ptr()), int(dn_in_sf.data_ptr()),
                M, F, cs())
        else:
            fvk.bias_gelu_inplace_bf16(
                int(up_out.data_ptr()), up_bias_ptr, M, F, cs())
            _quantize_act_to_nvfp4(up_out, dn_inv_s, dn_in_packed,
                                   dn_in_sf, M, F)
        dn_out = state.dn_out[:M]
        dn_bias_skipped = bool(getattr(down_site, 'bias_skip', False))
        use_streamk_dn = (
            os.environ.get('FLASH_RT_MOTUS_USE_WFFN_DN_STREAMK', '0') == '1'
            and dn_bias_skipped
            and hasattr(fvk, 'fp4_w4a16_gemm_dn_streamk_bf16out_sm120'))
        if use_streamk_dn:
            fvk.fp4_w4a16_gemm_dn_streamk_bf16out_sm120(
                int(dn_in_packed.data_ptr()), dn_w_p_ptr,
                int(dn_in_sf.data_ptr()), dn_w_sf_ptr, int(dn_out.data_ptr()),
                M, K, F, dn_alpha, cs())
        else:
            fvk.fp4_w4a16_gemm_sm120_bf16out(
                int(dn_in_packed.data_ptr()), dn_w_p_ptr,
                int(dn_out.data_ptr()),
                M, K, F,
                int(dn_in_sf.data_ptr()), dn_w_sf_ptr,
                dn_alpha, cs())
        if dn_bias_ptr and not dn_bias_skipped:
            fvk.add_bias_bf16(
                int(dn_out.data_ptr()), dn_bias_ptr, M, K, cs())
        return dn_out.view(*in_shape[:-1], K)

    return forward


def _make_nvfp4_ffn_step_mixed_forward(full_forward, down_forward,
                                       full_steps: set[int]):
    call_idx = 0

    def forward(x: torch.Tensor) -> torch.Tensor:
        nonlocal call_idx
        step = call_idx
        call_idx += 1
        if step in full_steps:
            return full_forward(x)
        return down_forward(x)

    return forward


def install_motus_nvfp4_ffn_video(model) -> dict:
    """Walk wan_model.blocks; replace each block.ffn.forward with NVFP4
    path. Must run AFTER G3d FP8 swap so we can read FP8 weights back."""
    counts = {'installed': 0, 'skipped': 0, 'reasons': {}}
    if os.environ.get('FLASH_RT_MOTUS_USE_NVFP4_FFN_VIDEO', '0') != '1':
        logger.info('[nvfp4.ffn_v] disabled by env (set USE=1)')
        return counts

    wan_blocks = model.video_model.wan_model.blocks
    layers = _selected_layers()
    full_layers = _layer_set_from_env(
        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_FULL_LAYERS')
    full_steps = _int_set_from_env(
        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_FULL_STEPS')
    # The full up+down NVFP4 path is faster but is not numerically acceptable
    # on Motus E2E. Keep the production default on down-only unless a probe
    # explicitly opts into full/layer-sweep mode.
    global_mode = os.environ.get(
        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_MODE', 'down').strip().lower()
    for i, blk in enumerate(wan_blocks):
        if layers is not None and i not in layers:
            counts['skipped'] += 1
            continue
        ffn = blk.ffn
        up_site = getattr(ffn, '_fp8_up_site', None)
        dn_site = getattr(ffn, '_fp8_down_site', None)
        if up_site is None or dn_site is None:
            counts['skipped'] += 1
            continue
        K = up_site.K   # input dim, e.g. 3072
        F = up_site.N   # ffn dim,   e.g. 14336
        if K % 16 != 0 or F % 16 != 0:
            reason = f'K={K} F={F} not %16'
            counts['skipped'] += 1
            counts['reasons'][reason] = counts['reasons'].get(reason, 0) + 1
            continue

        up_w_packed = getattr(up_site, 'nvfp4_w_packed', None)
        up_w_sf = getattr(up_site, 'nvfp4_w_sf', None)
        dn_w_packed = getattr(dn_site, 'nvfp4_w_packed', None)
        dn_w_sf = getattr(dn_site, 'nvfp4_w_sf', None)
        if any(t is None for t in (up_w_packed, up_w_sf, dn_w_packed, dn_w_sf)):
            reason = 'missing pre-G4 NVFP4 weights; set PREP env before load'
            counts['skipped'] += 1
            counts['reasons'][reason] = counts['reasons'].get(reason, 0) + 1
            continue

        up_bias = up_site.bias if up_site.has_bias else None
        dn_bias = dn_site.bias if dn_site.has_bias else None
        up_inv_s = None
        dn_inv_s = None
        up_alpha = 1.0
        dn_alpha = 1.0
        mode = 'full' if i in full_layers else global_mode
        if os.environ.get('FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ', '1') == '1':
            alpha_up = float(os.environ.get(
                'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_ALPHA_UP', '0.5'))
            alpha_dn = float(os.environ.get(
                'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_ALPHA_DN', '0.7'))
            awq_up_p = awq_up_sf = awq_up_inv = None
            if (mode != 'down'
                    and os.environ.get(
                        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_UP', '1') == '1'):
                awq_up_p, awq_up_sf, awq_up_inv = _smoothquant_nvfp4_from_fp8_site(
                    up_site, alpha_up)
            awq_dn_p = awq_dn_sf = awq_dn_inv = None
            if os.environ.get(
                    'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_DN', '1') == '1':
                awq_dn_p, awq_dn_sf, awq_dn_inv = _smoothquant_nvfp4_from_fp8_site(
                    dn_site, alpha_dn)
            if awq_up_p is not None:
                up_w_packed, up_w_sf, up_inv_s = awq_up_p, awq_up_sf, awq_up_inv
                up_site.nvfp4_w_packed = None
                up_site.nvfp4_w_sf = None
            if awq_dn_p is not None:
                dn_w_packed, dn_w_sf, dn_inv_s = awq_dn_p, awq_dn_sf, awq_dn_inv
                dn_site.nvfp4_w_packed = None
                dn_site.nvfp4_w_sf = None

        if up_inv_s is None:
            g = float(os.environ.get(
                'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_UP_GLOBAL_SCALE', '1.0'))
            if g != 1.0:
                up_inv_s = torch.full(
                    (K,), 1.0 / g, dtype=torch.bfloat16,
                    device=up_w_packed.device)
                up_alpha = g
        if dn_inv_s is None:
            g = float(os.environ.get(
                'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_DN_GLOBAL_SCALE', '1.0'))
            if g != 1.0:
                dn_inv_s = torch.full(
                    (F,), 1.0 / g, dtype=torch.bfloat16,
                    device=dn_w_packed.device)
                dn_alpha = g

        # Pin tensors on the FFN module so they don't get gc'd.
        buf = getattr(up_site, 'x_fp8_buf', None)
        M_capacity = (buf.numel() // K) if buf is not None and buf.numel() else 360
        dev = up_w_packed.device
        state = _Nvfp4VideoFfnState(
            up_w_packed, up_w_sf, dn_w_packed, dn_w_sf,
            up_inv_s, dn_inv_s,
            getattr(up_site, 'nvfp4_clip_group_amax', None),
            getattr(dn_site, 'nvfp4_clip_group_amax', None),
            up_alpha, dn_alpha,
            up_bias, dn_bias, dn_site, K, F,
            M_capacity, dev)
        ffn._nvfp4_state = state

        if mode == 'down':
            gemm = getattr(model, '_g3b_gemm', None) or fvk.GemmRunner()
            model._g3b_gemm = gemm
            ffn.forward = _make_nvfp4_ffn_down_forward(
                state, up_site, dn_site, gemm)
            if os.environ.get('FLASH_RT_MOTUS_NVFP4_FREE_FP8_SHADOW', '1') == '1':
                try:
                    down_mod = list(ffn)[2]
                    empty = torch.empty(0, dtype=dn_site.w_fp8.dtype,
                                        device=dn_site.w_fp8.device)
                    down_mod.weight.data = empty
                    dn_site.w_fp8 = empty
                except Exception:
                    pass
        else:
            full_forward = _make_nvfp4_ffn_forward(state)
            if full_steps:
                gemm = getattr(model, '_g3b_gemm', None) or fvk.GemmRunner()
                model._g3b_gemm = gemm
                down_forward = _make_nvfp4_ffn_down_forward(
                    state, up_site, dn_site, gemm)
                ffn.forward = _make_nvfp4_ffn_step_mixed_forward(
                    full_forward, down_forward, full_steps)
                if os.environ.get('FLASH_RT_MOTUS_NVFP4_FREE_FP8_SHADOW', '1') == '1':
                    try:
                        down_mod = list(ffn)[2]
                        empty_dn = torch.empty(0, dtype=dn_site.w_fp8.dtype,
                                               device=dn_site.w_fp8.device)
                        down_mod.weight.data = empty_dn
                        dn_site.w_fp8 = empty_dn
                    except Exception:
                        pass
            else:
                ffn.forward = full_forward
            free_fp8_shadow = (
                os.environ.get('FLASH_RT_MOTUS_NVFP4_FREE_FP8_SHADOW', '1') == '1'
                and not full_steps)
            if free_fp8_shadow:
                try:
                    up_mod, down_mod = list(ffn)[0], list(ffn)[2]
                    empty_up = torch.empty(0, dtype=up_site.w_fp8.dtype,
                                           device=up_site.w_fp8.device)
                    empty_dn = torch.empty(0, dtype=dn_site.w_fp8.dtype,
                                           device=dn_site.w_fp8.device)
                    up_mod.weight.data = empty_up
                    down_mod.weight.data = empty_dn
                    up_site.w_fp8 = empty_up
                    dn_site.w_fp8 = empty_dn
                except Exception:
                    pass
        counts['installed'] += 1

    logger.info(
        f'[nvfp4.ffn_v] installed NVFP4 FFN forward on '
        f'{counts["installed"]} layers (skipped {counts["skipped"]})')
    return counts

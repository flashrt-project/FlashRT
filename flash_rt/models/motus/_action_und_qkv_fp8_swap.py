"""G7.24 — Action/und QKV BAGEL packed-GEMM BF16 → FP8 W8A8 (SmoothQuant).

After G7.13a AWQ swap, action_o/und_o + action/und FFN are FP8. The
remaining BF16 GEMMs in the joint-attention path are the BAGEL packed
QKV projections introduced in G7.20:

    norm_action (M=B*L_a=8 , K=action_dim) @ wan_action_qkv_flat (K, 9216)
    norm_und    (M=B*L_u=138, K=und_dim)   @ wan_und_qkv_flat    (K, 9216)

Both are bandwidth-bound at small M. FP8 W8A8 halves weight load BW
and (with awq_quant_fp8_static_bf16) collapses act-scale + quant
into one launch. We follow G7.13a SmoothQuant exactly:

  s[k] = max(|x|_k)^alpha / max(|w|_K=k)^(1-alpha)  (alpha = 0.9)
  w' = w * s         (per-K, broadcast over N)
  inv_s = 1/s        (per-K bf16, broadcast over leading dims of x)
  w_fp8 = quantize_per_tensor(w')

Calibration: piggy-back AWQ's first BF16 pass — when calibrating, the
modulate_fuse BAGEL branch records norm_*.abs().amax over leading
dims into _G724State.act_amax_K. After AWQ install, we apply the
SmoothQuant scale + FP8 quantize. G4 calibration (the second pass
that AWQ already runs through) then uses our FP8 path with dynamic
quantize_fp8_device to capture per-site act_scale; final replay
uses static awq_quant_fp8_static_bf16 + fp8_nn_dev.

Toggle: FLASH_RT_MOTUS_NO_G7_24=1 (calibration hook + install + runtime
        path all gated; falls back to G7.20 BAGEL bf16_nn).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from flash_rt.models.motus._stream import cs
import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_EPS = 1e-5


class _G724State:
    """Per-block state. Two phases:

    * mode == 'calib_amax' : act_amax_K is being accumulated; runtime
      path is still BF16 (modulate_fuse uses bf16_nn).
    * mode == 'fp8_ready'  : w_fp8 / inv_s / act_scale are populated;
      runtime path uses awq_quant_fp8_static_bf16 + fp8_nn_dev.
    """
    __slots__ = ('mode', 'act_amax_K', 'count',
                 'K', 'N', 'w_fp8', 'w_scale', 'inv_s', 'act_scale',
                 'x_fp8_buf', 'label', 'alpha')

    def __init__(self, K: int, N: int, device, label: str, alpha: float):
        self.mode = 'calib_amax'
        self.act_amax_K = torch.zeros(K, dtype=torch.float32, device=device)
        self.count = 0
        self.K = K
        self.N = N
        self.label = label
        self.alpha = alpha
        self.w_fp8: Optional[torch.Tensor] = None
        self.w_scale: Optional[torch.Tensor] = None
        self.inv_s: Optional[torch.Tensor] = None
        self.act_scale: Optional[torch.Tensor] = None
        self.x_fp8_buf: Optional[torch.Tensor] = None

    def ensure_buf(self, M: int, device):
        n = M * self.K
        if self.x_fp8_buf is None or self.x_fp8_buf.numel() < n:
            self.x_fp8_buf = torch.empty(n, dtype=_FP8, device=device).contiguous()
        return self.x_fp8_buf


def _flatten_qkv_param(w: torch.Tensor) -> torch.Tensor:
    """wan_*_qkv (3, n, K=dim, d) → flat (K, 3*n*d) BF16 contiguous,
    matching G7.20 BAGEL `_action_qkv_flat` layout exactly so that
    output channels are interleaved (Q[0..n*d], K[n*d..2*n*d], V[..3*n*d]).
    """
    assert w.ndim == 4, f"expected (3, n, K, d), got {w.shape}"
    return w.permute(2, 0, 1, 3).contiguous().reshape(
        w.shape[2], 3 * w.shape[1] * w.shape[3]).contiguous()


def install_g724_calibration(model) -> Dict[str, _G724State]:
    """Attach _g724_state markers (mode='calib_amax') on every
    action_expert / und_expert block. Modulate_fuse picks these up
    and accumulates norm_*.abs().amax(dim=0) on each call.

    Returns label → state for caller convenience.
    """
    states: Dict[str, _G724State] = {}
    if os.environ.get('FLASH_RT_MOTUS_NO_G7_24', '0') == '1':
        logger.info('[g7.24] disabled')
        return states
    scope = os.environ.get(
        'FLASH_RT_MOTUS_G7_24_SCOPE', 'all').strip().lower()

    for tag, attr in (('action', 'action_expert'), ('und', 'und_expert')):
        if scope not in ('all', tag):
            continue
        ex = getattr(model, attr, None)
        if ex is None or not hasattr(ex, 'blocks'):
            continue
        for i, blk in enumerate(ex.blocks):
            qkv_attr = 'wan_action_qkv' if tag == 'action' else 'wan_und_qkv'
            w = getattr(blk, qkv_attr, None)
            if not isinstance(w, nn.Parameter):
                continue
            # w shape (3, n, K, d). K = K = block dim.
            K = int(w.shape[2])
            N = int(3 * w.shape[1] * w.shape[3])
            label = f'{tag}.blocks.{i}.{qkv_attr}'
            st = _G724State(K=K, N=N, device=w.device, label=label, alpha=0.9)
            blk._g724_state = st
            states[label] = st

    logger.info(f'[g7.24] installed calib markers on {len(states)} blocks')
    return states


def install_g724_fp8(model, alpha: float = 0.9) -> dict:
    """After AWQ BF16 calib pass (states accumulated act_amax_K), fold
    SmoothQuant scale into wan_*_qkv parameter, FP8-quantize the
    flattened weight, store on _g724_state with mode='fp8_ready'.

    Also pins _action_qkv_flat / _und_qkv_flat on each block (FP8) so
    the modulate_fuse runtime path can fetch w pointers.
    """
    counts = {'action': 0, 'und': 0, 'skip_no_calib': 0, 'skip_align': 0}
    if os.environ.get('FLASH_RT_MOTUS_NO_G7_24', '0') == '1':
        return counts

    alpha = float(os.environ.get('FLASH_RT_MOTUS_G7_24_ALPHA', str(alpha)))
    scope = os.environ.get(
        'FLASH_RT_MOTUS_G7_24_SCOPE', 'all').strip().lower()

    for tag, attr in (('action', 'action_expert'), ('und', 'und_expert')):
        if scope not in ('all', tag):
            continue
        ex = getattr(model, attr, None)
        if ex is None or not hasattr(ex, 'blocks'):
            continue
        for i, blk in enumerate(ex.blocks):
            st: Optional[_G724State] = getattr(blk, '_g724_state', None)
            if st is None:
                continue
            if st.count == 0:
                counts['skip_no_calib'] += 1
                continue
            qkv_attr = 'wan_action_qkv' if tag == 'action' else 'wan_und_qkv'
            w_param: nn.Parameter = getattr(blk, qkv_attr)

            # Build (K, N) BF16 flat exactly matching G7.20 BAGEL.
            w_flat = _flatten_qkv_param(w_param.data)
            K, N = w_flat.shape
            assert K == st.K and N == st.N, (K, N, st.K, st.N)
            if (K % 16) or (N % 16):
                counts['skip_align'] += 1
                continue

            dev = w_flat.device
            # SmoothQuant scale: s[k] = a^alpha / w^(1-alpha)
            a = st.act_amax_K.float().clamp(min=_EPS)
            w_amax_K = w_flat.abs().amax(dim=1).float().clamp(min=_EPS)  # (K,)
            s = (a.pow(alpha) / w_amax_K.pow(1.0 - alpha)).clamp(min=_EPS)
            # Geometric-mean normalize (no global gain shift).
            s = s / s.log().mean().exp()
            inv_s = (1.0 / s).to(torch.bfloat16).contiguous()  # (K,) bf16
            # Apply to weight: w'[k, n] = w[k, n] * s[k]
            w_scaled = (w_flat.float() * s.unsqueeze(1)).to(torch.bfloat16)

            max_abs = float(w_scaled.abs().max().item())
            scale = max(max_abs / _FP8_MAX, 1e-12)
            w_q = (w_scaled / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).contiguous()

            st.w_fp8 = w_q
            st.w_scale = torch.tensor([scale], dtype=torch.float32, device=dev)
            st.inv_s = inv_s
            # Derive static act_scale from the SmoothQuant-scaled amax:
            # after x' = x * inv_s, per-tensor max ≈ max_k(act_amax_K[k] * inv_s[k]).
            # This avoids a dedicated G4 calibration pass for these sites
            # (BAGEL eligibility is gated off during calibrating=True).
            scaled_amax = float((st.act_amax_K.float()
                                  * inv_s.float()).max().item())
            act_scale_val = max(scaled_amax / _FP8_MAX, 1e-12)
            st.act_scale = torch.tensor([act_scale_val],
                                          dtype=torch.float32, device=dev)
            st.mode = 'fp8_ready'

            # NOTE: do NOT replace w_param.data. The non-BAGEL einsum
            # fallback in modulate_fuse_swap (used during _FP8_STATE.
            # calibrating=True) still reads wan_*_qkv as (3, n, K, d)
            # BF16. Leaving it intact costs a few MB but keeps the
            # fallback path correct.

            counts[tag] += 1

    logger.info(f'[g7.24] FP8 swap: action={counts["action"]}, '
                f'und={counts["und"]}, '
                f'no-calib-skip={counts["skip_no_calib"]}, '
                f'align-skip={counts["skip_align"]}')
    return counts


# ──────────────────────────────────────────────────────────────────
# Runtime hooks invoked from _modulate_fuse_swap.py BAGEL branch.
# ──────────────────────────────────────────────────────────────────

def g724_capture_amax(state: _G724State, norm_flat: torch.Tensor) -> None:
    """Accumulate per-K activation amax during the AWQ-calib BF16 pass.
    norm_flat: (M, K) bf16. Run from modulate_fuse_swap.py.
    """
    if state.mode != 'calib_amax':
        return
    abs_max_K = norm_flat.detach().abs().amax(dim=0).float()
    torch.maximum(state.act_amax_K, abs_max_K, out=state.act_amax_K)
    state.count += 1


def g724_fp8_gemm(state: _G724State,
                   norm_flat: torch.Tensor,
                   packed_out: torch.Tensor,
                   gemm: 'fvk.GemmRunner',
                   stream,
                   calibrating: bool) -> None:
    """FP8 replacement for `gemm.bf16_nn(norm_flat, w_flat, packed_out)`.

    Must be called only when state.mode == 'fp8_ready'. During G4
    calibration (`calibrating=True`) the dynamic quantize_fp8_device
    captures running max into state.act_scale; otherwise we use the
    G7.14 fused awq_quant_fp8_static_bf16 path.
    """
    assert state.mode == 'fp8_ready', state.mode
    M, K = norm_flat.shape
    assert K == state.K, (K, state.K)
    N = state.N
    device = norm_flat.device

    x_fp8 = state.ensure_buf(M, device)
    n_act = M * K
    if calibrating:
        # Dynamic act_scale capture (mirrors G4 path used by Wan/AWQ).
        x_scaled = norm_flat * state.inv_s
        fvk.quantize_fp8_device(
            int(x_scaled.data_ptr()), int(x_fp8.data_ptr()),
            int(state.act_scale.data_ptr()), n_act, stream)
    else:
        fvk.awq_quant_fp8_static_bf16(
            int(norm_flat.data_ptr()), int(state.inv_s.data_ptr()),
            int(x_fp8.data_ptr()), int(state.act_scale.data_ptr()),
            M, K, stream)

    gemm.fp8_nn_dev(
        int(x_fp8.data_ptr()), int(state.w_fp8.data_ptr()),
        int(packed_out.data_ptr()),
        M, N, K,
        int(state.act_scale.data_ptr()), int(state.w_scale.data_ptr()),
        stream)

"""G7.13 v4 — AWQ / SmoothQuant FP8 PTQ for action_expert + und_expert.

Naive per-tensor FP8 PTQ on motus action expert collapses cos to 0.964
(G7.13 v1) because the M=8 token chunks have wide per-channel activation
magnitude variance — a single shared scale clips small-magnitude
channels. SmoothQuant rescues this by absorbing per-input-channel
activation magnitude into the weight tensor:

    s[k] = max(|x|_k)^alpha / max(|w|_:,k)^(1-alpha)    (per-K vector)
    x'   = x  / s                                       (broadcast over M)
    w'   = w  * s                                       (broadcast over N)

After this transformation the math is identical (x' @ w'^T == x @ w^T)
but both x' and w' have flatter per-channel magnitude distributions,
so a single per-tensor FP8 scale captures both well.

Cost at runtime: 1 extra elementwise launch per call (x *= inv_s)
before quantize_fp8_static. With 1200 calls/replay across 30 layers
this is ~1.2 ms overhead.

Calibration: one forward pass on real-distribution input (same hook
as G4 calibration); records `act_max[k] = max over (M, calls) of
|x[..., k]|`. Run BEFORE FP8 install so we work on the original BF16
weights.

Toggle: FLASH_RT_MOTUS_NO_G7_13=1 (pre-pass + swap both gated by this).
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


# ──────────────────────────────────────────────────────────────────
# 1) Calibration pre-pass — record per-K activation amax for target
#    Linears. Module-level state; install_awq_calibration_hooks
#    attaches forward pre-hooks; run_calibration_passes does inference
#    to collect; then we fold scales into weights.
# ──────────────────────────────────────────────────────────────────

class _AwqCalibState:
    """Per-Linear running activation amax along input-channel (K) dim."""
    __slots__ = ('act_amax_K', 'count', 'enabled')

    def __init__(self, K: int, device):
        self.act_amax_K = torch.zeros(K, dtype=torch.float32, device=device)
        self.count = 0
        self.enabled = True


def _make_calib_pre_hook(state: _AwqCalibState):
    def hook(module, args):
        if not state.enabled:
            return None
        x = args[0]
        if not torch.is_tensor(x):
            return None
        # x: (..., K). reduce across all leading dims.
        flat = x.detach().reshape(-1, x.shape[-1])
        # Use max(abs) across rows — robust to outlier rows.
        abs_max_K = flat.abs().amax(dim=0).float()
        torch.maximum(state.act_amax_K, abs_max_K, out=state.act_amax_K)
        state.count += 1
        return None
    return hook


def _select_targets(model) -> list:
    """Return list of (label, module, alpha) for AWQ-quantize candidates.

    Default scope: 60 O-proj Linears (wan_action_o + wan_und_o) at
    SmoothQuant α=0.5. With ``FLASH_RT_MOTUS_AWQ_FFN=1`` extends to
    FFN sites (action/und ffn[0] + ffn[2]) at α controlled by
    ``FLASH_RT_MOTUS_AWQ_FFN_ALPHA``.

    Per-site alpha returned so install_awq_fp8_swap can apply
    different SmoothQuant strengths per site type.
    """
    # G7.13a: O-proj sites use SmoothQuant α=0.5 (closed-form midpoint).
    # FFN AWQ is opt-in: it is faster, but current trajectory checks show
    # higher action drift than the BF16 FFN path.
    default_alpha = float(os.environ.get(
        'FLASH_RT_MOTUS_AWQ_O_ALPHA', '0.5'))
    ffn_enabled = os.environ.get('FLASH_RT_MOTUS_AWQ_FFN', '0') == '1'
    ffn_scope = os.environ.get(
        'FLASH_RT_MOTUS_AWQ_FFN_SCOPE', 'all').strip().lower()
    ffn_alpha = float(os.environ.get('FLASH_RT_MOTUS_AWQ_FFN_ALPHA', '0.9'))
    out = []
    a_blocks = getattr(model.action_expert, 'blocks', None)
    if a_blocks is not None:
        for i, blk in enumerate(a_blocks):
            o_mod = getattr(blk, 'wan_action_o', None)
            if isinstance(o_mod, nn.Linear):
                out.append((f'action.blocks.{i}.wan_action_o',
                            o_mod, default_alpha))
            if ffn_enabled and ffn_scope in ('all', 'action'):
                ffn = getattr(blk, 'ffn', None)
                if isinstance(ffn, nn.Sequential) and len(ffn) == 3:
                    if isinstance(ffn[0], nn.Linear):
                        out.append((f'action.blocks.{i}.ffn.0',
                                    ffn[0], ffn_alpha))
                    if isinstance(ffn[2], nn.Linear):
                        out.append((f'action.blocks.{i}.ffn.2',
                                    ffn[2], ffn_alpha))
    u_blocks = getattr(model.und_expert, 'blocks', None) if hasattr(
        model, 'und_expert') else None
    if u_blocks is not None:
        for i, blk in enumerate(u_blocks):
            o_mod = getattr(blk, 'wan_und_o', None)
            if isinstance(o_mod, nn.Linear):
                out.append((f'und.blocks.{i}.wan_und_o',
                            o_mod, default_alpha))
            if ffn_enabled and ffn_scope in ('all', 'und'):
                ffn = getattr(blk, 'ffn', None)
                if isinstance(ffn, nn.Sequential) and len(ffn) == 3:
                    if isinstance(ffn[0], nn.Linear):
                        out.append((f'und.blocks.{i}.ffn.0',
                                    ffn[0], ffn_alpha))
                    if isinstance(ffn[2], nn.Linear):
                        out.append((f'und.blocks.{i}.ffn.2',
                                    ffn[2], ffn_alpha))
    return out


def install_awq_calibration_hooks(model) -> Dict[str, _AwqCalibState]:
    """Attach forward pre-hooks to each target Linear to record per-K
    activation amax on the next inference pass.

    G3d FFN bypass replaces Sequential.forward with a closure that
    calls gemm.bf16_nn directly WITHOUT invoking ffn[0]/ffn[2].__call__,
    so naive pre-hooks on the Linear children never fire during the
    G3d-bypassed Sequential. We work around this by ALSO temporarily
    saving the original Sequential.forward and replacing it with a
    plain "for child in self: x = child(x)" path during calibration.
    Caller restores via remove_awq_calibration_hooks.

    Returns dict label -> _AwqCalibState.
    """
    states: Dict[str, _AwqCalibState] = {}
    for label, mod, _alpha in _select_targets(model):
        K = int(mod.weight.shape[0])
        st = _AwqCalibState(K, mod.weight.device)
        states[label] = st
        h = mod.register_forward_pre_hook(_make_calib_pre_hook(st))
        mod._awq_handle = h

    # Restore plain Sequential.forward on action/und FFN Sequentials
    # so the child Linear hooks fire during calibration. We pin the
    # G3d-replaced forward on ``_g3d_forward`` for restore.
    n_ffn_restored = 0
    for tag, attr in (('action', 'action_expert'), ('und', 'und_expert')):
        ex = getattr(model, attr, None)
        if ex is None or not hasattr(ex, 'blocks'):
            continue
        for blk in ex.blocks:
            ffn = getattr(blk, 'ffn', None)
            if isinstance(ffn, nn.Sequential):
                # Save the closure forward (G3d-installed) and
                # temporarily restore the class-level Sequential.forward.
                ffn._g3d_forward = ffn.forward
                # Plain Sequential.forward semantics. Bind to instance.
                def _seq_fwd(self, x):
                    for module in self:
                        x = module(x)
                    return x
                ffn.forward = _seq_fwd.__get__(ffn, type(ffn))
                n_ffn_restored += 1

    logger.info(f'[g7.13.awq] installed calibration hooks on {len(states)} '
                f'sites; restored plain forward on {n_ffn_restored} FFN '
                f'Sequentials for hook visibility')
    return states


def remove_awq_calibration_hooks(model) -> None:
    for label, mod, _alpha in _select_targets(model):
        h = getattr(mod, '_awq_handle', None)
        if h is not None:
            h.remove()
            del mod._awq_handle
    # Restore G3d FFN bypass forward.
    for tag, attr in (('action', 'action_expert'), ('und', 'und_expert')):
        ex = getattr(model, attr, None)
        if ex is None or not hasattr(ex, 'blocks'):
            continue
        for blk in ex.blocks:
            ffn = getattr(blk, 'ffn', None)
            if isinstance(ffn, nn.Sequential) and hasattr(ffn, '_g3d_forward'):
                ffn.forward = ffn._g3d_forward
                del ffn._g3d_forward


# ──────────────────────────────────────────────────────────────────
# 2) Apply SmoothQuant scales: fold per-K scale s into weight
#    (w' = w * s along K) and store inv_s for runtime activation
#    scaling. Then quantize w' to FP8.
# ──────────────────────────────────────────────────────────────────

class _AwqFp8Site:
    __slots__ = ('w_fp8', 'w_scale', 'inv_s', 'act_scale', 'x_fp8_buf',
                 'K', 'N', 'has_bias', 'bias', 'label', 'bias_skip',
                 'x_scaled_buf')

    def __init__(self, weight_param: nn.Parameter,
                 bias: Optional[torch.Tensor],
                 act_amax_K: torch.Tensor,
                 label: str, alpha: float = 0.5):
        w = weight_param.data           # (K, N) post-G3b transpose
        K, N = int(w.shape[0]), int(w.shape[1])
        self.K, self.N = K, N
        self.label = label
        self.has_bias = bias is not None
        self.bias = bias
        self.bias_skip = False
        dev = w.device

        # SmoothQuant scale: s[k] = (act_amax[k])^alpha / (w_amax_K[k])^(1-alpha)
        # Both clamped > 1e-5 to avoid div-by-zero.
        EPS = 1e-5
        w_amax_K = w.abs().amax(dim=1).float()    # (K,)
        a = act_amax_K.float().clamp(min=EPS)
        b = w_amax_K.clamp(min=EPS)
        s = (a.pow(alpha) / b.pow(1.0 - alpha)).clamp(min=EPS)  # (K,)
        # Normalize so geometric mean of s ~= 1 (avoids global gain shift).
        s = s / s.log().mean().exp()
        self.inv_s = (1.0 / s).to(torch.bfloat16).contiguous()    # (K,) bf16

        # Apply scale to weight along K dim: w'[k, n] = w[k, n] * s[k]
        s_bf = s.to(torch.bfloat16)
        w_scaled = (w.float() * s.unsqueeze(1)).to(torch.bfloat16)  # (K, N)

        # Per-tensor FP8 quantize on w_scaled.
        max_abs = float(w_scaled.abs().max().item())
        scale = max(max_abs / _FP8_MAX, 1e-12)
        w_q = (w_scaled / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).contiguous()
        self.w_fp8 = w_q
        weight_param.data = w_q  # frees the original bf16 storage

        self.w_scale = torch.tensor([scale], dtype=torch.float32, device=dev)
        # act_scale init 1.0; G4 calibration step will overwrite via
        # quantize_fp8_device path during the first post-AWQ infer().
        self.act_scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
        self.x_fp8_buf: Optional[torch.Tensor] = None
        self.x_scaled_buf: Optional[torch.Tensor] = None

    def ensure_buf(self, M: int, device):
        n = M * self.K
        if self.x_fp8_buf is None or self.x_fp8_buf.numel() < n:
            self.x_fp8_buf = torch.empty(
                n, dtype=_FP8, device=device).contiguous()
        if self.x_scaled_buf is None or self.x_scaled_buf.numel() < n:
            self.x_scaled_buf = torch.empty(
                n, dtype=torch.bfloat16, device=device).contiguous()
        return self.x_fp8_buf, self.x_scaled_buf


def _make_awq_fp8_linear_forward(site: _AwqFp8Site, gemm: fvk.GemmRunner):
    K, N = site.K, site.N
    w_ptr = int(site.w_fp8.data_ptr())
    w_scale_ptr = int(site.w_scale.data_ptr())
    act_scale_ptr = int(site.act_scale.data_ptr())
    inv_s = site.inv_s            # (K,) bf16, broadcast over leading dims
    inv_s_ptr = int(inv_s.data_ptr())
    bias = site.bias
    bias_ptr = int(bias.data_ptr()) if bias is not None else 0
    # G7.14: collapse (flat * inv_s) + quantize_fp8_static into 1 launch.
    use_g7_14 = (os.environ.get('FLASH_RT_MOTUS_NO_G7_14', '0') != '1'
                 and hasattr(fvk, 'awq_quant_fp8_static_bf16'))
    dynamic_act = os.environ.get(
        'FLASH_RT_MOTUS_AWQ_DYNAMIC_ACT', '0') == '1'

    # Local import to avoid circular ref.
    from flash_rt.models.motus._fp8_swap import _STATE

    def forward(x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K)
        M = flat.shape[0]
        device = flat.device

        x_fp8, _ = site.ensure_buf(M, device)
        n_act = M * K
        if _STATE.calibrating or dynamic_act:
            # Calibration runs once; preserve dynamic act_scale collection
            # via the original 2-launch chain.
            x_scaled = flat * inv_s
            fvk.quantize_fp8_device(
                int(x_scaled.data_ptr()), int(x_fp8.data_ptr()),
                act_scale_ptr, n_act, cs())
        elif use_g7_14:
            fvk.awq_quant_fp8_static_bf16(
                int(flat.data_ptr()), inv_s_ptr,
                int(x_fp8.data_ptr()), act_scale_ptr,
                M, K, cs())
        else:
            x_scaled = flat * inv_s
            fvk.quantize_fp8_static(
                int(x_scaled.data_ptr()), int(x_fp8.data_ptr()),
                act_scale_ptr, n_act, cs())

        out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        gemm.fp8_nn_dev(
            int(x_fp8.data_ptr()), w_ptr, int(out.data_ptr()),
            M, N, K, act_scale_ptr, w_scale_ptr, cs())

        if bias_ptr and not site.bias_skip:
            fvk.add_bias_bf16(int(out.data_ptr()), bias_ptr, M, N, cs())

        if in_dtype != torch.bfloat16:
            out = out.to(in_dtype)
        return out.view(*in_shape[:-1], N)

    return forward


def _make_awq_fp8_ffn_forward(up_site: _AwqFp8Site, dn_site: _AwqFp8Site,
                                gemm: fvk.GemmRunner):
    K_up, N_up = up_site.K, up_site.N
    K_dn, N_dn = dn_site.K, dn_site.N
    assert N_up == K_dn

    # up
    up_w = int(up_site.w_fp8.data_ptr())
    up_ws = int(up_site.w_scale.data_ptr())
    up_as = int(up_site.act_scale.data_ptr())
    up_inv_s = up_site.inv_s
    up_inv_s_ptr = int(up_inv_s.data_ptr())
    up_bias_ptr = int(up_site.bias.data_ptr()) if up_site.has_bias else 0
    # down
    dn_w = int(dn_site.w_fp8.data_ptr())
    dn_ws = int(dn_site.w_scale.data_ptr())
    dn_as = int(dn_site.act_scale.data_ptr())
    dn_inv_s = dn_site.inv_s
    dn_inv_s_ptr = int(dn_inv_s.data_ptr())
    dn_bias_ptr = int(dn_site.bias.data_ptr()) if dn_site.has_bias else 0
    # G7.14: collapse (x * inv_s) + quantize_fp8_static into 1 launch
    # for both up-proj entry and down-proj entry (post-GELU).
    use_g7_14 = (os.environ.get('FLASH_RT_MOTUS_NO_G7_14', '0') != '1'
                 and hasattr(fvk, 'awq_quant_fp8_static_bf16'))
    dynamic_act = os.environ.get(
        'FLASH_RT_MOTUS_AWQ_DYNAMIC_ACT', '0') == '1'

    from flash_rt.models.motus._fp8_swap import _STATE

    def forward(x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K_up)
        M = flat.shape[0]
        device = flat.device

        # up: scale + quant + GEMM + bias
        x_fp8, _ = up_site.ensure_buf(M, device)
        n_in = M * K_up
        if _STATE.calibrating or dynamic_act:
            x_scaled = flat * up_inv_s
            fvk.quantize_fp8_device(int(x_scaled.data_ptr()),
                                     int(x_fp8.data_ptr()),
                                     up_as, n_in, cs())
        elif use_g7_14:
            fvk.awq_quant_fp8_static_bf16(
                int(flat.data_ptr()), up_inv_s_ptr,
                int(x_fp8.data_ptr()), up_as, M, K_up, cs())
        else:
            x_scaled = flat * up_inv_s
            fvk.quantize_fp8_static(int(x_scaled.data_ptr()),
                                     int(x_fp8.data_ptr()),
                                     up_as, n_in, cs())
        up_out = torch.empty(M, N_up, dtype=torch.bfloat16, device=device)
        gemm.fp8_nn_dev(
            int(x_fp8.data_ptr()), up_w, int(up_out.data_ptr()),
            M, N_up, K_up, up_as, up_ws, cs())
        if up_bias_ptr:
            fvk.add_bias_bf16(int(up_out.data_ptr()), up_bias_ptr,
                              M, N_up, cs())

        # GELU inplace
        fvk.gelu_inplace(int(up_out.data_ptr()), M * N_up, cs())

        # down: gelu_out × dn_inv_s → quant + GEMM + bias
        up_fp8, _ = dn_site.ensure_buf(M, device)
        n_mid = M * K_dn
        if _STATE.calibrating or dynamic_act:
            gelu_scaled = up_out * dn_inv_s
            fvk.quantize_fp8_device(int(gelu_scaled.data_ptr()),
                                     int(up_fp8.data_ptr()),
                                     dn_as, n_mid, cs())
        elif use_g7_14:
            fvk.awq_quant_fp8_static_bf16(
                int(up_out.data_ptr()), dn_inv_s_ptr,
                int(up_fp8.data_ptr()), dn_as, M, K_dn, cs())
        else:
            gelu_scaled = up_out * dn_inv_s
            fvk.quantize_fp8_static(int(gelu_scaled.data_ptr()),
                                     int(up_fp8.data_ptr()),
                                     dn_as, n_mid, cs())
        dn_out = torch.empty(M, N_dn, dtype=torch.bfloat16, device=device)
        gemm.fp8_nn_dev(
            int(up_fp8.data_ptr()), dn_w, int(dn_out.data_ptr()),
            M, N_dn, K_dn, dn_as, dn_ws, cs())
        if dn_bias_ptr:
            fvk.add_bias_bf16(int(dn_out.data_ptr()), dn_bias_ptr,
                              M, N_dn, cs())

        if in_dtype != torch.bfloat16:
            dn_out = dn_out.to(in_dtype)
        return dn_out.view(*in_shape[:-1], N_dn)

    return forward


def _aligned(K: int, N: int) -> bool:
    return K % 16 == 0 and N % 16 == 0


def install_awq_fp8_swap(model, calib_states: Dict[str, _AwqCalibState],
                          gemm: Optional[fvk.GemmRunner] = None,
                          alpha: float = 0.5) -> dict:
    """Apply SmoothQuant-folded FP8 to action_expert + und_expert.

    Must be called AFTER calibration pass populated `calib_states`.
    """
    counts = {'linear': 0, 'ffn': 0,
              'skip_align': 0, 'skip_no_calib': 0}
    if os.environ.get('FLASH_RT_MOTUS_NO_G7_13', '0') == '1':
        logger.info('[g7.13.awq] disabled')
        return counts

    if gemm is None:
        gemm = getattr(model, '_g3b_gemm', None) or fvk.GemmRunner()

    # Build mapping from module to (label, alpha).
    targets = _select_targets(model)
    mod_to_meta = {id(m): (lab, a) for (lab, m, a) in targets}
    mod_to_label = {mid: meta[0] for mid, meta in mod_to_meta.items()}

    for tag, blocks_attr in (('action', 'action_expert'),
                              ('und', 'und_expert')):
        ex = getattr(model, blocks_attr, None)
        if ex is None or not hasattr(ex, 'blocks'):
            continue
        for i, blk in enumerate(ex.blocks):
            o_attr = 'wan_action_o' if tag == 'action' else 'wan_und_o'
            o_mod = getattr(blk, o_attr, None)
            if isinstance(o_mod, nn.Linear):
                label = mod_to_label.get(id(o_mod))
                K, N = int(o_mod.weight.shape[0]), int(o_mod.weight.shape[1])
                if not _aligned(K, N):
                    counts['skip_align'] += 1
                elif label not in calib_states or calib_states[label].count == 0:
                    counts['skip_no_calib'] += 1
                else:
                    _, a_site = mod_to_meta[id(o_mod)]
                    site = _AwqFp8Site(
                        o_mod.weight, o_mod.bias,
                        calib_states[label].act_amax_K,
                        label=label, alpha=a_site)
                    if (tag == 'action'
                            and os.environ.get('FLASH_RT_MOTUS_NO_G7_12',
                                               '0') != '1'
                            and site.has_bias):
                        # G7.12 originally toggled the BF16 Linear wrapper's
                        # bias_skip flag before this AWQ swap replaced the
                        # forward. Preserve the same contract for the
                        # AWQ-FP8 path so joint residual fusion can fold
                        # action_o bias into the gated residual.
                        site.bias_skip = True
                    o_mod.forward = _make_awq_fp8_linear_forward(site, gemm)
                    o_mod._awq_fp8_site = site
                    counts['linear'] += 1

            ffn = getattr(blk, 'ffn', None)
            if (isinstance(ffn, nn.Sequential) and len(ffn) == 3
                    and isinstance(ffn[0], nn.Linear)
                    and isinstance(ffn[2], nn.Linear)):
                up, _gelu, dn = ffn
                Ku, Nu = int(up.weight.shape[0]), int(up.weight.shape[1])
                Kd, Nd = int(dn.weight.shape[0]), int(dn.weight.shape[1])
                lab_up = mod_to_label.get(id(up))
                lab_dn = mod_to_label.get(id(dn))
                if not (_aligned(Ku, Nu) and _aligned(Kd, Nd)):
                    counts['skip_align'] += 1
                    continue
                if (lab_up not in calib_states or lab_dn not in calib_states
                        or calib_states[lab_up].count == 0
                        or calib_states[lab_dn].count == 0):
                    counts['skip_no_calib'] += 1
                    continue
                _, a_up = mod_to_meta[id(up)]
                _, a_dn = mod_to_meta[id(dn)]
                up_site = _AwqFp8Site(
                    up.weight, up.bias, calib_states[lab_up].act_amax_K,
                    label=lab_up, alpha=a_up)
                dn_site = _AwqFp8Site(
                    dn.weight, dn.bias, calib_states[lab_dn].act_amax_K,
                    label=lab_dn, alpha=a_dn)
                ffn.forward = _make_awq_fp8_ffn_forward(up_site, dn_site, gemm)
                ffn._awq_up_site = up_site
                ffn._awq_dn_site = dn_site
                counts['ffn'] += 1

    logger.info(f'[g7.13.awq] FP8 swap: linear={counts["linear"]}, '
                f'ffn={counts["ffn"]}, '
                f'align-skip={counts["skip_align"]}, '
                f'no-calib-skip={counts["skip_no_calib"]}')
    return counts

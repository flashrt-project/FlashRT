"""G4 — FP8 W8A8 static quantization for Motus hot-path GEMMs.

Replaces every hot-path nn.Linear and FFN bypass (from G3b / G3d) with
an FP8 W8A8 path:

    Steady state (post-calibrate):
        1. quantize_fp8_static(x_bf16, x_fp8, act_scale_dev, n)
        2. gemm.fp8_nn_dev(x_fp8, w_fp8_t, out_bf16, M, N, K,
                            act_scale_dev, w_scale_dev)
        3. (FFN only) repeat for down-proj after gelu_inplace +
            quantize_fp8_static of the up_out

    Calibration (first inference):
        1. quantize_fp8_device(x_bf16, x_fp8, layer_scale_dev, n)
            → writes max(|x|)/448 directly into the persistent
              layer_scale_dev buffer
        2. same gemm.fp8_nn_dev call

The ONE buffer (layer_scale_dev) serves both modes — calibration
writes it, steady-state reads it. This matches the pi05 pattern in
flash_rt/models/pi05/pipeline_rtx.py:_fp8_gemm.

Numerical contract (per docs/calibration.md §2.3):
    alpha = np.float32(act_scale) * np.float32(weight_scale)
    Strictly fp32 multiply — implicit f64 promotion historically
    cost ~0.001 cosine drift on Pi0.5. The fp8_nn_dev kernel reads
    the two scale ptrs and applies them separately on device, so
    alpha is computed inside the kernel in fp32 for free.

Scope:
    * 493 standalone Linears under hot path (G3b set, minus the 91
      whose forwards are now bypassed by FFN swap)
    * 91 FFN bypasses (G3d set) — both up_proj AND down_proj inside
      each replaced with FP8 path

Skipped (same as G3b _SKIP_PREFIXES):
    * VLM (frozen, runs once outside denoise loop)
    * VAE (BF16 only; the encode/decode is small, leave alone)
    * action_expert.time_embedding / time_projection — fp32 storage
    * Wan time_embedding / text_embedding — they are also fp32
      under autocast(fp32). Skipped (would need separate fp32->fp8
      handling).

Note on cuBLASLt sm_120: ``fp8_nn_bias`` (host alpha epilogue) and
``fp8_nn_gelu_bias`` (host alpha + GELU + bias) returned NOT_SUPPORTED
when probed on Wan FFN shapes during G3b/d debug. We use ``fp8_nn_dev``
(device scale ptrs, no epilogue) and add bias separately via
fvk.add_bias_bf16. Modest extra launch; does not affect compute roof.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from flash_rt.models.motus._stream import cs
import torch.nn as nn

import flash_rt.flash_rt_kernels as fvk

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_TRACE = os.environ.get('FLASH_RT_MOTUS_FP8_TRACE', '0') == '1'


def _nvfp4_awq_channel_stat(x: torch.Tensor) -> torch.Tensor:
    q = float(os.environ.get(
        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ_PERCENTILE', '1.0'))
    a = x.detach().abs().float()
    if q >= 0.999999:
        return a.amax(dim=0).float()
    rows = int(a.shape[0])
    k = int(torch.ceil(torch.tensor(q * rows)).item())
    k = max(1, min(rows, k))
    return a.kthvalue(k, dim=0).values.float()


def _nvfp4_group16_stat(x: torch.Tensor) -> torch.Tensor:
    q = float(os.environ.get(
        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_CLIP_PERCENTILE', '0.995'))
    a = x.detach().abs().float()
    rows, cols = int(a.shape[0]), int(a.shape[1])
    if cols % 16 != 0:
        return torch.empty(0, dtype=torch.float32, device=x.device)
    group_amax = a.reshape(rows, cols // 16, 16).amax(dim=2)
    if q >= 0.999999:
        return group_amax.amax(dim=0).float()
    k = int(torch.ceil(torch.tensor(q * rows)).item())
    k = max(1, min(rows, k))
    return group_amax.kthvalue(k, dim=0).values.float()


# ──────────────────────────────────────────────────────────────────
# Per-site state (one per Linear / FFN-stage)
# ──────────────────────────────────────────────────────────────────

class _Fp8Site:
    """One quantized GEMM site.

    Owns:
      - w_fp8: FP8 weight tensor in [K, N] layout (post-G3b transpose)
      - w_scale: 1-element fp32 device buffer (weight max_abs / 448)
      - act_scale: 1-element fp32 device buffer (activation max_abs / 448);
                   filled by calibration, read at steady state
      - x_fp8_buf: scratch FP8 activation buffer; sized to max
                   M*K seen in the live forward; reallocated lazily
    """

    __slots__ = ('w_fp8', 'w_scale', 'act_scale', 'x_fp8_buf', 'K', 'N',
                 'label', 'has_bias', 'bias', 'bias_skip',
                 '_x_fp8_prefilled', '_last_packed_qkv',
                 '_qkv_bias_skip_once',
                 'nvfp4_w_packed', 'nvfp4_w_sf', 'nvfp4_w_bf16_cpu',
                 'nvfp4_awq_act_amax_K', 'nvfp4_clip_group_amax',
                 'nvfp4_ready', 'nvfp4_inv_s',
                 'nvfp4_in_packed', 'nvfp4_in_sf', 'nvfp4_out')

    def __init__(self, weight_param: nn.Parameter,
                 bias: Optional[torch.Tensor],
                 label: str):
        """Quantize the weight in place; pin scale buffers.

        The weight Parameter's underlying storage is REPLACED with an
        FP8 [K, N] tensor; the BF16 storage is freed at construction
        time. This keeps peak install memory bounded — installing 675
        sites in series with NO transient FP32 intermediate
        (avoids the 32 GB OOM hit on first attempt).
        """
        w = weight_param.data
        K, N = int(w.shape[0]), int(w.shape[1])
        self.K, self.N = K, N
        self.label = label
        self.has_bias = bias is not None
        self.bias = bias  # held for closure lifetime + ptr access
        dev = w.device
        self.nvfp4_w_packed = None
        self.nvfp4_w_sf = None
        self.nvfp4_w_bf16_cpu = None
        self.nvfp4_awq_act_amax_K = None
        self.nvfp4_clip_group_amax = None
        self.nvfp4_ready = False
        self.nvfp4_inv_s = None
        self.nvfp4_in_packed = None
        self.nvfp4_in_sf = None
        self.nvfp4_out = None

        if (os.environ.get('FLASH_RT_MOTUS_PREP_NVFP4_VIDEO_FFN', '0') == '1'
                and '.ffn.' in label
                and label.startswith('video_model.wan_model.blocks.')):
            from flash_rt.models.motus._motus_nvfp4_ffn_video_swap import (
                quantize_weight_bf16_to_nvfp4_swz)
            # _Fp8Site stores linear weights in [K, N] layout for
            # cuBLASLt NN GEMM. The NVFP4 CUTLASS path expects persistent
            # weights as [N, K] and applies B^T in the GEMM.
            self.nvfp4_w_packed, self.nvfp4_w_sf = (
                quantize_weight_bf16_to_nvfp4_swz(w.t().contiguous()))
            if os.environ.get('FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_AWQ', '1') == '1':
                if (os.environ.get(
                        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_NATIVE_AWQ',
                        '1') == '1'):
                    self.nvfp4_w_bf16_cpu = (
                        w.detach().to('cpu', dtype=torch.bfloat16)
                        .contiguous())
                self.nvfp4_awq_act_amax_K = torch.zeros(
                    K, dtype=torch.float32, device=dev)
            if (os.environ.get(
                    'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_CLIP_UP', '0') == '1'
                    or os.environ.get(
                        'FLASH_RT_MOTUS_NVFP4_FFN_VIDEO_CLIP_DOWN', '0') == '1'):
                self.nvfp4_clip_group_amax = torch.zeros(
                    K // 16, dtype=torch.float32, device=dev)

        # Per-tensor symmetric. amax computed in bf16 (one extra
        # reduction; no fp32 copy of the whole weight needed).
        max_abs = float(w.abs().max().item())
        scale = max(max_abs / 448.0, 1e-12)

        # In-place divide on the existing bf16 storage, then clamp,
        # then cast. Cast to fp8 allocates a NEW tensor; the bf16
        # storage is freed when we rebind weight.data below.
        w.div_(scale).clamp_(-448.0, 448.0)
        self.w_fp8 = w.to(_FP8).contiguous()
        weight_param.data = self.w_fp8     # frees bf16

        # Persistent fp32 scalar buffers (both single floats).
        self.w_scale = torch.tensor([scale], dtype=torch.float32, device=dev)
        # act_scale init to 1.0; calibration will overwrite. Steady
        # state without calibration would clip badly — invariant
        # enforced by frontend's `_fp8_calibrated` flag.
        self.act_scale = torch.tensor([1.0], dtype=torch.float32, device=dev)

        # Activation FP8 scratch — lazy allocation on first call.
        self.x_fp8_buf: Optional[torch.Tensor] = None

        # G6.7: when True, the FP8 forward returns the raw bf16 GEMM
        # output WITHOUT adding bias. Used by downstream fused kernels
        # (bias_gate_mul_residual_bf16) that fold bias + gate + residual
        # into a single launch. Default False = upstream behaviour.
        self.bias_skip: bool = False

    def ensure_x_fp8(self, M: int, device, dtype=_FP8) -> torch.Tensor:
        n = M * self.K
        if self.x_fp8_buf is None or self.x_fp8_buf.numel() < n:
            self.x_fp8_buf = torch.empty(
                n, dtype=dtype, device=device).contiguous()
        return self.x_fp8_buf


# Global state for calibration mode.
class _Fp8State:
    calibrating = False  # if True, sites use quantize_fp8_device


_STATE = _Fp8State()


def set_calibrating(on: bool) -> None:
    """Toggle global calibration mode. Should bracket exactly one
    forward pass with calibrating=True before steady state.
    """
    _STATE.calibrating = bool(on)
    logger.info(f"[g4] calibrating={_STATE.calibrating}")


def autotune_motus_hot_fp8_gemms(
    model,
    num_algos: int = 32,
) -> dict:
    """Autotune cuBLASLt algo cache for Motus hot FP8 GEMM shapes.

    This runs once after FP8 calibration and before CUDA Graph capture.
    It only mutates ``GemmRunner``'s per-shape algo cache; model weights,
    activation scales, and graph-visible buffers are untouched.
    """
    if os.environ.get('FLASH_RT_MOTUS_NO_G7_29_GEMM_AUTOTUNE', '0') == '1':
        return {'enabled': False, 'reason': 'disabled_by_env'}
    gemm = getattr(model, '_g3b_gemm', None)
    if gemm is None or not hasattr(gemm, 'autotune_fp8_nn_dev'):
        return {'enabled': False, 'reason': 'no_gemm_runner'}

    dev = torch.device('cuda')
    shapes = [
        # Video path, 300 calls/inference each.
        ('video_ffn_up', 360, 14336, 3072),
        ('video_ffn_down', 360, 3072, 14336),
        ('video_qkv', 360, 9216, 3072),
        ('video_o', 360, 3072, 3072),
        # Joint action/und path. These are small-M and launch/heuristic
        # sensitive; autotune once before graph capture and reuse the cache.
        ('action_qkv', 8, 9216, 1024),
        ('und_qkv', 138, 9216, 512),
        ('action_o', 8, 1024, 3072),
        ('und_o', 138, 512, 3072),
        # FFN experts.
        ('action_ffn_up', 8, 4096, 1024),
        ('action_ffn_down', 8, 1024, 4096),
        ('und_ffn_up', 138, 2048, 512),
        ('und_ffn_down', 138, 512, 2048),
    ]
    tuned = []
    for label, M, N, K in shapes:
        A = torch.empty(M, K, dtype=_FP8, device=dev)
        B = torch.empty(K, N, dtype=_FP8, device=dev)
        D = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        a_scale = torch.ones(1, dtype=torch.float32, device=dev)
        w_scale = torch.ones(1, dtype=torch.float32, device=dev)
        try:
            gemm.autotune_fp8_nn_dev(
                int(A.data_ptr()), int(B.data_ptr()), int(D.data_ptr()),
                M, N, K,
                int(a_scale.data_ptr()), int(w_scale.data_ptr()),
                int(num_algos))
            tuned.append(label)
        except Exception as exc:
            logger.warning(
                "[g7.29] fp8 GEMM autotune failed for %s "
                "(M=%d,N=%d,K=%d): %s", label, M, N, K, exc)
    torch.cuda.synchronize()
    return {'enabled': True, 'tuned': tuned, 'num_algos': int(num_algos)}


# ──────────────────────────────────────────────────────────────────
# Forward factories
# ──────────────────────────────────────────────────────────────────

def _make_fp8_linear_forward(site: _Fp8Site, gemm: fvk.GemmRunner):
    """Single-GEMM Linear: x_bf16 -> quantize -> fp8_nn_dev -> bf16 + bias."""
    K, N = site.K, site.N
    w_fp8_ptr = int(site.w_fp8.data_ptr())
    w_scale_ptr = int(site.w_scale.data_ptr())
    act_scale_ptr = int(site.act_scale.data_ptr())
    bias = site.bias
    bias_ptr = int(bias.data_ptr()) if bias is not None else 0
    label = site.label

    def forward(x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K)
        M = flat.shape[0]
        device = flat.device

        x_fp8 = site.ensure_x_fp8(M, device)
        n_act = M * K

        if _STATE.calibrating:
            fvk.quantize_fp8_device(
                int(flat.data_ptr()), int(x_fp8.data_ptr()),
                act_scale_ptr, n_act, cs())
        else:
            fvk.quantize_fp8_static(
                int(flat.data_ptr()), int(x_fp8.data_ptr()),
                act_scale_ptr, n_act, cs())

        out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        gemm.fp8_nn_dev(
            int(x_fp8.data_ptr()), w_fp8_ptr, int(out.data_ptr()),
            M, N, K, act_scale_ptr, w_scale_ptr, cs(),
        )

        # G6.7: skip in-wrapper bias if a downstream kernel will fuse
        # it (bias_skip flag is read at every call so it can be flipped
        # post-install by _modulate_fuse_swap).
        if bias_ptr and not site.bias_skip:
            fvk.add_bias_bf16(int(out.data_ptr()), bias_ptr, M, N, cs())

        if in_dtype != torch.bfloat16:
            out = out.to(in_dtype)
        return out.view(*in_shape[:-1], N)

    return forward


def _make_fp8_ffn_forward(
    up_site: _Fp8Site, down_site: _Fp8Site, gemm: fvk.GemmRunner,
):
    """FFN bypass: 2 FP8 GEMMs with a GELU(tanh) sandwich."""
    K_up, N_up = up_site.K, up_site.N
    K_down, N_down = down_site.K, down_site.N
    assert N_up == K_down, (
        f"FFN dim mismatch: up.N={N_up} vs down.K={K_down}")

    up_w_ptr = int(up_site.w_fp8.data_ptr())
    up_w_scale = int(up_site.w_scale.data_ptr())
    up_act_scale = int(up_site.act_scale.data_ptr())
    up_bias_ptr = int(up_site.bias.data_ptr()) if up_site.has_bias else 0

    down_w_ptr = int(down_site.w_fp8.data_ptr())
    down_w_scale = int(down_site.w_scale.data_ptr())
    down_act_scale = int(down_site.act_scale.data_ptr())
    down_bias_ptr = int(down_site.bias.data_ptr()) if down_site.has_bias else 0

    def forward(x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if in_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x_c = x if x.is_contiguous() else x.contiguous()
        in_shape = x_c.shape
        flat = x_c.reshape(-1, K_up)
        M = flat.shape[0]
        device = flat.device

        # 1) input quantize
        x_fp8 = up_site.ensure_x_fp8(M, device)
        n_in = M * K_up
        # G7.18: when joint_attn body or process_ffn pre-fills x_fp8 via
        # the fused ada_layer_norm_fp8 kernel, skip the redundant
        # quantize step here. Flag is reset to False after consumption.
        prefilled = bool(getattr(up_site, '_x_fp8_prefilled', False))
        if prefilled:
            up_site._x_fp8_prefilled = False
        elif _STATE.calibrating:
            if up_site.nvfp4_awq_act_amax_K is not None:
                amax = _nvfp4_awq_channel_stat(flat)
                torch.maximum(up_site.nvfp4_awq_act_amax_K, amax,
                              out=up_site.nvfp4_awq_act_amax_K)
            if up_site.nvfp4_clip_group_amax is not None:
                gmax = _nvfp4_group16_stat(flat)
                if gmax.numel() == up_site.nvfp4_clip_group_amax.numel():
                    torch.maximum(up_site.nvfp4_clip_group_amax, gmax,
                                  out=up_site.nvfp4_clip_group_amax)
            fvk.quantize_fp8_device(
                int(flat.data_ptr()), int(x_fp8.data_ptr()),
                up_act_scale, n_in, cs())
        else:
            fvk.quantize_fp8_static(
                int(flat.data_ptr()), int(x_fp8.data_ptr()),
                up_act_scale, n_in, cs())

        # 2) up_proj GEMM (FP8 -> bf16)
        up_out = torch.empty(M, N_up, dtype=torch.bfloat16, device=device)
        gemm.fp8_nn_dev(
            int(x_fp8.data_ptr()), up_w_ptr, int(up_out.data_ptr()),
            M, N_up, K_up, up_act_scale, up_w_scale, cs(),
        )

        # 3-5) bias + GELU + FP8 quant.
        #   Steady state (G7.10): one fused kernel ``bias_gelu_quantize_fp8_static_bf16``
        #   reads up_out + up_bias, applies tanh-GELU, divides by
        #   down_act_scale (set during calibration), clamps, casts to FP8.
        #   Calibration (rare): keep the 3-launch chain so quantize_fp8_device
        #   can compute and write down_act_scale = max(|gelu(x+bias)|)/448.
        up_fp8 = down_site.ensure_x_fp8(M, device)
        n_mid = M * K_down
        if _STATE.calibrating:
            if up_bias_ptr:
                fvk.add_bias_bf16(
                    int(up_out.data_ptr()), up_bias_ptr, M, N_up, cs())
            fvk.gelu_inplace(int(up_out.data_ptr()), M * N_up, cs())
            if down_site.nvfp4_awq_act_amax_K is not None:
                amax = _nvfp4_awq_channel_stat(up_out)
                torch.maximum(down_site.nvfp4_awq_act_amax_K, amax,
                              out=down_site.nvfp4_awq_act_amax_K)
            if down_site.nvfp4_clip_group_amax is not None:
                gmax = _nvfp4_group16_stat(up_out)
                if gmax.numel() == down_site.nvfp4_clip_group_amax.numel():
                    torch.maximum(down_site.nvfp4_clip_group_amax, gmax,
                                  out=down_site.nvfp4_clip_group_amax)
            fvk.quantize_fp8_device(
                int(up_out.data_ptr()), int(up_fp8.data_ptr()),
                down_act_scale, n_mid, cs())
        else:
            fvk.bias_gelu_quantize_fp8_static_bf16(
                int(up_out.data_ptr()),
                up_bias_ptr,                       # 0 if no bias
                int(up_fp8.data_ptr()),
                down_act_scale,
                M, N_up, cs())

        # 6) down_proj GEMM
        down_out = torch.empty(M, N_down, dtype=torch.bfloat16, device=device)
        gemm.fp8_nn_dev(
            int(up_fp8.data_ptr()), down_w_ptr, int(down_out.data_ptr()),
            M, N_down, K_down, down_act_scale, down_w_scale, cs(),
        )

        # 7) down bias — G7.9: skip if a downstream kernel will fold
        # the bias into the gated residual (bias_gate_mul_residual_bf16).
        # bias_skip flag is read dynamically so it can be flipped post-
        # install by _modulate_fuse_swap.
        if down_bias_ptr and not down_site.bias_skip:
            fvk.add_bias_bf16(
                int(down_out.data_ptr()), down_bias_ptr, M, N_down, cs())

        if in_dtype != torch.bfloat16:
            down_out = down_out.to(in_dtype)
        return down_out.view(*in_shape[:-1], N_down)

    return forward


# ──────────────────────────────────────────────────────────────────
# Install
# ──────────────────────────────────────────────────────────────────

def install_fp8_swap(model, gemm: Optional[fvk.GemmRunner] = None) -> dict:
    """Walk the model and swap all hot-path Linear / FFN forwards from
    BF16 (G3b/d) to FP8 W8A8.

    Must run AFTER install_fvk_linears (G3b) and install_fvk_ffns (G3d):
    those put weights in [K,N] layout; G4 reads from there.

    Returns a stats dict.
    """
    from flash_rt.models.motus._linear_swap import (
        _HOT_PATH_PREFIXES, _SKIP_PREFIXES,
    )
    if gemm is None:
        gemm = getattr(model, '_g3b_gemm', None) or fvk.GemmRunner()
        model._g3b_gemm = gemm

    counts = {
        'linear_replaced': 0, 'ffn_replaced': 0,
        'skipped_scope': 0, 'skipped_dtype': 0, 'skipped_struct': 0,
        'skipped_bypassed_by_ffn': 0, 'skipped_align': 0,
        'skipped_g4_scope': 0,
    }

    # cuBLASLt FP8 GEMM requires N and K to be multiples of 16 on
    # sm_120 (and typically 8 elsewhere). Linears that fail this stay
    # on the BF16 path from G3b. Encountered at G4 debug with
    # action_encoder.0 (K=14 = action_dim).
    def _fp8_aligned(K: int, N: int) -> bool:
        return (K % 16 == 0) and (N % 16 == 0)

    # G4 scope is INTENTIONALLY narrower than G3b. Per-tensor FP8 on
    # the small Action / Und expert paths catastrophically degraded
    # action cosine (0.33 — verified at G4 debug); the action chunk
    # is only 8 tokens wide, and any per-tensor scale that clips a
    # small-magnitude signal in one of the 30 layers destroys the
    # whole prediction. Video DiT, by contrast, has 360 tokens and
    # tolerates per-tensor FP8 cleanly (cos(frames) = 0.9999 with
    # full quant). Keep G4 to where it's safe; revisit Action / Und
    # in a separate gate (per-channel or NVFP4 group quant).
    G4_FP8_PREFIXES = (
        'video_model.wan_model.blocks.',  # heavy DiT layers (30 blocks)
        # G7.13 follow-up: extend to wan_model.head (M=360, called 10x
        # per replay) — same layout as Wan blocks (per-tensor FP8 OK).
        # Excluded text_embedding / time_embedding / time_projection
        # because those are fp32 under autocast (would need separate
        # fp32->fp8 path).
        'video_model.wan_model.head.',
    )

    def _in_g4_scope(name: str) -> bool:
        return any(name.startswith(p) for p in G4_FP8_PREFIXES)

    # First pass: collect FFN Sequential children so we don't double-quantize
    # them (the FFN bypass owns these Linears' GEMMs now).
    ffn_owned_linears: set[int] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        in_hot = any(name.startswith(p) for p in _HOT_PATH_PREFIXES)
        in_skip = any(name.startswith(p) for p in _SKIP_PREFIXES)
        if (not in_hot) or in_skip:
            continue
        children = list(module)
        if len(children) != 3:
            continue
        if not (isinstance(children[0], nn.Linear)
                and isinstance(children[1], nn.GELU)
                and isinstance(children[2], nn.Linear)):
            continue
        ffn_owned_linears.add(id(children[0]))
        ffn_owned_linears.add(id(children[2]))

    # Second pass: install FP8 sites.
    sites_by_module: dict[int, _Fp8Site] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_hot = any(name.startswith(p) for p in _HOT_PATH_PREFIXES)
            in_skip = any(name.startswith(p) for p in _SKIP_PREFIXES)
            if (not in_hot) or in_skip:
                counts['skipped_scope'] += 1
                continue
            if module.weight.dtype != torch.bfloat16 \
                    or not module.weight.is_cuda:
                counts['skipped_dtype'] += 1
                continue
            if id(module) in ffn_owned_linears:
                # Owned by an FFN Sequential — handled in the FFN pass.
                counts['skipped_bypassed_by_ffn'] += 1
                continue
            # G4 narrow scope: only video_model.wan_model.blocks.*
            if not _in_g4_scope(name):
                counts['skipped_g4_scope'] += 1
                continue
            # Alignment check — weight is [K, N] post-G3b transpose.
            K_mod = int(module.weight.shape[0])
            N_mod = int(module.weight.shape[1])
            if not _fp8_aligned(K_mod, N_mod):
                logger.info(
                    f"[g4] SKIP align {name}: K={K_mod} N={N_mod}; "
                    f"keeping G3b BF16 path")
                counts['skipped_align'] += 1
                continue
            # _Fp8Site replaces module.weight.data with FP8 in-place.
            site = _Fp8Site(module.weight, module.bias, label=name)
            module.forward = _make_fp8_linear_forward(site, gemm)
            # G6.7: pin the site on the module for downstream lookup
            # (e.g. _modulate_fuse_swap toggles bias_skip).
            module._fp8_site = site
            sites_by_module[id(module)] = site
            counts['linear_replaced'] += 1
            if _TRACE:
                logger.info(
                    f"[g4] linear {name}: K={site.K}, N={site.N}, "
                    f"w_scale={site.w_scale.item():.4e}")

    # FFNs.
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        in_hot = any(name.startswith(p) for p in _HOT_PATH_PREFIXES)
        in_skip = any(name.startswith(p) for p in _SKIP_PREFIXES)
        if (not in_hot) or in_skip:
            continue
        children = list(module)
        if len(children) != 3:
            continue
        up, gelu, down = children
        if not (isinstance(up, nn.Linear) and isinstance(gelu, nn.GELU)
                and isinstance(down, nn.Linear)):
            continue
        if up.weight.dtype != torch.bfloat16 or not up.weight.is_cuda:
            counts['skipped_dtype'] += 1
            continue
        if down.weight.dtype != torch.bfloat16 or not down.weight.is_cuda:
            counts['skipped_dtype'] += 1
            continue
        # G4 narrow scope: only video_model.wan_model.blocks.*
        if not _in_g4_scope(name):
            counts['skipped_g4_scope'] += 1
            continue
        # Alignment check (both Linears in [K,N] post-G3b transpose).
        if not (_fp8_aligned(int(up.weight.shape[0]), int(up.weight.shape[1]))
                and _fp8_aligned(int(down.weight.shape[0]),
                                  int(down.weight.shape[1]))):
            logger.info(
                f"[g4] SKIP align FFN {name}: up=({up.weight.shape}) "
                f"down=({down.weight.shape}); keeping G3d BF16 path")
            counts['skipped_align'] += 1
            continue
        try:
            # Each _Fp8Site replaces its Parameter's data with FP8 in-place.
            up_site = _Fp8Site(up.weight, up.bias, label=f"{name}.up")
            down_site = _Fp8Site(down.weight, down.bias, label=f"{name}.down")
            module.forward = _make_fp8_ffn_forward(up_site, down_site, gemm)
            # G7.9: pin sites on module so downstream fusion swaps
            # (e.g. _modulate_fuse_swap toggling down_site.bias_skip)
            # can find them by walking the model.
            module._fp8_up_site = up_site
            module._fp8_down_site = down_site
            counts['ffn_replaced'] += 1
        except AssertionError as e:
            logger.warning(f"[g4] skip FFN {name}: {e}")
            counts['skipped_struct'] += 1

    logger.info(
        f"[g4] FP8 swap: linear={counts['linear_replaced']}, "
        f"ffn={counts['ffn_replaced']}, "
        f"scope-skip={counts['skipped_scope']}, "
        f"dtype-skip={counts['skipped_dtype']}, "
        f"align-skip={counts['skipped_align']}, "
        f"bypassed-by-ffn={counts['skipped_bypassed_by_ffn']}")
    return counts

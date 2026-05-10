"""FlashRT -- RTX GROOT N1.6 torch frontend.

Loads HuggingFace GR00T-N1.6-3B safetensors checkpoints and drives the
framework-agnostic ``GrootSigLIP2`` / ``GrootQwen3`` / ``GrootDiT`` classes
in :mod:`flash_rt.models.groot.pipeline_rtx`.

Usage::

    from flash_rt.hardware.rtx import GrootTorchFrontendRtx
    pipe = GrootTorchFrontendRtx(
        "/path/to/GR00T-N1.6-3B",
        num_views=2,
        embodiment_tag="libero_panda",
    )
    pipe.set_prompt("pick up the red block")
    out = pipe.infer({
        "image": img1,        # numpy uint8 (224, 224, 3)
        "wrist_image": img2,
        "state": np.zeros(state_dim, dtype=np.float32),
    })
    actions = out["actions"]  # numpy (action_horizon, action_dim)

The 3 sub-pipelines (vision, qwen3, dit) each get their own CUDA Graph,
matching the validated Thor design (and Pi0.5 rtx where the whole thing
fits in one graph).
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.hardware.rtx.attn_backend_groot import RtxFlashAttnBackendGroot
from flash_rt.models.groot.pipeline_rtx import (
    GrootDiT,
    GrootQwen3,
    GrootSigLIP2,
    DIT_D,
    DIT_H,
    DIT_HD,
    DIT_L,
    DIT_NH,
    DIT_OUTPUT_DIM,
    QWEN3_D,
    QWEN3_H,
    QWEN3_HD,
    QWEN3_L,
    QWEN3_NHKV,
    QWEN3_NHQ,
    QWEN3_QKV_DIM,
    VIS_D,
    VIS_H,
    VIS_HD,
    VIS_L,
    VIS_MLP1_IN,
    VIS_NH,
    VIS_PATCH_FLAT,
    VIS_SPV,
    VIS_SPV_RAW,
    NUM_FLOW_STEPS,
    ACTION_DIM,
    STATE_DIM,
    ACTION_HORIZON_MAX,
)

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

# ── GROOT N1.6 checkpoint key prefixes ──
VIS_PREFIX = "backbone.model.vision_model.vision_model"
LLM_PREFIX = "backbone.model.language_model.model"
MLP1_PREFIX = "backbone.model.mlp1"
DIT_PREFIX = "action_head.model"
AH_PREFIX = "action_head"

# ── Embodiment id mapping (shared between Thor and rtx, see
#    flash_rt/hardware/groot_embodiments.py) ──
from flash_rt.models.groot.embodiments import (
    EMBODIMENT_TAG_TO_INDEX,
    PUBLIC_TRAINED_TAGS,
    is_embodiment_trained,
)


def _quant_fp8(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Per-tensor FP8 E4M3 quantization."""
    a = float(w.float().abs().max().item())
    s = max(a / 448.0, 1e-12)
    return (w.float() / s).clamp(-448, 448).to(fp8), s


# ════════════════════════════════════════════════════════════════════
#   FP8 calibration — inline (was models/groot/calibration.py)
# ════════════════════════════════════════════════════════════════════

_FP8_MAX = 448.0
_SCALE_FLOOR = 1e-12


def _cal_scale(amax: float) -> float:
    return max(float(amax), _SCALE_FLOOR) / _FP8_MAX


def _load_groot_calibration(checkpoint_path, Se: int) -> Optional[dict]:
    """Return cached GROOT calibration for (checkpoint, Se) or None."""
    from flash_rt.core.quant.calibrator import _checkpoint_hash, _cache_path
    try:
        ckpt_hash = _checkpoint_hash(str(checkpoint_path))
    except FileNotFoundError:
        return None
    cache_file = _cache_path(ckpt_hash, Se)
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read calibration cache %s: %s", cache_file, e)
        return None
    if data.get("ckpt_hash") != ckpt_hash or data.get("Se") != Se:
        return None
    return data


def _save_groot_calibration(checkpoint_path, Se: int,
                            qwen3_scales=None, dit_scales=None) -> None:
    """Merge the given scales into the GROOT calibration cache file."""
    from flash_rt.core.quant.calibrator import _checkpoint_hash, _cache_path
    ckpt_hash = _checkpoint_hash(str(checkpoint_path))
    cache_file = _cache_path(ckpt_hash, Se)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
    data.setdefault("version", 1)
    data["ckpt_hash"] = ckpt_hash
    data["Se"] = Se
    if qwen3_scales is not None:
        data["qwen3_act_scales"] = [float(x) for x in qwen3_scales]
    if dit_scales is not None:
        data["dit_act_scales"] = [float(x) for x in dit_scales]
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved GROOT calibration to %s", cache_file)


def _build_rope_torch(Se: int, HD: int, theta: float = 1e6):
    """Build Qwen3 RoPE cos/sin tables."""
    freqs = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float64) / HD))
    pos = torch.arange(Se, dtype=torch.float64)
    angles = pos[:, None] * freqs[None, :]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(fp16).cuda()
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(fp16).cuda()
    return cos, sin


def _calibrate_qwen3_collect_amax(sd, gemm, fvk, qwen3, ie_fp16, stream=0):
    """FP16 shadow forward through Qwen3, returns 3*L raw activation amax.

    Stateless: each call processes one sample and returns its raw amax
    list (no FP8 scale conversion). Use ``_calibrate_qwen3_per_sample``
    to stack N calls for multi-sample calibration, or
    ``_calibrate_qwen3_impl`` for the legacy single-sample scale path.
    """
    Se = qwen3.Se
    D, H, L = QWEN3_D, QWEN3_H, QWEN3_L
    NHQ, NHKV, HD = QWEN3_NHQ, QWEN3_NHKV, QWEN3_HD
    QKV = QWEN3_QKV_DIM
    prefix = f"{LLM_PREFIX}.layers"

    w_ln, w_ln2 = [], []
    w_qkv, w_qn, w_kn, w_o = [], [], [], []
    w_gu, w_dn = [], []
    for i in range(L):
        lp = f"{prefix}.{i}"
        w_ln.append(sd[f"{lp}.input_layernorm.weight"].to(fp16).contiguous())
        w_ln2.append(sd[f"{lp}.post_attention_layernorm.weight"].to(fp16).contiguous())
        q_w = sd[f"{lp}.self_attn.q_proj.weight"]
        k_w = sd[f"{lp}.self_attn.k_proj.weight"]
        v_w = sd[f"{lp}.self_attn.v_proj.weight"]
        w_qkv.append(torch.cat([q_w, k_w, v_w], dim=0).T.contiguous().to(fp16))
        w_qn.append(sd[f"{lp}.self_attn.q_norm.weight"].to(fp16).contiguous())
        w_kn.append(sd[f"{lp}.self_attn.k_norm.weight"].to(fp16).contiguous())
        w_o.append(sd[f"{lp}.self_attn.o_proj.weight"].T.contiguous().to(fp16))
        g_w = sd[f"{lp}.mlp.gate_proj.weight"]
        u_w = sd[f"{lp}.mlp.up_proj.weight"]
        w_gu.append(torch.cat([g_w, u_w], dim=0).T.contiguous().to(fp16))
        w_dn.append(sd[f"{lp}.mlp.down_proj.weight"].T.contiguous().to(fp16))

    cos_full, sin_full = _build_rope_torch(max(Se, 1024), HD)
    cos_r = cos_full[:Se].unsqueeze(1)
    sin_r = sin_full[:Se].unsqueeze(1)

    def rotate_half(t):
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    x = ie_fp16[:Se].clone().contiguous()
    xn = torch.empty_like(x)
    amax_list = []

    for i in range(L):
        fvk.rms_norm_fp16(
            x.data_ptr(), w_ln[i].data_ptr(), xn.data_ptr(),
            Se, D, 1e-6, stream)
        torch.cuda.synchronize()
        amax_list.append(float(xn[:Se].float().abs().max().item()))

        qkv = torch.empty(Se, QKV, dtype=fp16, device="cuda")
        gemm.fp16_nn(
            xn.data_ptr(), w_qkv[i].data_ptr(), qkv.data_ptr(),
            Se, QKV, D, stream)

        Q = qkv[:, :NHQ * HD].contiguous().view(Se, NHQ, HD)
        K = qkv[:, NHQ * HD:NHQ * HD + NHKV * HD].contiguous().view(Se, NHKV, HD)
        V = qkv[:, NHQ * HD + NHKV * HD:].contiguous().view(Se, NHKV, HD)

        fvk.rms_norm_fp16(Q.data_ptr(), w_qn[i].data_ptr(), Q.data_ptr(),
                          Se * NHQ, HD, 1e-6, stream)
        fvk.rms_norm_fp16(K.data_ptr(), w_kn[i].data_ptr(), K.data_ptr(),
                          Se * NHKV, HD, 1e-6, stream)
        torch.cuda.synchronize()

        Qf, Kf = Q.float(), K.float()
        Q = (Qf * cos_r + rotate_half(Qf) * sin_r).to(fp16)
        K = (Kf * cos_r + rotate_half(Kf) * sin_r).to(fp16)

        q_s = Q.unsqueeze(0).transpose(1, 2)
        k_s = K.unsqueeze(0).transpose(1, 2).repeat_interleave(NHQ // NHKV, dim=1)
        v_s = V.to(fp16).unsqueeze(0).transpose(1, 2).repeat_interleave(NHQ // NHKV, dim=1)
        attn = F.scaled_dot_product_attention(q_s, k_s, v_s)
        attn_flat = attn.transpose(1, 2).reshape(Se, D).to(fp16).contiguous()

        o_out = torch.empty(Se, D, dtype=fp16, device="cuda")
        gemm.fp16_nn(attn_flat.data_ptr(), w_o[i].data_ptr(), o_out.data_ptr(),
                     Se, D, D, stream)
        fvk.residual_add_fp16(x.data_ptr(), o_out.data_ptr(), Se * D, stream)

        fvk.rms_norm_fp16(x.data_ptr(), w_ln2[i].data_ptr(), xn.data_ptr(),
                          Se, D, 1e-6, stream)
        torch.cuda.synchronize()
        amax_list.append(float(xn[:Se].float().abs().max().item()))

        gu = torch.empty(Se, 2 * H, dtype=fp16, device="cuda")
        gemm.fp16_nn(xn.data_ptr(), w_gu[i].data_ptr(), gu.data_ptr(),
                     Se, 2 * H, D, stream)
        gate = gu[:, :H].contiguous()
        up = gu[:, H:].contiguous()
        silu_up = (F.silu(gate.float()) * up.float()).to(fp16).contiguous()
        torch.cuda.synchronize()
        amax_list.append(float(silu_up.float().abs().max().item()))

        dn = torch.empty(Se, D, dtype=fp16, device="cuda")
        gemm.fp16_nn(silu_up.data_ptr(), w_dn[i].data_ptr(), dn.data_ptr(),
                     Se, D, H, stream)
        fvk.residual_add_fp16(x.data_ptr(), dn.data_ptr(), Se * D, stream)

    return amax_list


def _calibrate_qwen3_impl(sd, gemm, fvk, qwen3, ie_fp16, stream=0):
    """FP16 shadow forward through Qwen3, returns 3*L FP8 activation scales.

    Thin wrapper preserved for backward compatibility with the legacy
    single-sample calibration path. Bit-equal to the pre-refactor
    implementation.
    """
    return [_cal_scale(a) for a in _calibrate_qwen3_collect_amax(
        sd, gemm, fvk, qwen3, ie_fp16, stream)]


def _calibrate_qwen3_per_sample(sd, gemm, fvk, qwen3, ie_fp16_list, stream=0):
    """Run Qwen3 amax collection N times.

    Returns an ``[N, 3*L]`` float32 numpy array of raw amax values, ready
    for ``flash_rt.core.calibration.accumulate_amax`` percentile reduction.
    """
    if not ie_fp16_list:
        raise ValueError("ie_fp16_list must be non-empty")
    rows = [_calibrate_qwen3_collect_amax(sd, gemm, fvk, qwen3, ie, stream)
            for ie in ie_fp16_list]
    return np.asarray(rows, dtype=np.float32)


def _calibrate_dit_collect_amax(sd, gemm, fvk, dit, cal_tensors, state_feat,
                                actions_fp32, stream=0):
    """FP16 shadow forward through DiT (step=0), returns 3*L raw amax.

    Stateless: each call processes one sample and returns its raw amax
    list. Cross-attention layers (``l % 2 == 0``) have no learnable
    pre-attention scale; their first slot is filled with ``float('nan')``
    as a sentinel so multi-sample reducers can preserve the constant
    placeholder ``1.0 / _FP8_MAX`` that the FP8 path expects.
    """
    Se = dit.Se
    T = dit.T
    Sa = dit.Sa
    D, H, NH, HD, L = DIT_D, DIT_H, DIT_NH, DIT_HD, DIT_L
    prefix = f"{DIT_PREFIX}.transformer_blocks"

    ada_scales = cal_tensors["ada_scales"]
    ada_shifts = cal_tensors["ada_shifts"]
    ate = cal_tensors["ate"]
    pos_emb = cal_tensors["pos_emb"]
    ae_w1, ae_b1 = cal_tensors["action_enc_w1"], cal_tensors["action_enc_b1"]
    ae_w2, ae_b2 = cal_tensors["action_enc_w2"], cal_tensors["action_enc_b2"]
    ae_w3, ae_b3 = cal_tensors["action_enc_w3"], cal_tensors["action_enc_b3"]

    w_qkv_self, w_qkv_self_b = [None] * L, [None] * L
    w_q, w_q_b = [None] * L, [None] * L
    w_o, w_o_b = [None] * L, [None] * L
    w_ff_up, w_ff_up_b = [None] * L, [None] * L
    w_ff_dn, w_ff_dn_b = [None] * L, [None] * L
    for l in range(L):
        lp = f"{prefix}.{l}"
        q_w = sd[f"{lp}.attn1.to_q.weight"]
        k_w = sd[f"{lp}.attn1.to_k.weight"]
        v_w = sd[f"{lp}.attn1.to_v.weight"]
        o_w = sd[f"{lp}.attn1.to_out.0.weight"]
        q_b = sd[f"{lp}.attn1.to_q.bias"].to(fp16).contiguous()
        k_b = sd[f"{lp}.attn1.to_k.bias"].to(fp16).contiguous()
        v_b = sd[f"{lp}.attn1.to_v.bias"].to(fp16).contiguous()
        o_b = sd[f"{lp}.attn1.to_out.0.bias"].to(fp16).contiguous()
        w_q[l] = q_w.T.contiguous().to(fp16)
        w_q_b[l] = q_b
        w_o[l] = o_w.T.contiguous().to(fp16)
        w_o_b[l] = o_b
        if l % 2 == 1:
            qkv_m = torch.cat([q_w, k_w, v_w], dim=0).T.contiguous().to(fp16)
            qkv_b = torch.cat([q_b, k_b, v_b]).to(fp16).contiguous()
            w_qkv_self[l] = qkv_m
            w_qkv_self_b[l] = qkv_b
        up_w = sd[f"{lp}.ff.net.0.proj.weight"].T.contiguous().to(fp16)
        up_b = sd[f"{lp}.ff.net.0.proj.bias"].to(fp16).contiguous()
        dn_w = sd[f"{lp}.ff.net.2.weight"].T.contiguous().to(fp16)
        dn_b = sd[f"{lp}.ff.net.2.bias"].to(fp16).contiguous()
        w_ff_up[l], w_ff_up_b[l] = up_w, up_b
        w_ff_dn[l], w_ff_dn_b[l] = dn_w, dn_b

    actions_fp16 = actions_fp32[:T].to(fp16).contiguous()
    a_emb = torch.empty(T, D, dtype=fp16, device="cuda")
    gemm.fp16_nn(actions_fp16.data_ptr(), ae_w1.data_ptr(), a_emb.data_ptr(),
                 T, D, actions_fp16.shape[-1], stream)
    fvk.add_bias_fp16(a_emb.data_ptr(), ae_b1.data_ptr(), T, D, stream)

    concat = torch.empty(T, 2 * D, dtype=fp16, device="cuda")
    fvk.gpu_copy(concat.data_ptr(), a_emb.data_ptr(), T * D * 2, stream)
    fvk.gpu_copy(concat.data_ptr() + T * D * 2, ate[0].data_ptr(), T * D * 2, stream)

    enc_h = torch.empty(T, D, dtype=fp16, device="cuda")
    gemm.fp16_nn(concat.data_ptr(), ae_w2.data_ptr(), enc_h.data_ptr(),
                 T, D, 2 * D, stream)
    fvk.add_bias_fp16(enc_h.data_ptr(), ae_b2.data_ptr(), T, D, stream)
    fvk.silu_inplace_fp16(enc_h.data_ptr(), T * D, stream)

    gemm.fp16_nn(enc_h.data_ptr(), ae_w3.data_ptr(), a_emb.data_ptr(),
                 T, D, D, stream)
    fvk.add_bias_fp16(a_emb.data_ptr(), ae_b3.data_ptr(), T, D, stream)
    fvk.residual_add_fp16(a_emb.data_ptr(), pos_emb.data_ptr(), T * D, stream)

    hidden = torch.empty(Sa, D, dtype=fp16, device="cuda")
    fvk.gpu_copy(hidden.data_ptr(), state_feat.data_ptr(), D * 2, stream)
    fvk.gpu_copy(hidden.data_ptr() + D * 2, a_emb.data_ptr(), T * D * 2, stream)
    torch.cuda.synchronize()

    h_norm = torch.empty(Sa, D, dtype=fp16, device="cuda")
    amax_list = []

    for l in range(L):
        is_self = (l % 2 == 1)
        fvk.ada_layer_norm_fp16(
            hidden.data_ptr(),
            ada_scales[0, l].data_ptr(), ada_shifts[0, l].data_ptr(),
            h_norm.data_ptr(), Sa, D, 1e-5, stream)
        torch.cuda.synchronize()

        if is_self:
            amax_list.append(float(h_norm[:Sa].float().abs().max().item()))
        else:
            amax_list.append(float('nan'))

        if is_self:
            qkv = torch.empty(Sa, 3 * D, dtype=fp16, device="cuda")
            gemm.fp16_nn(h_norm.data_ptr(), w_qkv_self[l].data_ptr(),
                         qkv.data_ptr(), Sa, 3 * D, D, stream)
            fvk.add_bias_fp16(qkv.data_ptr(), w_qkv_self_b[l].data_ptr(),
                              Sa, 3 * D, stream)
            Q = qkv[:, :D].contiguous().view(Sa, NH, HD)
            K = qkv[:, D:2 * D].contiguous().view(Sa, NH, HD)
            V = qkv[:, 2 * D:].contiguous().view(Sa, NH, HD)
            q_s = Q.unsqueeze(0).transpose(1, 2)
            k_s = K.unsqueeze(0).transpose(1, 2)
            v_s = V.unsqueeze(0).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q_s, k_s, v_s)
            attn_flat = attn.transpose(1, 2).reshape(Sa, D).to(fp16).contiguous()
        else:
            Q = torch.empty(Sa, D, dtype=fp16, device="cuda")
            gemm.fp16_nn(h_norm.data_ptr(), w_q[l].data_ptr(), Q.data_ptr(),
                         Sa, D, D, stream)
            fvk.add_bias_fp16(Q.data_ptr(), w_q_b[l].data_ptr(), Sa, D, stream)
            cross_idx = l // 2
            K_pre = dit.attn.dit_cross_K[cross_idx][:Se]
            V_pre = dit.attn.dit_cross_V[cross_idx][:Se]
            q_s = Q.view(Sa, NH, HD).unsqueeze(0).transpose(1, 2)
            k_s = K_pre.unsqueeze(0).transpose(1, 2)
            v_s = V_pre.unsqueeze(0).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q_s, k_s, v_s)
            attn_flat = attn.transpose(1, 2).reshape(Sa, D).to(fp16).contiguous()

        o_out = torch.empty(Sa, D, dtype=fp16, device="cuda")
        gemm.fp16_nn(attn_flat.data_ptr(), w_o[l].data_ptr(), o_out.data_ptr(),
                     Sa, D, D, stream)
        fvk.add_bias_fp16(o_out.data_ptr(), w_o_b[l].data_ptr(), Sa, D, stream)
        fvk.residual_add_fp16(hidden.data_ptr(), o_out.data_ptr(), Sa * D, stream)

        fvk.layer_norm_no_affine_fp16(hidden.data_ptr(), h_norm.data_ptr(),
                                       Sa, D, 1e-5, stream)
        torch.cuda.synchronize()
        amax_list.append(float(h_norm[:Sa].float().abs().max().item()))

        ff_h = torch.empty(Sa, H, dtype=fp16, device="cuda")
        gemm.fp16_nn(h_norm.data_ptr(), w_ff_up[l].data_ptr(), ff_h.data_ptr(),
                     Sa, H, D, stream)
        fvk.add_bias_fp16(ff_h.data_ptr(), w_ff_up_b[l].data_ptr(), Sa, H, stream)
        fvk.gelu_inplace_fp16(ff_h.data_ptr(), Sa * H, stream)
        torch.cuda.synchronize()
        amax_list.append(float(ff_h[:Sa].float().abs().max().item()))

        ff_out = torch.empty(Sa, D, dtype=fp16, device="cuda")
        gemm.fp16_nn(ff_h.data_ptr(), w_ff_dn[l].data_ptr(), ff_out.data_ptr(),
                     Sa, D, H, stream)
        fvk.add_bias_fp16(ff_out.data_ptr(), w_ff_dn_b[l].data_ptr(), Sa, D, stream)
        fvk.residual_add_fp16(hidden.data_ptr(), ff_out.data_ptr(), Sa * D, stream)

    return amax_list


def _calibrate_dit_impl(sd, gemm, fvk, dit, cal_tensors, state_feat,
                        actions_fp32, stream=0):
    """FP16 shadow forward through DiT (step=0), returns 3*L FP8 activation scales.

    Thin wrapper preserved for backward compatibility with the legacy
    single-sample calibration path. ``NaN`` slots emitted by the
    collector for cross-attention layers are mapped to the constant
    ``1.0 / _FP8_MAX`` placeholder, matching the pre-refactor behavior.
    """
    raw = _calibrate_dit_collect_amax(
        sd, gemm, fvk, dit, cal_tensors, state_feat, actions_fp32, stream)
    return [(1.0 / _FP8_MAX) if math.isnan(a) else _cal_scale(a) for a in raw]


def _calibrate_dit_per_sample(sd, gemm, fvk, dit, cal_tensors_list,
                              state_feat_list, actions_fp32_list, stream=0):
    """Run DiT amax collection for N samples.

    Returns an ``[N, 3*L]`` float32 numpy array of raw amax values. Cross
    layers contribute ``NaN`` at their first slot in every row; downstream
    reducers must treat all-NaN columns as placeholders rather than
    propagating NaN into the final FP8 scale.

    All three list arguments (``cal_tensors_list``, ``state_feat_list``,
    ``actions_fp32_list``) must have the same length, one entry per
    calibration sample.
    """
    n = len(cal_tensors_list)
    if n == 0:
        raise ValueError("cal_tensors_list must be non-empty")
    if len(state_feat_list) != n or len(actions_fp32_list) != n:
        raise ValueError(
            f"length mismatch: cal_tensors={n}, "
            f"state_feat={len(state_feat_list)}, "
            f"actions_fp32={len(actions_fp32_list)}")
    rows = [_calibrate_dit_collect_amax(
                sd, gemm, fvk, dit, cal_tensors_list[i],
                state_feat_list[i], actions_fp32_list[i], stream)
            for i in range(n)]
    return np.asarray(rows, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════
#   GrootTorchFrontendRtx frontend
# ════════════════════════════════════════════════════════════════════


class GrootTorchFrontendRtx:
    """RTX consumer GPU GROOT N1.6 torch frontend.

    Mirrors the :class:`ThorPipelineTorchGroot` public API
    (``set_prompt`` + ``infer``) so the same eval scripts work on both
    hardware families.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        embodiment_tag: str = "new_embodiment",
        action_horizon: int = ACTION_HORIZON_MAX,
        use_fp8: bool = True,
    ):
        if not (1 <= int(action_horizon) <= ACTION_HORIZON_MAX):
            raise ValueError(
                f"action_horizon must be in [1, {ACTION_HORIZON_MAX}], "
                f"got {action_horizon}")
        if embodiment_tag not in EMBODIMENT_TAG_TO_INDEX:
            raise ValueError(
                f"Unknown embodiment_tag {embodiment_tag!r}. "
                f"Known tags: {sorted(EMBODIMENT_TAG_TO_INDEX.keys())}. "
                f"Trained in GR00T-N1.6-3B: {PUBLIC_TRAINED_TAGS}.")
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._num_views = int(num_views)
        self.use_fp8 = bool(use_fp8)
        self._embodiment_tag = embodiment_tag
        self._embodiment_id = EMBODIMENT_TAG_TO_INDEX[embodiment_tag]
        self._calibrated = False
        if not is_embodiment_trained(embodiment_tag):
            logger.warning(
                "embodiment_tag=%r (id=%d) is NOT trained in the GR00T-N1.6-3B "
                "base checkpoint — per-embodiment MLP weights are at "
                "initialization and the model will emit noise-like actions. "
                "Pick one of %s for a demo, or fine-tune this slot before "
                "deployment.",
                embodiment_tag, self._embodiment_id, PUBLIC_TRAINED_TAGS,
            )

        self.latency_records: list[float] = []
        self._graphs_built = False

        # ── Init kernels ──
        from flash_rt import flash_rt_kernels as fvk
        self._fvk = fvk
        self._gemm = fvk.GemmRunner()
        self._cudart = ctypes.CDLL("libcudart.so")

        # ── Load checkpoint into memory ──
        self._load_checkpoint()

        # ── Action / state dims (padded max — actual depends on embodiment) ──
        self._action_horizon = int(action_horizon)

        logger.info(
            "GrootTorchFrontendRtx initialised (num_views=%d, embodiment=%s id=%d)",
            self._num_views, embodiment_tag, self._embodiment_id,
        )

    # ─────────────────────────────────────────────────────────────
    #   Checkpoint loading
    # ─────────────────────────────────────────────────────────────

    def _load_checkpoint(self) -> None:
        """Load all safetensors files into a state dict on GPU."""
        from safetensors import safe_open

        st_files = sorted(self._checkpoint_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(
                f"No safetensors found in {self._checkpoint_dir}")
        logger.info("Loading %d safetensors files...", len(st_files))
        sd = {}
        for f in st_files:
            with safe_open(str(f), framework="pt", device="cuda") as sf:
                for k in sf.keys():
                    sd[k] = sf.get_tensor(k)
        logger.info("Loaded %d tensors", len(sd))
        self._sd = sd

        # Extract token embeddings (needed for set_prompt)
        self._qwen3_embed = sd[f"{LLM_PREFIX}.embed_tokens.weight"]

    # ─────────────────────────────────────────────────────────────
    #   SigLIP2 weight loading
    # ─────────────────────────────────────────────────────────────

    def _build_siglip_weights(self) -> dict:
        """Build the weights dict for GrootSigLIP2 from the loaded sd.

        FP8-quantizes QKV / O / FFN-up / FFN-down per layer; keeps norms
        and biases as fp16. Stores all tensors on the frontend (not
        deleted) so the data_ptr() values stay valid for the captured
        graph.
        """
        sd = self._sd
        store = self._weight_store
        logger.info("Building SigLIP2 weights (FP8 quantize)...")
        prefix = f"{VIS_PREFIX}.encoder.layers"

        ln_attn_w_list, ln_attn_b_list = [], []
        ln_ffn_w_list, ln_ffn_b_list = [], []
        qkv_b_list, o_b_list, up_b_list, down_b_list = [], [], [], []
        fp8_weights = {}

        def _stash(t):
            store.append(t)
            return t

        for i in range(VIS_L):
            lp = f"{prefix}.{i}"
            ln_attn_w_list.append(_stash(sd[f"{lp}.layer_norm1.weight"].to(fp16).contiguous()))
            ln_attn_b_list.append(_stash(sd[f"{lp}.layer_norm1.bias"].to(fp16).contiguous()))
            ln_ffn_w_list.append(_stash(sd[f"{lp}.layer_norm2.weight"].to(fp16).contiguous()))
            ln_ffn_b_list.append(_stash(sd[f"{lp}.layer_norm2.bias"].to(fp16).contiguous()))

            # QKV (cat then transpose) → FP8
            qkv_cat = torch.cat(
                [sd[f"{lp}.self_attn.{p}_proj.weight"] for p in ("q", "k", "v")],
                dim=0,
            )  # (3D, D)
            qkv_T = qkv_cat.T.contiguous()           # (D, 3D)
            qkv_fp8, qs = _quant_fp8(qkv_T)
            qkv_scale = torch.tensor([qs], dtype=torch.float32, device="cuda")
            _stash(qkv_fp8); _stash(qkv_scale)
            fp8_weights[f"vision_attn_qkv_w_{i}"] = (qkv_fp8.data_ptr(), qkv_scale.data_ptr())

            qkv_b_list.append(_stash(torch.cat(
                [sd[f"{lp}.self_attn.{p}_proj.bias"] for p in ("q", "k", "v")]).to(fp16).contiguous()))

            # O proj → FP8
            o_T = sd[f"{lp}.self_attn.out_proj.weight"].T.contiguous()
            o_fp8, os_ = _quant_fp8(o_T)
            o_scale = torch.tensor([os_], dtype=torch.float32, device="cuda")
            _stash(o_fp8); _stash(o_scale)
            fp8_weights[f"vision_attn_o_w_{i}"] = (o_fp8.data_ptr(), o_scale.data_ptr())
            o_b_list.append(_stash(sd[f"{lp}.self_attn.out_proj.bias"].to(fp16).contiguous()))

            # FFN up → FP8
            up_T = sd[f"{lp}.mlp.fc1.weight"].T.contiguous()
            up_fp8, us = _quant_fp8(up_T)
            up_scale = torch.tensor([us], dtype=torch.float32, device="cuda")
            _stash(up_fp8); _stash(up_scale)
            fp8_weights[f"vision_ffn_up_w_{i}"] = (up_fp8.data_ptr(), up_scale.data_ptr())
            up_b_list.append(_stash(sd[f"{lp}.mlp.fc1.bias"].to(fp16).contiguous()))

            # FFN down → FP8
            dn_T = sd[f"{lp}.mlp.fc2.weight"].T.contiguous()
            dn_fp8, ds = _quant_fp8(dn_T)
            dn_scale = torch.tensor([ds], dtype=torch.float32, device="cuda")
            _stash(dn_fp8); _stash(dn_scale)
            fp8_weights[f"vision_ffn_down_w_{i}"] = (dn_fp8.data_ptr(), dn_scale.data_ptr())
            down_b_list.append(_stash(sd[f"{lp}.mlp.fc2.bias"].to(fp16).contiguous()))

        # Stack per-layer lists into (L, ...) tensors so we can pass per-layer
        # data_ptr() via tensor.data_ptr() + i * stride. Or simpler: keep as
        # Python list of int ptrs.
        ln_attn_w_ptrs = [w.data_ptr() for w in ln_attn_w_list]
        ln_attn_b_ptrs = [w.data_ptr() for w in ln_attn_b_list]
        ln_ffn_w_ptrs = [w.data_ptr() for w in ln_ffn_w_list]
        ln_ffn_b_ptrs = [w.data_ptr() for w in ln_ffn_b_list]
        qkv_b_ptrs = [w.data_ptr() for w in qkv_b_list]
        o_b_ptrs = [w.data_ptr() for w in o_b_list]
        up_b_ptrs = [w.data_ptr() for w in up_b_list]
        down_b_ptrs = [w.data_ptr() for w in down_b_list]

        # Patch embedding (Linear, NOT Conv2d): HF stores (1152, 588) =
        # (out, in). For our row-major NN GEMM we want (in=588, out=1152) → .T
        pe_w = _stash(sd[f"{VIS_PREFIX}.embeddings.patch_embedding.weight"]
                       .T.contiguous().to(fp16))   # (588, 1152)
        pe_b = _stash(sd[f"{VIS_PREFIX}.embeddings.patch_embedding.bias"]
                       .to(fp16).contiguous())     # (1152,)
        pos_emb = _stash(sd[f"{VIS_PREFIX}.embeddings.position_embedding.weight"]
                          .to(fp16).contiguous())  # (256, 1152)

        # Post-LayerNorm
        post_ln_w = _stash(sd[f"{VIS_PREFIX}.post_layernorm.weight"].to(fp16).contiguous())
        post_ln_b = _stash(sd[f"{VIS_PREFIX}.post_layernorm.bias"].to(fp16).contiguous())

        # mlp1: LN(4608) → Linear(4608→2048) → GELU → Linear(2048→2048)
        mlp1_ln_w = _stash(sd[f"{MLP1_PREFIX}.0.weight"].to(fp16).contiguous())
        mlp1_ln_b = _stash(sd[f"{MLP1_PREFIX}.0.bias"].to(fp16).contiguous())
        mlp1_fc1_w = _stash(sd[f"{MLP1_PREFIX}.1.weight"].T.contiguous().to(fp16))
        mlp1_fc1_b = _stash(sd[f"{MLP1_PREFIX}.1.bias"].to(fp16).contiguous())
        mlp1_fc2_w = _stash(sd[f"{MLP1_PREFIX}.3.weight"].T.contiguous().to(fp16))
        mlp1_fc2_b = _stash(sd[f"{MLP1_PREFIX}.3.bias"].to(fp16).contiguous())

        return {
            "vision_patch_embedding_w": pe_w.data_ptr(),
            "vision_patch_embedding_b": pe_b.data_ptr(),
            "vision_position_embedding": pos_emb.data_ptr(),
            "vision_pre_attn_norm_w": ln_attn_w_ptrs,
            "vision_pre_attn_norm_b": ln_attn_b_ptrs,
            "vision_pre_ffn_norm_w": ln_ffn_w_ptrs,
            "vision_pre_ffn_norm_b": ln_ffn_b_ptrs,
            "vision_attn_qkv_b": qkv_b_ptrs,
            "vision_attn_o_b": o_b_ptrs,
            "vision_ffn_up_b": up_b_ptrs,
            "vision_ffn_down_b": down_b_ptrs,
            "vision_post_norm_w": post_ln_w.data_ptr(),
            "vision_post_norm_b": post_ln_b.data_ptr(),
            "mlp1_ln_w": mlp1_ln_w.data_ptr(),
            "mlp1_ln_b": mlp1_ln_b.data_ptr(),
            "mlp1_fc1_w": mlp1_fc1_w.data_ptr(),
            "mlp1_fc1_b": mlp1_fc1_b.data_ptr(),
            "mlp1_fc2_w": mlp1_fc2_w.data_ptr(),
            "mlp1_fc2_b": mlp1_fc2_b.data_ptr(),
            "fp8": fp8_weights,
        }

    # ─────────────────────────────────────────────────────────────
    #   Qwen3 weight loading
    # ─────────────────────────────────────────────────────────────

    def _build_qwen3_weights(self) -> dict:
        sd = self._sd
        store = self._weight_store
        logger.info("Building Qwen3 weights (16 layers, FP8 quantize)...")
        prefix = f"{LLM_PREFIX}.layers"

        ln_attn_w_ptrs, ln_ffn_w_ptrs = [], []
        q_norm_w_ptrs, k_norm_w_ptrs = [], []
        o_w_ptrs = []
        fp8_weights = {}

        def _stash(t):
            store.append(t)
            return t

        for i in range(QWEN3_L):
            lp = f"{prefix}.{i}"

            ln_attn_w_ptrs.append(_stash(sd[f"{lp}.input_layernorm.weight"].to(fp16).contiguous()).data_ptr())
            ln_ffn_w_ptrs.append(_stash(sd[f"{lp}.post_attention_layernorm.weight"].to(fp16).contiguous()).data_ptr())
            q_norm_w_ptrs.append(_stash(sd[f"{lp}.self_attn.q_norm.weight"].to(fp16).contiguous()).data_ptr())
            k_norm_w_ptrs.append(_stash(sd[f"{lp}.self_attn.k_norm.weight"].to(fp16).contiguous()).data_ptr())

            # QKV merged: HF stores per-proj (out, in); cat then .T → (D, QKV_DIM)
            q_w = sd[f"{lp}.self_attn.q_proj.weight"]
            k_w = sd[f"{lp}.self_attn.k_proj.weight"]
            v_w = sd[f"{lp}.self_attn.v_proj.weight"]
            qkv_T = torch.cat([q_w, k_w, v_w], dim=0).T.contiguous()
            qkv_fp8, qs = _quant_fp8(qkv_T)
            qkv_scale = torch.tensor([qs], dtype=torch.float32, device="cuda")
            _stash(qkv_fp8); _stash(qkv_scale)
            fp8_weights[f"qwen3_qkv_w_{i}"] = (qkv_fp8.data_ptr(), qkv_scale.data_ptr())

            # O proj fp16 (small M)
            o_w = sd[f"{lp}.self_attn.o_proj.weight"].T.contiguous().to(fp16)
            _stash(o_w)
            o_w_ptrs.append(o_w.data_ptr())

            # Gate+Up merged
            g_w = sd[f"{lp}.mlp.gate_proj.weight"]
            u_w = sd[f"{lp}.mlp.up_proj.weight"]
            gu_T = torch.cat([g_w, u_w], dim=0).T.contiguous()
            gu_fp8, gs = _quant_fp8(gu_T)
            gu_scale = torch.tensor([gs], dtype=torch.float32, device="cuda")
            _stash(gu_fp8); _stash(gu_scale)
            fp8_weights[f"qwen3_gate_up_w_{i}"] = (gu_fp8.data_ptr(), gu_scale.data_ptr())

            # Down
            dn_T = sd[f"{lp}.mlp.down_proj.weight"].T.contiguous()
            dn_fp8, ds = _quant_fp8(dn_T)
            dn_scale = torch.tensor([ds], dtype=torch.float32, device="cuda")
            _stash(dn_fp8); _stash(dn_scale)
            fp8_weights[f"qwen3_down_w_{i}"] = (dn_fp8.data_ptr(), dn_scale.data_ptr())

        final_norm_w = _stash(sd[f"{LLM_PREFIX}.norm.weight"].to(fp16).contiguous())
        vlln_w = _stash(sd[f"{AH_PREFIX}.vlln.weight"].to(fp16).contiguous())
        vlln_b = _stash(sd[f"{AH_PREFIX}.vlln.bias"].to(fp16).contiguous())

        return {
            "qwen3_ln_attn_w": ln_attn_w_ptrs,
            "qwen3_ln_ffn_w": ln_ffn_w_ptrs,
            "qwen3_q_norm_w": q_norm_w_ptrs,
            "qwen3_k_norm_w": k_norm_w_ptrs,
            "qwen3_o_w_fp16": o_w_ptrs,
            "qwen3_final_norm_w": final_norm_w.data_ptr(),
            "vlln_w": vlln_w.data_ptr(),
            "vlln_b": vlln_b.data_ptr(),
            "fp8": fp8_weights,
        }

    # ─────────────────────────────────────────────────────────────
    #   DiT weight loading
    # ─────────────────────────────────────────────────────────────

    def _build_dit_weights(self) -> dict:
        sd = self._sd
        store = self._weight_store
        logger.info("Building DiT weights (32 blocks)...")
        prefix = f"{DIT_PREFIX}.transformer_blocks"

        q_w_fp16_ptrs, q_b_ptrs = [], []
        k_w_fp16_ptrs, k_b_ptrs = [], []
        v_w_fp16_ptrs, v_b_ptrs = [], []
        o_w_fp16_ptrs, o_b_ptrs = [], []
        ff_up_b_ptrs, ff_down_b_ptrs = [], []
        qkv_b_self_ptrs = []
        fp8_weights = {}

        def _stash(t):
            store.append(t)
            return t

        # Per-step / per-layer norm1.linear (used during _precompute_conditioning)
        norm1_lin_w_list = []
        norm1_lin_b_list = []

        for l in range(DIT_L):
            is_self = (l % 2 == 1)
            lp = f"{prefix}.{l}"

            # Q/K/V/O fp16 weights
            q_w_T = sd[f"{lp}.attn1.to_q.weight"].T.contiguous().to(fp16)
            k_w_T = sd[f"{lp}.attn1.to_k.weight"].T.contiguous().to(fp16)
            v_w_T = sd[f"{lp}.attn1.to_v.weight"].T.contiguous().to(fp16)
            o_w_T = sd[f"{lp}.attn1.to_out.0.weight"].T.contiguous().to(fp16)
            q_b = sd[f"{lp}.attn1.to_q.bias"].to(fp16).contiguous()
            k_b = sd[f"{lp}.attn1.to_k.bias"].to(fp16).contiguous()
            v_b = sd[f"{lp}.attn1.to_v.bias"].to(fp16).contiguous()
            o_b = sd[f"{lp}.attn1.to_out.0.bias"].to(fp16).contiguous()
            for t in (q_w_T, k_w_T, v_w_T, o_w_T, q_b, k_b, v_b, o_b):
                _stash(t)
            q_w_fp16_ptrs.append(q_w_T.data_ptr())
            k_w_fp16_ptrs.append(k_w_T.data_ptr())
            v_w_fp16_ptrs.append(v_w_T.data_ptr())
            o_w_fp16_ptrs.append(o_w_T.data_ptr())
            q_b_ptrs.append(q_b.data_ptr())
            k_b_ptrs.append(k_b.data_ptr())
            v_b_ptrs.append(v_b.data_ptr())
            o_b_ptrs.append(o_b.data_ptr())

            # Self-attn merged QKV (only at odd layers)
            if is_self:
                qkv_m = torch.cat([
                    sd[f"{lp}.attn1.to_q.weight"],
                    sd[f"{lp}.attn1.to_k.weight"],
                    sd[f"{lp}.attn1.to_v.weight"],
                ], dim=0).T.contiguous()
                qkv_fp8, qs = _quant_fp8(qkv_m)
                qkv_scale = torch.tensor([qs], dtype=torch.float32, device="cuda")
                _stash(qkv_fp8); _stash(qkv_scale)
                fp8_weights[f"dit_qkv_w_{l}"] = (qkv_fp8.data_ptr(), qkv_scale.data_ptr())
                qkv_b = torch.cat([q_b, k_b, v_b]).to(fp16).contiguous()
                _stash(qkv_b)
                qkv_b_self_ptrs.append(qkv_b.data_ptr())
            else:
                # Dummy entry; the pipeline only reads at odd indices
                qkv_b_self_ptrs.append(0)

            # FFN up + down → FP8
            up_T = sd[f"{lp}.ff.net.0.proj.weight"].T.contiguous()
            dn_T = sd[f"{lp}.ff.net.2.weight"].T.contiguous()
            up_fp8, us = _quant_fp8(up_T)
            dn_fp8, ds = _quant_fp8(dn_T)
            up_scale = torch.tensor([us], dtype=torch.float32, device="cuda")
            dn_scale = torch.tensor([ds], dtype=torch.float32, device="cuda")
            _stash(up_fp8); _stash(dn_fp8); _stash(up_scale); _stash(dn_scale)
            fp8_weights[f"dit_ff_up_w_{l}"] = (up_fp8.data_ptr(), up_scale.data_ptr())
            fp8_weights[f"dit_ff_down_w_{l}"] = (dn_fp8.data_ptr(), dn_scale.data_ptr())

            ff_up_b = sd[f"{lp}.ff.net.0.proj.bias"].to(fp16).contiguous()
            ff_dn_b = sd[f"{lp}.ff.net.2.bias"].to(fp16).contiguous()
            _stash(ff_up_b); _stash(ff_dn_b)
            ff_up_b_ptrs.append(ff_up_b.data_ptr())
            ff_down_b_ptrs.append(ff_dn_b.data_ptr())

            # norm1.linear (for conditioning precompute, not used inside the
            # captured graph — the pipeline reads pre-baked ada_scales/shifts)
            norm1_lin_w = sd[f"{lp}.norm1.linear.weight"].T.contiguous().to(fp16)
            norm1_lin_b = sd[f"{lp}.norm1.linear.bias"].to(fp16).contiguous()
            _stash(norm1_lin_w); _stash(norm1_lin_b)
            norm1_lin_w_list.append(norm1_lin_w)
            norm1_lin_b_list.append(norm1_lin_b)

        # Output projection
        proj_out_1_w = _stash(sd[f"{DIT_PREFIX}.proj_out_1.weight"].T.contiguous().to(fp16))
        proj_out_1_b = _stash(sd[f"{DIT_PREFIX}.proj_out_1.bias"].to(fp16).contiguous())
        proj_out_2_w = _stash(sd[f"{DIT_PREFIX}.proj_out_2.weight"].T.contiguous().to(fp16))
        proj_out_2_b = _stash(sd[f"{DIT_PREFIX}.proj_out_2.bias"].to(fp16).contiguous())

        # Timestep encoder weights (used during conditioning precompute)
        ts_pre = f"{DIT_PREFIX}.timestep_encoder.timestep_embedder"
        ts_l1_w = sd[f"{ts_pre}.linear_1.weight"].T.contiguous().to(fp16)
        ts_l1_b = sd[f"{ts_pre}.linear_1.bias"].to(fp16)
        ts_l2_w = sd[f"{ts_pre}.linear_2.weight"].T.contiguous().to(fp16)
        ts_l2_b = sd[f"{ts_pre}.linear_2.bias"].to(fp16)

        # ── Per-step + per-layer pre-computed conditioning ──
        ada_scales, ada_shifts, out_scales, out_shifts, ate = \
            self._precompute_conditioning(
                ts_l1_w, ts_l1_b, ts_l2_w, ts_l2_b,
                norm1_lin_w_list, norm1_lin_b_list,
                proj_out_1_w, proj_out_1_b,
            )
        for t in (ada_scales, ada_shifts, out_scales, out_shifts, ate):
            _stash(t)

        # ── Per-embodiment MLPs ──
        eid = self._embodiment_id

        def _emb(name):
            t = sd[f"{AH_PREFIX}.{name}"][eid].contiguous().to(fp16)
            _stash(t)
            return t

        def _emb_b(name):
            t = sd[f"{AH_PREFIX}.{name}"][eid].to(fp16).contiguous()
            _stash(t)
            return t

        state_enc_w1 = _emb("state_encoder.layer1.W")
        state_enc_b1 = _emb_b("state_encoder.layer1.b")
        state_enc_w2 = _emb("state_encoder.layer2.W")
        state_enc_b2 = _emb_b("state_encoder.layer2.b")
        action_enc_w1 = _emb("action_encoder.W1.W")
        action_enc_b1 = _emb_b("action_encoder.W1.b")
        action_enc_w2 = _emb("action_encoder.W2.W")
        action_enc_b2 = _emb_b("action_encoder.W2.b")
        action_enc_w3 = _emb("action_encoder.W3.W")
        action_enc_b3 = _emb_b("action_encoder.W3.b")
        action_dec_w1 = _emb("action_decoder.layer1.W")
        action_dec_b1 = _emb_b("action_decoder.layer1.b")
        action_dec_w2 = _emb("action_decoder.layer2.W")
        action_dec_b2 = _emb_b("action_decoder.layer2.b")

        pos_emb = _stash(sd[f"{AH_PREFIX}.position_embedding.weight"].to(fp16).contiguous())

        # Save state encoder for runtime use
        self._state_enc_w1 = state_enc_w1
        self._state_enc_b1 = state_enc_b1
        self._state_enc_w2 = state_enc_w2
        self._state_enc_b2 = state_enc_b2

        # Keep torch tensors alive for the FP8 calibration pass (groot_calibration
        # needs the raw tensors, not just device pointers, to run a shadow forward).
        self._cal_dit_tensors = {
            "ada_scales": ada_scales,
            "ada_shifts": ada_shifts,
            "ate": ate,
            "pos_emb": pos_emb,
            "action_enc_w1": action_enc_w1,
            "action_enc_b1": action_enc_b1,
            "action_enc_w2": action_enc_w2,
            "action_enc_b2": action_enc_b2,
            "action_enc_w3": action_enc_w3,
            "action_enc_b3": action_enc_b3,
        }

        return {
            "dit_q_w_fp16": q_w_fp16_ptrs,
            "dit_q_b": q_b_ptrs,
            "dit_k_w_fp16": k_w_fp16_ptrs,
            "dit_k_b": k_b_ptrs,
            "dit_v_w_fp16": v_w_fp16_ptrs,
            "dit_v_b": v_b_ptrs,
            "dit_o_w_fp16": o_w_fp16_ptrs,
            "dit_o_b": o_b_ptrs,
            "dit_qkv_b_self": qkv_b_self_ptrs,
            "dit_ff_up_b": ff_up_b_ptrs,
            "dit_ff_down_b": ff_down_b_ptrs,
            "ada_scales": ada_scales.data_ptr(),
            "ada_shifts": ada_shifts.data_ptr(),
            "out_scales": out_scales.data_ptr(),
            "out_shifts": out_shifts.data_ptr(),
            "action_time_embeds": ate.data_ptr(),
            "action_enc_w1": action_enc_w1.data_ptr(),
            "action_enc_b1": action_enc_b1.data_ptr(),
            "action_enc_w2": action_enc_w2.data_ptr(),
            "action_enc_b2": action_enc_b2.data_ptr(),
            "action_enc_w3": action_enc_w3.data_ptr(),
            "action_enc_b3": action_enc_b3.data_ptr(),
            "action_dec_w1": action_dec_w1.data_ptr(),
            "action_dec_b1": action_dec_b1.data_ptr(),
            "action_dec_w2": action_dec_w2.data_ptr(),
            "action_dec_b2": action_dec_b2.data_ptr(),
            "pos_emb": pos_emb.data_ptr(),
            "proj_out_2_w": proj_out_2_w.data_ptr(),
            "proj_out_2_b": proj_out_2_b.data_ptr(),
            "fp8": fp8_weights,
        }

    def _precompute_conditioning(
        self, ts_l1_w, ts_l1_b, ts_l2_w, ts_l2_b,
        norm1_lin_w_list, norm1_lin_b_list,
        proj_out_1_w, proj_out_1_b,
    ) -> tuple:
        """Pre-compute (4 steps × 32 layers) of AdaLN scale/shift,
        plus per-step output conditioning, plus action time embeddings.

        Returns 5 fp16 cuda tensors, each contiguously laid out so the
        pipeline can index by (step * L + l) byte offset.
        """
        D = DIT_D
        T = self._action_horizon
        steps = NUM_FLOW_STEPS
        L = DIT_L

        # Sinusoidal time encoding (matches diffusers Timesteps with
        # flip_sin_to_cos=True, downscale_freq_shift=1)
        half_dim = 128
        exp = -torch.arange(half_dim, dtype=torch.float32, device="cuda") * \
            (math.log(10000.0) / half_dim)
        emb_freqs = exp.exp()

        ada_scales = torch.empty(steps, L, D, dtype=fp16, device="cuda")
        ada_shifts = torch.empty(steps, L, D, dtype=fp16, device="cuda")
        out_scales = torch.empty(steps, D, dtype=fp16, device="cuda")
        out_shifts = torch.empty(steps, D, dtype=fp16, device="cuda")
        ate = torch.empty(steps, T, D, dtype=fp16, device="cuda")

        half_d = D // 2
        exp_d = (-torch.arange(half_d, dtype=torch.float, device="cuda") *
                 (math.log(10000.0) / half_d)).exp()

        with torch.no_grad():
            for step in range(steps):
                t_disc = int(step / float(steps) * 1000)
                t_t = torch.tensor([t_disc], dtype=torch.float32, device="cuda")
                args = t_t[:, None] * emb_freqs[None, :]
                sincos = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(fp16)

                temb = F.silu(sincos @ ts_l1_w + ts_l1_b) @ ts_l2_w + ts_l2_b
                silu_temb = F.silu(temb)

                # Per-layer scale/shift
                for l in range(L):
                    ada_out = silu_temb @ norm1_lin_w_list[l] + norm1_lin_b_list[l]
                    sc, sh = ada_out.squeeze(0).chunk(2, dim=0)
                    ada_scales[step, l] = sc
                    ada_shifts[step, l] = sh

                # Output conditioning: shift FIRST, scale SECOND
                out_cond = silu_temb @ proj_out_1_w + proj_out_1_b
                osh, osc = out_cond.squeeze(0).chunk(2, dim=0)
                out_scales[step] = osc
                out_shifts[step] = osh

                # Action time embedding (T, D)
                t_expanded = torch.full((T,), t_disc, device="cuda")
                freqs = t_expanded.unsqueeze(-1).float() * exp_d
                te = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).to(fp16)
                ate[step] = te

        return ada_scales, ada_shifts, out_scales, out_shifts, ate

    # ─────────────────────────────────────────────────────────────
    #   Public API
    # ─────────────────────────────────────────────────────────────

    def set_prompt(self, prompt: str) -> None:
        """Tokenize and prepare text embeddings for Qwen3.

        GROOT N1.6's tokenizer is the standard Qwen3-1.7B tokenizer plus
        three Eagle special tokens (<img>, </img>, <IMG_CONTEXT>) at IDs
        151670, 151671, 151669. The GROOT checkpoint's embed_tokens matrix
        is 151680 rows (= 151643 base vocab + 37 added special tokens), so
        these IDs are valid.

        We don't need a fancy "Eagle tokenizer" — we just load the base
        Qwen3-1.7B tokenizer (downloaded once to a local cache) and bypass
        it for the three special tokens which we splice in by ID.
        """
        from transformers import AutoTokenizer

        if not hasattr(self, "_tokenizer"):
            tokenizer_candidates = [
                str(self._checkpoint_dir),                    # if user dropped one in
                str(self._checkpoint_dir / "tokenizer"),
                "/tmp/qwen3_tok",                              # local cache
                "/root/.cache/qwen3_tok",
                "Qwen/Qwen3-1.7B",                             # HF download (~7 MB)
            ]
            for tok_path in tokenizer_candidates:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        tok_path, trust_remote_code=True)
                    logger.info("Loaded Qwen3 tokenizer from %s", tok_path)
                    break
                except Exception:
                    continue
            if not hasattr(self, "_tokenizer"):
                raise RuntimeError(
                    "Cannot load Qwen3 tokenizer. Pre-download it via "
                    "`hf download Qwen/Qwen3-1.7B --include 'tokenizer*' "
                    "'vocab*' 'merges*' --local-dir /tmp/qwen3_tok`")
            self._img_token_id = 151669
            self._img_start_id = 151670
            self._img_end_id = 151671

        S_img = self._num_views * VIS_SPV
        text_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = (text_ids + [self._img_start_id] +
                    [self._img_token_id] * S_img + [self._img_end_id])

        self._input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
        self._text_len = len(text_ids)
        self._Se = len(full_ids)
        self._prompt_text = prompt

        # Pre-compute text embeddings (image positions get filled at infer time)
        self._text_embeds = F.embedding(self._input_ids, self._qwen3_embed)

        self._image_mask = (self._input_ids == self._img_token_id)
        self._backbone_mask = torch.ones(1, self._Se, dtype=torch.bool, device="cuda")
        self._non_img_mask = (~self._image_mask) & self._backbone_mask
        self._img_in_text_mask = self._image_mask & self._backbone_mask

        # v0 limitation: once the pipeline is built, prompts can only be
        # changed if their token length matches. The graph captures
        # encoder_seq_max-sized buffers and the qwen3 + dit graphs depend
        # on the exact Se. A safer-but-slower fallback would tear everything
        # down and rebuild — that requires keeping _sd alive. Punt to v1.
        if self._graphs_built:
            raise RuntimeError(
                "set_prompt() after the pipeline is built is not supported in "
                "v0; construct a new GrootTorchFrontendRtx instance for a new prompt")

        logger.info(
            "Prompt set: '%s' (%d text + %d img = %d total tokens)",
            prompt[:50], self._text_len, S_img, self._Se,
        )

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Unified FP8 calibration entry point.

        Args:
            observations: single observation dict, list of dicts, or any
                iterable of dicts. ``N=1`` falls back to the legacy
                single-sample calibration path (bit-equal). ``N>=2``
                runs per-sample shadow forwards through Qwen3 and DiT,
                collects per-tensor amax for both stages, reduces along
                the sample axis via ``numpy.percentile`` and writes the
                final scales back into the device buffers before graph
                replay.
            percentile: percentile to apply in multi-sample mode.
                ``100.0`` is equivalent to a traditional max reduction;
                ``99.9`` (default) clips outlier frames. Ignored for
                ``N=1``.
            max_samples: optional cap on samples consumed from the
                iterable.
            verbose: if True, log per-stage amax dispersion summaries
                after reduction.
        """
        if not hasattr(self, "_input_ids"):
            raise RuntimeError("set_prompt() must be called before calibrate()")
        if self._calibrated:
            logger.warning(
                "calibrate() called a second time; returning without re-running.")
            return

        if isinstance(observations, dict):
            obs_list = [observations]
        elif isinstance(observations, list):
            obs_list = observations
        else:
            obs_list = list(observations)
        if max_samples is not None:
            obs_list = obs_list[:max_samples]
        n = len(obs_list)
        if n == 0:
            raise ValueError("observations must contain at least 1 sample")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")

        if n == 1:
            self._calibrate_single_frame(obs_list[0])
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    @property
    def precision_spec(self):
        """:class:`ModelPrecisionSpec` captured at calibration time, or ``None``."""
        return getattr(self, "_precision_spec", None)

    def infer(self, obs: dict) -> dict:
        """Run full GROOT E2E inference: images → actions.

        First call captures graphs (~5 s). Subsequent calls use graph replay.

        Args:
            obs: dict with
                'image': numpy uint8 (224, 224, 3)
                'wrist_image': numpy uint8 (224, 224, 3)
                'state': numpy float32 (state_dim,)

        Returns:
            dict with 'actions': numpy (action_horizon, action_dim)
        """
        if not hasattr(self, "_input_ids"):
            raise RuntimeError("set_prompt() must be called before infer()")

        if not self._graphs_built:
            self._build_pipeline_and_capture(obs)

        t0 = time.perf_counter()

        # ── 1. Patch embed images + run SigLIP graph ──
        self._upload_images(obs)
        self._sig_graph.replay()
        torch.cuda.synchronize()

        # ── 2. Pixel unshuffle + mlp1 (torch outside graph) ──
        self._run_pixel_unshuffle_mlp1()

        # ── 3. Build input embeddings + replay Qwen3 graph ──
        self._g_ie_buf.copy_(self._text_embeds.squeeze(0).to(fp16))
        self._g_ie_buf[self._image_mask[0]] = self._g_vision_out.to(fp16)
        # Copy ie buf into pipeline's qwen3.bufs[x]
        self._cudart.cudaMemcpyAsync(
            self._qwen3.bufs["x"].ptr,
            ctypes.c_void_p(self._g_ie_buf.data_ptr()),
            self._Se * QWEN3_D * 2, 3, 0)  # D2D
        self._qwen3_graph.replay()
        torch.cuda.synchronize()

        # ── 4. Build kv_text/kv_img + precompute cross-attn KV + DiT graph replay ──
        # backbone_features is in qwen3.bufs[backbone_features]; mask split here
        self._build_dit_kv_inputs()

        # State encoding (outside graph)
        state = obs.get("state", np.zeros(STATE_DIM, dtype=np.float32))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(torch.float32).cuda()
        self._encode_state_into_buf(state)

        # Init noise
        self._dit_actions_torch.normal_()
        self._cudart.cudaMemcpyAsync(
            self._dit.bufs["actions"].ptr,
            ctypes.c_void_p(self._dit_actions_torch.data_ptr()),
            self._action_horizon * ACTION_DIM * 4, 3, 0)

        # Precompute cross K/V (NOT in graph — depends on kv_text/kv_img which
        # change per-prompt). Same as Thor.
        self._dit.precompute_cross_kv()

        self._dit_graph.replay()
        torch.cuda.synchronize()

        # Read actions back
        self._cudart.cudaMemcpy(
            ctypes.c_void_p(self._dit_actions_torch.data_ptr()),
            self._dit.bufs["actions"].ptr,
            self._action_horizon * ACTION_DIM * 4, 2)  # D2H

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        return {
            "actions": self._dit_actions_torch.cpu().numpy(),
        }

    # ─────────────────────────────────────────────────────────────
    #   Helpers (image upload, pixel unshuffle, state encode, KV split)
    # ─────────────────────────────────────────────────────────────

    def _upload_images(self, obs: dict) -> None:
        """Convert observation images to fp16, write into pipeline's input slot."""
        views = [obs["image"]]
        if "wrist_image" in obs and self._num_views >= 2:
            views.append(obs["wrist_image"])
        nv = self._num_views
        # Normalize per ImageNet-like (-1, 1) — same as Thor
        imgs = []
        for im in views[:nv]:
            arr = im.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            imgs.append(arr)
        stacked = np.stack(imgs).astype(np.float16)  # (nv, 224, 224, 3) fp16
        # Copy to GPU then into pipeline buffer
        gpu = torch.from_numpy(stacked).cuda()
        self._cudart.cudaMemcpyAsync(
            self._sig.bufs["input_images"].ptr,
            ctypes.c_void_p(gpu.data_ptr()),
            gpu.numel() * 2, 3, 0)
        # Save reference so it survives until after replay
        self._last_img_gpu = gpu

    def _run_pixel_unshuffle_mlp1(self) -> None:
        """Pixel unshuffle (256→64) + mlp1 (4608→2048→2048).

        Done in torch outside the graph because pixel_unshuffle is a 4D
        rearrangement that doesn't map to a single fvk strided copy.
        Output written into self._g_vision_out (S_img, 2048) fp16.
        """
        nv = self._num_views
        S = nv * VIS_SPV_RAW       # 512
        S_img = nv * VIS_SPV       # 128
        D = VIS_D
        nH = nW = int(math.sqrt(VIS_SPV_RAW))   # 16

        # Read post-LN output from pipeline's sig_postln buffer into a torch view
        post_ln_torch = torch.empty(S, D, dtype=fp16, device="cuda")
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(post_ln_torch.data_ptr()),
            self._sig.bufs["sig_postln"].ptr,
            S * D * 2, 3, 0)
        # Per-view pixel unshuffle: each view's 256 patches → 64 unshuffled tokens
        post_ln_torch = post_ln_torch.view(nv, VIS_SPV_RAW, D)
        per_view_out = []
        for v in range(nv):
            spatial = post_ln_torch[v].view(1, nH, nW, D).permute(0, 3, 1, 2)
            unsh = F.pixel_unshuffle(spatial, 2)
            flatv = unsh.reshape(VIS_MLP1_IN, -1).T.contiguous()  # (64, 4608)
            per_view_out.append(flatv)
        flat = torch.cat(per_view_out, dim=0)  # (S_img, 4608)

        # mlp1: LN → fp16_nn → bias → GELU → fp16_nn → bias
        # Run via fvk kernels into a fresh torch buffer
        fvk = self._fvk
        gemm = self._gemm
        ln_out = torch.empty(S_img, VIS_MLP1_IN, dtype=fp16, device="cuda")
        fvk.layer_norm_fp16(
            flat.data_ptr(),
            self._mlp1_w["ln_w"], self._mlp1_w["ln_b"],
            ln_out.data_ptr(),
            S_img, VIS_MLP1_IN, 1e-5, 0)

        fc1_out = torch.empty(S_img, QWEN3_D, dtype=fp16, device="cuda")
        gemm.fp16_nn(
            ln_out.data_ptr(),
            self._mlp1_w["fc1_w"],
            fc1_out.data_ptr(),
            S_img, QWEN3_D, VIS_MLP1_IN, 0)
        fvk.add_bias_fp16(
            fc1_out.data_ptr(),
            self._mlp1_w["fc1_b"],
            S_img, QWEN3_D, 0)
        fvk.gelu_inplace_fp16(fc1_out.data_ptr(), S_img * QWEN3_D, 0)

        fc2_out = torch.empty(S_img, QWEN3_D, dtype=fp16, device="cuda")
        gemm.fp16_nn(
            fc1_out.data_ptr(),
            self._mlp1_w["fc2_w"],
            fc2_out.data_ptr(),
            S_img, QWEN3_D, QWEN3_D, 0)
        fvk.add_bias_fp16(
            fc2_out.data_ptr(),
            self._mlp1_w["fc2_b"],
            S_img, QWEN3_D, 0)
        self._g_vision_out = fc2_out

    def _build_dit_kv_inputs(self) -> None:
        """Split Qwen3 backbone_features into kv_text / kv_img by mask."""
        Se = self._Se
        bb_ptr = self._qwen3.bufs["backbone_features"].ptr
        # Read into torch
        bb = torch.empty(Se, QWEN3_D, dtype=fp16, device="cuda")
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(bb.data_ptr()),
            bb_ptr,
            Se * QWEN3_D * 2, 3, 0)
        non_img = self._non_img_mask[0].to(fp16).unsqueeze(-1)
        img = self._img_in_text_mask[0].to(fp16).unsqueeze(-1)
        kv_text = (bb * non_img).contiguous()
        kv_img = (bb * img).contiguous()
        self._cudart.cudaMemcpyAsync(
            self._dit.bufs["kv_text"].ptr,
            ctypes.c_void_p(kv_text.data_ptr()),
            Se * QWEN3_D * 2, 3, 0)
        self._cudart.cudaMemcpyAsync(
            self._dit.bufs["kv_img"].ptr,
            ctypes.c_void_p(kv_img.data_ptr()),
            Se * QWEN3_D * 2, 3, 0)
        self._last_kv_text = kv_text
        self._last_kv_img = kv_img

    def _encode_state_into_buf(self, state: torch.Tensor) -> None:
        """Run state_encoder (2-layer MLP) and write into dit.bufs[state_feat]."""
        state = state.to(fp16).contiguous()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = torch.empty(1, 1024, dtype=fp16, device="cuda")
        self._gemm.fp16_nn(
            state.data_ptr(),
            self._state_enc_w1.data_ptr(),
            h.data_ptr(),
            1, 1024, STATE_DIM, 0)
        self._fvk.add_bias_fp16(
            h.data_ptr(),
            self._state_enc_b1.data_ptr(),
            1, 1024, 0)
        self._fvk.relu_inplace_fp16(h.data_ptr(), 1024, 0)
        sf = torch.empty(1, DIT_D, dtype=fp16, device="cuda")
        self._gemm.fp16_nn(
            h.data_ptr(),
            self._state_enc_w2.data_ptr(),
            sf.data_ptr(),
            1, DIT_D, 1024, 0)
        self._fvk.add_bias_fp16(
            sf.data_ptr(),
            self._state_enc_b2.data_ptr(),
            1, DIT_D, 0)
        self._cudart.cudaMemcpyAsync(
            self._dit.bufs["state_feat"].ptr,
            ctypes.c_void_p(sf.data_ptr()),
            DIT_D * 2, 3, 0)
        self._last_state_feat = sf

    # ─────────────────────────────────────────────────────────────
    #   Pipeline construction + graph capture
    # ─────────────────────────────────────────────────────────────

    def _build_pipeline_and_capture(self, obs: dict,
                                    release_sd: bool = True) -> None:
        """Lazy: build pipeline, run once for FP8 calibration, capture graphs.

        Args:
            obs: a single observation used to drive the initial single-sample
                calibration through SigLIP, Qwen3, and DiT.
            release_sd: if True (default), free ``self._sd`` after capture
                to reclaim GPU memory. Set to False when the caller intends
                to refine the calibration with additional samples (the
                shadow-forward helpers require the raw state dict to be
                alive).
        """
        Se = self._Se
        T = self._action_horizon
        logger.info("Building pipeline + capturing graphs (Se=%d, T=%d)...", Se, T)

        # Stash for weight refs (kept alive)
        self._weight_store = []

        # Build all weights
        sig_weights = self._build_siglip_weights()
        qwen3_weights = self._build_qwen3_weights()
        dit_weights = self._build_dit_weights()

        # Save mlp1 ptrs (used by torch-side pixel_unshuffle path)
        self._mlp1_w = {
            "ln_w": sig_weights["mlp1_ln_w"],
            "ln_b": sig_weights["mlp1_ln_b"],
            "fc1_w": sig_weights["mlp1_fc1_w"],
            "fc1_b": sig_weights["mlp1_fc1_b"],
            "fc2_w": sig_weights["mlp1_fc2_w"],
            "fc2_b": sig_weights["mlp1_fc2_b"],
        }

        # Attention backend
        self._attn = RtxFlashAttnBackendGroot(
            num_views=self._num_views,
            encoder_seq_max=Se,
            num_dit_actions=1 + T,         # Sa
            dit_kv_seq=Se,
        )

        # Build sub-pipelines
        self._sig = GrootSigLIP2(
            self._gemm, self._fvk, self._attn, sig_weights,
            num_views=self._num_views,
        )
        self._qwen3 = GrootQwen3(
            self._gemm, self._fvk, self._attn, qwen3_weights,
            encoder_seq_max=Se,
        )
        self._qwen3.set_seq_len(Se)
        self._dit = GrootDiT(
            self._gemm, self._fvk, self._attn, dit_weights,
            action_horizon=T, encoder_seq=Se,
        )

        # Pre-allocate runtime torch buffers used in the hot path
        self._g_ie_buf = torch.empty(Se, QWEN3_D, dtype=fp16, device="cuda")
        self._dit_actions_torch = torch.zeros(
            T, ACTION_DIM, dtype=torch.float32, device="cuda")

        # ── 1. Run SigLIP once to populate sig_x → calibrate Qwen3 from real input ──
        self._upload_images(obs)
        self._sig.forward(stream=0)
        torch.cuda.synchronize()
        self._run_pixel_unshuffle_mlp1()

        # Build input embeddings
        self._g_ie_buf.copy_(self._text_embeds.squeeze(0).to(fp16))
        self._g_ie_buf[self._image_mask[0]] = self._g_vision_out.to(fp16)

        # ── 2. Calibrate Qwen3 FP8 activation scales ──
        self._calibrate_qwen3()

        # ── 3. Run Qwen3 once with calibrated scales ──
        self._cudart.cudaMemcpy(
            self._qwen3.bufs["x"].ptr,
            ctypes.c_void_p(self._g_ie_buf.data_ptr()),
            Se * QWEN3_D * 2, 3)
        self._qwen3.forward(stream=0)
        torch.cuda.synchronize()

        # ── 4. Build DiT KV inputs + state + initial noise ──
        self._build_dit_kv_inputs()
        state = obs.get("state", np.zeros(STATE_DIM, dtype=np.float32))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(torch.float32).cuda()
        self._encode_state_into_buf(state)
        self._dit_actions_torch.normal_()
        self._cudart.cudaMemcpy(
            self._dit.bufs["actions"].ptr,
            ctypes.c_void_p(self._dit_actions_torch.data_ptr()),
            T * ACTION_DIM * 4, 3)

        # ── 5. Calibrate DiT FP8 activation scales ──
        self._calibrate_dit()

        # ── 6. Capture graphs ──
        self._capture_siglip_graph()
        self._capture_qwen3_graph()
        self._capture_dit_graph()

        # Free the raw state dict (we have all weights extracted into _weight_store)
        if release_sd:
            del self._sd
            torch.cuda.empty_cache()

        self._graphs_built = True
        logger.info("All 3 CUDA Graphs ready")

    # ─────────────────────────────────────────────────────────────
    #   FP8 calibration (Qwen3 + DiT)
    # ─────────────────────────────────────────────────────────────

    def _calibrate_qwen3(self) -> None:
        """Measure per-layer FP8 activation scales for Qwen3 and upload."""
        cached = _load_groot_calibration(self._checkpoint_dir, self._Se)
        if cached and cached.get("qwen3_act_scales"):
            scales = cached["qwen3_act_scales"]
            if len(scales) == QWEN3_L * 3:
                self._qwen3.set_act_scales(scales)
                logger.info("Qwen3 FP8 calibration: loaded %d scales from cache",
                            len(scales))
                return
            logger.warning("Cached Qwen3 scales have wrong length (%d); recomputing",
                           len(scales))

        logger.info("Qwen3 FP8 calibration: running shadow forward...")
        t0 = time.perf_counter()
        scales = _calibrate_qwen3_impl(
            sd=self._sd, gemm=self._gemm, fvk=self._fvk,
            qwen3=self._qwen3, ie_fp16=self._g_ie_buf, stream=0)
        elapsed = time.perf_counter() - t0
        self._qwen3.set_act_scales(scales)
        logger.info("Qwen3 FP8 calibration: %d scales in %.2fs",
                    len(scales), elapsed)

        try:
            _save_groot_calibration(self._checkpoint_dir, self._Se,
                                    qwen3_scales=scales)
        except Exception as e:
            logger.warning("Failed to save Qwen3 calibration cache: %s", e)

    def _calibrate_dit(self) -> None:
        """Measure per-layer FP8 activation scales for DiT and upload."""
        cached = _load_groot_calibration(self._checkpoint_dir, self._Se)
        if cached and cached.get("dit_act_scales"):
            scales = cached["dit_act_scales"]
            if len(scales) == DIT_L * 3:
                self._dit.set_act_scales(scales)
                logger.info("DiT FP8 calibration: loaded %d scales from cache",
                            len(scales))
                return
            logger.warning("Cached DiT scales have wrong length (%d); recomputing",
                           len(scales))

        self._dit.precompute_cross_kv()
        torch.cuda.synchronize()

        logger.info("DiT FP8 calibration: running shadow forward (step=0)...")
        t0 = time.perf_counter()
        scales = _calibrate_dit_impl(
            sd=self._sd, gemm=self._gemm, fvk=self._fvk,
            dit=self._dit, cal_tensors=self._cal_dit_tensors,
            state_feat=self._last_state_feat,
            actions_fp32=self._dit_actions_torch, stream=0)
        elapsed = time.perf_counter() - t0
        self._dit.set_act_scales(scales)
        logger.info("DiT FP8 calibration: %d scales in %.2fs",
                    len(scales), elapsed)

        try:
            _save_groot_calibration(self._checkpoint_dir, self._Se,
                                    dit_scales=scales)
        except Exception as e:
            logger.warning("Failed to save DiT calibration cache: %s", e)
        self._dit._rebuild_alpha_cache()

    # ─────────────────────────────────────────────────────────────
    #   Public calibration entry-point implementations
    # ─────────────────────────────────────────────────────────────

    def _calibrate_single_frame(self, obs: dict) -> None:
        """N=1 path: build pipeline (which calibrates) and snapshot the spec."""
        if not self._graphs_built:
            self._build_pipeline_and_capture(obs)
        self._calibrated = True
        self._precision_spec = self._snapshot_precision_spec(
            method="single_frame", n=1, percentile=None)
        self._warn_if_scale_ceiling_exceeded()
        logger.info("GROOT single-frame calibration complete")

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        """N>=2 path: refine FP8 scales using N samples with percentile reduce.

        Bootstraps the pipeline on the first sample (single-frame path),
        then drives Qwen3 and DiT shadow forwards over all N samples,
        reduces per-tensor amax via ``numpy.percentile`` along the sample
        axis, and uploads the final scales into the existing graph
        buffers (no graph re-capture needed; the captured kernels read
        scales from the buffers at replay time).
        """
        from flash_rt.core.calibration import (
            accumulate_amax,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Calibrating GROOT FP8 across %d samples (percentile=%.2f)...",
            n, percentile)

        if not self._graphs_built:
            self._build_pipeline_and_capture(obs_list[0], release_sd=False)

        # ── Pass A: collect per-sample Qwen3 amax ──
        qwen3_rows = [
            self._collect_qwen3_amax_for_obs(obs) for obs in obs_list
        ]
        qwen3_reduced = accumulate_amax(qwen3_rows, percentile=percentile)
        if verbose:
            logger.info("Qwen3 amax dispersion:\n%s", format_summary(
                summarize_amax_dispersion(qwen3_rows, qwen3_reduced)))
        qwen3_scales = [_cal_scale(float(a)) for a in qwen3_reduced.tolist()]
        self._qwen3.set_act_scales(qwen3_scales)

        # ── Pass B: collect per-sample DiT amax (uses updated Qwen3 scales) ──
        dit_rows = [
            self._collect_dit_amax_for_obs(obs) for obs in obs_list
        ]
        dit_reduced = self._reduce_amax_with_placeholders(
            dit_rows, percentile=percentile)
        dit_scales = [(1.0 / _FP8_MAX) if math.isnan(a) else _cal_scale(float(a))
                      for a in dit_reduced.tolist()]
        self._dit.set_act_scales(dit_scales)

        # ── Cleanup deferred state-dict release ──
        if hasattr(self, "_sd"):
            del self._sd
            torch.cuda.empty_cache()

        self._calibrated = True
        self._precision_spec = self._snapshot_precision_spec(
            method="percentile", n=n, percentile=percentile)
        self._warn_if_scale_ceiling_exceeded(label=f"groot_rtx_N{n}")
        logger.info(
            "GROOT multi-frame calibration complete (N=%d, percentile=%.2f)",
            n, percentile)

    def _collect_qwen3_amax_for_obs(self, obs: dict) -> np.ndarray:
        """Drive vision/embed for ``obs`` then collect Qwen3 raw amax."""
        self._upload_images(obs)
        self._sig.forward(stream=0)
        torch.cuda.synchronize()
        self._run_pixel_unshuffle_mlp1()
        self._g_ie_buf.copy_(self._text_embeds.squeeze(0).to(fp16))
        self._g_ie_buf[self._image_mask[0]] = self._g_vision_out.to(fp16)
        amax = _calibrate_qwen3_collect_amax(
            self._sd, self._gemm, self._fvk, self._qwen3, self._g_ie_buf, 0)
        return np.asarray(amax, dtype=np.float32)

    def _collect_dit_amax_for_obs(self, obs: dict) -> np.ndarray:
        """Run quantized backbone for ``obs`` then collect DiT raw amax."""
        # Vision + text embed → ie_fp16
        self._upload_images(obs)
        self._sig.forward(stream=0)
        torch.cuda.synchronize()
        self._run_pixel_unshuffle_mlp1()
        self._g_ie_buf.copy_(self._text_embeds.squeeze(0).to(fp16))
        self._g_ie_buf[self._image_mask[0]] = self._g_vision_out.to(fp16)

        # Quantized Qwen3 backbone → backbone features in qwen3.bufs
        self._cudart.cudaMemcpy(
            self._qwen3.bufs["x"].ptr,
            ctypes.c_void_p(self._g_ie_buf.data_ptr()),
            self._Se * QWEN3_D * 2, 3)
        self._qwen3.forward(stream=0)
        torch.cuda.synchronize()

        # DiT setup: KV split + state encode + noise + cross-KV precompute
        self._build_dit_kv_inputs()
        state = obs.get("state", np.zeros(STATE_DIM, dtype=np.float32))
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(torch.float32).cuda()
        self._encode_state_into_buf(state)
        self._dit_actions_torch.normal_()
        self._cudart.cudaMemcpy(
            self._dit.bufs["actions"].ptr,
            ctypes.c_void_p(self._dit_actions_torch.data_ptr()),
            self._action_horizon * ACTION_DIM * 4, 3)
        self._dit.precompute_cross_kv()
        torch.cuda.synchronize()

        amax = _calibrate_dit_collect_amax(
            self._sd, self._gemm, self._fvk, self._dit,
            self._cal_dit_tensors, self._last_state_feat,
            self._dit_actions_torch, stream=0)
        return np.asarray(amax, dtype=np.float32)

    @staticmethod
    def _reduce_amax_with_placeholders(rows, *, percentile: float) -> np.ndarray:
        """Percentile-reduce per-sample amax rows, preserving NaN placeholders.

        Columns whose value is ``NaN`` in every row (DiT cross-attention
        layers' first slot) are passed through as ``NaN`` so the FP8
        scale wrapper can map them to the constant placeholder. All
        other columns reduce via ``numpy.percentile``.
        """
        if not rows:
            raise ValueError("rows must contain at least one entry")
        stacked = np.stack(rows, axis=0).astype(np.float64)
        if stacked.ndim != 2:
            raise ValueError(
                f"each amax row must be 1-D, got shapes {[r.shape for r in rows]}")
        out = np.empty(stacked.shape[1], dtype=np.float32)
        for j in range(stacked.shape[1]):
            col = stacked[:, j]
            mask = np.isnan(col)
            if mask.all():
                out[j] = np.float32(np.nan)
            else:
                out[j] = np.float32(np.percentile(col[~mask], percentile))
        return out

    def _read_qwen3_scales(self) -> np.ndarray:
        """Snapshot the current Qwen3 FP8 scale buffer to host."""
        return self._qwen3.act_scales_buf.download_new(
            (QWEN3_L * 3,), np.float32)

    def _read_dit_scales(self) -> np.ndarray:
        """Snapshot the current DiT FP8 scale buffer to host."""
        return self._dit.act_scales_buf.download_new(
            (DIT_L * 3,), np.float32)

    def _warn_if_scale_ceiling_exceeded(self, label: str = "groot_rtx") -> None:
        """Diagnostic warning if any FP8 scale is far above the median."""
        from flash_rt.core.calibration import check_scale_ceiling
        scales: dict = {}
        for i, s in enumerate(self._read_qwen3_scales().tolist()):
            scales[f"qwen3_{i}"] = float(s)
        # Skip the constant placeholder slots in DiT cross-attention layers,
        # whose value is fixed at 1/_FP8_MAX and would distort the median.
        for i, s in enumerate(self._read_dit_scales().tolist()):
            if not math.isclose(float(s), 1.0 / _FP8_MAX):
                scales[f"dit_{i}"] = float(s)
        if scales:
            check_scale_ceiling(scales, label=label)

    def _snapshot_precision_spec(self, *, method: str, n: int,
                                 percentile: Optional[float]):
        """Build a :class:`ModelPrecisionSpec` from current FP8 scales."""
        from flash_rt.core.precision_spec import (
            ModelPrecisionSpec,
            PrecisionSpec,
        )

        spec = ModelPrecisionSpec(source="calibration")

        def _entry(scale_val: float):
            entry = PrecisionSpec(
                dtype="fp8_e4m3",
                granularity="per_tensor",
                scheme="symmetric",
                scale_source="calibration",
                scale=np.array([scale_val], dtype=np.float32),
                calibration_method=method,
                calibration_samples=n,
                calibration_percentile=percentile,
            )
            entry.validate()
            return entry

        for i, s in enumerate(self._read_qwen3_scales().tolist()):
            spec.encoder_layer_specs[f"qwen3_{i}"] = _entry(float(s))
        for i, s in enumerate(self._read_dit_scales().tolist()):
            spec.decoder_layer_specs[f"dit_{i}"] = _entry(float(s))
        return spec

    # ─────────────────────────────────────────────────────────────
    #   Graph capture
    # ─────────────────────────────────────────────────────────────

    def _capture_siglip_graph(self) -> None:
        logger.info("Capturing SigLIP graph...")
        self._sig_torch_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._sig_torch_stream):
            stream_int = self._sig_torch_stream.cuda_stream
            for _ in range(3):
                self._sig.forward(stream=stream_int)
        torch.cuda.synchronize()

        self._sig_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._sig_torch_stream):
            stream_int = self._sig_torch_stream.cuda_stream
            self._sig_graph.capture_begin()
            self._sig.forward(stream=stream_int)
            self._sig_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("  SigLIP graph captured")

    def _capture_qwen3_graph(self) -> None:
        logger.info("Capturing Qwen3 graph...")
        self._qwen3_torch_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._qwen3_torch_stream):
            stream_int = self._qwen3_torch_stream.cuda_stream
            for _ in range(3):
                self._qwen3.forward(stream=stream_int)
        torch.cuda.synchronize()

        self._qwen3_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._qwen3_torch_stream):
            stream_int = self._qwen3_torch_stream.cuda_stream
            self._qwen3_graph.capture_begin()
            self._qwen3.forward(stream=stream_int)
            self._qwen3_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("  Qwen3 graph captured")

    def _capture_dit_graph(self) -> None:
        logger.info("Capturing DiT graph...")
        self._dit_torch_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._dit_torch_stream):
            stream_int = self._dit_torch_stream.cuda_stream
            for _ in range(3):
                self._dit.run_steps(stream=stream_int)
        torch.cuda.synchronize()

        self._dit_graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self._dit_torch_stream):
            stream_int = self._dit_torch_stream.cuda_stream
            self._dit_graph.capture_begin()
            self._dit.run_steps(stream=stream_int)
            self._dit_graph.capture_end()
        torch.cuda.synchronize()
        logger.info("  DiT graph captured")

    # ─────────────────────────────────────────────────────────────
    #   Latency stats
    # ─────────────────────────────────────────────────────────────

    def get_latency_stats(self) -> dict:
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {
            "count": len(lat),
            "mean_ms": float(np.mean(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "min_ms": float(np.min(lat)),
            "hz": float(1000 / np.mean(lat)),
        }

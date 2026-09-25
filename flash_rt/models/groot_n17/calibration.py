"""Static FP8 calibration for the GR00T N1.7 pipeline.

For each FP8 GEMM in the inference graph there are two scales that meet:

  * ``weight_scale`` — baked into the FP8 weights at WeightLoader time
    (Quant transform; stored as ``_<block>_alpha[*]`` as host floats).
  * ``act_scale`` — captured here by running a fp32 shadow forward on
    the WHOLE pipeline (using dequantized FP8 weights as fp16-equivalent
    operands) and recording ``max(|x|) / fp8_max`` at every quant input.

After this routine returns, the frontend (Phase 3c.b2) bakes
``alpha = act_scale * weight_scale`` for every FP8 GEMM and stores the
host-side d_act_scale fp32 device tensors used by
``quantize_fp8_static_fp16``.

Layout note: production forwards expect FP8 weights as the **transposed**
``[K, N]`` matrix (per ``T()`` in the spec). The shadow forward dequants
back to fp16 in that same ``[K, N]`` layout, so ``x @ w_fp16`` gives the
correct production-equivalent GEMM result.

This module is the ONE place that knows about the FP8 quant-point
ordering. It mirrors the per-stage spec scale-list layouts:

  ``_vit_alpha[i]``    — (qkv, o, fc1, fc2)             24 layers
  ``_llm_alpha[i]``    — (qkv, o, gate, up, down)       16 layers
  ``_vlsa_alpha[i]``   — (q, k, v, o, fc1, fc2)          4 layers
  ``_dsm_alpha[i]``    — (fc1, fc2)                      3 mergers
  ``_dit_alpha[i]``    — (q, k, v, o, ada, ff_proj, ff_down)  32 layers — bf16 path
                          (DiT runs bf16 in production; alphas unused)

Returns a dict of ``act_scale`` lists (host floats) keyed by stage:

  ``"vit_act_qkv": [24]``, ``"vit_act_o": [24]``,
  ``"vit_act_fc1": [24]``, ``"vit_act_fc2": [24]``,
  ``"llm_act_qkv": [16]``, ``"llm_act_o": [16]``,
  ``"llm_act_gateup": [16]``, ``"llm_act_down": [16]``,
  ``"vlsa_act_qkv": [4]``, ``"vlsa_act_o": [4]``,
  ``"vlsa_act_fc1": [4]``, ``"vlsa_act_fc2": [4]``,
  ``"deepstack_act_fc1": [3]``, ``"deepstack_act_fc2": [3]``,

plus the post-stage hidden states (used by the next stage's shadow):

  ``"backbone_features": (1, S, 2048)`` — vlsa output (vl_self_attn final).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


FP8_MAX = 448.0


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL ViT 2D rotary position embedding (host-side cos/sin builder)
# ─────────────────────────────────────────────────────────────────────────


def build_vit_rope_tables(
    grid_thw,
    *,
    head_dim: int = 64,
    theta: float = 10000.0,
    spatial_merge_size: int = 2,
    device: str = "cuda",
):
    """Replicate ``Qwen3VLVisionModel.rot_pos_emb(grid_thw)`` host-side.

    Returns ``(cos, sin)`` each contiguous fp16 ``(total_tokens, head_dim)``
    tensors suitable for ``fvk.rope_rotate_half_fp16(x, cos, sin, S, NH, HD)``.

    Args:
        grid_thw: iterable of ``(t, h, w)`` per image group. Example fixture
            (2 views × 2 frames): ``[(1, 16, 16)] * 4``.
    """
    rope_dim_half = head_dim // 2
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, rope_dim_half, 2, dtype=torch.float32) / rope_dim_half
    ))
    max_hw = 0
    for _, h, w in grid_thw:
        max_hw = max(max_hw, int(h), int(w))
    seq = torch.arange(max_hw, dtype=torch.float32)
    freq_table = torch.outer(seq, inv_freq)

    pos_chunks = []
    for num_frames, h, w in grid_thw:
        m = spatial_merge_size
        merged_h, merged_w = int(h) // m, int(w) // m
        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(m)
        intra_col = torch.arange(m)
        row_idx = (block_rows[:, None, None, None] * m
                   + intra_row[None, None, :, None])
        col_idx = (block_cols[None, :, None, None] * m
                   + intra_col[None, None, None, :])
        row_idx = row_idx.expand(merged_h, merged_w, m, m).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, m, m).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if int(num_frames) > 1:
            coords = coords.repeat(int(num_frames), 1)
        pos_chunks.append(coords)
    pos_ids = torch.cat(pos_chunks, dim=0).long()

    embeddings = freq_table[pos_ids]
    embeddings = embeddings.flatten(1)
    emb = torch.cat((embeddings, embeddings), dim=-1)
    cos = emb.cos().to(dtype=torch.float16, device=device).contiguous()
    sin = emb.sin().to(dtype=torch.float16, device=device).contiguous()
    return cos, sin


def _amax(x: torch.Tensor) -> float:
    return float(x.detach().abs().max().item())


def _dequant_fp8(
    w_fp8: torch.Tensor,
    w_scale: float,
    *,
    shadow: dict | None = None,
    key: tuple | None = None,
) -> torch.Tensor:
    """Resolve a shadow weight to fp32 [K, N] for shadow matmul.

    If ``shadow`` is provided and ``key`` is present in it, returns the
    real fp16 weight cast to fp32 (no FP8 quant noise). Otherwise falls
    back to dequantizing the FP8 production weight.
    """
    if shadow is not None and key is not None and key in shadow:
        return shadow[key].float()
    return w_fp8.float() * float(w_scale)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def calibrate_vit(
    fe,
    pixel_features: torch.Tensor,    # (S=1024, 1024) post-patch-embed+pos-embed, fp32
    cos: torch.Tensor, sin: torch.Tensor,    # (S, HD) fp32
    num_views: int,
    deepstack_taps: tuple[int, ...] = (5, 11, 17),
) -> dict:
    """Shadow forward of 24-layer ViT. Returns scales + ViT final + deepstack taps."""
    out: dict = {
        "vit_act_qkv": [], "vit_act_o": [], "vit_act_fc1": [], "vit_act_fc2": [],
        "deepstack_taps": {},
    }
    h = pixel_features
    S, D = h.shape
    NH, HD, FF = 16, 64, 4096
    Sper = S // num_views
    shadow = getattr(fe, "_fp16_shadow_weights", None)

    for li in range(24):
        norm1_w = fe._vit_ln1_w[li].float(); norm1_b = fe._vit_ln1_b[li].float()
        norm2_w = fe._vit_ln2_w[li].float(); norm2_b = fe._vit_ln2_b[li].float()
        qkv_w = _dequant_fp8(fe._vit_qkv_w[li], fe._vit_alpha[li * 4 + 0], shadow=shadow, key=("vit", li, "qkv"))
        qkv_b = fe._vit_qkv_b[li].float()
        o_w = _dequant_fp8(fe._vit_o_w[li], fe._vit_alpha[li * 4 + 1], shadow=shadow, key=("vit", li, "o"))
        o_b = fe._vit_o_b[li].float()
        fc1_w = _dequant_fp8(fe._vit_fc1_w[li], fe._vit_alpha[li * 4 + 2], shadow=shadow, key=("vit", li, "fc1"))
        fc1_b = fe._vit_fc1_b[li].float()
        fc2_w = _dequant_fp8(fe._vit_fc2_w[li], fe._vit_alpha[li * 4 + 3], shadow=shadow, key=("vit", li, "fc2"))
        fc2_b = fe._vit_fc2_b[li].float()

        xn = F.layer_norm(h, (D,), norm1_w, norm1_b, eps=1e-6)
        out["vit_act_qkv"].append(_amax(xn))

        qkv = xn @ qkv_w + qkv_b   # (S, 3*D)
        Q = qkv[:, :D].view(S, NH, HD)
        K = qkv[:, D:2 * D].view(S, NH, HD)
        V = qkv[:, 2 * D:].view(S, NH, HD)
        ce = cos.unsqueeze(-2); se = sin.unsqueeze(-2)
        Q = Q * ce + _rotate_half(Q) * se
        K = K * ce + _rotate_half(K) * se
        # Multi-view per-image attention (4 chunks of Sper)
        Qg = Q.view(num_views, Sper, NH, HD).permute(0, 2, 1, 3)
        Kg = K.view(num_views, Sper, NH, HD).permute(0, 2, 1, 3)
        Vg = V.view(num_views, Sper, NH, HD).permute(0, 2, 1, 3)
        scores = (Qg @ Kg.transpose(-2, -1)) / (HD ** 0.5)
        attn_o = (scores.softmax(-1) @ Vg).permute(0, 2, 1, 3).reshape(S, D)
        out["vit_act_o"].append(_amax(attn_o))

        o = attn_o @ o_w + o_b
        h = h + o

        xn2 = F.layer_norm(h, (D,), norm2_w, norm2_b, eps=1e-6)
        out["vit_act_fc1"].append(_amax(xn2))

        fc1 = F.gelu(xn2 @ fc1_w + fc1_b, approximate="tanh")
        out["vit_act_fc2"].append(_amax(fc1))

        h = h + (fc1 @ fc2_w + fc2_b)

        if li in deepstack_taps:
            out["deepstack_taps"][li] = h.clone()

    out["vit_final"] = h
    return out


def calibrate_deepstack(fe, vit_taps: dict[int, torch.Tensor]) -> dict:
    """Shadow forward of 3 deepstack mergers. Returns scales + 3 features."""
    out: dict = {"deepstack_act_fc1": [], "deepstack_act_fc2": [], "features": []}
    shadow = getattr(fe, "_fp16_shadow_weights", None)
    for j, layer in enumerate((5, 11, 17)):
        x = vit_taps[layer].view(-1, 4096)   # spatial 4:1 merge → (256, 4096)
        norm_w = getattr(fe, f"_dsm{j}_norm_w").float()
        norm_b = getattr(fe, f"_dsm{j}_norm_b").float()
        fc1_w = _dequant_fp8(getattr(fe, f"_dsm{j}_fc1_w"), fe._dsm_alpha[j * 2 + 0], shadow=shadow, key=("dsm", j, "fc1"))
        fc1_b = getattr(fe, f"_dsm{j}_fc1_b").float()
        fc2_w = _dequant_fp8(getattr(fe, f"_dsm{j}_fc2_w"), fe._dsm_alpha[j * 2 + 1], shadow=shadow, key=("dsm", j, "fc2"))
        fc2_b = getattr(fe, f"_dsm{j}_fc2_b").float()

        xn = F.layer_norm(x, (4096,), norm_w, norm_b, eps=1e-6)
        out["deepstack_act_fc1"].append(_amax(xn))

        fc1 = F.gelu(xn @ fc1_w + fc1_b, approximate="tanh")
        out["deepstack_act_fc2"].append(_amax(fc1))

        out["features"].append(fc1 @ fc2_w + fc2_b)
    return out


def calibrate_llm(
    fe,
    llm_input: torch.Tensor,           # (1, S, D=2048) fp32
    rope_cos: torch.Tensor,            # (S, HD=128)
    rope_sin: torch.Tensor,
    visual_pos_mask: torch.Tensor,     # (S,) bool
    deepstack_features: list[torch.Tensor],   # 3 × (visual_count, D)
) -> dict:
    """Shadow forward of 16-layer truncated LLM with M-RoPE + DeepStack inject."""
    out: dict = {
        "llm_act_qkv": [], "llm_act_o": [],
        "llm_act_gateup": [], "llm_act_down": [],
    }
    h = llm_input.squeeze(0)   # (S, D)
    S, D = h.shape
    NHQ, NHKV, HD, FF = 16, 8, 128, 6144
    GQA = NHQ // NHKV
    ce = rope_cos.unsqueeze(-2); se = rope_sin.unsqueeze(-2)
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=h.device), diagonal=1)
    shadow = getattr(fe, "_fp16_shadow_weights", None)

    for li in range(16):
        in_ln = fe._llm_input_ln_w[li].float()
        post_ln = fe._llm_post_ln_w[li].float()
        q_n = fe._llm_q_norm_w[li].float()
        k_n = fe._llm_k_norm_w[li].float()
        # qkv weight is fused [D, NHQ*HD + 2*NHKV*HD]; but we have separate q/k/v
        # in the spec? Actually the spec has Cat(q,k,v) → fused. So _llm_qkv_w is fused.
        # _llm_qkv_w shape after Cat(dim=0)+T()+Quant() is (D, qkv_out_dim).
        qkv_w = _dequant_fp8(fe._llm_qkv_w[li], fe._llm_alpha[li * 5 + 0], shadow=shadow, key=("llm", li, "qkv"))  # (D, 4096)
        o_w = _dequant_fp8(fe._llm_o_w[li], fe._llm_alpha[li * 5 + 1], shadow=shadow, key=("llm", li, "o"))
        gate_w = _dequant_fp8(fe._llm_gate_w[li], fe._llm_alpha[li * 5 + 2], shadow=shadow, key=("llm", li, "gate"))
        up_w = _dequant_fp8(fe._llm_up_w[li], fe._llm_alpha[li * 5 + 3], shadow=shadow, key=("llm", li, "up"))
        down_w = _dequant_fp8(fe._llm_down_w[li], fe._llm_alpha[li * 5 + 4], shadow=shadow, key=("llm", li, "down"))

        xn = _rms_norm(h, in_ln, eps=1e-6)
        out["llm_act_qkv"].append(_amax(xn))

        qkv = xn @ qkv_w   # (S, 4096)
        Q = qkv[:, :NHQ * HD].view(S, NHQ, HD)
        K = qkv[:, NHQ * HD:NHQ * HD + NHKV * HD].view(S, NHKV, HD)
        V = qkv[:, NHQ * HD + NHKV * HD:].view(S, NHKV, HD)
        Q = _rms_norm(Q, q_n, eps=1e-6)
        K = _rms_norm(K, k_n, eps=1e-6)
        Q = Q * ce + _rotate_half(Q) * se
        K = K * ce + _rotate_half(K) * se
        # GQA expand 8→16
        K_e = K.unsqueeze(2).expand(S, NHKV, GQA, HD).reshape(S, NHQ, HD)
        V_e = V.unsqueeze(2).expand(S, NHKV, GQA, HD).reshape(S, NHQ, HD)
        Qa = Q.permute(1, 0, 2)
        Ka = K_e.permute(1, 0, 2)
        Va = V_e.permute(1, 0, 2)
        scores = (Qa @ Ka.transpose(-2, -1)) / (HD ** 0.5)
        scores = scores.masked_fill(causal, float("-inf"))
        attn_o = (scores.softmax(-1) @ Va).permute(1, 0, 2).reshape(S, NHQ * HD)
        out["llm_act_o"].append(_amax(attn_o))
        o = attn_o @ o_w
        h = h + o

        xn2 = _rms_norm(h, post_ln, eps=1e-6)
        out["llm_act_gateup"].append(_amax(xn2))
        gate_v = xn2 @ gate_w
        up_v = xn2 @ up_w
        gu = F.silu(gate_v) * up_v
        out["llm_act_down"].append(_amax(gu))
        h = h + gu @ down_w

        if li in (0, 1, 2):
            ds = deepstack_features[li]
            h = h.clone()
            h[visual_pos_mask] = h[visual_pos_mask] + ds

    out["llm_final"] = h.unsqueeze(0)   # (1, S, D)
    return out


def calibrate_vlsa(fe, llm_final: torch.Tensor) -> dict:
    """Shadow forward of vlln + 4-layer vl_self_attention. Returns scales + backbone."""
    out: dict = {
        "vlsa_act_qkv": [], "vlsa_act_o": [],
        "vlsa_act_fc1": [], "vlsa_act_fc2": [],
    }
    # vlln: LayerNorm(2048, eps=1e-5) on llm_final
    vlln_w = fe._vlln_w.float(); vlln_b = fe._vlln_b.float()
    h = F.layer_norm(llm_final, (2048,), vlln_w, vlln_b, eps=1e-5).squeeze(0)   # (S, D)
    S, D = h.shape
    NH, HD, FF = 32, 64, 8192
    shadow = getattr(fe, "_fp16_shadow_weights", None)

    for li in range(4):
        n1w = fe._vlsa_norm1_w[li].float(); n1b = fe._vlsa_norm1_b[li].float()
        n3w = fe._vlsa_norm3_w[li].float(); n3b = fe._vlsa_norm3_b[li].float()
        q_w = _dequant_fp8(fe._vlsa_q_w[li], fe._vlsa_alpha[li * 6 + 0], shadow=shadow, key=("vlsa", li, "q"))
        k_w = _dequant_fp8(fe._vlsa_k_w[li], fe._vlsa_alpha[li * 6 + 1], shadow=shadow, key=("vlsa", li, "k"))
        v_w = _dequant_fp8(fe._vlsa_v_w[li], fe._vlsa_alpha[li * 6 + 2], shadow=shadow, key=("vlsa", li, "v"))
        o_w = _dequant_fp8(fe._vlsa_o_w[li], fe._vlsa_alpha[li * 6 + 3], shadow=shadow, key=("vlsa", li, "o"))
        fc1_w = _dequant_fp8(fe._vlsa_fc1_w[li], fe._vlsa_alpha[li * 6 + 4], shadow=shadow, key=("vlsa", li, "fc1"))
        fc2_w = _dequant_fp8(fe._vlsa_fc2_w[li], fe._vlsa_alpha[li * 6 + 5], shadow=shadow, key=("vlsa", li, "fc2"))
        q_b = fe._vlsa_q_b[li].float(); k_b = fe._vlsa_k_b[li].float()
        v_b = fe._vlsa_v_b[li].float(); o_b = fe._vlsa_o_b[li].float()
        fc1_b = fe._vlsa_fc1_b[li].float(); fc2_b = fe._vlsa_fc2_b[li].float()

        xn = F.layer_norm(h, (D,), n1w, n1b, eps=1e-5)
        out["vlsa_act_qkv"].append(_amax(xn))

        Q = (xn @ q_w + q_b).view(S, NH, HD).permute(1, 0, 2)
        K = (xn @ k_w + k_b).view(S, NH, HD).permute(1, 0, 2)
        V = (xn @ v_w + v_b).view(S, NH, HD).permute(1, 0, 2)
        scores = (Q @ K.transpose(-2, -1)) / (HD ** 0.5)
        attn_o = (scores.softmax(-1) @ V).permute(1, 0, 2).reshape(S, D)
        out["vlsa_act_o"].append(_amax(attn_o))
        o = attn_o @ o_w + o_b
        h = h + o

        xn2 = F.layer_norm(h, (D,), n3w, n3b, eps=1e-5)
        out["vlsa_act_fc1"].append(_amax(xn2))
        fc1 = F.gelu(xn2 @ fc1_w + fc1_b, approximate="tanh")
        out["vlsa_act_fc2"].append(_amax(fc1))
        h = h + (fc1 @ fc2_w + fc2_b)

    out["backbone_features"] = h.unsqueeze(0)   # (1, S, D)
    return out


AMAX_KEYS: tuple[str, ...] = (
    "vit_act_qkv", "vit_act_o", "vit_act_fc1", "vit_act_fc2",
    "deepstack_act_fc1", "deepstack_act_fc2",
    "llm_act_qkv", "llm_act_o", "llm_act_gateup", "llm_act_down",
    "vlsa_act_qkv", "vlsa_act_o", "vlsa_act_fc1", "vlsa_act_fc2",
)


def calibrate_pipeline_amax(fe, aux: dict) -> dict:
    """Run the full 4-stage shadow forward (vit → deepstack → llm → vlsa)
    on a single ``aux`` dict and return only the per-quant-point amax
    lists (no backbone / hidden-state outputs).

    Used by ``GrootN17TorchFrontendThor.calibrate(aux_list, ...)`` to
    accumulate amax across N samples before percentile-reducing. Mirrors
    the per-aux work that ``set_prompt`` already does, minus the
    backbone / DeepStack / cross-KV downstream wiring (those stay tied
    to the ``set_prompt`` aux, not to the calibration set).

    The frontend must already have ``_fp16_shadow_weights`` populated.
    The 4 ``calibrate_*`` helpers consume it transparently via
    ``_dequant_fp8(shadow=...)``.
    """
    device = fe.device

    # ── M-RoPE cos/sin (per-aux, since position_ids vary) ──────────────
    mrope_cos = aux["rope_cos"][0].to(device).float().contiguous()
    mrope_sin = aux["rope_sin"][0].to(device).float().contiguous()

    # ── ViT 2D rope cos/sin from grid_thw ──────────────────────────────
    grid_thw = [tuple(int(x) for x in row) for row in aux["grid_thw"].tolist()]
    vit_cos, vit_sin = build_vit_rope_tables(
        grid_thw, head_dim=64, theta=10000.0, spatial_merge_size=2,
        device=device,
    )
    num_views = len(grid_thw)

    visual_pos_masks = aux["visual_pos_masks"][0].to(device)
    llm_input = aux["llm_input_embeds"].to(device).float()
    pixel_features = aux["pixel_features"].to(device).float()

    out_vit = calibrate_vit(
        fe, pixel_features, vit_cos.float(), vit_sin.float(),
        num_views=num_views,
    )
    out_ds = calibrate_deepstack(fe, out_vit["deepstack_taps"])
    out_llm = calibrate_llm(
        fe, llm_input, mrope_cos, mrope_sin,
        visual_pos_masks, out_ds["features"],
    )
    out_vlsa = calibrate_vlsa(fe, out_llm["llm_final"])

    return {
        "vit_act_qkv": list(out_vit["vit_act_qkv"]),
        "vit_act_o":   list(out_vit["vit_act_o"]),
        "vit_act_fc1": list(out_vit["vit_act_fc1"]),
        "vit_act_fc2": list(out_vit["vit_act_fc2"]),
        "deepstack_act_fc1": list(out_ds["deepstack_act_fc1"]),
        "deepstack_act_fc2": list(out_ds["deepstack_act_fc2"]),
        "llm_act_qkv":    list(out_llm["llm_act_qkv"]),
        "llm_act_o":      list(out_llm["llm_act_o"]),
        "llm_act_gateup": list(out_llm["llm_act_gateup"]),
        "llm_act_down":   list(out_llm["llm_act_down"]),
        "vlsa_act_qkv": list(out_vlsa["vlsa_act_qkv"]),
        "vlsa_act_o":   list(out_vlsa["vlsa_act_o"]),
        "vlsa_act_fc1": list(out_vlsa["vlsa_act_fc1"]),
        "vlsa_act_fc2": list(out_vlsa["vlsa_act_fc2"]),
    }


def amax_to_dev_scale(
    amax: float, *, device: str = "cuda", fp8_max: float = FP8_MAX,
) -> torch.Tensor:
    """fp32 device scalar consumed by ``quantize_fp8_static_fp16``."""
    s = max(amax / fp8_max, 1e-8)
    return torch.tensor([s], dtype=torch.float32, device=device).contiguous()


def alpha(
    amax_act: float, weight_scale: float, *, fp8_max: float = FP8_MAX,
) -> float:
    """Compose host alpha for ``GemmRunner.fp8_nn_*``."""
    s_act = max(amax_act / fp8_max, 1e-8)
    return s_act * weight_scale

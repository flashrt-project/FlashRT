"""LaWAM inference forwards for RTX SM120 (pointer-only, graph-capturable).

The policy step, every stage on FlashRT kernels:

  camera frames (uint8)        lawam.preprocess_u8 -> Qwen3-VL / DINOv3 patch rows
  Qwen3-VL ViT (24 layers)     FP8 per-tensor GEMMs on an FP16 residual stream,
    + DeepStack / merger        the fused producers of groot_n17.pipeline_rtx_fp8_fused,
  Qwen3-VL LLM (16 layers)      FA2 attention (FP8 O epilogue), erf-GELU mergers
  VLM -> latent QFormer        lawam.qformer_latent (+ LAM AdaLN conditioning)
  DINOv3 ViT-B (11 layers)     FP8 GEMMs, LayerScale folded into the weights
  LAM decoder (12 AdaLN)       FP8 GEMMs, gated residual + AdaLN rows
  flow conditioning            FP8 enc_vlm, BF16 cross K/V of all 8 cross blocks
  flow DiT x num_steps         skinny NVFP4 W4A4 GEMMs + fused consumers (PDL),
                                FP32 Euler sample, MIN_MAX unnormalization

FP8 activation scales are static device scalars. ``prec="cal"`` runs the same
GEMM sequence with every FP8 producer written in FP16 and quantized by
``quantize_fp8_device*`` (dynamic per-tensor), which records amax / 448 into
the very scale slot the static path reads: one calibration pass per real
frame yields the per-site amax vector. ``prec="fp16"`` keeps FP16 operands
and weights (accuracy reference / fallback for a stage).
"""

from __future__ import annotations

EPS_VIT = 1e-6
EPS_LLM = 1e-6
EPS_DINO = 1e-5
EPS_LAM = 1e-6
EPS_LN = 1e-5


def _ck(rc, what):
    if rc:
        raise RuntimeError(f"LaWAM {what} failed (status {rc})")


try:
    from flash_rt import flash_rt_lawam as _LK
except ImportError:   # resolved by the frontend, which raises a clear build error
    _LK = None


def _quant_f16(fvk, src, out8, scale, n, st):
    fvk.quantize_fp8_device_fp16(src, out8, scale, n, st)


# Precision modes of an FP8-capable stage:
#   "fp8"  static per-tensor FP8 operands written by the fused producers;
#   "cal"  the producers write FP16, quantize_fp8_device* records amax / 448
#          into the site's scale slot and quantizes (calibration pass);
#   "fp16" FP16 operands and FP16 weights, FP32 accumulation;
#   "f16acc" FP16 operands and weights with FP16 accumulation (lawam.gemm_f16acc,
#          twice the FP32-accumulate MMA rate on GeForce Blackwell).
# In "cal" / "fp16" / "f16acc" the FP16 operand lives in the stage's ``f16`` scratch.
_F16 = ("fp16", "f16acc")

def _norm8(fvk, x, delta, bias, delta2, w, b, mode, eps, out8, scale, rows, dim, st, prec, scratch):
    """fp8_residual_norm_quant producing the operand of the next GEMM."""
    if prec == "fp8":
        _ck(fvk.fp8_residual_norm_quant(x, delta, bias, delta2, 0, w, b, mode, eps, out8, scale, 0,
                                        rows, dim, st), "norm")
        return
    _ck(fvk.fp8_residual_norm_quant(x, delta, bias, delta2, 0, w, b, mode, eps, 0, 0, scratch,
                                    rows, dim, st), "norm")
    if prec == "cal":
        _quant_f16(fvk, scratch, out8, scale, rows * dim, st)


def _gnorm8(lk, fvk, x, delta, bias, gate, delta2, mode, eps, w, b, scale_m, shift_m, out8, scale, rows, dim, st,
            prec, scratch, out_bf16=0):
    """lawam.gated_residual_norm producing the operand of the next GEMM."""
    if prec == "fp8":
        _ck(lk.gated_residual_norm(x, delta, bias, gate, delta2, mode, eps, w, b, scale_m, shift_m, out8, scale, 0,
                                   out_bf16, rows, dim, st), "gated norm")
        return
    _ck(lk.gated_residual_norm(x, delta, bias, gate, delta2, mode, eps, w, b, scale_m, shift_m, 0, 0, scratch,
                               out_bf16, rows, dim, st), "gated norm")
    if prec == "cal":
        _quant_f16(fvk, scratch, out8, scale, rows * dim, st)


def _gelu8(lk, fvk, x, ld, bias, out8, scale, rows, n, st, prec, scratch, *, tanh=False):
    gelu = lk.bias_gelu_tanh if tanh else lk.bias_gelu_erf
    if prec == "fp8":
        if tanh:
            _ck(fvk.fp8_bias_gelu_quant(x, ld, bias, out8, scale, rows, n, st), "gelu")
        else:
            _ck(gelu(x, ld, bias, out8, scale, 0, rows, n, st), "gelu")
        return
    _ck(gelu(x, ld, bias, 0, 0, scratch, rows, n, st), "gelu")
    if prec == "cal":
        _quant_f16(fvk, scratch, out8, scale, rows * n, st)


def _mm(gemm, prec, x8, x16, L, name, out, M, N, K, st):
    """out [M, N] FP16 = operand @ W (FP8 descale GEMM, or an FP16 GEMM)."""
    if prec == "fp16":
        gemm.fp16_nn(x16, L[name + "_w16"], out, M, N, K, st)
    elif prec == "f16acc":
        _ck(_LK.gemm_f16acc(x16, L[name + "_w16"], out, 0, M, N, K, st), "f16acc gemm")
    else:
        gemm.fp8_descale_fp16(x8, L[name + "_w"], out, M, N, K, L["a_" + name], L[name + "_ws"], st)


def _attn(fa2, fvk, Q, K, V, O, lse, out8, scale, *, batch, seq, heads, hd, st, prec):
    """Non-causal self-attention over [batch, seq, heads, hd] operands written
    by fp8_qkv_split (BF16, FP16 in "fp16"); O -> the o-projection operand:
    FP8 (fused FA2 epilogue / dynamic quantize) or FP16 in O itself."""
    ld = heads * hd
    strides = (seq * ld, ld, hd)
    kw = dict(Q=Q, K=K, V=V, O=O, softmax_lse=lse, softmax_lse_accum=0, o_accum=0, batch=batch, seqlen_q=seq,
              seqlen_k=seq, num_heads_q=heads, num_heads_kv=heads, head_dim=hd, q_strides=strides,
              k_strides=strides, v_strides=strides, o_strides=strides, softmax_scale=hd ** -0.5, num_sms=0,
              stream=st)
    if prec in _F16:
        fa2.fwd_fp16(**kw)
    elif prec == "cal":
        fa2.fwd_bf16(**kw)
        fvk.quantize_fp8_device(O, out8, scale, batch * seq * ld, st)
    else:
        fa2.fwd_bf16_fp8out(**kw, O_fp8=out8, o_fp8_scale=scale, o_fp8_strides=strides, skip_bf16_output=True)


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL vision tower (24 layers, DeepStack taps 5 / 11 / 17) + mergers
# ─────────────────────────────────────────────────────────────────────────

VIT_TAPS = (5, 11, 17)


def merger_forward(gemm, fvk, lk, B, M, x, rows_in, dim_in, post_norm, out, st, prec):
    """Qwen3-VL patch merger on ``x`` [rows_in, dim_in] (2x2 merge groups are
    4 consecutive rows): LayerNorm (before or after the 4-row shuffle) -> fc1
    (+bias, erf-GELU) -> fc2 -> ``out`` [rows_in / 4, 2048] without the fc2
    bias (added where the rows are scattered)."""
    n = rows_in // 4
    D4 = 4 * dim_in
    s8, fc1, f16 = B["m_fp8"], B["m_fc1"], B["m_f16"]
    rows, dim = (n, D4) if post_norm else (rows_in, dim_in)
    _norm8(fvk, x, 0, 0, 0, M["norm_w"], M["norm_b"], 0, 1e-6, s8, M["a_fc1"], rows, dim, st, prec, f16)
    _mm(gemm, prec, s8, f16, M, "fc1", fc1, n, D4, D4, st)
    _gelu8(lk, fvk, fc1, D4, M["fc1_b"], s8, M["a_fc2"], n, D4, st, prec, f16)
    _mm(gemm, prec, s8, f16, M, "fc2", out, n, 2048, D4, st)


def qwen_vit_forward(gemm, fvk, lk, fa2, B, W, dims, *, stream=0, prec="fp8"):
    """Patch embedding + 24 ViT layers + DeepStack mergers at the taps + the
    final merger. B["ds_out"][j] / B["merger_out"] receive [S/4, 2048] rows
    (fc2 bias pending)."""
    S, nv = int(dims["S"]), int(dims["views"])
    Sper, D, NH, HD, FF = S // nv, 1024, 16, 64, 4096
    st = int(stream)
    h, x8, qkv = B["h"], B["x8"], B["qkv"]
    Q, K, V, O, lse = B["Q"], B["K"], B["V"], B["O"], B["lse"]
    o_out, fc1, f16 = B["o_out"], B["fc1"], B["f16"]
    bf = prec not in _F16
    gemm.fp16_nn_bias(B["patch"], W["patch_w"], h, W["patch_b"], S, D, 768, st)
    pend = (W["pos"], 0)
    for li in range(24):
        L = W["layers"][li]
        _norm8(fvk, h, pend[0], pend[1], 0, L["n1_w"], L["n1_b"], 0, EPS_VIT, x8, L["a_qkv"], S, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "qkv", qkv, S, 3 * D, D, st)
        qb = L["qkv_b"]
        _ck(fvk.fp8_qkv_split(qkv, qkv + 2 * D, qkv + 4 * D, 3 * D, qb, qb + 2 * D, qb + 4 * D, 1.0, 1.0, 0, 0,
                              1e-6, W["cos"], W["sin"], Q, K, V, bf, S, NH, NH, HD, st), "vit qkv")
        _attn(fa2, fvk, Q, K, V, O, lse, x8, L["a_o"], batch=nv, seq=Sper, heads=NH, hd=HD, st=st, prec=prec)
        _mm(gemm, prec, x8, O, L, "o", o_out, S, D, D, st)
        _norm8(fvk, h, o_out, L["o_b"], 0, L["n2_w"], L["n2_b"], 0, EPS_VIT, x8, L["a_fc1"], S, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "fc1", fc1, S, FF, D, st)
        _gelu8(lk, fvk, fc1, FF, L["fc1_b"], x8, L["a_fc2"], S, FF, st, prec, f16, tanh=True)
        _mm(gemm, prec, x8, f16, L, "fc2", o_out, S, D, FF, st)
        pend = (o_out, L["fc2_b"])
        if li in VIT_TAPS:
            _ck(fvk.fp8_residual_norm_quant(h, pend[0], pend[1], 0, 0, 0, 0, 0, 0.0, 0, 0, 0, S, D, st), "vit res")
            pend = (0, 0)
            j = VIT_TAPS.index(li)
            merger_forward(gemm, fvk, lk, B, W["ds"][j], h, S, D, True, B["ds_out"][j], st, prec)
    _ck(fvk.fp8_residual_norm_quant(h, pend[0], pend[1], 0, 0, 0, 0, 0, 0.0, 0, 0, 0, S, D, st), "vit res")
    merger_forward(gemm, fvk, lk, B, W["merger"], h, S, D, False, B["merger_out"], st, prec)


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL language model (16 layers kept by LaWAM) + final RMSNorm
# ─────────────────────────────────────────────────────────────────────────

def qwen_llm_forward(gemm, fvk, lk, fa2, B, W, dims, *, stream=0, prec="fp8", num_sms=0, out_prec="fp8"):
    """Embedding assembly, 16 decoder layers (fused QKV, q/k RMSNorm + M-RoPE,
    causal GQA FA2, SwiGLU, DeepStack adds after layers 0..2) and the final
    RMSNorm: B["vlm"] gets the FP16 VLM states, B["vlm8"] their FP8 copy (the
    flow's enc_vlm operand)."""
    S, n_vis = int(dims["S"]), int(dims["n_vis"])
    D, NHQ, NHKV, HD, FF = 2048, 16, 8, 128, 6144
    NQKV = (NHQ + 2 * NHKV) * HD
    st = int(stream)
    h, x8, qkv = B["h"], B["x8"], B["qkv"]
    Q, K, V, O, lse = B["Q"], B["K"], B["V"], B["O"], B["lse"]
    o_out, gate, up, f16 = B["o_out"], B["gate"], B["up"], B["f16"]
    # inputs_embeds: prompt template (text + query rows), image rows from the merger
    fvk.gpu_copy(h, W["embeds"], S * D * 2, st)
    _ck(lk.scatter_rows(B["merger_out"], W["merger_b"], W["vis_idx"], h, n_vis, D, st), "embed scatter")
    for j in range(3):
        _ck(lk.scatter_rows(B["ds_out"][j], W["ds_b"][j], W["vis_idx"], B["inject"][j], n_vis, D, st),
            "deepstack scatter")
    pend, pend2 = 0, 0
    for li in range(16):
        L = W["layers"][li]
        _norm8(fvk, h, pend, 0, pend2, L["in_ln_w"], 0, 1, EPS_LLM, x8, L["a_qkv"], S, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "qkv", qkv, S, NQKV, D, st)
        _ck(fvk.fp8_qkv_split(qkv, qkv + 2 * NHQ * HD, qkv + 2 * (NHQ + NHKV) * HD, NQKV, 0, 0, 0, 1.0, 1.0,
                              L["q_norm_w"], L["k_norm_w"], 1e-6, W["cos"], W["sin"], Q, K, V, True, S,
                              NHQ, NHKV, HD, st), "llm qkv")
        fa2.fwd_bf16_causal(Q=Q, K=K, V=V, O=O, softmax_lse=lse, softmax_lse_accum=B["lse_acc"], o_accum=B["o_acc"],
             batch=1, seqlen_q=S, seqlen_k=S, num_heads_q=NHQ, num_heads_kv=NHKV, head_dim=HD,
             q_strides=(S * NHQ * HD, NHQ * HD, HD), k_strides=(S * NHKV * HD, NHKV * HD, HD),
             v_strides=(S * NHKV * HD, NHKV * HD, HD), o_strides=(S * NHQ * HD, NHQ * HD, HD),
             softmax_scale=HD ** -0.5, num_sms=num_sms, stream=st)
        if prec == "cal":
            fvk.quantize_fp8_device(O, x8, L["a_o"], S * NHQ * HD, st)
        elif prec == "fp8":
            fvk.quantize_fp8_static(O, x8, L["a_o"], S * NHQ * HD, st)
        else:
            fvk.cast_bf16_to_fp16(O, f16, S * NHQ * HD, st)
        _mm(gemm, prec, x8, f16, L, "o", o_out, S, D, NHQ * HD, st)
        _norm8(fvk, h, o_out, 0, 0, L["post_ln_w"], 0, 1, EPS_LLM, x8, L["a_gate"], S, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "gate", gate, S, FF, D, st)
        _mm(gemm, prec, x8, f16, L, "up", up, S, FF, D, st)
        if prec == "fp8":
            fvk.silu_mul_split_fp8_fp16(gate, up, x8, S * FF, L["a_down"], st)
        else:
            _ck(lk.silu_mul(gate, up, FF, f16, S, FF, st), "llm silu")
            if prec == "cal":
                _quant_f16(fvk, f16, x8, L["a_down"], S * FF, st)
        _mm(gemm, prec, x8, f16, L, "down", o_out, S, D, FF, st)
        pend = o_out
        pend2 = B["inject"][li] if li < 3 else 0
    # final RMSNorm -> FP16 VLM states (+ the FP8 enc_vlm operand when the
    # flow conditioning consumes FP8: ``out_prec``)
    if out_prec == "fp8":
        _ck(fvk.fp8_residual_norm_quant(h, pend, 0, pend2, 0, W["norm_w"], 0, 1, EPS_LLM, B["vlm8"], W["a_enc"],
                                        B["vlm"], S, D, st), "llm norm")
    else:
        _ck(fvk.fp8_residual_norm_quant(h, pend, 0, pend2, 0, W["norm_w"], 0, 1, EPS_LLM, 0, 0, B["vlm"],
                                        S, D, st), "llm norm")
        if out_prec == "cal":
            _quant_f16(fvk, B["vlm"], B["vlm8"], W["a_enc"], S * D, st)


# ─────────────────────────────────────────────────────────────────────────
# LAM vision encoder: DINOv3 ViT-B/16 + 4 registers, hidden state of layer 11
# ─────────────────────────────────────────────────────────────────────────

DINO_PREFIX = 5      # cls + 4 register tokens
DINO_LAYERS = 11     # hidden_states[-2]: the 12th layer has no consumer


def dino_forward(gemm, fvk, lk, fa2, B, W, *, stream=0, prec="fp8", out_prec="fp8"):
    """Patch embedding, 11 layers (LayerScale folded into o / fc2), then the
    latent LayerNorm (no affine) of the patch rows: the LAM decoder operand
    (FP8 B["ht8"] / FP16 B["ht16"]) and BF16 into B["img_kv_in"][:256] (flow
    image tokens)."""
    T, P, D, NH, HD, FF = 261, 256, 768, 12, 64, 3072
    st = int(stream)
    h, x8, qkv = B["h"], B["x8"], B["qkv"]
    Q, K, V, O, lse = B["Q"], B["K"], B["V"], B["O"], B["lse"]
    o_out, fc1, f16 = B["o_out"], B["fc1"], B["f16"]
    fvk.gpu_copy(h, W["prefix"], DINO_PREFIX * D * 2, st)
    gemm.fp16_nn_bias(B["patch"], W["patch_w"], h + DINO_PREFIX * D * 2, W["patch_b"], P, D, 768, st)
    pend = (0, 0)
    for li in range(DINO_LAYERS):
        L = W["layers"][li]
        _norm8(fvk, h, pend[0], pend[1], 0, L["n1_w"], L["n1_b"], 0, EPS_DINO, x8, L["a_qkv"], T, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "qkv", qkv, T, 3 * D, D, st)
        qb = L["qkv_b"]
        _ck(fvk.fp8_qkv_split(qkv, qkv + 2 * D, qkv + 4 * D, 3 * D, qb, 0, qb + 4 * D, 1.0, 1.0, 0, 0, 1e-6,
                              W["cos"], W["sin"], Q, K, V, prec not in _F16, T, NH, NH, HD, st), "dino qkv")
        _attn(fa2, fvk, Q, K, V, O, lse, x8, L["a_o"], batch=1, seq=T, heads=NH, hd=HD, st=st, prec=prec)
        _mm(gemm, prec, x8, O, L, "o", o_out, T, D, D, st)
        _norm8(fvk, h, o_out, L["o_b"], 0, L["n2_w"], L["n2_b"], 0, EPS_DINO, x8, L["a_fc1"], T, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "fc1", fc1, T, FF, D, st)
        _gelu8(lk, fvk, fc1, FF, L["fc1_b"], x8, L["a_fc2"], T, FF, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "fc2", o_out, T, D, FF, st)
        pend = (o_out, L["fc2_b"])
    off = DINO_PREFIX * D * 2
    # latent norm -> BF16 flow tokens + the LAM decoder operand in its precision (``out_prec``)
    _gnorm8(lk, fvk, h + off, pend[0] + off, pend[1], 0, 0, 0, EPS_LN, 0, 0, 0, 0, B["ht8"], W["a_pin"], P, D, st,
            out_prec, B["ht16"], out_bf16=B["img_kv_in"])


# ─────────────────────────────────────────────────────────────────────────
# LAM decoder: 12 AdaLN blocks conditioned on the predicted latent action
# ─────────────────────────────────────────────────────────────────────────

def lam_decoder_forward(gemm, fvk, lk, fa2, B, W, *, stream=0, prec="fp8"):
    """h_t1 = last_ln(project_output(blocks(project_input(h_t) + pos, c))) into
    B["img_kv_in"][256:] (BF16). The AdaLN modulation of all 12 blocks is one
    GEMV of silu(c) (B["cond_silu"], from qformer_latent)."""
    T, D, NH, HD, FF = 256, 1024, 16, 64, 4096
    st = int(stream)
    x, x8, qkv = B["x"], B["x8"], B["qkv"]
    Q, K, V, O, lse = B["Q"], B["K"], B["V"], B["O"], B["lse"]
    o_out, fc1, f16, mods = B["o_out"], B["fc1"], B["f16"], B["mods"]
    gemm.fp16_nn_bias(B["cond_silu"], W["mod_w"], mods, W["mod_b"], 1, 12 * 6 * D, D, st)
    _mm(gemm, prec, B["ht8"], B["ht16"], W, "pin", x, T, D, 768, st)
    pend = (0, 0, 0, W["pos_b"])   # delta, bias, gate, delta2
    for li in range(12):
        L = W["layers"][li]
        m = mods + li * 6 * D * 2
        shift_msa, scale_msa, gate_msa = m, m + 2 * D, m + 4 * D
        shift_mlp, scale_mlp, gate_mlp = m + 6 * D, m + 8 * D, m + 10 * D
        _gnorm8(lk, fvk, x, pend[0], pend[1], pend[2], pend[3], 0, EPS_LAM, 0, 0, scale_msa, shift_msa, x8,
                L["a_qkv"], T, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "qkv", qkv, T, 3 * D, D, st)
        qb = L["qkv_b"]
        _ck(fvk.fp8_qkv_split(qkv, qkv + 2 * D, qkv + 4 * D, 3 * D, qb, qb + 2 * D, qb + 4 * D, 1.0, 1.0, 0, 0,
                              1e-6, 0, 0, Q, K, V, prec not in _F16, T, NH, NH, HD, st), "lam qkv")
        _attn(fa2, fvk, Q, K, V, O, lse, x8, L["a_o"], batch=1, seq=T, heads=NH, hd=HD, st=st, prec=prec)
        _mm(gemm, prec, x8, O, L, "o", o_out, T, D, D, st)
        _gnorm8(lk, fvk, x, o_out, L["o_b"], gate_msa, 0, 0, EPS_LAM, 0, 0, scale_mlp, shift_mlp, x8, L["a_fc1"],
                T, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "fc1", fc1, T, FF, D, st)
        _gelu8(lk, fvk, fc1, FF, L["fc1_b"], x8, L["a_fc2"], T, FF, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "fc2", o_out, T, D, FF, st)
        pend = (o_out, L["fc2_b"], gate_mlp, 0)
    _gnorm8(lk, fvk, x, pend[0], pend[1], pend[2], 0, 2, 0.0, 0, 0, 0, 0, x8, W["a_pout"], T, D, st, prec, f16)
    _mm(gemm, prec, x8, f16, W, "pout", B["pout"], T, 768, D, st)
    _ck(lk.gated_residual_norm(B["pout"], 0, W["pout_b"], 0, 0, 0, EPS_LN, W["last_w"], W["last_b"], 0, 0, 0, 0, 0,
                               B["img_kv_in"] + 256 * 768 * 2, T, 768, st), "lam last_ln")


# ─────────────────────────────────────────────────────────────────────────
# Flow head: conditioning (enc_vlm + cross K/V) and the DiT denoising loop
# ─────────────────────────────────────────────────────────────────────────

D_DIT, FF_DIT, LAYERS_DIT, HEADS_DIT, HD_DIT = 1024, 4096, 16, 16, 64
KV_LD = 4 * 2 * D_DIT     # 4 cross blocks per token kind, K | V each


def flow_condition_forward(gemm, fvk, lk, B, W, dims, *, stream=0, prec="fp8"):
    """enc_vlm (FP8 GEMM + LayerNorm) of the VLM states, then the K / V of the
    four text and four image cross blocks (BF16 [rows, KV_LD])."""
    S = int(dims["S"])
    st = int(stream)
    _mm(gemm, prec, B["vlm8"], B["vlm"], W, "enc", B["enc_out"], S, 768, 2048, st)
    _ck(lk.gated_residual_norm(B["enc_out"], 0, W["enc_b"], 0, 0, 0, EPS_LN, W["enc_ln_w"], W["enc_ln_b"], 0, 0, 0,
                               0, 0, B["cond_vlm"], S, 768, st), "enc_vlm ln")
    gemm.bf16_nt_bias(B["cond_vlm"], W["kv_text_w"], B["kv_text"], W["kv_text_b"], S, KV_LD, 768, st)
    gemm.bf16_nt_bias(B["img_kv_in"], W["kv_img_w"], B["kv_img"], W["kv_img_b"], 512, KV_LD, 768, st)


def _skinny(fvk, a, a_sf, site, part, rows, n, k, cfg, pdl, st, what, *, bf16_act=False, one=0):
    w, sf = site["w"], site["sf"]
    c = cfg[(n, k)]
    if bf16_act:
        _ck(fvk.nvfp4_skinny_gemm_bf16_act(a, one, w, sf, part, rows, n, k, c, pdl, st), what)
    else:
        _ck(fvk.nvfp4_skinny_gemm(a, a_sf, w, sf, part, rows, n, k, c, pdl, st), what)
    return k // fvk.nvfp4_skinny_config_k_chunk(c)


def dit_step_forward(fvk, lk, fa2, B, W, T, dims, *, step, stream=0, pdl=True, cfg):
    """One Euler step of the flow: action encoder, 16 DiT blocks (odd blocks
    self-attention over the action rows, even blocks cross-attention to the
    text (li % 4 == 0) or image tokens), output AdaLN, the folded
    proj_out_2 / decoder layer 1 (+ReLU), decoder layer 2 and the FP32 update
    of B["x"]. T holds the per-step AdaLN tables, W the step-invariant NVFP4
    sites (w, sf, gs, bias)."""
    rows, S_text = int(dims["rows"]), int(dims["S"])
    D, FF = D_DIT, FF_DIT
    st = int(stream)
    h, xn, xn_sf = B["h"], B["xn"], B["xn_sf"]
    hid, hid_sf, part, one = B["hid"], B["hid_sf"], B["part"], B["one"]
    O, hmod, ae = B["O"], B["hmod"], B["ae"]
    scale, shift = T["scale"][step], T["shift"][step]
    pos = T["pos"]
    _ck(lk.action_encode(B["x"], W["ae_w1"], W["ae_b1"], ae, rows, 32, D, st), "action encode")
    site = W["ae2"]
    sp = _skinny(fvk, ae, 0, site, part, rows, D, D, cfg, pdl, st, "action enc w2", bf16_act=True, one=one)
    _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, False, scale[0], shift[0], pos, xn,
                                      xn_sf, one, 0, rows, D, EPS_LN, pdl, st), "dit in-norm")
    for li in range(LAYERS_DIT):
        if li % 2 == 1:
            site = W["qkv"][li // 2]
            sp = _skinny(fvk, xn, xn_sf, site, part, rows, 3 * D, D, cfg, pdl, st, "self qkv")
            _ck(fvk.nvfp4_skinny_partials_attention(part, sp, one, site["gs"], site["b"], 3 * D, 0, 0, 0, rows,
                                                    False, O, rows, HEADS_DIT, HD_DIT, HD_DIT ** -0.5, pdl, st),
                "self attention")
        else:
            site = W["q"][li // 2]
            sp = _skinny(fvk, xn, xn_sf, site, part, rows, D, D, cfg, pdl, st, "cross q")
            g = li // 4
            if li % 4 == 0:
                kv = B["kv_text"] + g * 2 * D * 2
                _ck(fvk.nvfp4_skinny_partials_attention(part, sp, one, site["gs"], site["b"], D, kv, kv + D * 2,
                                                        KV_LD, S_text, True, O, rows, HEADS_DIT, HD_DIT,
                                                        HD_DIT ** -0.5, pdl, st), "cross attention")
            else:
                # 512 image keys exceed the partials-attention shared-memory
                # budget: BF16 Q, then the small-query FA2 tile kernel
                kv = B["kv_img"] + g * 2 * D * 2
                _ck(lk.dit_sum_bias_bf16(part, sp, one, site["gs"], site["b"], B["q_bf16"], rows, D, pdl, st),
                    "cross q store")
                fa2.fwd_bf16_tile(Q=B["q_bf16"], K=kv, V=kv + D * 2, O=O, softmax_lse=B["lse"], seqused_k=0,
                                  batch=1, seqlen_q=rows, seqlen_k=512, num_heads_q=HEADS_DIT,
                                  num_heads_kv=HEADS_DIT, head_dim=HD_DIT, q_strides=(rows * D, D, HD_DIT),
                                  k_strides=(512 * KV_LD, KV_LD, HD_DIT), v_strides=(512 * KV_LD, KV_LD, HD_DIT),
                                  o_strides=(rows * D, D, HD_DIT), softmax_scale=HD_DIT ** -0.5, num_sms=0,
                                  stream=st)
        site = W["o"][li]
        sp = _skinny(fvk, O, 0, site, part, rows, D, D, cfg, pdl, st, "o proj", bf16_act=True, one=one)
        _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, True, 0, 0, 0, xn, xn_sf, one, 0,
                                          rows, D, EPS_LN, pdl, st), "o residual norm")
        site = W["ff_up"][li]
        sp = _skinny(fvk, xn, xn_sf, site, part, rows, FF, D, cfg, pdl, st, "ffn up")
        _ck(fvk.nvfp4_skinny_sum_bias_gelu(part, sp, one, site["gs"], site["b"], hid, hid_sf, one, rows, FF, pdl,
                                           st), "ffn gelu")
        site = W["ff_down"][li]
        sp = _skinny(fvk, hid, hid_sf, site, part, rows, D, FF, cfg, pdl, st, "ffn down")
        if li < LAYERS_DIT - 1:
            _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, True, scale[li + 1],
                                              shift[li + 1], pos, xn, xn_sf, one, 0, rows, D, EPS_LN, pdl, st),
                "ffn residual adaln")
        else:
            _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, True, T["out_scale"][step],
                                              T["out_shift"][step], 0, 0, 0, 0, hmod, rows, D, 1e-6, pdl, st),
                "ffn residual out-norm")
    site = W["head1"]
    sp = _skinny(fvk, hmod, 0, site, part, rows, D, D, cfg, pdl, st, "head fold", bf16_act=True, one=one)
    _ck(lk.dit_sum_bias_act(part, sp, one, site["gs"], site["b"], 1, hid, hid_sf, one, rows, D, pdl, st), "head relu")
    site = W["head2"]
    sp = _skinny(fvk, hid, hid_sf, site, part, rows, 32, D, cfg, pdl, st, "head out")
    _ck(lk.dit_sum_bias_euler(part, sp, one, site["gs"], site["b32"], B["x"], float(dims["dt"]), rows, 32, pdl, st),
        "euler")

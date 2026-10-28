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

import numpy as np

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

# "f16acc" GEMM kernel: "cutlass" (lawam.gemm_f16acc_act: CUTLASS tiles, bias +
# GELU / SwiGLU fused into the epilogue) or "cublas" (lawam.gemm_f16acc).
# F16ACC_SPLIT: tile / serial split-K of the CUTLASS GEMMs: "tuned" (the
# F16ACC_PLANS table measured on RTX 5060 Ti for the production shapes, so every
# process runs the same plans and FP16 sums; other shapes are timed on their
# first call), "timed" (time every shape) or "cublas" (the cuBLASLt plan's
# split: the same FP16 sums as lawam.gemm_f16acc).
F16ACC_KERNEL = "cutlass"
F16ACC_SPLIT = "tuned"
_F16ACC_SPLIT = {}

# (ceil(M / 64), N, K, act or "swiglu" / "gateup" / "pswiglu", bias) -> (tile, split); tiles
# 8..10 are CUTLASS stream-K (split ignored), "gateup" runs the LLM gate | up as one GEMM +
# silu_mul; tiles 16.. are lawam.gemm_f16acc_persistent configurations (tile - 16, split =
# schedule), "pswiglu" its SwiGLU epilogue over the 8-column interleaved gate | up weight.
F16ACC_PLANS = {
    (2, 2048, 4096, 0, False): (1, 1),
    (4, 768, 768, 0, True): (1, 1),
    (4, 768, 1024, 0, False): (3, 1),
    (4, 768, 2048, 0, False): (3, 1),
    (4, 1024, 768, 0, False): (3, 1),
    (4, 1024, 1024, 0, False): (1, 1),
    (4, 1024, 4096, 0, False): (3, 1),
    (4, 2048, 2048, 0, False): (0, 1),
    (4, 2048, 6144, 0, False): (8, 1),
    (4, 3072, 1024, 0, False): (3, 1),
    (4, 3072, 1024, 0, True): (3, 1),
    (4, 4096, 2048, 0, False): (5, 1),
    (5, 768, 768, 0, False): (1, 1),
    (5, 768, 3072, 0, False): (1, 1),
    (5, 2304, 768, 0, False): (1, 1),
    (8, 1024, 768, 0, True): (0, 1),
    (8, 1024, 1024, 0, False): (2, 1),
    (8, 1024, 4096, 0, False): (2, 1),
    (8, 3072, 1024, 0, False): (2, 1),
    (8, 3072, 1024, 0, True): (2, 1),
    (8, 4096, 1024, 1, True): (5, 1),
    (2, 4096, 4096, 2, True): (3, 1),
    (4, 4096, 1024, 2, True): (6, 1),
    (5, 3072, 768, 2, True): (7, 1),
    (4, 6144, 2048, 'swiglu', False): (0, 2),
    (4, 6144, 2048, 'gateup', False): (8, 1),
    (4, 6144, 2048, 'pswiglu', False): (16, 0),
}


def _plan_for(M, N, K, act, bias):
    if F16ACC_SPLIT != "tuned":
        return None
    return F16ACC_PLANS.get(((M + 63) // 64, N, K, act, bool(bias)))


def _split_for(key):
    if F16ACC_SPLIT in ("tuned", "timed"):
        return 0
    split = _F16ACC_SPLIT.get(key)
    if split is None:
        split = _F16ACC_SPLIT[key] = max(1, _LK.gemm_f16acc_split(*key))
    return split


def _f16acc(a, w, out, bias, M, N, K, act, st):
    """out [M, N] = act(a @ w + bias), FP16 accumulation; act 0 identity, 1
    tanh-GELU, 2 erf-GELU (the reference GEMM has no bias for act 2: its bias +
    GELU ran as a separate pass on the FP16 GEMM output)."""
    if F16ACC_KERNEL == "cublas":
        _ck(_LK.gemm_f16acc(a, w, out, bias if act < 2 else 0, M, N, K, act == 1, st), "f16acc gemm")
        if act == 2:
            _ck(_LK.bias_gelu_erf(out, N, bias, 0, 0, out, M, N, st), "gelu")
        return
    plan = _plan_for(M, N, K, act, bias)
    if plan is not None and plan[0] >= 16:
        _ck(_LK.gemm_f16acc_persistent(a, w, out, bias, M, N, K, N, act, plan[0] - 16, plan[1], st), "f16acc gemm")
        return
    if plan is not None:
        _ck(_LK.gemm_f16acc_act_cfg(a, w, out, bias, M, N, K, act, plan[0], plan[1], st), "f16acc gemm")
        return
    split = _split_for((M, N, K, bool(bias) and act < 2, act == 1))
    _ck(_LK.gemm_f16acc_act(a, w, out, bias, M, N, K, act, split, st), "f16acc gemm")

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


def swiglu_interleaved(FF, K):
    """Whether the f16acc SwiGLU of [K, 2 FF] runs the persistent kernel's
    epilogue, which reads the gate | up weight interleaved in blocks of 8
    columns (interleave_gate_up)."""
    return (F16ACC_KERNEL == "cutlass" and F16ACC_SPLIT == "tuned" and
            any(k[1:4] == (FF, K, "pswiglu") for k in F16ACC_PLANS))


def interleave_gate_up(w):
    """[K, 2 FF] gate | up weight -> [K, 2 FF] with 8-column blocks alternating gate, up."""
    K, F2 = w.shape
    return w.view(K, 2, F2 // 16, 8).transpose(1, 2).reshape(K, F2).contiguous()


def _f16acc_swiglu(a, w, gu, out, M, FF, K, st, interleaved=False):
    """out [M, FF] = silu(a @ w[:, :FF]) * (a @ w[:, FF:]) (w [K, 2 FF], gate |
    up sums in gu [M, 2 FF]): the "cutlass" f16acc kernel runs it as one dual
    GEMM, "cublas" as the gate|up GEMM + silu_mul. ``out`` must not alias ``a``.
    ``interleaved``: w is the interleave_gate_up weight and the persistent
    GEMM writes out directly (gu unused)."""
    if interleaved:
        plan = _plan_for(M, FF, K, "pswiglu", False) or (16, 0)
        _ck(_LK.gemm_f16acc_persistent(a, w, out, 0, M, 2 * FF, K, FF, 3, plan[0] - 16, plan[1], st), "llm swiglu")
        return
    if F16ACC_KERNEL == "cublas":
        _f16acc(a, w, gu, 0, M, 2 * FF, K, 0, st)
        _ck(_LK.silu_mul(gu, gu + FF * 2, 2 * FF, out, M, FF, st), "llm silu")
        return
    plan = _plan_for(M, FF, K, "gateup", False)
    if plan is not None:
        # gate | up as one (stream-K) GEMM over the concatenated weight, then silu_mul
        _ck(_LK.gemm_f16acc_act_cfg(a, w, gu, 0, M, 2 * FF, K, 0, plan[0], plan[1], st), "llm gate|up")
        _ck(_LK.silu_mul(gu, gu + FF * 2, 2 * FF, out, M, FF, st), "llm silu")
        return
    plan = _plan_for(M, FF, K, "swiglu", False)
    if plan is not None:
        _ck(_LK.gemm_f16acc_swiglu_cfg(a, w, w + FF * 2, 2 * FF, gu, gu + FF * 2, 2 * FF, out, M, FF, K, plan[0],
                                       plan[1], st), "llm swiglu")
        return
    split = _split_for((M, 2 * FF, K, False, False))
    _ck(_LK.gemm_f16acc_swiglu(a, w, w + FF * 2, 2 * FF, gu, gu + FF * 2, 2 * FF, out, M, FF, K, split, st),
        "llm swiglu")


def _fc1_gelu(gemm, lk, fvk, prec, x8, x16, L, fc1, scale, M, N, K, st, scratch):
    """fc1 (+bias, erf-GELU); returns the FP16 operand of fc2 (fp8 / cal: also
    written to x8). f16acc runs the bias and GELU in the GEMM epilogue."""
    if prec == "f16acc":
        _f16acc(x16, L["fc1_w16"], fc1, L["fc1_b"], M, N, K, 2, st)
        return fc1
    _mm(gemm, prec, x8, x16, L, "fc1", fc1, M, N, K, st)
    _gelu8(lk, fvk, fc1, N, L["fc1_b"], x8, scale, M, N, st, prec, scratch)
    return scratch


def _mm(gemm, prec, x8, x16, L, name, out, M, N, K, st):
    """out [M, N] FP16 = operand @ W (FP8 descale GEMM, or an FP16 GEMM)."""
    if prec == "fp16":
        gemm.fp16_nn(x16, L[name + "_w16"], out, M, N, K, st)
    elif prec == "f16acc":
        _f16acc(x16, L[name + "_w16"], out, 0, M, N, K, 0, st)
    else:
        gemm.fp8_descale_fp16(x8, L[name + "_w"], out, M, N, K, L["a_" + name], L[name + "_ws"], st)


def _attn(fa2, fvk, Q, K, V, O, lse, out8, scale, *, batch, seq, heads, hd, st, prec, qkv_ld=0):
    """Non-causal self-attention over [batch, seq, heads, hd] operands written
    by fp8_qkv_split (BF16, FP16 in "fp16"), or read in place from a fused
    [seq, qkv_ld] QKV GEMM output (``qkv_ld``); O -> the o-projection operand:
    FP8 (fused FA2 epilogue / dynamic quantize) or FP16 in O itself."""
    ld = heads * hd
    strides = (seq * ld, ld, hd)
    in_strides = (seq * qkv_ld, qkv_ld, hd) if qkv_ld else strides
    kw = dict(Q=Q, K=K, V=V, O=O, softmax_lse=lse, softmax_lse_accum=0, o_accum=0, batch=batch, seqlen_q=seq,
              seqlen_k=seq, num_heads_q=heads, num_heads_kv=heads, head_dim=hd, q_strides=in_strides,
              k_strides=in_strides, v_strides=in_strides, o_strides=strides, softmax_scale=hd ** -0.5, num_sms=0,
              stream=st)
    if prec in _F16:
        fa2.lawam_fwd_fp16(**kw)
    elif prec == "cal":
        fa2.lawam_fwd_bf16(**kw)
        fvk.quantize_fp8_device(O, out8, scale, batch * seq * ld, st)
    else:
        fa2.lawam_fwd_bf16_fp8out(**kw, O_fp8=out8, o_fp8_scale=scale, o_fp8_strides=strides, skip_bf16_output=True)


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL vision tower (24 layers, DeepStack taps 5 / 11 / 17) + mergers
# ─────────────────────────────────────────────────────────────────────────

VIT_TAPS = (5, 11, 17)
VIT_ROPE_INPLACE = True


def merger_forward(gemm, fvk, lk, B, M, x, rows_in, dim_in, post_norm, out, st, prec, *, side=None):
    """Qwen3-VL patch merger on ``x`` [rows_in, dim_in] (2x2 merge groups are
    4 consecutive rows): LayerNorm (before or after the 4-row shuffle) -> fc1
    (+bias, erf-GELU) -> fc2 -> ``out`` [rows_in / 4, 2048] without the fc2
    bias (added where the rows are scattered). ``side`` = (stream, fork, bufs)
    (FP16 precisions): the LayerNorm runs on ``st`` into the private
    bufs["in16"], ``fork()`` orders the side stream after it and the GEMMs
    (DRAM bound) run there on bufs["fc1"]."""
    n = rows_in // 4
    D4 = 4 * dim_in
    s8, fc1, f16 = B["m_fp8"], B["m_fc1"], B["m_f16"]
    rows, dim = (n, D4) if post_norm else (rows_in, dim_in)
    if side is not None:
        assert prec in _F16
        side_st, fork, bufs = side
        _norm8(fvk, x, 0, 0, 0, M["norm_w"], M["norm_b"], 0, 1e-6, 0, 0, rows, dim, st, prec, bufs["in16"])
        fork()
        h16 = _fc1_gelu(gemm, lk, fvk, prec, 0, bufs["in16"], M, bufs["fc1"], 0, n, D4, D4, side_st, 0)
        _mm(gemm, prec, 0, h16, M, "fc2", out, n, 2048, D4, side_st)
        return
    _norm8(fvk, x, 0, 0, 0, M["norm_w"], M["norm_b"], 0, 1e-6, s8, M["a_fc1"], rows, dim, st, prec, f16)
    h16 = _fc1_gelu(gemm, lk, fvk, prec, s8, f16, M, fc1, M["a_fc2"], n, D4, D4, st, f16)
    _mm(gemm, prec, s8, h16, M, "fc2", out, n, 2048, D4, st)


def _site_prec(prec, fp8_sites, cal):
    """Per-GEMM-site precision: sites in ``fp8_sites`` run static FP8 ("cal"
    while calibrating) inside a 16-bit stage."""
    def sp(name, layer=None):
        if name in fp8_sites or (layer is not None and f"{name}@{layer}" in fp8_sites):
            return "cal" if cal else "fp8"
        return prec
    return sp


def qwen_vit_forward(gemm, fvk, lk, fa2, B, W, dims, *, stream=0, prec="fp8", fp8_sites=(), cal=False,
                     ds_side=None):
    """Patch embedding + 24 ViT layers + DeepStack mergers at the taps + the
    final merger. B["ds_out"][j] / B["merger_out"] receive [S/4, 2048] rows
    (fc2 bias pending). ``fp8_sites`` (qkv / o / fc1 / fc2) run FP8 in a
    16-bit tower. ``ds_side`` = (stream, fork(j), join()): the GEMMs of the
    DeepStack mergers run on that stream next to the following layers (their
    outputs are first read by the LLM); join() precedes the final merger."""
    S, nv = int(dims["S"]), int(dims["views"])
    Sper, D, NH, HD, FF = S // nv, 1024, 16, 64, 4096
    st = int(stream)
    h, x8, qkv = B["h"], B["x8"], B["qkv"]
    Q, K, V, O, lse = B["Q"], B["K"], B["V"], B["O"], B["lse"]
    o_out, fc1, f16 = B["o_out"], B["fc1"], B["f16"]
    sp = _site_prec(prec, fp8_sites, cal)
    p_qkv, p_o, p_fc1, p_fc2 = sp("qkv"), sp("o"), sp("fc1"), sp("fc2")
    if prec == "f16acc":   # FP16 accumulation like the tower GEMMs (cuBLASLt accumulates in FP32 at half rate)
        _f16acc(B["patch"], W["patch_w"], h, W["patch_b"], S, D, 768, 0, st)
    else:
        gemm.fp16_nn_bias(B["patch"], W["patch_w"], h, W["patch_b"], S, D, 768, st)
    pend = (W["pos"], 0)
    for li in range(24):
        L = W["layers"][li]
        p_qkv, p_o, p_fc1, p_fc2 = sp("qkv", li), sp("o", li), sp("fc1", li), sp("fc2", li)
        _norm8(fvk, h, pend[0], pend[1], 0, L["n1_w"], L["n1_b"], 0, EPS_VIT, x8, L["a_qkv"], S, D, st, p_qkv, f16)
        if p_qkv == "f16acc" and p_o == "f16acc" and VIT_ROPE_INPLACE:
            # bias in the GEMM epilogue, RoPE in place on Q | K, attention reads Q / K / V through strides
            _f16acc(f16, L["qkv_w16"], qkv, L["qkv_b"], S, 3 * D, D, 0, st)
            _ck(lk.qk_rope_inplace(qkv, 3 * D, W["cos"], W["sin"], S, 2 * NH, HD, st), "vit rope")
            _attn(fa2, fvk, qkv, qkv + 2 * D, qkv + 4 * D, O, lse, x8, L["a_o"], batch=nv, seq=Sper, heads=NH, hd=HD,
                  st=st, prec=p_o, qkv_ld=3 * D)
        else:
            _mm(gemm, p_qkv, x8, f16, L, "qkv", qkv, S, 3 * D, D, st)
            qb = L["qkv_b"]
            _ck(lk.qkv_rope_split(qkv, 3 * D, qb, qb + 2 * D, qb + 4 * D, 1.0, 1.0, W["cos"], W["sin"], Q, K,
                                  V, p_o not in _F16, S, NH, NH, HD, st), "vit qkv")
            _attn(fa2, fvk, Q, K, V, O, lse, x8, L["a_o"], batch=nv, seq=Sper, heads=NH, hd=HD, st=st, prec=p_o)
        _mm(gemm, p_o, x8, O, L, "o", o_out, S, D, D, st)
        _norm8(fvk, h, o_out, L["o_b"], 0, L["n2_w"], L["n2_b"], 0, EPS_VIT, x8, L["a_fc1"], S, D, st, p_fc1, f16)
        if p_fc1 == "f16acc" and p_fc2 == "f16acc":
            # fc1 bias + tanh-GELU in the GEMM epilogue; fc2 reads it directly
            _f16acc(f16, L["fc1_w16"], fc1, L["fc1_b"], S, FF, D, 1, st)
            _mm(gemm, p_fc2, x8, fc1, L, "fc2", o_out, S, D, FF, st)
        else:
            _mm(gemm, p_fc1, x8, f16, L, "fc1", fc1, S, FF, D, st)
            _gelu8(lk, fvk, fc1, FF, L["fc1_b"], x8, L["a_fc2"], S, FF, st, p_fc2, f16, tanh=True)
            _mm(gemm, p_fc2, x8, f16, L, "fc2", o_out, S, D, FF, st)
        pend = (o_out, L["fc2_b"])
        if li in VIT_TAPS:
            _ck(fvk.fp8_residual_norm_quant(h, pend[0], pend[1], 0, 0, 0, 0, 0, 0.0, 0, 0, 0, S, D, st), "vit res")
            pend = (0, 0)
            j = VIT_TAPS.index(li)
            side = None
            if ds_side is not None:
                side = (ds_side[0], lambda: ds_side[1](j), {"in16": B["ds_in16"][j], "fc1": B["ds_fc1"]})
            merger_forward(gemm, fvk, lk, B, W["ds"][j], h, S, D, True, B["ds_out"][j], st, prec, side=side)
    _ck(fvk.fp8_residual_norm_quant(h, pend[0], pend[1], 0, 0, 0, 0, 0, 0.0, 0, 0, 0, S, D, st), "vit res")
    if ds_side is not None:
        ds_side[2]()
    merger_forward(gemm, fvk, lk, B, W["merger"], h, S, D, False, B["merger_out"], st, prec)


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL language model (16 layers kept by LaWAM) + final RMSNorm
# ─────────────────────────────────────────────────────────────────────────

def qwen_llm_forward(gemm, fvk, lk, fa2, B, W, dims, *, stream=0, prec="fp8", num_sms=0, out_prec="fp8",
                     fp8_sites=(), cal=False, attn_dtype="bf16", embeds_ready=False):
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
    # inputs_embeds: prompt template (text + query rows; copied by the caller when ``embeds_ready``),
    # image rows from the merger
    if not embeds_ready:
        fvk.gpu_copy(h, W["embeds"], S * D * 2, st)
    _ck(lk.scatter_rows(B["merger_out"], W["merger_b"], W["vis_idx"], h, n_vis, D, st), "embed scatter")
    for j in range(3):
        _ck(lk.scatter_rows(B["ds_out"][j], W["ds_b"][j], W["vis_idx"], B["inject"][j], n_vis, D, st),
            "deepstack scatter")
    sp = _site_prec(prec, fp8_sites, cal)
    p_qkv, p_o, p_gu, p_dn = sp("qkv"), sp("o"), sp("gateup"), sp("down")
    # BF16 attention operands (the reference rounding) with the output cast to
    # FP16 for a 16-bit o projection, or FP16 operands end to end ("fp16")
    pend, pend2 = 0, 0
    for li in range(16):
        L = W["layers"][li]
        p_qkv, p_o, p_gu, p_dn = sp("qkv", li), sp("o", li), sp("gateup", li), sp("down", li)
        f16_attn = p_o in _F16 and attn_dtype == "fp16"
        attn = fa2.lawam_fwd_fp16_causal if f16_attn else fa2.lawam_fwd_bf16_causal
        _norm8(fvk, h, pend, 0, pend2, L["in_ln_w"], 0, 1, EPS_LLM, x8, L["a_qkv"], S, D, st, p_qkv, f16)
        _mm(gemm, p_qkv, x8, f16, L, "qkv", qkv, S, NQKV, D, st)
        _ck(fvk.fp8_qkv_split(qkv, qkv + 2 * NHQ * HD, qkv + 2 * (NHQ + NHKV) * HD, NQKV, 0, 0, 0, 1.0, 1.0,
                              L["q_norm_w"], L["k_norm_w"], 1e-6, W["cos"], W["sin"], Q, K, V, not f16_attn, S,
                              NHQ, NHKV, HD, st), "llm qkv")
        attn(Q=Q, K=K, V=V, O=O, softmax_lse=lse, softmax_lse_accum=B["lse_acc"], o_accum=B["o_acc"],
             batch=1, seqlen_q=S, seqlen_k=S, num_heads_q=NHQ, num_heads_kv=NHKV, head_dim=HD,
             q_strides=(S * NHQ * HD, NHQ * HD, HD), k_strides=(S * NHKV * HD, NHKV * HD, HD),
             v_strides=(S * NHKV * HD, NHKV * HD, HD), o_strides=(S * NHQ * HD, NHQ * HD, HD),
             softmax_scale=HD ** -0.5, num_sms=num_sms, stream=st)
        if p_o == "cal":
            fvk.quantize_fp8_device(O, x8, L["a_o"], S * NHQ * HD, st)
        elif p_o == "fp8":
            fvk.quantize_fp8_static(O, x8, L["a_o"], S * NHQ * HD, st)
        elif not f16_attn:
            fvk.cast_bf16_to_fp16(O, f16, S * NHQ * HD, st)
        _mm(gemm, p_o, x8, O if f16_attn else f16, L, "o", o_out, S, D, NHQ * HD, st)
        _norm8(fvk, h, o_out, 0, 0, L["post_ln_w"], 0, 1, EPS_LLM, x8, L["a_gate"], S, D, st, p_gu, f16)
        if p_gu == "f16acc" and p_dn in _F16:
            # SwiGLU in the gate|up GEMM; down reads its output
            _f16acc_swiglu(f16, L["gateup_w16"], gate, up, S, FF, D, st, interleaved=L.get("gateup_il", False))
            _mm(gemm, p_dn, x8, up, L, "down", o_out, S, D, FF, st)
            pend = o_out
            pend2 = B["inject"][li] if li < 3 else 0
            continue
        if p_gu in _F16:
            # gate | up as one [K, 2 FF] GEMM, SwiGLU reads both halves of the row
            _mm(gemm, p_gu, x8, f16, L, "gateup", gate, S, 2 * FF, D, st)
            g_ptr, u_ptr, g_ld = gate, gate + FF * 2, 2 * FF
        else:
            _mm(gemm, p_gu, x8, f16, L, "gate", gate, S, FF, D, st)
            _mm(gemm, p_gu, x8, f16, L, "up", up, S, FF, D, st)
            g_ptr, u_ptr, g_ld = gate, up, FF
        if p_dn == "fp8" and g_ld == FF:
            fvk.silu_mul_split_fp8_fp16(gate, up, x8, S * FF, L["a_down"], st)
        else:
            _ck(lk.silu_mul(g_ptr, u_ptr, g_ld, f16, S, FF, st), "llm silu")
            if p_dn == "cal":
                _quant_f16(fvk, f16, x8, L["a_down"], S * FF, st)
            elif p_dn == "fp8":
                fvk.quantize_fp8_static_fp16(f16, x8, L["a_down"], S * FF, st)
        _mm(gemm, p_dn, x8, f16, L, "down", o_out, S, D, FF, st)
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
    if prec == "f16acc":
        _f16acc(B["patch"], W["patch_w"], h + DINO_PREFIX * D * 2, W["patch_b"], P, D, 768, 0, st)
    else:
        gemm.fp16_nn_bias(B["patch"], W["patch_w"], h + DINO_PREFIX * D * 2, W["patch_b"], P, D, 768, st)
    pend = (0, 0)
    for li in range(DINO_LAYERS):
        L = W["layers"][li]
        _norm8(fvk, h, pend[0], pend[1], 0, L["n1_w"], L["n1_b"], 0, EPS_DINO, x8, L["a_qkv"], T, D, st, prec, f16)
        _mm(gemm, prec, x8, f16, L, "qkv", qkv, T, 3 * D, D, st)
        qb = L["qkv_b"]
        _ck(lk.qkv_rope_split(qkv, 3 * D, qb, 0, qb + 4 * D, 1.0, 1.0, W["cos"], W["sin"], Q, K,
                              V, prec not in _F16, T, NH, NH, HD, st), "dino qkv")
        _attn(fa2, fvk, Q, K, V, O, lse, x8, L["a_o"], batch=1, seq=T, heads=NH, hd=HD, st=st, prec=prec)
        _mm(gemm, prec, x8, O, L, "o", o_out, T, D, D, st)
        _norm8(fvk, h, o_out, L["o_b"], 0, L["n2_w"], L["n2_b"], 0, EPS_DINO, x8, L["a_fc1"], T, D, st, prec, f16)
        h16 = _fc1_gelu(gemm, lk, fvk, prec, x8, f16, L, fc1, L["a_fc2"], T, FF, D, st, f16)
        _mm(gemm, prec, x8, h16, L, "fc2", o_out, T, D, FF, st)
        pend = (o_out, L["fc2_b"])
    off = DINO_PREFIX * D * 2
    # latent norm -> BF16 flow tokens + the LAM decoder operand in its precision (``out_prec``)
    _gnorm8(lk, fvk, h + off, pend[0] + off, pend[1], 0, 0, 0, EPS_LN, 0, 0, 0, 0, B["ht8"], W["a_pin"], P, D, st,
            out_prec, B["ht16"], out_bf16=B["img_kv_in"])


# ─────────────────────────────────────────────────────────────────────────
# LAM decoder: 12 AdaLN blocks conditioned on the predicted latent action
# ─────────────────────────────────────────────────────────────────────────

LAM_BLOCKS, LAM_MOD_ROWS = 12, 6 * 1024   # AdaLN modulation rows per block (shift / scale / gate x 2)


def lam_modulation(gemm, lk, B, W, b0, b1, *, stream=0, max_ctas=0):
    """AdaLN modulation rows of LAM blocks [b0, b1): GEMV of silu(c)
    (B["cond_silu"], from qformer_latent). Every row is computed the same way
    whatever the range, so the blocks can be split over launches; the FP16
    weight path runs all blocks at b0 == 0. ``max_ctas`` caps the NVFP4 GEMV's
    grid (a background launch next to the LAM blocks)."""
    st, D, r0, n = int(stream), 1024, b0 * LAM_MOD_ROWS, (b1 - b0) * LAM_MOD_ROWS
    mods = B["mods"] + r0 * 2
    if "mod_w4" in W:
        _ck(lk.gemv_nvfp4w(B["cond_silu"], W["mod_w4"] + r0 * D // 2, W["mod_sf"] + r0 * D // 16, W["mod_gs"],
                           W["mod_b"] + r0 * 2, mods, n, D, st, max_ctas), "lam adaln gemv")
    elif "mod_w8" in W:
        _ck(lk.gemv_i8w(B["cond_silu"], W["mod_w8"] + r0 * D, W["mod_ws"] + r0 * 4, W["mod_b"] + r0 * 2, mods, n, D,
                        st), "lam adaln gemv")
    elif b0 == 0:
        gemm.fp16_nn_bias(B["cond_silu"], W["mod_w"], B["mods"], W["mod_b"], 1, LAM_BLOCKS * LAM_MOD_ROWS, D, st)


def lam_input_forward(gemm, B, W, *, stream=0, prec="fp8"):
    """project_input of the DINOv3 latents (B["x"]; the positional table and
    its bias are added by the first block's norm). Frame-only: it can run on
    the DINOv3 branch."""
    _mm(gemm, prec, B["ht8"], B["ht16"], W, "pin", B["x"], 256, 1024, 768, int(stream))


def lam_decoder_forward(gemm, fvk, lk, fa2, B, W, *, stream=0, prec="fp8", inputs=True, before_block=None):
    """h_t1 = last_ln(project_output(blocks(project_input(h_t) + pos, c))) into
    B["img_kv_in"][256:] (BF16). ``inputs=False``: the caller ran
    lam_modulation and lam_input_forward (``before_block(li)`` orders block li
    after its modulation rows)."""
    T, D, NH, HD, FF = 256, 1024, 16, 64, 4096
    st = int(stream)
    x, x8, qkv = B["x"], B["x8"], B["qkv"]
    Q, K, V, O, lse = B["Q"], B["K"], B["V"], B["O"], B["lse"]
    o_out, fc1, f16, mods = B["o_out"], B["fc1"], B["f16"], B["mods"]
    if inputs:
        lam_modulation(gemm, lk, B, W, 0, LAM_BLOCKS, stream=st)
        lam_input_forward(gemm, B, W, stream=st, prec=prec)
    pend = (0, 0, 0, W["pos_b"])   # delta, bias, gate, delta2
    for li in range(LAM_BLOCKS):
        if before_block is not None:
            before_block(li)
        L = W["layers"][li]
        m = mods + li * 6 * D * 2
        shift_msa, scale_msa, gate_msa = m, m + 2 * D, m + 4 * D
        shift_mlp, scale_mlp, gate_mlp = m + 6 * D, m + 8 * D, m + 10 * D
        _gnorm8(lk, fvk, x, pend[0], pend[1], pend[2], pend[3], 0, EPS_LAM, 0, 0, scale_msa, shift_msa, x8,
                L["a_qkv"], T, D, st, prec, f16)
        if prec == "f16acc":
            # bias in the GEMM epilogue (the same FP16 rounding as the split); attention reads Q / K / V in place
            _f16acc(f16, L["qkv_w16"], qkv, L["qkv_b"], T, 3 * D, D, 0, st)
            _attn(fa2, fvk, qkv, qkv + 2 * D, qkv + 4 * D, O, lse, x8, L["a_o"], batch=1, seq=T, heads=NH, hd=HD,
                  st=st, prec=prec, qkv_ld=3 * D)
        else:
            _mm(gemm, prec, x8, f16, L, "qkv", qkv, T, 3 * D, D, st)
            qb = L["qkv_b"]
            _ck(lk.qkv_rope_split(qkv, 3 * D, qb, qb + 2 * D, qb + 4 * D, 1.0, 1.0, 0, 0, Q, K,
                                  V, prec not in _F16, T, NH, NH, HD, st), "lam qkv")
            _attn(fa2, fvk, Q, K, V, O, lse, x8, L["a_o"], batch=1, seq=T, heads=NH, hd=HD, st=st, prec=prec)
        _mm(gemm, prec, x8, O, L, "o", o_out, T, D, D, st)
        _gnorm8(lk, fvk, x, o_out, L["o_b"], gate_msa, 0, 0, EPS_LAM, 0, 0, scale_mlp, shift_mlp, x8, L["a_fc1"],
                T, D, st, prec, f16)
        h16 = _fc1_gelu(gemm, lk, fvk, prec, x8, f16, L, fc1, L["a_fc2"], T, FF, D, st, f16)
        _mm(gemm, prec, x8, h16, L, "fc2", o_out, T, D, FF, st)
        pend = (o_out, L["fc2_b"], gate_mlp, 0)
    _gnorm8(lk, fvk, x, pend[0], pend[1], pend[2], 0, 2, 0.0, 0, 0, 0, 0, x8, W["a_pout"], T, D, st, prec, f16)
    _mm(gemm, prec, x8, f16, W, "pout", B["pout"], T, 768, D, st)
    # BF16 flow image tokens, and their FP16 copy for the FP16-accumulating K / V projection
    _ck(lk.gated_residual_norm(B["pout"], 0, W["pout_b"], 0, 0, 0, EPS_LN, W["last_w"], W["last_b"], 0, 0, 0, 0,
                               B["img16"] + 256 * 768 * 2 if "img16" in B else 0, B["img_kv_in"] + 256 * 768 * 2, T,
                               768, st), "lam last_ln")


# ─────────────────────────────────────────────────────────────────────────
# Flow head: conditioning (enc_vlm + cross K/V) and the DiT denoising loop
# ─────────────────────────────────────────────────────────────────────────

D_DIT, FF_DIT, LAYERS_DIT, HEADS_DIT, HD_DIT = 1024, 4096, 16, 16, 64
KV_LD = 4 * 2 * D_DIT     # 4 cross blocks per token kind, K | V each
KV_TILE = 1               # lawam.gemm_f16acc_bf16out tile (128 x 256)


def flow_condition_forward(gemm, fvk, lk, B, W, dims, *, stream=0, prec="fp8"):
    """enc_vlm (FP8 GEMM + LayerNorm) of the VLM states, then the K / V of the
    four text and four image cross blocks (BF16 [rows, KV_LD])."""
    flow_condition_text(gemm, lk, B, W, dims, stream=stream, prec=prec)
    flow_condition_img(gemm, lk, B, W, 0, 512, stream=stream)


def flow_condition_text(gemm, lk, B, W, dims, *, stream=0, prec="fp8"):
    """enc_vlm of the VLM states and the K / V of the four text cross blocks
    (depends on the VLM only)."""
    S = int(dims["S"])
    st = int(stream)
    _mm(gemm, prec, B["vlm8"], B["vlm"], W, "enc", B["enc_out"], S, 768, 2048, st)
    if "kv_text_w16" in W:
        # FP16 operands with FP16 accumulation (the BF16 path accumulates in FP32 at half the
        # tensor rate on sm_120), BF16 K / V out
        _ck(lk.gated_residual_norm(B["enc_out"], 0, W["enc_b"], 0, 0, 0, EPS_LN, W["enc_ln_w"], W["enc_ln_b"], 0,
                                   0, 0, 0, B["cond16"], 0, S, 768, st), "enc_vlm ln")
        _ck(lk.gemm_f16acc_bf16out(B["cond16"], W["kv_text_w16"], B["kv_text"], W["kv_text_b"], S, KV_LD, 768,
                                   KV_TILE, st), "kv text")
        return
    _ck(lk.gated_residual_norm(B["enc_out"], 0, W["enc_b"], 0, 0, 0, EPS_LN, W["enc_ln_w"], W["enc_ln_b"], 0, 0, 0,
                               0, 0, B["cond_vlm"], S, 768, st), "enc_vlm ln")
    gemm.bf16_nt_bias(B["cond_vlm"], W["kv_text_w"], B["kv_text"], W["kv_text_b"], S, KV_LD, 768, st)


def flow_condition_img(gemm, lk, B, W, r0, r1, *, stream=0):
    """K / V of the four image cross blocks for image tokens [r0, r1) (0..255
    DINOv3 latents, 256..511 the LAM prediction). The FP16-accumulate GEMM
    computes every row the same way for any row range (whole 128-row tiles);
    the BF16 path runs all 512 rows at once only."""
    st = int(stream)
    if "kv_text_w16" in W:
        assert r0 % 128 == 0 and r1 % 128 == 0
        _ck(lk.gemm_f16acc_bf16out(B["img16"] + r0 * 768 * 2, W["kv_img_w16"], B["kv_img"] + r0 * KV_LD * 2,
                                   W["kv_img_b"], r1 - r0, KV_LD, 768, KV_TILE, st), "kv img")
    elif (r0, r1) == (0, 512):
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
    of B["x"]. T holds the per-step AdaLN tables, W the step-invariant GEMM
    sites (w, sf / w16, gs, bias).

    W["mode"] "nvfp4": skinny W4A4 GEMMs with K-split FP32 partials, block
    inputs quantized to NVFP4 by the consumers. "w8a16": skinny weight-only
    INT8 GEMMs (per-channel scales) on BF16 block inputs, same partial layout
    (alpha = 1). "bf16": BF16 weights, FP32-output cuBLASLt GEMMs whose result
    the consumers read as a single partial."""
    rows, S_text = int(dims["rows"]), int(dims["S"])
    D, FF = D_DIT, FF_DIT
    st = int(stream)
    mode = W["mode"]
    bf = mode in ("bf16", "w8a16", "w4a16")        # BF16 block inputs
    ffn4 = W.get("ffn_mode") == "nvfp4"            # FFN on skinny NVFP4 (W4A4) inside a w8a16 DiT
    h, part, one = B["h"], B["part"], B["one"]
    O, hmod, ae = B["O"], B["hmod"], B["ae"]
    if bf:
        xn, xn_sf, xb = 0, 0, B["xb"]           # block input as BF16
        hid, hid_sf, hb = 0, 0, B["hb"]
        gs_out = 0
    else:
        xn, xn_sf, xb = B["xn"], B["xn_sf"], 0
        hid, hid_sf, hb = B["hid"], B["hid_sf"], 0
        gs_out = one
    scale, shift = T["scale"][step], T["shift"][step]
    pos = T["pos"]

    def gemm(a, a_sf, a_bf16, site, n, k, what, bf16_act=False):
        if "w4" in site:          # NVFP4 weight-only site (BF16 block inputs)
            c = W["w4_cfg"][(n, k)]
            _ck(lk.w4a16_skinny_gemm(a_bf16, site["w4"], site["sf4"], part, rows, n, k, c, pdl, st), what)
            return k // lk.w4a16_skinny_config_k_chunk(c)
        if "w" in site and mode != "nvfp4":   # NVFP4 site (FFN of the mixed tier)
            return _skinny(fvk, a, a_sf, site, part, rows, n, k, cfg, pdl, st, what)
        if mode == "w8a16":
            c = W["w8_cfg"][(n, k)]
            _ck(lk.w8a16_skinny_gemm(a_bf16, site["w8"], site["ws"], part, rows, n, k, c, pdl, st), what)
            return k // lk.w8a16_skinny_config_k_chunk(c)
        if mode == "bf16":
            _ck(lk.gemm_bf16_f32out(a_bf16, site["w16"], part, rows, n, k, st), what)
            return 1
        if bf16_act:
            return _skinny(fvk, a_bf16, 0, site, part, rows, n, k, cfg, pdl, st, what, bf16_act=True, one=one)
        return _skinny(fvk, a, a_sf, site, part, rows, n, k, cfg, pdl, st, what)

    _ck(lk.action_encode(B["x"], W["ae_w1"], W["ae_b1"], ae, rows, 32, D, st), "action encode")
    site = W["ae2"]
    sp = gemm(0, 0, ae, site, D, D, "action enc w2", bf16_act=True)
    _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, False, scale[0], shift[0], pos, xn,
                                      xn_sf, gs_out, xb, rows, D, EPS_LN, pdl, st), "dit in-norm")
    for li in range(LAYERS_DIT):
        if li % 2 == 1:
            site = W["qkv"][li // 2]
            sp = gemm(xn, xn_sf, xb, site, 3 * D, D, "self qkv")
            _ck(fvk.nvfp4_skinny_partials_attention(part, sp, one, site["gs"], site["b"], 3 * D, 0, 0, 0, rows,
                                                    False, O, rows, HEADS_DIT, HD_DIT, HD_DIT ** -0.5, pdl, st),
                "self attention")
        else:
            site = W["q"][li // 2]
            sp = gemm(xn, xn_sf, xb, site, D, D, "cross q")
            g = li // 4
            if li % 4 == 0:
                kv = B["kv_text"] + g * 2 * D * 2
                _ck(fvk.nvfp4_skinny_partials_attention(part, sp, one, site["gs"], site["b"], D, kv, kv + D * 2,
                                                        KV_LD, S_text, True, O, rows, HEADS_DIT, HD_DIT,
                                                        HD_DIT ** -0.5, pdl, st), "cross attention")
            elif W.get("img_slices"):
                # 512 image keys exceed the partials-attention shared-memory
                # budget: key slices over CTAs (merged by the last), Q read
                # from the partials like the partials attention
                kv = B["kv_img"] + g * 2 * D * 2
                _ck(lk.dit_cross_attention(part, sp, one, site["gs"], site["b"], kv, kv + D * 2, KV_LD, 512,
                                           W["img_slices"], O, rows, HEADS_DIT, HD_DIT, HD_DIT ** -0.5,
                                           B["xattn_state"], B["xattn_cnt"], pdl, st), "image cross attention")
            else:
                # or a BF16 Q store, then the small-query FA2 tile kernel
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
        sp = gemm(0, 0, O, site, D, D, "o proj", bf16_act=True)
        if ffn4:
            _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, True, 0, 0, 0, B["xn"],
                                              B["xn_sf"], one, 0, rows, D, EPS_LN, pdl, st), "o residual norm")
            site = W["ff_up"][li]
            sp = gemm(B["xn"], B["xn_sf"], 0, site, FF, D, "ffn up")
            _ck(fvk.nvfp4_skinny_sum_bias_gelu(part, sp, one, site["gs"], site["b"], B["hid"], B["hid_sf"], one, rows,
                                               FF, pdl, st), "ffn gelu")
            site = W["ff_down"][li]
            sp = gemm(B["hid"], B["hid_sf"], 0, site, D, FF, "ffn down")
        else:
            _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, True, 0, 0, 0, xn, xn_sf,
                                              gs_out, xb, rows, D, EPS_LN, pdl, st), "o residual norm")
            site = W["ff_up"][li]
            sp = gemm(xn, xn_sf, xb, site, FF, D, "ffn up")
            if bf:
                _ck(lk.dit_sum_bias_act(part, sp, one, site["gs"], site["b"], 2, 0, 0, 0, hb, rows, FF, pdl, st),
                    "ffn gelu")
            else:
                _ck(fvk.nvfp4_skinny_sum_bias_gelu(part, sp, one, site["gs"], site["b"], hid, hid_sf, one, rows, FF,
                                                   pdl, st), "ffn gelu")
            site = W["ff_down"][li]
            sp = gemm(hid, hid_sf, hb, site, D, FF, "ffn down")
        if li < LAYERS_DIT - 1:
            _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, True, scale[li + 1],
                                              shift[li + 1], pos, xn, xn_sf, gs_out, xb, rows, D, EPS_LN, pdl, st),
                "ffn residual adaln")
        else:
            _ck(lk.dit_sum_bias_residual_norm(part, sp, one, site["gs"], site["b"], h, True, T["out_scale"][step],
                                              T["out_shift"][step], 0, 0, 0, 0, hmod, rows, D, 1e-6, pdl, st),
                "ffn residual out-norm")
    site = W["head1"]
    sp = gemm(0, 0, hmod, site, D, D, "head fold", bf16_act=True)
    _ck(lk.dit_sum_bias_act(part, sp, one, site["gs"], site["b"], 1, hid, hid_sf, gs_out, hb, rows, D, pdl, st),
        "head relu")
    site = W["head2"]
    sp = gemm(hid, hid_sf, hb, site, 32, D, "head out")
    _ck(lk.dit_sum_bias_euler(part, sp, one, site["gs"], site["b32"], B["x"], float(dims["dt"]), rows, 32, pdl, st),
        "euler")


# ─────────────────────────────────────────────────────────────────────────
# Flow loop as one persistent kernel (lawam_dit_persistent_sm120.cu)
# ─────────────────────────────────────────────────────────────────────────

DIT_UNIT = 2304          # packed weight unit / ring slot (bytes)
DIT_WARPS = 8            # warps of the persistent kernel
DIT_CHUNK = 256          # packed NVFP4 unit (k)
_OP_GEMM, _OP_ATTN, _OP_AE, _OP_KVT, _OP_UNNORM, _OP_LN = range(6)
_A_BUF, _A_LN = range(2)     # host-side only: an LN GEMM input becomes an OP_LN into xa + a buffer GEMM
_EP_RESID_INIT, _EP_RESID, _EP_GELU, _EP_RELU, _EP_QKV, _EP_Q, _EP_EULER, _EP_PARTIAL = range(8)
(_I_KIND, _I_BAR, _I_SITE, _I_WTYPE, _I_N, _I_K, _I_EP, _I_KVLEN, _I_KLD, _I_VTLD, _I_KSPLIT,
 _I_OWNERS) = range(12)
_F_EPS, _F_DT, _F_QSCALE = range(3)
(_P_ABUF, _P_LNSC, _P_LNSH, _P_POS, _P_BIAS, _P_BIAS32, _P_GS, _P_WS, _P_OUT0, _P_OUT1, _P_OUT2, _P_Q, _P_K,
 _P_VT, _P_PART, _P_STATS) = range(16)
DIT_KSPLIT_FC2 = 4       # ffn down (K 4096): 4 k groups of 1024 x column groups, partials summed by the next LN
assert DIT_KSPLIT_FC2 <= 4, "the flow kernel's LayerNorm phase sums at most 4 K-split partials"


def dit_sites(W):
    """GEMM sites of one flow step in stream order: ae2, (in-proj, o, fc1, fc2) x 16, head1, head2."""
    sites = [W["ae2"]]
    for li in range(LAYERS_DIT):
        sites += [W["qkv"][li // 2] if li % 2 == 1 else W["q"][li // 2], W["o"][li], W["ff_up"][li], W["ff_down"][li]]
    return sites + [W["head1"], W["head2"]]


def _w4_units(torch, w4, sf, N, K):
    """NVFP4 weight-only site -> [N / 16, K / 256, 2304] units (fragment-ordered E2M1 pairs + UE4M3 scales)."""
    dev = w4.device
    b = torch.arange(2048, device=dev)
    lane, byte = (b % 512) // 16, b % 16
    G, s, which = 2 * (b // 512) + byte // 8, (byte % 8) // 4, byte % 4
    g, t = lane // 4, lane % 4
    row = g + 8 * (which & 1)
    k = 32 * G + 8 * t + 4 * s + 2 * (which >> 1)
    tiles, chunks = N // 16, K // DIT_CHUNK
    frag = w4.view(tiles, 16, chunks, DIT_CHUNK // 2)[:, row, :, k // 2]          # [2048, tiles, chunks]
    scales = sf.view(tiles, 16, chunks, DIT_CHUNK // 16).permute(0, 2, 1, 3).reshape(tiles, chunks, 256)
    return torch.cat([frag.permute(1, 2, 0), scales], dim=2).contiguous()


def _w8_units(torch, w8, N, K):
    """INT8 weight-only site -> [N / 16, K / 256, 2, 2304] units (two 128-wide k halves per chunk)."""
    dev = w8.device
    b = torch.arange(2048, device=dev)
    lane, byte = (b % 512) // 16, b % 16
    G, s, which, e = b // 512, byte // 8, (byte % 8) // 2, byte % 2
    g, t = lane // 4, lane % 4
    row = g + 8 * (which & 1)
    k = 32 * G + 8 * t + 4 * s + 2 * (which >> 1) + e
    tiles = N // 16
    frag = w8.view(torch.uint8).view(tiles, 16, K // 128, 128)[:, row, :, k]      # [2048, tiles, K / 128]
    units = torch.zeros(tiles, K // 128, DIT_UNIT, dtype=torch.uint8, device=dev)
    units[:, :, :2048] = frag.permute(1, 2, 0)
    return units.view(tiles, K // DIT_CHUNK, 2, DIT_UNIT)


def dit_site_ksplit(si):
    """K groups of GEMM site si (the ffn down projections are split 4 ways)."""
    return DIT_KSPLIT_FC2 if 1 <= si <= 4 * LAYERS_DIT and (si - 1) % 4 == 3 else 1


def _run_order(U, warps=7):
    """Stream order of a CTA's U GEMM units: contiguous runs per warp (the
    first U % warps take one more), interleaved round by round."""
    q, a = divmod(U, warps)
    runs = [(w * q + min(w, a), q + (1 if w < a else 0)) for w in range(warps)]
    return [s0 + r for r in range(q + (1 if a else 0)) for s0, n in runs if r < n]


def validate_dit_num_ctas(num_ctas):
    # Attention needs 16 heads x two query tiles; four-way K splits and
    # the 64-owner LN reduction define the remaining grid constraints.
    if not isinstance(num_ctas, int) or isinstance(num_ctas, bool) or not 32 <= num_ctas <= 64 or num_ctas % 4:
        raise ValueError("LaWAM persistent DiT needs 32..64 CTAs in multiples of 4")
    return num_ctas


def select_dit_num_ctas(num_sms):
    return validate_dit_num_ctas(min(int(num_sms), 64) // 4 * 4)


def pack_dit_persistent(torch, W, num_ctas, device):
    """Pack the step-invariant DiT GEMM sites for the persistent flow kernel.

    Every site's 16-column tiles are split over the CTAs (or, for a K-split
    site, over the column groups of num_ctas / ksplit CTAs, CTA c taking k
    group c % ksplit) in contiguous ranges, the remainder tiles rotating from
    site to site; each CTA's units are laid out in consumption order (site,
    k chunk, tile[, k half]). Returns (stream [num_ctas, units, 2304] uint8,
    site table int32 [sites, num_ctas, 4], units per CTA int32 [num_ctas])."""
    validate_dit_num_ctas(num_ctas)
    sites = dit_sites(W)
    per_cta = [[] for _ in range(num_ctas)]
    count = [0] * num_ctas
    tab = np.zeros((len(sites), num_ctas, 4), dtype=np.int32)
    rot = 0
    for si, site in enumerate(sites):
        N, K = int(site["N"]), int(site["K"])
        if "w4" in site:
            units = _w4_units(torch, site["w4_t"], site["sf4_t"], N, K)                # [tiles, chunks, U]
        else:
            units = _w8_units(torch, site["w8_t"].contiguous(), N, K)                   # [tiles, chunks, 2, U]
        ks = dit_site_ksplit(si)
        groups = num_ctas // ks
        nk = K // DIT_CHUNK // ks
        tiles = N // 16
        base, rem = divmod(tiles, groups)
        for c in range(num_ctas):
            grp, kg = divmod(c, ks)
            r = (grp - rot) % groups
            n = base + (1 if r < rem else 0)
            t0 = r * base + min(r, rem)
            tab[si, c] = (t0, n, count[c], kg * nk)
            if n:
                u = units[t0:t0 + n, kg * nk:(kg + 1) * nk].reshape(n * nk, -1, DIT_UNIT)   # u = tile * nk + chunk
                u = u[torch.as_tensor(_run_order(n * nk, DIT_WARPS), device=u.device)].reshape(-1, DIT_UNIT)
                per_cta[c].append(u)
                count[c] += u.shape[0]
        rot = (rot + rem) % groups
    stream = torch.zeros(num_ctas, max(count), DIT_UNIT, dtype=torch.uint8, device=device)
    for c in range(num_ctas):
        stream[c, :count[c]] = torch.cat(per_cta[c])
    return stream, tab, np.asarray(count, dtype=np.int32)


def dit_persistent_ops(W, T, B, dims, *, num_steps, lo, hi, action_dim):
    """Op table of the persistent flow kernel: [n_ops, 224] uint8 (DitOp)."""
    S = int(dims["S"])
    spad = int(dims["spad"])
    ops = []

    def op(kind, bar=1, i=None, f=None, p=None):
        e = (np.zeros(16, np.int32), np.zeros(8, np.float32), np.zeros(16, np.uint64))
        e[0][_I_KIND], e[0][_I_BAR] = kind, bar
        for k, v in (i or {}).items():
            e[0][k] = v
        for k, v in (f or {}).items():
            e[1][k] = v
        for k, v in (p or {}).items():
            e[2][k] = int(v)
        ops.append(e)

    D, qscale = D_DIT, HD_DIT ** -0.5 * 1.4426950408889634
    kvt, kvi = B["kv_text"], B["kv_img"]
    op(_OP_KVT, 0, {_I_KVLEN: S, _I_VTLD: spad, _I_KLD: KV_LD}, p={0: kvt, 1: B["vt_text"]})
    op(_OP_KVT, 0, {_I_KVLEN: 512, _I_VTLD: 512, _I_KLD: KV_LD}, p={0: kvi, 1: B["vt_img"]})

    # owners of the residual columns (row statistics partials) after the previous residual update,
    # or the K-split GEMM site whose partials are still to be added into the residual
    owners = {"n": 0, "pending": None}
    # the residual stream alternates between h and h2: the K-split residual update of an LN phase
    # reads whole old rows while other CTAs write their new segments
    hb_ = {"cur": B["h"], "alt": B["h2"]}

    def gemm(si, site, wtype, N, K, amode, ep, abuf, out0, *, eps=0.0, sc=0, sh=0, pos=0, out1=0, out2=0, dt=0.0):
        if amode == _A_LN:
            # the block-input LayerNorm runs once (distributed) into the FP16 operand xa
            pend = owners["pending"]
            if pend is not None:
                assert "w4" in pend[1], "the deferred K-split residual update expects an NVFP4 site"
                assert abuf == hb_["cur"]
                op(_OP_LN, 1, {_I_KSPLIT: dit_site_ksplit(pend[0])}, {_F_EPS: eps},
                   {_P_LNSC: sc, _P_LNSH: sh, _P_POS: pos, _P_OUT0: B["xa"], _P_OUT1: abuf, _P_OUT2: hb_["alt"],
                    _P_PART: B["part"], _P_GS: pend[1]["gs"], _P_BIAS: pend[1]["b"]})
                hb_["cur"], hb_["alt"] = hb_["alt"], hb_["cur"]
                owners["pending"] = None
            else:
                op(_OP_LN, 1, {_I_OWNERS: owners["n"]}, {_F_EPS: eps},
                   {_P_ABUF: abuf, _P_LNSC: sc, _P_LNSH: sh, _P_POS: pos, _P_OUT0: B["xa"], _P_STATS: B["rowstat"]})
            amode, abuf = _A_BUF, B["xa"]
        if ep == _EP_RESID and dit_site_ksplit(si) > 1:
            ep = _EP_PARTIAL                      # residual update deferred to the next LayerNorm phase
            owners["pending"] = (si, site)
        p = {_P_ABUF: abuf, _P_LNSC: 0, _P_LNSH: 0, _P_POS: 0, _P_BIAS: site["b"], _P_BIAS32: site["b32"],
             _P_OUT0: out0, _P_OUT1: out1, _P_OUT2: out2, _P_STATS: B["rowstat"]}
        if ep in (_EP_RESID, _EP_RESID_INIT):
            owners["n"] = B["num_ctas"] // dit_site_ksplit(si)
        if ep == _EP_PARTIAL:
            p[_P_PART] = B["part"]
        if wtype == 0:
            p[_P_GS] = site["gs"]
        else:
            p[_P_WS] = site["ws"]
        ks = dit_site_ksplit(si)
        assert ks == 1 or ep == _EP_PARTIAL, "a K-split site defers its residual update to the next LayerNorm"
        op(_OP_GEMM, 1, {_I_SITE: si, _I_WTYPE: wtype, _I_N: N, _I_K: K, _I_EP: ep, _I_KSPLIT: ks},
           {_F_EPS: eps, _F_DT: dt, _F_QSCALE: qscale}, p)

    sites = dit_sites(W)
    wt = lambda s: 0 if "w4" in s else 1   # noqa: E731
    for s in range(num_steps):
        op(_OP_AE, 1, {_I_N: 24, _I_K: 32}, p={0: B["x"], 1: W["ae_w1"], 2: W["ae_b1"], 3: B["ae"]})
        gemm(0, sites[0], 1, D, D, _A_BUF, _EP_RESID_INIT, B["ae"], hb_["cur"])
        for li in range(LAYERS_DIT):
            si = 1 + 4 * li
            ln = dict(eps=EPS_LN, sc=T["scale"][s][li], sh=T["shift"][s][li], pos=T["pos"])
            if li % 2 == 1:
                gemm(si, sites[si], wt(sites[si]), 3 * D, D, _A_LN, _EP_QKV, hb_["cur"], B["Qs"], out1=B["Ks"],
                     out2=B["Vts"], **ln)
                op(_OP_ATTN, 1, {_I_KVLEN: 24, _I_KLD: 1024, _I_VTLD: 32},
                   p={_P_Q: B["Qs"], _P_K: B["Ks"], _P_VT: B["Vts"], _P_OUT0: B["O"]})
            else:
                gemm(si, sites[si], wt(sites[si]), D, D, _A_LN, _EP_Q, hb_["cur"], B["Qs"], **ln)
                g = li // 4
                if li % 4 == 0:
                    op(_OP_ATTN, 1, {_I_KVLEN: S, _I_KLD: KV_LD, _I_VTLD: spad},
                       p={_P_Q: B["Qs"], _P_K: kvt + g * 2 * D * 2, _P_VT: B["vt_text"] + g * D * spad * 2,
                          _P_OUT0: B["O"]})
                else:
                    op(_OP_ATTN, 1, {_I_KVLEN: 512, _I_KLD: KV_LD, _I_VTLD: 512},
                       p={_P_Q: B["Qs"], _P_K: kvi + g * 2 * D * 2, _P_VT: B["vt_img"] + g * D * 512 * 2,
                          _P_OUT0: B["O"]})
            gemm(si + 1, sites[si + 1], wt(sites[si + 1]), D, D, _A_BUF, _EP_RESID, B["O"], hb_["cur"])
            gemm(si + 2, sites[si + 2], wt(sites[si + 2]), FF_DIT, D, _A_LN, _EP_GELU, hb_["cur"], B["hb"], eps=EPS_LN)
            gemm(si + 3, sites[si + 3], wt(sites[si + 3]), D, FF_DIT, _A_BUF, _EP_RESID, B["hb"], hb_["cur"])
        gemm(65, sites[65], wt(sites[65]), D, D, _A_LN, _EP_RELU, hb_["cur"], B["hb"], eps=1e-6,
             sc=T["out_scale"][s], sh=T["out_shift"][s])
        gemm(66, sites[66], wt(sites[66]), 32, D, _A_BUF, _EP_EULER, B["hb"], B["x"], dt=1.0 / num_steps)
    op(_OP_UNNORM, 0, {_I_N: 24, _I_K: 32, _I_KVLEN: action_dim}, p={0: B["x"], 1: lo, 2: hi, 3: B["actions"]})
    rows = [np.concatenate([e[0].view(np.uint8), e[1].view(np.uint8), e[2].view(np.uint8)]) for e in ops]
    return np.stack(rows)

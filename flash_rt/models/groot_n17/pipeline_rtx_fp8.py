"""GR00T N1.7 FP8 backbone forward pipeline for RTX (SM120 / SM89).

SM120-safe FP8 variant of ``pipeline_thor.py``'s backbone forwards. The Thor
path uses the fused cuBLAS FP8 epilogues (``fp8_nn_bias`` /
``fp8_nn_gelu_bias`` / ``fp8_nn_dev``); those hit ``CUBLAS_STATUS_NOT_SUPPORTED``
on SM120. This module uses the same SM120-proven pattern as the N1.6 RTX
pipeline (``flash_rt/models/groot/pipeline_rtx.py``):

    quantize_fp8_static_fp16(x, x_fp8, act_scale_devptr)
    gemm.fp8_descale_fp16(x_fp8, w_fp8, out, M, N, K,
                          act_scale_devptr, w_scale_devptr)   # out = act·w·(A@B), fp16
    fvk.add_bias_fp16(out, b)                                 # separate bias
    fvk.gelu_inplace_fp16(out, ...)                           # separate activation

Attention stays FP16 (the descale GEMM emits FP16 Q/K/V), so the existing
``RtxGrootN17BackboneAttn`` backend is reused unchanged.

Both per-tensor scales are device fp32 scalar pointers:
  * ``act_scale``  — from calibration (``self._<stage>_act_<point>_dev[li]``,
    or the disk calibration cache).
  * ``w_scale``    — the weight scale baked at load time
    (``self._<stage>_alpha[...]`` host float, uploaded to a device scalar by
    the frontend).

Stages mirror ``pipeline_thor.py``: ``qwen3vl_vit_forward`` →
``deepstack_merge_forward`` → ``qwen3vl_llm_forward`` → ``vlln_forward`` →
``vl_self_attn_forward``. Each forward is pointer-only (every device tensor is
an int ``data_ptr`` supplied by the frontend; no allocation, no host↔device
traffic).
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────
# Stage 4: VLLN — LayerNorm on backbone_features (no FP8; identical to thor)
# ─────────────────────────────────────────────────────────────────────────


def vlln_forward(gemm, fvk, bufs, weights, dims,
                 scales_dev=None, *, attn=None, stream: int = 0) -> None:
    """LayerNorm(2048, eps=1e-5) on backbone features ``(B, S, 2048)``.

    Required:
        bufs["x"], bufs["out"]      — fp16 (S × D)
        weights["vlln_w"], ["vlln_b"]
        dims["S"], dims["D"]
    """
    fvk.layer_norm_fp16(
        int(bufs["x"]), int(weights["vlln_w"]), int(weights["vlln_b"]),
        int(bufs["out"]), int(dims["S"]), int(dims["D"]), 1e-5, int(stream),
    )


# ─────────────────────────────────────────────────────────────────────────
# Stage 1: Qwen3-VL ViT (24 layers)
# ─────────────────────────────────────────────────────────────────────────


def qwen3vl_vit_forward(gemm, fvk, bufs, weights, dims,
                        scales_dev, *, attn, stream: int = 0,
                        layers_subset=None,
                        deepstack_taps=(5, 11, 17),
                        deepstack_capture=None) -> None:
    """24-layer Qwen3-VL ViT, FP8 GEMMs via SM120-safe descale.

    Per layer (in-place residual on ``h``): LayerNorm → quantize → 3 split FP8
    Q/K/V descale GEMMs (+bias) → split-half RoPE → multi-view FMHA → quantize
    O → FP8 o-proj (+bias) → residual → LayerNorm → quantize → FP8 fc1 (+bias,
    +GELU tanh) → quantize → FP8 fc2 (+bias) → residual.

    Q/K/V share the fused-qkv weight scale (``q_ws == k_ws == v_ws``).

    bufs:        h, xn (fp16 S×D); xn_fp8 (fp8 S×D); o_proj_out (fp16 S×D);
                 fc1_out (fp16 S×FF); fc1_fp8 (fp8 S×FF)
    weights:     norm1/2_w/b; q/k/v/o_w, q/k/v/o_b; fc1/fc2_w, fc1/fc2_b
                 (fp8 weight ptrs + fp16 bias ptrs); q/k/v/o_ws, fc1/fc2_ws
                 (weight-scale dev ptrs); cos, sin (fp16 dev S×HD)
    scales_dev:  act_qkv, act_o, act_fc1, act_fc2 (lists of fp32 dev ptrs)
    dims:        S, D, NH, HD, ff_inner, Sper_view
    """
    S  = int(dims["S"])
    D  = int(dims["D"])
    NH = int(dims["NH"])
    HD = int(dims["HD"])
    FF = int(dims["ff_inner"])

    h_ptr       = int(bufs["h"])
    xn_ptr      = int(bufs["xn"])
    xn_fp8_ptr  = int(bufs["xn_fp8"])
    o_proj_out  = int(bufs["o_proj_out"])
    fc1_out_ptr = int(bufs["fc1_out"])
    fc1_fp8_ptr = int(bufs["fc1_fp8"])
    cos_ptr     = int(weights["cos"])
    sin_ptr     = int(weights["sin"])

    layer_iter = range(24) if layers_subset is None else list(layers_subset)
    Sper = int(dims.get("Sper_view", S))

    for li in layer_iter:
        slots = attn.get_slot_ptrs("vit", li)
        Q_ptr, K_ptr, V_ptr, O_ptr = slots["Q"], slots["K"], slots["V"], slots["O"]
        a_qkv = int(scales_dev["act_qkv"][li])
        a_o   = int(scales_dev["act_o"][li])
        a_fc1 = int(scales_dev["act_fc1"][li])
        a_fc2 = int(scales_dev["act_fc2"][li])

        # ── Pre-attn LayerNorm ──
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
            xn_ptr, S, D, 1e-6, int(stream))

        # ── Quantize xn once for Q/K/V; 3 split FP8 descale GEMMs + bias ──
        fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_qkv, S * D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr,
                              S, D, D, a_qkv, int(weights["q_ws"][li]), int(stream))
        fvk.add_bias_fp16(Q_ptr, int(weights["q_b"][li]), S, D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["k_w"][li]), K_ptr,
                              S, D, D, a_qkv, int(weights["k_ws"][li]), int(stream))
        fvk.add_bias_fp16(K_ptr, int(weights["k_b"][li]), S, D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["v_w"][li]), V_ptr,
                              S, D, D, a_qkv, int(weights["v_ws"][li]), int(stream))
        fvk.add_bias_fp16(V_ptr, int(weights["v_b"][li]), S, D, int(stream))

        # ── Split-half RoPE on Q and K ──
        fvk.rope_rotate_half_fp16(Q_ptr, cos_ptr, sin_ptr, S, NH, HD, int(stream))
        fvk.rope_rotate_half_fp16(K_ptr, cos_ptr, sin_ptr, S, NH, HD, int(stream))

        # ── Multi-view batched FMHA ──
        attn.run("vit", li, q_seq=Sper, kv_seq=Sper, stream=int(stream))

        # ── O projection (FP8) ──
        fvk.quantize_fp8_static_fp16(O_ptr, xn_fp8_ptr, a_o, S * D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out,
                              S, D, D, a_o, int(weights["o_ws"][li]), int(stream))
        fvk.add_bias_fp16(o_proj_out, int(weights["o_b"][li]), S, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, S * D, int(stream))

        # ── Pre-FF LayerNorm ──
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm2_w"][li]), int(weights["norm2_b"][li]),
            xn_ptr, S, D, 1e-6, int(stream))

        # ── FF: D → FF (GELU) → D ──
        fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_fc1, S * D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["fc1_w"][li]), fc1_out_ptr,
                              S, FF, D, a_fc1, int(weights["fc1_ws"][li]), int(stream))
        fvk.add_bias_fp16(fc1_out_ptr, int(weights["fc1_b"][li]), S, FF, int(stream))
        fvk.gelu_inplace_fp16(fc1_out_ptr, S * FF, int(stream))
        fvk.quantize_fp8_static_fp16(fc1_out_ptr, fc1_fp8_ptr, a_fc2, S * FF, int(stream))
        gemm.fp8_descale_fp16(fc1_fp8_ptr, int(weights["fc2_w"][li]), o_proj_out,
                              S, D, FF, a_fc2, int(weights["fc2_ws"][li]), int(stream))
        fvk.add_bias_fp16(o_proj_out, int(weights["fc2_b"][li]), S, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, S * D, int(stream))

        # ── DeepStack tap callback ──
        if deepstack_capture is not None and li in deepstack_taps:
            deepstack_capture[deepstack_taps.index(li)](h_ptr)


# ─────────────────────────────────────────────────────────────────────────
# Stage 2: DeepStack mergers (3)
# ─────────────────────────────────────────────────────────────────────────


def deepstack_merge_forward(gemm, fvk, bufs, weights, dims,
                            scales_dev, *, attn=None, stream: int = 0) -> None:
    """3 DeepStack mergers (taps ViT [5, 11, 17]) → 3 features for LLM [0,1,2].

    Per merger j: LayerNorm(4096) → quantize → FP8 fc1 (+bias, +GELU tanh) →
    quantize → FP8 fc2 (+bias) → (Nout, 2048).

    bufs:       in (list[3]), ln_out, fp8_scratch, fc1_out, out (list[3])
    weights:    norm_w/b[j]; fc1/fc2_w[j] (fp8) + fc1/fc2_b[j]; fc1/fc2_ws[j]
    scales_dev: act_fc1[j], act_fc2[j]
    dims:       Nin, Din, Nout, Dmid, Dout
    """
    Nout = int(dims["Nout"])
    Dmid = int(dims["Dmid"])
    Dout = int(dims["Dout"])

    ln_out      = int(bufs["ln_out"])
    fp8_scratch = int(bufs["fp8_scratch"])
    fc1_out     = int(bufs["fc1_out"])

    for j in range(3):
        in_ptr  = int(bufs["in"][j])
        out_ptr = int(bufs["out"][j])
        a_fc1 = int(scales_dev["act_fc1"][j])
        a_fc2 = int(scales_dev["act_fc2"][j])

        fvk.layer_norm_fp16(
            in_ptr, int(weights["norm_w"][j]), int(weights["norm_b"][j]),
            ln_out, Nout, Dmid, 1e-6, int(stream))

        fvk.quantize_fp8_static_fp16(ln_out, fp8_scratch, a_fc1, Nout * Dmid, int(stream))
        gemm.fp8_descale_fp16(fp8_scratch, int(weights["fc1_w"][j]), fc1_out,
                              Nout, Dmid, Dmid, a_fc1, int(weights["fc1_ws"][j]), int(stream))
        fvk.add_bias_fp16(fc1_out, int(weights["fc1_b"][j]), Nout, Dmid, int(stream))
        fvk.gelu_inplace_fp16(fc1_out, Nout * Dmid, int(stream))

        fvk.quantize_fp8_static_fp16(fc1_out, fp8_scratch, a_fc2, Nout * Dmid, int(stream))
        gemm.fp8_descale_fp16(fp8_scratch, int(weights["fc2_w"][j]), out_ptr,
                              Nout, Dout, Dmid, a_fc2, int(weights["fc2_ws"][j]), int(stream))
        fvk.add_bias_fp16(out_ptr, int(weights["fc2_b"][j]), Nout, Dout, int(stream))


# ─────────────────────────────────────────────────────────────────────────
# Stage 3: Qwen3-VL truncated LLM (16 layers, causal, GQA)
# ─────────────────────────────────────────────────────────────────────────


def qwen3vl_llm_forward(gemm, fvk, bufs, weights, dims,
                        scales_dev, *, attn, stream: int = 0,
                        layers_subset=None) -> None:
    """16 truncated Qwen3-VL LLM decoder layers, FP8 GEMMs via SM120-safe descale.

    Per layer: RMSNorm → quantize → 3 split FP8 Q/K/V descale GEMMs (no bias) →
    per-head q/k RMSNorm → M-RoPE → GQA expand → causal MHA → quantize O →
    FP8 o-proj → residual → RMSNorm → quantize → FP8 gate/up → SiLU(gate)*up
    fused-to-FP8 → FP8 down → residual → optional DeepStack inject.

    Q/K/V share the fused-qkv weight scale.

    bufs:       h, xn (fp16 S×D); xn_fp8 (fp8 S×D); Q (fp16 S×NHQ·HD);
                K, V (fp16 S×NHKV·HD); K_exp, V_exp (fp16 S×NHQ·HD);
                o_proj_out (fp16 S×D); gate_out, up_out (fp16 S×FF);
                gu_fp8 (fp8 S×FF)
    weights:    in_ln_w, post_ln_w, q_norm_w, k_norm_w; q/k/v/o_w, gate/up/down_w
                (fp8); q/k/v/o_ws, gate/up/down_ws (weight-scale dev ptrs);
                cos, sin; deepstack_inject (list[16], 0 = none)
    scales_dev: act_qkv, act_o, act_gateup, act_down
    dims:       S, D, NHQ, NHKV, HD, FF
    """
    S    = int(dims["S"])
    D    = int(dims["D"])
    NHQ  = int(dims["NHQ"])
    NHKV = int(dims["NHKV"])
    HD   = int(dims["HD"])
    FF   = int(dims["FF"])
    GQA  = NHQ // NHKV

    h_ptr      = int(bufs["h"])
    xn_ptr     = int(bufs["xn"])
    xn_fp8_ptr = int(bufs["xn_fp8"])
    Q_ptr      = int(bufs["Q"])
    K_ptr      = int(bufs["K"])
    V_ptr      = int(bufs["V"])
    K_exp_ptr  = int(bufs["K_exp"])
    V_exp_ptr  = int(bufs["V_exp"])
    o_out_ptr  = int(bufs["o_proj_out"])
    gate_ptr   = int(bufs["gate_out"])
    up_ptr     = int(bufs["up_out"])
    gu_fp8_ptr = int(bufs["gu_fp8"])
    cos_ptr    = int(weights["cos"])
    sin_ptr    = int(weights["sin"])

    inject_ptrs = weights.get("deepstack_inject", [0] * 16)
    layer_iter = range(16) if layers_subset is None else list(layers_subset)

    for li in layer_iter:
        slots = attn.get_slot_ptrs("llm", li)
        a_qkv = int(scales_dev["act_qkv"][li])
        a_o   = int(scales_dev["act_o"][li])
        a_gu  = int(scales_dev["act_gateup"][li])
        a_dn  = int(scales_dev["act_down"][li])

        # ── Pre-attn RMSNorm + quantize ──
        fvk.rms_norm_fp16(h_ptr, int(weights["in_ln_w"][li]), xn_ptr,
                          S, D, 1e-6, int(stream))
        fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_qkv, S * D, int(stream))

        # ── 3 split FP8 descale GEMMs (no bias — Qwen3 QKV) ──
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr,
                              S, NHQ * HD, D, a_qkv, int(weights["q_ws"][li]), int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["k_w"][li]), K_ptr,
                              S, NHKV * HD, D, a_qkv, int(weights["k_ws"][li]), int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["v_w"][li]), V_ptr,
                              S, NHKV * HD, D, a_qkv, int(weights["v_ws"][li]), int(stream))

        # ── Per-head q_norm / k_norm (BEFORE M-RoPE) ──
        fvk.rms_norm_fp16(Q_ptr, int(weights["q_norm_w"][li]), Q_ptr,
                          S * NHQ, HD, 1e-6, int(stream))
        fvk.rms_norm_fp16(K_ptr, int(weights["k_norm_w"][li]), K_ptr,
                          S * NHKV, HD, 1e-6, int(stream))

        # ── M-RoPE on Q and K ──
        fvk.rope_rotate_half_fp16(Q_ptr, cos_ptr, sin_ptr, S, NHQ,  HD, int(stream))
        fvk.rope_rotate_half_fp16(K_ptr, cos_ptr, sin_ptr, S, NHKV, HD, int(stream))

        # ── GQA expand: K, V from NHKV → NHQ heads ──
        fvk.gpu_repeat_interleave_heads(K_ptr, K_exp_ptr, S, NHKV, HD, GQA, int(stream))
        fvk.gpu_repeat_interleave_heads(V_ptr, V_exp_ptr, S, NHKV, HD, GQA, int(stream))

        # ── Causal MHA via attn backend ──
        attn.run("llm", li, q_seq=S, kv_seq=S, stream=int(stream))

        # ── O projection (FP8) ──
        fvk.quantize_fp8_static_fp16(int(slots["O"]), xn_fp8_ptr, a_o,
                                     S * NHQ * HD, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["o_w"][li]), o_out_ptr,
                              S, D, NHQ * HD, a_o, int(weights["o_ws"][li]), int(stream))
        fvk.residual_add_fp16(h_ptr, o_out_ptr, S * D, int(stream))

        # ── Pre-FFN RMSNorm + quantize ──
        fvk.rms_norm_fp16(h_ptr, int(weights["post_ln_w"][li]), xn_ptr,
                          S, D, 1e-6, int(stream))
        fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_gu, S * D, int(stream))

        # ── gate / up FP8 GEMMs → fp16 ──
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["gate_w"][li]), gate_ptr,
                              S, FF, D, a_gu, int(weights["gate_ws"][li]), int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["up_w"][li]), up_ptr,
                              S, FF, D, a_gu, int(weights["up_ws"][li]), int(stream))

        # ── SiLU(gate) * up → directly FP8 (fused) → down GEMM ──
        fvk.silu_mul_split_fp8_fp16(gate_ptr, up_ptr, gu_fp8_ptr, S * FF,
                                    a_dn, int(stream))
        gemm.fp8_descale_fp16(gu_fp8_ptr, int(weights["down_w"][li]), o_out_ptr,
                              S, D, FF, a_dn, int(weights["down_ws"][li]), int(stream))
        fvk.residual_add_fp16(h_ptr, o_out_ptr, S * D, int(stream))

        # ── DeepStack injection (HF: layers 0, 1, 2) ──
        inject_ptr = int(inject_ptrs[li]) if li < len(inject_ptrs) else 0
        if inject_ptr != 0:
            fvk.residual_add_fp16(h_ptr, inject_ptr, S * D, int(stream))


# ─────────────────────────────────────────────────────────────────────────
# Stage 5: VL self-attention (4 layers)
# ─────────────────────────────────────────────────────────────────────────


def vl_self_attn_forward(gemm, fvk, bufs, weights, dims,
                         scales_dev, *, attn, stream: int = 0,
                         layers_subset=None) -> None:
    """4-layer SelfAttentionTransformer, FP8 GEMMs via SM120-safe descale.

    Per layer: LayerNorm → quantize → FP8 Q/K/V (+bias, separate weight scales)
    → MHA → quantize O → FP8 o-proj (+bias) → residual → LayerNorm → quantize →
    FP8 fc1 (+bias, +GELU tanh) → quantize → FP8 fc2 (+bias) → residual.

    Q/K/V each have their OWN weight scale (separate projections, not fused).

    bufs:       h, xn (fp16 T×D); xn_fp8 (fp8 T×D); o_proj_out (fp16 T×D);
                fc1_out (fp16 T×FF); fc1_fp8 (fp8 T×FF)
    weights:    norm1/3_w/b; q/k/v/o_w, q/k/v/o_b; fc1/fc2_w, fc1/fc2_b (fp8);
                q/k/v/o_ws, fc1/fc2_ws (weight-scale dev ptrs)
    scales_dev: act_qkv, act_o, act_fc1, act_fc2
    dims:       T, D, NH, HD, ff_inner
    """
    T  = int(dims["T"])
    D  = int(dims["D"])
    FF = int(dims["ff_inner"])

    h_ptr       = int(bufs["h"])
    xn_ptr      = int(bufs["xn"])
    xn_fp8_ptr  = int(bufs["xn_fp8"])
    o_proj_out  = int(bufs["o_proj_out"])
    fc1_out_ptr = int(bufs["fc1_out"])
    fc1_fp8_ptr = int(bufs["fc1_fp8"])

    layer_iter = range(4) if layers_subset is None else list(layers_subset)

    for li in layer_iter:
        slots = attn.get_slot_ptrs("vl_self_attn", li)
        Q_ptr, K_ptr, V_ptr, O_ptr = slots["Q"], slots["K"], slots["V"], slots["O"]
        a_qkv = int(scales_dev["act_qkv"][li])
        a_o   = int(scales_dev["act_o"][li])
        a_fc1 = int(scales_dev["act_fc1"][li])
        a_fc2 = int(scales_dev["act_fc2"][li])

        # ── Pre-attn LayerNorm + quantize ──
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
            xn_ptr, T, D, 1e-5, int(stream))
        fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_qkv, T * D, int(stream))

        # ── Q / K / V FP8 descale GEMMs + bias (separate weight scales) ──
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr,
                              T, D, D, a_qkv, int(weights["q_ws"][li]), int(stream))
        fvk.add_bias_fp16(Q_ptr, int(weights["q_b"][li]), T, D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["k_w"][li]), K_ptr,
                              T, D, D, a_qkv, int(weights["k_ws"][li]), int(stream))
        fvk.add_bias_fp16(K_ptr, int(weights["k_b"][li]), T, D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["v_w"][li]), V_ptr,
                              T, D, D, a_qkv, int(weights["v_ws"][li]), int(stream))
        fvk.add_bias_fp16(V_ptr, int(weights["v_b"][li]), T, D, int(stream))

        # ── MHA ──
        attn.run("vl_self_attn", li, q_seq=T, kv_seq=T, stream=int(stream))

        # ── O projection (FP8) ──
        fvk.quantize_fp8_static_fp16(O_ptr, xn_fp8_ptr, a_o, T * D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out,
                              T, D, D, a_o, int(weights["o_ws"][li]), int(stream))
        fvk.add_bias_fp16(o_proj_out, int(weights["o_b"][li]), T, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, T * D, int(stream))

        # ── Pre-FF LayerNorm + FF (GELU) ──
        fvk.layer_norm_fp16(
            h_ptr, int(weights["norm3_w"][li]), int(weights["norm3_b"][li]),
            xn_ptr, T, D, 1e-5, int(stream))
        fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_fc1, T * D, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["fc1_w"][li]), fc1_out_ptr,
                              T, FF, D, a_fc1, int(weights["fc1_ws"][li]), int(stream))
        fvk.add_bias_fp16(fc1_out_ptr, int(weights["fc1_b"][li]), T, FF, int(stream))
        fvk.gelu_inplace_fp16(fc1_out_ptr, T * FF, int(stream))
        fvk.quantize_fp8_static_fp16(fc1_out_ptr, fc1_fp8_ptr, a_fc2, T * FF, int(stream))
        gemm.fp8_descale_fp16(fc1_fp8_ptr, int(weights["fc2_w"][li]), o_proj_out,
                              T, D, FF, a_fc2, int(weights["fc2_ws"][li]), int(stream))
        fvk.add_bias_fp16(o_proj_out, int(weights["fc2_b"][li]), T, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, T * D, int(stream))

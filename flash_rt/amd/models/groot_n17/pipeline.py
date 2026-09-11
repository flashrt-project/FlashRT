"""GR00T N1.7 FP8 backbone forward pipeline for AMD CDNA4 (MI350X, gfx950).

Text mirror of ``flash_rt/models/groot_n17/pipeline_rtx_fp8.py`` — the
validated SM120-safe FP8 decomposed form — retargeted at the AMD kernel
surface (``flash_rt.amd.flash_rt_amd_kernels``). The decomposition is
numerics-identical to the RTX production path:

    quantize_fp8_static_fp16(x, x_fp8, act_scale_devptr)
    gemm.fp8_descale_fp16(x_fp8, w_fp8, out, M, N, K,
                          act_scale_devptr, w_scale_devptr)   # out = act·w·(A@B), fp16
    fvk.add_bias_fp16(out, b)                                 # separate bias
    fvk.gelu_inplace_fp16(out, ...)                           # separate activation

hipBLASLt does support fused fp8 bias/gelu epilogues (``fp8_nn_bias`` /
``fp8_nn_gelu_bias``), and the biased forwards (vit / deepstack /
vl_self_attn) now carry a FUSED-EPILOGUE tier behind the keyword-only
``fused_epilogue`` switch (default False — the decomposed form stays the
directly-RTX-comparable baseline). Fused form per biased GEMM site:

    quantize_fp8_static_fp16(x, x_fp8, act_scale_devptr)          # unchanged
    gemm.fp8_nn_bias(x_fp8, w_fp8, out, bias, M, N, K, alpha)     # bias in epilogue
    gemm.fp8_nn_gelu_bias(...)                                    # bias+GELU sites

with HOST ``alpha = act_scale * w_scale`` (python float) supplied through
the ``alphas`` dict (key names parallel to ``scales_dev``; required when
``fused_epilogue=True``). Numerics: the fused epilogue adds the bias on
the FP32 accumulator BEFORE the fp16 round, whereas the decomposed form
adds it AFTER rounding — slightly different, judged by the E2E gate, not
bit-parity. The llm stage is biasless (Qwen3) and keeps descale GEMMs;
its ``fused_epilogue`` flag instead fuses the norm/residual elementwise
chains (see its docstring).

The fused tier additionally collapses the norm→quantize front-ends into
single kernels (same FVK_AMD_FUSED_EPILOGUE gate): every
``layer_norm_fp16 → quantize_fp8_static_fp16`` pair whose fp16 normed
output feeds ONLY the quantize runs as ``layer_norm_fp8_static_fp16_vec``
(bit-matching — fp16 round-through before the quantize; falls back to
the pair when the vec preconditions return rc != 0), and the llm's
RMSNorm chains run as ``rms_norm_fp8_fp16`` /
``residual_add_rms_norm_fp8_fp16`` (last-ULP deltas, see the llm
docstring).

Attention stays FP16 (the descale GEMM emits FP16 Q/K/V) and delegates
to :class:`flash_rt.amd.hardware.cdna4.attn_backend_groot_n17.Cdna4GrootN17AttnBackend`.

Sanctioned delta vs the RTX source (the ONLY computation difference):

  * llm GQA expand is SKIPPED. The AMD backend's llm K/V slots hold the
    model's NATIVE 8 KV heads and aiter's ``mha_fwd`` performs GQA
    internally, so the RTX ``gpu_repeat_interleave_heads`` K/V → 16-head
    expand step is dropped and the Q/K/V GEMMs write straight into the
    backend slots (Q 16 heads, K/V 8 heads).

Both per-tensor scales are device fp32 scalar pointers:
  * ``act_scale``  — from calibration (``flash_rt.models.groot_n17.calibration``
    is pure torch / hardware-independent; the AMD frontend imports it
    directly, along with ``mrope_table``).
  * ``w_scale``    — the weight scale baked at load time, uploaded to a
    device fp32 scalar by the frontend.

Stages mirror ``pipeline_rtx_fp8.py``: ``qwen3vl_vit_forward`` →
``deepstack_merge_forward`` → ``qwen3vl_llm_forward`` → ``vlln_forward`` →
``vl_self_attn_forward``. Each forward is pointer-only (every device tensor
is an int ``data_ptr`` supplied by the frontend; no allocation, no
host↔device traffic; streams are raw ints).
"""

from __future__ import annotations

import os


# ─────────────────────────────────────────────────────────────────────────
# Stage 4: VLLN — LayerNorm on backbone_features (no FP8; identical to RTX)
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
                        deepstack_capture=None,
                        fused_epilogue: bool = False,
                        alphas=None) -> None:
    """24-layer Qwen3-VL ViT, FP8 GEMMs via decomposed descale.

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

    fused_epilogue: replace every descale-GEMM + add_bias (+gelu) pair with
        a single hipBLASLt fused-epilogue GEMM — ``fp8_nn_bias`` for the
        Q/K/V/o/fc2 sites, ``fp8_nn_gelu_bias`` for fc1. Same buffers, no
        extra scratch; the quantize front-end is unchanged. NUMERICS: the
        epilogue adds bias on the FP32 accumulator BEFORE the fp16 round
        (decomposed adds it after) — judged by the E2E gate, not bit-parity.
    alphas: required iff ``fused_epilogue`` — per-site per-layer HOST float
        lists, keys parallel to ``scales_dev``: act_qkv (one alpha for
        q/k/v — shared fused-qkv weight scale), act_o, act_fc1, act_fc2;
        each alpha = act_scale × w_scale.
    """
    if fused_epilogue and alphas is None:
        raise ValueError(
            "qwen3vl_vit_forward: fused_epilogue=True requires host `alphas` "
            "(act_qkv/act_o/act_fc1/act_fc2 per-layer float lists)")
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

        # ── Pre-attn LayerNorm (+ quantize for Q/K/V) ──
        if fused_epilogue:
            # AMD FUSED: LayerNorm + static FP8 quantize in ONE kernel.
            # The kernel rounds the norm output through fp16 BEFORE the
            # fp8 quantize, bit-matching the decomposed pair; the normed
            # fp16 buffer (xn) is consumed only by the quantize here, so
            # its write is dropped entirely. rc != 0 (dim%8 / alignment)
            # falls back to the decomposed pair.
            rc = fvk.layer_norm_fp8_static_fp16_vec(
                h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
                xn_fp8_ptr, a_qkv, S, D, 1e-6, int(stream))
            if rc != 0:
                fvk.layer_norm_fp16(
                    h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
                    xn_ptr, S, D, 1e-6, int(stream))
                fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_qkv,
                                             S * D, int(stream))
        else:
            fvk.layer_norm_fp16(
                h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
                xn_ptr, S, D, 1e-6, int(stream))
            # ── Quantize xn once for Q/K/V ──
            fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_qkv, S * D, int(stream))

        # ── 3 split FP8 GEMMs + bias ──
        if fused_epilogue:
            # FUSED: bias in the hipBLASLt epilogue, host alpha = act·w scale.
            al_qkv = float(alphas["act_qkv"][li])
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr,
                             int(weights["q_b"][li]), S, D, D, al_qkv, int(stream))
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["k_w"][li]), K_ptr,
                             int(weights["k_b"][li]), S, D, D, al_qkv, int(stream))
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["v_w"][li]), V_ptr,
                             int(weights["v_b"][li]), S, D, D, al_qkv, int(stream))
        else:
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
        if fused_epilogue:
            # FUSED: o-proj bias in epilogue.
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out,
                             int(weights["o_b"][li]), S, D, D,
                             float(alphas["act_o"][li]), int(stream))
        else:
            gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out,
                                  S, D, D, a_o, int(weights["o_ws"][li]), int(stream))
            fvk.add_bias_fp16(o_proj_out, int(weights["o_b"][li]), S, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, S * D, int(stream))

        # ── Pre-FF LayerNorm (+ quantize) then FF: D → FF (GELU) → D ──
        if fused_epilogue:
            # AMD FUSED: LayerNorm + static FP8 quantize in ONE kernel
            # (bit-matching; xn write dropped; rc != 0 falls back).
            rc = fvk.layer_norm_fp8_static_fp16_vec(
                h_ptr, int(weights["norm2_w"][li]), int(weights["norm2_b"][li]),
                xn_fp8_ptr, a_fc1, S, D, 1e-6, int(stream))
            if rc != 0:
                fvk.layer_norm_fp16(
                    h_ptr, int(weights["norm2_w"][li]), int(weights["norm2_b"][li]),
                    xn_ptr, S, D, 1e-6, int(stream))
                fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_fc1,
                                             S * D, int(stream))
        else:
            fvk.layer_norm_fp16(
                h_ptr, int(weights["norm2_w"][li]), int(weights["norm2_b"][li]),
                xn_ptr, S, D, 1e-6, int(stream))
            fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_fc1, S * D, int(stream))
        if fused_epilogue:
            # FUSED: fc1 bias+GELU in epilogue; fc2 bias in epilogue.
            gemm.fp8_nn_gelu_bias(xn_fp8_ptr, int(weights["fc1_w"][li]),
                                  fc1_out_ptr, int(weights["fc1_b"][li]),
                                  S, FF, D, float(alphas["act_fc1"][li]),
                                  int(stream))
            fvk.quantize_fp8_static_fp16(fc1_out_ptr, fc1_fp8_ptr, a_fc2,
                                         S * FF, int(stream))
            gemm.fp8_nn_bias(fc1_fp8_ptr, int(weights["fc2_w"][li]), o_proj_out,
                             int(weights["fc2_b"][li]), S, D, FF,
                             float(alphas["act_fc2"][li]), int(stream))
        else:
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
                            scales_dev, *, attn=None, stream: int = 0,
                            fused_epilogue: bool = False,
                            alphas=None) -> None:
    """3 DeepStack mergers (taps ViT [5, 11, 17]) → 3 features for LLM [0,1,2].

    Per merger j: LayerNorm(4096) → quantize → FP8 fc1 (+bias, +GELU tanh) →
    quantize → FP8 fc2 (+bias) → (Nout, 2048).

    bufs:       in (list[3]), ln_out, fp8_scratch, fc1_out, out (list[3])
    weights:    norm_w/b[j]; fc1/fc2_w[j] (fp8) + fc1/fc2_b[j]; fc1/fc2_ws[j]
    scales_dev: act_fc1[j], act_fc2[j]
    dims:       Nin, Din, Nout, Dmid, Dout

    fused_epilogue: fold bias(+GELU) into the hipBLASLt epilogue —
        ``fp8_nn_gelu_bias`` for fc1, ``fp8_nn_bias`` for fc2. Same buffers.
        NUMERICS: bias is added on the FP32 accumulator BEFORE the fp16
        round (decomposed adds it after) — judged by the E2E gate.
    alphas: required iff ``fused_epilogue`` — HOST float lists keyed
        act_fc1 / act_fc2 (per merger; alpha = act_scale × w_scale).
    """
    if fused_epilogue and alphas is None:
        raise ValueError(
            "deepstack_merge_forward: fused_epilogue=True requires host "
            "`alphas` (act_fc1/act_fc2 per-merger float lists)")
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

        if fused_epilogue:
            # AMD FUSED: LayerNorm + static FP8 quantize in ONE kernel.
            # Bit-matches the decomposed pair (fp16 round-through before
            # the quantize); ln_out is consumed only by the quantize, so
            # its write is dropped. rc != 0 falls back to the pair.
            rc = fvk.layer_norm_fp8_static_fp16_vec(
                in_ptr, int(weights["norm_w"][j]), int(weights["norm_b"][j]),
                fp8_scratch, a_fc1, Nout, Dmid, 1e-6, int(stream))
            if rc != 0:
                fvk.layer_norm_fp16(
                    in_ptr, int(weights["norm_w"][j]), int(weights["norm_b"][j]),
                    ln_out, Nout, Dmid, 1e-6, int(stream))
                fvk.quantize_fp8_static_fp16(ln_out, fp8_scratch, a_fc1,
                                             Nout * Dmid, int(stream))
        else:
            fvk.layer_norm_fp16(
                in_ptr, int(weights["norm_w"][j]), int(weights["norm_b"][j]),
                ln_out, Nout, Dmid, 1e-6, int(stream))
            fvk.quantize_fp8_static_fp16(ln_out, fp8_scratch, a_fc1, Nout * Dmid, int(stream))
        if fused_epilogue:
            # FUSED: fc1 bias+GELU in epilogue; fc2 bias in epilogue.
            gemm.fp8_nn_gelu_bias(fp8_scratch, int(weights["fc1_w"][j]), fc1_out,
                                  int(weights["fc1_b"][j]), Nout, Dmid, Dmid,
                                  float(alphas["act_fc1"][j]), int(stream))
            fvk.quantize_fp8_static_fp16(fc1_out, fp8_scratch, a_fc2,
                                         Nout * Dmid, int(stream))
            gemm.fp8_nn_bias(fp8_scratch, int(weights["fc2_w"][j]), out_ptr,
                             int(weights["fc2_b"][j]), Nout, Dout, Dmid,
                             float(alphas["act_fc2"][j]), int(stream))
        else:
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
                        layers_subset=None,
                        fused_epilogue: bool = False) -> None:
    """16 truncated Qwen3-VL LLM decoder layers, FP8 GEMMs via decomposed descale.

    Per layer: RMSNorm → quantize → 3 split FP8 Q/K/V descale GEMMs (no bias) →
    per-head q/k RMSNorm → M-RoPE → causal GQA MHA (aiter native — NO K/V head
    expand, see below) → quantize O → FP8 o-proj → residual → RMSNorm →
    quantize → FP8 gate/up → SiLU(gate)*up fused-to-FP8 → FP8 down → residual
    → optional DeepStack inject.

    Q/K/V share the fused-qkv weight scale.

    AMD delta vs pipeline_rtx_fp8 (the ONLY sanctioned computation change):
    the backend's llm K/V slots hold the NATIVE NHKV=8 heads and aiter's
    ``mha_fwd`` handles GQA internally, so the Q/K/V descale GEMMs write
    straight into the backend slots (``attn.get_slot_ptrs("llm", li)``) and
    the RTX ``gpu_repeat_interleave_heads`` expand step is SKIPPED. The RTX
    ``bufs`` keys Q/K/V/K_exp/V_exp are therefore NOT read here (the frontend
    may omit them).

    bufs:       h, xn (fp16 S×D); xn_fp8 (fp8 S×D); o_proj_out (fp16 S×D);
                gate_out, up_out (fp16 S×FF); gu_fp8 (fp8 S×FF)
    weights:    in_ln_w, post_ln_w, q_norm_w, k_norm_w; q/k/v/o_w, gate/up/down_w
                (fp8); q/k/v/o_ws, gate/up/down_ws (weight-scale dev ptrs);
                cos, sin; deepstack_inject (list[16], 0 = none)
    scales_dev: act_qkv, act_o, act_gateup, act_down
    dims:       S, D, NHQ, NHKV, HD, FF

    fused_epilogue: the llm stage is biasless (Qwen3) so the GEMMs always
        stay on descale form — here the flag instead fuses the elementwise
        chains around them (same FVK_AMD_FUSED_EPILOGUE gate as the other
        forwards; no ``alphas`` needed):
          * pre-attn  ``rms_norm_fp16 + quantize`` → ``rms_norm_fp8_fp16``
          * o-proj    ``residual_add_fp16 + rms_norm_fp16 + quantize`` →
            ``residual_add_rms_norm_fp8_fp16`` (h updated in place exactly
            as the decomposed chain: the fp16-rounded residual is written
            back before the norm output is quantized)
        NUMERICS (last-ULP class, judged by the E2E gate like the GEMM
        epilogues): the fused kernels quantize the fp32 normed value
        directly (no fp16 round-through of xn) and use 1/scale without the
        quantize kernel's 1e-12 guard; the residual variant additionally
        accumulates the sum-of-squares from the UNROUNDED fp32 residual
        sums (the decomposed rms_norm re-reads the fp16-rounded residual).
        The per-head q/k norms and the FFN-tail residual (which crosses
        the layer boundary into the next layer's pre-attn norm and may be
        followed by a DeepStack inject) are NOT substituted.
    """
    S    = int(dims["S"])
    D    = int(dims["D"])
    NHQ  = int(dims["NHQ"])
    NHKV = int(dims["NHKV"])
    HD   = int(dims["HD"])
    FF   = int(dims["FF"])

    h_ptr      = int(bufs["h"])
    xn_ptr     = int(bufs["xn"])
    xn_fp8_ptr = int(bufs["xn_fp8"])
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
        # AMD: GEMM outputs land directly in the backend slots — Q holds
        # NHQ=16 heads, K/V hold the native NHKV=8 heads (aiter GQA).
        Q_ptr = int(slots["Q"])
        K_ptr = int(slots["K"])
        V_ptr = int(slots["V"])
        O_ptr = int(slots["O"])
        a_qkv = int(scales_dev["act_qkv"][li])
        a_o   = int(scales_dev["act_o"][li])
        a_gu  = int(scales_dev["act_gateup"][li])
        a_dn  = int(scales_dev["act_down"][li])

        # ── Pre-attn RMSNorm + quantize ──
        if fused_epilogue:
            # AMD FUSED: RMSNorm + static FP8 quantize in ONE kernel; xn
            # is consumed only by the quantize here, so its write is
            # dropped. Numeric deltas vs the decomposed pair (no fp16
            # round-through, no 1e-12 scale guard) are last-ULP — see the
            # docstring; judged by the E2E gate.
            fvk.rms_norm_fp8_fp16(h_ptr, int(weights["in_ln_w"][li]),
                                  xn_fp8_ptr, S, D, 1e-6, a_qkv, int(stream))
        else:
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

        # ── (RTX had GQA expand K/V → NHQ heads here — SKIPPED on AMD:
        #     aiter consumes the native 8 KV heads.) ──

        # ── Causal MHA via attn backend ──
        attn.run("llm", li, q_seq=S, kv_seq=S, stream=int(stream))

        # ── O projection (FP8) ──
        fvk.quantize_fp8_static_fp16(O_ptr, xn_fp8_ptr, a_o,
                                     S * NHQ * HD, int(stream))
        gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["o_w"][li]), o_out_ptr,
                              S, D, NHQ * HD, a_o, int(weights["o_ws"][li]), int(stream))

        # ── o-proj residual + Pre-FFN RMSNorm + quantize ──
        if fused_epilogue:
            # AMD FUSED: residual += o_proj, RMSNorm(residual), FP8
            # quantize in ONE kernel. Same op sequence as the decomposed
            # chain: h is updated in place (fp16-rounded residual written
            # back) and o_out is not read again. Numeric deltas (unrounded
            # sum-of-squares, no fp16 round-through, no 1e-12 guard) are
            # last-ULP — see the docstring; judged by the E2E gate.
            fvk.residual_add_rms_norm_fp8_fp16(
                h_ptr, o_out_ptr, int(weights["post_ln_w"][li]), xn_fp8_ptr,
                S, D, 1e-6, a_gu, int(stream))
        else:
            fvk.residual_add_fp16(h_ptr, o_out_ptr, S * D, int(stream))
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
                         layers_subset=None,
                         fused_epilogue: bool = False,
                         alphas=None) -> None:
    """4-layer SelfAttentionTransformer, FP8 GEMMs via decomposed descale.

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

    fused_epilogue: fold bias(+GELU for fc1) into the hipBLASLt epilogue —
        ``fp8_nn_bias`` for Q/K/V/o/fc2, ``fp8_nn_gelu_bias`` for fc1. Same
        buffers, no extra scratch. NUMERICS: bias added on the FP32
        accumulator BEFORE the fp16 round (decomposed adds it after) —
        judged by the E2E gate, not bit-parity.
    alphas: required iff ``fused_epilogue`` — HOST floats, keys parallel to
        ``scales_dev``: act_qkv is a per-layer list of ``(a_q, a_k, a_v)``
        3-tuples (Q/K/V have SEPARATE weight scales here, so one shared
        float cannot cover them); act_o / act_fc1 / act_fc2 are per-layer
        float lists. Each alpha = act_scale × w_scale.
    """
    if fused_epilogue and alphas is None:
        raise ValueError(
            "vl_self_attn_forward: fused_epilogue=True requires host `alphas` "
            "(act_qkv 3-tuples + act_o/act_fc1/act_fc2 per-layer float lists)")
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
        if fused_epilogue:
            # AMD FUSED: LayerNorm + static FP8 quantize in ONE kernel
            # (bit-matching fp16 round-through; xn consumed only by the
            # quantize, its write dropped; rc != 0 falls back).
            rc = fvk.layer_norm_fp8_static_fp16_vec(
                h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
                xn_fp8_ptr, a_qkv, T, D, 1e-5, int(stream))
            if rc != 0:
                fvk.layer_norm_fp16(
                    h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
                    xn_ptr, T, D, 1e-5, int(stream))
                fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_qkv,
                                             T * D, int(stream))
        else:
            fvk.layer_norm_fp16(
                h_ptr, int(weights["norm1_w"][li]), int(weights["norm1_b"][li]),
                xn_ptr, T, D, 1e-5, int(stream))
            fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_qkv, T * D, int(stream))

        # ── Q / K / V FP8 GEMMs + bias (separate weight scales) ──
        if fused_epilogue:
            # FUSED: per-projection alphas (separate weight scales).
            al_q, al_k, al_v = alphas["act_qkv"][li]
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["q_w"][li]), Q_ptr,
                             int(weights["q_b"][li]), T, D, D,
                             float(al_q), int(stream))
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["k_w"][li]), K_ptr,
                             int(weights["k_b"][li]), T, D, D,
                             float(al_k), int(stream))
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["v_w"][li]), V_ptr,
                             int(weights["v_b"][li]), T, D, D,
                             float(al_v), int(stream))
        else:
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
        if fused_epilogue:
            # FUSED: o-proj bias in epilogue (vlsa).
            gemm.fp8_nn_bias(xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out,
                             int(weights["o_b"][li]), T, D, D,
                             float(alphas["act_o"][li]), int(stream))
        else:
            gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["o_w"][li]), o_proj_out,
                                  T, D, D, a_o, int(weights["o_ws"][li]), int(stream))
            fvk.add_bias_fp16(o_proj_out, int(weights["o_b"][li]), T, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, T * D, int(stream))

        # ── Pre-FF LayerNorm + FF (GELU) ──
        if fused_epilogue:
            # AMD FUSED: LayerNorm + static FP8 quantize in ONE kernel
            # (bit-matching; xn write dropped; rc != 0 falls back).
            rc = fvk.layer_norm_fp8_static_fp16_vec(
                h_ptr, int(weights["norm3_w"][li]), int(weights["norm3_b"][li]),
                xn_fp8_ptr, a_fc1, T, D, 1e-5, int(stream))
            if rc != 0:
                fvk.layer_norm_fp16(
                    h_ptr, int(weights["norm3_w"][li]), int(weights["norm3_b"][li]),
                    xn_ptr, T, D, 1e-5, int(stream))
                fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_fc1,
                                             T * D, int(stream))
        else:
            fvk.layer_norm_fp16(
                h_ptr, int(weights["norm3_w"][li]), int(weights["norm3_b"][li]),
                xn_ptr, T, D, 1e-5, int(stream))
            fvk.quantize_fp8_static_fp16(xn_ptr, xn_fp8_ptr, a_fc1, T * D, int(stream))
        if fused_epilogue:
            # FUSED: fc1 bias+GELU in epilogue; fc2 bias in epilogue.
            gemm.fp8_nn_gelu_bias(xn_fp8_ptr, int(weights["fc1_w"][li]),
                                  fc1_out_ptr, int(weights["fc1_b"][li]),
                                  T, FF, D, float(alphas["act_fc1"][li]),
                                  int(stream))
            fvk.quantize_fp8_static_fp16(fc1_out_ptr, fc1_fp8_ptr, a_fc2,
                                         T * FF, int(stream))
            gemm.fp8_nn_bias(fc1_fp8_ptr, int(weights["fc2_w"][li]), o_proj_out,
                             int(weights["fc2_b"][li]), T, D, FF,
                             float(alphas["act_fc2"][li]), int(stream))
        else:
            gemm.fp8_descale_fp16(xn_fp8_ptr, int(weights["fc1_w"][li]), fc1_out_ptr,
                                  T, FF, D, a_fc1, int(weights["fc1_ws"][li]), int(stream))
            fvk.add_bias_fp16(fc1_out_ptr, int(weights["fc1_b"][li]), T, FF, int(stream))
            fvk.gelu_inplace_fp16(fc1_out_ptr, T * FF, int(stream))
            fvk.quantize_fp8_static_fp16(fc1_out_ptr, fc1_fp8_ptr, a_fc2, T * FF, int(stream))
            gemm.fp8_descale_fp16(fc1_fp8_ptr, int(weights["fc2_w"][li]), o_proj_out,
                                  T, D, FF, a_fc2, int(weights["fc2_ws"][li]), int(stream))
            fvk.add_bias_fp16(o_proj_out, int(weights["fc2_b"][li]), T, D, int(stream))
        fvk.residual_add_fp16(h_ptr, o_proj_out, T * D, int(stream))


# ─────────────────────────────────────────────────────────────────────────
# DiT (bf16) — AMD FUSED-EPILOGUE variant of pipeline_thor.dit_forward
# ─────────────────────────────────────────────────────────────────────────


def dit_forward(gemm, fvk, bufs, weights, dims,
                *, attn, stream: int = 0, layers_subset=None,
                fvk_fp4=None) -> None:
    """AMD copy of :func:`flash_rt.models.groot_n17.pipeline_thor.dit_forward`
    with the bf16 GEMM+bias pairs fused into hipBLASLt epilogues.

    Same signature, same buffers, same order. Two changes vs the Thor
    source, both marked ``# AMD FUSED``:

      * every ``gemm.bf16_nn`` followed by ``fvk.add_bias_bf16`` on the
        SAME output becomes one ``gemm.bf16_nn_bias`` (and
        ``bf16_nn_bias_gelu`` where the pair is followed by
        ``gelu_inplace`` on that output) — Q, K, V (self), FFN up (+GELU);
      * where that biased GEMM's output is then only ``residual_add``-ed
        into ``h`` (O proj, FFN down), the pair collapses further into
        ``gemm.bf16_nn_bias_res`` writing straight into ``h`` (hipBLASLt
        beta=1 residual accumulate; the ``o_proj_out`` intermediate write
        and the res_add launch are dropped).

    NUMERICS: the epilogue adds bias — and, for the ``_res`` sites, the
    residual — on the FP32 accumulator BEFORE the bf16 round (the
    decomposed forms round first) — judged by the E2E gate, not
    bit-parity. The FP8/FP4 branch bodies are kept byte-identical for
    fidelity (dead on AMD: ``_DIT_USE_FP8=False`` and no fp4 weights are
    supplied); their shared trailing FFN ``residual_add`` remains as a
    separate launch reachable only from those branches. See the
    pipeline_thor docstring for the full per-layer contract.
    """
    Sa = int(dims["Sa"])
    D = int(dims["D"])
    FF = int(dims["FF"])
    Skv_text = int(dims.get("Skv_text", 0))
    Skv_image = int(dims.get("Skv_image", 0))

    h_ptr      = int(bufs["h"])
    xn_ptr     = int(bufs["xn"])
    o_out_ptr  = int(bufs["o_proj_out"])
    ff_out_ptr = int(bufs["ff_proj_out"])

    layer_iter = range(32) if layers_subset is None else list(layers_subset)

    # ── Small-M MFMA packed-GEMM routing (measured list, the pi05
    # FVK_AMD_DEC_GEMM precedent) ────────────────────────────────────
    # The gfx950 packed bf16 kernel (csrc/amd/gemm/smallm_mfma_bf16.h)
    # wins ONLY on the square (41, 1536, 1536) projection shape
    # (standalone gate vs autotuned hipBLASLt: bias 1.41x, bias_res
    # 1.12x); it LOSES on ff1 (41,6144,1536) and ff2 (41,1536,6144),
    # which stay on hipBLASLt. Routed sites: Q (32 layers), K/V (16
    # self layers), O (32 layers) — each only when the frontend supplied
    # an MFMA-packed copy of that weight in weights["smallm_packed"].
    # FVK_AMD_DIT_GEMM: "smallm" (default) enables the routing,
    # "hipblaslt" keeps every site on the library path. Env read at
    # graph-build time only — this function never runs post-capture.
    smallm_packed = weights.get("smallm_packed") or {}
    if os.environ.get("FVK_AMD_DIT_GEMM", "smallm").strip().lower() \
            != "smallm":
        smallm_packed = {}

    def _dd_proj_bias(wkey: str, li: int, out_ptr: int, s: int) -> None:
        """(Sa, D, D) bias-epilogue projection (Q/K/V, input xn).

        Variant 3 = w4_split, hardcoded per the standalone gate: the
        measured bias-epilogue winner at (41, 1536, 1536), 1.41x vs
        autotuned hipBLASLt (auto's heuristic would not pick it here).
        """
        wp = smallm_packed.get(f"{wkey}_w_{li}")
        if wp is not None:
            fvk.smallm_mfma_bf16_nn_bias(
                xn_ptr, int(wp), int(weights[f"{wkey}_b"][li]), out_ptr,
                Sa, D, D, 3, s)
        else:
            gemm.bf16_nn_bias(xn_ptr, int(weights[f"{wkey}_w"][li]),
                              out_ptr, int(weights[f"{wkey}_b"][li]),
                              Sa, D, D, s)

    # NVFP4 fast path: every DiT GEMM (fused QKV / cross-Q / O / FFN up /
    # FFN down) runs as a block-scaled NVFP4 GEMM with a fused bf16 bias
    # epilogue. At M=Sa=41 the DiT is weight-bandwidth-bound, so halving
    # the weight bytes is the dominant win; the fused epilogues (bias,
    # bias+residual, bias+GELU+fp4out) and the fused norm->fp4 front-ends
    # additionally remove most of the per-layer elementwise launches.
    use_fp4 = fvk_fp4 is not None and "ff_proj_w_fp4" in weights

    def _ck(rc, what, li):
        if rc != 0:
            raise RuntimeError(
                f"N1.7 DiT FP4 {what} layer {li} failed rc={rc}")

    for li in layer_iter:
        is_self = (li % 2 == 1)
        # Backend's ``dit_self`` and ``dit_cross`` sites are indexed
        # cross-only / self-only (16 entries each, NOT the full 0..31
        # layer index). Map here.
        j_attn = (li - 1) // 2 if is_self else li // 2

        if use_fp4:
            xn_fp4, xn_sfa = int(bufs["xn_fp4"]), int(bufs["xn_sfa"])
            slots = attn.get_slot_ptrs("dit_self" if is_self else "dit_cross",
                                       j_attn)
            Q_ptr, K_ptr, V_ptr, O_ptr = (slots["Q"], slots["K"], slots["V"],
                                          slots["O"])
            # AdaLN-modulated norm1 -> fp4 + SFA (one fused kernel).
            rc = fvk_fp4.ada_layer_norm_fp4_sfa_bf16(
                h_ptr, int(weights["scale_msa"][li]),
                int(weights["shift_msa"][li]),
                xn_fp4, xn_sfa, Sa, D, 1e-5, int(stream))
            _ck(rc, "adaln", li)
            if is_self:
                j = (li - 1) // 2
                rc = fvk_fp4.cutlass_fp4_gemm_bias_bf16(
                    xn_fp4, xn_sfa,
                    int(weights["qkv_w_fp4"][j]), int(weights["qkv_sfb"][j]),
                    int(weights["qkv_b_fp4"][j]), int(bufs["qkv_buf"]),
                    Sa, 3 * D, D, int(stream))
                _ck(rc, "qkv", li)
                if not bufs.get("qkv_strided"):
                    # The self-attn slots normally alias the fused QKV
                    # output (token stride 3D); split into packed per-slot
                    # buffers only when they do not.
                    fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), Q_ptr, Sa, D, 3 * D, 0, int(stream))
                    fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), K_ptr, Sa, D, 3 * D, D, int(stream))
                    fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), V_ptr, Sa, D, 3 * D, 2 * D, int(stream))
                attn.run("dit_self", j_attn, q_seq=Sa, kv_seq=Sa,
                         stream=int(stream))
            else:
                rc = fvk_fp4.cutlass_fp4_gemm_bias_bf16(
                    xn_fp4, xn_sfa,
                    int(weights["q_w_fp4"][j_attn]),
                    int(weights["q_sfb"][j_attn]),
                    int(weights["q_b"][li]), Q_ptr,
                    Sa, D, D, int(stream))
                _ck(rc, "q", li)
                target_text = (li % 4 == 0)
                kv_seq = Skv_text if target_text else Skv_image
                attn.run("dit_cross", j_attn, q_seq=Sa, kv_seq=kv_seq,
                         stream=int(stream))
            # O projection: quantize the attention output, then fused
            # bias + residual straight into h.
            rc = fvk_fp4.quantize_fp4_dynamic_sfa_bf16_vec(
                O_ptr, int(bufs["octx_fp4"]), int(bufs["octx_sfa"]),
                Sa, D, False, int(stream))
            _ck(rc, "o-quant", li)
            rc = fvk_fp4.cutlass_fp4_gemm_bias_res_bf16(
                int(bufs["octx_fp4"]), int(bufs["octx_sfa"]),
                int(weights["o_w_fp4"][li]), int(weights["o_sfb"][li]),
                int(weights["o_b"][li]), h_ptr, h_ptr,
                Sa, D, D, int(stream))
            _ck(rc, "o", li)
            # FFN: LN -> fp4, up GEMM with fused bias+GELU+fp4out, down
            # GEMM with fused bias+residual into h.
            rc = fvk_fp4.layer_norm_no_affine_fp4_sfa_bf16(
                h_ptr, xn_fp4, xn_sfa, Sa, D, 1e-5, int(stream))
            _ck(rc, "ffn-ln", li)
            rc = fvk_fp4.cutlass_fp4_gemm_bias_gelu_fp4out_bf16(
                xn_fp4, xn_sfa,
                int(weights["ff_proj_w_fp4"][li]),
                int(weights["ff_proj_sfb"][li]),
                int(weights["ff_proj_b"][li]),
                int(bufs["hid_fp4"]), int(bufs["hid_sfa"]),
                Sa, FF, D, int(stream))
            _ck(rc, "ffn-up", li)
            rc = fvk_fp4.cutlass_fp4_gemm_bias_res_bf16(
                int(bufs["hid_fp4"]), int(bufs["hid_sfa"]),
                int(weights["ff_down_w_fp4"][li]),
                int(weights["ff_down_sfb"][li]),
                int(weights["ff_down_b"][li]), h_ptr, h_ptr,
                Sa, D, FF, int(stream))
            _ck(rc, "ffn-down", li)
            continue

        # ── AdaLN modulated norm1 ─────────────────────────────────────
        # For self-attn FP8 QKV, fuse the AdaLN and the FP8 quantize into one
        # kernel — the AdaLN output feeds only the QKV projection, so it can
        # be emitted directly as fp8 (one kernel instead of AdaLN + quantize).
        # Cross-attn and the bf16 fallback keep the bf16 AdaLN.
        is_self_fp8_sm120 = is_self and "qkv_w_fp8_nt" in weights
        is_self_fp8 = is_self and ("qkv_w_fp8" in weights or is_self_fp8_sm120)
        j_self = (li - 1) // 2
        if is_self_fp8:
            fvk.ada_layer_norm_fp8(
                h_ptr, int(weights["scale_msa"][li]), int(weights["shift_msa"][li]),
                int(bufs["qkv_xn_fp8"]), int(weights["act_qkv_scale"][j_self]),
                Sa, D, 1e-5, int(stream),
            )
        else:
            fvk.ada_layer_norm_bf16(
                h_ptr,
                int(weights["scale_msa"][li]), int(weights["shift_msa"][li]),
                xn_ptr, Sa, D, 1e-5, int(stream),
            )

        # ── attention projections ─────────────────────────────────────
        if is_self:
            slots = attn.get_slot_ptrs("dit_self", j_attn)
        else:
            slots = attn.get_slot_ptrs("dit_cross", j_attn)
        Q_ptr, K_ptr, V_ptr, O_ptr = slots["Q"], slots["K"], slots["V"], slots["O"]

        # AMD delta vs pipeline_thor: hipBLASLt on gfx950 supports the bias
        # (and bias+GELU) epilogue at M=Sa=41 — parity-validated GemmRunner
        # entry points — so the bf16_nn + add_bias_bf16 pairs below run as
        # single fused-epilogue GEMMs (the cuBLASLt M-alignment limitation
        # that forced the decomposed form on Thor does not apply here).
        if is_self_fp8_sm120:
            gemm.fp8_nt_dev(
                int(bufs["qkv_xn_fp8"]), int(weights["qkv_w_fp8_nt"][j_self]),
                int(bufs["qkv_buf"]), Sa, 3 * D, D,
                int(weights["act_qkv_scale"][j_self]),
                int(weights["qkv_weight_scale"][j_self]), int(stream))
            fvk.add_bias_bf16(
                int(bufs["qkv_buf"]), int(weights["qkv_b"][j_self]),
                Sa, 3 * D, int(stream))
            fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), Q_ptr, Sa, D, 3 * D, 0, int(stream))
            fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), K_ptr, Sa, D, 3 * D, D, int(stream))
            fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), V_ptr, Sa, D, 3 * D, 2 * D, int(stream))
            attn.run("dit_self", j_attn, q_seq=Sa, kv_seq=Sa, stream=int(stream))
        elif is_self_fp8:
            # Fused FP8 QKV (self-attn): q/k/v share the post-AdaLN input,
            # so one [D, 3D] GEMM (compute-bound, unlike 3 launch-bound D→D
            # GEMMs) + a strided split into the Q/K/V slots. Cross-attn keeps
            # a single Q GEMM (K/V come from the backbone-projected cross-KV).
            qkv_fp8_layout = str(weights.get("qkv_fp8_layout", "kn"))
            if qkv_fp8_layout == "nk":
                gemm.fp8_nt_dev(
                    int(bufs["qkv_xn_fp8"]), int(weights["qkv_w_fp8"][j_self]),
                    int(bufs["qkv_buf"]), Sa, 3 * D, D,
                    int(weights["act_qkv_scale"][j_self]),
                    int(weights["w_qkv_scale"][j_self]), int(stream))
            else:
                gemm.fp8_nn_dev(
                    int(bufs["qkv_xn_fp8"]), int(weights["qkv_w_fp8"][j_self]),
                    int(bufs["qkv_buf"]), Sa, 3 * D, D,
                    int(weights["act_qkv_scale"][j_self]),
                    int(weights["w_qkv_scale"][j_self]), int(stream))
            fvk.add_bias_bf16(
                int(bufs["qkv_buf"]), int(weights["qkv_b"][j_self]),
                Sa, 3 * D, int(stream))
            fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), Q_ptr, Sa, D, 3 * D, 0, int(stream))
            fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), K_ptr, Sa, D, 3 * D, D, int(stream))
            fvk.gpu_strided_copy_fp16(int(bufs["qkv_buf"]), V_ptr, Sa, D, 3 * D, 2 * D, int(stream))
            attn.run("dit_self", j_attn, q_seq=Sa, kv_seq=Sa, stream=int(stream))
        else:
            # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (Q);
            # smallm-routed when a packed q_w copy exists (see
            # _dd_proj_bias / FVK_AMD_DIT_GEMM above).
            _dd_proj_bias("q", li, Q_ptr, int(stream))
            if is_self:
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (K).
                _dd_proj_bias("k", li, K_ptr, int(stream))
                # AMD FUSED: bf16_nn + add_bias_bf16 → bf16_nn_bias (V).
                _dd_proj_bias("v", li, V_ptr, int(stream))
                attn.run("dit_self", j_attn, q_seq=Sa, kv_seq=Sa, stream=int(stream))
            else:
                target_text = (li % 4 == 0)
                kv_seq = Skv_text if target_text else Skv_image
                attn.run("dit_cross", j_attn, q_seq=Sa, kv_seq=kv_seq,
                         stream=int(stream))

        # AMD FUSED: bf16_nn_bias + residual_add → bf16_nn_bias_res writing
        # straight into h (beta=1 residual on the FP32 accumulator; the
        # o_proj_out intermediate write and the res_add launch are both
        # dropped — o_proj_out was consumed only by the res_add here).
        # NUMERICS: the residual is added BEFORE the single bf16 round
        # (decomposed rounds the GEMM output first) — same class as the
        # bias epilogues, judged by the E2E gate.
        _o_wp = smallm_packed.get(f"o_w_{li}")
        if _o_wp is not None:
            # smallm-routed O proj (variant 4 = w8_split, hardcoded per
            # the standalone gate: measured bias_res winner at
            # (41, 1536, 1536), 1.12x vs autotuned hipBLASLt). Same
            # D += acc + bias semantics as bf16_nn_bias_res (beta=1).
            fvk.smallm_mfma_bf16_nn_bias_res(
                O_ptr, int(_o_wp), int(weights["o_b"][li]), h_ptr,
                Sa, D, D, 4, int(stream))
        else:
            gemm.bf16_nn_bias_res(O_ptr, int(weights["o_w"][li]),
                                  h_ptr, int(weights["o_b"][li]),
                                  Sa, D, D, int(stream))

        # ── Pre-FF LayerNorm (no affine — DiT default) ───────────────
        fvk.layer_norm_no_affine_bf16(
            h_ptr, xn_ptr, Sa, D, 1e-5, int(stream),
        )

        # ── FFN: GELU(tanh-approx) ────────────────────────────────────
        # The FFN GEMMs are the compute-bound part of the (M=41) DiT, so an
        # FP8 path here is a real win (≈1.8× on the up-projection) and fuses
        # the bias+GELU into the GEMM epilogue. Activated when calibrated FP8
        # FFN weights/scales are supplied; otherwise the bf16 path runs. The
        # attention GEMMs stay bf16 — at M=41 they are launch-bound, so FP8
        # gives no speedup there.
        if "ff_proj_w_fp8_sm120" in weights:
            fvk.quantize_fp8_static(
                xn_ptr, int(bufs["xn_fp8"]),
                int(weights["act_fc1_scale"][li]), Sa * D, int(stream))
            gemm.fp8_descale_fp16(
                int(bufs["xn_fp8"]), int(weights["ff_proj_w_fp8_sm120"][li]),
                int(bufs["ff_fp16"]), Sa, FF, D,
                int(weights["act_fc1_scale"][li]),
                int(weights["ff_proj_weight_scale"][li]), int(stream))
            fvk.add_bias_fp16(
                int(bufs["ff_fp16"]), int(weights["ff_proj_b"][li]),
                Sa, FF, int(stream))
            fvk.gelu_inplace_fp16(int(bufs["ff_fp16"]), Sa * FF, int(stream))
            fvk.quantize_fp8_static_fp16(
                int(bufs["ff_fp16"]), int(bufs["ff_fp8"]),
                int(weights["act_fc2_scale"][li]), Sa * FF, int(stream))
            gemm.fp8_nt_dev(
                int(bufs["ff_fp8"]), int(weights["ff_down_w_fp8_nt"][li]),
                o_out_ptr, Sa, D, FF,
                int(weights["act_fc2_scale"][li]),
                int(weights["ff_down_weight_scale"][li]), int(stream))
            fvk.add_bias_bf16(o_out_ptr, int(weights["ff_down_b"][li]),
                              Sa, D, int(stream))
        elif "ff_proj_w_fp8" in weights:
            ff_fp8_layout = str(weights.get("ff_fp8_layout", "kn"))
            fvk.quantize_fp8_static(
                xn_ptr, int(bufs["xn_fp8"]),
                int(weights["act_fc1_scale"][li]), Sa * D, int(stream))
            if ff_fp8_layout == "nk":
                gemm.fp8_nt_dev(
                    int(bufs["xn_fp8"]), int(weights["ff_proj_w_fp8"][li]),
                    ff_out_ptr, Sa, FF, D,
                    int(weights["act_fc1_scale"][li]),
                    int(weights["w_fc1_scale"][li]), int(stream))
            else:
                gemm.fp8_nn_dev(
                    int(bufs["xn_fp8"]), int(weights["ff_proj_w_fp8"][li]),
                    ff_out_ptr, Sa, FF, D,
                    int(weights["act_fc1_scale"][li]),
                    int(weights["w_fc1_scale"][li]), int(stream))
            fvk.bias_gelu_quantize_fp8_static_bf16(
                ff_out_ptr, int(weights["ff_proj_b"][li]),
                int(bufs["ff_fp8"]), int(weights["act_fc2_scale"][li]),
                Sa, FF, int(stream))
            if ff_fp8_layout == "nk":
                gemm.fp8_nt_dev(
                    int(bufs["ff_fp8"]), int(weights["ff_down_w_fp8"][li]),
                    o_out_ptr, Sa, D, FF,
                    int(weights["act_fc2_scale"][li]),
                    int(weights["w_fc2_scale"][li]), int(stream))
            else:
                gemm.fp8_nn_dev(
                    int(bufs["ff_fp8"]), int(weights["ff_down_w_fp8"][li]),
                    o_out_ptr, Sa, D, FF,
                    int(weights["act_fc2_scale"][li]),
                    int(weights["w_fc2_scale"][li]), int(stream))
            fvk.add_bias_bf16(o_out_ptr, int(weights["ff_down_b"][li]),
                              Sa, D, int(stream))
        else:
            # AMD FUSED: bf16_nn + add_bias_bf16 + gelu_inplace →
            # bf16_nn_bias_gelu (FFN up).
            gemm.bf16_nn_bias_gelu(xn_ptr, int(weights["ff_proj_w"][li]),
                                   ff_out_ptr, int(weights["ff_proj_b"][li]),
                                   Sa, FF, D, int(stream))
            # AMD FUSED: bf16_nn_bias + residual_add → bf16_nn_bias_res
            # writing straight into h (FFN down; o_proj_out intermediate
            # and the res_add launch dropped, residual added on the FP32
            # accumulator — judged by the E2E gate).
            gemm.bf16_nn_bias_res(ff_out_ptr, int(weights["ff_down_w"][li]),
                                  h_ptr, int(weights["ff_down_b"][li]),
                                  Sa, D, FF, int(stream))
            continue
        # FP8 FFN branches (dead on AMD, kept for fidelity): the down
        # projection still lands in o_out_ptr, so the residual runs as
        # the original separate launch.
        fvk.residual_add(h_ptr, o_out_ptr, Sa * D, int(stream))

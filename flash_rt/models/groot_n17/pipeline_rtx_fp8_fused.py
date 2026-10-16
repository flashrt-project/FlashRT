"""GR00T N1.7 FP8 backbone forwards for RTX SM120 with fused producers.

Same math as :mod:`pipeline_rtx_fp8` (per-tensor static FP8 GEMMs on an FP16
residual stream, cuBLASLt descale GEMMs), with every bias / residual / norm /
activation / quantize step folded into the kernel that produces the next GEMM
operand (``fvk.fp8_*``, kernels/fp8_prefill_fp16.cuh):

  * a layer's last residual (``h += fc2 + b``) is applied by the next layer's
    norm kernel, which also emits the FP8 operand of the following GEMM;
  * the ViT / LLM Q/K/V come from one fused projection GEMM, and bias,
    per-head q/k RMSNorm and RoPE run in one split kernel that writes the
    attention operands;
  * the LLM uses FA2 causal GQA attention (BF16 operands, no K/V expansion);
  * fc1 bias + GELU + FP8 quantize is one kernel.

Each forward is pointer-only and issues no host synchronization, so the whole
backbone records into one CUDA graph.
"""

from __future__ import annotations


def _ck(rc, what):
    if rc:
        raise RuntimeError(f"{what} failed (status {rc})")


def residual(fvk, x, delta, bias, rows, dim, stream, *, delta2=0, copy=0):
    """x += delta + bias (+ delta2); optionally copy the updated rows."""
    _ck(fvk.fp8_residual_norm_quant(x, delta, bias, delta2, copy, 0, 0, 0, 0.0, 0, 0, 0,
                                    rows, dim, stream), "residual")


def qwen3vl_vit_forward(gemm, fvk, bufs, weights, dims, scales_dev, *, attn,
                        stream: int = 0, layers=range(24), taps=(),
                        tap_hook=None) -> None:
    """Qwen3-VL ViT layers (pre-LN, fused QKV with bias, rotate-half RoPE).

    bufs:    h (fp16 S×D), xn_fp8 (S×D), qkv (fp16 S×3D), o_out (fp16 S×D),
             fc1_out (fp16 S×FF), fc1_fp8 (S×FF)
    weights: norm1/2_w/b, qkv_w (fp8 [D, 3D]), qkv_b (3D), qkv_ws, o_w/o_b/o_ws,
             fc1_w/fc1_b/fc1_ws, fc2_w/fc2_b/fc2_ws (per-layer lists); cos, sin
    tap_hook(j, h_ptr) runs after layer ``taps[j]`` with its residual applied.
    """
    S, D, NH, HD, FF = (int(dims[k]) for k in ("S", "D", "NH", "HD", "ff_inner"))
    Sper = int(dims.get("Sper_view", S))
    h, xn8, qkv = int(bufs["h"]), int(bufs["xn_fp8"]), int(bufs["qkv"])
    o_out, fc1, fc1_8 = int(bufs["o_out"]), int(bufs["fc1_out"]), int(bufs["fc1_fp8"])
    cos, sin = int(weights["cos"]), int(weights["sin"])
    st = int(stream)
    pend = (0, 0)   # (delta, bias) still to be added to h
    for li in layers:
        sl = attn.get_slot_ptrs("vit", li)
        a_qkv, a_o = int(scales_dev["act_qkv"][li]), int(scales_dev["act_o"][li])
        a_fc1, a_fc2 = int(scales_dev["act_fc1"][li]), int(scales_dev["act_fc2"][li])
        _ck(fvk.fp8_residual_norm_quant(h, pend[0], pend[1], 0, 0, int(weights["norm1_w"][li]),
                                        int(weights["norm1_b"][li]), 0, 1e-6, xn8, a_qkv, 0, S, D, st),
            "vit norm1")
        gemm.fp8_descale_fp16(xn8, int(weights["qkv_w"][li]), qkv, S, 3 * D, D, a_qkv,
                              int(weights["qkv_ws"][li]), st)
        qb = int(weights["qkv_b"][li])
        _ck(fvk.fp8_qkv_split(qkv, qkv + 2 * D, qkv + 4 * D, 3 * D, qb, qb + 2 * D, qb + 4 * D,
                              1.0, 1.0, 0, 0, 1e-6, cos, sin, sl["Q"], sl["K"], sl["V"], False,
                              S, NH, NH, HD, st), "vit qkv")
        attn.run("vit", li, q_seq=Sper, kv_seq=Sper, stream=st)
        fvk.quantize_fp8_static_fp16(sl["O"], xn8, a_o, S * D, st)
        gemm.fp8_descale_fp16(xn8, int(weights["o_w"][li]), o_out, S, D, D, a_o,
                              int(weights["o_ws"][li]), st)
        _ck(fvk.fp8_residual_norm_quant(h, o_out, int(weights["o_b"][li]), 0, 0,
                                        int(weights["norm2_w"][li]), int(weights["norm2_b"][li]),
                                        0, 1e-6, xn8, a_fc1, 0, S, D, st), "vit norm2")
        gemm.fp8_descale_fp16(xn8, int(weights["fc1_w"][li]), fc1, S, FF, D, a_fc1,
                              int(weights["fc1_ws"][li]), st)
        _ck(fvk.fp8_bias_gelu_quant(fc1, FF, int(weights["fc1_b"][li]), fc1_8, a_fc2, S, FF, st),
            "vit gelu")
        gemm.fp8_descale_fp16(fc1_8, int(weights["fc2_w"][li]), o_out, S, D, FF, a_fc2,
                              int(weights["fc2_ws"][li]), st)
        pend = (o_out, int(weights["fc2_b"][li]))
        if li in taps:
            residual(fvk, h, pend[0], pend[1], S, D, st)
            pend = (0, 0)
            if tap_hook is not None:
                tap_hook(taps.index(li), h)
    if pend[0]:
        residual(fvk, h, pend[0], pend[1], S, D, st)


def deepstack_merger_forward(gemm, fvk, bufs, weights, dims, scales_dev, j, x_ptr, out_ptr, *,
                             stream: int = 0) -> None:
    """DeepStack merger ``j`` on the ViT rows at ``x_ptr`` viewed as [Nout, Dmid]:
    LayerNorm → FP8 fc1 (+bias, GELU) → FP8 fc2 (+bias) → out [Nout, Dout]."""
    Nout, Dmid, Dout = int(dims["Nout"]), int(dims["Dmid"]), int(dims["Dout"])
    s8, fc1 = int(bufs["fp8_scratch"]), int(bufs["fc1_out"])
    a1, a2 = int(scales_dev["act_fc1"][j]), int(scales_dev["act_fc2"][j])
    st = int(stream)
    _ck(fvk.fp8_residual_norm_quant(int(x_ptr), 0, 0, 0, 0, int(weights["norm_w"][j]),
                                    int(weights["norm_b"][j]), 0, 1e-6, s8, a1, 0, Nout, Dmid, st),
        "deepstack norm")
    gemm.fp8_descale_fp16(s8, int(weights["fc1_w"][j]), fc1, Nout, Dmid, Dmid, a1,
                          int(weights["fc1_ws"][j]), st)
    _ck(fvk.fp8_bias_gelu_quant(fc1, Dmid, int(weights["fc1_b"][j]), s8, a2, Nout, Dmid, st),
        "deepstack gelu")
    gemm.fp8_descale_fp16(s8, int(weights["fc2_w"][j]), int(out_ptr), Nout, Dout, Dmid, a2,
                          int(weights["fc2_ws"][j]), st)
    fvk.add_bias_fp16(int(out_ptr), int(weights["fc2_b"][j]), Nout, Dout, st)


def qwen3vl_llm_forward(gemm, fvk, fa2, bufs, weights, dims, scales_dev, *, stream: int = 0,
                        num_sms: int = 0) -> int:
    """16 truncated Qwen3-VL LLM layers (RMSNorm, fused QKV, q/k norm + M-RoPE,
    causal GQA FA2, SwiGLU, DeepStack injection after layers 0..2).

    bufs:    h (fp16 S×D), xn_fp8, qkv (fp16 S×(NHQ+2NHKV)HD), Q/K/V/O (bf16
             attention operands), lse (fp32), optional lse_accum / o_accum
             (fp32 split-KV scratch for ceil(S/128) splits; the short
             prefill fills the GPU only when split), o_out (fp16 S×D), gate_out,
             up_out (fp16 S×FF), gu_fp8
    weights: in_ln_w, post_ln_w, q_norm_w, k_norm_w, qkv_w/qkv_ws, o_w/o_ws,
             gate_w/gate_ws, up_w/up_ws, down_w/down_ws; cos, sin;
             deepstack_inject (per-layer FP16 [S, D] ptr or 0)
    Returns the pending residual delta pointer (the last down projection),
    which the caller folds into the following norm.
    """
    S, D, NHQ, NHKV, HD, FF = (int(dims[k]) for k in ("S", "D", "NHQ", "NHKV", "HD", "FF"))
    NQKV = (NHQ + 2 * NHKV) * HD
    h, xn8, qkv = int(bufs["h"]), int(bufs["xn_fp8"]), int(bufs["qkv"])
    Q, K, V, O, lse = (int(bufs[k]) for k in ("Q", "K", "V", "O", "lse"))
    lse_acc, o_acc = int(bufs.get("lse_accum", 0)), int(bufs.get("o_accum", 0))
    o_out, gate, up, gu8 = (int(bufs[k]) for k in ("o_out", "gate_out", "up_out", "gu_fp8"))
    cos, sin = int(weights["cos"]), int(weights["sin"])
    inject = weights["deepstack_inject"]
    st = int(stream)
    pend, pend2 = 0, 0
    for li in range(16):
        a_qkv, a_o = int(scales_dev["act_qkv"][li]), int(scales_dev["act_o"][li])
        a_gu, a_dn = int(scales_dev["act_gateup"][li]), int(scales_dev["act_down"][li])
        _ck(fvk.fp8_residual_norm_quant(h, pend, 0, pend2, 0, int(weights["in_ln_w"][li]), 0, 1, 1e-6,
                                        xn8, a_qkv, 0, S, D, st), "llm in_ln")
        gemm.fp8_descale_fp16(xn8, int(weights["qkv_w"][li]), qkv, S, NQKV, D, a_qkv,
                              int(weights["qkv_ws"][li]), st)
        _ck(fvk.fp8_qkv_split(qkv, qkv + 2 * NHQ * HD, qkv + 2 * (NHQ + NHKV) * HD, NQKV, 0, 0, 0,
                              1.0, 1.0, int(weights["q_norm_w"][li]), int(weights["k_norm_w"][li]), 1e-6,
                              cos, sin, Q, K, V, True, S, NHQ, NHKV, HD, st), "llm qkv")
        fa2.fwd_bf16_causal(
            Q=Q, K=K, V=V, O=O, softmax_lse=lse, softmax_lse_accum=lse_acc, o_accum=o_acc,
            batch=1, seqlen_q=S, seqlen_k=S, num_heads_q=NHQ, num_heads_kv=NHKV, head_dim=HD,
            q_strides=(S * NHQ * HD, NHQ * HD, HD), k_strides=(S * NHKV * HD, NHKV * HD, HD),
            v_strides=(S * NHKV * HD, NHKV * HD, HD), o_strides=(S * NHQ * HD, NHQ * HD, HD),
            softmax_scale=1.0 / (HD ** 0.5), num_sms=num_sms, stream=st)
        fvk.quantize_fp8_static(O, xn8, a_o, S * NHQ * HD, st)
        gemm.fp8_descale_fp16(xn8, int(weights["o_w"][li]), o_out, S, D, NHQ * HD, a_o,
                              int(weights["o_ws"][li]), st)
        _ck(fvk.fp8_residual_norm_quant(h, o_out, 0, 0, 0, int(weights["post_ln_w"][li]), 0, 1, 1e-6,
                                        xn8, a_gu, 0, S, D, st), "llm post_ln")
        gemm.fp8_descale_fp16(xn8, int(weights["gate_w"][li]), gate, S, FF, D, a_gu,
                              int(weights["gate_ws"][li]), st)
        gemm.fp8_descale_fp16(xn8, int(weights["up_w"][li]), up, S, FF, D, a_gu,
                              int(weights["up_ws"][li]), st)
        fvk.silu_mul_split_fp8_fp16(gate, up, gu8, S * FF, a_dn, st)
        gemm.fp8_descale_fp16(gu8, int(weights["down_w"][li]), o_out, S, D, FF, a_dn,
                              int(weights["down_ws"][li]), st)
        pend = o_out
        pend2 = int(inject[li]) if li < len(inject) else 0
    if pend2:
        residual(fvk, h, 0, 0, S, D, st, delta2=pend2)
    return pend


def vlln_vl_self_attn_forward(gemm, fvk, bufs, weights, dims, scales_dev, *, attn, llm_h, llm_delta,
                              stream: int = 0) -> None:
    """vlln LayerNorm of the LLM output (+ its pending residual) into the VL
    stream, then the 4-layer VL self-attention transformer.

    bufs:    h (fp16 T×D, the VL stream / backbone features), xn_fp8, o_out,
             fc1_out (fp16 T×FF), fc1_fp8
    weights: vlln_w/b; norm1/3_w/b; q/k/v/o_w, q/k/v/o_b, q/k/v/o_ws;
             fc1/fc2_w, fc1/fc2_b, fc1/fc2_ws; k_mul, v_mul unused (separate GEMMs)
    """
    T, D, NH, HD, FF = (int(dims[k]) for k in ("T", "D", "NH", "HD", "ff_inner"))
    h, xn8, o_out = int(bufs["h"]), int(bufs["xn_fp8"]), int(bufs["o_out"])
    fc1, fc1_8 = int(bufs["fc1_out"]), int(bufs["fc1_fp8"])
    st = int(stream)
    _ck(fvk.fp8_residual_norm_quant(int(llm_h), int(llm_delta), 0, 0, 0, int(weights["vlln_w"]),
                                    int(weights["vlln_b"]), 0, 1e-5, 0, 0, h, T, D, st), "vlln")
    pend = (0, 0)
    for li in range(4):
        sl = attn.get_slot_ptrs("vl_self_attn", li)
        a_qkv, a_o = int(scales_dev["act_qkv"][li]), int(scales_dev["act_o"][li])
        a_fc1, a_fc2 = int(scales_dev["act_fc1"][li]), int(scales_dev["act_fc2"][li])
        _ck(fvk.fp8_residual_norm_quant(h, pend[0], pend[1], 0, 0, int(weights["norm1_w"][li]),
                                        int(weights["norm1_b"][li]), 0, 1e-5, xn8, a_qkv, 0, T, D, st),
            "vlsa norm1")
        for n in ("q", "k", "v"):
            gemm.fp8_descale_fp16(xn8, int(weights[f"{n}_w"][li]), sl[n.upper()], T, D, D, a_qkv,
                                  int(weights[f"{n}_ws"][li]), st)
        _ck(fvk.fp8_qkv_split(sl["Q"], sl["K"], sl["V"], D, int(weights["q_b"][li]), int(weights["k_b"][li]),
                              int(weights["v_b"][li]), 1.0, 1.0, 0, 0, 1e-5, 0, 0, sl["Q"], sl["K"], sl["V"],
                              False, T, NH, NH, HD, st), "vlsa qkv bias")
        attn.run("vl_self_attn", li, q_seq=T, kv_seq=T, stream=st)
        fvk.quantize_fp8_static_fp16(sl["O"], xn8, a_o, T * D, st)
        gemm.fp8_descale_fp16(xn8, int(weights["o_w"][li]), o_out, T, D, D, a_o,
                              int(weights["o_ws"][li]), st)
        _ck(fvk.fp8_residual_norm_quant(h, o_out, int(weights["o_b"][li]), 0, 0,
                                        int(weights["norm3_w"][li]), int(weights["norm3_b"][li]),
                                        0, 1e-5, xn8, a_fc1, 0, T, D, st), "vlsa norm3")
        gemm.fp8_descale_fp16(xn8, int(weights["fc1_w"][li]), fc1, T, FF, D, a_fc1,
                              int(weights["fc1_ws"][li]), st)
        _ck(fvk.fp8_bias_gelu_quant(fc1, FF, int(weights["fc1_b"][li]), fc1_8, a_fc2, T, FF, st),
            "vlsa gelu")
        gemm.fp8_descale_fp16(fc1_8, int(weights["fc2_w"][li]), o_out, T, D, FF, a_fc2,
                              int(weights["fc2_ws"][li]), st)
        pend = (o_out, int(weights["fc2_b"][li]))
    residual(fvk, h, pend[0], pend[1], T, D, st)

"""GROOT N1.7 DiT action head on the skinny NVFP4 family (RTX SM120).

The 32-layer ``AlternateVLDiT`` of ``pipeline_thor.dit_forward`` with every
GEMM (fused self-attention QKV, cross-attention Q, output projection, FFN up
and down) as a block-scaled W4A4 skinny GEMM (``nvfp4_skinny_gemm``): each
CTA streams its weight slice before the programmatic-dependent-launch wait
and writes FP32 K-split partials. The consumers that sum those partials
finish the block step in the same launch and emit the next GEMM's NVFP4
operand directly, so no standalone quantize, bias, activation, residual or
copy launch remains:

    AdaLN(h) -> fp4                       (layer 0 only; later layers get it
                                           from the previous layer's consumer)
    QKV / Q GEMM -> attention straight off the partials (bias added on
      load; cross blocks read the frame's cross K/V)
    O GEMM (BF16 attention output quantized on load)
      -> bias + residual into h, pre-FF LayerNorm -> fp4
    FFN up GEMM -> bias + tanh-GELU -> fp4
    FFN down GEMM -> bias + residual into h, then the next layer's AdaLN
      -> fp4 (last layer: the output AdaLN -> BF16 for the output projection)

Activations use NVFP4 with per-16 dynamic block scales and global scale 1
(no calibration); weights carry a per-tensor global scale.

The backbone functions further down are the experimental NVFP4 backbone
(``FLASHRT_GROOT_N17_BB_NVFP4``); the default backbone is the FP8 one.
"""

from __future__ import annotations

D = 1536
FF = 6144
LAYERS = 32
EPS = 1e-5
HEADS = 32
HEAD_DIM = 48

# Tile configuration per (N, K) at M = 41, chosen on the captured DiT graph
# (RTX 5090): fewer K splits keep the fused consumers' partial reads short.
SKINNY_CFG = {
    (3 * D, D): 11,    # fused self-attn QKV  256x512, 8 warps
    (D, D): 7,         # cross Q, O           256x256, 8 warps
    (FF, D): 3,        # FFN up               128x512
    (D, FF): 11,       # FFN down             256x512, 8 warps
}


def partial_floats(fvk, rows: int, cfg: dict | None = None) -> int:
    """FP32 partial scratch needed by the DiT GEMMs at ``rows`` rows."""
    cfg = cfg or SKINNY_CFG
    return max(k // fvk.nvfp4_skinny_config_k_chunk(c) * rows * n for (n, k), c in cfg.items())


def dit_forward_nvfp4(fvk, bufs: dict, weights: dict, dims: dict, *, attn,
                      stream: int = 0, pdl: bool = True, cfg: dict | None = None) -> None:
    """One denoising step of the DiT on NVFP4.

    Args:
        bufs: ``h`` (bf16 [Sa, D], residual stream, read and updated),
            ``xn`` / ``xn_sf`` (NVFP4 [Sa, D] block input), ``hid`` /
            ``hid_sf`` (NVFP4 [Sa, FF]), ``partials`` (fp32 scratch of
            ``partial_floats``), ``one`` (fp32 device scalar 1.0), ``hmod``
            (bf16 [Sa, D], output-AdaLN result for the output projection).
        weights: per-step pointer table — ``scale_msa`` / ``shift_msa``
            (32 bf16 [D] AdaLN vectors), ``out_scale`` / ``out_shift`` (output
            AdaLN), and the step-invariant NVFP4 sites ``qkv`` (16 self
            layers), ``q`` (16 cross layers), ``o``, ``ff_up``, ``ff_down``
            (32 layers), each entry ``(w_packed, sf_packed, w_gs, bias)``.
        dims: ``Sa``, ``Skv_text``, ``Skv_image``, optional ``kv_ld`` (row
            stride of the cross K / V, default ``D``).
    """
    cfg = cfg or SKINNY_CFG
    Sa = int(dims["Sa"])
    Skv_text = int(dims["Skv_text"])
    Skv_image = int(dims["Skv_image"])
    kv_ld = int(dims.get("kv_ld", D))   # row stride of the cross K / V
    h = int(bufs["h"])
    xn, xn_sf = int(bufs["xn"]), int(bufs["xn_sf"])
    hid, hid_sf = int(bufs["hid"]), int(bufs["hid_sf"])
    part = int(bufs["partials"])
    one = int(bufs["one"])
    hmod = int(bufs["hmod"])
    scale, shift = weights["scale_msa"], weights["shift_msa"]

    def _ck(rc: int, what: str, li: int) -> None:
        if rc != 0:
            raise RuntimeError(f"N1.7 NVFP4 DiT {what} layer {li} failed: cudaError {rc}")

    def _gemm(a, a_sf, site, n, k, li, what):
        w, sf, _, _ = site
        c = cfg[(n, k)]
        _ck(fvk.nvfp4_skinny_gemm(a, a_sf, w, sf, part, Sa, n, k, c, pdl, stream), what, li)
        return k // fvk.nvfp4_skinny_config_k_chunk(c)

    _ck(fvk.nvfp4_skinny_layer_norm(h, int(scale[0]), int(shift[0]), xn, xn_sf, one, 0,
                                    Sa, D, EPS, pdl, stream), "adaln", 0)
    for li in range(LAYERS):
        is_self = li % 2 == 1
        j = (li - 1) // 2 if is_self else li // 2
        slots = attn.get_slot_ptrs("dit_self" if is_self else "dit_cross", j)
        # Attention straight off the projection partials (Q, and K/V for the
        # self blocks); the cross K/V are per-frame inputs of this graph.
        if is_self:
            site = weights["qkv"][j]
            splits = _gemm(xn, xn_sf, site, 3 * D, D, li, "qkv gemm")
            _ck(fvk.nvfp4_skinny_partials_attention(part, splits, one, site[2], site[3], 3 * D, 0, 0, 0, Sa,
                                                    False, slots["O"], Sa, HEADS, HEAD_DIM, HEAD_DIM ** -0.5,
                                                    pdl, stream), "self attention", li)
        else:
            site = weights["q"][j]
            splits = _gemm(xn, xn_sf, site, D, D, li, "q gemm")
            kv = Skv_text if li % 4 == 0 else Skv_image
            _ck(fvk.nvfp4_skinny_partials_attention(part, splits, one, site[2], site[3], D, slots["K"],
                                                    slots["V"], kv_ld, kv, True, slots["O"], Sa, HEADS, HEAD_DIM,
                                                    HEAD_DIM ** -0.5, pdl, stream), "cross attention", li)

        # O projection (attention output quantized on load) -> bias + residual
        # -> pre-FF LayerNorm (no affine) -> fp4.
        w, sf, gs, b = weights["o"][li]
        c = cfg[(D, D)]
        _ck(fvk.nvfp4_skinny_gemm_bf16_act(slots["O"], one, w, sf, part, Sa, D, D, c, pdl, stream), "o gemm", li)
        splits = D // fvk.nvfp4_skinny_config_k_chunk(c)
        _ck(fvk.nvfp4_skinny_sum_bias_residual_norm(part, splits, one, gs, b, h, 0, 0, xn, xn_sf, one, 0,
                                                    Sa, D, EPS, pdl, stream), "o residual norm", li)

        # FFN up -> bias + GELU -> fp4.
        site = weights["ff_up"][li]
        splits = _gemm(xn, xn_sf, site, FF, D, li, "ffn up gemm")
        _ck(fvk.nvfp4_skinny_sum_bias_gelu(part, splits, one, site[2], site[3], hid, hid_sf, one,
                                           Sa, FF, pdl, stream), "ffn gelu", li)

        # FFN down -> bias + residual -> next AdaLN (fp4) or output AdaLN (bf16).
        site = weights["ff_down"][li]
        splits = _gemm(hid, hid_sf, site, D, FF, li, "ffn down gemm")
        if li < LAYERS - 1:
            _ck(fvk.nvfp4_skinny_sum_bias_residual_norm(
                part, splits, one, site[2], site[3], h, int(scale[li + 1]), int(shift[li + 1]),
                xn, xn_sf, one, 0, Sa, D, EPS, pdl, stream), "ffn residual adaln", li)
        else:
            _ck(fvk.nvfp4_skinny_sum_bias_residual_norm(
                part, splits, one, site[2], site[3], h, int(weights["out_scale"]), int(weights["out_shift"]),
                0, 0, 0, hmod, Sa, D, EPS, pdl, stream), "ffn residual out-norm", li)


# ─────────────────────────────────────────────────────────────────────────
# Backbone on NVFP4 (ViT taps + DeepStack, LLM, VLLN + VL self-attention).
# Experimental, opt-in (FLASHRT_GROOT_N17_BB_NVFP4): W4A4 error compounds
# through these stages, so the default backbone stays FP8.
# ─────────────────────────────────────────────────────────────────────────
#
# Every projection is a CUTLASS SM120 block-scaled W4A4 GEMM whose epilogue
# carries the bias, the residual add (C = D = residual stream), GELU / SwiGLU
# with an NVFP4 output, so the only non-GEMM launches per layer are the norm
# that emits the next NVFP4 operand, RoPE (fused with the per-head q/k norm)
# on the fused QKV output, attention reading Q/K/V in place, and the NVFP4
# quantization of the attention output. Residual streams are BF16;
# activation block scales are dynamic (global scale 1).

VIT_TAPS = (5, 11, 17)


class NvFp4Linear:
    """Packed NVFP4 weight + swizzled scales + per-tensor alpha (host value
    and the device fp32 scalar the CUTLASS bias / residual GEMM reads) + bias."""

    __slots__ = ("w", "sf", "alpha", "alpha_ptr", "bias", "n", "k")

    def __init__(self, w: int, sf: int, alpha: float, alpha_ptr: int, bias: int, n: int, k: int):
        self.w, self.sf, self.alpha, self.alpha_ptr = w, sf, float(alpha), alpha_ptr
        self.bias, self.n, self.k = bias, n, k


def _ck(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"N1.7 NVFP4 backbone {what} failed: status {rc}")


def _schedule(lin: NvFp4Linear) -> int:
    # stream-K for the long-K projections (down / fc2 at K >= 6144), whose
    # persistent tiling otherwise leaves SMs idle at these row counts.
    return 1 if lin.k >= 6144 else 0


def linear_bf16(fvk, a, a_sf, lin: NvFp4Linear, out, M, stream, *, residual: bool = False) -> None:
    """out = a @ W^T * alpha + bias (+ out when residual)."""
    _ck(fvk.nvfp4_gemm_bias_res_bf16out(a, lin.w, a_sf, lin.sf, lin.bias, out if residual else 0, out,
                                        M, lin.n, lin.k, lin.alpha_ptr, 1.0 if residual else 0.0,
                                        _schedule(lin), stream), f"gemm {M}x{lin.n}x{lin.k}")


def linear_gelu_fp4(fvk, a, a_sf, lin: NvFp4Linear, out, out_sf, M, stream) -> None:
    """NVFP4 out = quant(gelu(a @ W^T * alpha + bias)); M must be a multiple of
    128 (the block-scale-factor epilogue runs whole tiles: pass the padded row
    count of zero-padded operands)."""
    fvk.fp4_w4a16_gemm_bias_gelu_fp4out_sm120(a, lin.w, a_sf, lin.sf, lin.bias, out, out_sf,
                                             M, lin.n, lin.k, lin.alpha, stream)


def attention(fa2, q, k, v, o, lse, *, batch, seqlen, hq, hkv, hd, ld_qkv, ld_o, causal, num_sms, stream):
    fwd = fa2.fwd_bf16_causal if causal else fa2.fwd_bf16
    fwd(Q=q, K=k, V=v, O=o, softmax_lse=lse, softmax_lse_accum=0, o_accum=0,
        batch=batch, seqlen_q=seqlen, seqlen_k=seqlen, num_heads_q=hq, num_heads_kv=hkv, head_dim=hd,
        q_strides=(seqlen * ld_qkv, ld_qkv, hd), k_strides=(seqlen * ld_qkv, ld_qkv, hd),
        v_strides=(seqlen * ld_qkv, ld_qkv, hd), o_strides=(seqlen * ld_o, ld_o, hd),
        softmax_scale=1.0 / (hd ** 0.5), num_sms=num_sms, stream=stream)


def vit_forward_nvfp4(fvk, fa2, B: dict, W: dict, dims: dict, *, stream: int, num_sms: int) -> None:
    """Qwen3-VL ViT layers 0..17 with the three DeepStack mergers run at their
    taps (the backbone consumes only the taps; the merger of the final layer
    is not part of the action pipeline)."""
    S, nv = int(dims["S"]), int(dims["views"])
    Sper, D, FF, NH, HD = S // nv, 1024, 4096, 16, 64
    Nout = S // 4
    h, xq, xsf = B["vit_h"], B["vit_xq"], B["vit_xsf"]
    qkv, o, lse, hq, hsf = B["vit_qkv"], B["vit_o"], B["vit_lse"], B["vit_hq"], B["vit_hsf"]
    one = B["one"]
    for li in range(VIT_TAPS[-1] + 1):
        L = W["vit"][li]
        _ck(fvk.nvfp4_norm_quant(h, L["ln1_w"], L["ln1_b"], 0, 1e-6, xq, xsf, one, S, D, stream), "vit ln1")
        linear_bf16(fvk, xq, xsf, L["qkv"], qkv, S, stream)
        _ck(fvk.nvfp4_qk_norm_rope(qkv, 3 * D, 0, D, NH, NH, HD, W["vit_cos"], W["vit_sin"], 0, 0, 1e-6,
                                   S, stream), "vit rope")
        attention(fa2, qkv, qkv + 2 * D, qkv + 4 * D, o, lse, batch=nv, seqlen=Sper, hq=NH, hkv=NH, hd=HD,
                  ld_qkv=3 * D, ld_o=D, causal=False, num_sms=num_sms, stream=stream)
        _ck(fvk.nvfp4_quant(o, D, xq, xsf, one, S, D, stream), "vit o quant")
        linear_bf16(fvk, xq, xsf, L["o"], h, S, stream, residual=True)
        _ck(fvk.nvfp4_norm_quant(h, L["ln2_w"], L["ln2_b"], 0, 1e-6, xq, xsf, one, S, D, stream), "vit ln2")
        linear_gelu_fp4(fvk, xq, xsf, L["fc1"], hq, hsf, S, stream)
        linear_bf16(fvk, hq, hsf, L["fc2"], h, S, stream, residual=True)
        if li in VIT_TAPS:
            j = VIT_TAPS.index(li)
            M = W["ds"][j]
            # spatial-merge groups of 4 consecutive patch tokens: [S, D] viewed as [S/4, 4D]
            _ck(fvk.nvfp4_norm_quant(h, M["ln_w"], M["ln_b"], 0, 1e-6, B["ds_xq"], B["ds_xsf"], one,
                                     Nout, 4 * D, stream), "deepstack ln")
            linear_gelu_fp4(fvk, B["ds_xq"], B["ds_xsf"], M["fc1"], B["ds_hq"], B["ds_hsf"], Nout, stream)
            linear_bf16(fvk, B["ds_hq"], B["ds_hsf"], M["fc2"], B["ds_out"][j], Nout, stream)


def llm_forward_nvfp4(fvk, fa2, B: dict, W: dict, dims: dict, *, stream: int, num_sms: int) -> None:
    """16 truncated Qwen3-VL decoder layers (causal GQA, q/k norm, M-RoPE,
    SwiGLU), DeepStack features added at the visual rows after layers 0..2."""
    S, D, FF, NHQ, NHKV, HD = int(dims["S"]), 2048, 6144, 16, 8, 128
    Sp = int(dims["S_pad"])
    QKV = (NHQ + 2 * NHKV) * HD
    h, xq, xsf = B["llm_h"], B["llm_xq"], B["llm_xsf"]
    qkv, o, lse, gq, gsf = B["llm_qkv"], B["llm_o"], B["llm_lse"], B["llm_gq"], B["llm_gsf"]
    one = B["one"]
    n_vis = int(dims["n_vis"])
    for li in range(16):
        L = W["llm"][li]
        _ck(fvk.nvfp4_norm_quant(h, L["in_ln_w"], 0, 1, 1e-6, xq, xsf, one, S, D, stream), "llm in-norm")
        linear_bf16(fvk, xq, xsf, L["qkv"], qkv, S, stream)
        _ck(fvk.nvfp4_qk_norm_rope(qkv, QKV, 0, NHQ * HD, NHQ, NHKV, HD, W["llm_cos"], W["llm_sin"],
                                   L["q_norm_w"], L["k_norm_w"], 1e-6, S, stream), "llm qk norm rope")
        attention(fa2, qkv, qkv + 2 * NHQ * HD, qkv + 2 * (NHQ + NHKV) * HD, o, lse, batch=1, seqlen=S,
                  hq=NHQ, hkv=NHKV, hd=HD, ld_qkv=QKV, ld_o=NHQ * HD, causal=True, num_sms=num_sms,
                  stream=stream)
        _ck(fvk.nvfp4_quant(o, NHQ * HD, xq, xsf, one, S, NHQ * HD, stream), "llm o quant")
        linear_bf16(fvk, xq, xsf, L["o"], h, S, stream, residual=True)
        _ck(fvk.nvfp4_norm_quant(h, L["post_ln_w"], 0, 1, 1e-6, xq, xsf, one, S, D, stream), "llm post-norm")
        # SwiGLU fold: one GEMM on the row-interleaved gate|up weight, silu(g)*u
        # and the NVFP4 block scales in its epilogue (full-width output), then
        # the even-column compaction into the down projection's operand.
        gu = L["gate_up"]
        fvk.fp4_w4a16_dual_gemm_silu_fp4out_sm120(xq, gu.w, gu.w, xsf, gu.sf, gu.sf, B["llm_fold"], gsf,
                                                 Sp, 2 * FF, D, gu.alpha_gate, gu.alpha_up, stream)
        fvk.fp4_swiglu_even_col_compact_v2(B["llm_fold"], gq, S, FF, stream)
        linear_bf16(fvk, gq, gsf, L["down"], h, S, stream, residual=True)
        if li < 3:
            _ck(fvk.nvfp4_scatter_add_rows(h, B["ds_out"][li], B["vis_idx"], n_vis, D, stream), "deepstack add")


def vlsa_forward_nvfp4(fvk, fa2, B: dict, W: dict, dims: dict, *, stream: int, num_sms: int) -> None:
    """VLLN, then the 4-layer VL self-attention transformer; the result is
    written as FP16 into ``out_fp16`` for the DiT cross-KV refresh."""
    S, D, NH, HD, FF = int(dims["S"]), 2048, 32, 64, 8192
    Sp = int(dims["S_pad"])
    h, xq, xsf = B["vlsa_h"], B["vlsa_xq"], B["vlsa_xsf"]
    qkv, o, lse, hq, hsf = B["vlsa_qkv"], B["vlsa_o"], B["vlsa_lse"], B["vlsa_hq"], B["vlsa_hsf"]
    one = B["one"]
    # VLLN: LayerNorm(w, b) == no-affine norm modulated by (1 + (w - 1)) and shift b.
    _ck(fvk.nvfp4_skinny_layer_norm(B["llm_h"], W["vlln_wm1"], W["vlln_b"], 0, 0, 0, h, S, D, 1e-5,
                                    False, stream), "vlln")
    for li in range(4):
        L = W["vlsa"][li]
        _ck(fvk.nvfp4_norm_quant(h, L["n1_w"], L["n1_b"], 0, 1e-5, xq, xsf, one, S, D, stream), "vlsa norm1")
        linear_bf16(fvk, xq, xsf, L["qkv"], qkv, S, stream)
        attention(fa2, qkv, qkv + 2 * D, qkv + 4 * D, o, lse, batch=1, seqlen=S, hq=NH, hkv=NH, hd=HD,
                  ld_qkv=3 * D, ld_o=D, causal=False, num_sms=num_sms, stream=stream)
        _ck(fvk.nvfp4_quant(o, D, xq, xsf, one, S, D, stream), "vlsa o quant")
        linear_bf16(fvk, xq, xsf, L["o"], h, S, stream, residual=True)
        _ck(fvk.nvfp4_norm_quant(h, L["n3_w"], L["n3_b"], 0, 1e-5, xq, xsf, one, S, D, stream), "vlsa norm3")
        linear_gelu_fp4(fvk, xq, xsf, L["fc1"], hq, hsf, Sp, stream)
        linear_bf16(fvk, hq, hsf, L["fc2"], h, S, stream, residual=True)
    fvk.cast_bf16_to_fp16(h, B["out_fp16"], S * D, stream)

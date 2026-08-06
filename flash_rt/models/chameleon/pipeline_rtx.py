"""FlashRT — Chameleon-7B VLM forward for Jetson AGX Orin (SM87).

Standalone text-generating forward on the Chameleon-7B INT8/QuaRot-INT4
kernel set (SM80 CUTLASS rowwise GEMMs + Hadamard rotations). Key design
points:

1. **No attention bias.** Upstream Chameleon has ``attention_bias = false`` and
   ``mlp_bias = false``, so the no-bias GEMM entries are used unconditionally.
2. **A real KV cache.** The K and V GEMMs write straight into the attention
   backend's per-layer slab — legal because CUTLASS hard-wires its output row
   stride to ``N``, which equals the cache's row stride. No staging, no copy.
3. **One code path for prefill and decode.** ``pos is None`` means prefill
   (``S`` rows at the slab base, RoPE from position 0); ``pos`` set means a
   single decode row written at ``pos``, with the cos/sin pointers advanced by
   ``pos`` rows. ``qk_norm_rope_fused_fp16`` needs no modification for this:
   it derives position as ``row / num_heads``, which is 0 at ``S = 1``, so the
   position lives entirely in the table pointer.
4. **lm_head tail instead of an action head.** Final RMSNorm, then per-row INT8
   quantization of the wanted row(s), then an INT8 GEMM to BF16 logits. The
   ``mask_image_logits`` step and the argmax live in the frontend (they are
   torch ops on the logits view, which are graph-safe — verified).

lm_head stays INT8 in both precision tiers: it is 268 MB/token = 1.74 ms =
3.7 % of the INT8 decode budget, and dropping to 15 INT4 levels over a
65536-row output is not worth 0.8 ms.

Raw-pointer interface only (int pointers + Python primitives) for CUDA-Graph
safety: no torch ops, no allocation, no sync inside the forward.
"""

from __future__ import annotations

import ctypes

_CUDART = None


def _gpu_copy(dst_ptr: int, src_ptr: int, nbytes: int, stream: int) -> None:
    """Async D2D copy — the in-graph layer-probe mechanism."""
    global _CUDART
    if _CUDART is None:
        _CUDART = ctypes.CDLL("libcudart.so")
    _CUDART.cudaMemcpyAsync(
        ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(nbytes), 3, ctypes.c_void_p(stream))


def _check(status, name: str, shape) -> None:
    if status != 0:
        raise RuntimeError(f"{name} failed: status={status} shape={shape}")


def chameleon_forward(
    fvk, bufs, weights, dims, scales_dev,
    *, attn, S: int, pos=None, stream: int = 0,
    use_int4: bool = False, use_int4_down: bool = False,
    use_hadamard: bool = False,
    ffn_down_clamp_value: float = 60000.0,
    ffn_down_clamp_last_n: int = 4,
    logits_all: bool = False, probe=None,
) -> None:
    """Run the 32-layer Chameleon-7B decoder and the lm_head.

    Args:
        fvk: ``flash_rt.flash_rt_kernels`` module.
        bufs / weights / dims / scales_dev: int-pointer dicts from the frontend.
        attn: ``ChameleonAttnBackend`` (owns the KV cache).
        S: rows to process. ``1`` on the decode path.
        pos: ``None`` for prefill; otherwise the absolute KV position of the
            single decode row. RoPE and the KV write both key off this.
        ffn_down_clamp_value: symmetric clamp applied to the down-projection
            output before the residual add; ``<= 0`` disables it.
            **This is required for correctness, not a tuning knob.** Measured on
            this checkpoint (ISL=1032), the reference's magnitudes are tiny
            through L30 and then explode at L31 only:

                layer      L28    L29    L30      L31
                residual  1616   1720   2032   266240
                down_out  1120   1032   1880   264192

            2.6e5 is far beyond FP16's 65504, so the store becomes ``inf`` and
            the final RMSNorm then poisons that row's logits. Because the
            pre-L31 residual is only ~2032, clamping the down output at 60000
            keeps the residual at ~62000 < 65504 — so no BF16 residual stream is
            needed. See ``docs/chameleon7b_rtx_sm87.md``. Unlike the
            Thor path we do **not** clamp the down *input*: ours is BF16
            (``cutlass_int8_silu_gated_bf16out``), whose range absorbs the
            151552 without issue.
        logits_all: compute logits for **all** S rows (teacher-forced
            precision comparison). Requires an ``[S, vocab]`` logits buffer.
        probe: ``{"layers": [...], "bufs": [...], "final_buf": ptr}`` to snapshot
            the post-residual hidden state; ``None`` disables it at zero cost.
    """
    decode = pos is not None
    if decode and S != 1:
        raise ValueError(f"decode path requires S == 1, got {S}")

    D = int(dims["D"])
    Dff = int(dims["Dff"])
    L = int(dims["L"])
    H = int(dims["H"])
    Hd = int(dims["Hd"])
    V = int(dims["vocab"])

    x_ptr = int(bufs["x"])
    xn_ptr = int(bufs["xn"])
    int8_act_d_ptr = int(bufs["int8_act_d"])
    int8_act_ff_ptr = int(bufs["int8_act_ff"])
    int4_act_d_ptr = int(bufs.get("int4_act_d", 0))
    int4_act_ff_ptr = int(bufs.get("int4_act_ff", 0))
    bf16_gate_ptr = int(bufs["bf16_gate_ff"])
    bf16_xn_ff_ptr = int(bufs["bf16_xn_ff"])
    o_proj_out_ptr = int(bufs["o_proj_out"])
    logits_ptr = int(bufs["logits"])
    lm_act_ptr = int(bufs["lm_act"])
    lm_scale_ptr = int(bufs["lm_act_scale"])

    # RoPE position enters purely through the table pointer (see module doc).
    rope_off = (pos if decode else 0) * Hd * 2
    cos_ptr = int(weights["rope_cos"]) + rope_off
    sin_ptr = int(weights["rope_sin"]) + rope_off

    probe_map = None
    probe_final = None
    if probe is not None:
        layers = probe.get("layers") or []
        pbufs = probe.get("bufs") or []
        if len(layers) != len(pbufs):
            raise ValueError("probe['layers'] and probe['bufs'] length mismatch")
        probe_map = {int(li): int(p) for li, p in zip(layers, pbufs)}
        probe_final = int(probe.get("final_buf") or 0)

    # ── entry: fused RMSNorm + quantize (layer 0's input_layernorm) ──
    if use_int4:
        fvk.rms_norm_fht_int4_fp16(
            x_ptr, int(weights["input_ln_w"][0]),
            int4_act_d_ptr, int(scales_dev["act_qkv"][0]),
            S, D, 1e-5, int(stream))
    elif use_hadamard:
        fvk.rms_norm_fht_int8_fp16(
            x_ptr, int(weights["input_ln_w"][0]),
            int8_act_d_ptr, int(scales_dev["act_qkv"][0]),
            S, D, 1e-5, int(stream))
    else:
        fvk.rms_norm_int8_rowwise_fp16(
            x_ptr, int(weights["input_ln_w"][0]),
            int8_act_d_ptr, int(scales_dev["act_qkv"][0]),
            S, D, 1e-5, int(stream))

    act_d_ptr = int4_act_d_ptr if use_int4 else int8_act_d_ptr

    for li in range(L):
        if decode:
            K_ptr, V_ptr = attn.kv_row_ptrs(li, pos)
            Q_ptr = int(attn.get_slot_ptrs("llm", li)["Q"])
        else:
            slots = attn.get_slot_ptrs("llm", li)
            Q_ptr, K_ptr, V_ptr = int(slots["Q"]), int(slots["K"]), int(slots["V"])

        a_qkv = int(scales_dev["act_qkv"][li])
        a_o = int(scales_dev["act_o"][li])
        a_gu = int(scales_dev["act_gu"][li])
        a_d = int(scales_dev["act_down"][li])

        # ── Q/K/V: K and V land directly in the KV cache ──
        qkv_gemm = (fvk.cutlass_int4_rowwise_fp16out if use_int4
                    else fvk.cutlass_int8_rowwise_fp16out)
        for name, out_ptr in (("q_w", Q_ptr), ("k_w", K_ptr), ("v_w", V_ptr)):
            _check(qkv_gemm(act_d_ptr, int(weights[name][li]), a_qkv,
                            int(weights[name + "_scale"][li]), out_ptr,
                            S, D, D, int(stream)),
                   f"{'int4' if use_int4 else 'int8'} {name}", (S, D, D))

        # ── fused per-head QK LayerNorm(+bias) + rotate-half RoPE, in place ──
        fvk.qk_norm_rope_fused_fp16(
            Q_ptr, K_ptr,
            int(weights["q_norm_w"][li]), int(weights["q_norm_b"][li]),
            int(weights["k_norm_w"][li]), int(weights["k_norm_b"][li]),
            cos_ptr, sin_ptr,
            S, H, Hd, 1e-5, int(stream))

        # ── causal MHA (result written back into the Q slot) ──
        if decode:
            attn.run_decode(li, pos, stream=int(stream))
        else:
            attn.run_prefill(li, S, stream=int(stream))

        # ── O projection ──
        if use_int4:
            fvk.fht_int4_quant_fp16(Q_ptr, int4_act_d_ptr, a_o, S, D, int(stream))
        elif use_hadamard:
            fvk.fht_int8_quant_fp16(Q_ptr, int8_act_d_ptr, a_o, S, D, int(stream))
        else:
            fvk.quantize_int8_rowwise_fp16(Q_ptr, int8_act_d_ptr, a_o,
                                           S, D, int(stream))
        o_gemm = (fvk.cutlass_int4_rowwise_fp16out if use_int4
                  else fvk.cutlass_int8_rowwise_fp16out)
        _check(o_gemm(act_d_ptr, int(weights["o_w"][li]), a_o,
                      int(weights["o_w_scale"][li]), o_proj_out_ptr,
                      S, D, D, int(stream)), "o_proj", (S, D, D))

        # ── residual_1 + post-attention RMSNorm + quantize ──
        if use_int4:
            fvk.residual_add_rms_norm_fht_int4_fp16(
                x_ptr, o_proj_out_ptr, int(weights["post_ln_w"][li]),
                int4_act_d_ptr, a_gu, S, D, 1e-5, int(stream))
        elif use_hadamard:
            fvk.residual_add_rms_norm_fht_int8_fp16(
                x_ptr, o_proj_out_ptr, int(weights["post_ln_w"][li]),
                int8_act_d_ptr, a_gu, S, D, 1e-5, int(stream))
        else:
            fvk.residual_add_rms_norm_int8_rowwise_fp16(
                x_ptr, o_proj_out_ptr, int(weights["post_ln_w"][li]),
                int8_act_d_ptr, a_gu, S, D, 1e-5, int(stream))

        # ── FFN: gate -> BF16, up with fused SiLU(gate)* -> BF16 ──
        if use_int4:
            _check(fvk.cutlass_int4_rowwise_bf16out(
                int4_act_d_ptr, int(weights["gate_w"][li]), a_gu,
                int(weights["gate_w_scale"][li]), bf16_gate_ptr,
                S, Dff, D, int(stream)), "int4 gate", (S, Dff, D))
            _check(fvk.cutlass_int4_silu_gated_bf16out(
                int4_act_d_ptr, int(weights["up_w"][li]), a_gu,
                int(weights["up_w_scale"][li]), bf16_gate_ptr, bf16_xn_ff_ptr,
                S, Dff, D, int(stream)), "int4 up+silu", (S, Dff, D))
        else:
            _check(fvk.cutlass_int8_rowwise_bf16out(
                int8_act_d_ptr, int(weights["gate_w"][li]), a_gu,
                int(weights["gate_w_scale"][li]), bf16_gate_ptr,
                S, Dff, D, int(stream)), "int8 gate", (S, Dff, D))
            _check(fvk.cutlass_int8_silu_gated_bf16out(
                int8_act_d_ptr, int(weights["up_w"][li]), a_gu,
                int(weights["up_w_scale"][li]), bf16_gate_ptr, bf16_xn_ff_ptr,
                S, Dff, D, int(stream)), "int8 up+silu", (S, Dff, D))

        # ── down projection ──
        if use_int4_down:
            fvk.fht128_int4_quant_bf16(bf16_xn_ff_ptr, int4_act_ff_ptr, a_d,
                                       S, Dff, int(stream))
            _check(fvk.cutlass_int4_rowwise_fp16out(
                int4_act_ff_ptr, int(weights["d_w"][li]), a_d,
                int(weights["d_w_scale"][li]), o_proj_out_ptr,
                S, D, Dff, int(stream)), "int4 down", (S, D, Dff))
        else:
            fvk.quantize_int8_rowwise(bf16_xn_ff_ptr, int8_act_ff_ptr, a_d,
                                      S, Dff, int(stream))
            _check(fvk.cutlass_int8_rowwise_fp16out(
                int8_act_ff_ptr, int(weights["d_w"][li]), a_d,
                int(weights["d_w_scale"][li]), o_proj_out_ptr,
                S, D, Dff, int(stream)), "int8 down", (S, D, Dff))

        # Guard the FP16 residual against L31's massive down output (see the
        # measured table in the docstring). Restricted to the last
        # ``ffn_down_clamp_last_n`` layers: the magnitude grows monotonically
        # with depth and L28 measures 1616, i.e. 37x below the clamp, so the
        # earlier layers cannot reach it. Clamping all 32 layers instead costs
        # 6.2 ms of a 281 ms prefill (2.2 %) for no effect. The Gate-1 harness
        # reports per-layer clamp saturation, so a checkpoint that violates the
        # assumption is detectable — raise this to L if that ever happens.
        if ffn_down_clamp_value > 0.0 and li >= L - ffn_down_clamp_last_n:
            fvk.clamp_inplace_fp16(o_proj_out_ptr, float(ffn_down_clamp_value),
                                   S * D, int(stream))

        # ── residual_2 (+ next layer's input_layernorm + quantize) ──
        if li < L - 1:
            if use_int4:
                fvk.residual_add_rms_norm_fht_int4_fp16(
                    x_ptr, o_proj_out_ptr, int(weights["input_ln_w"][li + 1]),
                    int4_act_d_ptr, int(scales_dev["act_qkv"][li + 1]),
                    S, D, 1e-5, int(stream))
            elif use_hadamard:
                fvk.residual_add_rms_norm_fht_int8_fp16(
                    x_ptr, o_proj_out_ptr, int(weights["input_ln_w"][li + 1]),
                    int8_act_d_ptr, int(scales_dev["act_qkv"][li + 1]),
                    S, D, 1e-5, int(stream))
            else:
                fvk.residual_add_rms_norm_int8_rowwise_fp16(
                    x_ptr, o_proj_out_ptr, int(weights["input_ln_w"][li + 1]),
                    int8_act_d_ptr, int(scales_dev["act_qkv"][li + 1]),
                    S, D, 1e-5, int(stream))
        else:
            fvk.residual_add_fp16(x_ptr, o_proj_out_ptr, S * D, int(stream))

        if probe_map is not None and li in probe_map:
            _gpu_copy(probe_map[li], x_ptr, S * D * 2, stream)

    # ── final RMSNorm ──
    fvk.rms_norm_fp16(x_ptr, int(weights["final_norm_w"]), xn_ptr,
                      S, D, 1e-5, int(stream))
    if probe_final:
        _gpu_copy(probe_final, xn_ptr, S * D * 2, stream)

    # ── lm_head: INT8 W8A8 -> BF16 logits ──
    # Next-token prediction reads the last row; logits_all is for the
    # teacher-forced precision comparison.
    rows, row_off = (S, 0) if logits_all else (1, S - 1)
    fvk.quantize_int8_rowwise_fp16(xn_ptr + row_off * D * 2, lm_act_ptr,
                                   lm_scale_ptr, rows, D, int(stream))
    _check(fvk.cutlass_int8_rowwise_bf16out(
        lm_act_ptr, int(weights["lm_head_w"]), lm_scale_ptr,
        int(weights["lm_head_w_scale"]), logits_ptr,
        rows, V, D, int(stream)), "lm_head", (rows, V, D))


__all__ = ["chameleon_forward"]

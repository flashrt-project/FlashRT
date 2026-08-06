"""FP4 encoder forward variant for Pi0.5 (Phase 4.3 Step C, additive).

Mirrors `shared_primitives.encoder_forward` but allows a configurable subset
of layers to run Gate+Up and Down GEMMs in NVFP4 instead of FP8. Non-FP4
layers execute BIT-IDENTICAL to the original encoder_forward.

ARCHITECTURE (borrowed from NVIDIA enable_llm_nvfp4, see sample_code/
pytorch_to_onnx.py:485-501):
    In FP4 layers we keep the residual stream in fp16 (output_quantizer
    disabled), so the FP4 GEMM sees fp16 activation input — no double-lossy
    fp8→fp16 dequant. The QKV / Attention / O path stays FP8 throughout.

DOES NOT MODIFY:
    - shared_primitives.encoder_forward  (unchanged)
    - flash_rt_kernels.so               (unchanged)
    - any existing kernel source         (unchanged)

Uses:
    - fvk (flash_rt_kernels)  for all FP8 kernels + gate_geglu_merged_fp16
    - fvk_fp4 (flash_rt_fp4)  for NVFP4 GEMM + norm_fp16 + quant/reshape
"""

import math


def _check(rc, kernel, layer, **shape):
    """Raise on a non-zero NVFP4 GEMM status.

    The NVFP4 GEMM bindings report can_implement, initialize, workspace and
    run failures through a return code rather than an exception, so an
    unchecked call would leave the destination buffer untouched and let the
    pipeline consume stale data. (The fused elementwise bindings validate in
    C++ and raise directly; they return nothing and are not wrapped here.)
    """
    if rc != 0:
        dims = ", ".join(f"{name}={value}" for name, value in shape.items())
        raise RuntimeError(
            f"{kernel} failed at encoder layer {layer} ({dims}) rc={rc}")


def encoder_forward_with_fp4_subset(gemm, fvk, fvk_fp4, bufs, weights, dims,
                                     stream=0, *, attn=None,
                                     fp4_layers: set = None,
                                     fp4_weights: dict = None,
                                     fp4_scratch: dict = None,
                                     use_p1_split_gu: bool = False,
                                     fp4_attn_weights: dict = None):
    """Encoder forward with FP4 Gate+Up / Down on selected layers.

    When fp4_layers is empty / None, behaves identically to encoder_forward.

    Args:
        gemm, fvk, bufs, weights, dims, stream, attn: same as encoder_forward.
        fp4_layers: iterable of layer indices (0..L-1) where Gate+Up and Down
            should run in NVFP4. Other layers stay FP8.
        fp4_weights: dict { layer_idx: { 'gate_up': {packed, sfb, N, K},
                                         'down':    {packed, sfb, N, K} } }
            Output of fp4_utils.quant_weight_nvfp4 on each weight.
        fp4_scratch: dict with keys:
            'gu_act'  — FP4ActScratch for Gate+Up input  (max_M=Se, K=D)
            'down_act'— FP4ActScratch for Down input     (max_M=Se, K=H)
            'x_normed'— fp16 scratch [Se, D]   (post-rms buffer for Gate+Up)
            'gate_out'— fp16 scratch [Se, 2H]  (Gate+Up GEMM output)
            'hid_fp16'— fp16 scratch [Se, H]   (post-silu_mul output, Down input)
            'fg_fp16' — fp16 scratch [Se, D]   (Down GEMM output)
            'variant_gu' — int NVFP4 variant idx for Gate+Up
            'variant_dn' — int NVFP4 variant idx for Down
    """
    fp4_layers = set(fp4_layers or ())
    if fp4_layers and (fp4_weights is None or fp4_scratch is None):
        raise ValueError("fp4_weights and fp4_scratch required when fp4_layers non-empty")
    attn_fp4 = fp4_attn_weights or {}
    if attn_fp4 and not fp4_layers:
        raise ValueError(
            "fp4_attn_weights requires the FP4 encoder scratch "
            "(fp4_layers must be non-empty)")

    Se = dims['Se']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    total_keys = dims['total_keys']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    o_fp8 = bufs['o_fp8']
    gate = bufs['gate']
    hid_fp8 = bufs['hid_fp8']
    fg = bufs['fg']

    act_scales = weights['act_scales']
    alpha_host = weights['alpha_host']

    # Pre-grab FP4 scratch if any layer uses it
    if fp4_layers:
        sc_gu = fp4_scratch['gu_act']
        sc_dn = fp4_scratch['down_act']
        x_normed_ptr = fp4_scratch['x_normed']
        gate_fp16_ptr = fp4_scratch['gate_out']
        hid_fp16_ptr = fp4_scratch['hid_fp16']
        fg_fp16_ptr = fp4_scratch['fg_fp16']
        variant_gu = fp4_scratch['variant_gu']
        variant_dn = fp4_scratch['variant_dn']
        if attn_fp4:
            sc_at = fp4_scratch['attn_act']
            attn_variant = fp4_scratch['attn_variant']
        # P1 split-GU scratch (only used when use_p1_split_gu=True)
        if use_p1_split_gu:
            p1_combiner = fp4_scratch['p1_combiner']
            if p1_combiner == 'epilogue':
                p1_il_p4     = fp4_scratch['p1_il_p4']
                p1_il_sfa    = fp4_scratch['p1_il_sfa']
                variant_dn_x = fp4_scratch['variant_dn_x']
            elif p1_combiner == 'epilogue_hw':
                p1_dummy = fp4_scratch['p1_dummy']
            else:
                p1_gate_p4   = fp4_scratch['p1_gate_p4']
                p1_gate_sfa  = fp4_scratch['p1_gate_sfa']
                p1_up_p4     = fp4_scratch['p1_up_p4']
                p1_up_sfa    = fp4_scratch['p1_up_sfa']
            # Down input scratch for P1 (replaces sc_dn in this branch — same buffers)

    for l in range(L):
        last = (l == L - 1)
        is_fp4 = l in fp4_layers

        as_qkv = act_scales + (l * 4 + 0) * 4
        as_o   = act_scales + (l * 4 + 1) * 4
        as_gu  = act_scales + (l * 4 + 2) * 4
        as_d   = act_scales + (l * 4 + 3) * 4

        aw_attn = attn_fp4.get(l)

        # 1+2. RMSNorm + QKV GEMM. Default: FP8 with static scales. With
        # an AWQ-calibrated NVFP4 QKV weight ('qkv' in the layer entry),
        # the input runs weightless RMSNorm x per-channel AWQ inverse
        # scale straight into NVFP4 (input_layernorm stays folded into
        # the quantized weight).
        #
        # NOTE (FP8 path): on paper, step 1 for layer l>=1 is redundant
        # with the previous layer's step 11 (same scale, unchanged x).
        # However, the FP8 calibration scales were collected by
        # encoder_forward_calibrate WITH step 1 in place. Step 11 uses
        # fp32 (r+x) intermediate; step 1 re-reads the fp16-cast residual
        # and recomputes rms from the rounded value. The resulting x_fp8
        # differs at the fp16 rounding level — calibration was tuned to
        # step-1's output, so skipping it introduces a systematic
        # scale-mismatch that compounds across 17 layers. Empirically cos
        # drops from 0.997 to 0.91. Keep the call.
        if aw_attn is not None and 'qkv' in aw_attn:
            fvk_fp4.rms_norm_mul_fp4_sfa_fp16(
                x, aw_attn['qkv_inv_s'],
                sc_at.packed.data_ptr(), sc_at.sfa.data_ptr(),
                Se, D, stream)
            rc = fvk_fp4.cutlass_fp4_gemm_variant(
                attn_variant,
                sc_at.packed.data_ptr(), sc_at.sfa.data_ptr(),
                aw_attn['qkv']['packed'].data_ptr(),
                aw_attn['qkv']['sfb'].data_ptr(),
                qkv, Se, 2560, D, 1.0, 0.0, stream)
            if rc != 0:
                raise RuntimeError(
                    f"encoder FP4 qkv GEMM layer {l} failed rc={rc}")
        else:
            fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, Se, D, as_qkv, stream)
            fvk.cutlass_fp8_sq(x_fp8, weights['qkv_w'][l], qkv,
                               Se, 2560, D, alpha_host[l * 4 + 0], 0.0,
                               stream)

        # 3+4. QKV split + RoPE + KV cache write (vectorized, bit-exact)
        kv_elem_off = l * total_keys * HD
        rc = fvk.qkv_split_rope_kvcache_fp16_vec(
            qkv, weights['rope'], attn_out,
            weights['Kc'], weights['Vc'],
            Se, Q_dim, K_dim, HD, 2560,
            kv_elem_off, HD, stream)
        if rc != 0:
            raise RuntimeError(
                f"encoder qkv split layer {l} failed rc={rc}")

        if not last:
            # 5. Attention (unchanged)
            if attn is not None:
                attn.run("encoder", l, q_seq=Se, stream=stream)
            else:
                K_ptr = weights['Kc'] + kv_elem_off * 2
                V_ptr = weights['Vc'] + kv_elem_off * 2
                fvk.attention_qkv_fp16(bufs['ctx'], attn_out, K_ptr, V_ptr,
                                        logits, attn_out,
                                        Se, Se, NH, HD, attn_scale, stream)

            # 6. quantize attn output + O GEMM (FP8 static scales, or
            # dynamic NVFP4 with fp4_attn_weights).
            if aw_attn is not None and 'o' in aw_attn:
                rc = fvk_fp4.quantize_fp4_dynamic_sfa_fp16_vec(
                    attn_out, sc_at.packed.data_ptr(),
                    sc_at.sfa.data_ptr(), Se, D, False, stream)
                if rc != 0:
                    raise RuntimeError(
                        f"encoder FP4 O quantize layer {l} failed rc={rc}")
                rc = fvk_fp4.cutlass_fp4_gemm_variant(
                    attn_variant,
                    sc_at.packed.data_ptr(), sc_at.sfa.data_ptr(),
                    aw_attn['o']['packed'].data_ptr(),
                    aw_attn['o']['sfb'].data_ptr(),
                    fg, Se, D, D, 1.0, 0.0, stream)
                if rc != 0:
                    raise RuntimeError(
                        f"encoder FP4 O GEMM layer {l} failed rc={rc}")
            else:
                fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, Se * D,
                                             stream)
                fvk.cutlass_fp8_sq(o_fp8, weights['o_w'][l], fg,
                                   Se, D, D, alpha_host[l * 4 + 1], 0.0,
                                   stream)

            if is_fp4:
                # ── FP4 path. Residual stream stays fp16.
                w_gu = fp4_weights[l]['gate_up']
                w_dn = fp4_weights[l]['down']
                awq_gu = (fp4_scratch.get('awq_inv_s_gu') or {}).get(l)
                awq_dn = (fp4_scratch.get('awq_inv_s_dn') or {}).get(l)

                # Pre-GEMM: F3 / F3+mul → x_fp4 + SFA at sc_gu
                if awq_gu is None:
                    fvk_fp4.residual_add_rms_norm_fp4_sfa_fp16(
                        x, fg,
                        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                        Se, D, stream)
                else:
                    fvk_fp4.residual_add_rms_norm_mul_fp4_sfa_fp16(
                        x, fg, awq_gu,
                        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                        Se, D, stream)

                if use_p1_split_gu and 'gu_il' in fp4_weights[l]:
                    # ── P1 fused-epilogue path: one interleaved GeGLU GEMM ──
                    # The gate/up weight is pairwise interleaved along N and
                    # the epilogue computes gelu(gate)*up per column pair;
                    # the combiner kernel and the separate gate/up GEMMs
                    # disappear. 'epilogue' duplicates the result full-width
                    # (Down reads it via a K-expanded weight below);
                    # 'epilogue_hw' quantizes at compact granularity in the
                    # epilogue and writes Down's stock input buffer directly.
                    w_il = fp4_weights[l]['gu_il']
                    if p1_combiner == 'epilogue_hw':
                        _check(fvk_fp4.cutlass_fp4_gemm_geglu_il_hw(
                            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                            w_il['packed'].data_ptr(), w_il['sfb'].data_ptr(),
                            p1_dummy,
                            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                            Se, 2 * H, D, stream),
                            "cutlass_fp4_gemm_geglu_il_hw", l,
                            M=Se, N_il=2 * H, K=D)
                    else:
                        _check(fvk_fp4.cutlass_fp4_gemm_geglu_il(
                            sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                            w_il['packed'].data_ptr(), w_il['sfb'].data_ptr(),
                            p1_il_p4, p1_il_sfa,
                            Se, 2 * H, D, stream),
                            "cutlass_fp4_gemm_geglu_il", l,
                            M=Se, N_il=2 * H, K=D)
                elif use_p1_split_gu and 'gate' in fp4_weights[l]:
                    # ── P1 split-GU path: 2× fp4out GEMM + geglu_two_fp4 ──
                    w_g = fp4_weights[l]['gate']
                    w_u = fp4_weights[l]['up']
                    _check(fvk_fp4.cutlass_fp4_gemm_fp4out(
                        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                        w_g['packed'].data_ptr(), w_g['sfb'].data_ptr(),
                        p1_gate_p4, p1_gate_sfa,
                        Se, H, D, stream),
                        "cutlass_fp4_gemm_fp4out[gate]", l, M=Se, N=H, K=D)
                    _check(fvk_fp4.cutlass_fp4_gemm_fp4out(
                        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                        w_u['packed'].data_ptr(), w_u['sfb'].data_ptr(),
                        p1_up_p4, p1_up_sfa,
                        Se, H, D, stream),
                        "cutlass_fp4_gemm_fp4out[up]", l, M=Se, N=H, K=D)
                    # silu_mul → fp4 + SFA, write to sc_dn so the Down GEMM
                    # consumes it identically to the non-P1 path. With AWQ,
                    # apply Down inv_s in the same kernel.
                    if awq_dn is None:
                        fvk_fp4.geglu_two_fp4_to_fp4(
                            p1_gate_p4, p1_gate_sfa,
                            p1_up_p4,   p1_up_sfa,
                            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                            Se, H, stream)
                    elif p1_combiner == 'lut':
                        fvk_fp4.geglu_two_mul_fp4_to_fp4_lut(
                            p1_gate_p4, p1_gate_sfa,
                            p1_up_p4,   p1_up_sfa,
                            awq_dn,
                            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                            Se, H, stream)
                    elif p1_combiner == 'lut_native':
                        fvk_fp4.geglu_two_mul_fp4_to_fp4_lut_native(
                            p1_gate_p4, p1_gate_sfa,
                            p1_up_p4,   p1_up_sfa,
                            awq_dn,
                            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                            Se, H, stream)
                    elif p1_combiner == 'direct':
                        fvk_fp4.geglu_two_mul_fp4_to_fp4(
                            p1_gate_p4, p1_gate_sfa,
                            p1_up_p4,   p1_up_sfa,
                            awq_dn,
                            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                            Se, H, stream)
                    else:
                        raise ValueError(
                            f"unknown P1 combiner: {p1_combiner!r}")
                else:
                    # ── Original AWQ-fused (or non-AWQ) path ──
                    _check(fvk_fp4.cutlass_fp4_gemm_variant(
                        variant_gu,
                        sc_gu.packed.data_ptr(), sc_gu.sfa.data_ptr(),
                        w_gu['packed'].data_ptr(), w_gu['sfb'].data_ptr(),
                        gate_fp16_ptr,
                        Se, H * 2, D, 1.0, 0.0, stream),
                        f"cutlass_fp4_gemm_variant[gate_up,v{variant_gu}]", l,
                        M=Se, N=H * 2, K=D)

                    if awq_dn is None:
                        fvk_fp4.gate_geglu_fp4_sfa_v2_fp16(
                            gate_fp16_ptr,
                            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                            Se, H, stream)
                    else:
                        fvk_fp4.gate_geglu_mul_fp4_sfa_v2_fp16(
                            gate_fp16_ptr, awq_dn,
                            sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                            Se, H, stream)

                if (use_p1_split_gu and 'gu_il' in fp4_weights[l]
                        and p1_combiner == 'epilogue'):
                    w_dx = fp4_weights[l]['down_x']
                    _check(fvk_fp4.cutlass_fp4_gemm_variant(
                        variant_dn_x,
                        p1_il_p4, p1_il_sfa,
                        w_dx['packed'].data_ptr(), w_dx['sfb'].data_ptr(),
                        fg_fp16_ptr,
                        Se, D, 2 * H, 1.0, 0.0, stream),
                        f"cutlass_fp4_gemm_variant[down_x,v{variant_dn_x}]", l,
                        M=Se, N=D, K=2 * H)
                else:
                    _check(fvk_fp4.cutlass_fp4_gemm_variant(
                        variant_dn,
                        sc_dn.packed.data_ptr(), sc_dn.sfa.data_ptr(),
                        w_dn['packed'].data_ptr(), w_dn['sfb'].data_ptr(),
                        fg_fp16_ptr,
                        Se, D, H, 1.0, 0.0, stream),
                        f"cutlass_fp4_gemm_variant[down,v{variant_dn}]", l,
                        M=Se, N=D, K=H)

                # 11. residual + RMSNorm → FP8 for next layer (unchanged)
                # Uses fg_fp16 (Down GEMM fp16 output) as the residual delta.
                as_next = act_scales + ((l + 1) * 4 + 0) * 4
                fvk.residual_add_rms_norm_fp8_noweight_fp16(
                    x, fg_fp16_ptr, x_fp8, Se, D, as_next, stream)
            else:
                # ── FP8 path: identical to original encoder_forward ──
                # 7. residual + RMSNorm → FP8
                fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                              Se, D, as_gu, stream)

                # 8. Gate+Up FP8 GEMM (T1 tile)
                fvk.cutlass_fp8_t1(x_fp8, weights['gate_w'][l], gate,
                                   Se, H * 2, D, alpha_host[l * 4 + 2], 0.0, stream)

                # 9. GELU(gate) × up → FP8
                fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, Se, H,
                                                   as_d, stream)

                # 10. Down FP8 GEMM
                fvk.cutlass_fp8_wide(hid_fp8, weights['down_w'][l], fg,
                                      Se, D, H, alpha_host[l * 4 + 3], 0.0, stream)

                # 11. residual + RMSNorm → FP8 for next layer
                as_next = act_scales + ((l + 1) * 4 + 0) * 4
                fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                              Se, D, as_next, stream)


def siglip_forward_with_fp4_ffn(gemm, fvk, fvk_fp4, bufs, weights, dims,
                                stream=0, *, attn=None,
                                fp4_weights: dict = None,
                                fp4_scratch: dict = None):
    """SigLIP vision encoder with the FFN in NVFP4.

    The attention half of every layer is identical to
    ``shared_primitives.siglip_forward`` (FP8 with static scales). The FFN
    runs LayerNorm straight into NVFP4 activations, an Up GEMM with fused
    bias + tanh-GELU + fp4/SFA output, and a Down GEMM with fused bias +
    residual accumulate. The FFN hidden dimension is zero-padded to a
    multiple of 32 (fp4 TMA alignment); pad rows/columns carry zero
    weights and biases so the padding is mathematically inert.

    Args:
        fp4_weights: {layer: {'up': quantized, 'up_bias': ptr,
                              'down': quantized}} where the quantized
            entries are ``quant_weight_nvfp4`` dicts over the PADDED
            hidden dimension and ``up_bias`` points to the padded fp16
            bias. The Down bias reuses ``weights['down_b']``.
        fp4_scratch: {'ln_act': FP4ActScratch(S, D),
                      'hid_act': FP4ActScratch(S, H_pad), 'H_pad': int}
    """
    if fp4_weights is None or fp4_scratch is None:
        raise ValueError(
            "siglip_forward_with_fp4_ffn requires fp4_weights and "
            "fp4_scratch")
    S = dims['S']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    nv = dims['num_views']
    spv = dims['seq_per_view']

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    attn_out = bufs['attn_out']
    hidden = bufs['hidden']
    hid_fp8 = bufs['hid_fp8']

    alpha = weights['alpha']
    sc_ln = fp4_scratch['ln_act']
    sc_hid = fp4_scratch['hid_act']
    H_pad = int(fp4_scratch['H_pad'])

    for l in range(L):
        a_qkv = alpha[l * 4 + 0]
        a_o = alpha[l * 4 + 1]
        a_up = alpha[l * 4 + 2]
        a_down = alpha[l * 4 + 3]

        # ── Attention half: identical to siglip_forward (FP8) ──
        # Vectorized single-pass LayerNorm; falls back to the original
        # kernel on unsupported dims.
        rc = fvk_fp4.layer_norm_fp8_vec_fp16(
            x, weights['ln_attn_w'][l], weights['ln_attn_b'][l], x_fp8,
            S, D, 1e-5, stream)
        if rc != 0:
            fvk.layer_norm_fp8(x, x_fp8, weights['ln_attn_w'][l],
                               weights['ln_attn_b'][l], S, D, 1e-5, stream)
        gemm.fp8_nn_bias(x_fp8, weights['qkv_w'][l], qkv,
                         weights['qkv_b'][l], S, 3 * D, D, a_qkv, stream)
        if attn is not None:
            attn.run("siglip", 0, q_seq=spv, stream=stream)
        else:
            stride = 3 * D
            fvk.fmha_strided_full(qkv, qkv + D * 2, qkv + 2 * D * 2,
                                  attn_out, nv, spv, spv, NH, NH, HD,
                                  stride, stride, stream)
        fvk.quantize_fp8_static_fp16(attn_out, x_fp8,
                                     weights['unit_scale'], S * D, stream)
        gemm.fp8_nn_bias_res(x_fp8, weights['o_w'][l], x,
                             weights['o_b'][l], S, D, D, a_o, stream)

        # ── FFN: NVFP4 when the layer is quantized, FP8 otherwise ──
        w = fp4_weights.get(l)
        if w is None:
            fvk.layer_norm_fp8(x, x_fp8, weights['ln_ffn_w'][l],
                               weights['ln_ffn_b'][l], S, D, 1e-5, stream)
            gemm.fp8_nn_gelu_bias(x_fp8, weights['up_w'][l], hidden,
                                  weights['up_b'][l], S, H, D, a_up, stream)
            fvk.quantize_fp8_static_fp16(hidden, hid_fp8,
                                         weights['unit_scale'], S * H,
                                         stream)
            gemm.fp8_nn_bias_res(hid_fp8, weights['down_w'][l], x,
                                 weights['down_b'][l], S, D, H, a_down,
                                 stream)
            continue
        rc = fvk_fp4.layer_norm_mul_fp4_sfa_vec_fp16(
            x, weights['ln_ffn_w'][l], weights['ln_ffn_b'][l],
            w.get('ln_inv_s', 0),
            sc_ln.packed.data_ptr(), sc_ln.sfa.data_ptr(),
            S, D, 1e-5, stream)
        if rc != 0:
            rc = fvk_fp4.layer_norm_mul_fp4_sfa_fp16(
                x, weights['ln_ffn_w'][l], weights['ln_ffn_b'][l],
                w.get('ln_inv_s', 0),
                sc_ln.packed.data_ptr(), sc_ln.sfa.data_ptr(),
                S, D, 1e-5, stream)
        if rc != 0:
            raise RuntimeError(
                f"SigLIP FP4 FFN LayerNorm layer {l} failed rc={rc}")
        rc = fvk_fp4.cutlass_fp4_gemm_bias_gelu_fp4out(
            sc_ln.packed.data_ptr(), sc_ln.sfa.data_ptr(),
            w['up']['packed'].data_ptr(), w['up']['sfb'].data_ptr(),
            w['up_bias'],
            sc_hid.packed.data_ptr(), sc_hid.sfa.data_ptr(),
            S, H_pad, D, stream)
        if rc != 0:
            raise RuntimeError(
                f"SigLIP FP4 FFN Up GEMM layer {l} failed rc={rc}")
        rc = fvk_fp4.cutlass_fp4_gemm_bias_res_fp16(
            sc_hid.packed.data_ptr(), sc_hid.sfa.data_ptr(),
            w['down']['packed'].data_ptr(), w['down']['sfb'].data_ptr(),
            weights['down_b'][l], x, x, S, D, H_pad, stream)
        if rc != 0:
            raise RuntimeError(
                f"SigLIP FP4 FFN Down GEMM layer {l} failed rc={rc}")

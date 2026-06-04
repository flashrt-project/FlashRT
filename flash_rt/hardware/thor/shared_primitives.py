"""FlashRT -- Thor SM110 model-agnostic primitives, B=1 main-line path.

Closed set of Thor SM110 helpers + forward functions reused across
all models. Per the unified pipeline contract (docs/adding_new_model
.md §0), model-specific forwards must NOT be added here -- they
belong in ``models/<m>/pipeline_thor.py``.

This file holds the **B=1 hot path** (the production single-sample
inference). The B>=1 batched companions live in
:mod:`flash_rt.hardware.thor.shared_primitives_batched`
(``encoder_forward_b2``) so the B=1 file stays small and easy to
reason about — single-sample inference is the main-line product;
batched is opt-in.

In scope (model-agnostic, reused by >= 2 models):

    _gpu_alloc / _gpu_free / _gpu_zero / _gpu_copy / _gpu_sync
    _d2h_float / _d2h_floats
    _measure_scale_gpu           FP8 amax measurement
    siglip_forward               SigLIP vision (pi05/pi0/pi0fast/groot)
    postln_project               Paligemma post-LN + mm proj
                                 (pi05/pi0/pi0fast)
    encoder_forward              Paligemma transformer encoder
                                 (pi05/pi0)
    encoder_forward_calibrate    Paligemma encoder FP8 calibration
                                 (pi05/pi0)

NOT in scope (live elsewhere):

    encoder_forward_b2 (B>=1 batched encoder)
        -> hardware/thor/shared_primitives_batched.py
    Pi0.5 decoder_forward / decoder_forward_calibrate (AdaRMSNorm)
        -> models/pi05/pipeline_thor.py
    Pi0.5 decoder_forward_b2 (B>=1 batched decoder)
        -> models/pi05/pipeline_thor_batched.py
    Pi0 decoder_forward_pi0 / decoder_forward_calibrate_pi0
        -> models/pi0/pipeline_thor.py
    Pi0-FAST autoregressive prefill/decode
        -> models/pi0fast/pipeline.py (deprecated mixed file)
    GROOT Qwen3 / DiT decoders
        -> models/groot/pipeline_thor.py
"""

import math
import os
import numpy as np


# ══════════════════════════════════════════════════════════════════
# SigLIP Vision Encoder (27 layers)
# ══════════════════════════════════════════════════════════════════

def siglip_forward(gemm, fvk, bufs, weights, dims, stream=0, *, attn=None,
                   use_fp8=True):
    """Full SigLIP vision encoder ≡ siglip_forward_fp8_v7.

    Per-view independent attention (v7 strided FMHA).
    LayerNorm → FP8 → QKV GEMM → FMHA → FP8 → O GEMM+res → LN → FP8 → Up GELU → FP8 → Down+res

    Args:
        gemm: GemmRunner instance
        fvk: flash_rt_kernels module
        bufs: dict of GPU buffer pointers
            x (S,D), x_fp8 (S,D), qkv (S,3D), attn_out (S,D),
            hidden (S,H), hid_fp8 (S,H), scratch (S, max(D,H))
            When use_fp8=False: also expects ``x_norm`` (S,D fp16) and
            ``fg`` (S,D fp16) scratch buffers; ``x_fp8`` / ``hid_fp8``
            may be ignored.
        weights: dict — per-layer lists of GPU pointers
            ln_attn_w[L], ln_attn_b[L], qkv_w[L], qkv_b[L],
            o_w[L], o_b[L], ln_ffn_w[L], ln_ffn_b[L],
            up_w[L], up_b[L], down_w[L], down_b[L],
            alpha[L*4] (host floats; unused when use_fp8=False)
        dims: dict — S, D, H, NH, HD, L, num_views, seq_per_view
        use_fp8: When False, run a pure FP16 forward (no FP8 cast, no
            descale alpha; weights are FP16 [K,N] in the same layout).
    """
    if not use_fp8:
        return _siglip_forward_fp16(gemm, fvk, bufs, weights, dims, stream,
                                     attn=attn)
    S = dims['S']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    L = dims['L']
    nv = dims['num_views']
    spv = dims['seq_per_view']  # 256

    x = bufs['x']
    x_fp8 = bufs['x_fp8']
    qkv = bufs['qkv']
    attn_out = bufs['attn_out']
    hidden = bufs['hidden']
    hid_fp8 = bufs['hid_fp8']

    alpha = weights['alpha']  # host float array, len = L*4

    for l in range(L):
        a_qkv = alpha[l * 4 + 0]
        a_o = alpha[l * 4 + 1]
        a_up = alpha[l * 4 + 2]
        a_down = alpha[l * 4 + 3]

        # ── Attention LayerNorm → FP8 ──
        fvk.layer_norm_fp8(x, x_fp8, weights['ln_attn_w'][l], weights['ln_attn_b'][l],
                           S, D, 1e-5, stream)

        # ── QKV GEMM (FP8 → FP16, with bias) ──
        # x_fp8[S,D] @ qkv_w[D,3D] + qkv_b[3D] → qkv[S,3D]
        gemm.fp8_nn_bias(x_fp8, weights['qkv_w'][l], qkv, weights['qkv_b'][l],
                         S, 3 * D, D, a_qkv, stream)

        # ── Strided FMHA (per-view independent attention) ──
        # qkv[S, 3D] with Q/K/V interleaved: Q at offset 0, K at offset D, V at offset 2D
        # Each view has spv=256 tokens, nv views treated as batch
        scale = 1.0 / math.sqrt(float(HD))
        if attn is not None:
            # Stage 1.3: dispatch through AttentionBackend protocol.
            # Backend derives Q/K/V from the interleaved qkv buffer and
            # writes into attn_out (same fvk call under the hood).
            attn.run("siglip", 0, q_seq=spv, stream=stream)
        else:
            stride = 3 * D  # QKV interleaved stride
            Q_ptr = qkv
            K_ptr = qkv + D * 2    # byte offset for fp16
            V_ptr = qkv + 2 * D * 2
            fvk.fmha_strided_full(Q_ptr, K_ptr, V_ptr, attn_out,
                                  nv, spv, spv, NH, NH, HD,
                                  stride, stride, stream)

        # ── Cast attention output → FP8 (scale=1.0, LN output already normalized) ──
        fvk.quantize_fp8_static_fp16(attn_out, x_fp8, weights['unit_scale'], S * D, stream)

        # ── O projection + residual ──
        # x[S,D] += alpha * x_fp8[S,D] @ o_w[D,D] + o_b[D]
        gemm.fp8_nn_bias_res(x_fp8, weights['o_w'][l], x, weights['o_b'][l],
                             S, D, D, a_o, stream)

        # ── FFN LayerNorm → FP8 ──
        fvk.layer_norm_fp8(x, x_fp8, weights['ln_ffn_w'][l], weights['ln_ffn_b'][l],
                           S, D, 1e-5, stream)

        # ── Up GEMM with fused GELU + bias ──
        # hidden[S,H] = GELU(alpha * x_fp8[S,D] @ up_w[D,H] + up_b[H])
        gemm.fp8_nn_gelu_bias(x_fp8, weights['up_w'][l], hidden, weights['up_b'][l],
                              S, H, D, a_up, stream)

        # ── Cast FFN hidden → FP8 (scale=1.0) ──
        fvk.quantize_fp8_static_fp16(hidden, hid_fp8, weights['unit_scale'], S * H, stream)

        # ── Down GEMM + residual ──
        # x[S,D] += alpha * hid_fp8[S,H] @ down_w[H,D] + down_b[D]
        gemm.fp8_nn_bias_res(hid_fp8, weights['down_w'][l], x, weights['down_b'][l],
                             S, D, H, a_down, stream)

    # x[S, D] now contains final SigLIP output


# ══════════════════════════════════════════════════════════════════
# PostLN + Projection + Language Concat
# ══════════════════════════════════════════════════════════════════

def postln_project(gemm, fvk, bufs, weights, dims, stream=0):
    """Post-SigLIP LayerNorm + projection + language concat ≡ postln_project_concat.

    LayerNorm(siglip_out) → projection [D_sig → D_enc] → concat with lang_emb

    Args:
        bufs: x_sig (S_sig, D_sig), enc_x (Se, D_enc), scratch (S_sig, D_sig)
        weights: ln_w (D_sig,), ln_b (D_sig,), proj_w (D_enc, D_sig), proj_b (D_enc,), lang_emb
        dims: S_sig, D_sig, D_enc, S_lang
    """
    import ctypes

    S_sig = dims['S_sig']
    D_sig = dims['D_sig']
    D_enc = dims['D_enc']
    S_lang = dims['S_lang']

    x_sig = bufs['x_sig']
    enc_x = bufs['enc_x']
    scratch = bufs['scratch']

    # Step 1: LayerNorm
    fvk.layer_norm_fp16(x_sig, weights['ln_w'], weights['ln_b'], scratch,
                        S_sig, D_sig, 1e-6, stream)

    # Step 2: Projection [S_sig, D_sig] → [S_sig, D_enc]
    # proj_w stored as [D_sig, D_enc] (pre-transposed for NN layout)
    gemm.fp16_nn(scratch, weights['proj_w'], enc_x, S_sig, D_enc, D_sig, stream)
    fvk.add_bias_fp16(enc_x, weights['proj_b'], S_sig, D_enc, stream)

    # Step 3: Concat language embeddings after projected vision
    # enc_x[S_sig : S_sig+S_lang, :] = lang_emb[S_lang, D_enc]
    nbytes = S_lang * D_enc * 2  # fp16
    dst = enc_x + S_sig * D_enc * 2  # byte offset
    _crt.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(weights['lang_emb']),
                          ctypes.c_size_t(nbytes), 3, ctypes.c_void_p(stream))


# ══════════════════════════════════════════════════════════════════
# Encoder (18 layers, GQA, static FP8)
# ══════════════════════════════════════════════════════════════════

def _make_ones(D):
    """Create a ones tensor on GPU (4KB, negligible cost)."""
    import torch
    return torch.ones(D, dtype=torch.float16, device='cuda')

def _siglip_forward_fp16(gemm, fvk, bufs, weights, dims, stream=0, *, attn=None):
    """FP16-only SigLIP forward. Mirrors siglip_forward structure with
    every FP8 op split into its FP16 equivalent.
    """
    S = dims['S']; D = dims['D']; H = dims['H']
    NH = dims['NH']; HD = dims['HD']; L = dims['L']
    nv = dims['num_views']
    spv = dims['seq_per_view']

    x = bufs['x']
    x_norm = bufs['x_norm']       # S*D fp16 scratch (FP16 path only)
    qkv = bufs['qkv']
    attn_out = bufs['attn_out']
    hidden = bufs['hidden']
    fg = bufs['fg']               # S*D fp16 scratch for O/Down GEMM output

    # SigLIP FP16 weights stay in [K, N] row-major (cuBLAS NN convention).
    # Tested 2026-05-18 swapping to CUTLASS NT (cutlass_fp16_wide) after
    # dropping T() in the spec — bit-exact correctness (cos=1.0) but no
    # net hot-regime win at Pi0.5 SigLIP shape (84.13 vs 83.87 cublas, in
    # the ±0.5 ms noise band).  Keep cuBLAS for now.
    for l in range(L):
        # ── Attention LayerNorm → x_norm ──
        fvk.layer_norm_fp16(x, weights['ln_attn_w'][l], weights['ln_attn_b'][l],
                            x_norm, S, D, 1e-5, stream)

        # ── QKV GEMM (FP16 NN) + bias ──
        gemm.fp16_nn(x_norm, weights['qkv_w'][l], qkv, S, 3 * D, D, stream)
        fvk.add_bias_fp16(qkv, weights['qkv_b'][l], S, 3 * D, stream)

        # ── Strided FMHA ──
        scale = 1.0 / math.sqrt(float(HD))
        if attn is not None:
            attn.run("siglip", 0, q_seq=spv, stream=stream)
        else:
            stride = 3 * D
            Q_ptr = qkv
            K_ptr = qkv + D * 2
            V_ptr = qkv + 2 * D * 2
            fvk.fmha_strided_full(Q_ptr, K_ptr, V_ptr, attn_out,
                                  nv, spv, spv, NH, NH, HD,
                                  stride, stride, stream)

        # ── O projection + residual + bias ──
        gemm.fp16_nn(attn_out, weights['o_w'][l], fg, S, D, D, stream)
        fvk.bias_residual_strict_fp16(x, fg, weights['o_b'][l], S, D, stream)

        # ── FFN LayerNorm → x_norm ──
        fvk.layer_norm_fp16(x, weights['ln_ffn_w'][l], weights['ln_ffn_b'][l],
                            x_norm, S, D, 1e-5, stream)

        # ── Up GEMM + bias + GELU ──
        gemm.fp16_nn(x_norm, weights['up_w'][l], hidden, S, H, D, stream)
        fvk.add_bias_fp16(hidden, weights['up_b'][l], S, H, stream)
        fvk.gelu_inplace_fp16(hidden, S * H, stream)

        # ── Down GEMM + residual + bias ──
        gemm.fp16_nn(hidden, weights['down_w'][l], fg, S, D, H, stream)
        fvk.bias_residual_strict_fp16(x, fg, weights['down_b'][l], S, D, stream)


def _encoder_forward_fp16(gemm, fvk, bufs, weights, dims, stream=0, *, attn=None):
    """FP16-only Paligemma encoder forward. Mirrors encoder_forward
    structure with FP8 cast/descale dropped. GEMMs use the CUTLASS-FP16
    NT path (weights stored ``[N, K]`` row-major, matching the FP8 NT
    convention) rather than cuBLASLt; FP8-equivalent shape-specific
    tactics (sq/t1/wide) are chosen per-GEMM.
    """
    Se = dims['Se']; D = dims['D']; H = dims['H']
    NH = dims['NH']; HD = dims['HD']; L = dims['L']
    total_keys = dims['total_keys']
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    x = bufs['x']
    x_norm = bufs['x_norm']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    gate = bufs['gate']
    hidden = bufs['hidden']
    fg = bufs['fg']
    ones = bufs['ones']

    # Kernel-dispatch env switches, read once per call so graph capture sees
    # a fixed path. SKIP_LAST_QKV is kept off by default: setting it to 1 was
    # tested 2026-05-18 and showed inconsistent (sometimes reversed) timing,
    # suggesting it also affects downstream output via the decoder's K/V cache.
    skip_last_qkv = os.environ.get("SKIP_LAST_QKV", "0") == "1"
    qkv_kernel = os.environ.get("QKV_KERNEL", "k64")
    use_rms_qkv_bundle = os.environ.get("USE_RMS_QKV_BUNDLE", "1") == "1"
    enc_path = os.environ.get("ENC_PATH", "head")
    use_ffn_block = os.environ.get("USE_FFN_BLOCK", "0") == "1"
    o_kernel = os.environ.get("O_KERNEL", "plain")
    g8_kernel = os.environ.get("G8_KERNEL", "bundled")

    for l in range(L):
        last = (l == L - 1)

        if not last or not skip_last_qkv:
            # 1+2. RMSNorm + QKV (k64) — env-switchable bundle.
            _qkv = qkv_kernel
            if use_rms_qkv_bundle and _qkv == "k64":
                fvk.flashrt_rms_qkv_fp16(
                    x, ones, x_norm,
                    weights['qkv_w'][l], qkv,
                    Se, D, 2560, 1e-6, stream)
            else:
                fvk.rms_norm_fp16(x, ones, x_norm, Se, D, 1e-6, stream)
                if   _qkv == "sq":    fvk.cutlass_fp16_sq   (x_norm, weights['qkv_w'][l], qkv, Se, 2560, D, 1.0, 0.0, stream)
                elif _qkv == "wide":  fvk.cutlass_fp16_wide (x_norm, weights['qkv_w'][l], qkv, Se, 2560, D, 1.0, 0.0, stream)
                elif _qkv == "t1":    fvk.cutlass_fp16_t1   (x_norm, weights['qkv_w'][l], qkv, Se, 2560, D, 1.0, 0.0, stream)
                elif _qkv == "plain": fvk.cutlass_fp16_plain(x_norm, weights['qkv_w'][l], qkv, Se, 2560, D, 1.0, 0.0, stream)
                else:                 fvk.cutlass_fp16_k64  (x_norm, weights['qkv_w'][l], qkv, Se, 2560, D, 1.0, 0.0, stream)

            # 3+4. Split+RoPE+KV
            kv_elem_off = l * total_keys * HD
            fvk.qkv_split_rope_kvcache_fp16(
                qkv, weights['rope'], attn_out,
                weights['Kc'], weights['Vc'],
                Se, Q_dim, K_dim, HD, 2560,
                kv_elem_off, HD, stream)
        else:
            kv_elem_off = l * total_keys * HD  # unused but kept for code-flow parity

        if not last:
            # 5. Attention
            if attn is not None:
                attn.run("encoder", l, q_seq=Se, stream=stream)
            else:
                K_ptr = weights['Kc'] + kv_elem_off * 2
                V_ptr = weights['Vc'] + kv_elem_off * 2
                fvk.attention_qkv_fp16(bufs['ctx'], attn_out, K_ptr, V_ptr,
                                        logits, attn_out,
                                        Se, Se, NH, HD, attn_scale, stream)

            # Env-switch ENC_PATH: "pre_mk1" runs the unfused split GEMMs +
            # separate residual_add (the pre-megakernel baseline).  Default
            # "head" fuses the O-proj residual and runs rms + the GeGLU
            # megakernel + down GEMM with the residual folded in.
            up_w_ptr = weights['gate_w'][l] + H * D * 2
            if enc_path == "pre_mk1":
                # 6. O proj (no resid fuse)
                fvk.cutlass_fp16_k64(attn_out, weights['o_w'][l], fg,
                                      Se, D, D, 1.0, 0.0, stream)
                # 7. resid + rmsnorm
                fvk.residual_add_fp16(x, fg, Se * D, stream)
                fvk.rms_norm_fp16(x, ones, x_norm, Se, D, 1e-6, stream)
                # 8. sq_gelu(x_norm @ W_gate) → gate
                fvk.cutlass_fp16_sq_gelu(x_norm, weights['gate_w'][l],
                                          gate, Se, H, D, 1.0, 0.0, stream)
                # 9. sq(x_norm @ W_up) → bufs['hidden'] as up scratch
                fvk.cutlass_fp16_sq(x_norm, up_w_ptr, hidden,
                                     Se, H, D, 1.0, 0.0, stream)
                # mul: gate * hidden → hidden (use hidden as in-place mul out)
                fvk.mul_fp16(gate, hidden, hidden, Se * H, stream)
                # 10. 2sm21 G8 (no resid fuse)
                fvk.cutlass_fp16_2sm21(hidden, weights['down_w'][l], fg,
                                        Se, D, H, 1.0, 0.0, stream)
                # 11. resid add
                fvk.residual_add_fp16(x, fg, Se * D, stream)
            else:
                # head: O resid fuse + (rms + mega + G8 resid fuse).
                # plain won 2026-05-18 sweep at FP16 Pi0.5 O shape
                # (Se=768 N=K=D=2048): -0.5 ms hot regime vs k64.
                #
                # USE_FFN_BLOCK=1: bundle rms+mega+G8 in one C entry
                # (flashrt_encoder_ffn_block_fp16).  Default 0 = current
                # head with rms separate and mega+G8 bundled.
                _ffn_bundle = use_ffn_block
                _o = o_kernel
                if _o == "sq":
                    fvk.cutlass_fp16_sq   (attn_out, weights['o_w'][l], x, Se, D, D, 1.0, 1.0, stream)
                elif _o == "wide":
                    fvk.cutlass_fp16_wide (attn_out, weights['o_w'][l], x, Se, D, D, 1.0, 1.0, stream)
                elif _o == "t1":
                    fvk.cutlass_fp16_t1   (attn_out, weights['o_w'][l], x, Se, D, D, 1.0, 1.0, stream)
                elif _o == "plain":
                    fvk.cutlass_fp16_plain(attn_out, weights['o_w'][l], x, Se, D, D, 1.0, 1.0, stream)
                else:
                    fvk.cutlass_fp16_k64  (attn_out, weights['o_w'][l], x, Se, D, D, 1.0, 1.0, stream)
                if _ffn_bundle:
                    fvk.flashrt_encoder_ffn_block_fp16(
                        x, ones, x_norm,
                        weights['gate_w'][l], up_w_ptr, weights['down_w'][l],
                        gate, hidden,
                        Se, H, D, 1e-6, stream)
                    continue
                fvk.rms_norm_fp16(x, ones, x_norm, Se, D, 1e-6, stream)
                # Bundled GeGLU megakernel + down GEMM (2sm21 beta=1) in one C
                # entry.  The bundle wins ~0.5 ms in the hot regime over the
                # same two launches issued separately.  G8 kernel via env (bundled
                # entry always uses 2sm21 internally; G8_KERNEL switches
                # to unbundled python path with a different kernel).
                _g8 = g8_kernel
                if _g8 == "bundled":
                    fvk.flashrt_megakernel_geglu_g8_fp16(
                        x_norm, weights['gate_w'][l], up_w_ptr,
                        weights['down_w'][l],
                        hidden, x,
                        Se, H, D, stream)
                else:
                    fvk.flashrt_megakernel_geglu_fp16(
                        x_norm, weights['gate_w'][l], up_w_ptr,
                        gate, hidden, Se, H, D, stream)
                    if   _g8 == "sq":    fvk.cutlass_fp16_sq   (hidden, weights['down_w'][l], x, Se, D, H, 1.0, 1.0, stream)
                    elif _g8 == "wide":  fvk.cutlass_fp16_wide (hidden, weights['down_w'][l], x, Se, D, H, 1.0, 1.0, stream)
                    elif _g8 == "k64":   fvk.cutlass_fp16_k64  (hidden, weights['down_w'][l], x, Se, D, H, 1.0, 1.0, stream)
                    elif _g8 == "plain": fvk.cutlass_fp16_plain(hidden, weights['down_w'][l], x, Se, D, H, 1.0, 1.0, stream)
                    elif _g8 == "t1":    fvk.cutlass_fp16_t1   (hidden, weights['down_w'][l], x, Se, D, H, 1.0, 1.0, stream)
                    else:                fvk.cutlass_fp16_2sm21(hidden, weights['down_w'][l], x, Se, D, H, 1.0, 1.0, stream)


def encoder_forward(gemm, fvk, bufs, weights, dims, stream=0, *, attn=None,
                    use_fp8=True):
    """Full encoder forward ≡ encoder_full_static_cutlass.

    Static FP8 descale scheme:
      - RMSNorm→FP8 with per-layer calibrated act_scale
      - CUTLASS GEMM with alpha = alpha_host (act_scale * w_scale, precomputed)
      - quantize_fp8_static with act_scale
      - GELU(gate) × up with act_scale
      - All descaling fused into GEMM alpha

    Args:
        gemm: GemmRunner
        fvk: flash_rt_kernels module
        bufs: dict — x, x_fp8, qkv, logits, attn_out, o_fp8, gate, hidden, hid_fp8, fg, ctx
        weights: dict — qkv_w[L], o_w[L], gate_w[L], down_w[L], rope, Kc, Vc,
                        act_scales (device float ptr), alpha_host (host list[L*4])
        dims: dict — Se, D, H, NH, HD, L, total_keys
        use_fp8: When False, run a pure FP16 forward. Bufs must
            additionally provide ``x_norm`` (Se*D fp16) and ``ones``
            (D fp16, all 1.0) for noweight RMSNorm; weights must point
            at FP16 [K,N] tensors and ``act_scales``/``alpha_host`` are
            ignored.
    """
    if not use_fp8:
        return _encoder_forward_fp16(gemm, fvk, bufs, weights, dims, stream,
                                      attn=attn)
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
    gate = bufs['gate']   # [Se, 2H] for merged gate+up output
    hid_fp8 = bufs['hid_fp8']
    fg = bufs['fg']

    act_scales = weights['act_scales']  # device float ptr (base of calib tensor)
    alpha_host = weights['alpha_host']  # host float list [L*4]

    for l in range(L):
        last = (l == L - 1)

        # Per-layer act_scale device pointers (float32 = 4 bytes each)
        as_qkv = act_scales + (l * 4 + 0) * 4
        as_o   = act_scales + (l * 4 + 1) * 4
        as_gu  = act_scales + (l * 4 + 2) * 4
        as_d   = act_scales + (l * 4 + 3) * 4

        # ── 1. RMSNorm → FP8 with act_scale (noweight, matches production) ──
        fvk.rms_norm_fp8_noweight_fp16(x, x_fp8, Se, D, as_qkv, stream)

        # ── 2. QKV GEMM (alpha = act_scale * w_scale) ──
        fvk.cutlass_fp8_sq(x_fp8, weights['qkv_w'][l], qkv,
                           Se, 2560, D, alpha_host[l * 4 + 0], 0.0, stream)

        # ── 3+4. Fused: QKV split + RoPE + KV cache write (FP16) ──
        kv_elem_off = l * total_keys * HD  # element offset for kernel
        fvk.qkv_split_rope_kvcache_fp16(
            qkv, weights['rope'], attn_out,
            weights['Kc'], weights['Vc'],
            Se, Q_dim, K_dim, HD, 2560,
            kv_elem_off, HD, stream)

        if not last:
            # ── 5. Attention (cuBLAS) ──
            if attn is not None:
                attn.run("encoder", l, q_seq=Se, stream=stream)
            else:
                K_ptr = weights['Kc'] + kv_elem_off * 2  # byte offset (fp16)
                V_ptr = weights['Vc'] + kv_elem_off * 2
                fvk.attention_qkv_fp16(bufs['ctx'], attn_out, K_ptr, V_ptr,
                                        logits, attn_out,
                                        Se, Se, NH, HD, attn_scale, stream)

            # ── 6. Quantize attn→FP8 with act_scale + O proj GEMM ──
            fvk.quantize_fp8_static_fp16(attn_out, o_fp8, as_o, Se * D, stream)
            fvk.cutlass_fp8_sq(o_fp8, weights['o_w'][l], fg,
                               Se, D, D, alpha_host[l * 4 + 1], 0.0, stream)

            # ── 7. Residual + RMSNorm → FP8 with act_scale (noweight) ──
            fvk.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                          Se, D, as_gu, stream)

            # ── 8. Gate+Up merged GEMM (T1 tile for L2 optimization) ──
            fvk.cutlass_fp8_t1(x_fp8, weights['gate_w'][l], gate,
                               Se, H * 2, D, alpha_host[l * 4 + 2], 0.0, stream)

            # ── 9. GELU(gate) × up → FP8 with act_scale ──
            fvk.gate_geglu_merged_fp8_fp16(gate, hid_fp8, Se, H,
                                               as_d, stream)

            # ── 10. Down GEMM ──
            fvk.cutlass_fp8_wide(hid_fp8, weights['down_w'][l], fg,
                                  Se, D, H, alpha_host[l * 4 + 3], 0.0, stream)

            # ── 11. Residual writeback. The next layer recomputes C1
            # RMSNorm→FP8, so no FP8 output is consumed here.
            fvk.residual_add_fp16(x, fg, Se * D, stream)

    # x[Se, D] now contains final encoder output



# Pi0.5-specific decoder_forward / decoder_forward_calibrate moved to
# models/pi05/pipeline_thor.py (per unified pipeline_<hw>.py contract).
# Pi0-specific decoder fns live in models/pi0/pipeline_thor.py.


# ══════════════════════════════════════════════════════════════════
# Calibration helpers (framework-agnostic, pure pointer ops)
# ══════════════════════════════════════════════════════════════════

import ctypes
_crt = ctypes.CDLL('libcudart.so')


def _gpu_alloc(nbytes):
    """cudaMalloc → return int pointer."""
    ptr = ctypes.c_void_p()
    _crt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
    return ptr.value


def _gpu_free(ptr):
    _crt.cudaFree(ctypes.c_void_p(ptr))


def _gpu_zero(ptr, nbytes, stream=0):
    _crt.cudaMemsetAsync(ctypes.c_void_p(ptr), 0, ctypes.c_size_t(nbytes),
                         ctypes.c_void_p(stream))


def _gpu_copy(dst, src, nbytes, stream=0):
    """Device-to-device copy."""
    _crt.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                          ctypes.c_size_t(nbytes), 3, ctypes.c_void_p(stream))


def _gpu_sync(stream=0):
    if stream:
        _crt.cudaStreamSynchronize(ctypes.c_void_p(stream))
    else:
        _crt.cudaDeviceSynchronize()


def _d2h_float(device_ptr, stream=0):
    """Read single float32 from device → host."""
    val = ctypes.c_float()
    _crt.cudaMemcpy(ctypes.byref(val), ctypes.c_void_p(device_ptr),
                    ctypes.c_size_t(4), 2)  # D2H
    return val.value


def _d2h_floats(device_ptr, count, stream=0):
    """Read float32 array from device → host list."""
    arr = (ctypes.c_float * count)()
    _crt.cudaMemcpy(arr, ctypes.c_void_p(device_ptr),
                    ctypes.c_size_t(count * 4), 2)
    return [float(arr[i]) for i in range(count)]


def _measure_scale_gpu(fvk_mod, fp16_ptr, n_elements, d_scale_ptr, d_fp8_scratch, stream=0):
    """GPU-only amax measurement: absmax → compute_scale → d_scale.

    Uses quantize_fp8_device_fp16 which does absmax + scale + quantize.
    The quantize output goes to d_fp8_scratch (discarded), scale goes to d_scale_ptr.
    """
    fvk_mod.quantize_fp8_device_fp16(fp16_ptr, d_fp8_scratch, d_scale_ptr, n_elements, stream)


def encoder_forward_calibrate(gemm, fvk_mod, bufs, weights, dims,
                              calib_scales_ptr, stream=0, *, attn=None):
    """Calibrate encoder FP8 scales. Framework-agnostic (pure pointers).

    Two-pass per quantization point:
      1. FP16 kernel → measure amax on GPU → compute scale
      2. FP8 kernel with that scale (identical to inference)
    """
    Se = dims['Se']; D = dims['D']; H = dims['H']
    NH = dims['NH']; HD = dims['HD']; L = dims['L']
    total_keys = dims['total_keys']
    Q_dim = NH * HD; K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    x = bufs['x']; x_fp8 = bufs['x_fp8']; qkv = bufs['qkv']
    logits = bufs['logits']; attn_out = bufs['attn_out']
    o_fp8 = bufs['o_fp8']; gate = bufs['gate']
    hidden = bufs['hidden']; hid_fp8 = bufs['hid_fp8']; fg = bufs['fg']

    w_scales_dev = weights['w_scales']

    # Read w_scales to host (ctypes D2H)
    ws_host = _d2h_floats(w_scales_dev, L * 4)

    # Scratch buffers — provided by caller via bufs dict
    norm_scratch = bufs['norm_scratch']    # Se*D fp16
    x_scratch = bufs['x_scratch']          # Se*D fp16
    calib_buf = bufs['calib_buf']          # L*4 float32
    d_scale = bufs['d_scale']              # 1 float32
    fp8_scratch = bufs['fp8_scratch']      # Se*max(D,H) fp8
    ones_buf = bufs['ones']                # D fp16 (all 1.0)
    _gpu_zero(calib_buf, L * 4 * 4, stream)
    for l in range(L):
        last = (l == L - 1)

        # 1. RMSNorm FP16 → measure amax → scale
        fvk_mod.rms_norm_fp16(x, ones_buf, norm_scratch, Se, D, 1e-6, stream)
        _gpu_sync(stream)
        _measure_scale_gpu(fvk_mod, norm_scratch, Se * D, d_scale, fp8_scratch, stream)
        _gpu_sync(stream)
        as_qkv = _d2h_float(d_scale)
        cs_qkv = calib_buf + (l * 4 + 0) * 4
        _gpu_copy(cs_qkv, d_scale, 4, stream)

        # RMSNorm → FP8 with calibrated scale
        fvk_mod.rms_norm_fp8_noweight_fp16(x, x_fp8, Se, D, cs_qkv, stream)

        # 2. QKV GEMM
        alpha_qkv = float(np.float32(as_qkv) * np.float32(ws_host[l * 4 + 0]))
        fvk_mod.cutlass_fp8_sq(x_fp8, weights['qkv_w'][l], qkv,
                                Se, 2560, D, alpha_qkv, 0.0, stream)

        # 3. Split+RoPE+KV cache
        kv_off = l * total_keys * HD
        fvk_mod.qkv_split_rope_kvcache_fp16(qkv, weights['rope'], attn_out,
                                             weights['Kc'], weights['Vc'],
                                             Se, Q_dim, K_dim, HD, 2560,
                                             kv_off, HD, stream)

        if not last:
            # 5. Attention
            if attn is not None:
                attn.run("encoder", l, q_seq=Se, stream=stream)
            else:
                K_ptr = weights['Kc'] + kv_off * 2
                V_ptr = weights['Vc'] + kv_off * 2
                fvk_mod.attention_qkv_fp16(bufs['ctx'], attn_out, K_ptr, V_ptr,
                                            logits, attn_out,
                                            Se, Se, NH, HD, attn_scale, stream)

            # 6. O proj: measure attn amax → quantize → GEMM
            _measure_scale_gpu(fvk_mod, attn_out, Se * Q_dim, d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            as_o = _d2h_float(d_scale)
            cs_o = calib_buf + (l * 4 + 1) * 4
            _gpu_copy(cs_o, d_scale, 4, stream)
            fvk_mod.quantize_fp8_static_fp16(attn_out, o_fp8, cs_o, Se * D, stream)
            alpha_o = float(np.float32(as_o) * np.float32(ws_host[l * 4 + 1]))
            fvk_mod.cutlass_fp8_sq(o_fp8, weights['o_w'][l], fg,
                                    Se, D, D, alpha_o, 0.0, stream)

            # 7. Residual + RMSNorm → FP8
            _gpu_copy(x_scratch, x, Se * D * 2, stream)
            fvk_mod.residual_add_fp16(x_scratch, fg, Se * D, stream)
            fvk_mod.rms_norm_fp16(x_scratch, ones_buf, norm_scratch, Se, D, 1e-6, stream)
            _measure_scale_gpu(fvk_mod, norm_scratch, Se * D, d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            as_gu = _d2h_float(d_scale)
            cs_gu = calib_buf + (l * 4 + 2) * 4
            _gpu_copy(cs_gu, d_scale, 4, stream)
            fvk_mod.residual_add_rms_norm_fp8_noweight_fp16(x, fg, x_fp8,
                                                              Se, D, cs_gu, stream)

            # 8. Gate+Up GEMM
            alpha_gu = float(np.float32(as_gu) * np.float32(ws_host[l * 4 + 2]))
            fvk_mod.cutlass_fp8_t1(x_fp8, weights['gate_w'][l], gate,
                                    Se, H * 2, D, alpha_gu, 0.0, stream)

            # 9. GELU FP16 → measure → FP8
            fvk_mod.gate_geglu_merged_fp16(gate, hidden, Se, H, stream)
            _measure_scale_gpu(fvk_mod, hidden, Se * H, d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            as_down = _d2h_float(d_scale)
            cs_down = calib_buf + (l * 4 + 3) * 4
            _gpu_copy(cs_down, d_scale, 4, stream)
            fvk_mod.gate_geglu_merged_fp8_fp16(gate, hid_fp8, Se, H, cs_down, stream)

            # 10. Down GEMM
            alpha_down = float(np.float32(as_down) * np.float32(ws_host[l * 4 + 3]))
            fvk_mod.cutlass_fp8_wide(hid_fp8, weights['down_w'][l], fg,
                                      Se, D, H, alpha_down, 0.0, stream)

            # 11. Match inference residual writeback. The next layer
            # calibration step measures and quantizes its own C1 input.
            fvk_mod.residual_add_fp16(x, fg, Se * D, stream)

    # Copy calib scales to output
    _gpu_copy(calib_scales_ptr, calib_buf, L * 4 * 4, stream)
    _gpu_sync(stream)

    # Free scratch

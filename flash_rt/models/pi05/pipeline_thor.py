"""FlashRT — Pi0.5 Thor SM110 decoder pipeline (B=1, main-line path).

Pi0.5-specific Thor compute (AdaRMSNorm action expert decoder + the
calibration twin). SigLIP / encoder live in
``hardware/thor/shared_primitives`` because they are reused by Pi0,
GROOT, and Pi0-FAST. ``decoder_forward`` is Pi0.5-specific
(AdaRMSNorm with style modulation; Pi0 and GROOT have different
decoders) so it must NOT live in shared_primitives — see the unified
pipeline_<hw>.py contract in docs/adding_new_model.md §0.

This file holds the **B=1 main-line single-sample inference path**.
The B>=1 batched companion lives in
:mod:`flash_rt.models.pi05.pipeline_thor_batched` (mirrors the RTX
``pipeline_rtx`` / ``pipeline_rtx_batched`` split). The CFG variants
live in :mod:`flash_rt.models.pi05.pipeline_thor_cfg` (serial) and
:mod:`flash_rt.models.pi05.pipeline_thor_cfg_batched` (B=2 fused).

Functions:
    decoder_forward            — Pi0.5 decoder inference (static FP8)
    decoder_forward_calibrate  — Pi0.5 decoder FP8 scale calibration

Classes:
    Pi05ThorPipeline           — B=1 facade for SigLIP + enc_ae replay
"""

import math
import torch

from flash_rt.hardware.thor.shared_primitives import (
    _measure_scale_gpu,
    _gpu_copy,
    _gpu_sync,
    _gpu_zero,
)


# ══════════════════════════════════════════════════════════════════
# Decoder (18 layers, 10 diffusion steps, static FP8)
# ══════════════════════════════════════════════════════════════════

def decoder_forward(ctx, fvk, bufs, weights, dims, stream=0, *, attn=None,
                    use_fp8=True):
    """Full AE decoder forward pass ≡ pi05 ae_forward_static.

    Args:
        ctx: FvkContext (C++ object with cuBLAS handle)
        fvk: flash_rt_kernels module
        bufs: dict of GPU buffer pointers (uintptr_t)
            noise, x, xn, gate, qkv, logits, attn_out, hid, fg,
            xn_fp8, hid_fp8, ctx_fp8     [xn_fp8/hid_fp8/ctx_fp8 only when use_fp8=True]
        weights: dict of GPU buffer pointers
            ain_w, ain_b, sa, qw, Kc, Vc, ow, sf, gw, dw,
            aow, aob, fs, rope, w_scales, act_scales
            (w_scales/act_scales unused when use_fp8=False — weights are FP16)
        dims: dict
            S, D, H, NH, HD, steps, layers, enc_seq, total_keys
        stream: CUDA stream (int)
        use_fp8: When True (default) run the production static-FP8 path.
            When False, every GEMM uses ``fvk.gmm_fp16`` and the fused
            FP8 norm/quantize kernels are split into their FP16
            equivalents (``adarms_fp16``, ``gate_res_fp16``,
            ``gate_geglu_merged_fp16``). Used for FP16 baseline runs.
    """
    if not use_fp8:
        return _decoder_forward_fp16(ctx, fvk, bufs, weights, dims, stream,
                                      attn=attn)
    S = dims['S']
    D = dims['D']
    H = dims['H']
    NH = dims['NH']
    HD = dims['HD']
    steps = dims['steps']
    layers = dims['layers']
    enc_seq = dims['enc_seq']
    total_keys = dims['total_keys']
    D3 = 3 * D
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    # Buffer pointers
    noise = bufs['noise']
    x = bufs['x']
    xn = bufs['xn']
    gate = bufs['gate']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    hid = bufs['hid']
    fg = bufs['fg']
    noise_t = bufs.get('noise_t')
    xn_t = bufs.get('xn_t')
    delta_t = bufs.get('delta_t')
    xn_f32_t = bufs.get('xn_f32_t')
    delta_f32_t = bufs.get('delta_f32_t')
    xn_fp8 = bufs['xn_fp8']
    hid_fp8 = bufs['hid_fp8']
    ctx_fp8 = bufs['ctx_fp8']

    # Weight pointers
    ain_w = weights['ain_w']
    ain_b = weights['ain_b']
    sa = weights['sa']
    qw = weights['qw']
    Kc = weights['Kc']
    Vc = weights['Vc']
    ow = weights['ow']
    sf = weights['sf']
    gw = weights['gw']
    dw = weights['dw']
    aow = weights['aow']
    aob = weights['aob']
    aow_t = weights.get('aow_t')
    aob_t = weights.get('aob_t')
    aow_f32_t = weights.get('aow_f32_t')
    aob_f32_t = weights.get('aob_f32_t')
    dt_t = weights.get('dt_t')
    fs = weights['fs']
    rope = weights['rope']
    w_scales = weights['w_scales']
    act_scales = weights['act_scales']

    for s in range(steps):
        step_scale_base = s * layers * 4
        # ── Action input: noise → x ──
        fvk.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream)
        fvk.add_bias_fp16(x, ain_b, S, D, stream)

        for l in range(layers):
            si = (s * layers + l) * S * D3
            sa_ptr = sa + si * 2
            sf_ptr = sf + si * 2

            # ── C1: Fused AdaRMSNorm → FP8 with static scale ──
            act_scale_qkv = act_scales + (step_scale_base + l * 4 + 0) * 4
            if l == 0:
                fvk.fused_adarms_fp8_static_fp16(
                    x, sa_ptr, xn_fp8, gate, S, D, act_scale_qkv, stream)

            # ── C2: QKV GEMM with descale ──
            w_scale_qkv = w_scales + (l * 4 + 0) * 4
            qw_ptr = qw + l * D * 2560
            fvk.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D,
                                       act_scale_qkv, w_scale_qkv, stream)

            # ── C2b: Fused RoPE + QKV split + KV cache ──
            kv_offset = l * total_keys * HD + enc_seq * HD
            fvk.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                             S, Q_dim, K_dim, HD, 2560,
                                             kv_offset, HD, stream)

            # ── C3: Cross-attention ──
            if attn is not None:
                attn.run("decoder", l, q_seq=S, kv_seq=total_keys, stream=stream)
            else:
                K_ptr = Kc + l * total_keys * HD * 2
                V_ptr = Vc + l * total_keys * HD * 2
                fvk.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                        logits, attn_out,
                                        S, total_keys, NH, HD, attn_scale, stream)

            # ── C4: O proj ──
            act_scale_o = act_scales + (step_scale_base + l * 4 + 1) * 4
            w_scale_o = w_scales + (l * 4 + 1) * 4
            fvk.quantize_fp8_static_fp16(attn_out, ctx_fp8, act_scale_o, S * NH * HD, stream)
            ow_ptr = ow + l * NH * HD * D
            fvk.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S, D, NH * HD,
                                       act_scale_o, w_scale_o, stream)

            # ── C4→C5: gate×residual + AdaRMSNorm → FP8 ──
            act_scale_gu = act_scales + (step_scale_base + l * 4 + 2) * 4
            fvk.gate_res_adarms_fp8_static_fp16(fg, gate, x, sf_ptr,
                                                  xn_fp8, gate, S, D, act_scale_gu, stream)

            # ── C5: Gate+Up merged GEMM ──
            w_scale_gu = w_scales + (l * 4 + 2) * 4
            gw_ptr = gw + l * D * H * 2
            fvk.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S, H * 2, D,
                                       act_scale_gu, w_scale_gu, stream)

            # ── C6: SiLU(gate) × up → FP8 ──
            act_scale_down = act_scales + (step_scale_base + l * 4 + 3) * 4
            fvk.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S, H, act_scale_down, stream)

            # ── C6: Down GEMM ──
            w_scale_down = w_scales + (l * 4 + 3) * 4
            dw_ptr = dw + l * H * D
            fvk.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, S, D, H,
                                       act_scale_down, w_scale_down, stream)

            # ── C7→C1_next: gate×residual + next AdaRMSNorm → FP8 ──
            if l < layers - 1:
                si_next = (s * layers + l + 1) * S * D3
                sa_next_ptr = sa + si_next * 2
                act_scale_next = act_scales + (step_scale_base + (l + 1) * 4 + 0) * 4
                fvk.gate_res_adarms_fp8_static_fp16(fg, gate, x, sa_next_ptr,
                                                      xn_fp8, gate, S, D, act_scale_next, stream)
            else:
                fvk.gate_res_fp16(fg, gate, x, S * D, stream)

        # ── Final: AdaRMSNorm + action output ──
        fi = s * S * D3
        fs_ptr = fs + fi * 2
        fvk.adarms_fp16(x, fs_ptr, xn, gate, S, D, stream)

        if (noise_t is not None and xn_t is not None and delta_t is not None
                and xn_f32_t is not None and delta_f32_t is not None
                and aow_f32_t is not None and aob_f32_t is not None
                and dt_t is not None):
            xn_f32_t.copy_(xn_t)
            torch.addmm(aob_f32_t, xn_f32_t, aow_f32_t, out=delta_f32_t)
            delta_t.copy_(delta_f32_t)
            delta_t.mul_(dt_t)
            noise_t.add_(delta_t)
        elif (noise_t is not None and xn_t is not None and delta_t is not None
                and aow_t is not None and aob_t is not None and dt_t is not None):
            torch.addmm(aob_t, xn_t, aow_t, out=delta_t)
            delta_t.mul_(dt_t)
            noise_t.add_(delta_t)
        else:
            fvk.gmm_fp16(ctx, xn, aow, noise, S, 32, D, 1.0, stream)
            fvk.add_bias_fp16(noise, aob, S, 32, stream)


# ══════════════════════════════════════════════════════════════════
# FP16 decoder path (no quantization, FP16 weights, baseline only)
# ══════════════════════════════════════════════════════════════════

def _decoder_forward_fp16(ctx, fvk, bufs, weights, dims, stream=0, *, attn=None):
    """FP16-only decoder forward. Structure mirrors the FP8 path; every
    GEMM is ``fvk.gmm_fp16`` and the fused FP8 norm kernels are split.

    Weight pointers (``qw/ow/gw/dw``) must reference FP16 [K,N] tensors
    (NN layout — same as the FP8 layout, just unquantized; see the
    ``use_fp8=False`` weight spec).
    """
    S = dims['S']; D = dims['D']; H = dims['H']
    NH = dims['NH']; HD = dims['HD']
    steps = dims['steps']; layers = dims['layers']
    enc_seq = dims['enc_seq']; total_keys = dims['total_keys']
    D3 = 3 * D
    Q_dim = NH * HD
    K_dim = HD
    attn_scale = 1.0 / math.sqrt(float(HD))

    noise = bufs['noise']; x = bufs['x']; xn = bufs['xn']
    gate = bufs['gate']; qkv = bufs['qkv']; logits = bufs['logits']
    attn_out = bufs['attn_out']; fg = bufs['fg']; hid = bufs['hid']
    noise_t = bufs.get('noise_t')
    xn_t = bufs.get('xn_t')
    delta_t = bufs.get('delta_t')
    xn_f32_t = bufs.get('xn_f32_t')
    delta_f32_t = bufs.get('delta_f32_t')

    ain_w = weights['ain_w']; ain_b = weights['ain_b']
    sa = weights['sa']; qw = weights['qw']
    Kc = weights['Kc']; Vc = weights['Vc']
    ow = weights['ow']; sf = weights['sf']
    gw = weights['gw']; dw = weights['dw']
    aow = weights['aow']; aob = weights['aob']
    aow_t = weights.get('aow_t')
    aob_t = weights.get('aob_t')
    aow_f32_t = weights.get('aow_f32_t')
    aob_f32_t = weights.get('aob_f32_t')
    dt_t = weights.get('dt_t')
    fs = weights['fs']; rope = weights['rope']

    for s in range(steps):
        fvk.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream)
        fvk.add_bias_fp16(x, ain_b, S, D, stream)

        for l in range(layers):
            si = (s * layers + l) * S * D3
            sa_ptr = sa + si * 2
            sf_ptr = sf + si * 2

            # C1: AdaRMSNorm (FP16, no FP8 quantize)
            fvk.adarms_fp16(x, sa_ptr, xn, gate, S, D, stream)

            # C2: QKV GEMM (FP16 NN; weight is [K, 2560])
            qw_ptr = qw + l * D * 2560 * 2  # FP16 = 2 bytes/elem
            fvk.gmm_fp16(ctx, xn, qw_ptr, qkv, S, 2560, D, 0.0, stream)

            # C2b: Split + RoPE + KV cache write
            kv_offset = l * total_keys * HD + enc_seq * HD
            fvk.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                             S, Q_dim, K_dim, HD, 2560,
                                             kv_offset, HD, stream)

            # C3: Attention
            if attn is not None:
                attn.run("decoder", l, q_seq=S, kv_seq=total_keys, stream=stream)
            else:
                K_ptr = Kc + l * total_keys * HD * 2
                V_ptr = Vc + l * total_keys * HD * 2
                fvk.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                        logits, attn_out,
                                        S, total_keys, NH, HD, attn_scale, stream)

            # C4: O proj
            ow_ptr = ow + l * NH * HD * D * 2  # FP16
            fvk.gmm_fp16(ctx, attn_out, ow_ptr, fg, S, D, NH * HD, 0.0, stream)

            # C4→C5: gated residual + post-attn AdaRMSNorm
            fvk.mul_fp16(fg, gate, fg, S * D, stream)
            fvk.residual_add_fp16(x, fg, S * D, stream)
            fvk.adarms_fp16(x, sf_ptr, xn, gate, S, D, stream)

            # C5: Gate+Up merged GEMM (weight is [K, 2H])
            gw_ptr = gw + l * D * H * 2 * 2  # FP16 = 2 bytes/elem; weight has 2H cols
            fvk.gmm_fp16(ctx, xn, gw_ptr, fg, S, H * 2, D, 0.0, stream)

            # C6: GELU(gate) × up (FP16)
            fvk.gate_geglu_merged_fp16(fg, hid, S, H, stream)

            # C6: Down GEMM (weight is [H, D])
            dw_ptr = dw + l * H * D * 2  # FP16
            fvk.gmm_fp16(ctx, hid, dw_ptr, fg, S, D, H, 0.0, stream)

            # C7: residual into x (next layer's C1 will do AdaRMS)
            fvk.mul_fp16(fg, gate, fg, S * D, stream)
            fvk.residual_add_fp16(x, fg, S * D, stream)

        # Final: AdaRMSNorm + action output
        fi = s * S * D3
        fs_ptr = fs + fi * 2
        fvk.adarms_fp16(x, fs_ptr, xn, gate, S, D, stream)
        if (noise_t is not None and xn_t is not None and delta_t is not None
                and xn_f32_t is not None and delta_f32_t is not None
                and aow_f32_t is not None and aob_f32_t is not None
                and dt_t is not None):
            xn_f32_t.copy_(xn_t)
            torch.addmm(aob_f32_t, xn_f32_t, aow_f32_t, out=delta_f32_t)
            delta_t.copy_(delta_f32_t)
            delta_t.mul_(dt_t)
            noise_t.add_(delta_t)
        elif (noise_t is not None and xn_t is not None and delta_t is not None
                and aow_t is not None and aob_t is not None and dt_t is not None):
            torch.addmm(aob_t, xn_t, aow_t, out=delta_t)
            delta_t.mul_(dt_t)
            noise_t.add_(delta_t)
        else:
            fvk.gmm_fp16(ctx, xn, aow, noise, S, 32, D, 1.0, stream)
            fvk.add_bias_fp16(noise, aob, S, 32, stream)


# ══════════════════════════════════════════════════════════════════
# Calibration (framework-agnostic, pure pointer ops)
# ══════════════════════════════════════════════════════════════════

def decoder_forward_calibrate(ctx, fvk_mod, bufs, weights, dims,
                               calib_scales_ptr, stream=0, *, attn=None):
    """Calibrate decoder FP8 scales. Framework-agnostic (pure pointers).

    For each quantization point:
      1. FP16 kernel → measure amax on GPU
      2. FP8 kernel with that scale
    """
    S = dims['S']; D = dims['D']; H = dims['H']
    NH = dims['NH']; HD = dims['HD']
    steps = dims['steps']; layers = dims['layers']
    enc_seq = dims['enc_seq']; total_keys = dims['total_keys']
    Q_dim = NH * HD
    attn_scale = 1.0 / math.sqrt(float(HD))
    D3 = 3 * D

    noise = bufs['noise']; x = bufs['x']; xn = bufs['xn']
    gate_buf = bufs['gate']; qkv = bufs['qkv']; logits = bufs['logits']
    attn_out = bufs['attn_out']; hid = bufs['hid']; fg = bufs['fg']
    noise_t = bufs.get('noise_t')
    xn_t = bufs.get('xn_t')
    delta_t = bufs.get('delta_t')
    xn_f32_t = bufs.get('xn_f32_t')
    delta_f32_t = bufs.get('delta_f32_t')
    xn_fp8 = bufs['xn_fp8']; hid_fp8 = bufs['hid_fp8']; ctx_fp8 = bufs['ctx_fp8']

    ain_w = weights['ain_w']; ain_b = weights['ain_b']
    sa = weights['sa']; qw = weights['qw']
    Kc = weights['Kc']; Vc = weights['Vc']
    ow = weights['ow']; sf = weights['sf']
    gw = weights['gw']; dw = weights['dw']
    aow = weights['aow']; aob = weights['aob']
    aow_t = weights.get('aow_t')
    aob_t = weights.get('aob_t')
    aow_f32_t = weights.get('aow_f32_t')
    aob_f32_t = weights.get('aob_f32_t')
    dt_t = weights.get('dt_t')
    fs = weights['fs']; rope = weights['rope']
    w_scales = weights['w_scales']

    # Scratch buffers — provided by caller via bufs dict
    calib_buf = bufs['calib_buf']          # layers*4 float32
    d_scale = bufs['d_scale']              # 1 float32
    hidden_scratch = bufs['hidden_scratch']  # S*H fp16
    fp8_scratch = bufs['fp8_scratch']      # S*max(D,H) fp8
    _gpu_zero(calib_buf, steps * layers * 4 * 4, stream)

    for s in range(steps):
        step_scale_base = s * layers * 4
        fvk_mod.gmm_fp16(ctx, noise, ain_w, x, S, D, 32, 0.0, stream)
        fvk_mod.add_bias_fp16(x, ain_b, S, D, stream)

        for l in range(layers):
            si = (s * layers + l) * S * D3
            sa_ptr = sa + si * 2
            sf_ptr = sf + si * 2

            # C1: AdaRMSNorm FP16 → measure amax → FP8
            cs_qkv = calib_buf + (step_scale_base + l * 4 + 0) * 4
            if l == 0:
                fvk_mod.adarms_fp16(x, sa_ptr, xn, gate_buf, S, D, stream)
                _measure_scale_gpu(fvk_mod, xn, S * D, d_scale, fp8_scratch, stream)
                _gpu_sync(stream)
                _gpu_copy(cs_qkv, d_scale, 4, stream)
                fvk_mod.fused_adarms_fp8_static_fp16(
                    x, sa_ptr, xn_fp8, gate_buf, S, D, cs_qkv, stream)

            # C2: QKV GEMM
            ws_qkv = w_scales + (l * 4 + 0) * 4
            qw_ptr = qw + l * D * 2560
            fvk_mod.fp8_gemm_descale_fp16(xn_fp8, qw_ptr, qkv, S, 2560, D,
                                           cs_qkv, ws_qkv, stream)

            # C2b: Split+RoPE
            kv_offset = l * total_keys * HD + enc_seq * HD
            fvk_mod.qkv_split_rope_kvcache_fp16(qkv, rope, attn_out, Kc, Vc,
                                                  S, Q_dim, HD, HD, 2560,
                                                  kv_offset, HD, stream)

            # C3: Attention
            if attn is not None:
                attn.run("decoder", l, q_seq=S, kv_seq=total_keys, stream=stream)
            else:
                K_ptr = Kc + l * total_keys * HD * 2
                V_ptr = Vc + l * total_keys * HD * 2
                fvk_mod.attention_qkv_fp16(ctx, attn_out, K_ptr, V_ptr,
                                            logits, attn_out,
                                            S, total_keys, NH, HD, attn_scale, stream)

            # C4: O proj — measure attn amax → FP8 → GEMM
            _measure_scale_gpu(fvk_mod, attn_out, S * NH * HD, d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            cs_o = calib_buf + (step_scale_base + l * 4 + 1) * 4
            _gpu_copy(cs_o, d_scale, 4, stream)
            ws_o = w_scales + (l * 4 + 1) * 4
            fvk_mod.quantize_fp8_static_fp16(attn_out, ctx_fp8, cs_o, S * NH * HD, stream)
            ow_ptr = ow + l * NH * HD * D
            fvk_mod.fp8_gemm_descale_fp16(ctx_fp8, ow_ptr, fg, S, D, NH * HD,
                                           cs_o, ws_o, stream)

            # C4→C5: gate×residual + AdaRMSNorm → measure → FP8
            fvk_mod.gate_res_fp16(fg, gate_buf, x, S * D, stream)
            fvk_mod.adarms_fp16(x, sf_ptr, xn, gate_buf, S, D, stream)
            _measure_scale_gpu(fvk_mod, xn, S * D, d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            cs_gu = calib_buf + (step_scale_base + l * 4 + 2) * 4
            _gpu_copy(cs_gu, d_scale, 4, stream)
            fvk_mod.quantize_fp8_static_fp16(xn, xn_fp8, cs_gu, S * D, stream)

            # C5: Gate+Up GEMM
            ws_gu = w_scales + (l * 4 + 2) * 4
            gw_ptr = gw + l * D * H * 2
            fvk_mod.fp8_gemm_descale_fp16(xn_fp8, gw_ptr, fg, S, H * 2, D,
                                           cs_gu, ws_gu, stream)

            # C6: GELU → measure → FP8
            fvk_mod.gate_geglu_merged_fp16(fg, hidden_scratch, S, H, stream)
            _measure_scale_gpu(fvk_mod, hidden_scratch, S * H, d_scale, fp8_scratch, stream)
            _gpu_sync(stream)
            cs_down = calib_buf + (step_scale_base + l * 4 + 3) * 4
            _gpu_copy(cs_down, d_scale, 4, stream)
            fvk_mod.gate_geglu_merged_fp8_fp16(fg, hid_fp8, S, H, cs_down, stream)

            # C6: Down GEMM
            ws_down = w_scales + (l * 4 + 3) * 4
            dw_ptr = dw + l * H * D
            fvk_mod.fp8_gemm_descale_fp16(hid_fp8, dw_ptr, fg, S, D, H,
                                           cs_down, ws_down, stream)

            # C7: gate×residual + next layer prep
            if l < layers - 1:
                si_next = (s * layers + l + 1) * S * D3
                sa_next_ptr = sa + si_next * 2
                fvk_mod.gate_res_fp16(fg, gate_buf, x, S * D, stream)
                fvk_mod.adarms_fp16(x, sa_next_ptr, xn, gate_buf, S, D, stream)
                _measure_scale_gpu(fvk_mod, xn, S * D, d_scale, fp8_scratch, stream)
                _gpu_sync(stream)
                cs_next = calib_buf + (step_scale_base + (l + 1) * 4 + 0) * 4
                _gpu_copy(cs_next, d_scale, 4, stream)
                fvk_mod.quantize_fp8_static_fp16(xn, xn_fp8, cs_next, S * D, stream)
            else:
                fvk_mod.gate_res_fp16(fg, gate_buf, x, S * D, stream)

        fi = s * S * D3
        fs_ptr = fs + fi * 2
        fvk_mod.adarms_fp16(x, fs_ptr, xn, gate_buf, S, D, stream)
        if (noise_t is not None and xn_t is not None and delta_t is not None
                and xn_f32_t is not None and delta_f32_t is not None
                and aow_f32_t is not None and aob_f32_t is not None
                and dt_t is not None):
            xn_f32_t.copy_(xn_t)
            torch.addmm(aob_f32_t, xn_f32_t, aow_f32_t, out=delta_f32_t)
            delta_t.copy_(delta_f32_t)
            delta_t.mul_(dt_t)
            noise_t.add_(delta_t)
        elif (noise_t is not None and xn_t is not None and delta_t is not None
                and aow_t is not None and aob_t is not None and dt_t is not None):
            torch.addmm(aob_t, xn_t, aow_t, out=delta_t)
            delta_t.mul_(dt_t)
            noise_t.add_(delta_t)
        else:
            fvk_mod.gmm_fp16(ctx, xn, aow, noise, S, 32, D, 1.0, stream)
            fvk_mod.add_bias_fp16(noise, aob, S, 32, stream)

    _gpu_copy(calib_scales_ptr, calib_buf, steps * layers * 4 * 4, stream)
    _gpu_sync(stream)



class Pi05ThorPipeline:
    """Pi0.5 Thor SM110 inference pipeline base class (B=1).

    The Thor frontend owns all device buffers and captures the
    SigLIP + enc_ae CUDA graphs at B=1 shape. This class is a thin
    facade that exposes :meth:`run_pipeline` for orchestration, so the
    serial CFG and (Stage 2) batched subclasses can share the same
    contract.

    Args:
        batch_size: Hard contract — B=1 for the base class. Stage 2's
            :class:`Pi05ThorBatchedPipeline` accepts ``batch_size >= 1``
            and is the path to use for actual B>1 inference.

    Notes:
        * **No buffer ownership**: lifetime of all device buffers
          stays with the frontend. The pipeline never alloc/frees.
        * **No graph capture**: capture is the frontend's
          responsibility (it is intertwined with calibration). The
          pipeline only orchestrates ``replay``.
        * **Backend-agnostic**: the torch frontend uses
          ``torch.cuda.CUDAGraph`` and the JAX frontend uses
          ``flash_rt.core.cuda_graph.CUDAGraph``; both are wrapped
          by the ``replay_siglip`` / ``replay_enc_ae`` callbacks the
          frontend hands in. Each callback is responsible for any
          stream-sync the backend needs after replay.
    """

    def __init__(self, *, batch_size: int = 1):
        if batch_size != 1:
            raise ValueError(
                f"Pi05ThorPipeline base class supports only B=1; got "
                f"B={batch_size}. Use Pi05ThorBatchedPipeline for B>1.")
        self.batch_size = int(batch_size)

    def run_pipeline(self, *, replay_siglip, replay_enc_ae) -> None:
        """Replay the captured SigLIP graph followed by enc_ae graph.

        Args:
            replay_siglip: Callable ``() -> None`` that replays the
                frontend's captured SigLIP graph. Must include any
                stream synchronization the frontend's backend needs.
            replay_enc_ae: Callable ``() -> None`` for the encoder +
                decoder graph, same conventions.
        """
        replay_siglip()
        replay_enc_ae()

"""FlashRT — GROOT N1.6 Pipeline for Thor SM110.

Composes flash_rt_kernels into full GROOT inference pipeline.
Architecture: Eagle3-VL (SigLIP2 + Qwen3-1.7B) + AlternateVLDiT

Functions:
    siglip2_forward    — 27-layer SigLIP2 vision encoder (full attention, no sliding window)
    eagle_project      — Post-LN + mlp1 projection [1152 → 2048] + concat text
    qwen3_forward      — 28-layer Qwen3-1.7B LLM (GQA 16Q/8KV, q_norm/k_norm)
    dit_forward        — 32-layer AlternateVLDiT (alternating self/cross attention)
    embodiment_encode  — Per-embodiment state + action encoding MLPs
    embodiment_decode  — Per-embodiment action decoding MLP

Key API notes (from meta-tests):
    - bf16_nn: C = A @ B where B is [K, N] (NOT [N, K])
    - fp8_descale_fp16: scales are device pointers (not float values)
    - rope_apply: single num_heads arg — call separately for Q (16h) and K (8h) in GQA
    - gate_geglu_merged: actually GELU (use for DiT GEGLU FFN)
    - silu_mul_split_fp8_fp16: actual SiLU (use for Qwen3 FFN)
    - attention_qkv_fp16: GQA-only (single KV head) — NOT for DiT MHA
"""

import math


# ══════════════════════════════════════════════════════════════════
# SigLIP2 Vision Encoder (27 layers)
# Nearly identical to Pi0.5 SigLIP — use_windows_attn=false, use_rope=false
# Only difference: patch_embedding is Linear (vs Conv2d), handled in weight loading
# ══════════════════════════════════════════════════════════════════

def siglip2_forward(gemm, fvk, bufs, weights, dims, stream=0):
    """27-layer SigLIP2 vision encoder.

    Same computation as Pi0.5 siglip_forward (full attention, no RoPE).
    LayerNorm → FP8 → QKV GEMM → FMHA → FP8 → O+res → LN → FP8 → Up GELU → FP8 → Down+res

    Args:
        gemm: GemmRunner
        fvk: flash_rt_kernels module
        bufs: dict — x, x_fp8, qkv, attn_out, hidden, hid_fp8, scratch
        weights: dict — ln_attn_w/b[L], qkv_w/b[L], o_w/b[L], ln_ffn_w/b[L],
                        up_w/b[L], down_w/b[L], alpha[L*4]
        dims: dict — S, D=1152, H=4304, NH=16, HD=72, L=27, num_views, seq_per_view=256
    """
    # TODO: Implement — same as siglip_forward in pipeline_pi05.py
    raise NotImplementedError("siglip2_forward")


# ══════════════════════════════════════════════════════════════════
# Eagle VL Projection
# Post-LayerNorm → mlp1 Linear [1152 → 2048] → concat text embeddings
# ══════════════════════════════════════════════════════════════════

def eagle_project(gemm, fvk, bufs, weights, dims, stream=0):
    """Eagle backbone VL projection.

    Post-LayerNorm on SigLIP2 output → Linear [1152, 2048] → concat text embeddings.
    Output: [Se, 2048] where Se = S_vis + S_text.

    Args:
        dims: dict — S_vis, D_sig=1152, D_llm=2048, S_text (prompt_len)
    """
    # TODO: Implement
    raise NotImplementedError("eagle_project")


# ══════════════════════════════════════════════════════════════════
# Qwen3-1.7B LLM (28 layers)
# GQA 16Q/8KV, HD=128, FFN=6144, q_norm/k_norm, rope_theta=1e6
# ══════════════════════════════════════════════════════════════════

def qwen3_forward(gemm, fvk, bufs, weights, dims, stream=0):
    """28-layer Qwen3-1.7B with GQA and q_norm/k_norm.

    Per layer:
        1. RMSNorm → FP8 quantize
        2. QKV GEMM: [Se, 2048] @ [2048, 4096] → [Se, 4096]
           Split: Q [Se, 2048], K [Se, 1024], V [Se, 1024]
        3. q_norm: per-head RMSNorm on Q (16 heads × 128)
           k_norm: per-head RMSNorm on K (8 heads × 128)
        4. RoPE (theta=1e6) — Q and K separately (different head counts)
        5. FMHA: GQA 16Q/8KV, HD=128
        6. O GEMM: [Se, 2048] @ [2048, 2048] → [Se, 2048] + residual
        7. Post-attn RMSNorm → FP8
        8. FFN: gate_proj [2048→6144] + up_proj [2048→6144]
           SiLU(gate) * up → FP8 → down_proj [6144→2048] + residual

    Key differences from Pi0.5 encoder:
        - ★16 layers★ (select_layer=16, checkpoint truncated) (C1)
        - GQA 16/8 (vs 8/1), QKV_DIM=4096 (vs 2560)
        - q_norm/k_norm after QKV projection, before RoPE
        - FFN dim=6144 (vs 8192), activation=SiLU (Qwen3)
        - rope_theta=1e6 (vs 1e4)

    Args:
        dims: dict — Se, D=2048, H=6144, NHQ=16, NHKV=8, HD=128, L=16
    """
    Se = dims['Se']
    D  = dims['D']       # 2048
    H  = dims['H']       # 6144
    NHQ = dims['NHQ']    # 16 query heads
    NHKV = dims['NHKV']  # 8 KV heads
    HD = dims['HD']      # 128
    L  = dims['L']       # ★16★ (C1)
    QKV_DIM = NHQ * HD + NHKV * HD + NHKV * HD  # 4096

    x        = bufs['x']          # [Se, D] fp16
    residual = bufs['residual']   # [Se, D] fp16
    x_fp8    = bufs['x_fp8']      # [Se, D] fp8
    qkv      = bufs['qkv']        # [Se, QKV_DIM] fp16
    Q        = bufs['Q']          # [Se, NHQ, HD] fp16 (reshaped view of qkv)
    K        = bufs['K']          # [Se, NHKV, HD] fp16
    V        = bufs['V']          # [Se, NHKV, HD] fp16
    attn_out = bufs['attn_out']   # [Se, NHQ, HD] fp16
    ffn_merged = bufs['ffn_merged']  # [Se, 2*H] fp16/bf16
    ffn_fp8  = bufs['ffn_fp8']    # [Se, H] fp8

    act_scales = bufs['act_scales']  # device ptr: [L, 4] float32
    rope_weights = bufs['rope_weights']  # [max_seq, HD] bf16

    for l in range(L):
        # ── Pre-attention RMSNorm → FP8 ──
        as_qkv = act_scales + (l * 4 + 0) * 4  # byte offset for float32
        fvk.residual_add_rms_norm_fp8_noweight_fp16(
            residual, x, weights['ln_attn_w'][l], x_fp8, Se, D, as_qkv, stream)

        # ── QKV GEMM: [Se, D] @ [D, QKV_DIM] → [Se, 4096] ──
        alpha_qkv = weights['alpha_host'][l * 4 + 0]
        gemm.fp8_descale_fp16(
            x_fp8, weights['qkv_w'][l], qkv,
            Se, QKV_DIM, D,
            weights['act_scale_ptrs'][l * 4 + 0],
            weights['w_scale_ptrs']['qkv'][l],
            stream)

        # ── Split QKV → Q [Se, NHQ*HD], K [Se, NHKV*HD], V [Se, NHKV*HD] ──
        # Q, K, V are pointer views into qkv buffer
        Q_ptr = qkv
        K_ptr = qkv + NHQ * HD * 2     # byte offset (fp16 = 2 bytes)
        V_ptr = qkv + (NHQ + NHKV) * HD * 2

        # ── q_norm / k_norm: per-head RMSNorm ──
        for h in range(NHQ):
            q_h = Q_ptr + h * HD * 2
            fvk.rms_norm_fp16(q_h, weights['q_norm_w'][l], q_h, Se, HD, stream)
        for h in range(NHKV):
            k_h = K_ptr + h * HD * 2
            fvk.rms_norm_fp16(k_h, weights['k_norm_w'][l], k_h, Se, HD, stream)

        # ── RoPE (theta=1e6) — Q and K separately ──
        # rope_apply(rope_w, tensor, dummy, seq, num_heads, head_dim, stream)
        # For Q: 16 heads
        K_dummy = V_ptr  # reuse V as scratch (will be overwritten anyway? NO!)
        # FIXME: need a proper dummy buffer or call rope on Q/K together somehow
        # Alternative: apply rope in-place to Q only, then K only
        # The kernel modifies BOTH Q and K args. So we need two calls with dummies.
        # TODO: evaluate if we should add a rope_apply_single kernel

        # ── FMHA: GQA 16Q/8KV, HD=128 ──
        stride_q = NHQ * HD * 2    # bytes per token for Q
        stride_kv = NHKV * HD * 2  # bytes per token for K/V
        fvk.fmha_strided_full(
            Q_ptr, K_ptr, V_ptr, attn_out,
            1, Se, Se, NHQ, NHKV, HD,
            NHQ * HD, NHKV * HD,  # strides in elements
            stream)

        # ── O projection: FP8 → GEMM → residual ──
        as_o = act_scales + (l * 4 + 1) * 4
        fvk.quantize_fp8_static_fp16(attn_out, x_fp8, Se * NHQ * HD, as_o, stream)
        gemm.fp8_descale_fp16(
            x_fp8, weights['o_w'][l], x,
            Se, D, D,
            weights['act_scale_ptrs'][l * 4 + 1],
            weights['w_scale_ptrs']['o'][l],
            stream)
        # residual add happens in next iteration's residual_add_rms_norm

        # ── Post-attention RMSNorm → FP8 ──
        as_gu = act_scales + (l * 4 + 2) * 4
        fvk.residual_add_rms_norm_fp8_noweight_fp16(
            residual, x, weights['ln_ffn_w'][l], x_fp8, Se, D, as_gu, stream)

        # ── FFN: gate+up merged → SiLU → FP8 → down ──
        # gate_up merged: [D, 2*H] = [2048, 12288]
        gemm.fp8_descale_fp16(
            x_fp8, weights['gate_up_w'][l], ffn_merged,
            Se, 2 * H, D,
            weights['act_scale_ptrs'][l * 4 + 2],
            weights['w_scale_ptrs']['gate_up'][l],
            stream)

        # SiLU(gate) * up → FP8
        as_d = act_scales + (l * 4 + 3) * 4
        fvk.silu_mul_split_fp8_fp16(
            ffn_merged,                        # gate [Se, H]
            ffn_merged + Se * H * 2,           # up [Se, H] (offset in bytes)
            ffn_fp8, Se * H, as_d, stream)

        # down_proj → residual
        gemm.fp8_descale_fp16(
            ffn_fp8, weights['down_w'][l], x,
            Se, D, H,
            weights['act_scale_ptrs'][l * 4 + 3],
            weights['w_scale_ptrs']['down'][l],
            stream)


def qwen3_forward_calibrate(gemm, fvk, bufs, weights, dims, stream=0):
    """Qwen3 forward with activation scale collection for FP8 calibration."""
    # TODO: Implement — collect max activations per quantization point
    raise NotImplementedError("qwen3_forward_calibrate")


# ══════════════════════════════════════════════════════════════════
# AlternateVLDiT (32 layers × 4 flow-matching steps)
# ══════════════════════════════════════════════════════════════════

def dit_forward(gemm, fvk, bufs, weights, dims, stream=0):
    """32-layer AlternateVLDiT, single flow-matching step.

    Block pattern (32 blocks):
        Even idx (0,2,4,...): cross-attention to backbone features
            - idx % (2*attend_text_every_n_blocks) == 0: attend to non-image (text) tokens
            - else: attend to image tokens
            (with attend_text_every_n_blocks=2: text at 0,4,8,...; image at 2,6,10,...)
        Odd idx (1,3,5,...): self-attention

    Per block:
        1. AdaLayerNorm: LN(x) * (1 + scale) + shift  (precomputed from timestep)
           ★ norm has NO learnable parameters (elementwise_affine=False) (C7) ★
        2. Attention (self or cross, MHA 32h×48) + bias
        3. Residual add
        4. LayerNorm (NO params, elementwise_affine=False) (C8)
        5. ★ GELU FFN ★: Linear [1536→6144] → GELU(approx=tanh) → Linear [6144→1536]
           (NOT GEGLU — ff.net.0.proj is [6144,1536] not [12288,1536]) (C3)
        6. Residual add

    Self-attn: Q/K/V all from [Sa, 1536], S_q=S_kv=Sa (Sa=51 max)
    Cross-attn: Q from [Sa, 1536], K/V from backbone [S_kv, 2048→1536]

    Weight dimensions differ between block types:
        Self-attn (odd):  to_k/to_v [1536, 1536]
        Cross-attn (even): to_k/to_v [2048, 1536]  (input from backbone)

    Args:
        dims: dict — S=Sa, D=1536, H=6144, NH=32, HD=48, L=32,
                     S_img, S_txt (backbone token counts)
    """
    S  = dims['S']       # 17 (1 state + 16 actions)
    D  = dims['D']       # 1536
    H  = dims['H']       # 6144
    NH = dims['NH']      # 32
    HD = dims['HD']      # 48
    L  = dims['L']       # 32

    x            = bufs['x']              # [S, D] bf16
    x_normed     = bufs['x_normed']       # [S, D] bf16
    Q            = bufs['Q']              # [S, NH, HD] bf16
    K            = bufs['K']              # [max_kv, NH, HD] bf16
    V            = bufs['V']              # [max_kv, NH, HD] bf16
    attn_out     = bufs['attn_out']       # [S, NH, HD] bf16
    o_proj       = bufs['o_proj']         # [S, D] bf16
    ffn_merged   = bufs['ffn_merged']     # [S, 2*H] bf16
    ffn_out      = bufs['ffn_out']        # [S, H] bf16

    img_feats    = bufs['img_feats']      # [S_img, 2048] bf16  (precomputed)
    txt_feats    = bufs['txt_feats']      # [S_txt, 2048] bf16  (precomputed)

    stride_q  = NH * HD   # elements per token
    stride_kv = NH * HD

    for l in range(L):
        # ── AdaLayerNorm (no learnable norm params — C7) ──
        # Step 1: LayerNorm(x) with NO weight/bias (elementwise_affine=False)
        # Step 2: x_normed = LN(x) * (1 + scale[l]) + shift[l]
        # scale/shift are precomputed from timestep embedding
        fvk.layer_norm_fp16_no_affine(x, x_normed, S, D, stream)
        # TODO: Apply affine: x_normed = x_normed * (1 + scale) + shift
        # Precomputed scale/shift from _precompute_timesteps()
        scale_ptr = bufs['ada_scale'] + l * D * 2   # [D] bf16
        shift_ptr = bufs['ada_shift'] + l * D * 2

        if l % 2 == 1:
            # ── Self-Attention (odd blocks) ──
            # Q/K/V all from hidden_states [S, D]
            # QKV GEMM: [S, D] @ [D, 3D] → [S, 3D=4608]
            gemm.bf16_nn(x_normed, weights['qkv_w'][l], bufs['qkv'],
                         S, 3 * D, D, stream)
            # Split: Q [S, D], K [S, D], V [S, D]
            # Q/K/V are views into qkv buffer at offsets 0, D, 2D
            Q_ptr = bufs['qkv']
            K_ptr = bufs['qkv'] + D * 2       # byte offset
            V_ptr = bufs['qkv'] + 2 * D * 2

            # FMHA self-attention
            fvk.fmha_strided_full(
                Q_ptr, K_ptr, V_ptr, attn_out,
                1, S, S, NH, NH, HD,
                stride_q, stride_kv, stream)

        else:
            # ── Cross-Attention (even blocks) ──
            # Q from hidden_states, K/V from backbone
            if l % 4 == 0:
                kv_src = txt_feats    # text tokens
                S_kv = dims['S_txt']
            else:
                kv_src = img_feats    # image tokens
                S_kv = dims['S_img']

            # Q projection: [S, D] @ [D, D] → [S, D=1536]
            gemm.bf16_nn(x_normed, weights['q_w'][l], Q,
                         S, D, D, stream)
            # K projection: [S_kv, 2048] @ [2048, D] → [S_kv, D=1536]
            gemm.bf16_nn(kv_src, weights['k_w'][l], K,
                         S_kv, D, dims['D_backbone'], stream)
            # V projection: same
            gemm.bf16_nn(kv_src, weights['v_w'][l], V,
                         S_kv, D, dims['D_backbone'], stream)

            # FMHA cross-attention
            fvk.fmha_strided_full(
                Q, K, V, attn_out,
                1, S, S_kv, NH, NH, HD,
                stride_q, stride_kv, stream)

        # ── O projection + residual ──
        # attn_out [S, NH*HD=D] → [S, D]
        gemm.bf16_nn(attn_out, weights['o_w'][l], o_proj,
                     S, D, D, stream)
        # x = x + o_proj  (element-wise add)
        # TODO: fused residual add kernel

        # ── FFN: LayerNorm (no params — C8) → ★GELU★ (NOT GEGLU — C3) ──
        fvk.layer_norm_fp16_no_affine(x, x_normed, S, D, stream)
        # GELU up: [S, D] @ [D, H] → [S, 6144] (★ NOT 12288 ★)
        gemm.bf16_nn(x_normed, weights['ff_up_w'][l], ffn_out,
                     S, H, D, stream)
        # Add bias + GELU activation (approximate='tanh')
        # TODO: fused bias + GELU kernel, or compose:
        #   fvk.add_bias_fp16(ffn_out, weights['ff_up_b'][l], S, H, stream)
        #   fvk.gelu_approximate_inplace(ffn_out, S * H, stream)
        # Down: [S, H] @ [H, D] → [S, D]
        gemm.bf16_nn(ffn_out, weights['ff_down_w'][l], o_proj,
                     S, D, H, stream)
        # x = x + o_proj
        # TODO: fused residual add + bias

    # ── Final output projection ──
    # norm_out (NO params — C9) → affine (conditioning) → proj_out_2 [D → output_dim=1024]
    fvk.layer_norm_fp16_no_affine(x, x_normed, S, D, stream)
    # Apply final conditioning: x = x * (1 + scale_final) + shift_final (C11: shift first!)
    # TODO: element-wise affine with precomputed scale/shift from bufs['out_scale'], bufs['out_shift']
    # proj_out_2: [S, D] @ [D, 1024] → [S, 1024]
    gemm.bf16_nn(x_normed, weights['proj_out_2_w'], bufs['dit_output'],
                 S, dims['output_dim'], D, stream)


def dit_forward_calibrate(gemm, fvk, bufs, weights, dims, stream=0):
    """DiT forward with activation scale collection for calibration."""
    # TODO: Implement if FP8 DiT optimization is pursued
    raise NotImplementedError("dit_forward_calibrate")


# ══════════════════════════════════════════════════════════════════
# Per-Embodiment MLPs
# ══════════════════════════════════════════════════════════════════

def embodiment_encode_state(gemm, bufs, weights, dims, stream=0):
    """State encoding: [1, 128] → MLP → [1, 1536]. (C6: dim=128 not 29)

    CategorySpecificMLP: 2-layer with ReLU.
        Linear [128→1024] + ReLU → Linear [1024→1536]

    Weights pre-extracted for target embodiment.
    W layout is [in, out] — matches bf16_nn B=[K,N] directly (C13).
    """
    # state [1, 128] @ W1 [128, 1024] + b1 → ReLU → @ W2 [1024, 1536] + b2
    gemm.bf16_nn(bufs['state'], weights['state_enc_w1'], bufs['state_h1'],
                 1, 1024, 128, stream)
    # TODO: add bias + ReLU kernel
    gemm.bf16_nn(bufs['state_h1'], weights['state_enc_w2'], bufs['state_emb'],
                 1, 1536, 1024, stream)
    # TODO: add bias


def embodiment_encode_action(gemm, fvk, bufs, weights, dims, timestep_emb, stream=0):
    """Action encoding: [action_horizon, 128] + timestep → MLP → [action_horizon, 1536].

    Per-embodiment MultiEmbodimentActionEncoder (C6, C10):
        W1: [128→1536] per action token
        Sinusoidal time embedding → [1536] (same dim as hidden)
        Concat [action_emb, time_emb] → [3072]
        W2: [3072→1536] + Swish (=SiLU)
        W3: [1536→1536]

    Plus position embedding [action_horizon, 1536] added after encoding.
    """
    T = dims['action_horizon']  # 50 (padded max)
    D_act = 128                 # padded action dim (C6)
    D_hid = 1536                # hidden = input_embedding_dim (C10)
    D_time = D_hid              # sinusoidal time embedding dim

    # Step 1: Action embedding: [T, 128] @ W1 [128, 1536] → [T, 1536]
    gemm.bf16_nn(bufs['action'], weights['action_enc_w1'], bufs['action_emb'],
                 T, D_hid, D_act, stream)
    # TODO: add bias W1.b

    # Step 2: Sinusoidal time embedding (precomputed, broadcast to T tokens)
    # timestep_emb: [1, 1536] → expand to [T, 1536]
    # TODO: broadcast/copy timestep_emb to bufs['time_emb'] [T, 1536]

    # Step 3: Concat [action_emb, time_emb] → [T, 3072]
    # Then W2: [T, 3072] @ [3072, 1536] + b2 → Swish → [T, 1536]
    gemm.bf16_nn(bufs['action_time_concat'], weights['action_enc_w2'], bufs['action_h2'],
                 T, D_hid, 2 * D_hid, stream)
    # TODO: add bias + Swish (SiLU)

    # Step 4: W3: [T, 1536] @ [1536, 1536] + b3 → [T, 1536]
    gemm.bf16_nn(bufs['action_h2'], weights['action_enc_w3'], bufs['action_encoded'],
                 T, D_hid, D_hid, stream)
    # TODO: add bias

    # Step 5: Add position embedding
    # bufs['action_encoded'] += position_embedding[:T]


def embodiment_decode_action(gemm, bufs, weights, dims, stream=0):
    """Action decoding: [Sa, 1024] → MLP → [Sa, 128]. (C6, C12)

    CategorySpecificMLP: 2-layer with ReLU.
        Linear [1024→1024] + ReLU → Linear [1024→128]

    Only last action_horizon rows are used as velocity predictions.
    Sa = 1 + action_horizon = 51.
    """
    Sa = dims['Sa']  # 51 (C12)
    # dit_output [Sa, 1024] @ W1 [1024, 1024] + b1 → ReLU → @ W2 [1024, 128] + b2
    gemm.bf16_nn(bufs['dit_output'], weights['action_dec_w1'], bufs['dec_h1'],
                 Sa, 1024, 1024, stream)
    # TODO: add bias + ReLU
    gemm.bf16_nn(bufs['dec_h1'], weights['action_dec_w2'], bufs['velocity'],
                 Sa, 128, 1024, stream)
    # TODO: add bias
    # velocity[-action_horizon:] = predicted velocity for Euler step


# ══════════════════════════════════════════════════════════════════
# All-C-Kernel Graph-Compatible Classes
# ══════════════════════════════════════════════════════════════════

import torch
import torch.nn.functional as F
import flash_rt.flash_rt_kernels as fvk

fp16 = torch.float16
fp8 = torch.float8_e4m3fn

def _quant_fp8(w):
    a = w.float().abs().max().item()
    s = max(a / 448.0, 1e-12)
    return (w.float() / s).clamp(-448, 448).to(fp8), s


class CKernelQwen3:
    """Qwen3 16L all-C-kernel forward — CUDA Graph compatible.

    All operations use flash_rt_kernels: rms_norm, FP8 GEMM, RoPE,
    attention_mha (cuBLAS), residual_add. Internal FP32 precision.
    """

    def __init__(self, sd_or_path, Se, use_fp8=True):
        self.gemm = fvk.GemmRunner()
        self.use_fp8 = bool(use_fp8)
        self.D = 2048; self.NHQ = 16; self.NHKV = 8; self.HD = 128
        self.H = 6144; self.L = 16; self.Se = Se
        self.QKV = self.NHQ * self.HD + 2 * self.NHKV * self.HD
        # Optional AttentionBackend (ThorGrootAttnBackend); set post-construct
        # by the frontend in stage 4.3. When None, forward() uses the direct
        # fvk.attention_mha_fp16 call below (bit-identical legacy path).
        self.attn = None
        self._unit = torch.ones(1, dtype=torch.float32, device='cuda')
        # Per-layer activation scales: 3 per layer (QKV, Gate+Up, Down).
        # Default 1.0 = unit scale (backward compat). Set via set_act_scales().
        self.act_scales = torch.ones(self.L * 3, dtype=torch.float32, device='cuda')

        if isinstance(sd_or_path, dict):
            self._load_weights(sd_or_path)
        else:
            from safetensors import safe_open
            import pathlib
            sd = {}
            for f in sorted(pathlib.Path(sd_or_path).glob("*.safetensors")):
                with safe_open(str(f), framework="pt", device="cuda") as sf:
                    for key in sf.keys():
                        if "language_model" in key:
                            sd[key] = sf.get_tensor(key)
            self._load_weights(sd)
            del sd
        self._precompute_rope()
        self._alloc_buffers()

    def set_act_scales(self, scales_list):
        """Set per-layer activation scales from a list of L*3 floats."""
        self.act_scales.copy_(torch.tensor(scales_list, dtype=torch.float32, device='cuda'))

    def _load_weights(self, sd):
        prefix = "backbone.model.language_model.model.layers"
        self.layers = []
        for i in range(self.L):
            lp = f"{prefix}.{i}"
            w = {}
            w['ln_w'] = sd[f"{lp}.input_layernorm.weight"].to(fp16)
            qkv_T = torch.cat([sd[f"{lp}.self_attn.{p}_proj.weight"]
                                for p in ('q', 'k', 'v')], dim=0).T.contiguous()
            w['qkv_fp8'], s = _quant_fp8(qkv_T)
            w['qkv_s'] = torch.tensor([s], dtype=torch.float32, device='cuda')
            if not self.use_fp8:
                # FP16 reference uses the same [K, N] weight layout and the same
                # cuBLASLt fp16_nn GEMM as the o_proj below.
                w['qkv_fp16'] = qkv_T.to(fp16)
            w['q_norm_w'] = sd[f"{lp}.self_attn.q_norm.weight"].to(fp16)
            w['k_norm_w'] = sd[f"{lp}.self_attn.k_norm.weight"].to(fp16)
            w['o_w'] = sd[f"{lp}.self_attn.o_proj.weight"].T.contiguous().to(fp16)
            w['ln2_w'] = sd[f"{lp}.post_attention_layernorm.weight"].to(fp16)
            gate_up = torch.cat([sd[f"{lp}.mlp.gate_proj.weight"],
                                 sd[f"{lp}.mlp.up_proj.weight"]], dim=0).T.contiguous()
            w['gu_fp8'], s = _quant_fp8(gate_up)
            w['gu_s'] = torch.tensor([s], dtype=torch.float32, device='cuda')
            down_T = sd[f"{lp}.mlp.down_proj.weight"].T.contiguous()
            w['down_fp8'], s2 = _quant_fp8(down_T)
            w['down_s'] = torch.tensor([s2], dtype=torch.float32, device='cuda')
            if not self.use_fp8:
                w['gu_fp16'] = gate_up.to(fp16)
                w['down_fp16'] = down_T.to(fp16)
            self.layers.append(w)
        self.final_norm_w = sd["backbone.model.language_model.model.norm.weight"].to(fp16)

    def _precompute_rope(self):
        theta = 1000000.0; HD = self.HD
        freqs = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float32, device='cuda') / HD))
        angles = torch.outer(torch.arange(self.Se, dtype=torch.float32, device='cuda'), freqs)
        self.cos_table = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).to(fp16)
        self.sin_table = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).to(fp16)

    def _alloc_buffers(self):
        Se, D, H = self.Se, self.D, self.H
        NHQ, NHKV, HD = self.NHQ, self.NHKV, self.HD
        self.b_x = torch.empty(Se, D, dtype=fp16, device='cuda')
        self.b_xn = torch.empty(Se, D, dtype=fp16, device='cuda')
        self.b_fp8_qkv = torch.empty(Se * D, dtype=torch.uint8, device='cuda')
        self.b_fp8_ffn = torch.empty(max(Se * D, Se * H), dtype=torch.uint8, device='cuda')
        self.b_qkv = torch.empty(Se, self.QKV, dtype=fp16, device='cuda')
        self.b_q = torch.empty(Se, NHQ * HD, dtype=fp16, device='cuda')
        self.b_k = torch.empty(Se, NHKV * HD, dtype=fp16, device='cuda')
        self.b_attn = torch.empty(Se, D, dtype=fp16, device='cuda')
        self.b_o = torch.empty(Se, D, dtype=fp16, device='cuda')
        self.b_gu = torch.empty(Se, 2 * H, dtype=fp16, device='cuda')
        self.b_gate = torch.empty(Se, H, dtype=fp16, device='cuda')
        self.b_up = torch.empty(Se, H, dtype=fp16, device='cuda')
        self.b_down = torch.empty(Se, D, dtype=fp16, device='cuda')
        self.b_k_exp = torch.empty(Se, NHQ * HD, dtype=fp16, device='cuda')
        self.b_v_exp = torch.empty(Se, NHQ * HD, dtype=fp16, device='cuda')
        Se_pad = ((Se + 7) // 8) * 8
        self.b_logits = torch.full((NHQ * Se, Se_pad), float('-inf'), dtype=fp16, device='cuda')
        self.ctx = fvk.FvkContext()

    def forward(self, x_in, s=0):
        Se, D, H = self.Se, self.D, self.H
        NHQ, NHKV, HD = self.NHQ, self.NHKV, self.HD
        fvk.gpu_copy(self.b_x.data_ptr(), x_in.data_ptr(), Se * D * 2, s)
        for i in range(self.L):
            w = self.layers[i]
            # Per-layer act scale pointers (3 scales per layer: QKV, Gate+Up, Down)
            as_qkv = self.act_scales.data_ptr() + (i * 3 + 0) * 4
            as_gu  = self.act_scales.data_ptr() + (i * 3 + 1) * 4
            as_dn  = self.act_scales.data_ptr() + (i * 3 + 2) * 4
            fvk.rms_norm_fp16(self.b_x.data_ptr(), w['ln_w'].data_ptr(),
                              self.b_xn.data_ptr(), Se, D, 1e-6, s)
            if self.use_fp8:
                fvk.quantize_fp8_static_fp16(self.b_xn.data_ptr(), self.b_fp8_qkv.data_ptr(),
                                              as_qkv, Se * D, s)
                self.gemm.fp8_descale_fp16(self.b_fp8_qkv.data_ptr(), w['qkv_fp8'].data_ptr(),
                                            self.b_qkv.data_ptr(), Se, self.QKV, D,
                                            as_qkv, w['qkv_s'].data_ptr(), s)
            else:
                self.gemm.fp16_nn(self.b_xn.data_ptr(), w['qkv_fp16'].data_ptr(),
                                  self.b_qkv.data_ptr(), Se, self.QKV, D, s)
            fvk.gpu_strided_copy_fp16(self.b_qkv.data_ptr(), self.b_q.data_ptr(), Se, NHQ*HD, self.QKV, 0, s)
            fvk.gpu_strided_copy_fp16(self.b_qkv.data_ptr(), self.b_k.data_ptr(), Se, NHKV*HD, self.QKV, NHQ*HD, s)
            fvk.rms_norm_fp16(self.b_q.data_ptr(), w['q_norm_w'].data_ptr(), self.b_q.data_ptr(), Se*NHQ, HD, 1e-6, s)
            fvk.rms_norm_fp16(self.b_k.data_ptr(), w['k_norm_w'].data_ptr(), self.b_k.data_ptr(), Se*NHKV, HD, 1e-6, s)
            fvk.rope_rotate_half_fp16(self.b_q.data_ptr(), self.cos_table.data_ptr(), self.sin_table.data_ptr(), Se, NHQ, HD, s)
            fvk.rope_rotate_half_fp16(self.b_k.data_ptr(), self.cos_table.data_ptr(), self.sin_table.data_ptr(), Se, NHKV, HD, s)
            fvk.gpu_strided_copy_fp16(self.b_qkv.data_ptr(), self.b_attn.data_ptr(), Se, NHKV*HD, self.QKV, NHQ*HD+NHKV*HD, s)
            fvk.gpu_repeat_interleave_heads(self.b_k.data_ptr(), self.b_k_exp.data_ptr(), Se, NHKV, HD, NHQ//NHKV, s)
            fvk.gpu_repeat_interleave_heads(self.b_attn.data_ptr(), self.b_v_exp.data_ptr(), Se, NHKV, HD, NHQ//NHKV, s)
            fvk.gpu_fill_neginf_fp16(self.b_logits.data_ptr(), self.b_logits.nelement(), s)
            if self.attn is not None:
                self.attn.run("qwen3", i, q_seq=Se, stream=s)
            else:
                fvk.attention_mha_fp16(self.ctx, self.b_q.data_ptr(), self.b_k_exp.data_ptr(), self.b_v_exp.data_ptr(),
                                        self.b_logits.data_ptr(), self.b_o.data_ptr(), Se, Se, NHQ, HD, 1.0/math.sqrt(HD), s)
            self.gemm.fp16_nn(self.b_o.data_ptr(), w['o_w'].data_ptr(), self.b_xn.data_ptr(), Se, D, D, s)
            fvk.residual_add_fp16(self.b_x.data_ptr(), self.b_xn.data_ptr(), Se * D, s)
            fvk.rms_norm_fp16(self.b_x.data_ptr(), w['ln2_w'].data_ptr(), self.b_xn.data_ptr(), Se, D, 1e-6, s)
            if self.use_fp8:
                fvk.quantize_fp8_static_fp16(self.b_xn.data_ptr(), self.b_fp8_ffn.data_ptr(), as_gu, Se*D, s)
                self.gemm.fp8_descale_fp16(self.b_fp8_ffn.data_ptr(), w['gu_fp8'].data_ptr(), self.b_gu.data_ptr(),
                                            Se, 2*H, D, as_gu, w['gu_s'].data_ptr(), s)
            else:
                self.gemm.fp16_nn(self.b_xn.data_ptr(), w['gu_fp16'].data_ptr(),
                                  self.b_gu.data_ptr(), Se, 2*H, D, s)
            fvk.gpu_strided_copy_fp16(self.b_gu.data_ptr(), self.b_gate.data_ptr(), Se, H, 2*H, 0, s)
            fvk.gpu_strided_copy_fp16(self.b_gu.data_ptr(), self.b_up.data_ptr(), Se, H, 2*H, H, s)
            if self.use_fp8:
                fvk.silu_mul_split_fp8_fp16(self.b_gate.data_ptr(), self.b_up.data_ptr(),
                                             self.b_fp8_ffn.data_ptr(), Se*H, as_dn, s)
                self.gemm.fp8_descale_fp16(self.b_fp8_ffn.data_ptr(), w['down_fp8'].data_ptr(), self.b_down.data_ptr(),
                                            Se, D, H, as_dn, w['down_s'].data_ptr(), s)
            else:
                fvk.silu_inplace_fp16(self.b_gate.data_ptr(), Se*H, s)
                # SiLU(gate) * up -> b_gu (free after the gate/up split), so the
                # down GEMM input does not alias either operand.
                fvk.mul_fp16(self.b_gate.data_ptr(), self.b_up.data_ptr(), self.b_gu.data_ptr(), Se*H, s)
                self.gemm.fp16_nn(self.b_gu.data_ptr(), w['down_fp16'].data_ptr(),
                                  self.b_down.data_ptr(), Se, D, H, s)
            fvk.residual_add_fp16(self.b_x.data_ptr(), self.b_down.data_ptr(), Se * D, s)
        fvk.rms_norm_fp16(self.b_x.data_ptr(), self.final_norm_w.data_ptr(), self.b_xn.data_ptr(), Se, D, 1e-6, s)
        return self.b_xn


class CKernelDiTHead:
    """DiT 32L x 4-step action head, all C kernels — CUDA Graph compatible.

    Includes: embodiment MLPs (state/action encode/decode),
    precomputed timestep/AdaLN conditioning, FP8 hybrid GEMMs.
    """

    def __init__(self, sd_or_path, embodiment_id, action_horizon, backbone_shape,
                 use_fp8=True):
        self.gemm = fvk.GemmRunner()
        self.ctx = fvk.FvkContext()
        self.use_fp8 = bool(use_fp8)
        self.D = 1536; self.H = 6144; self.NH = 32; self.HD = 48
        self.L = 32; self.action_dim = 128; self.num_steps = 4
        self.T = action_horizon; self.Sa = 1 + action_horizon
        self.S_kv = backbone_shape[1]; self.D_kv = backbone_shape[2]
        # Optional AttentionBackend (ThorGrootAttnBackend); set post-construct
        # by the frontend. When None, _run_step uses the direct
        # fvk.attention_mha_fp16 calls below (bit-identical legacy path).
        self.attn = None

        if isinstance(sd_or_path, dict):
            sd = sd_or_path
        else:
            from safetensors import safe_open
            import pathlib
            sd = {}
            for f in sorted(pathlib.Path(sd_or_path).glob("*.safetensors")):
                with safe_open(str(f), framework="pt", device="cuda") as sf:
                    for key in sf.keys():
                        if "action_head" in key:
                            sd[key] = sf.get_tensor(key)
        self._load_weights(sd, embodiment_id)
        self._precompute(sd)
        self._alloc_buffers()
        if not isinstance(sd_or_path, dict):
            del sd

    def _load_weights(self, sd, eid):
        to = lambda t: t.to(fp16).contiguous()
        toT = lambda t: t.T.contiguous().to(fp16)
        self.se_w1=to(sd["action_head.state_encoder.layer1.W"][eid]); self.se_b1=to(sd["action_head.state_encoder.layer1.b"][eid])
        self.se_w2=to(sd["action_head.state_encoder.layer2.W"][eid]); self.se_b2=to(sd["action_head.state_encoder.layer2.b"][eid])
        self.ae_w1=to(sd["action_head.action_encoder.W1.W"][eid]); self.ae_b1=to(sd["action_head.action_encoder.W1.b"][eid])
        self.ae_w2=to(sd["action_head.action_encoder.W2.W"][eid]); self.ae_b2=to(sd["action_head.action_encoder.W2.b"][eid])
        self.ae_w3=to(sd["action_head.action_encoder.W3.W"][eid]); self.ae_b3=to(sd["action_head.action_encoder.W3.b"][eid])
        self.ad_w1=to(sd["action_head.action_decoder.layer1.W"][eid]); self.ad_b1=to(sd["action_head.action_decoder.layer1.b"][eid])
        self.ad_w2=to(sd["action_head.action_decoder.layer2.W"][eid]); self.ad_b2=to(sd["action_head.action_decoder.layer2.b"][eid])
        self.pos_emb = to(sd["action_head.position_embedding.weight"])
        self.proj_out_2_w = toT(sd["action_head.model.proj_out_2.weight"])
        self.proj_out_2_b = to(sd["action_head.model.proj_out_2.bias"])
        self.dit = []
        for l in range(self.L):
            is_self = (l % 2 == 1)
            prefix = f"action_head.model.transformer_blocks.{l}"
            w = {}
            w['q_w']=toT(sd[f"{prefix}.attn1.to_q.weight"]); w['q_b']=to(sd[f"{prefix}.attn1.to_q.bias"])
            w['k_w']=toT(sd[f"{prefix}.attn1.to_k.weight"]); w['k_b']=to(sd[f"{prefix}.attn1.to_k.bias"])
            w['v_w']=toT(sd[f"{prefix}.attn1.to_v.weight"]); w['v_b']=to(sd[f"{prefix}.attn1.to_v.bias"])
            w['o_w']=toT(sd[f"{prefix}.attn1.to_out.0.weight"]); w['o_b']=to(sd[f"{prefix}.attn1.to_out.0.bias"])
            if is_self:
                qkv_m = torch.cat([sd[f"{prefix}.attn1.to_q.weight"], sd[f"{prefix}.attn1.to_k.weight"],
                                   sd[f"{prefix}.attn1.to_v.weight"]], dim=0).T.contiguous()
                w['qkv_fp8'], s = _quant_fp8(qkv_m)
                w['qkv_s'] = torch.tensor([s], dtype=torch.float32, device='cuda')
                w['qkv_b'] = torch.cat([w['q_b'], w['k_b'], w['v_b']], dim=0)
                if not self.use_fp8:
                    w['qkv_fp16'] = qkv_m.to(fp16)
            up_T = sd[f"{prefix}.ff.net.0.proj.weight"].T.contiguous()
            dn_T = sd[f"{prefix}.ff.net.2.weight"].T.contiguous()
            w['ff_up_fp8'], s = _quant_fp8(up_T); w['ff_up_s'] = torch.tensor([s], dtype=torch.float32, device='cuda'); w['ff_up_alpha'] = float(s)
            w['ff_up_b'] = to(sd[f"{prefix}.ff.net.0.proj.bias"])
            w['ff_dn_fp8'], s = _quant_fp8(dn_T); w['ff_dn_s'] = torch.tensor([s], dtype=torch.float32, device='cuda'); w['ff_dn_alpha'] = float(s)
            w['ff_dn_b'] = to(sd[f"{prefix}.ff.net.2.bias"])
            if not self.use_fp8:
                w['ff_up_fp16'] = up_T.to(fp16)
                w['ff_dn_fp16'] = dn_T.to(fp16)
            self.dit.append(w)

    def _precompute(self, sd):
        D, T = self.D, self.T
        ts_prefix = "action_head.model.timestep_encoder.timestep_embedder"
        ts_l1_w = sd[f"{ts_prefix}.linear_1.weight"].T.contiguous().to(fp16)
        ts_l1_b = sd[f"{ts_prefix}.linear_1.bias"].to(fp16)
        ts_l2_w = sd[f"{ts_prefix}.linear_2.weight"].T.contiguous().to(fp16)
        ts_l2_b = sd[f"{ts_prefix}.linear_2.bias"].to(fp16)
        proj_out_1_w = sd["action_head.model.proj_out_1.weight"].T.contiguous().to(fp16)
        proj_out_1_b = sd["action_head.model.proj_out_1.bias"].to(fp16)
        half_dim = 128
        exp = -torch.arange(half_dim, dtype=torch.float32, device='cuda') * (math.log(10000.0) / half_dim)
        half_d = D // 2
        exp_d = (-torch.arange(half_d, dtype=torch.float, device='cuda') * (math.log(10000.0) / half_d)).exp()
        self.ada_scales = []; self.ada_shifts = []
        self.out_scales = []; self.out_shifts = []
        self.action_time_embeds = []
        with torch.no_grad():
            for step in range(self.num_steps):
                t_disc = int(step / float(self.num_steps) * 1000)
                t_t = torch.tensor([t_disc], dtype=torch.float32, device='cuda')
                args = t_t[:, None] * exp.exp()
                sincos = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(fp16)
                temb = F.silu(sincos @ ts_l1_w + ts_l1_b) @ ts_l2_w + ts_l2_b
                silu_temb = F.silu(temb)
                scales, shifts = [], []
                for l in range(self.L):
                    ada_w = sd[f"action_head.model.transformer_blocks.{l}.norm1.linear.weight"].T.contiguous().to(fp16)
                    ada_b = sd[f"action_head.model.transformer_blocks.{l}.norm1.linear.bias"].to(fp16)
                    ada_out = silu_temb @ ada_w + ada_b
                    sc, sh = ada_out.squeeze(0).chunk(2, dim=0)
                    scales.append(sc); shifts.append(sh)
                self.ada_scales.append(torch.stack(scales))
                self.ada_shifts.append(torch.stack(shifts))
                out_cond = silu_temb @ proj_out_1_w + proj_out_1_b
                o_sh, o_sc = out_cond.squeeze(0).chunk(2, dim=0)
                self.out_scales.append(o_sc); self.out_shifts.append(o_sh)
                t_expanded = torch.full((T,), t_disc, device='cuda')
                freqs = t_expanded.unsqueeze(-1).float() * exp_d
                te = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).to(fp16)
                self.action_time_embeds.append(te)
        self.ada_scales = torch.stack(self.ada_scales); self.ada_shifts = torch.stack(self.ada_shifts)
        self.out_scales = torch.stack(self.out_scales); self.out_shifts = torch.stack(self.out_shifts)
        self.action_time_embeds = torch.stack(self.action_time_embeds)

    def precompute_cross_kv(self, s=0):
        """Precompute K/V projections for all cross-attention blocks.

        Cross-attention blocks (even l) project backbone features (kv_text/kv_img)
        through per-block K/V weight matrices. The backbone features are CONSTANT
        across diffusion steps, so these projections only need to run once.

        Without this: 16 blocks × 2 projections × 4 steps = 128 GEMMs
        With this:    16 blocks × 2 projections × 1 time  = 32 GEMMs (save 96)
        """
        D, S_kv = self.D, self.S_kv
        for block_idx in range(self.L // 2):
            l = block_idx * 2  # cross-attention layers: 0, 2, 4, ..., 30
            w = self.dit[l]
            kv_src = self.b_kv_text if l % 4 == 0 else self.b_kv_img
            self._fp16_gemm(kv_src.data_ptr(), w['k_w'].data_ptr(),
                            self._precomp_k[block_idx].data_ptr(), S_kv, D, self.D_kv, s)
            fvk.add_bias_fp16(self._precomp_k[block_idx].data_ptr(),
                              w['k_b'].data_ptr(), S_kv, D, s)
            self._fp16_gemm(kv_src.data_ptr(), w['v_w'].data_ptr(),
                            self._precomp_v[block_idx].data_ptr(), S_kv, D, self.D_kv, s)
            fvk.add_bias_fp16(self._precomp_v[block_idx].data_ptr(),
                              w['v_b'].data_ptr(), S_kv, D, s)

    def _alloc_buffers(self):
        D, H, T, Sa = self.D, self.H, self.T, self.Sa
        NH, HD = self.NH, self.HD; S_kv = self.S_kv
        self._unit = torch.ones(1, dtype=torch.float32, device='cuda')
        self._fp8_buf = torch.empty(max(Sa * max(D, H, 3*D), S_kv * D), dtype=torch.uint8, device='cuda')
        self.b_actions = torch.zeros(1, T, self.action_dim, dtype=torch.float32, device='cuda')
        self.b_state_feat = torch.empty(1, D, dtype=fp16, device='cuda')
        self.b_kv_text = torch.empty(S_kv, self.D_kv, dtype=fp16, device='cuda')
        self.b_kv_img = torch.empty(S_kv, self.D_kv, dtype=fp16, device='cuda')
        # Precomputed K/V projections for 16 cross-attention blocks (persistent across steps)
        n_cross = self.L // 2  # 16 cross-attention blocks
        self._precomp_k = [torch.empty(S_kv, D, dtype=fp16, device='cuda') for _ in range(n_cross)]
        self._precomp_v = [torch.empty(S_kv, D, dtype=fp16, device='cuda') for _ in range(n_cross)]

        # Per-block activation scales for FP8 GEMMs. 3 scales per block:
        #   [0] QKV input (self-attn blocks only, ignored for cross-attn)
        #   [1] FFN-up input (post layer-norm)
        #   [2] FFN-down input (post GELU)
        # Default 1.0 (unit scale). Set via set_dit_act_scales().
        self._dit_act_scales_dev = torch.ones(self.L * 3, dtype=torch.float32, device='cuda')
        self._dit_alpha_cache = {}  # l → (qkv_alpha, ffn_up_alpha, ffn_dn_alpha)
        self._rebuild_dit_alpha_cache()
        self.b_a_emb = torch.empty(T, D, dtype=fp16, device='cuda')
        self.b_concat = torch.empty(T, 2*D, dtype=fp16, device='cuda')
        self.b_enc_h = torch.empty(T, D, dtype=fp16, device='cuda')
        self.b_hidden = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_h_norm = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_qkv = torch.empty(Sa, 3*D, dtype=fp16, device='cuda')
        self.b_q_self = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_k_self = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_v_self = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_actions_fp16 = torch.empty(T, self.action_dim, dtype=fp16, device='cuda')
        Sa_pad = ((Sa + 7) // 8) * 8; Skv_pad = ((S_kv + 7) // 8) * 8
        self.b_attn_logits_self = torch.full((NH*Sa, Sa_pad), float('-inf'), dtype=fp16, device='cuda')
        self.b_attn_logits_cross = torch.full((NH*Sa, Skv_pad), float('-inf'), dtype=fp16, device='cuda')
        self.b_attn_out = torch.empty(Sa, NH*HD, dtype=fp16, device='cuda')
        self.b_o = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_ff_h = torch.empty(Sa, H, dtype=fp16, device='cuda')
        self.b_ff_out = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_q_cross = torch.empty(Sa, D, dtype=fp16, device='cuda')
        self.b_k_proj = torch.empty(S_kv, D, dtype=fp16, device='cuda')
        self.b_v_proj = torch.empty(S_kv, D, dtype=fp16, device='cuda')
        self.b_model_out = torch.empty(Sa, 1024, dtype=fp16, device='cuda')
        self.b_dec_h = torch.empty(Sa, 1024, dtype=fp16, device='cuda')
        self.b_velocity = torch.empty(Sa, self.action_dim, dtype=fp16, device='cuda')

    def _rebuild_dit_alpha_cache(self):
        """Recompute host alpha = act_scale * w_scale for each FP8 GEMM."""
        for l in range(self.L):
            w = self.dit[l]
            is_self = (l % 2 == 1)
            as_qkv = self._dit_act_scales_dev[l * 3 + 0].item()
            as_up = self._dit_act_scales_dev[l * 3 + 1].item()
            as_dn = self._dit_act_scales_dev[l * 3 + 2].item()
            qkv_alpha = as_qkv * w['qkv_s'].item() if is_self else 0.0
            up_alpha = as_up * w['ff_up_alpha']
            dn_alpha = as_dn * w['ff_dn_alpha']
            self._dit_alpha_cache[l] = (qkv_alpha, up_alpha, dn_alpha)

    def set_dit_act_scales(self, scales_list):
        """Set per-block activation scales from a list of L*3 floats."""
        self._dit_act_scales_dev.copy_(
            torch.tensor(scales_list, dtype=torch.float32, device='cuda'))
        self._rebuild_dit_alpha_cache()

    def _fp16_gemm(self, A, B, C, M, N, K, s=0):
        self.gemm.fp16_nn(A, B, C, M, N, K, s)

    def _fp8_gemm(self, A_fp16, B_fp8, w_scale_ptr, M, N, K, C, act_scale_ptr=None, s=0):
        as_ptr = act_scale_ptr if act_scale_ptr is not None else self._unit.data_ptr()
        fvk.quantize_fp8_static_fp16(A_fp16, self._fp8_buf.data_ptr(), as_ptr, M*K, s)
        self.gemm.fp8_descale_fp16(self._fp8_buf.data_ptr(), B_fp8, C, M, N, K,
                                    as_ptr, w_scale_ptr, s)

    def _fp8_gemm_bias(self, A_fp16, B_fp8, alpha, M, N, K, C, bias, act_scale_ptr=None, s=0):
        """FP8 GEMM + bias epilogue (1 kernel instead of 2)."""
        as_ptr = act_scale_ptr if act_scale_ptr is not None else self._unit.data_ptr()
        fvk.quantize_fp8_static_fp16(A_fp16, self._fp8_buf.data_ptr(), as_ptr, M*K, s)
        self.gemm.fp8_nn_bias(self._fp8_buf.data_ptr(), B_fp8, C, bias, M, N, K, alpha, s)

    def _fp8_gemm_gelu_bias(self, A_fp16, B_fp8, alpha, M, N, K, C, bias, act_scale_ptr=None, s=0):
        """FP8 GEMM + bias + GELU epilogue (1 kernel instead of 3)."""
        as_ptr = act_scale_ptr if act_scale_ptr is not None else self._unit.data_ptr()
        fvk.quantize_fp8_static_fp16(A_fp16, self._fp8_buf.data_ptr(), as_ptr, M*K, s)
        self.gemm.fp8_nn_gelu_bias(self._fp8_buf.data_ptr(), B_fp8, C, bias, M, N, K, alpha, s)

    def _run_step(self, step, s=0):
        D, H, T, Sa = self.D, self.H, self.T, self.Sa
        NH, HD = self.NH, self.HD; S_kv = self.S_kv; dt = 1.0/self.num_steps
        fvk.gpu_cast_fp32_to_fp16(self.b_actions.data_ptr(), self.b_actions_fp16.data_ptr(), T*self.action_dim, s)
        self._fp16_gemm(self.b_actions_fp16.data_ptr(), self.ae_w1.data_ptr(), self.b_a_emb.data_ptr(), T, D, self.action_dim, s)
        fvk.add_bias_fp16(self.b_a_emb.data_ptr(), self.ae_b1.data_ptr(), T, D, s)
        fvk.gpu_copy(self.b_concat.data_ptr(), self.b_a_emb.data_ptr(), T*D*2, s)
        fvk.gpu_copy(self.b_concat.data_ptr()+T*D*2, self.action_time_embeds[step].data_ptr(), T*D*2, s)
        self._fp16_gemm(self.b_concat.data_ptr(), self.ae_w2.data_ptr(), self.b_enc_h.data_ptr(), T, D, 2*D, s)
        fvk.add_bias_fp16(self.b_enc_h.data_ptr(), self.ae_b2.data_ptr(), T, D, s)
        fvk.silu_inplace_fp16(self.b_enc_h.data_ptr(), T*D, s)
        self._fp16_gemm(self.b_enc_h.data_ptr(), self.ae_w3.data_ptr(), self.b_a_emb.data_ptr(), T, D, D, s)
        fvk.add_bias_fp16(self.b_a_emb.data_ptr(), self.ae_b3.data_ptr(), T, D, s)
        fvk.residual_add_fp16(self.b_a_emb.data_ptr(), self.pos_emb[:T].data_ptr(), T*D, s)
        fvk.gpu_copy(self.b_hidden.data_ptr(), self.b_state_feat.data_ptr(), D*2, s)
        fvk.gpu_copy(self.b_hidden.data_ptr()+D*2, self.b_a_emb.data_ptr(), T*D*2, s)
        for l in range(self.L):
            is_self = (l % 2 == 1); w = self.dit[l]
            fvk.ada_layer_norm_fp16(self.b_hidden.data_ptr(), self.ada_scales[step,l].data_ptr(),
                                     self.ada_shifts[step,l].data_ptr(), self.b_h_norm.data_ptr(), Sa, D, 1e-5, s)
            as_qkv_ptr = self._dit_act_scales_dev.data_ptr() + (l * 3 + 0) * 4
            as_up_ptr  = self._dit_act_scales_dev.data_ptr() + (l * 3 + 1) * 4
            as_dn_ptr  = self._dit_act_scales_dev.data_ptr() + (l * 3 + 2) * 4
            qkv_alpha, up_alpha, dn_alpha = self._dit_alpha_cache[l]
            if is_self:
                if self.use_fp8:
                    self._fp8_gemm(self.b_h_norm.data_ptr(), w['qkv_fp8'].data_ptr(), w['qkv_s'].data_ptr(), Sa, 3*D, D, self.b_qkv.data_ptr(), as_qkv_ptr, s)
                else:
                    self._fp16_gemm(self.b_h_norm.data_ptr(), w['qkv_fp16'].data_ptr(), self.b_qkv.data_ptr(), Sa, 3*D, D, s)
                fvk.add_bias_fp16(self.b_qkv.data_ptr(), w['qkv_b'].data_ptr(), Sa, 3*D, s)
                fvk.gpu_strided_copy_fp16(self.b_qkv.data_ptr(), self.b_q_self.data_ptr(), Sa, D, 3*D, 0, s)
                fvk.gpu_strided_copy_fp16(self.b_qkv.data_ptr(), self.b_k_self.data_ptr(), Sa, D, 3*D, D, s)
                fvk.gpu_strided_copy_fp16(self.b_qkv.data_ptr(), self.b_v_self.data_ptr(), Sa, D, 3*D, 2*D, s)
                fvk.gpu_fill_neginf_fp16(self.b_attn_logits_self.data_ptr(), self.b_attn_logits_self.nelement(), s)
                if self.attn is not None:
                    # dit_self site layer_idx: we're at DiT layer l (odd), which
                    # is the (l // 2)-th self layer in the spec's 16-layer list.
                    self.attn.run("dit_self", l // 2, q_seq=Sa, stream=s)
                else:
                    fvk.attention_mha_fp16(self.ctx, self.b_q_self.data_ptr(), self.b_k_self.data_ptr(), self.b_v_self.data_ptr(),
                                           self.b_attn_logits_self.data_ptr(), self.b_attn_out.data_ptr(), Sa, Sa, NH, HD, 1.0/math.sqrt(HD), s)
            else:
                self._fp16_gemm(self.b_h_norm.data_ptr(), w['q_w'].data_ptr(), self.b_q_cross.data_ptr(), Sa, D, D, s)
                fvk.add_bias_fp16(self.b_q_cross.data_ptr(), w['q_b'].data_ptr(), Sa, D, s)
                # Use precomputed K/V projections (computed once before step loop)
                cross_idx = l // 2
                k_ptr = self._precomp_k[cross_idx].data_ptr()
                v_ptr = self._precomp_v[cross_idx].data_ptr()
                fvk.gpu_fill_neginf_fp16(self.b_attn_logits_cross.data_ptr(), self.b_attn_logits_cross.nelement(), s)
                if self.attn is not None:
                    # dit_cross site: cross_idx = l // 2 indexes the 16 cross layers
                    # (even DiT layers 0, 2, ..., 30).
                    self.attn.run("dit_cross", cross_idx,
                                   q_seq=Sa, kv_seq=S_kv, stream=s)
                else:
                    fvk.attention_mha_fp16(self.ctx, self.b_q_cross.data_ptr(), k_ptr, v_ptr,
                                           self.b_attn_logits_cross.data_ptr(), self.b_attn_out.data_ptr(), Sa, S_kv, NH, HD, 1.0/math.sqrt(HD), s)
            self._fp16_gemm(self.b_attn_out.data_ptr(), w['o_w'].data_ptr(), self.b_o.data_ptr(), Sa, D, D, s)
            fvk.add_bias_fp16(self.b_o.data_ptr(), w['o_b'].data_ptr(), Sa, D, s)
            fvk.residual_add_fp16(self.b_hidden.data_ptr(), self.b_o.data_ptr(), Sa*D, s)
            fvk.layer_norm_no_affine_fp16(self.b_hidden.data_ptr(), self.b_h_norm.data_ptr(), Sa, D, 1e-5, s)
            if self.use_fp8:
                # FFN-up + bias + GELU fused (3 kernels → 1)
                self._fp8_gemm_gelu_bias(self.b_h_norm.data_ptr(), w['ff_up_fp8'].data_ptr(),
                                          up_alpha, Sa, H, D, self.b_ff_h.data_ptr(),
                                          w['ff_up_b'].data_ptr(), as_up_ptr, s)
                # FFN-down + bias fused (2 kernels → 1)
                self._fp8_gemm_bias(self.b_ff_h.data_ptr(), w['ff_dn_fp8'].data_ptr(),
                                     dn_alpha, Sa, D, H, self.b_ff_out.data_ptr(),
                                     w['ff_dn_b'].data_ptr(), as_dn_ptr, s)
            else:
                self._fp16_gemm(self.b_h_norm.data_ptr(), w['ff_up_fp16'].data_ptr(), self.b_ff_h.data_ptr(), Sa, H, D, s)
                fvk.add_bias_fp16(self.b_ff_h.data_ptr(), w['ff_up_b'].data_ptr(), Sa, H, s)
                fvk.gelu_inplace_fp16(self.b_ff_h.data_ptr(), Sa*H, s)
                self._fp16_gemm(self.b_ff_h.data_ptr(), w['ff_dn_fp16'].data_ptr(), self.b_ff_out.data_ptr(), Sa, D, H, s)
                fvk.add_bias_fp16(self.b_ff_out.data_ptr(), w['ff_dn_b'].data_ptr(), Sa, D, s)
            fvk.residual_add_fp16(self.b_hidden.data_ptr(), self.b_ff_out.data_ptr(), Sa*D, s)
        fvk.ada_layer_norm_fp16(self.b_hidden.data_ptr(), self.out_scales[step].data_ptr(),
                                 self.out_shifts[step].data_ptr(), self.b_h_norm.data_ptr(), Sa, D, 1e-6, s)
        self._fp16_gemm(self.b_h_norm.data_ptr(), self.proj_out_2_w.data_ptr(), self.b_model_out.data_ptr(), Sa, 1024, D, s)
        fvk.add_bias_fp16(self.b_model_out.data_ptr(), self.proj_out_2_b.data_ptr(), Sa, 1024, s)
        self._fp16_gemm(self.b_model_out.data_ptr(), self.ad_w1.data_ptr(), self.b_dec_h.data_ptr(), Sa, 1024, 1024, s)
        fvk.add_bias_fp16(self.b_dec_h.data_ptr(), self.ad_b1.data_ptr(), Sa, 1024, s)
        fvk.relu_inplace_fp16(self.b_dec_h.data_ptr(), Sa*1024, s)
        self._fp16_gemm(self.b_dec_h.data_ptr(), self.ad_w2.data_ptr(), self.b_velocity.data_ptr(), Sa, self.action_dim, 1024, s)
        fvk.add_bias_fp16(self.b_velocity.data_ptr(), self.ad_b2.data_ptr(), Sa, self.action_dim, s)
        vel_offset = (self.Sa - self.T) * self.action_dim
        fvk.gpu_euler_step(self.b_actions.data_ptr(), self.b_velocity.data_ptr(), self.T, self.action_dim, dt, vel_offset, s)

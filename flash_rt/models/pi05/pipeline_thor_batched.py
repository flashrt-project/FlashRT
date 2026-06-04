"""FlashRT — Pi0.5 Thor SM110 batched (B>=1) inference pipeline.

Companion to :mod:`flash_rt.models.pi05.pipeline_thor` which holds
the B=1 main-line single-sample inference path. This module isolates
the B>=1 batched orchestration so the B=1 file stays small and easy
to reason about — single-sample inference is the production hot
path; batched is opt-in (used by the fused-CFG B=2 pipeline and the
generic infer_batch entry point).

Mirrors the RTX layout split between
:mod:`flash_rt.models.pi05.pipeline_rtx` and
:mod:`flash_rt.models.pi05.pipeline_rtx_batched`, and the existing
:mod:`flash_rt.models.pi05.pipeline_thor_cfg` /
:mod:`flash_rt.models.pi05.pipeline_thor_cfg_batched` cfg / cfg-
batched split.

Class hierarchy::

    Pi05ThorPipeline (pipeline_thor.py)                   # B=1 base
      └── Pi05ThorBatchedPipeline (this file)             # B>=1
            └── Pi05ThorCFGBatchedPipeline                # B=2 fused CFG
                (pipeline_thor_cfg_batched.py)

These classes do NOT own the buffer state (the Thor frontends own
all encoder / decoder / KV-cache buffers and capture the CUDA
Graphs). The pipeline class is a thin facade that:

  * holds ``batch_size`` so subclasses can advertise their B contract
  * provides ``run_pipeline(replay_siglip, replay_enc_ae)`` —
    backend-agnostic via callbacks, like
    :class:`Pi05ThorCFGPipeline`.

Functions:
    decoder_forward_b2  — Pi0.5 batched action-expert decoder forward.
"""

import math

from .pipeline_thor import Pi05ThorPipeline


def decoder_forward_b2(ctx, fvk, bufs, weights, dims, stream=0, *,
                       attn=None, B=2, cfg_beta=None):
    """Batched action-expert decoder forward for ``B`` independent samples.

    Stage 2 of the Thor batched-CFG port. Same kernel-class split as
    :func:`flash_rt.hardware.thor.shared_primitives_batched.encoder_forward_b2`:

      * Flat-elementwise + GEMM kernels scale via ``M = B*S``.
      * ``qkv_split_rope_kvcache_fp16`` and ``attention_qkv_fp16`` go
        through a per-sample inline loop with adjusted byte offsets
        and per-sample KV slabs (``Kc_b2[b]`` / ``Vc_b2[b]``).
      * AdaRMSNorm-style ops (``fused_adarms_fp8_static_fp16``,
        ``gate_res_adarms_fp8_static_fp16``, ``adarms_fp16``) read
        per-token style at row index. The ``sa`` / ``sf`` / ``fs``
        buffers MUST be **B-tiled** (each (step, layer) slice
        replicated B times along the row dim) before this function
        is called — same trick as RTX
        :class:`flash_rt.models.pi05.pipeline_rtx_batched.Pi05BatchedPipeline`
        Bug 6 fix at pipeline_rtx_batched.py:195–220. With B-tiled
        styles each kernel call uses the B*S row count and naturally
        walks both samples' style data.

    Args:
        bufs: same key set as
            :func:`flash_rt.models.pi05.pipeline_thor.decoder_forward`;
            row dims are B*S (flat). If ``cfg_beta`` is set, must
            additionally include ``v_b2`` — a fresh velocity output
            buffer of size ``B*S*32*2`` bytes used to hold per-step
            ``v_cond`` / ``v_uncond`` before the per-step CFG combine.
        weights: same as ``decoder_forward`` plus ``Kc_b2`` and
            ``Vc_b2`` — lists of B device pointers, each pointing at
            the sample's ``[La * total_keys * HD]`` flat KV slab.
            ``Kc`` / ``Vc`` (the B=1 keys) are ignored.
            ``sa``, ``sf``, ``fs`` MUST be B-tiled (see above).
        dims: same as ``decoder_forward``. ``S`` is per-sample
            sequence length (NOT B*S).
        attn: Optional. Same caveat as encoder_forward_b2 — the
            backend's encoder/decoder slot is single-batch, so we
            call ``fvk.attention_qkv_fp16`` directly per-sample
            in Stage 2.
        cfg_beta: Optional float. When given, requires ``B == 2`` and
            switches the per-step action-output stage from "integrate
            both slots' velocity into noise independently" to the
            per-step classifier-free-guidance schedule (matches RTX
            :meth:`Pi05CFGBatchedPipeline.transformer_decoder_batched`
            and arXiv:2511.14759 Appendix E):

              v_b2[slot 0]  ← v_cond  = action_head(x[slot 0])
              v_b2[slot 1]  ← v_uncond = action_head(x[slot 1])
              noise[slot 0] += v_uncond + cfg_beta * (v_cond - v_uncond)
              noise[slot 1]  ← noise[slot 0]   (mirror so both slots
                                                see the same x_t at
                                                step k+1)

            All ops are graph-capturable on ``stream``. ``cfg_beta=1.0``
            collapses to ``noise[slot 0] += v_cond`` (cond-only),
            matching the standard non-CFG path bit-exactly modulo the
            extra D2D mirror.
    """
    if cfg_beta is not None and B != 2:
        raise ValueError(
            f"cfg_beta requires B == 2 (got B={B}); CFG schedule is "
            "defined for cond-slot 0 / uncond-slot 1 pairs only")
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
    BS = B * S

    noise = bufs['noise']
    x = bufs['x']
    xn = bufs['xn']
    gate = bufs['gate']
    qkv = bufs['qkv']
    logits = bufs['logits']
    attn_out = bufs['attn_out']
    fg = bufs['fg']
    xn_fp8 = bufs['xn_fp8']
    hid_fp8 = bufs['hid_fp8']
    ctx_fp8 = bufs['ctx_fp8']

    ain_w = weights['ain_w']
    ain_b = weights['ain_b']
    sa = weights['sa']                 # B-tiled: (steps, layers, B*S, D3)
    qw = weights['qw']
    ow = weights['ow']
    sf = weights['sf']                 # B-tiled: (steps, layers, B*S, D3)
    gw = weights['gw']
    dw = weights['dw']
    aow = weights['aow']
    aob = weights['aob']
    fs = weights['fs']                 # B-tiled: (steps, B*S, D3)
    rope = weights['rope']
    w_scales = weights['w_scales']
    act_scales = weights['act_scales']

    Kc_b2 = weights['Kc_b2']
    Vc_b2 = weights['Vc_b2']
    if len(Kc_b2) != B or len(Vc_b2) != B:
        raise ValueError(
            f"Kc_b2/Vc_b2 must each have B={B} entries; "
            f"got {len(Kc_b2)} / {len(Vc_b2)}")

    qkv_stride_bytes = S * 2560 * 2
    attn_q_stride_bytes = S * Q_dim * 2

    for s in range(steps):
        step_scale_base = s * layers * 4
        # ── Action input: noise → x (M = B*S) ──
        fvk.gmm_fp16(ctx, noise, ain_w, x, BS, D, 32, 0.0, stream)
        fvk.add_bias_fp16(x, ain_b, BS, D, stream)

        for l in range(layers):
            # Style offsets (B-tiled buffer: sa[step, layer] starts at
            # ``(s * layers + l) * (B * S) * D3`` element offset).
            si = (s * layers + l) * BS * D3
            sa_ptr = sa + si * 2
            sf_ptr = sf + si * 2

            # ── C1: Fused AdaRMSNorm + style → FP8 (M = B*S) ──
            act_scale_qkv = act_scales + (step_scale_base + l * 4 + 0) * 4
            if l == 0:
                fvk.fused_adarms_fp8_static_fp16(
                    x, sa_ptr, xn_fp8, gate, BS, D, act_scale_qkv, stream)

            # ── C2: QKV GEMM (M = B*S) ──
            w_scale_qkv = w_scales + (l * 4 + 0) * 4
            qw_ptr = qw + l * D * 2560
            fvk.fp8_gemm_descale_fp16(
                xn_fp8, qw_ptr, qkv, BS, 2560, D,
                act_scale_qkv, w_scale_qkv, stream)

            # ── C2b: Per-sample RoPE + QKV split + KV cache ──
            kv_offset = l * total_keys * HD + enc_seq * HD
            for b in range(B):
                fvk.qkv_split_rope_kvcache_fp16(
                    qkv + b * qkv_stride_bytes,
                    rope,
                    attn_out + b * attn_q_stride_bytes,
                    Kc_b2[b], Vc_b2[b],
                    S, Q_dim, K_dim, HD, 2560,
                    kv_offset, HD, stream)

            # ── C3: Per-sample cross-attention ──
            for b in range(B):
                K_ptr = Kc_b2[b] + l * total_keys * HD * 2
                V_ptr = Vc_b2[b] + l * total_keys * HD * 2
                fvk.attention_qkv_fp16(
                    ctx,
                    attn_out + b * attn_q_stride_bytes,
                    K_ptr, V_ptr,
                    logits,  # scratch reused
                    attn_out + b * attn_q_stride_bytes,
                    S, total_keys, NH, HD, attn_scale, stream)

            # ── C4: Quantize attn → FP8 + O proj GEMM (flat, M = B*S) ──
            act_scale_o = act_scales + (step_scale_base + l * 4 + 1) * 4
            w_scale_o = w_scales + (l * 4 + 1) * 4
            fvk.quantize_fp8_static_fp16(
                attn_out, ctx_fp8, act_scale_o, BS * NH * HD, stream)
            ow_ptr = ow + l * NH * HD * D
            fvk.fp8_gemm_descale_fp16(
                ctx_fp8, ow_ptr, fg, BS, D, NH * HD,
                act_scale_o, w_scale_o, stream)

            # ── C4→C5: gate × residual + AdaRMSNorm → FP8 (M = B*S) ──
            act_scale_gu = act_scales + (step_scale_base + l * 4 + 2) * 4
            fvk.gate_res_adarms_fp8_static_fp16(
                fg, gate, x, sf_ptr,
                xn_fp8, gate, BS, D, act_scale_gu, stream)

            # ── C5: Gate+Up GEMM (M = B*S) ──
            w_scale_gu = w_scales + (l * 4 + 2) * 4
            gw_ptr = gw + l * D * H * 2
            fvk.fp8_gemm_descale_fp16(
                xn_fp8, gw_ptr, fg, BS, H * 2, D,
                act_scale_gu, w_scale_gu, stream)

            # ── C6: SiLU(gate) × up → FP8 (flat, M = B*S*H) ──
            act_scale_down = act_scales + (step_scale_base + l * 4 + 3) * 4
            fvk.gate_geglu_merged_fp8_fp16(
                fg, hid_fp8, BS, H, act_scale_down, stream)

            # ── C6: Down GEMM (M = B*S) ──
            w_scale_down = w_scales + (l * 4 + 3) * 4
            dw_ptr = dw + l * H * D
            fvk.fp8_gemm_descale_fp16(
                hid_fp8, dw_ptr, fg, BS, D, H,
                act_scale_down, w_scale_down, stream)

            # ── C7→C1_next: gate × residual + next-layer AdaRMSNorm → FP8 ──
            if l < layers - 1:
                si_next = (s * layers + l + 1) * BS * D3
                sa_next_ptr = sa + si_next * 2
                act_scale_next = act_scales + (step_scale_base + (l + 1) * 4 + 0) * 4
                fvk.gate_res_adarms_fp8_static_fp16(
                    fg, gate, x, sa_next_ptr,
                    xn_fp8, gate, BS, D, act_scale_next, stream)
            else:
                fvk.gate_res_fp16(fg, gate, x, BS * D, stream)

        # ── Final: AdaRMSNorm + action output (M = B*S) ──
        fi = s * BS * D3
        fs_ptr = fs + fi * 2
        fvk.adarms_fp16(x, fs_ptr, xn, gate, BS, D, stream)

        if cfg_beta is None:
            # Standard batched flow-matching integration: each slot
            # accumulates its own velocity into ``noise``.
            #   noise[i, :] = noise[i, :] + xn[i, :] @ aow + aob
            fvk.gmm_fp16(ctx, xn, aow, noise, BS, 32, D, 1.0, stream)
            fvk.add_bias_fp16(noise, aob, BS, 32, stream)
        else:
            # Per-step CFG (paper-correct, arXiv:2511.14759 App. E;
            # mirrors RTX
            # :meth:`Pi05CFGBatchedPipeline.transformer_decoder_batched`).
            # Write velocity into a SEPARATE buffer ``v_b2`` (β=0
            # GEMM = overwrite) so we can blend the two slots before
            # integrating into noise:
            v_b2 = bufs['v_b2']
            fvk.gmm_fp16(ctx, xn, aow, v_b2, BS, 32, D, 0.0, stream)
            fvk.add_bias_fp16(v_b2, aob, BS, 32, stream)
            # noise[slot 0] += v_uncond + cfg_beta * (v_cond - v_uncond)
            #   v_cond   = v_b2[0:S, :]   (slot 0)
            #   v_uncond = v_b2[S:2S, :]  (slot 1)
            per_slot_n = S * 32
            per_slot_bytes = per_slot_n * 2  # fp16
            v_cond_ptr = v_b2                                    # slot 0
            v_uncond_ptr = v_b2 + per_slot_bytes                 # slot 1
            noise_cond_ptr = noise                               # slot 0
            noise_uncond_ptr = noise + per_slot_bytes            # slot 1
            fvk.cfg_combine_into_residual_fp16(
                noise_cond_ptr, v_cond_ptr, v_uncond_ptr,
                cfg_beta, per_slot_n, stream)
            # Mirror the guided noise into the uncond slot so both
            # slots enter the next denoise step with identical x_t.
            # Without this the uncond trajectory drifts (plain Euler)
            # and per-step CFG degrades to per-chunk over 10 steps.
            fvk.gpu_copy(noise_uncond_ptr, noise_cond_ptr,
                         per_slot_bytes, stream)


class Pi05ThorBatchedPipeline(Pi05ThorPipeline):
    """B=N batched Pi0.5 Thor inference pipeline.

    Stage 2 of the Thor batched-CFG port. Runs ``B`` independent
    samples through the encoder + 10-step decoder in a single fused
    forward, sharing the GEMMs (M = B*Seq) across samples and only
    splitting per-sample at the per-token-indexed kernels
    (``qkv_split_rope_kvcache_fp16`` and ``attention_qkv_fp16``) via
    a python inline loop.

    The frontend captures a separate ``_enc_ae_graph_b2`` at B*Seq
    shapes and hands its replay callback in here. The base class's
    ``run_pipeline`` contract (replay_siglip + replay_enc_ae) is
    unchanged — only the underlying graphs differ.

    Args:
        batch_size: Must be ``>= 1``. Stage 3's
            :class:`flash_rt.models.pi05.pipeline_thor_cfg_batched.Pi05ThorCFGBatchedPipeline`
            sets ``batch_size = 2`` and assigns slot 0 = cond,
            slot 1 = uncond for fused CFG.

    Notes:
        * No buffer ownership; the frontend allocates ``_b2``-suffixed
          buffers and captures the B*Seq graph.
        * No new Thor kernels — see
          :func:`flash_rt.hardware.thor.shared_primitives_batched.encoder_forward_b2`
          and :func:`decoder_forward_b2` (this file)
          for the inline-loop strategy.
    """

    def __init__(self, *, batch_size: int = 2):
        if batch_size < 1:
            raise ValueError(
                f"Pi05ThorBatchedPipeline batch_size must be >= 1; "
                f"got B={batch_size}")
        # Skip the base-class B=1 guard via super().__init__(batch_size=1)
        # — we deliberately advertise the actual B>=1 contract.
        self.batch_size = int(batch_size)

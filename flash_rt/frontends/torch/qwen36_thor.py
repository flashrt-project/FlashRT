"""FlashRT — Qwen3.6-27B NVFP4 Thor frontend (SM110).

Per-hardware split required by ``docs/adding_new_model.md`` rule 2:
one ``(model, framework, hardware)`` file. The RTX frontend at
:mod:`flash_rt.frontends.torch.qwen36_rtx` is the canonical compute
path on RTX 5090; this module hosts the parallel Thor entry point.

Construction strategy
---------------------
The RTX frontend is monolithic (~10k LOC). The single Thor-incompatible
construction step is the attention-backend ctor at
``qwen36_rtx.py:1314``, which directly imports the vendored FA2
extension (``flash_rt_fa2`` — not built on Thor).

We swap that one construction site by patching the RTX attention
backend symbol to the Thor backend
(:class:`ThorFlashAttnBackendQwen36`) for the duration of the parent
``__init__``. Every other load step (NVFP4 weight extraction, MTP
head conversion, tokenizer, large buffer allocation, CUDA Graph
mempool setup) runs unchanged on Thor.

What still needs work after ctor lands cleanly
----------------------------------------------
The RTX frontend has four hot paths that bypass ``self._attn.run()``
and call ``self._attn._fa2_fwd[_causal]`` directly. The Thor
backend's ``_fa2_fwd`` / ``_fa2_fwd_causal`` attributes are currently
``None``; calling any of the four hot paths today will raise. Wiring
them to ``fvk.qwen36_flashinfer_xqa_bf16_fp8kv_spec`` with bf16->fp8
paged staging is the focused next milestone. See
``dev_log_qwen36_thor/step3b_corrected_per_token_budget.md`` for the
kernel-level cost data the integration is targeting.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

# Thor-only default for the long-context chunked prefill's GDN backend.
# The RTX default ``wy_lt`` (cublasLt WY-decomposition chunk scan) is
# numerically non-equivalent to the per-token recurrent path on SM110:
# every linear-attention layer drifts by ~5e-4 vs. the per-token output,
# the drift compounds, and downstream MTP spec acceptance collapses.
# The ``native`` per-step recurrent backend is bit-exact to the
# per-token path on Thor (see step5e_chunked_prefill_drift_root_cause.md
# for the per-layer cosine measurement). Setting via ``setdefault`` lets
# the user still pin a specific backend for bisection work.
os.environ.setdefault(
    "FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND", "native")


class _ThorFirstChunkReplay:
    """Stand-in for a captured CUDA graph at positions inside the
    Thor first-chunk pre-fill window. Implements ``.replay()`` so the
    parent's prefill loop can call it the same way it calls a real
    captured graph; the body just copies the corresponding per-position
    hidden out of the pre-computed batched K-row output into the M=1
    decode buffer the parent expects to read from.
    """
    __slots__ = ("_src", "_dst")

    def __init__(self, src, dst):
        self._src = src
        self._dst = dst

    def replay(self) -> None:
        self._dst.copy_(self._src)


@contextmanager
def _use_thor_attn_backend():
    """Replace the RTX attn backend symbol with the Thor backend for
    the duration of ``Qwen36TorchFrontendRtx.__init__``."""
    import flash_rt.hardware.rtx.attn_backend_qwen36 as rtx_mod
    from flash_rt.hardware.thor.attn_backend_qwen36 import (
        ThorFlashAttnBackendQwen36,
    )
    saved = rtx_mod.RtxFlashAttnBackendQwen36
    rtx_mod.RtxFlashAttnBackendQwen36 = ThorFlashAttnBackendQwen36
    try:
        yield
    finally:
        rtx_mod.RtxFlashAttnBackendQwen36 = saved


class Qwen36TorchFrontendThor(Qwen36TorchFrontendRtx):
    """Qwen3.6-27B NVFP4 Torch frontend for Jetson Thor (SM110).

    Inherits the RTX frontend's loader, weight handling, MTP draft
    chain, sampling, and generate loop. Swaps in the Thor attention
    backend at ctor time. The Thor backend mirrors the RTX backend's
    buffer surface (K_cache, V_cache, Q_buf, O_buf, lse_buf, ...) so
    the loader does not need per-arch branches.

    Routing policy override
    -----------------------
    The RTX frontend's ``_should_use_long_ctx_route`` carries a
    "128-token exception" that forces prompts in ``[128, 192)`` through
    the chunked FP8-KV long-ctx path because on RTX 5090 the chunked
    forward is materially faster than the per-token spec walk
    (``docs/qwen36_nvfp4.md`` §5: TTFT 31 ms vs. seconds on the per-
    token path). On Thor, profiling shows the opposite trade: the
    chunked forward on Thor leaves a per-position hidden-state drift
    versus the per-token forward (mean cosine 0.94, min 0.51 over a
    128-token prompt), which collapses MTP spec acceptance from
    AL=4.07 (per-token path) to AL=1.21 (chunked path). Until that
    drift is root-caused at the kernel level, this subclass takes the
    "AL > TTFT" branch on Thor for short prompts: anything short
    enough to fit the BF16 spec window keeps the per-token spec path.
    The chunked path stays available for prompts that exceed the
    BF16 window (long-ctx serving was the original target of that
    mode anyway).

    See ``dev_log_qwen36_thor/step5c_thor_complete_AB_findings.md``
    for the diff numbers and the per-position cosine trace that
    motivated this override.

    Linear-attention chunked backend (Thor default)
    -----------------------------------------------
    The module-level ``setdefault`` for
    ``FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND`` pins the per-step
    recurrent GDN backend on Thor. The RTX default ``wy_lt`` chunk
    backend produces a measurable per-position drift on SM110
    (mean cos 0.999979 → 0.999721 at layer 2; compounds layer-by-layer
    to a cos = 0.43 floor by layer 63) while ``native`` is bit-exact
    to the per-token recurrent path. See
    ``dev_log_qwen36_thor/step5e_chunked_prefill_drift_root_cause.md``
    for the per-layer hidden-state cosine table.
    """

    # Default cap below which the Thor frontend forces the short-ctx
    # (per-token spec) route. Above this the long-ctx chunked path
    # engages — that's still the only viable option for prompts past
    # the BF16 spec window.
    _THOR_LONG_CTX_FORCE_MIN_SEQ: int = 8192

    # K-row layer dispatch threshold on Thor.
    #
    # The K-row layer kernel chain on SM110 (NVFP4 Qwen3.6) shares the
    # SM120 implementation but produces a different hidden state vs the
    # per-token forward at certain (cur_pos, K) combinations — every
    # individual kernel is row-deterministic at M=K vs M=1, yet the
    # composite chain at M=K behaves differently. The same source on
    # SM120 (5090) is bit-exact across all (cur_pos, K). Root-causing
    # the kernel-chain divergence at the SASS / PTX level is a separate
    # workstream; the production path takes the mathematically
    # equivalent rewrite below instead.
    #
    # For K ≤ THRESHOLD the parent's K-row method is used unchanged —
    # this preserves the production short-ctx spec-verify decode (K=7-8
    # at cur_pos > prompt_len), which has been verified to match
    # per-token by direct AL measurement. For K > THRESHOLD (i.e.
    # prefill chunks) we dispatch to K sequential single-token forwards
    # of the parent class. Single-token semantics are the ground truth
    # in this build and are bit-exact at every (cur_pos, K=1) we have
    # tested on Thor, so the chunked prefill state evolution is
    # bit-exact to the legacy per-token prefill walk.
    # Threshold matches ``_K_save_max=8`` from the parent: at K ≤ 8 the
    # parent's K-row enters the per-step ``save_steps>0`` branch, which
    # uses the in/out-state conv1d + recurrent kernels for exact-step
    # state checkpointing — production spec-verify decode (K=7-8) relies
    # on this path and is known bit-exact on Thor (AL=3.93 in prod).
    # At K > 8 the parent's K-row enters the ``save_steps=0`` chunk
    # branch, which is where the SM110 kernel-chain divergence lives;
    # the dispatch below replaces that branch with K sequential
    # single-token forwards.
    _THOR_K_ROW_FAST_PATH_MAX: int = 7

    # First-chunk prefill K-row size for short-ctx TTFT optimization.
    # The Thor K-row at K=22 cur_pos=0 is verified byte-for-byte
    # equivalent to running K=22 single-token forwards (probe
    # ``probe_thor_K_row_bit_exact`` reports zero diff across all
    # layers and all four state caches). Running the first 22 prefill
    # positions through one batched K-row call instead of 22 graph
    # replays is a 12.7x speedup (≈ 113 ms vs 1430 ms) per chunk on
    # the Thor BW-bound regime, so the prompt is short enough to fit
    # the BF16 retention window we trade off 22 graph replays for one.
    # Production sweet spot: K=128 first-chunk for ctx=128 prompts —
    # entire prefill in one K-row call, AL=3.93 preserved (verified
    # across the K∈{96, 110, 120, 124, 126, 127, 128} sweep). At
    # K=128 the row 42 of layer 63 picks up ~ULP drift vs per-token
    # (the only divergent layer), but MTP head absorbs that noise.
    # Override via env if needed for bisection.
    _THOR_FIRST_CHUNK_K: int = int(
        os.environ.get("FLASHRT_QWEN36_THOR_FIRST_CHUNK_K", "128"))

    def __init__(self, *args, **kwargs):
        with _use_thor_attn_backend():
            super().__init__(*args, **kwargs)
        self._thor_alloc_K_row_scratch()
        self._thor_first_chunk_active: bool = False
        self._thor_first_chunk_input_ids = None

    # ---------- Thor-native K-row scratch ----------
    #
    # The parent class allocates a number of K-row scratch buffers
    # (``_K_lin_*`` / ``_K_full_*`` / ``_K_mlp_*``) under env flags whose
    # default values flip with ``_long_ctx_mode``. In particular,
    # ``_mlp_up_out`` and the ``_nvfp4_scratch[(17408, 5120)][2]`` gate
    # output both shrink to ``(1, 17408)`` when
    # ``_enable_mlp_gate_up_fusion=True``, which is the default in
    # long-context mode. That makes the parent's M=1 buffers unsafe for
    # writes at M=K. The buffers below are owned by this subclass and
    # always sized for the K-row's M=K upper bound, so the Thor K-row
    # implementations below never alias a fusion-shrunk parent buffer.
    def _thor_alloc_K_row_scratch(self) -> None:
        import torch
        device = self._h_b.device
        bf16 = torch.bfloat16
        Kmax = self.MAX_Q_SEQ
        # MLP gate / up / silu(gate)*up — own (Kmax, 17408) outputs.
        self._thor_gate_K = torch.empty(
            Kmax, 17408, device=device, dtype=bf16)
        self._thor_up_K = torch.empty(
            Kmax, 17408, device=device, dtype=bf16)
        self._thor_silu_K = torch.empty(
            Kmax, 17408, device=device, dtype=bf16)

    # ---------- K-row layer overrides (Thor-only) ----------
    #
    # Linear-attention K-row forward at K above the fast-path threshold.
    # Routes to a from-scratch Thor implementation that batches every
    # GEMM / norm / quantize / element-wise op at M=K while keeping the
    # state-bearing ops (causal_conv1d_update, GDN recurrent) on a
    # per-position sub-loop. Bit-exact to running K sequential single-
    # token forwards (see DESIGN §4.5 for the leaf-kernel set).
    def _layer_forward_lin_K_nvfp4(self, L, h_in_K, K):
        if K <= self._THOR_K_ROW_FAST_PATH_MAX:
            return super()._layer_forward_lin_K_nvfp4(L, h_in_K, K)
        if K > self.MAX_Q_SEQ:
            return self._thor_lin_K_dispatch(L, h_in_K, K)
        return self._thor_lin_K_forward(L, h_in_K, K)

    def _layer_forward_full_K_nvfp4(
            self, L, h_in_K, cos_K, sin_K, cur_pos, K):
        if K <= self._THOR_K_ROW_FAST_PATH_MAX:
            return super()._layer_forward_full_K_nvfp4(
                L, h_in_K, cos_K, sin_K, cur_pos, K)
        if K > self.MAX_Q_SEQ:
            return self._thor_full_K_dispatch(
                L, h_in_K, cos_K, sin_K, cur_pos, K)
        return self._thor_full_K_forward(
            L, h_in_K, cos_K, sin_K, cur_pos, K)

    # ---------- Thor-native lin-attn K-row layer ----------
    #
    # Mirrors the per-token ``_layer_forward_lin_nvfp4`` math step by
    # step, scaled to M=K. The leaf-kernel set is the per-token-
    # equivalent subset that has been verified row-deterministic at
    # M=K on Thor (no pingpong, no fused norm+quant, no fused
    # mlp_gate_up, no ab96 paired kernel, no WY chunk scan). State-
    # bearing ops walk per-position so the recurrent state evolves
    # exactly as in the per-token path.
    def _thor_lin_K_forward(self, L, h_in_K, K):
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'linear_attention', (
            f'_thor_lin_K_forward layer {L} type {lw["type"]!r}'
        )
        eps = float(self._cfg['rms_norm_eps'])

        h2 = h_in_K.view(K, 5120)
        # (1) input rms_norm @ M=K. _h_b is (max_seq, 5120) so [:K] is safe.
        x_norm = self._h_b[:K].view(K, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(),
            K, 5120, eps, s,
        )

        # (2) NVFP4 quantize x_norm — reused by in_proj_qkv / in_proj_z.
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(10240, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), K, 5120, s,
        )

        # (3) in_proj_qkv NVFP4 GEMM @ M=K, N=10240.
        out_qkv_buf = self._nvfp4_scratch[(10240, 5120)][2]
        out_qkv_K = out_qkv_buf[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['in_proj_qkv_packed']),
            out_qkv_K.data_ptr(),
            K, 10240, 5120,
            sf_5120.data_ptr(), int(lw['in_proj_qkv_sf']),
            float(lw['in_proj_qkv_alpha']),
            s,
        )

        # (4) in_proj_z NVFP4 GEMM @ M=K, N=6144.
        out_z_buf = self._nvfp4_scratch[(6144, 5120)][2]
        out_z_K = out_z_buf[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['in_proj_z_packed']),
            out_z_K.data_ptr(),
            K, 6144, 5120,
            sf_5120.data_ptr(), int(lw['in_proj_z_sf']),
            float(lw['in_proj_z_alpha']),
            s,
        )

        # (5) in_proj_ab BF16 matmul @ M=K, N=96, K_in=5120.
        # Always go through the K-row of bf16_matmul_qwen36_bf16 (not
        # the ab96 paired SM120 kernel), so per-row reduction order
        # matches the per-token bf16_matvec_qwen36_bf16 byte-for-byte.
        a_vec_K = self._K_lin_a_vec[:K]
        b_vec_K = self._K_lin_b_vec[:K]
        fvk.bf16_matmul_qwen36_bf16(
            x_norm.data_ptr(), int(lw['in_proj_ab_w']),
            self._K_lin_ab_vec[:K].data_ptr(), K, 96, 5120, s,
        )

        # (6) causal_conv1d_update — chunk variant (1 launch for K
        # iters). Byte-equal to per-token at K=22 cur_pos=0; tiny
        # ULP drift at larger K but well below MTP tolerance.
        lin_rank = self._linear_layer_rank(L)
        conv_state = self._lin_conv_state[lin_rank]
        conv_out_K = self._K_lin_conv_out[:K]
        fvk.causal_conv1d_qwen36_update_chunk_bf16(
            out_qkv_K.data_ptr(), int(lw['conv1d_w']),
            int(lw['conv1d_b']),
            conv_out_K.data_ptr(), conv_state.data_ptr(),
            1, K, 10240, 4, True, s,
        )

        # (7-9) Fused conv_out -> split + Q/K broadcast + GDN gating
        # + GDN chunk recurrent in one launch. Replaces three separate
        # launches with the fused chunk-scan kernel.
        rec_state = self._lin_state[lin_rank]
        attn_out_K = self._K_lin_attn_out[:K]
        a_stride = a_vec_K.stride(0)
        b_stride = b_vec_K.stride(0)
        fvk.qwen36_gdn_chunk_from_conv_smem_strided_bf16(
            conv_out_K.data_ptr(),
            a_vec_K.data_ptr(), b_vec_K.data_ptr(),
            lw['neg_A_log_exp_fp32_t'].data_ptr(),
            lw['dt_bias_fp32_t'].data_ptr(),
            rec_state.data_ptr(),
            attn_out_K.data_ptr(),
            K, 48, a_stride, b_stride, True, s,
        )

        # (10) rms_norm_gated_silu @ M=K*48, dim=128.
        attn_out_flat = attn_out_K.view(K * 48, 128)
        z_flat = out_z_K.view(K * 48, 128)
        norm_out_K = self._K_lin_norm_out[:K]
        norm_out_flat = norm_out_K.view(K * 48, 128)
        fvk.rms_norm_gated_silu_qwen36_bf16(
            attn_out_flat.data_ptr(), z_flat.data_ptr(),
            int(lw['head_norm_w']),
            norm_out_flat.data_ptr(),
            K * 48, 128, eps, s,
        )

        # (11) Quantize norm_out (K, 6144) -> ap_6144, sf_6144.
        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        norm_out_2d = norm_out_K.view(K, 6144)
        fvk.quantize_bf16_to_nvfp4_swizzled(
            norm_out_2d.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), K, 6144, s,
        )

        # (12) out_proj NVFP4 GEMM @ M=K, N=5120, K_in=6144.
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        out_op_K = out_op_buf[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(lw['out_proj_packed']),
            out_op_K.data_ptr(),
            K, 5120, 6144,
            sf_6144.data_ptr(), int(lw['out_proj_sf']),
            float(lw['out_proj_alpha']),
            s,
        )

        # (13) Post-attn residual h_in + attn_proj -> _K_res_mid.
        # NB: parent K-row fuses (add + rms_norm + quant) into one
        # ``residual_add_rms_norm_to_nvfp4_swizzled_bf16`` launch, but
        # the per-token forward (which production reads
        # ``_prefill_h_cache`` from) uses the split path; the fused
        # kernel keeps the intermediate in FP32 while split rounds
        # through BF16, which yields slightly different SF entries
        # and hence different MLP-input hidden. The K-row first-chunk
        # writes into the same ``_prefill_h_cache`` that the MTP head
        # then consumes, so we MUST match the per-token kernel choice
        # exactly — otherwise MTP sees a distribution shift and AL
        # collapses (measured: 3.93 -> 2.15 at K=6 when swapping in
        # the fused kernel).
        attn_proj = out_op_K.view(1, K, 5120)
        res_mid_K = self._K_res_mid[:, :K]
        fvk.add_bf16_out(
            h_in_K.data_ptr(), attn_proj.data_ptr(),
            res_mid_K.data_ptr(), K * 5120, s,
        )
        h_post = res_mid_K

        # (14) post-attn rms_norm.
        x_mlp = self._h_b[:K].view(K, 5120)
        h_post_view = h_post.view(K, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(),
            K, 5120, eps, s,
        )

        # (15) Quantize x_mlp for MLP gate / up.
        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_mlp.data_ptr(), ap_mlp.data_ptr(),
            sf_mlp.data_ptr(), K, 5120, s,
        )

        # (16-17) MLP gate / up — separate NVFP4 widen GEMMs @ M=K.
        # ``_thor_gate_K`` / ``_thor_up_K`` are this subclass's owned
        # (Kmax, 17408) buffers so they're safe regardless of the
        # parent's ``_enable_mlp_gate_up_fusion`` shape collapse.
        gate_out_K = self._thor_gate_K[:K]
        up_out_K = self._thor_up_K[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_K.data_ptr(),
            K, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_up_packed']),
            up_out_K.data_ptr(),
            K, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )

        # (18) silu(gate) * up.
        silu_out = self._thor_silu_K[:K]
        fvk.silu_mul_qwen36_bf16(
            gate_out_K.data_ptr(), up_out_K.data_ptr(),
            silu_out.data_ptr(), K * 17408, s,
        )

        # (19) Quantize silu_out for MLP down.
        ap_dn, sf_dn, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            silu_out.data_ptr(), ap_dn.data_ptr(),
            sf_dn.data_ptr(), K, 17408, s,
        )

        # (20) MLP down NVFP4 GEMM @ M=K, N=5120, K_in=17408.
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_dn.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            K, 5120, 17408,
            sf_dn.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:K].view(1, K, 5120)

        # (21) Final residual h_post + mlp_out -> _K_layer_out_{a,b}[:K].
        h_out_full = (self._K_layer_out_a if (L % 2 == 0)
                      else self._K_layer_out_b)
        h_out_K = h_out_full[:, :K]
        fvk.add_bf16_out(
            h_post.data_ptr(), mlp_out.data_ptr(),
            h_out_K.data_ptr(), K * 5120, s,
        )
        return h_out_K

    # ---------- Thor-native full-attn K-row layer ----------
    #
    # Mirrors the per-token ``_layer_forward_full_nvfp4`` math step by
    # step at M=K. The K projections / norms / quantize ops batch
    # cleanly; partial_rope + V write + attention walk per-position so
    # each K row of Q lands at ``Q_buf[:, :1]`` exactly like the per-
    # token forward — letting ``_attn.run('full', q_seq=1, ...)`` read
    # the right Q without any extra copies. The corresponding K row of
    # the layer output is captured from ``O_buf[:, :1]`` before the next
    # iteration overwrites it.
    def _thor_full_K_forward(self, L, h_in_K, cos_K, sin_K, cur_pos, K):
        import torch
        from flash_rt import flash_rt_kernels as fvk

        s = torch.cuda.current_stream().cuda_stream
        lw = self._weights.ptrs['layers'][L]
        assert lw['type'] == 'full_attention', (
            f'_thor_full_K_forward layer {L} type {lw["type"]!r}'
        )
        eps = float(self._cfg['rms_norm_eps'])
        full_rank = self._full_layer_rank(L)

        h2 = h_in_K.view(K, 5120)
        # (1) input rms_norm @ M=K. Per-token full-attn at line 1687
        # uses the SPLIT (rms_norm + separate quant) path even though
        # the fused kernel exists — keeping the same kernel choice
        # here so _prefill_h_cache fed to MTP head sees the same
        # rounding profile.
        x_norm = self._h_b[:K].view(K, 5120)
        fvk.rms_norm(
            h2.data_ptr(), int(lw['input_norm_eff_w']),
            x_norm.data_ptr(),
            K, 5120, eps, s,
        )

        # (2) NVFP4 quantize x_norm — reused for q/k/v projections.
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(12288, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), K, 5120, s,
        )

        # (3) q_proj NVFP4 GEMM @ M=K, N=12288 (Q + output_gate fused).
        q_proj_out_buf = self._nvfp4_scratch[(12288, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['q_proj_packed']),
            q_proj_out_buf.data_ptr(),
            K, 12288, 5120,
            sf_5120.data_ptr(), int(lw['q_proj_sf']),
            float(lw['q_proj_alpha']),
            s,
        )
        q_pre_2d = self._K_full_q_rot[:, :K].view(K * 24, 256)
        gate_flat = self._K_full_gate_sig[:, :K]
        fvk.qwen36_split_q_gate_bf16(
            q_proj_out_buf[:K].data_ptr(), q_pre_2d.data_ptr(),
            gate_flat.data_ptr(), K, s,
        )

        # (4) k_proj NVFP4 GEMM @ M=K, N=1024.
        kv_proj_out_buf = self._nvfp4_scratch[(1024, 5120)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['k_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            K, 1024, 5120,
            sf_5120.data_ptr(), int(lw['k_proj_sf']),
            float(lw['k_proj_alpha']),
            s,
        )
        k_pre_K = kv_proj_out_buf[:K].view(K, 4, 256).clone()

        # (5) q_norm / k_norm (per-head RMSNorm, row-independent).
        q_norm_out = self._K_full_q_norm_out[:K * 24]
        fvk.rms_norm(
            q_pre_2d.data_ptr(), int(lw['q_norm_eff_w']),
            q_norm_out.data_ptr(),
            K * 24, 256, eps, s,
        )
        k_pre_2d = k_pre_K.view(K * 4, 256)
        k_norm_out = self._K_full_k_norm_out[:K * 4]
        fvk.rms_norm(
            k_pre_2d.data_ptr(), int(lw['k_norm_eff_w']),
            k_norm_out.data_ptr(),
            K * 4, 256, eps, s,
        )

        # (6) v_proj NVFP4 GEMM @ M=K, N=1024 (overwrites kv_proj scratch
        # — k_pre_K already cloned above).
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(lw['v_proj_packed']),
            kv_proj_out_buf.data_ptr(),
            K, 1024, 5120,
            sf_5120.data_ptr(), int(lw['v_proj_sf']),
            float(lw['v_proj_alpha']),
            s,
        )
        v_new_K = kv_proj_out_buf[:K].view(K, 4, 256)

        # Pre-compute view of q_norm laid out per-position for the loop.
        q_norm_K = q_norm_out.view(K, 24, 256)
        k_norm_K = k_norm_out.view(K, 4, 256)

        # Long-context FP8 KV mirror is required when the parent's
        # chunked prefill set ``_fp8_kv_verify_active=True``: the
        # decode-time spec verify reads from ``_fp8_K_cache`` /
        # ``_fp8_V_cache`` so we have to mirror after writing BF16.
        write_fp8 = bool(getattr(self, "_fp8_kv_verify_active", False))

        # (7+8) Per-position partial RoPE, V copy, attention. Each
        # iteration writes Q to ``Q_buf[:, :1]`` and runs q_seq=1 XQA,
        # capturing the output row before the next iteration overwrites
        # ``O_buf[:, :1]``.
        d = self._rope_dim
        cos_3d = cos_K.view(1, K, d)
        sin_3d = sin_K.view(1, K, d)
        scaling = float(self._cfg['head_dim']) ** -0.5
        # Stage per-iteration attention outputs into _K_full_q_rot —
        # already consumed by partial_rope above (Q rows landed in
        # _attn.Q_buf), so the (1, Kmax, 24, 256) slot is free for the
        # rest of the layer. Keeping this distinct from _K_full_gated
        # avoids aliasing with the sigmoid_mul output buffer.
        attn_out_K = self._K_full_q_rot[:, :K].view(K, 24, 256)
        for k in range(K):
            pos_k = cur_pos + k
            cos_k = cos_3d[:, k].contiguous()
            sin_k = sin_3d[:, k].contiguous()
            q_dst = self._attn.Q_buf[:, :1]
            k_dst = self._attn.K_cache[full_rank, pos_k:pos_k + 1]
            fvk.qwen36_partial_rope_qk_bf16(
                q_norm_K[k].data_ptr(), k_norm_K[k].data_ptr(),
                cos_k.data_ptr(), sin_k.data_ptr(),
                q_dst.data_ptr(), k_dst.data_ptr(),
                1, 24, 4, 256, 64, s,
            )
            fvk.gpu_copy(
                self._attn.V_cache[
                    full_rank, pos_k:pos_k + 1].data_ptr(),
                v_new_K[k:k + 1].data_ptr(), 4 * 256 * 2, s,
            )
            if write_fp8:
                self._fp8_write_kv(
                    full_rank, pos_k, pos_k + 1,
                    k_dst.view(1, 4, 256),
                    v_new_K[k:k + 1].view(1, 4, 256),
                )
            self._attn.run(
                'full', layer_idx=full_rank, q_seq=1,
                kv_seq=pos_k + 1, stream=s, softmax_scale=scaling,
            )
            attn_out_K[k].copy_(
                self._attn.O_buf[:, 0].view(24, 256))

        # (9) Output gate: attn * sigmoid(gate). K rows in one launch.
        attn_flat = attn_out_K.view(1, K, 24 * 256)
        gated = self._K_full_gated[:, :K].view(1, K, 24 * 256)
        fvk.sigmoid_mul_qwen36_bf16(
            gate_flat.data_ptr(), attn_flat.data_ptr(),
            gated.data_ptr(), K * 24 * 256, s,
        )

        # (10) o_proj NVFP4 GEMM @ M=K, N=5120, K_in=6144.
        ap_6144, sf_6144, _ = self._nvfp4_scratch[(5120, 6144)]
        gated_2d = gated.view(K, 6144)
        fvk.quantize_bf16_to_nvfp4_swizzled(
            gated_2d.data_ptr(), ap_6144.data_ptr(),
            sf_6144.data_ptr(), K, 6144, s,
        )
        out_op_buf = self._nvfp4_scratch[(5120, 6144)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_6144.data_ptr(), int(lw['o_proj_packed']),
            out_op_buf.data_ptr(),
            K, 5120, 6144,
            sf_6144.data_ptr(), int(lw['o_proj_sf']),
            float(lw['o_proj_alpha']),
            s,
        )

        # (11) Post-attn residual h_in + attn_proj -> _K_res_mid.
        # See lin-attn note above on why we keep split (matches per-
        # token kernel choice).
        attn_proj = out_op_buf[:K].view(1, K, 5120)
        res_mid_K = self._K_res_mid[:, :K]
        fvk.add_bf16_out(
            h_in_K.data_ptr(), attn_proj.data_ptr(),
            res_mid_K.data_ptr(), K * 5120, s,
        )
        h_post = res_mid_K

        # (12) post-attn rms_norm.
        x_mlp = self._h_b[:K].view(K, 5120)
        h_post_view = h_post.view(K, 5120)
        fvk.rms_norm(
            h_post_view.data_ptr(), int(lw['post_attn_norm_eff_w']),
            x_mlp.data_ptr(),
            K, 5120, eps, s,
        )

        # (13) Quantize x_mlp for MLP gate / up.
        ap_mlp, sf_mlp, _ = self._nvfp4_scratch[(17408, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_mlp.data_ptr(), ap_mlp.data_ptr(),
            sf_mlp.data_ptr(), K, 5120, s,
        )

        # (14-15) MLP gate / up — separate widen NVFP4 GEMMs @ M=K.
        gate_out_K = self._thor_gate_K[:K]
        up_out_K = self._thor_up_K[:K]
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_gate_packed']),
            gate_out_K.data_ptr(),
            K, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_gate_sf']),
            float(lw['mlp_gate_alpha']),
            s,
        )
        fvk.fp4_w4a16_gemm_sm120_bf16out_widen(
            ap_mlp.data_ptr(), int(lw['mlp_up_packed']),
            up_out_K.data_ptr(),
            K, 17408, 5120,
            sf_mlp.data_ptr(), int(lw['mlp_up_sf']),
            float(lw['mlp_up_alpha']),
            s,
        )

        # (16) silu(gate) * up.
        silu_out = self._thor_silu_K[:K]
        fvk.silu_mul_qwen36_bf16(
            gate_out_K.data_ptr(), up_out_K.data_ptr(),
            silu_out.data_ptr(), K * 17408, s,
        )

        # (17) Quantize silu_out for MLP down.
        ap_dn, sf_dn, _ = self._nvfp4_scratch[(5120, 17408)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            silu_out.data_ptr(), ap_dn.data_ptr(),
            sf_dn.data_ptr(), K, 17408, s,
        )

        # (18) MLP down NVFP4 GEMM @ M=K, N=5120, K_in=17408.
        down_out_buf = self._nvfp4_scratch[(5120, 17408)][2]
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_dn.data_ptr(), int(lw['mlp_down_packed']),
            down_out_buf.data_ptr(),
            K, 5120, 17408,
            sf_dn.data_ptr(), int(lw['mlp_down_sf']),
            float(lw['mlp_down_alpha']),
            s,
        )
        mlp_out = down_out_buf[:K].view(1, K, 5120)

        # (19) Final residual h_post + mlp_out -> _K_layer_out_{a,b}[:K].
        h_out_full = (self._K_layer_out_a if (L % 2 == 0)
                      else self._K_layer_out_b)
        h_out_K = h_out_full[:, :K]
        fvk.add_bf16_out(
            h_post.data_ptr(), mlp_out.data_ptr(),
            h_out_K.data_ptr(), K * 5120, s,
        )
        return h_out_K

    # ---------- Dispatch fallback (K > MAX_Q_SEQ panic path) ----------
    #
    # When the requested K-row chunk exceeds the K-row scratch capacity,
    # fall back to a per-position single-token walk. Bit-exact to a
    # per-token forward; used as a safety hatch only.
    def _thor_lin_K_dispatch(self, L, h_in_K, K):
        import torch
        from flash_rt import flash_rt_kernels as fvk

        hidden = self._cfg["hidden_size"]
        lin_rank = self._linear_layer_rank(L)
        s = torch.cuda.current_stream().cuda_stream
        save_steps = K if K <= self._K_save_max else 0
        lin_state_slot = self._lin_state[lin_rank]
        lin_conv_slot = self._lin_conv_state[lin_rank]
        ls_bytes = lin_state_slot.numel() * 2
        lc_bytes = lin_conv_slot.numel() * 2
        h_out_K = (self._K_layer_out_a if (L % 2 == 0)
                   else self._K_layer_out_b)[:, :K]
        for r in range(K):
            h_in_r = h_in_K[:, r:r + 1, :].view(1, hidden).contiguous()
            h_out_r = super()._layer_forward_lin_nvfp4(L, h_in_r)
            h_out_K[:, r:r + 1, :].copy_(h_out_r.view(1, 1, hidden))
            if save_steps > 0:
                fvk.gpu_copy(
                    self._K_lin_state_per_step[r, lin_rank].data_ptr(),
                    lin_state_slot.data_ptr(), ls_bytes, s)
                fvk.gpu_copy(
                    self._K_lin_conv_state_per_step[r, lin_rank].data_ptr(),
                    lin_conv_slot.data_ptr(), lc_bytes, s)
        return h_out_K

    def _thor_full_K_dispatch(self, L, h_in_K, cos_K, sin_K, cur_pos, K):
        hidden = self._cfg["hidden_size"]
        d = self._rope_dim
        cos_3d = cos_K.view(1, K, d)
        sin_3d = sin_K.view(1, K, d)
        h_out_K = (self._K_layer_out_a if (L % 2 == 0)
                   else self._K_layer_out_b)[:, :K]
        write_fp8 = bool(getattr(self, "_fp8_kv_verify_active", False))
        full_rank = self._full_layer_rank(L) if write_fp8 else None
        for r in range(K):
            h_in_r = h_in_K[:, r:r + 1, :].view(1, hidden).contiguous()
            cos_r = cos_3d[:, r].contiguous()
            sin_r = sin_3d[:, r].contiguous()
            h_out_r = super()._layer_forward_full_nvfp4(
                L, h_in_r, cos_r, sin_r, cur_pos + r)
            h_out_K[:, r:r + 1, :].copy_(h_out_r.view(1, 1, hidden))
            if write_fp8:
                pos = cur_pos + r
                k_row = self._attn.K_cache[
                    full_rank, pos:pos + 1].view(1, 4, 256)
                v_row = self._attn.V_cache[
                    full_rank, pos:pos + 1].view(1, 4, 256)
                self._fp8_write_kv(full_rank, pos, pos + 1, k_row, v_row)
        return h_out_K

    # ---------- Short-ctx first-chunk K-row prefill optimization ----------
    #
    # The parent's ``generate_own_speculative_KN_nvfp4`` iterates one
    # captured single-token graph per prompt position. On Thor that
    # walks the 9.4 s BW-bound TTFT roofline at ctx=128. The Thor-
    # native K-row layer at K=22 cur_pos=0 is byte-equivalent to that
    # walk (probe verified) and runs in ~113 ms instead of ~1430 ms
    # for the first 22 positions (12.7x speedup). We splice that
    # speedup in by:
    #
    #   1. Overriding ``generate_own_speculative_KN_nvfp4`` to set a
    #      pending flag and stash ``input_ids`` before delegating to
    #      the parent.
    #   2. Overriding ``_ensure_graph_for_pos_nvfp4`` so that the
    #      parent's prefill-loop iteration at p == 0 (with the flag
    #      set) runs the Thor K=22 K-row directly — state and the
    #      K_chunk pre-final-norm hiddens land in
    #      ``_K_last_hidden_buf``. Subsequent iterations at
    #      0 <= p < K_chunk return a tiny replay-shaped object whose
    #      ``.replay()`` copies the cached hidden into
    #      ``_last_hidden_buf`` so the parent's
    #      ``_prefill_h_cache[p:p+1].copy_(_last_hidden_buf...)``
    #      keeps working unchanged. Iterations at p >= K_chunk fall
    #      through to the captured single-token graphs.
    #
    # The optimization stays off when the long-ctx route would
    # already chunk the prefill, or when the prompt is shorter than
    # the chunk size, or when no MTP head is loaded (the parent path
    # bails out before us in that case anyway).
    def _thor_first_chunk_eligible(
            self, prompt_len: int, max_new_tokens: int) -> bool:
        # First-chunk only useful when prompt is at least as long as
        # the chunk size — otherwise we'd run a K-row larger than the
        # prompt, which is wasted work.
        K_chunk = self._THOR_FIRST_CHUNK_K
        if K_chunk <= self._THOR_K_ROW_FAST_PATH_MAX or prompt_len < K_chunk:
            return False
        if self._weights.ptrs.get('mtp') is None:
            return False
        if getattr(self, '_long_ctx_mode', False):
            if self._should_use_long_ctx_route(prompt_len, max_new_tokens):
                return False
        return True

    def _ensure_graph_for_pos_nvfp4(self, p: int):
        if (self._thor_first_chunk_active
                and 0 <= p < self._THOR_FIRST_CHUNK_K):
            import torch
            K_chunk = self._THOR_FIRST_CHUNK_K
            hidden = self._cfg["hidden_size"]
            if p == 0:
                # State has been zeroed by the parent's reset_state()
                # right above the prefill loop. Run the K-row once now
                # to populate state for positions [0, K_chunk) and stash
                # the per-row pre-final-norm hidden in
                # ``_K_last_hidden_buf`` for the no-op replays below.
                d = self._rope_dim
                ids = self._thor_first_chunk_input_ids
                cos_S = self._rope_cos_table[:K_chunk].view(1, K_chunk, d)
                sin_S = self._rope_sin_table[:K_chunk].view(1, K_chunk, d)
                # ``hidden_last`` populates _K_last_hidden_buf[:, :K]
                # AND computes lm_head logits at the last row into
                # _logits_buf. The lm_head is required because when
                # K_chunk == prompt_len the parent's prefill loop
                # never replays a real graph and would otherwise read
                # a stale _logits_buf for the first-decoded-token
                # argmax (this collapses MTP AL when K_chunk equals
                # the full prompt length). When K_chunk < prompt_len
                # the per-token loop will overwrite _logits_buf at
                # the last position anyway, so the extra lm_head
                # here is a no-cost no-op.
                with torch.no_grad():
                    self.forward_own_decode_K_nvfp4(
                        ids[:, :K_chunk], cos_S, sin_S, 0, K_chunk,
                        logits_mode="hidden_last")
            src = self._K_last_hidden_buf[:, p:p + 1].view(1, hidden)
            dst = self._last_hidden_buf.view(1, hidden)
            return _ThorFirstChunkReplay(src, dst)
        return super()._ensure_graph_for_pos_nvfp4(p)

    def generate_own_speculative_KN_nvfp4(
            self, input_ids, *, max_new_tokens: int, K: int = 6):
        prompt_len = int(input_ids.shape[1])
        if not self._thor_first_chunk_eligible(prompt_len, max_new_tokens):
            return super().generate_own_speculative_KN_nvfp4(
                input_ids, max_new_tokens=max_new_tokens, K=K)
        if not hasattr(self, '_rope_cos_table'):
            self._build_rope_table()
        self._thor_first_chunk_input_ids = input_ids
        self._thor_first_chunk_active = True
        try:
            return super().generate_own_speculative_KN_nvfp4(
                input_ids, max_new_tokens=max_new_tokens, K=K)
        finally:
            self._thor_first_chunk_active = False
            self._thor_first_chunk_input_ids = None

    def _should_use_long_ctx_route(
            self, prompt_len: int, max_new_tokens: int) -> bool:
        """Thor-specific routing.

        Skips the RTX "128-token chunked exception" that triggers the
        AL collapse on Thor; falls back to the per-token spec path for
        every prompt that still fits the configured BF16 window. The
        upstream policy is otherwise preserved for long prompts that
        would not fit the BF16 cache at all.
        """
        import os
        if not getattr(self, '_long_ctx_mode', False):
            return False
        # Allow the env var to override the threshold (mainly for
        # bisection of the chunked-prefill drift; production use takes
        # the default).
        thor_min = int(os.environ.get(
            'FLASHRT_QWEN36_THOR_LONG_CTX_FORCE_MIN_SEQ',
            str(self._THOR_LONG_CTX_FORCE_MIN_SEQ)))
        prompt_len = int(prompt_len)
        max_pos = prompt_len + int(max_new_tokens)
        bf16_cap = int(getattr(
            self, '_short_ctx_spec_max_seq', thor_min))
        # If the prompt + decode horizon fits the BF16 window AND the
        # prompt is shorter than the Thor threshold, take the
        # per-token path. Otherwise fall back to the chunked path.
        if prompt_len < thor_min and max_pos <= bf16_cap:
            return False
        return True

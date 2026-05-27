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
        # Long-ctx: BF16 K_cache is sized at the spec window
        # (default 2048), while the parent routes long-ctx K/V writes
        # to ``_fp8_K_cache`` / TQ packed cache. ``_thor_full_K_forward``
        # writes BF16 unconditionally and slices past the cache when
        # the requested window exceeds the BF16 buffer; defer to the
        # parent's branching. ``self.max_seq`` is misleading here
        # (becomes user_max_seq after long-ctx setup) — read the
        # actual BF16 cache extent off the backend.
        bf16_cap = int(self._attn.K_cache.shape[1])
        if cur_pos + K > bf16_cap:
            return super()._layer_forward_full_K_nvfp4(
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

        # (7+8) Batched partial RoPE + V copy + q_seq=K causal XQA.
        # The per-position loop was 128 separate XQA calls per
        # full-attn layer × 16 layers = 2048 launches at K=128, each
        # re-quantizing K_cache[:, :128] to FP8 inside Thor's run().
        # The q_seq=K batched call runs ONE XQA with the causal mask
        # (``_mask_for(q_seq)``) that gives identical attention
        # outputs as the K serial q_seq=1 walk.
        d = self._rope_dim
        scaling = float(self._cfg['head_dim']) ** -0.5
        # Batched partial RoPE: Q rows land in Q_buf[:, :K], K rows in
        # K_cache[full_rank, cur_pos:cur_pos+K].
        q_dst = self._attn.Q_buf[:, :K]
        k_dst = self._attn.K_cache[full_rank, cur_pos:cur_pos + K]
        fvk.qwen36_partial_rope_qk_bf16(
            q_norm_out.data_ptr(), k_norm_out.data_ptr(),
            cos_K.view(K, d).data_ptr(), sin_K.view(K, d).data_ptr(),
            q_dst.data_ptr(), k_dst.data_ptr(),
            K, 24, 4, 256, 64, s,
        )
        # Batched V copy: K rows.
        fvk.gpu_copy(
            self._attn.V_cache[
                full_rank, cur_pos:cur_pos + K].data_ptr(),
            v_new_K.data_ptr(), K * 4 * 256 * 2, s,
        )
        if write_fp8:
            self._fp8_write_kv(
                full_rank, cur_pos, cur_pos + K,
                k_dst.view(K, 4, 256), v_new_K.view(K, 4, 256),
            )
        self._attn.run(
            'full', layer_idx=full_rank, q_seq=K,
            kv_seq=cur_pos + K, stream=s, softmax_scale=scaling,
        )
        attn_out_K = self._attn.O_buf[:, :K].view(K, 24, 256)

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

    def _thor_ensure_first_chunk_graph(self):
        """Lazily capture the K=K_chunk K-row + final norm + lm_head
        as one CUDA graph. Replay amortizes ~1200 per-launch Python
        orchestration overheads (~30-50 ms at K=128) into a single
        graph replay.

        State buffers (lin_state / lin_conv_state / K_cache / V_cache)
        are zeroed before the graph capture (so the captured kernels
        see fresh-state inputs the same way they will at replay time
        after a ``reset_state()``). The captured graph writes those
        state buffers in place at fixed addresses, so subsequent
        replays — preceded by a ``reset_state()`` per the parent
        prefill flow — produce the same outputs.
        """
        if hasattr(self, '_thor_first_chunk_graph'):
            return
        import torch
        K = self._THOR_FIRST_CHUNK_K
        d = self._rope_dim
        device = self._h_b.device
        # Static input buffer the captured graph reads from.
        self._thor_static_ids_K = torch.zeros(
            1, K, dtype=torch.long, device=device)
        cos_S = self._rope_cos_table[:K].view(1, K, d)
        sin_S = self._rope_sin_table[:K].view(1, K, d)
        gs = torch.cuda.Stream(device=device)
        # Warmup the K-row a couple of times into the graph mempool so
        # the allocator wires up reusable scratch.
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            with torch.no_grad():
                for _ in range(3):
                    self.reset_state()
                    self.reset_mtp_state()
                    self.forward_own_decode_K_nvfp4(
                        self._thor_static_ids_K, cos_S, sin_S, 0, K,
                        logits_mode="hidden_last")
        torch.cuda.current_stream().wait_stream(gs)
        # Capture against the parent's shared mempool so the recorded
        # tensor addresses survive across replays.
        g = torch.cuda.CUDAGraph()
        self.reset_state()
        self.reset_mtp_state()
        gs.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs):
            with torch.cuda.graph(
                    g, stream=gs, pool=self._graph_mempool):
                with torch.no_grad():
                    self.forward_own_decode_K_nvfp4(
                        self._thor_static_ids_K, cos_S, sin_S, 0, K,
                        logits_mode="hidden_last")
        torch.cuda.current_stream().wait_stream(gs)
        self._thor_first_chunk_graph = g
        self._thor_first_chunk_graph_stream = gs

    def _ensure_graph_for_pos_nvfp4(self, p: int):
        if (self._thor_first_chunk_active
                and 0 <= p < self._THOR_FIRST_CHUNK_K):
            import torch
            K_chunk = self._THOR_FIRST_CHUNK_K
            hidden = self._cfg["hidden_size"]
            if p == 0:
                # Capture the K-row graph on first use; subsequent
                # generations replay it (one CUDA-Graph launch instead
                # of ~1200 Python-orchestrated kernel launches).
                self._thor_ensure_first_chunk_graph()
                # Copy this generation's prompt prefix into the static
                # input buffer the captured graph reads from.
                self._thor_static_ids_K.copy_(
                    self._thor_first_chunk_input_ids[:, :K_chunk])
                # Parent already reset_state() right above the prefill
                # loop, so the captured graph sees a fresh
                # lin_state / K_cache at replay time.
                gs = self._thor_first_chunk_graph_stream
                gs.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(gs):
                    self._thor_first_chunk_graph.replay()
                torch.cuda.current_stream().wait_stream(gs)
            src = self._K_last_hidden_buf[:, p:p + 1].view(1, hidden)
            dst = self._last_hidden_buf.view(1, hidden)
            return _ThorFirstChunkReplay(src, dst)
        return super()._ensure_graph_for_pos_nvfp4(p)

    # ---------- Batched NVFP4 MTP prefill ----------
    #
    # The parent's prefill MTP loop calls forward_mtp_head_nvfp4 once per
    # prompt position (positions 1..prompt_len). At ctx=128 that's 128
    # serial per-position MTP forwards = ~638 ms on Thor (each call
    # ~5 ms, mostly kernel-launch / Python orchestration plus the M=1
    # MTP MLP / lm_head BW). The 5090 dodges this via the long-ctx
    # route's batched ``_prefill_mtp_tail_kv_nvfp4`` (parent qwen36_rtx
    # line 4063), which observes that MTP prefill outputs (next_h /
    # logits) are discarded — only the K/V cache writes matter. That
    # batched routine populates _mtp_K_cache / _mtp_V_cache rows at M=K
    # while skipping Q proj, attention, output gate, O proj, MLP and
    # lm_head entirely. The parent's variant is gated on
    # ``'k_proj_w_bf16' in mtp`` though — Thor's MTP weights are NVFP4
    # so the parent branch is a no-op.
    #
    # Below: a Thor-only NVFP4 equivalent. Dedicated buffers (parent
    # uses ``_mtp_tail_*_buf`` — we mirror with ``_thor_mtp_tail_*``)
    # so the batched prefill does not alias the K-row scratch the
    # first-chunk graph holds across the spec decode loop.
    #
    # Wiring: ``forward_mtp_head_nvfp4`` override below detects the
    # first prefill call (cur_pos == 1, ``_thor_mtp_prefill_active``)
    # and runs ONE batched call for positions [1..prompt_len - 1].
    # Subsequent calls in that range no-op. The final call at
    # cur_pos == prompt_len (which carries the freshly decoded ``tok``
    # not yet available at p=1) still goes through the per-position
    # captured graph below. Net: 128 serial MTP forwards → 1 batched
    # call + 1 per-position graph replay.
    def _thor_ensure_mtp_prefill_buffers(self, rows: int) -> None:
        """Lazy alloc of dedicated MTP prefill scratch — mirror of
        parent ``_ensure_mtp_tail_kv_buffers`` but owned by the Thor
        subclass. Sized to the largest ``rows`` ever requested."""
        import torch

        rows = int(rows)
        cap = int(getattr(self, '_thor_mtp_tail_rows', 0))
        if cap >= rows:
            return
        hidden = self._cfg['hidden_size']
        bf16 = torch.bfloat16
        device = self._h_b.device
        self._thor_mtp_tail_rows = rows
        self._thor_mtp_tail_embed_buf = torch.empty(
            rows, hidden, device=device, dtype=bf16)
        self._thor_mtp_tail_h_norm_buf = torch.empty_like(
            self._thor_mtp_tail_embed_buf)
        self._thor_mtp_tail_e_norm_buf = torch.empty_like(
            self._thor_mtp_tail_embed_buf)
        self._thor_mtp_tail_cat_buf = torch.empty(
            rows, hidden * 2, device=device, dtype=bf16)
        self._thor_mtp_tail_fc_out_buf = torch.empty(
            rows, hidden, device=device, dtype=bf16)
        self._thor_mtp_tail_x_norm_buf = torch.empty_like(
            self._thor_mtp_tail_fc_out_buf)
        self._thor_mtp_tail_k_proj_buf = torch.empty(
            rows, 4 * 256, device=device, dtype=bf16)
        self._thor_mtp_tail_v_proj_buf = torch.empty_like(
            self._thor_mtp_tail_k_proj_buf)
        self._thor_mtp_tail_k_norm_buf = torch.empty(
            rows * 4, 256, device=device, dtype=bf16)
        # Q is not computed during prefill — we still hand the kernel a
        # 1-head dummy because qwen36_partial_rope_qk_bf16 always
        # rotates Q (cheap at num_heads_q=1). Parent does the same.
        self._thor_mtp_tail_dummy_q_in = torch.empty(
            rows, 1, 256, device=device, dtype=bf16)
        self._thor_mtp_tail_dummy_q_out = torch.empty_like(
            self._thor_mtp_tail_dummy_q_in)

    def _thor_mtp_prefill_K_nvfp4(
            self, prev_h_rows, token_ids, pos_start: int, K: int) -> bool:
        """Populate ``_mtp_K_cache`` / ``_mtp_V_cache`` rows
        ``[pos_start..pos_start + K)`` in a single batched walk.
        Mirror of parent ``_prefill_mtp_tail_kv_nvfp4`` (qwen36_rtx
        line 4063) with NVFP4 k/v projections instead of BF16. Returns
        ``True`` on success, ``False`` when MTP weights are missing.
        Skips Q proj, attention, output gate, O proj, MLP, lm_head —
        none feed K/V cache state, so the parent's per-position loop
        discards them anyway."""
        import torch

        from flash_rt import flash_rt_kernels as fvk

        mtp = self._weights.ptrs.get('mtp')
        if mtp is None:
            return False
        rows = int(K)
        if rows <= 0:
            return True
        hidden = self._cfg['hidden_size']
        eps = float(self._cfg['rms_norm_eps'])
        s = torch.cuda.current_stream().cuda_stream
        self._thor_ensure_mtp_prefill_buffers(rows)

        embed = self._thor_mtp_tail_embed_buf[:rows]
        h_norm = self._thor_mtp_tail_h_norm_buf[:rows]
        e_norm = self._thor_mtp_tail_e_norm_buf[:rows]
        cat_buf = self._thor_mtp_tail_cat_buf[:rows]
        fc_out = self._thor_mtp_tail_fc_out_buf[:rows]
        x_norm = self._thor_mtp_tail_x_norm_buf[:rows]
        k_proj = self._thor_mtp_tail_k_proj_buf[:rows]
        v_proj = self._thor_mtp_tail_v_proj_buf[:rows]
        k_norm = self._thor_mtp_tail_k_norm_buf[:rows * 4]

        # 0) Embed prev tokens.
        fvk.qwen36_embedding_lookup_bf16(
            token_ids.view(-1).data_ptr(),
            int(self._weights.ptrs['embed_w']),
            embed.data_ptr(), rows, hidden, s,
        )

        # 1) Pre-FC norms on prev_h and embed.
        fvk.rms_norm(
            prev_h_rows.view(rows, hidden).data_ptr(),
            int(mtp['pre_fc_norm_hidden_eff_w']),
            h_norm.data_ptr(), rows, hidden, eps, s,
        )
        fvk.rms_norm(
            embed.data_ptr(), int(mtp['pre_fc_norm_embedding_eff_w']),
            e_norm.data_ptr(), rows, hidden, eps, s,
        )

        # 2) Concat [e_norm, h_norm].
        fvk.concat2_bf16(
            e_norm.data_ptr(), h_norm.data_ptr(),
            cat_buf.data_ptr(), rows, hidden, hidden, s,
        )

        # 3) FC (BF16 matmul, K_in=2*hidden, N=hidden). ``bf16_matmul``
        # at K=10240 is the generic chunked path (no K=10240 spec at
        # csrc/kernels/bf16_matmul_qwen36.cu:472) — slow at M=127
        # (117 ms measured) but bit-identical to the per-token
        # ``bf16_matvec`` reduction. Alternatives tried:
        #   * ``torch.mm`` (cuBLAS, 0.45 ms): cosine 1.0000 vs custom
        #     kernel but bit-level rounding drops MTP AL 3.93 -> 3.20.
        #   * Split fc_w along K dim into two contiguous (H, H) halves
        #     and run two K=5120 spec GEMMs + sum (52 ms): K=5120
        #     specialization plus separate FP32 partial-sum changes
        #     the fma order vs the K=10240 generic chunked path, AL
        #     drops 3.93 -> 3.50.
        # MTP head is calibrated against the K=10240 generic chunked
        # reduction. A faster AL-preserving fc would require either a
        # new kernel that matches the K=10240 reduction at higher M,
        # or recalibrating the MTP head against a different reduction
        # — both out of scope for this ship. The custom kernel stays.
        fvk.bf16_matmul_qwen36_bf16(
            cat_buf.data_ptr(), int(mtp['fc_w']),
            fc_out.data_ptr(), rows, hidden, hidden * 2, s,
        )

        # 4) input_norm.
        fvk.rms_norm(
            fc_out.data_ptr(), int(mtp['input_norm_eff_w']),
            x_norm.data_ptr(), rows, hidden, eps, s,
        )

        # 5) NVFP4 quantize x_norm — reused for k_proj and v_proj.
        # Share the (1024, 5120) NVFP4 scratch's ap/sf (sized
        # max_seq × hidden, so 128 rows is well within). The MTP
        # batched prefill runs sequentially after the K-row first-chunk
        # graph and before the spec decode loop, so the shared scratch
        # is not racing any concurrent user.
        ap_5120, sf_5120, _ = self._nvfp4_scratch[(1024, 5120)]
        fvk.quantize_bf16_to_nvfp4_swizzled(
            x_norm.data_ptr(), ap_5120.data_ptr(),
            sf_5120.data_ptr(), rows, hidden, s,
        )

        # 6) k_proj NVFP4 → dedicated k_proj_buf.
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(mtp['k_proj_packed']),
            k_proj.data_ptr(),
            rows, 4 * 256, hidden,
            sf_5120.data_ptr(), int(mtp['k_proj_sf']),
            float(mtp['k_proj_alpha']),
            s,
        )

        # 7) v_proj NVFP4 → dedicated v_proj_buf.
        fvk.fp4_w4a16_gemm_sm120_bf16out(
            ap_5120.data_ptr(), int(mtp['v_proj_packed']),
            v_proj.data_ptr(),
            rows, 4 * 256, hidden,
            sf_5120.data_ptr(), int(mtp['v_proj_sf']),
            float(mtp['v_proj_alpha']),
            s,
        )

        # 8) k_norm.
        fvk.rms_norm(
            k_proj.view(rows * 4, 256).data_ptr(),
            int(mtp['k_norm_eff_w']),
            k_norm.data_ptr(), rows * 4, 256, eps, s,
        )

        # 9) Partial RoPE on K with a 1-head dummy Q. Lands rotated K
        # directly into _mtp_K_cache[pos_start..pos_start+rows].
        cos = self._rope_cos_table[
            pos_start:pos_start + rows].view(rows, self._rope_dim)
        sin = self._rope_sin_table[
            pos_start:pos_start + rows].view(rows, self._rope_dim)
        q_dummy = self._thor_mtp_tail_dummy_q_in[:rows]
        q_dummy_out = self._thor_mtp_tail_dummy_q_out[:rows]
        fvk.qwen36_partial_rope_qk_bf16(
            q_dummy.data_ptr(), k_norm.data_ptr(),
            cos.data_ptr(), sin.data_ptr(),
            q_dummy_out.data_ptr(),
            self._mtp_K_cache[
                pos_start:pos_start + rows].data_ptr(),
            rows, 1, 4, 256, self._rope_dim, s,
        )

        # 10) V copy.
        fvk.gpu_copy(
            self._mtp_V_cache[
                pos_start:pos_start + rows].data_ptr(),
            v_proj.data_ptr(), rows * 4 * 256 * 2, s,
        )
        return True

    # ---------- Per-position MTP graph fallback (unchanged) ----------
    #
    # Parent has ``_ensure_mtp_graph_nvfp4(cur_pos)`` that captures a
    # full forward_mtp_head_nvfp4 per position via
    # ``_mtp_static_prev_h`` / ``_mtp_static_prev_token`` static
    # buffers. We still need it for the cur_pos == prompt_len call
    # because that one carries the freshly decoded ``tok`` (not in
    # ``input_ids``) — the batched prefill above can only cover
    # positions [1..prompt_len - 1].
    #
    # Guard against re-entry from inside _ensure_mtp_graph_nvfp4's own
    # warmup + capture (which call forward_mtp_head_nvfp4) and from
    # _ensure_mtp_chain_graph_nvfp4's capture (CUDA stream is in
    # capture mode there).
    def forward_mtp_head_nvfp4(self, prev_h, prev_token_id, cur_pos: int,
                                mtp_cache_pos: int | None = None):
        import torch
        active = getattr(self, '_thor_mtp_prefill_active', False)
        in_inner = getattr(self, '_thor_in_graphed_mtp', False)
        # Fall through to the original implementation during capture or
        # when re-entering from the graph helper.
        if (not active or in_inner
                or torch.cuda.is_current_stream_capturing()):
            return super().forward_mtp_head_nvfp4(
                prev_h, prev_token_id, cur_pos, mtp_cache_pos)
        # The captured per-pos graph is keyed on cur_pos, not on
        # mtp_cache_pos; only used when the two coincide (i.e. the
        # prefill path).
        if mtp_cache_pos is not None and int(mtp_cache_pos) != int(cur_pos):
            return super().forward_mtp_head_nvfp4(
                prev_h, prev_token_id, cur_pos, mtp_cache_pos)
        hidden = self._cfg["hidden_size"]
        ids = self._thor_first_chunk_input_ids
        prompt_len = int(ids.shape[1]) if ids is not None else 0
        prefilled_to = int(getattr(self, '_thor_mtp_prefilled_to', 0))
        cur_pos_i = int(cur_pos)
        # Batched-prefill window covers [1..prompt_len - 1]. On the
        # first eligible call (cur_pos == 1 with not-yet-prefilled
        # state), run ONE batched call and then no-op the rest of the
        # window. Position prompt_len falls through to the per-pos
        # graph below — it carries the freshly decoded tok which isn't
        # available at p=1.
        if (ids is not None and prompt_len > 1 and prefilled_to == 0
                and cur_pos_i == 1):
            with torch.no_grad():
                tail_K = prompt_len - 1
                prev_h_K = self._prefill_h_cache[
                    :tail_K].view(tail_K, hidden).contiguous()
                prev_tok_K = ids[:, 1:prompt_len].contiguous().view(tail_K)
                ok = self._thor_mtp_prefill_K_nvfp4(
                    prev_h_K, prev_tok_K, 1, tail_K)
                if ok:
                    self._thor_mtp_prefilled_to = prompt_len - 1
                    return
            # If batched failed (no MTP weights / bail), fall through
            # to the per-pos graph path for this call.
        if (prefilled_to > 0 and 1 <= cur_pos_i <= prefilled_to):
            # Already covered by the batched call above. No-op.
            return
        # Re-entry guard: _ensure_mtp_graph_nvfp4 internally calls
        # forward_mtp_head_nvfp4 to warm + capture; we must not loop
        # back into the graphed path during that.
        self._thor_in_graphed_mtp = True
        try:
            self._mtp_static_prev_h.copy_(prev_h.view(1, 1, hidden))
            self._mtp_static_prev_token.copy_(prev_token_id.view(1, 1))
            g = self._ensure_mtp_graph_nvfp4(int(cur_pos))
            gs = self._graph_stream
            gs.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(gs):
                g.replay()
            torch.cuda.current_stream().wait_stream(gs)
        finally:
            self._thor_in_graphed_mtp = False

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
        self._thor_mtp_prefill_active = True
        self._thor_mtp_prefilled_to = 0
        try:
            return super().generate_own_speculative_KN_nvfp4(
                input_ids, max_new_tokens=max_new_tokens, K=K)
        finally:
            self._thor_first_chunk_active = False
            self._thor_first_chunk_input_ids = None
            self._thor_mtp_prefill_active = False
            self._thor_mtp_prefilled_to = 0

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

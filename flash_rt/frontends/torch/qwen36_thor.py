"""FlashRT — Qwen3.6-27B NVFP4 Thor frontend (SM110).

Per-hardware split required by ``docs/adding_new_model.md`` rule 2:
one ``(model, framework, hardware)`` file. The RTX frontend at
:mod:`flash_rt.frontends.torch.qwen36_rtx` is the canonical compute
path on RTX 5090; this module hosts the parallel Thor entry point.

Construction strategy
---------------------
The RTX frontend is monolithic (~10k LOC). The single Thor-incompatible
construction step is the attention-backend ctor inside the parent
``__init__``, which directly imports the vendored FA2 extension
(``flash_rt_fa2`` — not built on Thor). ``_use_thor_attn_backend``
patches the RTX attention-backend symbol to
:class:`ThorFlashAttnBackendQwen36` for the duration of parent init.
Every other load step (NVFP4 weight extraction, MTP head conversion,
tokenizer, buffer allocation, CUDA Graph mempool setup) runs
unchanged on Thor.

Dispatch (mirrors 5090's ``_should_use_long_ctx_route`` exactly)
--------------------------------------------------------------
``prompt_len < 128``                : short-ctx legacy per-pos walk.
``128 ≤ prompt_len < 192``          : long-ctx route (5090's exception).
``prompt_len ≥ LONG_CTX_THRESHOLD``  : long-ctx route.
``max_pos > bf16_cap``              : long-ctx route.

This module owns five Thor-specific overrides on top of the parent:

  * ``_layer_forward_lin_K_nvfp4`` / ``_layer_forward_full_K_nvfp4``:
    route K > 7 to the from-scratch ``_thor_lin_K_forward`` /
    ``_thor_full_K_forward`` (parent's K-row kernel chain at M=K
    diverges from M=1 on SM110 due to fused-kernel reductions;
    ours uses split kernels that match per-token byte-for-byte at
    K=128, and stays cos > 0.99 through K=2048).
  * ``_thor_mtp_prefill_K_nvfp4``: NVFP4 batched MTP K/V tail prefill.
    Mirrors parent's ``_prefill_mtp_tail_kv_nvfp4`` (which requires
    BF16 shadow MTP weights) for the NVFP4-only Thor MTP head.
  * ``_long_tq_effective_k``: caps adaptive K at 5 for
    ``prompt_len ≥ 12288`` (5090 picks 7 there; Thor's M=K rounding
    profile cannot sustain a K=7 chain — measured AL 1.06 → 4.00
    when K is capped at 5).
  * ``_long_mtp_prefill_tail_for_prompt`` / ``_prefill_mtp_tail_kv_nvfp4``:
    NVFP4 MTP variant of parent's bucketed tail-prefill helpers
    (parent gates them on BF16 shadow MTP weights).

Backend init also bumps Thor's BF16 K/V cache + internal FP8 paged
cache to ``user_max_seq`` (parent sizes them at the 2048-row spec
window). This is mandatory: without the bump every chunk at
``cur_pos+K > 2048`` falls back to the per-position FA2 loop and
TTFT regresses 4–5× at long ctx.
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

    Linear-attention chunked backend (Thor default)
    -----------------------------------------------
    The module-level ``setdefault`` for
    ``FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND`` pins the per-step
    recurrent GDN backend on Thor. The RTX default ``wy_lt`` chunk
    backend produces a measurable per-position drift on SM110
    (mean cos 0.999979 → 0.999721 at layer 2; compounds layer-by-layer
    to a cos = 0.43 floor by layer 63) while ``native`` is mathematically
    equivalent (cos > 0.9999) to the per-token recurrent path. See
    ``dev_log_qwen36_thor/step5e_chunked_prefill_drift_root_cause.md``
    for the per-layer hidden-state cosine table.
    """

    # K-row layer dispatch threshold on Thor.
    #
    # At K ≤ 7 (the spec-verify chain length range used by short-ctx
    # spec decode) parent's K-row enters the ``save_steps>0`` branch
    # which uses per-step recurrent kernels and stays per-token-
    # equivalent on SM110 — that path is byte-stable on Thor (AL=3.86
    # in prod at ctx=128 K=6), so we delegate.
    # At K > 7 parent's K-row enters the ``save_steps=0`` chunk
    # branch with fused kernels (residual_add_rms_norm_to_nvfp4,
    # mlp_gate_up_packed, silu_mul_to_nvfp4) whose M=K BF16 rounding
    # diverges from M=1 on SM110. The Thor-native ``_thor_lin_K_forward``
    # and ``_thor_full_K_forward`` below replace that branch with
    # split kernels that match per-token reduction order — verified
    # cos ≥ 0.999999 at K=128, ≥ 0.99 for the bulk of layers at K=2048.
    _THOR_K_ROW_FAST_PATH_MAX: int = 7

    def __init__(self, *args, **kwargs):
        with _use_thor_attn_backend():
            super().__init__(*args, **kwargs)
        self._thor_alloc_K_row_scratch()
        # Bump Thor backend's BF16 K/V cache + internal FP8 paged
        # cache from the parent's long-ctx spec-window size (default
        # 2048 rows) up to ``user_max_seq``. Parent leaves the BF16
        # cache small to save 5090 memory; on Thor with 128 GB unified
        # memory the cost (~3 GB at max_seq=32768) is fine and the
        # benefit is large: the K-row's batched XQA path
        # (``_attn.run('full', q_seq=K)``) reads from ``self.K_cache``
        # — without the bump, long-ctx prompts past cur_pos > 2048
        # would have to fall back to the per-pos FA2 loop in the
        # FP8-paged branch (measured: ~32K per-pos FA2 calls instead
        # of 16 batched XQA calls at ctx=2K K=2048).
        self._attn.ensure_kv_capacity(int(self._user_max_seq))
        # Pre-grow ``_fa2_fp8_K`` paged scratch too — used by FA2
        # adapter path (MTP attention etc.). Same rationale: avoid
        # mid-graph-capture reallocations.
        self._attn.ensure_fa2_paged_capacity(int(self._user_max_seq))

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
        # Rotated K staging buffer used by ``_thor_full_K_forward``'s
        # safety-net FP8-paged branch (cur_pos + K > BF16 K_cache cap).
        # In normal production this branch never fires — Thor frontend
        # __init__ bumps the K_cache to user_max_seq so the BF16 fast
        # path covers every chunk. The staging buffer is the fallback
        # path for callers that construct the frontend at a smaller
        # ``max_seq`` than the actual prompt length.
        self._thor_full_k_stage = torch.empty(
            Kmax, self._attn.NUM_KV_HEADS, self._attn.HEAD_DIM,
            device=device, dtype=bf16)

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
        # ``_thor_full_K_forward`` picks the BF16-cache fast path
        # (rotated K → K_cache, batched XQA via _attn.run) when
        # cur_pos+K fits the BF16 K_cache extent. The frontend ctor
        # bumps the cache to user_max_seq, so this branch fires for
        # every chunk in normal production. The fallback FP8-paged
        # per-position FA2 branch exists only for callers that under-
        # size max_seq at construction.
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

        # (7+8) Attention. Two paths:
        # * BF16-cache (short-ctx): write rotated K to ``_attn.K_cache``
        #   and V to ``_attn.V_cache``, then one batched ``_attn.run``
        #   call (Thor XQA at q_seq=K). Optionally mirror to FP8 paged
        #   cache for downstream verify-time FP8-KV reads.
        # * FP8 paged (long-ctx, cur_pos + K exceeds the BF16 spec
        #   window): rotated K lands in ``_thor_full_k_stage`` only,
        #   ``_fp8_write_kv`` populates parent's persistent FP8 paged
        #   cache for the new [cur_pos..cur_pos+K] rows, then
        #   ``_fp8_stage_for_layer`` dequantizes [0..cur_pos+K] back
        #   into BF16 staging for the attention loop. Per-position FA2
        #   is the only available path on Thor (``_fa2_fwd_causal``
        #   is None — XQA at q_seq=K is unavailable when reading from
        #   the staging buffer because XQA expects FP8 paged input,
        #   not BF16).
        d = self._rope_dim
        scaling = float(self._cfg['head_dim']) ** -0.5
        write_fp8 = bool(getattr(self, "_fp8_kv_verify_active", False))
        bf16_cap = int(self._attn.K_cache.shape[1])
        # BF16-cache fast path applies whenever ``cur_pos + K`` fits
        # in the BF16 K_cache, regardless of ``_fp8_kv_verify_active``.
        # In FP8-KV mode we additionally mirror the rotated K + V to
        # the persistent FP8 paged cache so subsequent chunks past
        # bf16_cap can still read the [0..cur_pos+K] window via the
        # staging buffer. Without this, even ctx=2K (which fits in
        # bf16_cap entirely) would degrade to the per-pos FA2 loop
        # below — measured 32K FA2 calls per prefill at ctx=2K.
        use_bf16_cache = (cur_pos + K) <= bf16_cap

        q_dst = self._attn.Q_buf[:, :K]
        if use_bf16_cache:
            # Fast path: rotated K straight into K_cache.
            # When ``_fp8_kv_verify_active`` is set (long-ctx generate
            # where the first chunk happens to fit in BF16 cap), we
            # still mirror to the persistent FP8 paged cache so the
            # later chunks that exceed bf16_cap see initialised FP8
            # positions when they read [0..cur_pos+K] via the staging
            # buffer.
            k_dst = self._attn.K_cache[full_rank, cur_pos:cur_pos + K]
            fvk.qwen36_partial_rope_qk_bf16(
                q_norm_out.data_ptr(), k_norm_out.data_ptr(),
                cos_K.view(K, d).data_ptr(), sin_K.view(K, d).data_ptr(),
                q_dst.data_ptr(), k_dst.data_ptr(),
                K, 24, 4, 256, 64, s,
            )
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
        else:
            # Long-ctx FP8-paged path: rotated K to Thor staging buf
            # (BF16 K_cache too small at cur_pos + K > bf16_cap).
            k_stage_new = self._thor_full_k_stage[:K]
            fvk.qwen36_partial_rope_qk_bf16(
                q_norm_out.data_ptr(), k_norm_out.data_ptr(),
                cos_K.view(K, d).data_ptr(), sin_K.view(K, d).data_ptr(),
                q_dst.data_ptr(), k_stage_new.data_ptr(),
                K, 24, 4, 256, 64, s,
            )
            # Populate parent's persistent FP8 paged cache for the
            # [cur_pos..cur_pos+K] window. v_new_K is BF16 in the
            # NVFP4 GEMM output buffer.
            self._fp8_write_kv(
                full_rank, cur_pos, cur_pos + K,
                k_stage_new.view(K, 4, 256),
                v_new_K.view(K, 4, 256),
            )
            # Dequant [0..cur_pos+K] from FP8 paged cache into BF16
            # staging. ``_fp8_stage_for_layer`` caches the prior valid
            # range so only the new K rows actually dequantize.
            k_stage, v_stage = self._fp8_stage_for_layer(
                full_rank, cur_pos + K)
            # Per-position FA2 loop reading from BF16 staging
            # (Thor's _fa2_fwd_causal is None, so q_seq=K causal
            # FA2 is unavailable).
            for kk in range(K):
                q_view = self._attn.Q_buf[:, kk:kk + 1]
                kv_seq_k = cur_pos + kk + 1
                k_view = k_stage[:kv_seq_k].view(1, kv_seq_k, 4, 256)
                v_view = v_stage[:kv_seq_k].view(1, kv_seq_k, 4, 256)
                o_view = self._attn.O_buf[:, kk:kk + 1]
                self._attn._fa2_fwd(
                    Q=q_view.data_ptr(), K=k_view.data_ptr(),
                    V=v_view.data_ptr(), O=o_view.data_ptr(),
                    softmax_lse=self._attn.lse_buf.data_ptr(),
                    softmax_lse_accum=self._attn.lse_accum.data_ptr(),
                    o_accum=self._attn.o_accum.data_ptr(),
                    batch=1, seqlen_q=1, seqlen_k=kv_seq_k,
                    num_heads_q=24, num_heads_kv=4, head_dim=256,
                    q_strides=(q_view.stride(0), q_view.stride(1),
                               q_view.stride(2)),
                    k_strides=(k_view.stride(0), k_view.stride(1),
                               k_view.stride(2)),
                    v_strides=(v_view.stride(0), v_view.stride(1),
                               v_view.stride(2)),
                    o_strides=(o_view.stride(0), o_view.stride(1),
                               o_view.stride(2)),
                    softmax_scale=scaling,
                    num_sms=self._attn._num_sms,
                    stream=s,
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

    # ---------- Batched NVFP4 MTP prefill ----------
    #
    # Mirror of parent's ``_prefill_mtp_tail_kv_nvfp4`` for NVFP4 MTP
    # weights. Parent gates that function on ``'k_proj_w_bf16' in mtp``
    # (5090 keeps BF16 shadow weights alongside the NVFP4 packed
    # weights). On Thor we only load NVFP4 MTP weights so parent's
    # function always returns False; this NVFP4 batched variant fills
    # the same role.
    #
    # The Thor override of ``_prefill_mtp_tail_kv_nvfp4`` (below) calls
    # this; the override of ``_long_mtp_prefill_tail_for_prompt``
    # (below) drops parent's BF16-shadow gate so the bucket table
    # applies to Thor too.
    #
    # Dedicated ``_thor_mtp_tail_*`` buffers so the batched prefill
    # never aliases the K-row scratch buffers.
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
            self, prev_h_rows, token_ids, pos_start: int, K: int,
            cache_base_pos: int | None = None) -> bool:
        """Populate ``_mtp_K_cache`` / ``_mtp_V_cache`` rows
        ``[cache_base_pos..cache_base_pos + K)`` (defaults to
        ``pos_start`` for the absolute-RoPE path) in a single batched
        walk. Mirror of parent ``_prefill_mtp_tail_kv_nvfp4`` (qwen36_rtx
        line 4063) with NVFP4 k/v projections instead of BF16.
        ``pos_start`` is the absolute RoPE position; ``cache_base_pos``
        is the MTP K/V cache row offset (long-ctx TQ uses a compact
        MTP cache where this differs from ``pos_start``). Returns
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

        # 9) Partial RoPE on K with a 1-head dummy Q. Rotated K lands
        # into _mtp_K_cache[cache_base..cache_base+rows]. RoPE position
        # is pos_start (absolute prompt position).
        cache_base = (int(cache_base_pos)
                      if cache_base_pos is not None else int(pos_start))
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
                cache_base:cache_base + rows].data_ptr(),
            rows, 1, 4, 256, self._rope_dim, s,
        )

        # 10) V copy.
        fvk.gpu_copy(
            self._mtp_V_cache[
                cache_base:cache_base + rows].data_ptr(),
            v_proj.data_ptr(), rows * 4 * 256 * 2, s,
        )
        return True

    # ---------- Adaptive K override ----------
    #
    # Parent's ``_long_tq_effective_k`` picks K=7 for prompt buckets
    # [12288, 24576) and [49152, 160000) — matching 5090 measurements
    # where K=7 gives the best decode tok/s. On Thor the M=K NVFP4
    # GEMM rounding accumulates per-layer noise that's tighter than
    # 5090's; the longer K=7 draft chain ends up with rejections that
    # collapse AL (measured ctx=12K K=7 AL=1.13, K=5 AL=3.12; ctx=16K
    # K=7 AL=1.06, K=5 AL=4.00). Override to cap at K=5 in those
    # buckets where parent picks 7 — Thor measured numbers preserve
    # AL ≥ 3.0 across the full long-ctx range.
    def _long_tq_effective_k(self, prompt_len: int, K: int) -> int:
        target_k = super()._long_tq_effective_k(prompt_len, K)
        # Parent's bucket overrides via FLASHRT_QWEN36_TQ_SPEC_K env
        # already short-circuit before reaching this cap; keep the
        # cap purely a Thor-specific bucket adjustment.
        import os
        if os.environ.get('FLASHRT_QWEN36_TQ_SPEC_K', ''):
            return target_k
        if target_k > 5 and int(prompt_len) >= 12288:
            return 5
        return target_k

    # ---------- Long-ctx MTP prefill integration ----------
    #
    # Parent's _long_mtp_prefill_tail_for_prompt returns 0 when MTP
    # weights lack a ``_w_bf16`` shadow — the long-ctx generate then
    # writes only one MTP K/V row (cur_pos = prompt_len) and leaves
    # positions [1..prompt_len-1] uninitialised. The spec loop then
    # attends to zeros for the first generated token's MTP cache
    # window and AL collapses (ctx=128 K=6: 3.93 -> 1.75 measured).
    #
    # Thor's NVFP4 MTP head has no BF16 shadow but the math is the
    # same. Override so the bucket logic applies regardless of the
    # weight format, then route ``_prefill_mtp_tail_kv_nvfp4`` to the
    # NVFP4 batched helper above.
    def _long_mtp_prefill_tail_for_prompt(self, prompt_len: int) -> int:
        import os
        raw = os.environ.get(
            'FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL', 'auto') or 'auto'
        if raw.lower() != 'auto':
            return max(0, int(raw))
        mtp = self._weights.ptrs.get('mtp') if self._weights else None
        if not isinstance(mtp, dict):
            return 0
        # Mirror parent's bucket table (qwen36_rtx.py:7867) but drop
        # the BF16-shadow gate. The NVFP4 batched MTP function provides
        # the equivalent K/V cache fill.
        prompt_len = int(prompt_len)
        if prompt_len >= 128 and prompt_len < 512:
            return min(128, prompt_len)
        if prompt_len < 512:
            return 0
        if prompt_len < 768:
            return 512
        if prompt_len < 3072:
            return 2048
        if prompt_len < 6144:
            return 512
        return 2048

    def _prefill_mtp_tail_kv_nvfp4(
            self, prev_h_rows, token_ids, pos_start: int,
            cache_base_pos: int) -> bool:
        """Thor override of parent's MTP K/V tail prefill.

        Parent's variant requires BF16 MTP projection weights and
        returns False when only NVFP4 weights are loaded. Route to
        our NVFP4 batched helper instead so long-ctx generate seeds
        the MTP cache properly and AL is preserved at the bucket
        sizes parent assumes."""
        mtp = self._weights.ptrs.get('mtp')
        if mtp is None:
            return False
        # If BF16 shadow weights are present, defer to parent (matches
        # the original behaviour byte-for-byte on 5090-style ckpts).
        if 'k_proj_w_bf16' in mtp:
            return super()._prefill_mtp_tail_kv_nvfp4(
                prev_h_rows, token_ids, pos_start, cache_base_pos)
        rows = int(token_ids.numel())
        if rows <= 0:
            return True
        return self._thor_mtp_prefill_K_nvfp4(
            prev_h_rows, token_ids, pos_start, rows,
            cache_base_pos=cache_base_pos)


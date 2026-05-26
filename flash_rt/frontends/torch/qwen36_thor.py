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

    def __init__(self, *args, **kwargs):
        with _use_thor_attn_backend():
            super().__init__(*args, **kwargs)

    # ---------- K-row layer dispatch (Thor-only) ----------
    #
    # Linear-attention K-row forward at K above the fast-path threshold.
    # Each iteration calls the parent's single-token linear-attention
    # forward, which mutates ``_lin_state[lin_rank]`` and
    # ``_lin_conv_state[lin_rank]`` in place via the in-place
    # ``causal_conv1d_qwen36_update_bf16`` and
    # ``gated_deltanet_recurrent_qwen36_bf16`` kernels. After each
    # iteration we also snapshot lin_state / lin_conv_state into the
    # per-step buffers so spec-verify partial-accept recovery still
    # finds matching state at slot N (only meaningful when K ≤
    # ``_K_save_max``, which the prefill chunks here exceed — the
    # snapshot block is a no-op in that regime).
    def _layer_forward_lin_K_nvfp4(self, L, h_in_K, K):
        if K <= self._THOR_K_ROW_FAST_PATH_MAX:
            return super()._layer_forward_lin_K_nvfp4(L, h_in_K, K)
        return self._thor_lin_K_dispatch(L, h_in_K, K)

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

    # Full-attention K-row forward at K above the fast-path threshold.
    # Each iteration calls the parent's single-token full-attn forward
    # at ``cur_pos + r``, which writes its K/V row into the BF16 K/V
    # cache and runs attention via the Thor backend's ``run('full',
    # q_seq=1, kv_seq=cur_pos+r+1)`` entry. Bit-exact to a per-token
    # walk that processes the same K positions sequentially.
    def _layer_forward_full_K_nvfp4(
            self, L, h_in_K, cos_K, sin_K, cur_pos, K):
        if K <= self._THOR_K_ROW_FAST_PATH_MAX:
            return super()._layer_forward_full_K_nvfp4(
                L, h_in_K, cos_K, sin_K, cur_pos, K)
        return self._thor_full_K_dispatch(
            L, h_in_K, cos_K, sin_K, cur_pos, K)

    def _thor_full_K_dispatch(self, L, h_in_K, cos_K, sin_K, cur_pos, K):
        import torch
        hidden = self._cfg["hidden_size"]
        d = self._rope_dim
        cos_3d = cos_K.view(1, K, d)
        sin_3d = sin_K.view(1, K, d)
        h_out_K = (self._K_layer_out_a if (L % 2 == 0)
                   else self._K_layer_out_b)[:, :K]
        # When the parent's chunked long-ctx prefill set
        # ``_fp8_kv_verify_active=True`` the K-row layer normally writes
        # FP8 K/V into ``_fp8_K_cache`` / ``_fp8_V_cache`` via
        # ``_fp8_write_kv`` so the decode-time spec-verify XQA reads
        # FP8 values. The per-token forward writes BF16 K/V into
        # ``_attn.K_cache`` instead, so we mirror the FP8 cache update
        # ourselves after each per-position call to keep the long-ctx
        # decode path consistent.
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

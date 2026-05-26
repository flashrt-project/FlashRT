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

from contextlib import contextmanager

from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx


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
    """

    # Default cap below which the Thor frontend forces the short-ctx
    # (per-token spec) route. Above this the long-ctx chunked path
    # engages — that's still the only viable option for prompts past
    # the BF16 spec window.
    _THOR_LONG_CTX_FORCE_MIN_SEQ: int = 8192

    def __init__(self, *args, **kwargs):
        with _use_thor_attn_backend():
            super().__init__(*args, **kwargs)

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

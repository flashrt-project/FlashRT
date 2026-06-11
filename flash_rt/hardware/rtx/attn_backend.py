"""FlashRT — RTX attention backend protocol + implementation.

The rtx Pi0 / Pi0.5 pipeline is framework-agnostic except for the
scratch tensor allocator. This module provides the attention backend
injection point: the pipeline calls into an :class:`AttnBackend`
instance for Q/K/V→O, and the backend decides how to run it.

Design:
  - The backend **owns** all Q/K/V/O memory. Today that is torch
    tensors (``torch.empty(...)``) for both torch and jax frontends
    — the torch dependency is for allocation only, the attention
    kernel itself doesn't see torch. A future pass will swap the
    allocator for :class:`flash_rt.core.cuda_buffer.CudaBuffer` to
    remove the torch dep entirely.
  - The pipeline asks the backend for raw device pointers via
    :meth:`AttnBackend.get_ptrs` and writes Q/K/V via ``fvk`` kernels
    directly into those pointers (shared memory, no copy).
  - When attention needs to run, the pipeline calls
    :meth:`AttnBackend.vision_attn` / ``encoder_attn`` / ``decoder_attn``
    with layer + shape info; the backend uses its own tensor views.

Ships :class:`RtxFlashAttnBackend` which dispatches to the vendored
Flash-Attention 2 in :mod:`flash_rt.flash_rt_fa2` (fp16 and bf16).
The legacy name ``TorchFlashAttnBackend`` is kept as a deprecated
alias for external plugins; see the class docstring.
"""

from __future__ import annotations

import os
from typing import Protocol


def _make_flash_attn_proxy(need_legacy: bool):
    """Return a callable that resolves to ``flash_attn.flash_attn_func``.

    The upstream ``flash-attn`` pip wheel is *only* required when either
    ``FVK_RTX_FA2=0`` is set (legacy backend) or ``FVK_RTX_FA2_SITES``
    excludes one of the three FA2 call sites. When ``need_legacy`` is
    ``False`` we still return a proxy so attribute capture in
    ``__init__`` succeeds, but no import is attempted until the proxy
    is actually called — which it never is on the default fast path.

    On environments without a prebuilt flash-attn wheel (Modal cloud,
    older CUDA images) this lets ``import flash_rt`` and the default
    RTX inference path succeed without paying the 30–60 min sdist
    compile or hitting an ImportError at module load.
    """
    if need_legacy:
        try:
            from flash_attn import flash_attn_func
        except ImportError as e:
            raise ImportError(
                "FVK_RTX_FA2=0 (or FVK_RTX_FA2_SITES excludes a site) "
                "selected the legacy upstream flash-attn path, but the "
                "`flash-attn` pip package is not installed. Either:\n"
                "  - unset FVK_RTX_FA2 / FVK_RTX_FA2_SITES to use the "
                "vendored flash_rt_fa2 (default), or\n"
                "  - install flash-attn (prebuilt wheels at "
                "https://github.com/Dao-AILab/flash-attention/releases)."
            ) from e
        return flash_attn_func

    def _missing_legacy(*_args, **_kw):
        raise RuntimeError(
            "Default RTX FA2 path was reconfigured at runtime to require "
            "flash_attn.flash_attn_func, but the proxy was constructed "
            "with need_legacy=False. This indicates the FVK_RTX_FA2 / "
            "FVK_RTX_FA2_SITES env vars changed after backend init."
        )
    return _missing_legacy


class AttnBackend(Protocol):
    """Protocol for Pi0.5 attention backends.

    The backend owns Q/K/V input memory for all three attention blocks
    (vision, encoder, decoder) and exposes raw device pointers via
    :meth:`get_ptrs` so the pipeline can write into them from fvk kernels.

    Output is **not** pre-allocated by the backend for the legacy
    pip-flash-attn path (``FVK_RTX_FA2=0``). In that mode each
    attention call returns the raw device pointer of whatever tensor
    ``flash_attn_func`` allocated internally; torch's caching
    allocator pins that pointer for the life of the captured graph,
    so it's safe to pass straight into the next GEMM.

    On the default in-SO FA2 path (``FVK_RTX_FA2=1``), output IS
    pre-allocated by the backend at ``__init__`` time; the same
    pointer is returned on every call. The vendored kernel writes
    into that pointer directly and the pipeline reads it the same
    way. This is strictly more stable across CUDA Graph re-capture
    because we don't rely on torch's allocator heuristics.
    """

    def get_ptrs(self) -> dict:
        """Return raw device pointer ints for every attention INPUT buffer.

        Expected keys (all int):
            vis_Q, vis_K, vis_V    — (num_views, 256, 16, 72) bf16
            enc_Q                  — (enc_seq, 8, 256) bf16
            enc_K, enc_V           — (18, enc_seq+chunk, 1, 256) bf16
            dec_Q                  — (chunk, 8, 256) bf16

        The K/V cache is shared across encoder + decoder layers: encoder
        writes K/V into ``enc_K[i, :enc_seq]`` / ``enc_V[i, :enc_seq]`` for
        layer ``i``; the decoder then writes ``enc_K[i, enc_seq:enc_seq+chunk]``
        for the chunk tokens before cross-attention.

        Layer stride (for computing per-layer offsets) is returned as
        ``enc_k_layer_stride_bytes`` / ``enc_v_layer_stride_bytes``.
        """
        ...

    def vision_attn(self, stream: int = 0) -> int:
        """Run per-view vision attention: Q/K/V (nv,256,16,72) → O.

        Returns the raw device pointer of the attention output, shape
        ``(num_views * 256, num_heads * head_dim) = (nv*256, 1152)`` bf16,
        row-major.
        """
        ...

    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        """Run GQA encoder attention for one layer.

        Reads from ``enc_Q[:seq]``, ``enc_K[layer_idx, :seq]``,
        ``enc_V[layer_idx, :seq]``. Returns the raw device pointer of the
        attention output, shape ``(seq, 8*256) = (seq, 2048)`` bf16.
        """
        ...

    def decoder_attn(self, layer_idx: int, enc_seq: int, dec_seq: int,
                     stream: int = 0) -> int:
        """Run cross-attention for one decoder layer.

        Q comes from ``dec_Q[:dec_seq]``; K/V come from
        ``enc_K[layer_idx, :enc_seq+dec_seq]`` (shared encoder+decoder cache).
        Returns the raw device pointer of the attention output, shape
        ``(dec_seq, 8*256) = (dec_seq, 2048)`` bf16.
        """
        ...


class RtxFlashAttnBackend:
    """Pi0 / Pi0.5 attention backend for the RTX family (SM80/86/89/120).

    Framework-agnostic at the attention-call layer: the actual kernel is
    the vendored Flash-Attention 2 in ``flash_rt.flash_rt_fa2`` (fp16
    and bf16 entries, picked per-dtype at ``__init__`` time). Pi0 on
    5090 / 4090, Pi0.5 on either, and the GROOT vision path all go
    through the same ``run(site, layer_idx, ...)`` call regardless of
    whether the frontend is torch or jax.

    Backend still uses ``torch.empty(...)`` for the Q/K/V/O/LSE scratch
    tensors because the pipeline reads their ``.data_ptr()`` into fvk
    kernel calls. That's a torch dependency for allocation only — the
    attention kernel itself does not see torch. A future pass will
    swap the allocator for :class:`flash_rt.core.cuda_buffer.CudaBuffer`
    to remove the torch dep entirely (not urgent — torch is already a
    RTX-path transitive dep via the frontend-level image preprocessor).

    Naming history
    --------------
    Previously called ``TorchFlashAttnBackend`` — the ``Torch`` prefix
    reflected the earlier implementation that delegated to
    ``flash_attn.flash_attn_func`` (pip wheel, torch-typed). After the
    switch to the vendored in-SO FA2 (see
    ``csrc/attention/flash_attn_2_src/``) the attention call is
    framework-neutral; the rename drops the misleading ``Torch``
    prefix. A deprecated module-level alias
    ``TorchFlashAttnBackend = RtxFlashAttnBackend`` is kept for
    external plugins pinned to the old name.

    FA2 / legacy dispatch
    ---------------------
    Controlled by env var ``FVK_RTX_FA2`` (default ``"1"``):
      * ``"1"`` — attention call goes to ``flash_rt_fa2.fwd_{fp16,bf16}``.
      * ``"0"`` — attention call goes to pip ``flash_attn.flash_attn_func``
        (only as a safety-net fallback; the pip wheel is not a runtime
        dependency any more, you'd need to install it separately).
      * ``FVK_RTX_FA2_SITES="siglip,encoder,decoder"`` — per-site fa2
        toggles for bisecting integration bugs.

    **Protocol compatibility**: this class implements the
    :class:`flash_rt.hardware.backend.AttentionBackend` protocol via
    :meth:`get_slot_ptrs` and :meth:`run`. The legacy methods
    (:meth:`vision_attn` / :meth:`encoder_attn` / :meth:`decoder_attn`
    / :meth:`get_ptrs`) remain live for the current pipelines in
    ``flash_rt.models.pi05.pipeline_rtx`` and will be retired once
    those pipelines migrate to the protocol methods. Both surfaces
    wrap the same underlying torch tensors and ``flash_attn_func``
    calls — adding the new methods did not change any runtime
    behavior.

    Site mapping
    ------------
    The new protocol sees this backend as having three sites:

      * ``"siglip"`` — 27 layers, per-view self-attention, GQA disabled
        (num_q=num_kv=16). Shape: (num_views, 256, 16, 72).
      * ``"encoder"`` — 18 PaliGemma encoder layers, GQA 8Q/1KV,
        head_dim=256, self-attention.
      * ``"decoder"`` — 18 decoder layers, GQA 8Q/1KV, head_dim=256,
        **cross-attention** against the shared encoder KV cache. The
        pipeline writes the chunk's fresh K/V tokens into
        ``enc_K[l, enc_seq:enc_seq+chunk]`` before calling
        ``run("decoder", l, q_seq=chunk, kv_seq=enc_seq+chunk)``.

    Layer indexing for ``"decoder"`` site is still 0..num_encoder_layers-1;
    K/V rows are the shared encoder cache, which is why the decoder at
    layer ``l`` reads ``enc_K[l, :]`` and not some separate decoder
    cache. This matches the production pipeline layout.
    """

    def __init__(self, num_views: int, encoder_seq_max: int, chunk_size: int,
                 num_encoder_layers: int = 18, dtype=None):
        import torch
        import os
        self._torch = torch
        # ``dtype`` selects the 16-bit tensor type used for Q/K/V/O
        # buffers. Defaults to bfloat16 (pi05/groot). Pi0 on RTX needs
        # float16 to match the pi0_thor FP16 math path for FP8 stability.
        bf16 = dtype if dtype is not None else torch.bfloat16
        d = "cuda"

        # Select attention implementation:
        #   FVK_RTX_FA2 env var: "1" (default) = use vendored FA2 from
        #   flash_rt_fa2.so (drops pip flash-attn wheel dep), "0" =
        #   keep legacy pip flash_attn_func path.
        # The vendored module ships both fp16 and bf16 instantiations
        # (hdim 96/128/256 × regular + splitkv), so every RTX frontend
        # can go through fvk FA2 regardless of dtype.
        fa2_env = os.environ.get("FVK_RTX_FA2", "1") == "1"
        self._use_fvk_fa2 = fa2_env
        self._is_fp16 = (bf16 == torch.float16)
        # Per-site debug toggles for bisecting integration regressions —
        # FVK_RTX_FA2_SITES="siglip,encoder,decoder" (default all) picks
        # which sites route through fvk FA2 when _use_fvk_fa2 is on.
        _sites_env = os.environ.get("FVK_RTX_FA2_SITES", "siglip,encoder,decoder")
        _enabled = set(s.strip() for s in _sites_env.split(",") if s.strip())
        self._fa2_sites = {
            "siglip":  self._use_fvk_fa2 and "siglip" in _enabled,
            "encoder": self._use_fvk_fa2 and "encoder" in _enabled,
            "decoder": self._use_fvk_fa2 and "decoder" in _enabled,
        }

        # Vision attention INPUTS (per-view batched, no cache)
        self.vis_Q = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=d)
        self.vis_K = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=d)
        self.vis_V = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=d)

        # Encoder Q (reused across layers — no per-layer cache on query side)
        # Encoder K/V shared layer cache (also used by decoder cross-attn)
        total_kv = encoder_seq_max + chunk_size
        self.enc_Q = torch.empty(encoder_seq_max, 8, 256, dtype=bf16, device=d)
        self.enc_K = torch.empty(num_encoder_layers, total_kv, 1, 256,
                                 dtype=bf16, device=d)
        self.enc_V = torch.empty(num_encoder_layers, total_kv, 1, 256,
                                 dtype=bf16, device=d)

        # Decoder Q
        self.dec_Q = torch.empty(chunk_size, 8, 256, dtype=bf16, device=d)

        # Pre-allocated decoder O buffer for state-masked cross-attention
        # (Pi0: row 0 is the state query with a shorter KV window). Only
        # populated and used when ``run("decoder", ..., state_nk=<int>)``
        # is called; Pi0.5 and other models leave this slot unused and
        # pay zero overhead.
        self.dec_O_masked = torch.empty(chunk_size, 8, 256, dtype=bf16, device=d)

        # ── Fixed-shape (seqused/devpos) state-prompt support ──
        # One captured graph at the MAX prefix length serves any prompt length.
        # The valid prefix length is read from device buffers at replay, so FA2
        # (seqused_k) masks the padded keys and the decoder appends its action
        # K/V right after the valid prefix (qkv_split_rope_devpos). Updated per
        # set_prompt via :meth:`set_fixed_valid_len` — runtime inputs, never a
        # recapture. batch=1, so each is a single int32.
        self._fixed_shape = False
        self.enc_seqused = torch.zeros(1, dtype=torch.int32, device=d)  # vis_enc+plen
        self.dec_seqused = torch.zeros(1, dtype=torch.int32, device=d)  # +chunk
        self.dec_devpos = torch.zeros(1, dtype=torch.int32, device=d)   # = enc_seqused

        # Cached shape metadata
        self._num_views = num_views
        self._encoder_seq_max = encoder_seq_max
        self._chunk_size = chunk_size
        self._num_encoder_layers = num_encoder_layers
        # enc_K/V layer stride in bytes (bf16 = 2 bytes)
        self._enc_kv_layer_stride_bytes = (
            total_kv * 1 * 256 * self.enc_K.element_size())

        # Output tensor refs — populated during the first warmup run so the
        # torch caching allocator assigns stable pointers that survive graph
        # capture. We hold references here so the allocator doesn't reclaim
        # and reassign them between warmup and replay.
        self._vis_out_ref = None
        self._enc_out_ref = None
        self._dec_out_ref = None

        # Lazy-import either fvk FA2 (in-SO vendored) or pip flash_attn
        # depending on the dispatch chosen above. In fvk FA2 mode we
        # also pre-allocate hdim96-padded scratch buffers for SigLIP
        # (the vendored kernel has hdim96 but not hdim72 — the native
        # SigLIP head_dim — so we pad Q/K/V to 96 by zeroing the last
        # 24 columns and slice the first 72 of the output back).
        if self._use_fvk_fa2:
            try:
                from flash_rt import flash_rt_fa2 as _fa2
            except ImportError as e:
                raise RuntimeError(
                    "FVK_RTX_FA2=1 but flash_rt_fa2 module is not built. "
                    "Rebuild with ENABLE_FA2 (SM80/86/89/120). Set "
                    f"FVK_RTX_FA2=0 to use pip flash_attn. Import error: {e}")
            self._fa2 = _fa2
            # Pick fp16 or bf16 entry based on the backend's dtype;
            # both live in the same flash_rt_fa2 module.
            self._fa2_fwd = _fa2.fwd_fp16 if self._is_fp16 else _fa2.fwd_bf16
            # SM count used by the splitkv heuristic — matches upstream's
            # get_num_sm(current_device) behaviour in flash_api.cpp.
            self._num_sms = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count

            # Padded hdim=96 SigLIP buffers. Last 24 cols kept zero
            # after construction; we only write cols [:72] per call.
            self._vis_Q96 = torch.zeros(num_views, 256, 16, 96,
                                        dtype=bf16, device=d)
            self._vis_K96 = torch.zeros(num_views, 256, 16, 96,
                                        dtype=bf16, device=d)
            self._vis_V96 = torch.zeros(num_views, 256, 16, 96,
                                        dtype=bf16, device=d)
            self._vis_O96 = torch.empty(num_views, 256, 16, 96,
                                        dtype=bf16, device=d)
            self._vis_O72 = torch.empty(num_views, 256, 16, 72,
                                        dtype=bf16, device=d)
            self._vis_lse = torch.empty(num_views, 16, 256,
                                        dtype=torch.float32, device=d)

            # Output + LSE for encoder (max seq), decoder cross-attn,
            # and state-masked row-0 / rows-1+. All (B=1, S, H, D) fp16.
            self._enc_O = torch.empty(1, encoder_seq_max, 8, 256,
                                      dtype=bf16, device=d)
            # ceil(S/128)*128 — worst-case seq_q_rounded buffer size
            enc_sq_r = ((encoder_seq_max + 127) // 128) * 128
            self._enc_lse = torch.empty(1, 8, enc_sq_r,
                                        dtype=torch.float32, device=d)

            self._dec_O = torch.empty(1, chunk_size, 8, 256,
                                      dtype=bf16, device=d)
            dec_sq_r = ((chunk_size + 127) // 128) * 128
            self._dec_lse = torch.empty(1, 8, dec_sq_r,
                                        dtype=torch.float32, device=d)

            # State-masked row-0 path: Sq=1
            self._dec_O_row0 = torch.empty(1, 1, 8, 256,
                                           dtype=bf16, device=d)
            self._dec_lse_row0 = torch.empty(1, 8, 128,
                                             dtype=torch.float32, device=d)
            # State-masked rows 1+: Sq = chunk_size - 1 (pi0 has state
            # token at row 0, chunk_size-1 action rows after).
            _rows1_sq = max(chunk_size - 1, 1)
            self._dec_O_rows1 = torch.empty(1, _rows1_sq, 8, 256,
                                            dtype=bf16, device=d)
            r1_sq_r = ((_rows1_sq + 127) // 128) * 128
            self._dec_lse_rows1 = torch.empty(1, 8, r1_sq_r,
                                              dtype=torch.float32, device=d)

            # ──────────────────────────────────────────────────────────
            # SplitKV scratch buffers (worst-case num_splits per site).
            # Matches flash_api.cpp:set_params_splitkv allocations. Only
            # consumed when num_splits > 1 (the wrapper's num_splits
            # heuristic decides per call). max_splits bound is
            # min(128, num_sms*2, n_blocks) where n_blocks depends on
            # block_n per head_dim (HD<=64: 256, <=128: 128, else 64).
            # ──────────────────────────────────────────────────────────
            # SigLIP: D=72 → block_n=128, Sk=256 → n_blocks=2 → splits≤2.
            _sig_splits = 2
            self._vis_lse_accum = torch.empty(_sig_splits, num_views, 16, 256,
                                              dtype=torch.float32, device=d)
            self._vis_o_accum = torch.empty(_sig_splits, num_views, 16, 256, 96,
                                            dtype=torch.float32, device=d)
            # Encoder: D=256 → block_n=64, Sk=encoder_seq_max → n_blocks = ceil(S/64).
            _enc_splits = min(128, (encoder_seq_max + 63) // 64)
            self._enc_lse_accum = torch.empty(_enc_splits, 1, 8, encoder_seq_max,
                                              dtype=torch.float32, device=d)
            self._enc_o_accum = torch.empty(_enc_splits, 1, 8, encoder_seq_max, 256,
                                            dtype=torch.float32, device=d)
            # Decoder: D=256, Sk up to encoder_seq_max + chunk_size.
            _dec_splits = min(128, (total_kv + 63) // 64)
            self._dec_lse_accum = torch.empty(_dec_splits, 1, 8, chunk_size,
                                              dtype=torch.float32, device=d)
            self._dec_o_accum = torch.empty(_dec_splits, 1, 8, chunk_size, 256,
                                            dtype=torch.float32, device=d)
            # State-masked row-0: Sq=1
            self._dec_lse_accum_row0 = torch.empty(_dec_splits, 1, 8, 1,
                                                   dtype=torch.float32, device=d)
            self._dec_o_accum_row0 = torch.empty(_dec_splits, 1, 8, 1, 256,
                                                 dtype=torch.float32, device=d)
            # State-masked rows 1+: Sq=chunk_size-1
            self._dec_lse_accum_rows1 = torch.empty(_dec_splits, 1, 8, _rows1_sq,
                                                    dtype=torch.float32, device=d)
            self._dec_o_accum_rows1 = torch.empty(_dec_splits, 1, 8, _rows1_sq, 256,
                                                  dtype=torch.float32, device=d)
        # ``flash_attn`` (the upstream pip wheel) is needed only when:
        #   - ``FVK_RTX_FA2=0`` (legacy backend), or
        #   - ``FVK_RTX_FA2_SITES`` excludes some sites during bisection.
        # The default (``_use_fvk_fa2=True`` + all three sites enabled) goes
        # entirely through the vendored ``flash_rt_fa2.so``, so we make the
        # import lazy: environments without a prebuilt flash-attn wheel
        # (Modal / older CUDA images) can still use FlashRT for the
        # default RTX path. ``_flash_attn_call`` raises a clear error if
        # the legacy path is actually invoked without the package.
        self._flash_attn_func = _make_flash_attn_proxy(
            need_legacy=not self._use_fvk_fa2 or not all(self._fa2_sites.values()))

    # ── Pointer interface (for pipeline's fvk kernel calls) ──

    def get_ptrs(self) -> dict:
        return {
            "vis_Q": self.vis_Q.data_ptr(),
            "vis_K": self.vis_K.data_ptr(),
            "vis_V": self.vis_V.data_ptr(),
            "enc_Q": self.enc_Q.data_ptr(),
            "enc_K": self.enc_K.data_ptr(),
            "enc_V": self.enc_V.data_ptr(),
            "dec_Q": self.dec_Q.data_ptr(),
            "enc_k_layer_stride_bytes": self._enc_kv_layer_stride_bytes,
            "enc_v_layer_stride_bytes": self._enc_kv_layer_stride_bytes,
        }

    # ── Attention calls ──
    #
    # Each method returns the raw device pointer of the attention output.
    # ``flash_attn_func`` allocates a new tensor on every call; torch's
    # caching allocator reuses the same slot once warmed up, so the
    # returned pointer is stable across CUDA graph replays and can be fed
    # directly into the next GEMM without a copy (this saves ~1.4 ms per
    # full Pi0.5 inference on RTX 5090 vs copying into a fixed O buffer).

    def _call_fvk_fa2(self, q, k, v, o, lse, *, stream: int = 0,
                       softmax_scale=None,
                       lse_accum=None, o_accum=None):
        """Thin adapter around fvk.attention_fa2_fwd_fp16.

        q, k, v, o are (B, S, H, D) fp16 contiguous cuda tensors;
        lse is (B, H, seqlen_q) fp32. ``lse_accum`` and ``o_accum``
        are the splitkv scratch buffers (see
        flash_api.cpp:set_params_splitkv); pass them to let the
        wrapper's num_splits heuristic dispatch to the splitkv
        kernel when it improves SM occupancy. Pass None on both to
        force num_splits=1.

        ``softmax_scale`` defaults to 1/sqrt(head_dim) from the
        tensor shape. Override required for head_dim mismatch cases
        (e.g. SigLIP semantic HD=72 passed through HD=72 directly —
        the kernel pads internally to kHeadDim=96 but scale should
        still be 1/sqrt(72)).
        """
        B, Sq, Hq, D = q.shape
        Sk, Hk = k.shape[1], k.shape[2]
        if softmax_scale is None:
            softmax_scale = 1.0 / (D ** 0.5)
        lse_accum_ptr = lse_accum.data_ptr() if lse_accum is not None else 0
        o_accum_ptr = o_accum.data_ptr() if o_accum is not None else 0
        self._fa2_fwd(
            Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
            O=o.data_ptr(), softmax_lse=lse.data_ptr(),
            softmax_lse_accum=lse_accum_ptr,
            o_accum=o_accum_ptr,
            batch=B, seqlen_q=Sq, seqlen_k=Sk,
            num_heads_q=Hq, num_heads_kv=Hk, head_dim=D,
            q_strides=(q.stride(0), q.stride(1), q.stride(2)),
            k_strides=(k.stride(0), k.stride(1), k.stride(2)),
            v_strides=(v.stride(0), v.stride(1), v.stride(2)),
            o_strides=(o.stride(0), o.stride(1), o.stride(2)),
            softmax_scale=softmax_scale,
            num_sms=self._num_sms,
            stream=stream,
        )

    def set_fixed_shape(self, enabled: bool) -> None:
        """Enable/disable fixed-shape (seqused/devpos) state-prompt execution.

        Fixed shape masks the padded prompt prefix with FlashAttention-2
        ``seqused_k`` (``fwd_bf16_seqused``) on the encoder + decoder sites, so
        it is ONLY correct on the vendored bf16 FA2 path with those sites
        enabled. The legacy pip-flash-attn fallback (``FVK_RTX_FA2=0`` or a
        site excluded via ``FVK_RTX_FA2_SITES``) does NOT mask the padded keys
        and would silently produce wrong output — so refuse to enable there
        instead of falling back. Call with ``False`` to return to per-length
        execution (the shared backend is reused across pipelines, so the active
        pipeline must sync this each time it changes).
        """
        if enabled:
            reasons = []
            if self._is_fp16:
                reasons.append("backend is fp16 (seqused path is bf16-only)")
            if not self._use_fvk_fa2:
                reasons.append("FVK_RTX_FA2=0 (vendored FA2 disabled)")
            if not self._fa2_sites.get("encoder", False):
                reasons.append("encoder excluded from FVK_RTX_FA2_SITES")
            if not self._fa2_sites.get("decoder", False):
                reasons.append("decoder excluded from FVK_RTX_FA2_SITES")
            if not hasattr(getattr(self, "_fa2", None), "fwd_bf16_seqused"):
                reasons.append("flash_rt_fa2.fwd_bf16_seqused unavailable")
            if reasons:
                raise RuntimeError(
                    "Pi0.5 fixed-shape state-prompt mode needs the vendored "
                    "bf16 FlashAttention-2 seqused path on the encoder+decoder "
                    "sites; refusing because: " + "; ".join(reasons) + ". Use "
                    "state_prompt_mode='exact' (per-length capture), or enable "
                    "FA2 (unset FVK_RTX_FA2 / FVK_RTX_FA2_SITES).")
        self._fixed_shape = bool(enabled)

    def set_fixed_valid_len(self, valid_prefix_len: int) -> None:
        """Update the fixed-shape valid prefix length (host->device).

        ``valid_prefix_len`` = vision tokens + valid prompt tokens. Drives:
          - encoder self-attn seqused_k  = valid_prefix_len
          - decoder cross-attn seqused_k = valid_prefix_len + chunk
          - decoder K/V append row offset (devpos) = valid_prefix_len
        Called once per prompt (outside the captured graph); the graph reads
        these device buffers at replay, so no recapture as the length drifts.
        """
        import torch
        v = int(valid_prefix_len)
        self.enc_seqused.fill_(v)
        self.dec_seqused.fill_(v + self._chunk_size)
        self.dec_devpos.fill_(v)
        # Cold path (once per prompt): make the device writes visible before
        # the next graph replay reads them (the fills run on the current stream,
        # the graph replays on its own captured stream).
        torch.cuda.synchronize()

    def _call_fvk_fa2_seqused(self, q, k, v, o, lse, seqused, *,
                              stream: int = 0, softmax_scale=None):
        """FA2 with a device-side valid K length (``seqused_k``): the kernel
        early-exits past ``seqused[0]`` keys, so one fixed-shape (max-length)
        launch masks the padded prefix bit-exactly. bf16-only."""
        B, Sq, Hq, D = q.shape
        Sk, Hk = k.shape[1], k.shape[2]
        if softmax_scale is None:
            softmax_scale = 1.0 / (D ** 0.5)
        self._fa2.fwd_bf16_seqused(
            Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
            O=o.data_ptr(), softmax_lse=lse.data_ptr(),
            seqused_k=seqused.data_ptr(),
            batch=B, seqlen_q=Sq, seqlen_k=Sk,
            num_heads_q=Hq, num_heads_kv=Hk, head_dim=D,
            q_strides=(q.stride(0), q.stride(1), q.stride(2)),
            k_strides=(k.stride(0), k.stride(1), k.stride(2)),
            v_strides=(v.stride(0), v.stride(1), v.stride(2)),
            o_strides=(o.stride(0), o.stride(1), o.stride(2)),
            softmax_scale=softmax_scale, num_sms=self._num_sms, stream=stream)

    def vision_attn(self, stream: int = 0) -> int:
        # (batch=nv, seq=256, heads=16, head_dim=72) → per-view attention
        if self._fa2_sites["siglip"]:
            # No external padding — the wrapper sets params.d=72 and
            # params.d_rounded=96, then dispatches to the kHeadDim=96
            # template. The FA2 kernel itself handles the 72..96 col
            # zero-padding in smem, matching upstream flash_api.cpp
            # for HD=72 inputs exactly bit-for-bit.
            self._call_fvk_fa2(
                self.vis_Q, self.vis_K, self.vis_V,
                self._vis_O72, self._vis_lse, stream=stream,
                lse_accum=self._vis_lse_accum,
                o_accum=self._vis_o_accum)
            return self._vis_O72.data_ptr()

        out = self._flash_attn_func(
            self.vis_Q, self.vis_K, self.vis_V, causal=False)
        self._vis_out_ref = out
        return out.data_ptr()

    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        q = self.enc_Q[:seq].unsqueeze(0)                # (1, seq, 8, 256)
        k = self.enc_K[layer_idx, :seq].unsqueeze(0)     # (1, seq, 1, 256)
        v = self.enc_V[layer_idx, :seq].unsqueeze(0)     # (1, seq, 1, 256)
        if self._fa2_sites["encoder"]:
            o = self._enc_O[:, :seq].contiguous()
            if self._fixed_shape:
                # Fixed max shape; padded prefix keys masked via seqused_k.
                self._call_fvk_fa2_seqused(
                    q, k, v, o, self._enc_lse, self.enc_seqused, stream=stream)
            else:
                self._call_fvk_fa2(q, k, v, o, self._enc_lse, stream=stream,
                                   lse_accum=self._enc_lse_accum,
                                   o_accum=self._enc_o_accum)
            return o.data_ptr()
        out = self._flash_attn_func(q, k, v, causal=False)
        self._enc_out_ref = out
        return out.data_ptr()

    def decoder_attn(self, layer_idx: int, enc_seq: int, dec_seq: int,
                     stream: int = 0) -> int:
        total_kv = enc_seq + dec_seq
        q = self.dec_Q[:dec_seq].unsqueeze(0)                # (1, chunk, 8, 256)
        k = self.enc_K[layer_idx, :total_kv].unsqueeze(0)    # (1, total, 1, 256)
        v = self.enc_V[layer_idx, :total_kv].unsqueeze(0)    # (1, total, 1, 256)
        if self._fa2_sites["decoder"]:
            o = self._dec_O[:, :dec_seq].contiguous()
            if self._fixed_shape:
                # The decoder's action K/V were appended right after the valid
                # prefix (qkv_split_rope_devpos), so [0 : valid+chunk] is one
                # contiguous valid range — single seqused call masks padding.
                self._call_fvk_fa2_seqused(
                    q, k, v, o, self._dec_lse, self.dec_seqused, stream=stream)
            else:
                self._call_fvk_fa2(q, k, v, o, self._dec_lse, stream=stream,
                                   lse_accum=self._dec_lse_accum,
                                   o_accum=self._dec_o_accum)
            return o.data_ptr()
        out = self._flash_attn_func(q, k, v, causal=False)
        self._dec_out_ref = out
        return out.data_ptr()

    def decoder_attn_state_masked(self, layer_idx: int, kv_seq: int,
                                  dec_seq: int, state_nk: int,
                                  stream: int = 0) -> int:
        """Pi0 cross-attention with per-query KV windows.

        Row 0 of ``dec_Q`` is the state token and attends only to
        ``enc_K[layer_idx, :state_nk]``. Rows ``1..dec_seq`` are action
        tokens and attend to the full ``enc_K[layer_idx, :kv_seq]``. Two
        FA2 invocations are dispatched (row-0 with KV=state_nk, rows-1+
        with full KV) and their outputs copied into a pre-allocated
        ``dec_O_masked`` slot so the returned pointer is stable across
        CUDA graph replays.
        """
        q_state = self.dec_Q[:1].unsqueeze(0)
        k_state = self.enc_K[layer_idx, :state_nk].unsqueeze(0)
        v_state = self.enc_V[layer_idx, :state_nk].unsqueeze(0)

        if self._fa2_sites["decoder"]:
            self._call_fvk_fa2(
                q_state.contiguous(), k_state.contiguous(),
                v_state.contiguous(),
                self._dec_O_row0, self._dec_lse_row0, stream=stream,
                lse_accum=self._dec_lse_accum_row0,
                o_accum=self._dec_o_accum_row0)
            self.dec_O_masked[:1].copy_(self._dec_O_row0[0])

            if dec_seq > 1:
                q_act = self.dec_Q[1:dec_seq].unsqueeze(0).contiguous()
                k_act = self.enc_K[layer_idx, :kv_seq].unsqueeze(0)
                v_act = self.enc_V[layer_idx, :kv_seq].unsqueeze(0)
                act_rows = dec_seq - 1
                o_act = self._dec_O_rows1[:, :act_rows].contiguous()
                self._call_fvk_fa2(
                    q_act, k_act, v_act, o_act,
                    self._dec_lse_rows1, stream=stream,
                    lse_accum=self._dec_lse_accum_rows1,
                    o_accum=self._dec_o_accum_rows1)
                self.dec_O_masked[1:dec_seq].copy_(o_act[0])
            return self.dec_O_masked.data_ptr()

        out_state = self._flash_attn_func(q_state, k_state, v_state,
                                          causal=False)
        self.dec_O_masked[:1].copy_(out_state[0])

        if dec_seq > 1:
            q_act = self.dec_Q[1:dec_seq].unsqueeze(0)
            k_act = self.enc_K[layer_idx, :kv_seq].unsqueeze(0)
            v_act = self.enc_V[layer_idx, :kv_seq].unsqueeze(0)
            out_act = self._flash_attn_func(q_act, k_act, v_act,
                                            causal=False)
            self.dec_O_masked[1:dec_seq].copy_(out_act[0])

        return self.dec_O_masked.data_ptr()

    # ──────────────────────────────────────────────────────────────
    # AttentionBackend protocol implementation. Delegates to the same
    # flash_attn_func calls as the legacy vision/encoder/decoder_attn
    # methods above; both surfaces wrap identical state. The legacy
    # half will be retired once all pipelines route through the
    # protocol methods.
    # ──────────────────────────────────────────────────────────────

    _PROTOCOL_SITES = ("siglip", "encoder", "decoder")

    def sites(self) -> tuple[str, ...]:
        return self._PROTOCOL_SITES

    def head_dim(self, site: str) -> int:
        if site == "siglip":
            return 72
        if site in ("encoder", "decoder"):
            return 256
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")

    def num_q_heads(self, site: str) -> int:
        if site == "siglip":
            return 16
        if site in ("encoder", "decoder"):
            return 8
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")

    def num_kv_heads(self, site: str) -> int:
        if site == "siglip":
            return 16
        if site in ("encoder", "decoder"):
            return 1
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        """Return pointer dict for one (site, layer) pair.

        For Pi0.5 the SigLIP and encoder Q slots are shared across
        all layers (single fixed scratch). Encoder K/V are per-layer
        slices of the shared 3D cache. The decoder Q slot is a
        separate fixed scratch; decoder K/V slice the same encoder
        cache — so ``get_slot_ptrs("decoder", l)["K"]`` points at
        exactly the same rows as ``get_slot_ptrs("encoder", l)["K"]``.
        """
        if site == "siglip":
            # Vision attention: all 27 layers share one batched slot.
            # layer_idx is accepted for protocol uniformity but ignored.
            return {
                "Q": self.vis_Q.data_ptr(),
                "K": self.vis_K.data_ptr(),
                "V": self.vis_V.data_ptr(),
            }
        if site == "encoder":
            # Shared per-layer KV cache; Q is shared across all layers.
            # layer stride in ELEMENTS (not bytes): total_kv * 1 * 256
            layer_stride_elts = self.enc_K.shape[1] * self.enc_K.shape[2] * self.enc_K.shape[3]
            layer_off_elts = layer_idx * layer_stride_elts
            # bf16 == 2 bytes
            layer_off_bytes = layer_off_elts * 2
            return {
                "Q": self.enc_Q.data_ptr(),
                "K": self.enc_K.data_ptr() + layer_off_bytes,
                "V": self.enc_V.data_ptr() + layer_off_bytes,
            }
        if site == "decoder":
            layer_stride_elts = self.enc_K.shape[1] * self.enc_K.shape[2] * self.enc_K.shape[3]
            layer_off_bytes = layer_idx * layer_stride_elts * 2
            return {
                "Q": self.dec_Q.data_ptr(),
                "K": self.enc_K.data_ptr() + layer_off_bytes,
                "V": self.enc_V.data_ptr() + layer_off_bytes,
            }
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")

    def run(
        self,
        site: str,
        layer_idx: int,
        q_seq: int,
        *,
        kv_seq=None,
        stream: int = 0,
        state_nk=None,
    ) -> int:
        """Dispatch to the legacy attention call for the given site.

        Identical kernel invocation as the legacy vision/encoder/
        decoder_attn methods — this is a thin dispatcher so that
        pipeline code can use a uniform API across models.

        ``state_nk`` is only accepted at the ``"decoder"`` site and
        activates the Pi0 state-masked variant (row 0 query sees only
        ``state_nk`` keys, remaining rows see ``kv_seq`` keys).
        """
        if site == "siglip":
            # SigLIP is per-view batched self-attention; q_seq is
            # tokens-per-view (256) and is already baked into the
            # fixed-shape vis_Q tensor, so the parameter is accepted
            # for protocol uniformity but not used to slice.
            return self.vision_attn(stream=stream)
        if site == "encoder":
            if kv_seq is not None and kv_seq != q_seq:
                raise ValueError(
                    f"encoder site is self-attention; kv_seq must be "
                    f"None or equal to q_seq, got kv_seq={kv_seq} "
                    f"q_seq={q_seq}"
                )
            return self.encoder_attn(layer_idx, q_seq, stream=stream)
        if site == "decoder":
            if kv_seq is None:
                raise ValueError(
                    "decoder site is cross-attention against the "
                    "shared encoder KV cache; kv_seq (the total KV "
                    "length including freshly-written chunk rows) "
                    "must be supplied"
                )
            dec_seq = q_seq
            if state_nk is not None:
                return self.decoder_attn_state_masked(
                    layer_idx, kv_seq, dec_seq, int(state_nk),
                    stream=stream)
            enc_seq = kv_seq - dec_seq
            if enc_seq < 0:
                raise ValueError(
                    f"decoder kv_seq ({kv_seq}) must be >= q_seq "
                    f"({q_seq}) — the chunk is appended to the "
                    f"encoder cache"
                )
            return self.decoder_attn(layer_idx, enc_seq, dec_seq, stream=stream)
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")


# ─────────────────────────────────────────────────────────────────────────
# Backwards-compatible alias — old name before the Torch-prefix drop.
# External plugins pinned to ``from flash_rt.hardware.rtx.attn_backend
# import TorchFlashAttnBackend`` continue to work; the name was
# misleading because the backend is used by the jax frontend too (JAX
# has no dependency on torch in the attention call path — it only
# needs torch transitively for the scratch-tensor allocator, which is
# tangentially related to the attention kernel itself).
TorchFlashAttnBackend = RtxFlashAttnBackend

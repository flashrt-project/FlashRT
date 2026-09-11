"""FlashRT AMD — CDNA4 attention backend for the Pi0.5 pipeline.

Mirror of :class:`flash_rt.hardware.rtx.attn_backend.RtxFlashAttnBackend`
as consumed by :mod:`flash_rt.amd.models.pi05.pipeline`: the backend owns
all Q/K/V/O attention memory, exposes raw device pointers via
:meth:`get_ptrs` so the pipeline can write into them from
``flash_rt_amd_kernels`` calls, and runs attention via
``run(site, layer_idx, ...)``.

Buffer ownership, shapes, dtypes and the shared encoder/decoder K/V cache
layout are IDENTICAL to the RTX backend:

    vis_Q/K/V   — (num_views, 256, 16, 72) bf16
    enc_Q       — (encoder_seq_max, 8, 256) bf16
    enc_K/V     — (num_encoder_layers, encoder_seq_max + chunk, 1, 256) bf16
    dec_Q       — (chunk_size, 8, 256) bf16

The K/V cache is shared across encoder + decoder: the encoder writes
``enc_K[i, :enc_seq]`` for layer ``i``; the decoder appends its chunk
rows at ``enc_K[i, enc_seq:enc_seq+chunk]`` (or at the runtime ``devpos``
offset in fixed-shape mode) before cross-attention. All buffers are
allocated ONCE at ``__init__`` and never reallocated — the pipeline bakes
their pointers into a captured HIP graph.

INTERIM ATTENTION MATH
----------------------
This backend does NOT yet dispatch to a vendored Flash-Attention kernel
(the RTX backend uses ``flash_rt.flash_rt_fa2``). Until the CDNA4
attention kernel lands in ``flash_rt_amd_kernels``, each site is computed
with torch ops over the SAME owned buffers:

  * views / permutes / expands of the owned Q/K/V tensors (no copies),
  * ``torch.nn.functional.scaled_dot_product_attention`` with
    ``dropout_p=0.0``,
  * a final ``copy_`` into the pre-allocated, pointer-stable O tensor
    (sdpa has no ``out=`` argument, so the copy is the last op).

Masking semantics match the FA2 call semantics the pipeline relies on:

  * "siglip"  — per-view bidirectional self-attention, no mask.
  * "encoder" — bidirectional self-attention over the first ``seq``
    tokens; in fixed-shape mode the padded prompt keys are masked with a
    boolean key mask driven by :meth:`set_fixed_valid_len` (the SDPA twin
    of FA2 ``seqused_k``).
  * "decoder" — chunk queries attend the full [valid prefix + chunk]
    K/V range bidirectionally; fixed-shape mode masks keys past
    ``valid + chunk``.

GQA (8 Q heads over 1 KV head on encoder/decoder) uses ``expand`` — no
materialized repeat. Every op in the per-call path is legal under HIP
stream capture given the pipeline's warmup contract: the pipeline runs
3 warmup iterations on the capture stream before ``begin_capture`` so
torch's caching allocator reaches steady state and sdpa's internal
workspace allocations are served from cached blocks (the same contract
the RTX legacy pip-flash-attn path relied on). The mask/seqused buffers
are updated OUTSIDE the captured graph (cold path, once per prompt) and
are read by pointer at replay — a changing prompt length never forces a
re-capture.

Stream note: like the RTX backend's torch-side ops, the sdpa call runs
on torch's CURRENT stream. The frontend wraps calibration + capture +
replay in ``torch.cuda.stream(...)`` with the same stream it passes to
the pipeline, so all work lands on the capture stream. (torch on ROCm
keeps the "cuda" device / stream API names.)
"""

from __future__ import annotations


class Cdna4AttnBackend:
    """Pi0.5 attention backend for AMD CDNA4 (MI350X).

    Implements the surface consumed by
    :class:`flash_rt.amd.models.pi05.pipeline.Pi05Pipeline`:

      * :meth:`get_ptrs` — raw device pointers for every attention INPUT
        buffer plus the enc K/V per-layer stride,
      * :meth:`run` — dispatcher over the "siglip" / "encoder" /
        "decoder" sites (delegating to :meth:`vision_attn` /
        :meth:`encoder_attn` / :meth:`decoder_attn`),
      * :meth:`set_fixed_shape` / :meth:`set_fixed_valid_len` — the
        fixed-shape (max-length graph) state-prompt support, including
        the ``dec_devpos`` device buffer read by
        ``qkv_split_rope_devpos``.

    Attention output pointers are stable across graph replays: each site
    writes into a pre-allocated O tensor owned by this backend and
    returns the same pointer on every call.
    """

    def __init__(self, num_views: int, encoder_seq_max: int, chunk_size: int,
                 num_encoder_layers: int = 18, dtype=None):
        import torch
        self._torch = torch
        # ``dtype`` selects the 16-bit tensor type used for Q/K/V/O
        # buffers. Defaults to bfloat16 (pi05).
        bf16 = dtype if dtype is not None else torch.bfloat16
        # torch on ROCm keeps the "cuda" device string.
        d = "cuda"

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

        # ── Fixed-shape (seqused/devpos) state-prompt support ──
        # One captured graph at the MAX prefix length serves any prompt length.
        # The valid prefix length is pushed into device buffers per set_prompt
        # (set_fixed_valid_len — runtime inputs, never a recapture):
        #   - enc_seqused / dec_seqused mirror the FA2 seqused_k contract and
        #     additionally drive the boolean key masks consumed by sdpa here,
        #   - dec_devpos is read by the pipeline's qkv_split_rope_devpos kernel
        #     to append the chunk K/V right after the valid prefix.
        # batch=1, so each is a single int32.
        self._fixed_shape = False
        self._calibrating = False
        self.enc_seqused = torch.zeros(1, dtype=torch.int32, device=d)  # vis_enc+plen
        self.dec_seqused = torch.zeros(1, dtype=torch.int32, device=d)  # +chunk
        self.dec_devpos = torch.zeros(1, dtype=torch.int32, device=d)   # = enc_seqused

        # Boolean key masks (True = attend) for the fixed-shape path.
        # Shape (1, 1, 1, S) broadcasts over batch/heads/queries in sdpa.
        # Contents are rewritten by set_fixed_valid_len (cold path); the
        # captured graph reads them by pointer at replay.
        self._enc_key_mask = torch.ones(1, 1, 1, encoder_seq_max,
                                        dtype=torch.bool, device=d)
        self._dec_key_mask = torch.ones(1, 1, 1, total_kv,
                                        dtype=torch.bool, device=d)
        # Pre-allocated iota used to rebuild the masks without host loops.
        self._kv_iota = torch.arange(total_kv, dtype=torch.int32, device=d)

        # Cached shape metadata
        self._num_views = num_views
        self._encoder_seq_max = encoder_seq_max
        self._chunk_size = chunk_size
        self._num_encoder_layers = num_encoder_layers
        # enc_K/V layer stride in bytes (bf16 = 2 bytes)
        self._enc_kv_layer_stride_bytes = (
            total_kv * 1 * 256 * self.enc_K.element_size())

        # Pre-allocated attention OUTPUT buffers. The pipeline reads the
        # returned pointer as a flat row-major (rows, heads*head_dim)
        # bf16 tensor, so O keeps the (rows, H, D) input layout and the
        # sdpa result — (B, H, S, D) — is copy_'d through a transposed
        # view as the last op of every call.
        self._vis_O = torch.empty(num_views, 256, 16, 72, dtype=bf16, device=d)
        self._enc_O = torch.empty(encoder_seq_max, 8, 256, dtype=bf16, device=d)
        self._dec_O = torch.empty(chunk_size, 8, 256, dtype=bf16, device=d)

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

    # ── Fixed-shape state-prompt support ──

    def set_fixed_shape(self, enabled: bool) -> None:
        """Enable/disable fixed-shape (seqused/devpos) state-prompt execution.

        The interim sdpa path masks padded keys with the boolean key masks
        maintained by :meth:`set_fixed_valid_len`, so — unlike the RTX
        backend, which must validate that the vendored FA2 seqused entries
        exist — fixed-shape is always available here. The shared backend is
        reused across pipelines, so the active pipeline re-syncs this flag
        each time it changes.
        """
        self._fixed_shape = bool(enabled)

    def set_calibrating(self, enabled: bool) -> None:
        """Mark the FP8-calibration window (deterministic-attention hint).

        Quantization scales are derived from activation amax during the
        calibration pass, so any run-to-run nondeterminism in an attention
        site becomes permanent scale jitter. Backends whose attention
        library is nondeterministic across processes (e.g. aiter) override
        their site dispatch to a deterministic kernel while this flag is
        set. The sdpa math here is already deterministic, so the base
        implementation only records the flag.
        """
        self._calibrating = bool(enabled)

    def set_fixed_valid_len(self, valid_prefix_len: int) -> None:
        """Update the fixed-shape valid prefix length (host->device).

        ``valid_prefix_len`` = vision tokens + valid prompt tokens. Drives:
          - encoder self-attn key mask   = [0, valid_prefix_len)
          - decoder cross-attn key mask  = [0, valid_prefix_len + chunk)
          - decoder K/V append row offset (devpos) = valid_prefix_len
        Called once per prompt (outside the captured graph); the graph reads
        these device buffers at replay, so no recapture as the length drifts.
        """
        torch = self._torch
        v = int(valid_prefix_len)
        self.enc_seqused.fill_(v)
        self.dec_seqused.fill_(v + self._chunk_size)
        self.dec_devpos.fill_(v)
        # Rebuild the boolean key masks read by the captured sdpa calls.
        self._enc_key_mask[0, 0, 0].copy_(
            self._kv_iota[:self._encoder_seq_max] < v)
        self._dec_key_mask[0, 0, 0].copy_(
            self._kv_iota < (v + self._chunk_size))
        # Cold path (once per prompt): make the device writes visible before
        # the next graph replay reads them (the fills run on the current
        # stream, the graph replays on its own captured stream).
        torch.cuda.synchronize()

    # ── Attention calls ──
    #
    # Each method returns the raw device pointer of the attention output;
    # the pointer is the same on every call (pre-allocated O tensors), so
    # it can be baked into the captured HIP graph and fed directly into
    # the next GEMM without a copy.

    def _sdpa(self, q, k, v, attn_mask=None):
        """scaled_dot_product_attention with the pipeline's fixed contract:
        no dropout, no causal mask (all Pi0.5 sites are bidirectional),
        default softmax scale 1/sqrt(head_dim)."""
        import torch.nn.functional as F
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    def vision_attn(self, stream: int = 0) -> int:
        # (batch=nv, seq=256, heads=16, head_dim=72) → per-view attention.
        # sdpa expects (B, H, S, D); permute is a view, no copy.
        q = self.vis_Q.permute(0, 2, 1, 3)
        k = self.vis_K.permute(0, 2, 1, 3)
        v = self.vis_V.permute(0, 2, 1, 3)
        out = self._sdpa(q, k, v)                     # (nv, 16, 256, 72)
        self._vis_O.permute(0, 2, 1, 3).copy_(out)    # → (nv, 256, 16, 72)
        return self._vis_O.data_ptr()

    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        # GQA 8Q/1KV: expand the single KV head across the 8 query heads
        # (stride-0 broadcast, no materialized repeat).
        q = self.enc_Q[:seq].permute(1, 0, 2).unsqueeze(0)       # (1, 8, seq, 256)
        k = self.enc_K[layer_idx, :seq].permute(1, 0, 2).unsqueeze(0)
        v = self.enc_V[layer_idx, :seq].permute(1, 0, 2).unsqueeze(0)
        k = k.expand(1, 8, seq, 256)
        v = v.expand(1, 8, seq, 256)
        mask = self._enc_key_mask[..., :seq] if self._fixed_shape else None
        out = self._sdpa(q, k, v, attn_mask=mask)                # (1, 8, seq, 256)
        self._enc_O[:seq].transpose(0, 1).copy_(out[0])
        return self._enc_O.data_ptr()

    def decoder_attn(self, layer_idx: int, enc_seq: int, dec_seq: int,
                     stream: int = 0) -> int:
        total_kv = enc_seq + dec_seq
        q = self.dec_Q[:dec_seq].permute(1, 0, 2).unsqueeze(0)   # (1, 8, chunk, 256)
        k = self.enc_K[layer_idx, :total_kv].permute(1, 0, 2).unsqueeze(0)
        v = self.enc_V[layer_idx, :total_kv].permute(1, 0, 2).unsqueeze(0)
        k = k.expand(1, 8, total_kv, 256)
        v = v.expand(1, 8, total_kv, 256)
        # Fixed-shape: the chunk K/V were appended right after the valid
        # prefix (qkv_split_rope_devpos), so [0 : valid+chunk] is one
        # contiguous valid range — the key mask hides the padding rows.
        mask = self._dec_key_mask[..., :total_kv] if self._fixed_shape else None
        out = self._sdpa(q, k, v, attn_mask=mask)                # (1, 8, chunk, 256)
        self._dec_O[:dec_seq].transpose(0, 1).copy_(out[0])
        return self._dec_O.data_ptr()

    # ── Uniform site dispatcher (surface consumed by the pipeline) ──

    _PROTOCOL_SITES = ("siglip", "encoder", "decoder")

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
        """Dispatch to the attention call for the given site.

        Mirrors the RTX backend's dispatcher. ``state_nk`` (the Pi0
        state-masked decoder variant) is not implemented on the interim
        CDNA4 path — Pi0.5 never passes it.
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
            if state_nk is not None:
                raise NotImplementedError(
                    "state_nk (Pi0 state-masked decoder attention) is not "
                    "implemented on the CDNA4 interim backend")
            dec_seq = q_seq
            enc_seq = kv_seq - dec_seq
            if enc_seq < 0:
                raise ValueError(
                    f"decoder kv_seq ({kv_seq}) must be >= q_seq "
                    f"({q_seq}) — the chunk is appended to the "
                    f"encoder cache"
                )
            return self.decoder_attn(layer_idx, enc_seq, dec_seq, stream=stream)
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")

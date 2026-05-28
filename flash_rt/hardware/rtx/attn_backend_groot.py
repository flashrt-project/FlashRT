"""FlashRT — RTX GROOT N1.6 attention backend.

GROOT has four distinct attention shapes:

  - SigLIP2     :  (batch=num_views, seq=256, heads=16,  head_dim=72)   self
  - Qwen3 GQA   :  (1, Se, 16q/8kv, head_dim=128) full self-attn
  - DiT self    :  (1, Sa=51,  heads=32, head_dim=48)  self
  - DiT cross   :  Q (1, Sa, 32, 48)  vs  K/V (1, Se, 32, 48)  cross

FlashRT's vendored FA2 covers all four head_dims (48 → kHeadDim=96,
72 → 96, 128 direct) so we can route everything through the
``flash_rt_fa2`` backend on RTX 5090 without requiring the upstream
``flash-attn`` pip wheel.

This is a separate backend from ``RtxFlashAttnBackend`` (the Pi0/Pi0.5
one) because the attention sites and shapes are entirely different.
The protocol matches: ``get_ptrs()`` returns a dict of input pointers
for the pipeline to write into via fvk kernels, and each ``*_attn``
method returns the raw device pointer of a backend-owned output tensor.

Future revision will replace this with a pure-C
``CFlashAttnBackend`` that calls into ``libfmha_fp16.so``. The
pipeline doesn't care which backend is used.
"""

from __future__ import annotations


# Pi0.5 uses bfloat16; GROOT N1.6 uses float16 throughout (matches
# Thor's reference implementation, lets us use the existing FP16
# fvk kernel set including layer_norm_no_affine_fp16 and
# ada_layer_norm_fp16 which don't have bf16 variants).


class RtxFlashAttnBackendGroot:
    """FA2-backed attention for GROOT N1.6 (FP16) on RTX hardware.

    Name parallel to :class:`RtxFlashAttnBackend` (the Pi0/Pi0.5
    backend). Previously called ``TorchFlashAttnBackendGroot`` — the
    ``Torch`` prefix was misleading since jax frontends use this
    backend too. A deprecated module-level alias is preserved at the
    bottom of this file for external plugins pinned to the old name.

    **Protocol compatibility**: implements
    :class:`flash_rt.hardware.backend.AttentionBackend` via
    :meth:`sites` / :meth:`get_slot_ptrs` / :meth:`run`. Legacy
    methods (:meth:`vision_attn` / :meth:`qwen3_attn` /
    :meth:`dit_self_attn` / :meth:`dit_cross_attn` / :meth:`get_ptrs`)
    remain live for the current pipeline and will be retired once the
    pipeline migrates to the protocol API.

    Site mapping
    ------------
    GROOT has four distinct attention shapes:

      * ``"siglip"`` — 27 SigLIP layers, batched per-view self-attn,
        16 heads, head_dim=72, 256 tokens/view.
      * ``"qwen3"`` — 16 Qwen3 encoder layers, GQA 16Q/8KV, head_dim=128.
      * ``"dit_self"`` — 16 DiT self-attention blocks (DiT layers
        1, 3, 5, …, 31 — the odd layers). 32 heads, head_dim=48.
      * ``"dit_cross"`` — 16 DiT cross-attention blocks (DiT layers
        0, 2, 4, …, 30 — the even layers). Q over action seq,
        K/V over the Qwen3 backbone features precomputed once per
        diffusion loop.

    DiT layer indexing
    ------------------
    The production pipeline iterates DiT layers ``l`` in ``range(32)``
    and decides ``is_self = (l % 2 == 1)`` per iteration. The new
    protocol wants the pipeline to pass ``run("dit_self", l, …)`` or
    ``run("dit_cross", l, …)`` with the **full DiT layer index**
    ``l`` (0..31). This backend maps internally:

      * self site: ``layer_idx`` is passed straight through; there
        are effectively 16 distinct layer indices (the odd ones) but
        the shared Q/K/V torch tensors don't care — a single slot is
        reused across all self-attn layers, the layer_idx is accepted
        for protocol uniformity.
      * cross site: ``block_idx = layer_idx // 2``. Each cross block
        has its own precomputed K/V in ``dit_cross_K[block_idx]`` /
        ``dit_cross_V[block_idx]``; Q is shared across all blocks.
        The pipeline must write Q into the shared slot before every
        cross attention call but writes K/V into
        ``dit_cross_K/V[block_idx]`` **once** before the diffusion
        loop (that's the ``precompute_cross_kv`` step).
    """

    def __init__(self, num_views: int, encoder_seq_max: int,
                 num_dit_actions: int, dit_kv_seq: int,
                 num_dit_cross_blocks: int = 16):
        import torch
        self._torch = torch
        fp16 = torch.float16
        d = "cuda"

        # ── SigLIP2 vision attention ──
        self.vis_Q = torch.empty(num_views, 256, 16, 72, dtype=fp16, device=d)
        self.vis_K = torch.empty(num_views, 256, 16, 72, dtype=fp16, device=d)
        self.vis_V = torch.empty(num_views, 256, 16, 72, dtype=fp16, device=d)

        # ── Qwen3 backbone GQA ──
        self.qwen3_Q = torch.empty(encoder_seq_max, 16, 128, dtype=fp16, device=d)
        self.qwen3_K = torch.empty(encoder_seq_max, 8, 128, dtype=fp16, device=d)
        self.qwen3_V = torch.empty(encoder_seq_max, 8, 128, dtype=fp16, device=d)

        # ── DiT self-attention ──
        self.dit_self_Q = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)
        self.dit_self_K = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)
        self.dit_self_V = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)

        # ── DiT cross-attention ──
        # Q is the action stream (rewritten per layer by the pipeline).
        # K/V are 16 per-block precomputed projections of the Qwen3 backbone,
        # owned by the backend so vendored FA2 can read them as torch tensors.
        # The pipeline writes these via fp16_nn GEMMs in precompute_cross_kv().
        self.dit_cross_Q = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)
        self.dit_cross_K = [
            torch.empty(dit_kv_seq, 32, 48, dtype=fp16, device=d)
            for _ in range(num_dit_cross_blocks)
        ]
        self.dit_cross_V = [
            torch.empty(dit_kv_seq, 32, 48, dtype=fp16, device=d)
            for _ in range(num_dit_cross_blocks)
        ]

        self._num_views = num_views
        self._encoder_seq_max = encoder_seq_max
        self._num_dit_actions = num_dit_actions
        self._dit_kv_seq = dit_kv_seq
        self._num_dit_cross_blocks = num_dit_cross_blocks

        try:
            from flash_rt import flash_rt_fa2 as fa2
        except ImportError as e:
            raise RuntimeError(
                "GROOT RTX backend requires FlashRT's vendored FA2 module "
                "(`flash_rt.flash_rt_fa2`). Build it with CMake on RTX "
                "targets; do not install/use the upstream `flash_attn` pip "
                f"package for this path. Import error: {e}"
            ) from e
        self._fa2 = fa2
        self._fa2_fwd = fa2.fwd_fp16
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count

        def lse(batch: int, heads: int, seq: int):
            return torch.empty(
                batch, heads, self._round_up_128(seq),
                dtype=torch.float32, device=d)

        def split_scratch(batch: int, heads: int, q_seq: int,
                          kv_seq: int, head_dim: int):
            block_n = 256 if head_dim <= 64 else (128 if head_dim <= 128 else 64)
            max_splits = max(1, min(128, (kv_seq + block_n - 1) // block_n))
            lse_accum = torch.empty(
                max_splits, batch, heads, q_seq,
                dtype=torch.float32, device=d)
            o_accum = torch.empty(
                max_splits, batch, heads, q_seq,
                self._round_head_dim(head_dim),
                dtype=torch.float32, device=d)
            return lse_accum, o_accum

        # Preallocated outputs and LSE/splitkv scratch. Returned pointers stay
        # stable across CUDA Graph capture and replay.
        self.vis_O = torch.empty_like(self.vis_Q)
        self._vis_lse = lse(num_views, 16, 256)
        self._vis_lse_accum, self._vis_o_accum = split_scratch(
            num_views, 16, 256, 256, 72)

        self.qwen3_O = torch.empty_like(self.qwen3_Q)
        self._qwen3_lse = lse(1, 16, encoder_seq_max)
        self._qwen3_lse_accum, self._qwen3_o_accum = split_scratch(
            1, 16, encoder_seq_max, encoder_seq_max, 128)

        self.dit_self_O = torch.empty_like(self.dit_self_Q)
        self._dit_self_lse = lse(1, 32, num_dit_actions)
        self._dit_self_lse_accum, self._dit_self_o_accum = split_scratch(
            1, 32, num_dit_actions, num_dit_actions, 48)

        self.dit_cross_O = torch.empty_like(self.dit_cross_Q)
        self._dit_cross_lse = lse(1, 32, num_dit_actions)
        self._dit_cross_lse_accum, self._dit_cross_o_accum = split_scratch(
            1, 32, num_dit_actions, dit_kv_seq, 48)

    def _run_fa2(self, q, k, v, o, lse, *, stream: int = 0,
                 lse_accum=None, o_accum=None) -> int:
        B, Sq, Hq, D = q.shape
        Sk, Hk = k.shape[1], k.shape[2]
        self._fa2_fwd(
            Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
            O=o.data_ptr(), softmax_lse=lse.data_ptr(),
            softmax_lse_accum=(lse_accum.data_ptr()
                               if lse_accum is not None else 0),
            o_accum=(o_accum.data_ptr() if o_accum is not None else 0),
            batch=B, seqlen_q=Sq, seqlen_k=Sk,
            num_heads_q=Hq, num_heads_kv=Hk, head_dim=D,
            q_strides=(q.stride(0), q.stride(1), q.stride(2)),
            k_strides=(k.stride(0), k.stride(1), k.stride(2)),
            v_strides=(v.stride(0), v.stride(1), v.stride(2)),
            o_strides=(o.stride(0), o.stride(1), o.stride(2)),
            softmax_scale=1.0 / (D ** 0.5),
            num_sms=self._num_sms,
            stream=stream,
        )
        return o.data_ptr()

    @staticmethod
    def _round_up_128(x: int) -> int:
        return ((int(x) + 127) // 128) * 128

    @staticmethod
    def _round_head_dim(x: int) -> int:
        return (int(x) + 31) & ~31

    # ── Pointer interface (for pipeline's fvk kernel calls) ──

    def get_ptrs(self) -> dict:
        return {
            "vis_Q": self.vis_Q.data_ptr(),
            "vis_K": self.vis_K.data_ptr(),
            "vis_V": self.vis_V.data_ptr(),
            "qwen3_Q": self.qwen3_Q.data_ptr(),
            "qwen3_K": self.qwen3_K.data_ptr(),
            "qwen3_V": self.qwen3_V.data_ptr(),
            "dit_self_Q": self.dit_self_Q.data_ptr(),
            "dit_self_K": self.dit_self_K.data_ptr(),
            "dit_self_V": self.dit_self_V.data_ptr(),
            "dit_cross_Q": self.dit_cross_Q.data_ptr(),
            "dit_cross_K": [t.data_ptr() for t in self.dit_cross_K],
            "dit_cross_V": [t.data_ptr() for t in self.dit_cross_V],
        }

    # ── Attention calls ──
    #
    # All four return stable pointers to backend-owned output tensors.

    def vision_attn(self, stream: int = 0) -> int:
        """Per-view SigLIP2 self-attention.

        Returns the device pointer of the (num_views, 256, 16, 72) output.
        Downstream the pipeline reshapes/views this as (num_views * 256, 1152)
        and feeds it to the next FP8 GEMM (out projection).
        """
        return self._run_fa2(
            self.vis_Q, self.vis_K, self.vis_V, self.vis_O,
            self._vis_lse, stream=stream,
            lse_accum=self._vis_lse_accum,
            o_accum=self._vis_o_accum)

    def qwen3_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        """Qwen3 GQA self-attention for one layer (16Q/8KV, head_dim=128).

        Reads from ``qwen3_Q[:seq]``, ``qwen3_K[:seq]``, ``qwen3_V[:seq]``.
        Returns the (1, seq, 16, 128) output pointer.
        """
        q = self.qwen3_Q[:seq].unsqueeze(0)        # (1, seq, 16, 128)
        k = self.qwen3_K[:seq].unsqueeze(0)        # (1, seq, 8,  128)
        v = self.qwen3_V[:seq].unsqueeze(0)        # (1, seq, 8,  128)
        o = self.qwen3_O[:seq].unsqueeze(0)
        return self._run_fa2(
            q, k, v, o, self._qwen3_lse, stream=stream,
            lse_accum=self._qwen3_lse_accum,
            o_accum=self._qwen3_o_accum)

    def dit_self_attn(self, layer_idx: int, sa: int, stream: int = 0) -> int:
        """DiT self-attention (32 heads × 48 head_dim) over Sa tokens.

        Returns the (1, Sa, 32, 48) output pointer.
        """
        q = self.dit_self_Q[:sa].unsqueeze(0)
        k = self.dit_self_K[:sa].unsqueeze(0)
        v = self.dit_self_V[:sa].unsqueeze(0)
        o = self.dit_self_O[:sa].unsqueeze(0)
        return self._run_fa2(
            q, k, v, o, self._dit_self_lse, stream=stream,
            lse_accum=self._dit_self_lse_accum,
            o_accum=self._dit_self_o_accum)

    def dit_cross_attn(self, cross_block_idx: int, sa: int,
                       kv_seq: int, stream: int = 0) -> int:
        """DiT cross-attention for one cross block.

        ``cross_block_idx`` is in [0, 16) — the cross blocks are at DiT
        layers 0, 2, 4, ..., 30. The backend owns persistent K/V tensors
        for each cross block (precomputed by the pipeline before each
        diffusion run loop). Q is the action stream which the pipeline
        rewrites per layer.
        """
        q = self.dit_cross_Q[:sa].unsqueeze(0)
        k = self.dit_cross_K[cross_block_idx][:kv_seq].unsqueeze(0)
        v = self.dit_cross_V[cross_block_idx][:kv_seq].unsqueeze(0)
        o = self.dit_cross_O[:sa].unsqueeze(0)
        return self._run_fa2(
            q, k, v, o, self._dit_cross_lse, stream=stream,
            lse_accum=self._dit_cross_lse_accum,
            o_accum=self._dit_cross_o_accum)

    # ──────────────────────────────────────────────────────────────
    # AttentionBackend protocol implementation. Wraps the same vendored
    # FA2 dispatches as the legacy methods above. The legacy half will
    # be retired once the GROOT pipeline routes entirely through the
    # protocol methods.
    # ──────────────────────────────────────────────────────────────

    _PROTOCOL_SITES = ("siglip", "qwen3", "dit_self", "dit_cross")

    def sites(self) -> tuple[str, ...]:
        return self._PROTOCOL_SITES

    def head_dim(self, site: str) -> int:
        return {
            "siglip": 72,
            "qwen3": 128,
            "dit_self": 48,
            "dit_cross": 48,
        }[site]

    def num_q_heads(self, site: str) -> int:
        return {
            "siglip": 16,
            "qwen3": 16,
            "dit_self": 32,
            "dit_cross": 32,
        }[site]

    def num_kv_heads(self, site: str) -> int:
        return {
            "siglip": 16,  # SigLIP is MHA
            "qwen3": 8,    # Qwen3 is GQA 16Q/8KV
            "dit_self": 32,
            "dit_cross": 32,
        }[site]

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        """Return pointer dict for one (site, layer) pair.

        Notes on layer_idx semantics per site:

          * ``siglip``: all 27 layers share one batched Q/K/V slot.
            layer_idx accepted for uniformity but ignored.
          * ``qwen3``: all 16 layers share one flat Q/K/V slot
            (GQA). layer_idx ignored.
          * ``dit_self``: all 16 self-attn DiT blocks share one flat
            Q/K/V slot. ``layer_idx`` is the full DiT layer index
            (1, 3, ..., 31) and is accepted for uniformity but only
            used to sanity-check the odd-layer contract.
          * ``dit_cross``: 16 per-block K/V slots (one tensor each).
            ``layer_idx`` is the full DiT layer index (0, 2, 4, ..., 30)
            which we map to ``block_idx = layer_idx // 2``. Q is
            shared across all 16 cross blocks.
        """
        if site == "siglip":
            return {
                "Q": self.vis_Q.data_ptr(),
                "K": self.vis_K.data_ptr(),
                "V": self.vis_V.data_ptr(),
            }
        if site == "qwen3":
            return {
                "Q": self.qwen3_Q.data_ptr(),
                "K": self.qwen3_K.data_ptr(),
                "V": self.qwen3_V.data_ptr(),
            }
        if site == "dit_self":
            return {
                "Q": self.dit_self_Q.data_ptr(),
                "K": self.dit_self_K.data_ptr(),
                "V": self.dit_self_V.data_ptr(),
            }
        if site == "dit_cross":
            if layer_idx % 2 != 0:
                raise ValueError(
                    f"dit_cross layer_idx must be even (DiT cross "
                    f"blocks are at layers 0, 2, 4, ..., 30), got "
                    f"layer_idx={layer_idx}"
                )
            block_idx = layer_idx // 2
            if not (0 <= block_idx < self._num_dit_cross_blocks):
                raise IndexError(
                    f"dit_cross block_idx {block_idx} out of range "
                    f"[0, {self._num_dit_cross_blocks})"
                )
            return {
                "Q": self.dit_cross_Q.data_ptr(),
                "K": self.dit_cross_K[block_idx].data_ptr(),
                "V": self.dit_cross_V[block_idx].data_ptr(),
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
    ) -> int:
        """Dispatch to the legacy attention call for the given site.

        Wraps the same vendored FA2 invocations as the legacy methods.
        Returns a backend-owned output tensor pointer that is stable
        across CUDA Graph capture and replay.
        """
        if site == "siglip":
            # Pi0.5's SigLIP was 2-view × 256 tokens, GROOT's SigLIP
            # (27L) uses a different, per-view batched API. The
            # legacy vision_attn doesn't take q_seq at all.
            return self.vision_attn(stream=stream)
        if site == "qwen3":
            if kv_seq is not None and kv_seq != q_seq:
                raise ValueError(
                    f"qwen3 site is self-attention; kv_seq must be "
                    f"None or equal to q_seq, got kv_seq={kv_seq} "
                    f"q_seq={q_seq}"
                )
            return self.qwen3_attn(layer_idx, q_seq, stream=stream)
        if site == "dit_self":
            if kv_seq is not None and kv_seq != q_seq:
                raise ValueError(
                    f"dit_self site is self-attention; kv_seq must be "
                    f"None or equal to q_seq, got kv_seq={kv_seq} "
                    f"q_seq={q_seq}"
                )
            return self.dit_self_attn(layer_idx, q_seq, stream=stream)
        if site == "dit_cross":
            if kv_seq is None:
                raise ValueError(
                    "dit_cross site is cross-attention against the "
                    "Qwen3 backbone features; kv_seq (= Se, the "
                    "encoder sequence length) must be supplied"
                )
            if layer_idx % 2 != 0:
                raise ValueError(
                    f"dit_cross layer_idx must be even, got "
                    f"layer_idx={layer_idx}"
                )
            block_idx = layer_idx // 2
            return self.dit_cross_attn(block_idx, q_seq, kv_seq,
                                       stream=stream)
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")


# Backwards-compatible alias — see the class docstring for the rename
# history. External plugins pinned to the old name continue to work.
TorchFlashAttnBackendGroot = RtxFlashAttnBackendGroot

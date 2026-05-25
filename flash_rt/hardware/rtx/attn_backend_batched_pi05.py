"""Batched (B=2) Pi0.5 RTX attention backend.

Subclass of :class:`flash_rt.hardware.rtx.attn_backend.RtxFlashAttnBackend`
that adds B=2 sample-batched Q/K/V/output buffers for use by
:class:`flash_rt.models.pi05.pipeline_rtx_batched.Pi05BatchedPipeline`.

The parent backend's B=1 buffers and methods are left untouched: all
existing pipelines (``Pi05Pipeline``, ``Pi05CFGPipeline``) continue to
use the original ``vision_attn`` / ``encoder_attn`` / ``decoder_attn``
entry points unchanged. The batched pipeline routes its attention calls
through the new ``*_batched`` methods added here, which read from the
new B=2 buffers (suffixed ``_b2``) and dispatch to the same FA2 wrapper
the parent uses.

Hardcoded B=2 for v0.1.0 — chosen specifically to fuse the cond + uncond
forwards of classifier-free guidance into a single batched pass
(arXiv:2511.14759 Appendix E). Wider batch sizes are not exposed today;
multi-robot RL rollout style B=N use cases are tracked separately as a
future workstream.
"""

from __future__ import annotations

import logging

from .attn_backend import RtxFlashAttnBackend

logger = logging.getLogger(__name__)

# Hardcoded sample-batch size. The buffers here are sized for exactly
# this many samples; the pipeline subclass that uses this backend asserts
# the same value at construction time so the two stay locked.
PI05_BATCH_SIZE = 2


class RtxFlashAttnBatchedBackendPi05(RtxFlashAttnBackend):
    """Pi0.5-specific RTX attention backend with B=2 sample batching.

    Adds these slots on top of the parent (all bf16/fp16 per the
    backend's selected dtype):

      vis_Q_b2 / vis_K_b2 / vis_V_b2 : (B*num_views, 256, 16, 72)
      enc_Q_b2                       : (B, encoder_seq_max, 8, 256)
      enc_K_b2 / enc_V_b2            : (num_layers, B, total_kv, 1, 256)
      dec_Q_b2                       : (B, chunk_size, 8, 256)

    plus the corresponding ``_b2_O`` / ``_b2_lse`` / splitkv accumulator
    buffers used by the FA2 wrapper. Vision is "B*num_views" in the
    leading dim (matching the parent's "num_views"-fold layout) so the
    same kernel call with a larger leading dim covers two samples.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch = self._torch
        # Pull the same dtype as the parent's vision Q so we stay
        # consistent across the whole stack.
        bf16 = self.vis_Q.dtype
        d = "cuda"
        B = PI05_BATCH_SIZE
        nv = self._num_views
        es_max = self._encoder_seq_max
        ds = self._chunk_size
        L = self._num_encoder_layers
        total_kv = es_max + ds

        # ── Inputs (mirrors parent's Q/K/V layouts, B-folded) ──
        # Vision: (B*nv, 256, 16, 72) — natural per-view-per-sample fold.
        self.vis_Q_b2 = torch.empty(B * nv, 256, 16, 72, dtype=bf16, device=d)
        self.vis_K_b2 = torch.empty(B * nv, 256, 16, 72, dtype=bf16, device=d)
        self.vis_V_b2 = torch.empty(B * nv, 256, 16, 72, dtype=bf16, device=d)

        # Encoder Q (reused across layers, B-folded)
        self.enc_Q_b2 = torch.empty(B, es_max, 8, 256, dtype=bf16, device=d)
        # Encoder K/V cache: per-layer × per-sample
        self.enc_K_b2 = torch.empty(L, B, total_kv, 1, 256, dtype=bf16, device=d)
        self.enc_V_b2 = torch.empty(L, B, total_kv, 1, 256, dtype=bf16, device=d)

        # Decoder Q (B-folded)
        self.dec_Q_b2 = torch.empty(B, ds, 8, 256, dtype=bf16, device=d)

        # Per-layer KV stride (bytes) for B=2
        # Per-layer slice in enc_K_b2 is (B, total_kv, 1, 256) elements.
        elem_size = self.enc_K_b2.element_size()
        self._enc_kv_layer_stride_bytes_b2 = (
            B * total_kv * 1 * 256 * elem_size)
        # Per-sample stride within a layer slice (bytes)
        self._enc_kv_sample_stride_bytes_b2 = total_kv * 1 * 256 * elem_size
        # Per-sample stride for enc_Q_b2 (bytes). The Q buffer is sized
        # for ``encoder_seq_max`` along the time dim (so a single
        # backend instance can serve any prompt length up to that
        # bound), but the per-sample qkv_split_rope loop in the
        # pipeline only writes ``encoder_seq_len`` rows per slot.
        # Callers must offset slot 1 by the FULL es_max-based stride,
        # NOT by ``encoder_seq_len`` rows; otherwise slot 1's writes
        # land in slot 0's tail (which is silently unused) and slot 1
        # itself stays uninitialised — FA2 then reads garbage Q for
        # batch index 1, wrecking slot 1's attention output. This was
        # the root cause of the slot-asymmetric drift seen during CFG
        # batched validation.
        self._enc_q_sample_stride_bytes_b2 = es_max * 8 * 256 * elem_size

        # ── FA2 output / LSE buffers (B-folded) ──
        # Vision (always uses fvk FA2 in pi05 path)
        self._vis_O72_b2 = torch.empty(B * nv, 256, 16, 72,
                                        dtype=bf16, device=d)
        self._vis_lse_b2 = torch.empty(B * nv, 16, 256,
                                        dtype=torch.float32, device=d)

        # Encoder output: (B, S, H, D)
        self._enc_O_b2 = torch.empty(B, es_max, 8, 256,
                                      dtype=bf16, device=d)
        enc_sq_r = ((es_max + 127) // 128) * 128
        self._enc_lse_b2 = torch.empty(B, 8, enc_sq_r,
                                        dtype=torch.float32, device=d)

        # Decoder output: (B, chunk, H, D)
        self._dec_O_b2 = torch.empty(B, ds, 8, 256,
                                      dtype=bf16, device=d)
        dec_sq_r = ((ds + 127) // 128) * 128
        self._dec_lse_b2 = torch.empty(B, 8, dec_sq_r,
                                        dtype=torch.float32, device=d)

        # ── SplitKV scratch (B-folded leading dim).
        # Reuses the parent's per-site num_splits choices.
        # SigLIP: 2 splits, leading B*nv
        _sig_splits = 2
        self._vis_lse_accum_b2 = torch.empty(_sig_splits, B * nv, 16, 256,
                                              dtype=torch.float32, device=d)
        self._vis_o_accum_b2 = torch.empty(_sig_splits, B * nv, 16, 256, 96,
                                            dtype=torch.float32, device=d)
        # Encoder
        _enc_splits = min(128, (es_max + 63) // 64)
        self._enc_lse_accum_b2 = torch.empty(_enc_splits, B, 8, es_max,
                                              dtype=torch.float32, device=d)
        self._enc_o_accum_b2 = torch.empty(_enc_splits, B, 8, es_max, 256,
                                            dtype=torch.float32, device=d)
        # Decoder
        _dec_splits = min(128, (total_kv + 63) // 64)
        self._dec_lse_accum_b2 = torch.empty(_dec_splits, B, 8, ds,
                                              dtype=torch.float32, device=d)
        self._dec_o_accum_b2 = torch.empty(_dec_splits, B, 8, ds, 256,
                                            dtype=torch.float32, device=d)

    # ──────────────────────────────────────────────────────────────
    # Pointer helpers (additive — parent's ``get_ptrs`` is untouched)
    # ──────────────────────────────────────────────────────────────

    def get_ptrs_b2(self) -> dict:
        """Pointer dict for the B=2 buffers (used by Pi05BatchedPipeline)."""
        return {
            "vis_Q": self.vis_Q_b2.data_ptr(),
            "vis_K": self.vis_K_b2.data_ptr(),
            "vis_V": self.vis_V_b2.data_ptr(),
            "enc_Q": self.enc_Q_b2.data_ptr(),
            "enc_K": self.enc_K_b2.data_ptr(),
            "enc_V": self.enc_V_b2.data_ptr(),
            "dec_Q": self.dec_Q_b2.data_ptr(),
            "enc_k_layer_stride_bytes": self._enc_kv_layer_stride_bytes_b2,
            "enc_v_layer_stride_bytes": self._enc_kv_layer_stride_bytes_b2,
            "enc_k_sample_stride_bytes": self._enc_kv_sample_stride_bytes_b2,
            "enc_v_sample_stride_bytes": self._enc_kv_sample_stride_bytes_b2,
            "enc_q_sample_stride_bytes": self._enc_q_sample_stride_bytes_b2,
        }

    @property
    def batch_size(self) -> int:
        """Hardcoded sample batch dimension (B=2 for the v0.1.0 CFG path)."""
        return PI05_BATCH_SIZE

    # ──────────────────────────────────────────────────────────────
    # Batched attention dispatch (additive — parent methods untouched)
    # ──────────────────────────────────────────────────────────────

    # ── Held references for ``.contiguous()`` outputs (Bug 7) ──
    #
    # The B=2 attention buffers are sliced ``[:, :seq]`` (vision uses
    # the full leading dim and is already contiguous; encoder /
    # decoder slice the time dim < ``es_max`` / ``ds`` respectively).
    # Slicing along time when ``B > 1`` produces a NON-contiguous view
    # — the next ``.contiguous()`` call therefore allocates a fresh
    # tensor and copies into it. If we then return only ``data_ptr()``
    # without holding a Python reference, the local goes out of scope
    # the moment the method returns; torch's caching allocator can
    # immediately recycle that memory and hand it back to a subsequent
    # small allocation. The caller (``transformer_encoder_batched`` /
    # ``transformer_decoder_batched``) is still trying to read from
    # the freed pointer for the attention output projection — slot 0
    # tends to survive (sits at the start of the freed block) but
    # slot 1 (at the tail) is the most likely region for the allocator
    # to reuse, producing the slot-asymmetric drift this fix addresses.
    #
    # Holding the most recent contiguous output per site keeps the
    # storage alive across method returns. The reference is
    # overwritten on each call (one tensor per site), so steady-state
    # memory cost is bounded.
    #
    # The parent ``RtxFlashAttnBackend`` does not hit this bug because
    # at B=1 the slice ``[:, :seq]`` IS contiguous (no inter-batch
    # gaps), so ``.contiguous()`` returns the same view backed by
    # ``self._enc_O`` / ``self._dec_O`` — both held by ``self``.

    def vision_attn_batched(self, stream: int = 0) -> int:
        """SigLIP self-attention over B*num_views = 2*num_views slots.

        SigLIP attention is per-view to begin with (the parent method
        already runs at batch=num_views); the batched variant simply
        increases the leading dim to ``B * num_views`` so a single
        FA2 launch covers both samples.
        """
        if not self._fa2_sites["siglip"]:
            raise RuntimeError(
                "vision_attn_batched requires fvk FA2 (FVK_RTX_FA2_SITES "
                "must include 'siglip'); legacy pip flash_attn fallback "
                "is not implemented for the batched path")
        self._call_fvk_fa2(
            self.vis_Q_b2, self.vis_K_b2, self.vis_V_b2,
            self._vis_O72_b2, self._vis_lse_b2, stream=stream,
            lse_accum=self._vis_lse_accum_b2,
            o_accum=self._vis_o_accum_b2)
        return self._vis_O72_b2.data_ptr()

    def encoder_attn_batched(self, layer_idx: int, seq: int,
                              stream: int = 0) -> int:
        """Gemma-2B encoder self-attention with B=2 sample batching."""
        if not self._fa2_sites["encoder"]:
            raise RuntimeError(
                "encoder_attn_batched requires fvk FA2 ('encoder' site)")
        # (B, seq, 8, 256) for Q; (B, seq, 1, 256) for K/V at this layer
        q = self.enc_Q_b2[:, :seq]
        k = self.enc_K_b2[layer_idx, :, :seq]
        v = self.enc_V_b2[layer_idx, :, :seq]
        o = self._enc_O_b2[:, :seq].contiguous()
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        self._call_fvk_fa2(q_c, k_c, v_c,
                           o, self._enc_lse_b2, stream=stream,
                           lse_accum=self._enc_lse_accum_b2,
                           o_accum=self._enc_o_accum_b2)
        if not hasattr(self, "_enc_qkv_o_refs_b2"):
            self._enc_qkv_o_refs_b2 = {}
        self._enc_qkv_o_refs_b2[layer_idx] = (q_c, k_c, v_c, o)
        return o.data_ptr()

    def decoder_attn_batched(self, layer_idx: int, enc_seq: int,
                              dec_seq: int, stream: int = 0) -> int:
        """Gemma-300M decoder cross-attention with B=2 sample batching."""
        if not self._fa2_sites["decoder"]:
            raise RuntimeError(
                "decoder_attn_batched requires fvk FA2 ('decoder' site)")
        total_kv = enc_seq + dec_seq
        q = self.dec_Q_b2[:, :dec_seq]
        k = self.enc_K_b2[layer_idx, :, :total_kv]
        v = self.enc_V_b2[layer_idx, :, :total_kv]
        o = self._dec_O_b2[:, :dec_seq].contiguous()
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        self._call_fvk_fa2(q_c, k_c, v_c,
                           o, self._dec_lse_b2, stream=stream,
                           lse_accum=self._dec_lse_accum_b2,
                           o_accum=self._dec_o_accum_b2)
        if not hasattr(self, "_dec_qkv_o_refs_b2"):
            self._dec_qkv_o_refs_b2 = []
        self._dec_qkv_o_refs_b2.append((q_c, k_c, v_c, o))
        return o.data_ptr()

    def run_batched(
        self,
        site: str,
        layer_idx: int,
        q_seq: int,
        *,
        kv_seq=None,
        stream: int = 0,
    ) -> int:
        """Batched dispatch (B=2) — sibling to :meth:`run`.

        Same contract as the parent's :meth:`run` but routes to the
        ``*_batched`` methods. Pi0-specific state-masked decoder mode
        is intentionally not supported here (Pi0.5 has no state token).
        """
        if site == "siglip":
            return self.vision_attn_batched(stream=stream)
        if site == "encoder":
            if kv_seq is not None and kv_seq != q_seq:
                raise ValueError(
                    "encoder site is self-attention; kv_seq must be "
                    "None or equal to q_seq")
            return self.encoder_attn_batched(layer_idx, q_seq, stream=stream)
        if site == "decoder":
            if kv_seq is None:
                raise ValueError(
                    "decoder site requires kv_seq (encoder cache + chunk)")
            dec_seq = q_seq
            enc_seq = kv_seq - dec_seq
            if enc_seq < 0:
                raise ValueError(
                    f"decoder kv_seq ({kv_seq}) must be >= q_seq ({q_seq})")
            return self.decoder_attn_batched(layer_idx, enc_seq, dec_seq,
                                              stream=stream)
        raise KeyError(f"unknown site {site!r}")

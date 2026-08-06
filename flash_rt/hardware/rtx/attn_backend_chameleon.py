"""FlashRT — Chameleon-7B VLM attention backend for Jetson Orin (SM87).

This backend owns a **real per-layer KV cache** so the LLM can decode
autoregressively (a prefill-only backend could instead share one K/V scratch
across all 32 layers, ``layer_stride = 0``).

Two load-bearing design points, both measured — see
``docs/chameleon7b_rtx_sm87.md`` §2.1 and §3.1:

1. **The K/V GEMMs write straight into the cache.** CUTLASS hard-wires the
   output row stride to ``N`` (``cutlass_sm80_int8_rowwise_fp16out.cu:169-171``),
   and a per-layer slab of ``[max_seq, num_kv_heads*head_dim]`` has exactly that
   row stride, so no staging buffer or copy is needed for either prefill
   (``M = S`` at the slab base) or decode (``M = 1`` at ``+ pos*row_stride``).
   ``qk_norm_rope_fused_fp16`` then does QK-LayerNorm + RoPE in place.

2. **Split-KV must be forced on with a biased ``num_sms``.** FA2's heuristic
   (``fa2_wrapper_causal.cu:41-43,152-158``) returns ``num_splits = 1`` whenever
   ``batch*num_q_heads*num_m_blocks >= 0.8 * (num_sms*2)``. Chameleon decode is
   ``1*32*1 = 32`` against ``0.8*32 = 25.6``, so at the true SM count split-KV
   silently does nothing (measured: bit-identical output, 1.05x). ``num_sms`` is
   a pure heuristic knob in this wrapper, so ``split_kv_bias`` multiplies it;
   bias 4 measured 204.9 -> 141.8 us (**1.44x**) at fp16-rounding-level delta.
   This is why the Qwen3-VL split-KV lever does not transfer as-is —
   that model has 16 Q heads and lands under the threshold naturally.

FA2 ``fwd_fp16_causal`` is **mandatory** for decode: its causal mask is
bottom-right aligned (``fa2_wrapper_causal.cu:126-138``) so ``q=1, kv=N``
attends all N keys. The cuBLAS ``attention_mha_causal_fp16`` fallback is **top-left** aligned (``softmax.cu:182-191`` masks with
``q = row % S_q``), so at ``S_q = 1`` only column 0 survives — it is silently
wrong rather than merely slow. This backend therefore raises instead of
degrading to it.
"""

from __future__ import annotations

import logging

import torch

from flash_rt.hardware.backend import AttentionBackendBase, AttentionSpec

logger = logging.getLogger(__name__)

SITE = "llm"

#: FA2's own cap on the split count (``fa2_wrapper_causal.cu`` passes 128).
_MAX_SPLITS = 128


class ChameleonAttnBackend(AttentionBackendBase):
    """Chameleon-7B self-attention with a per-layer FP16 KV cache.

    Owns every buffer it needs (KV cache, Q/O staging, softmax LSE, split-KV
    accumulators). Pointers are stable for the object's lifetime, so they are
    safe to bake into a captured CUDA graph — but the backend must be kept
    alive for as long as any such graph exists.
    """

    def __init__(self, spec: AttentionSpec, *, max_seq: int,
                 split_kv_bias: int = 4) -> None:
        super().__init__(spec)
        if set(spec.sites.keys()) != {SITE}:
            raise ValueError(
                f"ChameleonAttnBackend expects exactly the {SITE!r} site, "
                f"got {set(spec.sites.keys())}")

        s = spec.site(SITE)
        self.num_layers = int(s.num_layers)
        self.num_q_heads_ = int(s.num_q_heads)
        self.num_kv_heads_ = int(s.num_kv_heads)
        self.head_dim_ = int(s.head_dim)
        self.max_seq = int(max_seq)
        self.split_kv_bias = int(split_kv_bias)
        self.scale = float(self.head_dim_) ** -0.5

        if self.head_dim_ != 128:
            raise ValueError(
                f"head_dim must be 128 (FA2 fp16 causal aborts otherwise, "
                f"fa2_wrapper_causal.cu:235-243); got {self.head_dim_}")

        try:
            import flash_rt.flash_rt_fa2 as _fa2
        except ImportError as e:                                  # pragma: no cover
            raise RuntimeError(
                "Chameleon decode requires flash_rt_fa2 (the cuBLAS MHA "
                "fallback is top-left-aligned causal and therefore WRONG at "
                "q_seq=1). Rebuild with -DFLASHRT_ENABLE_FA2=ON.") from e
        if not hasattr(_fa2, "fwd_fp16_causal"):
            raise RuntimeError(
                "flash_rt_fa2 lacks fwd_fp16_causal; Chameleon decode cannot "
                "fall back to cuBLAS MHA (top-left-aligned mask is wrong at "
                "q_seq=1). Rebuild with -DFA2_DTYPES='fp16;bf16' "
                "-DFA2_HDIMS='128;256'.")
        self._fa2_fwd = _fa2.fwd_fp16_causal

        dev, fp16, fp32 = "cuda", torch.float16, torch.float32
        kv_row = self.num_kv_heads_ * self.head_dim_        # 4096 == GEMM N
        q_row = self.num_q_heads_ * self.head_dim_

        # Per-layer KV cache. The [max_seq, kv_row] slab per layer is exactly a
        # legal CUTLASS destination (row stride == N), which is what lets the
        # K/V GEMMs write into it directly.
        self.K_cache = torch.zeros(self.num_layers, self.max_seq, kv_row,
                                   dtype=fp16, device=dev)
        self.V_cache = torch.zeros(self.num_layers, self.max_seq, kv_row,
                                   dtype=fp16, device=dev)
        # Q staging (prefill writes S rows, decode row 0). O aliases Q, matching
        # the caller convention: the pipeline reads its result back in place.
        self.Q_buf = torch.zeros(self.max_seq, q_row, dtype=fp16, device=dev)

        lse_rows = ((self.max_seq + 127) // 128) * 128
        self.lse_buf = torch.zeros(1, self.num_q_heads_, lse_rows,
                                   dtype=fp32, device=dev)
        # Split-KV accumulators, sized for the decode case (seqlen_q == 1).
        # Empty splits are self-initialised by the kernel, so no pre-fill.
        self.lse_accum = torch.zeros(_MAX_SPLITS, 1, self.num_q_heads_, 1,
                                     dtype=fp32, device=dev)
        self.o_accum = torch.zeros(_MAX_SPLITS, 1, self.num_q_heads_, 1,
                                   self.head_dim_, dtype=fp32, device=dev)

        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count

        kv_gb = 2 * self.K_cache.numel() * 2 / 2 ** 30
        logger.info(
            "ChameleonAttnBackend: L=%d %dQ/%dKV hd=%d max_seq=%d "
            "KV=%.2f GB split_kv_bias=%d (num_sms %d->%d)",
            self.num_layers, self.num_q_heads_, self.num_kv_heads_,
            self.head_dim_, self.max_seq, kv_gb, self.split_kv_bias,
            self._num_sms, self._num_sms * self.split_kv_bias)

    # ------------------------------------------------------------------
    # Layout / pointers
    # ------------------------------------------------------------------

    @property
    def kv_layer_stride_bytes(self) -> int:
        return self.max_seq * self.num_kv_heads_ * self.head_dim_ * 2

    @property
    def kv_row_stride_bytes(self) -> int:
        return self.num_kv_heads_ * self.head_dim_ * 2

    def _check_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict:
        """Prefill slots: Q/O staging plus the base of this layer's KV slab."""
        if site != SITE:
            raise KeyError(f"unknown site {site!r}")
        self._check_layer(layer_idx)
        off = layer_idx * self.kv_layer_stride_bytes
        q = self.Q_buf.data_ptr()
        return {"Q": q, "O": q,
                "K": self.K_cache.data_ptr() + off,
                "V": self.V_cache.data_ptr() + off}

    def kv_row_ptrs(self, layer_idx: int, pos: int) -> tuple:
        """Decode slots: the single KV row for absolute position ``pos``."""
        self._check_layer(layer_idx)
        if not 0 <= pos < self.max_seq:
            raise IndexError(
                f"pos {pos} out of range [0, max_seq={self.max_seq}); raise "
                f"max_seq at construction time")
        off = layer_idx * self.kv_layer_stride_bytes + pos * self.kv_row_stride_bytes
        return (self.K_cache.data_ptr() + off, self.V_cache.data_ptr() + off)

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------

    def _fa2(self, layer_idx: int, q_seq: int, kv_seq: int, *,
             use_split_kv: bool, stream: int) -> int:
        qr = self.num_q_heads_ * self.head_dim_
        kr = self.num_kv_heads_ * self.head_dim_
        off = layer_idx * self.kv_layer_stride_bytes
        q = self.Q_buf.data_ptr()
        # num_sms == 0 disables the split-KV heuristic entirely; a biased value
        # is the only way to reach num_splits > 1 at 32 Q heads (see module doc).
        num_sms = self._num_sms * self.split_kv_bias if use_split_kv else 0
        self._fa2_fwd(
            q, self.K_cache.data_ptr() + off, self.V_cache.data_ptr() + off, q,
            self.lse_buf.data_ptr(),
            self.lse_accum.data_ptr() if use_split_kv else 0,
            self.o_accum.data_ptr() if use_split_kv else 0,
            batch=1, seqlen_q=q_seq, seqlen_k=kv_seq,
            num_heads_q=self.num_q_heads_, num_heads_kv=self.num_kv_heads_,
            head_dim=self.head_dim_,
            q_strides=(q_seq * qr, qr, self.head_dim_),
            k_strides=(kv_seq * kr, kr, self.head_dim_),
            v_strides=(kv_seq * kr, kr, self.head_dim_),
            o_strides=(q_seq * qr, qr, self.head_dim_),
            softmax_scale=self.scale, num_sms=num_sms, stream=stream)
        return q

    def run_prefill(self, layer_idx: int, seq_len: int, *, stream: int = 0) -> int:
        """Square causal attention over ``seq_len`` tokens. Result lands in Q_buf."""
        self._check_layer(layer_idx)
        if seq_len > self.max_seq:
            raise ValueError(f"seq_len {seq_len} > max_seq {self.max_seq}")
        # Prefill already fills every SM (num_m_blocks = ceil(S/64)), so
        # splitting KV would only add a combine pass.
        return self._fa2(layer_idx, seq_len, seq_len,
                         use_split_kv=False, stream=stream)

    def run_decode(self, layer_idx: int, pos: int, *, stream: int = 0) -> int:
        """One query row attending keys ``[0, pos]``. Result lands in Q_buf row 0."""
        self._check_layer(layer_idx)
        return self._fa2(layer_idx, 1, pos + 1,
                         use_split_kv=self.split_kv_bias > 1, stream=stream)

    # No generic ``run(site, layer_idx, q_seq, ...)`` on purpose. Prefill and
    # decode differ in more than q_seq here (split-KV on/off, and the caller
    # must supply the absolute KV position rather than a length), so a shim that
    # inferred the mode from ``q_seq == 1`` would be an untested footgun. The
    # inherited ``AttentionBackendBase.run`` raises NotImplementedError, which
    # is the behaviour we want if a generic caller ever appears.

    def reset_cache(self) -> None:
        self.K_cache.zero_()
        self.V_cache.zero_()


def make_chameleon_attention_spec(*, num_layers: int, num_q_heads: int,
                                  num_kv_heads: int, head_dim: int,
                                  max_seq: int) -> AttentionSpec:
    spec = AttentionSpec()
    spec.add_site(SITE, num_layers=num_layers, num_q_heads=num_q_heads,
                  num_kv_heads=num_kv_heads, head_dim=head_dim,
                  max_q_seq=max_seq, max_kv_seq=max_seq, causal=True)
    return spec


__all__ = ["ChameleonAttnBackend", "make_chameleon_attention_spec", "SITE"]

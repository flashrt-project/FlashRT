"""FlashRT -- RTX Nex-N2-mini full-attention backend.

Nex-N2-mini (qwen3_5_moe) has two attention regimes per decoder layer:

  * ``full_attention`` (10 layers): GQA 16Q / 2KV, head_dim=256, causal
    self-attention with a per-layer KV cache of shape
    (max_seq, 2, 256) bf16. Q carries partial RoPE (first 64 dims) and
    the attention output is post-multiplied by a sigmoid gate; both are
    applied by the pipeline outside this backend.
  * ``linear_attention`` (30 layers): Gated DeltaNet, runs through the
    fvk recurrent kernel with its own (state, conv) caches. Does NOT
    flow through this backend.

This backend exposes a single site ``"full"`` with 10 layers. It mirrors
the surface of :class:`RtxFlashAttnBackendQwen36`, specialised to the
Nex-N2 GQA shape.

Decode contract (q_seq=1):
  * Pipeline writes the new token's K/V into ``K_cache[layer, cur_pos]``
    / ``V_cache[layer, cur_pos]`` via ``get_slot_ptrs(layer)``.
  * Pipeline writes Q for the new token into ``Q_buf[:, :1]``.
  * ``run("full", layer_idx, q_seq=1, kv_seq=cur_pos+1)`` runs the
    vendored FA2 fwd_bf16 over (1, 1, 16, 256) Q against
    (1, kv_seq, 2, 256) K/V, writing output to a fixed ``O_buf`` pointer.
"""

from __future__ import annotations


class RtxFlashAttnBackendNexn2:
    """Nex-N2-mini full-attention backend (BF16 attention math).

    Built atop the vendored FA2 (``flash_rt.flash_rt_fa2.fwd_bf16``).
    KV cache lives in this backend so the pipeline can write new tokens
    via ``get_slot_ptrs(layer)['K'] + cur_pos * row_stride`` from a fused
    RoPE+write kernel without going through Python.

    The 30 linear-attn layers are NOT addressed here -- the pipeline
    routes them directly to the fvk Gated DeltaNet kernels with their own
    state cache (managed at the pipeline level).
    """

    SITES = ("full",)
    NUM_FULL_LAYERS = 10
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 2
    HEAD_DIM = 256

    def __init__(self, max_seq: int, max_q_seq: int = 1, dtype=None):
        import os
        import torch

        self._torch = torch
        bf16 = dtype if dtype is not None else torch.bfloat16
        d = "cuda"

        self._max_seq = int(max_seq)
        self._max_q_seq = int(max_q_seq)
        self._dtype = bf16

        # Per-layer KV cache: (NUM_FULL_LAYERS, max_seq, NUM_KV_HEADS, HEAD_DIM)
        # bf16. The pipeline owns the per-layer cur_pos cursor; this
        # backend only exposes pointers + strides.
        self.K_cache = torch.empty(
            self.NUM_FULL_LAYERS, self._max_seq,
            self.NUM_KV_HEADS, self.HEAD_DIM, dtype=bf16, device=d,
        )
        self.V_cache = torch.empty_like(self.K_cache)

        # Q scratch (single-token decode by default; sized to max_q_seq
        # in case a future step batches multiple new tokens).
        self.Q_buf = torch.empty(
            1, self._max_q_seq, self.NUM_Q_HEADS, self.HEAD_DIM,
            dtype=bf16, device=d,
        )
        self.O_buf = torch.empty_like(self.Q_buf)

        # softmax_lse: fp32 (B, Hq, Sq_rounded). FA2 requires Sq rounded
        # up to a multiple of 128.
        sq_rounded = ((self._max_q_seq + 127) // 128) * 128
        self.lse_buf = torch.empty(
            1, self.NUM_Q_HEADS, sq_rounded,
            dtype=torch.float32, device=d,
        )

        # SplitKV scratch. head_dim=256 -> block_n=64 in FA2; with
        # max_seq tokens we get up to ceil(max_seq/64) splits, capped 128.
        n_splits = min(128, (self._max_seq + 63) // 64)
        self._n_splits = n_splits
        self.lse_accum = torch.empty(
            n_splits, 1, self.NUM_Q_HEADS, self._max_q_seq,
            dtype=torch.float32, device=d,
        )
        self.o_accum = torch.empty(
            n_splits, 1, self.NUM_Q_HEADS, self._max_q_seq, self.HEAD_DIM,
            dtype=torch.float32, device=d,
        )

        from flash_rt import flash_rt_fa2 as _fa2
        self._fa2 = _fa2
        self._fa2_fwd = _fa2.fwd_bf16
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

        # Optional bisect knob: env FVK_NEXN2_FA2=0 routes through pip
        # flash_attn_func instead (single-call fallback for debugging cos
        # drifts). Not used in production.
        self._use_fvk_fa2 = os.environ.get("FVK_NEXN2_FA2", "1") == "1"
        if not self._use_fvk_fa2:
            from flash_attn import flash_attn_func
            self._flash_attn_func = flash_attn_func

    # ── Layer cache pointer math ──

    @property
    def kv_layer_stride_bytes(self) -> int:
        return self._max_seq * self.NUM_KV_HEADS * self.HEAD_DIM * 2

    @property
    def kv_row_stride_bytes(self) -> int:
        return self.NUM_KV_HEADS * self.HEAD_DIM * 2

    # ── AttentionBackend protocol ──

    def sites(self) -> tuple[str, ...]:
        return self.SITES

    def _check_site(self, site: str) -> None:
        if site != "full":
            raise KeyError(
                f"nexn2 backend only knows site='full', got {site!r}")

    def head_dim(self, site: str) -> int:
        self._check_site(site)
        return self.HEAD_DIM

    def num_q_heads(self, site: str) -> int:
        self._check_site(site)
        return self.NUM_Q_HEADS

    def num_kv_heads(self, site: str) -> int:
        self._check_site(site)
        return self.NUM_KV_HEADS

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict:
        self._check_site(site)
        layer_off_bytes = layer_idx * self.kv_layer_stride_bytes
        return {
            "Q": self.Q_buf.data_ptr(),
            "K": self.K_cache.data_ptr() + layer_off_bytes,
            "V": self.V_cache.data_ptr() + layer_off_bytes,
            "kv_layer_stride_bytes": self.kv_layer_stride_bytes,
            "kv_row_stride_bytes": self.kv_row_stride_bytes,
        }

    def reset_cache(self) -> None:
        """Zero-fill caches between independent prompts (data-side reset)."""
        self.K_cache.zero_()
        self.V_cache.zero_()

    # ── Attention call ──

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: int, stream: int = 0,
            softmax_scale: float | None = None) -> int:
        """Run FA2 over Q[:q_seq] against K/V[layer_idx, :kv_seq].

        Returns the device pointer of the output tensor (always
        ``self.O_buf.data_ptr()`` for the fvk-FA2 path -- stable across
        CUDA graph replays).
        """
        self._check_site(site)
        if not (1 <= q_seq <= self._max_q_seq):
            raise ValueError(f"q_seq={q_seq} out of range [1, {self._max_q_seq}]")
        if not (1 <= kv_seq <= self._max_seq):
            raise ValueError(f"kv_seq={kv_seq} out of range [1, {self._max_seq}]")

        q = self.Q_buf[:, :q_seq]                           # (1, q_seq, 16, 256)
        k = self.K_cache[layer_idx:layer_idx + 1, :kv_seq]  # (1, kv_seq, 2, 256)
        v = self.V_cache[layer_idx:layer_idx + 1, :kv_seq]  # (1, kv_seq, 2, 256)
        o = self.O_buf[:, :q_seq]                           # (1, q_seq, 16, 256)

        if softmax_scale is None:
            softmax_scale = 1.0 / (self.HEAD_DIM ** 0.5)

        if self._use_fvk_fa2:
            self._fa2_fwd(
                Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
                O=o.data_ptr(), softmax_lse=self.lse_buf.data_ptr(),
                softmax_lse_accum=self.lse_accum.data_ptr(),
                o_accum=self.o_accum.data_ptr(),
                batch=1, seqlen_q=q_seq, seqlen_k=kv_seq,
                num_heads_q=self.NUM_Q_HEADS,
                num_heads_kv=self.NUM_KV_HEADS,
                head_dim=self.HEAD_DIM,
                q_strides=(q.stride(0), q.stride(1), q.stride(2)),
                k_strides=(k.stride(0), k.stride(1), k.stride(2)),
                v_strides=(v.stride(0), v.stride(1), v.stride(2)),
                o_strides=(o.stride(0), o.stride(1), o.stride(2)),
                softmax_scale=softmax_scale,
                num_sms=self._num_sms,
                stream=stream,
            )
            return o.data_ptr()

        out = self._flash_attn_func(q, k, v, causal=False,
                                    softmax_scale=softmax_scale)
        o.copy_(out)
        return o.data_ptr()


def make_nexn2_attention_spec(*, max_seq: int, max_q_seq: int = 1) -> dict:
    """Static metadata describing Nex-N2-mini's full-attn + linear-attn sites."""
    return {
        "sites": [
            {
                "name": "full",
                "layer_count": RtxFlashAttnBackendNexn2.NUM_FULL_LAYERS,
                "num_q_heads": RtxFlashAttnBackendNexn2.NUM_Q_HEADS,
                "num_kv_heads": RtxFlashAttnBackendNexn2.NUM_KV_HEADS,
                "head_dim": RtxFlashAttnBackendNexn2.HEAD_DIM,
                "max_q_seq": int(max_q_seq),
                "max_kv_seq": int(max_seq),
                "kernel": "fvk_fa2_bf16",
            },
        ],
        "linear_attn": {
            "layer_count": 30,
            "num_k_heads": 16,
            "num_v_heads": 32,
            "head_dim": 128,
            "conv_kernel": 4,
            "kernel": "fvk_gated_deltanet_recurrent_bf16",
        },
    }

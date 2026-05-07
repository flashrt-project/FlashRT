"""FlashRT — RTX Qwen3 (plain dense) full-attention backend.

Qwen3-8B's 36 layers are all standard causal GQA self-attention with
shape::

    Q  : (1, max_q_seq, NUM_Q_HEADS=32, HEAD_DIM=128)
    KV : (NUM_LAYERS=36, max_seq, NUM_KV_HEADS=8, HEAD_DIM=128)

Backed by the vendored FlashAttention-2 ``flash_rt_fa2.fwd_bf16``.
The class mirrors :class:`RtxFlashAttnBackendQwen36` with the lin-attn
half removed and the GQA shape updated for plain Qwen3.

Decode contract (q_seq=1):

  * Pipeline writes the new token's K/V into ``K_cache[layer, cur_pos]``
    / ``V_cache[layer, cur_pos]`` — pointer math via
    ``get_slot_ptrs(layer)``, cur_pos owned by the pipeline.
  * Pipeline writes Q for the new token into ``Q_buf[:, :1]``.
  * ``run("full", layer_idx, q_seq=1, kv_seq=cur_pos+1)`` runs FA2
    over (1, 1, 32, 128) Q vs (1, kv_seq, 8, 128) K/V, writing output
    to ``O_buf`` and returning that pointer for the next GEMM.

Prefill (S=N) contract:

  * Pipeline stages Q[:, :S] into ``Q_buf[:, :S]`` (S ≤ max_q_seq).
  * Pipeline writes K/V[layer, :S] into the cache in one shot.
  * ``run("full", layer_idx, q_seq=S, kv_seq=S, causal=True)`` runs
    FA2 in causal mode for the whole prompt.

The ``max_q_seq`` constructor arg sizes Q/O/lse buffers; for the Qwen3
prefill path we'll use a few discrete buckets (32 / 64 / ... / 1024).
"""
from __future__ import annotations


class RtxFlashAttnBackendQwen3:
    """Qwen3 full-attention backend (NVFP4 weights, BF16 attention math).

    All 36 layers route through this one backend; there is no lin-attn
    site (unlike the Qwen3.6-27B sibling).
    """

    SITES = ("full",)
    NUM_FULL_LAYERS = 36
    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 8                # GQA 4:1
    HEAD_DIM = 128

    def __init__(self, max_seq: int, max_q_seq: int = 1, dtype=None):
        import os

        import torch

        self._torch = torch
        bf16 = dtype if dtype is not None else torch.bfloat16
        d = "cuda"

        self._max_seq = int(max_seq)
        self._max_q_seq = int(max_q_seq)
        self._dtype = bf16

        # Per-layer KV cache:
        #   (NUM_FULL_LAYERS, max_seq, NUM_KV_HEADS, HEAD_DIM) bf16
        # 36 × max_seq × 8 × 128 × 2 bytes per K (and same for V).
        # At max_seq=2048: 36 × 2048 × 8 × 128 × 2 × 2 = 384 MiB total.
        # At max_seq=40960 (max_pos): ~7.7 GiB total — borderline; the
        # frontend will surface a max_seq knob.
        self.K_cache = torch.empty(
            self.NUM_FULL_LAYERS, self._max_seq,
            self.NUM_KV_HEADS, self.HEAD_DIM, dtype=bf16, device=d,
        )
        self.V_cache = torch.empty_like(self.K_cache)

        # Q / O scratch.
        self.Q_buf = torch.empty(
            1, self._max_q_seq, self.NUM_Q_HEADS, self.HEAD_DIM,
            dtype=bf16, device=d,
        )
        self.O_buf = torch.empty_like(self.Q_buf)

        # softmax_lse: fp32 (1, Hq, Sq_rounded) — FA2 rounds to mult of 128.
        sq_rounded = ((self._max_q_seq + 127) // 128) * 128
        self.lse_buf = torch.empty(
            1, self.NUM_Q_HEADS, sq_rounded,
            dtype=torch.float32, device=d,
        )

        # SplitKV scratch. head_dim=128 → block_n=64 in FA2; cap splits 128.
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

        # FA2 module. Fail loud if missing — Qwen3 path requires the
        # vendored fp16/bf16 FA2 build.
        from flash_rt import flash_rt_fa2 as _fa2
        self._fa2 = _fa2
        self._fa2_fwd = _fa2.fwd_bf16
        # Causal binding for q_seq>1 prefill. Built only for
        # (bf16, head_dim=128); other shapes fall back to the SDPA path.
        self._fa2_fwd_causal = getattr(_fa2, 'fwd_bf16_causal', None)
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

        # Optional debug bisect knob (mirror qwen36 sibling).
        self._use_fvk_fa2 = os.environ.get("FVK_QWEN3_FA2", "1") == "1"
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

    def head_dim(self, site: str) -> int:
        if site != "full":
            raise KeyError(
                f"qwen3 backend only knows site='full', got {site!r}")
        return self.HEAD_DIM

    def num_q_heads(self, site: str) -> int:
        if site != "full":
            raise KeyError(
                f"qwen3 backend only knows site='full', got {site!r}")
        return self.NUM_Q_HEADS

    def num_kv_heads(self, site: str) -> int:
        if site != "full":
            raise KeyError(
                f"qwen3 backend only knows site='full', got {site!r}")
        return self.NUM_KV_HEADS

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict:
        if site != "full":
            raise KeyError(
                f"qwen3 backend only knows site='full', got {site!r}")
        layer_off_bytes = layer_idx * self.kv_layer_stride_bytes
        return {
            "Q": self.Q_buf.data_ptr(),
            "K": self.K_cache.data_ptr() + layer_off_bytes,
            "V": self.V_cache.data_ptr() + layer_off_bytes,
            "kv_layer_stride_bytes": self.kv_layer_stride_bytes,
            "kv_row_stride_bytes": self.kv_row_stride_bytes,
        }

    def reset_cache(self) -> None:
        self.K_cache.zero_()
        self.V_cache.zero_()

    # ── Attention call ──

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: int, stream: int = 0,
            softmax_scale: float | None = None,
            causal: bool = True) -> int:
        """FA2 over Q[:q_seq] vs K/V[layer_idx, :kv_seq].

        Returns ``self.O_buf.data_ptr()`` (graph-replay-stable).
        ``causal=True`` is the right default: prefill uses causal masking
        and decode (q_seq=1) is unaffected by the flag.
        """
        if site != "full":
            raise KeyError(
                f"qwen3 backend only knows site='full', got {site!r}")
        if not (1 <= q_seq <= self._max_q_seq):
            raise ValueError(
                f"q_seq={q_seq} out of range [1, {self._max_q_seq}]")
        if not (1 <= kv_seq <= self._max_seq):
            raise ValueError(
                f"kv_seq={kv_seq} out of range [1, {self._max_seq}]")

        q = self.Q_buf[:, :q_seq]
        k = self.K_cache[layer_idx:layer_idx + 1, :kv_seq]
        v = self.V_cache[layer_idx:layer_idx + 1, :kv_seq]
        o = self.O_buf[:, :q_seq]

        if softmax_scale is None:
            softmax_scale = 1.0 / (self.HEAD_DIM ** 0.5)

        # Decode (q_seq=1): single-query, causal vs non-causal identical,
        # so use the existing non-causal fwd_bf16 (template Is_causal=false).
        # Prefill (q_seq>1, causal=True): use the fwd_bf16_causal
        # binding (template Is_causal=true). Both paths handle GQA
        # natively via FA2's h_h_k_ratio — no repeat_interleave / SDPA
        # detour.
        if self._use_fvk_fa2 and q_seq == 1:
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

        if (self._use_fvk_fa2 and causal and q_seq > 1
                and self._fa2_fwd_causal is not None):
            # Prefill causal via FA2 native (head_dim=128 only).
            # num_splits=1 forced (softmax_lse_accum=0, o_accum=0): the
            # splitkv heuristic is tuned for decode, and at S>=64 with
            # 36 layers there's enough work per-launch to fill SMs without
            # split. Keeps the path simple + deterministic.
            self._fa2_fwd_causal(
                Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
                O=o.data_ptr(), softmax_lse=self.lse_buf.data_ptr(),
                softmax_lse_accum=0, o_accum=0,
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

        if causal and q_seq > 1:
            # SDPA fallback (only hit when fwd_bf16_causal is not built or
            # head_dim != 128). Kept for forward compatibility.
            tt = self._torch
            q_h = q.transpose(1, 2).contiguous()
            k_h = k.transpose(1, 2)
            v_h = v.transpose(1, 2)
            ratio = self.NUM_Q_HEADS // self.NUM_KV_HEADS
            k_h = k_h.repeat_interleave(ratio, dim=1).contiguous()
            v_h = v_h.repeat_interleave(ratio, dim=1).contiguous()
            out_h = tt.nn.functional.scaled_dot_product_attention(
                q_h, k_h, v_h,
                attn_mask=None, is_causal=True, scale=softmax_scale,
            )
            o.copy_(out_h.transpose(1, 2))
            return o.data_ptr()

        # pip flash_attn fallback (debug only).
        out = self._flash_attn_func(
            q, k, v, causal=causal, softmax_scale=softmax_scale)
        o.copy_(out)
        return o.data_ptr()


def make_qwen3_8b_attention_spec(*, max_seq: int, max_q_seq: int = 1) -> dict:
    """Static metadata describing Qwen3-8B's full-attn site."""
    return {
        "sites": [
            {
                "name": "full",
                "layer_count": RtxFlashAttnBackendQwen3.NUM_FULL_LAYERS,
                "num_q_heads": RtxFlashAttnBackendQwen3.NUM_Q_HEADS,
                "num_kv_heads": RtxFlashAttnBackendQwen3.NUM_KV_HEADS,
                "head_dim": RtxFlashAttnBackendQwen3.HEAD_DIM,
                "max_q_seq": int(max_q_seq),
                "max_kv_seq": int(max_seq),
                "kernel": "fvk_fa2_bf16",
            },
        ],
    }

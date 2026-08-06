"""FlashRT — Thor (SM110) Qwen3 dense full-attention backend.

Thor analogue of ``flash_rt.hardware.rtx.attn_backend_qwen3``. It keeps exactly
the buffer surface the Qwen3-VL frontends rely on (``Q_buf`` / ``K_cache`` /
``V_cache`` / ``O_buf`` + ``get_slot_ptrs`` + ``kv_layer_stride_bytes`` /
``kv_row_stride_bytes`` + ``reset_cache`` + ``run``), so the frontend's KV-write
pointer math and attention call stay arch-independent.

The one thing that differs from the RTX backend is the attention kernel: the
vendored FA2 is not built on sm_110, so ``run`` uses a GQA
``scaled_dot_product_attention``. SDPA is numerically faithful here (BF16 math,
fp32 softmax) and is the correctness baseline for the Thor bring-up. A later
phase can swap in the Thor CUTLASS causal FMHA without changing this surface.

Cache layout (identical to the RTX backend)::

    Q  : (1, max_q_seq, NUM_Q_HEADS,  HEAD_DIM) bf16
    KV : (NUM_LAYERS,   max_seq,      NUM_KV_HEADS, HEAD_DIM) bf16

Decode (q_seq=1): the frontend writes the new token's K/V into
``K_cache[L, cur_pos]`` / ``V_cache[L, cur_pos]`` and Q into ``Q_buf[:, :1]``,
then calls ``run("full", L, q_seq=1, kv_seq=cur_pos + 1)``.

Prefill (q_seq=S): the frontend writes K/V[L, :S] and Q[:, :S], then calls
``run("full", L, q_seq=S, kv_seq=S, causal=True)``.
"""
from __future__ import annotations


class ThorAttnBackendQwen3:
    """Qwen3 dense GQA full-attention backend for Jetson Thor (SM110).

    BF16 attention math via SDPA. Mirrors the RTX backend's buffer surface and
    per-model dim kwargs (num_layers / num_q_heads / num_kv_heads / head_dim)
    so the 2B (28/16/8/128) and larger variants share one backend.
    """

    SITES = ("full",)
    NUM_FULL_LAYERS = 28
    NUM_Q_HEADS = 16
    NUM_KV_HEADS = 8                # GQA 2:1 for the 2B variant
    HEAD_DIM = 128

    def __init__(self, max_seq: int, max_q_seq: int = 1, dtype=None,
                 *, num_layers: int | None = None,
                 num_q_heads: int | None = None,
                 num_kv_heads: int | None = None,
                 head_dim: int | None = None,
                 device: str = "cuda") -> None:
        import torch

        self._torch = torch
        bf16 = dtype if dtype is not None else torch.bfloat16
        d = torch.device(device)

        self.NUM_FULL_LAYERS = (
            int(num_layers) if num_layers is not None
            else type(self).NUM_FULL_LAYERS)
        self.NUM_Q_HEADS = (
            int(num_q_heads) if num_q_heads is not None
            else type(self).NUM_Q_HEADS)
        self.NUM_KV_HEADS = (
            int(num_kv_heads) if num_kv_heads is not None
            else type(self).NUM_KV_HEADS)
        self.HEAD_DIM = (
            int(head_dim) if head_dim is not None
            else type(self).HEAD_DIM)
        if self.NUM_Q_HEADS % self.NUM_KV_HEADS != 0:
            raise ValueError(
                f"num_q_heads ({self.NUM_Q_HEADS}) must be a multiple of "
                f"num_kv_heads ({self.NUM_KV_HEADS})")

        self._max_seq = int(max_seq)
        self._max_q_seq = int(max_q_seq)
        self._dtype = bf16

        # Per-layer KV cache: (NUM_FULL_LAYERS, max_seq, NUM_KV_HEADS, HEAD_DIM).
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

        # Probe native-GQA SDPA support here rather than on the first run():
        # the probe itself launches an SDPA, which must not happen inside a
        # CUDA Graph capture. When unsupported, run() expands K/V instead.
        self._sdpa_gqa = True
        if self.NUM_Q_HEADS > self.NUM_KV_HEADS:
            probe = torch.empty(
                1, self.NUM_Q_HEADS, 1, self.HEAD_DIM, dtype=bf16, device=d)
            probe_kv = torch.empty(
                1, self.NUM_KV_HEADS, 1, self.HEAD_DIM, dtype=bf16, device=d)
            try:
                torch.nn.functional.scaled_dot_product_attention(
                    probe, probe_kv, probe_kv, enable_gqa=True)
            except TypeError:
                self._sdpa_gqa = False

    # ── Layer cache pointer math ──

    @property
    def kv_layer_stride_bytes(self) -> int:
        return self._max_seq * self.kv_row_stride_bytes

    @property
    def kv_row_stride_bytes(self) -> int:
        return (self.NUM_KV_HEADS * self.HEAD_DIM
                * self.K_cache.element_size())

    # ── AttentionBackend protocol ──

    def sites(self) -> tuple[str, ...]:
        return self.SITES

    def _check_site(self, site: str) -> None:
        if site != "full":
            raise KeyError(
                f"qwen3 thor backend only knows site='full', got {site!r}")

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
        self.K_cache.zero_()
        self.V_cache.zero_()

    # ── Attention call ──

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: int, stream: int = 0,
            softmax_scale: float | None = None,
            causal: bool = True) -> int:
        """GQA SDPA over Q[:q_seq] vs K/V[layer_idx, :kv_seq].

        Writes into ``O_buf[:, :q_seq]`` and returns its pointer. ``causal`` is
        honoured for q_seq>1 prefill; q_seq==1 decode is causal-invariant.
        """
        self._check_site(site)
        if not (1 <= q_seq <= self._max_q_seq):
            raise ValueError(
                f"q_seq={q_seq} out of range [1, {self._max_q_seq}]")
        if not (1 <= kv_seq <= self._max_seq):
            raise ValueError(
                f"kv_seq={kv_seq} out of range [1, {self._max_seq}]")
        if causal and q_seq > 1 and kv_seq != q_seq:
            # A q-block that ends mid-sequence needs a bottom-right-aligned
            # mask, but SDPA's is_causal is top-left-aligned. Building the mask
            # here would allocate on the hot path, so refuse instead of
            # silently attending to the wrong positions.
            raise NotImplementedError(
                "Thor Qwen3 attention supports causal prefill only with "
                f"kv_seq == q_seq; got q_seq={q_seq} kv_seq={kv_seq}")

        tt = self._torch
        if softmax_scale is None:
            softmax_scale = 1.0 / (self.HEAD_DIM ** 0.5)

        # (1, q_seq, Hq, hd) -> (1, Hq, q_seq, hd)
        q = self.Q_buf[:, :q_seq].transpose(1, 2)
        # (1, kv_seq, Hkv, hd) -> (1, Hkv, kv_seq, hd)
        k = self.K_cache[layer_idx:layer_idx + 1, :kv_seq].transpose(1, 2)
        v = self.V_cache[layer_idx:layer_idx + 1, :kv_seq].transpose(1, 2)
        ratio = self.NUM_Q_HEADS // self.NUM_KV_HEADS
        is_causal = bool(causal and q_seq > 1)
        sdpa = tt.nn.functional.scaled_dot_product_attention

        if ratio > 1 and self._sdpa_gqa:
            out_h = sdpa(q.contiguous(), k, v, attn_mask=None,
                         is_causal=is_causal, scale=softmax_scale,
                         enable_gqa=True)
        else:
            if ratio > 1:
                # Expansion allocates; on this path it lands in the graph's
                # private pool during capture.
                k = k.repeat_interleave(ratio, dim=1)
                v = v.repeat_interleave(ratio, dim=1)
            out_h = sdpa(q.contiguous(), k.contiguous(), v.contiguous(),
                         attn_mask=None, is_causal=is_causal,
                         scale=softmax_scale)
        # (1, Hq, q_seq, hd) -> (1, q_seq, Hq, hd)
        self.O_buf[:, :q_seq].copy_(out_h.transpose(1, 2))
        return self.O_buf.data_ptr()


def make_qwen3_thor_attention_spec(*, num_layers: int, num_q_heads: int,
                                   num_kv_heads: int, head_dim: int,
                                   max_seq: int, max_q_seq: int = 1) -> dict:
    """Static metadata describing the Thor Qwen3 full-attn site."""
    return {
        "sites": [
            {
                "name": "full",
                "layer_count": int(num_layers),
                "num_q_heads": int(num_q_heads),
                "num_kv_heads": int(num_kv_heads),
                "head_dim": int(head_dim),
                "max_q_seq": int(max_q_seq),
                "max_kv_seq": int(max_seq),
                "kernel": "sdpa_bf16",
            },
        ],
    }

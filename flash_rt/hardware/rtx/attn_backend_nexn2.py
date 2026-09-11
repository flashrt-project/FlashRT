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


def _load_optional_fa2():
    """Load FA2, treating only this module's absence as optional."""
    import importlib

    try:
        return importlib.import_module("flash_rt.flash_rt_fa2")
    except ModuleNotFoundError as exc:
        if exc.name != "flash_rt.flash_rt_fa2":
            raise
        return None


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

    def __init__(self, max_seq: int, max_q_seq: int = 1, dtype=None,
                 num_full_layers: int | None = None,
                 use_fa2: bool | None = None):
        import torch

        self._torch = torch
        bf16 = dtype if dtype is not None else torch.bfloat16
        d = "cuda"
        # The model's ten full-attention layers by default. A speculative
        # draft head is one more full-attention layer carrying its own KV, so
        # it asks for an extra slot rather than owning a second cache.
        self.NUM_FULL_LAYERS = (
            int(num_full_layers) if num_full_layers is not None
            else type(self).NUM_FULL_LAYERS)

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

        # A target may not build FA2 at all. Absence is a fallback, not an
        # error, because the reference path below computes the same thing.
        #
        # Prefill and decode want different answers about FA2. Prefill gains
        # 20x on a chunked block's non-square window. At the decode shape the
        # two are the same answer to bf16 precision -- measured against an
        # fp32 reference, 2.0e-3 relative for both at kv=64, 2.2e-3 against
        # 2.1e-3 at kv=2048 -- so taking it there buys about 1% of a step and
        # moves two of the sixteen golden tokens, because a bf16-level
        # difference in one step flips a later one. That is the fixture losing
        # its meaning in exchange for 1%, which is the wrong trade.
        #
        # So the caller says. ``use_fa2=None`` keeps what each target already
        # validated: on the arch that has always had FA2 in decode, keep it;
        # on sm_110, where FA2 has only just started building and the fixture
        # was recorded through the reference path, decline it.
        # FLASHRT_NEXN2_DECODE_FA2 overrides either way.
        import os as _os
        _env_override = "FLASHRT_NEXN2_DECODE_FA2" in _os.environ
        if use_fa2 is None:
            _cap = torch.cuda.get_device_capability()
            _default = "0" if _cap == (11, 0) else "1"
            want_fa2 = _os.environ.get(
                "FLASHRT_NEXN2_DECODE_FA2", _default) != "0"
        else:
            want_fa2 = bool(use_fa2)
        self._require_fa2 = bool(
            use_fa2 is True or (_env_override and want_fa2))
        _fa2 = _load_optional_fa2()
        if want_fa2 and _fa2 is None and self._require_fa2:
            raise RuntimeError(
                "FA2 was explicitly requested, but flash_rt.flash_rt_fa2 "
                "is not installed. Build the FA2 module or disable the "
                "explicit FA2 request.")
        self._fa2 = _fa2 if want_fa2 else None
        self._fa2_fwd = _fa2.fwd_bf16 if self._fa2 is not None else None
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self._fa2_usable = (
            self._fa2_fwd is not None and self._probe_fa2())

    def _probe_fa2(self) -> bool:
        """Does the vendored kernel actually compute on this device?

        Importing it and finding its symbols proves neither. Its own arch
        handling can leave a build that links, loads, prints a complaint to
        stdout and returns without writing the output -- which downstream looks
        like plausible-but-wrong attention rather than a failure. Measured on
        an SM110 part: the module imported, every symbol was present, and the
        kernel refused at run time.

        So run one small case against a reference and compare. The cost is one
        launch at construction.
        """
        # Imported here, not at module scope, for the same reason as F: this
        # module is written to import without torch present. The body had
        # never run on a target that builds FA2, so the missing name sat
        # unnoticed until this arch started building one.
        import torch
        import torch.nn.functional as F

        q_seq, kv_seq = 1, 8
        generator = torch.Generator(device=self.Q_buf.device).manual_seed(1)
        q = torch.randn(
            1, q_seq, self.NUM_Q_HEADS, self.HEAD_DIM, generator=generator,
            device=self.Q_buf.device, dtype=torch.bfloat16)
        k = torch.randn(
            1, kv_seq, self.NUM_KV_HEADS, self.HEAD_DIM, generator=generator,
            device=self.Q_buf.device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        self.Q_buf[:, :q_seq].copy_(q)
        self.K_cache[0:1, :kv_seq].copy_(k)
        self.V_cache[0:1, :kv_seq].copy_(v)
        self.O_buf[:, :q_seq].zero_()
        try:
            self._launch_fa2(0, q_seq, kv_seq, 0,
                             1.0 / (self.HEAD_DIM ** 0.5))
            torch.cuda.synchronize()
        except Exception as exc:                             # noqa: BLE001
            if self._require_fa2:
                raise RuntimeError(
                    "FA2 was explicitly requested, but its runtime probe "
                    "failed.") from exc
            import warnings

            warnings.warn(
                "FA2 runtime probe failed; using the SDPA attention "
                f"fallback: {exc!r}", RuntimeWarning, stacklevel=2)
            return False
        produced = self.O_buf[:, :q_seq].float().clone()

        groups = self.NUM_Q_HEADS // self.NUM_KV_HEADS
        kr = k.repeat_interleave(groups, dim=2)
        vr = v.repeat_interleave(groups, dim=2)
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2).float(), kr.transpose(1, 2).float(),
            vr.transpose(1, 2).float()).transpose(1, 2)
        if not torch.isfinite(produced).all():
            usable = False
        else:
            reference = expected.norm().clamp_min(1e-6)
            usable = bool(
                ((produced - expected).norm() / reference).item() < 0.05)
        if not usable:
            message = (
                "FA2 runtime probe produced invalid or mismatched output.")
            if self._require_fa2:
                raise RuntimeError(message)
            import warnings

            warnings.warn(
                message + " Using the SDPA attention fallback.",
                RuntimeWarning, stacklevel=2)
        return usable

    def _sdpa(self, layer_idx: int, q_seq: int, kv_seq: int,
              softmax_scale: float) -> None:
        """Bottom-right causal reference for a rejected/absent FA2 kernel."""
        import torch
        import torch.nn.functional as F

        # BF16 throughout. The cache is bf16, so upcasting it materialises the
        # whole history in fp32 every step -- hundreds of MB of temporaries per
        # token at a long context, which is what made decode fall from 26 to 11
        # tok/s between 4k and 10k. SDPA accumulates in fp32 regardless.
        q = self.Q_buf[:, :q_seq].transpose(1, 2)
        k = self.K_cache[layer_idx:layer_idx + 1, :kv_seq]
        v = self.V_cache[layer_idx:layer_idx + 1, :kv_seq]
        # Broadcasting the KV to the query head count materialises it: at
        # decode that is 8x2xkv_seqx256 floats twice per layer, ~48 MB a step
        # across the ten full-attention layers, purely to be read once. Native
        # GQA does the same thing without the copy. fp32 either way, so the
        # numerics are untouched -- this path seeds a token-exact decode.
        groups = self.NUM_Q_HEADS // self.NUM_KV_HEADS
        if q_seq > kv_seq:
            raise ValueError(
                f"q_seq={q_seq} cannot exceed kv_seq={kv_seq} for causal "
                "attention")
        mask = None
        if 1 < q_seq < kv_seq:
            q_positions = torch.arange(
                kv_seq - q_seq, kv_seq, device=q.device).unsqueeze(1)
            mask = torch.arange(
                kv_seq, device=q.device).unsqueeze(0) <= q_positions
        try:
            out = F.scaled_dot_product_attention(
                q, k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=mask, is_causal=q_seq == kv_seq and q_seq > 1,
                scale=softmax_scale, enable_gqa=True,
            ).transpose(1, 2)
        except TypeError:                       # torch without native GQA
            out = F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(groups, dim=2).transpose(1, 2),
                v.repeat_interleave(groups, dim=2).transpose(1, 2),
                attn_mask=mask, is_causal=q_seq == kv_seq and q_seq > 1,
                scale=softmax_scale,
            ).transpose(1, 2)
        self.O_buf[:, :q_seq].copy_(out.to(self.O_buf.dtype))

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

        if not self._fa2_usable:
            self._sdpa(layer_idx, q_seq, kv_seq, softmax_scale)
            return o.data_ptr()

        self._launch_fa2(layer_idx, q_seq, kv_seq, stream, softmax_scale)
        return o.data_ptr()

    def _launch_fa2(self, layer_idx: int, q_seq: int, kv_seq: int,
                    stream: int, softmax_scale: float) -> None:
        """One vendored-FA2 launch. Shared with the construction-time probe so
        the probe exercises the same call the hot path makes."""
        q = self.Q_buf[:, :q_seq]
        k = self.K_cache[layer_idx:layer_idx + 1, :kv_seq]
        v = self.V_cache[layer_idx:layer_idx + 1, :kv_seq]
        o = self.O_buf[:, :q_seq]
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

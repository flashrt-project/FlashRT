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

    def __init__(self, max_seq: int, max_q_seq: int = 1, dtype=None,
                 *, num_layers: int | None = None,
                 num_q_heads: int | None = None,
                 num_kv_heads: int | None = None,
                 head_dim: int | None = None,
                 device: str = "cuda",
                 use_int8_kv: bool = False):
        import os

        import torch

        self._torch = torch
        bf16 = dtype if dtype is not None else torch.bfloat16
        d = torch.device(device)

        # Per-instance attention geometry. Defaults match Qwen3-VL-8B
        # (36/32/8/128); the 2B variant passes 28/16/8/128. Class constants
        # are kept for the static spec helper / backward compat.
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

        # ── INT8 KV cache (opt-in) ──
        # int8 mirrors of K/V with one bf16 scale per (position, kv-head) row.
        # q=1 decode reads these instead of the bf16 cache, halving the KV
        # bytes per step; prefill still runs FA2 against the bf16 cache.
        # Everything is allocated here so the decode path stays graph-safe.
        self._use_int8_kv = bool(use_int8_kv)
        if self._use_int8_kv:
            from flash_rt import flash_rt_qwen3_vl_kernels as _vlk
            for fn in ('qwen3_kv_rows_quant_int8',
                       'qwen3_attn_decode_int8kv_partial',
                       'qwen3_attn_decode_int8kv_combine'):
                if not hasattr(_vlk, fn):
                    raise RuntimeError(
                        f'use_int8_kv=True requires {fn} in '
                        'flash_rt_qwen3_vl_kernels (rebuild with '
                        '-DFLASHRT_BUILD_QWEN3_VL=ON)')
            self._vlk = _vlk
            if (self.NUM_Q_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM) != (
                    16, 8, 128):
                raise RuntimeError(
                    'the int8-KV decode kernel is specialized for GQA 16Q/8KV '
                    f'head_dim=128; got {self.NUM_Q_HEADS}Q/'
                    f'{self.NUM_KV_HEADS}KV head_dim={self.HEAD_DIM}')
            self.K8 = torch.empty(
                self.NUM_FULL_LAYERS, self._max_seq,
                self.NUM_KV_HEADS, self.HEAD_DIM,
                dtype=torch.int8, device=d)
            self.V8 = torch.empty_like(self.K8)
            self.KS = torch.empty(
                self.NUM_FULL_LAYERS, self._max_seq, self.NUM_KV_HEADS,
                dtype=bf16, device=d)
            self.VS = torch.empty_like(self.KS)
            n_chunks_max = (self._max_seq + 127) // 128
            self._i8_part_o = torch.empty(
                n_chunks_max, self.NUM_Q_HEADS, self.HEAD_DIM,
                dtype=torch.float32, device=d)
            self._i8_part_m = torch.empty(
                n_chunks_max, self.NUM_Q_HEADS,
                dtype=torch.float32, device=d)
            self._i8_part_l = torch.empty_like(self._i8_part_m)

        # FA2 module. Fail loud if missing — Qwen3 path requires the
        # vendored fp16/bf16 FA2 build.
        from flash_rt import flash_rt_fa2 as _fa2
        self._fa2 = _fa2
        self._fa2_fwd = _fa2.fwd_bf16
        # Causal binding for q_seq>1 prefill. Built only for
        # (bf16, head_dim=128); other shapes fall back to the SDPA path.
        self._fa2_fwd_causal = getattr(_fa2, 'fwd_bf16_causal', None)
        self._num_sms = torch.cuda.get_device_properties(
            d
        ).multi_processor_count

        # Optional debug bisect knob (mirror qwen36 sibling).
        self._use_fvk_fa2 = os.environ.get("FVK_QWEN3_FA2", "1") == "1"
        if not self._use_fvk_fa2:
            from flash_attn import flash_attn_func
            self._flash_attn_func = flash_attn_func

        # fp8-QK prefill attention (env-gated, additive). Quantizes Q/K to e4m3
        # (per-warp / per-block-64) and casts V to fp16, then runs the sage2
        # fp8-QK / fp16-PV causal-GQA kernel. Only fires for q_seq a multiple of
        # 128 (the kernel's per-warp scale stride is ceil(Lq/128)*4 — matched by
        # the quant only at that alignment); FA2 handles every other case.
        self._fp8_prefill = (
            os.environ.get("FLASH_RT_QWEN3_FP8_ATTN", "0") == "1"
            and self._max_q_seq >= 128
        )
        self._fp8_attn_fn = None
        if self._fp8_prefill:
            from flash_rt import flash_rt_kernels as _fvk
            fn = getattr(
                _fvk, "sage2_qk_f8_sv_f16_bf16_gqa_nhd_d128_causal", None)
            if fn is None or not hasattr(_fvk, "quant_per_token_fp8_bf16_d128"):
                self._fp8_prefill = False
            else:
                self._fvk = _fvk
                self._fp8_attn_fn = fn
                sp = ((self._max_q_seq + 127) // 128) * 128
                self._fp8_sp_max = sp
                hq, hkv, hd = self.NUM_Q_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM
                self._fp8_q8 = torch.zeros(
                    1, sp, hq, hd, dtype=torch.float8_e4m3fn, device=d)
                self._fp8_k8 = torch.zeros(
                    1, sp, hkv, hd, dtype=torch.float8_e4m3fn, device=d)
                self._fp8_vf = torch.zeros(
                    1, sp, hkv, hd, dtype=torch.float16, device=d)
                # per-token scales: one per (token, head).
                self._fp8_qs = torch.zeros(hq * sp, dtype=torch.float32, device=d)
                self._fp8_ks = torch.zeros(hkv * sp, dtype=torch.float32, device=d)

        # All-fp8 prefill attention (env-gated, additive): raw e4m3 Q/K/V from
        # the direct-cast norm_rope fold, single all-fp8 causal-GQA kernel (no
        # per-token scales — post-norm Q/K sit inside e4m3's dynamic range).
        # Takes precedence over FLASH_RT_QWEN3_FP8_ATTN when both are set.
        self._fp8_direct = False
        if (os.environ.get("FLASH_RT_QWEN3_FP8_FMHA", "1") == "1"
                and self._max_q_seq >= 128):
            from flash_rt import flash_rt_kernels as _fvk
            fn = getattr(_fvk, "fmha_fp8_causal_gqa_nhd_d128", None)
            have_quant = (
                hasattr(_fvk, "qwen3_q_norm_rope_qstage_prefill_v3_fp8_direct")
                and hasattr(_fvk, "qwen3_k_norm_rope_kvwrite_prefill_v3_fp8_direct"))
            if fn is not None and have_quant:
                self._fvk = _fvk
                self._fp8_direct = True
                self._fp8_prefill = True
                self._fp8_attn_fn = fn
                # fp4-out epilogue: O emitted directly as the o_proj GEMM's
                # NVFP4 A-operand (skips the bf16 O + quantize launch). The
                # frontend hands the packed/SF scratch pointers before run().
                self._fp8_fp4out_fn = getattr(
                    _fvk, "fmha_fp8_causal_gqa_nhd_d128_fp4out", None)
                self._fp8_o_fp4 = 0
                self._fp8_o_sf = 0
                sp = ((self._max_q_seq + 127) // 128) * 128
                self._fp8_sp_max = sp
                hq, hkv, hd = self.NUM_Q_HEADS, self.NUM_KV_HEADS, self.HEAD_DIM
                self._fp8_q8 = torch.zeros(
                    1, sp, hq, hd, dtype=torch.float8_e4m3fn, device=d)
                self._fp8_k8 = torch.zeros(
                    1, sp, hkv, hd, dtype=torch.float8_e4m3fn, device=d)
                self._fp8_v8 = torch.zeros(
                    1, sp, hkv, hd, dtype=torch.float8_e4m3fn, device=d)

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

    # ── INT8 KV maintenance (no-ops unless use_int8_kv) ──

    def quantize_kv_rows(self, layer_idx: int, pos: int,
                         stream: int = 0) -> None:
        """Mirror the K/V rows a decode step just wrote at (layer, pos) into
        int8. Graph-safe: fixed pointers, one tiny launch each."""
        if not self._use_int8_kv:
            return
        nkv = self.NUM_KV_HEADS
        self._vlk.qwen3_kv_rows_quant_int8(
            self.K_cache[layer_idx, pos].data_ptr(),
            self.K8[layer_idx, pos].data_ptr(),
            self.KS[layer_idx, pos].data_ptr(), nkv, stream)
        self._vlk.qwen3_kv_rows_quant_int8(
            self.V_cache[layer_idx, pos].data_ptr(),
            self.V8[layer_idx, pos].data_ptr(),
            self.VS[layer_idx, pos].data_ptr(), nkv, stream)

    def quantize_kv_prefix(self, kv_len: int, stream: int = 0) -> None:
        """Bulk-mirror KV[0:kv_len] for every layer after prefill."""
        if not self._use_int8_kv:
            return
        n_rows = int(kv_len) * self.NUM_KV_HEADS
        for L in range(self.NUM_FULL_LAYERS):
            self._vlk.qwen3_kv_rows_quant_int8(
                self.K_cache[L, 0].data_ptr(), self.K8[L, 0].data_ptr(),
                self.KS[L, 0].data_ptr(), n_rows, stream)
            self._vlk.qwen3_kv_rows_quant_int8(
                self.V_cache[L, 0].data_ptr(), self.V8[L, 0].data_ptr(),
                self.VS[L, 0].data_ptr(), n_rows, stream)

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

        # fp8-QK prefill path (env-gated). q_seq is the captured S-bucket; the
        # buckets that reach here (>=128) are powers of two, hence multiples of
        # 128 -> Sp == q_seq, no staging/padding. Causal masking excludes the
        # prompt's pad tokens (positions >= real_S) for every real query.
        if (self._fp8_prefill and q_seq > 1 and causal
                and q_seq % 128 == 0 and q_seq <= self._fp8_sp_max):
            return self._run_fp8_prefill(
                layer_idx, q_seq, softmax_scale, stream)

        # Decode (q_seq=1): single-query, causal vs non-causal identical,
        # so use the existing non-causal fwd_bf16 (template Is_causal=false).
        # Prefill (q_seq>1, causal=True): use the fwd_bf16_causal
        # binding (template Is_causal=true). Both paths handle GQA
        # natively via FA2's h_h_k_ratio — no repeat_interleave / SDPA
        # detour.
        if self._use_int8_kv and q_seq == 1:
            # Flash-decoding over the int8 KV mirrors: half the KV HBM bytes
            # of the bf16 FA2 path. Partials + combine buffers were
            # pre-allocated in __init__, so this is graph-safe.
            n_chunks = (kv_seq + 127) // 128
            self._vlk.qwen3_attn_decode_int8kv_partial(
                self.Q_buf[0, 0].data_ptr(),
                self.K8[layer_idx].data_ptr(),
                self.V8[layer_idx].data_ptr(),
                self.KS[layer_idx].data_ptr(),
                self.VS[layer_idx].data_ptr(),
                self._i8_part_o.data_ptr(),
                self._i8_part_m.data_ptr(),
                self._i8_part_l.data_ptr(),
                kv_seq, n_chunks, float(softmax_scale), stream)
            self._vlk.qwen3_attn_decode_int8kv_combine(
                self._i8_part_o.data_ptr(),
                self._i8_part_m.data_ptr(),
                self._i8_part_l.data_ptr(),
                self.O_buf[0, 0].data_ptr(), n_chunks, stream)
            return o.data_ptr()

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

    def _run_fp8_prefill(self, layer_idx: int, q_seq: int,
                         softmax_scale: float, stream: int) -> int:
        """fp8-QK / fp16-PV causal-GQA prefill attention. q_seq is a multiple of
        128, so Sp == q_seq. The per-token e4m3 Q/K (+scale) and fp16 V are ALL
        produced by the FOLDED norm_rope kernels (q8/k8/vf/qs/ks) — this gate is
        identical to the prefill fold gate, so they always fire together — so the
        attention here is a single launch, no quant/cast. Writes O_buf directly.
        Graph-safe: scratch pre-allocated in __init__, kernel uses `stream`."""
        Sp = q_seq
        if self._fp8_direct:
            # All-fp8 path: raw e4m3 Q/K/V, no scales.
            if self._fp8_fp4out_fn is not None and self._fp8_o_fp4:
                rc = self._fp8_fp4out_fn(
                    q_fp8=self._fp8_q8.data_ptr(),
                    k_fp8=self._fp8_k8.data_ptr(),
                    v_fp8=self._fp8_v8.data_ptr(),
                    out_fp4=self._fp8_o_fp4, out_sf=self._fp8_o_sf,
                    Lq=Sp, Lk=Sp,
                    Hq=self.NUM_Q_HEADS, Hkv=self.NUM_KV_HEADS,
                    softmax_scale=softmax_scale, stream=stream)
                if rc != 0:
                    raise RuntimeError(f"qwen3 fp8 prefill attn rc={rc}")
                return self.O_buf.data_ptr()
            rc = self._fp8_attn_fn(
                q_fp8=self._fp8_q8.data_ptr(), k_fp8=self._fp8_k8.data_ptr(),
                v_fp8=self._fp8_v8.data_ptr(), out_bf16=self.O_buf.data_ptr(),
                Lq=Sp, Lk=Sp, Hq=self.NUM_Q_HEADS, Hkv=self.NUM_KV_HEADS,
                softmax_scale=softmax_scale, stream=stream)
            if rc != 0:
                raise RuntimeError(f"qwen3 fp8 prefill attn rc={rc}")
            return self.O_buf.data_ptr()
        rc = self._fp8_attn_fn(
            q_fp8=self._fp8_q8.data_ptr(), k_fp8=self._fp8_k8.data_ptr(),
            v_fp16=self._fp8_vf.data_ptr(), out_bf16=self.O_buf.data_ptr(),
            q_scale=self._fp8_qs.data_ptr(), k_scale=self._fp8_ks.data_ptr(),
            B=1, Lq=Sp, Lk=Sp, Hq=self.NUM_Q_HEADS, Hkv=self.NUM_KV_HEADS,
            softmax_scale=softmax_scale, per_token=True, stream=stream)
        if rc != 0:
            raise RuntimeError(f"qwen3 fp8 prefill attn rc={rc}")
        return self.O_buf.data_ptr()


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

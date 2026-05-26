"""FlashRT -- Thor Qwen3.6-27B full-attention backend (SM110).

Mirrors the surface of :class:`RtxFlashAttnBackendQwen36` (see
``flash_rt.hardware.rtx.attn_backend_qwen36``) so the Qwen3.6 frontend
can stay model-shape-agnostic, but routes full-attention calls through
the FlashInfer XQA kernel
(``fvk.qwen36_flashinfer_xqa_bf16_fp8kv_spec``) instead of the
RTX-only vendored FA2 ``fwd_bf16``.

Architecture
------------
Qwen3.6 has two attention regimes per decoder layer:

  * ``full_attention`` (16 layers): GQA 24Q / 4KV, head_dim=256.
    On Thor this site goes through XQA with paged FP8 K/V cache.
  * ``linear_attention`` (48 layers): Gated DeltaNet recurrent. Runs
    through the existing ``fvk.gated_deltanet_recurrent_qwen36_bf16``
    kernels (graph-capture-safe, no FA2 dependency on either GPU
    target). This backend does not own those tensors.

Implementation status
---------------------
Step 1c on the Thor branch wires the FlashInfer XQA kernel into
``flash_rt_kernels`` on sm_110a. The XQA C-side dispatch is verified
runnable on Thor (see ``dev_log_qwen36_thor/step1bc_thor_build_and_smoke.md``).

What remains to wire here:
  * Construction-time allocation of the FP8 paged K/V slabs (same
    formulas as RTX's ``_fp8_setup_long_kv`` block).
  * Per-token K/V quantize + page-write on the decode hot path.
  * ``run("full", ...)`` that constructs the XQA arguments
    (page_table, seq_lens, mask, semaphores, scratch) and calls
    ``fvk.qwen36_flashinfer_xqa_bf16_fp8kv_spec`` directly.
  * Adapter for prefill-chunk calls that today reach
    ``self._attn._fa2_fwd_causal`` inside the RTX frontend (we either
    re-route the matching frontend method or expose a thin
    ``_xqa_fwd_causal`` shim with the same call signature).

To keep the dev branch's smoke tests honest, the current ``run``
method raises NotImplementedError; the kernel surface is in place,
the integration of the four call sites is the focused next milestone.
The frontend split (this file + :mod:`flash_rt.frontends.torch.qwen36_thor`)
is in place so per-arch hardware contracts are obeyed.
"""

from __future__ import annotations


class ThorFlashAttnBackendQwen36:
    """Qwen3.6 full-attention backend on Thor (FP8 KV via XQA).

    Surface mirrors :class:`RtxFlashAttnBackendQwen36`. Allocations
    that the frontend addresses by attribute name (``K_cache``,
    ``V_cache``, ``Q_buf``, ``O_buf``, ``lse_buf``, ``_num_sms``) are
    populated so the RTX-side weight loader and KV-write helpers can
    address them without per-arch branches.

    The actual attention call lives in ``run`` and dispatches to
    ``fvk.qwen36_flashinfer_xqa_bf16_fp8kv_spec``.
    """

    SITES = ("full",)
    NUM_FULL_LAYERS = 16
    NUM_Q_HEADS = 24
    NUM_KV_HEADS = 4
    HEAD_DIM = 256

    # XQA build-time constants from the CMake target
    # (see qwen36_flashinfer_xqa_obj in CMakeLists.txt):
    #   TOKENS_PER_PAGE=128, HEAD_ELEMS=256, HEAD_GRP_SIZE=6,
    #   DTYPE=__nv_bfloat16, CACHE_ELEM_ENUM=2 (FP8 e4m3).
    XQA_TOKENS_PER_PAGE = 128

    def __init__(self, max_seq: int, max_q_seq: int = 1, dtype=None):
        import torch

        from flash_rt import flash_rt_kernels as fvk
        if not hasattr(fvk, "qwen36_flashinfer_xqa_bf16_fp8kv_spec"):
            raise RuntimeError(
                "FlashRT Thor build is missing "
                "qwen36_flashinfer_xqa_bf16_fp8kv_spec. Rebuild "
                "flash_rt_kernels with -DGPU_ARCH=110.")

        bf16 = dtype if dtype is not None else torch.bfloat16
        device = "cuda"
        page = self.XQA_TOKENS_PER_PAGE
        # Round max_seq up to a page boundary (XQA contract).
        max_seq = ((int(max_seq) + page - 1) // page) * page

        self._torch = torch
        self._fvk = fvk
        self._dtype = bf16
        self._max_seq = max_seq
        self._max_q_seq = int(max_q_seq)
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count

        # BF16 K/V cache slabs (frontend's per-layer writer addresses
        # K_cache[layer, cur_pos] etc.). Same shape as the RTX backend.
        self.K_cache = torch.empty(
            self.NUM_FULL_LAYERS, max_seq,
            self.NUM_KV_HEADS, self.HEAD_DIM,
            dtype=bf16, device=device)
        self.V_cache = torch.empty_like(self.K_cache)

        # Q / O scratch.
        self.Q_buf = torch.empty(
            1, self._max_q_seq, self.NUM_Q_HEADS, self.HEAD_DIM,
            dtype=bf16, device=device)
        self.O_buf = torch.empty_like(self.Q_buf)

        # FA2-style softmax LSE buffers; XQA does not consume these
        # but the frontend's prefill code path captures shapes off
        # them, so we allocate matching layouts.
        sq_rounded = ((self._max_q_seq + 127) // 128) * 128
        self.lse_buf = torch.empty(
            1, self.NUM_Q_HEADS, sq_rounded,
            dtype=torch.float32, device=device)
        n_splits = min(128, (self._max_seq + 63) // 64)
        self._n_splits = n_splits
        self.lse_accum = torch.empty(
            n_splits, 1, self.NUM_Q_HEADS, self._max_q_seq,
            dtype=torch.float32, device=device)
        self.o_accum = torch.empty(
            n_splits, 1, self.NUM_Q_HEADS, self._max_q_seq, self.HEAD_DIM,
            dtype=torch.float32, device=device)

        # The FA2 attribute is exposed as a Python callable that
        # adapts the FA2 forward contract to a bf16 -> fp8 paged
        # quantize + XQA call. This lets the RTX frontend's
        # FA2-direct call sites (MTP head verify, prefill chunks)
        # run unchanged on Thor. ``_fa2_fwd_causal`` is left ``None``
        # so the RTX frontend falls back to the non-causal per-q-step
        # path that uses the same adapter — XQA enforces causality
        # via its row-wise mask.
        self._fa2_fwd = self._fa2_fwd_adapter
        self._fa2_fwd_causal = None

    # ── Layer cache pointer math (matches RTX surface) ──

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
                f"qwen36 thor backend only knows site='full', got {site!r}")
        return self.HEAD_DIM

    def num_q_heads(self, site: str) -> int:
        if site != "full":
            raise KeyError(
                f"qwen36 thor backend only knows site='full', got {site!r}")
        return self.NUM_Q_HEADS

    def num_kv_heads(self, site: str) -> int:
        if site != "full":
            raise KeyError(
                f"qwen36 thor backend only knows site='full', got {site!r}")
        return self.NUM_KV_HEADS

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict:
        if site != "full":
            raise KeyError(
                f"qwen36 thor backend only knows site='full', got {site!r}")
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

    def _ensure_xqa_paged(self):
        """Lazy allocation of the XQA argument tensors. Defers GPU
        memory pressure until the first attention call so an
        unused Thor backend (e.g. a frontend ctor smoke test) does
        not eat the FP8 KV / scratch / semaphores VRAM."""
        if getattr(self, "_xqa_inited", False):
            return
        torch = self._torch
        fp8 = torch.float8_e4m3fn
        device = "cuda"

        # FP8 paged K/V — one slab per full-attn layer.
        page = self.XQA_TOKENS_PER_PAGE
        n_pages = self._max_seq // page
        self._fp8_K = torch.empty(
            self.NUM_FULL_LAYERS, n_pages, page,
            self.NUM_KV_HEADS, self.HEAD_DIM,
            dtype=fp8, device=device)
        self._fp8_V = torch.empty_like(self._fp8_K)

        self._page_table = torch.arange(
            n_pages, dtype=torch.int32, device=device).view(1, n_pages)
        # Cached per-end_pos seq_len + per-q_seq causal mask tensors.
        self._seq_lens_cache: dict[int, "torch.Tensor"] = {}
        self._mask_cache: dict[int, "torch.Tensor"] = {}

        # Semaphore + scratch (sized like the RTX long-ctx XQA path).
        head_grp = self.NUM_Q_HEADS // self.NUM_KV_HEADS
        sem_count = self.NUM_KV_HEADS * (
            (self._max_q_seq * head_grp + 31) // 32)
        self._semaphores = torch.zeros(
            max(256, sem_count), dtype=torch.uint32, device=device)
        # 256 MB scratch — matches the RTX default; sized for the
        # worst-case long-context buckets.
        self._scratch = torch.zeros(
            256 << 20, dtype=torch.uint8, device=device)

        self._k_stride_page = page * self.NUM_KV_HEADS * self.HEAD_DIM
        self._k_stride_token = self.NUM_KV_HEADS * self.HEAD_DIM
        self._k_stride_head = self.HEAD_DIM

        # FP8 paged scratch used by the FA2 adapter (separate from
        # the per-layer FP8 cache so callers can pass arbitrary
        # bf16 K/V slabs from the frontend — e.g. _mtp_K_cache —
        # without colliding with the main attention path).
        self._fa2_fp8_K = torch.empty(
            n_pages, page, self.NUM_KV_HEADS, self.HEAD_DIM,
            dtype=fp8, device=device)
        self._fa2_fp8_V = torch.empty_like(self._fa2_fp8_K)

        # Per-tensor scale = 1.0 for the FP8 conversion.
        # Matches XQA's kv_scale=1.0 contract on this build.
        self._scale_one = torch.ones(1, dtype=torch.float32, device=device)

        self._xqa_inited = True

    def _seq_lens_for(self, end_pos: int):
        t = self._seq_lens_cache.get(end_pos)
        if t is None:
            t = self._torch.full(
                (1, 1), end_pos, dtype=self._torch.uint32, device="cuda")
            self._seq_lens_cache[end_pos] = t
        return t

    def _mask_for(self, q_seq: int):
        t = self._mask_cache.get(q_seq)
        if t is None:
            torch = self._torch
            words = (q_seq + 31) // 32
            rows = torch.zeros((q_seq, words), dtype=torch.int32)
            for i in range(q_seq):
                upto = i + 1
                full = upto // 32
                rem = upto % 32
                if full:
                    rows[i, :full] = -1
                if rem:
                    rows[i, full] = (1 << rem) - 1
            t = rows.to(device="cuda")
            self._mask_cache[q_seq] = t
        return t

    def run(self, site: str, layer_idx: int, q_seq: int,
            *, kv_seq: int, stream: int = 0,
            softmax_scale: float | None = None) -> int:
        """Run full-attention on Thor via the FlashInfer XQA kernel.

        The pipeline has already written the per-token BF16 K/V into
        ``self.K_cache[layer, cur_pos]`` / ``self.V_cache[layer, cur_pos]``
        (same contract as the RTX backend). This method
        copy-quantizes the relevant slab into the FP8 paged storage
        on demand, then issues the XQA call.

        For the initial Thor bring-up the per-call quantize is wasteful
        (re-quantizes the full ``[:kv_seq]`` prefix every call).
        Production will move this onto the K/V writer side so each new
        token quantizes once. Logged as Step 5 optimization on the dev
        branch.
        """
        if site != "full":
            raise KeyError(
                f"qwen36 thor backend only knows site='full', got {site!r}")
        if not (1 <= q_seq <= self._max_q_seq):
            raise ValueError(
                f"q_seq={q_seq} out of range [1, {self._max_q_seq}]")
        if not (1 <= kv_seq <= self._max_seq):
            raise ValueError(
                f"kv_seq={kv_seq} out of range [1, {self._max_seq}]")

        self._ensure_xqa_paged()
        torch = self._torch

        # Round kv_seq up to a page boundary; XQA scans the full pages
        # but only reads up to ``seq_lens`` rows from the last page.
        page = self.XQA_TOKENS_PER_PAGE
        max_seq_len = ((kv_seq + page - 1) // page) * page

        # Re-quantize the bf16 K/V slab for this layer into fp8 paged
        # storage. The bf16 slab is already contiguous along (kv_seq,
        # NUM_KV_HEADS, HEAD_DIM), so a single view + .to(fp8) gives the
        # paged layout XQA expects after a reshape.
        k_bf16 = self.K_cache[layer_idx, :max_seq_len]
        v_bf16 = self.V_cache[layer_idx, :max_seq_len]
        # to(fp8_e4m3fn) saturates BF16 values into FP8 e4m3; this
        # matches the RTX FP8-KV long-ctx path's kv_scale=1.0 contract.
        self._fp8_K[layer_idx, :max_seq_len // page] = k_bf16.view(
            max_seq_len // page, page, self.NUM_KV_HEADS, self.HEAD_DIM
        ).to(torch.float8_e4m3fn)
        self._fp8_V[layer_idx, :max_seq_len // page] = v_bf16.view(
            max_seq_len // page, page, self.NUM_KV_HEADS, self.HEAD_DIM
        ).to(torch.float8_e4m3fn)

        # Slice the static buffers down to the live q/o.
        q_view = self.Q_buf[:, :q_seq].view(
            1, 1, q_seq, self.NUM_Q_HEADS, self.HEAD_DIM)
        o_view = self.O_buf[:, :q_seq].view(
            1, 1, q_seq, self.NUM_Q_HEADS, self.HEAD_DIM)

        self._fvk.qwen36_flashinfer_xqa_bf16_fp8kv_spec(
            q_view.data_ptr(),
            self._fp8_K[layer_idx].data_ptr(),
            self._fp8_V[layer_idx].data_ptr(),
            self._page_table.data_ptr(),
            self._seq_lens_for(kv_seq).data_ptr(),
            self._mask_for(q_seq).data_ptr(),
            o_view.data_ptr(),
            self._semaphores.data_ptr(),
            self._scratch.data_ptr(),
            max_seq_len,
            int(q_seq),
            int(self._num_sms),
            1.0,   # q_scale
            1.0,   # kv_scale
            True,  # enable_pdl
            self._k_stride_page,
            self._k_stride_token,
            self._k_stride_head,
            int(stream),
        )
        return self.O_buf[:, :q_seq].data_ptr()

    def _fa2_fwd_adapter(
        self,
        *,
        Q, K, V, O,
        softmax_lse=0, softmax_lse_accum=0, o_accum=0,
        batch, seqlen_q, seqlen_k,
        num_heads_q, num_heads_kv, head_dim,
        q_strides=None, k_strides=None,
        v_strides=None, o_strides=None,
        softmax_scale=None,
        num_sms=None,
        stream=0,
    ):
        """Drop-in for ``flash_rt_fa2.fwd_bf16`` on Thor.

        The RTX frontend reaches this entry point for the MTP head
        verify and a couple of prefill chunk sites where it has its
        own bf16 K/V cache (separate from the main per-layer cache).
        Here we quantize the bf16 K/V slab to FP8 e4m3 on demand,
        reshape into the XQA paged layout, and issue an XQA call.

        Q, K, V, O are raw device pointers (``data_ptr()`` integers).
        K/V are expected contiguous as ``(batch, seqlen_k,
        num_heads_kv, head_dim)`` bf16. The FA2 LSE buffers are
        accepted for ABI compatibility but unused on the XQA path
        (XQA computes the softmax internally without exposing LSE).

        Causality: XQA enforces the causal mask via the row-wise
        ``mask`` argument so an FA2 ``causal=True`` request becomes
        the same call as ``causal=False`` here.
        """
        if batch != 1:
            raise ValueError(
                f"thor _fa2_fwd_adapter: batch={batch} unsupported")
        if num_heads_q != self.NUM_Q_HEADS \
                or num_heads_kv != self.NUM_KV_HEADS \
                or head_dim != self.HEAD_DIM:
            raise ValueError(
                f"thor _fa2_fwd_adapter: GQA shape mismatch "
                f"q={num_heads_q} kv={num_heads_kv} hd={head_dim}")
        if seqlen_q < 1:
            raise ValueError(f"seqlen_q={seqlen_q} must be >= 1")
        if seqlen_k < 1:
            raise ValueError(f"seqlen_k={seqlen_k} must be >= 1")

        self._ensure_xqa_paged()
        fvk = self._fvk

        page = self.XQA_TOKENS_PER_PAGE
        max_seq_len = ((seqlen_k + page - 1) // page) * page

        # Grow the FP8 paged scratch on demand. The frontend's long-ctx
        # mode pipes K/V slabs whose seqlen_k can exceed the Thor
        # backend's own ``self._max_seq`` (which sized the per-layer
        # cache from ctor) — those slabs come from the frontend's TQ /
        # FP8-packed cache instead. Allocate enough pages for the
        # largest seqlen_k we have seen.
        torch = self._torch
        needed_pages = max_seq_len // page
        if self._fa2_fp8_K.shape[0] < needed_pages:
            self._fa2_fp8_K = torch.empty(
                needed_pages, page, self.NUM_KV_HEADS, self.HEAD_DIM,
                dtype=torch.float8_e4m3fn, device="cuda")
            self._fa2_fp8_V = torch.empty_like(self._fa2_fp8_K)
            # The XQA page_table also needs to cover the new range —
            # rebuild lazily for the larger size.
            self._page_table = torch.arange(
                needed_pages, dtype=torch.int32, device="cuda"
            ).view(1, needed_pages)

        # Convert bf16 K/V at the caller's pointer into FP8 e4m3 in
        # the adapter's paged scratch. Read only the actually-populated
        # seqlen_k tokens; padding beyond seqlen_k is irrelevant to
        # XQA because seq_lens caps the kernel's range.
        n_real = int(seqlen_k) * num_heads_kv * head_dim
        fvk.quantize_fp8_static(
            int(K), self._fa2_fp8_K.data_ptr(),
            self._scale_one.data_ptr(), n_real, int(stream))
        fvk.quantize_fp8_static(
            int(V), self._fa2_fp8_V.data_ptr(),
            self._scale_one.data_ptr(), n_real, int(stream))

        sms = int(num_sms) if num_sms is not None else int(self._num_sms)
        fvk.qwen36_flashinfer_xqa_bf16_fp8kv_spec(
            int(Q),
            self._fa2_fp8_K.data_ptr(),
            self._fa2_fp8_V.data_ptr(),
            self._page_table.data_ptr(),
            self._seq_lens_for(int(seqlen_k)).data_ptr(),
            self._mask_for(int(seqlen_q)).data_ptr(),
            int(O),
            self._semaphores.data_ptr(),
            self._scratch.data_ptr(),
            max_seq_len,
            int(seqlen_q),
            sms,
            1.0,   # q_scale
            1.0,   # kv_scale
            True,  # enable_pdl
            self._k_stride_page,
            self._k_stride_token,
            self._k_stride_head,
            int(stream),
        )


def make_qwen36_thor_attention_spec(
        *, max_seq: int, max_q_seq: int = 1) -> dict:
    """Static metadata describing Qwen3.6's full-attn site on Thor.

    Mirrors :func:`make_qwen36_attention_spec` from the RTX backend
    but reports the kernel as ``fvk_flashinfer_xqa_bf16_fp8kv``.
    """
    return {
        "sites": [
            {
                "name": "full",
                "layer_count": ThorFlashAttnBackendQwen36.NUM_FULL_LAYERS,
                "num_q_heads": ThorFlashAttnBackendQwen36.NUM_Q_HEADS,
                "num_kv_heads": ThorFlashAttnBackendQwen36.NUM_KV_HEADS,
                "head_dim": ThorFlashAttnBackendQwen36.HEAD_DIM,
                "max_q_seq": int(max_q_seq),
                "max_kv_seq": int(max_seq),
                "kernel": "fvk_flashinfer_xqa_bf16_fp8kv",
            },
        ],
        "linear_attn": {
            "layer_count": 48,
            "num_k_heads": 16,
            "num_v_heads": 48,
            "head_dim": 128,
            "conv_kernel": 4,
            "kernel": "fvk_gated_deltanet_recurrent_bf16",
        },
    }

"""Spark-X2.5-4B decode/prefill pipeline for RTX SM120.

Hybrid attention (27 sliding-window + 9 full), NVFP4 weights, batch = 1. This
module owns the per-layer kernel sequence: the KV write (partial RoPE + E4M3
quantisation in one pass), the two-pass decode attention, the boundary
norm/quantiser, the per-head output gate, gproj and the greedy argmax. Every
steady-state launch is a raw-pointer call into ``flash_rt_sparkx25`` /
``flash_rt_fa2`` / ``flash_rt_kernels``, so the whole decode loop captures into
one CUDA Graph and replays with no framework dispatch.

The attention is served by this model's own kernels rather than by the RTX
attention backend: FA2's decode entry launches a 1 x kv_heads x 1 grid, four
blocks on a 36-SM part, and saturates at 154 GB/s against this part's measured
425.8 GB/s ceiling. See docs/spark_x25_rtx.md for the measurements.

``SparkX25Runtime`` does both the frontend's job (weight load, buffer
allocation, graph capture) and the pipeline's (the layer sequence). Qwen3.6's
pipeline_rtx.py has the same shape, so the split here is by file, not by class.
"""
from __future__ import annotations

import math

import torch

# The three extension modules are optional at import time: this file is
# reachable from ``import flash_rt.models.spark_x25`` (config parsing, and the
# checkpoint validation the frontend shares), which must work on a machine with
# no SM120 build. Nothing at module scope dereferences them, so they are bound
# to None and checked in ``_require_kernels`` from the runtime constructor --
# the fail-fast style ``frontends/torch/nexn2_rtx.py`` uses.
try:
    import flash_rt.flash_rt_fa2 as fa2
except ImportError:                                     # pragma: no cover
    fa2 = None
try:
    import flash_rt.flash_rt_kernels as fvk
except ImportError:                                     # pragma: no cover
    fvk = None
try:
    from flash_rt import flash_rt_sparkx25 as sk
except ImportError:                                     # pragma: no cover
    sk = None

from flash_rt.models.spark_x25.config import load_config
from flash_rt.models.spark_x25.weights import load_weights

#: What to tell a caller whose build does not carry the modules above.
_BUILD_HINT = (
    "Build them with:\n"
    "    cmake -B build -S . -DGPU_ARCH=120 -DFLASHRT_ENABLE_SPARK_X25=ON\n"
    "    cmake --build build -j4 --target flash_rt_kernels flash_rt_fa2 "
    "flash_rt_sparkx25\n"
    "flash_rt_sparkx25 is SM120 (RTX 50-series) only and is skipped on every "
    "other GPU_ARCH. See docs/spark_x25_usage.md."
)


def _require_kernels() -> None:
    """Raise unless every extension this pipeline drives is importable.

    ``flash_rt_kernels`` supplies the NVFP4 W4A4 GEMMs, ``flash_rt_fa2`` the
    prefill attention and ``flash_rt_sparkx25`` the decode path; all three are
    needed, so a missing one is a refusal here rather than an AttributeError
    mid-capture.
    """
    missing = [name for name, mod in (
        ("flash_rt_kernels", fvk),
        ("flash_rt_fa2", fa2),
        ("flash_rt_sparkx25", sk)) if mod is None]
    if missing:
        raise RuntimeError(
            "Spark-X2.5 decode needs the compiled extensions "
            f"{', '.join(missing)}, which are not importable here. "
            + _BUILD_HINT)


def _sf_bytes(rows: int, k: int) -> int:
    return int(fvk.nvfp4_sf_swizzled_bytes(rows, k))


# Decode GEMM kernel choice, keyed by (N, K): (warps, stages) for FlashRT's
# warp-split-K kernel, or (0, 0) for `full_n`. See `_gemm` for the measurement
# that picks these. The warp-split kernel requires (K/64) % warps == 0, which
# every entry below satisfies; a shape that is not listed keeps `full_n`.
_DECODE_SPLIT = {
    (2560, 4096): (4, 4),    # o_proj
    (2560, 10240): (4, 3),   # down
    (6144, 2560): (2, 3),    # qkv
}


class SparkX25Runtime:
    """All-native Spark-X2.5 inference, batch = 1, fixed shapes throughout."""

    def __init__(self, ckpt_dir: str, max_seq: int = 32768,
                 prefill_cap: int = 8192, device: str = "cuda",
                 prefill_chunk: int | None = None,
                 split_kv_sms: int | None = None,
                 attn_impl: str = "native",
                 attn_splits: int | None = None,
                 attn_splits_slide: int | None = None):
        _require_kernels()
        self.cfg = load_config(ckpt_dir)
        self.device = device
        self.max_seq = max_seq
        self.prefill_cap = prefill_cap
        # Both split counts are partitions of a key range, so overriding them
        # cannot change a result -- only the block count. `None` keeps the
        # device-aware default chosen in `_alloc`.
        self._attn_splits = attn_splits
        self._attn_splits_slide = attn_splits_slide
        # Rows of the activation working set. Prefill walks the prompt in
        # chunks of this size and only the last chunk's logits are kept, so
        # this -- not `max_seq` -- is what sizes every activation buffer.
        self.prefill_chunk = int(prefill_chunk or min(prefill_cap, 2048))
        # Split the full layers' decode key range across this many of the SMs.
        # Measured in-graph on the 36-SM part, whole decode step:
        #   140 keys   7.181 -> 7.295 ms/token   (+1.6%, split loses)
        #   553 keys   7.716 -> 7.800            (+1.1%)
        #   1943 keys  8.334 -> 8.062            (-3.3%)
        #   8192 keys 11.293 -> 11.284           (a wash at num_sms=18)
        #   22000      17.702 -> 12.630          (-29%, split wins)
        # The plain entry walks the key range in blocks whose cost is fixed per
        # block, so at long context it is latency-bound rather than
        # bandwidth-bound -- 4x the bytes at 16 KV heads take the same time as
        # at 4. The split buys back the parallelism; below ~8k keys it costs
        # more than it buys. The cutoff is on the context budget because the
        # entry is baked into a captured graph.
        # 128 splits beats 18 once the range is long: at 22000 keys the split
        # kernel is 5197 us against 5499. It is also why the split now wins
        # down to ~1900 keys; below ~600 it still loses, so the cutoff stays on
        # the context budget -- which is what keeps the 128- and 512-token
        # buckets, and therefore the equal-scope median, on the plain entry.
        # The optimum moves with the key count, and it is measurable at both
        # ends: 128 splits wins at 22 000 keys (5197 us against 18's 5499), but
        # at 176 000 fewer is better -- 36 gives 51.3/51.6 ms per token against
        # 128's 54.9/55.0, two interleaved reps each. 6 is far worse (90 ms):
        # too few splits and the per-split range stops fitting the walk.
        self.split_kv_sms = int(
            split_kv_sms if split_kv_sms is not None
            else (0 if max_seq < 16384
                  else (36 if max_seq >= 131072 else 128)))
        # "native" runs the two-pass decode attention in this repo's own
        # kernel source; "fa2" keeps the FlashRT entry for A/B in-graph.
        self.attn_impl = attn_impl
        self.stream = torch.cuda.current_stream().cuda_stream
        self.w = load_weights(ckpt_dir, self.cfg, device=device, stream=self.stream)
        self._alloc()
        self._build_rope()

    # ── buffers ──────────────────────────────────────────────────────────
    def _alloc(self) -> None:
        c = self.cfg
        R = self.max_seq
        d = self.device
        bf = torch.bfloat16

        # The activation working set is sized by the prefill chunk, not by the
        # context budget. Every buffer below holds *rows of the step being
        # computed* -- one row during decode, up to `prefill_chunk` during
        # prefill -- and nothing in them persists between chunks, so sizing them
        # by `max_seq` bought 0.37 MB per context token for no reason. Only the
        # KV caches and `tokens_out` are indexed by absolute position and must
        # span the whole context.
        A = self.act_rows = max(1, min(self.prefill_chunk, R))

        def zbuf(*shape, dtype=bf):
            return torch.zeros(shape, dtype=dtype, device=d)

        # residual stream ping-pong, plus the intermediate the MLP half writes
        self.h_res = zbuf(A, c.hidden_size)
        self.h_tmp = zbuf(A, c.hidden_size)
        self.h_norm = zbuf(A, c.hidden_size)
        self.gate = zbuf(A, c.num_attention_heads)

        self.qkv = zbuf(A, c.qkv_dim)
        self.o_raw = zbuf(A, c.q_dim)
        self.o_gated = zbuf(A, c.q_dim)
        self.o_proj = zbuf(A, c.hidden_size)
        self.gate_up = zbuf(A, 2 * c.intermediate_size)
        self.down = zbuf(A, c.hidden_size)
        self.logits = zbuf(A, c.vocab_size)
        self.zero = zbuf(A, c.hidden_size)

        self.q_buf = zbuf(A, c.num_attention_heads, c.head_dim)
        self.lse = zbuf(A, c.num_attention_heads, 1, dtype=torch.float32)
        # Split-KV scratch. A decode step is one query row, so the plain entry
        # has only the KV blocks to hide memory latency behind; the split entry
        # cuts each key range into `num_sms` pieces that run concurrently, and
        # reduces them in these two buffers. Only the full layers use it -- a
        # 512-key window is already short enough that the split loses.
        self.lse_accum = zbuf(128, c.num_attention_heads, 1, dtype=torch.float32)
        # Native two-pass attention scratch. The whole score row S[q_head][key]
        # is materialised, which is what lets pass 2 be a plain weighted sum of
        # V rows instead of an online softmax; S and P together cost 8 bytes per
        # head per key, 7% of the KV bytes they save.
        self.attn_s = zbuf(c.num_attention_heads, self.max_seq, dtype=torch.float32)
        self.attn_p = zbuf(c.num_attention_heads, self.max_seq, dtype=torch.float32)
        self.attn_max = zbuf(c.num_attention_heads, dtype=torch.float32)
        self.attn_sum = zbuf(c.num_attention_heads, dtype=torch.float32)
        # Both are parallelism knobs, not correctness ones -- the split is a
        # partition of the key range and the combine is a sum, so neither can
        # change a result. The split count is not a monotone-in-context knob:
        # the per-key segment has to be long enough to stream (so the count
        # cannot be too small) but the combine reads nsplit x q_heads x head_dim
        # partials per layer whatever the key count is (so it cannot be too
        # large either).
        #
        # The two rules below are per-part, because the optimum moves with the
        # SM count. The 36-SM rule is the one the original sweeps picked
        # (scripts/attn_split_sweep.py): one split per 32 tokens of budget,
        # floored at 32 and capped at 256. Re-swept on a 170-SM 5090 (see
        # docs/spark_x25_rtx.md), the optimum moves to one split per 128 tokens
        # of budget, floored at 128 and capped at 1024; the capped-at-256 rule
        # cost 18% at 128k and 51% at 1M there.
        sm_count = int(torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count)
        if self._attn_splits is not None:
            nsplit = int(self._attn_splits)
        elif sm_count <= 64:
            nsplit = min(256, max(32, self.max_seq // 32))
        else:
            nsplit = min(1024, max(128, self.max_seq // 128))
        nsplit_cap = max(256, nsplit)
        self.attn_nsplit = nsplit
        # The sliding window is only W keys, but 8 splits gave it 8 blocks of
        # 64 threads -- 512 threads for 2 MB, which measured 794 us/token for
        # 27 layers against a ~150 us floor. 32 is the balance point on the
        # 36-SM part; on a 170-SM part the same 512-key window has only 64
        # blocks across 32 splits, so 64 splits buys ~2% there.
        if self._attn_splits_slide is not None:
            self.attn_nsplit_slide = int(self._attn_splits_slide)
        else:
            self.attn_nsplit_slide = 32 if sm_count <= 64 else 64
        self.attn_nchunk = 16
        self.attn_op = zbuf(nsplit_cap, c.num_attention_heads, c.head_dim,
                            dtype=torch.float32)
        self.o_accum = zbuf(128, c.num_attention_heads, 1, c.head_dim,
                            dtype=torch.float32)
        # KV storage. Full layers keep a plain growing linear cache. Sliding
        # layers keep the linear cache *and* a mirrored 2W ring: the linear part
        # is what prefill's per-query window reads, the ring is what decode
        # reads, and the ring is what makes decode's key pointer independent of
        # the position (so a captured graph can serve any step).
        W = c.sliding_window
        # A sliding layer's linear cache only ever serves prefill's per-query
        # window, which is at most `rows + W - 1 <= prefill_chunk + W - 1`
        # positions long. One mirrored period of that width therefore holds any
        # single prefill window, so the cache is that wide instead of
        # `max_seq` -- 4 KB per layer per context token that bought nothing.
        # Decode still reads the separate W-slot ring, whose residue-set
        # semantics are what keep its base pointer position-independent.
        # Full layers keep absolute indexing; their key range really does grow.
        self.lin_w = A + W
        #
        # Two modes, chosen by whether both caches fit.
        #
        #   mirror      bf16 K+V for every layer, plus an E4M3 copy of the full
        #               layers that decode reads. Prefill's FA2 sees exact bf16,
        #               so the long-context logit cosine stays ~0.999.
        #   kv8-only    the full layers keep E4M3 alone and prefill expands one
        #               layer's prefix into a reusable staging buffer. FA2 then
        #               sees the quantised cache expanded, which costs the
        #               long-context cosine (~0.99 at 8k/32k) but is the only way
        #               a 262k window runs at all: measured, the mirror needs
        #               4.77 GiB more than is free there.
        #
        # Decode is identical in both; they differ only in what prefill reads.
        n_full = sum(1 for lt in c.layer_types if lt == "full_attention")
        sliding_elems = sum((2 * self.lin_w + W) * c.kv_dim
                            for lt in c.layer_types if lt == "sliding_attention")
        mirror_extra = n_full * R * c.kv_dim * 6      # bf16 K,V (x4) + E4M3 K,V (x2)
        kv8_extra = n_full * R * c.kv_dim * 2 + 2 * R * c.kv_dim * 2
        free, _ = torch.cuda.mem_get_info()
        self.kv8_only = (mirror_extra + (1 << 30)) > free - sliding_elems * 2
        self.kv_offset = []          # (bf16_lin | None, ring | None, fp8_lin | None)
        off = 0                      # bf16 elements
        off8 = 0                     # E4M3 bytes
        for i in range(c.num_hidden_layers):
            sliding = c.layer_types[i] == "sliding_attention"
            lin = ring = lin8 = None
            if sliding:
                lin = off
                off += 2 * self.lin_w * c.kv_dim
                ring = off
                off += W * c.kv_dim
            else:
                if not self.kv8_only:
                    lin = off
                    off += R * c.kv_dim
                lin8 = off8
                off8 += R * c.kv_dim
            self.kv_offset.append((lin, ring, lin8))
        self.k_cache = zbuf(off)
        self.v_cache = zbuf(off)
        self.k8_cache = torch.zeros(off8, dtype=torch.uint8, device=d)
        self.v8_cache = torch.zeros(off8, dtype=torch.uint8, device=d)
        nslot8 = off8 // c.kv_dim
        self.k8_scale = torch.zeros(nslot8 * c.num_key_value_heads,
                                    dtype=torch.float32, device=d)
        self.v8_scale = torch.zeros(nslot8 * c.num_key_value_heads,
                                    dtype=torch.float32, device=d)
        # Used only in kv8-only mode, one layer at a time.
        if self.kv8_only:
            self.k_stage = zbuf(R, c.kv_dim)
            self.v_stage = zbuf(R, c.kv_dim)
        else:
            self.k_stage = self.v_stage = None
        # E4M3 mirror of the full layers' KV, with a per-(row, KV head) amax
        # scale. Only decode reads it -- prefill keeps the bf16 cache for FA2.
        # The KV bytes are the whole of decode's long-context cost and 4.83 GB
        # of bf16 at 131k is a 11.3 ms floor, already past what 2x Ollama
        # leaves for attention, so halving them is the only way through.
        #
        self.ring_w = W

        # decode-loop scratch
        # The decode position is a device value, not a launch argument, so the
        # captured graph's launch descriptors do not depend on it. There is
        # exactly one such word, and both the KV write (which rotates Q and K at
        # this position and mirrors K/V into the ring slot it selects) and the
        # decode loop's own advance read and write it. A second copy would drift:
        # `forward` arms the word from the host, and `step_positions` advances it
        # on device for every replayed step.
        self.pos_i = torch.zeros(1, dtype=torch.int32, device=d)
        self.full_klen = torch.zeros(1, dtype=torch.int32, device=d)
        self.slide_klen = torch.zeros(1, dtype=torch.int32, device=d)
        self.next_token = torch.zeros(1, dtype=torch.int64, device=d)
        self.tokens_out = torch.zeros(R, dtype=torch.int64, device=d)
        self._loop_graph = None
        self._loop_steps = 0

        # NVFP4 activation scratch, one (packed, sf) pair per distinct K
        self.act = {}
        for name, k in (("qkv", c.hidden_size), ("o", c.q_dim),
                        ("down", c.intermediate_size)):
            self.act[name] = (
                zbuf(A, k // 2, dtype=torch.uint8),
                torch.zeros(_sf_bytes(A, k), dtype=torch.uint8, device=d),
            )

        self._capture_stream = None
        self._in_loop = False

        self.q_row = c.q_dim          # elements per row of q_buf / o_raw
        self.kv_row = c.kv_dim        # elements per row of the KV caches

    # ── RoPE tables ──────────────────────────────────────────────────────
    def _build_rope(self) -> None:
        """fp32 cos/sin tables, one pair per layer type.

        The checkpoint's `compute_rope_cos_sin` builds ``outer(pos, inv_freq)``
        and then concatenates it with itself, so ``cos[j] == cos[j + rd/2]`` and
        only half needs storing. Tables stay fp32 because the reference rotates
        in fp32 and rounds the *result* to bf16, not the table.
        """
        c = self.cfg
        self.rope = {}
        for lt in ("full_attention", "sliding_attention"):
            rd = c.rope_dim(lt)
            inv = 1.0 / (c.rope_theta(lt)
                         ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
            freqs = torch.outer(torch.arange(self.max_seq, dtype=torch.float64), inv)
            self.rope[lt] = (freqs.cos().float().contiguous().cuda(),
                             freqs.sin().float().contiguous().cuda())

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _p(t: torch.Tensor) -> int:
        return t.data_ptr()

    def _gemm(self, lin, act_name: str, out: torch.Tensor, rows: int) -> None:
        """One NVFP4 W4A4 GEMM, dispatching on the row count.

        FlashRT ships two kernels for this: a hand-written single-row MMA
        (`full_n`, grid over 8-column N tiles) that is the decode hot path, and
        a CUTLASS multi-row GEMM for prefill. Both take packed E2M1 operands
        with swizzled UE4M3 block scales, and row 0 of the multi-row kernel
        matches the single-row kernel bit for bit, so prefill and decode apply
        the same arithmetic.

        Decode picks between the two single-row kernels per shape. `full_n`
        launches ONE warp per block over N/8 blocks -- 320 warps at N=2560 on a
        36-SM part -- so on the narrow-N shapes it cannot keep enough of the
        weight stream in flight: measured cold it runs at 282-325 GB/s where a
        plain reduction over the same bytes reaches 297-362. FlashRT's own
        `warpsplit` kernel covers the same 8-column strip but splits K across
        `warps` warps inside the block (grid = N/8, so `warps` times the
        blocks) and reduces the partials in shared memory with no cross-kernel
        intermediate, which is what the decode shapes need. The pair below is
        the per-shape winner of a cold sweep over warps in {2,4,8} x stages in
        {3,4,6}; gate_up is absent because its gated epilogue is already at the
        machine's sustained rate, and the wide lm_head shape needs no help.
        """
        packed, sf = self.act[act_name]
        if rows == 1:
            wp, st = _DECODE_SPLIT.get((lin.n, lin.k), (0, 0))
            if wp:
                fvk.fp4_w4a4_mma_sm120_warpsplit_bf16out(
                    self._p(packed), self._p(lin.packed), self._p(out),
                    lin.n, lin.k, self._p(sf), self._p(lin.sf), lin.alpha,
                    wp, st, self.stream)
            else:
                fvk.fp4_w4a4_mma_sm120_full_n_bf16out(
                    self._p(packed), self._p(lin.packed), self._p(out),
                    lin.n, lin.k, self._p(sf), self._p(lin.sf), lin.alpha,
                    self.stream)
        else:
            fvk.fp4_w4a16_gemm_sm120_bf16out(
                self._p(packed), self._p(lin.packed), self._p(out),
                rows, lin.n, lin.k, self._p(sf), self._p(lin.sf), lin.alpha,
                self.stream)

    def _attention(self, i: int, rows: int, pos: int) -> None:
        """Prefill attention. Reads the linear cache at absolute positions."""
        c = self.cfg
        lin, _, lin8 = self.kv_offset[i]
        if c.layer_types[i] == "sliding_attention":
            # One call for the whole prompt. FA2's mask can express the window
            # directly: with Is_causal false and window_right = 0 over
            # seqlen_k == seqlen_q, Mask clamps the right edge to row+1 (so it is
            # causal) and the left edge to row - window_left, which is exactly
            # [q-W+1, q]. FlashRT pins both window fields to -1, so the model
            # applies a small additive patch to its FA2 wrapper
            # (scripts/apply_flashrt_window_patch.py) that exposes them on the
            # non-causal entry -- FA2 statically forbids Is_causal && Is_local,
            # so the causal entry cannot carry the window at all.
            W = c.sliding_window
            # The keys these rows need are [pos - W + 1, pos + rows - 1], which
            # is `rows + W - 1` long, intersected with everything written so
            # far. The cache is one mirrored period wide, so that run is
            # contiguous from slot (pos+rows-klen) mod lin_w whatever the
            # position. FA2's local mask is bottom-right aligned, so
            # window_left = W-1 over a key length of `klen` selects exactly
            # [q-W+1, q] for each query. Reading past what was written would
            # feed the attention zeros, so `klen` never exceeds `total`.
            total = pos + rows
            klen = min(total, rows + W - 1)
            s0 = (total - klen) % self.lin_w
            kbase = lin + s0 * self.kv_row
            fa2.fwd_bf16_window(
                self._p(self.q_buf), self._p(self.k_cache) + kbase * 2,
                self._p(self.v_cache) + kbase * 2, self._p(self.o_raw),
                self._p(self.lse), 0, 0,
                1, rows, klen,
                c.num_attention_heads, c.num_key_value_heads, c.head_dim,
                (rows * self.q_row, self.q_row, c.head_dim),
                (klen * self.kv_row, self.kv_row, c.head_dim),
                (klen * self.kv_row, self.kv_row, c.head_dim),
                (rows * self.q_row, self.q_row, c.head_dim),
                1.0 / math.sqrt(c.head_dim), W - 1, 0, 0, self.stream)
        elif self.kv8_only:
            klen = pos + rows
            # Expand this layer's E4M3 prefix into the shared staging buffer and
            # run FA2 against that. ~4.4 ms per layer at 262k against a 131 s
            # TTFT.
            s_off = (lin8 // c.kv_dim) * c.num_key_value_heads
            sk.kv_dequant_bf16(self._p(self.k8_cache) + lin8,
                               self._p(self.k8_scale) + s_off,
                               self._p(self.k_stage), klen,
                               c.num_key_value_heads, c.head_dim, self.stream)
            sk.kv_dequant_bf16(self._p(self.v8_cache) + lin8,
                               self._p(self.v8_scale) + s_off,
                               self._p(self.v_stage), klen,
                               c.num_key_value_heads, c.head_dim, self.stream)
            self._fa2_stage(i, rows, klen)
        else:
            klen = pos + rows
            self._fa2_lin(i, lin, 0, 0, 0, 1, rows, klen, rows > 1,
                          rows * self.q_row, klen * self.kv_row)

    def _fa2_stage(self, layer_i, rows: int, klen: int) -> None:
        """FA2 prefill over the dequantised staging buffers (kv8-only mode)."""
        c = self.cfg
        (fa2.fwd_bf16_causal if rows > 1 else fa2.fwd_bf16)(
            self._p(self.q_buf), self._p(self.k_stage), self._p(self.v_stage),
            self._p(self.o_raw), self._p(self.lse), 0, 0,
            1, rows, klen,
            c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            (rows * self.q_row, self.q_row, c.head_dim),
            (klen * self.kv_row, self.kv_row, c.head_dim),
            (klen * self.kv_row, self.kv_row, c.head_dim),
            (rows * self.q_row, self.q_row, c.head_dim),
            1.0 / math.sqrt(c.head_dim), 0, self.stream)

    def _fa2_lin(self, layer_i, lin, q_off, o_off, k_off, batch, seqlen_q,
                 seqlen_k, causal, q_batch_stride, k_batch_stride) -> None:
        c = self.cfg
        fa2_fn = fa2.fwd_bf16_causal if causal else fa2.fwd_bf16
        fa2_fn(self._p(self.q_buf) + q_off * self.q_row * 2,
               self._p(self.k_cache) + (lin + k_off * self.kv_row) * 2,
               self._p(self.v_cache) + (lin + k_off * self.kv_row) * 2,
               self._p(self.o_raw) + o_off * self.q_row * 2,
               self._p(self.lse), 0, 0,
               batch, seqlen_q, seqlen_k,
               c.num_attention_heads, c.num_key_value_heads, c.head_dim,
               (q_batch_stride, self.q_row, c.head_dim),
               (k_batch_stride, self.kv_row, c.head_dim),
               (k_batch_stride, self.kv_row, c.head_dim),
               (q_batch_stride, self.q_row, c.head_dim),
               1.0 / math.sqrt(c.head_dim), 0, self.stream)

    def _attention_decode(self, i: int) -> None:
        """Decode attention with a device-side key length and a fixed pointer.

        Sliding layers read the W-slot ring from its base: each slot holds the
        most recent position with that residue, so the ring's W entries are
        exactly the last W positions, and a decode query has no causal mask over
        them -- attention over the set is the sliding window at any position.
        Full layers read the growing linear cache from its base with a
        device-side length.
        """
        c = self.cfg
        lin, ring, lin8 = self.kv_offset[i]
        if ring is None:
            base, seq_host, seq_dev = lin8, self.max_seq, self.full_klen
            nsplit = self.attn_nsplit
        else:
            base, seq_host, seq_dev = ring, self.ring_w, self.slide_klen
            nsplit = self.attn_nsplit_slide
        if self.attn_impl == "native":
            self._attention_decode_native(i, base, seq_host, seq_dev, nsplit,
                                          ring is None)
            return

        # A/B reference only: the shipped path returns above, and the frontend
        # does not expose `attn_impl`, so this is reached only when a caller
        # asks for it explicitly. It is what the native two-pass attention was
        # measured against (docs/spark_x25_rtx.md); the two agree to 7.6e-06 on
        # attention outputs of magnitude 0.60 (cos = 0.9999999), so the gap is
        # the split reduction's fp32 association rather than a different result.
        # `split_kv_sms > 0` selects the split-KV entry, 0 the plain one.
        args = (
            self._p(self.q_buf), self._p(self.k_cache) + base * 2,
            self._p(self.v_cache) + base * 2, self._p(self.o_raw),
            self._p(self.lse), self._p(seq_dev))
        strides = (
            (self.q_row, self.q_row, c.head_dim),
            (self.kv_row, self.kv_row, c.head_dim),
            (self.kv_row, self.kv_row, c.head_dim),
            (self.q_row, self.q_row, c.head_dim))
        common = (1, 1, seq_host, c.num_attention_heads, c.num_key_value_heads,
                  c.head_dim) + strides + (1.0 / math.sqrt(c.head_dim),)
        if ring is None and self.split_kv_sms > 0:
            fa2.fwd_bf16_seqused_splitkv(
                *args, self._p(self.lse_accum), self._p(self.o_accum),
                *common, self.split_kv_sms, self.stream)
        else:
            fa2.fwd_bf16_seqused(*args, *common, 0, self.stream)

    def _attention_decode_native(self, i: int, base: int, seq_host: int,
                                 seq_dev, nsplit: int, full: bool) -> None:
        """Decode attention over this repo's own kernels.

        Five launches: shift the softmax state, score every key, exponentiate
        and sum, weight the V rows per split, then reduce and normalise. Every
        one is a plain streaming pass, which is the point -- the FA2 entry this
        replaces reaches 154 GB/s at the 131k shape against this part's 425.8
        GB/s ceiling because a single decode query row only gives it a
        1 x kv_heads x 1 grid, four blocks on a 36-SM part. At 131 072 keys
        this chain runs the same layer in 1.40 ms against 3.48 ms, at 90% of the
        ceiling, and agrees with the fp32 oracle to 1.2e-05.
        """
        c = self.cfg
        hd = c.head_dim
        nchunk = max(1, min(self.attn_nchunk, (seq_host + 511) // 512))
        if full:
            k = self._p(self.k8_cache) + base
            v = self._p(self.v8_cache) + base
            s_off = (base // c.kv_dim) * c.num_key_value_heads
            ks = self._p(self.k8_scale) + s_off
            vs = self._p(self.v8_scale) + s_off
            kv8 = 1
        else:
            k = self._p(self.k_cache) + base * 2
            v = self._p(self.v_cache) + base * 2
            ks = vs = 0
            kv8 = 0
        sk.attn_state_init_bf16(self._p(self.attn_max), self._p(self.attn_sum),
                                c.num_attention_heads, self.stream)
        sk.attn_scores_bf16(
            self._p(self.q_buf), k, ks, self._p(self.attn_s), self._p(self.attn_max),
            self._p(seq_dev), seq_host, self.max_seq, c.num_attention_heads,
            c.num_key_value_heads, hd, c.num_kv_groups, 1.0 / math.sqrt(hd),
            kv8, self.stream)
        sk.attn_softmax_bf16(
            self._p(self.attn_s), self._p(self.attn_p), self._p(self.attn_max),
            self._p(self.attn_sum), self._p(seq_dev), seq_host, self.max_seq,
            c.num_attention_heads, nchunk, self.stream)
        sk.attn_pv_bf16(
            v, vs, self._p(self.attn_p), self._p(self.attn_op), self._p(seq_dev),
            self.max_seq, c.num_attention_heads, c.num_key_value_heads, hd,
            c.num_kv_groups, nsplit, kv8, self.stream)
        sk.attn_pv_combine_bf16(
            self._p(self.attn_op), self._p(self.attn_sum), self._p(self.o_raw),
            nsplit, c.num_attention_heads, hd, self.stream)

    def _boundary_norm(self, h_in, x, h_post, rms_w, rows: int) -> None:
        """Residual add + RMSNorm + NVFP4 quantize for one sub-block boundary.

        FlashRT's v2 entry covers the prefill rows. The single decode row is a
        different problem -- 73 launches per token, ~20 KB each -- and v2's
        shape (one block of 256 threads walking the row in 16 strided scalar
        steps, staging the normalized row through shared memory) costs more than
        the bytes justify, so this repository's kernel covers it. The two are
        byte-compatible: tests/test_spark_x25_kernels.py compares the packed E2M1 nibbles
        and the swizzled UE4M3 scales against v2 directly, at three input
        magnitudes, and they match exactly. So prefill and decode hand the same
        activation format to the same GEMM.
        """
        c = self.cfg
        act_p, act_sf = self.act["qkv"]
        if rows == 1:
            sk.residual_add_rms_norm_to_nvfp4_bf16(
                self._p(h_in), self._p(x), self._p(h_post), self._p(rms_w),
                self._p(act_p), self._p(act_sf), c.hidden_size,
                c.rms_norm_eps, self.stream)
        else:
            fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16_v2(
                self._p(h_in), self._p(x), self._p(h_post), self._p(rms_w),
                self._p(act_p), self._p(act_sf), rows, c.hidden_size,
                c.rms_norm_eps, self.stream)

    # ── one decoder layer ────────────────────────────────────────────────
    def _layer(self, i: int, rows: int, pos: int, prev_mlp: torch.Tensor) -> None:
        c = self.cfg
        lw = self.w.layers[i]
        lt = c.layer_types[i]
        eps = c.rms_norm_eps
        st = self.stream
        R = rows

        # ── attention half ──
        # h_tmp = h_res + prev_mlp, then RMSNorm + NVFP4 quant -> QKV activation
        self._boundary_norm(self.h_res[:R], prev_mlp, self.h_tmp[:R],
                            lw.input_norm, R)

        # bf16 normalized row, needed only by the tiny attention output gate.
        # Fusing this + g_proj into the boundary kernel above was implemented and
        # measured: the merged kernel becomes latency-bound (one block per row)
        # and costs 21.1 us against 13.7 us for the three separate launches, so
        # decode drops from 119.3 to 115.8 tok/s. Kept separate deliberately.
        fvk.rms_norm(self._p(self.h_tmp[:R]), self._p(lw.input_norm),
                     self._p(self.h_norm[:R]), R, c.hidden_size, eps, st)
        sk.gproj_bf16(self._p(self.h_norm[:R]), self._p(self.w.g_proj_bf16[i]),
                      self._p(self.gate[:R]), R, c.num_attention_heads,
                      c.hidden_size, st)

        self._gemm(lw.qkv, "qkv", self.qkv[:R], R)

        cos_t, sin_t = self.rope[lt]
        # The KV caches are one flat (layer, position, head, dim) buffer, so the
        # write must be offset to this layer's slice -- the same offset the
        # attention call below reads from.
        lin, ring_off, lin8 = self.kv_offset[i]
        # Multi-row writes into a ring would race (position p and p+W share a
        # slot and nothing orders the blocks), so prefill writes only the linear
        # cache and seeds the ring afterwards; single-row decode writes directly.
        write_ring = (R == 1) and (ring_off is not None)
        rp = (self._p(self.k_cache) + ring_off * 2) if write_ring else 0
        rv = (self._p(self.v_cache) + ring_off * 2) if write_ring else 0
        # Full layers always write E4M3 for decode; they write bf16 as well only
        # in mirror mode, where prefill reads it. One scale per (row, KV head).
        kb = vb = 0
        if ring_off is None:
            s_off = (lin8 // c.kv_dim) * c.num_key_value_heads
            k8 = self._p(self.k8_cache) + lin8
            v8 = self._p(self.v8_cache) + lin8
            k8s = self._p(self.k8_scale) + s_off
            v8s = self._p(self.v8_scale) + s_off
            if lin is not None:
                kb = self._p(self.k_cache) + lin * 2
                vb = self._p(self.v_cache) + lin * 2
        else:
            kb = self._p(self.k_cache) + lin * 2
            vb = self._p(self.v_cache) + lin * 2
            k8 = v8 = k8s = v8s = 0
        sk.qkv_post_rope_kvwrite_bf16(
            self._p(self.qkv[:R]), self._p(cos_t), self._p(sin_t),
            self._p(self.q_buf[:R]), kb, vb,
            rp, rv, k8, v8, k8s, v8s,
            R, c.num_attention_heads, c.num_key_value_heads, c.head_dim,
            c.rope_dim(lt), self._p(self.pos_i), self.ring_w,
            self.lin_w if ring_off is not None else 0, st)

        if R == 1 and self._in_loop:
            self._attention_decode(i)
        else:
            self._attention(i, R, pos)

        # gate and pack in one pass: the gated row is out_proj's activation, so
        # quantizing it here removes a separate launch over 4096 elements
        sk.attn_out_gate_to_nvfp4_bf16(
            self._p(self.o_raw[:R]), self._p(self.gate[:R]), 0,
            self._p(self.act["o"][0]), self._p(self.act["o"][1]),
            R, c.num_attention_heads, c.head_dim, st)
        self._gemm(lw.o_proj, "o", self.o_proj[:R], R)

        # ── MLP half ──
        # h_res = h_tmp + o_proj, then RMSNorm + NVFP4 quant -> gate/up activation
        self._boundary_norm(self.h_tmp[:R], self.o_proj[:R], self.h_res[:R],
                            lw.post_attn_norm, R)

        inter = c.intermediate_size
        if R == 1:
            # Decode: the gated product rides the GEMM epilogue. The kernel's
            # output is bit-identical to gate_up GEMM + gelu_mul here (verified
            # in tests/test_spark_x25_kernels.py) and costs 63.5 us against 67.7 us.
            fvk.fp4_w4a4_mma_sm120_gated_geglu_fp4out(
                self._p(self.act["qkv"][0]), self._p(lw.gate_up_il.packed),
                self._p(self.act["down"][0]), self._p(self.act["down"][1]),
                self._p(self.act["qkv"][1]), self._p(lw.gate_up_il.sf),
                lw.gate_up_il.alpha, 2 * inter, c.hidden_size, st)
            self._gemm(lw.down, "down", self.down[:R], R)
            return
        self._gemm(lw.gate_up, "qkv", self.gate_up[:R], R)
        ld = 2 * inter
        sk.gelu_mul_to_nvfp4_swizzled_bf16(
            self._p(self.gate_up), self._p(self.gate_up) + inter * 2,
            self._p(self.act["down"][0]), self._p(self.act["down"][1]),
            R, inter, ld, ld, st)

        self._gemm(lw.down, "down", self.down[:R], R)

    # ── CUDA Graph decode loop ───────────────────────────────────────────
    def _decode_iteration(self, first: bool) -> None:
        """One decode step whose only variable inputs live in device memory.

        The position, both attention key lengths, the token fed to the embedding
        lookup and the token sampled out are all device values, so the launch
        descriptors are identical for every step and the whole loop captures
        into a single graph.
        """
        c = self.cfg
        # Bind to the current stream, as `forward` does: this is what puts every
        # launch of the step inside the capture region.
        self.stream = torch.cuda.current_stream().cuda_stream
        sk.step_positions_bf16(
            self._p(self.pos_i), self._p(self.full_klen), self._p(self.slide_klen),
            self._p(self.tokens_out), self._p(self.next_token),
            self.ring_w, 1 if first else 0, self.stream)
        fvk.embedding_lookup_bf16(self._p(self.next_token), self._p(self.w.embed),
                                  self._p(self.h_res), 1, c.hidden_size, self.stream)
        prev = self.zero[:1]
        for i in range(c.num_hidden_layers):
            self._layer(i, 1, 0, prev)
            prev = self.down[:1]
        self._boundary_norm(self.h_res[:1], prev, self.h_norm[:1],
                            self.w.final_norm, 1)
        self._gemm(self.w.lm_head, "qkv", self.logits[:1], 1)
        # Greedy sampling on device: no CPU synchronization anywhere in the loop.
        # This repo's argmax, not FlashRT's: qwen36_argmax_bf16 scans with a
        # scalar stride, which leaves one 2-byte load in flight per warp and
        # costs 42 us/token to read 262 KB (6 GB/s). See spark_kernels.cu.
        sk.argmax_bf16(self._p(self.logits), self._p(self.next_token),
                       c.vocab_size, self.stream)

    def _capture_state_views(self, steps: int) -> list[torch.Tensor]:
        """Persistent slots capture can touch when started at position zero.

        Save only the affected prefix/ring slots, not the entire long-context
        cache. Scratch activations are overwritten by each decode iteration.
        """
        views = [self.pos_i, self.full_klen, self.slide_klen,
                 self.next_token, self.tokens_out[:steps]]
        width = self.cfg.kv_dim
        for lin, ring, lin8 in self.kv_offset:
            if ring is not None:
                count = min(steps, self.lin_w) * width
                for cache in (self.k_cache, self.v_cache):
                    views.extend((cache[lin:lin + count],
                                  cache[lin + self.lin_w * width:
                                        lin + self.lin_w * width + count],
                                  cache[ring:ring + min(steps, self.ring_w) * width]))
            else:
                if lin is not None:
                    views.extend(cache[lin:lin + steps * width]
                                 for cache in (self.k_cache, self.v_cache))
                views.extend(cache[lin8:lin8 + steps * width]
                             for cache in (self.k8_cache, self.v8_cache))
                offset = lin8 // width * self.cfg.num_key_value_heads
                count = steps * self.cfg.num_key_value_heads
                views.extend(cache[offset:offset + count]
                             for cache in (self.k8_scale, self.v8_scale))
        return views

    def capture_decode_loop(self, steps: int) -> torch.cuda.CUDAGraph:
        """Capture without changing the caller's KV, token or position state."""
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= self.max_seq:
            raise ValueError("decode steps must be in [1, max_seq]")
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream()
        s = self._capture_stream
        cur = torch.cuda.current_stream()
        s.wait_stream(cur)
        self._in_loop = True
        try:
            with torch.cuda.stream(s):
                views = self._capture_state_views(steps)
                saved = [value.clone() for value in views]
                try:
                    self.pos_i.zero_()
                    for _ in range(2):
                        self._decode_iteration(first=True)
                    self.pos_i.zero_()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=s):
                        for k in range(steps):
                            self._decode_iteration(first=(k == 0))
                finally:
                    for value, original in zip(views, saved):
                        value.copy_(original)
        finally:
            self._in_loop = False
            cur.wait_stream(s)
        self._loop_steps = steps
        return g

    def decode_loop(self, pos: int, steps: int) -> torch.Tensor:
        """Replay `steps` decode iterations starting at absolute position `pos`.

        Consumes ``next_token`` at ``pos`` and returns its ``steps`` successor
        predictions. The input token itself is not part of the returned tensor.
        """
        if (not isinstance(pos, int) or isinstance(pos, bool) or pos < 0
                or not isinstance(steps, int) or isinstance(steps, bool)
                or steps < 1 or pos + steps > self.max_seq):
            raise ValueError("decode positions must fit within max_seq with positive steps")
        if self._loop_graph is None or self._loop_steps != steps:
            self._loop_graph = self.capture_decode_loop(steps)
        self.pos_i.fill_(pos)
        self._loop_graph.replay()
        # Each step records the token the *previous* step produced, so after N
        # steps tokens_out holds the first N-1 of them and the last still sits in
        # next_token.
        if steps == 1:
            return self.next_token.clone()
        return torch.cat([self.tokens_out[pos:pos + steps - 1], self.next_token])

    def _seed_rings(self, end_pos: int) -> None:
        """Copy the surviving window from each sliding layer's linear cache into
        its ring after a prefill, so decode starts with a valid ring."""
        c = self.cfg
        W = self.ring_w
        count = min(end_pos, W)
        if count <= 0:
            return
        base = end_pos - count
        for i in range(c.num_hidden_layers):
            lin, ring, _ = self.kv_offset[i]
            if ring is None:
                continue
            sk.seed_ring_bf16(
                self._p(self.k_cache) + lin * 2, self._p(self.v_cache) + lin * 2,
                self._p(self.k_cache) + ring * 2, self._p(self.v_cache) + ring * 2,
                base, count, c.kv_dim, W, self.lin_w, self.stream)

    # ── entry points ─────────────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor, pos: int = 0) -> torch.Tensor:
        """Run ``input_ids`` (contiguous, absolute position ``pos``) -> logits.

        The prompt is walked in chunks of ``prefill_chunk`` rows. Only the last
        chunk feeds the lm_head, because the only logits a caller reads are the
        final position's; the rest of the chunk's hidden states exist solely to
        fill the KV cache. Chunking is what lets the activation working set be
        sized by the chunk instead of by the context budget -- it changes no
        arithmetic, since every chunk attends over the same absolute key range
        it would have in a single pass.
        """
        # Bind to whichever stream is current at entry. Every kernel below takes
        # a raw stream handle, so this is what makes the whole step land inside a
        # CUDA Graph capture -- and what makes replay follow the caller's stream.
        self.stream = torch.cuda.current_stream().cuda_stream
        rows = int(input_ids.numel())
        # Raises rather than asserts: the buffers below are sized from these
        # two bounds and `python -O` would strip an assert.
        if rows > self.prefill_cap:
            raise ValueError(
                f"prefill: prompt of {rows} rows exceeds prefill_cap="
                f"{self.prefill_cap}; raise it in the constructor or chunk the "
                "prompt yourself")
        if pos + rows > self.max_seq:
            raise ValueError(
                f"prefill: positions [{pos}, {pos + rows}) exceed max_seq="
                f"{self.max_seq}; raise max_seq in the constructor")
        ids = input_ids.to(torch.int64).contiguous()

        A = self.act_rows
        out = None
        for start in range(0, rows, A):
            n = min(A, rows - start)
            out = self._prefill_chunk(ids[start:start + n], pos + start, n,
                                      last=(start + n == rows))

        # Sliding layers keep a linear cache that prefill's per-query window
        # reads; seed their rings from its surviving tail once the whole prompt
        # is in, so decode starts with a valid ring.
        self._seed_rings(pos + rows)
        return out

    def _prefill_chunk(self, ids: torch.Tensor, pos: int, rows: int,
                       *, last: bool) -> torch.Tensor | None:
        """One chunk of the prefill: rows r = 0..rows-1 sit at ``pos + r``."""
        c = self.cfg
        # Arm the decode position from the host. Prefill rows r sit at absolute
        # positions pos + r, which the KV write derives from this word; the
        # captured decode loop advances the same word on device.
        # A native kernel, not a framework fill: the equal-scope profile must
        # show zero ATen ops.
        sk.set_int32(self._p(self.pos_i), int(pos), self.stream)

        fvk.embedding_lookup_bf16(self._p(ids), self._p(self.w.embed),
                                  self._p(self.h_res), rows, c.hidden_size, self.stream)

        prev_mlp = self.zero[:rows]
        for i in range(c.num_hidden_layers):
            self._layer(i, rows, pos, prev_mlp)
            prev_mlp = self.down[:rows]

        if not last:
            return None

        # `_layer` leaves the *post-attention* residual in h_res and the layer's
        # MLP output in self.down, so the final hidden state is their sum. The
        # fused residual-add + RMSNorm does both in one launch.
        fvk.residual_add_rms_norm_to_nvfp4_swizzled_bf16_v2(
            self._p(self.h_res[:rows]), self._p(prev_mlp), self._p(self.h_norm[:rows]),
            self._p(self.w.final_norm),
            self._p(self.act["qkv"][0]), self._p(self.act["qkv"][1]),
            rows, c.hidden_size, c.rms_norm_eps, self.stream)
        # NVFP4 lm_head. FlashRT's bf16 matmul is M-major: a prefill of S rows
        # re-streams the whole 671 MB table once per row (S x 671 MB of traffic).
        # The block-scaled GEMM tiles over M, so the same table is read once.
        self._gemm(self.w.lm_head, "qkv", self.logits[:rows], rows)
        return self.logits[:rows]

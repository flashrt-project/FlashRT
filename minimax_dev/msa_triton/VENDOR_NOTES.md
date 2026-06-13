# MiniMax-M3 block-sparse attention Triton kernels — vendor notes

Standalone, sm_121-ready copy of the MiniMax-M3 ("MSA") lightning-indexer +
block-sparse GQA attention Triton kernels, decoupled from SGLang so they import
with **only `torch` + `triton`** (CUDA). Drop-in candidate for the FlashRT
runtime's `> 2048`-ctx sparse-attention path on the DGX Spark (GB10, sm_121).

Triton is architecture-agnostic (it lowers through LLVM/PTX), so these compile on
sm_121 unmodified. This is the gap-filler for MiniMax-AI's official MSA kernels,
which are SM100-only (they use `tcgen05`).

---

## 1. Provenance

**Primary source — SGLang PR #27944 "Support MiniMax-M3"** (author XunhaoLai,
NSA-triton; these are the better-tested kernels and ship a naive reference).

- PR head repo/ref: `JustinTong0323/sglang` @ `minimax-m3-upstream`
- PR head commit: `cbe1ffddcc8f31da7f6f805c6833df5e511a578f`
- Upstream path: `python/sglang/srt/layers/attention/minimax_sparse_ops/`
- License: Apache-2.0. Files carry `# Copyright 2025 XunhaoLai. All rights
  reserved.` headers, preserved verbatim.

| vendored file | upstream file | role |
|---|---|---|
| `kernels/prefill/flash_with_topk_idx.py` | `prefill/flash_with_topk_idx.py` | **lightning indexer (prefill)** — dense flash attn that emits per-block scores, then a bitonic top-k kernel selects block ids. `flash_prefill_with_topk_index` |
| `kernels/prefill/topk_sparse.py` | `prefill/topk_sparse.py` | **block-sparse GQA attention (prefill)** — `flash_prefill_with_gqa_share_sparse` |
| `kernels/decode/flash_with_topk_idx.py` | `decode/flash_with_topk_idx.py` | **lightning indexer (decode, split-K)** — `flash_decode_with_topk_idx` |
| `kernels/decode/topk_sparse.py` | `decode/topk_sparse.py` | **block-sparse GQA attention (decode, split-K)** — `flash_decode_with_gqa_share_sparse` |
| `kernels/common/utils.py` | `common/utils.py` | `get_cu_seqblocks`, `robust_allocator`, `tensor_cache` (torch+triton only) |
| `kernels/naive/topk_sparse.py` | `naive/topk_sparse.py` | PyTorch reference: sparse attn given `topk_idx` |
| `kernels/naive/flash_with_topk_idx.py` | `naive/flash_with_topk_idx.py` | PyTorch reference: joint indexer + attn (rewritten to drop `einops`) |
| `kernels/_compat.py` | *(new)* | `is_hip()` / `envs` stubs replacing the two stripped framework imports |

> Note: the SGLang indexer top-k selection lives in
> `prefill/flash_with_topk_idx.py` (`_topk_index_kernel`, bitonic) and
> `decode/flash_with_topk_idx.py` (`_topk_index_partial_kernel` +
> `_topk_index_merge_kernel`), **not** in `topk_sparse.py` — `topk_sparse.py` is
> the attention kernel that *consumes* the selected block ids. The task brief had
> these two swapped; the table above reflects the actual upstream layout.

**Secondary source — vLLM PR #45381 "[Model] Add MiniMax M3 support"**
(same M3 semantics, independent impl). Kept under `_src_vllm/` for cross-check,
**not** wired into the package.

- PR head repo/ref: `vllm-project/vllm` @ `m3_release`
- PR head commit: `0c6d468d5c470d2797a1a86a4d079e589d25b42f`
- Files: `vllm/model_executor/models/minimax_m3/common/ops/index_topk.py`
  (`minimax_m3_index_score`, `minimax_m3_index_topk`),
  `.../common/ops/sparse_attn.py` (`minimax_m3_sparse_attn`,
  `minimax_m3_sparse_attn_decode`), `common/indexer.py`,
  `common/sparse_attention.py`. License: Apache-2.0.
  vLLM differs from SGLang in that it **forces KV page_size == sparse block_size
  (128)** so one sparse block maps to exactly one page, and the index-K cache is
  laid out `(num_blocks, 128, idx_head_dim)`.

**Standalone fallback (not vendored):**
`github.com/XunhaoLai/native-sparse-attention-triton`
(`ops/triton/topk_sparse_attention.py`) — the non-paged NSA origin of these
kernels; useful if a paged-cache-free variant is wanted later.

**Raw `_src_*/` staging dirs** (`_src_sglang/`, `_src_vllm/`) hold the untouched
fetched files for audit/diff. The shipped package is `kernels/`.

---

## 2. Framework coupling that was stripped

Only three import sites touched the framework; all replaced via `kernels/_compat.py`:

1. `from sglang.srt.utils import is_hip` → `from .._compat import is_hip`
   (in `prefill/topk_sparse.py`, `decode/topk_sparse.py`). `is_hip()` returns
   `torch.version.hip is not None` → **False on CUDA sm_121**, so the `IS_FP8`
   paged-KV widening branch (a HIP-only path for `--kv-cache-dtype fp8_*`) stays
   off and the CUDA dtype contract (K/V cache dtype == Q dtype) is enforced
   exactly as upstream.

2. `from sglang.srt.environ import envs` → `from .._compat import envs`
   (in `decode/flash_with_topk_idx.py`). The stub exposes
   `SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = False`, which routes decode top-k
   through the **pure-Triton 2-stage fallback** instead of a CUDA JIT radix kernel.

3. Two deferred `from sglang.jit_kernel.minimax_decode_topk import …` inside
   `decode/flash_with_topk_idx.py` (the `use_jit_topk` and `use_dense_main_attn`
   fast paths) → replaced with `raise NotImplementedError(...)`. These require an
   un-vendored CUDA kernel (`minimax_decode_topk.cuh`) and are **off by default**
   (`use_dense_main_attn=False`, radix env False), so they are never reached on
   the default Triton path. If you want them later, vendor that `.cuh` too.

`kernels/naive/flash_with_topk_idx.py` was additionally rewritten to drop
`einops` (the only third-party dep beyond torch/triton): `einops.einsum/rearrange`
→ `torch.einsum`/`torch.reshape`, logic unchanged (originals kept in comments).

No `.cu`/binding/CMake changes; everything is additive Python under `kernels/`.

---

## 3. Exact call interface

All four Triton entrypoints operate on a **paged KV cache**. Common layout:

```
k_cache / v_cache : [max_slots, num_kv_heads, head_dim]   (Q dtype: bf16/fp16; same dtype as q)
req_to_token      : [max_reqs, max_kv_len]  int32   logical-position -> physical-slot map
slot_ids          : [batch]                 int64   per-request row index into req_to_token
seq_lens          : [batch]                 int32   live KV length per request
topk_idx          : [num_kv_heads, n, topk] int32   selected block ids, valid ids LEFT-packed,
                                                     -1 RIGHT-padding   (the M3 indexer contract)
sink (optional)   : [num_q_heads, head_dim] bf16/fp16   per-head attention sink (softmax denom only)
sm_scale          : float, default head_dim ** -0.5
```

`num_q_heads % num_kv_heads == 0`; all q heads in a GQA group share one kv head's
`topk_idx[kh]` ("gqa_share"). `head_dim <= 256`. block ids index blocks of
`block_size` keys in *logical* position space (resolved to physical slots through
`req_to_token`).

### (a) Block-sparse GQA attention — decode
```python
o = flash_decode_with_gqa_share_sparse(
        q,            # [batch, num_q_heads, head_dim]
        sink,         # [num_q_heads, head_dim] or None
        k_cache, v_cache,         # [max_slots, num_kv_heads, head_dim]
        req_to_token, # [max_reqs, max_kv_len] int32
        seq_lens,     # [batch] int32
        slot_ids,     # [batch] int64
        block_size,   # int, power of 2 (M3: 128)
        topk_idx,     # [num_kv_heads, batch, topk] int32, -1 right-pad
        sm_scale=None, use_tma=True)
# -> o: [batch, num_q_heads, head_dim]   (split-K internally over the top-k blocks,
#       NUM_TOPK_CHUNKS auto-sized to ~256-CTA grid, merged by _merge_topk_attn_out_kernel)
```
Constraints: `triton.next_power_of_2(block_size) == block_size`.

### (b) Block-sparse GQA attention — prefill (causal, varlen-packed)
```python
o = flash_prefill_with_gqa_share_sparse(
        q,            # [total_q, num_q_heads, head_dim]  (varlen-packed across batch)
        k_cache, v_cache, sink,
        req_to_token, slot_ids,
        topk_idx,     # [num_kv_heads, total_q, topk] int32, -1 right-pad (per query token)
        block_size_q, # in {1,2,4,8,16,32,64};  gqa_group_size * block_size_q <= 128
        block_size_k, # in {16,32,64,128}  (M3: 128)
        cu_seqlens,   # [batch+1] int32  cumulative token counts
        seq_lens,     # [batch] int32
        prefix_lens,  # [batch] int32  (prefix already in cache; absolute query pos = prefix+local)
        max_seqlen_q, # int
        sm_scale=None, use_tma=True,
        cu_seqblocks_q=None, max_seqblock_q=None)   # auto-derived via get_cu_seqblocks
# -> o: [total_q, num_q_heads, head_dim]
```
Causal mask is applied per query position (`off_q_k >= c` against block start)
*in addition to* the block selection.

### (c) Lightning indexer — decode (q.k blockmax score -> top-k block ids)
```python
o, topk_idx, real_seq_lens = flash_decode_with_topk_idx(
        q,            # [batch, num_index_heads, index_head_dim]  (M3: 1 shared index head, D=128)
        sink,         # or None
        k_cache,      # [max_slots, num_index_kv_heads, index_head_dim]  (M3: 1 index kv head)
        v_cache,      # or None when disable_index_value=True (M3 default)
        req_to_token, seq_lens, max_seqlen, slot_ids,
        block_size,   # M3: 128
        topk,         # M3: 16
        init_blocks,  # forced-visible leading blocks (M3 uses local boost; see §5)
        local_blocks, # forced-visible trailing/local blocks
        sm_scale=None, use_tma=True, score_type="max",   # M3 uses "max" (blockmax)
        disable_index_value=True,                        # M3: indexer is selection-only, no value out
        use_dense_main_attn=False, page_size=1)          # keep False (Triton-only vendor)
# disable_index_value=True -> o is None; topk_idx: [num_index_heads, batch, topk] int32, -1 right-pad
#                              real_seq_lens is None unless use_dense_main_attn=True
```
`assert init_blocks + local_blocks <= topk`. With `score_type="max"` the per-block
score is the max over the block's per-key q.k scores — the M3 blockmax. The
returned `topk_idx` is exactly the format the attention kernels in (a)/(b)
consume.

### (d) Lightning indexer — prefill
```python
o, topk_idx = flash_prefill_with_topk_index(
        q, k_cache, v_cache, sink, req_to_token, slot_ids,
        cu_seqlens, seq_lens, prefix_lens, max_seqlen_q, max_seqlen_k,
        block_size_q, block_size_k, topk,
        init_blocks=1, local_blocks=2, sm_scale=None, use_tma=False,
        score_type="max", disable_index_value=False, ...)
# -> (o or None, topk_idx[num_index_heads, all_seqblock_q, topk] int32)
```

### Pipeline wiring (indexer output -> attention input)
```
topk_idx = flash_{decode,prefill}_with_topk_index(... score_type="max",
                                                   disable_index_value=True)[1]
o        = flash_{decode,prefill}_with_gqa_share_sparse(q_main, ..., topk_idx)
```
`topk_idx` is `[num_kv_heads, n, topk]` int32, valid block ids left-packed, `-1`
right-padded. The attention kernel `tl.sum(valid_idx != -1)`-counts the valid
entries then reads them sequentially, so duplicate ids would be double-counted —
selections **must be deduplicated** (the M3 indexer dedups via the local boost).

### PyTorch references (correctness oracle)
```python
naive_flash_decode_with_gqa_share_sparse(q, sink, kv_cache, seq_lens, slot_ids,
                                          block_size, topk_idx, sm_scale=None)
#   kv_cache here is the *contiguous* 5-D buffer [max_slots, 2(k/v), max_len, num_kv_heads, head_dim]
naive_flash_decode_with_topk_idx(q, sink, kv_cache, seq_lens, max_seqlen, slot_ids,
                                 block_size, topk, sm_scale=None,
                                 init_blocks=0, local_blocks=0)
#   joint indexer + attention -> (o, topk_idx)
```
`test_msa_standalone.py` also carries paged fp32 references
(`pytorch_sparse_gqa_decode_reference`, `pytorch_sparse_gqa_prefill_reference`)
that match the paged 3-D `k_cache`/`v_cache` + `req_to_token` layout of the Triton
kernels directly.

---

## 4. sm_121 (GB10 / consumer Blackwell) readiness

- **No SM100 intrinsics.** Unlike MiniMax-AI's official MSA (tcgen05, SM100-only),
  these kernels use only `tl.dot`, `tl.make_block_ptr`, `tl.exp2`, bitonic sort —
  all arch-agnostic Triton. Expected to compile and run on sm_121 unmodified.
- **No real TMA dependency.** `use_tma`/`USE_TMA` is plumbed as a constexpr but the
  kernel bodies address memory with `tl.make_block_ptr`, **not** TMA tensor
  descriptors. `common/utils.py` imports `tl.make_tensor_descriptor` and sets a
  `robust_allocator`, but it is dead weight on this path. If `make_tensor_descriptor`
  is unavailable in the installed Triton, the import has an
  `except: tl._experimental_make_tensor_descriptor` fallback; if both are missing,
  delete that try-block in `common/utils.py` (it is unused by these kernels).
- **`tl.dot` dtypes.** QK and PV `tl.dot`s run in the Q dtype (bf16/fp16) with fp32
  accumulation (`acc_o`/`qk` are fp32). bf16 `tl.dot` is well-supported on Blackwell.
  No fp8 `tl.dot` on the CUDA path (the fp8 widening branch is HIP-only and off here),
  so no sm_120/121 fp8-MMA scale-vec constraints apply.
- **Autotune grids may need pruning on consumer Blackwell.** The kernels autotune
  over `num_warps ∈ {2,4,8}` × `num_stages ∈ {2,3,4(,5)}` (decode attn adds ns=5;
  prefill indexer uses fixed 64/128 Q×K tiles). High `num_stages` × large
  `BLOCK_SIZE_K=128` can exceed the **smaller shared-memory budget** of GB10 vs
  datacenter Blackwell; Triton skips configs that fail to compile, so the autotuner
  *should* self-prune, but if you hit `OutOfResources` / long autotune, trim the
  `num_stages` upper end (drop 4/5) in the `@triton.autotune` configs of
  `decode/topk_sparse.py` and `prefill/flash_with_topk_idx.py`.
- **`triton.set_allocator(robust_allocator)`** is called inside each host wrapper
  (for TMA descriptor scratch). Harmless without TMA; keep it.
- **HIP/fp8 path** (`IS_FP8`) is compiled out on CUDA (`is_hip()==False`).

---

## 5. Mapping to our runtime's MSA semantics

Our reference indexer is `MiniMaxM3VLIndexer` in
`upstream/modular_minimax_m3_vl.py` (transformers). Config:
`index_n_heads=4`, `index_head_dim=128`, `index_block_size=128`,
`index_topk_blocks=16`, partial-RoPE first 64 dims, Gemma RMSNorm on q/k.
Main attention: GQA 64Q/4KV, head_dim 128, scale `128**-0.5`.

| our runtime (`MiniMaxM3VLIndexer.forward`) | vendored kernel equivalent |
|---|---|
| `score = q.float() @ k.float().T`, causal `masked_fill(future, -inf)` | `_decode_score_kernel` / `_flash_attn_fwd_with_block_score_kernel`, `score_type="max"` |
| `scores.view(...,num_blocks,block_size).amax(-1)` (blockmax over 128) **then `.amax(dim=1)` over the 4 index heads** | per-block `tl.max` with `score_type="max"`; **see GAP below re: head reduction** |
| `local_blocks` scatter `+inf` so local blocks always win | `local_blocks` arg (forced `LOCAL_SCORE=1e29`); `init_blocks` → `INIT_SCORE=1e30` |
| `block_scores.topk(16)`, `-1` where score `== -inf` | bitonic / 2-stage top-k → `topk_idx` left-packed, `-1` right-pad |
| `topk_indices.masked_fill(topk_scores==-inf, -1)` (right-pad) | identical `-1` right-pad contract |
| `build_block_mask` (eager SDPA path: dense `[B,1,Sq,Sk]` additive mask) | replaced by the block-sparse attention kernel that reads `topk_idx` directly (no dense mask materialized) |

This is a faithful match for the **per-(query, kv-head) block-max top-16 with
forced-local, `-1` right-padded** contract our runtime expects, and the
attention kernels consume that `topk_idx` with no dense mask.

### Gaps / caveats (be concrete before wiring into FlashRT)

1. **Index-head reduction (the main semantic gap).** Our runtime indexer has
   **4 index heads** and reduces blockmax across them (`.amax(dim=1)`) to one score
   per (query, block) *before* top-k. These kernels' indexer scores a **single
   index (kv) head** per `topk_idx` row. To reproduce M3 exactly you must either
   (a) run the indexer with `num_index_heads=1` after pre-reducing the 4 heads
   yourself, or (b) feed a 4-head index q/k and add a max-over-heads reduction
   (the kernels already take `num_q_heads`/`gqa` for the *main* attention, but the
   indexer's per-head max-pool reduction is over the block dim, not the head dim —
   confirm before relying on multi-head indexer scores). Simplest: compute the
   4-head blockmax fusion in a small pre-pass / our own kernel and call the
   indexer / attention with the reduced single-head selection.

2. **Paged KV cache assumed.** All four kernels require
   `k_cache/v_cache [max_slots, num_kv_heads, head_dim]` + `req_to_token` +
   `slot_ids`. **Our FlashRT M3 runtime does not yet have a paged cache** (HANDOFF
   §P3 plans a resident KV + expert-cache design, not a vLLM-style paged pool).
   To use these as-is you must build a `req_to_token` map. For a *contiguous*
   single-request cache this is trivial:
   `req_to_token = torch.arange(seq_len).view(1, -1)`, `slot_ids = [0]` — the test
   harness's prefill path does exactly this. The decode tests use a `randperm`
   slot map to exercise true paging.

3. **Right-padding only / slot-relative blocking.** Both these kernels and our
   runtime block on **absolute key slots**, so only **right-padding** is
   equivalent to an unpadded run (left-padding shifts block boundaries). Matches
   our runtime's documented limitation (`test_left_padding_compatibility` skipped).

4. **`disable_index_value` / no index residual.** M3's indexer is a pure selection
   branch (no value projection, no residual output) — call the decode indexer with
   `disable_index_value=True` (returns `o=None`). The prefill indexer's
   `disable_index_value=True` likewise skips V.

5. **CUDA JIT fast paths disabled.** `use_jit_topk` and `use_dense_main_attn` need
   the un-vendored `minimax_decode_topk.cuh`; they raise `NotImplementedError`
   here. The pure-Triton 2-stage top-k runs instead (correct, slightly slower at
   very long ctx). Profile before deciding whether to vendor the `.cuh`.

6. **sink tokens.** Optional `sink [num_q_heads, head_dim]` adds a per-head sink to
   the softmax denominator only (no value contribution). Our runtime's M3 config
   should confirm whether attention sinks are enabled; pass `sink=None` if not.

---

## 6. How to run the correctness harness

Requires a CUDA GPU with torch + triton (the kernels JIT-compile on first call).
**Not runnable on a CPU-only box** (no GPU here — it was authored, not executed).

```bash
cd /home/heima/suliang/PI/minimax_m3_work/msa_triton
python test_msa_standalone.py            # script mode, prints cos / max_abs_err table
python test_msa_standalone.py --quick    # skip the 32768-ctx cases (faster)

# pytest mode — disable host pytest-plugin autoload (this box has a ROS
# launch_testing plugin that errors at collection; unrelated to our code):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest test_msa_standalone.py -v -s
```

The harness builds random q/k/v at M3 shapes (Hq=64, Hkv=4, D=128, block=128,
topk=16) over ctx ∈ {128, 2048, 4096, 32768}, runs the Triton kernels against the
PyTorch references, and asserts **cos ≥ 0.999** and **max_abs_err ≤ 5e-2** for the
attention kernels, and **≥ 0.99 top-k block-set overlap** for the indexer.
Thresholds reflect bf16-kernel-vs-fp32-reference error on randn activations; tune
if you feed real M3 activations.
```
```

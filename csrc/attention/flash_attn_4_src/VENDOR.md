# Vendored: FlashAttention-4 (CuTe-DSL) — Thor / SM100 forward only

This is a trimmed vendor of the `flash_attn/cute` (FlashAttention-4, CuTe-DSL)
forward path, used by the LingBot-VLA Thor backend for prefix + denoise
attention. It contains **only** what the Blackwell (SM100, including sm_110 /
Thor) forward kernel needs for inference.

## Source

- Upstream:  https://github.com/Dao-AILab/flash-attention  (`flash_attn/cute/`)
- Variant:   FlashAttention-4 CuTe-DSL implementation (Hopper + Blackwell).
             Header dates the snapshot `2025-07-04`; authored against
             `nvidia-cutlass-dsl` 4.2.0 (we run it on 4.5.1).
- License:   BSD-3-Clause (see `flashrt_fa4/cute/LICENSE`, `flashrt_fa4/cute/AUTHORS`).

The CuTe-DSL forward path is a Python/JIT kernel, so unlike the FA2 vendor
there are no `.cu`/`.cpp` sources here and nothing is compiled into
`flash_rt_kernels`. The runtime deps (`nvidia-cutlass-dsl`, `quack-kernels`)
ship via the `thor-fa4` pip extra.

## Private package name (isolation)

The package is renamed `flash_attn` → **`flashrt_fa4`** so that importing it
never shadows a pip-installed `flash_attn` (the RTX backends use
`from flash_attn import flash_attn_func`, the FA2 wheel). All internal
`flash_attn.cute.*` imports were rewritten to `flashrt_fa4.cute.*`. The loader
(`flash_rt/hardware/thor/fa4_backend.py`) adds this directory to `sys.path`
only transiently and imports `flashrt_fa4.cute`.

## Scope (what was kept)

Forward, SM100-class, inference only. The kept files are exactly the import
closure of the SM100 forward kernel + the forward-only public entry:

- `flash_fwd_sm100.py`            — the SM100 forward kernel
- `flash_fwd_combine.py`          — split-KV partial-result combine
- `pack_gqa.py`, `paged_kv.py`    — GQA packing / paged-KV
- `mask.py`, `softmax.py`, `seqlen_info.py`, `block_info.py`
- `block_sparsity.py`, `block_sparse_utils.py`  (imported by the fwd kernel)
- `mma_sm100_desc.py`, `blackwell_helpers.py`, `named_barrier.py`,
  `tile_scheduler.py`, `pipeline.py`, `barrier.py`
- `cute_dsl_utils.py`, `cache_utils.py`, `copy_utils.py`, `utils.py`,
  `fast_math.py`, `fa_logging.py`, `testing.py`, `ampere_helpers.py`,
  `cute_dsl_ptxas.py`
- `interface_fwd_sm100.py`        — **local** forward-only entry (see below)

## Removed from upstream

- **All backward kernels**: `flash_bwd*.py`, `flash_bwd_sm{90,100,120}.py`,
  `flash_bwd_pre/postprocess.py`, the 2CTA backward kernels.
- **Non-SM100 forward kernels**: `flash_fwd.py` (SM80), `flash_fwd_sm90.py`,
  `flash_fwd_sm120.py`.
- **MLA** forward: `flash_fwd_mla_sm100.py`, `topk_gather_kv.py`.
- **head_dim=256 2CTA** kernels: `sm100_hd256_2cta_fmha_*.py`.
- Benchmarks / search: `benchmark*.py`, `bench_utils.py`,
  `sm90_config_search.py`, `compute_block_sparsity.py`.
- All non-`cute` packages of the wheel: `models/`, `modules/`, `layers/`,
  `losses/`, `ops/` (Triton), `utils/`, `bert_padding.py`,
  `flash_attn_interface.py`, `flash_attn_triton*.py`, `flash_blocksparse*.py`.

## Local patches

1. **`flashrt_fa4/cute/interface_fwd_sm100.py`** (renamed from `interface.py`):
   - The heavy unconditional top-level imports of the backward / SM80 / SM90 /
     SM120 / MLA / 2CTA kernels were removed. Upstream `interface.py` imported
     all of them at module load even for a single forward call.
   - The removed forward classes (`FlashAttentionForwardSm80/90/120`,
     `FlashAttentionMLAForwardSm100`, `BlackwellFusedMultiHeadAttentionForward`)
     are replaced with stubs that raise a clear `RuntimeError`. They are only
     referenced by dead arch-dispatch branches inside `_flash_attn_fwd`; on
     Blackwell those branches never execute. The SM100 forward control flow is
     **byte-identical** to upstream (verified bit-exact, see below).
   - All `_flash_attn_bwd*` helpers were deleted; `FlashAttnFunc.backward` /
     `FlashAttnVarlenFunc.backward` raise `NotImplementedError`.
2. **`flashrt_fa4/cute/__init__.py`**: imports from `interface_fwd_sm100`
   instead of the (removed) `interface`.
3. Package rename `flash_attn` → `flashrt_fa4` (isolation; see above).
4. Trailing whitespace was stripped from the kept `.py` files (inert for
   Python) so the tree passes `git diff --check`.

Trim result: the full `flash_attn` wheel (100+ files across `cute/`, `models/`,
`modules/`, `layers/`, `ops/`, …) is reduced to **32 vendored files** — 30 under
`flashrt_fa4/cute/` (27 `.py` plus `LICENSE`, `AUTHORS`, `.flake8`), the
`flashrt_fa4/__init__.py` namespace shim, and this `VENDOR.md`. The SM100
forward output is **bit-exact** vs the full upstream FA4 forward
(`cosine = 1.00000000`, `max_abs_diff = 0`).

## Update procedure

1. Pull the desired upstream `flash_attn/cute/` snapshot.
2. Re-apply the trim: keep the SM100-forward import closure listed above;
   delete backward / SM80 / SM90 / SM120 / MLA / 2CTA / benchmarks / non-cute.
3. Re-apply the three local patches (forward-only `interface_fwd_sm100.py`,
   `__init__.py` entry, `flash_attn`→`flashrt_fa4` rename:
   `grep -rl 'flash_attn\.cute' | xargs sed -i 's/flash_attn\.cute/flashrt_fa4.cute/g'`).
4. Verify forward is bit-exact vs the untrimmed source on a tiny tensor, then
   run the LingBot FA4-vs-fmha A/B (`docs/lingbot_usage.md`).

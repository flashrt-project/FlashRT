# Vendored: Flash-Attention 2 forward kernels (fp16, SM80)

## Sources

### flash_attn/
- Upstream:  https://github.com/Dao-AILab/flash-attention
- Tag:       v2.7.4.post1
- License:   BSD-3-Clause (see `flash_attn/LICENSE`)
- Subset:    `csrc/flash_attn/src/` headers + fwd fp16 SM80 kernel
             instantiations. bwd, CK (AMD) path, Hopper FA3,
             alibi/rotary/dropout runtime paths (headers kept for
             compile; p_dropout=0 in inference means inert), and the
             torch-coupled `flash_api.cpp` pybind wrapper are excluded
             — we ship our own `csrc/attention/fa2_wrapper.cu`.

### cutlass/
- Upstream:  https://github.com/NVIDIA/cutlass
- Commit:    c506e16788cb08416a4a57e11a9067beeee29420  (2025-01-08,
             between v3.7.0 and v3.8.0)
- License:   BSD-3-Clause (see `cutlass/LICENSE`)
- Subset:    `include/` subtree only. This is the submodule pin of
             Flash-Attention 2 v2.7.4.post1 — FA2 was authored
             against this CUTLASS snapshot, so we pin the same commit.

We keep a dedicated CUTLASS 3.x tree here separate from
`third_party/cutlass/` (which is v4.4.2, used by our FP4/FP8 kernels)
because CUTLASS 4.x has breaking CuTe layout-algebra changes that FA2
2.7.x does not support.

## Local patches

Three PyTorch-decoupling patches in `flash_attn/` (the whole point of
the vendor: ship FA2 without the torch wheel environment):

1. `flash_fwd_launch_template.h`: replaced `#include <c10/cuda/CUDAException.h>`
   with inline CUDA-runtime stubs for `C10_CUDA_CHECK` and
   `C10_CUDA_KERNEL_LAUNCH_CHECK`.

2. `flash.h`: replaced `#include <ATen/cuda/CUDAGeneratorImpl.h>` with a
   minimal POD `at::PhiloxCudaState` struct (dropout RNG state is
   carried through but never read in inference because `p_dropout=0`).

3. `philox_unpack.cuh`: replaced `#include <ATen/cuda/detail/UnpackRaw.cuh>`
   with a ~10-line inline stub for `at::cuda::philox::unpack()`.

One feature patch:

4. `flash.h` / `flash_fwd_kernel.h`: optional static-FP8 (e4m3) copy of the
   attention output, written from the registers that already feed the bf16
   store, plus a flag that drops the bf16 store when that copy is the only
   consumer. Reached through the `fvk_attention_fa2_fwd_bf16{,_causal}_fp8out`
   entries; with a null FP8 pointer the kernel is byte-for-byte upstream. The
   per-element arithmetic is copied from `csrc/kernels/quantize.cu`
   (`quantize_fp8_kernel_generic`) so the result matches a standalone
   `quantize_fp8_static` pass bit for bit, which
   `tests/test_fa2_fp8_output_epilogue.py` asserts on both the single-pass and
   split-KV combine paths.

Total patch footprint: ~30 LoC for the decoupling patches, ~90 LoC for the
output epilogue. See commit history on this path for exact diffs.

## X-VLA tile-set and ragged-tail patches

On top of the upstream pin above, the X-VLA RTX path adds:

1. `csrc/attention/fa2_tile_inst/flash_fwd_tile64_bf16_sm80.cu` — FlashRT
   instantiations of the vendored templates for head_dim 64:
   `run_mha_fwd_tile64_bf16` (`<64,64,64,4>`),
   `run_mha_fwd_tile64m32_bf16` (`<64,32,64,4>`) and
   `run_mha_fwd_tile64m32w2_bf16` (`<64,32,64,2>`). Additive: the upstream
   dispatch in `flash_fwd_launch_template.h` keeps its existing tiles; the new
   entries are reached only through the internal
   `fvk_attention_fa2_fwd_bf16_tile64*` bindings. The raw-library allow-list
   exports these symbols so the native-mode Python adapter can resolve them.
2. `flash_fwd_kernel.h` / `utils.h` — the ragged-tail path, enabled by the
   `SkipDeadTail` compile-time flag on `run_flash_fwd` / `flash_fwd_kernel`.
   The three X-VLA tile instantiations pass `true`; every existing entry
   uses the default `false`, so its kernel is compiled without the skip and
   keeps the unmodified upstream MMA sequence. When enabled, a warp whose 16
   query rows are all past `seqlen_q` skips its MMAs (`warp_live`), and the
   last partial KV tile only issues the live 16-key slices (`gemm_ncap` /
   `gemm_rs_kcap`). The skipped S columns are overwritten with -inf by the
   mask and the skipped P.V slices multiply P == 0, so outputs are
   bit-identical. The path is gated to the non-causal, non-local, ragged-N
   case.
3. `flash.h` / `flash_fwd_launch_template.h` — declarations and launch
   templates for the new tile entries; no upstream dispatch changes.

The tile templates remain under the upstream BSD-3-Clause license; the
instantiating source carries the notice and points back to this file.

## Backporting upstream bugfixes

FA2 main line is in maintenance (each FA2 release is 5–50 LoC of
toolchain fixes, no algorithmic changes; big work goes to FA3 which
is SM90+ only).

Procedure when we want a fix from upstream:

```bash
# In /tmp, fetch upstream + diff
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/fa-upstream
cd /tmp/fa-upstream
git log v2.7.4.post1..HEAD -- csrc/flash_attn/src/

# For each relevant commit, generate a patch and apply here
git format-patch -1 <SHA> --stdout > /tmp/fa-fix.patch
cd <FlashRT>
cd csrc/attention/flash_attn_2_src/flash_attn
patch -p4 < /tmp/fa-fix.patch   # strip leading csrc/flash_attn/src/
# verify: rebuild + cos test
```

Record each applied fix in the commit log on this path. Do NOT edit
the CUTLASS submodule without bumping the pin commit above.

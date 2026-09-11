# Development guide

## Architecture

The adapter has two halves with a hard boundary:

- **Framework-free half** (`fr_repack.cu`, `fr_quant_act.cu`,
  `fr_qkv_post.cu`, `fr_ada.cu`, `fr_decode_attn.cu`, `fr_fa4_vit.cu`):
  plain CUDA translation units, entry points declared in `fr_kernels.h`
  (raw pointers + `cudaStream_t`, no ggml or CUTLASS types in the
  header). GEMMs with fused epilogues live in `csrc/gemm/fp4/` and are
  compiled alongside.
- **ggml-facing half** (`fr_ggml.cuh`, `fr_dispatch.cu`): window
  predicates (`ggml_cuda_flashrt_should_fuse_*`) and executors that speak
  `ggml_tensor`, plus the caches. The host's `ggml-cuda.cu` calls the
  predicates from its fuse hook; that call-site code lives in the host
  tree, not here.

Keep the boundary: nothing under `csrc/` or in the framework-free half
may include ggml headers, and `fr_kernels.h` must stay consumable from a
plain C++ translation unit.

## Caches (all capture-safe)

- **Weight repack cache** — keyed by weight data pointer, never evicted
  (weights are immortal in a loaded model). ggml's split-nibble NVFP4
  blocks are repacked to the CUTLASS wire format (adjacent-pair nibbles,
  scale bytes in the Sm1xx atom layout) on first use.
- **Per-evaluation activation cache** — a producer (e.g. the fused adaLN)
  can register its already-quantized output; later GEMMs in the same
  evaluation reuse it. Keyed by tensor pointer + an evaluation counter so
  recycled addresses can never alias. Slots are grow-only so device
  addresses stay stable for captured CUDA graphs.
- **One-shot handoffs** (f16 Q from the QKV window to the decode
  attention) — single grow-only slot, key cleared on consumption.
- **CUTLASS workspaces** — shape-keyed, grown only outside capture.

Rule for all of them: no allocation while a CUDA graph is being captured.
Check `cudaStreamIsCapturing` and fall back to the unfused path (or pool
memory) when growth would be needed mid-capture.

## Adding a fusion window

1. Express the executor in the framework-free half with a C entry point
   in `fr_kernels.h`; consume existing `csrc` GEMMs where possible.
2. Add the predicate/executor pair to `fr_ggml.cuh` / `fr_dispatch.cu`.
   The predicate must pin every assumption the kernel makes: dtypes,
   shapes, strides (element-exact, not just "contiguous"), op params
   (`max_bias`, softcap), and use counts where the window elides
   intermediates.
3. Add the call site to the host's fuse hook, and a
   `GGML_FLASHRT_NO_<NAME>` switch in the predicate.
4. Validate per TESTING.md (trigger proof, A/B/A, parity or judge).

Invariants and pitfalls learned the hard way:

- **The fuse hook's return contract**: returning 0 means "not fused" and
  the anchor node executes normally afterwards — a window that replaces a
  single node must also consume the pure-view node that follows it and
  return ≥1, or its work is silently overwritten (symptom: identical
  results, slower).
- **Overlap checks are allocator-sensitive.** The generic fusion memory
  range check vetoes a window when the destination aliases an
  outside-window source. The allocator legitimately hands a window's
  output the block of an input that dies inside the window; whether that
  alias is safe depends on the fused implementation's read-before-write
  order, so exemptions are per-window and must be argued in a comment
  (see the GeGLU window: the activation is fully consumed by the quantize
  kernel before the down GEMM writes). Any change that shifts allocation
  (new nodes, another sched) can re-trigger vetoes elsewhere — symptom is
  a silent GPU-time regression; diagnose with a kernel census diff.
- **Numeric equivalences must be argued or measured**, e.g. an epilogue
  that converts the fp32 accumulator to f16 once is bit-equal to f32
  output plus a separate cast; a fused kernel writing the same values
  through the same conversion is bit-identical to the copy chain it
  replaces. Anything weaker goes through the real-observation judge.
- **RoPE in `fr_qkv_post.cu` mirrors ggml's `rope_neox`** (yarn
  corrections included) and must stay bit-exact with it; the predicate
  rejects non-NEOX modes.
- Windows only ever fire on `cc == 1100` (checked at the call site).

## AOT FlashAttention-4 modules

`fa4_aot/` holds ahead-of-time exports of the vendored FA4 forward
(vision shape: padded head_dim 80, MHA; prefill shape: head_dim 256, GQA
with one KV head). Regeneration and the export mechanics are documented
in `fa4_aot/README.md`; the short version:

- CuTe-DSL's `export_to_c` emits a C header (host launch entry, tensor
  argument structs, embedded cubin) plus a host object. The tvm-ffi
  compile variant only exports a TVM ABI, so the export script strips
  `--enable-tvm-ffi` and never executes the resulting object in-process
  (its calling convention differs).
- `fr_fa4_shims.c` supplies the small `_cuda*` runtime aliases the object
  expects, so neither the build nor the runtime depends on any CuTe-DSL
  library.
- Module loading must happen outside CUDA graph capture; the adapter
  preloads from `ggml_cuda_flashrt_begin_eval`, which always runs before
  a capture can begin.
- The wrapper takes dynamic shapes/strides per tensor, so one export per
  (head_dim, GQA config) covers all sequence lengths. The prefill
  window's mask handling relies on the pi0.5 prefix-LM property that the
  mask is row-uniform pad-only and the real KV length equals the query
  count; the padded tail is excluded by the dynamic shape instead of by
  a mask.

## Single-source rule

Structure changes (GEMM tiles, epilogues, attention decompositions)
belong in `csrc/` or the structures catalog so every host adapter
inherits them; this directory only translates. Nothing here may be
copy-pasted into a host tree, and the host integration must stay behind
its own opt-in build flag so stock builds are unaffected.

## SM120 target: additional invariants (LLM decode, speculative)

Learned on the Qwen3.6-35B window set; they generalize to any stateful or
speculative host integration.

- **Host launch overlap is a capability, not a constant.** llama.cpp's CUDA
  backend overlaps every launch through programmatic dependent launch
  (sm90+); on such a host every adapter kernel must join the chain (device
  trigger/sync + the launch attribute) or it stalls the pipeline — and once
  the chain holds, pure launch-count reduction has near-zero marginal value,
  so fusions must win on memory round-trips, byte reduction, or batch size.
  On hosts without PDL the same fusions re-rank. Treat PDL as a per-target
  capability flag (the csrc entries take a `pdl` bool).
- **Runtime dimensions out of hot loops.** A token-batch count as a kernel
  argument instead of a template parameter costs measurable time even when
  the value is 1; heavier instantiations degrade more. Dispatch runtime M
  onto compile-time specializations at the launch boundary.
- **Speculative verify batches are a correctness regime, not a batch size.**
  Stateful regions (recurrent state, conv windows) must write per-token
  snapshots and leave the source slot pristine so the host can roll back to
  any accepted position; in-place update produces degenerate output with
  *inflated* acceptance and throughput, and perplexity-style gates do not
  cover the speculative graphs at all. Judge on end-to-end text plus a
  duplicated-token bit-exact replay across batch variants.
- **Zero-sized graph nodes can become real.** Checkpoint save nodes sit in
  the host graph at zero size on most steps and materialize on checkpoint
  steps; a region that silently skips them corrupts rollback invisibly.
  Replay them inside the region or decline the whole span.
- **A second model shares the host's name scheme.** The speculative draft
  model's tensors reuse the target's naming at shifted layer indices and can
  collide with window/pack shapes; anything swept in from the draft model is
  an acceptance-only substitution (never output-visible), but it is a
  separate judgment — gate it explicitly instead of letting shape
  coincidence decide.

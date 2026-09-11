# Usage

## Building a host against the adapter

The adapter sources live at `hosts/ggml/` in the FlashRT checkout (they
were under `flash_rt/structures/adapters/ggml/` before the structures
layer moved to its own repository). A host CMake that lists the
translation units by path must use `hosts/ggml/`.

The reference host is the Jetson-PI-Edge llama.cpp tree, which carries the
integration side (CMake wiring, fuse-hook call sites, pi0 graph changes)
on its FlashRT branch and consumes this repository as a submodule at
`ggml/src/ggml-cuda/flashrt/flashrt-public`:

```bash
git clone --recursive -b feat/flashrt-thor-kernels <host repo>
cd Jetson-PI-Edge
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FLASHRT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

Options:

- `GGML_CUDA_FLASHRT` (OFF by default) — enables the layer. Without it the
  build is stock llama.cpp; every integration point is compiled out.
- `GGML_CUDA_FLASHRT_PUBLIC_DIR` — path to a FlashRT checkout, overriding
  the submodule location.
- `GGML_CUDA_FLASHRT_CUTLASS_DIR` — CUTLASS override; defaults to
  `third_party/cutlass` inside this repository.

The adapter is built as a separate CMake OBJECT library with
`-arch=sm_110a` and CUTLASS headers; the rest of ggml-cuda compiles
unchanged. The AOT FlashAttention-4 windows enable automatically when the
`fa4_aot/*.o` artifacts are present (they are checked in; see
`fa4_aot/README.md` to regenerate).

## Model preparation

- LLM weights: quantize with the host's `llama-quantize` to the `NVFP4`
  target (exposed by the FlashRT branch). Setting `GGML_NVFP4_MSE=1`
  during quantization selects per-block scales by reconstruction-MSE
  search instead of plain absmax (slower to quantize, more accurate).
- The mmproj (vision tower) is quantized to NVFP4 the same way.

## Running

```bash
PI_MODEL=pi05 ./build/bin/llama-server -m <llm.gguf> --mmproj <mmproj.gguf> \
    -ngl 99 --flash-attn on --port <port>
```

The server exposes the host's action-chunk HTTP protocol (reset → images →
state → infer). Warm-up matters on Thor: latency reaches its steady state
after roughly 15 inferences.

## Runtime switches

All switches are environment variables; unset means enabled/default.

| variable | effect |
|---|---|
| `GGML_CUDA_FLASHRT_DISABLE=1` | disable the whole layer at runtime (stock kernels) |
| `GGML_FLASHRT_NO_RMS_GEMMA=1` | disable the Gemma norm-chain window |
| `GGML_FLASHRT_NO_QKV_PREFILL=1` | disable the fused prefill QKV window |
| `GGML_FLASHRT_NO_DEC_ATTN=1` | disable the decomposed decode attention |
| `GGML_FLASHRT_NO_VIT_FA4=1` | disable the AOT FA4 vision attention |
| `GGML_FLASHRT_NO_PREFILL_FA4=1` | disable the AOT FA4 prefill attention |
| `GGML_FLASHRT_NO_KV_TAIL=1` | disable the batched persistent-KV tail copies |
| `GGML_FLASHRT_NO_VIS_F16=1` | disable the vision QKV window's direct f16 K/V outputs |
| `GGML_CUDA_FLASHRT_NO_CACHE=1` | disable the pointer-keyed weight repack cache (required for `test-backend-ops`, see TESTING.md) |
| `GGML_FLASHRT_DEBUG=1` | print window-match failure diagnostics |
| `GGML_FLASHRT_DUMP=<file>` / `GGML_FLASHRT_DUMP_MAX=<n>` | dump the first n evaluated graphs' node sequences |

Host-side switches on the FlashRT branch (outside this repository):
`GGML_PI05_MOD_PRECOMP=0` disables the denoise-schedule modulation
precompute, `LLAMA_GRAPH_REUSE_DISABLE=1` disables graph reuse,
`GGML_CUDA_DISABLE_GRAPHS=1` disables CUDA graphs (useful for profiling:
kernels are invisible to nsys while CUDA graphs replay).

Every window degrades gracefully: when its predicate does not match (or
its switch is set) the nodes run on stock ggml kernels, so the switches
bisect regressions window by window.

## SM120 / Qwen3.6 target

Build a llama.cpp tree against this checkout:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_CUDA_FLASHRT_SM120=ON -DGGML_CUDA_FLASHRT_PUBLIC_DIR=<FlashRT checkout> \
  -DCMAKE_CUDA_FLAGS="-I<cutlass>/include -gencode=arch=compute_120a,code=sm_120a"
```

The safe tier is the zero-configuration default: a compiled-in build runs
every quality-neutral window (fused packs, GDN span, MoE span, out-proj,
draft-head serving, in-process repack) with no environment at all —
`./llama-server -m <stock gguf> [-md <draft gguf> -bs]` is the whole story.
Switch semantics, most specific wins:

| layer | variable |
|---|---|
| whole layer off (stock llama.cpp) | `GGML_CUDA_FLASHRT_DISABLE=1` |
| per-window disable | `GGML_FLASHRT_NO_{INPROJ,ATTNQKV,GDN,MOEGLUE,MOEFUSE,SHEXP_FOLD,OUTNATIVE,HEAD_DRAFT,ONLINE_REPACK}=1` |
| per-window A/B override | historic `FRT_<X>_SWAP=0/1` |

Opt-in extras: `FRT_HEAD_SWAP=1 FRT_HEAD_PACK=<pack>` (full tier — the FP4
lm-head trades a measured perplexity increment for speed, so it never
defaults on); `FRT_DRAFT_REGIONS=1` (also FP4-serve the draft model's own
qkv projections; judged flat, off by default); archive windows
(`FRT_SHEXP_SWAP`, `FRT_OUTPROJ_SWAP`, `FRT_MOE_SWAP`, `FRT_ATTNGATE_SWAP`,
`FRT_GDN_NORMFOLD`) stay opt-in. Recommended host-side flags for
speculative decode: `LLAMA_GRAPH_SLOTS=6` + `--backend-sampling`.

**Model artifacts** (sm120 target): the safe tier runs any stock GGUF
as-is. The speed tier is itself just a GGUF — the FlashRT edition splices
an NVFP4 lm-head (quantized from the BF16 checkpoint via
`llama-quantize --output-tensor-type NVFP4` on a bf16 conversion) into the
shipping body with `tools/splice_nvfp4_head.py`; no side-band packs, no
environment. Do not requantize the whole body from scratch for this: the
FP4 regions inherit the source tensors' quantization quality, so the best
shipping body stays the best base.

FP4 region weights repack **in-process by default** (set `FRT_REGIONS_PACK`
to use an offline pack instead): the pre-capture hook dequantizes the GGUF
members on device and rebuilds the wire format byte-identically to the
offline packer (validated by `FRT_REPACK_CHECK=1` with
`FRT_REGIONS_PACK_REF=<file>`). The
lm-head is the exception: the shipped head pack is quantized from the BF16
checkpoint (the GGUF only holds Q6_K), and the BF16-sourced pack drafts and
scores measurably better than an online Q6_K-sourced rebuild — keep
`FRT_HEAD_PACK` for the head (both tiers); the online head build is a
fallback only. Diagnostics: `FRT_STATS=1`, `FRT_MOEFUSE_DBG=<n>`,
`FRT_MOEFUSE_SELFTEST=1`, `FRT_DUMP_GRAPH=1` + `FRT_DUMP_M=<m>`
(+ `FRT_DUMP_PATH`).

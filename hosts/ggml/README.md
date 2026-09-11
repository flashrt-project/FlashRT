# ggml host adapter (native)

Native C++/CUDA host adapter that maps FlashRT structures onto ggml's CUDA
backend (llama.cpp family), targeting Jetson AGX Thor (SM110). Unlike the
Python runtime adapters (`vllm_engine.py`, `sglang_engine.py`), this adapter
is consumed at **build time**: the host's CMake compiles these translation
units inside its own build tree and only C symbols cross the boundary. The
host never links against Python, PyTorch, or any FlashRT runtime.

Documentation:

- [USAGE.md](USAGE.md) — building a host against this adapter, runtime
  switches, deployment notes.
- [TESTING.md](TESTING.md) — operator tests, the qualification gates, and
  the benchmarking / parity methodology every change must pass.
- [DEVELOPMENT.md](DEVELOPMENT.md) — layer architecture, how to add a
  fusion window, invariants and known pitfalls, AOT FlashAttention-4
  regeneration.

## What it is

The adapter is the native host of the `flash_rt/catalog` structure catalog. The
same structures that the torch frontend and the vllm/sglang adapters
consume — block-scaled NVFP4 GEMMs with fused epilogues, fused
norm/modulation producers, the decomposed tiny-M decode attention, the
FlashAttention-4 forward — are mapped here onto ggml's graph executor
through pattern-matched subgraph windows. Heavy math is single-source:

- **NVFP4 GEMMs** come from `csrc/gemm/fp4/` in this repository
  (GeGLU-interleaved, SigLIP-FFN pair, bias/f16-out variants). Nothing is
  vendored into the host.
- **FlashAttention-4** is the vendored CuTe-DSL forward under
  `csrc/attention/flash_attn_4_src`, consumed as ahead-of-time compiled
  modules (see `fa4_aot/`), so the host build needs no CuTe-DSL toolchain.
- The `fr_*.cu` files here are the translation layer only: wire-format
  repack (ggml split-nibble NVFP4 → CUTLASS atom layout), activation
  quantize, fused RoPE/norm producers, and the dispatch/caching half that
  speaks `ggml_tensor`.

## Layout

- `fr_kernels.h` — pure C entry points (no ggml, no CUTLASS in the header).
- `fr_gemm_f32out.cu`, `fr_ada.cu`, `fr_qkv_post.cu`, `fr_quant_act.cu`,
  `fr_repack.cu`, `fr_decode_attn.cu`, `fr_fa4_vit.cu`, `fr_fa4_shims.c` —
  framework-free CUDA translation units.
- `fr_dispatch.cu`, `fr_ggml.cuh` — the ggml-facing half: subgraph window
  predicates and executors over `ggml_tensor` chains, weight/activation
  caches. Requires ggml-cuda's internal headers on the include path.
- `fa4_aot/` — AOT FlashAttention-4 modules (vision and prefill shapes)
  plus their regeneration script and provenance notes.
- `qualification/` — the release gates (see TESTING.md).
- `flash_rt/catalog/bindings/jetson_pi_edge_pi05.yaml` — the pipeline binding that
  maps the host's hot path onto catalog structures under the
  complete-hot-path contract.

## Measured performance (Jetson AGX Thor, pi0.5, 2 camera views)

| metric | stock llama.cpp (BF16) | with this adapter (NVFP4) |
|---|---|---|
| `llama_encode` + `llama_decode` (host `total_ms`, P50 warm) | 202.7 ms | 35.5 ms (**5.7×**) |
| end-to-end action chunk (ViT + prefill + 10 denoise steps) | — | **42.5 ms** |
| phase split | — | ViT 6.7 + prefill 15.6 + decode 19.8 |

For context, the FlashRT torch frontend runs the same checkpoint at
36.4 ms end-to-end on the same device; the remaining gap is dominated by
the host graph's fp32 activation dtype (the torch pipeline holds
activations in fp16).

Numerics: the adapter is bitwise deterministic across processes after
warmup; changes are gated by an exact e2e action golden plus a
real-observation parity protocol against an f16 reference (see
TESTING.md).

## Second target: RTX 5090 (SM120) + Qwen3.6-35B-A3B

`fr_win_qwen36_sm120.cu` carries the adapter's second (arch, model-family)
target: an LLM decode window set for the Qwen3.6 hybrid (GDN + attention,
256-expert MoE, MTP speculative decode) on SM120, consuming ggml's native
K-quant weights in place plus NVFP4 W4A4 fused regions. It follows the same
two-half discipline with one deliberate difference: its MoE/out-proj/router
kernels reproduce ggml's mmvq numerics through ggml's own `vec_dot_*_q8_1`
device functions (bit-exact q8_1 activation clone), so those kernels live in
the ggml-facing half by construction. Windows are M<=4 aware (speculative
verify batches) and carry the recurrent-state snapshot/checkpoint discipline
documented in DEVELOPMENT.md.

Binding: `flash_rt/catalog/bindings/llamacpp_qwen36_35b_sm120.yaml`;
gates: `qualification/pins_qwen36_sm120.yaml`.

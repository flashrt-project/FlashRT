# Qwen3-VL official BF16 on Jetson Orin (SM87)

This path brings up Qwen3-VL on Jetson Orin using the official BF16
checkpoint weights. It is the SM87 baseline counterpart to the optimized
SM89 FP8 and SM120 NVFP4 Qwen3-VL paths: the model stays in BF16, while the
runtime still uses FlashRT fixed-shape CUDA Graph replay and a small set of
Orin-friendly BF16 kernels.

The fully validated target for this path is `Qwen3-VL-2B-Instruct` on Jetson
AGX Orin 32G. The frontend is config-driven and can load
`Qwen3-VL-8B-Instruct`, but practical memory headroom is tight on Orin 32G;
8B has only been checked with a constrained low-resolution 1-token smoke test.

## Checkpoint

Use an official BF16 checkpoint:

```text
Qwen3-VL-2B-Instruct
Qwen3-VL-8B-Instruct
```

The language stack tensors are stored under `model.language_model.layers.*`.
Linear weights remain BF16. Sharded and single-file safetensors checkpoints
are both supported. For checkpoints with tied embeddings, such as the 2B
release, the BF16 loader synthesizes `lm_head` from `embed_tokens` when no
separate `lm_head.weight` exists.

## Build

Build the regular FlashRT kernels, FlashAttention module, and Qwen3-VL helper
module for SM87:

```bash
cmake -B build -S . \
  -DGPU_ARCH=87 \
  -DFA2_ARCH_NATIVE_ONLY=ON \
  -DFLASHRT_BUILD_QWEN3_VL=ON
cmake --build build -j4 \
  --target flash_rt_kernels flash_rt_fa2 flash_rt_qwen3_vl_kernels
```

On SM87, `flash_rt_qwen3_vl_kernels` provides BF16 Qwen3-VL helper kernels.
It does not build the SM89 FP8 activation-quantization sources.

## Runtime Architecture

The runtime frontend is:

```python
flash_rt.frontends.torch.qwen3_vl_rtx_bf16.Qwen3VlTorchFrontendRtxBF16
```

The dtype mapping is:

| Component | SM87 BF16 path |
|---|---|
| Language weights | Official BF16 |
| Language activations | BF16 |
| Language GEMM output | BF16 |
| Attention Q/K/V cache | BF16 |
| Attention backend | BF16 FA2 |
| Residual stream and norms | BF16 |
| Vision tower | BF16 |
| `lm_head` | BF16 |

The language stack uses the generic FlashRT BF16 Qwen3 helpers:
`bf16_matmul_bf16`, `rms_norm`, `residual_add_rms_norm`,
`silu_mul_qwen36_bf16`, and fused Q/K norm + RoPE + KV-write kernels.

The vision tower reuses `Qwen3VlVisionRtx` in BF16 mode. On SM87, Qwen3-VL
BF16 prefill GEMMs use a cuBLASLt helper from `flash_rt_qwen3_vl_kernels`.
Decode-time `M=1` language GEMMs use a Qwen3-VL-specific BF16 GEMV helper for
the 2B model's dominant `K=2048` and `K=6144` projections.

The cuBLASLt autotune change is limited to callers of
`bf16_matmul_cublaslt_bf16`; the regular `bf16_matmul_bf16`, INT8, FP8, and
NVFP4 kernels are unchanged. Autotuning is skipped during CUDA Graph capture.

The BF16 frontend currently supports single-image chat prompts and greedy
generation. It stages single-image prompt tensors into fixed buffers, captures
one prefill graph per `(patch_count, seq_len, image_span)` bucket, and captures
decode graphs per `(cache_pos, rope_pos)` bucket.

## Quickstart

```bash
python examples/orin/qwen3_vl_quickstart.py \
  --checkpoint /path/to/Qwen3-VL-2B-Instruct \
  --image FlashRT.png \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 32
```

Use `--no-graph` to run the eager correctness path without CUDA Graph replay.

Both decode-quantization flags default to `bf16`, so the command above is the
plain BF16 path. Add `--weight-mode int8 --kv-mode int8` for the recommended
Orin decode configuration; see [Decode weight quantization](#decode-weight-quantization-weight_mode)
below for the full menu and why the FP8/FP4 tiers are the wrong choice here.

The frontend is registered for `('qwen3_vl', 'torch', 'rtx_sm87')`, so
`flash_rt.hardware.resolve_pipeline_class` finds it. `load_model()` still
raises a redirect, because Qwen3-VL exposes a chat surface
(`generate(messages) -> str`) rather than the VLA `predict()` surface.

For the full-resolution comparison workload used by the existing Qwen3-VL
FP8/NVFP4 reports:

```bash
python examples/orin/qwen3_vl_quickstart.py \
  --checkpoint /path/to/Qwen3-VL-2B-Instruct \
  --image FlashRT.png \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 4 \
  --benchmark 3
```

`FlashRT.png` at full resolution produces 6256 vision patches and 1581 prompt
tokens. Pass `--max-pixels` only when deliberately trading visual resolution
for latency; the BF16 frontend forwards this through the Qwen3-VL processor's
smart-resize policy rather than manually resizing the image.

## Jetson Orin Validation

Environment:

- Device: Jetson AGX Orin 32G, SM87
- L4T: R36.4.7
- CUDA Toolkit: 12.6.68
- PyTorch: 2.8.0 + CUDA 12.6
- Checkpoint: `/path/to/Qwen3-VL-2B-Instruct`
- Workload: `FlashRT.png`, prompt `Describe this image in one sentence.`

Local checks:

```bash
python -m py_compile \
  flash_rt/frontends/torch/qwen3_vl_rtx_bf16.py \
  examples/orin/qwen3_vl_quickstart.py
python -m pytest tests/test_qwen3_vl_rtx_bf16.py tests/test_build_inventory.py -q
git diff --check
```

Runtime smoke validation compared the same prompt through HuggingFace BF16,
FlashRT BF16 eager (`--no-graph`), and FlashRT BF16 graph paths. FlashRT eager
and graph produced the same short continuation on the small smoke prompt.

Full-resolution Orin BF16 result with cuBLASLt autotuning and the M=1 BF16
GEMV decode helper enabled:

```text
vision patches: 6256
prompt tokens: 1581
max_new_tokens: 4
generate latency: 5768.0 ms cold / 1050.5 ms warm
prefill graph P50: 927.5 ms
decode throughput (warm graph): 36.8 tok/s
```

With cuBLASLt autotuning disabled via
`FLASHRT_BF16_CUBLASLT_AUTOTUNE_ALGOS=1`, the same binary measured:

```text
generate latency: 4973.0 ms cold / 1066.7 ms warm
prefill graph P50: 953.8 ms
decode throughput (warm graph): 36.7 tok/s
```

The two optimization effects were measured separately:

| Change | Before | After |
|---|---:|---:|
| M=1 BF16 GEMV decode helper | ~16.2 tok/s | 36.8 tok/s |
| cuBLASLt autotune for M>1 prefill GEMMs | 953.8 ms prefill P50 | 927.5 ms prefill P50 |

The fixed-K M=1 GEMV helper provides the larger decode win. The cuBLASLt
autotune mainly affects M>1 prefill replay; it has little effect on decode
throughput once the M=1 GEMV helper is enabled.

### Decode weight quantization (`weight_mode`)

Decode runs at M=1, so every step reads the whole weight set once and is bound
by weight bandwidth rather than math. Shrinking the weights converts almost
directly into tokens/s. `weight_mode` picks a weight-only tier; the GEMV
dequantizes to BF16 in-kernel, so no FP8/FP4 tensor core is involved. Prefill
always keeps using the BF16 weights.

| `weight_mode` | format | weight bytes/element | notes |
|---|---|---:|---|
| `bf16` (default) | BF16 | 2.0 | unchanged baseline |
| `int8` | INT8 symmetric, scale = amax/127 | 1.125 | recommended on Orin |
| `int4` | INT4 symmetric, scale = amax/7 | 0.625 | chat-grade precision |
| `w8` | FP8 e4m3, scale = amax/448 | 1.125 | **not recommended on Orin** |
| `w4` | NVFP4 e2m1, scale = amax/6 | 0.625 | **not recommended on Orin** |

Scales are per 16 elements, stored BF16.

**Why the integer tiers and not the float ones.** Orin is Ampere (sm_87), which
has no hardware FP8 conversion instruction. The e4m3/e2m1 dequant therefore
compiles to a software bit sequence and the GEMV flips from bandwidth-bound to
ALU-bound, so `w8` and `w4` measure *slower than BF16* here despite moving fewer
bytes. `int8`/`int4` dequantize with a hardware integer-to-float conversion and
keep the bandwidth win. The `w8`/`w4` tiers remain available because they are
the right choice on sm_89+, where that conversion exists.

`int8` and `int4` use K-templated kernels built for the Qwen3-VL-2B dimensions
(K in {2048, 6144}). The constructor rejects other dimensions up front rather
than failing partway through the first decode step; use `w8`/`w4` or `bf16` for
8B.

### INT8 KV cache (`kv_mode`)

`kv_mode='int8'` keeps INT8 mirrors of the KV cache with one BF16 scale per
(position, KV-head) row and runs q=1 decode attention over those mirrors with a
two-pass flash-decoding kernel, halving the KV bytes each decode step reads.
That matters more as the prompt grows, since attention reads the whole KV cache
every token. Prefill still runs FA2 against the BF16 cache.

The decode kernel is specialized for the 2B attention geometry (GQA 16Q/8KV,
head_dim 128) and is rejected at construction otherwise.

### Measured tiers

Measured during development on Jetson AGX Orin with a Qwen3-VL-2B-architecture
checkpoint at prompt = 1581 tokens (6256 vision patches). Absolute throughput
depends on device, prompt length and clocks, so treat the ratios as the
portable result and re-measure locally:

```bash
python examples/orin/qwen3_vl_quickstart.py --checkpoint <ckpt> \
  --image FlashRT.png --prompt "Describe this image in one sentence." \
  --max-new-tokens 64 --benchmark 20 --weight-mode int8 --kv-mode int8
```

| `weight_mode` | vs BF16 decode | logit cosine vs BF16 |
|---|---:|---:|
| `bf16` | 1.00× | — |
| `int8` | ~1.7× | 0.99985 (effectively lossless) |
| `int4` | ~2.3× | 0.989 (chat-grade) |
| `w8` | ~0.8× (regression) | 0.99932 |
| `w4` | ~0.65× (regression) | not measured |

`kv_mode='int8'` is orthogonal to the weight tier. Its speedup grows with prompt
length, since attention reads the whole KV cache every token, and it does not
change prefill. Its effect on output quality has not been characterised yet —
validate it against the BF16 KV path for your workload before relying on it.

Two results are worth recording because they look like obvious wins and are
not: staging the activation in shared memory *cost* throughput (Orin has 16 SMs,
so the occupancy loss outweighed the saved L2 re-reads, and the activation was
already L2-hot), and the FP8/FP4 tiers regress for the hardware reason above.

### In-graph argmax decode

`generate(use_graph=True)` captures the argmax and the token feedback into each
decode graph, so a replay yields the next token with no host round-trip. EOS is
checked from a pinned async copy one step behind, so decoding can overshoot EOS
by at most one token; the returned text is trimmed at the first EOS either way,
so callers see no difference.

The argmax runs on the BF16 logits directly. FP32 is a superset of BF16, so the
ordering — and therefore the argmax — is identical to an FP32 argmax. This is
exact, not an approximation.

### Resolution knob

Resolution capping is an explicit deployment knob. It reduces both vision
patches and LLM prefill tokens:

| `max_pixels` | vision patches | prompt tokens | warm generate | prefill graph P50 | decode throughput |
|---|---:|---:|---:|---:|---:|
| none | 6256 | 1581 | 1050.5 ms | 927.5 ms | 36.8 tok/s |
| 1.0 M | 3888 | 989 | 600.2 ms | 503.6 ms | 41.4 tok/s |
| 0.5 M | 1824 | 473 | 317.3 ms | 216.1 ms | 39.0 tok/s |
| 0.25 M | 972 | 260 | 234.6 ms | 127.2 ms | 39.6 tok/s |

These capped-resolution numbers are not the full-resolution baseline. They
only show the expected prefill scaling when the processor emits fewer visual
tokens.

### 8B support

The same BF16 frontend can load `Qwen3-VL-8B-Instruct`. On Jetson AGX Orin
32G, memory headroom is tight, so validation was limited to a low-resolution
1-token smoke:

```text
max_pixels: 250000
max_seq: 2048
max_new_tokens: 1
latency: 2790.9 ms
```

This confirms the checkpoint path is loadable, but the validated target for
this BF16 path remains Qwen3-VL-2B on Orin 32G. Orin configurations with more
memory should have more room for larger `max_pixels`, longer sequences, and
more decode tokens.

## Profiling Notes

A replay-only Nsight profile was collected on the full-resolution workload
after graph capture and warmup:

```bash
nsys profile --trace=cuda \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node \
  -o /path/to/qwen3_vl_bf16_prefill_replay \
  python ...
```

The captured range replayed the full-resolution prefill graph three times.
Top GPU kernel groups were:

| Kernel group | Time share |
|---|---:|
| FA2 BF16 prefill attention, vision tower | 32.9% |
| cuBLASLt BF16 GEMM, 128x128 family | 23.7% |
| cuBLASLt BF16 GEMM, 128x256 family | 12.7% |
| cuBLASLt BF16 GEMM, 256x128 family | 6.3% |
| QKV split / bias | 4.0% |
| BF16 bias+GELU | 3.3% |
| copy / staging elementwise | 2.8% |
| residual+bias | 2.6% |
| FA2 BF16 prefill attention, language stack | 2.1% |
| SiLU multiply | 2.0% |

The remaining full-resolution prefill bottleneck is split between attention
and large BF16 GEMMs. Another one-off elementwise fusion is unlikely to move
the full path much; larger future work would need to target attention/prefill
structure, larger-grain BF16 GEMM scheduling, or a separate Orin-friendly
quantized path.

## Limits

- The first fully validated target is Qwen3-VL-2B on Jetson Orin / SM87.
- Qwen3-VL-8B is supported by the config-driven BF16 path, but Orin 32G was
  only validated with a constrained low-resolution smoke because memory
  pressure is high.
- Single-image prompts are supported. Multi-image and video are not part of
  this BF16 bring-up.
- The frontend can be instantiated directly or resolved through
  `resolve_pipeline_class`; `load_model()` deliberately redirects, since this is
  a chat VLM rather than a VLA. Server integration is not included.
- Weights and prefill are BF16. Decode weight quantization and the INT8 KV
  cache are opt-in and off by default; the `int8`/`int4` tiers additionally
  require the Qwen3-VL-2B dimensions. This path is a correctness and
  portability baseline for official checkpoints, not a replacement for the
  optimized FP8/NVFP4 paths on sm_89+.
- Decode graphs are captured per cache position and RoPE position.

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
  --checkpoint /root/models/Qwen3-VL-2B-Instruct \
  --image FlashRT.png \
  --prompt "Describe this image in one sentence." \
  --max-new-tokens 32
```

Use `--no-graph` to run the eager correctness path without CUDA Graph replay.

For the full-resolution comparison workload used by the existing Qwen3-VL
FP8/NVFP4 reports:

```bash
python examples/orin/qwen3_vl_quickstart.py \
  --checkpoint /root/models/Qwen3-VL-2B-Instruct \
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
- Checkpoint: `/root/models/Qwen3-VL-2B-Instruct`
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
  -o /root/qwen3_vl_bf16_prefill_replay \
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
- The frontend is instantiated directly; server integration and `load_model()`
  registration are not included.
- The path is BF16-only. It is a correctness and portability baseline for
  official checkpoints, not a replacement for the optimized FP8/NVFP4 paths.
- Decode graphs are captured per cache position and RoPE position.

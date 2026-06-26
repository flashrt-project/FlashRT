# Qwen3-VL-8B official FP8 on RTX 4090 (SM89)

This path brings up Qwen3-VL-8B on Ada RTX GPUs using the official
Qwen3-VL FP8 checkpoint. It is the SM89 counterpart to the SM120/NVFP4
Qwen3-VL path, but it intentionally uses the checkpoint's native block-scaled
FP8 language weights instead of NVFP4.

## Checkpoint

Use the official FP8 checkpoint:

```text
Qwen3-VL-8B-Instruct-FP8
```

The language stack tensors are stored under
`model.language_model.layers.*`. Linear weights are
`torch.float8_e4m3fn` with `weight_scale_inv` block scales in a 128x128
layout. The vision tower in this checkpoint is BF16.

## Build

Build the regular FlashRT kernels and the Qwen3-VL helper module for SM89:

```bash
cmake -S . -B build -DGPU_ARCH=89 -DFLASHRT_BUILD_QWEN3_VL=ON
cmake --build build -j --target flash_rt_kernels flash_rt_fa2 flash_rt_qwen3_vl_kernels
```

The SM89 Qwen3-VL path uses:

- `flash_rt_kernels` for generic helpers shared with other RTX paths, such as
  embedding lookup, RMSNorm, residual add, and the base attention/runtime
  support.
- `flash_rt_fa2` for BF16 FlashAttention.
- `flash_rt_qwen3_vl_kernels` for Qwen3-VL-specific kernels: SM89 FP8
  block-128 GEMV/GEMM, fused norm/activation quantization, QK norm/RoPE/KV
  write, BF16 cuBLASLt ViT matmul wrappers, Qwen3 postprocessing, and the
  existing SM120 vision RoPE / layernorm-GELU-to-FP8 / BF16 bias epilogues.

## Runtime Architecture

The runtime frontend is
`flash_rt.frontends.torch.qwen3_vl_fp8_sm89_multimodal.Qwen3VlFp8Sm89Frontend`.

The text-only language core is
`flash_rt.frontends.torch.qwen3_vl_fp8_sm89.Qwen3VlFp8Sm89TextFrontend`.

The dtype mapping is:

| Component | SM89 dtype path |
|---|---|
| Language weights | Official block-scaled FP8 e4m3, 128x128 scales |
| Language activations before linears | BF16 -> per-token/per-128 FP8 |
| Language GEMV/GEMM output | BF16 |
| Attention Q/K/V cache | BF16 |
| Attention backend | BF16 FA2 |
| Residual stream and norms | BF16 |
| Vision protected blocks | BF16 |
| Vision bulk GEMMs | FP8 block-128 activation/weight, BF16 output |
| `lm_head` default | BF16 |

`use_fp8_lm_head=True` remains an explicit experimental mode. Single-step
logits can match closely, but multi-token generation can diverge after the
first generated tokens, so BF16 `lm_head` is the default.

## Quickstart

```bash
python examples/qwen3_vl_quickstart.py \
  --arch sm89 \
  --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8 \
  --image FlashRT.png \
  --prompt "Describe this image in one sentence." \
  --max-seq 2048
```

By default, the SM89 prefill buffer is sized to `max_seq`, matching the SM120
Qwen3-VL frontend. Pass `--max-prefill-seq` only to deliberately lower that
capacity.

For the development smoke benchmark:

```bash
python scripts/smoke_qwen3_vl_fp8_sm89.py \
  --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8 \
  --multimodal \
  --iters 5 \
  --generate-tokens 0
```

The full-resolution `FlashRT.png` workload produces 6256 vision patches and
1581 language tokens. This is the workload used for the SM120 Qwen3-VL report,
so it is the preferred comparison point for full TTFT numbers.

## RTX 4090 Validation

Environment:

- GPU: NVIDIA GeForce RTX 4090, SM89
- Checkpoint: official Qwen3-VL-8B-Instruct-FP8
- Workload: `FlashRT.png`, prompt `Describe this image in one sentence.`
- Prompt shape: 6256 vision patches, 1581 language tokens

Command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_qwen3_vl_fp8_sm89.py \
  --checkpoint /path/to/Qwen3-VL-8B-Instruct-FP8 \
  --multimodal \
  --iters 3 \
  --generate-tokens 0
```

Result:

```text
S=1581 pixel_shape=(6256, 1536) spans=[(4, 1568)]
set_prompt_warm median=15.711 ms
vision_only median=88.990 ms
language_only_no_mm_scatter median=120.634 ms
prefill median=206.987 ms top=785 finite=True
prefill_graph median=206.214 ms cos_vs_eager=0.998522 top=785 finite=True
graph_decode_cache_pos=1581 median=10.681 ms cos_vs_eager=1.000000
top_eager=2168 top_graph=2168
```

The same workload on the SM120/NVFP4 path is faster because it uses Blackwell
NVFP4 language kernels. The SM89 path is intended to use the best available
Ada-compatible dtype path: official FP8 language weights, BF16 attention and
residual state, and BF16 `lm_head`.

## Limits

- This is not an NVFP4 implementation. SM89 uses official FP8 weights because
  NVFP4 is a Blackwell path.
- `lm_head` stays BF16 by default for generation quality.
- Single-image prefill has a CUDA Graph replay path. Multi-image and video
  should use the eager path unless separately validated.
- Decode graphs are captured per cache position because the FA2 call captures
  host-side sequence length.
- The development scripts under `scripts/*qwen3_vl_fp8_sm89*` are local
  validation and profiling entry points, not CI replacements.

## Why this path is hand-written (no TMA on Ada)

The SM120 Qwen3-VL path runs its block-128 FP8 GEMM through the CUTLASS
`KernelTmaWarpSpecializedBlockwise` collective: TMA bulk-copy engines feed a
warp-specialized producer/consumer pipeline, and CUTLASS auto-selects a deep
(3-4 stage) software pipeline. That design hides global-load latency *without*
spending registers or occupancy on the staging.

Ada (sm_89) has **no TMA and no warp-specialized CUTLASS collective** (those
builders are Sm90+/Sm120 only), so that design does not port. Two consequences
shape this kernel:

- **GEMM** is a hand-written `cp.async` + `ldmatrix.x4` + block-128-scaling MMA
  kernel (`fp8_block128_gemm_mma_sm89.cu`). With `cp.async` instead of TMA, each
  extra pipeline stage costs a full shared-memory buffer, and at the tuned
  64x64 tile the kernel is already register-limited to 4 CTA/SM — so a 2-stage
  pipeline halves occupancy and regresses. It runs single-stage and is
  latency-bound; the obvious memory-pattern optimizations (load/store
  coalescing, `ldmatrix`) are applied, and CUTLASS's own Ada blockwise example
  is slower than this kernel, so it is not used.
- **Attention** uses the vendored FlashAttention-2 (Tri Dao) kernels, including
  an added hdim64 instantiation for the 2B ViT; the FA2 kernel body is upstream
  and is not modified here.

Net: the SM89 path matches the SM120 design *intent* (block-128 FP8, fused
act/norm-quant, ldmatrix-based MMA) but cannot reuse the TMA/warp-specialized
machinery, which is the main reason its prefill is slower than the Blackwell
NVFP4 path.

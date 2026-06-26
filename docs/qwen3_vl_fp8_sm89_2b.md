# Qwen3-VL-2B block-128 FP8 on RTX 4090 (SM89)

This brings up Qwen3-VL-2B on Ada RTX GPUs using the same SM89 block-128 FP8
language path as the 8B variant. The 2B model shares every kernel, the vision
tower, the MRoPE geometry, and the CUDA-Graph prefill/decode machinery with
the 8B path documented in `qwen3_vl_fp8_sm89.md` — only the language-stack
dimensions differ.

| | 2B | 8B |
|---|---|---|
| layers | 28 | 36 |
| Q / KV heads | 16 / 8 (GQA 2:1) | 32 / 8 (GQA 4:1) |
| hidden | 2048 | 4096 |
| intermediate | 6144 | 12288 |
| fused qkv N | 4096 | 6144 |
| `rope_theta` / `mrope_section` | 5e6 / [24,20,20] | 5e6 / [24,20,20] |
| `tie_word_embeddings` | true | false |

The frontend reads all of these from `config.json`; it enforces only the
kernel constraints (head_dim == 128, every GEMM N/K a multiple of 128, Q heads
a multiple of KV heads). No code is 2B-specific.

## Checkpoint

The official Qwen3-VL-2B release ships **BF16 only** (single-file
`model.safetensors`, tied embeddings, no `lm_head.weight`). Produce a
block-128 FP8 checkpoint in the official layout with the offline quantizer:

```bash
python scripts/quantize_qwen3_vl_to_fp8_block128.py \
  --src /path/to/Qwen3-VL-2B-Instruct \
  --dst /path/to/Qwen3-VL-2B-Instruct-FP8
```

This quantizes the 196 language-stack linears (`model.language_model.layers.*`
q/k/v/o + gate/up/down) to e4m3 `weight` + fp32 `weight_scale_inv` (128x128
blocks, `amax/448` multiply-to-dequant) and copies norms, `embed_tokens`, and
the BF16 vision tower through unchanged. Tied `lm_head` is **not** materialized
— the loader synthesizes it from `embed_tokens`. Output is ~2.85 GB.

## Build

Identical to the 8B path:

```bash
cmake -S . -B build -DGPU_ARCH=89 -DFLASHRT_BUILD_QWEN3_VL=ON
cmake --build build -j --target flash_rt_kernels flash_rt_fa2 flash_rt_qwen3_vl_kernels
```

## lm_head: optional FP8 mode for 2B

The frontend default is BF16 `lm_head` (shared with the 8B path). For
throughput-sensitive local validation, pass `use_fp8_lm_head=True` or
`--fp8-lm-head`. The 152k-vocab projection is a large decode-time weight read,
so FP8 `lm_head` can reduce bandwidth pressure on 2B. Treat it as a performance
mode: validate logits and generation quality against the default BF16 head on
the target checkpoint before using it in production.

## Quickstart

```bash
python scripts/smoke_qwen3_vl_fp8_sm89.py \
  --checkpoint /path/to/Qwen3-VL-2B-Instruct-FP8 \
  --multimodal --fp8-lm-head --iters 10 --generate-tokens 32
```

## RTX 4090 Validation

Environment: NVIDIA GeForce RTX 4090 (SM89); checkpoint
`Qwen3-VL-2B-Instruct-FP8` (quantized as above); FP8 `lm_head`.

Text-only (S=79 prefill, decode at cache_pos=63):

```text
S=79 prefill median=8.489 ms
prefill_speedup=52.09x logit_cos=0.999426 top_prefill=198 top_loop=198
graph_decode_pos=63 median=2.653 ms (377 tok/s) top=19564 finite=True
```

Multimodal (`FlashRT.png`, `Describe this image in one sentence.`, S=1581):

```text
S=1581 pixel_shape=(6256, 1536) spans=[(4, 1568)]
vision_only median=69.670 ms
language_only_no_mm_scatter median=29.827 ms
prefill median=102.991 ms top=32 finite=True
prefill_graph median=99.743 ms cos_vs_eager=0.997508 top=32 finite=True
graph_decode_cache_pos=1581 median=2.966 ms (337 tok/s) cos_vs_eager=1.000000
  top_eager=3691 top_graph=3691
generate_tokens=32 text='A black background features the "FlashRT" logo, with
  an orange lightning bolt symbol next to the stylized text "FlashRT" in white
  and orange.'
```

These numbers are local validation points, not CI guarantees. Re-benchmark on
the target GPU, driver, and build flags before treating them as deployment
latency targets.

## Limits

- Inherits all 8B-path limits (single-image graph prefill, per-cache-position
  decode graphs, FP8-not-NVFP4 on Ada). See `qwen3_vl_fp8_sm89.md`.
- The 2B FP8 checkpoint is produced offline by the repo quantizer; there is no
  official 2B FP8 release.
- The `scripts/*qwen3_vl_fp8_sm89*` entry points are local validation, not CI.

# GR00T N1.7 on NVFP4 (RTX 5090, sm_120a)

`use_fp4=True` on `rtx_sm120` runs the GR00T N1.7 action head — the 32-layer
DiT and its cross-attention K/V projections — on NVFP4 (e2m1 values, UE4M3
scales per 16 elements, a per-tensor global scale) and keeps the VLM backbone on
FP8. Both tiers share the fused FP8 backbone described below. Quantization
always happens inside the kernel that produces the operand; no standalone
quantize / dequantize pass runs around the 4-bit GEMMs.

```python
model = flash_rt.load_model(
    "/path/to/GR00T-N1.7-3B", framework="torch", config="groot_n17",
    hardware="rtx_sm120", num_views=2,
    embodiment_tag="oxe_droid_relative_eef_relative_joint",
    use_fp4=True)                     # default: the FP8 tier
model.set_prompt(aux=aux)
actions = model.infer(state_normalized, aux=aux, initial_noise=noise)
```

Needs a `GPU_ARCH=120` build (the skinny NVFP4 family and the SM120
block-scaled GEMMs are built for it unconditionally).

## What runs where

| stage | tier | per layer |
|---|---|---|
| ViT (to the last DeepStack tap, 18 layers) | FP8 | LayerNorm (+ the previous layer's residual and bias) → FP8; one fused QKV GEMM; bias + RoPE head split; FA2; O GEMM; residual + bias + LayerNorm → FP8; fc1 GEMM; bias + GELU → FP8; fc2 GEMM |
| DeepStack mergers | FP8 | run at their tap, straight off the ViT residual stream |
| LLM (16 layers) | FP8 | residual (+ DeepStack injection) + RMSNorm → FP8; fused QKV GEMM; per-head q/k RMSNorm + M-RoPE writing BF16 attention operands; FA2 causal GQA with split-KV (no K/V expansion, no logits buffer); O GEMM; residual + RMSNorm → FP8; gate / up GEMMs; SiLU-mul → FP8; down GEMM |
| vlln + VL self-attention (4 layers) | FP8 | vlln fused with the LLM's last residual; same producer pattern as the ViT |
| cross K/V (16 blocks) | NVFP4 | two W4A4 GEMMs per frame — the text rows and the image rows each against the concatenated K\|V weights of their 8 blocks (N = 24576, bias in the epilogue); each block reads its K / V as a column slice |
| DiT (32 layers x 4 steps) | NVFP4 | skinny W4A4 GEMMs (weights prefetched into registers before the programmatic-dependent-launch wait, K-split partials) and fused consumers: AdaLN → e2m1, attention straight off the Q / QKV projection partials (`nvfp4_skinny_partials_attention`, mma.sync), O GEMM quantizing its BF16 input on load, bias + residual + LayerNorm → e2m1, bias + GELU → e2m1 |

The fused FP8 producers live in `kernels/fp8_prefill_fp16.cu`
(`fp8_residual_norm_quant`, `fp8_qkv_split`, `fp8_bias_gelu_quant`); the
backbone forward is `models/groot_n17/pipeline_rtx_fp8_fused.py`.

### Why the backbone stays FP8

A torch emulator of the W4A4 kernels (it reproduces their outputs) on real DROID
frames, cosine against the HF activations:

| backbone precision | DeepStack features (taps 5 / 11 / 17) | VL self-attention output |
|---|---|---|
| FP8 | 0.997 / 0.980 / 0.978 | 0.957 |
| NVFP4 everywhere | 0.973 / 0.853 / 0.872 | 0.69–0.75 |
| NVFP4 ViT FFNs only (layers 2–17) | 0.994 / 0.966 / 0.956 | — |

The errors compound through the ViT and the LLM, with or without static
activation global scales, so 4-bit is limited to the action head.
`FLASHRT_GROOT_N17_BB_NVFP4=vit,llm,vlsa` builds the NVFP4 backbone stages for
experiments only.

## Numbers

RTX 5090, 12 real DROID 2-view observations (`infer(aux=...)`: backbone graph
replay with the fresh observation, then the action graph; four denoising
steps; GPU time per call, median of 200):

| tier | latency |
|---|---:|
| FP8 before this tier (all 24 ViT layers, unfused backbone) | 16.51 ms |
| FP8 | 14.01 ms |
| NVFP4 (`use_fp4=True`) | 8.64 ms |

Agreement, same inputs and noise:

| tier | vs FP16 frontend, normalized actions (mean / min cosine) | vs official policy, relative actions (mean / min cosine) |
|---|---|---|
| FP16 frontend | — | 0.932 / 0.222 |
| FP8 | 0.99979 / 0.99811 | 0.995 / 0.979 |
| NVFP4 | 0.99976 / 0.99846 | 0.988 / 0.967 |

Two of the twelve frames are numerically sensitive against the official policy
(relative actions near zero); which implementation misses them varies, which is
why the FP16 frontend's minimum is the lowest.

## Switches

| variable | effect |
|---|---|
| `FLASHRT_GROOT_N17_BB_FUSED=0` | unfused FP8 backbone sequence |
| `FLASHRT_GROOT_N17_CK_NVFP4=0` | BF16 cross K/V projections |
| `FLASHRT_GROOT_N17_DIT_FP4_MSE=<n>` | block-scale search steps of the DiT weights (default 4) |
| `FLASHRT_GROOT_N17_DIT_PDL=0` | DiT without programmatic dependent launch |
| `FLASHRT_GROOT_N17_BB_NVFP4=<stages>` | experimental NVFP4 backbone stages (inaccurate, see above) |

## Limits

- sm_120a only.
- The DiT is bound by its chain of eight dependent kernels per layer
  (~27 us per layer against ~8 us of weight streaming).

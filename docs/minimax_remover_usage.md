# MiniMax-Remover — FlashRT Inference Pipeline

MiniMax-Remover video inpainting (subtitle / object removal) with NVFP4
(W4A4) kernelized inference on Blackwell SM120.

## Build

The pipeline reuses the **generic** FlashRT SM120 NVFP4 kernels — it
ships **no model-specific CUDA operators**. Building for Blackwell
auto-enables the NVFP4 kernels (the NVFP4 quantise / W4A4 GEMM /
fused-bias-gelu entry points land in `flash_rt_kernels.so`):

```bash
cd FlashRT
cmake -S . -B build -DGPU_ARCH=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target flash_rt_kernels
pip install -e ".[torch,minimax-remover]"
```

`GPU_ARCH=120` (RTX 5090) or `121` selects the Blackwell target; the
NVFP4 surface is compiled in automatically (internally gated by
`ENABLE_CUTLASS_SM120_NVFP4_W4A16`, which is set from `GPU_ARCH`, not a
flag users pass). Then install the runtime extras:

```bash
pip install -e ".[minimax-remover]"   # diffusers + einops + sageattention
```

Importing `flash_rt.models.minimax_remover` always succeeds — it needs
**none** of `diffusers` / `einops` / `triton` / `sageattention`. The
kernel surface is validated lazily in
`MiniMaxRemoverPipeline.__init__` via `_load_kernels()`, and the runtime
deps are resolved at construction via `_import_runtime()`. If a required
symbol or dep is missing the constructor raises a clear `RuntimeError`
with the rebuild/install hint, so a non-NVFP4 build or a bare
environment fails fast instead of crashing mid-run. The required generic
symbols are:

| Symbol | Role |
|--------|------|
| `nvfp4_sf_swizzled_bytes` | block-scale-factor byte layout helper |
| `bf16_weight_to_nvfp4_swizzled` | one-shot weight -> NVFP4 quantise |
| `quantize_bf16_to_nvfp4_swizzled` | per-call dynamic activation quantise |
| `fp4_w4a16_gemm_sm120_bf16out_pingpong` | SM120-native W4A4 MMA -> bf16 |
| `add_bias_bf16` | in-place bias add on bf16 GEMM output |
| `fp4_w4a16_gemm_bias_gelu_fp4out_sm120` | fused FFN-up GEMM + bias + GELU -> FP4 |

The attention backend (`FLASHRT_ATTN_MODE`) optionally pulls in
`sageattention` (Sage); `fa2` uses the vendored `flash_rt_fa2.so` and is
the dependency-light fallback. The fused norm / RoPE / Euler-step
elementwise kernels are self-contained Triton JIT kernels shipped in the
package (`_kernels.py`) and need no build step.

## Pipeline

`flash_rt/models/minimax_remover/pipeline.py` — `MiniMaxRemoverPipeline`.
It wraps a loaded diffusers MiniMax-Remover `pipe` and consumes it in
place:

- every eligible transformer Linear -> NVFP4 W4A4 GEMM (weight quantised
  once at load time; activation quantised **dynamically** per call with
  per-16-element UE4M3 block scales computed on-GPU — no offline
  calibration, no CPU sync);
- transformer switched to bf16 (NVFP4-native, eliminates the fp16<->bf16
  cast pair);
- RoPE freqs cached as complex<float>;
- `torch.nn.functional.scaled_dot_product_attention` -> FA2 / SageAttention;
- per-block LayerNorm + adaLN modulation + gate-residual fused into a
  single fp32-stat Triton kernel;
- the N-step flow-matching denoise loop replaced by a manual, graph-
  capturable pointer-based loop (`ManualRemoverPipeline`). QKV quantises
  the norm output **once** and reuses it for all three projections; the
  FFN-up GEMM fuses bias + GELU straight to FP4 output so the FFN-down
  projection skips re-quantisation. With `FLASHRT_MANUAL_GRAPH=1` the
  whole N-step x N-block loop is captured as a single CUDA Graph;
  inside the captured graph there are **zero** torch elementwise ops —
  every operation is a kernel launch.

The VAE encode / decode run unchanged from the loaded diffusers model
(one-shot per segment, outside the graph). No MiniMax-Remover source is
imported; the `pipe` is duck-typed through `.transformer` / `.vae` /
`.scheduler` / `.video_processor` and the `expand_masks` / `resize`
helpers.

## Performance (RTX 5060 Ti, SM120, CUDA 13)

### End-to-end (123 frames, 3 segments, fp16 reference baseline)

| Stack | Wall time | Speedup | PSNR vs fp16 |
|-------|-----------|---------|--------------|
| fp16 reference (diffusers) | 30.7 s | 1.0× | — |
| FlashRT NVFP4 (this pipeline) | 11.9 s | **2.6×** | 52.0 / 45.2 dB |

At 24 fps the 123-frame clip is 5.1 s, so the FlashRT stack runs at
**RTF ≈ 2.3** (processing-time / clip-duration); the fp16 reference is
RTF ≈ 6.0.

### Transformer GEMM (NVFP4 vs fp16 matmul, single layer)

| Linear | fp16 matmul | NVFP4 W4A4 | per-layer speedup |
|--------|-------------|------------|-------------------|
| FFN up [5120 -> 13824] | 1.095 ms | 0.840 ms | 1.30× |
| FFN down [13824 -> 5120] | 1.020 ms | 0.864 ms | 1.18× |
| QKV / out [5120 -> 5120] | 0.409 ms | 0.359 ms | 1.14× |

Including cast + quantise overhead, the isolated FP4 GEMM is 4–9× faster
than the fp16 matmul on the large FFN projections (e.g. ffn_up
3.95 ms -> 0.47 ms). NVFP4 is also 1.14–1.30× faster per layer than the
static-quant FP8 GEMM.

### Precision specification

The pipeline keeps the math reference-equivalent on the precision-critical
path (fp32-stat LayerNorm / RMSNorm, interleaved RoPE) and confines the
loss to the quantised GEMMs and the attention backend.

| Component | Metric | Value |
|-----------|--------|-------|
| Attention — SageAttention QK-int8 PV-fp8 (`sage_fp8`, default) | cosine vs SDPA | 0.9993 |
| Attention — SageAttention QK-int8 PV-fp16 (`sage_fp16`) | cosine vs SDPA | 0.9999 |
| NVFP4 W4A4 GEMM | cosine vs fp16 matmul | >= 0.999 |
| End-to-end (full pipeline) | PSNR vs fp16 | 52.0 dB (mean) / 45.2 dB (worst frame) |
| End-to-end (full pipeline) | max abs diff vs fp16 | bounded by the FP4 grid; per-block relative, scales with activation magnitude (median per-pixel deviation < 2 / 255 on 8-bit output) |

The default `sage_fp8` attention gives the best latency at cosine 0.9993;
switch to `FLASHRT_ATTN_MODE=sage_fp16` for cosine 0.9999 at a small
latency cost. NVFP4 needs no calibration, so the first call is already
in the steady state (no warm-up / calibration pass).

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `FLASHRT_ATTN_MODE` | `sage_fp8` | attention backend (`sage_fp8`/`sage_fp16`/`sage`/`fa2`) |
| `FLASHRT_FP4_GEMM` | `pingpong` | GEMM kernel variant (`pingpong`/`plain`/`widen`) |
| `FLASHRT_FUSED_BLOCK` | `1` | fused QKV-quant-once + fused FFN-up GEMM+bias+gelu block (`0` = per-projection re-quant debug path) |
| `FLASHRT_MANUAL_GRAPH` | `0` | capture the whole denoise loop as one CUDA Graph |
| `FLASHRT_NUM_STEPS` | unset | override the denoise step count (default 12) |

## Usage

```python
from flash_rt.models.minimax_remover import MiniMaxRemoverPipeline

# `pipe` is a loaded diffusers Minimax_Remover_Pipeline (transformer +
# vae + scheduler). The FlashRT pipeline consumes it in place.
pipeline = MiniMaxRemoverPipeline(pipe)
output = pipeline(
    images=frames,        # [F, H, W, 3] uint8/np, 0..255
    masks=masks,          # [F, H, W, 1] np, 0/1
    num_frames=len(frames),
    height=720, width=1280,
    num_inference_steps=12,
)
video = output.frames
```

## Model weights

MiniMax-Remover checkpoint + the `Transformer3DModel` / `AutoencoderKLWan`
definitions are loaded by the reference project (unmodified, via the
loaded diffusers `pipe`). This FlashRT pipeline module imports no
MiniMax-Remover source.

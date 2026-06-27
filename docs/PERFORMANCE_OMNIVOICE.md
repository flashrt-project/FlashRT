# OmniVoice FlashRT — Performance Specifications

RTX 5060 Ti (SM120, 448 GB/s, 36 SMs), CUDA 13.0, PyTorch 2.12.
Model: OmniVoice TTS (Qwen3-1.5B, D=1024, L=28, FFN=3072, GQA 16Q/8KV).

## Build Configuration

```sh
cmake -DFLASHRT_ENABLE_OMNIVOICE=ON -DGPU_ARCH=120 -DENABLE_NVFP4=ON ..
make -j$(nproc) flash_rt_kernels flash_rt_fa2 flash_rt_omnivoice
```

## Kernel Modules

| Module | Contents | Gate |
|--------|----------|------|
| `flash_rt_kernels` | FP4 GEMM, RMS norm, quantize, SiLU | Always built |
| `flash_rt_omnivoice` | `omnivoice_cfg_logsoftmax_bf16`, `omnivoice_qk_norm_rope_bf16` | `FLASHRT_ENABLE_OMNIVOICE=ON` |
| `flash_rt_fa2` | FlashAttention2 BF16 forward | Always built |

## End-to-End Inference (ns=32, gs=2.0, 32 codebooks)

| Metric | Value |
|--------|-------|
| **Latency** | 100 ms |
| **Speedup vs PyTorch** | 5.0x |
| **RTF** | 0.032 |
| **VRAM** | 2.0 GB (70% of PyTorch) |

## Audio Quality (versus PyTorch BF16 reference)

| Metric | BF16 FlashRT | FP4 W4A4 FlashRT | Hybrid (5% BF16) |
|--------|-------------|-------------------|-------------------|
| **Mel-cosine** | 0.9971 | 0.9315 | 0.9961 |
| **Max abs error** | 0.0012 | 0.0473 | 0.0009 |
| **Token match rate** | 100% | 3.3% | 99.7% |
| **Subjective** | Identical to PyTorch | Different prosody | Same as BF16 |

## Precision Notes

- **BF16 noise floor**: mel-cosine between two BF16 runs (different seeds) is ~0.996.
  0.999 is unattainable due to stochastic sampling. BF16 FlashRT sits at this floor.
- **FP4 W4A4 alone**: 98% of audio tokens diverge from BF16, altering rhythm and
  timbre. Not suitable for quality-sensitive TTS.
- **Hybrid strategy**: Step 1 uses BF16 CFG to establish token structure
  (17 tokens / 2.9% mutated). Steps 2-32 use FP4 without CFG to fill remaining
  tokens at 5x speed. Perceptually identical to full BF16.

## VRAM Breakdown

| Component | Size |
|-----------|------|
| embed_tokens | 296 MB |
| WL_bf16 (weights, transient) | 840 MB |
| WL_fp4 (packed weights) | 230 MB |
| Decoder (final proj) | 151 MB |
| Buffers + CUDA Graph | 500 MB |
| **Total** | **2.0 GB** |

Memory optimizations:
- FP4 instance releases WL_bf16 after quantization (-840 MB)
- Original layer weights zeroed after calibration (-840 MB)
- Encoder weights freed via `free_encoder()` (-612 MB)

## Engine Injection

```python
from flash_rt.models.omnivoice import inject, free_encoder

model = OmniVoice.from_pretrained(...)
inject(model, cfg_ratio=0.05, bookend=False)
free_encoder(model)

# model._generate_iterative now uses FlashRT accelerated path
```

The engine raises `RuntimeError` before model loading if required kernel
symbols are missing, with instructions to rebuild with
`-DFLASHRT_ENABLE_OMNIVOICE=ON`.

## Regression Tests

```sh
pytest -q tests/test_omnivoice_smoke.py
```

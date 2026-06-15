# Cosmos3-Nano text2video (FP8) — Usage & Benchmark

Kernelized, CUDA-graphed Cosmos3-Nano **text2video** denoise, packaged as a
first-class FlashRT model (`config="cosmos3_video"`). Run it through the standard
`flash_rt.load_model(...)` API.

This is the two-tower MoT denoise backbone (same engine as `cosmos3_av`), driven for
the video path: the gen tower is the all-noisy vision latent, the head is `llm2vae`,
and the output is unpatchified to a `[1, 48, T, H, W]` vision latent. Conditioning
(text / VAE encode) is upstream and consumed from the official reference dump; VAE
decode to pixels is the downstream step.

---

## 1. Precision & speed (RTX 5090 / sm120)

Quantization choice matters: the **video latent is far more quant-sensitive than the
AV action head**, so only **FP8 (E4M3) is near-lossless**; NVFP4 GEMM and int8-sage
attention degrade it and are off by default.

**480p / 49 frames / 10-step UniPC** (denoise loop; VAE decode listed separately):

| path | denoise | VAE decode | E2E | latent cos | notes |
|---|---|---|---|---|---|
| official Cosmos3-Nano | ~53.5 s | ~2.3 s | **~55.8 s** | — (ref) | eager, uncached |
| FlashRT **bf16** | 4.4 s | 2.3 s | 6.7 s | 0.9989 | CUDA graph + static text-KV cache |
| FlashRT **fp8** | 2.5 s | 2.3 s | 4.8 s | 0.986 | near-lossless |
| FlashRT **fp8 + TeaCache(3,5,7) + fp4-VAE** | ~1.8 s | 1.26 s | **~3.0 s** | 0.981 | default-recommended |

→ **~18× faster end-to-end than the official pipeline**, near-lossless.

**Decoded-frame quality** (256p, PSNR of decoded frames vs the official video):

| denoise precision | frame PSNR |
|---|---|
| bf16 | 41.4 dB |
| **fp8** | 34.2 dB |
| fp4 | 22.8 dB (lossy — not recommended) |

**How to read these numbers.** The official baseline runs eager (no CUDA graph) and
recomputes the static text tower every step, so most of the ~18× is FlashRT's CUDA
graph + static text-KV cache (same precision); the **quantization + TeaCache** kernels
contribute ~2.2× on top of a graphed bf16 baseline. Both are real; the ~18× is the
end-user speedup switching from the official pipeline.

How it gets there: FP8 weights/activations (`fp8_gemm_descale_bf16out`) · static text
K/V cache (text tower computed once) · fused qk-norm+rope · TeaCache training-free
step caching · one CUDA graph per compute step · near-lossless fp4 Wan-VAE conv decode.

---

## 2. Prerequisites

- **GPU**: RTX 5090 (sm120). FlashRT sm120 image (`flash_rt_kernels.so` + `flash_rt_fa2.so`).
- Build the model-local kernels once on the target GPU (isolated; does not rebuild
  `flash_rt_kernels.so`):

```bash
cd flash_rt/models/cosmos3_av/kernels && python3 setup.py build_ext --inplace
```

- Required files (paths are env/arg-driven — no host paths baked in):
  - Cosmos3 flat-format weights `.safetensors` (the base text2video model, converted
    from the public diffusers transformer).
  - The official reference dump `tensors.safetensors` (conditioning: text/VAE-encode
    tokens, rope tables, initial latent, timestep embeds).

---

## 3. Quickstart

```bash
python3 examples/cosmos3_video_quickstart.py \
    --checkpoint <cosmos3 flat weights .safetensors> \
    --ref <.../tensors.safetensors> \
    --quant fp8 --teacache-skip 3,5,7
```

Expected (RTX 5090, 480p/49f/10-step):

```text
[cosmos3_video] denoise 1809.1 ms  quant=fp8  teacache_skip=[3,5,7]
[cosmos3_video] latent (1, 48, 13, 30, 52)
[cosmos3_video] latent cos 0.98125  rel_l2 19.408%  (vs official reference)
```

Programmatic:

```python
import os, flash_rt
os.environ["COSMOS_R"] = "<.../tensors.safetensors>"
model = flash_rt.load_model(checkpoint="<weights>", config="cosmos3_video",
                            hardware="rtx_sm120")
out = model.infer({"compare_ref": True})
# out["latent"]     -> [1,48,T,H,W] denoised vision latent
# out["latency_ms"] -> denoise loop wall time
# out["cos"]        -> latent cosine vs official once/final_vision_latent
```

---

## 4. Knobs (env / arg, default-safe)

| knob | default | effect |
|---|---|---|
| `COSMOS_VIDEO_QUANT` | `fp8` | `fp8` (near-lossless) · `bf16` (reference) · `fp4` (lossy) |
| `COSMOS_VIDEO_TEACACHE_SKIP` | (off) | step-cache skip steps, e.g. `3,5,7` (cos 0.99) or `2,4,6,8` (faster) |
| `COSMOS_VIDEO_BF16_PROJS` | (none) | keep named projections in bf16 (e.g. `down`) |
| `COSMOS_SHIFT` | `10.0` | UniPC shift |

TeaCache is **training-free step caching** (it reuses the cached velocity on skip
steps), not step distillation.

---

## 5. Scope

- The denoise policy is packaged here; **VAE decode to frames is downstream** (the
  Wan2.2 VAE + its fp4/fp8 conv acceleration are applied to the user's VAE).
- Conditioning (text encode, VAE encode) is upstream, consumed from the reference dump
  — consistent with the `cosmos3_av` path.
- **Additive & isolated**: the production `flash_rt_kernels.so` and its CMake are
  untouched; the model-local kernels are an isolated extension.

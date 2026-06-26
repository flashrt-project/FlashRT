# MelBandRoformer — FlashRT Inference Pipeline

MelBandRoformer audio source separation (vocals / background) with kernelized
FP8 inference on Blackwell SM120.

## Build

Enable the MelBandRoformer kernels at CMake configure time:

```bash
cd FlashRT
mkdir build && cd build
cmake .. -DFLASHRT_ENABLE_MELBAND_ROFORMER=ON
make -j$(nproc)
```

The kernels are **OFF by default** — they only compile when
`-DFLASHRT_ENABLE_MELBAND_ROFORMER=ON` is passed, so non-MelBand builds are
unaffected. Bindings are gated by `#ifdef FLASHRT_HAVE_MELBAND_ROFORMER`.

## Custom fused kernels

Five fused CUDA kernels in `csrc/kernels/mbr_kernels.{cu,cuh}`, compiled into
`flash_rt_kernels.so`:

| Kernel | Fused operation |
|--------|----------------|
| `mbr_qkv_split_rope` | QKV split + interleaved RoPE → (B,H,S,D) |
| `mbr_gated_attn_quant` | sigmoid gate · attn + reshape + FP8 quant |
| `mbr_fp8_dequant_bf16` | FP8 E4M3 → BF16 dequantize |
| `mbr_resadd_rmsnorm_fp8_keepres` | residual add + RMSNorm → FP8 (keeps residual) |
| `mbr_fused_add_rmsnorm_bf16` | residual add + RMSNorm (BF16 in/out) |

The pipeline also reuses existing FlashRT kernels such as `rms_norm_fp8` and
`bias_gelu_quantize_fp8_static_bf16`. All MelBandRoformer kernels follow
FlashRT conventions: `template<typename T>`, `common.cuh` reuse (`to_f32`,
`from_f32`, `packed2`, `block_reduce_sum`), typed launchers.

## Pipeline

`flash_rt/models/melband_roformer/pipeline.py` — `MelBandRoformerPipeline`:

- FP8 GEMM via `torch._scaled_mm` (cuBLASLt)
- Flash SDPA attention
- FP8 activation scales from offline calibration (`fp8_calibration.json`)
- Deletes original BF16 Linear weights after FP8 quantization (~400 MB saved)
- All compute through FlashRT `fvk.*` and `mbr_*` kernel interfaces

## Performance (RTX 5060 Ti, SM120, CUDA 13)

120 s stereo audio, overlap-add (num_overlap=2), batch_size=4:

| Metric | Baseline (PyTorch) | Optimized (FP8) |
|--------|-------------------|-----------------|
| RTF | 0.077 | 0.025 |
| Speedup | 1.0× | **3.1×** |
| Peak VRAM | 1056 MB | 1507 MB |
| Cosine similarity | — | 0.9997 |
| Max abs diff | — | 0.074 |

## Usage

```python
from flash_rt.models.melband_roformer import MelBandRoformerPipeline

pipeline = MelBandRoformerPipeline(
    frontend, max_seq_len=stft_len,
    model_dir="/path/to/checkpoint")
output = pipeline.forward(audio_batch_bf16)
```

## Model weights

Checkpoint from `poiqazwsx/melband-roformer-denoise` (HuggingFace) or
`cpadyun/melband-roformer` (ModelScope). Requires `melband-roformer-infer`
package (unmodified, loaded via runtime monkeypatch).

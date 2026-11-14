# FlashRT Examples

## Quick Start (Hardware-Agnostic)

```bash
# Set $CHECKPOINT_DIR to the directory holding your model weights
# (Pi0 / Pi0.5 / GROOT format; the library auto-detects config).
export CHECKPOINT_DIR=/path/to/your/pi05_checkpoint
python examples/quickstart.py --checkpoint "$CHECKPOINT_DIR"
```

Auto-detects GPU (SM110 → Thor backend, SM120 → Blackwell backend) and runs one inference pass with precision check.

## Hardware-Specific Examples

### AMD RDNA (W7900D, Radeon 8060S / Halo, R9700)

[Higgs Audio v3 quickstart](amd/higgs_audio_v3_quickstart.py) uses the model's
`create_frontend()` factory to detect the RDNA device. R9700 defaults to native
FP8 W8A8; W7900D and Halo use BF16. `--bf16` selects BF16 explicitly.

```bash
export HIGGS_CHECKPOINT=/path/to/higgs-tts-3-4b
python examples/amd/higgs_audio_v3_quickstart.py --out auto.wav
python examples/amd/higgs_audio_v3_quickstart.py --bf16 --out bf16.wav
```

Use `--hardware amd_rdna3` for W7900D, `amd_rdna35` for Halo, or `amd_rdna4`
for R9700 to check an explicit target. `--device cuda:N` selects the ROCm device.
See [Higgs RDNA setup and precision](../docs/deployment_amd_higgs_audio_v3.md).

### Apple Silicon (experimental MPS BF16)

```bash
python examples/apple/pi05.py \
    --checkpoint /Volumes/SSD/FlashRT/models/pi05-libero-pytorch \
    --tokenizer /Volumes/SSD/FlashRT/tokenizers/paligemma_tokenizer.model \
    --output /Volumes/SSD/FlashRT/runs/apple_mps_pi05 \
    --cpu-reference
```

Runs Pi0.5 on real camera/state fixtures and saves fixed-noise actions and
latency measurements. See [Apple setup and downloads](../docs/deployment_apple.md).

### Thor (Jetson AGX Thor, SM110)

Run inside Docker container (`<your_container>`):

```bash
# Precision check: cosine similarity vs PyTorch reference
python examples/thor/eval_precision.py \
    --checkpoint_dir $CHECKPOINT_DIR

# LIBERO benchmark (full evaluation)
python examples/thor/eval_libero.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --task_suite libero_spatial
```

| Metric | Value |
|--------|-------|
| E2E Latency | 44.3 ms |
| Cosine vs PyTorch | 0.9998 |
| LIBERO Spatial | 98.2% (491/500) |

### Blackwell (RTX 5090, SM120)

```bash
# Precision check
python examples/blackwell/eval_precision.py \
    --checkpoint /path/to/checkpoint

# LIBERO benchmark
python examples/blackwell/eval_libero.py \
    --checkpoint /path/to/checkpoint \
    --task_suite libero_spatial
```

| Metric | Value |
|--------|-------|
| Latency (2-view) | 17.88 ms |
| Cosine vs PyTorch | 0.999638 |
| LIBERO Spatial | 100% (50/50) |

## Adding a New Hardware Target

1. Create `examples/<hardware>/` directory
2. Copy an existing example and adjust engine/FMHA paths
3. The library auto-detects SM version; you may also force backend:
   ```python
   model = flash_rt.load_model("pi05", ckpt, backend="thor")
   ```

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

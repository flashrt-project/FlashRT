# GR00T N1.7 on Thor

First complete [Docker or native setup](README.md). The validated model is GR00T N1.7 3B with its Cosmos-Reason2-2B backbone. Download both, including processor/tokenizer files and action statistics.

## 1. Prepare the models

Docker:

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e PIP_INDEX_URL -v "$MODELS:/models" "$IMAGE" prepare groot
```

Native:

```bash
bash prepare-models.sh groot
```

The script downloads pinned complete snapshots from ModelScope and verifies every file. Existing matching files are reused. If already downloaded, use `$MODELS/GR00T-N1.7-3B` and `$MODELS/Cosmos-Reason2-2B`. GR00T weights/config/statistics matched the NVIDIA checkpoint in this audit; Cosmos was verified against the mirror, without independent gated-Hugging-Face verification.

## 2. Run reference and FlashRT checks

Docker:

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" validate groot
```

Native:

```bash
python local_groot_checkpoint.py --checkpoint "$MODELS/GR00T-N1.7-3B" \
  --cosmos "$MODELS/Cosmos-Reason2-2B" --out "$MODELS/GR00T-local"
export GROOT="$MODELS/GR00T-local"
bash run-validation.sh groot
```

Native overlay creation needs a new output directory; reuse an existing valid overlay on subsequent runs. Docker creates it automatically in the result directory. Only local model paths are changed; original weights remain unchanged. Reference execution runs offline.

The official policy reruns from raw RGB, state and language. FlashRT compares against these newly generated official actions. The fixture has two cameras with two historical frames each, four diffusion steps, and 40 output steps. FP4 means FP8 backbone plus NVFP4 DiT.

## 3. Read the result

Inspect `groot-fp8.json` and `groot-fp4.json` in the result directory. Checks compare all 40×17 physical actions after official decoding: EEF 9, gripper 1 and joints 7. Mean cosine must be ≥0.999, worst ≥0.995, EEF/joint cosine ≥0.995 and gripper maximum absolute error ≤0.05. Reports also include RMSE, maximum errors and repeated-input differences. Tested Docker FP4 cosine: 0.999844.

`last_action_decoder_out` is velocity, not the integrated final action. Comparing it to final actions gives a misleading low score. Near-zero gripper outputs also need absolute-error checks.

## 4. State the timing boundary

The default check now times raw RGB, state and language through fresh official processor preprocessing, FlashRT patch embedding/visual merger/backbone/action head, and physical action decoding. Prompt/grid setup and fixed-sample calibration occur once before timing. Each timed call processes the actual raw images; it does not replay official model embeddings. The processor patches, token IDs and grid are checked against the fresh official capture.

Use `--boundary feature` with `verify_groot_fixture.py` only when deliberately measuring the older post-patch-feature graph boundary. Its approximately 29 ms result excludes preprocessing and embedding generation; the newly verified complete FP4 boundary is approximately 57 ms on the second native machine. Official instrumented capture is not a speed baseline.

The same-sample raw-input FP8/FP4 checks passed all physical-action and repeat gates. This is still a fixed-sample regression with calibration on that sample. Expanding samples, changing prompts/grids or preprocessing requires new setup and evaluation. See the [comparison contract](comparison.md) for the matched official eager-PyTorch measurement and why the JAL TensorRT result uses a different checkpoint/configuration.

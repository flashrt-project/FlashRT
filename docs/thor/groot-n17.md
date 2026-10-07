# GR00T N1.7 on Thor

Complete [Docker or native setup](README.md). The default model is **GR00T-N1.7-LIBERO/libero_10**, with its Cosmos-Reason2-2B backbone, one camera, batch 1 and four denoising steps.

## 1. Prepare the models

Docker:

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e PIP_INDEX_URL -v "$MODELS:/models" "$IMAGE" prepare groot
```

Native, from `repro/thor`:

```bash
bash prepare-models.sh groot
```

The downloader uses versioned ModelScope mirrors, verifies checksums and reuses matching local files. It selects the `libero_10` inference files, excluding other task suites and training optimizer states. Keep both `$MODELS/GR00T-N1.7-LIBERO/libero_10` and `$MODELS/Cosmos-Reason2-2B`.

## 2. Run reference and FlashRT checks

Docker:

```bash
docker run --rm --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -v "$MODELS:/models" -v "$OUT:/results" "$IMAGE" validate groot
```

Native:

```bash
python local_groot_checkpoint.py \
  --checkpoint "$MODELS/GR00T-N1.7-LIBERO/libero_10" \
  --cosmos "$MODELS/Cosmos-Reason2-2B" --out "$MODELS/GR00T-local"
export GROOT="$MODELS/GR00T-local"
bash run-validation.sh groot
```

Use a new output directory when first creating the overlay; reuse it on later runs. Docker creates it automatically. Original weights and statistics remain unchanged. The overlay points to the local backbone and explicitly selects the `image` camera: the checkpoint processor originally configures both `image` and `wrist_image`.

The bundled fixture contains a real LIBERO observation. The official reference runs afresh from RGB, state and language. FlashRT uses the same input, prompt, initial noise and four denoising steps. Processor patches, token IDs and image grid must exactly match the fresh reference. The model generates a padded horizon of 40; official decoding delivers **16 × 7** physical actions.

## 3. Read the accuracy reports

Inspect `groot-fp8.json` and `groot-fp4.json`. Reports include overall and per-action-group cosine, RMSE, maximum absolute errors and repeated-input stability. Overall numerical acceptance requires mean cosine ≥0.999 and worst-sample cosine ≥0.995. Repeated normalized outputs use the existing cosine ≥0.9999 and maximum absolute error ≤0.05 checks; bitwise identity is also reported. Strict per-group diagnostics are reported separately, including failures.

On the verified release-container FP4 fixture, overall cosine is **0.999694**. Rotation cosine is **0.96167**, with maximum absolute error **0.00760**; near-zero components need absolute-error interpretation as well as cosine. This is a fixed-sample numerical regression, not a robot task-success evaluation or proof that every action-group diagnostic passes. FP4 uses an FP8 backbone and NVFP4 action head.

`last_action_decoder_out` is velocity, not the integrated final action. Compare decoded physical actions against the official output.

## 4. Read the latency reports

The default uses process-local **two CPU threads**, with no image or state cache. Setup, prompt preparation, calibration and graph capture occur before warmup. Each measured call freshly processes RGB and state.

| Measurement | Scope |
|---|---|
| Preprocessing | Official CPU processor: image transforms, patch preparation, text/state processing and collation |
| Model inference | Prepared processor input through FlashRT input transfer, backbone and four-step action head |
| Complete call | Raw RGB/state/language through preprocessing, FlashRT inference and decoded physical actions |

The release-container FP4 measurement was **24.26 ms model inference**, **3.25 ms preprocessing** and **27.89 ms complete call**. The separate controlled native two-thread comparison reduced complete-call latency from 30.19 to 29.00 ms (1.19 ms / 3.9%), with bitwise-identical output. Component medians need not sum to the complete-call median. See [verified measurements](results.md) for measured model latency and release-container results, and [comparison contract](comparison.md) for the JAL configuration.

To time the official eager policy without capture hooks:

```bash
"$REF/groot-venv/bin/python" benchmark_groot_reference.py \
  --checkpoint "$GROOT" --fixture fixtures/groot-libero10-onecam.pt \
  --reference-records "$OUT/groot-reference.pt" --embodiment LIBERO_PANDA --cpu-threads 2 \
  --out "$OUT/groot-official-eager.json"
```

Use `--boundary feature` only for the older captured-feature boundary. It excludes image/text embedding generation and cannot substitute for model-inference or complete-call latency.

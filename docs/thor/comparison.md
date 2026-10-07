# Thor comparison contract

The shipped checks reproduce FlashRT numerical results and include a matched raw-input GR00T comparison against official eager PyTorch. They do **not** establish a matched TensorRT speedup. Run both backends on the same Thor, checkpoint, input and output boundary before publishing a TensorRT ratio. Official reference capture is instrumented; its diagnostic latency is not a performance baseline.

## π0.5

Use the OpenPI `pi05_libero` checkpoint and its normalization assets, two real camera images, batch 1, 10 denoising steps, horizon 10 and seven physical output dimensions. Match state handling, prompt, initial noise and calibration samples. Record the converted checkpoint identity, resolved quantization options and software/power settings.

The verified FlashRT command, from the prepared reproduction directory, is:

```bash
export PI05="$MODELS/pi05-openpi"
bash run-validation.sh pi05
```

The fixed-frame benchmark times `infer(observation)` from before the Python call to after GPU synchronization, including image processing and delivered-action decoding inside that call. Prompt setup and eight-frame calibration occur before timing. It uses 20 warmups and 100 measured calls. `pi05-fp4.json` records resolved options; `pi05-fp4-accuracy.json` records the physical-action comparison against official OpenPI using the same observations/noise. Confirm the comparator uses this same boundary rather than only a TensorRT engine call.

The [JAL OpenPI tutorial](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) reports both total and model latency for `pi05_libero`, horizon 10. Its published totals are useful context, but the supplied FlashRT audit did not execute its TensorRT engine with identical images, noise, timing wrapper and device settings. The approximately 20 ms FlashRT result therefore has no validated apples-to-apples TensorRT speedup attached. Closed-loop `predict()` rollout latency also has a different calibration and measurement protocol.

## GR00T N1.7

The currently verified configurations differ:

| Setting | FlashRT reproduction | JAL published TensorRT benchmark |
|---|---|---|
| Checkpoint | GR00T-N1.7-3B | GR00T-N1.7-LIBERO/libero_10 |
| Embodiment/input | DROID, two cameras × two historical frames | LIBERO, one camera |
| Denoising | 4 steps | 4 steps |
| FlashRT FP4 recipe | FP8 backbone, NVFP4 DiT | Mixed NVFP4 also quantizes the LLM |
| Default timed input/output | Raw RGB/state/language → physical actions | Full pipeline inference |

The JAL configuration is documented in the [official GR00T tutorial](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/). Its number cannot be divided by FlashRT's older feature-input number, or by the new raw-input result using a different checkpoint and input configuration.

Verified FlashRT command after the [model overlay setup](groot-n17.md):

```bash
export GROOT="$MODELS/GR00T-local"
bash run-validation.sh groot
```

This freshly executes the official policy from raw observations to obtain reference actions. Default `raw-full` FlashRT validation reruns the official processor on the raw RGB/state/language each call, computes embeddings and backbone/action head with FlashRT, and includes physical decode. Static prompt/grid setup and calibration occur before timing. Captured embeddings are calibration inputs only. Processor patches, token IDs and grid must exactly match the fresh reference. Warmup=20, measured calls=100, batch=1; both ends of each call are GPU synchronized.

The independent native check passed both raw-input tiers: FP8 cosine 0.999835 at 76.74 ms; FP4 cosine 0.999751 at 56.81 ms. Official eager PyTorch on the same raw sample/checkpoint measured 133.38 ms, with no capture hooks and exactly matching reference actions. These are fixed-sample observations on the recorded native environment, not a TensorRT or optimized-PyTorch comparison.

To repeat the clean eager measurement after reference capture, use the installed official reference environment:

```bash
"$REF/groot-venv/bin/python" benchmark_groot_reference.py \
  --checkpoint "$GROOT" --fixture "$GROOT_FIXTURE" \
  --reference-records "$OUT/groot-reference.pt" \
  --out "$OUT/groot-official-eager.json"
```

The explicit optional `--boundary feature` validator starts from captured post-patch features and image/text embeddings and ends at normalized action. Keep that result separate from raw-input timing; physical decoding is compared but excluded from its timer.

There is no executed, validated TensorRT harness for the exact FlashRT feature boundary in this release. Matching that boundary would require a separate instrumented TensorRT runner and a common fixture; no untested TensorRT command is provided here. See [verified results](results.md) for what has actually passed.

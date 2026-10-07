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

The default reproduction now uses **GR00T-N1.7-LIBERO/libero_10**, one camera, four denoising steps and batch 1, matching the model and input configuration stated in the [JAL tutorial](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/). Both FlashRT and the fresh official reference explicitly use the `image` camera. The overlay changes the processor's original two-camera configuration without changing weights.

| Measurement | FlashRT report | JAL report |
|---|---|---|
| Model inference | Input transfer + backbone + action head; excludes CPU processor and physical decode | Backbone and action head reported separately |
| Preprocessing | Fresh official processor on each RGB/state/language input | CPU data processing measured separately and shared across backends |
| Complete call | Direct wall-clock measurement including physical decode | Per-iteration sum of processing, backbone and action-head measurements; excludes physical decode |

FlashRT reports its own processor time, rather than substituting a separately measured shared processing array. The README model-latency column retains the model-inference boundary; preprocessing and complete-call latency have separate columns. Use the complete-call result when discussing raw-image response time.

The official PyTorch source revision is managed by the installer. Fresh capture verifies the same processor patches, tokens, grid and physical actions. Performance measurements run separately without capture hooks. The default validator uses two CPU threads; report the same thread count for the comparator. The JAL timing protocol is five warmups and 20 measured iterations; validator JSON records its actual settings.

To repeat:

```bash
export GROOT="$MODELS/GR00T-local"
bash run-validation.sh groot
```

See [model preparation](groot-n17.md) and [verified results](results.md). Earlier DROID two-camera feature-input results remain historical records and are not the current LIBERO comparison. The published JAL TensorRT result is approximately 40 ms, but its engine has not been executed on the same test device. Do not publish a matched TensorRT speedup ratio from these different-machine measurements.

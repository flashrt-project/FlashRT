# Pi0.5 on Ascend

The NPU backend keeps its code in `flash_rt/npu`, native kernels in
`csrc/npu`, and build scripts in `scripts/npu`. Its build does not enter the
CUDA or AMD build trees. Tested environment: Ascend 910B4, CANN 8.5.2,
PyTorch 2.7.1 and matching torch_npu 2.7.1.post2.

## Build and run

Activate the matching PyTorch environment and source the CANN toolkit's
`set_env.sh`, then build the optional INT8 kernels:

```bash
bash scripts/npu/build.sh
```

`ASCEND_TOOLKIT_HOME` selects the toolkit, `ASCEND_SOC_VERSION` selects the
compiler target (default `Ascend910B4`), and `FLASHRT_NPU_BUILD_DIR` selects
the output directory. A nondefault library location can be supplied using
`FLASHRT_NPU_LIBRARY`.

```python
from flash_rt.npu.frontends.torch.pi05 import Pi05TorchFrontendNpu

pipe = Pi05TorchFrontendNpu(checkpoint_dir, num_views=2, use_int8=True)
pipe.set_prompt(task, state=normalized_state)
pipe.calibrate_with_real_data(real_calibration_observations)
result = pipe.infer(observation, noise=fixed_noise)
pipe.precision_spec.to_json("precision.json")
```

Each observation contains `image` and `wrist_image`: real `uint8` HWC
camera arrays of shape `(224,224,3)`. Resize/pad using the model's official
preprocessing before calling this API. Calibration samples can supply
`prompt`, normalized `state`, and `(chunk_size,32)` float32 `noise`.
Otherwise the current prompt and deterministic model diffusion noise are
used. Samples must be nonempty real observations; no synthetic camera
fallback exists. Perform calibration and prompt changes while the frontend
is idle. Set `use_int8=False` for the BF16 path, which needs no calibration.

## Precision and execution

Encoder linear weights use symmetric INT8 per output channel. Real
calibration collects per-image-token activation maxima and a shared maximum
for language tokens, first across calls within each sample and then through
`flash_rt.core.calibration.accumulate_amax` across samples. Scales are frozen
before capture. An unseen prompt length reuses the calibrated language
scale; it does not collect statistics from a serving request. The decoder
and vision tower remain BF16, with explicit FP32 setup/reference operations.

This follows the useful parts of the Orin W8A8 pipeline: row/channel scale
separation, native quantization producers, shape-aware kernels, and immutable
replay buffers. The Ascend implementation uses its own kernels and ABI.
Missing native libraries fail explicitly rather than selecting a fallback.
INT8 inference before real-data calibration also fails explicitly.

`set_prompt` computes language embeddings and constructs a graph for the
prompt length. Cached prompt updates complete their producer stream before
replay. The graph covers image normalization, all vision and encoder layers,
all denoising steps, and action unnormalization. No temporal KV reuse or
step reduction is enabled.

Steady-state execution uses AscendCL host copies and native graph replay.
`NativeReplay.enqueue()` submits without a CPU wait; `wait()` completes
host output access. Synchronous `infer()` necessarily waits once before
returning independent NumPy output arrays. Graph construction and tensor
allocation occur during setup. Replay retains its graph dependencies and
pinned buffers even if the builder is released.

## Validation and measurement scope

Use an independent official full-model reference and disjoint calibration
and heldout episodes. Judge per-sample full raw-action cosine, and also
report the seven action channels and unnormalized robot actions. Layer
comparisons are diagnostics, not the final accuracy gate.

The corrected BF16 path passed 16 real LIBERO frames against an official
FP32 host with minimum raw-action cosine 0.9999829. A paired vision precision
change reduced median infer latency from about 79.4 to 71.9 ms on the tested
machine. These timings include host image/noise upload, normalization,
complete model execution, output unnormalization and download. They exclude
checkpoint loading, camera resize, tokenization, calibration and capture.
They are not a claim of a hardware limit or task-success validation.

The historical 200-to-86 ms report used a local golden that shared model
errors with its serving path: incorrect vision attention axes and an omitted
encoder attention output projection. Do not use that golden or its cosine
as validation for the corrected model. Preserve historical artifacts when
reproducing the investigation, and generate independent references for new
checkpoints.

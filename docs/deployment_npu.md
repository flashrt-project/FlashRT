# Pi0.5 on Ascend

The NPU backend keeps its code in `flash_rt/npu`, native kernels in
`csrc/npu`, and build scripts in `scripts/npu`. Its build does not enter the
CUDA or AMD build trees. Tested environment: Ascend 910B4, CANN 8.5.2,
PyTorch 2.7.1 and matching torch_npu 2.7.1.post2.

## Build and run

Activate the matching PyTorch environment and source the CANN toolkit's
`set_env.sh`, then build the native kernels:

```bash
bash scripts/npu/build.sh
```

`ASCEND_TOOLKIT_HOME` selects the toolkit, `ASCEND_SOC_VERSION` selects the
compiler target (default `Ascend910B4`), and `FLASHRT_NPU_BUILD_DIR` selects
the output directory. A nondefault library location can be supplied using
`FLASHRT_NPU_LIBRARY`.

The compiled target must match the current device. A stale library or an
architecture mismatch fails before the checkpoint is loaded.

```python
from flash_rt.npu.frontends.torch.pi05 import Pi05TorchFrontendNpu

pipe = Pi05TorchFrontendNpu(checkpoint_dir, num_views=2, use_int8=True)
pipe.set_prompt(task, state=normalized_state)
pipe.calibrate_with_real_data(real_calibration_observations, percentile=90)
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
QKV and gate/up projections share quantized inputs while keeping separate
GEMMs. Their inputs come directly from fused RMSNorm and frozen row
quantization, including the attention residual update before the MLP.
The encoder down projection consumes INT8 produced directly by fused
GELU, multiplication and static quantization. Encoder attention writes INT8
directly for its output projection, using the maximum of frozen row scales
after house aggregation as its scalar output scale. Q/K rotary embedding
shares one native kernel and retains FP32 arithmetic before BF16 rounding. Decoder kernels fuse rotary
embedding with KV writes and gated residual updates with AdaRMS normalization.

Vision attention weights are padded at setup from 72 to 80 channels per
head, including zero QKV bias channels and zero input columns in the output
projection. Attention still uses the original 72-channel scale. The aligned
GEMM outputs eliminate runtime padding, clearing and output slicing without
changing the real-valued attention expression. Vendor kernel tiling can alter
floating-point rounding, so the complete padded path is qualified separately.
Uint8 normalization and CHW patch gathering share a native input kernel. A
256-entry setup lookup preserves the original FP32 divide/subtract and BF16
rounding exactly for every byte value. The kernel writes FP32 patch tokens
directly for the patch projection.

Vision normalization parameters are converted to BF16 once during setup; a
persistent full-size zero operand also avoids per-normalization broadcasting.
This constant preparation was bit-exact on all 56 BF16 and 56 INT8 frames.

This follows the useful parts of the Orin W8A8 pipeline: row/channel scale
separation, native quantization producers, shape-aware kernels, and immutable
replay buffers. The Ascend implementation uses its own kernels and ABI.
Missing native libraries fail explicitly rather than selecting a fallback.
INT8 inference before real-data calibration also fails explicitly.

`set_prompt` computes language embeddings and constructs a graph for the
prompt length. Cached prompt updates complete their producer stream before
replay. The graph covers image normalization, all vision and encoder layers,
all denoising steps, and action unnormalization. All 18 encoder K/V pairs
are preserved. Once the final pair is produced, its unused attention output
and MLP calculations are omitted because the action API never consumes the
final encoder hidden state. Calibration still executes every observer site;
precision metadata describes only live serving bindings. In the INT8 path,
MLP residual additions enter the following layer's fused RMS quantizer.
No temporal KV reuse or step reduction is enabled.

The decoder treats action queries as separate single-query batches sharing
the same current-frame K/V page table. This retains the original fully
bidirectional attention expression with one KV head. It uses CANN's
incremental attention kernel; its rounding is qualified by full-model E2E,
not assumed bit-exact. The vendor API requires page capacity summed over
queries even when page indices are shared. For the tested ten-query shapes,
K/V storage is about 112.5 MiB per prompt bucket, roughly 102 MiB more than
the contiguous baseline. This implementation is specific to Pi0.5's single
KV head and does not claim a general multi-KV-head paging layout.

The action input weight is rounded to BF16 and promoted to FP32 once at
setup. The biased action output GEMM rounds to BF16; the update kernel
promotes that velocity and performs multiplication and subtraction
separately in FP32. It introduces no BF16 rounding of the scaled velocity.

Only `state_prompt_mode="exact"` is implemented: warm representative state
prompt buckets before serving to avoid capture on a new length. Fixed padded
prompts and reduced vision depth raise explicitly. This preserves the AMD
frontend's API boundary without silently changing the requested behavior.

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

The INT8 and BF16 paths passed 96 real LIBERO frames against an independent
official FP32 host. INT8 calibration used 80 real frames covering 40 tasks at
house percentile 90. The first 56 evaluation frames were episode-disjoint
from calibration but were used for percentile selection. A further 40
frames, one per task from separate episodes, provided independent confirmation.
The minimum cosine across all 96 frames was 0.993905 for full raw actions,
0.993986 for the seven action channels and 0.997725 for robot actions.
Every raw/action7 result passed the 0.99 INT8 hard gate; one confirmation
frame did not reach the 0.995 target.

The default percentile remains 99.9. The original eight-frame calibration
recipe also passed all 96 frames (88 heldout), with minimum raw cosine
0.993202; one confirmation frame missed the 0.995 target. Expanding
calibration to 80 frames at percentile 99.9 failed the E2E gate. More
calibration samples alone do not guarantee better scales. The BF16 path
passed all 96 frames with minimum raw cosine 0.999981, above its 0.9999 gate.
These are numerical agreement results, not task-success measurements.

On the tested 910B4, the median of per-frame latency medians was 44.31 ms
for the 16-frame paired benchmark, versus 44.44 ms with the previous image
preprocessing path in the same process. All 96 INT8 outputs were bitwise
identical to that parent. The frozen corrected BF16 baseline at the same
boundary measured 91.08 ms, giving approximately 2.06x acceleration.
Timings include host image/noise upload, normalization, complete model
execution, output unnormalization and download. They exclude checkpoint
loading, camera resize, tokenization, calibration and capture.

The captured profile contains 2207 device kernels, including 122 INT8 GEMMs,
35 RMSNorm/quantization producers and 17 GELU/product/quantization producers.
It has no standalone row quantizers and retains one cast. This result does
not establish a hardware limit or complete fusion; normalized compute and
HBM utilization have not been measured for this exact implementation.

The historical 200-to-86 ms report used a local golden that shared model
errors with its serving path: incorrect vision attention axes and an omitted
encoder attention output projection. Do not use that golden or its cosine
as validation for the corrected model. Preserve historical artifacts when
reproducing the investigation, and generate independent references for new
checkpoints.

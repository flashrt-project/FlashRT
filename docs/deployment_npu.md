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

That produces four shared objects in `flash_rt/npu/lib`. They are separate
because the CANN kernel headers define a per-translation-unit tiling symbol and
a cube-only unit cannot share a file scope with a mixed cube/vector one: the
vector dispatch unit, the gate/up cube unit, the decoder GEMM and the decode
attention.

`ASCEND_TOOLKIT_HOME` selects the toolkit and `FLASHRT_NPU_BUILD_DIR` the output
directory. `ASCEND_SOC_VERSION` exists but accepts only `Ascend910B4`: the host
tiling names that part and several kernels divide work by its twenty cube cores,
so the build refuses to emit libraries whose tiling would be wrong for another
target. Individual libraries can be overridden with `FLASHRT_NPU_LIBRARY`,
`FLASHRT_NPU_CUBE_LIBRARY`, `FLASHRT_NPU_DECODER_LIBRARY` and
`FLASHRT_NPU_ATTENTION_LIBRARY`.

Every library exports its SoC target and an ABI number, and every loader checks
both against each other and against the running device before binding an entry
point. A partial rebuild or a mismatched override therefore fails at load with a
message naming the library, rather than calling a stale contract behind
identical symbol names.

## Selecting a tier

```python
from flash_rt.api import load_model

model = load_model(checkpoint_dir, hardware="npu", precision="int8", num_views=2)
model.calibrate(real_calibration_observations, percentile=90)
actions = model.predict(images, prompt=task, state=normalized_state)
```

`precision` takes `"bf16"` (the default tier for this part), `"int8"`, or
`"auto"`. 910/A2 parts have no FP8 hardware, so `precision="fp8"` is refused
rather than silently downgraded.

The INT8 tier freezes its scales at setup and cannot serve until it has seen
real data, so `predict()` raises until `calibrate()` has run. Give it
observations from the distribution the robot will actually run: one frame will
produce scales, and they will be bad ones.

Parity work needs the diffusion noise pinned to the reference's draw, which
`predict()` deliberately does not expose. `model.pipeline` reaches the
frontend's own `set_prompt`/`infer`/`precision_spec` surface for that:

```python
pipe = model.pipeline
pipe.set_prompt(task, state=normalized_state)
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
scale; it does not collect statistics from a serving request. The vision
tower remains BF16, with explicit FP32 setup/reference operations.
QKV and gate/up projections share quantized inputs while keeping separate
GEMMs. Their inputs come directly from fused RMSNorm and frozen row
quantization, including the attention residual update before the MLP.
The encoder down projection consumes INT8 produced directly by fused
GELU, multiplication and static quantization. Encoder attention writes INT8
directly for its output projection, using the maximum of frozen row scales
after house aggregation as its scalar output scale. Q/K rotary embedding
shares one native kernel and retains FP32 arithmetic before BF16 rounding. Decoder kernels fuse rotary
embedding with KV writes and gated residual updates with AdaRMS normalization.

The action decoder's four projections — QKV, attention output, gate/up and the
MLP down projection — are INT8 as well, on the same frozen-scale contract, with
one activation scale per (layer, denoise step, projection). Their quantization
rides in the AdaRMS and the GELU that already write the activation rather than
adding a kernel to feed them, and the FP16 the cube emits is consumed directly
by the rotary. The decoder's cross attention runs on the raw cube path over a
KV cache that keeps the values transposed in fractal NZ; that layout is what
lets both of its GEMMs take the operand form a raw `Mmad` wants, with nothing
to convert. Selecting `precision="bf16"` leaves the whole decoder, and the
encoder, in BF16.

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

Both tiers were built through `load_model` and judged against an independent
official FP32 host, on a 910B4. INT8 calibration used 80 real frames covering
40 tasks at house percentile 90. Two evaluation groups: 16 in-distribution
frames, and 40 held-out frames one per task from separate episodes. No
evaluation frame appears in the calibration set; the check is by content hash,
not by construction.

Per-frame full raw-action cosine:

| tier | group | minimum | median | below 0.995 |
|---|---|---|---|---|
| bf16 | in-distribution (16) | 0.999983 | 0.999997 | 0 |
| bf16 | held-out (40) | 0.999992 | 0.999997 | 0 |
| int8 | in-distribution (16) | 0.996836 | 0.999293 | 0 |
| int8 | held-out (40) | **0.994491** | 0.999231 | **1** |

The seven action channels track raw within 1e-4 throughout. All outputs finite.

**The accepted standard, stated rather than implied.** The BF16 tier clears the
0.995 bring-up target on every frame with three orders of magnitude to spare,
and it is the default. The INT8 tier is opt-in, clears 0.99 everywhere, and has
a median of 0.9992 — but one of the 40 held-out frames sits at 0.9945, below
0.995. That frame's BF16 result is 0.99999, so the shortfall belongs to the
INT8 tier, not to this port's numerics. Ship INT8 where a 1.25x frame-rate gain
is worth a tail frame at 0.9945 and measure task success before relying on it;
otherwise stay on BF16. Raising percentile to 99.9, and the earlier
eight-frame calibration recipe, were both measured and neither removed the
frame. These are numerical agreement results, not task-success measurements.

Steady-state frame latency, production form — one prompt, one observation, graph
captured, timed after warm-up, 50 frames:

| tier | median | p90 | frame-0 cosine |
|---|---|---|---|
| bf16 | 45.131 ms | 45.231 ms | 0.9999963 |
| int8 | **36.028 ms** | 36.176 ms | 0.9992928 |

Timings include host image/noise upload, normalization, complete model
execution, output unnormalization and download. They exclude checkpoint
loading, camera resize, tokenization, calibration and capture. A per-frame
benchmark that changes prompt every frame reads higher — 41.0 ms INT8 and
49.9 ms BF16 over the 56 evaluation frames — because a new token length pays
graph capture.

Replay is bit-exact: 300 replays of the captured INT8 frame produce one
distinct answer, with the BF16 decoder arm through the same harness as the
control. Judge determinism on at least 300 replays; a five-replay check passes
about half the time on a broken arrangement and has signed one off before.

This result does not establish a hardware limit. `cube_utilization` on the
decode attention reads 24% and the kernel is still the frame's most idle
bucket; splitting its key range takes occupancy to 95% and does not move the
wall, because 9.4 us of its 19 us is spent before the kernel does any work.

The historical 200-to-86 ms report used a local golden that shared model
errors with its serving path: incorrect vision attention axes and an omitted
encoder attention output projection. Do not use that golden or its cosine
as validation for the corrected model. Preserve historical artifacts when
reproducing the investigation, and generate independent references for new
checkpoints.

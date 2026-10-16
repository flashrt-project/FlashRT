# DM05-LIBERO FlashRT — Recorded Performance

These are the author's measurements before the frontend integration follow-up.
They do not establish numerical parity or latency for the current revision.

RTX 5060 Ti (SM120, 16 GB, 36 SMs, ~387 GB/s measured), CUDA 13.2, driver 595.84,
PyTorch 2.11.0+cu130, Python 3.12.

Model: DM05-LIBERO. SigLIP vision tower (27 layers, 448 px, 2 views, head_dim 72,
16 heads), Gemma3 prefix (34 layers, D=2560, FF=10240, 8 query / 4 KV heads,
head_dim 256, 5 full + 29 sliding layers) and an action expert (34 layers, D=1024,
FF=4096, 8 query / 4 KV heads, head_dim 256, action dim 32, 10 Euler steps).

The compute pipeline lives in `flash_rt/models/dm05/pipeline_rtx.py`, with a
dedicated Torch frontend and dispatch entry. DM05's Gemma3 prefix is distinct
from Pi0.5's PaliGemma prefix; overlapping kernel shapes do not imply identical
model semantics. `_ext` adapts calls to this repository's compiled kernels.

## Build Configuration

```sh
cmake -S . -B build -DFLASHRT_ENABLE_DM05=ON -DGPU_ARCH=120
cmake --build build -j4 --target flash_rt_kernels flash_rt_dm05 flash_rt_fa2
```

## Kernel Modules

| Module | Contents | Gate |
|--------|----------|------|
| `flash_rt_kernels` | FP8/NVFP4 GEMM, the norm/quantize family, and the three DM05 fusions below | Always built |
| `flash_rt_fa2` | FlashAttention2 BF16/FP8 forward | RTX attention build |
| `flash_rt_dm05` | DM05 attention and W8A16 kernels | `FLASHRT_ENABLE_DM05=ON`, SM120 |

## Entry Point

```python
from flash_rt import load_model
model = load_model(checkpoint, config="dm05", hardware="rtx_sm120", use_fp4=True)
model.pipeline.load_calibration_state(calibration_state)
actions = model.predict(images=images, prompt=prompt, state=state)
```

The frontend owns the checkpoint, preprocessing wrappers, calibration and graph
capture. No external runtime methods or transform lists are replaced. The public
API example above is the current integration; the measurements below describe the
earlier implementation and need a fresh complete-action and latency run.

## End-to-End Inference

Equal-scope E2E: 8 real observations, warm-up, then 7 timed `_predict` +
`synchronize`, all 3675 kernels in one CUDA graph, 0 eager, 0 ATen, 0 Triton.
Measured with the same profiler on the same session as the native control — the
delivered number was taken on an earlier day on a shared card, so a cross-day
comparison would not attribute anything.

| Configuration | E2E median | Speedup vs PyTorch | Kernels/predict | GPU busy |
|---------------|-----------|--------------------|-----------------|----------|
| PyTorch (`inference_config._predict`, unmodified) | 253.0 ms | 1.00x | — | — |
| DM05 native pipeline (delivered) | 70.62 ms | 3.58x | 3675 | — |
| DM05 native pipeline, same-session control | 70.93 ms | 3.57x | 3700 | 0.9945 |
| **This pipeline** | **70.91 ms** | **3.57x** | **3675** | **0.9967** |

Operator provenance, same profiler, same observations:

| | E2E | GPU kernel | Kernels/predict | ATen model math | Triton model math | Graph-external model math |
|---|---|---|---|---|---|---|
| PyTorch | 253.02 ms | 225.05 ms | 25543 | 28 | 34 | 922 |
| This pipeline | 70.91 ms | 70.67 ms | 3675 | 0 | 0 | 0 |

The PyTorch path leaves 922 model-math ops outside the graph and issues 28 ATen and 34
Triton model-math invocations; this pipeline runs every one of its 3675 kernels inside
a single CUDA graph replay, with zero framework dispatch. That is where most of the
3.57x comes from, not from a faster kernel.

**Compare against the same-session control, not the delivered figure.** The two native
rows are the *same configuration* measured on different days and they differ by 0.31 ms,
which is fifteen times the gap between this pipeline and the native control. That drift
is why the control exists: without it, cross-day movement would be read as the port's
cost. The number that answers "did the port make this slower" is 70.91 against 70.93.

The PyTorch row is the runtime's own `_predict` with nothing installed — the same
reference object used by the earlier integration — measured through
the same profiler on the same observations. It is worth stating because the 106.28 ms
figure that appears in this model's optimization history is **not** that baseline: it
is the campaign's own (already-optimized) starting point, and using it here would have
overstated the speedup by roughly 2.4x.

The port is level with the native control (70.91 against 70.93, inside noise), carries
five fewer kernels, and sits 0.29 ms from the delivered figure measured on an earlier
day.

### Where the time went

The port initially measured 75.7 ms. Three substitutions accounted for the whole gap;
each was found by diffing two profiler receipts by **kernel time**, then confirmed with
a no-model A/B before being wired in.

| Change | E2E | Kernels |
|--------|-----|---------|
| — | 75.7 ms | 4904 |
| fuse `qk_norm_rope_v8` | 73.9 ms | 3782 |
| vectorize `bias_gelu_quant_fp8_v8` | 71.8 ms | 3782 |
| fuse `res_ln_quant` | 70.91 ms | 3675 |

## Accuracy (444 real LIBERO observations)

The reference is the **unmodified `inference_config._predict`** -- the PyTorch path.
Verified rather than assumed: running that path over the same 444 observations
reproduces the gate reference bit for bit (`min cosine 1.000000`), which is what makes
this gate a PyTorch comparison and not merely a self-consistency check.

Metric is the 6-DoF arm, as in the gate.

| Metric | This pipeline | Native pipeline (delivered) |
|--------|---------------|------------------------------|
| Minimum cosine | 0.990153 | 0.990423 |
| 1st percentile | 0.996626 | -- |
| Median | 0.999921 | -- |
| Observations below 0.99 | 0 | 0 |

Quality against the PyTorch reference, over the same 8 observations:

| | cosine | max abs | rel-L2 |
|---|---|---|---|
| This pipeline vs PyTorch | 0.99990916 | 6.855e-2 | 1.351e-2 |
| Native pipeline vs PyTorch | 0.99991268 | 6.855e-2 | 1.325e-2 |

The two rows are the point: this pipeline deviates from PyTorch by essentially exactly
what the native pipeline already deviated by. The distance comes from the static
FP8/NVFP4 quantization the model was calibrated for, not from the port — if the port
had introduced its own error it would show up as a gap between these rows.

**The gripper dimension is not bit-exact and the gate does not cover it.** Over the 444
observations no single one matches the PyTorch gripper value exactly (mean absolute
difference 0.00529). Four flip outright, at a maximum difference of 2.0. The gripper is
a near-binary signal, so a last-ulp difference in the arm logits can cross its decision
boundary; the 6-DoF arm metric above is unaffected, which is why the gate uses it. If a
downstream consumer is sensitive to gripper timing, this is the number to look at.

The gate reads identically across every configuration above — the composition, the
fused kernels and the vectorized kernel all give min 0.990153 / median 0.999921. That
is what makes the latency series attributable: the output never moved, so the clock
only measured kernel work.

## The Three DM05 Fusions

Each of these replaced a composition of this repository's kernels that was already
**bit-exact** with the reference. They are optimisations, not correctness fixes, and
the gate confirms it by not moving.

| Kernel | Replaces | Gain |
|--------|----------|------|
| `gemma3_qk_norm_rope_v8_bf16` | `gemma3_rms_norm_v8_bf16` ×2 + `rope_hf_bf16` ×2 | 1 launch instead of 4; +1191 of the +1204 extra launches |
| `bias_gelu_quantize_fp8_static_bf16_rounded_v8` | the same kernel at one element per thread | 306 → 756 GB/s, 2.47× |
| `res_ln_quant_v8_bf16` | `bias_residual` + `layer_norm` + `quantize_fp8_static` | 3 launches → 1, and removes a 4.7 MB/call scratch |

`gemma3_qk_norm_rope_v8_bf16` was necessary because the existing
`csrc/kernels/qk_norm_rope_fused.cu` is fp16 with a layer norm, laid out for
head_dim 128 (the Chameleon shape); DM05 is bf16 with a `(1+w)` RMSNorm at
head_dim 256, so none of it applies. `bias_gelu_quantize_fp8_static_bf16_rounded_v8`
keeps the sibling's arithmetic verbatim and changes only the access pattern to 8
columns per thread. `res_ln_quant_v8_bf16` keeps the residual and normalised values in
`v0/v1` registers across both passes, which is why it can leave `xn` untouched — the
composition had to write it to DRAM purely because `layer_norm` and
`quantize_fp8_static` are separate kernels.

## Precision Notes

**Division is not IEEE in this repository's kernels.** `flash_rt_kernels` is compiled
with `--prec-div=false`; an externally-built reference is compiled with nvcc defaults,
where division *is* correctly rounded. The two agree exactly when the divisor is a
power of two and differ by an ulp otherwise. The DM05 path pins the division sites
with `__fdiv_rn` — in `resize_norm.cu`, in six places in the `v8_block_sum` family, and
in the branch revision originally measured, in `layer_norm_kernel`'s mean and inv_std. The last two were found only after a 444
gate run came back at min 0.989149 against a 0.99 threshold with a median of 0.999922:
a handful of elements per 2.36M moved across a bf16 boundary, which at the real
calibration scales flipped a few quantized FP8 values. A tail effect, never a median
one — if the median moves, look for a topology error instead.

**Preprocessing ownership.** `build_transforms` returns frontend-local image and
tokenization wrappers. It leaves the model package's transform list unchanged.
State-bearing prompts rebuild state tokens instead of using a text-only cache hit.

**The reference's fast-math primitives are pinned in the fusions.**
`rcp.approx.ftz.f32` and `mul.rn.ftz.f32` are reproduced as inline asm in
`res_ln_quant_v8_bf16`; `1.0f / x` is not the same operation. Its reduction likewise
reproduces `frt_block_sum`'s tree rather than calling `block_reduce_sum`.


## Integration follow-up (2026-10-17)

The shared `layer_norm_kernel` template has been restored to current main for both
FP16 and BF16. The new DM05 fused kernels remain separate additions; this model
integration does not change existing LayerNorm arithmetic. The measurements above
are the author's earlier revision records, not a new validation of this follow-up.

The integration now routes `("dm05", "torch", "rtx_sm120")` to its own frontend
and `models/dm05/pipeline_rtx.py`. The frontend owns loading, input/output
transforms, calibration and graph capture. The serving example calls the public
VLA API; it never replaces external runtime methods.

The 444-observation complete-action gate is still pending. In particular, the four
reported gripper decision flips have not been retested or declared fixed.
See [the validation procedure](dm05_sm120.md#complete-action-validation).

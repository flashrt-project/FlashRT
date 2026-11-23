# Pi0.5 SM87 strict visual consumers

This opt-in path leaves BF16 GEMMs unchanged and fuses their consumers:
patch bias/position embedding/LayerNorm, QKV bias/split, and projector
bias/first encoder RMSNorm/dynamic token INT8. Existing strict FFN bias/GELU
and residual/LayerNorm kernels stay unchanged.

## Build and enable

```bash
cmake -S . -B build -DGPU_ARCH=87 -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DFA2_ARCH_NATIVE_ONLY=ON -DFA2_HDIMS="96;128;256" -DFA2_DTYPES=bf16 \
  -DFLASHRT_ENABLE_PI05_STRICT_VISUAL_SM87=ON
cmake --build build -j4 --target flash_rt_kernels flash_rt_fa2
python -c "from flash_rt import flash_rt_kernels as k; print(k.__file__); print(k.pi05_strict_visual_available())"
python -m pytest tests/test_pi05_strict_visual.py -q
```

Set both runtime options before constructing the frontend:

```bash
export FVK_PI05_RTX_FORCE_INT8=1
export FLASHRT_PI05_STRICT_VISUAL=1
```

`FVK_PI05_RTX_FORCE_INT8=1` selects the required dynamic INT8 encoder/decoder
path; `FLASHRT_PI05_STRICT_VISUAL=1` enables the visual fusion. Both the fusion
build option and its runtime option default OFF. This changes no public
`load_model`/`predict` signature. The tested mode is BF16 vision, dynamic INT8
encoder and decoder, exact prompt lengths, no visual pooling. Fixed-shape,
FP8, visual INT8 and static encoder modes fail explicitly when selected with
the new option; they do not silently inherit this behavior.

The model pipeline calls normal `flash_rt_kernels` bindings directly.
There is no pointer-interception proxy, external shared-library loader,
temporary tensor address, or benchmark dependency in the hot path.

## Numerical contract and ownership

- QKV reuses `csrc/kernels/bias_epilogue_bf16.cu`; the main extension owns
  its source only with the new build gate. Qwen3-VL's separate extension
  keeps its own compilation and binding. Both can coexist.
- Patch reuses `bias_residual_layer_norm_bf16`, preserving the original
  BF16 residual write before LayerNorm reductions.
- `pi05_projector_bias_rms_norm_int8` extends the algorithm from
  `norm.cu::rms_norm_int8_rowwise_kernel`. Only the visual prefix rows get
  bias; their sum is rounded to BF16 and stored back for subsequent residual
  reads. Language rows are untouched. RMSNorm, FP32 amax/scale and INT8
  rounding retain the original operation order.
- The projector launcher requires positive rows, even cols <=8192,
  0<=image_rows<=rows and positive eps. Input x is mutable; outputs and
  scales must have separate, correctly sized device storage.
- All newly exposed optional functions have explicit not-built errors.
  Their source inclusion and definitions use the same build gate.
- Device scale and residual buffers are existing persistent pipeline buffers;
  no new calibration artifact or cache representation is introduced.

## Validation and historical measurements

Single-kernel tests check residuals, INT8 values and FP32 scales exactly,
including language-only rows and changed-input graph replay. Full model
validation should use the same checkpoint, inputs, noise and all other
algorithms, asserting both raw and decoded actions match.

On the prior pinned Orin producer, 16 observation/noise cases were bit-equal;
three paired comparisons totaled 1536 timed executions. Fusion alone saved a
paired median 1.583 ms (1.113%) in graph replay and 1.649 ms (1.118%) in full
offline requests. Each forward removes 29 target launches (patch 1, QKV 27,
projector 1). These are prior experiment results; fresh-main build and
validation results belong in the PR description. Individual-site latency and
multi-day stability have not been established. This does not claim parity
with official BF16 or closed-loop robot quality.

The repository-only model smoke compares both paths in one instance, with the
same weight/buffer addresses and BF16 algorithms:

```bash
python benchmarks/pi05_sm87_validate.py --checkpoint /path/to/checkpoint \
  --compare-strict
```

It uses two deterministic synthetic camera inputs and fixed noises, tests
changed input after capture, and compares raw and public actions exactly.
It does not drive a robot or download data. Its small timing sample is not
a replacement for repeated real-input performance measurements.

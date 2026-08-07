# Hy-Embodied-0.5-VLA on Jetson Orin SM87

> Orin SM87 adaptation of the existing HyVLA Thor path. SM87 has no native
> FP8/FP4 tensor cores, so Thor FP8/NVFP4 kernels are not compiled or used.
> The Orin frontend keeps the validated HyVLA IO/prefix/graph path and maps the
> lower-precision GEMM slots to SM80-family INT8 W8A8 rowwise CUTLASS kernels.

## Platform

| Field | Value |
|---|---|
| Device | Jetson AGX Orin / SM87 |
| GPU family | Ampere, SM87 |
| Native FP8 / FP4 | No |
| Build target | `-DGPU_ARCH=87` |
| Attention | PyTorch SDPA efficient backend baseline |
| Low-bit GEMM | SM80 INT8 rowwise W8A8 (`ENABLE_SM80_INT8_CUTLASS`) |

## Dispatch

`flash_rt/hardware/__init__.py` registers:

```python
("hyvla", "torch", "rtx_sm87") -> (
    "flash_rt.frontends.torch.hyvla_orin",
    "HyVLATorchFrontendOrin",
)
```

The key is also included in `_SM87_ALLOWED`, so
`flash_rt.load_model(ckpt, config="hyvla", framework="torch")` resolves on
Orin.

## Files

| File | Purpose |
|---|---|
| `flash_rt/frontends/torch/hyvla_orin.py` | Orin frontend; inherits Thor tokenizer/preprocess/prefix/graph orchestration; disables Thor-only FP8/FP4 fused options; materializes INT8 per-row weights. |
| `flash_rt/models/hyvla/pipeline_orin.py` | Orin pipeline subclass; inherits BF16 math and overrides the lower-precision GEMM slot with INT8 W8A8 rowwise CUTLASS; fused ViT forward (pending-add+LN, efficient SDPA). |
| `csrc/kernels/hyvla_vit_fuse.cu` | Fused ViT residual-add + LayerNorm kernel (SM87 + SM110 builds). |
| `tests/test_orin_hyvla05_e2e_check.py` | BF16 baseline vs default-INT8 fixed-noise action cosine gate (>= 0.999) plus state/noise/prompt input boundaries. |
| `tests/test_orin_hyvla05_graphsafe.py` | Graph-vs-eager, replay-stability, and fused-vs-unfused (attention-prep, ViT add+LN) gates. |
| `tests/test_orin_hyvla05_arch_gate.py` | SM87 fail-fast hardware gate and FP4 rejection (mocked CUDA, no device needed). |

## Precision policy

| Component | Orin v2 precision (current default) |
|---|---|
| Embeddings / token assembly | BF16 |
| HYViT2 + merger | BF16 graph (INT8 opt-in, fails gate — see dead ends) |
| MoT VLM prefill QKV/O | BF16 (outputs feed the KV cache read by 10 denoise steps) |
| MoT VLM prefill FFN (gate/up/down) | **INT8 W8A8 by default** (`use_int8_vlm_ffn`) |
| Expert denoise QKV/O/FFN GEMMs | **INT8 W8A8 by default** (`use_int8_exp`) |
| RMSNorm / RoPE / QK-Norm | BF16 |
| Attention | BF16/SDPA |
| State/time/action head | BF16/FP32 update as in Thor path |
| FP4 / Thor FP8 fused kernels | Unsupported on SM87 |

`HyVLATorchFrontendOrin(..., use_fp8=True)` maps to the precision-safe Orin
INT8 tier: expert denoise INT8 **plus prefill FFN INT8** (gate/up/down; QKV/O
stay BF16 to protect the KV cache). Pass `use_fp8=False, use_int8=False` for
the BF16 baseline. Diagnostic flags: `use_int8_vlm=True` (all-tower prefill
INT8) and `use_int8_vit=True` (ViT INT8) both fail the 0.999 gate — see dead
ends.

## Measured gates

Measured on the downloaded checkpoint at
`/path/to/checkpoint/Hy-Embodied-0.5-VLA-RoboTwin`, fixed
noise, three random 6-frame cameras, prompt `pick up the bottle`.

| Config | Reference action cosine | MAE | Graph safety |
|---|---:|---:|---|
| BF16 baseline | 0.999911 | 2.45e-3 | PASS (`graph==eager`, replay max diff 0) |
| **BF16 + ViT fusion (Stage 4)** | **0.999931** | 1.75e-3 | PASS |
| INT8 default + Stage 3 fused kernels | 0.999531 | 6.46e-3 | PASS (`graph==eager`, replay max diff 0) |
| **INT8 + ViT fusion (Stage 4, current default)** | **0.999442** | 6.69e-3 | PASS (`graph==eager`, replay max diff 0) |
| INT8 default (expert + prefill FFN) | 0.999580 | 6.12e-3 | PASS |
| INT8 expert-only (Stage 1) | 0.999681 | 5.56e-3 | PASS |
| all-tower INT8 (`use_int8_vlm=True`) | 0.998843 | 8.77e-3 | graph-safe but below gate |
| ViT INT8 (`use_int8_vit=True`) | 0.997459 | 1.38e-2 | FAIL — dead end |

Stage latency, sequential median of 20 iterations:

| Config | E2E `predict_actions` | ViT+merger graph | Prefix assembly* | Prefill+denoise graph |
|---|---:|---:|---:|---:|
| BF16 baseline | 403.6 ms | 170.7 ms | 7.6 ms | 205.2 ms |
| INT8 Stage 1 (expert-only) | 384.4 ms | 173.9 ms | 7.2 ms | 187.5 ms |
| INT8 Stage 2 (expert + prefill FFN + prefix cache) | 353.6 ms | 171.4 ms | — | 167.3 ms |
| INT8 Stage 3 (Stage 2 + fused norm/rope kernels) | 293.2 ms | 170.9 ms | 7.1 ms | 107.0 ms |
| **INT8 Stage 4 (Stage 3 + ViT fusion, default)** | **277.5 ms** | 156.9 ms | 7.3 ms | 106.9 ms |
| BF16 Stage 4 (ViT fusion) | 315.2 ms | 156.5 ms | 7.8 ms | 143.8 ms |
| BF16 Stage 2 (prefix cache only) | 391.9 ms | 172.6 ms | — | 204.9 ms |

\* stageprof's assembly line re-runs the uncached reproduction for
measurement; inside `predict_actions` the Orin frontend caches the
prompt/camera-static artifacts (segment mask, permutation, bf16-rounded RoPE
tables, suffix mask) per `(prompt, num_cam)`, which removed ~12 ms/call from
E2E in both BF16 and INT8 tiers (403.6→391.9 and 365.7→353.6).

Stage 2 levers: prefill FFN INT8 took the main graph 187.5→167.3 ms; the
static-prefix cache removed the per-call mask/RoPE recomputation. Both
preserve the 0.999 action-cosine gate.

Stage 3 levers (main graph 167.3→107.0 ms, E2E 353.6→293.2 ms):

1. **Kernel-backed RMSNorm shim** — torch 2.3 has no `F.rms_norm`; the eager
   Python fallback was a hot elementwise chain. Dispatching the existing
   `fvk.rms_norm` (bit-equal math: fp32 sum-of-squares, single bf16 round)
   removed ~38 ms from the main graph.
2. **Fused residual-add + RMSNorm** (`fvk.residual_add_rms_norm`) in the
   expert denoise loop: one kernel replaces add + norm, and the fp32
   accumulation before rounding keeps drift at ≤1 ULP vs torch.
3. **Fused RoPE + QK-Norm + KV-write megakernel** — Thor's
   `hyvla_rope_qknorm_kvwrite_bf16` is plain CUDA and was compiled for SM87
   (CMake gate `FLASHRT_HAVE_HYVLA_ORIN`; `csrc/bindings.cpp` needed the
   HyVLA def block moved out of the `ENABLE_NVFP4` guard, which silently
   excluded it on SM87). Collapses ~11 launches per attention block × 352
   blocks/step into one kernel. Auto-enabled in the Orin frontend when the
   symbol is present (`use_fused=True`).

All three preserve graph safety (max diff 0) and the 0.999 gate (cos drops
0.999580→0.999531 from the fused RoPE rounding order, still comfortably
above gate).

Stage 4 levers (ViT 170.9→156.9 ms, E2E 293.2→277.5 ms):

1. **Memory-efficient SDPA for ViT spatial attention** — the q/k/v slices
   from the packed QKV GEMM are strided; the flash backend force-copies all
   three (plus the output) contiguous, ~25 ms of `copy_` per ViT pass. The
   memory-efficient backend reads the strides natively (2.89→0.62 ms per
   attention segment in µbench). Same finding as Thor's efficient-SDPA lever:
   accuracy *improves* vs flash (BF16 E2E cos 0.999911→0.999931).
2. **Fused residual-add + LayerNorm** — new kernel
   `hyvla_vit_add_layer_norm_bf16` (`csrc/kernels/hyvla_vit_fuse.cu`):
   in-place bf16-rounded residual add (bit-equal to torch add) + LayerNorm
   matching this repo's `layer_norm_kernel`. The Orin `vit_forward` override
   carries each block's MLP output as `pending` and fuses it into the next
   block's entry LN; spacetime blocks keep the torch path (their entry LN
   also adds the time positional embedding). LN deviates from torch's Welford
   reduction by ≤1 bf16 ULP on ~2e-5 of elements — unbiased rounding noise,
   no gate impact.

Note: the "drop history frames after the final spacetime block" lever from
Thor is already inherited via the base pipeline (blocks 24-26 run on 3
frames instead of 18) — the Stage 3 ViT estimate of 171 ms already included
it.

## Dead ends (Stage 2, measured)

| Scheme | Result | Mechanism |
|---|---|---|
| **ViT INT8** (`use_int8_vit`) | FAIL: E2E cos 0.997459; only −7 ms (166.7 vs 173.9 ms) | ViT merged-output cos only 0.999006 vs BF16 (MAE 3.4e-2); INT8 GEMM saves ~52 ms but per-GEMM activation quant (+14 ms) and extra elementwise (+28 ms) eat most of it; ViT errors propagate through 32-layer prefill + 10 denoise steps. MLP-only variant worse (merged cos 0.998638) |
| **All-tower prefill INT8** (`use_int8_vlm`) | cos 0.998843 < 0.999 | QKV INT8 error lands in the KV cache and compounds over 10 denoise steps; FFN-only variant avoids the KV path and passes |
| ViT INT8 t64x128 tile | slower than 128×128 at ViT shapes | µbench: (588,3456,1152) 0.556 vs 0.436 ms |
| **ViT elementwise fusion via existing kernels** | rejected pre-integration | `bias_gelu_bf16_strict` uses tanh-approx gelu (the reference uses exact erf; bit-mismatch max 0.0156); `bias_residual_layer_norm_bf16` LN drifts 1-2 ULP vs `F.layer_norm` on ViT shapes — both would erode the 0.999 gate |

**Reference eager anchor (measured on this Orin, same fixed inputs, warmup 3 +
median of 5): 3137.3 ms.** FlashRT speedups vs the reference eager path:

| Config | E2E | Speedup vs reference eager |
|---|---:|---:|
| **INT8 + ViT fusion (Stage 4, default)** | **277.5 ms** | **11.3×** |
| INT8 Stage 3 (fused norm/rope kernels) | 293.2 ms | 10.7× |
| INT8 Stage 2 (expert + prefill FFN + prefix cache) | 353.6 ms | 8.87× |
| INT8 Stage 2 before prefix cache | 365.7 ms | 8.58× |
| INT8 Stage 1 (expert-only) | 384.4 ms | 8.16× |
| BF16 Stage 2 | 391.9 ms | 8.00× |
| BF16 baseline | 403.6 ms | 7.77× |

Cross-platform comparison (Thor numbers from `hyvla05_thor_sm110.md`):

| Platform | Reference eager | Native BF16 (graph) | Production | Speedup |
|---|---:|---:|---:|---:|
| Thor SM110 | ~930 ms | 248.6 ms | 158.3 ms (FP8+fused+autotune) | 5.9× |
| Orin SM87 | 3137 ms | 315.2 ms | 277.5 ms (INT8 + fused norm/rope/ViT kernels) | 11.3× |

The reference eager path is much slower on Orin than Thor (3.1 s vs 0.93 s) because the
eager PyTorch path is launch/dispatch-bound on 16 SMs and falls back to the
slow SDPA math backend; the CUDA-Graph capture collapses most of that. The
Orin absolute latency (~278 ms) is ~1.8× Thor's because SM87 lacks FP8 and
has lower bandwidth (~204 vs 243 GB/s) and compute; the remaining gap is
dominated by the ViT stage (~157 ms, 57% of E2E), whose ~100 ms of BF16
GEMMs run near the cuBLAS compute ceiling on both platforms.

## Roofline (Stage 4, measured anchors)

Hardware anchors measured on this unit (warm, unlocked clocks):

| Anchor | Value |
|---|---:|
| >>L2 read bandwidth (1 GB buffer) | **97 GB/s** |
| bf16 cuBLAS peak (4096³, warm) | 29.9 TFLOPS |
| bf16 at ViT qkv shape (3528,1152,3456) | 9.5 TFLOPS |
| bf16 at FFN shapes (3528,·,4304/1152) | 21–24 TFLOPS |
| INT8 CUTLASS peak (4096³) | 38.7 TOPS |
| INT8 at prefill FFN (240,2048,12288) | 18.9 TOPS |
| INT8 at denoise M=41 shapes | 6–8 TOPS, weight-streaming at ~71–95 GB/s |

Stage floors vs measured (median of 20):

| Stage | Dominant regime | Floor | Measured | Headroom |
|---|---|---:|---:|---:|
| ViT+merger (156.9 ms) | GEMM compute: ~106 ms cuBLAS bf16 (profiler) | ~150 ms | 156.9 ms | ~5% |
| Prefill graph | FFN INT8 ≈47 ms (18.9 TOPS) + QKV/O bf16 + attn + quant | ~65 ms | — | — |
| Denoise graph (×10 steps) | INT8 weight streaming: 369 MB/step ≈ 97 GB/s → ~38 ms floor + quant/attn | ~70 ms | prefill+denoise combined: 106.9 ms | — |
| Main graph total | | ~135 ms | 106.9 ms graph replay (faster than eager-shape µbench sum) | ~0–20% |
| E2E | | ~230 ms | 277.5 ms | ~17% |

Reading: the ViT stage sits within ~5% of its measured cuBLAS+bandwidth
floor — its qkv GEMM (9.5 TFLOPS, N=3456 K=1152) is the single slowest
shape and alone costs ~71 ms of the 106 ms GEMM budget. The main graph's
denoise portion is weight-streaming-bound at M=41 (INT8 weights read at
71–95 GB/s, i.e. already at the measured memory ceiling); the prefill
portion is compute-bound in the FFN INT8 GEMMs. The ~17% E2E headroom
splits between prefix assembly (7.3 ms eager), graph-replay gaps, and
elementwise tails (GELU ~9 ms, spacetime mixes ~14 ms) that are already
near-bandwidth. Conclusion: no >1 ms framework-side lever remains; the
quantifiable floor for this model shape on SM87 is ~230–250 ms.

## Assets

Official model repository: see the Hy-Embodied-0.5-VLA project page on
Hugging Face.

Checkpoint target:

```bash
/path/to/checkpoint/Hy-Embodied-0.5-VLA-RoboTwin
```

Direct Hugging Face access may be reset; use `HF_ENDPOINT=https://hf-mirror.com`
when needed.

## Build

```bash
cmake -B build_orin_sm87 -S . \
  -DGPU_ARCH=87 \
  -DFLASHRT_ENABLE_HYVLA=ON \
  -DFA2_ARCH_NATIVE_ONLY=ON \
  -DFA2_HDIMS='128;256' \
  -DFA2_DTYPES='bf16'
cmake --build build_orin_sm87 -j4
```

For SM87, CMake enables `ENABLE_SM80_INT8_CUTLASS` and builds the existing
rowwise INT8 CUTLASS kernels into `flash_rt_kernels`.

## Verification

Checkpoint-gated precision and graph-safety gates (skipped automatically when
`FLASHRT_HYVLA_CHECKPOINT` is unset):

```bash
FLASHRT_HYVLA_CHECKPOINT=/path/to/Hy-Embodied-0.5-VLA-RoboTwin \
  PYTHONPATH=. python -m pytest \
  tests/test_orin_hyvla05_e2e_check.py \
  tests/test_orin_hyvla05_graphsafe.py -v
```

Hardware fail-fast and dispatch gates run without a device or checkpoint:

```bash
PYTHONPATH=. python -m pytest \
  tests/test_orin_hyvla05_arch_gate.py \
  tests/test_orin_hyvla05_dispatch.py -v
```

## Current caveats

- The fused RoPE/QKNorm/KV-write megakernel (plain CUDA) is compiled for SM87
  and enabled by default (`use_fused=True`); Thor-only fused FP8 quant, FFN
  megakernels, and NVFP4 remain deliberately disabled.
- INT8 uses dynamic per-row activation scales and BF16 GEMM outputs; static
  calibration and FP16-output epilogues are possible next steps if profiling
  shows the quant kernels on the critical path.
- `predict_actions` caches prompt/camera-static prefix artifacts; the first
  call per prompt pays the full mask/RoPE build (~7 ms), later calls reuse it.
- Remaining ceiling: ViT (~157 ms, 57% of E2E) is the largest stage — ~100 ms
  of BF16 GEMMs near the cuBLAS compute ceiling plus near-bandwidth-bound
  elementwise (GELU, spacetime ops). INT8 there fails the precision gate, so
  further ViT cuts need model-side changes (fewer history frames, smaller
  ViT) rather than framework-side work.
- The reference E2E oracle requires the official HyVLA repo and checkpoint assets.

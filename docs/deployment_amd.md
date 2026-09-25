# FlashRT on AMD GPUs

This is the model-independent guide to the AMD backend: supported
hardware, how to build the extension, how the tree is laid out, how
hardware routing works, and how to run the tests.

Per-model deployment guides:

| Model | Guide | Status |
|---|---|---|
| Pi0.5 | [deployment_amd_pi05.md](deployment_amd_pi05.md) | CDNA4 FP8/BF16; RDNA 3.5 BF16 |
| GROOT N1.7 | [deployment_amd_groot_n17.md](deployment_amd_groot_n17.md) | FP8 backbone + bf16 DiT |

## Supported hardware

The native HIP extension has architecture-selected source sets for gfx942
(CDNA3 / MI300), gfx950 (CDNA4 / MI350), and gfx1151 (RDNA 3.5 / Radeon
8060S). RDNA implementations remain in the existing `csrc/amd` layout and
use an `_rdna` filename suffix; they do not compile or reuse the CDNA
wave64/MFMA kernels.

The extension is built for exactly one architecture. CDNA3 uses E4M3 FNUZ
(maximum finite value 240), CDNA4 uses OCP E4M3 (maximum finite value 448),
and the RDNA backend is BF16-only. Using a mismatched source set can produce
wrong results rather than a slow fallback, so the restriction is enforced at
build time and frontend initialization:

- **At build time** — `scripts/amd/build_amd.sh` accepts the exact gfx942,
  gfx950, and gfx1151 source sets and rejects other `GPU_ARCH` values. Set
  `FLASHRT_AMD_ALLOW_ARCH=1` to override when bringing up a port to a
  future architecture.
- **At CDNA frontend init** — the frontend reads the extension's
  `device_arch()` (the running device's `gcnArchName`, e.g.
  `gfx950:sramecc+:xnack-`) and `build_info()["gpu_arch"]` (the
  compile-time target) and raises `RuntimeError` unless both exactly match.
  This fires before the checkpoint is touched, so forcing
  an explicit AMD hardware target on the other generation fails immediately instead
  of computing garbage.
- **At RDNA frontend init** — the Pi0.5 frontend requires an extension whose
  `build_info()` reports the RDNA source set, wave32, and gfx1151, and also
  requires the visible device to be gfx1151. A CDNA extension cannot be
  loaded through the RDNA route, or vice versa.

## Build

The AMD sources live in a standalone tree with their own build entry
point; the root (CUDA) CMake project is not involved and does not need
CUDA or CUTLASS present.

```bash
bash scripts/amd/build_amd.sh gfx950
# or, on MI300X:
bash scripts/amd/build_amd.sh gfx942
# or, on RDNA 3.5:
PYTHON=.venv/bin/python bash scripts/amd/build_amd.sh gfx1151
```

Output: `flash_rt/amd/flash_rt_amd_kernels*.so`. CMake selects the source
set for the target generation.

Dependencies:

- **hipBLASLt** — ships with ROCm and is required by both AMD source sets.
  RDNA 3.5 uses an instance-local BF16 algorithm cache populated by runtime
  measurements before HIP Graph capture.
- **No vendored third-party code.** There is no CUTLASS/CK checkout to
  manage; everything else is HIP C++ plus compiler MFMA intrinsics on CDNA4
  or WMMA builtins on RDNA 3.5.
- **aiter** (strongly recommended for CDNA4) — AMD's assembly flash-attention
  library. When importable it serves the attention sites; otherwise the
  backend falls back to torch SDPA. Attention is the largest kernel
  bucket on this backend, so the fallback is expensive: measured on the
  Pi0.5 quickstart, one variable changed, same process and node —

  | Attention path | Median |
  |---|---|
  | aiter (default when importable) | 16.3 ms |
  | torch SDPA (aiter absent, or `FVK_AMD_ATTN=sdpa`) | 22.2 ms |

  If a deployment measures roughly 6 ms above the published numbers,
  check that aiter is importable in the serving environment first.

## Layout

```
csrc/amd/                    standalone HIP tree (own CMake entry point)
  bindings.cpp               pybind module flash_rt_amd_kernels
  bindings_rdna.cpp          gfx1151-only pybind surface
  gemm/                      hipBLASLt + CDNA MFMA / RDNA WMMA kernels
  arch.h                     FP8 format, datatype, and capability selection
  attention/                 CDNA and `_rdna` encoder/decoder attention
  kernels/                   norm / activation / quantize / fusion families;
                              `_rdna` files are selected only for gfx1151
flash_rt/amd/
  core/hip_buffer.py         ctypes device memory over libamdhip64
  core/hip_graph.py          ctypes HIP graph capture / instantiate / replay
  hardware/cdna3|cdna4/      generation-specific attention backends
  hardware/rdna35/           BF16 attention and GEMM providers
  models/<model>/pipeline.py pointer-only forward passes
  frontends/torch/<model>.py weight load, calibration, capture, infer
  models/pi05_rdna35/        BF16 RDNA Pi0.5 pipeline
  frontends/torch/pi05_rdna35.py
```

The pybind entry points keep the same `uintptr_t` pointer + stream ABI as the
CUDA module. RDNA-only entries carry an `_rdna` suffix so they cannot be
silently routed on CDNA. CDNA model frontends warm up and replay a captured
HIP graph on their production path. The RDNA Pi0.5 frontend defaults to eager
execution and `cache_frames=1`; explicit values greater than one enable
decoder-only frames that reuse the last encoded K/V prefix.

## Hardware routing

Auto-detection uses the device's `gcnArchName`: gfx942 maps to `amd_cdna3`,
gfx950 maps to `amd_cdna4`, and gfx1151 maps to `amd_rdna35`. An unregistered
ROCm architecture raises rather than falling through to an NVIDIA table entry.

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch")            # auto-detected
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch",
                            hardware="amd_cdna3")         # explicit MI300X
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch",
                            hardware="amd_cdna4")         # explicit
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch",
                            hardware="amd_rdna35",
                            use_fp8=False)                 # gfx1151 BF16
```

Expected failures:

| Situation | Error |
|---|---|
| Extension not built | `ImportError` naming `flash_rt_amd_kernels` and the build command |
| CDNA build/device mismatch | `RuntimeError` naming the device and build architectures |
| RDNA build/backend/wave size/device mismatch | `RuntimeError` naming the required gfx1151 RDNA source set |
| Model/framework not ported to AMD | `RuntimeError` from pipeline resolution |
| Thor-only options (`use_fp4_decoder`, `use_fa4`) | `ValueError` naming the supported hardware |

## Build environment knob

Model and architecture-specific runtime knobs are documented in the
per-model guides. The only shared build escape hatch is:

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_ATTN` | `aiter` | attention backend: `aiter` or `sdpa` (torch fallback) |
| `FLASHRT_FP8_NT_AUTOTUNE` | `auto` | timed hipBLASLt algorithm selection at setup; `off` uses heuristic top-1 |
| `FLASHRT_FP8_ALGO_POOL` | `16` | candidate pool depth for the timed selection. Deeper pools (64/128) sometimes find faster algorithms but widen run-to-run pick variance, and a single mis-timed trial can ship a slow algorithm |
| `FLASHRT_AMD_ALLOW_ARCH` | `0` | build-script escape hatch for an unregistered AMD ISA port |

`FVK_AMD_ATTN`, `FLASHRT_FP8_NT_AUTOTUNE`, and `FLASHRT_FP8_ALGO_POOL`
belong to the CDNA runtime. The RDNA Pi0.5 backend has its own
`FLASHRT_RDNA35_*` controls and never reads those CDNA FP8/aiter settings.

## Tests

```bash
python -m pytest tests/test_amd_routing.py tests/test_amd_extension.py \
                 tests/test_amd_hip_graph.py tests/test_amd_kernel_parity.py \
                 tests/test_amd_pi05_model.py \
                 tests/test_amd_rdna35_routing.py \
                 tests/test_amd_rdna35_ops.py \
                 tests/test_amd_rdna35_model.py -v
```

| File | Covers | Skips when |
|---|---|---|
| `test_amd_routing.py` | pipeline-map entries, `detect_arch()` ROCm branches (including unregistered-target rejection), `load_model` failure modes | mostly runs anywhere; extension-dependent cases skip without the `.so` |
| `test_amd_extension.py` | `build_info()` / `device_arch()` coherence, required-symbol inventory of every bound kernel | extension not importable |
| `test_amd_hip_graph.py` | buffer round-trips with pattern data, capture → instantiate → replay, byte-identical repeat replays | no ROCm device or extension |
| `test_amd_kernel_parity.py` | numerical parity of the kernel surface against torch references on real-distribution inputs, including FP8 byte-exactness and the seqused fixed-shape attention path | no ROCm device or extension |
| `test_amd_pi05_model.py` | end-to-end model load, graph capture, pinned-noise determinism, exact/fixed prompt modes, FP8 and BF16 | no checkpoint (see the per-model guide for the environment variables) |
| `test_amd_rdna35_routing.py` | gfx1151 routing, lazy imports, BF16/profile contracts, and CDNA/Triton isolation | torch unavailable; hardware-only cases require gfx1151 |
| `test_amd_rdna35_ops.py` | RDNA native-kernel parity, validation, graph replay, and library fallbacks | no gfx1151 ROCm device or RDNA extension |
| `test_amd_rdna35_model.py` | RDNA Pi0.5 finite output, determinism, full-graph parity, and optimized/fallback parity | no gfx1151 checkpoint |
| `test_amd_groot_routing.py` | GROOT N1.7 pipeline-map entry, precision-tier contracts, attention-backend site/layer validation | extension-dependent cases skip without the `.so` |
| `test_amd_groot_model.py` | GROOT N1.7 end-to-end: kernel backbone, finite actions, pinned-noise determinism, optional reference cosine | no checkpoint |

Everything skips cleanly with a stated reason on machines without ROCm,
so the suite is safe to run in a CUDA-only CI.

## Measuring

- Report **medians** after warmup. Report a minimum only as a lower
  bound, never as the headline.
- On CDNA4, timed hipBLASLt algorithm selection moves cross-run medians by roughly
  ±0.2–0.35 ms. Compare arms **inside one process** where possible.
- When judging output cosine against a saved reference, **pin the
  denoise noise** to the exact array the reference was generated with.
  A fresh random draw shifts cosine by about 1e-3, which is the same
  magnitude as a real numerical regression and will mask it.
- Some ROCm profiler reports fold dispatch gaps into the reported
  kernel time. Use them for ranking buckets; take absolute per-call
  numbers from isolated in-graph chains.

## HIP versus CUDA notes

- CDNA4 uses wave64 reductions from `common_hip.h`; RDNA 3.5 uses wave32
  reductions from `common_hip_rdna.h`. HIP has no CUDA-style `*_sync`
  shuffle variants.
- The three-argument graph instantiate is `hipGraphInstantiateWithFlags`.
- Memcpy-kind and capture-mode enum values match CUDA numerically
  (validated on hardware by the runtime-seam smoke test).
- FP8 storage comes from `csrc/amd/arch.h`: E4M3 FNUZ with
  `HIP_R_8F_E4M3_FNUZ` on CDNA3, and OCP E4M3 with `HIP_R_8F_E4M3` on
  CDNA4. Weights and activation scales must be regenerated for the target.
- hipBLASLt matmul is column-major; the GEMM runner uses the
  operand-swap form (`D_col = B_col @ A_col`), the same trick the CUDA
  FP8 paths use.
- gfx950 exposes `V_MFMA_F32_16X16X32_FP8_FP8` and `..._BF16` with
  per-lane contiguous 8-byte fragments; the hand-written GEMMs repack
  weights at setup into per-lane consumption order so each workgroup
  streams its weight tile linearly.
- On gfx942 the packed FNUZ FP8 and packed BF16 Pi0.5 decoder paths are
  available. MXFP4 remains unavailable; unsupported forms route through
  hipBLASLt. Encoder attention uses AITER.
- Every HIP runtime call in the Python layer is return-code checked; a
  failed launch, copy or synchronise raises instead of letting a stale
  buffer be read back as a result.

# FlashRT on AMD Instinct (MI350X / CDNA4)

This is the model-independent guide to the AMD backend: supported
hardware, how to build the extension, how the tree is laid out, how
hardware routing works, and how to run the tests.

Per-model deployment guides:

| Model | Guide | Status |
|---|---|---|
| Pi0.5 | [deployment_amd_pi05.md](deployment_amd_pi05.md) | FP8 default, BF16 available |
| GROOT N1.7 | [deployment_amd_groot_n17.md](deployment_amd_groot_n17.md) | FP8 backbone + bf16 DiT |

## Supported hardware

**gfx950 only** — CDNA4, i.e. the Instinct MI350 series, with ROCm 7.x
and a ROCm build of PyTorch. The kernels use gfx950-specific MFMA shapes
and FP8 (OCP `e4m3`) paths that produce wrong results, not a slow
fallback, on other AMD architectures. The restriction is therefore
enforced twice:

- **At build time** — `scripts/amd/build_amd.sh` rejects a `GPU_ARCH`
  argument that does not start with `gfx950`. Set
  `FLASHRT_AMD_ALLOW_ARCH=1` to override when bringing up a port to a
  future architecture.
- **At frontend init** — the frontend reads the extension's
  `device_arch()` (the running device's `gcnArchName`, e.g.
  `gfx950:sramecc+:xnack-`) and `build_info()["gpu_arch"]` (the
  compile-time target) and raises `RuntimeError` unless both are gfx950.
  This fires before the checkpoint is touched, so forcing
  `hardware="amd_cdna4"` on another AMD card fails immediately instead
  of computing garbage.

## Build

The AMD sources live in a standalone tree with their own build entry
point; the root (CUDA) CMake project is not involved and does not need
CUDA or CUTLASS present.

```bash
bash scripts/amd/build_amd.sh gfx950
```

Output: `flash_rt/amd/flash_rt_amd_kernels*.so`. The script prefers
CMake and falls back to a direct `hipcc` one-shot compile when no usable
CMake is available.

Dependencies:

- **hipBLASLt** — ships with ROCm; the GEMM engine and the baseline that
  hand-written kernels must beat before they are routed.
- **No vendored third-party code.** There is no CUTLASS/CK checkout to
  manage; everything else is HIP C++ plus the `__builtin_amdgcn_mfma_*`
  intrinsics from the compiler.
- **aiter** (strongly recommended) — AMD's assembly flash-attention
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
  gemm/                      hipBLASLt runner + hand-written MFMA GEMMs
  attention/                 hand-written CDNA4 attention kernels
  kernels/                   norm / activation / quantize / fusion families
flash_rt/amd/
  core/hip_buffer.py         ctypes device memory over libamdhip64
  core/hip_graph.py          ctypes HIP graph capture / instantiate / replay
  hardware/cdna4/            attention backends (aiter, SDPA fallback)
  models/<model>/pipeline.py pointer-only forward passes
  frontends/torch/<model>.py weight load, calibration, capture, infer
```

The pybind entry points keep the same names and the same
`uintptr_t` pointer + stream ABI as the CUDA module, so pipeline code is
portable text between the two backends. The execution contract is also
the same: warm up, capture a HIP graph once, then replay it — no Python
or framework operations on the inference path.

## Hardware routing

Auto-detection: when `torch.version.hip` is set and the device's
`gcnArchName` is gfx950, `detect_arch()` returns `"amd_cdna4"`. Another
ROCm architecture raises rather than falling through to an NVIDIA table
entry.

```python
import flash_rt

model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch")            # auto-detected
model = flash_rt.load_model(checkpoint_dir, config="pi05",
                            framework="torch",
                            hardware="amd_cdna4")         # explicit
```

Expected failures:

| Situation | Error |
|---|---|
| Extension not built | `ImportError` naming `flash_rt_amd_kernels` and the build command |
| Non-gfx950 AMD device | `RuntimeError` naming the device and build architectures |
| Model/framework not ported to AMD | `RuntimeError` from pipeline resolution |
| Thor-only options (`use_fp4_decoder`, `use_fa4`) | `ValueError` naming the supported hardware |

## Shared environment knobs

These apply to every model on this backend. Model-specific knobs are
documented in the per-model guides.

| Env | Default | Meaning |
|---|---|---|
| `FVK_AMD_ATTN` | `aiter` | attention backend: `aiter` or `sdpa` (torch fallback) |
| `FLASHRT_FP8_NT_AUTOTUNE` | `auto` | timed hipBLASLt algorithm selection at setup; `off` uses heuristic top-1 |
| `FLASHRT_FP8_ALGO_POOL` | `16` | candidate pool depth for the timed selection. Deeper pools (64/128) sometimes find faster algorithms but widen run-to-run pick variance, and a single mis-timed trial can ship a slow algorithm |
| `FLASHRT_AMD_ALLOW_ARCH` | `0` | build-script escape hatch for a non-gfx950 port |

## Tests

```bash
python -m pytest tests/test_amd_routing.py tests/test_amd_extension.py \
                 tests/test_amd_hip_graph.py tests/test_amd_kernel_parity.py \
                 tests/test_amd_pi05_model.py -v
```

| File | Covers | Skips when |
|---|---|---|
| `test_amd_routing.py` | pipeline-map entry, `detect_arch()` ROCm branch (including non-gfx950 rejection), `load_model` failure modes | mostly runs anywhere; extension-dependent cases skip without the `.so` |
| `test_amd_extension.py` | `build_info()` / `device_arch()` coherence, required-symbol inventory of every bound kernel | extension not importable |
| `test_amd_hip_graph.py` | buffer round-trips with pattern data, capture → instantiate → replay, byte-identical repeat replays | no ROCm device or extension |
| `test_amd_kernel_parity.py` | numerical parity of the kernel surface against torch references on real-distribution inputs, including FP8 byte-exactness and the seqused fixed-shape attention path | no ROCm device or extension |
| `test_amd_pi05_model.py` | end-to-end model load, graph capture, pinned-noise determinism, exact/fixed prompt modes, FP8 and BF16 | no checkpoint (see the per-model guide for the environment variables) |
| `test_amd_groot_routing.py` | GROOT N1.7 pipeline-map entry, precision-tier contracts, attention-backend site/layer validation | extension-dependent cases skip without the `.so` |
| `test_amd_groot_model.py` | GROOT N1.7 end-to-end: kernel backbone, finite actions, pinned-noise determinism, optional reference cosine | no checkpoint |

Everything skips cleanly with a stated reason on machines without ROCm,
so the suite is safe to run in a CUDA-only CI.

## Measuring

- Report **medians** after warmup. Report a minimum only as a lower
  bound, never as the headline.
- Timed hipBLASLt algorithm selection moves cross-run medians by roughly
  ±0.2–0.35 ms. Compare arms **inside one process** where possible.
- When judging output cosine against a saved reference, **pin the
  denoise noise** to the exact array the reference was generated with.
  A fresh random draw shifts cosine by about 1e-3, which is the same
  magnitude as a real numerical regression and will mask it.
- Profiler kernel durations on ROCm fold dispatch gaps into the reported
  kernel time. Use them for ranking buckets; take absolute per-call
  numbers from isolated in-graph chains.

## HIP versus CUDA notes

- Wavefront is 64: reductions use `__shfl_down` wave64 helpers
  (`csrc/amd/kernels/common_hip.h`); there are no `*_sync` shuffle
  variants.
- The three-argument graph instantiate is `hipGraphInstantiateWithFlags`.
- Memcpy-kind and capture-mode enum values match CUDA numerically
  (validated on hardware by the runtime-seam smoke test).
- FP8 storage is OCP `e4m3` (`__hip_fp8_e4m3`, `HIP_R_8F_E4M3`) — never
  the CDNA3 `fnuz` variant.
- hipBLASLt matmul is column-major; the GEMM runner uses the
  operand-swap form (`D_col = B_col @ A_col`), the same trick the CUDA
  FP8 paths use.
- gfx950 exposes `V_MFMA_F32_16X16X32_FP8_FP8` and `..._BF16` with
  per-lane contiguous 8-byte fragments; the hand-written GEMMs repack
  weights at setup into per-lane consumption order so each workgroup
  streams its weight tile linearly.
- Every HIP runtime call in the Python layer is return-code checked; a
  failed launch, copy or synchronise raises instead of letting a stale
  buffer be read back as a result.

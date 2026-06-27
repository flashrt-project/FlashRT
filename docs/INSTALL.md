# Installing FlashRT

One-page install guide. Picks up where the README leaves off and
covers the details the README keeps short to stay readable.

For the full build overview (what .so files are produced, which arch
enables which kernels), read the "Build" section of the top-level
[README](../README.md) first.

---

## 1. Two supported paths

| Path | When to use | Entry point |
|---|---|---|
| **Prebuilt Docker image** | Fastest path. Cloud (Modal / RunPod / Vast) or local. CUDA + kernels already compiled. | [README §Option A](../README.md) + [`docker/README.md`](../docker/README.md) |
| **Build Docker yourself** | Custom GPU arch / pinned commit / vetting the recipe | [README §Option B](../README.md) |
| **Native Linux** | Existing venv on a CUDA host, no Docker | [README §Option C](../README.md) + this doc below |
| **Native Jetson Thor** | SM110, ARM64, JetPack — Docker not recommended on Jetson | this doc below + [`docs/deployment_rtx4090.md`](deployment_rtx4090.md) for cross-ref |

Both paths end at the same verification step — `import flash_rt;
flash_rt.__version__` returns the installed version, and
`flash_rt.flash_rt_kernels` is importable.

---

## 2. Prerequisites (native path)

| Component | Minimum | Notes |
|---|---|---|
| GPU | SM80+ | A100 / RTX 30-series / 40-series / Thor / 5090 / DGX Spark. Pre-SM80 (V100, 20-series) is unsupported — FA2 vendored code requires Ampere. |
| NVIDIA driver | 525+ (CUDA 12.4) / 545+ (CUDA 13) | 5090 needs 550+ |
| CUDA Toolkit | 12.4+ on Thor/Ada/Hopper, 12.8+ on Blackwell | CUDA 13 is the NGC-image default |
| Python | 3.10 / 3.11 / 3.12 | One venv; the interpreter that runs `cmake` MUST match the interpreter that later imports `flash_rt` |
| GCC / G++ | 11+ (C++17) | |
| CMake | 3.24+ | |

## 3. Python environment

**Always use a fresh venv or conda env.** The build step resolves
`pybind11` via `python3 -m pybind11 --cmakedir`, and the `.so` files
ship with an ABI tag tied to the interpreter they were compiled
against. Mixing a system Python at build time with a conda Python at
import time is the #1 native-install failure mode.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## 4. CUTLASS dependency

FlashRT's main FP8/FP4 GEMM path is built against **CUTLASS 4.x**, not
bundled in the repo to keep clone size small. Clone it before
running `cmake`:

```bash
git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

CMake now fails with a clear message if this step is missing (see
`CMakeLists.txt` near the top of the "Paths" section).

> **Note**: FA2 uses a vendored CUTLASS 3.x under
> `csrc/attention/flash_attn_2_src/`. That one IS checked in — only
> the CUTLASS 4.x for the main kernels needs a manual clone.

## 5. Editable install is required

```bash
pip install -e ".[torch]"       # or "[jax]" / "[all]"
```

`-e` is not optional. The CMake build drops compiled `.so` files into
the `flash_rt/` source tree; only editable install makes that
directory importable without an extra copy step. A plain
`pip install .` would snapshot `flash_rt/` BEFORE the kernels are
built, and `import flash_rt` later would fail with a missing
`flash_rt_kernels` error.

## 6. Build

```bash
cmake -B build -S .                  # auto-detects GPU arch via nvidia-smi
# Or override: cmake -B build -S . -DGPU_ARCH=121   (121=Spark, 120=5090, 110=Thor, 89=4090, 80=A100)
cmake --build build -j$(nproc)       # equivalent to: ninja -C build, or make -C build
```

That's it — no separate `cp`, `make install`, or `ninja install`
step. CMake writes every `.so` directly into `flash_rt/` at build
time via `LIBRARY_OUTPUT_DIRECTORY`, so a single `cmake --build`
leaves the package importable. (The legacy `install(TARGETS …)`
rule is still present for wheel-packaging users who run
`cmake --install build`.)

Per-arch produced shared libraries:

| Target  | `flash_rt_kernels.so` | `flash_rt_fp4.so` | `flash_rt_fa2.so` | `libfmha_fp16_strided.so` |
|---------|:----------------------:|:------------------:|:------------------:|:-------------------------:|
| Thor (SM110) | ✅ | ✅ | — | ✅ (SigLIP fast path) |
| Hopper (SM100) | ✅ | ✅ | — | ✅ |
| DGX Spark / GB10 (SM121) | ✅ | ✅ | ✅ (in-SO FA2) | — |
| RTX 5090 (SM120) | ✅ | ✅ | ✅ (in-SO FA2) | — |
| RTX 4090 (SM89) | ✅ | — | ✅ (in-SO FA2) | — |

### 6.1 Building on CUDA < 12.8

The default vendor build of Flash-Attention 2 emits Blackwell PTX
fallbacks alongside the per-arch SASS so a single ``.so`` covers all
listed gencodes — including Blackwell SM120/SM121 targets that need
CUDA 12.8+. On older toolchains (e.g. an L40S running a CUDA-12.4
image) ``nvcc`` rejects the Blackwell PTX target with a
``Value 'compute_120' is not defined`` or
``Value 'compute_121' is not defined`` error and the build aborts.

If you only need a binary for the GPU detected on the build host
(typical for cloud / self-hosted users that aren't shipping the
``.so`` to a different arch), set ``FA2_ARCH_NATIVE_ONLY=ON`` to
skip the cross-arch SASS + PTX fallback. The build emits SASS for
the current arch only, runs ~66 % faster, and works on any CUDA
toolchain that supports that arch:

```bash
cmake -B build -S . -DFA2_ARCH_NATIVE_ONLY=ON
cmake --build build -j$(nproc)
```

### 6.2 Slim VLA build (optional)

The default build keeps FlashRT's broad compatibility surface. It compiles
shared kernels plus several model- or architecture-specific translation units
so existing model paths keep their historical bindings.

For deployment builds that only need the current VLA-oriented surface, you can
opt into a smaller compile surface:

```bash
cmake -B build -S . -DGPU_ARCH=<arch> -DFLASHRT_SLIM_BUILD=ON
cmake --build build -j$(nproc) --target flash_rt_kernels
```

`FLASHRT_SLIM_BUILD` is OFF by default and only changes what is compiled into
`flash_rt_kernels`. It does not change kernel math, launch parameters, dtype
selection, graph capture, runtime routing, or fallback policy.

In slim mode, the build drops kernel groups that the current VLA deployment
surface does not need:

- Motus VAE FP8 quantize kernels.
- Qwen3.6 / linear-attention kernels and their legacy Qwen3.6 binding names.
- SM120/NVFP4-named helper translation units on non-NVFP4 builds.

Neutral shared helpers stay compiled in both modes, including
`bf16_matmul_bf16` and `embedding_lookup_bf16`. Architecture-required kernels
also stay compiled when their architecture macro is enabled; for example,
SM120/NVFP4 builds retain NVFP4-required sources even with slim mode enabled.

Do not use `FLASHRT_SLIM_BUILD=ON` for compatibility builds or for model paths
that require the gated bindings, such as Qwen3.6 / Nex-N2, Motus FP8/VAE, or
non-VLA NVFP4 conversion flows. Those paths should use the default build until
they have their own documented build profile.

This option is a first step toward explicit build profiles. It is not yet a
general `vla` / `llm` / `vlm` / `tts` / `video` profile system; it is a
conservative opt-in compile-time reduction with tests covering the exported
binding surface.

## 7. Verify

```bash
python -c "
import flash_rt, torch, numpy
print('flash_rt:', flash_rt.__version__)
print('torch    :', torch.__version__, torch.cuda.get_device_capability())
print('numpy    :', numpy.__version__)
from flash_rt import flash_rt_kernels
print('kernels CUTLASS SM100:', flash_rt_kernels.has_cutlass_sm100())
"
```

Expected (Thor example):
```
flash_rt: 0.1.0
torch    : 2.9.0+cu124 (11, 0)
numpy    : 1.26.x
kernels CUTLASS SM100: True
```

If `import flash_rt` fails with "no module named flash_rt_kernels",
either (a) `cmake --build` didn't produce the `.so` (re-run with
`-v` and check the link step succeeded), or (b) you installed
non-editable (`pip install .` instead of `pip install -e .`) and
the import is hitting a stale site-packages copy. Check in order.

## 7.1 `flash-attn` (optional)

The default RTX Pi0 / Pi0.5 path routes attention through the
vendored `flash_rt_fa2.so` (built from `csrc/attention/flash_attn_2_src/`)
and does **not** require the upstream `flash-attn` pip package.
You only need to install `flash-attn` if:

- You set `FVK_RTX_FA2=0` to fall back to the legacy upstream path, or
- You set `FVK_RTX_FA2_SITES=…` to bisect a subset of attention
  sites against the upstream reference.

The GROOT N1.6 / N1.7 RTX backends also use FlashRT's vendored
attention modules by default; they should not require the upstream
`flash-attn` wheel.

When you do need it, prefer a prebuilt wheel matching your
torch / CUDA / Python combo from
[the flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases) —
building the source distribution typically takes 30+ minutes on a
cold cloud image (Modal, RunPod, etc.).

## 8. JAX frontend (optional)

The JAX path uses a specific Orbax / jaxlib / PJRT plugin combo. Pins
below are what we test against — don't upgrade one without the others:

```bash
pip install jax==0.5.3 jax-cuda12-pjrt==0.5.3 jax-cuda12-plugin==0.5.3 ml_dtypes==0.5.3 orbax-checkpoint flax
```

Upgrade path (tracked, not yet done):

- jax 0.6+ needs the `jax-cuda12-plugin` name to stay aligned (no
  rename expected but verify); check the PJRT plugin registers
  cleanly with `python -c "import jax; jax.devices()"`.
- Orbax 0.6+ changed the default metadata layout for `StandardRestore`;
  our `load_from_cache` path in `flash_rt/frontends/jax/` expects
  the 0.5.x layout.

## 9. `transformers` version constraint

`transformers<4.56` is pinned because the Pi0.5 PaliGemma tokenizer
was broken by internal refactors in 4.56+. This affects ONLY the
Pi0.5 torch frontend; Pi0 / GROOT / Pi0-FAST are unaffected. Plan
is to upgrade the pin once we port the tokenizer call-site.

## 10. Checkpoints

FlashRT does not bundle model weights. Bring your own Pi0 / Pi0.5 /
GROOT checkpoint in whichever format your trainer produced:

- `safetensors` (HuggingFace / PyTorch format) — used by the torch
  frontends
- Orbax (JAX native) — used by the JAX frontends

See [USAGE.md](../USAGE.md) §Loading a model for the per-frontend
`load_model` call.

## 11. Troubleshooting quick reference

| Symptom | Likely cause |
|---|---|
| `CMake Error ... CUTLASS headers not found` | Step 4 skipped |
| `No module named 'flash_rt_kernels'` | Step 6's `cp *.so` step skipped, OR non-editable install |
| `PJRT plugin ... not found` at JAX import | JAX / jax-cuda12-plugin version mismatch (Step 8) |
| `cuBLAS error code=13` when loading second model | Ran two model loads in one process; subprocess-isolate per model |
| cos regression right after calibrate | `act_scale * weight_scale` alpha computed in f64 somewhere; see `docs/calibration.md` §2.3 |

## 12. Known runtime issue: cuBLASLt FP8 heuristic `code=15`

Some FP8 cuBLASLt descriptors are sensitive to the cuBLASLt runtime
patch version. `CUDA 13`, `CUDA 12.4`, or `libcublasLt.so.13` alone is
not specific enough to identify the runtime behavior.

If an FP8 GEMM fails with:

```text
cublasLtMatmulAlgoGetHeuristic(...), CUBLAS_STATUS_NOT_SUPPORTED / code=15
```

first print the exact cuBLASLt runtime version from the same Python
environment that imports `flash_rt`:

```python
import ctypes
import ctypes.util

lib = ctypes.CDLL(ctypes.util.find_library("cublasLt"))
lib.cublasLtGetVersion.restype = ctypes.c_size_t
print(lib.cublasLtGetVersion())
```

Known local result on the same RTX 5090 and the same FP8 descriptor:

| cuBLASLt runtime | Result |
|---|---|
| `13.0.2` (`cublasLtGetVersion() == 130002`) | one SM120 FP8 NN descriptor returned `code=15` |
| `13.1.0` (`cublasLtGetVersion() == 130100`) | the same descriptor succeeded |

On SM89, FP8 fused epilogue descriptors can show the same class of
failure on older CUDA/cuBLASLt stacks. Treat this as a runtime-library
capability issue until the exact cuBLASLt version, GPU, descriptor
shape/layout, and epilogue have been checked. Prefer the validated
Docker/NGC stack when debugging FP8 cuBLASLt heuristic failures, and
include `cublasLtGetVersion()` in bug reports.

# ROCm Deployment

FlashRT keeps the AMD backend additive. CUDA kernels stay under `csrc/`, while
ROCm/HIP kernels live under `rsrc/` and build into:

```text
flash_rt/flash_rt_rocm_kernels*.so
```

The first supported public route is Pi0.5 with the PyTorch frontend:

```python
import flash_rt

model = flash_rt.load_model(
    checkpoint="/path/to/pi05/checkpoint",
    framework="torch",
    config="pi05",
    hardware="rocm",
    num_views=3,
)
```

`hardware="auto"` also resolves to `rocm` when PyTorch reports a HIP runtime
through `torch.version.hip`.

## Build

Install a ROCm PyTorch environment with `hipcc`, `pybind11`, and hipBLASLt
available, then build the extension:

```bash
GPU_ARCH=gfx942 bash scripts/build_rocm_kernels.sh
```

Set `GPU_ARCH` to the target offload architecture for the deployment card.
The default is `gfx942`.

## Hardware Tiers

ROCm support is capability-gated by the GPU architecture and the installed
ROCm libraries.

Compatibility tier:

- Typical architecture: `gfx1100` / Navi 31 / Radeon PRO W7900-class devices.
- Expected use: build validation, BF16 kernels, HIP buffers, ROCm attention,
  Pi0.5 BF16 pipeline, and public frontend routing.
- FP8 status: PyTorch may expose FP8 dtypes, but hipBLASLt may not return a
  usable FP8 GEMM/Linear algorithm. FP8 tests and benchmarks report this as a
  runtime capability boundary.

Performance tier:

- Typical architecture: `gfx942` / Instinct MI300-class devices.
- Expected use: BF16 plus FP8 GEMM/Linear performance validation.
- FP8 status: this is the target tier for enabling and tuning the static FP8
  Pi0.5 pipeline.

## Validation

Run the ROCm unit tests from the repository root:

```bash
PYTHONPATH=. pytest \
  tests/test_rocm_extension_smoke.py \
  tests/test_rocm_hardware_backend.py \
  tests/test_pi05_rocm_pipeline.py \
  tests/test_rocm_frontend_dispatch.py
```

The smoke tests treat FP8 hipBLASLt GEMM/Linear as a runtime capability. BF16
kernels and the Pi0.5 BF16 pipeline are hard requirements. FP8 GEMM tests run
only when hipBLASLt returns a usable FP8 algorithm for the current ROCm runtime
and GPU.

## Benchmark

Use the synthetic Pi0.5 ROCm benchmark for quick first-light timing:

```bash
PYTHONPATH=. python scripts/benchmark_rocm_pi05.py
```

The default benchmark is intentionally lightweight:

- ROCm extension import and hipBLASLt probe
- representative BF16 hipBLASLt Linear
- ROCm SDPA attention backend for SigLIP, encoder, and decoder shapes

To include the larger Pi0.5 GEMM bake path:

```bash
PYTHONPATH=. python scripts/benchmark_rocm_pi05.py --include-pipeline-bake
```

To check FP8 capability in the same run:

```bash
PYTHONPATH=. python scripts/benchmark_rocm_pi05.py --fp8
```

If FP8 hipBLASLt algorithms are unavailable, the script reports that state
instead of treating it as a BF16 pipeline failure.

## Current Boundaries

- `flash_rt.hardware.rocm` owns ROCm runtime capability helpers and the ROCm
  attention backend.
- `flash_rt.core.hip_buffer.HipBuffer` mirrors the CUDA buffer contract for
  stable HIP device addresses.
- `flash_rt.models.pi05.pipeline_rocm.Pi05PipelineRocm` owns the optimized
  Pi0.5 ROCm kernel pipeline.
- `flash_rt.frontends.torch.pi05_rocm.Pi05TorchFrontendRocm` provides the
  public Pi0.5 PyTorch ROCm route with real prompt and observation handling.

Qwen and other decoder-only LLM paths should be integrated in separate
hardware/model PRs rather than mixed into the Pi0.5 ROCm path.

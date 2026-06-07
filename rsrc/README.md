# FlashRT ROCm Source

`rsrc` contains the AMD/ROCm backend. The existing `csrc` tree remains CUDA-only.

The first ROCm artifact is intentionally separate:

```text
flash_rt/flash_rt_rocm_kernels*.so
```

This keeps AMD development independent from the CUDA build while the APIs converge.

Build from the repository root:

```bash
GPU_ARCH=gfx942 bash scripts/build_rocm_kernels.sh
```

See `docs/deployment_rocm.md` for validation and benchmark commands.

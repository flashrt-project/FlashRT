# FlashRT ROCm Source

`rsrc` contains the AMD/ROCm backend. The existing `csrc` tree remains CUDA-only.

The first ROCm artifact is intentionally separate:

```text
flash_rt/flash_rt_rocm_kernels*.so
```

This keeps AMD bring-up independent from the CUDA build while the APIs converge.

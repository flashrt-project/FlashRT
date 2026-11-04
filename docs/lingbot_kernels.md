# LingBot-VA / Wan kernels module

`flash_rt_lingbot_va_kernels` is an **opt-in, SM89-only** module that owns the
model-specific kernels for LingBot-VA LIBERO-long (LeRobot `lingbot_va`) and
the shared Wan diffusion primitives it uses. It is separate from
`flash_rt_kernels` so these kernels do not compile into the shared module on
every architecture.

```bash
cmake -S . -B build -DGPU_ARCH=89 \
  -DFLASHRT_BUILD_LINGBOT_VA_KERNELS=ON
cmake --build build --target flash_rt_lingbot_va_kernels -j
```

The flag defaults **OFF**; the default build, `flash_rt_kernels` ownership, and
all existing routes are unchanged. The module is built only when
`GPU_ARCH=89`; other architectures fail the gate explicitly and do not inherit
these compile units.

## What it owns

- Wan/LingBot AdaLN FP8 and gated-residual entry points (`wan_adaln_*`,
  `wan_gated_residual_*`).
- Sage2 SM89 attention, K-cache ring/tail merge and QKV+rope split entries
  (`sage2_attn_self*`, `kv_*`, `qkv_split_bias_norm_rope_v_cat2_*`,
  `v_transpose_f8*`).
- FP4/FP8 weight and activation quantizers, NVFP4 tile-major packing, FP8
  block-128 GEMM epilogues.
- Streaming-VAE norm / AvgDown / NCDHW-to-NDHWC entries.
- Flow-Euler window and cross-attention probability entries.

The model-owned copies under `csrc/lingbot_va/` are the versions of the shared
source files that contain these new entry points; the shared files stay at
their base revisions. The shared `flash_rt_kernels` module does **not** export
any of these symbols (guarded by `tests/test_lingbot_va_kernels.py`).

## Contracts

- Head dimension for the Sage2/QKV entries is 128; the kernels assume the
  layouts documented in their headers.
- FP8 block-128 / NVFP4 scale tensors are caller-owned and persistent.
- CUDA Graph capture: the entry points take raw device pointers and a stream;
  they do not allocate or synchronize on the host.

## Tests

`tests/test_lingbot_va_kernels.py` imports the module, checks all required
symbols, and asserts the shared module does not export them. It skips when the
module is not built. `tests/test_fa2_lingbot_slim.py` covers the separate FA2
slim module.

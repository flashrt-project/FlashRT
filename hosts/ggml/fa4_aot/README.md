# AOT FlashAttention-4 module (SigLIP vision attention, Thor SM110)

`fa4_siglip_fwd.h` / `fa4_siglip_fwd.o` are the CuTe-DSL ahead-of-time
export of the vendored FA4 SM100-compatible forward
(`csrc/attention/flash_attn_4_src/flashrt_fa4`) compiled for `sm_110a`
at head_dim 80 — the padded-head layout the ggml adapter's vision path
uses. `fa4_prefill_fwd.h` / `fa4_prefill_fwd.o` are the same export at
the pi0.5 prefill shape (head_dim 256, GQA with one KV head, full
attention): the prefill's row-uniform pad mask is reproduced by passing
the real sequence length as the KV dynamic shape. Sequence length, head count and batch stay dynamic; the softmax
scale is a runtime argument. The `.o` contains the embedded cubin plus
the host launch entry; `fr_fa4_shims.c` provides the small `_cuda*`
runtime aliases the object expects, so no CuTe-DSL runtime library is
needed at build or run time.

The ggml adapter build enables the FA4 vision-attention window
automatically when these files are present (see the host build's
`GGML_CUDA_FLASHRT` integration); delete them or set
`GGML_FLASHRT_NO_VIT_FA4=1` / `GGML_FLASHRT_NO_PREFILL_FA4=1` to fall
back to the host's own flash attention per site.

## Regeneration

Requires the `thor-fa4` runtime deps (`nvidia-cutlass-dsl`,
`quack-kernels`) and PyTorch with CUDA, on the target device:

```bash
CUTE_DSL_ARCH=sm_110a python export_fa4_siglip.py
```

The script compiles the vendored FA4 forward once at the head_dim-80
shape with `--enable-tvm-ffi` stripped (the plain JIT object carries the
classic C-header exporter) and writes both files into this directory.

# FlashAttention-4 AOT modules

Generated from FlashRT's vendored FA4 CuTe DSL forward
(`csrc/attention/flash_attn_4_src`) by `backends/tensorrt/tools/export_fa4_aot.py`, through the
same loader and entry point the FlashRT torch pipeline uses. Each module is a C
header plus an object file with the kernel binary embedded.

| module | attention | pi0.5 site |
|---|---|---|
| `fa4_hd256_fwd` | head_dim 256, GQA with 1 KV head, Sq x heads > 128 | encoder prefill |
| `fa4_hd256_q1_fwd` | same, Sq x heads <= 128 | decoder (10 queries) |
| `fa4_hd72_fwd` | head_dim 72, MHA, Sq > 128 | SigLIP |

Batch and sequence lengths are runtime values; the only length-dependent
compile-key field is the query stage count, hence the two hd256 modules.
`backends/tensorrt/tests/fa4_aot_parity.py` checks each module is bitwise equal to the JIT FA4.

Build environment: Jetson Thor, nvidia-cutlass-dsl 4.5.1,
`CUTE_DSL_ARCH=sm_101a`, `FLASH_ATTENTION_ARCH=sm_100a`, FlashRT
perf/pi05-thor-limit 340b750.

Regenerate:

    CUTE_DSL_ARCH=sm_101a python backends/tensorrt/tools/export_fa4_aot.py csrc/attention/fa4_aot

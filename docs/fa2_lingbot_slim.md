# LingBot slim FA2 module

`FLASHRT_FA2_LINGBOT_SLIM` builds a `flash_rt_fa2` module that contains only the
two FA2 instantiations the LingBot-VA native transformer calls. It is opt-in;
the default build keeps the generic matrix.

```bash
cmake -S . -B build -DGPU_ARCH=89 \
  -DFLASHRT_ENABLE_LINGBOT_VA_SM89=ON \
  -DFLASHRT_FA2_LINGBOT_SLIM=ON -DFA2_ARCH_NATIVE_ONLY=ON
cmake --build build --target flash_rt_fa2 -j
```

The module exports `fwd_bf16` (`Flash_fwd_kernel_traits<128,128,32,4,...>`) and
`fwd_bf16_tile` (`<128,64,64,4,...>`), both bf16 / `head_dim=128` / MHA /
non-causal / no window / no split-KV. The C entries reject other head dims and
split-KV scratch. The vendor `ALIBI`, `SOFTCAP`, `LOCAL`, `UNEVEN_K` and
`DROPOUT` switches are collapsed at compile time because that configuration is
fixed, so the fatbin contains one cubin instead of the full template matrix.

## Shape contract

- `Q` is `(batch, seqlen_q, num_heads_q, 128)`; `K`/`V` are
  `(batch, seqlen_k, num_heads_kv, 128)`; `O` has `Q`'s shape. Strides are the
  tensors' `.stride()` triples.
- `seqlen_q > 64` selects `fwd_bf16`; `seqlen_q <= 64` selects
  `fwd_bf16_tile`.
- `num_heads_q == num_heads_kv` (MHA). `seqused_k = 0` means full attention.
- `softmax_scale = 1 / sqrt(128)` at the LingBot call sites.

## Build and tests

- The slim module is about 0.94 MiB (one sm_89 cubin) versus 73 MiB for the
  generic module with the current vendor sources.
- Generic vs slim outputs are bit-identical at
  `B=2, S_q=128/16, S_k=2160`.
- `tests/test_fa2_lingbot_slim.py` checks the two exports, rejects generic
  entries, and compares with PyTorch SDPA; it skips on the generic module and
  on machines without CUDA.
- With the flag OFF (default), the build, bindings, and module are unchanged.
  Model integration is outside this module.


All LingBot-VA kernel additions are controlled by
`FLASHRT_ENABLE_LINGBOT_VA_SM89` (default OFF, requires `GPU_ARCH=89`).
This gates the Wan/AdaLN, Sage2, streaming-VAE helpers, new fused norm and
QKV/quantization entries, and the SM89 GEMM/conv additions in the Qwen3-VL
module. Enable `FLASHRT_BUILD_QWEN3_VL=ON` when those GEMM/conv entries are
needed. The slim FA2 module additionally requires its own opt-in flag.
Slim core builds also enable the four existing NVFP4 packing-helper sources
required by the delivery; this does not change LingBot-OFF builds.
Legacy normalization remains in the original translation unit; the new fused
profiles have a separate source file. Invalid slim-entry arguments raise an
exception rather than terminating the process.

The bundle was removed by the author before this follow-up. This PR supplies
kernel/build support only; it does not deliver a native model frontend or
claim that the external model integration follows this repository's pipeline
contract. The published end-to-end measurements are from that external
integration and were not rerun in this build-isolation follow-up.

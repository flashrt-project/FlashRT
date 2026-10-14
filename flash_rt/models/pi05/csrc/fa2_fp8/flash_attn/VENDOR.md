# Vendored: Flash-Attention 2 forward headers, with an FP8 output epilogue

Copied from FlashRT `csrc/attention/flash_attn_2_src/flash_attn/` (upstream
Dao-AILab/flash-attention v2.7.4.post1, BSD-3-Clause, see LICENSE).

Only the headers and the four head_dim-256 bf16 instantiations are vendored; the CUTLASS 3.x tree
FA2 is pinned against and the build flags come from the FlashRT checkout at
build time (see `../../_fa2_fp8.py`). The headers have to live here rather than
being included from FlashRT because a quoted `#include` resolves against the
*including* file's directory first: `flash_fwd_launch_template.h` sitting in
FlashRT's tree would pick up FlashRT's unpatched `flash.h`, and the two
`Flash_fwd_params` layouts would disagree at the ABI boundary.

## Local patches

`flash.h`           adds `o_fp8_ptr`, `o_fp8_scale_ptr`, the three FP8 strides
                    and `o_bf16_skip` to `Flash_fwd_params`.
`flash_fwd_kernel.h` adds `write_o_fp8()` and calls it at the three sites that
                    write the final attention output (non-split epilogue,
                    non-split early-exit, split combine), and makes the bf16
                    store skippable. The FP8 arithmetic is copied from FlashRT
                    `csrc/kernels/quantize.cu::quantize_fp8_kernel_generic` so
                    the result is bit-identical to a `quantize_fp8_static` pass.

Everything else is byte-identical to the FlashRT copy.

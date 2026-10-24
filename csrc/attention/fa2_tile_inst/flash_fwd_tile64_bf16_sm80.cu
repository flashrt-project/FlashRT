// ================================================================
// FlashRT — head_dim 64 FA2 tile set (bf16, non-causal).
//
// The delivered denoise attention reaches the vendored short-query dispatch,
// which routes head_dim <= 96 to Flash_fwd_kernel_traits<96, 64, 32, 4>: the
// template head dim is 96 while the model's head is 64 wide, and the query/KV
// tiles are 64x32. Both cost real work here — a 96-wide template wastes shared
// memory and halves the K/V load vector (kBlockKGmem falls to 32 for 96, where
// 64 would give 64), and a 32-wide KV tile means nine mainloop iterations for
// a 262-token sequence.
//
// This entry keeps the head width the model actually has and offers the two
// tile shapes that follow from it:
//
//   run_mha_fwd_tile64_bf16   : <64, 64, 64, 4>  (kBlockM 64, kBlockN 64)
//   run_mha_fwd_tile64m32_bf16: <64, 32, 64, 4>  (kBlockM 32, kBlockN 64)
//   run_mha_fwd_tile64m32w2_bf16: <64, 32, 64, 2> (kBlockM 32, two warps)
//
// FA2 gives each warp 16 query rows and walks the KV tiles identically for
// every row, so the two-warp 32-row tile computes every output row with the
// same instructions as the 64-row tile. What changes is the grid: at the
// X-VLA denoise shape (262 queries x 16 heads) 144 CTAs (exactly four per SM
// on a 36-SM part, 288 padded rows) instead of 80 (2.2 per SM, 320 rows).
//
// Shape contract: head_dim == 64 (the caller rejects anything else). Additive:
// the upstream dispatch in flash_fwd_launch_template.h is untouched and the
// existing entries keep their tiles. These three instantiations opt in to the
// ragged-tail/dead-warp skip (`SkipDeadTail=true`); every existing FA2 entry
// uses the default false and keeps the unmodified upstream MMA sequence.
// ================================================================

// The kernel templates instantiated here are the vendored Flash-Attention 2
// sources (BSD-3-Clause, v2.7.4.post1); see
// csrc/attention/flash_attn_2_src/VENDOR.md for the upstream pin and patches.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

void run_mha_fwd_tile64_bf16(Flash_fwd_params& params, cudaStream_t stream) {
  run_flash_fwd<Flash_fwd_kernel_traits<64, 64, 64, 4, false, false,
                                        cutlass::bfloat16_t>,
                false, false, true>(params, stream);
}

void run_mha_fwd_tile64m32_bf16(Flash_fwd_params& params, cudaStream_t stream) {
  run_flash_fwd<Flash_fwd_kernel_traits<64, 32, 64, 4, false, false,
                                        cutlass::bfloat16_t>,
                false, false, true>(params, stream);
}

void run_mha_fwd_tile64m32w2_bf16(Flash_fwd_params& params, cudaStream_t stream) {
  run_flash_fwd<Flash_fwd_kernel_traits<64, 32, 64, 2, false, false,
                                        cutlass::bfloat16_t>,
                false, false, true>(params, stream);
}

}  // namespace FLASH_NAMESPACE

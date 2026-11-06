// ================================================================
// FlashRT — short-query FA2 query-tile specialisation (bf16).
//
// Generic FA2 capability, not model-specific: any model whose attention
// has a short seqlen_q pays for the vendored default's 128 query rows per
// CTA. The vendored default for head_dim <= 96 and <= 128 non-causal is
// Flash_fwd_kernel_traits<H, 128, 64, 4>. This entry instead dispatches on
// head_dim to a narrower query tile:
//
//   head_dim <= 64 : Flash_fwd_kernel_traits< 64, 64, 64, 4>   (seqlen_q <= 64)
//                    Flash_fwd_kernel_traits< 64, 64,128, 4>   (otherwise)
//   head_dim <= 96 : Flash_fwd_kernel_traits< 96, 64, 32, 4>   (kBlockM=64)
//   head_dim <= 128: Flash_fwd_kernel_traits<128, 64, 64, 4>   (kBlockM=64)
//
// Shape contract: head_dim in (0, 128]. Out-of-range head_dim is rejected by
// the caller (fa2_wrapper.cu). Only the query tiles above are specialised;
// this is an additive entry and the upstream dispatch in
// flash_fwd_launch_template.h is untouched.
//
// hd64 was swept on VLA-JEPA (sm_120, Qwen3-VL ViT 2 x 256 patches x 16 heads;
// DiT-B action head 40 queries x 12 heads, 40 / 32 keys) over 15 traits:
//   ViT  : <64,128,4> 15.25 us vs vendored <128,128,4> 16.30 us (bit-identical)
//   DiT  : <64, 64,4>  2.66 /  2.57 us vs 5.53 / 5.53 us     (bit-identical)
//
// The tile sweep below was measured on the first consuming model (Hy-VLA;
// ViT hd96 seqlen_q=196 x18 batches, denoise hd128 seqlen_q=41) with flat
// 300-iter CUDA-event timings, same FA2 kernel body, only the traits differ:
//   hd96  (ViT,     N=196): <64,32,4> = 144.7 us vs <128,64,4> = 169.1 us
//   hd128 (denoise, S=41 ): <64,64,4> =   9.64 us vs <128,64,4> = 16.84 us
// hd128 <64,64,4> is bit-identical to the default (kBlockN unchanged, so the
// per-row softmax reduction order is unchanged); hd96 <64,32,4> changes only
// the reduction order and moves the output by ~2e-4.
// ================================================================

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

void run_mha_fwd_smallq_bf16(int head_dim, Flash_fwd_params& params,
                             cudaStream_t stream, bool vla_hd64) {
  if (vla_hd64 && head_dim <= 64) {
#ifdef FA2_HAS_HDIM_64
    // kBlockN=128 is the vendored hd64 default, so <64,128,4> only halves the
    // query tile (bit-identical); queries that fit one 64-row tile also take
    // the 64-wide key tile, bit-identical whenever seqlen_k <= 64.
    if (params.seqlen_q <= 64) {
      run_flash_fwd<Flash_fwd_kernel_traits<64, 64, 64, 4, false, false,
                                            cutlass::bfloat16_t>,
                    false, false>(params, stream);
    } else {
      run_flash_fwd<Flash_fwd_kernel_traits<64, 64, 128, 4, false, false,
                                            cutlass::bfloat16_t>,
                    false, false>(params, stream);
    }
    return;
#endif
    // A build that does not instantiate head-dim 64 keeps the generic
    // hd<=96 entry below instead of silently launching nothing.
  }
  if (head_dim <= 96) {
#ifdef FA2_HAS_HDIM_96
    run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 32, 4, false, false,
                                          cutlass::bfloat16_t>,
                  false, false>(params, stream);
#endif
  } else {
#ifdef FA2_HAS_HDIM_128
    run_flash_fwd<Flash_fwd_kernel_traits<128, 64, 64, 4, false, false,
                                          cutlass::bfloat16_t>,
                  false, false>(params, stream);
#endif
  }
}

}  // namespace FLASH_NAMESPACE

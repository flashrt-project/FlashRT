// FlashRT — FA2 causal instantiation for (bf16, head_dim=256).
//
// Add-only sibling of flash_attn_2_src/flash_attn/
// flash_fwd_hdim256_bf16_sm80.cu. Qwen3.6 full-attention prefill uses
// head_dim=256; exposing Is_causal=true lets one S-token prefill chunk
// run as a single FA2 call instead of S serial q_seq=1 calls.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 256, true>(
    Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim256<cutlass::bfloat16_t, true>(params, stream);
}

}  // namespace FLASH_NAMESPACE

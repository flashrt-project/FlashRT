// FlashVLA — FA2 causal instantiation for (bf16, head_dim=128).
//
// Add-only sibling of csrc/attention/flash_attn_2_src/flash_attn/
// flash_fwd_hdim128_bf16_sm80.cu (which only specializes
// run_mha_fwd_<bf16, 128, false>). The vendored launch template
// already supports Is_causal=true (line 251 of
// flash_fwd_launch_template.h), so this file just adds the
// matching specialization for the causal path. The vendored tree
// is not edited.
//
// Used by csrc/attention/fa2_wrapper_causal.cu (the FlashVLA
// raw-pointer entry point that exposes Is_causal=true to Python).
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::bfloat16_t, true>(params, stream);
}

}  // namespace FLASH_NAMESPACE

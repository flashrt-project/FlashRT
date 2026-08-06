// FlashRT — FA2 causal instantiation for (fp16, head_dim=128).
//
// Sibling of flash_fwd_hdim128_bf16_sm80_causal.cu — adds the fp16
// specialization needed by the Chameleon-7B (Orin SM87) causal
// attention path. The vendored launch template already supports
// Is_causal=true; this file just provides the matching fp16 spec.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t, true>(params, stream);
}

}  // namespace FLASH_NAMESPACE

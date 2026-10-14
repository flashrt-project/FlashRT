// FlashRT — FA2 causal splitkv instantiation for (bf16, head_dim=256).
//
// Add-only sibling of flash_attn_2_src/flash_attn/
// flash_fwd_split_hdim256_bf16_sm80.cu.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<
    cutlass::bfloat16_t, 256, true>(
        Flash_fwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE

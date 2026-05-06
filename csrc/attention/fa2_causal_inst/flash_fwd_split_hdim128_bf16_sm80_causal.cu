// FlashVLA — FA2 causal splitkv instantiation for (bf16, head_dim=128).
//
// Add-only sibling of csrc/attention/flash_attn_2_src/flash_attn/
// flash_fwd_split_hdim128_bf16_sm80.cu (which only instantiates
// run_mha_fwd_splitkv_dispatch<bf16, 128, false>).
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE

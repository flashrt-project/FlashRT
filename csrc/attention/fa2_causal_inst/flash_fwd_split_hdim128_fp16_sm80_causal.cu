// FlashRT — FA2 causal splitkv instantiation for (fp16, head_dim=128).
//
// Sibling of flash_fwd_split_hdim128_bf16_sm80_causal.cu — provides
// the fp16 splitkv dispatch for causal attention. Used by the
// Chameleon-7B (Orin SM87) path when the splitkv heuristic kicks in.
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE

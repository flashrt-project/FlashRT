// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE router top-8-of-256 (M=1 decode). One block, iterative
// block-argmax -> 8 (index, logit) pairs, replacing torch.topk's general
// bitonic sort (~7% of the decode step). EXACT top-k (no precision change).
// Re-normalising the top-8 of softmax(256) equals softmax(top-8 logits), so
// the caller softmaxes only the 8 returned logits.

#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// logits (n_experts,) bf16 -> out_idx (k,) int32 + out_val (k,) fp32 (the
// top-k logits, descending). k must be <= 32. Returns 0 on success.
int nexn2_router_topk_bf16(const void* logits, void* out_idx, void* out_val,
                           int n_experts, int k, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

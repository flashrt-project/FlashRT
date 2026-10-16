// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 Gemma3 prefix attention with a per-row key limit.
//
// The Gemma3 prefix mask is causal over text and bidirectional inside each image
// block, so row i attends to keys [0, lim[i]) where lim is i + 1 on a text row
// and the end of its image block on an image row. lim is not monotone across the
// sequence. One call per layer covers the whole prefix, which is what this entry
// is for.
//
// The same mask is expressible through FA2 by decomposing the prefix into plain
// causal and non-causal calls, and that is what the DM05 pipeline does by
// default: it selects this kernel only under its `prefix_attn_single_call`
// option, which is off because the decomposition measured faster there. FA2
// cannot express the per-row limit directly -- its masks are causal or
// sliding-window and `seqused_k` is per batch, not per row -- so this entry
// exists for a caller that wants the mask in one launch, not as a claim that it
// is the faster route.
//
// head_dim is fixed at 256 (4 key/value heads of 8 query heads each).

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt {
namespace dm05 {

// q, k, v, out are bf16 with contiguous rows: q/out are [S, NH*256] and k/v are
// [S, NKV*256], so the row strides are implied by the head counts. `lim` is a
// device int array of at least S entries, one per query row: row i attends to
// keys [0, lim[i]) and lim[i] must be in [0, S].
void prefix_attn(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                 __nv_bfloat16* out, const int* lim,
                 int S, int NH, int NKV, float scale, cudaStream_t stream);

// Dynamic shared memory the kernel requests per block, in bytes.
inline int prefix_attn_smem_bytes() { return 3 * 64 * 256 * 2; }

}  // namespace dm05
}  // namespace flash_rt

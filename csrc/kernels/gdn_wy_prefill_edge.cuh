#pragma once

#include <cuda_runtime.h>

namespace flash_rt {
namespace kernels {

// WY chunked delta-rule front matter for a batched prefill, with the head
// counts as runtime arguments.
//
// The sibling 27B path has equivalent kernels, but its v-head count is a
// compile-time constant and its gate cumulative sum runs one block of
// `num_v_heads` threads serially over S -- a decode shape. These take the
// counts as arguments and parallelise the cumulative sum over chunks, which is
// what a prefill of a few thousand tokens needs.
//
// Layout conventions match the mma WY kernels:
//   packed:   (chunks, num_v_heads, 64, head_dim), chunks = ceil(S / 64),
//             pack[c, h, i, d] = x[c * 64 + i, h, d], zero past S.
//   g_cumsum: (S, num_v_heads), cumulative within each 64-token chunk.

// Fuses the q/k l2 normalisation, the GQA broadcast of q into v-head slots,
// the chunk-major packing of q, and the gate cumulative sum.
//
// `q` and `k` are read as (S, num_v_heads, head_dim) already broadcast across
// the GQA group -- the form the conv split kernel writes -- and only the group
// leaders are touched, so no strided host-side slice is needed.
//
//   q, k      (S, num_v_heads, head_dim) bf16, GQA-broadcast
//   g         (S, num_v_heads) bf16
//   k_l2      (S, num_k_heads, head_dim) bf16   out, unique heads only
//   q_pack    (chunks, num_v_heads, 64, head_dim) bf16  out
//   g_cumsum  (S, num_v_heads) bf16   out
//
// head_dim must be 128. qk_group = num_v_heads / num_k_heads.
void gdn_wy_norm_pack_q_cumsum_edge_bf16(
    const void* q,
    const void* k,
    const void* g,
    void*       k_l2,
    void*       q_pack,
    void*       g_cumsum,
    int S,
    int num_k_heads,
    int num_v_heads,
    int head_dim,
    int qk_group,
    cudaStream_t stream);

// Chunk-major packing of the un-decayed v the chunk_h stage produces.
//   v      (S, num_v_heads, head_dim) bf16
//   v_pack (chunks, num_v_heads, 64, head_dim) bf16  out, zero past S
void gdn_wy_pack_v_edge_bf16(
    const void* v,
    void*       v_pack,
    int S,
    int num_v_heads,
    int head_dim,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

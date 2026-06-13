// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace flashrt_minimax_msa {

// Tensor-core (mma m16n8k16) fragment-resident block-sparse GQA decode
// attention. Same contract as msa_decode_sparse_attn_cuda but specialized for
// the MiniMax M3 MSA shape: head_dim D = 128 and GQA group Hq/Hkv = 16.
//   q         [B, Hq, D]                      bf16
//   kv_cache  [max_slots, 2, max_len, Hkv, D] bf16  (k=0, v=1)
//   seq_lens  [B] int32, slot_ids [B] int64
//   topk_idx  [Hkv, B, topk] int32            (block ids, -1 padded)
//   out       [B, Hq, D]                      bf16
// block_size must be a multiple of 64.
void msa_decode_sparse_attn_mma_cuda(const void* q, const void* kv_cache,
                                     const int* seq_lens,
                                     const int64_t* slot_ids,
                                     const int* topk_idx, void* out,
                                     int B, int Hq, int Hkv, int D,
                                     int max_slots, int max_len,
                                     int block_size, int topk,
                                     float sm_scale, cudaStream_t stream);

// Paged variant: K/V in separate [max_slots, Hkv, D] caches, logical token ->
// physical slot via req_to_token [max_reqs, max_kv_len]. Same M3 shape limits.
void msa_decode_sparse_attn_mma_paged_cuda(
    const void* q, const void* k_cache, const void* v_cache,
    const int* req_to_token, const int* seq_lens, const int64_t* slot_ids,
    const int* topk_idx, void* out, int B, int Hq, int Hkv, int D,
    int max_slots, int max_kv_len, int block_size, int topk, float sm_scale,
    cudaStream_t stream);

}  // namespace flashrt_minimax_msa

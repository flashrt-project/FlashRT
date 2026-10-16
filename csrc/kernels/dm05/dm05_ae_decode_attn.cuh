// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 action-expert decode attention (grouped-query, head_dim 256).
//
// The action expert attends a short block of query rows (the served action
// chunk, 10 rows) to the prefix key/value cache once per denoising step, 34
// layers deep. FlashRT's head_dim-256 decode attention (pi05_dec_skinny_attn)
// serves a single shared key/value head and combines its splits in a second
// launch; this kernel packs the G query heads that share a key/value head into
// one tensor-core tile so each key/value byte is read once per key/value head,
// and combines the key splits inside the last CTA of each head.
//
// Shapes it is instantiated for (see the dispatch in the .cu):
//   R = G * T <= 32, where T is the query-row count and G the query heads per
//   key/value head. R must be one of the instantiated values.
//
// Workspace the caller owns and must size from these functions:
//   ws_o   NKV * ns_max * 32 * HD floats   (split partials)
//   ws_ml  NKV * ns_max * 32 * 2 floats    (per-split running max and sum)
//   cnt    NKV int32, which the kernel zeroes again after each use, so a
//          captured CUDA graph replays without a memset in between
// with ns_max the largest split count the caller will pass.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace flash_rt {
namespace dm05 {

// Attention of T query rows over the prefix cache, head_dim fixed at 256.
//
//   q      — [T, G * NKV * 256] bf16, row stride q_ld elements; query head
//            (hk * G + gg) sits at column (hk * G + gg) * 256
//   k, v   — [sk, NKV * 256] bf16, row stride kv_ld elements; key/value head
//            hk sits at column hk * 256
//   out    — [T, G * NKV * 256] bf16, row stride o_ld elements, same head
//            ordering as q
//   T      — query rows (the served action chunk)
//   G      — query heads per key/value head
//   NKV    — key/value heads
//   sk     — number of cached keys
//   ns     — key splits; 1 runs the single-pass path and writes `out` directly
//   scale  — softmax scale
//
// The per-row softmax is online in the exp2 domain with the scale folded into
// log2 e, P is rounded to bf16 before the PV product, and all accumulation is
// fp32. G * T must be an instantiated R.
void ae_decode_attn(const __nv_bfloat16* q, long long q_ld,
                    const __nv_bfloat16* k, const __nv_bfloat16* v, long long kv_ld,
                    __nv_bfloat16* out, long long o_ld,
                    float* ws_o, float* ws_ml, int* cnt,
                    int T, int G, int NKV, int sk, int ns, float scale,
                    cudaStream_t stream);

// Instantiated G * T values this build supports.
bool ae_decode_attn_supports_r(int r);

// Dynamic shared memory the kernel requests per block, in bytes.
inline int ae_decode_attn_smem_bytes() {
    return 32 * 264 * 2 + 4 * 32 * 264 * 2 + 32 * 32 * 4 + 32 * 40 * 2 + 3 * 32 * 4;
}

// Floats the caller must allocate for ws_o and ws_ml at a given split count.
inline long long ae_decode_attn_ws_o_floats(int NKV, int ns_max) {
    return (long long)NKV * ns_max * 32 * 256;
}
inline long long ae_decode_attn_ws_ml_floats(int NKV, int ns_max) {
    return (long long)NKV * ns_max * 32 * 2;
}

}  // namespace dm05
}  // namespace flash_rt

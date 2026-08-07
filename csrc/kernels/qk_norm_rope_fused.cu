// ================================================================
// FlashRT — Fused QK LayerNorm + Rotate-Half RoPE (FP16)
//
// Replaces the per-Chameleon-7B-layer chain:
//   qk_layer_norm_fast_fp16(Q, K, q_w/b, k_w/b, Se*H, Hd, ...)
//   rope_rotate_half_fp16(Q, cos, sin, Se, H, Hd, ...)
//   rope_rotate_half_fp16(K, cos, sin, Se, H, Hd, ...)
// with a single kernel launch, saving ~3 launches/layer × 32 layers.
//
// Layout (matches Chameleon-7B prefill):
//   Q, K           : [Se*H, Hd] FP16 (head-interleaved, viewed as [Se, H, Hd])
//   q_w/b, k_w/b   : [Hd] FP16 (per-head LayerNorm shares params across heads)
//   cos_table      : [Se, Hd] FP16 — RoPE cos, tiled cat([c, c], dim=-1)
//   sin_table      : [Se, Hd] FP16 — RoPE sin, tiled cat([s, s], dim=-1)
//   Output: in-place if {q_out, k_out} == {q, k}.
//
// Math (LayerNorm with bias, then rotate_half RoPE):
//   x_n[d] = (x[d] - mean) * inv_std * w[d] + b[d]
//   out[d]      = x_n[d]      * cos[d]      - x_n[d + Hd/2] * sin[d]      (d < Hd/2)
//   out[d+Hd/2] = x_n[d+Hd/2] * cos[d+Hd/2] + x_n[d]        * sin[d+Hd/2]
//
// Kernel layout (matches qk_layer_norm_fast_fp16):
//   Grid:  ((2 * Se * H + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK)
//   Block: dim3(32, ROWS_PER_BLOCK = 8) — 256 threads/CTA
//   Rows [0,         Se*H) → Q
//   Rows [Se*H,    2*Se*H) → K
//
// Per-lane register cache holds the LayerNorm output for both halves of
// the head_dim simultaneously, so rotate_half pairs are co-located in
// registers — no warp shuffle needed for HD=128.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flash_rt {
namespace kernels {

template<int ROWS_PER_BLOCK>
__global__ void qk_norm_rope_fused_fp16_kernel(
        const __half* __restrict__ q,    const __half* __restrict__ k,
        const __half* __restrict__ q_w,  const __half* __restrict__ q_b,
        const __half* __restrict__ k_w,  const __half* __restrict__ k_b,
        const __half* __restrict__ cos_t, const __half* __restrict__ sin_t,
        __half* __restrict__ q_out, __half* __restrict__ k_out,
        int rows_per_qk,           // = Se * num_heads (rows in Q or K)
        int num_heads,             // for computing RoPE seq position from row
        int dim, float eps) {
    constexpr int MAX_PER_LANE = 4;  // covers dim ≤ 256
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int global_row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (global_row >= 2 * rows_per_qk) return;

    const bool is_k = (global_row >= rows_per_qk);
    const int row    = is_k ? (global_row - rows_per_qk) : global_row;  // [0, Se*H)
    const int seq_pos = row / num_heads;                                 // [0, Se)
    // (head_idx = row % num_heads is implicit; LayerNorm params are shared.)

    const __half* x_ptr = is_k ? k    : q;
    const __half* w_ptr = is_k ? k_w  : q_w;
    const __half* b_ptr = is_k ? k_b  : q_b;
    __half*       o_ptr = is_k ? k_out : q_out;

    const __half2* x2 = reinterpret_cast<const __half2*>(x_ptr + (size_t)row * dim);
    __half2*       o2 = reinterpret_cast<__half2*>(      o_ptr + (size_t)row * dim);
    const __half2* w2 = reinterpret_cast<const __half2*>(w_ptr);
    const __half2* b2 = reinterpret_cast<const __half2*>(b_ptr);
    const __half2* c2 = reinterpret_cast<const __half2*>(cos_t + (size_t)seq_pos * dim);
    const __half2* s2 = reinterpret_cast<const __half2*>(sin_t + (size_t)seq_pos * dim);
    const int dim2 = dim >> 1;

    // ── Pass 1: load x into per-lane register cache, accumulate sum for mean.
    __half2 cache[MAX_PER_LANE];
    float local_sum = 0.0f;
    int n = 0;
    #pragma unroll
    for (int it = 0; it < MAX_PER_LANE; ++it) {
        int i = lane + it * 32;
        if (i < dim2) {
            __half2 v = x2[i];
            cache[it] = v;
            local_sum += __half2float(v.x) + __half2float(v.y);
            ++n;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, off);
    const float mean = local_sum / static_cast<float>(dim);

    // ── Pass 2: variance from cached values.
    float local_var = 0.0f;
    #pragma unroll
    for (int it = 0; it < MAX_PER_LANE; ++it) {
        if (it < n) {
            __half2 v = cache[it];
            float d0 = __half2float(v.x) - mean;
            float d1 = __half2float(v.y) - mean;
            local_var += d0 * d0 + d1 * d1;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_var += __shfl_xor_sync(0xffffffff, local_var, off);
    const float inv_std = rsqrtf(local_var / static_cast<float>(dim) + eps);

    // ── Pass 3: normalize + scale + bias → write back into cache[].
    // We re-purpose cache[] to hold the LayerNorm output before applying
    // RoPE, so rotate_half pairs are co-located in the same lane's regs.
    #pragma unroll
    for (int it = 0; it < MAX_PER_LANE; ++it) {
        if (it < n) {
            int i = lane + it * 32;
            __half2 xv = cache[it];
            __half2 wv = w2[i], bv = b2[i];
            float v0 = (__half2float(xv.x) - mean) * inv_std * __half2float(wv.x) + __half2float(bv.x);
            float v1 = (__half2float(xv.y) - mean) * inv_std * __half2float(wv.y) + __half2float(bv.y);
            cache[it] = __halves2half2(__float2half(v0), __float2half(v1));
        }
    }

    // ── Pass 4: rotate_half RoPE — pair-wise on cached (norm) halves.
    //
    // For HD=128 (the production Chameleon shape): dim2 = 64.
    //   it=0 covers half2 indices 0..31  (fp16 indices 0..63   = first half)
    //   it=1 covers half2 indices 32..63 (fp16 indices 64..127 = second half)
    //
    // Each lane holds:
    //   cache[0] = (norm[2*lane],     norm[2*lane+1])     ∈ first half
    //   cache[1] = (norm[2*lane+64],  norm[2*lane+65])    ∈ second half
    //
    // The rotate_half partner of fp16 index d (d < Hd/2) is d + Hd/2 — i.e.
    // cache[0].x partners with cache[1].x, cache[0].y with cache[1].y.
    // ZERO cross-lane communication required for HD=128.
    if (n >= 2) {
        int i_lo = lane;          // half2 index in first half
        int i_hi = lane + 32;     // half2 index in second half (= dim2/2 + lane)

        __half2 norm_lo = cache[0];
        __half2 norm_hi = cache[1];

        __half2 cos_lo = c2[i_lo];
        __half2 sin_lo = s2[i_lo];
        __half2 cos_hi = c2[i_hi];
        __half2 sin_hi = s2[i_hi];

        // First half: out_lo = norm_lo * cos_lo - norm_hi * sin_lo
        float lo_x = __half2float(norm_lo.x) * __half2float(cos_lo.x)
                   - __half2float(norm_hi.x) * __half2float(sin_lo.x);
        float lo_y = __half2float(norm_lo.y) * __half2float(cos_lo.y)
                   - __half2float(norm_hi.y) * __half2float(sin_lo.y);

        // Second half: out_hi = norm_hi * cos_hi + norm_lo * sin_hi
        float hi_x = __half2float(norm_hi.x) * __half2float(cos_hi.x)
                   + __half2float(norm_lo.x) * __half2float(sin_hi.x);
        float hi_y = __half2float(norm_hi.y) * __half2float(cos_hi.y)
                   + __half2float(norm_lo.y) * __half2float(sin_hi.y);

        o2[i_lo] = __halves2half2(__float2half(lo_x), __float2half(lo_y));
        o2[i_hi] = __halves2half2(__float2half(hi_x), __float2half(hi_y));
    } else if (n == 1) {
        // HD < 64 — should never hit in the Chameleon path.
        // Fall back to LayerNorm-only output (RoPE would need a separate pass).
        int i = lane;
        if (i < dim2) o2[i] = cache[0];
    }
}

void qk_norm_rope_fused_fp16(
        const __half* q, const __half* k,
        const __half* q_w, const __half* q_b,
        const __half* k_w, const __half* k_b,
        const __half* cos_t, const __half* sin_t,
        __half* q_out, __half* k_out,
        int seq_len, int num_heads, int dim, float eps,
        cudaStream_t stream) {
    constexpr int ROWS_PER_BLOCK = 8;
    const int rows_per_qk = seq_len * num_heads;     // = Se * H
    const int total_rows  = 2 * rows_per_qk;
    const int blocks = (total_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    const dim3 block(32, ROWS_PER_BLOCK);
    qk_norm_rope_fused_fp16_kernel<ROWS_PER_BLOCK><<<blocks, block, 0, stream>>>(
        q, k, q_w, q_b, k_w, k_b, cos_t, sin_t, q_out, k_out,
        rows_per_qk, num_heads, dim, eps);
}

}  // namespace kernels
}  // namespace flash_rt

// ── Public C-callable entry (consumed by bindings.cpp) ──
extern "C" void flash_rt_qk_norm_rope_fused_fp16(
        const __half* q, const __half* k,
        const __half* q_w, const __half* q_b,
        const __half* k_w, const __half* k_b,
        const __half* cos_t, const __half* sin_t,
        __half* q_out, __half* k_out,
        int seq_len, int num_heads, int dim, float eps,
        cudaStream_t stream) {
    flash_rt::kernels::qk_norm_rope_fused_fp16(
        q, k, q_w, q_b, k_w, k_b, cos_t, sin_t,
        q_out, k_out, seq_len, num_heads, dim, eps, stream);
}

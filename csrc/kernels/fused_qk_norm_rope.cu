// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — Fused Q/K RMSNorm + RoPE kernel.
//
// Single-kernel warp-per-head Q/K normalization + RoPE rotation.
// Each warp processes one attention head independently using shuffle-based
// RMS reduction — no shared memory, no __syncthreads() within a head.
// Replaces qkv_split + rms_norm + rope_apply (3 kernel launches → 1).
//
// Pipeline per warp:
//   1. Load head elements into registers from strided Dq buffer
//   2. Compute RMS sum via warp-shuffle butterfly reduction
//   3. Normalize with per-head Q/K weight, apply RoPE from registers
//   4. Write RoPE result to flat output buffer

#include "fused_qk_norm_rope.cuh"
#include "common.cuh"

namespace flash_rt {
namespace kernels {

__global__ void fused_qk_norm_rope_v4_kernel(
    const __nv_bfloat16* __restrict__ dq,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ q_temp,
    __nv_bfloat16* __restrict__ k_temp,
    int NH, int NKV, int HD, int QKVD, float eps)
{
    using T = __nv_bfloat16;

    constexpr int kWarpCount = 8;
    constexpr int kWarpSize = 32;
    constexpr int kElemsPerThread = 128 / kWarpSize;  // 4 for HD=128
    constexpr int kHalf = 128 >> 1;                    // 64
    constexpr int kRotSlotOffset = kHalf / kWarpSize;  // 2

    int row = blockIdx.x;
    int warp_id = threadIdx.x / kWarpSize;
    int lane_id = threadIdx.x % kWarpSize;
    int NQK = NH * HD;
    int KVD = NKV * HD;

    const T* cos_row = cos + row * HD;
    const T* sin_row = sin + row * HD;
    float cos_vals[kElemsPerThread];
    float sin_vals[kElemsPerThread];
    #pragma unroll
    for (int e = 0; e < kElemsPerThread; ++e) {
        int col = lane_id + e * kWarpSize;
        cos_vals[e] = to_f32(cos_row[col]);
        sin_vals[e] = to_f32(sin_row[col]);
    }

    // Phase 1: Q heads (NH=16, 2 iterations of 8)
    const T* dq_q = dq + row * QKVD;
    T* q_temp_row = q_temp + row * NQK;

    for (int iter = 0; iter < 2; ++iter) {
        int h = iter * kWarpCount + warp_id;
        if (h >= NH) continue;

        const T* head_src = dq_q + h * HD;
        T* head_dst = q_temp_row + h * HD;

        float vals[kElemsPerThread];
        float local_sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            vals[e] = to_f32(head_src[col]);
            local_sum += vals[e] * vals[e];
        }

        float total = local_sum;
        total += __shfl_xor_sync(0xffffffff, total, 16);
        total += __shfl_xor_sync(0xffffffff, total, 8);
        total += __shfl_xor_sync(0xffffffff, total, 4);
        total += __shfl_xor_sync(0xffffffff, total, 2);
        total += __shfl_xor_sync(0xffffffff, total, 1);
        float rms = rsqrtf(total / HD + eps);

        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            vals[e] = vals[e] * rms * to_f32(q_weight[col]);
        }

        #pragma unroll
        for (int e = 0; e < kElemsPerThread; ++e) {
            int col = lane_id + e * kWarpSize;
            float cv = cos_vals[e];
            float sv = sin_vals[e];
            float xv = vals[e];

            int rot_slot = (col < kHalf) ? (e + kRotSlotOffset) : (e - kRotSlotOffset);
            float rot_val = vals[rot_slot];
            if (col < kHalf) rot_val = -rot_val;

            float rot_sin_bf = to_f32(from_f32<T>(rot_val * sv));
            head_dst[col] = from_f32<T>(rot_sin_bf + xv * cv);
        }
    }

    // Phase 2: K heads (NKV=8, 1 iteration of 8)
    const T* dq_k = dq + row * QKVD + NQK;
    T* k_temp_row = k_temp + row * KVD;

    {
        int h = warp_id;
        if (h < NKV) {
            const T* head_src = dq_k + h * HD;
            T* head_dst = k_temp_row + h * HD;

            float vals[kElemsPerThread];
            float local_sum = 0.0f;
            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                vals[e] = to_f32(head_src[col]);
                local_sum += vals[e] * vals[e];
            }

            float total = local_sum;
            total += __shfl_xor_sync(0xffffffff, total, 16);
            total += __shfl_xor_sync(0xffffffff, total, 8);
            total += __shfl_xor_sync(0xffffffff, total, 4);
            total += __shfl_xor_sync(0xffffffff, total, 2);
            total += __shfl_xor_sync(0xffffffff, total, 1);
            float rms = rsqrtf(total / HD + eps);

            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                vals[e] = vals[e] * rms * to_f32(k_weight[col]);
            }

            #pragma unroll
            for (int e = 0; e < kElemsPerThread; ++e) {
                int col = lane_id + e * kWarpSize;
                float cv = cos_vals[e];
                float sv = sin_vals[e];
                float xv = vals[e];

                int rot_slot = (col < kHalf) ? (e + kRotSlotOffset) : (e - kRotSlotOffset);
                float rot_val = vals[rot_slot];
                if (col < kHalf) rot_val = -rot_val;

                float rot_sin_bf = to_f32(from_f32<T>(rot_val * sv));
                head_dst[col] = from_f32<T>(rot_sin_bf + xv * cv);
            }
        }
    }
}

void fused_qk_norm_rope_v4_bf16(
    const __nv_bfloat16* dq,
    const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight,
    const __nv_bfloat16* cos,
    const __nv_bfloat16* sin,
    __nv_bfloat16* q_temp,
    __nv_bfloat16* k_temp,
    int BS, int NH, int NKV, int HD, int QKVD, float eps,
    cudaStream_t stream)
{
    if (BS <= 0 || NH <= 0 || NKV <= 0 || HD <= 0) return;

    constexpr int kThreads = 256;
    dim3 block(kThreads);
    dim3 grid(BS);

    fused_qk_norm_rope_v4_kernel<<<grid, block, 0, stream>>>(
        dq, q_weight, k_weight, cos, sin,
        q_temp, k_temp,
        NH, NKV, HD, QKVD, eps);
}

}  // namespace kernels
}  // namespace flash_rt

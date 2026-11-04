// SPDX-License-Identifier: Apache-2.0
//
// Short private key segment + log-sum-exp merge for Sage2 SM89 attention.
//
// R query groups of T rows attend to one shared key segment and, each, to a
// private segment of T keys of their own. The shared part runs once for all
// R*T rows through qk_int8_sv_f8_bf16_nhd_d128_strided_lse (normalised bf16
// output plus a log2-domain LSE per (head, row)); this kernel adds the private
// part and merges the two softmax segments:
//
//   s_j  = (q_i8 . k_i8_j) * q_scale * k_scale * sm_scale * log2(e)
//   m    = max(lse, max_j s_j)
//   out  = (2^(lse-m) * o_shared + sum_j 2^(s_j-m) * v_j) / (2^(lse-m) + sum_j 2^(s_j-m))
//
// The output is rounded to bf16 and quantised exactly like the Sage FP8
// epilogue (one block per (row, head): s = max(amax/448, 1e-12),
// q = clamp(v / s, +-448), e4m3 RN), i.e. the activation the out-projection
// GEMM reads.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flash_rt {
namespace attention {
namespace sage2 {

namespace {

constexpr int kD = 128;
constexpr int kT = 16;
// The f8 core's LSE line subtracts S_FP8_OFFSET from a running max that
// already carries it, so the value it writes is the true log2 LSE minus this
// constant (measured: exact offset, max residual 1.5e-5 over 3072 rows).
constexpr float kSageLseOffset = 8.807f;

__global__ void tail_merge_fp8_kernel(const int8_t* __restrict__ q8,
                                      const int8_t* __restrict__ k8,
                                      const __half* __restrict__ v16,
                                      const __nv_bfloat16* __restrict__ o_shared,
                                      const float* __restrict__ lse_shared,
                                      const float* __restrict__ q_scale,
                                      const float* __restrict__ k_scale,
                                      int per_head, float sm_scale_log2,
                                      int H, int M,
                                      uint8_t* __restrict__ out8,
                                      float* __restrict__ out_scale,
                                      __nv_bfloat16* __restrict__ out_bf16,
                                      int rs_reduce) {
    __shared__ int32_t ks[kT][kD / 4];
    __shared__ __half vs[kT][kD];
    const int r = blockIdx.x;          // query group (one private key segment)
    const int h = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    const size_t row_stride = (size_t)H * kD;

    for (int i = threadIdx.x; i < kT * kD / 4; i += blockDim.x) {
        const int j = i / (kD / 4), c = i % (kD / 4);
        ks[j][c] = reinterpret_cast<const int32_t*>(
            k8 + ((size_t)(r * kT + j)) * row_stride + (size_t)h * kD)[c];
    }
    for (int i = threadIdx.x; i < kT * kD / 2; i += blockDim.x) {
        const int j = i / (kD / 2), c = i % (kD / 2);
        reinterpret_cast<__half2*>(&vs[j][0])[c] = reinterpret_cast<const __half2*>(
            v16 + ((size_t)(r * kT + j)) * row_stride + (size_t)h * kD)[c];
    }
    __syncthreads();

    const float sc_qk = (per_head ? q_scale[h] * k_scale[h] : q_scale[0] * k_scale[0]) *
                        sm_scale_log2;
    for (int t = warp; t < kT; t += nwarp) {
        const int row = r * kT + t;
        const size_t base = (size_t)row * row_stride + (size_t)h * kD;
        const int32_t qv = reinterpret_cast<const int32_t*>(q8 + base)[lane];
        float s[kT];
        float m = lse_shared[(size_t)h * M + row] + kSageLseOffset;
        const float lse = m;
        if (rs_reduce) {
            // the 16 dot products all-reduced as a reduce-scatter (8+4+2+1+1
            // shuffles) plus a 16-shuffle broadcast instead of 16 five-shuffle
            // butterflies; integer sums are exact in any order
            int p[kT];
#pragma unroll
            for (int j = 0; j < kT; ++j) p[j] = __dp4a(qv, ks[j][lane], 0);
#pragma unroll
            for (int w = kT / 2, o = 16; w >= 1; w >>= 1, o >>= 1) {
                const bool hi = (lane & o) != 0;
#pragma unroll
                for (int j = 0; j < w; ++j) {
                    const int send = hi ? p[j] : p[j + w];
                    const int keep = hi ? p[j + w] : p[j];
                    p[j] = keep + __shfl_xor_sync(0xffffffffu, send, o);
                }
            }
            const int tot = p[0] + __shfl_xor_sync(0xffffffffu, p[0], 1);
#pragma unroll
            for (int j = 0; j < kT; ++j) {
                s[j] = (float)__shfl_sync(0xffffffffu, tot, j << 1) * sc_qk;
                m = fmaxf(m, s[j]);
            }
        } else {
#pragma unroll
        for (int j = 0; j < kT; ++j) {
            int acc = __dp4a(qv, ks[j][lane], 0);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            s[j] = (float)acc * sc_qk;
            m = fmaxf(m, s[j]);
        }
        }
        const float w0 = exp2f(lse - m);
        float den = w0;
        float acc[4];
        const __nv_bfloat162* op = reinterpret_cast<const __nv_bfloat162*>(o_shared + base) + lane * 2;
        const float2 o01 = __bfloat1622float2(op[0]);
        const float2 o23 = __bfloat1622float2(op[1]);
        acc[0] = w0 * o01.x; acc[1] = w0 * o01.y; acc[2] = w0 * o23.x; acc[3] = w0 * o23.y;
#pragma unroll
        for (int j = 0; j < kT; ++j) {
            const float w = exp2f(s[j] - m);
            den += w;
            const __half2* vp = reinterpret_cast<const __half2*>(&vs[j][lane * 4]);
            const float2 v01 = __half22float2(vp[0]);
            const float2 v23 = __half22float2(vp[1]);
            acc[0] += w * v01.x; acc[1] += w * v01.y; acc[2] += w * v23.x; acc[3] += w * v23.y;
        }
        const float inv = 1.0f / den;
        float ob[4];
        float amax = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            ob[e] = __bfloat162float(__float2bfloat16(acc[e] * inv));
            amax = fmaxf(amax, fabsf(ob[e]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float sc = fmaxf(amax / 448.0f, 1.0e-12f);
        uint32_t packed = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float qv8 = fminf(fmaxf(ob[e] / sc, -448.0f), 448.0f);
            packed |= (uint32_t)__nv_fp8_e4m3(qv8).__x << (8 * e);
        }
        reinterpret_cast<uint32_t*>(out8 + base)[lane] = packed;
        if (lane == 0) out_scale[(size_t)row * H + h] = sc;
        if (out_bf16 != nullptr) {
            __nv_bfloat162* bp = reinterpret_cast<__nv_bfloat162*>(out_bf16 + base) + lane * 2;
            bp[0] = __floats2bfloat162_rn(ob[0], ob[1]);
            bp[1] = __floats2bfloat162_rn(ob[2], ob[3]);
        }
    }
}

}  // namespace

int sage2_tail_merge_fp8(const void* q8, const void* k8, const void* v16,
                         const void* o_shared, const float* lse_shared,
                         const float* q_scale, const float* k_scale, bool per_head,
                         float sm_scale, int groups, int group_rows, int num_heads,
                         void* out_fp8, float* out_scale, void* out_bf16,
                         cudaStream_t stream, int rs_reduce) {
    if (group_rows != kT || groups <= 0 || num_heads <= 0) return -1;
    if (!q8 || !k8 || !v16 || !o_shared || !lse_shared || !q_scale || !k_scale ||
        !out_fp8 || !out_scale)
        return -2;
    const float log2e = 1.4426950408889634f;
    dim3 grid(groups, num_heads);
    tail_merge_fp8_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const int8_t*>(q8), reinterpret_cast<const int8_t*>(k8),
        reinterpret_cast<const __half*>(v16),
        reinterpret_cast<const __nv_bfloat16*>(o_shared), lse_shared, q_scale, k_scale,
        per_head ? 1 : 0, sm_scale * log2e, num_heads, groups * kT,
        reinterpret_cast<uint8_t*>(out_fp8), out_scale,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), rs_reduce);
    return (int)cudaGetLastError();
}

}  // namespace sage2
}  // namespace attention
}  // namespace flash_rt

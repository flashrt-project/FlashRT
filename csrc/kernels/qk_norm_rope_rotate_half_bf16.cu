// ============================================================================
//  FlashRT — fused per-head RMSNorm + rotate-half RoPE (bf16, in-place).
//
//  GR00T N1.6 Qwen3 tier: replaces the torch chain per layer
//      q = q_norm(q_proj_out.view(1,S,NH,HD))          (RMSNorm over HD)
//      q, k = apply_rotary_pos_emb(q, k, cos, sin)     (cat/mul/rotate_half)
//  with one warp-per-(row, head) kernel. Launched once for Q (NHQ heads)
//  and once for K (NHKV heads), so GQA is handled by the caller.
//
//  x     : [S, NH*HD] bf16 row-major, modified in place
//  w     : [HD] bf16 RMSNorm weight
//  cos/sin: [S, HD] bf16 (HF rotary_emb output; duplicated halves)
//  Rounding follows the HF chain closely: norm output rounded to bf16
//  before the RoPE multiplies; RoPE accumulated in fp32, rounded once.
// ============================================================================
#include "qk_norm_rope_rotate_half_bf16.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace kernels {

namespace {

__global__ void qk_norm_rope_rotate_half_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    int S, int NH, int HD, float eps) {
    const int half = HD >> 1;                    // pairs per head (HD==128 -> 64)
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= S * NH) return;
    const int n = warp_id % NH;
    const int s = warp_id / NH;
    const int base = s * NH * HD + n * HD;

    // Lane l handles pairs p = l and p = l + 32 (half == 64).
    float vals[4];
    int p[2] = {lane, lane + 32};
    float ssq = 0.f;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int d = p[j];
        const float a = __bfloat162float(x[base + d]);
        const float b = __bfloat162float(x[base + d + half]);
        vals[2 * j] = a; vals[2 * j + 1] = b;
        ssq += a * a + b * b;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
    const float inv = rsqrtf(ssq / HD + eps);

    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int d = p[j];
        // norm, rounded to bf16 like the HF RMSNorm output
        const float n_lo = __bfloat162float(__float2bfloat16(
            vals[2 * j] * inv * __bfloat162float(w[d])));
        const float n_hi = __bfloat162float(__float2bfloat16(
            vals[2 * j + 1] * inv * __bfloat162float(w[d + half])));
        const float c = __bfloat162float(cos_t[s * HD + d]);
        const float si = __bfloat162float(sin_t[s * HD + d]);
        x[base + d] = __float2bfloat16(n_lo * c - n_hi * si);
        x[base + d + half] = __float2bfloat16(n_hi * c + n_lo * si);
    }
}

}  // namespace

int qk_norm_rope_rotate_half_bf16(
    void* x, const void* w, const void* cos_t, const void* sin_t,
    int S, int NH, int HD, float eps, cudaStream_t stream) {
    if (HD != 128) return -1;
    const int total_warps = S * NH;
    const int threads = 128;                     // 4 warps per block
    const int blocks = (total_warps * 32 + threads - 1) / threads;
    qk_norm_rope_rotate_half_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<const __nv_bfloat16*>(cos_t),
        reinterpret_cast<const __nv_bfloat16*>(sin_t),
        S, NH, HD, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace kernels
}  // namespace flash_rt

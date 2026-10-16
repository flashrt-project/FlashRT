// SPDX-License-Identifier: Apache-2.0
// Fused consumers for the skinny NVFP4 GEMM family on sm_120a. See the header.
#include "nvfp4_skinny_consumers_sm120.cuh"
#include "nvfp4_skinny_common.cuh"

namespace flash_rt {
namespace nvfp4_skinny {
namespace {

// Block-wide sum over dim / 4 threads (<= 1024): warp shuffles, then one warp.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    return warp_sum(t);
}

// LayerNorm of the four row values v (fp32 copies of BF16 values) with
// optional modulation, written as NVFP4 or BF16 at column c0.
__device__ __forceinline__ void norm_out(float v0, float v1, float v2, float v3, int dim, float eps,
                                         const __nv_bfloat16* __restrict__ scale,
                                         const __nv_bfloat16* __restrict__ shift, uint8_t* __restrict__ out,
                                         uint8_t* __restrict__ out_sf, const float* __restrict__ out_gs,
                                         __nv_bfloat16* __restrict__ out_bf16, size_t row, int c0, float* red) {
    const float mean = block_sum(v0 + v1 + v2 + v3, red) / dim;
    const float d0 = v0 - mean, d1 = v1 - mean, d2 = v2 - mean, d3 = v3 - mean;
    const float inv_std = rsqrtf(block_sum(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3, red) / dim + eps);
    float n[4] = {d0 * inv_std, d1 * inv_std, d2 * inv_std, d3 * inv_std};
    if (scale != nullptr) {
        const uint2 sc = *reinterpret_cast<const uint2*>(scale + c0);
        const uint2 sh = *reinterpret_cast<const uint2*>(shift + c0);
        const float2 s0 = bf2(sc.x), s1 = bf2(sc.y), h0 = bf2(sh.x), h1 = bf2(sh.y);
        n[0] = n[0] * (1.0f + s0.x) + h0.x;
        n[1] = n[1] * (1.0f + s0.y) + h0.y;
        n[2] = n[2] * (1.0f + s1.x) + h1.x;
        n[3] = n[3] * (1.0f + s1.y) + h1.y;
    }
    if (out_bf16 != nullptr) {
        *reinterpret_cast<uint2*>(out_bf16 + row * dim + c0) = make_uint2(pack_bf2(n[0], n[1]), pack_bf2(n[2], n[3]));
        return;
    }
    const float inv_gs = 1.0f / (*out_gs);
    quant_quad4(n[0] * inv_gs, n[1] * inv_gs, n[2] * inv_gs, n[3] * inv_gs, out + (row * dim + c0) / 2,
                out_sf + row * (dim / 16) + c0 / 16, (threadIdx.x & 3) == 0);
}

template <bool PDL>
__global__ void __launch_bounds__(1024)
layer_norm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ scale,
                  const __nv_bfloat16* __restrict__ shift, uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
                  const float* __restrict__ out_gs, __nv_bfloat16* __restrict__ out_bf16, int dim, float eps) {
    __shared__ float red[32];
    if constexpr (PDL) pdl_prologue();
    const size_t row = blockIdx.x;
    const int c0 = threadIdx.x * 4;
    const uint2 xv = *reinterpret_cast<const uint2*>(x + row * dim + c0);
    const float2 a = bf2(xv.x), b = bf2(xv.y);
    norm_out(a.x, a.y, b.x, b.y, dim, eps, scale, shift, out, out_sf, out_gs, out_bf16, row, c0, red);
}

template <bool PDL>
__global__ void __launch_bounds__(1024)
sum_bias_residual_norm_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                              const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias,
                              __nv_bfloat16* __restrict__ residual, const __nv_bfloat16* __restrict__ scale,
                              const __nv_bfloat16* __restrict__ shift, uint8_t* __restrict__ out,
                              uint8_t* __restrict__ out_sf, const float* __restrict__ out_gs,
                              __nv_bfloat16* __restrict__ out_bf16, int rows, int dim, float eps) {
    __shared__ float red[32];
    const size_t row = blockIdx.x;
    const int c0 = threadIdx.x * 4;
    const uint2 bb = *reinterpret_cast<const uint2*>(bias + c0);
    if constexpr (PDL) pdl_prologue();
    const float alpha = (*a_gs) * (*w_gs);
    const size_t elements = static_cast<size_t>(rows) * dim;
    const size_t pair0 = (row * dim + c0) / 2;
    const float2 x0 = sum_pair(partials, splits, elements, pair0, alpha);
    const float2 x1 = sum_pair(partials, splits, elements, pair0 + 1, alpha);
    const float2 b0 = bf2(bb.x), b1 = bf2(bb.y);
    // bias add rounded to BF16, then the BF16 residual add
    const float2 y0 = bf2(pack_bf2(x0.x + b0.x, x0.y + b0.y));
    const float2 y1 = bf2(pack_bf2(x1.x + b1.x, x1.y + b1.y));
    uint2* rp = reinterpret_cast<uint2*>(residual + row * dim + c0);
    const uint2 rv = *rp;
    const float2 r0 = bf2(rv.x), r1 = bf2(rv.y);
    const uint32_t k0 = pack_bf2(r0.x + y0.x, r0.y + y0.y);
    const uint32_t k1 = pack_bf2(r1.x + y1.x, r1.y + y1.y);
    *rp = make_uint2(k0, k1);
    const float2 h0 = bf2(k0), h1 = bf2(k1);
    norm_out(h0.x, h0.y, h1.x, h1.y, dim, eps, scale, shift, out, out_sf, out_gs, out_bf16, row, c0, red);
}

template <bool PDL>
__global__ void __launch_bounds__(256)
sum_bias_gelu_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                     const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias,
                     uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf, const float* __restrict__ out_gs,
                     int rows, int n) {
    if constexpr (PDL) pdl_prologue();
    const int quads = n >> 2;
    // lanes past the end compute a clamped duplicate so every quad shuffles with full membership
    const int raw = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = raw < rows * quads;
    const int index = valid ? raw : rows * quads - 4 + (raw & 3);
    const int row = index / quads;
    const int c0 = (index % quads) * 4;
    const float alpha = (*a_gs) * (*w_gs);
    const size_t elements = static_cast<size_t>(rows) * n;
    const size_t pair0 = (static_cast<size_t>(row) * n + c0) / 2;
    const float2 x0 = sum_pair(partials, splits, elements, pair0, alpha);
    const float2 x1 = sum_pair(partials, splits, elements, pair0 + 1, alpha);
    const uint2 bb = *reinterpret_cast<const uint2*>(bias + c0);
    const float2 b0 = bf2(bb.x), b1 = bf2(bb.y);
    const float2 y0 = bf2(pack_bf2(x0.x + b0.x, x0.y + b0.y));
    const float2 y1 = bf2(pack_bf2(x1.x + b1.x, x1.y + b1.y));
    const float inv_gs = 1.0f / (*out_gs);
    uint16_t scratch = 0;
    quant_quad4(tanh_gelu(y0.x) * inv_gs, tanh_gelu(y0.y) * inv_gs, tanh_gelu(y1.x) * inv_gs,
                tanh_gelu(y1.y) * inv_gs,
                valid ? out + (static_cast<size_t>(row) * n + c0) / 2 : reinterpret_cast<uint8_t*>(&scratch),
                out_sf + static_cast<size_t>(row) * (n / 16) + c0 / 16, valid && ((c0 >> 2) & 3) == 0);
}

bool row_dim_ok(int dim) { return dim % 128 == 0 && dim >= 128 && dim <= 4096; }

}  // namespace

int layer_norm_fp4(const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
                   uint8_t* out, uint8_t* out_sf, const float* out_gs, __nv_bfloat16* out_bf16, int rows,
                   int dim, float eps, bool pdl, cudaStream_t stream) {
    if (rows < 1 || !row_dim_ok(dim) || ((scale == nullptr) != (shift == nullptr)) ||
        (out_bf16 == nullptr && (out == nullptr || out_sf == nullptr || out_gs == nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)layer_norm_kernel<true> : (const void*)layer_norm_kernel<false>;
    return launch_ex(fn, dim3(rows), dim3(dim / 4), stream, pdl, x, scale, shift, out, out_sf, out_gs, out_bf16,
                     dim, eps);
}

int sum_bias_residual_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                           const __nv_bfloat16* bias, __nv_bfloat16* residual, const __nv_bfloat16* scale,
                           const __nv_bfloat16* shift, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                           __nv_bfloat16* out_bf16, int rows, int dim, float eps, bool pdl, cudaStream_t stream) {
    if (rows < 1 || !row_dim_ok(dim) || ((scale == nullptr) != (shift == nullptr)) ||
        (out_bf16 == nullptr && (out == nullptr || out_sf == nullptr || out_gs == nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)sum_bias_residual_norm_kernel<true>
                         : (const void*)sum_bias_residual_norm_kernel<false>;
    return launch_ex(fn, dim3(rows), dim3(dim / 4), stream, pdl, partials, splits, a_gs, w_gs, bias, residual,
                     scale, shift, out, out_sf, out_gs, out_bf16, rows, dim, eps);
}

int sum_bias_gelu_fp4(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      const __nv_bfloat16* bias, uint8_t* out, uint8_t* out_sf, const float* out_gs, int rows,
                      int n, bool pdl, cudaStream_t stream) {
    if (rows < 1 || n % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const int total = rows * (n >> 2);
    const void* fn = pdl ? (const void*)sum_bias_gelu_kernel<true> : (const void*)sum_bias_gelu_kernel<false>;
    return launch_ex(fn, dim3((total + 255) / 256), dim3(256), stream, pdl, partials, splits, a_gs, w_gs, bias,
                     out, out_sf, out_gs, rows, n);
}

}  // namespace nvfp4_skinny
}  // namespace flash_rt

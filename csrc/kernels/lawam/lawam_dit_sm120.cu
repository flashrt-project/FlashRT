// SPDX-License-Identifier: Apache-2.0
// Skinny NVFP4 consumers for the LaWAM flow DiT on sm_120a. Same numerics and
// layouts as gemm/fp4/nvfp4_skinny_consumers_sm120.cu (see that file), with
// the per-row positional embedding of every block input, the residual
// initialization of the first block, a ReLU / identity NVFP4 consumer and the
// FP32 Euler update of the flow sample.
#include "lawam_kernels.cuh"

#include "gemm/fp4/nvfp4_skinny_common.cuh"

namespace flash_rt {
namespace lawam {
namespace {

using nvfp4_skinny::bf2;
using nvfp4_skinny::launch_ex;
using nvfp4_skinny::pack_bf2;
using nvfp4_skinny::pdl_prologue;
using nvfp4_skinny::quant_quad4;
using nvfp4_skinny::sum_pair;
using nvfp4_skinny::warp_sum;

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float t = lane < nw ? red[lane] : 0.f;
    return warp_sum(t);
}

__device__ __forceinline__ void norm_out(float v0, float v1, float v2, float v3, int dim, float eps,
                                         const __nv_bfloat16* __restrict__ scale,
                                         const __nv_bfloat16* __restrict__ shift,
                                         const __nv_bfloat16* __restrict__ pos, uint8_t* __restrict__ out,
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
    if (pos != nullptr) {
        // the modulated norm is rounded to BF16 before the positional add,
        // as in the unfused chain (norm1 output + positional embedding)
        const uint2 pv = *reinterpret_cast<const uint2*>(pos + row * dim + c0);
        const float2 p0 = bf2(pv.x), p1 = bf2(pv.y);
        const float2 m0 = bf2(pack_bf2(n[0], n[1])), m1 = bf2(pack_bf2(n[2], n[3]));
        n[0] = m0.x + p0.x;
        n[1] = m0.y + p0.y;
        n[2] = m1.x + p1.x;
        n[3] = m1.y + p1.y;
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
                  const __nv_bfloat16* __restrict__ shift, const __nv_bfloat16* __restrict__ pos,
                  uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf, const float* __restrict__ out_gs,
                  __nv_bfloat16* __restrict__ out_bf16, int dim, float eps) {
    __shared__ float red[32];
    if constexpr (PDL) pdl_prologue();
    const size_t row = blockIdx.x;
    const int c0 = threadIdx.x * 4;
    const uint2 xv = *reinterpret_cast<const uint2*>(x + row * dim + c0);
    const float2 a = bf2(xv.x), b = bf2(xv.y);
    norm_out(a.x, a.y, b.x, b.y, dim, eps, scale, shift, pos, out, out_sf, out_gs, out_bf16, row, c0, red);
}

template <bool PDL>
__global__ void __launch_bounds__(1024)
sum_bias_residual_norm_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                              const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias,
                              __nv_bfloat16* __restrict__ residual, bool accumulate,
                              const __nv_bfloat16* __restrict__ scale, const __nv_bfloat16* __restrict__ shift,
                              const __nv_bfloat16* __restrict__ pos, uint8_t* __restrict__ out,
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
    const float2 y0 = bf2(pack_bf2(x0.x + b0.x, x0.y + b0.y));
    const float2 y1 = bf2(pack_bf2(x1.x + b1.x, x1.y + b1.y));
    uint2* rp = reinterpret_cast<uint2*>(residual + row * dim + c0);
    float2 r0 = make_float2(0.f, 0.f), r1 = make_float2(0.f, 0.f);
    if (accumulate) {
        const uint2 rv = *rp;
        r0 = bf2(rv.x);
        r1 = bf2(rv.y);
    }
    const uint32_t k0 = pack_bf2(r0.x + y0.x, r0.y + y0.y);
    const uint32_t k1 = pack_bf2(r1.x + y1.x, r1.y + y1.y);
    *rp = make_uint2(k0, k1);
    const float2 h0 = bf2(k0), h1 = bf2(k1);
    norm_out(h0.x, h0.y, h1.x, h1.y, dim, eps, scale, shift, pos, out, out_sf, out_gs, out_bf16, row, c0, red);
}

template <bool PDL>
__global__ void __launch_bounds__(256)
sum_bias_act_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                    const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias, int act,
                    uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf, const float* __restrict__ out_gs,
                    __nv_bfloat16* __restrict__ out_bf16, int rows, int n) {
    if constexpr (PDL) pdl_prologue();
    const int quads = n >> 2;
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
    float2 y0 = bf2(pack_bf2(x0.x + b0.x, x0.y + b0.y));
    float2 y1 = bf2(pack_bf2(x1.x + b1.x, x1.y + b1.y));
    if (act == 1) {
        y0 = make_float2(fmaxf(y0.x, 0.f), fmaxf(y0.y, 0.f));
        y1 = make_float2(fmaxf(y1.x, 0.f), fmaxf(y1.y, 0.f));
    } else if (act == 2) {
        y0 = make_float2(nvfp4_skinny::tanh_gelu(y0.x), nvfp4_skinny::tanh_gelu(y0.y));
        y1 = make_float2(nvfp4_skinny::tanh_gelu(y1.x), nvfp4_skinny::tanh_gelu(y1.y));
    }
    if (out_bf16 != nullptr) {
        if (valid)
            *reinterpret_cast<uint2*>(out_bf16 + static_cast<size_t>(row) * n + c0) =
                make_uint2(pack_bf2(y0.x, y0.y), pack_bf2(y1.x, y1.y));
        return;
    }
    const float inv_gs = 1.0f / (*out_gs);
    uint16_t scratch = 0;
    quant_quad4(y0.x * inv_gs, y0.y * inv_gs, y1.x * inv_gs, y1.y * inv_gs,
                valid ? out + (static_cast<size_t>(row) * n + c0) / 2 : reinterpret_cast<uint8_t*>(&scratch),
                out_sf + static_cast<size_t>(row) * (n / 16) + c0 / 16, valid && ((c0 >> 2) & 3) == 0);
}

template <bool PDL>
__global__ void __launch_bounds__(256)
sum_bias_bf16_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                     const float* __restrict__ w_gs, const __nv_bfloat16* __restrict__ bias,
                     __nv_bfloat16* __restrict__ out, int rows, int n) {
    if constexpr (PDL) pdl_prologue();
    const int quads = n >> 2;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * quads) return;
    const int row = index / quads;
    const int c0 = (index % quads) * 4;
    const float alpha = (*a_gs) * (*w_gs);
    const size_t elements = static_cast<size_t>(rows) * n;
    const size_t pair0 = (static_cast<size_t>(row) * n + c0) / 2;
    const float2 x0 = sum_pair(partials, splits, elements, pair0, alpha);
    const float2 x1 = sum_pair(partials, splits, elements, pair0 + 1, alpha);
    const uint2 bb = *reinterpret_cast<const uint2*>(bias + c0);
    const float2 b0 = bf2(bb.x), b1 = bf2(bb.y);
    *reinterpret_cast<uint2*>(out + static_cast<size_t>(row) * n + c0) =
        make_uint2(pack_bf2(x0.x + b0.x, x0.y + b0.y), pack_bf2(x1.x + b1.x, x1.y + b1.y));
}

template <bool PDL>
__global__ void __launch_bounds__(256)
sum_bias_euler_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                      const float* __restrict__ w_gs, const float* __restrict__ bias, float* __restrict__ x,
                      float dt, int rows, int n) {
    if constexpr (PDL) pdl_prologue();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const float alpha = (*a_gs) * (*w_gs);
    const size_t elements = static_cast<size_t>(rows) * n;
    float s = 0.f;
    for (int k = 0; k < splits; ++k) s += partials[k * elements + i];
    x[i] += dt * (s * alpha + bias[i % n]);
}

bool row_dim_ok(int dim) { return dim % 128 == 0 && dim >= 128 && dim <= 4096; }

}  // namespace

int dit_sum_bias_residual_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                               const __nv_bfloat16* bias, __nv_bfloat16* residual, bool accumulate,
                               const __nv_bfloat16* scale, const __nv_bfloat16* shift, const __nv_bfloat16* pos,
                               uint8_t* out, uint8_t* out_sf, const float* out_gs, __nv_bfloat16* out_bf16,
                               int rows, int dim, float eps, bool pdl, cudaStream_t stream) {
    if (rows < 1 || !row_dim_ok(dim) || ((scale == nullptr) != (shift == nullptr)) ||
        (out_bf16 == nullptr && (out == nullptr || out_sf == nullptr || out_gs == nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)sum_bias_residual_norm_kernel<true>
                         : (const void*)sum_bias_residual_norm_kernel<false>;
    return launch_ex(fn, dim3(rows), dim3(dim / 4), stream, pdl, partials, splits, a_gs, w_gs, bias, residual,
                     accumulate, scale, shift, pos, out, out_sf, out_gs, out_bf16, rows, dim, eps);
}

int dit_sum_bias_act(const float* partials, int splits, const float* a_gs, const float* w_gs,
                     const __nv_bfloat16* bias, int act, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                     __nv_bfloat16* out_bf16, int rows, int n, bool pdl, cudaStream_t stream) {
    if (rows < 1 || n % 16 != 0 || act < 0 || act > 2 ||
        (out_bf16 == nullptr && (out == nullptr || out_sf == nullptr || out_gs == nullptr)))
        return static_cast<int>(cudaErrorInvalidValue);
    const int total = rows * (n >> 2);
    const void* fn = pdl ? (const void*)sum_bias_act_kernel<true> : (const void*)sum_bias_act_kernel<false>;
    return launch_ex(fn, dim3((total + 255) / 256), dim3(256), stream, pdl, partials, splits, a_gs, w_gs, bias,
                     act, out, out_sf, out_gs, out_bf16, rows, n);
}

int dit_sum_bias_bf16(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      const __nv_bfloat16* bias, __nv_bfloat16* out, int rows, int n, bool pdl, cudaStream_t stream) {
    if (rows < 1 || n % 4 != 0 || out == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    const int total = rows * (n >> 2);
    const void* fn = pdl ? (const void*)sum_bias_bf16_kernel<true> : (const void*)sum_bias_bf16_kernel<false>;
    return launch_ex(fn, dim3((total + 255) / 256), dim3(256), stream, pdl, partials, splits, a_gs, w_gs, bias,
                     out, rows, n);
}

int dit_sum_bias_euler(const float* partials, int splits, const float* a_gs, const float* w_gs,
                       const float* bias, float* x, float dt, int rows, int n, bool pdl, cudaStream_t stream) {
    if (rows < 1 || n < 1) return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)sum_bias_euler_kernel<true> : (const void*)sum_bias_euler_kernel<false>;
    return launch_ex(fn, dim3((rows * n + 255) / 256), dim3(256), stream, pdl, partials, splits, a_gs, w_gs, bias,
                     x, dt, rows, n);
}

}  // namespace lawam
}  // namespace flash_rt

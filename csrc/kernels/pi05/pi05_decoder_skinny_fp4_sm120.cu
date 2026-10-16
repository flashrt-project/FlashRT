// SPDX-License-Identifier: Apache-2.0
// Pi0.5 action-decoder consumers for the skinny NVFP4 family on sm_120a. See the header.
#include "pi05_decoder_skinny_fp4_sm120.cuh"
#include "nvfp4_skinny_common.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace flash_rt {
namespace pi05_dec_skinny_fp4 {
namespace {

using namespace flash_rt::nvfp4_skinny;

// One 256-thread block per row of 1024; thread t owns elements [4t, 4t + 4).
template <bool PDL>
__global__ void __launch_bounds__(256)
residual_ada_norm_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                         const float* __restrict__ w_gs, __nv_bfloat16* __restrict__ residual,
                         const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ weight,
                         const __nv_bfloat16* __restrict__ style, uint8_t* __restrict__ out,
                         uint8_t* __restrict__ out_sf, const float* __restrict__ out_gs,
                         __nv_bfloat16* __restrict__ gate_out, int rows, float eps) {
    constexpr int DIM = 1024;
    __shared__ float warp_partial[8];
    const int row = blockIdx.x, tid = threadIdx.x, n0 = tid * 4;
    const size_t base = static_cast<size_t>(row) * DIM;
    const size_t elements = static_cast<size_t>(rows) * DIM;
    const __nv_bfloat16* style_row = style + static_cast<size_t>(row) * 3 * DIM;
    const uint2 wv = *reinterpret_cast<const uint2*>(weight + n0);
    const uint2 sc = *reinterpret_cast<const uint2*>(style_row + n0);
    const uint2 sh = *reinterpret_cast<const uint2*>(style_row + DIM + n0);
    const uint2 gt = *reinterpret_cast<const uint2*>(style_row + 2 * DIM + n0);
    if constexpr (PDL) pdl_prologue();
    const float alpha = (*a_gs) * (*w_gs);
    const float2 x0 = sum_pair(partials, splits, elements, base / 2 + 2 * tid, alpha);
    const float2 x1 = sum_pair(partials, splits, elements, base / 2 + 2 * tid + 1, alpha);
    const uint2 rr = *reinterpret_cast<const uint2*>(residual + base + n0);
    const uint2 gg = *reinterpret_cast<const uint2*>(gate + base + n0);
    const float2 r0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rr.x));
    const float2 r1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rr.y));
    const float2 g0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg.x));
    const float2 g1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gg.y));
    const float v[4] = {r0.x + x0.x * g0.x, r0.y + x0.y * g0.y, r1.x + x1.x * g1.x, r1.y + x1.y * g1.y};
    const __nv_bfloat162 k0 = __floats2bfloat162_rn(v[0], v[1]);
    const __nv_bfloat162 k1 = __floats2bfloat162_rn(v[2], v[3]);
    uint2 kept;
    kept.x = *reinterpret_cast<const uint32_t*>(&k0);
    kept.y = *reinterpret_cast<const uint32_t*>(&k1);
    *reinterpret_cast<uint2*>(residual + base + n0) = kept;
    float sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    sq = warp_sum(sq);
    if ((tid & 31) == 0) warp_partial[tid >> 5] = sq;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_partial[w];
    const float rms = rsqrtf(total / DIM + eps);
    const float inv_gs = 1.0f / (*out_gs);
    const float2 kf0 = __bfloat1622float2(k0), kf1 = __bfloat1622float2(k1);
    const float2 w0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.x));
    const float2 w1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.y));
    const float2 s0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sc.x));
    const float2 s1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sc.y));
    const float2 h0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sh.x));
    const float2 h1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sh.y));
    quant_quad4((kf0.x * rms * w0.x * (1.0f + s0.x) + h0.x) * inv_gs,
                (kf0.y * rms * w0.y * (1.0f + s0.y) + h0.y) * inv_gs,
                (kf1.x * rms * w1.x * (1.0f + s1.x) + h1.x) * inv_gs,
                (kf1.y * rms * w1.y * (1.0f + s1.y) + h1.y) * inv_gs,
                out + base / 2 + 2 * tid, out_sf + static_cast<size_t>(row) * (DIM / 16) + (tid >> 2),
                (tid & 3) == 0);
    *reinterpret_cast<uint2*>(gate_out + base + n0) = gt;
}

// GeGLU over merged [gate | up] partials; thread owns 4 consecutive outputs.
template <bool PDL>
__global__ void __launch_bounds__(256)
gate_gelu_kernel(const float* __restrict__ partials, int splits, const float* __restrict__ a_gs,
                 const float* __restrict__ w_gs, uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
                 int rows, int half, const float* __restrict__ out_gs) {
    if constexpr (PDL) pdl_prologue();
    const int quads = half >> 2;
    // lanes past the end compute a clamped duplicate so every quad shuffles with its full membership
    const int raw = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = raw < rows * quads;
    const int index = valid ? raw : rows * quads - 4 + (raw & 3);
    const int row = index / quads;
    const int q = index % quads;
    const size_t elements = static_cast<size_t>(rows) * 2 * half;
    const float alpha = (*a_gs) * (*w_gs);
    const size_t row_pairs = static_cast<size_t>(row) * half;
    const float2 ga = sum_pair(partials, splits, elements, row_pairs + 2 * q, alpha);
    const float2 gb = sum_pair(partials, splits, elements, row_pairs + 2 * q + 1, alpha);
    const float2 ua = sum_pair(partials, splits, elements, row_pairs + (half >> 1) + 2 * q, alpha);
    const float2 ub = sum_pair(partials, splits, elements, row_pairs + (half >> 1) + 2 * q + 1, alpha);
    const float inv_gs = 1.0f / (*out_gs);
    uint16_t scratch = 0;
    quant_quad4(tanh_gelu(ga.x) * ua.x * inv_gs, tanh_gelu(ga.y) * ua.y * inv_gs,
                tanh_gelu(gb.x) * ub.x * inv_gs, tanh_gelu(gb.y) * ub.y * inv_gs,
                valid ? out + (static_cast<size_t>(row) * half + 4 * q) / 2 : reinterpret_cast<uint8_t*>(&scratch),
                out_sf + static_cast<size_t>(row) * (half / 16) + (q >> 2), valid && (q & 3) == 0);
}

template <bool PDL>
__global__ void __launch_bounds__(256)
action_in_norm_kernel(const __nv_bfloat16* __restrict__ noise, const __nv_bfloat16* __restrict__ w_in,
                      const __nv_bfloat16* __restrict__ b_in, __nv_bfloat16* __restrict__ x,
                      const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ style,
                      uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
                      __nv_bfloat16* __restrict__ gate_out, const float* __restrict__ out_gs, float eps) {
    constexpr int DIM = 1024, KDIM = 32;
    __shared__ float warp_partial[8];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int n0 = tid * 4;
    const __nv_bfloat16* style_row = style + static_cast<size_t>(row) * 3 * DIM;
    float2 wq[KDIM][2];
#pragma unroll
    for (int k = 0; k < KDIM; ++k) {
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(w_in + k * DIM + n0);
        wq[k][0] = __bfloat1622float2(wp[0]);
        wq[k][1] = __bfloat1622float2(wp[1]);
    }
    const float2 bv0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(b_in + n0)[0]);
    const float2 bv1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(b_in + n0)[1]);
    const float2 wv0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(weight + n0)[0]);
    const float2 wv1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(weight + n0)[1]);
    const float2 s0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + n0)[0]);
    const float2 s1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + n0)[1]);
    const float2 h0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + DIM + n0)[0]);
    const float2 h1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(style_row + DIM + n0)[1]);
    const uint2 gt = *reinterpret_cast<const uint2*>(style_row + 2 * DIM + n0);
    if constexpr (PDL) pdl_prologue();
    float nv[KDIM];
#pragma unroll
    for (int k = 0; k < KDIM; k += 2) {
        const float2 v = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(noise + static_cast<size_t>(row) * KDIM + k));
        nv[k] = v.x; nv[k + 1] = v.y;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < KDIM; ++k) {
        acc[0] += nv[k] * wq[k][0].x; acc[1] += nv[k] * wq[k][0].y;
        acc[2] += nv[k] * wq[k][1].x; acc[3] += nv[k] * wq[k][1].y;
    }
    const float2 g0 = __bfloat1622float2(__floats2bfloat162_rn(acc[0], acc[1]));
    const float2 g1 = __bfloat1622float2(__floats2bfloat162_rn(acc[2], acc[3]));
    const __nv_bfloat162 x0 = __floats2bfloat162_rn(g0.x + bv0.x, g0.y + bv0.y);
    const __nv_bfloat162 x1 = __floats2bfloat162_rn(g1.x + bv1.x, g1.y + bv1.y);
    __nv_bfloat162* xrow = reinterpret_cast<__nv_bfloat162*>(x + static_cast<size_t>(row) * DIM + n0);
    xrow[0] = x0; xrow[1] = x1;
    const float2 xf0 = __bfloat1622float2(x0), xf1 = __bfloat1622float2(x1);
    float sq = xf0.x * xf0.x + xf0.y * xf0.y + xf1.x * xf1.x + xf1.y * xf1.y;
    sq = warp_sum(sq);
    if ((tid & 31) == 0) warp_partial[tid >> 5] = sq;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_partial[w];
    const float rms = rsqrtf(total / DIM + eps);
    const float inv_gs = 1.0f / (*out_gs);
    quant_quad4((xf0.x * rms * wv0.x * (1.0f + s0.x) + h0.x) * inv_gs,
                (xf0.y * rms * wv0.y * (1.0f + s0.y) + h0.y) * inv_gs,
                (xf1.x * rms * wv1.x * (1.0f + s1.x) + h1.x) * inv_gs,
                (xf1.y * rms * wv1.y * (1.0f + s1.y) + h1.y) * inv_gs,
                out + static_cast<size_t>(row) * (DIM / 2) + 2 * tid,
                out_sf + static_cast<size_t>(row) * (DIM / 16) + (tid >> 2), (tid & 3) == 0);
    *reinterpret_cast<uint2*>(gate_out + static_cast<size_t>(row) * DIM + n0) = gt;
}

}  // namespace

int residual_ada_norm(const float* partials, int splits, const float* a_gs, const float* w_gs,
                      __nv_bfloat16* residual, const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                      const __nv_bfloat16* style, uint8_t* out, uint8_t* out_sf, const float* out_gs,
                      __nv_bfloat16* gate_out, int rows, int dim, float eps, bool pdl, cudaStream_t stream) {
    if (dim != 1024 || rows < 1) return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)residual_ada_norm_kernel<true> : (const void*)residual_ada_norm_kernel<false>;
    return launch_ex(fn, dim3(rows), dim3(256), stream, pdl, partials, splits, a_gs, w_gs, residual, gate, weight,
                     style, out, out_sf, out_gs, gate_out, rows, eps);
}

int gate_gelu(const float* partials, int splits, const float* a_gs, const float* w_gs, uint8_t* out,
              uint8_t* out_sf, int rows, int half, const float* out_gs, bool pdl, cudaStream_t stream) {
    if (rows < 1 || half % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const int total = rows * (half >> 2);
    const dim3 grid((total + 255) / 256);
    const void* fn = pdl ? (const void*)gate_gelu_kernel<true> : (const void*)gate_gelu_kernel<false>;
    return launch_ex(fn, grid, dim3(256), stream, pdl, partials, splits, a_gs, w_gs, out, out_sf, rows, half, out_gs);
}

int action_in_norm(const __nv_bfloat16* noise, const __nv_bfloat16* w_in, const __nv_bfloat16* b_in,
                   __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* style, uint8_t* out,
                   uint8_t* out_sf, __nv_bfloat16* gate_out, const float* out_gs, int rows, float eps, bool pdl,
                   cudaStream_t stream) {
    if (rows < 1) return static_cast<int>(cudaErrorInvalidValue);
    const void* fn = pdl ? (const void*)action_in_norm_kernel<true> : (const void*)action_in_norm_kernel<false>;
    return launch_ex(fn, dim3(rows), dim3(256), stream, pdl, noise, w_in, b_in, x, weight, style, out, out_sf,
                     gate_out, out_gs, eps);
}

}  // namespace pi05_dec_skinny_fp4
}  // namespace flash_rt

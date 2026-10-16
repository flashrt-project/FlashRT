// SPDX-License-Identifier: Apache-2.0
// NVFP4 prefill row kernels on sm_120a. See the header.
#include "nvfp4_prefill_sm120.cuh"
#include "gemm/fp4/nvfp4_skinny_common.cuh"

namespace flash_rt {
namespace nvfp4_prefill {
namespace {

using namespace flash_rt::nvfp4_skinny;

// CUTLASS Sm1xxBlockScaledConfig<16> atom: 128 rows x 4 scale columns per 512 bytes.
__device__ __forceinline__ size_t sf_index(int row, int blk, int nb4) {
    return (static_cast<size_t>(row >> 7) * nb4 + (blk >> 2)) * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 +
           (blk & 3);
}

// Quad-cooperative NVFP4 quantization of four consecutive values (already
// divided by the global scale) with the scale written in the swizzled layout.
__device__ __forceinline__ void quant_quad4_swz(float v0, float v1, float v2, float v3, uint8_t* out2,
                                                uint8_t* sf, int row, int blk, int nb4, bool write_sf) {
    float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const uint8_t s = act_scale_code(amax * (1.0f / 6.0f));
    const float d = ue4m3_value(s);
    const float inv = d > 0.f ? 1.0f / d : 0.f;
    *reinterpret_cast<uint16_t*>(out2) = e2m1x4(v0 * inv, v1 * inv, v2 * inv, v3 * inv);
    if (write_sf) sf[sf_index(row, blk, nb4)] = s;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    return warp_sum(lane < nw ? red[lane] : 0.f);
}

__global__ void __launch_bounds__(1024)
norm_quant_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ residual,
                  const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ bias, int mode,
                  float eps, uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf, const float* __restrict__ gs,
                  int dim) {
    __shared__ float red[32];
    const int row = blockIdx.x, c0 = threadIdx.x * 4;
    uint2* xp = reinterpret_cast<uint2*>(x + static_cast<size_t>(row) * dim + c0);
    uint2 xv = *xp;
    if (residual != nullptr) {   // x += residual, BF16 residual stream updated in place
        const uint2 rv = *reinterpret_cast<const uint2*>(residual + static_cast<size_t>(row) * dim + c0);
        const float2 a = bf2(xv.x), b = bf2(xv.y), ra = bf2(rv.x), rb = bf2(rv.y);
        xv = make_uint2(pack_bf2(a.x + ra.x, a.y + ra.y), pack_bf2(b.x + rb.x, b.y + rb.y));
        *xp = xv;
    }
    const float2 a = bf2(xv.x), b = bf2(xv.y);
    float v[4] = {a.x, a.y, b.x, b.y};
    if (mode == 0) {
        const float mean = block_sum(v[0] + v[1] + v[2] + v[3], red) / dim;
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] -= mean;
    }
    const float inv = rsqrtf(block_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3], red) / dim + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= inv;
    if (weight != nullptr) {
        const uint2 wv = *reinterpret_cast<const uint2*>(weight + c0);
        const float2 w0 = bf2(wv.x), w1 = bf2(wv.y);
        v[0] *= w0.x; v[1] *= w0.y; v[2] *= w1.x; v[3] *= w1.y;
    }
    if (bias != nullptr) {
        const uint2 bv = *reinterpret_cast<const uint2*>(bias + c0);
        const float2 b0 = bf2(bv.x), b1 = bf2(bv.y);
        v[0] += b0.x; v[1] += b0.y; v[2] += b1.x; v[3] += b1.y;
    }
    const float ig = 1.0f / (*gs);
    quant_quad4_swz(v[0] * ig, v[1] * ig, v[2] * ig, v[3] * ig, out + (static_cast<size_t>(row) * dim + c0) / 2,
                    out_sf, row, c0 / 16, (dim / 16 + 3) / 4, (threadIdx.x & 3) == 0);
}

__global__ void __launch_bounds__(256)
quant_kernel(const __nv_bfloat16* __restrict__ x, int ld, uint8_t* __restrict__ out, uint8_t* __restrict__ out_sf,
             const float* __restrict__ gs, int cols) {
    const int row = blockIdx.x;
    const float ig = 1.0f / (*gs);
    const int nb4 = (cols / 16 + 3) / 4;
    // every quad of lanes owns one 16-element block; cols % 128 == 0 keeps whole warps in range
    for (int c0 = threadIdx.x * 4; c0 < cols; c0 += blockDim.x * 4) {
        const uint2 xv = *reinterpret_cast<const uint2*>(x + static_cast<size_t>(row) * ld + c0);
        const float2 a = bf2(xv.x), b = bf2(xv.y);
        quant_quad4_swz(a.x * ig, a.y * ig, b.x * ig, b.y * ig, out + (static_cast<size_t>(row) * cols + c0) / 2,
                        out_sf, row, c0 / 16, nb4, (threadIdx.x & 3) == 0);
    }
}

// One CTA per (row, head); hd / 2 threads, thread i owns the pair (i, i + hd / 2).
__global__ void qk_norm_rope_kernel(__nv_bfloat16* __restrict__ qkv, int ld, int q_off, int k_off, int n_q, int hd,
                                    const float* __restrict__ cos, const float* __restrict__ sin,
                                    const __nv_bfloat16* __restrict__ q_norm,
                                    const __nv_bfloat16* __restrict__ k_norm, float eps) {
    __shared__ float red[32];
    const int s = blockIdx.x, head = blockIdx.y, i = threadIdx.x, h = hd >> 1;
    const bool is_q = head < n_q;
    __nv_bfloat16* p = qkv + static_cast<size_t>(s) * ld + (is_q ? q_off + head * hd : k_off + (head - n_q) * hd);
    float x1 = __bfloat162float(p[i]), x2 = __bfloat162float(p[i + h]);
    const __nv_bfloat16* nw = is_q ? q_norm : k_norm;
    if (nw != nullptr) {
        const float r = rsqrtf(block_sum(x1 * x1 + x2 * x2, red) / hd + eps);
        x1 *= r * __bfloat162float(nw[i]);
        x2 *= r * __bfloat162float(nw[i + h]);
    }
    // rotate-half tables repeat their first half; both outputs use entry i
    const float c = cos[static_cast<size_t>(s) * hd + i];
    const float sn = sin[static_cast<size_t>(s) * hd + i];
    p[i] = __float2bfloat16(x1 * c - x2 * sn);
    p[i + h] = __float2bfloat16(x2 * c + x1 * sn);
}

__global__ void scatter_add_rows_kernel(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ src,
                                        const int64_t* __restrict__ idx, int dim) {
    const int r = blockIdx.x;
    const int c = threadIdx.x * 8;
    if (c >= dim) return;
    uint4* dst = reinterpret_cast<uint4*>(h + static_cast<size_t>(idx[r]) * dim + c);
    const uint4 a = *dst;
    const uint4 b = *reinterpret_cast<const uint4*>(src + static_cast<size_t>(r) * dim + c);
    const uint32_t* av = reinterpret_cast<const uint32_t*>(&a);
    const uint32_t* bv = reinterpret_cast<const uint32_t*>(&b);
    uint4 o;
    uint32_t* ov = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 x = bf2(av[k]), y = bf2(bv[k]);
        ov[k] = pack_bf2(x.x + y.x, x.y + y.y);
    }
    *dst = o;
}

}  // namespace

size_t sf_bytes(int rows, int cols) {
    return static_cast<size_t>((rows + 127) / 128) * ((cols / 16 + 3) / 4) * 512;
}

int norm_quant(__nv_bfloat16* x, const __nv_bfloat16* residual, const __nv_bfloat16* weight,
               const __nv_bfloat16* bias, int mode, float eps, uint8_t* out, uint8_t* out_sf, const float* gs,
               int rows, int dim, cudaStream_t stream) {
    if (rows < 1 || dim % 128 != 0 || dim > 4096 || (mode != 0 && mode != 1) || (mode == 1 && bias != nullptr))
        return static_cast<int>(cudaErrorInvalidValue);
    norm_quant_kernel<<<rows, dim / 4, 0, stream>>>(x, residual, weight, bias, mode, eps, out, out_sf, gs, dim);
    return static_cast<int>(cudaGetLastError());
}

int quant(const __nv_bfloat16* x, int ld, uint8_t* out, uint8_t* out_sf, const float* gs, int rows, int cols,
          cudaStream_t stream) {
    if (rows < 1 || cols % 128 != 0 || ld < cols || ld % 4 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const int threads = cols / 4 < 256 ? cols / 4 : 256;
    quant_kernel<<<rows, threads, 0, stream>>>(x, ld, out, out_sf, gs, cols);
    return static_cast<int>(cudaGetLastError());
}

int qk_norm_rope(__nv_bfloat16* qkv, int ld, int q_off, int k_off, int n_q, int n_k, int hd,
                 const float* cos, const float* sin, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                 float eps, int S, cudaStream_t stream) {
    if (S < 1 || hd % 64 != 0 || hd > 256 || n_q < 0 || n_k < 0 || n_q + n_k < 1)
        return static_cast<int>(cudaErrorInvalidValue);
    qk_norm_rope_kernel<<<dim3(S, n_q + n_k), hd / 2, 0, stream>>>(qkv, ld, q_off, k_off, n_q, hd, cos, sin, q_norm,
                                                                   k_norm, eps);
    return static_cast<int>(cudaGetLastError());
}

int scatter_add_rows(__nv_bfloat16* h, const __nv_bfloat16* src, const int64_t* idx, int n, int dim,
                     cudaStream_t stream) {
    if (n < 0 || dim % 8 != 0 || dim / 8 > 1024) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    scatter_add_rows_kernel<<<n, dim / 8, 0, stream>>>(h, src, idx, dim);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace nvfp4_prefill
}  // namespace flash_rt

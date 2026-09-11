// Fused adaLN kernels for the pi0.5 action expert (Thor).
//
// The ggml graph expresses each adaLN application as rms_norm + two
// broadcast repeats + mul + two adds (and the gated residual as repeat +
// mul + add), all on [M, C] tensors with M ~ 10. These kernels collapse
// each chain into one launch. The scale/shift/gate vectors are passed as
// direct pointers (the ggml view tensors' data pointers, which already
// include their byte offsets into the modulation vector).

#include "fr_kernels.h"

#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace ggml_cuda_flashrt {

namespace {

// out[m, c] = norm(x[m])[c] * (1 + scale[c]) + shift[c]
// where norm = rms-normalize when with_rms, identity otherwise.
template <bool with_rms>
__global__ void kernel_ada_rms(const float * __restrict__ x,
                               const float * __restrict__ scale,
                               const float * __restrict__ shift,
                               float * __restrict__ out,
                               int C, float eps) {
    const int m = blockIdx.x;
    const float * xr = x + (int64_t) m * C;
    float * orow = out + (int64_t) m * C;

    float inv_rms = 1.0f;
    if (with_rms) {
        float sumsq = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = xr[c];
            sumsq += v * v;
        }
        __shared__ float red[32];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
        }
        const int warp = threadIdx.x / 32;
        if (threadIdx.x % 32 == 0) {
            red[warp] = sumsq;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (threadIdx.x == 0) {
                red[0] = v;
            }
        }
        __syncthreads();
        inv_rms = rsqrtf(red[0] / C + eps);
    }

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float n = with_rms ? xr[c] * inv_rms : xr[c];
        orow[c] = n * (1.0f + scale[c]) + shift[c];
    }
}

using AdaCfg = cutlass::detail::Sm1xxBlockScaledConfig<16>;

__device__ __forceinline__ uint8_t ada_f32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t m;
    if      (ax <= 0.25f) m = 0u;
    else if (ax <= 0.75f) m = 1u;
    else if (ax <= 1.25f) m = 2u;
    else if (ax <= 1.75f) m = 3u;
    else if (ax <= 2.5f)  m = 4u;
    else if (ax <= 3.5f)  m = 5u;
    else if (ax <= 5.0f)  m = 6u;
    else                  m = 7u;
    return sign | m;
}

// Quantize one 16-element block held in registers by 16 consecutive threads?
// Simpler: each thread quantizes one 16-element block it re-reads from the
// just-written f32 output row (L2-hot), writing packed bytes + one SF byte.
template <class LayoutSF>
__device__ __forceinline__ void ada_quant_row(const float * __restrict__ orow,
                                              uint8_t * __restrict__ dst_packed,
                                              uint8_t * __restrict__ dst_sfa,
                                              LayoutSF layout,
                                              int m, int C) {
    const int n_blocks = C / 16;
    for (int blk = threadIdx.x; blk < n_blocks; blk += blockDim.x) {
        const float * v = orow + blk * 16;
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(v[i]));
        }
        float desired = amax / 6.f;
        if (desired < 1e-12f) desired = 1e-12f;
        __nv_fp8_e4m3 q(desired);
        dst_sfa[layout(m, blk * 16, 0)] = *reinterpret_cast<const uint8_t *>(&q);
        const float inv = 1.f / static_cast<float>(q);
        uint2 out;
        uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
        #pragma unroll
        for (int pIdx = 0; pIdx < 8; ++pIdx) {
            const uint8_t lo = ada_f32_to_e2m1(v[2 * pIdx]     * inv);
            const uint8_t hi = ada_f32_to_e2m1(v[2 * pIdx + 1] * inv);
            ob[pIdx] = static_cast<uint8_t>(lo | (hi << 4));
        }
        reinterpret_cast<uint2 *>(dst_packed)[(int64_t) m * n_blocks + blk] = out;
    }
}

// Fused adaLN modulate + NVFP4 quantize of the result.
template <bool with_rms, class LayoutSF>
__global__ void kernel_ada_rms_q(const float * __restrict__ x,
                                 const float * __restrict__ scale,
                                 const float * __restrict__ shift,
                                 float * __restrict__ out,
                                 uint8_t * __restrict__ dst_packed,
                                 uint8_t * __restrict__ dst_sfa,
                                 LayoutSF layout,
                                 int C, float eps) {
    const int m = blockIdx.x;
    const float * xr = x + (int64_t) m * C;
    float * orow = out + (int64_t) m * C;

    float inv_rms = 1.0f;
    if (with_rms) {
        float sumsq = 0.0f;
        for (int c = threadIdx.x; c < C; c += blockDim.x) {
            const float v = xr[c];
            sumsq += v * v;
        }
        __shared__ float red[32];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
        }
        const int warp = threadIdx.x / 32;
        if (threadIdx.x % 32 == 0) {
            red[warp] = sumsq;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (threadIdx.x < blockDim.x / 32) ? red[threadIdx.x] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (threadIdx.x == 0) {
                red[0] = v;
            }
        }
        __syncthreads();
        inv_rms = rsqrtf(red[0] / C + eps);
    }

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float n = with_rms ? xr[c] * inv_rms : xr[c];
        orow[c] = n * (1.0f + scale[c]) + shift[c];
    }
    __syncthreads();
    ada_quant_row(orow, dst_packed, dst_sfa, layout, m, C);
}

// Fused LayerNorm + affine + NVFP4 quantize of the result.
template <class LayoutSF>
__global__ void kernel_layer_norm_affine_q(const float * __restrict__ x,
                                           const float * __restrict__ w,
                                           const float * __restrict__ b,
                                           float * __restrict__ out,
                                           uint8_t * __restrict__ dst_packed,
                                           uint8_t * __restrict__ dst_sfa,
                                           LayoutSF layout,
                                           int C, float eps) {
    const int m = blockIdx.x;
    const float * xr = x + (int64_t) m * C;
    float * orow = out + (int64_t) m * C;

    float sum = 0.0f, sumsq = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float v = xr[c];
        sum   += v;
        sumsq += v * v;
    }
    __shared__ float red[2][32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum   += __shfl_xor_sync(0xffffffff, sum, off);
        sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
    }
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        red[0][warp] = sum;
        red[1][warp] = sumsq;
    }
    __syncthreads();
    if (warp == 0) {
        float s  = (threadIdx.x < blockDim.x / 32) ? red[0][threadIdx.x] : 0.0f;
        float s2 = (threadIdx.x < blockDim.x / 32) ? red[1][threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s  += __shfl_xor_sync(0xffffffff, s, off);
            s2 += __shfl_xor_sync(0xffffffff, s2, off);
        }
        if (threadIdx.x == 0) {
            red[0][0] = s;
            red[1][0] = s2;
        }
    }
    __syncthreads();
    const float mean = red[0][0] / C;
    const float var  = red[1][0] / C - mean * mean;
    const float rstd = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        orow[c] = (xr[c] - mean) * rstd * w[c] + b[c];
    }
    __syncthreads();
    ada_quant_row(orow, dst_packed, dst_sfa, layout, m, C);
}

// out[m, c] = residual[m, c] + branch[m, c] * gate[c]
__global__ void kernel_gated_residual(const float * __restrict__ residual,
                                      const float * __restrict__ branch,
                                      const float * __restrict__ gate,
                                      float * __restrict__ out,
                                      int C) {
    const int m = blockIdx.x;
    const int64_t off = (int64_t) m * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        out[off + c] = residual[off + c] + branch[off + c] * gate[c];
    }
}

// out[m, c] = (x[m, c] - mean(x[m])) * rstd(x[m]) * w[c] + b[c]
__global__ void kernel_layer_norm_affine(const float * __restrict__ x,
                                         const float * __restrict__ w,
                                         const float * __restrict__ b,
                                         float * __restrict__ out,
                                         int C, float eps) {
    const int m = blockIdx.x;
    const float * xr = x + (int64_t) m * C;
    float * orow = out + (int64_t) m * C;

    float sum = 0.0f, sumsq = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float v = xr[c];
        sum   += v;
        sumsq += v * v;
    }
    __shared__ float red[2][32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum   += __shfl_xor_sync(0xffffffff, sum, off);
        sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
    }
    const int warp = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        red[0][warp] = sum;
        red[1][warp] = sumsq;
    }
    __syncthreads();
    if (warp == 0) {
        float s  = (threadIdx.x < blockDim.x / 32) ? red[0][threadIdx.x] : 0.0f;
        float s2 = (threadIdx.x < blockDim.x / 32) ? red[1][threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s  += __shfl_xor_sync(0xffffffff, s, off);
            s2 += __shfl_xor_sync(0xffffffff, s2, off);
        }
        if (threadIdx.x == 0) {
            red[0][0] = s;
            red[1][0] = s2;
        }
    }
    __syncthreads();
    const float mean = red[0][0] / C;
    const float var  = red[1][0] / C - mean * mean;
    const float rstd = rsqrtf(var + eps);

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        orow[c] = (xr[c] - mean) * rstd * w[c] + b[c];
    }
}

// out[c] = a[c] + b[c]
__global__ void kernel_vec_add(const float * __restrict__ a,
                               const float * __restrict__ b,
                               float * __restrict__ out,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

} // namespace

int ada_rms_mod(const float * x, const float * scale, const float * shift,
                float * out, int M, int C, float eps, bool with_rms,
                cudaStream_t stream) {
    const int threads = 256;
    if (with_rms) {
        kernel_ada_rms<true><<<M, threads, 0, stream>>>(x, scale, shift, out, C, eps);
    } else {
        kernel_ada_rms<false><<<M, threads, 0, stream>>>(x, scale, shift, out, C, eps);
    }
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int gated_residual(const float * residual, const float * branch, const float * gate,
                   float * out, int M, int C, cudaStream_t stream) {
    kernel_gated_residual<<<M, 256, 0, stream>>>(residual, branch, gate, out, C);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int layer_norm_affine(const float * x, const float * w, const float * b,
                      float * out, int M, int C, float eps, cudaStream_t stream) {
    kernel_layer_norm_affine<<<M, 256, 0, stream>>>(x, w, b, out, C, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int ada_rms_mod_quant(const float * x, const float * scale, const float * shift,
                      float * out, void * dst_packed, void * dst_sfa,
                      int M, int C, float eps, bool with_rms, cudaStream_t stream) {
    if (C % 16 != 0) return -1;
    auto shape = cute::make_shape(M, 1, C, 1);
    auto layout = AdaCfg::tile_atom_to_shape_SFA(shape);
    if (with_rms) {
        kernel_ada_rms_q<true><<<M, 256, 0, stream>>>(x, scale, shift, out,
            (uint8_t *) dst_packed, (uint8_t *) dst_sfa, layout, C, eps);
    } else {
        kernel_ada_rms_q<false><<<M, 256, 0, stream>>>(x, scale, shift, out,
            (uint8_t *) dst_packed, (uint8_t *) dst_sfa, layout, C, eps);
    }
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int layer_norm_affine_quant(const float * x, const float * w, const float * b,
                            float * out, void * dst_packed, void * dst_sfa,
                            int M, int C, float eps, cudaStream_t stream) {
    if (C % 16 != 0) return -1;
    auto shape = cute::make_shape(M, 1, C, 1);
    auto layout = AdaCfg::tile_atom_to_shape_SFA(shape);
    kernel_layer_norm_affine_q<<<M, 256, 0, stream>>>(x, w, b, out,
        (uint8_t *) dst_packed, (uint8_t *) dst_sfa, layout, C, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int vec_add_f32(const float * a, const float * b, float * out, int n, cudaStream_t stream) {
    kernel_vec_add<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt

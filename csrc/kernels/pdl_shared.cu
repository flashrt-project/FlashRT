// FlashRT — opt-in PDL-aware shared elementwise/norm entries.
//
// The arithmetic is the same as the existing add_bias_bf16, residual_add and
// layer_norm entries; only the launch is PDL-aware and the reductions that
// need back-to-back shared[0] use block_reduce_*_safe.  The existing symbols
// are deliberately untouched so other models keep their current launch and
// build behavior.
#include "pdl_shared.cuh"
#include "pdl_switch.cuh"
#include "block_reduce_safe.cuh"
#include "common.cuh"

#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ── add_bias: x[i] += b[i % D] ─────────────────────────────────────────────
__global__ void add_bias_bf16_pdl_kernel(__nv_bfloat16* x,
                                         const __nv_bfloat16* b,
                                         int S, int D) {
    flashrt_pdl_wait_and_trigger();
    const int n = S * D;
    if ((D & 3) == 0) {
        const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
        if (i4 + 3 < n) {
            const int j = i4 % D;
            const __nv_bfloat162 x0 = *reinterpret_cast<__nv_bfloat162*>(x + i4);
            const __nv_bfloat162 x1 = *reinterpret_cast<__nv_bfloat162*>(x + i4 + 2);
            const __nv_bfloat162 b0 = *reinterpret_cast<const __nv_bfloat162*>(b + j);
            const __nv_bfloat162 b1 = *reinterpret_cast<const __nv_bfloat162*>(b + j + 2);
            *reinterpret_cast<__nv_bfloat162*>(x + i4) = __nv_bfloat162(
                __float2bfloat16(__bfloat162float(x0.x) + __bfloat162float(b0.x)),
                __float2bfloat16(__bfloat162float(x0.y) + __bfloat162float(b0.y)));
            *reinterpret_cast<__nv_bfloat162*>(x + i4 + 2) = __nv_bfloat162(
                __float2bfloat16(__bfloat162float(x1.x) + __bfloat162float(b1.x)),
                __float2bfloat16(__bfloat162float(x1.y) + __bfloat162float(b1.y)));
            return;
        }
        for (int i = i4; i < n; ++i) {
            x[i] = __float2bfloat16(__bfloat162float(x[i]) +
                                    __bfloat162float(b[i % D]));
        }
        return;
    }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = __float2bfloat16(__bfloat162float(x[i]) +
                                __bfloat162float(b[i % D]));
    }
}

void add_bias_bf16_pdl(__nv_bfloat16* x, const __nv_bfloat16* b,
                       int S, int D, cudaStream_t stream) {
    const int n = S * D;
    const dim3 grid = (D & 3) == 0 ? dim3((n / 4 + 255) / 256)
                                   : dim3((n + 255) / 256);
    flash_rt::pdl::launch(add_bias_bf16_pdl_kernel, grid, dim3(256), 0,
                          stream, x, b, S, D);
}

// ── residual_add: residual += x ───────────────────────────────────────────
template <typename T>
__global__ void residual_add_pdl_kernel(T* __restrict__ residual,
                                        const T* __restrict__ x, int n) {
    flashrt_pdl_wait_and_trigger();
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual);
    const T2* x2 = reinterpret_cast<const T2*>(x);
    const int n4 = n >> 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        const size_t o = static_cast<size_t>(idx) * 2;
        const T2 r0 = res2[o], r1 = res2[o + 1];
        const T2 x0 = x2[o], x1 = x2[o + 1];
        res2[o] = make_packed2<T>(from_f32<T>(to_f32(r0.x) + to_f32(x0.x)),
                                  from_f32<T>(to_f32(r0.y) + to_f32(x0.y)));
        res2[o + 1] = make_packed2<T>(from_f32<T>(to_f32(r1.x) + to_f32(x1.x)),
                                      from_f32<T>(to_f32(r1.y) + to_f32(x1.y)));
    }
    if (idx == 0 && (n & 3)) {
        for (int i = n & ~3; i < n; ++i) {
            residual[i] = from_f32<T>(to_f32(residual[i]) + to_f32(x[i]));
        }
    }
}

void residual_add_pdl(__nv_bfloat16* residual, const __nv_bfloat16* x,
                      int n, cudaStream_t stream) {
    const int n4 = n >> 2;
    const int blocks = n4 > 0 ? n4 : 1;
    flash_rt::pdl::launch(residual_add_pdl_kernel<__nv_bfloat16>,
                          dim3((blocks + 255) / 256), dim3(256), 0, stream,
                          residual, x, n);
}

// ── layer_norm ────────────────────────────────────────────────────────────
template <typename T>
__global__ void layer_norm_pdl_kernel(const T* __restrict__ x,
                                      const T* __restrict__ weight,
                                      const T* __restrict__ bias,
                                      T* __restrict__ out,
                                      int dim, float eps) {
    flashrt_pdl_wait_and_trigger();
    using T2 = typename packed2<T>::type;
    const int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    const int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const T2 val = x2[i];
        local_sum += to_f32(val.x) + to_f32(val.y);
    }
    const float mean = block_reduce_sum_safe(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const T2 val = x2[i];
        const float d0 = to_f32(val.x) - mean;
        const float d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    const float inv_std =
        rsqrtf(block_reduce_sum_safe(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const T2 xv = x2[i], wv = w2[i], bv = b2[i];
        const float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) +
                         to_f32(bv.x);
        const float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) +
                         to_f32(bv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

void layer_norm_pdl(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    const __nv_bfloat16* bias, __nv_bfloat16* out,
                    int seq_len, int dim, float eps, cudaStream_t stream) {
    flash_rt::pdl::launch(layer_norm_pdl_kernel<__nv_bfloat16>,
                          dim3(seq_len), dim3(256), 256 * sizeof(float),
                          stream, x, weight, bias, out, dim, eps);
}

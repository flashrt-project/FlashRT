// DiT bf16 helpers — Phase 5a-2 (R0: pure additions, no edits to existing
// fp16 kernels). Mirrors layer_norm_no_affine_fp16 / ada_layer_norm_fp16 /
// add_bias_fp16 but with __nv_bfloat162 vectorized math, plus cast helpers
// for fp16 ↔ bf16 boundaries.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

// ────────────────────────────────────────────────────────────────────
// LayerNorm (no affine) — bf16
// ────────────────────────────────────────────────────────────────────
__global__ void layer_norm_no_affine_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                                   __nv_bfloat16* __restrict__ out,
                                                   int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + row * dim);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 val = x2[i];
        local_sum += __bfloat162float(val.x) + __bfloat162float(val.y);
    }
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads();
    if (!threadIdx.x) shared[0] = val;
    __syncthreads();
    float mean = shared[0] / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 v = x2[i];
        float d0 = __bfloat162float(v.x) - mean, d1 = __bfloat162float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    val = local_var;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads();
    if (!threadIdx.x) shared[0] = val;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 xv = x2[i];
        float v0 = (__bfloat162float(xv.x) - mean) * inv_std;
        float v1 = (__bfloat162float(xv.y) - mean) * inv_std;
        out2[i] = __floats2bfloat162_rn(v0, v1);
    }
}

void layer_norm_no_affine_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                                int seq_len, int dim, float eps,
                                cudaStream_t stream) {
    layer_norm_no_affine_bf16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, out, dim, eps);
}

__global__ void layer_norm_no_affine_fp8_static_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ d_scale,
    int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat162* x2 =
        reinterpret_cast<const __nv_bfloat162*>(x + (long long)row * dim);
    __nv_fp8_e4m3* out_row = out + (long long)row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 val = x2[i];
        local_sum += __bfloat162float(val.x) + __bfloat162float(val.y);
    }
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) {
        val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    }
    __syncthreads();
    if (!threadIdx.x) shared[0] = val;
    __syncthreads();
    float mean = shared[0] / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 v = x2[i];
        float d0 = __bfloat162float(v.x) - mean;
        float d1 = __bfloat162float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    val = local_var;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) {
        val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    }
    __syncthreads();
    if (!threadIdx.x) shared[0] = val;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);
    float inv_a = 1.0f / fmaxf(*d_scale, 1e-12f);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 xv = x2[i];
        float v0 = (__bfloat162float(xv.x) - mean) * inv_std;
        float v1 = (__bfloat162float(xv.y) - mean) * inv_std;
        v0 = __bfloat162float(__float2bfloat16(v0)) * inv_a;
        v1 = __bfloat162float(__float2bfloat16(v1)) * inv_a;
        __nv_fp8_e4m3 pair[2];
        pair[0] = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        pair[1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
        *reinterpret_cast<uint16_t*>(out_row + 2 * i) =
            *reinterpret_cast<uint16_t*>(pair);
    }
}

void layer_norm_no_affine_fp8_static_bf16(
    const __nv_bfloat16* x, __nv_fp8_e4m3* out, const float* d_scale,
    int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_no_affine_fp8_static_bf16_kernel
        <<<seq_len, 256, 256 * sizeof(float), stream>>>(
            x, out, d_scale, dim, eps);
}

// ────────────────────────────────────────────────────────────────────
// AdaLayerNorm — bf16
// out = LN(x, no_affine) * (1 + scale) + shift
// ────────────────────────────────────────────────────────────────────
__global__ void ada_layer_norm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ scale,
                                            const __nv_bfloat16* __restrict__ shift,
                                            __nv_bfloat16* __restrict__ out,
                                            int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + row * dim);
    const __nv_bfloat162* sc2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* sh2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 val = x2[i];
        local_sum += __bfloat162float(val.x) + __bfloat162float(val.y);
    }
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float mean = shared[0] / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 v = x2[i];
        float d0 = __bfloat162float(v.x) - mean, d1 = __bfloat162float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    val = local_var;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i];
        float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
        float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
        float v0 = n0 * (1.0f + __bfloat162float(sv.x)) + __bfloat162float(hv.x);
        float v1 = n1 * (1.0f + __bfloat162float(sv.y)) + __bfloat162float(hv.y);
        out2[i] = __floats2bfloat162_rn(v0, v1);
    }
}

void ada_layer_norm_bf16(const __nv_bfloat16* x,
                          const __nv_bfloat16* scale, const __nv_bfloat16* shift,
                          __nv_bfloat16* out, int seq_len, int dim, float eps,
                          cudaStream_t stream) {
    ada_layer_norm_bf16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, scale, shift, out, dim, eps);
}

// Per-row AdaLayerNorm — bf16
// out[row] = LN(x[row], no_affine) * (1 + scale[row]) + shift[row]
__global__ void ada_layer_norm_bf16_per_token_kernel(
                                            const __nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ scale,
                                            const __nv_bfloat16* __restrict__ shift,
                                            __nv_bfloat16* __restrict__ out,
                                            int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + row * dim);
    const __nv_bfloat162* sc2 = reinterpret_cast<const __nv_bfloat162*>(scale + row * dim);
    const __nv_bfloat162* sh2 = reinterpret_cast<const __nv_bfloat162*>(shift + row * dim);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 val = x2[i];
        local_sum += __bfloat162float(val.x) + __bfloat162float(val.y);
    }
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float mean = shared[0] / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 v = x2[i];
        float d0 = __bfloat162float(v.x) - mean, d1 = __bfloat162float(v.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    val = local_var;
    for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o);
    if (!lane) shared[wid] = val;
    __syncthreads();
    if (!wid) { val = (lane < (blockDim.x >> 5)) ? shared[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xffffffff, val, o); }
    __syncthreads(); if (!threadIdx.x) shared[0] = val; __syncthreads();
    float inv_std = rsqrtf(shared[0] / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __nv_bfloat162 xv = x2[i], sv = sc2[i], hv = sh2[i];
        float n0 = (__bfloat162float(xv.x) - mean) * inv_std;
        float n1 = (__bfloat162float(xv.y) - mean) * inv_std;
        float v0 = n0 * (1.0f + __bfloat162float(sv.x)) + __bfloat162float(hv.x);
        float v1 = n1 * (1.0f + __bfloat162float(sv.y)) + __bfloat162float(hv.y);
        out2[i] = __floats2bfloat162_rn(v0, v1);
    }
}

void ada_layer_norm_bf16_per_token(
                          const __nv_bfloat16* x,
                          const __nv_bfloat16* scale, const __nv_bfloat16* shift,
                          __nv_bfloat16* out, int seq_len, int dim, float eps,
                          cudaStream_t stream) {
    ada_layer_norm_bf16_per_token_kernel
        <<<seq_len, 256, 256 * sizeof(float), stream>>>(
            x, scale, shift, out, dim, eps);
}

// ────────────────────────────────────────────────────────────────────
// Bias add: x[i] += b[i % D] — bf16
// ────────────────────────────────────────────────────────────────────
__global__ void add_bias_bf16_kernel(__nv_bfloat16* x, const __nv_bfloat16* b,
                                      int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * D) {
        x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(b[i % D]));
    }
}

void add_bias_bf16(__nv_bfloat16* x, const __nv_bfloat16* b,
                    int S, int D, cudaStream_t stream) {
    add_bias_bf16_kernel<<<(S * D + 255) / 256, 256, 0, stream>>>(x, b, S, D);
}

// ────────────────────────────────────────────────────────────────────
// Cast fp16 ↔ bf16 (DiT input/output boundary)
// ────────────────────────────────────────────────────────────────────
__global__ void cast_fp16_to_bf16_kernel(const __half* in, __nv_bfloat16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__half2float(in[i]));
}

void cast_fp16_to_bf16(const __half* in, __nv_bfloat16* out, int n,
                        cudaStream_t stream) {
    cast_fp16_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(in, out, n);
}

__global__ void cast_bf16_to_fp16_kernel(const __nv_bfloat16* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__bfloat162float(in[i]));
}

void cast_bf16_to_fp16(const __nv_bfloat16* in, __half* out, int n,
                        cudaStream_t stream) {
    cast_bf16_to_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(in, out, n);
}

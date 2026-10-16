// ================================================================
// FlashRT — Normalization kernels (dtype-generic)
// RMSNorm, LayerNorm, AdaRMSNorm
// Supports: __half (FP16), __nv_bfloat16 (BF16) via templates
// ================================================================

#include "norm.cuh"
#include "common.cuh"
#include <cuda_fp8.h>
#include <stdexcept>

// ── RMSNorm ──
template<typename T>
__global__ void rms_norm_kernel(const T* __restrict__ x,
                                const T* __restrict__ weight,
                                T* __restrict__ out,
                                int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

// Explicit instantiation
template __global__ void rms_norm_kernel<__half>(const __half*, const __half*, __half*, int, float);
template __global__ void rms_norm_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, float);

void rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
              __nv_bfloat16* out, int seq_len, int dim, float eps,
              cudaStream_t stream) {
    rms_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}
void rms_norm_fp16(const __half* x, const __half* weight,
                    __half* out, int seq_len, int dim, float eps,
                    cudaStream_t stream) {
    rms_norm_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps);
}

void rms_norm_inplace(const __nv_bfloat16* weight,
                      __nv_bfloat16* x, int seq_len, int dim, float eps,
                      cudaStream_t stream) {
    rms_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, x, dim, eps);
}

__global__ void bias_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int dim, float eps) {
    using T2 = typename packed2<__nv_bfloat16>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], bv = b2[i];
        __nv_bfloat16 xb0_b = __float2bfloat16(
            __bfloat162float(xv.x) + __bfloat162float(bv.x));
        __nv_bfloat16 xb1_b = __float2bfloat16(
            __bfloat162float(xv.y) + __bfloat162float(bv.y));
        float xb0 = __bfloat162float(xb0_b);
        float xb1 = __bfloat162float(xb1_b);
        local_sum += xb0 * xb0 + xb1 * xb1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], bv = b2[i], wv = w2[i];
        __nv_bfloat16 xb0_b = __float2bfloat16(
            __bfloat162float(xv.x) + __bfloat162float(bv.x));
        __nv_bfloat16 xb1_b = __float2bfloat16(
            __bfloat162float(xv.y) + __bfloat162float(bv.y));
        float v0 = __bfloat162float(xb0_b) * rms * __bfloat162float(wv.x);
        float v1 = __bfloat162float(xb1_b) * rms * __bfloat162float(wv.y);
        out2[i] = __halves2bfloat162(__float2bfloat16(v0),
                                     __float2bfloat16(v1));
    }
}

void bias_rms_norm_bf16(const __nv_bfloat16* x,
                        const __nv_bfloat16* bias,
                        const __nv_bfloat16* weight,
                        __nv_bfloat16* out,
                        int seq_len, int dim, float eps,
                        cudaStream_t stream) {
    bias_rms_norm_bf16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, bias, weight, out, dim, eps);
}

// ── LayerNorm ──
template<typename T>
__global__ void layer_norm_kernel(const T* __restrict__ x,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ bias,
                                  T* __restrict__ out,
                                  int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        local_sum += to_f32(val.x) + to_f32(val.y);
    }
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i], bv = b2[i];
        float v0 = (to_f32(xv.x) - mean) * inv_std * to_f32(wv.x) + to_f32(bv.x);
        float v1 = (to_f32(xv.y) - mean) * inv_std * to_f32(wv.y) + to_f32(bv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

template __global__ void layer_norm_kernel<__half>(const __half*, const __half*, const __half*, __half*, int, float);
template __global__ void layer_norm_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, float);

void layer_norm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                const __nv_bfloat16* bias, __nv_bfloat16* out,
                int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, bias, out, dim, eps);
}
void layer_norm_fp16(const __half* x, const __half* weight,
                      const __half* bias, __half* out,
                      int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, bias, out, dim, eps);
}

// ── LayerNorm → FP8 (fused, matches pi05 fused_layernorm_fp8) ──
// FP16 specialization.
__global__ void layer_norm_fp8_kernel_fp16(const __half* in, __nv_fp8_e4m3* out, const __half* gamma,
                                    const __half* beta, int R, int C, float eps) {
    int r=blockIdx.x; if(r>=R)return;
    const __half*row=in+r*C; __nv_fp8_e4m3*orow=out+r*C;
    float sum=0;
    for(int i=threadIdx.x;i<C;i+=blockDim.x) sum+=__half2float(row[i]);
    __shared__ float sh[32]; int l=threadIdx.x%32,w=threadIdx.x/32;
    for(int o=16;o>0;o>>=1) sum+=__shfl_xor_sync(0xffffffff,sum,o);
    if(!l)sh[w]=sum;__syncthreads();
    if(!w){sum=(l<(blockDim.x+31)/32)?sh[l]:0;for(int o=16;o>0;o>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,o);}
    __syncthreads();if(!threadIdx.x)sh[0]=sum;__syncthreads();
    float mean=sh[0]/C;
    float var=0;
    for(int i=threadIdx.x;i<C;i+=blockDim.x){float v=__half2float(row[i])-mean;var+=v*v;}
    for(int o=16;o>0;o>>=1)var+=__shfl_xor_sync(0xffffffff,var,o);
    if(!l)sh[w]=var;__syncthreads();
    if(!w){var=(l<(blockDim.x+31)/32)?sh[l]:0;for(int o=16;o>0;o>>=1)var+=__shfl_xor_sync(0xffffffff,var,o);}
    __syncthreads();if(!threadIdx.x)sh[0]=var;__syncthreads();
    float rstd=rsqrtf(sh[0]/C+eps);
    for(int i=threadIdx.x;i<C;i+=blockDim.x){
        float v=(__half2float(row[i])-mean)*rstd;
        float normed = v*__half2float(gamma[i])+__half2float(beta[i]);
        orow[i]=__nv_fp8_e4m3(normed);
    }
}

// BF16 generic version
template<typename T>
__global__ void layer_norm_fp8_kernel(const T* __restrict__ x,
                                       __nv_fp8_e4m3* __restrict__ out,
                                       const T* __restrict__ gamma,
                                       const T* __restrict__ beta,
                                       int dim, float eps) {
    int row = blockIdx.x;
    const T* x_row = x + row * dim;
    __nv_fp8_e4m3* o_row = out + row * dim;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_sum += to_f32(x_row[i]);
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = to_f32(x_row[i]) - mean;
        local_var += d * d;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normed = (to_f32(x_row[i]) - mean) * inv_std * to_f32(gamma[i]) + to_f32(beta[i]);
        o_row[i] = __nv_fp8_e4m3(normed);
    }
}

template __global__ void layer_norm_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_fp8_e4m3*, const __nv_bfloat16*, const __nv_bfloat16*, int, float);

void layer_norm_fp8(const __half* x, __nv_fp8_e4m3* out,
                     const __half* gamma, const __half* beta,
                     int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_fp8_kernel_fp16<<<seq_len, 256, 0, stream>>>(
        x, out, gamma, beta, seq_len, dim, eps);
}
void layer_norm_fp8_bf16(const __nv_bfloat16* x, __nv_fp8_e4m3* out,
                          const __nv_bfloat16* gamma, const __nv_bfloat16* beta,
                          int seq_len, int dim, float eps, cudaStream_t stream) {
    layer_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, out, gamma, beta, dim, eps);
}

__global__ void awq_layer_norm_fp8_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_fp8_e4m3* __restrict__ out,
    const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ inv_s,
    const float* __restrict__ d_scale,
    int dim, float eps) {
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + (long long)row * dim;
    __nv_fp8_e4m3* o_row = out + (long long)row * dim;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        local_sum += __bfloat162float(x_row[i]);
    float mean = block_reduce_sum(local_sum, shared) / dim;

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = __bfloat162float(x_row[i]) - mean;
        local_var += d * d;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, shared) / dim + eps);
    float inv_a = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float normed = (__bfloat162float(x_row[i]) - mean) * inv_std
            * __bfloat162float(gamma[i]) + __bfloat162float(beta[i]);
        float rounded = __bfloat162float(__float2bfloat16(normed));
        float q = rounded * __bfloat162float(inv_s[i]) * inv_a;
        q = fminf(fmaxf(q, -448.0f), 448.0f);
        o_row[i] = __nv_fp8_e4m3(q);
    }
}

void awq_layer_norm_fp8_bf16(const __nv_bfloat16* x,
                             __nv_fp8_e4m3* out,
                             const __nv_bfloat16* gamma,
                             const __nv_bfloat16* beta,
                             const __nv_bfloat16* inv_s,
                             const float* d_scale,
                             int seq_len, int dim, float eps,
                             cudaStream_t stream) {
    awq_layer_norm_fp8_bf16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, out, gamma, beta, inv_s, d_scale, dim, eps);
}

// ── AdaRMSNorm + Style ──
template<typename T>
__global__ void ada_rms_norm_style_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ style, T* __restrict__ out, T* __restrict__ gate_out,
    int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    T2* gate2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(
            from_f32<T>(n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)),
            from_f32<T>(n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)));
        gate2[i] = gv;
    }
}

template __global__ void ada_rms_norm_style_kernel<__half>(const __half*, const __half*, const __half*, __half*, __half*, int, float);
template __global__ void ada_rms_norm_style_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, float);

void ada_rms_norm_style(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                        const __nv_bfloat16* style,
                        __nv_bfloat16* out, __nv_bfloat16* gate_out,
                        int seq_len, int dim, float eps, cudaStream_t stream) {
    ada_rms_norm_style_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps);
}

void ada_rms_norm_style_fp16(const __half* x, const __half* weight,
                             const __half* style,
                             __half* out, __half* gate_out,
                             int seq_len, int dim, float eps,
                             cudaStream_t stream) {
    ada_rms_norm_style_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps);
}

// ── RMSNorm → FP8 ──
template<typename T>
__global__ void rms_norm_fp8_kernel(const T* __restrict__ x,
                                     const T* __restrict__ weight,
                                     __nv_fp8_e4m3* __restrict__ out,
                                     int dim, float eps,
                                     const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __nv_fp8_e4m3* out_row = out + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        float v0 = to_f32(xv.x) * rms * to_f32(wv.x) * inv_scale;
        float v1 = to_f32(xv.y) * rms * to_f32(wv.y) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

template __global__ void rms_norm_fp8_kernel<__half>(const __half*, const __half*, __nv_fp8_e4m3*, int, float, const float*);
template __global__ void rms_norm_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_fp8_e4m3*, int, float, const float*);

void rms_norm_fp8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                   __nv_fp8_e4m3* out, int seq_len, int dim, float eps,
                   const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps, d_scale);
}
void rms_norm_fp8_fp16(const __half* x, const __half* weight,
                        __nv_fp8_e4m3* out, int seq_len, int dim, float eps,
                        const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(x, weight, out, dim, eps, d_scale);
}

// ── AdaRMSNorm + Style → FP8 ──
template<typename T>
__global__ void ada_rms_norm_style_fp8_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ style, __nv_fp8_e4m3* __restrict__ out, T* __restrict__ gate_out,
    int dim, float eps, const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __nv_fp8_e4m3* out_row = out + row * dim;
    T2* gate2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate2[i] = gv;
    }
}

template __global__ void ada_rms_norm_style_fp8_kernel<__half>(const __half*, const __half*, const __half*, __nv_fp8_e4m3*, __half*, int, float, const float*);
template __global__ void ada_rms_norm_style_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_fp8_e4m3*, __nv_bfloat16*, int, float, const float*);

void ada_rms_norm_style_fp8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                             const __nv_bfloat16* style,
                             __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                             int seq_len, int dim, float eps,
                             const float* d_scale, cudaStream_t stream) {
    ada_rms_norm_style_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps, d_scale);
}

// ── Residual Add + RMSNorm → FP8 ──
template<typename T>
__global__ void residual_add_rms_norm_fp8_kernel(
    T* __restrict__ residual, const T* __restrict__ x,
    const T* __restrict__ weight, __nv_fp8_e4m3* __restrict__ out,
    int dim, float eps, const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    __nv_fp8_e4m3* out_row = out + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        float v0 = to_f32(rv.x) * rms * to_f32(wv.x) * inv_scale;
        float v1 = to_f32(rv.y) * rms * to_f32(wv.y) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

template __global__ void residual_add_rms_norm_fp8_kernel<__half>(__half*, const __half*, const __half*, __nv_fp8_e4m3*, int, float, const float*);
template __global__ void residual_add_rms_norm_fp8_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_fp8_e4m3*, int, float, const float*);

void residual_add_rms_norm_fp8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                const __nv_bfloat16* weight, __nv_fp8_e4m3* out,
                                int seq_len, int dim, float eps,
                                const float* d_scale, cudaStream_t stream) {
    residual_add_rms_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps, d_scale);
}
void residual_add_rms_norm_fp8_fp16(__half* residual, const __half* x,
                                     const __half* weight, __nv_fp8_e4m3* out,
                                     int seq_len, int dim, float eps,
                                     const float* d_scale, cudaStream_t stream) {
    residual_add_rms_norm_fp8_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps, d_scale);
}

// ── Residual Add + RMSNorm → T (same dtype output) ──
template<typename T>
__global__ void residual_add_rms_norm_kernel(
    T* __restrict__ residual, const T* __restrict__ x,
    const T* __restrict__ weight, T* __restrict__ out,
    int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        float v0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float v1 = to_f32(rv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

template __global__ void residual_add_rms_norm_kernel<__half>(__half*, const __half*, const __half*, __half*, int, float);
template __global__ void residual_add_rms_norm_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, float);

void residual_add_rms_norm(__nv_bfloat16* residual, const __nv_bfloat16* x,
                            const __nv_bfloat16* weight, __nv_bfloat16* out,
                            int seq_len, int dim, float eps,
                            cudaStream_t stream) {
    residual_add_rms_norm_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps);
}


// ================================================================
// Production-exact kernels (no weight, no d_scale variant)
// ================================================================

// RMSNorm → FP8 (no weight, no d_scale). Bit-exact reference variant.
__global__ void plain_rms_fp8_kernel(const __half* __restrict__ in,
                                      __nv_fp8_e4m3* __restrict__ out,
                                      int C) {
    int r = blockIdx.x;
    const __half* row = in + r * C;
    __nv_fp8_e4m3* orow = out + r * C;

    __shared__ float sh[32];
    float ssq = 0;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float v = __half2float(row[i]);
        ssq += v * v;
    }
    // Warp reduce
    for (int o = 16; o > 0; o >>= 1)
        ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    int l = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (!l) sh[w] = ssq;
    __syncthreads();
    if (!w) {
        ssq = (l < (blockDim.x + 31) / 32) ? sh[l] : 0;
        for (int o = 16; o > 0; o >>= 1)
            ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    }
    __syncthreads();
    if (!threadIdx.x) sh[0] = ssq;
    __syncthreads();
    float rms = rsqrtf(sh[0] / C + 1e-6f);

    for (int i = threadIdx.x; i < C; i += blockDim.x)
        orow[i] = __nv_fp8_e4m3(__half2float(row[i]) * rms);
}

void plain_rms_fp8_fp16(const __half* x, __nv_fp8_e4m3* out,
                         int seq_len, int dim, cudaStream_t stream) {
    plain_rms_fp8_kernel<<<seq_len, 256, 0, stream>>>(x, out, dim);
}

// Residual add + RMSNorm → FP8 (no weight, no d_scale). Identical to pi05 res_rms_fp8_k.
__global__ void plain_res_rms_fp8_kernel(__half* __restrict__ residual,
                                          const __half* __restrict__ x,
                                          __nv_fp8_e4m3* __restrict__ out,
                                          int D) {
    int row = blockIdx.x;
    __half2* res2 = reinterpret_cast<__half2*>(residual + row * D);
    const __half2* x2 = reinterpret_cast<const __half2*>(x + row * D);
    __nv_fp8_e4m3* out_row = out + row * D;
    int D2 = D >> 1;

    extern __shared__ float sh[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __half2 rv = res2[i], xv = x2[i];
        float r0 = __half2float(rv.x) + __half2float(xv.x);
        float r1 = __half2float(rv.y) + __half2float(xv.y);
        res2[i] = __halves2half2(__float2half(r0), __float2half(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    // Block reduce
    float val = local_sum;
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, o);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (!lane) sh[wid] = val;
    __syncthreads();
    int nw = blockDim.x >> 5;
    val = (threadIdx.x < nw) ? sh[threadIdx.x] : 0;
    if (!wid) {
        for (int o = 16; o > 0; o >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, o);
    }
    if (!threadIdx.x) sh[0] = val;
    __syncthreads();
    float rms = rsqrtf(sh[0] / D + 1e-6f);

    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __half2 rv = res2[i];
        float v0 = __half2float(rv.x) * rms;
        float v1 = __half2float(rv.y) * rms;
        out_row[2 * i]     = __nv_fp8_e4m3(v0);
        out_row[2 * i + 1] = __nv_fp8_e4m3(v1);
    }
}

void plain_res_rms_fp8_fp16(__half* residual, const __half* x,
                             __nv_fp8_e4m3* out, int seq_len, int dim,
                             cudaStream_t stream) {
    plain_res_rms_fp8_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, out, dim);
}

// ── RMSNorm → FP8 with d_scale, no weight (norm weight baked into GEMM weights) ──
// Verbatim copy of production rms_norm_fp8_static_k.
// NOT "equivalent" — literally the same source to get identical SASS.

#define RMS_NW_THREADS 256
#define RMS_NW_D_MAX 2048
#define RMS_NW_ELEMS_PER_THREAD (RMS_NW_D_MAX / RMS_NW_THREADS)  // 8

__global__ void rms_norm_fp8_noweight_kernel(const __half* in, __nv_fp8_e4m3* out, int R, int C,
                                       const float* descale_ptr) {
    int r = blockIdx.x; if (r >= R) return;
    const __half* row = in + r * C;
    __nv_fp8_e4m3* orow = out + r * C;

    const __half2* row2 = reinterpret_cast<const __half2*>(row);
    int C2 = C / 2;

    float cache[RMS_NW_ELEMS_PER_THREAD];  // 8 floats
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            __half2 v2 = row2[c2];
            cache[it*2]   = __half2float(v2.x);
            cache[it*2+1] = __half2float(v2.y);
            ssq += cache[it*2]*cache[it*2] + cache[it*2+1]*cache[it*2+1];
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float scale = __frsqrt_rn(sh[0] / C + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);

    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void rms_norm_fp8_noweight_fp16(const __half* x, __nv_fp8_e4m3* out,
                                 int seq_len, int dim,
                                 const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_noweight_kernel<<<seq_len, 256, 0, stream>>>(x, out, seq_len, dim, d_scale);
}

// ── Residual + RMSNorm → FP8 with d_scale, no weight ──
// Verbatim copy of production res_rms_fp8_static_k.

__global__ void res_rms_fp8_noweight_kernel(__half* residual, const __half* x, __nv_fp8_e4m3* out, int D,
                                      const float* descale_ptr) {
    int r = blockIdx.x;
    __half* res_row = residual + r * D;
    const __half* x_row = x + r * D;
    __nv_fp8_e4m3* orow = out + r * D;
    int D2 = D / 2;

    __half2* res2w = reinterpret_cast<__half2*>(res_row);
    const __half2* res2 = reinterpret_cast<const __half2*>(res_row);
    const __half2* x2 = reinterpret_cast<const __half2*>(x_row);

    float cache[RMS_NW_ELEMS_PER_THREAD];  // 8 floats
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            __half2 rv2 = res2[c2], xv2 = x2[c2];
            float r0 = __half2float(rv2.x) + __half2float(xv2.x);
            float r1 = __half2float(rv2.y) + __half2float(xv2.y);
            cache[it*2]   = r0;
            cache[it*2+1] = r1;
            res2w[c2] = __halves2half2(__float2half(r0), __float2half(r1));
            ssq += r0*r0 + r1*r1;
        } else {
            cache[it*2] = 0; cache[it*2+1] = 0;
        }
    }

    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();

    float scale = __frsqrt_rn(sh[0] / D + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);

    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void residual_add_rms_norm_fp8_noweight_fp16(__half* residual, const __half* x,
                                               __nv_fp8_e4m3* out,
                                               int seq_len, int dim,
                                               const float* d_scale, cudaStream_t stream) {
    res_rms_fp8_noweight_kernel<<<seq_len, 256, 0, stream>>>(
        residual, x, out, dim, d_scale);
}

// ── BF16 noweight variants ──
// For models with activations exceeding FP16 range (>65504).
// BF16 residual stream can store up to 3.4e38.

__global__ void rms_norm_fp8_noweight_bf16_kernel(const __nv_bfloat16* in, __nv_fp8_e4m3* out, int R, int C,
                                                    const float* descale_ptr) {
    int r = blockIdx.x; if (r >= R) return;
    const __nv_bfloat16* row = in + r * C;
    __nv_fp8_e4m3* orow = out + r * C;
    const __nv_bfloat162* row2 = reinterpret_cast<const __nv_bfloat162*>(row);
    int C2 = C / 2;
    float cache[RMS_NW_ELEMS_PER_THREAD];
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            __nv_bfloat162 v2 = row2[c2];
            cache[it*2]   = __bfloat162float(v2.x);
            cache[it*2+1] = __bfloat162float(v2.y);
            ssq += cache[it*2]*cache[it*2] + cache[it*2+1]*cache[it*2+1];
        } else { cache[it*2] = 0; cache[it*2+1] = 0; }
    }
    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();
    float scale = __frsqrt_rn(sh[0] / C + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < C2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void rms_norm_fp8_noweight_bf16(const __nv_bfloat16* x, __nv_fp8_e4m3* out,
                                 int seq_len, int dim,
                                 const float* d_scale, cudaStream_t stream) {
    rms_norm_fp8_noweight_bf16_kernel<<<seq_len, 256, 0, stream>>>(x, out, seq_len, dim, d_scale);
}

__global__ void res_rms_fp8_noweight_bf16_kernel(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                                   __nv_fp8_e4m3* out, int D,
                                                   const float* descale_ptr) {
    int r = blockIdx.x;
    __nv_bfloat16* res_row = residual + r * D;
    const __nv_bfloat16* x_row = x + r * D;
    __nv_fp8_e4m3* orow = out + r * D;
    int D2 = D / 2;
    __nv_bfloat162* res2w = reinterpret_cast<__nv_bfloat162*>(res_row);
    const __nv_bfloat162* res2 = reinterpret_cast<const __nv_bfloat162*>(res_row);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x_row);
    float cache[RMS_NW_ELEMS_PER_THREAD];
    float ssq = 0;
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            __nv_bfloat162 rv2 = res2[c2], xv2 = x2[c2];
            float r0 = __bfloat162float(rv2.x) + __bfloat162float(xv2.x);
            float r1 = __bfloat162float(rv2.y) + __bfloat162float(xv2.y);
            cache[it*2]   = r0;
            cache[it*2+1] = r1;
            res2w[c2] = __halves2bfloat162(__float2bfloat16(r0), __float2bfloat16(r1));
            ssq += r0*r0 + r1*r1;
        } else { cache[it*2] = 0; cache[it*2+1] = 0; }
    }
    __shared__ float sh[16];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o);
    if (!lane) sh[wid] = ssq; __syncthreads();
    if (!wid) { ssq = (lane < (blockDim.x/32)) ? sh[lane] : 0;
                for (int o = 16; o > 0; o >>= 1) ssq += __shfl_xor_sync(0xffffffff, ssq, o); }
    __syncthreads(); if (!threadIdx.x) sh[0] = ssq; __syncthreads();
    float scale = __frsqrt_rn(sh[0] / D + 1e-6f) / fmaxf(*descale_ptr, 1e-12f);
    #pragma unroll
    for (int it = 0; it < RMS_NW_ELEMS_PER_THREAD / 2; it++) {
        int c2 = threadIdx.x + it * blockDim.x;
        if (c2 < D2) {
            int c = c2 * 2;
            __nv_fp8_e4m3 pair[2];
            pair[0] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2]   * scale, -448.f), 448.f));
            pair[1] = __nv_fp8_e4m3(fminf(fmaxf(cache[it*2+1] * scale, -448.f), 448.f));
            *reinterpret_cast<uint16_t*>(orow + c) = *reinterpret_cast<uint16_t*>(pair);
        }
    }
}

void residual_add_rms_norm_fp8_noweight_bf16(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                               __nv_fp8_e4m3* out,
                                               int seq_len, int dim,
                                               const float* d_scale, cudaStream_t stream) {
    res_rms_fp8_noweight_bf16_kernel<<<seq_len, 256, 0, stream>>>(
        residual, x, out, dim, d_scale);
}

// Cast FP16 → FP8 (no scale). Identical to pi05 cast_fp16_fp8_k.
__global__ void cast_fp16_fp8_kernel(const __half* __restrict__ in,
                                      __nv_fp8_e4m3* __restrict__ out,
                                      int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = __nv_fp8_e4m3(__half2float(in[idx]));
}

void cast_fp16_fp8(const __half* input, __nv_fp8_e4m3* output,
                    int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cast_fp16_fp8_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// ================================================================
// LayerNorm without affine parameters (for DiT AdaLayerNorm / norm3 / norm_out)
// out = (x - mean) / sqrt(var + eps)
// No weight/bias — elementwise_affine=False
// ================================================================

__global__ void layer_norm_no_affine_fp16_kernel(const __half* __restrict__ x,
                                                   __half* __restrict__ out,
                                                   int dim, float eps) {
    int row = blockIdx.x;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + row * dim);
    __half2* out2 = reinterpret_cast<__half2*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        local_sum += __half2float(val.x) + __half2float(val.y);
    }
    // Block reduce for mean
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
        __half2 v = x2[i];
        float d0 = __half2float(v.x) - mean, d1 = __half2float(v.y) - mean;
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
        __half2 xv = x2[i];
        float v0 = (__half2float(xv.x) - mean) * inv_std;
        float v1 = (__half2float(xv.y) - mean) * inv_std;
        out2[i] = __halves2half2(__float2half(v0), __float2half(v1));
    }
}

void layer_norm_no_affine_fp16(const __half* x, __half* out,
                                int seq_len, int dim, float eps,
                                cudaStream_t stream) {
    layer_norm_no_affine_fp16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, out, dim, eps);
}

// ================================================================
// Fused AdaLayerNorm FP16 (for DiT)
// out = LN(x, no_affine) * (1 + scale) + shift
// scale, shift: [dim] per-layer precomputed from timestep
// ================================================================

__global__ void ada_layer_norm_fp16_kernel(const __half* __restrict__ x,
                                            const __half* __restrict__ scale,
                                            const __half* __restrict__ shift,
                                            __half* __restrict__ out,
                                            int dim, float eps) {
    int row = blockIdx.x;
    const __half2* x2 = reinterpret_cast<const __half2*>(x + row * dim);
    const __half2* sc2 = reinterpret_cast<const __half2*>(scale);
    const __half2* sh2 = reinterpret_cast<const __half2*>(shift);
    __half2* out2 = reinterpret_cast<__half2*>(out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        __half2 val = x2[i];
        local_sum += __half2float(val.x) + __half2float(val.y);
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
        __half2 v = x2[i];
        float d0 = __half2float(v.x) - mean, d1 = __half2float(v.y) - mean;
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
        __half2 xv = x2[i], sv = sc2[i], hv = sh2[i];
        float n0 = (__half2float(xv.x) - mean) * inv_std;
        float n1 = (__half2float(xv.y) - mean) * inv_std;
        float v0 = n0 * (1.0f + __half2float(sv.x)) + __half2float(hv.x);
        float v1 = n1 * (1.0f + __half2float(sv.y)) + __half2float(hv.y);
        out2[i] = __halves2half2(__float2half(v0), __float2half(v1));
    }
}

void ada_layer_norm_fp16(const __half* x, const __half* scale, const __half* shift,
                          __half* out, int seq_len, int dim, float eps,
                          cudaStream_t stream) {
    ada_layer_norm_fp16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, scale, shift, out, dim, eps);
}


// ---- Public Orin/INT8 helpers restored for API compatibility ----
template<typename T>
__global__ void ada_rms_norm_style_int8_kernel(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ style, int8_t* __restrict__ out, T* __restrict__ gate_out,
    int dim, float eps, float* __restrict__ d_scales) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* gate2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int8_t* out_row = out + row * dim;
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    float local_amax = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        float val0 = n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x);
        float val1 = n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y);
        local_amax = fmaxf(local_amax, fabsf(val0));
        local_amax = fmaxf(local_amax, fabsf(val1));
    }
    float amax = block_reduce_max(local_amax, shared);
    __shared__ float scale_s;
    if (threadIdx.x == 0) {
        float s = fmaxf(amax / 127.0f, 1e-10f);
        d_scales[row] = s;
        scale_s = s;
    }
    __syncthreads();
    float inv_scale = 1.0f / scale_s;

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(xv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        int q0 = __float2int_rn(val0);
        int q1 = __float2int_rn(val1);
        out_row[2 * i] = static_cast<int8_t>((q0 < -127) ? -127 : ((q0 > 127) ? 127 : q0));
        out_row[2 * i + 1] = static_cast<int8_t>((q1 < -127) ? -127 : ((q1 > 127) ? 127 : q1));
        gate2[i] = gv;
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void ada_rms_norm_style_int8_kernel<__half>(
    const __half*, const __half*, const __half*, int8_t*, __half*, int, float, float*))
FVK_KERNEL_INSTANTIATE(__global__ void ada_rms_norm_style_int8_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int8_t*, __nv_bfloat16*, int, float, float*))
void ada_rms_norm_style_int8(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                             const __nv_bfloat16* style,
                             int8_t* out, __nv_bfloat16* gate_out,
                             int seq_len, int dim, float eps,
                             float* d_scales, cudaStream_t stream) {
    ada_rms_norm_style_int8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, style, out, gate_out, dim, eps, d_scales);
}

// ── RMSNorm → FP8 ──
// rms_norm + quantize_int8_rowwise pair requires.  Each block
// handles one row; shared memory holds the float-converted values
// so the data is only loaded from global memory once.
//
// smem layout: [0..cols-1] float data; [cols..cols+31] warp partials.
// Launch: grid=(seq_len,), block=(256,), smem=(cols+32)*sizeof(float)
__global__ void rms_norm_int8_rowwise_kernel(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        int8_t*  __restrict__ out,
        float*   __restrict__ scales,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols;

    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* xr = x + (int64_t)row * cols;
    int8_t* outr = out + (int64_t)row * cols;

    // Pass 1: load x → smem, accumulate sum of squares
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = __bfloat162float(xr[i]);
        smem[i] = xi;
        sum_sq += xi * xi;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);

    // Pass 2: normalize (reuse smem), accumulate max_abs
    float max_abs = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * rms * __bfloat162float(weight[i]);
        smem[i] = v;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = fmaxf(block_reduce_max(max_abs, partial) / 127.f, 1e-12f);
    if (threadIdx.x == 0) scales[row] = scale;
    float inv_s = 1.f / scale;

    // Pass 3: write INT8
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * inv_s;
        outr[i] = (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

void rms_norm_int8_rowwise(const __nv_bfloat16* x,
                            const __nv_bfloat16* weight,
                            int8_t* out, float* scales,
                            int seq_len, int dim, float eps,
                            cudaStream_t stream) {
    int smem = (dim + 32) * sizeof(float);
    rms_norm_int8_rowwise_kernel<<<seq_len, 256, smem, stream>>>(
        x, weight, out, scales, seq_len, dim, eps);
}

// ── Fused residual-add + RMSNorm → INT8 (per-row dynamic scale) ──
//
// Fuses three operations that appear at encoder B4:
//   residual += x_norm
//   normed    = RMSNorm(residual, weight)
//   out_i8    = INT8_quantize_rowwise(normed)
// into a single kernel pass over global memory.
__global__ void residual_add_rms_norm_int8_rowwise_kernel(
        __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        int8_t* __restrict__ out,
        float*  __restrict__ scales,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols;

    int row = blockIdx.x;
    if (row >= rows) return;

    __nv_bfloat16* res_row = residual + (int64_t)row * cols;
    const __nv_bfloat16* x_row = x + (int64_t)row * cols;
    int8_t* out_row = out + (int64_t)row * cols;

    // Pass 1: residual += x, load into smem, accumulate sum_sq
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float ri = __bfloat162float(res_row[i]) + __bfloat162float(x_row[i]);
        res_row[i] = __float2bfloat16(ri);
        smem[i] = ri;
        sum_sq += ri * ri;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);

    // Pass 2: normalize, accumulate max_abs
    float max_abs = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * rms * __bfloat162float(weight[i]);
        smem[i] = v;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = fmaxf(block_reduce_max(max_abs, partial) / 127.f, 1e-12f);
    if (threadIdx.x == 0) scales[row] = scale;
    float inv_s = 1.f / scale;

    // Pass 3: write INT8
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * inv_s;
        out_row[i] = (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

void residual_add_rms_norm_int8_rowwise(
        __nv_bfloat16* residual, const __nv_bfloat16* x,
        const __nv_bfloat16* weight,
        int8_t* out, float* scales,
        int seq_len, int dim, float eps,
        cudaStream_t stream) {
    int smem = (dim + 32) * sizeof(float);
    residual_add_rms_norm_int8_rowwise_kernel<<<seq_len, 256, smem, stream>>>(
        residual, x, weight, out, scales, seq_len, dim, eps);
}

// ── Fused bias-residual + LayerNorm (strict bf16 round-trip in middle) ──
//
// Fuses three SigLIP ops that appear between consecutive sub-blocks:
//   residual += x + bias_pre        (bias_residual kernel, ncu L2 hit ~34%)
//   out      = LayerNorm(residual)  (layer_norm kernel,    ncu L2 hit ~52%)
// into a single block-per-row kernel. The intermediate `residual` write
// + read between the two original kernels is the worst-L2-hit (DRAM-
// bound) traffic in the SigLIP path; fusing eliminates that round-trip
// entirely.
//
// **Strict-precision contract** (matches the bias_residual + layer_norm
// pair bit-for-bit):
//   1. bias-add result is rounded to bf16 BEFORE the LN computation
//      (mirrors bias_residual storing bf16 to global, layer_norm
//      promoting bf16 → fp32 on read).
//   2. The LN sum/var reductions, normalization, weight/bias apply,
//      and final bf16 round are bit-identical to the existing
//      layer_norm_kernel.
//
// SMEM layout (for blockDim.x = 256):
//   smem[0..31] = block_reduce_sum partials (one per warp, max 8 for
//                 blockDim.x = 256). 1024 bytes total — same as
//                 layer_norm_kernel.
template<typename T>
__global__ void bias_residual_layer_norm_kernel(
        T* __restrict__ residual,
        const T* __restrict__ x,
        const T* __restrict__ bias_pre,
        const T* __restrict__ ln_weight,
        const T* __restrict__ ln_bias,
        T* __restrict__ out,
        int dim, float eps) {
    extern __shared__ float partial[];

    int row = blockIdx.x;
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual + (size_t)row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + (size_t)row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias_pre);
    const T2* w2 = reinterpret_cast<const T2*>(ln_weight);
    const T2* lnb2 = reinterpret_cast<const T2*>(ln_bias);
    T2* out2 = reinterpret_cast<T2*>(out + (size_t)row * dim);
    int dim2 = dim >> 1;

    // Pass 1: residual = bf16(residual + x + bias_pre) (write to global),
    // accumulate sum for mean. We re-read residual from global on passes
    // 2 and 3 — strictly matching layer_norm_kernel's behavior. (An
    // earlier smem-cached variant produced 1-ULP drift on the SigLIP
    // shape; subtle precision difference between 'fp32 stored to smem'
    // and 'bf16 stored to global, re-read'. Re-reading from global is
    // bit-equivalent to running the kernel pair sequentially.)
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], bv = b2[i];
        T r0 = from_f32<T>(to_f32(rv.x) + to_f32(xv.x) + to_f32(bv.x));
        T r1 = from_f32<T>(to_f32(rv.y) + to_f32(xv.y) + to_f32(bv.y));
        res2[i] = make_packed2<T>(r0, r1);
        local_sum += to_f32(r0) + to_f32(r1);
    }
    float mean = block_reduce_sum(local_sum, partial) / dim;

    // Pass 2: re-read residual from global (now bf16), compute
    // (val - mean)^2, accumulate var. Bit-identical to layer_norm_kernel.
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i];
        float d0 = to_f32(val.x) - mean, d1 = to_f32(val.y) - mean;
        local_var += d0 * d0 + d1 * d1;
    }
    float inv_std = rsqrtf(block_reduce_sum(local_var, partial) / dim + eps);

    // Pass 3: re-read residual, normalize, apply weight/bias, write out.
    // Identical to layer_norm_kernel's final loop.
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = res2[i];
        T2 wv = w2[i], lbv = lnb2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        float n0 = (v0 - mean) * inv_std * to_f32(wv.x) + to_f32(lbv.x);
        float n1 = (v1 - mean) * inv_std * to_f32(wv.y) + to_f32(lbv.y);
        out2[i] = make_packed2<T>(from_f32<T>(n0), from_f32<T>(n1));
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void bias_residual_layer_norm_kernel<__half>(__half*, const __half*, const __half*, const __half*, const __half*, __half*, int, float))
FVK_KERNEL_INSTANTIATE(__global__ void bias_residual_layer_norm_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, float))

void bias_residual_layer_norm_bf16(
        __nv_bfloat16* residual, const __nv_bfloat16* x,
        const __nv_bfloat16* bias_pre,
        const __nv_bfloat16* ln_weight, const __nv_bfloat16* ln_bias,
        __nv_bfloat16* out, int seq_len, int dim, float eps,
        cudaStream_t stream) {
    int smem = 256 * sizeof(float);
    bias_residual_layer_norm_kernel<__nv_bfloat16><<<seq_len, 256, smem, stream>>>(
        residual, x, bias_pre, ln_weight, ln_bias, out, dim, eps);
}

void bias_residual_layer_norm_fp16(
        __half* residual, const __half* x,
        const __half* bias_pre,
        const __half* ln_weight, const __half* ln_bias,
        __half* out, int seq_len, int dim, float eps,
        cudaStream_t stream) {
    int smem = 256 * sizeof(float);
    bias_residual_layer_norm_kernel<__half><<<seq_len, 256, smem, stream>>>(
        residual, x, bias_pre, ln_weight, ln_bias, out, dim, eps);
}

// ── Vision Token Spatial Average Pooling (BF16) ──
//
// Reduces SigLIP output from (nv * spv, dim) to (nv * spv / (f*f), dim)
// by applying f×f average pooling in the spatial (H, W) grid.
//
// Token layout: each view's spv tokens are arranged as an (H, W) grid
// with H = W = sqrt(spv). The kernel averages each f×f block of tokens
// into one output token, reducing spv to spv/(f*f) per view.
//
// Launch: gridDim = (nv * out_spv, 1), blockDim = (256, 1)
//         where out_spv = spv / (f*f)
// smem: not needed (pure BW-limited kernel)
__global__ void avg_pool_vision_tokens_kernel(
        const __nv_bfloat16* __restrict__ x,   // (nv * spv, dim) BF16
        __nv_bfloat16* __restrict__ out,        // (nv * out_spv, dim) BF16
        int nv, int H, int W, int dim, int f) {
    // output token index in [0, nv * (H/f) * (W/f))
    int out_tok = blockIdx.x;
    int H_out = H / f;
    int W_out = W / f;
    int spv_out = H_out * W_out;

    int v   = out_tok / spv_out;          // view index
    int rc  = out_tok % spv_out;          // spatial index within view
    int r_out = rc / W_out;
    int c_out = rc % W_out;

    float inv_f2 = 1.0f / float(f * f);

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.f;
        for (int dr = 0; dr < f; dr++) {
            for (int dc = 0; dc < f; dc++) {
                int r_in  = r_out * f + dr;
                int c_in  = c_out * f + dc;
                int in_tok = v * H * W + r_in * W + c_in;
                sum += __bfloat162float(x[in_tok * dim + d]);
            }
        }
        out[out_tok * dim + d] = __float2bfloat16(sum * inv_f2);
    }
}

// pool_factor: 1 = no-op, 2 = 2x2 (spv 256→64), 4 = 4x4 (spv 256→16)
// H = W = sqrt(spv) must be divisible by pool_factor.
void avg_pool_vision_tokens(
        const __nv_bfloat16* x, __nv_bfloat16* out,
        int nv, int H, int W, int dim, int pool_factor,
        cudaStream_t stream) {
    int H_out = H / pool_factor;
    int W_out = W / pool_factor;
    int out_tokens = nv * H_out * W_out;
    avg_pool_vision_tokens_kernel<<<out_tokens, 256, 0, stream>>>(
        x, out, nv, H, W, dim, pool_factor);

}

// RMSNorm that also block-reduces the abs-max of its own fp16 output into
// a caller-zeroed device scale accumulator (atomicMax across blocks). Lets
// a dynamic per-tensor FP8 quantize of the normalized output skip the
// separate absmax_kernel full read pass (fused into this kernel's existing
// write pass instead). `max_val` must be memset to 0 by the caller first.
template<typename T>
__global__ void rms_norm_amax_kernel(const T* __restrict__ x,
                                      const T* __restrict__ weight,
                                      T* __restrict__ out,
                                      float* __restrict__ max_val,
                                      int dim, float eps) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 val = x2[i];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        local_sum += v0 * v0 + v1 * v1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 xv = x2[i], wv = w2[i];
        float v0 = to_f32(xv.x) * rms * to_f32(wv.x);
        float v1 = to_f32(xv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
        // amax over the fp16-rounded stored values, matching absmax_kernel
        // reading the fp16 buffer afterwards (quantize consumes those).
        float q0 = to_f32(from_f32<T>(v0));
        float q1 = to_f32(from_f32<T>(v1));
        local_max = fmaxf(local_max, fmaxf(fabsf(q0), fabsf(q1)));
    }
    float block_max = block_reduce_max(local_max, shared);
    if (threadIdx.x == 0) atomicMax((int*)max_val, __float_as_int(block_max));
}

template __global__ void rms_norm_amax_kernel<__half>(const __half*, const __half*, __half*, float*, int, float);

void rms_norm_amax_fp16(const __half* x, const __half* weight,
                         __half* out, float* d_amax,
                         int seq_len, int dim, float eps,
                         cudaStream_t stream) {
    rms_norm_amax_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        x, weight, out, d_amax, dim, eps);
}

// Residual add + RMSNorm with amax fused into the xn write pass.
// residual[row,:] += x[row,:] (fp16-rounded, same as residual_add_fp16);
// ssq is computed over the ROUNDED fp16 residual values (same as rms_norm
// reading the fp16 residual buffer afterwards); the normalized xn output
// is computed from register-cached residuals (no global re-read); the
// abs-max of xn is block-reduced and atomically folded into max_val
// (caller must memset max_val to 0 first). One kernel replaces
// residual_add_fp16 + rms_norm_fp16 + absmax over xn.
template<typename T>
__global__ void residual_add_rms_norm_amax_kernel(
    T* __restrict__ residual, const T* __restrict__ x,
    const T* __restrict__ weight, T* __restrict__ xn_out,
    float* __restrict__ max_val, int dim, float eps) {
    using T2 = typename packed2<T>::type;
    constexpr int MAX_CHUNKS = 16;  // dim2/blockDim; D=4096 -> 8 at 256 threads
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    T2* out2 = reinterpret_cast<T2*>(xn_out + row * dim);
    int dim2 = dim >> 1;
    int chunks = (dim2 + blockDim.x - 1) / blockDim.x;
    T2 regs[MAX_CHUNKS];

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int c = 0; c < chunks && c < MAX_CHUNKS; c++) {
        int i = c * blockDim.x + threadIdx.x;
        if (i < dim2) {
            T2 rv = res2[i], xv = x2[i];
            float r0 = to_f32(rv.x) + to_f32(xv.x);
            float r1 = to_f32(rv.y) + to_f32(xv.y);
            T2 rn = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
            res2[i] = rn;
            regs[c] = rn;
            float q0 = to_f32(rn.x), q1 = to_f32(rn.y);
            local_sum += q0 * q0 + q1 * q1;
        }
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    float local_max = 0.0f;
    for (int c = 0; c < chunks && c < MAX_CHUNKS; c++) {
        int i = c * blockDim.x + threadIdx.x;
        if (i < dim2) {
            T2 rv = regs[c], wv = w2[i];
            float v0 = to_f32(rv.x) * rms * to_f32(wv.x);
            float v1 = to_f32(rv.y) * rms * to_f32(wv.y);
            out2[i] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
            // amax over the fp16-rounded stored values, matching
            // absmax_kernel reading the fp16 buffer afterwards.
            float q0 = to_f32(from_f32<T>(v0));
            float q1 = to_f32(from_f32<T>(v1));
            local_max = fmaxf(local_max, fmaxf(fabsf(q0), fabsf(q1)));
        }
    }
    float block_max = block_reduce_max(local_max, shared);
    if (threadIdx.x == 0) atomicMax((int*)max_val, __float_as_int(block_max));
}

template __global__ void residual_add_rms_norm_amax_kernel<__half>(__half*, const __half*, const __half*, __half*, float*, int, float);

void residual_add_rms_norm_amax_fp16(__half* residual, const __half* x,
                                      const __half* weight, __half* xn_out,
                                      float* d_amax, int seq_len, int dim,
                                      float eps, cudaStream_t stream) {
    residual_add_rms_norm_amax_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, xn_out, d_amax, dim, eps);
}

// FP16 host wrapper — re-uses the existing __half template instantiation.
void residual_add_rms_norm_fp16(__half* residual, const __half* x,
                                 const __half* weight, __half* out,
                                 int seq_len, int dim, float eps,
                                 cudaStream_t stream) {
    residual_add_rms_norm_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, weight, out, dim, eps);
}

#ifdef FLASHRT_ENABLE_CHAMELEON
// ── FP16 variants of the INT8-rowwise fused norms ──
// Same math as the bf16 kernels above, reading/writing FP16 residual
// streams (FP16-backbone models on Orin SM87).
__global__ void rms_norm_int8_rowwise_fp16_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ weight,
        int8_t*  __restrict__ out,
        float*   __restrict__ scales,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols;

    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* xr = x + (int64_t)row * cols;
    int8_t* outr = out + (int64_t)row * cols;

    // Pass 1: load x → smem, accumulate sum of squares
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xi = to_f32(xr[i]);
        smem[i] = xi;
        sum_sq += xi * xi;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);

    // Pass 2: normalize (reuse smem), accumulate max_abs
    float max_abs = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * rms * to_f32(weight[i]);
        smem[i] = v;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = fmaxf(block_reduce_max(max_abs, partial) / 127.f, 1e-12f);
    if (threadIdx.x == 0) scales[row] = scale;
    float inv_s = 1.f / scale;

    // Pass 3: write INT8
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * inv_s;
        outr[i] = (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

void rms_norm_int8_rowwise_fp16(const __half* x,
                                 const __half* weight,
                                 int8_t* out, float* scales,
                                 int seq_len, int dim, float eps,
                                 cudaStream_t stream) {
    int smem = (dim + 32) * sizeof(float);
    rms_norm_int8_rowwise_fp16_kernel<<<seq_len, 256, smem, stream>>>(
        x, weight, out, scales, seq_len, dim, eps);
}

__global__ void residual_add_rms_norm_int8_rowwise_fp16_kernel(
        __half* __restrict__ residual,
        const __half* __restrict__ x,
        const __half* __restrict__ weight,
        int8_t* __restrict__ out,
        float*  __restrict__ scales,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols;

    int row = blockIdx.x;
    if (row >= rows) return;

    __half* res_row = residual + (int64_t)row * cols;
    const __half* x_row = x + (int64_t)row * cols;
    int8_t* out_row = out + (int64_t)row * cols;

    // Pass 1: residual += x, load into smem, accumulate sum_sq
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float ri = to_f32(res_row[i]) + to_f32(x_row[i]);
        res_row[i] = from_f32<__half>(ri);
        smem[i] = ri;
        sum_sq += ri * ri;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);

    // Pass 2: normalize, accumulate max_abs
    float max_abs = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * rms * to_f32(weight[i]);
        smem[i] = v;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = fmaxf(block_reduce_max(max_abs, partial) / 127.f, 1e-12f);
    if (threadIdx.x == 0) scales[row] = scale;
    float inv_s = 1.f / scale;

    // Pass 3: write INT8
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * inv_s;
        out_row[i] = (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

void residual_add_rms_norm_int8_rowwise_fp16(
        __half* residual, const __half* x,
        const __half* weight,
        int8_t* out, float* scales,
        int seq_len, int dim, float eps,
        cudaStream_t stream) {
    int smem = (dim + 32) * sizeof(float);
    residual_add_rms_norm_int8_rowwise_fp16_kernel<<<seq_len, 256, smem, stream>>>(
        residual, x, weight, out, scales, seq_len, dim, eps);
}
#endif  // FLASHRT_ENABLE_CHAMELEON

// ── Gemma3 RMSNorm: y = bf16( fp32(x) * rsqrt(mean(fp32(x)^2) + eps) * (1 + fp32(w)) ) ──
//
// Gemma3 scales by (1 + w) rather than by w, and forms it in fp32 per element.
// Folding it into the weight ahead of time is not equivalent: bf16 near 1.0 has
// a 1/128 step, so a pre-rounded (1 + w) is visibly coarser than the fp32
// product.
//
// Rows may be strided, so the per-head q/k norm runs in place on a fused QKV
// buffer: row r lives at
//     (r / rows_per_group) * group_stride + (r % rows_per_group) * x_stride
// For plain contiguous rows pass rows_per_group = rows with group_stride = 0;
// rows_per_group = 1 would make r % rows_per_group always 0 and read row 0 for
// every row. See norm.cuh for the full contract.
//
// `residual`, when non-null, reproduces HF's ordering for a post-norm residual:
// the normalized value is rounded to bf16 first, then added to the residual, and
// the sum is rounded again — `residual = bf16(residual + bf16(norm(x)))`. `out`
// may be null when only the residual is wanted.
__device__ __forceinline__ float gemma3_bf(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__global__ void gemma3_rms_norm_kernel(const __nv_bfloat16* __restrict__ x, long long x_stride,
                                       const __nv_bfloat16* __restrict__ w,
                                       __nv_bfloat16* __restrict__ out, long long out_stride,
                                       __nv_bfloat16* __restrict__ residual, long long res_stride,
                                       int rows_per_group, long long group_stride,
                                       int dim, float eps) {
    const int r = blockIdx.x;
    const long long base_x = (long long)(r / rows_per_group) * group_stride +
                             (long long)(r % rows_per_group) * x_stride;
    const __nv_bfloat16* xr = x + base_x;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        ss += v * v;
    }
    __shared__ float red[32];
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x] : 0.f;
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float inv = rsqrtf(__fdiv_rn(red[0], (float)dim) + eps);
    const long long base_o = (long long)(r / rows_per_group) * group_stride +
                             (long long)(r % rows_per_group) * out_stride;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float y = gemma3_bf(__bfloat162float(xr[i]) * inv * (1.0f + __bfloat162float(w[i])));
        if (residual) {
            __nv_bfloat16* rr = residual + (long long)r * res_stride;
            rr[i] = __float2bfloat16(__bfloat162float(rr[i]) + y);
        }
        if (out) out[base_o + i] = __float2bfloat16(y);
    }
}

void gemma3_rms_norm_bf16(const __nv_bfloat16* x, long long x_stride,
                          const __nv_bfloat16* weight,
                          __nv_bfloat16* out, long long out_stride,
                          __nv_bfloat16* residual, long long res_stride,
                          int rows, int rows_per_group, long long group_stride,
                          int dim, float eps, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    if (rows_per_group <= 0)
        throw std::runtime_error("gemma3_rms_norm_bf16: rows_per_group must be positive");
    if (!out && !residual)
        throw std::runtime_error("gemma3_rms_norm_bf16: give out, residual, or both");
    const int threads = dim >= 1024 ? 1024 : (dim >= 256 ? 256 : 128);
    gemma3_rms_norm_kernel<<<rows, threads, 0, stream>>>(
        x, x_stride, weight, out, out_stride, residual, res_stride,
        rows_per_group, group_stride, dim, eps);
}

// The vectorized form's block reduction: 8 columns per thread, one value per
// thread, reduced warp-first and then across warps through shared memory.
//
// This is a different summation order from the strided-loop kernels'
// loop above, so the two can disagree in the last fp32 ulp of the sum of
// squares and a bf16 result can land either side of a rounding boundary. A
// caller that has to reproduce one of them bit for bit has to use that one;
// gemma3_rms_norm_bf16 deliberately implements the strided order.
// NOTE for every kernel below that reproduces a reference build's arithmetic:
// this target compiles with --prec-div=false, where `a / b` is an approximate
// division, while the DM05 extension is compiled with nvcc's defaults, where it
// is correctly rounded. The two agree exactly when the divisor is a power of two
// and can differ by an ulp otherwise -- enough to push a few elements per
// million across a bf16 rounding boundary. Every division that feeds an `inv`
// is therefore written as __fdiv_rn, which pins the reference's semantics
// regardless of this target's flags.
__device__ __forceinline__ float v8_block_sum(float v) {
    __shared__ float red[32];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = threadIdx.x < (blockDim.x + 31) / 32 ? red[threadIdx.x & 31] : 0.f;
    if (threadIdx.x < 32)
        for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    return red[0];
}

// Gemma3 RMSNorm whose result is consumed straight as a static per-tensor FP8
// (e4m3) GEMM input: one block per row, dim / 8 threads, 8 columns per thread.
// The normalized row never reaches memory as bf16, because its only consumer is
// the quantized GEMM and a bf16 copy would be written and never read.
//
// The quantization reproduces what a standalone quantize_fp8_static pass over
// the bf16 norm output would produce -- round to bf16, then clamp(x / a_scale)
// to e4m3 -- so the activation written here is byte-identical to the two-pass
// form's. `a_scale` is the same device scalar that pass reads.
__global__ void gemma3_rms_norm_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ w,
                                           const float* __restrict__ a_scale,
                                           __nv_fp8_e4m3* __restrict__ q,
                                           int dim, float eps) {
    union Gemma3V8 { uint4 u; __nv_bfloat16 h[8]; };
    const long long base = (long long)blockIdx.x * dim + threadIdx.x * 8;
    Gemma3V8 xv, wv;
    xv.u = *reinterpret_cast<const uint4*>(x + base);
    wv.u = *reinterpret_cast<const uint4*>(w + threadIdx.x * 8);
    float f[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = __bfloat162float(xv.h[j]);
        ss += f[j] * f[j];
    }
    const float inv = rsqrtf(__fdiv_rn(v8_block_sum(ss), (float)dim) + eps);
    const float qi = 1.0f / a_scale[0];
    __align__(8) __nv_fp8_e4m3 o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float y = __bfloat162float(__float2bfloat16(f[j] * inv * (1.0f + __bfloat162float(wv.h[j]))));
        o[j] = __nv_fp8_e4m3(fminf(fmaxf(y * qi, -448.f), 448.f));
    }
    reinterpret_cast<uint2*>(q + base)[0] = *reinterpret_cast<uint2*>(o);
}

void gemma3_rms_norm_fp8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                              __nv_fp8_e4m3* out, const float* a_scale,
                              int rows, int dim, float eps, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    if (dim % 8 != 0)
        throw std::runtime_error("gemma3_rms_norm_fp8_bf16: dim must be a multiple of 8");
    gemma3_rms_norm_fp8_kernel<<<rows, dim / 8, 0, stream>>>(x, weight, a_scale, out, dim, eps);
}

// ── Adaptive RMSNorm with one modulation shared by every row ──
//   y = bf16( bf16(x) * rsqrt(mean(fp32(x)^2) + eps) * (1 + scale) + shift )
//
// `mod` is a single 2*dim bf16 vector held for all rows: scale in mod[0:dim],
// shift in mod[dim:2*dim]. The product of x and the inverse RMS is carried in
// fp32 without an intermediate round; only the final sum is rounded.
//
// This is not ada_rms_norm_style. That one takes a per-row style of 3*dim
// (scale, shift, gate), applies a per-channel weight, and writes a gate output.
// Here the modulation is one shared pair, there is no weight -- the reference
// does not use one on this path -- and there is no gate. A caller holding a
// shared modulation cannot reach the styled kernel without materialising a
// per-row style, and the two also reduce the sum of squares in a different
// order, so this is the entry for that case rather than a second spelling of
// the other.
__global__ void ada_rms_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ mod,
                                    __nv_bfloat16* __restrict__ out,
                                    int dim, float eps) {
    const __nv_bfloat16* xr = x + (long long)blockIdx.x * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = __bfloat162float(xr[i]);
        ss += v * v;
    }
    const float inv = rsqrtf(__fdiv_rn(v8_block_sum(ss), (float)dim) + eps);
    __nv_bfloat16* orow = out + (long long)blockIdx.x * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float n = __bfloat162float(xr[i]) * inv;
        const float sc = __bfloat162float(mod[i]), sh = __bfloat162float(mod[dim + i]);
        orow[i] = __float2bfloat16(n * (1.0f + sc) + sh);
    }
}

// Gemma3 RMSNorm over contiguous rows, 8 columns per thread, bf16 output.
//
// This is the *same formula* as gemma3_rms_norm_bf16 above and a different
// summation order: it loads 8 columns per thread and reduces warp-first through
// shared memory, where that one runs a strided loop and reduces inline. The two
// can disagree in the last fp32 ulp of the sum of squares, which for a bf16
// result means an element landing either side of a rounding boundary. A caller
// reproducing one of them bit for bit has to use that one -- a caller whose
// fused norm+RoPE kernel uses this order cannot substitute the other.
//
// One block per row, dim / 8 threads; dim must be a multiple of 8. `residual`,
// when non-null, is updated with HF's two roundings as in the strided form.
__global__ void gemma3_rms_norm_v8_kernel(const __nv_bfloat16* __restrict__ x, long long x_stride,
                                          const __nv_bfloat16* __restrict__ w,
                                          __nv_bfloat16* __restrict__ out, long long out_stride,
                                          __nv_bfloat16* __restrict__ residual, long long res_stride,
                                          int rows_per_group, long long group_stride,
                                          int dim, float eps) {
    union Gemma3V8N { uint4 u; __nv_bfloat16 h[8]; };
    const int r = blockIdx.x;
    const long long base = (long long)(r / rows_per_group) * group_stride +
                           (long long)(r % rows_per_group) * x_stride + threadIdx.x * 8;
    Gemma3V8N xv, wv;
    xv.u = *reinterpret_cast<const uint4*>(x + base);
    wv.u = *reinterpret_cast<const uint4*>(w + threadIdx.x * 8);
    float f[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = __bfloat162float(xv.h[j]);
        ss += f[j] * f[j];
    }
    const float inv = rsqrtf(__fdiv_rn(v8_block_sum(ss), (float)dim) + eps);
    Gemma3V8N yv;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        yv.h[j] = __float2bfloat16(f[j] * inv * (1.0f + __bfloat162float(wv.h[j])));
    if (residual) {
        const long long rb = (long long)(r / rows_per_group) * group_stride +
                             (long long)(r % rows_per_group) * res_stride + threadIdx.x * 8;
        Gemma3V8N rv;
        rv.u = *reinterpret_cast<const uint4*>(residual + rb);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            rv.h[j] = __float2bfloat16(__bfloat162float(rv.h[j]) + __bfloat162float(yv.h[j]));
        *reinterpret_cast<uint4*>(residual + rb) = rv.u;
    }
    if (out) {
        const long long ob = (long long)(r / rows_per_group) * group_stride +
                             (long long)(r % rows_per_group) * out_stride + threadIdx.x * 8;
        *reinterpret_cast<uint4*>(out + ob) = yv.u;
    }
}

// The block reduction this uses is v8_block_sum, so its sum-of-squares order is
// the vectorized one; gemma3_rms_norm_bf16 above uses the strided loop's. Same
// addressing as that kernel -- rows_per_group rows of x_stride inside each
// group, groups group_stride apart, which is what lets a per-head norm run in
// place on a fused QKV buffer -- so the two differ only in reduction order and
// a caller picks the one its reference used.
// Gemma3 RMSNorm followed by a gated residual add, 8 columns per thread:
//
//   residual = bf16( residual + bf16( bf16(gemma3_norm(x)) * gate ) )
//
// Three roundings, one per store HF performs on this path: the norm rounds, the
// gate multiply rounds, and the residual add rounds. elementwise.cu's
// gate_mul_residual is the same shape of operation but takes `x` directly with a
// single rounding at the end -- it has no norm and no intermediate rounds, so it
// is not a substitute for this.
//
// The reduction is the vectorized one (v8_block_sum), which is what the caller's
// fused norm+gate+residual kernel uses; composing this with ada_rms_norm
// reproduces that fused kernel, and using the strided-order norm instead does
// not.
//
// Contiguous rows, one block per row, dim / 8 threads; dim must be a multiple
// of 8. `x` and `residual` are (rows, dim); `w` and `gate` are per-column, one
// dim-wide vector shared by every row -- the gate is a modulation, not a
// per-row mask. `residual` is read and written.
__global__ void gemma3_norm_gate_residual_v8_kernel(const __nv_bfloat16* __restrict__ x,
                                                    const __nv_bfloat16* __restrict__ w,
                                                    const __nv_bfloat16* __restrict__ gate,
                                                    __nv_bfloat16* __restrict__ residual,
                                                    int dim, float eps) {
    union Gemma3Ngr { uint4 u; __nv_bfloat16 h[8]; };
    const long long base = (long long)blockIdx.x * dim + threadIdx.x * 8;
    Gemma3Ngr xv, wv, gv, rv;
    xv.u = *reinterpret_cast<const uint4*>(x + base);
    wv.u = *reinterpret_cast<const uint4*>(w + threadIdx.x * 8);
    gv.u = *reinterpret_cast<const uint4*>(gate + threadIdx.x * 8);
    rv.u = *reinterpret_cast<const uint4*>(residual + base);
    float f[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = __bfloat162float(xv.h[j]);
        ss += f[j] * f[j];
    }
    const float inv = rsqrtf(__fdiv_rn(v8_block_sum(ss), (float)dim) + eps);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float y = __bfloat162float(__float2bfloat16(f[j] * inv * (1.0f + __bfloat162float(wv.h[j]))));
        const float t = __bfloat162float(__float2bfloat16(y * __bfloat162float(gv.h[j])));
        rv.h[j] = __float2bfloat16(__bfloat162float(rv.h[j]) + t);
    }
    *reinterpret_cast<uint4*>(residual + base) = rv.u;
}

// Gemma3 gate + residual + adaLN norm, fused, bf16 out with the residual updated
// in place. This is the reference's own fusion and it has to be one kernel: the
// adaLN tail's own RMSNorm is taken over the updated residual with the same
// 8-columns-per-thread reduction the first half uses, and `dim / 8` threads. Going
// through the standalone ada_rms_norm entry instead reduces the same sum in the
// strided order over a different block size, which moves about one element in
// ten thousand across a bf16 rounding boundary -- a difference that survives the
// whole action-expert loop.
//
//   y        = bf16(bf16(x * rsqrt(mean(x^2)+eps)) * (1 + w))
//   residual = bf16(residual + bf16(y * gate))
//   out      = bf16(residual * rsqrt(mean(residual^2)+eps) * (1 + mod[0:dim]) + mod[dim:2dim])
//
// with the two means reduced by v8_block_sum; the modulated norm has no weight,
// as the reference's does not. `mod` is 2*dim wide here, not 3*dim: the gate is a
// separate argument because the reference takes it from a different modulation.
__global__ void gemma3_norm_gate_residual_ada_v8_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
        const __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ residual,
        const __nv_bfloat16* __restrict__ mod, __nv_bfloat16* __restrict__ out,
        int dim, float eps) {
    union Gemma3Ngra { uint4 u; __nv_bfloat16 h[8]; };
    const long long base = (long long)blockIdx.x * dim + threadIdx.x * 8;
    const int c = threadIdx.x * 8;
    Gemma3Ngra xv, wv, gv, rv, sv, hv;
    xv.u = *reinterpret_cast<const uint4*>(x + base);
    wv.u = *reinterpret_cast<const uint4*>(w + c);
    gv.u = *reinterpret_cast<const uint4*>(gate + c);
    rv.u = *reinterpret_cast<const uint4*>(residual + base);
    sv.u = *reinterpret_cast<const uint4*>(mod + c);
    hv.u = *reinterpret_cast<const uint4*>(mod + dim + c);
    float f[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = __bfloat162float(xv.h[j]);
        ss += f[j] * f[j];
    }
    const float inv = rsqrtf(__fdiv_rn(v8_block_sum(ss), (float)dim) + eps);
    float r[8], ss2 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float y = __bfloat162float(__float2bfloat16(f[j] * inv * (1.0f + __bfloat162float(wv.h[j]))));
        const float t = __bfloat162float(__float2bfloat16(y * __bfloat162float(gv.h[j])));
        rv.h[j] = __float2bfloat16(__bfloat162float(rv.h[j]) + t);
        r[j] = __bfloat162float(rv.h[j]);
        ss2 += r[j] * r[j];
    }
    *reinterpret_cast<uint4*>(residual + base) = rv.u;
    const float inv2 = rsqrtf(__fdiv_rn(v8_block_sum(ss2), (float)dim) + eps);
    Gemma3Ngra ov;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        ov.h[j] = __float2bfloat16(r[j] * inv2 * (1.0f + __bfloat162float(sv.h[j]))
                                   + __bfloat162float(hv.h[j]));
    *reinterpret_cast<uint4*>(out + base) = ov.u;
}

void gemma3_norm_gate_residual_ada_v8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                           const __nv_bfloat16* gate, __nv_bfloat16* residual,
                                           const __nv_bfloat16* mod, __nv_bfloat16* out,
                                           int rows, int dim, float eps, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    if (!x || !weight || !gate || !residual || !mod || !out)
        throw std::runtime_error("gemma3_norm_gate_residual_ada_v8_bf16: null pointer argument");
    if (dim % 8 != 0)
        throw std::runtime_error("gemma3_norm_gate_residual_ada_v8_bf16: dim must be a multiple of 8");
    gemma3_norm_gate_residual_ada_v8_kernel<<<rows, dim / 8, 0, stream>>>(
        x, weight, gate, residual, mod, out, dim, eps);
}

void gemma3_norm_gate_residual_v8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                       const __nv_bfloat16* gate, __nv_bfloat16* residual,
                                       int rows, int dim, float eps, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    if (!x || !weight || !gate || !residual)
        throw std::runtime_error("gemma3_norm_gate_residual_v8_bf16: null pointer argument");
    if (dim % 8 != 0)
        throw std::runtime_error("gemma3_norm_gate_residual_v8_bf16: dim must be a multiple of 8");
    gemma3_norm_gate_residual_v8_kernel<<<rows, dim / 8, 0, stream>>>(x, weight, gate, residual, dim, eps);
}

// The conversions below match quantize.cu's; the fused form has to write the
// same scale factors and packed bytes as the unfused chain it replaces.
__device__ __forceinline__ uint8_t gemma3_fp4_e2m1(float v) {
    const float a = fabsf(v);
    const uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    return sign | (uint8_t)((a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) +
                            (a >= 2.5f) + (a >= 3.5f) + (a >= 5.0f));
}

__device__ __forceinline__ uint8_t gemma3_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    if (v > 240.0f) return 0xFE;
    const uint32_t bits = __float_as_uint(v);
    const int float_exp = ((bits >> 23) & 0xFF) - 127;
    const uint32_t frac = bits & 0x7FFFFF;
    int ue_exp = float_exp + 7;
    if (ue_exp <= 0) {
        int m = (int)ceilf(v * 512.0f);
        if (m > 7) return (1 << 3) | 0;
        if (m < 1) m = 1;
        return (uint8_t)m;
    }
    if (ue_exp >= 15) return 0xFE;
    int m = (int)(frac >> 20);
    if (frac & 0xFFFFF) m++;
    if (m >= 8) { m = 0; ue_exp++; }
    if (ue_exp >= 15) return 0xFE;
    return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float gemma3_ue4m3_to_float(uint8_t v) {
    const int e = (v >> 3) & 0xF, m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// Gemma3 RMSNorm whose bf16 result is written straight into a clipped static
// NVFP4 GEMM input: e2m1 values per 16-element group with a UE4M3 block scale
// in the CUTLASS sm120 swizzled layout, the group's amax clamped to `cap`
// before the scale is formed.
//
// Same encoding as quantize_bf16_to_nvfp4_swizzled_clipped, so this equals that
// quantizer run over the bf16 norm output. Two adjacent threads own one
// 16-element group: each takes the amax of its 8 columns, they exchange it, and
// the even thread writes the shared scale factor. One block per row,
// dim / 8 threads; dim must be a multiple of 16 and at most 8192, and the block
// must have at least 32 threads for the pair exchange to be defined.
__global__ void gemma3_rms_norm_nvfp4_kernel(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ w,
                                             const float* __restrict__ cap,
                                             uint8_t* __restrict__ q,
                                             uint8_t* __restrict__ sf,
                                             int dim, float eps) {
    union Gemma3Nv { uint4 u; __nv_bfloat16 h[8]; };
    const int row = blockIdx.x;
    const int nb = dim / 16, ncb = (nb + 3) / 4;
    const long long base = (long long)row * dim + threadIdx.x * 8;
    Gemma3Nv xv, wv;
    xv.u = *reinterpret_cast<const uint4*>(x + base);
    wv.u = *reinterpret_cast<const uint4*>(w + threadIdx.x * 8);
    float f[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = __bfloat162float(xv.h[j]);
        ss += f[j] * f[j];
    }
    const float inv = rsqrtf(__fdiv_rn(v8_block_sum(ss), (float)dim) + eps);
    float y[8], a = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        y[j] = __bfloat162float(__float2bfloat16(f[j] * inv * (1.0f + __bfloat162float(wv.h[j]))));
        a = fmaxf(a, fabsf(y[j]));
    }
    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
    const int b = threadIdx.x >> 1;
    float amax = a;
    const float clip = cap ? cap[b] : 0.0f;
    if (clip > 0.0f && isfinite(clip)) amax = fminf(amax, clip);
    const uint8_t ue = gemma3_ue4m3_ceil(amax / 6.0f);
    const int rb = row / 128, ri = row % 128;
    if ((threadIdx.x & 1) == 0)
        sf[(rb * ncb + b / 4) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + b % 4] = ue;
    const float scale = gemma3_ue4m3_to_float(ue);
    const float invs = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    uint32_t r = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t lo = gemma3_fp4_e2m1(y[2 * j] * invs);
        const uint8_t hi = gemma3_fp4_e2m1(y[2 * j + 1] * invs);
        r |= (uint32_t)(((hi << 4) | (lo & 0x0F)) & 0xFF) << (8 * j);
    }
    reinterpret_cast<uint32_t*>(q + (long long)row * (dim / 2))[threadIdx.x] = r;
}

// Bytes the caller must allocate for the scale factors.
long long gemma3_rms_norm_nvfp4_sf_bytes(int rows, int dim) {
    if (rows <= 0 || dim <= 0) return 0;
    return (long long)((rows + 127) / 128) * (((dim / 16) + 3) / 4) * 512;
}

void gemma3_rms_norm_nvfp4_swizzled(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                    const float* cap, uint8_t* packed, uint8_t* sf_swz,
                                    int rows, int dim, float eps, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    if (!x || !weight || !packed || !sf_swz)
        throw std::runtime_error("gemma3_rms_norm_nvfp4_swizzled: null pointer argument");
    if (dim % 16 != 0)
        throw std::runtime_error("gemma3_rms_norm_nvfp4_swizzled: dim must be a multiple of 16");
    if (dim / 8 > 1024)
        throw std::runtime_error("gemma3_rms_norm_nvfp4_swizzled: dim / 8 must not exceed 1024");
    gemma3_rms_norm_nvfp4_kernel<<<rows, dim / 8, 0, stream>>>(x, weight, cap, packed, sf_swz, dim, eps);
}

void gemma3_rms_norm_v8_bf16(const __nv_bfloat16* x, long long x_stride,
                             const __nv_bfloat16* weight,
                             __nv_bfloat16* out, long long out_stride,
                             __nv_bfloat16* residual, long long res_stride,
                             int rows, int rows_per_group, long long group_stride,
                             int dim, float eps, cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    if (!x || !weight || (!out && !residual))
        throw std::runtime_error("gemma3_rms_norm_v8_bf16: null pointer, or no destination");
    if (rows_per_group <= 0)
        throw std::runtime_error("gemma3_rms_norm_v8_bf16: rows_per_group must be positive");
    if (dim % 8 != 0)
        throw std::runtime_error("gemma3_rms_norm_v8_bf16: dim must be a multiple of 8");
    gemma3_rms_norm_v8_kernel<<<rows, dim / 8, 0, stream>>>(
        x, x_stride, weight, out, out_stride, residual, res_stride,
        rows_per_group, group_stride, dim, eps);
}

// (1+w) RMSNorm followed by HF rotate-half RoPE, fused over q and the cached k rows
// in one pass, head_dim 256. One warp per (row, head); lane l holds elements 8l..8l+7,
// so the rotate-half partner (j +- 128) sits in lane l ^ 16 and needs no extra traffic.
//
// The reference emits this as a single kernel. Composing it from two
// gemma3_rms_norm_v8_bf16 plus two rope_hf_bf16 calls is bit-identical but costs four
// launches and writes then re-reads every q and k; measured at +1191 launches and
// +4.8 ms end to end over a full predict, which is the whole of the port's regression.
//
// Per-element rounding matches the reference exactly: y = bf(f * inv * (1 + w)), then
// out = __float2bfloat16(bf(y*cos) + bf(+-yp*sin)). HD is 256, a power of two, so
// `ss / (float)HD` is exact even under --prec-div=false and needs no __fdiv_rn.
__global__ void gemma3_qk_norm_rope_v8_kernel(
        __nv_bfloat16* __restrict__ q, long long q_stride, int nhq,
        __nv_bfloat16* __restrict__ k, long long k_stride, int nhk,
        const __nv_bfloat16* __restrict__ wq, const __nv_bfloat16* __restrict__ wk,
        const __nv_bfloat16* __restrict__ cosT, const __nv_bfloat16* __restrict__ sinT,
        int seq, float eps) {
    constexpr int HD = 256;
    union Gemma3QkRope { uint4 u; __nv_bfloat16 h[8]; };
    const int lane = threadIdx.x & 31;
    const long long vec = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const long long nq = (long long)seq * nhq, total = nq + (long long)seq * nhk;
    if (vec >= total) return;
    const bool isq = vec < nq;
    const long long v = isq ? vec : vec - nq;
    const int nh = isq ? nhq : nhk;
    const int s = (int)(v / nh), h = (int)(v % nh);
    __nv_bfloat16* x = (isq ? q + (long long)s * q_stride : k + (long long)s * k_stride)
                       + (long long)h * HD;
    const int c = lane * 8;
    Gemma3QkRope xv, wv, cv, sv;
    xv.u = *reinterpret_cast<const uint4*>(x + c);
    wv.u = *reinterpret_cast<const uint4*>((isq ? wq : wk) + c);
    cv.u = *reinterpret_cast<const uint4*>(cosT + (long long)s * HD + c);
    sv.u = *reinterpret_cast<const uint4*>(sinT + (long long)s * HD + c);
    float f[8], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) { f[j] = __bfloat162float(xv.h[j]); ss += f[j] * f[j]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss / (float)HD + eps);
    const bool lo = lane < 16;
    Gemma3QkRope ov;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float y = gemma3_bf(f[j] * inv * (1.0f + __bfloat162float(wv.h[j])));
        const float yp = __shfl_xor_sync(0xffffffffu, y, 16);
        const float cc = __bfloat162float(cv.h[j]), sn = __bfloat162float(sv.h[j]);
        ov.h[j] = __float2bfloat16(gemma3_bf(y * cc) + (lo ? gemma3_bf(-yp * sn) : gemma3_bf(yp * sn)));
    }
    *reinterpret_cast<uint4*>(x + c) = ov.u;
}

void gemma3_qk_norm_rope_v8_bf16(__nv_bfloat16* q, long long q_stride, int nhq,
                                 __nv_bfloat16* k, long long k_stride, int nhk,
                                 const __nv_bfloat16* wq, const __nv_bfloat16* wk,
                                 const __nv_bfloat16* cosT, const __nv_bfloat16* sinT,
                                 int seq, float eps, cudaStream_t stream) {
    if (seq <= 0 || nhq <= 0 || nhk <= 0) return;
    const long long vecs = (long long)seq * (nhq + nhk);
    const int wpb = 4;                                  // 4 warps = 128 threads per block
    gemma3_qk_norm_rope_v8_kernel<<<(int)((vecs + wpb - 1) / wpb), wpb * 32, 0, stream>>>(
        q, q_stride, nhq, k, k_stride, nhk, wq, wk, cosT, sinT, seq, eps);
}

// residual add + LayerNorm + static FP8 quantize in a single pass -- the reference
// emits this as one kernel, and composing it from bias_residual + layer_norm +
// quantize_fp8_static costs three launches and forces the normalised values through
// DRAM in a scratch the reference never writes (4.7 MB per call, 54 calls per predict).
//
// Two details are load-bearing and neither is cosmetic:
//
//   * rcp.approx.ftz.f32 and mul.rn.ftz.f32 are pinned in the reference, so they are
//     pinned here. FlashRT has no equivalent and `1.0f / x` is not the same operation.
//   * the reduction reproduces frt_block_sum exactly (shfl_down 16..1, then warp 0 over
//     the per-warp sums). A different tree moves mean/inv_std by an ulp, which is
//     precisely what made this op fail the 444 gate before the division sites were
//     pinned -- the two are the same class of error.
//
// `dim / 2 <= 256 * MAXP`; D=1152 needs np <= 3, so MAXP=4 covers it. The residual and
// normalised values stay in v0/v1 registers across the two passes, which is what lets
// xn stay untouched when the caller passes 0.
namespace {
__device__ __forceinline__ float dm05_warp_sum(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 16);
    v += __shfl_down_sync(0xffffffffu, v, 8);
    v += __shfl_down_sync(0xffffffffu, v, 4);
    v += __shfl_down_sync(0xffffffffu, v, 2);
    v += __shfl_down_sync(0xffffffffu, v, 1);
    return v;
}
__device__ __forceinline__ float dm05_block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = dm05_warp_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) v = dm05_warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float dm05_rcp(float s) {
    float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s)); return r;
}
__device__ __forceinline__ float dm05_mul(float a, float b) {
    float r; asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
}  // namespace

template <int MAXP>
__global__ void __launch_bounds__(256)
res_ln_quant_v8_bf16_kernel(__nv_bfloat16* __restrict__ r, const __nv_bfloat16* __restrict__ y,
                            const __nv_bfloat16* __restrict__ b, const __nv_bfloat16* __restrict__ w,
                            const __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ xn,
                            __nv_fp8_e4m3* __restrict__ q, const float* __restrict__ scale,
                            int dim, float eps) {
    __shared__ float sh[32];
    const int row = blockIdx.x, dim2 = dim >> 1;
    __nv_bfloat162* r2 = reinterpret_cast<__nv_bfloat162*>(r + (long long)row * dim);
    float v0[MAXP], v1[MAXP];
    int np = 0;
    float ls = 0.f;
    for (int i = threadIdx.x; i < dim2; i += 256, ++np) {
        __nv_bfloat162 rv = r2[i];
        float a0 = __bfloat162float(rv.x), a1 = __bfloat162float(rv.y);
        if (y) {
            const __nv_bfloat162 yv = reinterpret_cast<const __nv_bfloat162*>(y + (long long)row * dim)[i];
            const __nv_bfloat162 bv = reinterpret_cast<const __nv_bfloat162*>(b)[i];
            a0 = gemma3_bf(a0 + __bfloat162float(yv.x) + __bfloat162float(bv.x));
            a1 = gemma3_bf(a1 + __bfloat162float(yv.y) + __bfloat162float(bv.y));
            r2[i] = __floats2bfloat162_rn(a0, a1);
        }
        v0[np] = a0; v1[np] = a1;
        ls += a0 + a1;
    }
    const float mean = __fdiv_rn(dm05_block_sum(ls, sh), (float)dim);
    float lv = 0.f;
    for (int j = 0; j < np; ++j) {
        const float d0 = v0[j] - mean, d1 = v1[j] - mean;
        lv += d0 * d0 + d1 * d1;
    }
    const float inv_std = rsqrtf(__fdiv_rn(dm05_block_sum(lv, sh), (float)dim) + eps);
    const float inv_s = q ? dm05_rcp(scale[0]) : 0.f;
    int j = 0;
    for (int i = threadIdx.x; i < dim2; i += 256, ++j) {
        const __nv_bfloat162 wv = reinterpret_cast<const __nv_bfloat162*>(w)[i];
        const __nv_bfloat162 bb = reinterpret_cast<const __nv_bfloat162*>(beta)[i];
        const float o0 = gemma3_bf((v0[j] - mean) * inv_std * __bfloat162float(wv.x) + __bfloat162float(bb.x));
        const float o1 = gemma3_bf((v1[j] - mean) * inv_std * __bfloat162float(wv.y) + __bfloat162float(bb.y));
        if (xn) reinterpret_cast<__nv_bfloat162*>(xn + (long long)row * dim)[i] = __floats2bfloat162_rn(o0, o1);
        if (q) {
            __nv_fp8_e4m3* qr = q + (long long)row * dim + 2 * i;
            qr[0] = __nv_fp8_e4m3(fminf(fmaxf(dm05_mul(o0, inv_s), -448.f), 448.f));
            qr[1] = __nv_fp8_e4m3(fminf(fmaxf(dm05_mul(o1, inv_s), -448.f), 448.f));
        }
    }
}

void res_ln_quant_v8_bf16(__nv_bfloat16* r, const __nv_bfloat16* y, const __nv_bfloat16* b,
                          const __nv_bfloat16* w, const __nv_bfloat16* beta, __nv_bfloat16* xn,
                          __nv_fp8_e4m3* q, const float* scale, int rows, int dim, float eps,
                          cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    if (dim % 2) throw std::runtime_error("res_ln_quant_v8_bf16: dim must be even");
    if (dim / 2 > 256 * 4) throw std::runtime_error("res_ln_quant_v8_bf16: dim too large for MAXP=4");
    res_ln_quant_v8_bf16_kernel<4><<<rows, 256, 0, stream>>>(r, y, b, w, beta, xn, q, scale, dim, eps);
}

void ada_rms_norm(const __nv_bfloat16* x, const __nv_bfloat16* mod,
                  __nv_bfloat16* out, int rows, int dim, float eps,
                  cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    const int threads = dim >= 1024 ? 1024 : (dim >= 256 ? 256 : 128);
    ada_rms_norm_kernel<<<rows, threads, 0, stream>>>(x, mod, out, dim, eps);
}

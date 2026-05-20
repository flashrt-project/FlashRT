// ================================================================
// FlashRT — Cross-layer fusion kernels (dtype-generic)
// Fused gate*residual + AdaRMSNorm -> FP8 (decoder optimization)
// Supports: __half (FP16), __nv_bfloat16 (BF16) via templates
// ================================================================

#include "fusion.cuh"
#include "common.cuh"

// ── Fused Gate*Residual + AdaRMSNorm + Style -> FP8 ──
// Pass 1: residual += x * gate; sum_sq += residual^2
// Reduce: rms = rsqrt(sum_sq / dim + eps)
// Pass 2: out_fp8 = clamp((norm(residual) * (1+scale) + shift) * inv_scale)
template<typename T>
__global__ void gate_residual_ada_norm_fp8_kernel(
    T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    const T* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ out,
    T* __restrict__ gate_out,
    int dim, float eps,
    const float* __restrict__ d_scale) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    __nv_fp8_e4m3* out_row = out + row * dim;
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];

    // Pass 1: residual += x * gate, compute sum of squares
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    // Pass 2: AdaRMSNorm with style -> FP8
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __nv_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __nv_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate_out2[i] = gv;
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_fp8_kernel<__half>(
    __half*, const __half*, const __half*, const __half*, const __half*,
    __nv_fp8_e4m3*, __half*, int, float, const float*))
FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_fp8_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_fp8_e4m3*, __nv_bfloat16*, int, float, const float*))
void gate_residual_ada_norm_fp8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 __nv_fp8_e4m3* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 const float* d_scale, cudaStream_t stream) {
    gate_residual_ada_norm_fp8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
}
void gate_residual_ada_norm_fp8_fp16(__half* residual, const __half* x,
                                      const __half* gate, const __half* weight,
                                      const __half* style,
                                      __nv_fp8_e4m3* out, __half* gate_out,
                                      int seq_len, int dim, float eps,
                                      const float* d_scale, cudaStream_t stream) {
    gate_residual_ada_norm_fp8_kernel<__half><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
}

// ── Fused Gate*Residual + AdaRMSNorm + Style -> FP16 ──
// Same math as gate_mul_residual_fp16 + ada_rms_norm_style_fp16, but
// keeps the residual update and RMS reduction in one launch.
__global__ void gate_residual_ada_norm_fp16_kernel(
    __half* __restrict__ residual,
    const __half* __restrict__ x,
    const __half* __restrict__ gate,
    const __half* __restrict__ weight,
    const __half* __restrict__ style,
    __half* __restrict__ out,
    __half* __restrict__ gate_out,
    int dim, float eps) {
    using T2 = typename packed2<__half>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const __half* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        __half rh0 = from_f32<__half>(r0);
        __half rh1 = from_f32<__half>(r1);
        res2[i] = make_packed2<__half>(rh0, rh1);
        float rr0 = to_f32(rh0);
        float rr1 = to_f32(rh1);
        local_sum += rr0 * rr0 + rr1 * rr1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        out2[i] = make_packed2<__half>(
            from_f32<__half>(n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)),
            from_f32<__half>(n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)));
        gate_out2[i] = gv;
    }
}

void gate_residual_ada_norm_fp16(__half* residual, const __half* x,
                                  const __half* gate, const __half* weight,
                                  const __half* style,
                                  __half* out, __half* gate_out,
                                  int seq_len, int dim, float eps,
                                  cudaStream_t stream) {
    gate_residual_ada_norm_fp16_kernel<<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps);
}

template<typename T>
__global__ void gate_residual_ada_norm_int8_kernel(
    T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    const T* __restrict__ style,
    int8_t* __restrict__ out,
    T* __restrict__ gate_out,
    int dim, float eps,
    float* __restrict__ d_scales) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    int8_t* out_row = out + row * dim;
    T2* gate_out2 = reinterpret_cast<T2*>(gate_out + row * dim);
    int dim2 = dim >> 1;

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
        local_sum += r0 * r0 + r1 * r1;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    float local_amax = 0.0f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
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
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        int q0 = __float2int_rn(val0);
        int q1 = __float2int_rn(val1);
        out_row[2 * i] = static_cast<int8_t>((q0 < -127) ? -127 : ((q0 > 127) ? 127 : q0));
        out_row[2 * i + 1] = static_cast<int8_t>((q1 < -127) ? -127 : ((q1 > 127) ? 127 : q1));
        gate_out2[i] = gv;
    }
}

FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_int8_kernel<__half>(
    __half*, const __half*, const __half*, const __half*, const __half*,
    int8_t*, __half*, int, float, float*))
FVK_KERNEL_INSTANTIATE(__global__ void gate_residual_ada_norm_int8_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    int8_t*, __nv_bfloat16*, int, float, float*))
void gate_residual_ada_norm_int8(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* gate, const __nv_bfloat16* weight,
                                 const __nv_bfloat16* style,
                                 int8_t* out, __nv_bfloat16* gate_out,
                                 int seq_len, int dim, float eps,
                                 float* d_scales, cudaStream_t stream) {
    gate_residual_ada_norm_int8_kernel<__nv_bfloat16><<<seq_len, 256, 256 * sizeof(float), stream>>>(
        residual, x, gate, weight, style, out, gate_out, dim, eps, d_scales);
}

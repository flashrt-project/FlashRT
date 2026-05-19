// ================================================================
// FlashRT — Elementwise kernels (dtype-generic)
// Residual add, gate multiply, bias residual
// Supports: __half (FP16), __nv_bfloat16 (BF16) via templates
// ================================================================

#include "elementwise.cuh"
#include "common.cuh"
#include "fp8_numeric_conversion.cuh"
#include <math_constants.h>

// ── Gate Multiply + Residual ──
template<typename T>
__global__ void gate_mul_res_kernel(T* __restrict__ residual,
                                    const T* __restrict__ x,
                                    const T* __restrict__ gate, int n) {
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual);
    const T2* x2 = reinterpret_cast<const T2*>(x);
    const T2* g2 = reinterpret_cast<const T2*>(gate);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 rv = res2[idx], xv = x2[idx], gv = g2[idx];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        res2[idx] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void gate_mul_res_kernel<__half>(__half*, const __half*, const __half*, int);
template __global__ void gate_mul_res_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int);

void gate_mul_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                       const __nv_bfloat16* gate, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    gate_mul_res_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, gate, n);
}
void gate_mul_residual_fp16(__half* residual, const __half* x,
                            const __half* gate, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    gate_mul_res_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, gate, n);
}

__global__ void ncdhw_to_blc_bf16_kernel(
        const __nv_bfloat16* __restrict__ x,
        __nv_bfloat16* __restrict__ out,
        int B, int C, int T, int H, int W) {
    int64_t total = static_cast<int64_t>(B) * T * H * W * C;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int S = T * H * W;
    for (; idx < total; idx += stride) {
        int c = static_cast<int>(idx % C);
        int s = static_cast<int>((idx / C) % S);
        int b = static_cast<int>(idx / (static_cast<int64_t>(S) * C));
        int w = s % W;
        int h = (s / W) % H;
        int t = s / (H * W);
        int64_t src = (((static_cast<int64_t>(b) * C + c) * T + t) * H + h) * W + w;
        out[idx] = x[src];
    }
}

void ncdhw_to_blc_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                       int B, int C, int T, int H, int W,
                       cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(B) * C * T * H * W;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 4096) blocks = 4096;
    ncdhw_to_blc_bf16_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, T, H, W);
}

template<typename T>
__global__ void gate_mul_res_out_kernel(const T* __restrict__ residual,
                                        const T* __restrict__ x,
                                        const T* __restrict__ gate,
                                        T* __restrict__ out, int n) {
    using T2 = typename packed2<T>::type;
    const T2* res2 = reinterpret_cast<const T2*>(residual);
    const T2* x2 = reinterpret_cast<const T2*>(x);
    const T2* g2 = reinterpret_cast<const T2*>(gate);
    T2* out2 = reinterpret_cast<T2*>(out);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 rv = res2[idx], xv = x2[idx], gv = g2[idx];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        out2[idx] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void gate_mul_res_out_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int);

void gate_mul_residual_out_bf16(const __nv_bfloat16* residual,
                                const __nv_bfloat16* x,
                                const __nv_bfloat16* gate,
                                __nv_bfloat16* out, int n,
                                cudaStream_t stream) {
    int n2 = n >> 1;
    gate_mul_res_out_kernel<__nv_bfloat16>
        <<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, gate, out, n);
}

template<typename T>
__global__ void gate_mul_res_out_g1d_kernel(
    const T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ gate_1d,
    T* __restrict__ out,
    int dim) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* res2 = reinterpret_cast<const T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate_1d);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
        out2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void gate_mul_res_out_g1d_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int);

void gate_mul_residual_out_bf16_g1d(const __nv_bfloat16* residual,
                                    const __nv_bfloat16* x,
                                    const __nv_bfloat16* gate_1d,
                                    __nv_bfloat16* out,
                                    int seq_len, int dim,
                                    cudaStream_t stream) {
    gate_mul_res_out_g1d_kernel<__nv_bfloat16>
        <<<seq_len, 256, 0, stream>>>(residual, x, gate_1d, out, dim);
}

__global__ void gate_mul_res_out_gate_fp8_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_fp8_e4m3* __restrict__ gate,
    const float* __restrict__ gate_scale,
    __nv_bfloat16* __restrict__ out,
    int n) {
    using T2 = packed2<__nv_bfloat16>::type;
    const T2* res2 = reinterpret_cast<const T2*>(residual);
    const T2* x2 = reinterpret_cast<const T2*>(x);
    T2* out2 = reinterpret_cast<T2*>(out);
    int n2 = n >> 1;
    float gs = *gate_scale;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n2; idx += blockDim.x * gridDim.x) {
        T2 rv = res2[idx], xv = x2[idx];
        int j = idx << 1;
        float g0 = static_cast<float>(gate[j]) * gs;
        float g1 = static_cast<float>(gate[j + 1]) * gs;
        float r0 = to_f32(rv.x) + to_f32(xv.x) * g0;
        float r1 = to_f32(rv.y) + to_f32(xv.y) * g1;
        out2[idx] = make_packed2<__nv_bfloat16>(
            from_f32<__nv_bfloat16>(r0), from_f32<__nv_bfloat16>(r1));
    }
}

void gate_mul_residual_out_bf16_gate_fp8(const __nv_bfloat16* residual,
                                         const __nv_bfloat16* x,
                                         const __nv_fp8_e4m3* gate,
                                         const float* gate_scale,
                                         __nv_bfloat16* out, int n,
                                         cudaStream_t stream) {
    int n2 = n >> 1;
    gate_mul_res_out_gate_fp8_kernel
        <<<(n2 + 255) / 256, 256, 0, stream>>>(
            residual, x, gate, gate_scale, out, n);
}

// ── Bias + Residual ──
template<typename T>
__global__ void bias_res_kernel(T* __restrict__ residual,
                                const T* __restrict__ x,
                                const T* __restrict__ bias,
                                int dim) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], bv = b2[i];
        float r0 = to_f32(rv.x) + to_f32(xv.x) + to_f32(bv.x);
        float r1 = to_f32(rv.y) + to_f32(xv.y) + to_f32(bv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void bias_res_kernel<__half>(__half*, const __half*, const __half*, int);
template __global__ void bias_res_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, int);

void bias_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                   const __nv_bfloat16* bias, int seq_len, int dim,
                   cudaStream_t stream) {
    bias_res_kernel<__nv_bfloat16><<<seq_len, 256, 0, stream>>>(residual, x, bias, dim);
}
void bias_residual_fp16(__half* residual, const __half* x,
                        const __half* bias, int seq_len, int dim,
                        cudaStream_t stream) {
    bias_res_kernel<__half><<<seq_len, 256, 0, stream>>>(residual, x, bias, dim);
}

template<typename T>
__global__ void bias_res_out_kernel(const T* __restrict__ residual,
                                    const T* __restrict__ x,
                                    const T* __restrict__ bias,
                                    T* __restrict__ out,
                                    int dim) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* res2 = reinterpret_cast<const T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], bv = b2[i];
        float xb0 = to_f32(from_f32<T>(to_f32(xv.x) + to_f32(bv.x)));
        float xb1 = to_f32(from_f32<T>(to_f32(xv.y) + to_f32(bv.y)));
        float r0 = to_f32(rv.x) + xb0;
        float r1 = to_f32(rv.y) + xb1;
        out2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void bias_res_out_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int);

void bias_residual_out_bf16(const __nv_bfloat16* residual,
                            const __nv_bfloat16* x,
                            const __nv_bfloat16* bias,
                            __nv_bfloat16* out,
                            int seq_len, int dim,
                            cudaStream_t stream) {
    bias_res_out_kernel<__nv_bfloat16><<<seq_len, 256, 0, stream>>>(
        residual, x, bias, out, dim);
}

// ── Residual Add ──
template<typename T>
__global__ void res_add_kernel(T* __restrict__ residual,
                               const T* __restrict__ x, int n) {
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual);
    const T2* x2 = reinterpret_cast<const T2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 rv = res2[idx], xv = x2[idx];
        res2[idx] = make_packed2<T>(
            from_f32<T>(to_f32(rv.x) + to_f32(xv.x)),
            from_f32<T>(to_f32(rv.y) + to_f32(xv.y)));
    }
}

template __global__ void res_add_kernel<__half>(__half*, const __half*, int);
template __global__ void res_add_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, int);

void residual_add(__nv_bfloat16* residual, const __nv_bfloat16* x, int n,
                  cudaStream_t stream) {
    int n2 = n >> 1;
    res_add_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, n);
}
void residual_add_fp16(__half* residual, const __half* x, int n,
                       cudaStream_t stream) {
    int n2 = n >> 1;
    res_add_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(residual, x, n);
}

template<typename T>
__global__ void add_out_kernel(const T* __restrict__ a,
                               const T* __restrict__ b,
                               T* __restrict__ out,
                               int n) {
    using T2 = typename packed2<T>::type;
    const T2* a2 = reinterpret_cast<const T2*>(a);
    const T2* b2 = reinterpret_cast<const T2*>(b);
    T2* out2 = reinterpret_cast<T2*>(out);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 av = a2[idx], bv = b2[idx];
        out2[idx] = make_packed2<T>(
            from_f32<T>(to_f32(av.x) + to_f32(bv.x)),
            from_f32<T>(to_f32(av.y) + to_f32(bv.y)));
    }
}

template __global__ void add_out_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);

void add_bf16_out(const __nv_bfloat16* a, const __nv_bfloat16* b,
                  __nv_bfloat16* out, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    add_out_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(
        a, b, out, n);
}

__global__ void euler_step_bf16_out_kernel(
    const __nv_bfloat16* __restrict__ latent,
    const __nv_bfloat16* __restrict__ velocity,
    __nv_bfloat16* __restrict__ out,
    float dt, int n2) {
    using T2 = packed2<__nv_bfloat16>::type;
    const T2* l2 = reinterpret_cast<const T2*>(latent);
    const T2* v2 = reinterpret_cast<const T2*>(velocity);
    T2* o2 = reinterpret_cast<T2*>(out);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n2; idx += blockDim.x * gridDim.x) {
        T2 lv = l2[idx], vv = v2[idx];
        float x0 = to_f32(lv.x) + to_f32(vv.x) * dt;
        float x1 = to_f32(lv.y) + to_f32(vv.y) * dt;
        o2[idx] = make_packed2<__nv_bfloat16>(
            from_f32<__nv_bfloat16>(x0),
            from_f32<__nv_bfloat16>(x1));
    }
}

void euler_step_bf16_out(const __nv_bfloat16* latent,
                         const __nv_bfloat16* velocity,
                         __nv_bfloat16* out,
                         float dt, int n,
                         cudaStream_t stream) {
    int n2 = n >> 1;
    int blocks = (n2 + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    euler_step_bf16_out_kernel<<<blocks, 256, 0, stream>>>(
        latent, velocity, out, dt, n2);
}

__global__ void teacher_force_first_frame_bf16_kernel(
    __nv_bfloat16* __restrict__ video_latent,
    const __nv_bfloat16* __restrict__ cond_latent,
    int C, int T, int H, int W, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        long long q = idx;
        int w = (int)(q % W); q /= W;
        int h = (int)(q % H); q /= H;
        int c = (int)(q % C); q /= C;
        long long b = q;
        long long dst = (((b * C + c) * (long long)T) * H + h) * W + w;
        video_latent[dst] = cond_latent[idx];
    }
}

void teacher_force_first_frame_bf16(__nv_bfloat16* video_latent,
                                    const __nv_bfloat16* cond_latent,
                                    int B, int C, int T, int H, int W,
                                    cudaStream_t stream) {
    long long n = (long long)B * C * H * W;
    int blocks = (int)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    teacher_force_first_frame_bf16_kernel<<<blocks, 256, 0, stream>>>(
        video_latent, cond_latent, C, T, H, W, n);
}

__global__ void motus_decode_postprocess_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ decoded,
    float* __restrict__ out,
    int C, int T_in, int H, int W, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    int T_out = T_in - 1;
    for (; idx < n; idx += stride) {
        long long q = idx;
        int w = (int)(q % W); q /= W;
        int h = (int)(q % H); q /= H;
        int t = (int)(q % T_out); q /= T_out;
        int c = (int)(q % C); q /= C;
        long long b = q;
        long long src = (((b * C + c) * (long long)T_in + (t + 1)) * H + h) * W + w;
        float v = (to_f32(decoded[src]) + 1.0f) * 0.5f;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        out[idx] = v;
    }
}

void motus_decode_postprocess_bf16_to_fp32(const __nv_bfloat16* decoded,
                                           float* out,
                                           int B, int C, int T_in,
                                           int H, int W,
                                           cudaStream_t stream) {
    long long n = (long long)B * C * (T_in - 1) * H * W;
    int blocks = (int)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    motus_decode_postprocess_bf16_to_fp32_kernel<<<blocks, 256, 0, stream>>>(
        decoded, out, C, T_in, H, W, n);
}

__global__ void cast_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += blockDim.x * gridDim.x) {
        dst[idx] = to_f32(src[idx]);
    }
}

void cast_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n,
                       cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    cast_bf16_to_fp32_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
}

__global__ void dup_up3d_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int Cin, int Cout, int T, int H, int W,
    int factor_t, int factor_s, int repeats,
    int out_T, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int out_W = W * factor_s;
    const int out_H = H * factor_s;

    int ow = idx % out_W;
    long long q = idx / out_W;
    int oh = q % out_H;
    q /= out_H;
    int ot = q % out_T;
    q /= out_T;
    int co = q % Cout;
    int b = q / Cout;

    const int temporal_skip = T * factor_t - out_T;
    const int ot_full = ot + temporal_skip;
    const int dt = ot_full % factor_t;
    const int it = ot_full / factor_t;
    const int dh = oh % factor_s;
    const int ih = oh / factor_s;
    const int dw = ow % factor_s;
    const int iw = ow / factor_s;

    const int sub = ((dt * factor_s) + dh) * factor_s + dw;
    const int p = co * factor_t * factor_s * factor_s + sub;
    const int ci = p / repeats;
    if (ci >= Cin) return;

    const long long in_idx =
        (((long long)b * Cin + ci) * T + it) * H * W
        + (long long)ih * W + iw;
    out[idx] = x[in_idx];
}

void dup_up3d_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                   int B, int Cin, int Cout, int T, int H, int W,
                   int factor_t, int factor_s, int repeats,
                   int first_chunk, cudaStream_t stream) {
    const int out_T = T * factor_t - (first_chunk ? (factor_t - 1) : 0);
    const long long n =
        (long long)B * Cout * out_T * (H * factor_s) * (W * factor_s);
    dup_up3d_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
        x, out, Cin, Cout, T, H, W, factor_t, factor_s, repeats, out_T, n);
}

__global__ void time_unshuffle2_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int C, int T, int H, int W, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const long long HW = (long long)H * W;
    const int ow = (int)(idx % W);
    long long q = idx / W;
    const int oh = (int)(q % H);
    q /= H;
    const int ot = (int)(q % (2 * T));
    q /= (2 * T);
    const int c = (int)(q % C);
    const long long b = q / C;

    const int src_group = ot & 1;
    const int src_t = ot >> 1;
    const int src_c = src_group * C + c;
    const long long src =
        (((b * (2LL * C) + src_c) * T + src_t) * HW)
        + (long long)oh * W + ow;
    out[idx] = x[src];
}

void time_unshuffle2_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                          int B, int C, int T, int H, int W,
                          cudaStream_t stream) {
    const long long n = (long long)B * C * (2LL * T) * H * W;
    time_unshuffle2_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
        x, out, C, T, H, W, n);
}

__global__ void add_bias_ncdhw_bf16_kernel(__nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ bias,
                                           int C, int T, int H, int W,
                                           long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long long inner = idx % ((long long)C * T * H * W);
    int c = (int)(inner / ((long long)T * H * W));
    float v = __bfloat162float(x[idx]) + __bfloat162float(bias[c]);
    x[idx] = __float2bfloat16(v);
}

void add_bias_ncdhw_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias,
                         int B, int C, int T, int H, int W,
                         cudaStream_t stream) {
    long long n = (long long)B * C * T * H * W;
    add_bias_ncdhw_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
        x, bias, C, T, H, W, n);
}

__global__ void update_cache2_ncdhw_bf16_kernel(
    const __nv_bfloat16* __restrict__ cur,
    const __nv_bfloat16* __restrict__ prev,
    __nv_bfloat16* __restrict__ out,
    int C, int T, int H, int W,
    long long n_out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_out) return;

    const long long HW = (long long)H * W;
    const long long cache_stride_c = 2LL * HW;
    const long long block = idx / HW;
    const int hw = (int)(idx - block * HW);
    const int t_cache = (int)(block % 2);
    const long long bc = block / 2;
    const int c = (int)(bc % C);
    const long long b = bc / C;

    __nv_bfloat16 v = __float2bfloat16(0.0f);
    if (T >= 2) {
        const int src_t = T - 2 + t_cache;
        long long src = (((b * C + c) * (long long)T + src_t) * HW) + hw;
        v = cur[src];
    } else if (T == 1) {
        if (t_cache == 1) {
            long long src = ((b * C + c) * (long long)T * HW) + hw;
            v = cur[src];
        } else if (prev != nullptr) {
            long long src = ((b * C + c) * cache_stride_c + HW) + hw;
            v = prev[src];
        }
    }
    out[idx] = v;
}

void update_cache2_ncdhw_bf16(const __nv_bfloat16* cur,
                              const __nv_bfloat16* prev,
                              __nv_bfloat16* out,
                              int B, int C, int T, int H, int W,
                              cudaStream_t stream) {
    long long n = (long long)B * C * 2LL * H * W;
    update_cache2_ncdhw_bf16_kernel
        <<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
            cur, prev, out, C, T, H, W, n);
}

__global__ void adaln_modulation6_bf16_kernel(
    const float* __restrict__ adaln_params,       // [B, S, 6, D]
    const float* __restrict__ layer_modulation,   // [6, D]
    __nv_bfloat16* __restrict__ out0,             // [B, S, D]
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    __nv_bfloat16* __restrict__ out3,
    __nv_bfloat16* __restrict__ out4,
    __nv_bfloat16* __restrict__ out5,
    int S, int D, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int d = (int)(idx % D);
    long long bs = idx / D;
    long long base = (bs * 6LL * D) + d;
    out0[idx] = __float2bfloat16(adaln_params[base + 0LL * D] +
                                 layer_modulation[0LL * D + d]);
    out1[idx] = __float2bfloat16(adaln_params[base + 1LL * D] +
                                 layer_modulation[1LL * D + d]);
    out2[idx] = __float2bfloat16(adaln_params[base + 2LL * D] +
                                 layer_modulation[2LL * D + d]);
    out3[idx] = __float2bfloat16(adaln_params[base + 3LL * D] +
                                 layer_modulation[3LL * D + d]);
    out4[idx] = __float2bfloat16(adaln_params[base + 4LL * D] +
                                 layer_modulation[4LL * D + d]);
    out5[idx] = __float2bfloat16(adaln_params[base + 5LL * D] +
                                 layer_modulation[5LL * D + d]);
}

void adaln_modulation6_bf16(const float* adaln_params,
                            const float* layer_modulation,
                            __nv_bfloat16* out0,
                            __nv_bfloat16* out1,
                            __nv_bfloat16* out2,
                            __nv_bfloat16* out3,
                            __nv_bfloat16* out4,
                            __nv_bfloat16* out5,
                            int B, int S, int D,
                            cudaStream_t stream) {
    long long n = (long long)B * S * D;
    adaln_modulation6_bf16_kernel
        <<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
            adaln_params, layer_modulation,
            out0, out1, out2, out3, out4, out5, S, D, n);
}

__global__ void concat3_qkv_bf16_kernel(
    const __nv_bfloat16* __restrict__ q0,
    const __nv_bfloat16* __restrict__ q1,
    const __nv_bfloat16* __restrict__ q2,
    const __nv_bfloat16* __restrict__ k0,
    const __nv_bfloat16* __restrict__ k1,
    const __nv_bfloat16* __restrict__ k2,
    const __nv_bfloat16* __restrict__ v0,
    const __nv_bfloat16* __restrict__ v1,
    const __nv_bfloat16* __restrict__ v2,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int L0, int L1, int L2, int H, int D,
    long long q0s0, long long q0s1, long long q0s2,
    long long q1s0, long long q1s1, long long q1s2,
    long long q2s0, long long q2s1, long long q2s2,
    long long k0s0, long long k0s1, long long k0s2,
    long long k1s0, long long k1s1, long long k1s2,
    long long k2s0, long long k2s1, long long k2s2,
    long long v0s0, long long v0s1, long long v0s2,
    long long v1s0, long long v1s1, long long v1s2,
    long long v2s0, long long v2s1, long long v2s2,
    long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const long long HD = (long long)H * D;
    const int L = L0 + L1 + L2;
    long long token = idx / HD;
    int hd = (int)(idx - token * HD);
    int h = hd / D;
    int d = hd - h * D;
    int pos = (int)(token % L);
    long long b = token / L;

    const __nv_bfloat16* qs;
    const __nv_bfloat16* ks;
    const __nv_bfloat16* vs;
    long long qs0, qs1, qs2, ks0, ks1, ks2, vs0, vs1, vs2;
    int src_pos;
    if (pos < L0) {
        qs = q0; ks = k0; vs = v0; src_pos = pos;
        qs0 = q0s0; qs1 = q0s1; qs2 = q0s2;
        ks0 = k0s0; ks1 = k0s1; ks2 = k0s2;
        vs0 = v0s0; vs1 = v0s1; vs2 = v0s2;
    } else if (pos < L0 + L1) {
        qs = q1; ks = k1; vs = v1; src_pos = pos - L0;
        qs0 = q1s0; qs1 = q1s1; qs2 = q1s2;
        ks0 = k1s0; ks1 = k1s1; ks2 = k1s2;
        vs0 = v1s0; vs1 = v1s1; vs2 = v1s2;
    } else {
        qs = q2; ks = k2; vs = v2; src_pos = pos - L0 - L1;
        qs0 = q2s0; qs1 = q2s1; qs2 = q2s2;
        ks0 = k2s0; ks1 = k2s1; ks2 = k2s2;
        vs0 = v2s0; vs1 = v2s1; vs2 = v2s2;
    }
    q_out[idx] = qs[b * qs0 + src_pos * qs1 + h * qs2 + d];
    k_out[idx] = ks[b * ks0 + src_pos * ks1 + h * ks2 + d];
    v_out[idx] = vs[b * vs0 + src_pos * vs1 + h * vs2 + d];
}

void concat3_qkv_bf16(const __nv_bfloat16* q0,
                      const __nv_bfloat16* q1,
                      const __nv_bfloat16* q2,
                      const __nv_bfloat16* k0,
                      const __nv_bfloat16* k1,
                      const __nv_bfloat16* k2,
                      const __nv_bfloat16* v0,
                      const __nv_bfloat16* v1,
                      const __nv_bfloat16* v2,
                      __nv_bfloat16* q_out,
                      __nv_bfloat16* k_out,
                      __nv_bfloat16* v_out,
                      int B, int L0, int L1, int L2,
                      int H, int D,
                      long long q0s0, long long q0s1, long long q0s2,
                      long long q1s0, long long q1s1, long long q1s2,
                      long long q2s0, long long q2s1, long long q2s2,
                      long long k0s0, long long k0s1, long long k0s2,
                      long long k1s0, long long k1s1, long long k1s2,
                      long long k2s0, long long k2s1, long long k2s2,
                      long long v0s0, long long v0s1, long long v0s2,
                      long long v1s0, long long v1s1, long long v1s2,
                      long long v2s0, long long v2s1, long long v2s2,
                      cudaStream_t stream) {
    long long n = (long long)B * (L0 + L1 + L2) * H * D;
    concat3_qkv_bf16_kernel
        <<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
            q0, q1, q2, k0, k1, k2, v0, v1, v2,
            q_out, k_out, v_out, L0, L1, L2, H, D,
            q0s0, q0s1, q0s2, q1s0, q1s1, q1s2, q2s0, q2s1, q2s2,
            k0s0, k0s1, k0s2, k1s0, k1s1, k1s2, k2s0, k2s1, k2s2,
            v0s0, v0s1, v0s2, v1s0, v1s1, v1s2, v2s0, v2s1, v2s2,
            n);
}

__global__ void concat3_qkv_bf16_fast_kernel(
    const __nv_bfloat16* __restrict__ q0,
    const __nv_bfloat16* __restrict__ q1,
    const __nv_bfloat16* __restrict__ q2,
    const __nv_bfloat16* __restrict__ k0,
    const __nv_bfloat16* __restrict__ k1,
    const __nv_bfloat16* __restrict__ k2,
    const __nv_bfloat16* __restrict__ v0,
    const __nv_bfloat16* __restrict__ v1,
    const __nv_bfloat16* __restrict__ v2,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int L0, int L1, int L2, int HD,
    long long q0s0, long long q0s1,
    long long q1s0, long long q1s1,
    long long q2s0, long long q2s1,
    long long k0s0, long long k0s1,
    long long k1s0, long long k1s1,
    long long k2s0, long long k2s1,
    long long v0s0, long long v0s1,
    long long v1s0, long long v1s1,
    long long v2s0, long long v2s1) {
    const int L = L0 + L1 + L2;
    const int row_kind = blockIdx.x;
    const int kind = row_kind / (gridDim.x / 3);  // 0=q, 1=k, 2=v
    const int row = row_kind - kind * (gridDim.x / 3);
    const int b = row / L;
    const int pos = row - b * L;

    const __nv_bfloat16* src = nullptr;
    long long s0 = 0, s1 = 0;
    int src_pos = pos;
    if (pos < L0) {
        if (kind == 0) { src = q0; s0 = q0s0; s1 = q0s1; }
        else if (kind == 1) { src = k0; s0 = k0s0; s1 = k0s1; }
        else { src = v0; s0 = v0s0; s1 = v0s1; }
    } else if (pos < L0 + L1) {
        src_pos = pos - L0;
        if (kind == 0) { src = q1; s0 = q1s0; s1 = q1s1; }
        else if (kind == 1) { src = k1; s0 = k1s0; s1 = k1s1; }
        else { src = v1; s0 = v1s0; s1 = v1s1; }
    } else {
        src_pos = pos - L0 - L1;
        if (kind == 0) { src = q2; s0 = q2s0; s1 = q2s1; }
        else if (kind == 1) { src = k2; s0 = k2s0; s1 = k2s1; }
        else { src = v2; s0 = v2s0; s1 = v2s1; }
    }

    const __nv_bfloat16* src_row = src + (long long)b * s0 + (long long)src_pos * s1;
    __nv_bfloat16* dst_base = (kind == 0) ? q_out : ((kind == 1) ? k_out : v_out);
    __nv_bfloat16* dst_row = dst_base + (long long)row * HD;

    const int vecs = HD >> 3;  // 8 bf16 = 16 bytes
    const uint4* src4 = reinterpret_cast<const uint4*>(src_row);
    uint4* dst4 = reinterpret_cast<uint4*>(dst_row);
    for (int i = threadIdx.x; i < vecs; i += blockDim.x) {
        dst4[i] = src4[i];
    }
}

void concat3_qkv_bf16_fast(const __nv_bfloat16* q0,
                           const __nv_bfloat16* q1,
                           const __nv_bfloat16* q2,
                           const __nv_bfloat16* k0,
                           const __nv_bfloat16* k1,
                           const __nv_bfloat16* k2,
                           const __nv_bfloat16* v0,
                           const __nv_bfloat16* v1,
                           const __nv_bfloat16* v2,
                           __nv_bfloat16* q_out,
                           __nv_bfloat16* k_out,
                           __nv_bfloat16* v_out,
                           int B, int L0, int L1, int L2, int H, int D,
                           long long q0s0, long long q0s1,
                           long long q1s0, long long q1s1,
                           long long q2s0, long long q2s1,
                           long long k0s0, long long k0s1,
                           long long k1s0, long long k1s1,
                           long long k2s0, long long k2s1,
                           long long v0s0, long long v0s1,
                           long long v1s0, long long v1s1,
                           long long v2s0, long long v2s1,
                           cudaStream_t stream) {
    const int HD = H * D;
    if (B <= 0 || HD <= 0 || ((HD & 7) != 0)) return;
    const int rows = B * (L0 + L1 + L2);
    concat3_qkv_bf16_fast_kernel<<<rows * 3, 256, 0, stream>>>(
        q0, q1, q2, k0, k1, k2, v0, v1, v2,
        q_out, k_out, v_out, L0, L1, L2, HD,
        q0s0, q0s1, q1s0, q1s1, q2s0, q2s1,
        k0s0, k0s1, k1s0, k1s1, k2s0, k2s1,
        v0s0, v0s1, v1s0, v1s1, v2s0, v2s1);
}

__device__ __forceinline__ const __nv_bfloat16* concat3_row_ptr(
    const __nv_bfloat16* p0, const __nv_bfloat16* p1,
    const __nv_bfloat16* p2, int B, int L0, int L1, int L2,
    long long s00, long long s01, long long s10, long long s11,
    long long s20, long long s21, int row) {
    const int L = L0 + L1 + L2;
    const int b = row / L;
    const int pos = row - b * L;
    if (pos < L0) return p0 + (long long)b * s00 + (long long)pos * s01;
    if (pos < L0 + L1) return p1 + (long long)b * s10 + (long long)(pos - L0) * s11;
    return p2 + (long long)b * s20 + (long long)(pos - L0 - L1) * s21;
}

__device__ __forceinline__ int8_t f32_to_i8_sat(float x) {
    int v = __float2int_rn(x);
    v = max(-127, min(127, v));
    return static_cast<int8_t>(v);
}

template<int BLOCK_TOKENS>
__global__ void concat3_quant_int8_d128_kernel(
    const __nv_bfloat16* __restrict__ x0,
    const __nv_bfloat16* __restrict__ x1,
    const __nv_bfloat16* __restrict__ x2,
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    int L0, int L1, int L2, int H,
    long long x0s0, long long x0s1,
    long long x1s0, long long x1s1,
    long long x2s0, long long x2s1) {
    constexpr int D = 128;
    constexpr int PACK = 8;
    constexpr int THREADS_PER_TOKEN = D / PACK;
    const int block_id = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int L = L0 + L1 + L2;
    const int tid = threadIdx.x;
    const int token_in_block = tid / THREADS_PER_TOKEN;
    const int d_pack = tid - token_in_block * THREADS_PER_TOKEN;
    const int pos = block_id * BLOCK_TOKENS + token_in_block;
    const int row = b * L + pos;
    float vals[PACK];
    float amax = 1.0e-7f;
    if (pos < L) {
        const __nv_bfloat16* src = concat3_row_ptr(
            x0, x1, x2, gridDim.z, L0, L1, L2,
            x0s0, x0s1, x1s0, x1s1, x2s0, x2s1, row);
        src += (long long)h * D + d_pack * PACK;
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            vals[i] = __bfloat162float(src[i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }
    } else {
#pragma unroll
        for (int i = 0; i < PACK; ++i) vals[i] = 0.0f;
    }

    __shared__ float smem[1024];
    smem[tid] = amax;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    const float s = smem[0] * (1.0f / 127.0f);
    if (tid == 0) {
        scale[((long long)b * H + h) * gridDim.x + block_id] = s;
    }
    if (pos < L) {
        const float inv_s = 127.0f / smem[0];
        int8_t* dst = out + (((long long)b * L + pos) * H + h) * D + d_pack * PACK;
        char4 lo = make_char4(
            f32_to_i8_sat(vals[0] * inv_s),
            f32_to_i8_sat(vals[1] * inv_s),
            f32_to_i8_sat(vals[2] * inv_s),
            f32_to_i8_sat(vals[3] * inv_s));
        char4 hi = make_char4(
            f32_to_i8_sat(vals[4] * inv_s),
            f32_to_i8_sat(vals[5] * inv_s),
            f32_to_i8_sat(vals[6] * inv_s),
            f32_to_i8_sat(vals[7] * inv_s));
        reinterpret_cast<char4*>(dst)[0] = lo;
        reinterpret_cast<char4*>(dst)[1] = hi;
    }
}

template<int BLOCK_TOKENS>
__global__ void quant_int8_bf16_nhd_d128_kernel(
    const __nv_bfloat16* __restrict__ x,
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    int L, int H) {
    constexpr int D = 128;
    constexpr int PACK = 8;
    constexpr int THREADS_PER_TOKEN = D / PACK;
    const int block_id = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int token_in_block = tid / THREADS_PER_TOKEN;
    const int d_pack = tid - token_in_block * THREADS_PER_TOKEN;
    const int pos = block_id * BLOCK_TOKENS + token_in_block;
    float vals[PACK];
    float amax = 1.0e-7f;
    if (pos < L) {
        const __nv_bfloat16* src =
            x + (((long long)b * L + pos) * H + h) * D + d_pack * PACK;
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            vals[i] = __bfloat162float(src[i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }
    } else {
#pragma unroll
        for (int i = 0; i < PACK; ++i) vals[i] = 0.0f;
    }
    __shared__ float smem[1024];
    smem[tid] = amax;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    const float s = smem[0] * (1.0f / 127.0f);
    if (tid == 0) {
        scale[((long long)b * H + h) * gridDim.x + block_id] = s;
    }
    if (pos < L) {
        const float inv_s = 127.0f / smem[0];
        int8_t* dst = out + (((long long)b * L + pos) * H + h) * D + d_pack * PACK;
        char4 lo = make_char4(
            f32_to_i8_sat(vals[0] * inv_s),
            f32_to_i8_sat(vals[1] * inv_s),
            f32_to_i8_sat(vals[2] * inv_s),
            f32_to_i8_sat(vals[3] * inv_s));
        char4 hi = make_char4(
            f32_to_i8_sat(vals[4] * inv_s),
            f32_to_i8_sat(vals[5] * inv_s),
            f32_to_i8_sat(vals[6] * inv_s),
            f32_to_i8_sat(vals[7] * inv_s));
        reinterpret_cast<char4*>(dst)[0] = lo;
        reinterpret_cast<char4*>(dst)[1] = hi;
    }
}

__global__ void concat3_v_bf16_to_fp16_d128_kernel(
    const __nv_bfloat16* __restrict__ v0,
    const __nv_bfloat16* __restrict__ v1,
    const __nv_bfloat16* __restrict__ v2,
    __half* __restrict__ v_out,
    int L0, int L1, int L2, int H,
    long long v0s0, long long v0s1,
    long long v1s0, long long v1s1,
    long long v2s0, long long v2s1) {
    constexpr int D = 128;
    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const __nv_bfloat16* src = concat3_row_ptr(
        v0, v1, v2, gridDim.z, L0, L1, L2,
        v0s0, v0s1, v1s0, v1s1, v2s0, v2s1, row);
    src += (long long)h * D;
    __half* dst = v_out + ((long long)row * H + h) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        dst[i] = __float2half_rn(__bfloat162float(src[i]));
    }
}

__device__ __forceinline__ int sage_v_perm64(int t) {
    const int base = (t >> 4) << 4;
    const int m = t & 15;
    return base + ((m >> 3) << 1) + (((m >> 1) & 3) << 2) + (m & 1);
}

__device__ __forceinline__ int sage_v_inv_perm64(int t) {
    const int base = (t >> 4) << 4;
    const int m = t & 15;
    const int inv = (m < 2) ? m :
                    (m < 4) ? (m + 6) :
                    (m < 6) ? (m - 2) :
                    (m < 8) ? (m + 4) :
                    (m < 10) ? (m - 4) :
                    (m < 12) ? (m + 2) :
                    (m < 14) ? (m - 6) : m;
    return base + inv;
}

__global__ void concat3_v_fp8_per_channel_d128_kernel(
    const __nv_bfloat16* __restrict__ v0,
    const __nv_bfloat16* __restrict__ v1,
    const __nv_bfloat16* __restrict__ v2,
    int8_t* __restrict__ v_fp8_out,
    float* __restrict__ v_scale,
    int L0, int L1, int L2, int H,
    long long v0s0, long long v0s1,
    long long v1s0, long long v1s1,
    long long v2s0, long long v2s1) {
    constexpr int D = 128;
    constexpr int PACK = 8;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int d = blockIdx.z;
    const int tid = threadIdx.x;
    const int L = L0 + L1 + L2;
    const int padded = ((L + 63) / 64) * 64;
    float max_v = -CUDART_INF_F;
    float min_v = CUDART_INF_F;

    for (int t = tid; t < L; t += blockDim.x) {
        const __nv_bfloat16* src = concat3_row_ptr(
            v0, v1, v2, gridDim.y, L0, L1, L2,
            v0s0, v0s1, v1s0, v1s1, v2s0, v2s1, b * L + t);
        const float x = __bfloat162float(src[(long long)h * D + d]);
        max_v = fmaxf(max_v, x);
        min_v = fminf(min_v, x);
    }

    __shared__ float smem_max[256];
    __shared__ float smem_min[256];
    smem_max[tid] = max_v;
    smem_min[tid] = min_v;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + stride]);
            smem_min[tid] = fminf(smem_min[tid], smem_min[tid + stride]);
        }
        __syncthreads();
    }
    const float amax = fmaxf(fabsf(smem_max[0]), fabsf(smem_min[0]));
    const float scale = fmaxf(amax, 1.0e-7f) * (1.0f / 448.0f);
    if (tid == 0) {
        v_scale[((long long)b * H + h) * D + d] = scale;
    }
    const float inv_scale = 448.0f / fmaxf(amax, 1.0e-7f);
    int8_t* out_base = v_fp8_out + (((long long)b * D + d) * H + h) * padded;

    for (int t0 = tid * PACK; t0 < padded; t0 += blockDim.x * PACK) {
        float vals[PACK];
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const int out_t = t0 + i;
            const int t = sage_v_inv_perm64(out_t);
            float x = 0.0f;
            if (t < L) {
                const __nv_bfloat16* src = concat3_row_ptr(
                    v0, v1, v2, gridDim.y, L0, L1, L2,
                    v0s0, v0s1, v1s0, v1s1, v2s0, v2s1, b * L + t);
                x = __bfloat162float(src[(long long)h * D + d]) * inv_scale;
            }
            vals[i] = x;
        }
        uint32_t fp8_pack[2];
        floatx4_to_e4m3x4(fp8_pack, vals, vals + 2);
        floatx4_to_e4m3x4(fp8_pack + 1, vals + 4, vals + 6);
        *reinterpret_cast<uint2*>(out_base + t0) = *reinterpret_cast<uint2*>(fp8_pack);
    }
}

void concat3_qk_int8_v_fp16_d128(const __nv_bfloat16* q0,
                                 const __nv_bfloat16* q1,
                                 const __nv_bfloat16* q2,
                                 const __nv_bfloat16* k0,
                                 const __nv_bfloat16* k1,
                                 const __nv_bfloat16* k2,
                                 const __nv_bfloat16* v0,
                                 const __nv_bfloat16* v1,
                                 const __nv_bfloat16* v2,
                                 int8_t* q_out,
                                 int8_t* k_out,
                                 __half* v_out,
                                 float* q_scale,
                                 float* k_scale,
                                 int B, int L0, int L1, int L2, int H,
                                 long long q0s0, long long q0s1,
                                 long long q1s0, long long q1s1,
                                 long long q2s0, long long q2s1,
                                 long long k0s0, long long k0s1,
                                 long long k1s0, long long k1s1,
                                 long long k2s0, long long k2s1,
                                 long long v0s0, long long v0s1,
                                 long long v1s0, long long v1s1,
                                 long long v2s0, long long v2s1,
                                 cudaStream_t stream) {
    const int L = L0 + L1 + L2;
    if (B <= 0 || L <= 0 || H <= 0) return;
    constexpr int D = 128;
    constexpr int PACK = 8;
    constexpr int Q_BLOCK = 32;
    constexpr int K_BLOCK = 64;
    dim3 q_grid((L + Q_BLOCK - 1) / Q_BLOCK, H, B);
    dim3 k_grid((L + K_BLOCK - 1) / K_BLOCK, H, B);
    concat3_quant_int8_d128_kernel<Q_BLOCK>
        <<<q_grid, Q_BLOCK * (D / PACK), 0, stream>>>(
            q0, q1, q2, q_out, q_scale, L0, L1, L2, H,
            q0s0, q0s1, q1s0, q1s1, q2s0, q2s1);
    concat3_quant_int8_d128_kernel<K_BLOCK>
        <<<k_grid, K_BLOCK * (D / PACK), 0, stream>>>(
            k0, k1, k2, k_out, k_scale, L0, L1, L2, H,
            k0s0, k0s1, k1s0, k1s1, k2s0, k2s1);
    concat3_v_bf16_to_fp16_d128_kernel<<<dim3(B * L, H, 1), 128, 0, stream>>>(
        v0, v1, v2, v_out, L0, L1, L2, H,
        v0s0, v0s1, v1s0, v1s1, v2s0, v2s1);
}

void concat3_qk_int8_v_fp8_d128(const __nv_bfloat16* q0,
                                const __nv_bfloat16* q1,
                                const __nv_bfloat16* q2,
                                const __nv_bfloat16* k0,
                                const __nv_bfloat16* k1,
                                const __nv_bfloat16* k2,
                                const __nv_bfloat16* v0,
                                const __nv_bfloat16* v1,
                                const __nv_bfloat16* v2,
                                int8_t* q_out,
                                int8_t* k_out,
                                int8_t* v_fp8_out,
                                float* q_scale,
                                float* k_scale,
                                float* v_scale,
                                int B, int L0, int L1, int L2, int H,
                                long long q0s0, long long q0s1,
                                long long q1s0, long long q1s1,
                                long long q2s0, long long q2s1,
                                long long k0s0, long long k0s1,
                                long long k1s0, long long k1s1,
                                long long k2s0, long long k2s1,
                                long long v0s0, long long v0s1,
                                long long v1s0, long long v1s1,
                                long long v2s0, long long v2s1,
                                cudaStream_t stream) {
    const int L = L0 + L1 + L2;
    if (B <= 0 || L <= 0 || H <= 0) return;
    constexpr int D = 128;
    constexpr int PACK = 8;
    constexpr int Q_BLOCK = 32;
    constexpr int K_BLOCK = 64;
    dim3 q_grid((L + Q_BLOCK - 1) / Q_BLOCK, H, B);
    dim3 k_grid((L + K_BLOCK - 1) / K_BLOCK, H, B);
    concat3_quant_int8_d128_kernel<Q_BLOCK>
        <<<q_grid, Q_BLOCK * (D / PACK), 0, stream>>>(
            q0, q1, q2, q_out, q_scale, L0, L1, L2, H,
            q0s0, q0s1, q1s0, q1s1, q2s0, q2s1);
    concat3_quant_int8_d128_kernel<K_BLOCK>
        <<<k_grid, K_BLOCK * (D / PACK), 0, stream>>>(
            k0, k1, k2, k_out, k_scale, L0, L1, L2, H,
            k0s0, k0s1, k1s0, k1s1, k2s0, k2s1);
    concat3_v_fp8_per_channel_d128_kernel<<<dim3(H, B, D), 256, 0, stream>>>(
        v0, v1, v2, v_fp8_out, v_scale, L0, L1, L2, H,
        v0s0, v0s1, v1s0, v1s1, v2s0, v2s1);
}

__global__ void concat3_v_tpp_bf16_d128_kernel(
    const __nv_bfloat16* __restrict__ v0,
    const __nv_bfloat16* __restrict__ v1,
    const __nv_bfloat16* __restrict__ v2,
    __nv_bfloat16* __restrict__ out,
    int L0, int L1, int L2, int H,
    long long v0s0, long long v0s1,
    long long v1s0, long long v1s1,
    long long v2s0, long long v2s1) {
    constexpr int D = 128;
    constexpr int CTA = 64;
    constexpr int PACK = 8;
    constexpr int THREADS_PER_TOKEN = D / PACK;
    __shared__ __nv_bfloat16 sm_load[CTA][D];
    __shared__ __nv_bfloat16 sm_store[D][CTA];

    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int token_lane = tid / THREADS_PER_TOKEN;
    const int d_pack = tid - token_lane * THREADS_PER_TOKEN;
    const int L = L0 + L1 + L2;
    const int padded = ((L + CTA - 1) / CTA) * CTA;
    const int src_t = tile * CTA + token_lane;
    const int perm_lane = ((token_lane / 16) * 16) +
                          ((token_lane % 16) / 8) * 2 +
                          (((token_lane % 16) / 2) % 4) * 4 +
                          (token_lane % 2);

    if (src_t < L) {
        const __nv_bfloat16* src = concat3_row_ptr(
            v0, v1, v2, gridDim.z, L0, L1, L2,
            v0s0, v0s1, v1s0, v1s1, v2s0, v2s1, b * L + src_t);
        src += (long long)h * D + d_pack * PACK;
        *reinterpret_cast<uint4*>(&sm_load[perm_lane][d_pack * PACK]) =
            *reinterpret_cast<const uint4*>(src);
    } else {
        uint4 zero = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&sm_load[perm_lane][d_pack * PACK]) = zero;
    }
    __syncthreads();

    const int row = tid & 63;
    const int d_base = tid >> 6;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm_store[d_base + i * (D / PACK)][row] =
            sm_load[row][d_base + i * (D / PACK)];
    }
    __syncthreads();

    const int d_out = tid / (CTA / PACK);
    const int seq_pack = tid - d_out * (CTA / PACK);
    __nv_bfloat16* dst =
        out + (((long long)b * D + d_out) * H + h) * padded + tile * CTA + seq_pack * PACK;
    *reinterpret_cast<uint4*>(dst) =
        *reinterpret_cast<uint4*>(&sm_store[d_out][seq_pack * PACK]);
}

__global__ void concat3_v_tpp_bf16_amax_d128_kernel(
    const __nv_bfloat16* __restrict__ v0,
    const __nv_bfloat16* __restrict__ v1,
    const __nv_bfloat16* __restrict__ v2,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ tile_amax,
    int L0, int L1, int L2, int H,
    long long v0s0, long long v0s1,
    long long v1s0, long long v1s1,
    long long v2s0, long long v2s1) {
    constexpr int D = 128;
    constexpr int CTA = 64;
    constexpr int PACK = 8;
    constexpr int THREADS_PER_TOKEN = D / PACK;
    __shared__ __nv_bfloat16 sm_load[CTA][D];
    __shared__ __nv_bfloat16 sm_store[D][CTA];
    __shared__ float sm_amax[D][CTA / PACK];

    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int token_lane = tid / THREADS_PER_TOKEN;
    const int d_pack = tid - token_lane * THREADS_PER_TOKEN;
    const int L = L0 + L1 + L2;
    const int padded = ((L + CTA - 1) / CTA) * CTA;
    const int src_t = tile * CTA + token_lane;
    const int perm_lane = ((token_lane / 16) * 16) +
                          ((token_lane % 16) / 8) * 2 +
                          (((token_lane % 16) / 2) % 4) * 4 +
                          (token_lane % 2);

    if (src_t < L) {
        const __nv_bfloat16* src = concat3_row_ptr(
            v0, v1, v2, gridDim.z, L0, L1, L2,
            v0s0, v0s1, v1s0, v1s1, v2s0, v2s1, b * L + src_t);
        src += (long long)h * D + d_pack * PACK;
        *reinterpret_cast<uint4*>(&sm_load[perm_lane][d_pack * PACK]) =
            *reinterpret_cast<const uint4*>(src);
    } else {
        uint4 zero = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(&sm_load[perm_lane][d_pack * PACK]) = zero;
    }
    __syncthreads();

    const int row = tid & 63;
    const int d_base = tid >> 6;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sm_store[d_base + i * (D / PACK)][row] =
            sm_load[row][d_base + i * (D / PACK)];
    }
    __syncthreads();

    const int d_out = tid / (CTA / PACK);
    const int seq_pack = tid - d_out * (CTA / PACK);
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < PACK; ++i) {
        a = fmaxf(a, fabsf(__bfloat162float(sm_store[d_out][seq_pack * PACK + i])));
    }
    sm_amax[d_out][seq_pack] = a;
    __syncthreads();
    if (seq_pack == 0) {
        float m = sm_amax[d_out][0];
#pragma unroll
        for (int i = 1; i < CTA / PACK; ++i) {
            m = fmaxf(m, sm_amax[d_out][i]);
        }
        tile_amax[(((long long)b * H + h) * D + d_out) * gridDim.x + tile] = m;
    }

    __nv_bfloat16* dst =
        out + (((long long)b * D + d_out) * H + h) * padded + tile * CTA + seq_pack * PACK;
    *reinterpret_cast<uint4*>(dst) =
        *reinterpret_cast<uint4*>(&sm_store[d_out][seq_pack * PACK]);
}

__global__ void v_tpp_quant_fp8_d128_kernel(
    const __nv_bfloat16* __restrict__ in,
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    int L, int H) {
    constexpr int D = 128;
    constexpr int PACK = 8;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int d = blockIdx.z;
    const int tid = threadIdx.x;
    const int padded = ((L + 63) / 64) * 64;
    const __nv_bfloat16* src = in + (((long long)b * D + d) * H + h) * padded;
    int8_t* dst = out + (((long long)b * D + d) * H + h) * padded;

    float max_v = -CUDART_INF_F;
    float min_v = CUDART_INF_F;
    for (int t = tid * PACK; t < padded; t += blockDim.x * PACK) {
        __nv_bfloat16 vals_bf16[PACK];
        *reinterpret_cast<uint4*>(vals_bf16) = *reinterpret_cast<const uint4*>(src + t);
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            const float x = __bfloat162float(vals_bf16[i]);
            max_v = fmaxf(max_v, x);
            min_v = fminf(min_v, x);
        }
    }
    __shared__ float smem_max[256];
    __shared__ float smem_min[256];
    smem_max[tid] = max_v;
    smem_min[tid] = min_v;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + stride]);
            smem_min[tid] = fminf(smem_min[tid], smem_min[tid + stride]);
        }
        __syncthreads();
    }
    const float amax = fmaxf(fabsf(smem_max[0]), fabsf(smem_min[0]));
    const float safe_amax = fmaxf(amax, 1.0e-7f);
    if (tid == 0) {
        scale[((long long)b * H + h) * D + d] = safe_amax * (1.0f / 448.0f);
    }
    const float inv_scale = 448.0f / safe_amax;
    for (int t = tid * PACK; t < padded; t += blockDim.x * PACK) {
        __nv_bfloat16 vals_bf16[PACK];
        float vals[PACK];
        *reinterpret_cast<uint4*>(vals_bf16) = *reinterpret_cast<const uint4*>(src + t);
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            vals[i] = __bfloat162float(vals_bf16[i]) * inv_scale;
        }
        uint32_t fp8_pack[2];
        floatx4_to_e4m3x4(fp8_pack, vals, vals + 2);
        floatx4_to_e4m3x4(fp8_pack + 1, vals + 4, vals + 6);
        *reinterpret_cast<uint2*>(dst + t) = *reinterpret_cast<uint2*>(fp8_pack);
    }
}

__global__ void v_tpp_quant_fp8_amax_d128_kernel(
    const __nv_bfloat16* __restrict__ in,
    const float* __restrict__ tile_amax,
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    int L, int H) {
    constexpr int D = 128;
    constexpr int PACK = 8;
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int d = blockIdx.z;
    const int tid = threadIdx.x;
    const int tiles = (L + 63) / 64;
    const int padded = tiles * 64;
    const __nv_bfloat16* src = in + (((long long)b * D + d) * H + h) * padded;
    int8_t* dst = out + (((long long)b * D + d) * H + h) * padded;
    const float* amax_src = tile_amax + (((long long)b * H + h) * D + d) * tiles;

    float amax = 0.0f;
    for (int i = tid; i < tiles; i += blockDim.x) {
        amax = fmaxf(amax, amax_src[i]);
    }
    __shared__ float smem[256];
    smem[tid] = amax;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    const float safe_amax = fmaxf(smem[0], 1.0e-7f);
    if (tid == 0) {
        scale[((long long)b * H + h) * D + d] = safe_amax * (1.0f / 448.0f);
    }
    const float inv_scale = 448.0f / safe_amax;
    for (int t = tid * PACK; t < padded; t += blockDim.x * PACK) {
        __nv_bfloat16 vals_bf16[PACK];
        float vals[PACK];
        *reinterpret_cast<uint4*>(vals_bf16) = *reinterpret_cast<const uint4*>(src + t);
#pragma unroll
        for (int i = 0; i < PACK; ++i) {
            vals[i] = __bfloat162float(vals_bf16[i]) * inv_scale;
        }
        uint32_t fp8_pack[2];
        floatx4_to_e4m3x4(fp8_pack, vals, vals + 2);
        floatx4_to_e4m3x4(fp8_pack + 1, vals + 4, vals + 6);
        *reinterpret_cast<uint2*>(dst + t) = *reinterpret_cast<uint2*>(fp8_pack);
    }
}

void concat3_v_transpose_pad_permute_bf16_d128(
                                const __nv_bfloat16* v0,
                                const __nv_bfloat16* v1,
                                const __nv_bfloat16* v2,
                                __nv_bfloat16* v_tpp_out,
                                int B, int L0, int L1, int L2, int H,
                                long long v0s0, long long v0s1,
                                long long v1s0, long long v1s1,
                                long long v2s0, long long v2s1,
                                cudaStream_t stream) {
    const int L = L0 + L1 + L2;
    if (B <= 0 || L <= 0 || H <= 0) return;
    const int tiles = (L + 63) / 64;
    concat3_v_tpp_bf16_d128_kernel<<<dim3(tiles, H, B), 1024, 0, stream>>>(
        v0, v1, v2, v_tpp_out, L0, L1, L2, H,
        v0s0, v0s1, v1s0, v1s1, v2s0, v2s1);
}

void v_tpp_bf16_quant_fp8_d128(const __nv_bfloat16* v_tpp,
                               int8_t* v_fp8,
                               float* v_scale,
                               int B, int L, int H,
                               cudaStream_t stream) {
    if (B <= 0 || L <= 0 || H <= 0) return;
    v_tpp_quant_fp8_d128_kernel<<<dim3(H, B, 128), 256, 0, stream>>>(
        v_tpp, v_fp8, v_scale, L, H);
}

void concat3_v_tpp_bf16_amax_d128(const __nv_bfloat16* v0,
                                  const __nv_bfloat16* v1,
                                  const __nv_bfloat16* v2,
                                  __nv_bfloat16* v_tpp_out,
                                  float* tile_amax,
                                  int B, int L0, int L1, int L2, int H,
                                  long long v0s0, long long v0s1,
                                  long long v1s0, long long v1s1,
                                  long long v2s0, long long v2s1,
                                  cudaStream_t stream) {
    const int L = L0 + L1 + L2;
    if (B <= 0 || L <= 0 || H <= 0) return;
    const int tiles = (L + 63) / 64;
    concat3_v_tpp_bf16_amax_d128_kernel<<<dim3(tiles, H, B), 1024, 0, stream>>>(
        v0, v1, v2, v_tpp_out, tile_amax, L0, L1, L2, H,
        v0s0, v0s1, v1s0, v1s1, v2s0, v2s1);
}

void v_tpp_bf16_quant_fp8_amax_d128(const __nv_bfloat16* v_tpp,
                                    const float* tile_amax,
                                    int8_t* v_fp8,
                                    float* v_scale,
                                    int B, int L, int H,
                                    cudaStream_t stream) {
    if (B <= 0 || L <= 0 || H <= 0) return;
    v_tpp_quant_fp8_amax_d128_kernel<<<dim3(H, B, 128), 256, 0, stream>>>(
        v_tpp, tile_amax, v_fp8, v_scale, L, H);
}

void quant_per_warp_int8_bf16_d128(const __nv_bfloat16* x,
                                   int8_t* out,
                                   float* scale,
                                   int B, int L, int H,
                                   cudaStream_t stream) {
    if (B <= 0 || L <= 0 || H <= 0) return;
    constexpr int D = 128;
    constexpr int PACK = 8;
    constexpr int BLOCK = 32;
    quant_int8_bf16_nhd_d128_kernel<BLOCK>
        <<<dim3((L + BLOCK - 1) / BLOCK, H, B), BLOCK * (D / PACK), 0, stream>>>(
            x, out, scale, L, H);
}

void quant_per_block_int8_bf16_d128(const __nv_bfloat16* x,
                                    int8_t* out,
                                    float* scale,
                                    int B, int L, int H,
                                    cudaStream_t stream) {
    if (B <= 0 || L <= 0 || H <= 0) return;
    constexpr int D = 128;
    constexpr int PACK = 8;
    constexpr int BLOCK = 64;
    quant_int8_bf16_nhd_d128_kernel<BLOCK>
        <<<dim3((L + BLOCK - 1) / BLOCK, H, B), BLOCK * (D / PACK), 0, stream>>>(
            x, out, scale, L, H);
}

// ── Bias + Gate × Residual (G6.7) ──
// Fused: residual[s, d] = residual[s, d] + (x[s, d] + bias[d]) * gate[s, d]
// Replaces the chain  add_bias_bf16(x, bias, S, D)  +  gate_mul_residual(
// residual, x, gate, S*D)  with a single launch.
//
// Shape contract:
//   residual : [S, D]   in-place updated
//   x        : [S, D]   read-only (e.g. fresh GEMM output, no bias yet)
//   bias     : [D]      broadcast across rows
//   gate     : [S, D]   per-token gate (Wan adaLN gamma)
//   S = seq_len = batch*tokens, D = dim (channels)
template<typename T>
__global__ void bias_gate_mul_residual_kernel(T* __restrict__ residual,
                                              const T* __restrict__ x,
                                              const T* __restrict__ bias,
                                              const T* __restrict__ gate,
                                              int dim) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i], bv = b2[i];
        float r0 = to_f32(rv.x)
                 + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
        float r1 = to_f32(rv.y)
                 + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
        res2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void bias_gate_mul_residual_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, int);
template __global__ void bias_gate_mul_residual_kernel<__half>(
    __half*, const __half*, const __half*, const __half*, int);

void bias_gate_mul_residual_bf16(__nv_bfloat16* residual, const __nv_bfloat16* x,
                                 const __nv_bfloat16* bias,
                                 const __nv_bfloat16* gate,
                                 int seq_len, int dim,
                                 cudaStream_t stream) {
    bias_gate_mul_residual_kernel<__nv_bfloat16><<<seq_len, 256, 0, stream>>>(
        residual, x, bias, gate, dim);
}

void bias_gate_mul_residual_fp16(__half* residual, const __half* x,
                                 const __half* bias,
                                 const __half* gate,
                                 int seq_len, int dim,
                                 cudaStream_t stream) {
    bias_gate_mul_residual_kernel<__half><<<seq_len, 256, 0, stream>>>(
        residual, x, bias, gate, dim);
}

template<typename T>
__global__ void bias_gate_mul_residual_out_kernel(
    const T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ bias,
    const T* __restrict__ gate,
    T* __restrict__ out,
    int dim) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* res2 = reinterpret_cast<const T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i], bv = b2[i];
        float r0 = to_f32(rv.x)
                 + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
        float r1 = to_f32(rv.y)
                 + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
        out2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void bias_gate_mul_residual_out_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, int);

void bias_gate_mul_residual_out_bf16(const __nv_bfloat16* residual,
                                     const __nv_bfloat16* x,
                                     const __nv_bfloat16* bias,
                                     const __nv_bfloat16* gate,
                                     __nv_bfloat16* out,
                                     int seq_len, int dim,
                                     cudaStream_t stream) {
    bias_gate_mul_residual_out_kernel<__nv_bfloat16>
        <<<seq_len, 256, 0, stream>>>(residual, x, bias, gate, out, dim);
}

template<typename T>
__global__ void bias_gate_mul_residual_out_g1d_kernel(
    const T* __restrict__ residual,
    const T* __restrict__ x,
    const T* __restrict__ bias,
    const T* __restrict__ gate,
    T* __restrict__ out,
    int dim) {
    using T2 = typename packed2<T>::type;
    int row = blockIdx.x;
    const T2* res2 = reinterpret_cast<const T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], gv = g2[i], bv = b2[i];
        float r0 = to_f32(rv.x)
                 + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
        float r1 = to_f32(rv.y)
                 + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
        out2[i] = make_packed2<T>(from_f32<T>(r0), from_f32<T>(r1));
    }
}

template __global__ void bias_gate_mul_residual_out_g1d_kernel<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, int);

void bias_gate_mul_residual_out_bf16_g1d(const __nv_bfloat16* residual,
                                         const __nv_bfloat16* x,
                                         const __nv_bfloat16* bias,
                                         const __nv_bfloat16* gate_1d,
                                         __nv_bfloat16* out,
                                         int seq_len, int dim,
                                         cudaStream_t stream) {
    bias_gate_mul_residual_out_g1d_kernel<__nv_bfloat16>
        <<<seq_len, 256, 0, stream>>>(residual, x, bias, gate_1d, out, dim);
}

__global__ void bias_gate_mul_residual_out_gate_fp8_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_fp8_e4m3* __restrict__ gate,
    const float* __restrict__ gate_scale,
    __nv_bfloat16* __restrict__ out,
    int dim) {
    using T2 = packed2<__nv_bfloat16>::type;
    int row = blockIdx.x;
    const T2* res2 = reinterpret_cast<const T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    T2* out2 = reinterpret_cast<T2*>(out + row * dim);
    const __nv_fp8_e4m3* g = gate + (long long)row * dim;
    int dim2 = dim >> 1;
    float gs = *gate_scale;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], xv = x2[i], bv = b2[i];
        int j = i << 1;
        float g0 = static_cast<float>(g[j]) * gs;
        float g1 = static_cast<float>(g[j + 1]) * gs;
        float r0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * g0;
        float r1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * g1;
        out2[i] = make_packed2<__nv_bfloat16>(
            from_f32<__nv_bfloat16>(r0), from_f32<__nv_bfloat16>(r1));
    }
}

void bias_gate_mul_residual_out_bf16_gate_fp8(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    const __nv_fp8_e4m3* gate,
    const float* gate_scale,
    __nv_bfloat16* out,
    int seq_len, int dim,
    cudaStream_t stream) {
    bias_gate_mul_residual_out_gate_fp8_kernel
        <<<seq_len, 256, 0, stream>>>(
            residual, x, bias, gate, gate_scale, out, dim);
}

__global__ void motus_joint_residual3_out_kernel(
    const __nv_bfloat16* __restrict__ v_residual,
    const __nv_bfloat16* __restrict__ v_x,
    const __nv_bfloat16* __restrict__ v_bias,
    const __nv_bfloat16* __restrict__ v_gate,
    __nv_bfloat16* __restrict__ v_out,
    int v_n2, int v_dim2,
    const __nv_bfloat16* __restrict__ a_residual,
    const __nv_bfloat16* __restrict__ a_x,
    const __nv_bfloat16* __restrict__ a_bias,
    const __nv_bfloat16* __restrict__ a_gate,
    __nv_bfloat16* __restrict__ a_out,
    int a_n2, int a_dim2,
    const __nv_bfloat16* __restrict__ u_residual,
    const __nv_bfloat16* __restrict__ u_x,
    __nv_bfloat16* __restrict__ u_out,
    int u_n2, int u_dim2) {
    using T2 = packed2<__nv_bfloat16>::type;
    int v_rows = v_n2 / v_dim2;
    int a_rows = a_n2 / a_dim2;
    int u_rows = u_n2 / u_dim2;
    int row = blockIdx.x;
    if (row < v_rows) {
        const T2* r = reinterpret_cast<const T2*>(v_residual);
        const T2* x = reinterpret_cast<const T2*>(v_x);
        const T2* g = reinterpret_cast<const T2*>(v_gate);
        const T2* b = reinterpret_cast<const T2*>(v_bias);
        T2* o = reinterpret_cast<T2*>(v_out);
        int base = row * v_dim2;
        for (int col = threadIdx.x; col < v_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[idx], bv = b[col];
            float o0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows) {
        int a_row = row - v_rows;
        const T2* r = reinterpret_cast<const T2*>(a_residual);
        const T2* x = reinterpret_cast<const T2*>(a_x);
        const T2* g = reinterpret_cast<const T2*>(a_gate);
        const T2* b = reinterpret_cast<const T2*>(a_bias);
        T2* o = reinterpret_cast<T2*>(a_out);
        int base = a_row * a_dim2;
        for (int col = threadIdx.x; col < a_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[idx], bv = b[col];
            float o0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows + u_rows) {
        int u_row = row - v_rows - a_rows;
        const T2* r = reinterpret_cast<const T2*>(u_residual);
        const T2* x = reinterpret_cast<const T2*>(u_x);
        T2* o = reinterpret_cast<T2*>(u_out);
        int base = u_row * u_dim2;
        for (int col = threadIdx.x; col < u_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx];
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(xv.x)),
                from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(xv.y)));
        }
    }
}

void motus_joint_residual3_out_bf16(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_bfloat16* v_gate,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_bias,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream) {
    int v_n2 = v_n >> 1;
    int a_n2 = a_n >> 1;
    int u_n2 = u_n >> 1;
    int total_rows = (v_n / v_dim) + (a_n / a_dim) + (u_n / u_dim);
    motus_joint_residual3_out_kernel<<<total_rows, 256, 0, stream>>>(
        v_residual, v_x, v_bias, v_gate, v_out, v_n2, v_dim >> 1,
        a_residual, a_x, a_bias, a_gate, a_out, a_n2, a_dim >> 1,
        u_residual, u_x, u_out, u_n2, u_dim >> 1);
}

__global__ void motus_joint_residual3_out_vgate_fp8_kernel(
    const __nv_bfloat16* __restrict__ v_residual,
    const __nv_bfloat16* __restrict__ v_x,
    const __nv_bfloat16* __restrict__ v_bias,
    const __nv_fp8_e4m3* __restrict__ v_gate,
    const float* __restrict__ v_gate_scale,
    __nv_bfloat16* __restrict__ v_out,
    int v_n2, int v_dim2,
    const __nv_bfloat16* __restrict__ a_residual,
    const __nv_bfloat16* __restrict__ a_x,
    const __nv_bfloat16* __restrict__ a_bias,
    const __nv_bfloat16* __restrict__ a_gate,
    __nv_bfloat16* __restrict__ a_out,
    int a_n2, int a_dim2,
    const __nv_bfloat16* __restrict__ u_residual,
    const __nv_bfloat16* __restrict__ u_x,
    __nv_bfloat16* __restrict__ u_out,
    int u_n2, int u_dim2) {
    using T2 = packed2<__nv_bfloat16>::type;
    int v_rows = v_n2 / v_dim2;
    int a_rows = a_n2 / a_dim2;
    int u_rows = u_n2 / u_dim2;
    int row = blockIdx.x;
    if (row < v_rows) {
        const T2* r = reinterpret_cast<const T2*>(v_residual);
        const T2* x = reinterpret_cast<const T2*>(v_x);
        const T2* b = reinterpret_cast<const T2*>(v_bias);
        T2* o = reinterpret_cast<T2*>(v_out);
        float gs = *v_gate_scale;
        int base = row * v_dim2;
        const __nv_fp8_e4m3* g = v_gate + (long long)row * (v_dim2 << 1);
        for (int col = threadIdx.x; col < v_dim2; col += blockDim.x) {
            int idx = base + col;
            int j = col << 1;
            T2 rv = r[idx], xv = x[idx], bv = b[col];
            float g0 = static_cast<float>(g[j]) * gs;
            float g1 = static_cast<float>(g[j + 1]) * gs;
            float o0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * g0;
            float o1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * g1;
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows) {
        int a_row = row - v_rows;
        const T2* r = reinterpret_cast<const T2*>(a_residual);
        const T2* x = reinterpret_cast<const T2*>(a_x);
        const T2* g = reinterpret_cast<const T2*>(a_gate);
        const T2* b = reinterpret_cast<const T2*>(a_bias);
        T2* o = reinterpret_cast<T2*>(a_out);
        int base = a_row * a_dim2;
        for (int col = threadIdx.x; col < a_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[idx], bv = b[col];
            float o0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows + u_rows) {
        int u_row = row - v_rows - a_rows;
        const T2* r = reinterpret_cast<const T2*>(u_residual);
        const T2* x = reinterpret_cast<const T2*>(u_x);
        T2* o = reinterpret_cast<T2*>(u_out);
        int base = u_row * u_dim2;
        for (int col = threadIdx.x; col < u_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx];
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(xv.x)),
                from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(xv.y)));
        }
    }
}

void motus_joint_residual3_out_bf16_vgate_fp8(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_fp8_e4m3* v_gate,
    const float* v_gate_scale,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_bias,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream) {
    int v_n2 = v_n >> 1;
    int a_n2 = a_n >> 1;
    int u_n2 = u_n >> 1;
    int total_rows = (v_n / v_dim) + (a_n / a_dim) + (u_n / u_dim);
    motus_joint_residual3_out_vgate_fp8_kernel<<<total_rows, 256, 0, stream>>>(
        v_residual, v_x, v_bias, v_gate, v_gate_scale, v_out, v_n2, v_dim >> 1,
        a_residual, a_x, a_bias, a_gate, a_out, a_n2, a_dim >> 1,
        u_residual, u_x, u_out, u_n2, u_dim >> 1);
}

__global__ void motus_joint_residual3_out_action_nobias_kernel(
    const __nv_bfloat16* __restrict__ v_residual,
    const __nv_bfloat16* __restrict__ v_x,
    const __nv_bfloat16* __restrict__ v_bias,
    const __nv_bfloat16* __restrict__ v_gate,
    __nv_bfloat16* __restrict__ v_out,
    int v_n2, int v_dim2,
    const __nv_bfloat16* __restrict__ a_residual,
    const __nv_bfloat16* __restrict__ a_x,
    const __nv_bfloat16* __restrict__ a_gate,
    __nv_bfloat16* __restrict__ a_out,
    int a_n2, int a_dim2,
    const __nv_bfloat16* __restrict__ u_residual,
    const __nv_bfloat16* __restrict__ u_x,
    __nv_bfloat16* __restrict__ u_out,
    int u_n2, int u_dim2) {
    using T2 = packed2<__nv_bfloat16>::type;
    int v_rows = v_n2 / v_dim2;
    int a_rows = a_n2 / a_dim2;
    int u_rows = u_n2 / u_dim2;
    int row = blockIdx.x;
    if (row < v_rows) {
        const T2* r = reinterpret_cast<const T2*>(v_residual);
        const T2* x = reinterpret_cast<const T2*>(v_x);
        const T2* g = reinterpret_cast<const T2*>(v_gate);
        const T2* b = reinterpret_cast<const T2*>(v_bias);
        T2* o = reinterpret_cast<T2*>(v_out);
        int base = row * v_dim2;
        for (int col = threadIdx.x; col < v_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[idx], bv = b[col];
            float o0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows) {
        int a_row = row - v_rows;
        const T2* r = reinterpret_cast<const T2*>(a_residual);
        const T2* x = reinterpret_cast<const T2*>(a_x);
        const T2* g = reinterpret_cast<const T2*>(a_gate);
        T2* o = reinterpret_cast<T2*>(a_out);
        int base = a_row * a_dim2;
        for (int col = threadIdx.x; col < a_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[idx];
            float o0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows + u_rows) {
        int u_row = row - v_rows - a_rows;
        const T2* r = reinterpret_cast<const T2*>(u_residual);
        const T2* x = reinterpret_cast<const T2*>(u_x);
        T2* o = reinterpret_cast<T2*>(u_out);
        int base = u_row * u_dim2;
        for (int col = threadIdx.x; col < u_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx];
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(xv.x)),
                from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(xv.y)));
        }
    }
}

void motus_joint_residual3_out_bf16_action_nobias(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_bfloat16* v_gate,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream) {
    int total_rows = (v_n / v_dim) + (a_n / a_dim) + (u_n / u_dim);
    motus_joint_residual3_out_action_nobias_kernel<<<total_rows, 256, 0, stream>>>(
        v_residual, v_x, v_bias, v_gate, v_out, v_n >> 1, v_dim >> 1,
        a_residual, a_x, a_gate, a_out, a_n >> 1, a_dim >> 1,
        u_residual, u_x, u_out, u_n >> 1, u_dim >> 1);
}

__global__ void motus_joint_residual3_out_g1d_action_nobias_kernel(
    const __nv_bfloat16* __restrict__ v_residual,
    const __nv_bfloat16* __restrict__ v_x,
    const __nv_bfloat16* __restrict__ v_bias,
    const __nv_bfloat16* __restrict__ v_gate_1d,
    __nv_bfloat16* __restrict__ v_out,
    int v_n2, int v_dim2,
    const __nv_bfloat16* __restrict__ a_residual,
    const __nv_bfloat16* __restrict__ a_x,
    const __nv_bfloat16* __restrict__ a_gate_1d,
    __nv_bfloat16* __restrict__ a_out,
    int a_n2, int a_dim2,
    const __nv_bfloat16* __restrict__ u_residual,
    const __nv_bfloat16* __restrict__ u_x,
    __nv_bfloat16* __restrict__ u_out,
    int u_n2, int u_dim2) {
    using T2 = packed2<__nv_bfloat16>::type;
    int v_rows = v_n2 / v_dim2;
    int a_rows = a_n2 / a_dim2;
    int u_rows = u_n2 / u_dim2;
    int row = blockIdx.x;
    if (row < v_rows) {
        const T2* r = reinterpret_cast<const T2*>(v_residual);
        const T2* x = reinterpret_cast<const T2*>(v_x);
        const T2* g = reinterpret_cast<const T2*>(v_gate_1d);
        const T2* b = reinterpret_cast<const T2*>(v_bias);
        T2* o = reinterpret_cast<T2*>(v_out);
        int base = row * v_dim2;
        for (int col = threadIdx.x; col < v_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[col], bv = b[col];
            float o0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows) {
        int a_row = row - v_rows;
        const T2* r = reinterpret_cast<const T2*>(a_residual);
        const T2* x = reinterpret_cast<const T2*>(a_x);
        const T2* g = reinterpret_cast<const T2*>(a_gate_1d);
        T2* o = reinterpret_cast<T2*>(a_out);
        int base = a_row * a_dim2;
        for (int col = threadIdx.x; col < a_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[col];
            float o0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows + u_rows) {
        int u_row = row - v_rows - a_rows;
        const T2* r = reinterpret_cast<const T2*>(u_residual);
        const T2* x = reinterpret_cast<const T2*>(u_x);
        T2* o = reinterpret_cast<T2*>(u_out);
        int base = u_row * u_dim2;
        for (int col = threadIdx.x; col < u_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx];
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(xv.x)),
                from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(xv.y)));
        }
    }
}

void motus_joint_residual3_out_bf16_g1d_action_nobias(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_bfloat16* v_gate_1d,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_gate_1d,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream) {
    int total_rows = (v_n / v_dim) + (a_n / a_dim) + (u_n / u_dim);
    motus_joint_residual3_out_g1d_action_nobias_kernel<<<total_rows, 256, 0, stream>>>(
        v_residual, v_x, v_bias, v_gate_1d, v_out, v_n >> 1, v_dim >> 1,
        a_residual, a_x, a_gate_1d, a_out, a_n >> 1, a_dim >> 1,
        u_residual, u_x, u_out, u_n >> 1, u_dim >> 1);
}

__global__ void motus_joint_residual3_out_vgate_fp8_action_nobias_kernel(
    const __nv_bfloat16* __restrict__ v_residual,
    const __nv_bfloat16* __restrict__ v_x,
    const __nv_bfloat16* __restrict__ v_bias,
    const __nv_fp8_e4m3* __restrict__ v_gate,
    const float* __restrict__ v_gate_scale,
    __nv_bfloat16* __restrict__ v_out,
    int v_n2, int v_dim2,
    const __nv_bfloat16* __restrict__ a_residual,
    const __nv_bfloat16* __restrict__ a_x,
    const __nv_bfloat16* __restrict__ a_gate,
    __nv_bfloat16* __restrict__ a_out,
    int a_n2, int a_dim2,
    const __nv_bfloat16* __restrict__ u_residual,
    const __nv_bfloat16* __restrict__ u_x,
    __nv_bfloat16* __restrict__ u_out,
    int u_n2, int u_dim2) {
    using T2 = packed2<__nv_bfloat16>::type;
    int v_rows = v_n2 / v_dim2;
    int a_rows = a_n2 / a_dim2;
    int u_rows = u_n2 / u_dim2;
    int row = blockIdx.x;
    if (row < v_rows) {
        const T2* r = reinterpret_cast<const T2*>(v_residual);
        const T2* x = reinterpret_cast<const T2*>(v_x);
        const T2* b = reinterpret_cast<const T2*>(v_bias);
        T2* o = reinterpret_cast<T2*>(v_out);
        float gs = *v_gate_scale;
        int base = row * v_dim2;
        const __nv_fp8_e4m3* g = v_gate + (long long)row * (v_dim2 << 1);
        for (int col = threadIdx.x; col < v_dim2; col += blockDim.x) {
            int idx = base + col;
            int j = col << 1;
            T2 rv = r[idx], xv = x[idx], bv = b[col];
            float g0 = static_cast<float>(g[j]) * gs;
            float g1 = static_cast<float>(g[j + 1]) * gs;
            float o0 = to_f32(rv.x) + (to_f32(xv.x) + to_f32(bv.x)) * g0;
            float o1 = to_f32(rv.y) + (to_f32(xv.y) + to_f32(bv.y)) * g1;
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows) {
        int a_row = row - v_rows;
        const T2* r = reinterpret_cast<const T2*>(a_residual);
        const T2* x = reinterpret_cast<const T2*>(a_x);
        const T2* g = reinterpret_cast<const T2*>(a_gate);
        T2* o = reinterpret_cast<T2*>(a_out);
        int base = a_row * a_dim2;
        for (int col = threadIdx.x; col < a_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx], gv = g[idx];
            float o0 = to_f32(rv.x) + to_f32(xv.x) * to_f32(gv.x);
            float o1 = to_f32(rv.y) + to_f32(xv.y) * to_f32(gv.y);
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(o0), from_f32<__nv_bfloat16>(o1));
        }
    } else if (row < v_rows + a_rows + u_rows) {
        int u_row = row - v_rows - a_rows;
        const T2* r = reinterpret_cast<const T2*>(u_residual);
        const T2* x = reinterpret_cast<const T2*>(u_x);
        T2* o = reinterpret_cast<T2*>(u_out);
        int base = u_row * u_dim2;
        for (int col = threadIdx.x; col < u_dim2; col += blockDim.x) {
            int idx = base + col;
            T2 rv = r[idx], xv = x[idx];
            o[idx] = make_packed2<__nv_bfloat16>(
                from_f32<__nv_bfloat16>(to_f32(rv.x) + to_f32(xv.x)),
                from_f32<__nv_bfloat16>(to_f32(rv.y) + to_f32(xv.y)));
        }
    }
}

void motus_joint_residual3_out_bf16_vgate_fp8_action_nobias(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_fp8_e4m3* v_gate,
    const float* v_gate_scale,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream) {
    int total_rows = (v_n / v_dim) + (a_n / a_dim) + (u_n / u_dim);
    motus_joint_residual3_out_vgate_fp8_action_nobias_kernel
        <<<total_rows, 256, 0, stream>>>(
            v_residual, v_x, v_bias, v_gate, v_gate_scale,
            v_out, v_n >> 1, v_dim >> 1,
            a_residual, a_x, a_gate, a_out, a_n >> 1, a_dim >> 1,
            u_residual, u_x, u_out, u_n >> 1, u_dim >> 1);
}

// ── Classifier-Free Guidance combine ──
// In-place: residual[i] += v_uncond[i] + beta * (v_cond[i] - v_uncond[i])
template<typename T>
__global__ void cfg_combine_kernel(T* __restrict__ residual,
                                   const T* __restrict__ v_cond,
                                   const T* __restrict__ v_uncond,
                                   float beta, int n) {
    using T2 = typename packed2<T>::type;
    T2* res2 = reinterpret_cast<T2*>(residual);
    const T2* vc2 = reinterpret_cast<const T2*>(v_cond);
    const T2* vu2 = reinterpret_cast<const T2*>(v_uncond);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 rv = res2[idx], vc = vc2[idx], vu = vu2[idx];
        float rx = to_f32(rv.x), ry = to_f32(rv.y);
        float vcx = to_f32(vc.x), vcy = to_f32(vc.y);
        float vux = to_f32(vu.x), vuy = to_f32(vu.y);
        float gx = vux + beta * (vcx - vux);
        float gy = vuy + beta * (vcy - vuy);
        res2[idx] = make_packed2<T>(
            from_f32<T>(rx + gx),
            from_f32<T>(ry + gy));
    }
}

template __global__ void cfg_combine_kernel<__half>(
    __half*, const __half*, const __half*, float, int);
template __global__ void cfg_combine_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, int);

void cfg_combine_into_residual(__nv_bfloat16* residual,
                               const __nv_bfloat16* v_cond,
                               const __nv_bfloat16* v_uncond,
                               float beta, int n,
                               cudaStream_t stream) {
    int n2 = n >> 1;
    cfg_combine_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(
        residual, v_cond, v_uncond, beta, n);
}

void cfg_combine_into_residual_fp16(__half* residual,
                                    const __half* v_cond,
                                    const __half* v_uncond,
                                    float beta, int n,
                                    cudaStream_t stream) {
    int n2 = n >> 1;
    cfg_combine_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(
        residual, v_cond, v_uncond, beta, n);
}

// ================================================================
// GPU memory/copy ops for CUDA Graph compatibility (DiT pipeline)
// These replace PyTorch .copy_()/.fill_()/.half() which don't
// submit to the correct CUDA stream during graph capture.
// ================================================================

void gpu_copy_async(void* dst, const void* src, size_t nbytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream);
}

// Fill FP16 buffer with large negative (for softmax masking in attention)
__global__ void fill_neginf_fp16_kernel(__half* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = __float2half(-1e30f);
}
void gpu_fill_neginf_fp16(__half* dst, int n, cudaStream_t stream) {
    fill_neginf_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, n);
}

// Strided copy: src[rows, src_cols] col_offset:col_offset+dst_cols → dst[rows, dst_cols]
// For QKV split: [Sa, 3D] → [Sa, D] at offsets 0, D, 2D
__global__ void strided_copy_fp16_kernel(const __half* src, __half* dst,
                                          int rows, int dst_cols, int src_stride, int col_offset) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    for (int cc = c; cc < dst_cols; cc += blockDim.x) {
        dst[r * dst_cols + cc] = src[r * src_stride + col_offset + cc];
    }
}
void gpu_strided_copy_fp16(const __half* src, __half* dst,
                            int rows, int dst_cols, int src_stride, int col_offset,
                            cudaStream_t stream) {
    strided_copy_fp16_kernel<<<rows, min(256, dst_cols), 0, stream>>>(
        src, dst, rows, dst_cols, src_stride, col_offset);
}

// Cast FP32 → FP16
__global__ void cast_fp32_fp16_kernel(const float* src, __half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(src[idx]);
}
void gpu_cast_fp32_to_fp16(const float* src, __half* dst, int n, cudaStream_t stream) {
    cast_fp32_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

// Euler step: actions_fp32[0:T*D] += dt * velocity_fp16[offset:offset+T*D]
__global__ void euler_step_kernel(float* actions, const __half* velocity,
                                   float dt, int n, int vel_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) actions[idx] += dt * __half2float(velocity[vel_offset + idx]);
}
// ── GQA KV repeat interleave (8→16 heads) ──
// src: [S, NH_src * HD], dst: [S, NH_dst * HD] where NH_dst = NH_src * repeat
// Each src head is copied `repeat` times to consecutive dst heads
__global__ void repeat_interleave_heads_kernel(
    const __half* src, __half* dst,
    int S, int NH_src, int HD, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    if (idx >= total) return;

    int d = idx % HD;
    int remainder = idx / HD;
    int h_dst = remainder % NH_dst;
    int s = remainder / NH_dst;

    int h_src = h_dst / repeat;
    dst[s * NH_dst * HD + h_dst * HD + d] = src[s * NH_src * HD + h_src * HD + d];
}

void gpu_repeat_interleave_heads(const __half* src, __half* dst,
                                  int S, int NH_src, int HD, int repeat,
                                  cudaStream_t stream) {
    int NH_dst = NH_src * repeat;
    int total = S * NH_dst * HD;
    repeat_interleave_heads_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        src, dst, S, NH_src, HD, repeat);
}

void gpu_euler_step(float* actions, const __half* velocity,
                     int T, int action_dim, float dt, int vel_elem_offset,
                     cudaStream_t stream) {
    int n = T * action_dim;
    euler_step_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        actions, velocity, dt, n, vel_elem_offset);
}

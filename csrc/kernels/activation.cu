// ================================================================
// FlashRT — Activation kernels (dtype-generic)
// GeLU, SiLU, Gate*Act*Mul (BF16/FP16 and fused FP8 variants)
// Supports: __half (FP16), __nv_bfloat16 (BF16) via templates
// ================================================================

#include "activation.cuh"
#include "common.cuh"

// ── Gate GELU Multiply ──
// GELU(x) approx: x * sigmoid(1.5957691216 * x * (1 + 0.044715 * x^2))
template<typename T>
__global__ void gate_silu_mul_kernel(const T* __restrict__ gate,
                                     const T* __restrict__ up,
                                     T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_f32(gate[idx]);
        float u = to_f32(up[idx]);
        float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
        out[idx] = from_f32<T>(gelu * u);
    }
}

template __global__ void gate_silu_mul_kernel<__half>(const __half*, const __half*, __half*, int);
template __global__ void gate_silu_mul_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int);

void gate_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   __nv_bfloat16* out, int n, cudaStream_t stream) {
    gate_silu_mul_kernel<__nv_bfloat16><<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}
void gate_silu_mul_fp16(const __half* gate, const __half* up,
                        __half* out, int n, cudaStream_t stream) {
    gate_silu_mul_kernel<__half><<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
}

// ── GELU in-place ──
template<typename T>
__global__ void gelu_kernel(T* __restrict__ x, int n) {
    using T2 = typename packed2<T>::type;
    T2* x2 = reinterpret_cast<T2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 val = x2[idx];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        float t0 = tanhf(0.7978845608f * (v0 + 0.044715f * v0 * v0 * v0));
        float t1 = tanhf(0.7978845608f * (v1 + 0.044715f * v1 * v1 * v1));
        x2[idx] = make_packed2<T>(
            from_f32<T>(v0 * 0.5f * (1.0f + t0)),
            from_f32<T>(v1 * 0.5f * (1.0f + t1)));
    }
}

template __global__ void gelu_kernel<__half>(__half*, int);
template __global__ void gelu_kernel<__nv_bfloat16>(__nv_bfloat16*, int);

void gelu_inplace(__nv_bfloat16* x, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    gelu_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}
void gelu_inplace_fp16(__half* x, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    gelu_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}

// ── G7.11 — Fused (bias add + GELU(tanh)) in-place ──
// Replaces add_bias_bf16 + gelu_inplace pair (2 launches -> 1) used
// by the BF16 FFN intermediate path (_make_fvk_ffn_forward_5op).
// In-place on x; reads bias broadcast over rows.
template<typename T>
__global__ void bias_gelu_kernel(T* __restrict__ x,
                                  const T* __restrict__ bias,
                                  int M, int N) {
    using T2 = typename packed2<T>::type;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total2 = (M * N) >> 1;        // pair index
    if (idx >= total2) return;
    int n2 = N >> 1;                  // pairs per row
    // Decode pair-of-elements offset to (m, col_pair).
    int row = idx / n2;
    int col_pair = idx - row * n2;
    T2* x2 = reinterpret_cast<T2*>(x);
    const T2* b2 = reinterpret_cast<const T2*>(bias);
    T2 v = x2[idx];
    T2 b = b2[col_pair];
    float v0 = to_f32(v.x) + to_f32(b.x);
    float v1 = to_f32(v.y) + to_f32(b.y);
    float t0 = tanhf(0.7978845608f * (v0 + 0.044715f * v0 * v0 * v0));
    float t1 = tanhf(0.7978845608f * (v1 + 0.044715f * v1 * v1 * v1));
    x2[idx] = make_packed2<T>(
        from_f32<T>(v0 * 0.5f * (1.0f + t0)),
        from_f32<T>(v1 * 0.5f * (1.0f + t1)));
    (void)row;  // silence unused warning if compiler optimizes it out
}

void bias_gelu_inplace_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias,
                              int M, int N, cudaStream_t stream) {
    int total2 = (M * N) >> 1;
    bias_gelu_kernel<__nv_bfloat16><<<(total2 + 255) / 256, 256, 0, stream>>>(
        x, bias, M, N);
}

// ── Gate GELU Mul Merged ──
// Input: (seq, 2*half_dim), gate = [:, :half_dim], up = [:, half_dim:]
template<typename T>
__global__ void gate_silu_mul_merged_kernel(const T* __restrict__ merged,
                                             T* __restrict__ out,
                                             int seq, int half_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * half_dim;
    if (idx < total) {
        int row = idx / half_dim;
        int col = idx % half_dim;
        int full_dim = half_dim * 2;
        float g = to_f32(merged[row * full_dim + col]);
        float u = to_f32(merged[row * full_dim + half_dim + col]);
        float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
        out[idx] = from_f32<T>(gelu * u);
    }
}

template __global__ void gate_silu_mul_merged_kernel<__half>(const __half*, __half*, int, int);
template __global__ void gate_silu_mul_merged_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int);

void gate_silu_mul_merged(const __nv_bfloat16* merged, __nv_bfloat16* out,
                           int seq, int half_dim, cudaStream_t stream) {
    int total = seq * half_dim;
    int blocks = (total + 255) / 256;
    gate_silu_mul_merged_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(merged, out, seq, half_dim);
}
void gate_silu_mul_merged_fp16(const __half* merged, __half* out,
                                int seq, int half_dim, cudaStream_t stream) {
    int total = seq * half_dim;
    int blocks = (total + 255) / 256;
    gate_silu_mul_merged_kernel<__half><<<blocks, 256, 0, stream>>>(merged, out, seq, half_dim);
}

// ── Gate GELU Mul Merged -> FP8 ──
// 4 elem/thread vectorized, matching production silu_mul_split_fp8_k throughput.
// Merged layout: merged[s, 0..H-1] = gate, merged[s, H..2H-1] = up
__global__ void gate_silu_mul_merged_fp8_kernel_fp16(const __half* merged, __nv_fp8_e4m3* out, int S, int H,
                                       const float* descale_ptr) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 4 elements per thread
    if (i >= S * H) return;
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);

    int s = i / H, h = i % H;
    int base = s * 2 * H;
    // Vectorized half2 loads from gate and up regions
    const __half2* gate2 = reinterpret_cast<const __half2*>(merged + base + h);
    const __half2* up2 = reinterpret_cast<const __half2*>(merged + base + H + h);
    __half2 gA = gate2[0], gB = gate2[1];
    __half2 uA = up2[0],   uB = up2[1];

    float gv[4] = {__half2float(gA.x), __half2float(gA.y),
                    __half2float(gB.x), __half2float(gB.y)};
    float uv[4] = {__half2float(uA.x), __half2float(uA.y),
                    __half2float(uB.x), __half2float(uB.y)};

    __nv_fp8_e4m3 fp8_pack[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        float gelu = gv[j] / (1.0f + __expf(-1.5957691216057308f * gv[j] * (1.0f + 0.044715f * gv[j] * gv[j])));
        fp8_pack[j] = __nv_fp8_e4m3(fminf(fmaxf(gelu * uv[j] * inv_scale, -448.f), 448.f));
    }
    *reinterpret_cast<uint32_t*>(out + i) = *reinterpret_cast<uint32_t*>(fp8_pack);
}

// BF16 generic version (non-encoder paths)
template<typename T>
__global__ void gate_silu_mul_merged_fp8_kernel(const T* __restrict__ merged,
                                                 __nv_fp8_e4m3* __restrict__ out,
                                                 int seq, int half_dim,
                                                 const float* __restrict__ d_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * half_dim;
    if (idx < total) {
        int row = idx / half_dim;
        int col = idx % half_dim;
        int full_dim = half_dim * 2;
        float g = to_f32(merged[row * full_dim + col]);
        float u = to_f32(merged[row * full_dim + half_dim + col]);
        float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
        float val = gelu * u;
        float inv_scale = 1.0f / (*d_scale);
        val = fminf(fmaxf(val * inv_scale, -448.0f), 448.0f);
        out[idx] = __nv_fp8_e4m3(val);
    }
}

template __global__ void gate_silu_mul_merged_fp8_kernel<__half>(const __half*, __nv_fp8_e4m3*, int, int, const float*);
template __global__ void gate_silu_mul_merged_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, __nv_fp8_e4m3*, int, int, const float*);

void gate_silu_mul_merged_fp8(const __nv_bfloat16* merged, __nv_fp8_e4m3* out,
                               int seq, int half_dim,
                               const float* d_scale, cudaStream_t stream) {
    int total = seq * half_dim;
    int blocks = (total + 255) / 256;
    gate_silu_mul_merged_fp8_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
        merged, out, seq, half_dim, d_scale);
}
void gate_silu_mul_merged_fp8_fp16(const __half* merged, __nv_fp8_e4m3* out,
                                    int seq, int half_dim,
                                    const float* d_scale, cudaStream_t stream) {
    // 4 elem/thread, matching production throughput
    int total = seq * half_dim;
    int blocks = (total / 4 + 255) / 256;
    gate_silu_mul_merged_fp8_kernel_fp16<<<blocks, 256, 0, stream>>>(
        merged, out, seq, half_dim, d_scale);
}

// ── Split SiLU × Up → FP8 (separate gate/up buffers) ──
// Matches pi05 silu_mul_split_fp8_k: gate and up from separate GEMMs
template<typename T>
__global__ void silu_mul_split_fp8_kernel(const T* __restrict__ gate,
                                           const T* __restrict__ up,
                                           __nv_fp8_e4m3* __restrict__ out,
                                           int n, const float* __restrict__ d_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float inv_scale = 1.0f / (*d_scale);
    float g = to_f32(gate[idx]);
    float u = to_f32(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    float val = silu_g * u * inv_scale;
    out[idx] = __nv_fp8_e4m3(fminf(fmaxf(val, -448.0f), 448.0f));
}

template __global__ void silu_mul_split_fp8_kernel<__half>(const __half*, const __half*, __nv_fp8_e4m3*, int, const float*);
template __global__ void silu_mul_split_fp8_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_fp8_e4m3*, int, const float*);

void silu_mul_split_fp8_fp16(const __half* gate, const __half* up,
                              __nv_fp8_e4m3* out, int n,
                              const float* d_scale, cudaStream_t stream) {
    silu_mul_split_fp8_kernel<__half><<<(n + 255) / 256, 256, 0, stream>>>(
        gate, up, out, n, d_scale);
}

// ── SiLU in-place (for DiT action encoder) ──
template<typename T>
__global__ void silu_inplace_kernel(T* __restrict__ x, int n) {
    using T2 = typename packed2<T>::type;
    T2* x2 = reinterpret_cast<T2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 val = x2[idx];
        float v0 = to_f32(val.x), v1 = to_f32(val.y);
        float s0 = v0 / (1.0f + expf(-v0));
        float s1 = v1 / (1.0f + expf(-v1));
        x2[idx] = make_packed2<T>(from_f32<T>(s0), from_f32<T>(s1));
    }
}

template __global__ void silu_inplace_kernel<__half>(__half*, int);

void silu_inplace_fp16(__half* x, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    silu_inplace_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}

// ── Fused add + SiLU: a = silu(a + b), used by Pi0 action_time_mlp ──
template<typename T>
__global__ void fused_add_silu_kernel(T* __restrict__ a,
                                       const T* __restrict__ b, int n) {
    using T2 = typename packed2<T>::type;
    T2* a2 = reinterpret_cast<T2*>(a);
    const T2* b2 = reinterpret_cast<const T2*>(b);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 va = a2[idx], vb = b2[idx];
        float v0 = to_f32(va.x) + to_f32(vb.x);
        float v1 = to_f32(va.y) + to_f32(vb.y);
        float s0 = v0 / (1.0f + expf(-v0));
        float s1 = v1 / (1.0f + expf(-v1));
        a2[idx] = make_packed2<T>(from_f32<T>(s0), from_f32<T>(s1));
    }
}

template __global__ void fused_add_silu_kernel<__half>(__half*, const __half*, int);
template __global__ void fused_add_silu_kernel<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, int);

void fused_add_silu_fp16(__half* a, const __half* b, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    fused_add_silu_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(a, b, n);
}

void fused_add_silu_bf16(__nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    fused_add_silu_kernel<__nv_bfloat16><<<(n2 + 255) / 256, 256, 0, stream>>>(a, b, n);
}

// ── ReLU in-place (for DiT action decoder) ──
template<typename T>
__global__ void relu_inplace_kernel(T* __restrict__ x, int n) {
    using T2 = typename packed2<T>::type;
    T2* x2 = reinterpret_cast<T2*>(x);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        T2 val = x2[idx];
        float v0 = fmaxf(to_f32(val.x), 0.0f);
        float v1 = fmaxf(to_f32(val.y), 0.0f);
        x2[idx] = make_packed2<T>(from_f32<T>(v0), from_f32<T>(v1));
    }
}

template __global__ void relu_inplace_kernel<__half>(__half*, int);

void relu_inplace_fp16(__half* x, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    relu_inplace_kernel<__half><<<(n2 + 255) / 256, 256, 0, stream>>>(x, n);
}

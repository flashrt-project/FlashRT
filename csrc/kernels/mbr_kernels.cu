// ================================================================
// FlashRT — MelBandRoformer custom fused kernels
//
// Conventions (verified against rotary_embedding_torch):
//   - RoPE is INTERLEAVED (pairs of adjacent elements); cos/sin tables (S, D/2).
//   - QKV layout: (B, S, 3*H*D) = [Q(H*D) | K(H*D) | V(H*D)], each H*D = h0(D)...
//   - melband RMSNorm = x/||x|| * sqrt(dim) * gamma == x * rms * gamma where
//     rms = rsqrt(mean(x^2)+eps) (sqrt(dim) folded into rms).
//
// V70: Added fused_add_rmsnorm_bf16 to eliminate final_norm overhead
// ================================================================

#include "mbr_kernels.cuh"
#include "common.cuh"

namespace flash_rt { namespace mbr {

// ── 1) fused QKV split + interleaved RoPE -> (B,H,S,D) for SDPA ──
template<typename T>
__global__ void qkv_split_rope_kernel(const T* __restrict__ qkv,
    const float* __restrict__ cosT, const float* __restrict__ sinT,
    T* __restrict__ Q, T* __restrict__ K, T* __restrict__ V,
    int B, int S, int H, int D) {
    using T2 = typename packed2<T>::type;
    const int HD = H * D, HALF = D >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * S * HALF) return;
    int pair = idx % HALF; idx /= HALF;
    int t = idx % S; idx /= S;
    int h = idx % H; int b = idx / H;

    int qbase  = (b * S + t) * (3 * HD);
    int hoff   = h * D + pair * 2;
    int outoff = ((b * H + h) * S + t) * D + pair * 2;
    float c = cosT[t * HALF + pair], s = sinT[t * HALF + pair];

    // Q — rotate
    T2 qp = reinterpret_cast<const T2*>(qkv + qbase + hoff)[0];
    float q0 = to_f32<T>(qp.x), q1 = to_f32<T>(qp.y);
    reinterpret_cast<T2*>(Q + outoff)[0] =
        make_packed2<T>(from_f32<T>(q0 * c - q1 * s),
                        from_f32<T>(q0 * s + q1 * c));

    // K — rotate
    T2 kp = reinterpret_cast<const T2*>(qkv + qbase + HD + hoff)[0];
    float k0 = to_f32<T>(kp.x), k1 = to_f32<T>(kp.y);
    reinterpret_cast<T2*>(K + outoff)[0] =
        make_packed2<T>(from_f32<T>(k0 * c - k1 * s),
                        from_f32<T>(k0 * s + k1 * c));

    // V — copy (no rotation)
    reinterpret_cast<T2*>(V + outoff)[0] =
        reinterpret_cast<const T2*>(qkv + qbase + 2 * HD + hoff)[0];
}

// ── 2) fused sigmoid(gates)*attn + reshape (B,H,S,D)->(B,S,H*D) + FP8 quant ──
template<typename T>
__global__ void gated_attn_quant_kernel(const T* __restrict__ o,
    const T* __restrict__ gates, __nv_fp8_e4m3* __restrict__ out_fp8,
    int B, int H, int S, int D, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H * D;
    if (idx >= total) return;
    int d = idx % D; int tmp = idx / D;
    int h = tmp % H; tmp /= H;
    int t = tmp % S; int b = tmp / S;
    float g = to_f32<T>(gates[(b * S + t) * H + h]);
    g = 1.0f / (1.0f + expf(-g));
    float val = to_f32<T>(o[((b * H + h) * S + t) * D + d]) * g;
    float q = fminf(fmaxf(val / scale, -448.0f), 447.0f);
    out_fp8[(b * S + t) * (H * D) + h * D + d] = __nv_fp8_e4m3(q);
}

// ── 3) FP8 -> T dequant with scalar scale ──
template<typename T>
__global__ void fp8_dequant_kernel(const __nv_fp8_e4m3* __restrict__ inp,
    float scale, T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __half h = static_cast<__half>(inp[idx]);
    out[idx] = from_f32<T>(__half2float(h) * scale);
}

// ── 4) fused residual_add + RMSNorm -> FP8 (keeps summed residual) ──
template<typename T>
__global__ void resadd_rmsnorm_fp8_keepres_kernel(const T* __restrict__ a,
    const T* __restrict__ b, const T* __restrict__ gamma,
    T* __restrict__ sum_out, __nv_fp8_e4m3* __restrict__ norm_fp8,
    int dim, float eps, float inv_scale) {
    using T2 = typename packed2<T>::type;
    extern __shared__ float shared[];
    int row = blockIdx.x, tid = threadIdx.x, off = row * dim;
    int dim2 = dim >> 1;

    // residual add + partial sum-of-squares (vectorised via packed2)
    const T2* a2 = reinterpret_cast<const T2*>(a + off);
    const T2* b2 = reinterpret_cast<const T2*>(b + off);
    T2* s2       = reinterpret_cast<T2*>(sum_out + off);
    float local  = 0.0f;
    for (int i = tid; i < dim2; i += blockDim.x) {
        T2 av = a2[i], bv = b2[i];
        float s0 = to_f32<T>(av.x) + to_f32<T>(bv.x);
        float s1 = to_f32<T>(av.y) + to_f32<T>(bv.y);
        s2[i] = make_packed2<T>(from_f32<T>(s0), from_f32<T>(s1));
        local += s0 * s0 + s1 * s1;
    }
    float rms = rsqrtf(block_reduce_sum(local, shared) / dim + eps);

    // normalize + quantize to FP8 (element-wise — no clean packed2 for FP8 writes)
    for (int i = tid; i < dim; i += blockDim.x) {
        float sv  = to_f32<T>(sum_out[off + i]);
        float gv  = to_f32<T>(gamma[i]);
        norm_fp8[off + i] = __nv_fp8_e4m3(
            fminf(fmaxf(sv * rms * gv * inv_scale, -448.0f), 447.0f));
    }
}

// ── 5) V70 NEW: Fused residual_add + RMSNorm (BF16 in/out, for final_norm) ──
template<typename T>
__global__ void fused_add_rmsnorm_bf16_kernel(
    const T* __restrict__ a,        // ff_out
    const T* __restrict__ b,        // x_new (residual)
    const T* __restrict__ gamma,    // norm weight
    T* __restrict__ out,            // final output
    int dim, float eps) {

    using T2 = typename packed2<T>::type;
    extern __shared__ char smem[];
    float* reduce_buf = reinterpret_cast<float*>(smem);
    T* sum_buf = reinterpret_cast<T*>(smem + 256 * sizeof(float));

    int row = blockIdx.x, tid = threadIdx.x, off = row * dim;
    int dim2 = dim >> 1;

    // Phase 1: Residual add + compute sum-of-squares
    const T2* a2 = reinterpret_cast<const T2*>(a + off);
    const T2* b2 = reinterpret_cast<const T2*>(b + off);
    T2* sum2 = reinterpret_cast<T2*>(sum_buf);

    float local = 0.0f;
    for (int i = tid; i < dim2; i += blockDim.x) {
        T2 av = a2[i], bv = b2[i];
        float s0 = to_f32<T>(av.x) + to_f32<T>(bv.x);
        float s1 = to_f32<T>(av.y) + to_f32<T>(bv.y);
        sum2[i] = make_packed2<T>(from_f32<T>(s0), from_f32<T>(s1));
        local += s0 * s0 + s1 * s1;
    }
    __syncthreads();

    // Reduce to get RMS
    float rms = rsqrtf(block_reduce_sum(local, reduce_buf) / dim + eps);

    // Phase 2: Normalize
    T2* out2 = reinterpret_cast<T2*>(out + off);
    const T2* g2 = reinterpret_cast<const T2*>(gamma);

    for (int i = tid; i < dim2; i += blockDim.x) {
        T2 sv = sum2[i];
        T2 gv = g2[i];
        float o0 = to_f32<T>(sv.x) * rms * to_f32<T>(gv.x);
        float o1 = to_f32<T>(sv.y) * rms * to_f32<T>(gv.y);
        out2[i] = make_packed2<T>(from_f32<T>(o0), from_f32<T>(o1));
    }

    // Handle odd dimension
    if ((dim & 1) && tid == 0) {
        int last = dim - 1;
        float s = to_f32<T>(a[off + last]) + to_f32<T>(b[off + last]);
        out[off + last] = from_f32<T>(s * rms * to_f32<T>(gamma[last]));
    }
}

// ── BF16 launchers (primary dtype) ──

void qkv_split_rope(const __nv_bfloat16* qkv, const float* cosT, const float* sinT,
                    __nv_bfloat16* Q, __nv_bfloat16* K, __nv_bfloat16* V,
                    int B, int S, int H, int D, cudaStream_t st) {
    int n = B * H * S * (D >> 1);
    qkv_split_rope_kernel<__nv_bfloat16><<<(n + 255) / 256, 256, 0, st>>>(
        qkv, cosT, sinT, Q, K, V, B, S, H, D);
}

void gated_attn_quant(const __nv_bfloat16* o, const __nv_bfloat16* gates,
                      __nv_fp8_e4m3* out_fp8,
                      int B, int H, int S, int D, float scale, cudaStream_t st) {
    int n = B * S * H * D;
    gated_attn_quant_kernel<__nv_bfloat16><<<(n + 255) / 256, 256, 0, st>>>(
        o, gates, out_fp8, B, H, S, D, scale);
}

void fp8_dequant_bf16(const __nv_fp8_e4m3* inp, float scale,
                      __nv_bfloat16* out, int n, cudaStream_t st) {
    fp8_dequant_kernel<__nv_bfloat16><<<(n + 255) / 256, 256, 0, st>>>(
        inp, scale, out, n);
}

void resadd_rmsnorm_fp8_keepres(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                const __nv_bfloat16* gamma,
                                __nv_bfloat16* sum_out, __nv_fp8_e4m3* norm_fp8,
                                int M, int dim, float scale, cudaStream_t st) {
    resadd_rmsnorm_fp8_keepres_kernel<__nv_bfloat16>
        <<<M, 256, 256 * sizeof(float), st>>>(
            a, b, gamma, sum_out, norm_fp8, dim, 1e-6f, 1.0f / scale);
}

// ── V70 NEW Launcher ──
void fused_add_rmsnorm_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b,
                            const __nv_bfloat16* gamma, __nv_bfloat16* out,
                            int M, int dim, cudaStream_t st) {
    int smem = 256 * sizeof(float) + dim * sizeof(__nv_bfloat16);
    fused_add_rmsnorm_bf16_kernel<__nv_bfloat16><<<M, 256, smem, st>>>(
        a, b, gamma, out, dim, 1e-6f);
}

// ══════════════════════════════════════════════════════════════════
// V72 NEW: Fused FP8 dequant + tiny Linear (for gates)  
// ══════════════════════════════════════════════════════════════════

// Kernel template (inside namespace)
template<typename T>
__global__ void fp8_dequant_tiny_linear_kernel(
    const __nv_fp8_e4m3* __restrict__ inp_fp8,
    float inp_scale,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ out,
    int M, int D_in, int D_out) {

    int m = blockIdx.x;
    if (m >= M) return;

    int tid = threadIdx.x;
    const __nv_fp8_e4m3* inp_row = inp_fp8 + m * D_in;
    T* out_row = out + m * D_out;

    for (int d = tid; d < D_out; d += blockDim.x) {
        float sum = 0.0f;
        const T* w_row = weight + d * D_in;

        #pragma unroll 8
        for (int i = 0; i < D_in; i++) {
            __half h = static_cast<__half>(inp_row[i]);
            float inp_val = __half2float(h) * inp_scale;
            float w_val = to_f32<T>(w_row[i]);
            sum += inp_val * w_val;
        }

        if (bias) {
            sum += to_f32<T>(bias[d]);
        }

        out_row[d] = from_f32<T>(sum);
    }
}

// Launcher
void fp8_dequant_tiny_linear_bf16(
    const __nv_fp8_e4m3* inp_fp8,
    float inp_scale,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* out,
    int M, int D_in, int D_out,
    cudaStream_t st) {

    fp8_dequant_tiny_linear_kernel<__nv_bfloat16>
        <<<M, 128, 0, st>>>(
            inp_fp8, inp_scale, weight, bias, out, M, D_in, D_out);
}

// ══════════════════════════════════════════════════════════════════
// V73 ULTIMATE FUSION: FP8 GEMM + bias + residual + RMSNorm
//
// 这是终极融合kernel！一次完成5个操作：
//   1. FP8 GEMM (hfp8 × w2)
//   2. + bias (w2.bias)
//   3. + residual (x_new)
//   4. RMSNorm
//   5. 输出 BF16
//
// 预期节省: 10-15ms (消除所有中间步骤和内存往返)
// ══════════════════════════════════════════════════════════════════

template<typename T>
__global__ void fp8_gemm_bias_residual_norm_kernel(
    const __nv_fp8_e4m3* __restrict__ inp_fp8,      // (M, K) - hfp8
    const __nv_fp8_e4m3* __restrict__ weight_fp8,   // (N, K) transposed - w2
    float scale_inp,
    float scale_weight,
    const T* __restrict__ bias,                      // (N) - w2.bias
    const T* __restrict__ residual,                  // (M, N) - x_new
    const T* __restrict__ gamma,                     // (N) - norm weight
    T* __restrict__ out,                             // (M, N) - output
    int M, int K, int N,
    float eps) {

    // 每个block处理一行
    int m = blockIdx.x;
    if (m >= M) return;

    int tid = threadIdx.x;
    extern __shared__ float shared[];

    const __nv_fp8_e4m3* inp_row = inp_fp8 + m * K;
    T* out_row = out + m * N;
    const T* res_row = residual + m * N;

    // Phase 1: GEMM + bias + residual (compute到shared memory)
    for (int n = tid; n < N; n += blockDim.x) {
        float sum = 0.0f;

        // Dot product for this output element
        const __nv_fp8_e4m3* w_col = weight_fp8 + n * K;
        #pragma unroll 8
        for (int k = 0; k < K; k++) {
            __half h_inp = static_cast<__half>(inp_row[k]);
            __half h_w = static_cast<__half>(w_col[k]);
            float inp_val = __half2float(h_inp) * scale_inp;
            float w_val = __half2float(h_w) * scale_weight;
            sum += inp_val * w_val;
        }

        // Add bias
        if (bias) {
            sum += to_f32<T>(bias[n]);
        }

        // Add residual
        sum += to_f32<T>(res_row[n]);

        // Store in shared memory for RMSNorm
        shared[n] = sum;
    }
    __syncthreads();

    // Phase 2: Compute RMS
    float local_sq = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float val = shared[n];
        local_sq += val * val;
    }

    // Reduce across block
    float rms_sq = block_reduce_sum(local_sq, shared + N) / N + eps;
    float rms_inv = rsqrtf(rms_sq);
    __syncthreads();

    // Phase 3: Normalize and write output
    for (int n = tid; n < N; n += blockDim.x) {
        float val = shared[n];
        float normed = val * rms_inv * to_f32<T>(gamma[n]);
        out_row[n] = from_f32<T>(normed);
    }
}

// Launcher
void fp8_gemm_bias_residual_norm_bf16(
    const __nv_fp8_e4m3* inp_fp8,
    const __nv_fp8_e4m3* weight_fp8,
    float scale_inp,
    float scale_weight,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* gamma,
    __nv_bfloat16* out,
    int M, int K, int N,
    cudaStream_t st) {

    // Shared memory: N floats for intermediate + 256 floats for reduction
    int smem = (N + 256) * sizeof(float);

    fp8_gemm_bias_residual_norm_kernel<__nv_bfloat16>
        <<<M, 256, smem, st>>>(
            inp_fp8, weight_fp8, scale_inp, scale_weight,
            bias, residual, gamma, out, M, K, N, 1e-6f);
}


// ══════════════════════════════════════════════════════════════════
// V74: 正确的融合策略 - 处理 GEMM 输出，而不是替换 GEMM
//
// 输入：PyTorch FP8 GEMM 的输出（BF16）
// 融合：bias + residual + RMSNorm
// 这个 kernel 简单且高效，因为不涉及 GEMM 计算！
// ══════════════════════════════════════════════════════════════════

template<typename T>
__global__ void gemm_output_bias_residual_norm_kernel(
    const T* __restrict__ gemm_out,     // (M, N) - GEMM 输出
    const T* __restrict__ bias,          // (N) - 可选
    const T* __restrict__ residual,      // (M, N)
    const T* __restrict__ gamma,         // (N) - norm weight
    T* __restrict__ out,                 // (M, N)
    int M, int N,
    float eps) {

    // 每个 block 处理一行
    int m = blockIdx.x;
    if (m >= M) return;

    int tid = threadIdx.x;
    extern __shared__ float shared[];

    const T* gemm_row = gemm_out + m * N;
    const T* res_row = residual + m * N;
    T* out_row = out + m * N;

    // Phase 1: bias + residual，并计算 sum of squares
    float local_sq = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float val = to_f32<T>(gemm_row[n]);

        // Add bias if present
        if (bias) {
            val += to_f32<T>(bias[n]);
        }

        // Add residual
        val += to_f32<T>(res_row[n]);

        // Store in shared memory
        shared[n] = val;
        local_sq += val * val;
    }
    __syncthreads();

    // Phase 2: Compute RMS
    float rms_sq = block_reduce_sum(local_sq, shared + N) / N + eps;
    float rms_inv = rsqrtf(rms_sq);
    __syncthreads();

    // Phase 3: Normalize and write output
    for (int n = tid; n < N; n += blockDim.x) {
        float val = shared[n];
        float normed = val * rms_inv * to_f32<T>(gamma[n]);
        out_row[n] = from_f32<T>(normed);
    }
}

// Launcher
void gemm_output_bias_residual_norm_bf16(
    const __nv_bfloat16* gemm_out,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* gamma,
    __nv_bfloat16* out,
    int M, int N,
    cudaStream_t st) {

    // Shared memory: N floats for intermediate + 256 floats for reduction
    int smem = (N + 256) * sizeof(float);

    gemm_output_bias_residual_norm_kernel<__nv_bfloat16>
        <<<M, 256, smem, st>>>(
            gemm_out, bias, residual, gamma, out, M, N, 1e-6f);
}

}}  // namespace flash_rt::mbr
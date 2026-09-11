// Fused QKV post-processing for the pi0.5 action expert (Thor).
//
// Consumes the fused QKV GEMM's f32 output [M, Nk + Nv + Nq] (sections
// [k | v | q]) and in one launch:
//   - RoPEs K and writes it as f16 into the persistent KV buffer's suffix
//   - writes V as f16 into the persistent KV buffer's suffix
//   - RoPEs and scales Q, writing the f32 tensor flash attention consumes
//
// The RoPE math mirrors ggml-cuda's rope_neox (yarn corrections included)
// so the fused path is numerically identical to the unfused graph.

#include "fr_kernels.h"

#include <cuda_fp16.h>

namespace ggml_cuda_flashrt {

namespace {

__device__ float qkv_rope_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// mirrors ggml-cuda rope_yarn (forward)
__device__ void qkv_rope_yarn(const float theta_extrap, const float freq_scale,
                              const float corr_low, const float corr_high,
                              const int i0, const float ext_factor, float mscale,
                              float & cos_theta, float & sin_theta) {
    float theta = freq_scale * theta_extrap;
    if (ext_factor != 0.0f) {
        const float ramp_mix = qkv_rope_ramp(corr_low, corr_high, i0) * ext_factor;
        theta = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

// one block per token; threads cover the K/Q rope pairs and the V copy
__global__ void kernel_qkv_post(const float * __restrict__ qkv,   // [M, Nk+Nv+Nq]
                                float * __restrict__ q_out,       // [M, Nq] f32 (head-major rows)
                                __half * __restrict__ q16_out,    // nullable: same values as f16
                                __half * __restrict__ k_out,      // suffix rows, head_dim per token
                                __half * __restrict__ v_out,
                                float * __restrict__ k_f32_out,   // nullable: rope'd K as f32 rows
                                float * __restrict__ v_f32_out,   // nullable: V as f32 rows
                                const int32_t * __restrict__ pos,
                                const float * __restrict__ freq_factors, // nullable
                                int Nk, int Nv, int Nq,
                                int head_dim, int n_dims,
                                float freq_scale, float ext_factor, float attn_factor,
                                float corr_low, float corr_high,
                                float theta_scale, float q_scale) {
    const int t = blockIdx.x;
    const float * row = qkv + (int64_t) t * (Nk + Nv + Nq);
    const float * krow = row;
    const float * vrow = row + Nk;
    const float * qrow = row + Nk + Nv;

    const int p = pos[t];

    // V: plain f16 copy (and optionally the f32 row for downstream readers)
    for (int d = threadIdx.x; d < Nv; d += blockDim.x) {
        v_out[(int64_t) t * Nv + d] = __float2half(vrow[d]);
        if (v_f32_out != nullptr) {
            v_f32_out[(int64_t) t * Nv + d] = vrow[d];
        }
    }

    // K: rope one head (Nk == head_dim)
    for (int i = threadIdx.x; i < Nk / 2; i += blockDim.x) {
        const int i0 = 2 * i; // pair index within the head
        if (i0 >= n_dims) {
            k_out[(int64_t) t * Nk + n_dims + (i0 - n_dims)]     = __float2half(krow[n_dims + (i0 - n_dims)]);
            k_out[(int64_t) t * Nk + n_dims + (i0 - n_dims) + 1] = __float2half(krow[n_dims + (i0 - n_dims) + 1]);
            if (k_f32_out != nullptr) {
                k_f32_out[(int64_t) t * Nk + n_dims + (i0 - n_dims)]     = krow[n_dims + (i0 - n_dims)];
                k_f32_out[(int64_t) t * Nk + n_dims + (i0 - n_dims) + 1] = krow[n_dims + (i0 - n_dims) + 1];
            }
            continue;
        }
        const float theta_base  = p * powf(theta_scale, i0 / 2.0f);
        const float freq_factor = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        float cos_t, sin_t;
        qkv_rope_yarn(theta_base / freq_factor, freq_scale, corr_low, corr_high,
                      i0, ext_factor, attn_factor, cos_t, sin_t);
        const float x0 = krow[i0 / 2];
        const float x1 = krow[i0 / 2 + n_dims / 2];
        const float k0 = x0 * cos_t - x1 * sin_t;
        const float k1 = x0 * sin_t + x1 * cos_t;
        k_out[(int64_t) t * Nk + i0 / 2]              = __float2half(k0);
        k_out[(int64_t) t * Nk + i0 / 2 + n_dims / 2] = __float2half(k1);
        if (k_f32_out != nullptr) {
            k_f32_out[(int64_t) t * Nk + i0 / 2]              = k0;
            k_f32_out[(int64_t) t * Nk + i0 / 2 + n_dims / 2] = k1;
        }
    }

    // Q: rope + scale per head
    const int n_head = Nq / head_dim;
    for (int hp = threadIdx.x; hp < n_head * head_dim / 2; hp += blockDim.x) {
        const int h  = hp / (head_dim / 2);
        const int i0 = 2 * (hp % (head_dim / 2));
        const float * qh = qrow + (int64_t) h * head_dim;
        float  * oh   = q_out + (int64_t) t * Nq + (int64_t) h * head_dim;
        __half * oh16 = q16_out != nullptr
                      ? q16_out + (int64_t) t * Nq + (int64_t) h * head_dim : nullptr;
        if (i0 >= n_dims) {
            const float v0 = qh[n_dims + (i0 - n_dims)]     * q_scale;
            const float v1 = qh[n_dims + (i0 - n_dims) + 1] * q_scale;
            oh[n_dims + (i0 - n_dims)]     = v0;
            oh[n_dims + (i0 - n_dims) + 1] = v1;
            if (oh16 != nullptr) {
                oh16[n_dims + (i0 - n_dims)]     = __float2half(v0);
                oh16[n_dims + (i0 - n_dims) + 1] = __float2half(v1);
            }
            continue;
        }
        const float theta_base  = p * powf(theta_scale, i0 / 2.0f);
        const float freq_factor = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        float cos_t, sin_t;
        qkv_rope_yarn(theta_base / freq_factor, freq_scale, corr_low, corr_high,
                      i0, ext_factor, attn_factor, cos_t, sin_t);
        const float x0 = qh[i0 / 2];
        const float x1 = qh[i0 / 2 + n_dims / 2];
        const float v0 = (x0 * cos_t - x1 * sin_t) * q_scale;
        const float v1 = (x0 * sin_t + x1 * cos_t) * q_scale;
        oh[i0 / 2]              = v0;
        oh[i0 / 2 + n_dims / 2] = v1;
        if (oh16 != nullptr) {
            oh16[i0 / 2]              = __float2half(v0);
            oh16[i0 / 2 + n_dims / 2] = __float2half(v1);
        }
    }
}

} // namespace

int qkv_post_full(const float * qkv_cat, float * q_out, void * q16_out,
                  void * k_out_f16, void * v_out_f16,
                  float * k_f32_out, float * v_f32_out,
                  const int32_t * pos, const float * freq_factors,
                  int M, int Nk, int Nv, int Nq, int head_dim, int n_dims,
                  float freq_scale, float ext_factor, float attn_factor,
                  float corr_low, float corr_high, float theta_scale, float q_scale,
                  cudaStream_t stream) {
    if (n_dims % 2 != 0 || Nk != head_dim || Nq % head_dim != 0) return -1;
    kernel_qkv_post<<<M, 256, 0, stream>>>(
        qkv_cat, q_out, (__half *) q16_out, (__half *) k_out_f16, (__half *) v_out_f16,
        k_f32_out, v_f32_out,
        pos, freq_factors, Nk, Nv, Nq, head_dim, n_dims,
        freq_scale, ext_factor, attn_factor, corr_low, corr_high, theta_scale, q_scale);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int qkv_post(const float * qkv_cat, float * q_out, void * q16_out,
             void * k_out_f16, void * v_out_f16,
             const int32_t * pos, const float * freq_factors,
             int M, int Nk, int Nv, int Nq, int head_dim, int n_dims,
             float freq_scale, float ext_factor, float attn_factor,
             float corr_low, float corr_high, float theta_scale, float q_scale,
             cudaStream_t stream) {
    return qkv_post_full(qkv_cat, q_out, q16_out, k_out_f16, v_out_f16, nullptr, nullptr,
                         pos, freq_factors, M, Nk, Nv, Nq, head_dim, n_dims,
                         freq_scale, ext_factor, attn_factor, corr_low, corr_high,
                         theta_scale, q_scale, stream);
}

} // namespace ggml_cuda_flashrt

// SPDX-License-Identifier: Apache-2.0
// ============================================================================
//  FlashRT — MolmoAct2 inference helper kernels (bf16, SM80+).
//
//  These are the small model-specific glue kernels the MolmoAct2 RTX pipeline
//  needs on top of the shared FlashRT GEMM / norm / FA2 / elementwise surface:
//
//    * molmo_mul_bf16_out           elementwise a*b (projector GELU gate)
//    * molmo_qk_norm_rope_bf16      per-head RMSNorm + rotate-half RoPE
//                                   (text HD=128 q/k, action HD=96 q/k)
//    * molmo_pool_mean_bf16         masked mean of the pooling window
//    * molmo_pool_attn_bf16         1-token x N-window masked adapter attention
//    * molmo_mod_rms_norm_bf16      broadcast adaLN: norm(x)*(1+s)+shift
//    * molmo_state_discretize_f32   q01/q99 state -> discrete token bins
//    * molmo_embedding_gather_bf16  token-id gather from the tied embedding table
//    * molmo_mask_dims_bf16         zero padded action dims
//    * molmo_resize_patchify_bf16   uint8 HWC -> normalized 14x14 patch rows
//
//  All entry points are stream-ordered and CUDA-Graph capturable.
// ============================================================================
#include "molmoact2_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <math.h>

namespace flash_rt {
namespace molmoact2 {

namespace {

__device__ __forceinline__ float b2f(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ __nv_bfloat16 f2b(float v) {
    return __float2bfloat16(v);
}

// ── elementwise multiply ──────────────────────────────────────────────────
__global__ void mul_bf16_kernel(const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                __nv_bfloat16* __restrict__ out, int n2,
                                __nv_fp8_e4m3* __restrict__ of8,
                                const float* __restrict__ fscale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(out);
    __nv_bfloat162 av = a2[idx], bv = b2[idx];
    o2[idx] = __nv_bfloat162(f2b(b2f(av.x) * b2f(bv.x)),
                             f2b(b2f(av.y) * b2f(bv.y)));
    if (of8 != nullptr) {
        const float inv = 1.0f / (*fscale);
        of8[2 * idx] = __nv_fp8_e4m3(fminf(fmaxf(
            b2f(o2[idx].x) * inv, -448.0f), 448.0f));
        of8[2 * idx + 1] = __nv_fp8_e4m3(fminf(fmaxf(
            b2f(o2[idx].y) * inv, -448.0f), 448.0f));
    }
}

// ── SiLU gate multiply (out = silu(g) * u) ────────────────────────────────
__global__ void silu_mul_bf16_kernel(const __nv_bfloat16* __restrict__ g,
                                     const __nv_bfloat16* __restrict__ u,
                                     __nv_bfloat16* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float gv = b2f(g[idx]);
    // torch computes silu on bf16 as float math rounded back to bf16, then the
    // multiply is a second bf16-rounded op.
    const float silu = b2f(f2b(gv / (1.0f + expf(-gv))));
    out[idx] = f2b(silu * b2f(u[idx]));
}

// ── fp32 accumulator + bf16 bias -> bf16 (matches a fused cuBLASLt epilogue) ─
__global__ void bias_round_bf16_kernel(const float* __restrict__ acc,
                                       const __nv_bfloat16* __restrict__ bias,
                                       __nv_bfloat16* __restrict__ out,
                                       int dim) {
    const int row = blockIdx.x;
    const float* ar = acc + (long)row * dim;
    __nv_bfloat16* orow = out + (long)row * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        orow[d] = f2b(ar[d] + b2f(bias[d]));
}

// bias_round_bf16 followed by an in-place bf16 residual add (res_add), in
// one pass: out = bf16(bf16(acc + bias) + add). Four elements per thread.
__global__ void bias_round_add_bf16_kernel(const float* __restrict__ acc,
                                           const __nv_bfloat16* __restrict__ bias,
                                           const __nv_bfloat16* __restrict__ add,
                                           __nv_bfloat16* __restrict__ out,
                                           int dim, long n4) {
    const long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n4) return;
    const long e = g * 4;
    const int d = (int)(e % dim);
    const float4 a = *reinterpret_cast<const float4*>(acc + e);
    const uint2 bb = *reinterpret_cast<const uint2*>(bias + d);
    const uint2 pp = *reinterpret_cast<const uint2*>(add + e);
    const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&bb);
    const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&pp);
    const float av[4] = {a.x, a.y, a.z, a.w};
    __nv_bfloat16 o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = f2b(b2f(f2b(av[i] + b2f(bv[i]))) + b2f(pv[i]));
    *reinterpret_cast<uint2*>(out + e) = *reinterpret_cast<const uint2*>(o);
}

// ── bf16 <-> fp16 casts (attention P-precision experiments) ──────────────
__global__ void cast_bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ src,
                                         __half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(b2f(src[idx]));
}
__global__ void cast_fp16_to_bf16_kernel(const __half* __restrict__ src,
                                         __nv_bfloat16* __restrict__ dst,
                                         int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = f2b(__half2float(src[idx]));
}

// ── in-place broadcast bias add ───────────────────────────────────────────
__global__ void add_bias_bf16_kernel(__nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ bias,
                                     int dim) {
    const int row = blockIdx.x;
    __nv_bfloat162* x2 = reinterpret_cast<__nv_bfloat162*>(x + (long)row * dim);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(bias);
    const int dim2 = dim >> 1;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i], bv = b2[i];
        x2[i] = __nv_bfloat162(f2b(b2f(xv.x) + b2f(bv.x)),
                               f2b(b2f(xv.y) + b2f(bv.y)));
    }
}

// ── per-head RMSNorm + rotate-half RoPE ───────────────────────────────────
// x: [S, NH*HD] bf16 row-major, modified in place.
// w: [HD] bf16 or nullptr (elementwise_affine=False).
// cos/sin: [S, HD] bf16 with duplicated halves (HF rotary_emb layout).
__global__ void qk_norm_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    int S, int NH, int HD, float eps, int has_weight) {
    const int half = HD >> 1;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= S * NH) return;
    const int n = warp_id % NH;
    const int s = warp_id / NH;
    const int base = s * NH * HD + n * HD;

    float ssq = 0.f;
    for (int d = lane; d < half; d += 32) {
        const float a = b2f(x[base + d]);
        const float b = b2f(x[base + d + half]);
        ssq += a * a + b * b;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
    const float inv = rsqrtf(ssq / (float)HD + eps);

    for (int d = lane; d < half; d += 32) {
        // HF chain: norm -> bf16, weight multiply -> bf16, each RoPE product
        // rounded to bf16, then the bf16 sum/difference.
        float nlo = b2f(f2b(b2f(x[base + d]) * inv));
        float nhi = b2f(f2b(b2f(x[base + d + half]) * inv));
        if (has_weight) {
            nlo = b2f(f2b(nlo * b2f(w[d])));
            nhi = b2f(f2b(nhi * b2f(w[d + half])));
        }
        const float c = b2f(cos_t[s * HD + d]);
        const float si = b2f(sin_t[s * HD + d]);
        const float lo = b2f(f2b(nlo * c));
        const float lo2 = b2f(f2b(nhi * si));
        const float hi = b2f(f2b(nhi * c));
        const float hi2 = b2f(f2b(nlo * si));
        x[base + d] = f2b(lo - lo2);
        x[base + d + half] = f2b(hi + hi2);
    }
}

// Fused QKV split (+bias) with per-head RMSNorm + rotate-half RoPE on Q/K.
// Matches qkv_split[_bias] followed by qk_norm_rope_bf16 bit-for-bit.
__global__ void qkv_split_bias_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ V, int qd, int kd, int vd, int nq, int nk,
    int hd, float eps, int has_bias, int has_rope,
    __nv_fp8_e4m3* __restrict__ Kf8, __nv_fp8_e4m3* __restrict__ Vf8,
    const float* __restrict__ kscale, const float* __restrict__ vscale,
    __nv_fp8_e4m3* __restrict__ Qf8, const float* __restrict__ qscale) {
    const int row = blockIdx.x;
    const int qkv_dim = qd + kd + vd;
    const __nv_bfloat16* src = qkv + (long)row * qkv_dim;
    const int tid = threadIdx.x;
    const int nwarps = blockDim.x >> 5;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int half = hd >> 1;
    // V: plain copy (+bias), optional static-FP8 copy for a context GEMM.
    // 8-wide vector form: identical per-element arithmetic (no reduction), so
    // the output bytes are unchanged, but the global accesses are 16 B/lane
    // instead of 2 B and the warp transactions are fully coalesced.
    for (int i8 = tid * 8; i8 < vd; i8 += blockDim.x * 8) {
        uint4 vs = *reinterpret_cast<const uint4*>(src + qd + kd + i8);
        const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(&vs);
        uint4 bs;
        const __nv_bfloat16* bb = nullptr;
        if (has_bias) {
            bs = *reinterpret_cast<const uint4*>(bias + qd + kd + i8);
            bb = reinterpret_cast<const __nv_bfloat16*>(&bs);
        }
        __nv_bfloat16 outv[8];
        __nv_fp8_e4m3 outf[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = b2f(vv[j]);
            if (has_bias) v = b2f(f2b(v + b2f(bb[j])));
            const float vb = b2f(f2b(v));
            outv[j] = f2b(vb);
            if (Vf8 != nullptr)
                outf[j] = __nv_fp8_e4m3(fminf(fmaxf(
                    vb * (1.0f / (*vscale)), -448.0f), 448.0f));
        }
        if (V != nullptr)
            *reinterpret_cast<uint4*>(V + (long)row * vd + i8) =
                *reinterpret_cast<uint4*>(outv);
        if (Vf8 != nullptr)
            *reinterpret_cast<uint2*>(Vf8 + (long)row * vd + i8) =
                *reinterpret_cast<uint2*>(outf);
    }
    // Q then K heads, one warp per head
    for (int h = warp; h < nq + nk; h += nwarps) {
        const int is_q = h < nq;
        const int head = is_q ? h : h - nq;
        const int off = is_q ? head * hd : qd + head * hd;
        __nv_bfloat16* dst = is_q ? Q + (long)row * qd + head * hd
                                  : K + (long)row * kd + head * hd;
        const __nv_bfloat16* w = is_q ? qw : kw;
        // A null Q / K skips the bf16 copy (FP8-only consumers).
        const bool wdst = (is_q ? Q : K) != nullptr;
        float ssq = 0.f;
        for (int d = lane; d < half; d += 32) {
            float a = b2f(src[off + d]);
            float b = b2f(src[off + d + half]);
            if (has_bias) {
                a = b2f(f2b(a + b2f(bias[off + d])));
                b = b2f(f2b(b + b2f(bias[off + d + half])));
            }
            ssq += a * a + b * b;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
        const float inv = rsqrtf(ssq / (float)hd + eps);
        for (int d = lane; d < half; d += 32) {
            float a = b2f(src[off + d]);
            float b = b2f(src[off + d + half]);
            if (has_bias) {
                a = b2f(f2b(a + b2f(bias[off + d])));
                b = b2f(f2b(b + b2f(bias[off + d + half])));
            }
            float nlo = b2f(f2b(a * inv));
            float nhi = b2f(f2b(b * inv));
            if (w != nullptr) {
                nlo = b2f(f2b(nlo * b2f(w[d])));
                nhi = b2f(f2b(nhi * b2f(w[d + half])));
            }
            if (has_rope) {
                const float c = b2f(cos_t[(long)row * hd + d]);
                const float si = b2f(sin_t[(long)row * hd + d]);
                const float lo = b2f(f2b(nlo * c));
                const float lo2 = b2f(f2b(nhi * si));
                const float hi = b2f(f2b(nhi * c));
                const float hi2 = b2f(f2b(nlo * si));
                const float o0 = b2f(f2b(lo - lo2));
                const float o1 = b2f(f2b(hi + hi2));
                if (wdst) {
                    dst[d] = f2b(o0);
                    dst[d + half] = f2b(o1);
                }
                if (!is_q && Kf8 != nullptr) {
                    Kf8[(long)row * kd + head * hd + d] = __nv_fp8_e4m3(
                        fminf(fmaxf(o0 * (1.0f / (*kscale)), -448.0f),
                              448.0f));
                    Kf8[(long)row * kd + head * hd + d + half] =
                        __nv_fp8_e4m3(fminf(fmaxf(
                            o1 * (1.0f / (*kscale)), -448.0f), 448.0f));
                }
                if (is_q && Qf8 != nullptr) {
                    Qf8[(long)row * qd + head * hd + d] = __nv_fp8_e4m3(
                        fminf(fmaxf(o0 * (1.0f / (*qscale)), -448.0f),
                              448.0f));
                    Qf8[(long)row * qd + head * hd + d + half] =
                        __nv_fp8_e4m3(fminf(fmaxf(
                            o1 * (1.0f / (*qscale)), -448.0f), 448.0f));
                }
            } else {
                const float o0 = b2f(f2b(nlo));
                const float o1 = b2f(f2b(nhi));
                if (wdst) {
                    dst[d] = f2b(o0);
                    dst[d + half] = f2b(o1);
                }
                if (!is_q && Kf8 != nullptr) {
                    Kf8[(long)row * kd + head * hd + d] = __nv_fp8_e4m3(
                        fminf(fmaxf(o0 * (1.0f / (*kscale)), -448.0f),
                              448.0f));
                    Kf8[(long)row * kd + head * hd + d + half] =
                        __nv_fp8_e4m3(fminf(fmaxf(
                            o1 * (1.0f / (*kscale)), -448.0f), 448.0f));
                }
                if (is_q && Qf8 != nullptr) {
                    Qf8[(long)row * qd + head * hd + d] = __nv_fp8_e4m3(
                        fminf(fmaxf(o0 * (1.0f / (*qscale)), -448.0f),
                              448.0f));
                    Qf8[(long)row * qd + head * hd + d + half] =
                        __nv_fp8_e4m3(fminf(fmaxf(
                            o1 * (1.0f / (*qscale)), -448.0f), 448.0f));
                }
            }
        }
    }
}

// Repo-exact block reduction (same tree as kernels/common.cuh) so the fused
// producers below round identically to layer_norm + quantize_fp8_static.
__device__ __forceinline__ float molmo_block_sum(float val, float* shared) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_down_sync(0xffffffffu, val, o);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    const int nwarps = blockDim.x >> 5;
    val = (threadIdx.x < nwarps) ? shared[threadIdx.x] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            val += __shfl_down_sync(0xffffffffu, val, o);
    }
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

// LayerNorm (affine) emitted directly as static-scale FP8 E4M3.
__global__ void layer_norm_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ b, __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ scale, int dim, float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (long)row * dim;
    __nv_fp8_e4m3* orow = out + (long)row * dim;
    extern __shared__ float shared[];
    const int dim2 = dim >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xr);
    float ls = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i];
        ls += b2f(v.x) + b2f(v.y);
    }
    const float mean = molmo_block_sum(ls, shared) / (float)dim;
    float lv = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i];
        const float d0 = b2f(v.x) - mean, d1 = b2f(v.y) - mean;
        lv += d0 * d0 + d1 * d1;
    }
    const float inv_std = rsqrtf(molmo_block_sum(lv, shared) / (float)dim + eps);
    const float inv_s = 1.0f / (*scale);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i], wv = w2[i], bv = b2[i];
        const float v0 = (b2f(v.x) - mean) * inv_std * b2f(wv.x) + b2f(bv.x);
        const float v1 = (b2f(v.y) - mean) * inv_std * b2f(wv.y) + b2f(bv.y);
        orow[2 * i] = __nv_fp8_e4m3(fminf(fmaxf(
            b2f(f2b(v0)) * inv_s, -448.0f), 448.0f));
        orow[2 * i + 1] = __nv_fp8_e4m3(fminf(fmaxf(
            b2f(f2b(v1)) * inv_s, -448.0f), 448.0f));
    }
}

// bias + tanh-GELU emitted directly as static-scale FP8 E4M3.
__global__ void bias_gelu_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ b,
    __nv_fp8_e4m3* __restrict__ out, const float* __restrict__ scale,
    int n, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int c = idx % dim;
    float v = b2f(x[idx]);
    if (b != nullptr) v = b2f(f2b(v + b2f(b[c])));
    const float t = tanhf(0.7978845608f * (v + 0.044715f * v * v * v));
    const float g = b2f(f2b(v * 0.5f * (1.0f + t)));
    out[idx] = __nv_fp8_e4m3(fminf(fmaxf(g * (1.0f / (*scale)),
                                          -448.0f), 448.0f));
}

// 8-wide vectorized form of bias_gelu_quant_fp8_kernel (dim % 8 == 0).
// Identical per-element arithmetic, 16-byte loads and 8-byte stores.
__global__ void bias_gelu_quant_fp8_vec_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ b,
    __nv_fp8_e4m3* __restrict__ out, const float* __restrict__ scale,
    int n8, int dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n8) return;
    const int idx = j << 3;
    const int c = idx % dim;
    const uint4 xv = *reinterpret_cast<const uint4*>(x + idx);
    const __nv_bfloat16* px = reinterpret_cast<const __nv_bfloat16*>(&xv);
    __nv_bfloat16 bx[8];
    if (b != nullptr) {
        const uint4 bv = *reinterpret_cast<const uint4*>(b + c);
        const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&bv);
#pragma unroll
        for (int k = 0; k < 8; ++k) bx[k] = pb[k];
    }
    const float inv = 1.0f / (*scale);
    uint2 o;
    uint8_t* ob = reinterpret_cast<uint8_t*>(&o);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = b2f(px[k]);
        if (b != nullptr) v = b2f(f2b(v + b2f(bx[k])));
        const float t = tanhf(0.7978845608f * (v + 0.044715f * v * v * v));
        const float g = b2f(f2b(v * 0.5f * (1.0f + t)));
        ob[k] = __nv_fp8_e4m3(fminf(fmaxf(g * inv, -448.0f), 448.0f)).__x;
    }
    *reinterpret_cast<uint2*>(out + idx) = o;
}

// pairwise concat with an optional static-scale FP8 copy (adapter input).
__global__ void concat2_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out, __nv_fp8_e4m3* __restrict__ of8,
    const float* __restrict__ fscale, int rows, int cols_a, int cols_b) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int cols = cols_a + cols_b;
    const float inv = of8 != nullptr ? 1.0f / (*fscale) : 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const __nv_bfloat16 vb = (c < cols_a)
            ? a[(long)r * cols_a + c] : b[(long)r * cols_b + (c - cols_a)];
        out[(long)r * cols + c] = vb;
        if (of8 != nullptr)
            of8[(long)r * cols + c] = __nv_fp8_e4m3(fminf(fmaxf(
                b2f(vb) * inv, -448.0f), 448.0f));
    }
}

// 8-wide vectorized QKV split (repository qkv_split semantics, pure copy).
__global__ void qkv_split_vec_kernel(const __nv_bfloat16* __restrict__ qkv,
                                     __nv_bfloat16* __restrict__ Q,
                                     __nv_bfloat16* __restrict__ K,
                                     __nv_bfloat16* __restrict__ V,
                                     int q_dim, int k_dim, int v_dim) {
    const int row = blockIdx.x;
    const int qkv_dim = q_dim + k_dim + v_dim;
    const __nv_bfloat16* src = qkv + (long)row * qkv_dim;
    const int n8 = qkv_dim >> 3;
    for (int j = threadIdx.x; j < n8; j += blockDim.x) {
        const int c = j << 3;
        const uint4 v = *reinterpret_cast<const uint4*>(src + c);
        if (c < q_dim) {
            *reinterpret_cast<uint4*>(Q + (long)row * q_dim + c) = v;
        } else if (c < q_dim + k_dim) {
            *reinterpret_cast<uint4*>(K + (long)row * k_dim + (c - q_dim)) = v;
        } else {
            *reinterpret_cast<uint4*>(V + (long)row * v_dim +
                                      (c - q_dim - k_dim)) = v;
        }
    }
}

// ── masked pooling-window mean ────────────────────────────────────────────
__global__ void pool_mean_bf16_kernel(
    const __nv_bfloat16* __restrict__ feats,  // [rows, D]
    const int* __restrict__ idx,              // [P, I], -1 = invalid
    __nv_bfloat16* __restrict__ out,          // [P, D]
    __nv_fp8_e4m3* __restrict__ of8, const float* __restrict__ fscale,
    int P, int I, int D) {
    const int p = blockIdx.x;
    int count = 0;
    for (int i = 0; i < I; ++i) count += (idx[p * I + i] >= 0);
    const float inv = count > 0 ? 1.0f / (float)count : 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < I; ++i) {
            const int r = idx[p * I + i];
            if (r >= 0) acc += b2f(feats[(long)r * D + d]);
        }
        const float vb = b2f(f2b(acc * inv));
        out[(long)p * D + d] = f2b(vb);
        if (of8 != nullptr)
            of8[(long)p * D + d] = __nv_fp8_e4m3(fminf(fmaxf(
                vb * (1.0f / (*fscale)), -448.0f), 448.0f));
    }
}

// ── adapter attention: one query token over an I-wide masked window ───────
// q: [P, NH, HD], k/v: [rows, NH, HD], idx: [P, I] (-1 = masked).
__global__ void pool_attn_bf16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ idx,
    __nv_bfloat16* __restrict__ out,
    __nv_fp8_e4m3* __restrict__ of8, const float* __restrict__ fscale,
    int P, int I, int NH, int HD, float scale) {
    const int p = blockIdx.x;
    const int h = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (h >= NH) return;

    const __nv_bfloat16* qrow = q + ((long)p * NH + h) * HD;
    __nv_bfloat16* orow = out + ((long)p * NH + h) * HD;

    float score[8];
    float mx = -INFINITY;
    int valid = 0;
    for (int i = 0; i < I; ++i) {
        const int r = idx[p * I + i];
        float acc = 0.f;
        if (r >= 0) {
            const __nv_bfloat16* krow = k + ((long)r * NH + h) * HD;
            for (int d = lane; d < HD; d += 32)
                acc += b2f(qrow[d]) * b2f(krow[d]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                acc += __shfl_xor_sync(0xffffffffu, acc, o);
            acc *= scale;
            mx = fmaxf(mx, acc);
            ++valid;
        }
        score[i] = acc;
    }
    if (valid == 0) {
        for (int d = lane; d < HD; d += 32) {
            orow[d] = f2b(0.f);
            if (of8 != nullptr)
                ((__nv_fp8_e4m3*)of8)[((long)p * NH + h) * HD + d] =
                    __nv_fp8_e4m3(0.f);
        }
        return;
    }
    float sum = 0.f;
    for (int i = 0; i < I; ++i) {
        if (idx[p * I + i] < 0) { score[i] = 0.f; continue; }
        score[i] = __expf(score[i] - mx);
        sum += score[i];
    }
    const float inv = 1.0f / sum;
    for (int d = lane; d < HD; d += 32) {
        float acc = 0.f;
        for (int i = 0; i < I; ++i) {
            const int r = idx[p * I + i];
            if (r < 0) continue;
            acc += score[i] * b2f(v[((long)r * NH + h) * HD + d]);
        }
        const float vb = b2f(f2b(acc * inv));
        orow[d] = f2b(vb);
        if (of8 != nullptr)
            of8[((long)p * NH + h) * HD + d] = __nv_fp8_e4m3(fminf(fmaxf(
                vb * (1.0f / (*fscale)), -448.0f), 448.0f));
    }
}

// ── broadcast adaLN: out = rmsnorm(x,w) * (1+scale) + shift ───────────────
__global__ void mod_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out, int dim, float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (long)row * dim;
    __nv_bfloat16* orow = out + (long)row * dim;

    extern __shared__ float shared[];
    const int dim2 = dim >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xr);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i];
        local_sum += b2f(v.x) * b2f(v.x) + b2f(v.y) * b2f(v.y);
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
    __shared__ float warp_sums[32];
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) shared[0] = rsqrtf(v / (float)dim + eps);
    }
    __syncthreads();
    const float rms = shared[0];

    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(orow);
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i];
        float vx = b2f(xv.x) * rms;
        float vy = b2f(xv.y) * rms;
        if (w != nullptr) {
            const __nv_bfloat162 wv = w2[i];
            vx *= b2f(wv.x);
            vy *= b2f(wv.y);
        }
        const __nv_bfloat162 sv = s2[i];
        const __nv_bfloat162 hv = h2[i];
        // Mirror torch: norm rounded to bf16, then bf16 (1+scale), multiply,
        // then bf16 add of shift.
        const float nx = b2f(f2b(vx));
        const float ny = b2f(f2b(vy));
        const float ax = b2f(f2b(1.0f + b2f(sv.x)));
        const float ay = b2f(f2b(1.0f + b2f(sv.y)));
        const float mx = b2f(f2b(nx * ax));
        const float my = b2f(f2b(ny * ay));
        o2[i] = __nv_bfloat162(f2b(mx + b2f(hv.x)), f2b(my + b2f(hv.y)));
    }
}

__global__ void rms_norm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     __nv_bfloat16* __restrict__ out,
                                     int dim, float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (long)row * dim;
    __nv_bfloat16* orow = out + (long)row * dim;
    extern __shared__ float shared[];
    __shared__ float warp_sums[32];
    const int dim2 = dim >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xr);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i];
        local_sum += b2f(v.x) * b2f(v.x) + b2f(v.y) * b2f(v.y);
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) shared[0] = rsqrtf(v / (float)dim + eps);
    }
    __syncthreads();
    const float rms = shared[0];
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(orow);
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i];
        __nv_bfloat162 r;
        r.x = f2b(b2f(xv.x) * rms);
        r.y = f2b(b2f(xv.y) * rms);
        if (w != nullptr) {
            const __nv_bfloat162 wv = w2[i];
            r.x = f2b(b2f(r.x) * b2f(wv.x));
            r.y = f2b(b2f(r.y) * b2f(wv.y));
        }
        o2[i] = r;
    }
}

// bias add (bf16) fused into a per-head RMSNorm producer.
// x/out: [rows, hd] bf16; bias: [heads*hd] broadcast over `heads`.
__global__ void bias_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
    int heads, int hd, float eps) {
    const int row = blockIdx.x;
    const int h = row % heads;
    const __nv_bfloat16* xr = x + (long)row * hd;
    const __nv_bfloat16* br = bias + (long)h * hd;
    __nv_bfloat16* orow = out + (long)row * hd;
    extern __shared__ float shared[];
    __shared__ float warp_sums[32];
    const int dim2 = hd >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xr);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(br);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i], bv = b2[i];
        const float v0 = b2f(f2b(b2f(xv.x) + b2f(bv.x)));
        const float v1 = b2f(f2b(b2f(xv.y) + b2f(bv.y)));
        local_sum += v0 * v0 + v1 * v1;
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) shared[0] = rsqrtf(v / (float)hd + eps);
    }
    __syncthreads();
    const float rms = shared[0];
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(orow);
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i], bv = b2[i];
        __nv_bfloat162 r;
        r.x = f2b(b2f(f2b(b2f(xv.x) + b2f(bv.x))) * rms);
        r.y = f2b(b2f(f2b(b2f(xv.y) + b2f(bv.y))) * rms);
        if (w != nullptr) {
            const __nv_bfloat162 wv = w2[i];
            r.x = f2b(b2f(r.x) * b2f(wv.x));
            r.y = f2b(b2f(r.y) * b2f(wv.y));
        }
        o2[i] = r;
    }
}

// ── fused producers (bias / static-FP8 quantization) ─────────────────────
// qkv split with a per-column bias add (the action expert's fused QKV).
__global__ void qkv_split_bias_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ V, int q_dim, int k_dim, int v_dim) {
    // 8-wide vectorized: every 16-byte chunk lies inside one of Q/K/V
    // because the three widths are multiples of 8.
    const int row = blockIdx.x;
    const int qkv_dim = q_dim + k_dim + v_dim;
    const __nv_bfloat16* src = qkv + (long)row * qkv_dim;
    const int n8 = qkv_dim >> 3;
    for (int j = threadIdx.x; j < n8; j += blockDim.x) {
        const int c = j << 3;
        const uint4 xv = *reinterpret_cast<const uint4*>(src + c);
        const uint4 bv = *reinterpret_cast<const uint4*>(bias + c);
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bv);
        uint4 ov;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            o2[i] = __nv_bfloat162(
                f2b(b2f(x2[i].x) + b2f(b2[i].x)),
                f2b(b2f(x2[i].y) + b2f(b2[i].y)));
        }
        if (c < q_dim) {
            *reinterpret_cast<uint4*>(Q + (long)row * q_dim + c) = ov;
        } else if (c < q_dim + k_dim) {
            *reinterpret_cast<uint4*>(K + (long)row * k_dim + (c - q_dim)) = ov;
        } else {
            *reinterpret_cast<uint4*>(V + (long)row * v_dim +
                                      (c - q_dim - k_dim)) = ov;
        }
    }
}

// out = silu(g + bias_g) * (u + bias_u), two bf16 roundings like torch.
__global__ void silu_mul_bias_kernel(
    const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ bg,
    const __nv_bfloat16* __restrict__ bu, __nv_bfloat16* __restrict__ out,
    int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int c = idx % dim;
    float gv = b2f(g[idx]);
    float uv = b2f(u[idx]);
    if (bg != nullptr) gv = b2f(f2b(gv + b2f(bg[c])));
    if (bu != nullptr) uv = b2f(f2b(uv + b2f(bu[c])));
    const float silu = b2f(f2b(gv / (1.0f + expf(-gv))));
    out[idx] = f2b(silu * uv);
}

// Warp-per-row RMS norm for the context projections: 8-wide loads, one warp
// reduction per row, no block-wide barrier. dim must be a multiple of 8 and
// at most 2048; weight may be null.
__global__ void rms_norm_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out, int rows, int dim, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int row = warp;
    if (row >= rows) return;
    const int n8 = dim >> 3;                 // 8-element (uint4) chunks
    float sum = 0.f;
    for (int j = lane; j < n8; j += 32) {
        const uint4 v = *reinterpret_cast<const uint4*>(
            x + (long)row * dim + (j << 3));
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a = b2f(p2[k].x), b = b2f(p2[k].y);
            sum += a * a + b * b;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
    sum = __shfl_sync(0xffffffffu, sum, 0);
    const float inv = rsqrtf(sum / (float)dim + eps);
    for (int j = lane; j < n8; j += 32) {
        const int c = j << 3;
        const uint4 v = *reinterpret_cast<const uint4*>(
            x + (long)row * dim + c);
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&v);
        uint4 o4;
        __nv_bfloat162* q2 = reinterpret_cast<__nv_bfloat162*>(&o4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float a = b2f(p2[k].x) * inv;
            float b = b2f(p2[k].y) * inv;
            if (w != nullptr) {
                a *= b2f(w[c + 2 * k]);
                b *= b2f(w[c + 2 * k + 1]);
            }
            q2[k] = __floats2bfloat162_rn(a, b);
        }
        *reinterpret_cast<uint4*>(out + (long)row * dim + c) = o4;
    }
}

// res = bf16(res + bf16(bf16(x + bias) * gate)); gate broadcasts per row.
__global__ void gate_residual_bias_kernel(
    __nv_bfloat16* __restrict__ res, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ bias, int rows, int dim) {
    // 8-wide vectorized (dim is a multiple of 8 in every production site).
    const int n8 = (rows * dim) >> 3;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n8) return;
    const int idx = j << 3;
    const int c = idx % dim;
    const uint4 xv = *reinterpret_cast<const uint4*>(x + idx);
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + c);
    const uint4 bv = *reinterpret_cast<const uint4*>(bias + c);
    const uint4 rv = *reinterpret_cast<const uint4*>(res + idx);
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bv);
    const __nv_bfloat162* r2 = reinterpret_cast<const __nv_bfloat162*>(&rv);
    uint4 ov;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float a0 = b2f(f2b(b2f(x2[i].x) + b2f(b2[i].x)));
        const float a1 = b2f(f2b(b2f(x2[i].y) + b2f(b2[i].y)));
        o2[i] = __nv_bfloat162(
            f2b(b2f(r2[i].x) + b2f(f2b(a0 * b2f(g2[i].x)))),
            f2b(b2f(r2[i].y) + b2f(f2b(a1 * b2f(g2[i].y)))));
    }
    *reinterpret_cast<uint4*>(res + idx) = ov;
}

// HF-exact modulated RMSNorm emitted directly as static-scale FP8 E4M3.
__global__ void mod_rms_norm_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_fp8_e4m3* __restrict__ out, const float* __restrict__ qscale,
    int dim, float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (long)row * dim;
    __nv_fp8_e4m3* orow = out + (long)row * dim;
    extern __shared__ float shared[];
    __shared__ float warp_sums[32];
    const int dim2 = dim >> 1;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(xr);
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i];
        local_sum += b2f(v.x) * b2f(v.x) + b2f(v.y) * b2f(v.y);
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) shared[0] = rsqrtf(v / (float)dim + eps);
    }
    __syncthreads();
    const float rms = shared[0];
    const float inv_s = 1.0f / (*qscale);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i];
        float vx = b2f(xv.x) * rms;
        float vy = b2f(xv.y) * rms;
        if (w != nullptr) {
            const __nv_bfloat162 wv = w2[i];
            vx = b2f(f2b(vx * b2f(wv.x)));
            vy = b2f(f2b(vy * b2f(wv.y)));
        }
        const __nv_bfloat162 sv = s2[i];
        const __nv_bfloat162 hv = h2[i];
        const float nx = b2f(f2b(vx));
        const float ny = b2f(f2b(vy));
        const float ax = b2f(f2b(1.0f + b2f(sv.x)));
        const float ay = b2f(f2b(1.0f + b2f(sv.y)));
        const float mx = b2f(f2b(nx * ax)) + b2f(hv.x);
        const float my = b2f(f2b(ny * ay)) + b2f(hv.y);
        orow[2 * i] = __nv_fp8_e4m3(fminf(fmaxf(
            b2f(f2b(mx)) * inv_s, -448.0f), 448.0f));
        orow[2 * i + 1] = __nv_fp8_e4m3(fminf(fmaxf(
            b2f(f2b(my)) * inv_s, -448.0f), 448.0f));
    }
}

// silu(g)*u with the two bf16 roundings, emitted as static-scale FP8.
__global__ void silu_mul_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
    const __nv_bfloat16* __restrict__ bg,
    const __nv_bfloat16* __restrict__ bu,
    __nv_fp8_e4m3* __restrict__ out, const float* __restrict__ qscale,
    int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int c = dim > 0 ? idx % dim : 0;
    float gv = b2f(g[idx]);
    float uv = b2f(u[idx]);
    if (bg != nullptr) gv = b2f(f2b(gv + b2f(bg[c])));
    if (bu != nullptr) uv = b2f(f2b(uv + b2f(bu[c])));
    const float silu = b2f(f2b(gv / (1.0f + expf(-gv))));
    const float v = b2f(f2b(silu * uv));
    out[idx] = __nv_fp8_e4m3(fminf(fmaxf(v / (*qscale), -448.0f), 448.0f));
}

// ── consumers for the skinny NVFP4 action-expert GEMMs ───────────────────
// Each sums the K-split FP32 partials, applies alpha = a_gs * w_gs, rounds
// the GEMM result to BF16 and then applies the block's bias/post-op with the
// same rounding sequence as the unfused BF16 chain.
static __device__ __forceinline__ float skinny_sum(const float* p, int splits,
                                                   long idx, int n,
                                                   float alpha) {
    float s = 0.f;
    for (int t = 0; t < splits; ++t) s += p[(long)t * n + idx];
    return __bfloat162float(__float2bfloat16(s * alpha));
}

__global__ void skinny_qkv_split_bias_kernel(
    const float* __restrict__ partials, int splits,
    const float* __restrict__ ags, const float* __restrict__ wgs,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int n,
    int q_dim, int k_dim, int v_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int row = idx / (q_dim + k_dim + v_dim);
    const int c = idx - row * (q_dim + k_dim + v_dim);
    const float alpha = (*ags) * (*wgs);
    const float v = __bfloat162float(__float2bfloat16(
        __bfloat162float(b2f(__float2bfloat16(
            skinny_sum(partials, splits, idx, n, alpha)))) + b2f(bias[c])));
    if (c < q_dim) {
        Q[(long)row * q_dim + c] = __float2bfloat16(v);
    } else if (c < q_dim + k_dim) {
        K[(long)row * k_dim + (c - q_dim)] = __float2bfloat16(v);
    } else {
        V[(long)row * v_dim + (c - q_dim - k_dim)] = __float2bfloat16(v);
    }
}

__global__ void skinny_gate_residual_kernel(
    const float* __restrict__ partials, int splits,
    const float* __restrict__ ags, const float* __restrict__ wgs,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ res,
    const __nv_bfloat16* __restrict__ gate, int n, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int c = idx % dim;
    const float alpha = (*ags) * (*wgs);
    const float a = __bfloat162float(__float2bfloat16(
        skinny_sum(partials, splits, idx, n, alpha) + b2f(bias[c])));
    const float p = __bfloat162float(
        __float2bfloat16(a * b2f(gate[c])));
    res[idx] = __float2bfloat16(b2f(res[idx]) + p);
}

__global__ void skinny_bias_rms_norm_kernel(
    const float* __restrict__ partials, int splits,
    const float* __restrict__ ags, const float* __restrict__ wgs,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int heads, int hd, int n, float eps) {
    const int row = blockIdx.x;
    const int h = row % heads;
    const long base = (long)row * hd;
    const __nv_bfloat16* b = bias + (long)h * hd;
    const float alpha = (*ags) * (*wgs);
    extern __shared__ float shared[];
    __shared__ float warp_sums[32];
    const int dim2 = hd >> 1;
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const int idx = (int)(base + 2 * i);
        const float v0 = __bfloat162float(__float2bfloat16(
            skinny_sum(partials, splits, idx, n, alpha) + b2f(b[2 * i])));
        const float v1 = __bfloat162float(__float2bfloat16(
            skinny_sum(partials, splits, idx + 1, n, alpha) +
            b2f(b[2 * i + 1])));
        local_sum += v0 * v0 + v1 * v1;
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) shared[0] = rsqrtf(v / (float)hd + eps);
    }
    __syncthreads();
    const float rms = shared[0];
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(out + base);
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const int idx = (int)(base + 2 * i);
        const float a0 = __bfloat162float(__float2bfloat16(
            skinny_sum(partials, splits, idx, n, alpha) + b2f(b[2 * i])));
        const float a1 = __bfloat162float(__float2bfloat16(
            skinny_sum(partials, splits, idx + 1, n, alpha) +
            b2f(b[2 * i + 1])));
        __nv_bfloat162 r;
        r.x = __float2bfloat16(a0 * rms);
        r.y = __float2bfloat16(a1 * rms);
        o2[i] = r;
    }
}

__global__ void skinny_silu_mul_kernel(
    const float* __restrict__ pg, int splits_g, const float* __restrict__ ags_g,
    const float* __restrict__ wgs_g, const float* __restrict__ pu,
    int splits_u, const float* __restrict__ ags_u,
    const float* __restrict__ wgs_u, const __nv_bfloat16* __restrict__ bg,
    const __nv_bfloat16* __restrict__ bu, __nv_bfloat16* __restrict__ out,
    int n, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int c = idx % dim;
    float gv = skinny_sum(pg, splits_g, idx, n, (*ags_g) * (*wgs_g));
    float uv = skinny_sum(pu, splits_u, idx, n, (*ags_u) * (*wgs_u));
    if (bg != nullptr) gv = b2f(f2b(gv + b2f(bg[c])));
    if (bu != nullptr) uv = b2f(f2b(uv + b2f(bu[c])));
    const float silu = b2f(f2b(gv / (1.0f + expf(-gv))));
    out[idx] = f2b(silu * uv);
}

// ── state q01/q99 normalization -> discrete bins (matches HF numpy chain) ─
__global__ void state_discretize_f32_kernel(
    const float* __restrict__ state,
    const float* __restrict__ q01,
    const float* __restrict__ q99,
    const signed char* __restrict__ mask,       // optional, 1 = normalize
    const signed char* __restrict__ zero_mask,  // optional, 1 = force 0
    long long* __restrict__ ids, int D, int num_tokens, int base_id) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const float e = 1e-6f;
    const float lo = q01[d];
    const float hi = q99[d];
    const float denom = fmaxf(hi - lo, e);
    float v = 2.0f * (state[d] - lo) / denom - 1.0f;
    v = fminf(fmaxf(v, -1.0f), 1.0f);
    if (mask != nullptr && mask[d] == 0) v = state[d];
    if (zero_mask != nullptr && zero_mask[d] != 0) v = 0.0f;
    if (isnan(v) || isinf(v)) v = 0.0f;
    v = fminf(fmaxf(v, -1.0f), 1.0f);
    const float scaled = (v + 1.0f) * 0.5f * (float)(num_tokens - 1);
    int bin = (int)rintf(scaled);
    bin = max(0, min(num_tokens - 1, bin));
    ids[d] = (long long)(base_id + bin);
}

// ── action unnormalization (q01/q99, mask passthrough) ───────────────────
__global__ void action_denorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ traj, float* __restrict__ out,
    const float* __restrict__ q01, const float* __restrict__ q99,
    const signed char* __restrict__ mask, int stride, int act_dim) {
    const int r = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= act_dim) return;
    float v = b2f(traj[(long)r * stride + d]);
    v = fminf(fmaxf(v, -1.0f), 1.0f);
    float o = (v + 1.0f) * (q99[d] - q01[d]) * 0.5f + q01[d];
    if (mask != nullptr && mask[d] == 0) o = v;
    out[(long)r * act_dim + d] = o;
}

// ── zero padded action dims ───────────────────────────────────────────────
// gate is a per-sample [dim] vector broadcast over ``rows`` tokens.
__global__ void gate_residual_bf16_kernel(__nv_bfloat16* __restrict__ res,
                                          const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ gate,
                                          int rows, int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = rows * dim;
    if (idx >= n) return;
    const int col = idx % dim;
    const float g = b2f(x[idx]) * b2f(gate[col]);
    res[idx] = f2b(b2f(res[idx]) + b2f(f2b(g)));
}

__global__ void mask_dims_bf16_kernel(__nv_bfloat16* __restrict__ x,
                                      int rows, int dim, int valid) {
    const int r = blockIdx.x;
    for (int d = valid + threadIdx.x; d < dim; d += blockDim.x)
        x[(long)r * dim + d] = f2b(0.f);
}

__device__ __forceinline__ float bilinear_sample(
    const uint8_t* img, int H, int W, int C, int c, float sy, float sx) {
    sy = fminf(fmaxf(sy, 0.f), (float)(H - 1));
    sx = fminf(fmaxf(sx, 0.f), (float)(W - 1));
    const int y0 = (int)floorf(sy);
    const int x0 = (int)floorf(sx);
    const int y1 = min(y0 + 1, H - 1);
    const int x1 = min(x0 + 1, W - 1);
    const float wy = sy - (float)y0;
    const float wx = sx - (float)x0;
    const float p00 = (float)img[((long)y0 * W + x0) * C + c];
    const float p01 = (float)img[((long)y0 * W + x1) * C + c];
    const float p10 = (float)img[((long)y1 * W + x0) * C + c];
    const float p11 = (float)img[((long)y1 * W + x1) * C + c];
    return (p00 * (1.f - wx) + p01 * wx) * (1.f - wy) +
           (p10 * (1.f - wx) + p11 * wx) * wy;
}

// ── uint8 HWC resize (bilinear, align_corners=False) + normalize + patch ──
__global__ void resize_patchify_bf16_kernel(
    const uint8_t* __restrict__ img, __nv_bfloat16* __restrict__ out,
    int H, int W, int C, int OH, int OW, int patch, int n_patch_y,
    int n_patch_w, int feat_per_patch, int out_stride) {
    const long total = (long)gridDim.x * blockDim.x;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < (long)n_patch_y * n_patch_w * feat_per_patch * 1;
         idx += total) {
        // Decoded per image by the caller via grid offset (one launch/image).
        const int f = (int)(idx % feat_per_patch);
        const int pidx = (int)(idx / feat_per_patch);
        const int ph = pidx / n_patch_w;
        const int pw = pidx % n_patch_w;
        // Feature order matches batch_pixels_to_patches: (py, px, c).
        const int c = f % C;
        const int px = (f / C) % patch;
        const int py = f / (C * patch);
        const int oy = ph * patch + py;
        const int ox = pw * patch + px;
        const float sy = ((float)oy + 0.5f) * (float)H / (float)OH - 0.5f;
        const float sx = ((float)ox + 0.5f) * (float)W / (float)OW - 0.5f;
        const float v = bilinear_sample(img, H, W, C, c, sy, sx);
        // torchvision rounds the uint8 resize result to the nearest integer
        // (round-half-to-even) before the reference normalizes it.
        const float q = fminf(fmaxf(rintf(v), 0.f), 255.f);
        out[(long)pidx * out_stride + f] = f2b(q * (2.0f / 255.0f) - 1.0f);
    }
}

}  // namespace

int mul_bf16_out(const void* a, const void* b, void* out, int n,
                 cudaStream_t stream, void* of8, const void* fscale) {
    const int n2 = n >> 1;
    mul_bf16_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(out), n2,
        reinterpret_cast<__nv_fp8_e4m3*>(of8),
        reinterpret_cast<const float*>(fscale));
    return (int)cudaGetLastError();
}

// rms_norm_bf16_kernel (dim 768, unweighted) followed by
// rms_norm_rows96_exact_kernel over its 8 heads, in one 256-thread block per
// row. The row reduction is the reference block reduction verbatim; the
// normalized bf16 row stays in shared memory and warp h replays the per-head
// replica on head h. Bit-identical to the two-launch sequence.
// Output (row, head h, d): row-major [rows, 768] when hm_rows == 0, else
// head-major [rows / hm_rows][8][hm_rows][hm_stride] (blocks of hm_rows
// rows), the layout the persistent action kernel bulk-copies per head.
__device__ __forceinline__ long ctx_out_index(int row, int h, int hm_rows,
                                              int hm_stride) {
    if (hm_rows == 0) return (long)row * 768 + h * 96;
    return ((long)(row / hm_rows) * 8 + h) * hm_rows * hm_stride +
           (long)(row % hm_rows) * hm_stride;
}

__global__ void __launch_bounds__(256) rms_norm768_heads96_kernel(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    float eps, float eps_head, int hm_rows, int hm_stride) {
    constexpr int dim = 768, dim2 = 384;
    const int row = blockIdx.x;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(
        x + (long)row * dim);
    __shared__ float warp_sums[32];
    __shared__ float rms_s;
    __shared__ __nv_bfloat162 nrm[dim2];
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i];
        local_sum += b2f(v.x) * b2f(v.x) + b2f(v.y) * b2f(v.y);
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) rms_s = rsqrtf(v / (float)dim + eps);
    }
    __syncthreads();
    const float rms = rms_s;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i];
        __nv_bfloat162 r;
        r.x = f2b(b2f(xv.x) * rms);
        r.y = f2b(b2f(xv.y) * rms);
        nrm[i] = r;
    }
    __syncthreads();
    // warp h: rms_norm_rows96_exact on head h (48 pairs)
    const __nv_bfloat162* h2 = nrm + wid * 48;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(
        out + ctx_out_index(row, wid, hm_rows, hm_stride));
    const __nv_bfloat162 va = h2[lane];
    float a = b2f(va.x) * b2f(va.x) + b2f(va.y) * b2f(va.y);
    float b = 0.f;
    if (lane < 16) {
        const __nv_bfloat162 vb = h2[32 + lane];
        b = b2f(vb.x) * b2f(vb.x) + b2f(vb.y) * b2f(vb.y);
    }
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, o);
        b += __shfl_down_sync(0xffffffffu, b, o);
    }
    const float total = __shfl_sync(0xffffffffu, a + b, 0);
    const float hr = rsqrtf(total / 96.0f + eps_head);
    const auto store = [&](int pair) {
        const __nv_bfloat162 v = h2[pair];
        __nv_bfloat162 r;
        r.x = f2b(b2f(v.x) * hr);
        r.y = f2b(b2f(v.y) * hr);
        o2[pair] = r;
    };
    store(lane);
    if (lane < 16) store(32 + lane);
}

int rms_norm768_heads96(const void* x, void* out, int rows, float eps,
                        float eps_head, cudaStream_t stream, int hm_rows,
                        int hm_stride) {
    if (rows < 1 || hm_rows < 0 || (hm_rows > 0 && hm_stride < 96)) return 1;
    rms_norm768_heads96_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(out), eps, eps_head, hm_rows,
        hm_stride);
    return (int)cudaGetLastError();
}

// rms_norm_bf16_kernel (dim 768, unweighted) with the head-major output of
// rms_norm768_heads96: same block reduction, so the values are identical.
__global__ void __launch_bounds__(256) rms_norm768_headmajor_kernel(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    float eps, int hm_rows, int hm_stride) {
    constexpr int dim = 768, dim2 = 384;
    const int row = blockIdx.x;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(
        x + (long)row * dim);
    __shared__ float warp_sums[32];
    __shared__ float rms_s;
    float local_sum = 0.f;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 v = x2[i];
        local_sum += b2f(v.x) * b2f(v.x) + b2f(v.y) * b2f(v.y);
    }
    for (int o = 16; o > 0; o >>= 1)
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
    const int wid = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        const int nwarps = blockDim.x >> 5;
        float v = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) rms_s = rsqrtf(v / (float)dim + eps);
    }
    __syncthreads();
    const float rms = rms_s;
    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        const __nv_bfloat162 xv = x2[i];
        __nv_bfloat162 r;
        r.x = f2b(b2f(xv.x) * rms);
        r.y = f2b(b2f(xv.y) * rms);
        const int c = 2 * i;
        *reinterpret_cast<__nv_bfloat162*>(
            out + ctx_out_index(row, c / 96, hm_rows, hm_stride) + c % 96) = r;
    }
}

// Warp-per-row replicas of rms_norm768_heads96_kernel (HEADS) and
// rms_norm768_headmajor_kernel: lane l plays threads l + 32 w of the 256-thread
// CTA, so the per-thread partials, each warp's shfl_down tree, warp 0's tree
// over the eight warp sums and (HEADS) every head's trees run in the same
// order; the output bytes are identical, without the CTA barriers.
template <bool HEADS>
__global__ void __launch_bounds__(256) rms_norm768_ctx_warp_kernel(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    int rows, float eps, float eps_head, int hm_rows, int hm_stride) {
    constexpr int dim = 768, dim2 = 384, NW = 8;
    __shared__ __nv_bfloat162 nrm_s[8][dim2];
    const int lane = threadIdx.x & 31, wslot = threadIdx.x >> 5;
    const int row = blockIdx.x * 8 + wslot;
    if (row >= rows) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(
        x + (long)row * dim);
    __nv_bfloat162 xv[12];
#pragma unroll
    for (int j = 0; j < 12; ++j) xv[j] = x2[lane + 32 * j];
    float r = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        // thread 32w + l owns pairs 32w + l and (w < 4) 32w + l + 256
        float local_sum = 0.f;
        local_sum += b2f(xv[w].x) * b2f(xv[w].x) + b2f(xv[w].y) * b2f(xv[w].y);
        if (w < 4)
            local_sum += b2f(xv[w + 8].x) * b2f(xv[w + 8].x) +
                         b2f(xv[w + 8].y) * b2f(xv[w + 8].y);
        for (int o = 16; o > 0; o >>= 1)
            local_sum += __shfl_down_sync(0xffffffffu, local_sum, o);
        const float ws = __shfl_sync(0xffffffffu, local_sum, 0);
        if (lane == w) r = ws;
    }
    for (int o = 16; o > 0; o >>= 1) r += __shfl_down_sync(0xffffffffu, r, o);
    const float rms = __shfl_sync(0xffffffffu, rsqrtf(r / (float)dim + eps), 0);
    if constexpr (!HEADS) {
#pragma unroll
        for (int j = 0; j < 12; ++j) {
            const int i = lane + 32 * j;
            __nv_bfloat162 o2;
            o2.x = f2b(b2f(xv[j].x) * rms);
            o2.y = f2b(b2f(xv[j].y) * rms);
            const int c = 2 * i;
            *reinterpret_cast<__nv_bfloat162*>(
                out + ctx_out_index(row, c / 96, hm_rows, hm_stride) + c % 96) = o2;
        }
        return;
    } else {
        __nv_bfloat162* nrm = nrm_s[wslot];
#pragma unroll
        for (int j = 0; j < 12; ++j) {
            __nv_bfloat162 o2;
            o2.x = f2b(b2f(xv[j].x) * rms);
            o2.y = f2b(b2f(xv[j].y) * rms);
            nrm[lane + 32 * j] = o2;
        }
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            const __nv_bfloat162* h2 = nrm + h * 48;
            __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(
                out + ctx_out_index(row, h, hm_rows, hm_stride));
            const __nv_bfloat162 va = h2[lane];
            float a = b2f(va.x) * b2f(va.x) + b2f(va.y) * b2f(va.y);
            float b = 0.f;
            if (lane < 16) {
                const __nv_bfloat162 vb = h2[32 + lane];
                b = b2f(vb.x) * b2f(vb.x) + b2f(vb.y) * b2f(vb.y);
            }
            for (int o = 16; o > 0; o >>= 1) {
                a += __shfl_down_sync(0xffffffffu, a, o);
                b += __shfl_down_sync(0xffffffffu, b, o);
            }
            const float total = __shfl_sync(0xffffffffu, a + b, 0);
            const float hr = rsqrtf(total / 96.0f + eps_head);
            const auto store = [&](int pair) {
                const __nv_bfloat162 v = h2[pair];
                __nv_bfloat162 rr;
                rr.x = f2b(b2f(v.x) * hr);
                rr.y = f2b(b2f(v.y) * hr);
                o2[pair] = rr;
            };
            store(lane);
            if (lane < 16) store(32 + lane);
        }
    }
}

int rms_norm768_ctx_warp(const void* x, void* out, int rows, float eps,
                         float eps_head, int heads, int hm_rows,
                         int hm_stride, cudaStream_t stream) {
    if (rows < 1 || hm_rows < 0 || (hm_rows > 0 && hm_stride < 96) ||
        (!heads && hm_rows < 1))
        return 1;
    const dim3 grid((rows + 7) / 8);
    if (heads)
        rms_norm768_ctx_warp_kernel<true><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(out), rows, eps, eps_head,
            hm_rows, hm_stride);
    else
        rms_norm768_ctx_warp_kernel<false><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(out), rows, eps, eps_head,
            hm_rows, hm_stride);
    return (int)cudaGetLastError();
}

int rms_norm768_headmajor(const void* x, void* out, int rows, float eps,
                          int hm_rows, int hm_stride, cudaStream_t stream) {
    if (rows < 1 || hm_rows < 1 || hm_stride < 96) return 1;
    rms_norm768_headmajor_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(out), eps, hm_rows, hm_stride);
    return (int)cudaGetLastError();
}

int rms_norm_bf16(const void* x, const void* w, void* out, int rows,
                  int dim, float eps, cudaStream_t stream) {
    rms_norm_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<__nv_bfloat16*>(out), dim, eps);
    return (int)cudaGetLastError();
}

int gate_residual_bf16(void* res, const void* x, const void* gate, int rows,
                       int dim, cudaStream_t stream) {
    const int n = rows * dim;
    gate_residual_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(res),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gate), rows, dim);
    return (int)cudaGetLastError();
}

int silu_mul_bf16(const void* g, const void* u, void* out, int n,
                  cudaStream_t stream) {
    silu_mul_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g),
        reinterpret_cast<const __nv_bfloat16*>(u),
        reinterpret_cast<__nv_bfloat16*>(out), n);
    return (int)cudaGetLastError();
}

int bias_round_bf16(const void* acc, const void* bias, void* out, int rows,
                    int dim, cudaStream_t stream) {
    bias_round_bf16_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const float*>(acc),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out), dim);
    return (int)cudaGetLastError();
}

int bias_round_add_bf16(const void* acc, const void* bias, const void* add,
                        void* out, int rows, int dim, cudaStream_t stream) {
    if (dim % 4 != 0) return 1;
    const long n4 = (long)rows * dim / 4;
    bias_round_add_bf16_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const float*>(acc),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<const __nv_bfloat16*>(add),
        reinterpret_cast<__nv_bfloat16*>(out), dim, n4);
    return (int)cudaGetLastError();
}

int bias_rms_norm_bf16(const void* x, const void* bias, const void* w,
                        void* out, int rows, int heads, int hd, float eps,
                        cudaStream_t stream) {
    bias_rms_norm_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<__nv_bfloat16*>(out), heads, hd, eps);
    return 0;
}

int skinny_qkv_split_bias(const void* partials, int splits, const void* ags,
                         const void* wgs, const void* bias, void* Q, void* K,
                         void* V, int rows, int qd, int kd, int vd,
                         cudaStream_t stream) {
    const int n = rows * (qd + kd + vd);
    skinny_qkv_split_bias_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float*>(partials), splits,
        reinterpret_cast<const float*>(ags),
        reinterpret_cast<const float*>(wgs),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(Q),
        reinterpret_cast<__nv_bfloat16*>(K),
        reinterpret_cast<__nv_bfloat16*>(V), n, qd, kd, vd);
    return (int)cudaGetLastError();
}

int skinny_gate_residual(const void* partials, int splits, const void* ags,
                         const void* wgs, const void* bias, void* res,
                         const void* gate, int rows, int dim,
                         cudaStream_t stream) {
    const int n = rows * dim;
    skinny_gate_residual_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float*>(partials), splits,
        reinterpret_cast<const float*>(ags),
        reinterpret_cast<const float*>(wgs),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(res),
        reinterpret_cast<const __nv_bfloat16*>(gate), n, dim);
    return (int)cudaGetLastError();
}

int skinny_bias_rms_norm(const void* partials, int splits, const void* ags,
                         const void* wgs, const void* bias, void* out,
                         int rows, int heads, int hd, float eps,
                         cudaStream_t stream) {
    skinny_bias_rms_norm_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const float*>(partials), splits,
        reinterpret_cast<const float*>(ags),
        reinterpret_cast<const float*>(wgs),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out), heads, hd, rows * heads * hd,
        eps);
    return (int)cudaGetLastError();
}

int skinny_silu_mul(const void* pg, int splits_g, const void* ags_g,
                    const void* wgs_g, const void* pu, int splits_u,
                    const void* ags_u, const void* wgs_u, const void* bg,
                    const void* bu, void* out, int rows, int dim,
                    cudaStream_t stream) {
    const int n = rows * dim;
    skinny_silu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float*>(pg), splits_g,
        reinterpret_cast<const float*>(ags_g),
        reinterpret_cast<const float*>(wgs_g),
        reinterpret_cast<const float*>(pu), splits_u,
        reinterpret_cast<const float*>(ags_u),
        reinterpret_cast<const float*>(wgs_u),
        reinterpret_cast<const __nv_bfloat16*>(bg),
        reinterpret_cast<const __nv_bfloat16*>(bu),
        reinterpret_cast<__nv_bfloat16*>(out), n, dim);
    return (int)cudaGetLastError();
}

// FP8-only form of qkv_split_bias_norm_rope for head_dim 128 (no bias, RoPE,
// q/k norm weights): a half-warp owns a head and lane l owns elements
// 4l..4l+3 of both rotate halves, so every access is 8 bytes (bf16) or 4
// bytes (FP8) and the head is read once. The per-element arithmetic is that
// of the generic kernel; only the sum-of-squares order differs (4-lane
// partials over 16 lanes instead of 2 per lane over 32).
__global__ void qkv_norm_rope_fp8_hd128_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ cos_t,
    const __nv_bfloat16* __restrict__ sin_t,
    __nv_fp8_e4m3* __restrict__ Qf8, __nv_fp8_e4m3* __restrict__ Kf8,
    __nv_fp8_e4m3* __restrict__ Vf8, const float* __restrict__ qscale,
    const float* __restrict__ kscale, const float* __restrict__ vscale,
    int nq, int nk, int vd, float eps) {
    constexpr int HD = 128, HALF = 64;
    const int row = blockIdx.x;
    const int qd = nq * HD, kd = nk * HD;
    const __nv_bfloat16* src = qkv + (long)row * (qd + kd + vd);
    const int tid = threadIdx.x;
    if (Vf8 != nullptr) {
        const float inv_v = 1.0f / (*vscale);
        for (int i8 = tid * 8; i8 < vd; i8 += blockDim.x * 8) {
            const uint4 vs = *reinterpret_cast<const uint4*>(src + qd + kd + i8);
            const __nv_bfloat16* vv = reinterpret_cast<const __nv_bfloat16*>(&vs);
            __nv_fp8_e4m3 outf[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                outf[j] = __nv_fp8_e4m3(fminf(fmaxf(
                    b2f(vv[j]) * inv_v, -448.0f), 448.0f));
            *reinterpret_cast<uint2*>(Vf8 + (long)row * vd + i8) =
                *reinterpret_cast<uint2*>(outf);
        }
    }
    const int lane = tid & 31, l16 = lane & 15;
    const int slot0 = (tid >> 5) * 2 + (lane >> 4), nslots = (blockDim.x >> 5) * 2;
    const int d0 = 4 * l16;
    const uint2 cs = *reinterpret_cast<const uint2*>(cos_t + (long)row * HD + d0);
    const uint2 sn = *reinterpret_cast<const uint2*>(sin_t + (long)row * HD + d0);
    const __nv_bfloat16* cv = reinterpret_cast<const __nv_bfloat16*>(&cs);
    const __nv_bfloat16* sv = reinterpret_cast<const __nv_bfloat16*>(&sn);
    const float inv_q = Qf8 != nullptr ? 1.0f / (*qscale) : 0.f;
    const float inv_k = 1.0f / (*kscale);
    // all 16 lanes of a half-warp share the slot, so the loop bound and the
    // shuffles stay half-warp uniform
    for (int h = slot0; h < nq + nk; h += nslots) {
        const bool is_q = h < nq;
        const int head = is_q ? h : h - nq;
        const int off = is_q ? head * HD : qd + head * HD;
        const uint2 lo = *reinterpret_cast<const uint2*>(src + off + d0);
        const uint2 hi = *reinterpret_cast<const uint2*>(src + off + HALF + d0);
        const __nv_bfloat16* lv = reinterpret_cast<const __nv_bfloat16*>(&lo);
        const __nv_bfloat16* hv = reinterpret_cast<const __nv_bfloat16*>(&hi);
        const __nv_bfloat16* w = is_q ? qw : kw;
        const uint2 wl = *reinterpret_cast<const uint2*>(w + d0);
        const uint2 wh = *reinterpret_cast<const uint2*>(w + HALF + d0);
        const __nv_bfloat16* wlv = reinterpret_cast<const __nv_bfloat16*>(&wl);
        const __nv_bfloat16* whv = reinterpret_cast<const __nv_bfloat16*>(&wh);
        float a[4], b[4];
        float ssq = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            a[j] = b2f(lv[j]);
            b[j] = b2f(hv[j]);
            ssq += a[j] * a[j] + b[j] * b[j];
        }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1)
            ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
        const float inv = rsqrtf(ssq / (float)HD + eps);
        const float sc = is_q ? inv_q : inv_k;
        __nv_fp8_e4m3 qlo[4], qhi[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float nlo = b2f(f2b(a[j] * inv));
            float nhi = b2f(f2b(b[j] * inv));
            nlo = b2f(f2b(nlo * b2f(wlv[j])));
            nhi = b2f(f2b(nhi * b2f(whv[j])));
            const float c = b2f(cv[j]);
            const float si = b2f(sv[j]);
            const float r0 = b2f(f2b(nlo * c));
            const float r1 = b2f(f2b(nhi * si));
            const float r2 = b2f(f2b(nhi * c));
            const float r3 = b2f(f2b(nlo * si));
            const float o0 = b2f(f2b(r0 - r1));
            const float o1 = b2f(f2b(r2 + r3));
            qlo[j] = __nv_fp8_e4m3(fminf(fmaxf(o0 * sc, -448.0f), 448.0f));
            qhi[j] = __nv_fp8_e4m3(fminf(fmaxf(o1 * sc, -448.0f), 448.0f));
        }
        __nv_fp8_e4m3* dst = is_q ? Qf8 + (long)row * qd + head * HD
                                  : Kf8 + (long)row * kd + head * HD;
        if (!is_q || Qf8 != nullptr) {
            *reinterpret_cast<uint32_t*>(dst + d0) = *reinterpret_cast<uint32_t*>(qlo);
            *reinterpret_cast<uint32_t*>(dst + HALF + d0) = *reinterpret_cast<uint32_t*>(qhi);
        }
    }
}

int qkv_norm_rope_fp8_hd128(const void* qkv, const void* qw, const void* kw,
                            const void* cos_t, const void* sin_t, void* Qf8,
                            void* Kf8, void* Vf8, const void* qscale,
                            const void* kscale, const void* vscale, int rows,
                            int nq, int nk, int vd, float eps,
                            cudaStream_t stream) {
    if (!qw || !kw || !Kf8 || (Qf8 && !qscale) || (Vf8 && !vscale) || vd % 8)
        return 1;
    qkv_norm_rope_fp8_hd128_kernel<<<rows, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(qw),
        reinterpret_cast<const __nv_bfloat16*>(kw),
        reinterpret_cast<const __nv_bfloat16*>(cos_t),
        reinterpret_cast<const __nv_bfloat16*>(sin_t),
        reinterpret_cast<__nv_fp8_e4m3*>(Qf8),
        reinterpret_cast<__nv_fp8_e4m3*>(Kf8),
        reinterpret_cast<__nv_fp8_e4m3*>(Vf8),
        reinterpret_cast<const float*>(qscale),
        reinterpret_cast<const float*>(kscale),
        reinterpret_cast<const float*>(vscale), nq, nk, vd, eps);
    return (int)cudaGetLastError();
}

int qkv_split_bias_norm_rope(const void* qkv, const void* bias,
                              const void* qw, const void* kw,
                              const void* cos_t, const void* sin_t, void* Q,
                              void* K, void* V, int rows, int qd, int kd,
                              int vd, int nq, int nk, int hd, float eps,
                              int has_bias, int has_rope, void* Kf8,
                              void* Vf8, const void* kscale,
                              const void* vscale, void* Qf8,
                              const void* qscale, cudaStream_t stream) {
    qkv_split_bias_norm_rope_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<const __nv_bfloat16*>(qw),
        reinterpret_cast<const __nv_bfloat16*>(kw),
        reinterpret_cast<const __nv_bfloat16*>(cos_t),
        reinterpret_cast<const __nv_bfloat16*>(sin_t),
        reinterpret_cast<__nv_bfloat16*>(Q),
        reinterpret_cast<__nv_bfloat16*>(K),
        reinterpret_cast<__nv_bfloat16*>(V), qd, kd, vd, nq, nk, hd, eps,
        has_bias, has_rope, reinterpret_cast<__nv_fp8_e4m3*>(Kf8),
        reinterpret_cast<__nv_fp8_e4m3*>(Vf8),
        reinterpret_cast<const float*>(kscale),
        reinterpret_cast<const float*>(vscale),
        reinterpret_cast<__nv_fp8_e4m3*>(Qf8),
        reinterpret_cast<const float*>(qscale));
    return (int)cudaGetLastError();
}

int layer_norm_quant_fp8(const void* x, const void* w, const void* b,
                         void* out, const void* scale, int rows, int dim,
                         float eps, cudaStream_t stream) {
    layer_norm_quant_fp8_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        reinterpret_cast<const float*>(scale), dim, eps);
    return (int)cudaGetLastError();
}

__global__ void reciprocal_f32_kernel(const float* __restrict__ src,
                                      float* __restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 1.0f / src[i];
}

int reciprocal_f32(const void* src, void* dst, int n, cudaStream_t stream) {
    if (n <= 0) return 1;
    reciprocal_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst),
        n);
    return (int)cudaGetLastError();
}

int bias_gelu_quant_fp8_vec(const void* x, const void* b, void* out,
                            const void* scale, int rows, int dim,
                            cudaStream_t stream) {
    if (rows <= 0 || dim <= 0 || (dim & 7) != 0) return 1;
    const int n8 = (rows * dim) >> 3;
    bias_gelu_quant_fp8_vec_kernel<<<(n8 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        reinterpret_cast<const float*>(scale), n8, dim);
    return (int)cudaGetLastError();
}

int bias_gelu_quant_fp8(const void* x, const void* b, void* out,
                        const void* scale, int rows, int dim,
                        cudaStream_t stream) {
    const int n = rows * dim;
    bias_gelu_quant_fp8_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        reinterpret_cast<const float*>(scale), n, dim);
    return (int)cudaGetLastError();
}

int qkv_split_vec(const void* qkv, void* Q, void* K, void* V, int rows,
                  int qd, int kd, int vd, cudaStream_t stream) {
    qkv_split_vec_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<__nv_bfloat16*>(Q),
        reinterpret_cast<__nv_bfloat16*>(K),
        reinterpret_cast<__nv_bfloat16*>(V), qd, kd, vd);
    return (int)cudaGetLastError();
}

int qkv_split_bias(const void* qkv, const void* bias, void* Q, void* K,
                   void* V, int seq, int qd, int kd, int vd,
                   cudaStream_t stream) {
    qkv_split_bias_kernel<<<seq, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(Q),
        reinterpret_cast<__nv_bfloat16*>(K),
        reinterpret_cast<__nv_bfloat16*>(V), qd, kd, vd);
    return (int)cudaGetLastError();
}

int silu_mul_bias(const void* g, const void* u, const void* bg,
                  const void* bu, void* out, int rows, int dim,
                  cudaStream_t stream) {
    const int n = rows * dim;
    silu_mul_bias_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g),
        reinterpret_cast<const __nv_bfloat16*>(u),
        reinterpret_cast<const __nv_bfloat16*>(bg),
        reinterpret_cast<const __nv_bfloat16*>(bu),
        reinterpret_cast<__nv_bfloat16*>(out), n, dim);
    return (int)cudaGetLastError();
}

// Bit-exact warp-per-row replica of rms_norm_bf16_kernel for dim == 96
// (unweighted): the reference sums one bf16x2 pair per lane of a 256-thread
// block, reduces warp 0 (pairs 0..31) and warp 1 (pairs 32..47) with the
// standard shuffle tree, then adds the two warp sums. Reproducing that exact
// grouping removes the block barrier without changing a single output bit.
__global__ void rms_norm_rows96_exact_kernel(
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
    int rows, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(
        x + (long)warp * 96);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(
        out + (long)warp * 96);
    const __nv_bfloat162 va = x2[lane];       // pair lane (0..31)
    float a = b2f(va.x) * b2f(va.x) + b2f(va.y) * b2f(va.y);
    float b = 0.f;
    if (lane < 16) {
        const __nv_bfloat162 vb = x2[32 + lane];
        b = b2f(vb.x) * b2f(vb.x) + b2f(vb.y) * b2f(vb.y);
    }
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, o);
        b += __shfl_down_sync(0xffffffffu, b, o);
    }
    const float total = __shfl_sync(0xffffffffu, a + b, 0);
    const float rms = rsqrtf(total / 96.0f + eps);
    const auto store = [&](int pair) {
        const __nv_bfloat162 v = x2[pair];
        __nv_bfloat162 r;
        r.x = f2b(b2f(v.x) * rms);
        r.y = f2b(b2f(v.y) * rms);
        o2[pair] = r;
    };
    store(lane);
    if (lane < 16) store(32 + lane);
}

// Bit-exact warp-per-row replica of mod_rms_norm_quant_fp8_kernel for
// dim == 768, unweighted. Same virtual-warp grouping as the bf16 replica
// above; the elementwise sequence (norm, modulation, static-scale FP8
// quantisation) is copied operation for operation.
__global__ void mod_rms_norm_rows768_quant_fp8_exact_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_fp8_e4m3* __restrict__ out, const float* __restrict__ qscale,
    int rows, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(
        x + (long)warp * 768);
    __nv_fp8_e4m3* orow = out + (long)warp * 768;
    float p[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const int i = 32 * w + lane;
        const __nv_bfloat162 v = x2[i];
        float s = b2f(v.x) * b2f(v.x) + b2f(v.y) * b2f(v.y);
        if (w < 4) {
            const __nv_bfloat162 u = x2[256 + i];
            s += b2f(u.x) * b2f(u.x) + b2f(u.y) * b2f(u.y);
        }
        p[w] = s;
    }
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        float a = p[w];
        for (int o = 16; o > 0; o >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, o);
        p[w] = a;
    }
    float total = 0.f;
    if (lane == 0) {
        const float t0 = p[0] + p[4], t1 = p[1] + p[5];
        const float t2 = p[2] + p[6], t3 = p[3] + p[7];
        total = (t0 + t2) + (t1 + t3);
    }
    const float rms = rsqrtf(__shfl_sync(0xffffffffu, total, 0) / 768.0f + eps);
    const float inv_s = 1.0f / (*qscale);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    const auto store = [&](int i) {
        const __nv_bfloat162 xv = x2[i];
        const float vx = b2f(xv.x) * rms;
        const float vy = b2f(xv.y) * rms;
        const __nv_bfloat162 sv = s2[i];
        const __nv_bfloat162 hv = h2[i];
        const float nx = b2f(f2b(vx));
        const float ny = b2f(f2b(vy));
        const float ax = b2f(f2b(1.0f + b2f(sv.x)));
        const float ay = b2f(f2b(1.0f + b2f(sv.y)));
        const float mx = b2f(f2b(nx * ax)) + b2f(hv.x);
        const float my = b2f(f2b(ny * ay)) + b2f(hv.y);
        const __nv_fp8_e4m3 q0(fminf(fmaxf(b2f(f2b(mx)) * inv_s, -448.0f),
                                     448.0f));
        const __nv_fp8_e4m3 q1(fminf(fmaxf(b2f(f2b(my)) * inv_s, -448.0f),
                                     448.0f));
        *reinterpret_cast<uint16_t*>(orow + 2 * i) =
            static_cast<uint16_t>(q0.__x) |
            (static_cast<uint16_t>(q1.__x) << 8);
    };
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        store(32 * w + lane);
        if (w < 4) store(256 + 32 * w + lane);
    }
}

// skinny_gate_residual_kernel followed by
// mod_rms_norm_rows768_quant_fp8_exact_kernel in one launch. One 256-thread
// block per row: warp w owns exactly the pairs that the norm replica's
// virtual warp w reads (32w + lane and, for w < 4, 256 + 32w + lane), so the
// per-w shuffle sums and their final combination are reproduced operation
// for operation; the residual stream and the FP8 operand are bit-identical
// to the two-launch sequence.
__global__ void __launch_bounds__(256)
skinny_gate_residual_mod_rms768_quant_fp8_kernel(
    const float* __restrict__ partials, int splits,
    const float* __restrict__ ags, const float* __restrict__ wgs,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ res,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_fp8_e4m3* __restrict__ out, const float* __restrict__ qscale,
    int rows, float eps) {
    __shared__ float wsum[8];
    const int row = blockIdx.x;
    const int w = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = rows * 768;
    const float alpha = (*ags) * (*wgs);
    __nv_bfloat162* x2 = reinterpret_cast<__nv_bfloat162*>(
        res + (long)row * 768);
    const auto update = [&](int i) -> __nv_bfloat162 {
        const __nv_bfloat162 xv = x2[i];
        float o[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int c = 2 * i + e;
            const long idx = (long)row * 768 + c;
            const float a = __bfloat162float(__float2bfloat16(
                skinny_sum(partials, splits, idx, n, alpha) + b2f(bias[c])));
            const float pv = __bfloat162float(
                __float2bfloat16(a * b2f(gate[c])));
            o[e] = b2f(e == 0 ? xv.x : xv.y) + pv;
        }
        __nv_bfloat162 r;
        r.x = f2b(o[0]);
        r.y = f2b(o[1]);
        x2[i] = r;
        return r;
    };
    const int ia = 32 * w + lane;
    const __nv_bfloat162 va = update(ia);
    float sq = b2f(va.x) * b2f(va.x) + b2f(va.y) * b2f(va.y);
    __nv_bfloat162 vb;
    if (w < 4) {
        vb = update(256 + ia);
        sq += b2f(vb.x) * b2f(vb.x) + b2f(vb.y) * b2f(vb.y);
    }
    for (int o = 16; o > 0; o >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, o);
    if (lane == 0) wsum[w] = sq;
    __syncthreads();
    const float t0 = wsum[0] + wsum[4], t1 = wsum[1] + wsum[5];
    const float t2 = wsum[2] + wsum[6], t3 = wsum[3] + wsum[7];
    const float total = (t0 + t2) + (t1 + t3);
    const float rms = rsqrtf(total / 768.0f + eps);
    const float inv_s = 1.0f / (*qscale);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    __nv_fp8_e4m3* orow = out + (long)row * 768;
    const auto store = [&](int i, __nv_bfloat162 xv) {
        const float vx = b2f(xv.x) * rms;
        const float vy = b2f(xv.y) * rms;
        const __nv_bfloat162 sv = s2[i];
        const __nv_bfloat162 hv = h2[i];
        const float nx = b2f(f2b(vx));
        const float ny = b2f(f2b(vy));
        const float ax = b2f(f2b(1.0f + b2f(sv.x)));
        const float ay = b2f(f2b(1.0f + b2f(sv.y)));
        const float mx = b2f(f2b(nx * ax)) + b2f(hv.x);
        const float my = b2f(f2b(ny * ay)) + b2f(hv.y);
        const __nv_fp8_e4m3 q0(fminf(fmaxf(b2f(f2b(mx)) * inv_s, -448.0f),
                                     448.0f));
        const __nv_fp8_e4m3 q1(fminf(fmaxf(b2f(f2b(my)) * inv_s, -448.0f),
                                     448.0f));
        *reinterpret_cast<uint16_t*>(orow + 2 * i) =
            static_cast<uint16_t>(q0.__x) |
            (static_cast<uint16_t>(q1.__x) << 8);
    };
    store(ia, va);
    if (w < 4) store(256 + ia, vb);
}

int skinny_gate_residual_mod_rms768_quant_fp8(
    const void* partials, int splits, const void* ags, const void* wgs,
    const void* bias, void* res, const void* gate, const void* scale,
    const void* shift, void* out, const void* qscale, int rows, float eps,
    cudaStream_t stream) {
    if (rows < 1) return 1;
    skinny_gate_residual_mod_rms768_quant_fp8_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const float*>(partials), splits,
        reinterpret_cast<const float*>(ags),
        reinterpret_cast<const float*>(wgs),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(res),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(shift),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        reinterpret_cast<const float*>(qscale), rows, eps);
    return (int)cudaGetLastError();
}

int mod_rms_norm_rows768_quant_fp8_exact(
    const void* x, const void* scale, const void* shift, void* out,
    const void* qscale, int rows, float eps, cudaStream_t stream) {
    if (rows < 1) return 1;
    mod_rms_norm_rows768_quant_fp8_exact_kernel<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(shift),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        reinterpret_cast<const float*>(qscale), rows, eps);
    return (int)cudaGetLastError();
}

// Bit-exact warp-per-row replica of mod_rms_norm_bf16_kernel for dim == 768
// (unweighted). The reference runs 256 threads per row: threads 0..127 sum
// one bf16x2 pair at i and i+256, threads 128..255 one pair at i; then the
// standard shuffle tree per warp and the 8-warp sum tree. One warp reproduces
// that grouping with eight virtual-warp accumulators per lane, so the output
// is bit-identical while the block barrier disappears.
__global__ void mod_rms_norm_rows768_exact_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out, int rows, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(
        x + (long)warp * 768);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(
        out + (long)warp * 768);
    float p[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const int i = 32 * w + lane;
        const __nv_bfloat162 v = x2[i];
        float s = b2f(v.x) * b2f(v.x) + b2f(v.y) * b2f(v.y);
        if (w < 4) {
            const __nv_bfloat162 u = x2[256 + i];
            s += b2f(u.x) * b2f(u.x) + b2f(u.y) * b2f(u.y);
        }
        p[w] = s;
    }
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        float a = p[w];
        for (int o = 16; o > 0; o >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, o);
        p[w] = a;
    }
    float total = 0.f;
    if (lane == 0) {
        const float t0 = p[0] + p[4], t1 = p[1] + p[5];
        const float t2 = p[2] + p[6], t3 = p[3] + p[7];
        total = (t0 + t2) + (t1 + t3);
    }
    const float rms = rsqrtf(__shfl_sync(0xffffffffu, total, 0) / 768.0f + eps);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    const auto store = [&](int i) {
        const __nv_bfloat162 xv = x2[i];
        const float vx = b2f(xv.x) * rms;
        const float vy = b2f(xv.y) * rms;
        const float nx = b2f(f2b(vx));
        const float ny = b2f(f2b(vy));
        const __nv_bfloat162 sv = s2[i];
        const float ax = b2f(f2b(1.0f + b2f(sv.x)));
        const float ay = b2f(f2b(1.0f + b2f(sv.y)));
        const float mx = b2f(f2b(nx * ax));
        const float my = b2f(f2b(ny * ay));
        const __nv_bfloat162 hv = h2[i];
        o2[i] = __nv_bfloat162(f2b(mx + b2f(hv.x)), f2b(my + b2f(hv.y)));
    };
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        store(32 * w + lane);
        if (w < 4) store(256 + 32 * w + lane);
    }
}

int mod_rms_norm_rows768_exact(const void* x, const void* scale,
                               const void* shift, void* out, int rows,
                               float eps, cudaStream_t stream) {
    if (rows < 1) return 1;
    // one warp per CTA: 30 rows occupy 30 SMs (the 8-warp form used 4)
    mod_rms_norm_rows768_exact_kernel<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(shift),
        reinterpret_cast<__nv_bfloat16*>(out), rows, eps);
    return (int)cudaGetLastError();
}

// Bit-exact warp-per-row replica of bias_rms_norm_bf16_kernel for hd == 96
// (unweighted). The reference sums one bf16x2 pair per lane of a 256-thread
// block after rounding the bias add to bf16, reduces warp 0 (pairs 0..31) and
// warp 1 (pairs 32..47) with the shuffle tree, then adds the two warp sums.
// Reproducing that grouping removes the block barrier without changing a bit.
__global__ void bias_rms_norm_rows96_exact_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out, int rows, int heads, float eps) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(
        x + (long)warp * 96);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(
        bias + (long)(warp % heads) * 96);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(out + (long)warp * 96);
    const auto biased = [&](int pair, float2& v) {
        const __nv_bfloat162 xv = x2[pair], bv = b2[pair];
        v.x = b2f(f2b(b2f(xv.x) + b2f(bv.x)));
        v.y = b2f(f2b(b2f(xv.y) + b2f(bv.y)));
    };
    float2 p;
    biased(lane, p);
    float a = p.x * p.x + p.y * p.y;
    float b = 0.f;
    if (lane < 16) {
        biased(32 + lane, p);
        b = p.x * p.x + p.y * p.y;
    }
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_down_sync(0xffffffffu, a, o);
        b += __shfl_down_sync(0xffffffffu, b, o);
    }
    const float total = __shfl_sync(0xffffffffu, a + b, 0);
    const float rms = rsqrtf(total / 96.0f + eps);
    const auto store = [&](int pair) {
        float2 v;
        biased(pair, v);
        __nv_bfloat162 r;
        r.x = f2b(v.x * rms);
        r.y = f2b(v.y * rms);
        o2[pair] = r;
    };
    store(lane);
    if (lane < 16) store(32 + lane);
}

int bias_rms_norm_rows96_exact(const void* x, const void* bias, void* out,
                               int rows, int heads, float eps,
                               cudaStream_t stream) {
    if (rows < 1 || heads < 1) return 1;
    const int warps = 256 / 32;
    const int blocks = (rows + warps - 1) / warps;
    bias_rms_norm_rows96_exact_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<__nv_bfloat16*>(out), rows, heads, eps);
    return (int)cudaGetLastError();
}

int rms_norm_rows96_exact(const void* x, void* out, int rows, float eps,
                          cudaStream_t stream) {
    if (rows < 1) return 1;
    const int warps = 256 / 32;
    const int blocks = (rows + warps - 1) / warps;
    rms_norm_rows96_exact_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(out), rows, eps);
    return (int)cudaGetLastError();
}

int rms_norm_rows_bf16(const void* x, const void* w, void* out, int rows,
                       int dim, float eps, cudaStream_t stream) {
    if (rows < 1 || dim < 8 || (dim & 7) != 0 || dim > 2048) return 1;
    const int warps = 256 / 32;
    const int blocks = (rows + warps - 1) / warps;
    rms_norm_rows_bf16_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<__nv_bfloat16*>(out), rows, dim, eps);
    return (int)cudaGetLastError();
}

int gate_residual_bias(void* res, const void* x, const void* gate,
                       const void* bias, int rows, int dim,
                       cudaStream_t stream) {
    const int n8 = (rows * dim) >> 3;
    gate_residual_bias_kernel<<<(n8 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(res),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(bias), rows, dim);
    return (int)cudaGetLastError();
}

int mod_rms_norm_quant_fp8(const void* x, const void* w, const void* scale,
                           const void* shift, void* out, const void* qscale,
                           int rows, int dim, float eps,
                           cudaStream_t stream) {
    mod_rms_norm_quant_fp8_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(shift),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        reinterpret_cast<const float*>(qscale), dim, eps);
    return (int)cudaGetLastError();
}

int silu_mul_quant_fp8(const void* g, const void* u, const void* bg,
                       const void* bu, void* out, const void* qscale,
                       int rows, int dim, cudaStream_t stream) {
    const int n = rows * dim;
    silu_mul_quant_fp8_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g),
        reinterpret_cast<const __nv_bfloat16*>(u),
        reinterpret_cast<const __nv_bfloat16*>(bg),
        reinterpret_cast<const __nv_bfloat16*>(bu),
        reinterpret_cast<__nv_fp8_e4m3*>(out),
        reinterpret_cast<const float*>(qscale), n, dim);
    return (int)cudaGetLastError();
}

int cast_bf16_to_fp16(const void* src, void* dst, int n,
                      cudaStream_t stream) {
    cast_bf16_to_fp16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src),
        reinterpret_cast<__half*>(dst), n);
    return (int)cudaGetLastError();
}

int cast_fp16_to_bf16(const void* src, void* dst, int n,
                      cudaStream_t stream) {
    cast_fp16_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(src),
        reinterpret_cast<__nv_bfloat16*>(dst), n);
    return (int)cudaGetLastError();
}

int add_bias_bf16(void* x, const void* bias, int rows, int dim,
                  cudaStream_t stream) {
    add_bias_bf16_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(bias), dim);
    return (int)cudaGetLastError();
}

int action_denorm_bf16(const void* traj, void* out, const void* q01,
                       const void* q99, const void* mask, int rows,
                       int stride, int act_dim, cudaStream_t stream) {
    action_denorm_bf16_kernel<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(traj),
        reinterpret_cast<float*>(out), reinterpret_cast<const float*>(q01),
        reinterpret_cast<const float*>(q99),
        reinterpret_cast<const signed char*>(mask), stride, act_dim);
    return (int)cudaGetLastError();
}

int qk_norm_rope_bf16(void* x, const void* w, const void* cos_t,
                      const void* sin_t, int S, int NH, int HD, float eps,
                      int has_weight, cudaStream_t stream) {
    if (HD <= 0 || (HD & 1) || S <= 0 || NH <= 0) return -1;
    const int total_warps = S * NH;
    const int threads = 128;
    const int blocks = (total_warps * 32 + threads - 1) / threads;
    qk_norm_rope_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<const __nv_bfloat16*>(cos_t),
        reinterpret_cast<const __nv_bfloat16*>(sin_t), S, NH, HD, eps,
        has_weight);
    return (int)cudaGetLastError();
}

int concat2_quant_fp8(const void* a, const void* b, void* out, int rows,
                      int cols_a, int cols_b, cudaStream_t stream,
                      void* of8, const void* fscale) {
    concat2_quant_fp8_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<__nv_fp8_e4m3*>(of8),
        reinterpret_cast<const float*>(fscale), rows, cols_a, cols_b);
    return (int)cudaGetLastError();
}

int pool_mean_bf16(const void* feats, const void* idx, void* out,
                   int P, int I, int D, cudaStream_t stream, void* of8,
                   const void* fscale) {
    pool_mean_bf16_kernel<<<P, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(feats),
        reinterpret_cast<const int*>(idx),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<__nv_fp8_e4m3*>(of8),
        reinterpret_cast<const float*>(fscale), P, I, D);
    return (int)cudaGetLastError();
}

int pool_attn_bf16(const void* q, const void* k, const void* v,
                   const void* idx, void* out, int P, int I, int NH, int HD,
                   float scale, cudaStream_t stream, void* of8,
                   const void* fscale) {
    if (I > 8) return -1;
    pool_attn_bf16_kernel<<<P, NH * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const int*>(idx),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<__nv_fp8_e4m3*>(of8),
        reinterpret_cast<const float*>(fscale), P, I, NH, HD, scale);
    return (int)cudaGetLastError();
}

int mod_rms_norm_bf16(const void* x, const void* w, const void* scale,
                      const void* shift, void* out, int rows, int dim,
                      float eps, cudaStream_t stream) {
    mod_rms_norm_bf16_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(shift),
        reinterpret_cast<__nv_bfloat16*>(out), dim, eps);
    return (int)cudaGetLastError();
}

int state_discretize_f32(const void* state, const void* q01, const void* q99,
                         const void* mask, const void* zero_mask, void* ids,
                         int D, int num_tokens, int base_id,
                         cudaStream_t stream) {
    const int threads = 64;
    state_discretize_f32_kernel<<<(D + threads - 1) / threads, threads, 0,
                                  stream>>>(
        reinterpret_cast<const float*>(state),
        reinterpret_cast<const float*>(q01),
        reinterpret_cast<const float*>(q99),
        reinterpret_cast<const signed char*>(mask),
        reinterpret_cast<const signed char*>(zero_mask),
        reinterpret_cast<long long*>(ids), D, num_tokens, base_id);
    return (int)cudaGetLastError();
}

int mask_dims_bf16(void* x, int rows, int dim, int valid,
                   cudaStream_t stream) {
    if (valid >= dim) return 0;
    mask_dims_bf16_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x), rows, dim, valid);
    return (int)cudaGetLastError();
}

int resize_patchify_bf16(const void* img, void* out, int H, int W, int C,
                         int OH, int OW, int patch, cudaStream_t stream,
                         int out_stride) {
    if (OH % patch != 0 || OW % patch != 0) return -1;
    const int n_patch_y = OH / patch;
    const int n_patch_w = OW / patch;
    const int feat_per_patch = patch * patch * C;
    if (out_stride == 0) out_stride = feat_per_patch;
    if (out_stride < feat_per_patch) return -1;
    const long total = (long)n_patch_y * n_patch_w * feat_per_patch;
    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);
    resize_patchify_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(img),
        reinterpret_cast<__nv_bfloat16*>(out), H, W, C, OH, OW, patch,
        n_patch_y, n_patch_w, feat_per_patch, out_stride);
    return (int)cudaGetLastError();
}


// ── Action-expert SwiGLU FFN producer ────────────────────────────────────
// One launch replaces [up GEMM] + [gate GEMM] + [silu_mul_bias]:
//   out[m, n] = bf16( silu(bf16(bf16(acc_g * alpha_g) + bg[n])) *
//                    bf16(bf16(acc_u * alpha_u) + bu[n]) )
// with acc_u = A_fp8 @ Wu_fp8^T and acc_g = A_fp8 @ Wg_fp8^T in fp32.
// A is [M, K] e4m3 with the calibrated static activation scale folded into
// alpha_u/alpha_g; Wu/Wg are [N, K] e4m3 (checkpoint row-major layout, so no
// transposed copy is needed). The per-element rounding sequence matches
// silu_mul_bias_kernel exactly.
//
// One CTA per 32-column N tile covering all M <= 32 rows (so each weight row
// is streamed once per inference), 4 warps, one 8-wide n-atom per warp and
// two 16-row m-atoms; 2-stage cp.async over 128-deep K chunks.
constexpr int kSwRows = 32;
constexpr int kSwN = 32;
constexpr int kSwBK = 128;
constexpr int kSwStages = 2;
constexpr int kSwThreads = 128;
constexpr int kSwPad = kSwBK + 16;
constexpr int kSwTile = kSwRows * kSwPad;

__device__ __forceinline__ uint32_t sw_smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void sw_cp_async16(uint32_t smem, const uint8_t* src) {
    int b = (src == nullptr) ? 0 : 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem), "l"(src), "r"(b));
}

__device__ __forceinline__ void sw_mma_fp8(float c[4], uint32_t a0, uint32_t a1,
                                           uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(kSwThreads, 4)
action_swiglu_kernel(const uint8_t* __restrict__ A,
                     const uint8_t* __restrict__ Wu,
                     const uint8_t* __restrict__ Wg,
                     const __nv_bfloat16* __restrict__ bu,
                     const __nv_bfloat16* __restrict__ bg,
                     __nv_bfloat16* __restrict__ out,
                     int M, int K, int N, float alpha_u, float alpha_g) {
    extern __shared__ __align__(16) uint8_t sw_smem[];
    uint8_t* As = sw_smem;
    uint8_t* Us = As + kSwStages * kSwTile;
    uint8_t* Gs = Us + kSwStages * kSwTile;
    const int n0 = blockIdx.x * kSwN;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int l = lane & 3, h = lane >> 2;

    auto issue = [&](int stage, int kb) {
        // 32 rows x 8 chunks of 16 B = 256 chunks per operand.
        for (int it = 0; it < 2; ++it) {
            int idx = it * kSwThreads + t;
            int ra = idx >> 3, ko = (idx & 7) * 16;
            const uint8_t* src = (ra < M) ? A + (size_t)ra * K + kb + ko
                                          : nullptr;
            sw_cp_async16(sw_smem_u32(
                &As[stage * kSwTile + ra * kSwPad + ko]), src);
        }
        for (int it = 0; it < 2; ++it) {
            int idx = it * kSwThreads + t;
            int rb = idx >> 3, ko = (idx & 7) * 16;
            int ng = n0 + rb;
            const uint8_t* su = (ng < N) ? Wu + (size_t)ng * K + kb + ko
                                         : nullptr;
            const uint8_t* sg = (ng < N) ? Wg + (size_t)ng * K + kb + ko
                                         : nullptr;
            sw_cp_async16(sw_smem_u32(
                &Us[stage * kSwTile + rb * kSwPad + ko]), su);
            sw_cp_async16(sw_smem_u32(
                &Gs[stage * kSwTile + rb * kSwPad + ko]), sg);
        }
    };

    float acc_u[2][4] = {}, acc_g[2][4] = {};
    int is = 0, k_issue = 0;
#pragma unroll
    for (int s = 0; s < kSwStages - 1; ++s) {
        if (k_issue < K) issue(is, k_issue);
        asm volatile("cp.async.commit_group;\n" ::);
        is = (is + 1) % kSwStages;
        k_issue += kSwBK;
    }
    int stage = 0;
    for (int kb = 0; kb < K; kb += kSwBK) {
        if (k_issue < K) issue(is, k_issue);
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group %0;\n" :: "n"(kSwStages - 1));
        __syncthreads();
        const int brow = warp * 8 + h;              // local B row (n in tile)
#pragma unroll
        for (int ki = 0; ki < kSwBK / 32; ++ki) {
            int kA0 = ki * 32 + 4 * l, kA2 = kA0 + 16;
            uint32_t b0u = *reinterpret_cast<const uint32_t*>(
                &Us[stage * kSwTile + brow * kSwPad + kA0]);
            uint32_t b1u = *reinterpret_cast<const uint32_t*>(
                &Us[stage * kSwTile + brow * kSwPad + kA2]);
            uint32_t b0g = *reinterpret_cast<const uint32_t*>(
                &Gs[stage * kSwTile + brow * kSwPad + kA0]);
            uint32_t b1g = *reinterpret_cast<const uint32_t*>(
                &Gs[stage * kSwTile + brow * kSwPad + kA2]);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                int r0 = mi * 16 + h, r1 = r0 + 8;
                const uint8_t* abase = &As[stage * kSwTile];
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(
                    abase + r0 * kSwPad + kA0);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(
                    abase + r1 * kSwPad + kA0);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(
                    abase + r0 * kSwPad + kA2);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(
                    abase + r1 * kSwPad + kA2);
                sw_mma_fp8(acc_u[mi], a0, a1, a2, a3, b0u, b1u);
                sw_mma_fp8(acc_g[mi], a0, a1, a2, a3, b0g, b1g);
            }
        }
        __syncthreads();
        stage = (stage + 1) % kSwStages;
        is = (is + 1) % kSwStages;
        k_issue += kSwBK;
    }
    asm volatile("cp.async.wait_all;\n" ::);

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int row = mi * 16 + ((j < 2) ? h : h + 8);
            int col = n0 + warp * 8 + 2 * l + (j & 1);
            if (row < M && col < N) {
                float gv = b2f(f2b(acc_g[mi][j] * alpha_g));
                float uv = b2f(f2b(acc_u[mi][j] * alpha_u));
                if (bg != nullptr) gv = b2f(f2b(gv + b2f(bg[col])));
                if (bu != nullptr) uv = b2f(f2b(uv + b2f(bu[col])));
                const float silu = b2f(f2b(gv / (1.0f + expf(-gv))));
                out[(size_t)row * N + col] = f2b(silu * uv);
            }
        }
    }
}

int action_swiglu_fp8(const void* A, const void* Wu, const void* Wg,
                      const void* bu, const void* bg, void* out,
                      int M, int K, int N, float alpha_u, float alpha_g,
                      cudaStream_t stream) {
    if (M <= 0 || M > kSwRows || K <= 0 || N <= 0) return 1;
    if ((K % kSwBK) != 0 || (N % kSwN) != 0) return 2;
    const int smem = kSwStages * 3 * kSwTile;
    action_swiglu_kernel<<<N / kSwN, kSwThreads, smem, stream>>>(
        reinterpret_cast<const uint8_t*>(A),
        reinterpret_cast<const uint8_t*>(Wu),
        reinterpret_cast<const uint8_t*>(Wg),
        reinterpret_cast<const __nv_bfloat16*>(bu),
        reinterpret_cast<const __nv_bfloat16*>(bg),
        reinterpret_cast<__nv_bfloat16*>(out), M, K, N,
        alpha_u, alpha_g);
    return (int)cudaGetLastError();
}

}  // namespace molmoact2
}  // namespace flash_rt

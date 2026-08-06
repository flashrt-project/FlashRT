// ================================================================
// FlashRT — Fast Hadamard Transform + INT4 pack kernels (Orin SM87,
// FP16-backbone QuaRot W4A4/W8A8 paths).
//
// The activation side of the rotated GEMMs: x' = (x @ H_K) / sqrt(K),
// then per-row symmetric int4 (qmax=7), packed 2 elems/byte (low
// nibble = even index, matching cutlass::int4b_t sub-byte order).
// The matching weight rotation (per stored [N,K] row: row @ H_K /
// sqrt(K), then per-row int4) is done offline in the frontend.
//
// K == 4096 fast path: H_4096 = H16 (x) H16 (x) H16 — three radix-16
// register-resident butterfly stages over a padded fp32 smem row, only
// 3 __syncthreads() (the naive 12-stage smem butterfly measured 2.3 ms
// at M=1214; latency-bound on 12 barriers). Other pow-2 K falls back
// to the generic staged butterfly.
//
// Padded smem layout: addr(i) = i + (i >> 4)  (one pad float per 16)
// keeps the stride-16 (stage 2) accesses bank-conflict-free.
//
// Three call sites (mirroring the INT8 pipeline):
//   residual_add_rms_norm_fht_int4_fp16  layer boundaries (2x/layer)
//   rms_norm_fht_int4_fp16               L0 entry
//   fht_int4_quant_fp16                  pre-O (attention output)
// ================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

#include "common.cuh"

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ int pad_idx(int i) { return i + (i >> 4); }

__device__ __forceinline__ void h16_registers(float v[16]) {
    #pragma unroll
    for (int len = 1; len < 16; len <<= 1) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if ((i & len) == 0) {
                float a = v[i];
                float b = v[i + len];
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
    }
}

// In-place FHT over the padded smem row. K == 4096 uses the radix-16
// x3 fast path; other pow-2 K uses the generic staged butterfly.
__device__ __forceinline__ void fht_padded(float* s, int K, int tid) {
    if (K == 4096) {
        float v[16];
        // Stage 1: bits 0-3 (stride 1). Thread t owns rows [16t, 16t+16).
        {
            int base = tid * 16;
            #pragma unroll
            for (int c = 0; c < 16; ++c) v[c] = s[pad_idx(base + c)];
            h16_registers(v);
            #pragma unroll
            for (int c = 0; c < 16; ++c) s[pad_idx(base + c)] = v[c];
        }
        __syncthreads();
        // Stage 2: bits 4-7 (stride 16). Thread t owns (a = t>>4, c = t&15).
        {
            int base = (tid >> 4) * 256 + (tid & 15);
            #pragma unroll
            for (int b = 0; b < 16; ++b) v[b] = s[pad_idx(base + b * 16)];
            h16_registers(v);
            #pragma unroll
            for (int b = 0; b < 16; ++b) s[pad_idx(base + b * 16)] = v[b];
        }
        __syncthreads();
        // Stage 3: bits 8-11 (stride 256). Thread t owns (b = t>>4, c = t&15).
        {
            int base = (tid >> 4) * 16 + (tid & 15);
            #pragma unroll
            for (int a = 0; a < 16; ++a) v[a] = s[pad_idx(base + a * 256)];
            h16_registers(v);
            #pragma unroll
            for (int a = 0; a < 16; ++a) s[pad_idx(base + a * 256)] = v[a];
        }
        __syncthreads();
        return;
    }
    for (int len = 1; len < K; len <<= 1) {
        for (int idx = tid; idx < (K >> 1); idx += kThreads) {
            int i = ((idx / len) * (len << 1)) + (idx % len);
            float a = s[pad_idx(i)];
            float b = s[pad_idx(i + len)];
            s[pad_idx(i)] = a + b;
            s[pad_idx(i + len)] = a - b;
        }
        __syncthreads();
    }
}

// amax over the padded smem row + quantize to packed int4.
__device__ __forceinline__ void quant_pack_int4(
        const float* s, int K, int tid,
        float* partial, float inv_sqrt_k,
        uint8_t* out_row, float* scale_out) {
    float local_max = 0.f;
    for (int i = tid; i < K; i += kThreads)
        local_max = fmaxf(local_max, fabsf(s[pad_idx(i)]));
    float amax = block_reduce_max(local_max, partial);
    float scale_u = fmaxf(amax / 7.0f, 1e-10f);      // unnormalised domain
    if (tid == 0) *scale_out = scale_u * inv_sqrt_k; // fold 1/sqrt(K)
    float inv_s = 1.0f / scale_u;
    for (int j = tid; j < (K >> 1); j += kThreads) {
        int q0 = __float2int_rn(s[pad_idx(2 * j)] * inv_s);
        int q1 = __float2int_rn(s[pad_idx(2 * j + 1)] * inv_s);
        q0 = (q0 < -7) ? -7 : ((q0 > 7) ? 7 : q0);
        q1 = (q1 < -7) ? -7 : ((q1 > 7) ? 7 : q1);
        out_row[j] = static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
}

// amax over the padded smem row + quantize to int8 (one byte per element).
//
// The INT8 twin of quant_pack_int4: same amax reduction, same 1/sqrt(K)
// folding into the row scale, qmax 127 instead of 7 and no nibble packing.
// This is what lets the W8A8+Hadamard path feed the *unmodified*
// cutlass_int8_rowwise_* GEMMs — the reason for choosing a rotation that
// preserves plain per-row scales over a block-scaled scheme that would need
// a bespoke (and measured-slower) GEMM.
__device__ __forceinline__ void quant_int8(
        const float* s, int K, int tid,
        float* partial, float inv_sqrt_k,
        int8_t* out_row, float* scale_out) {
    float local_max = 0.f;
    for (int i = tid; i < K; i += kThreads)
        local_max = fmaxf(local_max, fabsf(s[pad_idx(i)]));
    float amax = block_reduce_max(local_max, partial);
    float scale_u = fmaxf(amax / 127.0f, 1e-12f);    // unnormalised domain
    if (tid == 0) *scale_out = scale_u * inv_sqrt_k; // fold 1/sqrt(K)
    float inv_s = 1.0f / scale_u;
    for (int i = tid; i < K; i += kThreads) {
        int q = __float2int_rn(s[pad_idx(i)] * inv_s);
        q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
        out_row[i] = static_cast<int8_t>(q);
    }
}

// Emit dispatch, so the norm+FHT kernels below are shared verbatim between
// the INT4 and INT8 activation paths (identical rotation, different width).
template <bool kInt8>
__device__ __forceinline__ void quant_emit(
        const float* s, int K, int tid, float* partial, float inv_sqrt_k,
        void* out_base, int64_t row, float* scale_out) {
    if (kInt8) {
        quant_int8(s, K, tid, partial, inv_sqrt_k,
                   static_cast<int8_t*>(out_base) + row * K, scale_out);
    } else {
        quant_pack_int4(s, K, tid, partial, inv_sqrt_k,
                        static_cast<uint8_t*>(out_base) + row * (K >> 1),
                        scale_out);
    }
}

// residual += x (fp16, written back); h = RMSNorm(residual)*w; FHT(h);
// int4 pack. Vectorised 16B loads for the fp16 streams.
template <bool kInt8>
__global__ void residual_add_rms_norm_fht_kernel(
        __half* __restrict__ residual,
        const __half* __restrict__ x,
        const __half* __restrict__ weight,
        void* __restrict__ out,          // int8 [rows,cols] | s4 [rows,cols/2]
        float* __restrict__ scales,      // [rows]
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols + (cols >> 4);

    int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int n8 = cols >> 3;

    uint4* res4 = reinterpret_cast<uint4*>(residual + (int64_t)row * cols);
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (int64_t)row * cols);
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);

    float sum_sq = 0.f;
    for (int j = tid; j < n8; j += kThreads) {
        uint4 rv = res4[j], xv = x4[j];
        __half2* rp = reinterpret_cast<__half2*>(&rv);
        const __half2* xp = reinterpret_cast<const __half2*>(&xv);
        int base = j << 3;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float r0 = __half2float(rp[k].x) + __half2float(xp[k].x);
            float r1 = __half2float(rp[k].y) + __half2float(xp[k].y);
            rp[k] = __halves2half2(__float2half(r0), __float2half(r1));
            smem[pad_idx(base + 2 * k)] = r0;
            smem[pad_idx(base + 2 * k + 1)] = r1;
            sum_sq += r0 * r0 + r1 * r1;
        }
        res4[j] = rv;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);

    for (int j = tid; j < n8; j += kThreads) {
        uint4 wv = w4[j];
        const __half2* wp = reinterpret_cast<const __half2*>(&wv);
        int base = j << 3;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            smem[pad_idx(base + 2 * k)]     *= rms * __half2float(wp[k].x);
            smem[pad_idx(base + 2 * k + 1)] *= rms * __half2float(wp[k].y);
        }
    }
    __syncthreads();

    fht_padded(smem, cols, tid);
    quant_emit<kInt8>(smem, cols, tid, partial, rsqrtf((float)cols),
                      out, (int64_t)row, scales + row);
}

// h = RMSNorm(x)*w; FHT; int4 pack (no residual update). L0 entry.
template <bool kInt8>
__global__ void rms_norm_fht_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ weight,
        void* __restrict__ out,
        float* __restrict__ scales,
        int rows, int cols, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols + (cols >> 4);
    int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int n8 = cols >> 3;
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (int64_t)row * cols);
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);

    float sum_sq = 0.f;
    for (int j = tid; j < n8; j += kThreads) {
        uint4 xv = x4[j];
        const __half2* xp = reinterpret_cast<const __half2*>(&xv);
        int base = j << 3;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float v0 = __half2float(xp[k].x), v1 = __half2float(xp[k].y);
            smem[pad_idx(base + 2 * k)] = v0;
            smem[pad_idx(base + 2 * k + 1)] = v1;
            sum_sq += v0 * v0 + v1 * v1;
        }
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);
    for (int j = tid; j < n8; j += kThreads) {
        uint4 wv = w4[j];
        const __half2* wp = reinterpret_cast<const __half2*>(&wv);
        int base = j << 3;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            smem[pad_idx(base + 2 * k)]     *= rms * __half2float(wp[k].x);
            smem[pad_idx(base + 2 * k + 1)] *= rms * __half2float(wp[k].y);
        }
    }
    __syncthreads();
    fht_padded(smem, cols, tid);
    quant_emit<kInt8>(smem, cols, tid, partial, rsqrtf((float)cols),
                      out, (int64_t)row, scales + row);
}

// FHT(x) + int4 pack, raw fp16 input (pre-O site).
template <bool kInt8>
__global__ void fht_quant_kernel(
        const __half* __restrict__ x,
        void* __restrict__ out,
        float* __restrict__ scales,
        int rows, int cols) {
    extern __shared__ float smem[];
    float* partial = smem + cols + (cols >> 4);
    int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int n8 = cols >> 3;
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (int64_t)row * cols);
    for (int j = tid; j < n8; j += kThreads) {
        uint4 xv = x4[j];
        const __half2* xp = reinterpret_cast<const __half2*>(&xv);
        int base = j << 3;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            smem[pad_idx(base + 2 * k)] = __half2float(xp[k].x);
            smem[pad_idx(base + 2 * k + 1)] = __half2float(xp[k].y);
        }
    }
    __syncthreads();
    fht_padded(smem, cols, tid);
    quant_emit<kInt8>(smem, cols, tid, partial, rsqrtf((float)cols),
                      out, (int64_t)row, scales + row);
}

inline int smem_bytes(int cols) {
    return (cols + (cols >> 4) + 32) * (int)sizeof(float);
}

}  // namespace

extern "C" void residual_add_rms_norm_fht_int4_fp16(
        __half* residual, const __half* x, const __half* weight,
        uint8_t* out, float* scales, int seq_len, int dim, float eps,
        cudaStream_t stream) {
    residual_add_rms_norm_fht_kernel<false>
        <<<seq_len, kThreads, smem_bytes(dim), stream>>>(
            residual, x, weight, out, scales, seq_len, dim, eps);
}

extern "C" void rms_norm_fht_int4_fp16(
        const __half* x, const __half* weight,
        uint8_t* out, float* scales, int seq_len, int dim, float eps,
        cudaStream_t stream) {
    rms_norm_fht_kernel<false><<<seq_len, kThreads, smem_bytes(dim), stream>>>(
        x, weight, out, scales, seq_len, dim, eps);
}

extern "C" void fht_int4_quant_fp16(
        const __half* x, uint8_t* out, float* scales,
        int seq_len, int dim, cudaStream_t stream) {
    fht_quant_kernel<false><<<seq_len, kThreads, smem_bytes(dim), stream>>>(
        x, out, scales, seq_len, dim);
}

// ── W8A8 + Hadamard (QuaRot at 8 bits) ──
// Identical rotation to the INT4 entries above, emitting int8 so the
// *unmodified* cutlass_int8_rowwise_* GEMMs consume it. Conditions the
// Chameleon massive-activation channels (which destroy plain per-row INT8)
// without paying INT4's quantization noise.

extern "C" void residual_add_rms_norm_fht_int8_fp16(
        __half* residual, const __half* x, const __half* weight,
        int8_t* out, float* scales, int seq_len, int dim, float eps,
        cudaStream_t stream) {
    residual_add_rms_norm_fht_kernel<true>
        <<<seq_len, kThreads, smem_bytes(dim), stream>>>(
            residual, x, weight, out, scales, seq_len, dim, eps);
}

extern "C" void rms_norm_fht_int8_fp16(
        const __half* x, const __half* weight,
        int8_t* out, float* scales, int seq_len, int dim, float eps,
        cudaStream_t stream) {
    rms_norm_fht_kernel<true><<<seq_len, kThreads, smem_bytes(dim), stream>>>(
        x, weight, out, scales, seq_len, dim, eps);
}

extern "C" void fht_int8_quant_fp16(
        const __half* x, int8_t* out, float* scales,
        int seq_len, int dim, cudaStream_t stream) {
    fht_quant_kernel<true><<<seq_len, kThreads, smem_bytes(dim), stream>>>(
        x, out, scales, seq_len, dim);
}

// ── Block-diagonal H_128 FHT + per-row int4 pack, BF16 input ──
// For the FFN down input (K = 11008 = 86 x 128, not a power of two).
// Each warp transforms 128-element chunks fully in registers: lane l
// holds elements [4l, 4l+3] of the chunk; stages len=1,2 are in-lane,
// len=4..64 are shfl_xor butterflies. The transformed row is kept in
// registers (MAX_CHUNKS per warp), amax-reduced across the block, then
// quantised and packed 2/byte. 1/sqrt(128) is folded into the scale.

namespace {

__device__ __forceinline__ void fht128_chunk(
        const __nv_bfloat16* __restrict__ xrow, int c, int lane,
        float& a0, float& a1, float& a2, float& a3) {
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(
        xrow + (c << 7) + (lane << 2));
    __nv_bfloat162 p0 = p[0], p1 = p[1];
    a0 = __bfloat162float(p0.x); a1 = __bfloat162float(p0.y);
    a2 = __bfloat162float(p1.x); a3 = __bfloat162float(p1.y);
    // len=1: (0,1) (2,3)
    float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
    // len=2: (0,2) (1,3)
    a0 = b0 + b2; a1 = b1 + b3; a2 = b0 - b2; a3 = b1 - b3;
    // len=4..64: cross-lane butterflies (branchless: lower lane a+o,
    // upper lane o-a  ==  fma(a, sgn, o)).
    #pragma unroll
    for (int xm = 1; xm <= 16; xm <<= 1) {
        float sgn = (lane & xm) ? -1.f : 1.f;
        float o0 = __shfl_xor_sync(0xffffffff, a0, xm);
        float o1 = __shfl_xor_sync(0xffffffff, a1, xm);
        float o2 = __shfl_xor_sync(0xffffffff, a2, xm);
        float o3 = __shfl_xor_sync(0xffffffff, a3, xm);
        a0 = fmaf(a0, sgn, o0);
        a1 = fmaf(a1, sgn, o1);
        a2 = fmaf(a2, sgn, o2);
        a3 = fmaf(a3, sgn, o3);
    }
}

// Single pass: shuffle-transform each 128-chunk once, park the fp32
// result in smem (44 KB at Dff=11008), block-amax, then quantize from
// smem. Halves the global traffic vs the recompute variant.
__global__ void fht128_int4_quant_bf16_kernel(
        const __nv_bfloat16* __restrict__ x,
        uint8_t* __restrict__ out,       // [rows, cols/2]
        float* __restrict__ scales,      // [rows]
        int rows, int cols) {
    extern __shared__ float srow[];      // [cols] transformed fp32
    __shared__ float partial[64];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = blockDim.x >> 5;
    const int nchunks = cols >> 7;

    const __nv_bfloat16* xrow = x + (int64_t)row * cols;
    uint8_t* orow = out + (int64_t)row * (cols >> 1);

    float local_max = 0.f;
    for (int c = warp; c < nchunks; c += nwarp) {
        float a0, a1, a2, a3;
        fht128_chunk(xrow, c, lane, a0, a1, a2, a3);
        float* sc4 = srow + (c << 7) + (lane << 2);
        sc4[0] = a0; sc4[1] = a1; sc4[2] = a2; sc4[3] = a3;
        local_max = fmaxf(local_max,
            fmaxf(fmaxf(fabsf(a0), fabsf(a1)), fmaxf(fabsf(a2), fabsf(a3))));
    }

    float amax = block_reduce_max(local_max, partial);
    float scale_u = fmaxf(amax / 7.0f, 1e-10f);
    if (tid == 0) scales[row] = scale_u * 0.08838834764831845f;  // 1/sqrt(128)
    float inv_s = 1.0f / scale_u;

    const int n4 = cols >> 2;
    for (int j = tid; j < n4; j += blockDim.x) {
        const float* s4 = srow + (j << 2);
        int q0 = __float2int_rn(s4[0] * inv_s);
        int q1 = __float2int_rn(s4[1] * inv_s);
        int q2 = __float2int_rn(s4[2] * inv_s);
        int q3 = __float2int_rn(s4[3] * inv_s);
        q0 = (q0 < -7) ? -7 : ((q0 > 7) ? 7 : q0);
        q1 = (q1 < -7) ? -7 : ((q1 > 7) ? 7 : q1);
        q2 = (q2 < -7) ? -7 : ((q2 > 7) ? 7 : q2);
        q3 = (q3 < -7) ? -7 : ((q3 > 7) ? 7 : q3);
        uint16_t pk = (uint16_t)((q0 & 0xF) | ((q1 & 0xF) << 4)
                    | ((q2 & 0xF) << 8) | ((q3 & 0xF) << 12));
        *reinterpret_cast<uint16_t*>(&orow[j << 1]) = pk;
    }
}

}  // namespace

extern "C" void fht128_int4_quant_bf16(
        const __nv_bfloat16* x, uint8_t* out, float* scales,
        int seq_len, int dim, cudaStream_t stream) {
    int smem = dim * (int)sizeof(float);
    static bool attr_set = false;
    if (!attr_set && smem > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fht128_int4_quant_bf16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    fht128_int4_quant_bf16_kernel<<<seq_len, 512, smem, stream>>>(
        x, out, scales, seq_len, dim);
}

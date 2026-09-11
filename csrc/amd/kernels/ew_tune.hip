// ================================================================
// FlashRT AMD — elementwise launch-geometry tuning probe (CDNA4)
// See ew_tune.h for the contract. Variant 0 of each family is the
// production geometry replicated verbatim (control arm).
// ================================================================

#include "common_hip.h"
#include "ew_tune.h"
#include <hip/hip_fp8.h>
#include <stdexcept>
#include <string>

namespace {

// 16-byte packet = 8 bf16 (one dwordx4) — same pattern as the
// attention kernels' attn_bf16x8.
struct alignas(16) ew_bf16x8 {
    __hip_bfloat162 p[4];
};

// 8-byte packet = 8 FP8 bytes (one dwordx2 store).
struct alignas(8) ew_fp8x8 {
    unsigned char b[8];
};

__device__ __forceinline__ unsigned char fp8_byte(float v) {
    return __hip_fp8_e4m3(fminf(fmaxf(v, -448.0f), 448.0f)).__x;
}

// Butterfly all-reduce: every lane ends with the full wave sum.
__device__ __forceinline__ float wave_allreduce_sum(float val) {
    #pragma unroll
    for (int off = FVK_WAVE_SIZE / 2; off > 0; off >>= 1)
        val += __shfl_xor(val, off);
    return val;
}

// ────────────────────────────────────────────────────────────────
// Family Q: flat FP8 quantize
// ────────────────────────────────────────────────────────────────

// q0 — production geometry: one bf16x2 per thread, two byte stores.
__global__ void ew_quant_legacy_kernel(const __hip_bfloat16* __restrict__ input,
                                       __hip_fp8_e4m3* __restrict__ output,
                                       const float* __restrict__ scale, int n) {
    const __hip_bfloat162* in2 = reinterpret_cast<const __hip_bfloat162*>(input);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        float inv_s = 1.0f / (*scale);
        __hip_bfloat162 val = in2[idx];
        float v0 = __bfloat162float(val.x) * inv_s;
        float v1 = __bfloat162float(val.y) * inv_s;
        output[2*idx]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        output[2*idx+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

// q1 — legacy geometry, but the two FP8 bytes packed into one u16 store.
__global__ void ew_quant_pack16_kernel(const __hip_bfloat16* __restrict__ input,
                                       unsigned short* __restrict__ output,
                                       const float* __restrict__ scale, int n) {
    const __hip_bfloat162* in2 = reinterpret_cast<const __hip_bfloat162*>(input);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        float inv_s = 1.0f / (*scale);
        __hip_bfloat162 val = in2[idx];
        float v0 = __bfloat162float(val.x) * inv_s;
        float v1 = __bfloat162float(val.y) * inv_s;
        unsigned short lo = fp8_byte(v0);
        unsigned short hi = fp8_byte(v1);
        output[idx] = (unsigned short)(lo | (hi << 8));
    }
}

// q2..q6 — wide: dwordx4 load (8 bf16), dwordx2 store (8 FP8), ILP chunks.
template<int ILP>
__global__ void ew_quant_wide_kernel(const ew_bf16x8* __restrict__ in,
                                     ew_fp8x8* __restrict__ out,
                                     const float* __restrict__ scale,
                                     int nchunks, int stride) {
    int base = blockIdx.x * blockDim.x + threadIdx.x;
    ew_bf16x8 raw[ILP];
    bool valid[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
        int idx = base + j * stride;
        valid[j] = idx < nchunks;
        if (valid[j]) raw[j] = in[idx];
    }
    float inv_s = 1.0f / (*scale);
    #pragma unroll
    for (int j = 0; j < ILP; ++j) {
        if (!valid[j]) continue;
        ew_fp8x8 o;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            o.b[2*k]   = fp8_byte(__bfloat162float(raw[j].p[k].x) * inv_s);
            o.b[2*k+1] = fp8_byte(__bfloat162float(raw[j].p[k].y) * inv_s);
        }
        out[base + j * stride] = o;
    }
}

struct QuantVariant { const char* name; int ilp; int block; };
const QuantVariant kQuantVariants[] = {
    {"q0_legacy_t2_bytestore_b256", 0, 256},
    {"q1_legacy_t2_pack16_b256",    0, 256},
    {"q2_wide8_ilp1_b256",          1, 256},
    {"q3_wide8_ilp2_b256",          2, 256},
    {"q4_wide8_ilp4_b256",          4, 256},
    {"q5_wide8_ilp1_b64",           1, 64},
    {"q6_wide8_ilp1_b1024",         1, 1024},
};

} // namespace

int ew_tune_quant_variant_count() { return (int)(sizeof(kQuantVariants) / sizeof(kQuantVariants[0])); }
const char* ew_tune_quant_variant_name(int id) { return kQuantVariants[id].name; }

void ew_tune_quant(int variant, const __hip_bfloat16* in, __hip_fp8_e4m3* out,
                   const float* d_scale, int n, hipStream_t stream) {
    if (variant < 0 || variant >= ew_tune_quant_variant_count())
        throw std::runtime_error("ew_tune_quant: bad variant " + std::to_string(variant));
    const QuantVariant& v = kQuantVariants[variant];
    if (variant == 0) {
        int n2 = n >> 1;
        ew_quant_legacy_kernel<<<(n2 + v.block - 1) / v.block, v.block, 0, stream>>>(
            in, out, d_scale, n);
        return;
    }
    if (variant == 1) {
        int n2 = n >> 1;
        ew_quant_pack16_kernel<<<(n2 + v.block - 1) / v.block, v.block, 0, stream>>>(
            in, reinterpret_cast<unsigned short*>(out), d_scale, n);
        return;
    }
    if (n % 8 != 0 || ((uintptr_t)in & 15) || ((uintptr_t)out & 7))
        throw std::runtime_error("ew_tune_quant: wide variant needs n%8==0 and aligned ptrs");
    int nchunks = n >> 3;
    int stride = (nchunks + v.ilp - 1) / v.ilp;   // threads needed
    int blocks = (stride + v.block - 1) / v.block;
    const ew_bf16x8* in8 = reinterpret_cast<const ew_bf16x8*>(in);
    ew_fp8x8* out8 = reinterpret_cast<ew_fp8x8*>(out);
    switch (v.ilp) {
    case 1: ew_quant_wide_kernel<1><<<blocks, v.block, 0, stream>>>(in8, out8, d_scale, nchunks, stride); break;
    case 2: ew_quant_wide_kernel<2><<<blocks, v.block, 0, stream>>>(in8, out8, d_scale, nchunks, stride); break;
    case 4: ew_quant_wide_kernel<4><<<blocks, v.block, 0, stream>>>(in8, out8, d_scale, nchunks, stride); break;
    }
}

// ────────────────────────────────────────────────────────────────
// Family N: fused gate*residual + AdaRMSNorm + style → FP8
// ────────────────────────────────────────────────────────────────

namespace {

// n0 — production geometry: row-per-block 256T, bf16x2 accesses,
// LDS block reduce, byte FP8 stores. Verbatim copy of fusion.hip.
__global__ void ew_norm_legacy_kernel(
    __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ gate,
    const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ style,
    __hip_fp8_e4m3* __restrict__ out,
    __hip_bfloat16* __restrict__ gate_out,
    int dim, float eps,
    const float* __restrict__ d_scale) {
    using T = __hip_bfloat16;
    using T2 = __hip_bfloat162;
    int row = blockIdx.x;
    T2* res2 = reinterpret_cast<T2*>(residual + row * dim);
    const T2* x2 = reinterpret_cast<const T2*>(x + row * dim);
    const T2* g2 = reinterpret_cast<const T2*>(gate + row * dim);
    const T2* w2 = reinterpret_cast<const T2*>(weight);
    const T* style_row = style + row * 3 * dim;
    const T2* sc2 = reinterpret_cast<const T2*>(style_row);
    const T2* sh2 = reinterpret_cast<const T2*>(style_row + dim);
    const T2* gt2 = reinterpret_cast<const T2*>(style_row + 2 * dim);
    __hip_fp8_e4m3* out_row = out + row * dim;
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
    float inv_scale = 1.0f / (*d_scale);

    for (int i = threadIdx.x; i < dim2; i += blockDim.x) {
        T2 rv = res2[i], wv = w2[i];
        T2 sv = sc2[i], hv = sh2[i], gv = gt2[i];
        float n0 = to_f32(rv.x) * rms * to_f32(wv.x);
        float n1 = to_f32(rv.y) * rms * to_f32(wv.y);
        float val0 = (n0 * (1.0f + to_f32(sv.x)) + to_f32(hv.x)) * inv_scale;
        float val1 = (n1 * (1.0f + to_f32(sv.y)) + to_f32(hv.y)) * inv_scale;
        out_row[2*i]   = __hip_fp8_e4m3(fminf(fmaxf(val0, -448.0f), 448.0f));
        out_row[2*i+1] = __hip_fp8_e4m3(fminf(fmaxf(val1, -448.0f), 448.0f));
        gate_out2[i] = gv;
    }
}

// Wave-per-row core: one wave64 owns one full row. Wide dwordx4
// accesses, packed FP8 stores, single-wave butterfly reduce — no LDS,
// no __syncthreads. Pass 1 keeps the bf16-rounded residual in
// registers so pass 2 never re-reads it (bit-identical to the
// re-read: the register value IS the rounded value that was stored).
// MAXC * 64 lanes * 8 values = dim <= 4096.
constexpr int EW_NORM_MAXC = 8;

__device__ __forceinline__ void ew_norm_row_wave(
    int row, int lane,
    __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ gate,
    const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ style,
    __hip_fp8_e4m3* __restrict__ out,
    __hip_bfloat16* __restrict__ gate_out,
    int dim, float eps, float inv_scale) {
    int chunks = dim >> 3;
    ew_bf16x8* res8 = reinterpret_cast<ew_bf16x8*>(residual + (size_t)row * dim);
    const ew_bf16x8* x8 = reinterpret_cast<const ew_bf16x8*>(x + (size_t)row * dim);
    const ew_bf16x8* g8 = reinterpret_cast<const ew_bf16x8*>(gate + (size_t)row * dim);
    const ew_bf16x8* w8 = reinterpret_cast<const ew_bf16x8*>(weight);
    const __hip_bfloat16* style_row = style + (size_t)row * 3 * dim;
    const ew_bf16x8* sc8 = reinterpret_cast<const ew_bf16x8*>(style_row);
    const ew_bf16x8* sh8 = reinterpret_cast<const ew_bf16x8*>(style_row + dim);
    const ew_bf16x8* gt8 = reinterpret_cast<const ew_bf16x8*>(style_row + 2 * dim);
    ew_fp8x8* out8 = reinterpret_cast<ew_fp8x8*>(out + (size_t)row * dim);
    ew_bf16x8* gate_out8 = reinterpret_cast<ew_bf16x8*>(gate_out + (size_t)row * dim);

    // Pass 1: residual += x * gate (rounded to bf16, stored AND kept in
    // registers); accumulate sum of squares of the unrounded values.
    ew_bf16x8 rres[EW_NORM_MAXC];
    int cidx[EW_NORM_MAXC];
    int cnt = 0;
    float local_sum = 0.0f;
    for (int c = lane; c < chunks; c += FVK_WAVE_SIZE) {
        ew_bf16x8 rv = res8[c];
        ew_bf16x8 xv = x8[c];
        ew_bf16x8 gv = g8[c];
        ew_bf16x8 rounded;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float r0 = to_f32(rv.p[k].x) + to_f32(xv.p[k].x) * to_f32(gv.p[k].x);
            float r1 = to_f32(rv.p[k].y) + to_f32(xv.p[k].y) * to_f32(gv.p[k].y);
            rounded.p[k] = make_packed2<__hip_bfloat16>(
                from_f32<__hip_bfloat16>(r0), from_f32<__hip_bfloat16>(r1));
            local_sum += r0 * r0 + r1 * r1;
        }
        res8[c] = rounded;
        rres[cnt] = rounded;
        cidx[cnt] = c;
        ++cnt;
    }
    float total = wave_allreduce_sum(local_sum);
    float rms = rsqrtf(total / dim + eps);

    // Pass 2: AdaRMSNorm with style → packed FP8 + gate copy.
    for (int j = 0; j < cnt; ++j) {
        int c = cidx[j];
        ew_bf16x8 wv = w8[c];
        ew_bf16x8 sv = sc8[c];
        ew_bf16x8 hv = sh8[c];
        ew_bf16x8 gv = gt8[c];
        ew_fp8x8 o;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float n0 = to_f32(rres[j].p[k].x) * rms * to_f32(wv.p[k].x);
            float n1 = to_f32(rres[j].p[k].y) * rms * to_f32(wv.p[k].y);
            float v0 = (n0 * (1.0f + to_f32(sv.p[k].x)) + to_f32(hv.p[k].x)) * inv_scale;
            float v1 = (n1 * (1.0f + to_f32(sv.p[k].y)) + to_f32(hv.p[k].y)) * inv_scale;
            o.b[2*k]   = fp8_byte(v0);
            o.b[2*k+1] = fp8_byte(v1);
        }
        out8[c] = o;
        gate_out8[c] = gv;
    }
}

// n2 — one 64-thread WG per row.
__global__ void ew_norm_waverow_kernel(
    __hip_bfloat16* __restrict__ residual, const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ gate, const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ style, __hip_fp8_e4m3* __restrict__ out,
    __hip_bfloat16* __restrict__ gate_out, int dim, float eps,
    const float* __restrict__ d_scale) {
    float inv_scale = 1.0f / (*d_scale);
    ew_norm_row_wave(blockIdx.x, threadIdx.x, residual, x, gate, weight, style,
                     out, gate_out, dim, eps, inv_scale);
}

// n3 — 256-thread WG = 4 independent waves, one row per wave, no sync.
__global__ void ew_norm_wave4row_kernel(
    __hip_bfloat16* __restrict__ residual, const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ gate, const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ style, __hip_fp8_e4m3* __restrict__ out,
    __hip_bfloat16* __restrict__ gate_out, int seq_len, int dim, float eps,
    const float* __restrict__ d_scale) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= seq_len) return;
    float inv_scale = 1.0f / (*d_scale);
    ew_norm_row_wave(row, threadIdx.x & (FVK_WAVE_SIZE - 1), residual, x, gate,
                     weight, style, out, gate_out, dim, eps, inv_scale);
}

// n1/n4 — row-per-block with wide accesses and LDS block reduce
// (isolates the access-width axis from the reduction-geometry axis).
__global__ void ew_norm_wideblock_kernel(
    __hip_bfloat16* __restrict__ residual, const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ gate, const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ style, __hip_fp8_e4m3* __restrict__ out,
    __hip_bfloat16* __restrict__ gate_out, int dim, float eps,
    const float* __restrict__ d_scale) {
    int row = blockIdx.x;
    int chunks = dim >> 3;
    ew_bf16x8* res8 = reinterpret_cast<ew_bf16x8*>(residual + (size_t)row * dim);
    const ew_bf16x8* x8 = reinterpret_cast<const ew_bf16x8*>(x + (size_t)row * dim);
    const ew_bf16x8* g8 = reinterpret_cast<const ew_bf16x8*>(gate + (size_t)row * dim);
    const ew_bf16x8* w8 = reinterpret_cast<const ew_bf16x8*>(weight);
    const __hip_bfloat16* style_row = style + (size_t)row * 3 * dim;
    const ew_bf16x8* sc8 = reinterpret_cast<const ew_bf16x8*>(style_row);
    const ew_bf16x8* sh8 = reinterpret_cast<const ew_bf16x8*>(style_row + dim);
    const ew_bf16x8* gt8 = reinterpret_cast<const ew_bf16x8*>(style_row + 2 * dim);
    ew_fp8x8* out8 = reinterpret_cast<ew_fp8x8*>(out + (size_t)row * dim);
    ew_bf16x8* gate_out8 = reinterpret_cast<ew_bf16x8*>(gate_out + (size_t)row * dim);

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        ew_bf16x8 rv = res8[c];
        ew_bf16x8 xv = x8[c];
        ew_bf16x8 gv = g8[c];
        ew_bf16x8 rounded;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float r0 = to_f32(rv.p[k].x) + to_f32(xv.p[k].x) * to_f32(gv.p[k].x);
            float r1 = to_f32(rv.p[k].y) + to_f32(xv.p[k].y) * to_f32(gv.p[k].y);
            rounded.p[k] = make_packed2<__hip_bfloat16>(
                from_f32<__hip_bfloat16>(r0), from_f32<__hip_bfloat16>(r1));
            local_sum += r0 * r0 + r1 * r1;
        }
        res8[c] = rounded;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);
    float inv_scale = 1.0f / (*d_scale);

    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        ew_bf16x8 rv = res8[c];
        ew_bf16x8 wv = w8[c];
        ew_bf16x8 sv = sc8[c];
        ew_bf16x8 hv = sh8[c];
        ew_bf16x8 gv = gt8[c];
        ew_fp8x8 o;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float n0 = to_f32(rv.p[k].x) * rms * to_f32(wv.p[k].x);
            float n1 = to_f32(rv.p[k].y) * rms * to_f32(wv.p[k].y);
            float v0 = (n0 * (1.0f + to_f32(sv.p[k].x)) + to_f32(hv.p[k].x)) * inv_scale;
            float v1 = (n1 * (1.0f + to_f32(sv.p[k].y)) + to_f32(hv.p[k].y)) * inv_scale;
            o.b[2*k]   = fp8_byte(v0);
            o.b[2*k+1] = fp8_byte(v1);
        }
        out8[c] = o;
        gate_out8[c] = gv;
    }
}

// n5 — wide row-per-block b256 with PREFETCHED side inputs: the
// weight/style streams are only consumed in pass 2, but in production
// they are cold (evicted by ~40MB of weight streaming between uses)
// and their HBM round trip lands on the critical path after the
// reduce. Issuing those loads FIRST hides the cold latency behind
// pass 1's compute + reduce. Requires dim <= 8*8*blockDim... uses the
// register budget: chunks_per_thread <= EW_NORM_MAXC.
__global__ void ew_norm_wideblock_prefetch_kernel(
    __hip_bfloat16* __restrict__ residual, const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ gate, const __hip_bfloat16* __restrict__ weight,
    const __hip_bfloat16* __restrict__ style, __hip_fp8_e4m3* __restrict__ out,
    __hip_bfloat16* __restrict__ gate_out, int dim, float eps,
    const float* __restrict__ d_scale) {
    int row = blockIdx.x;
    int chunks = dim >> 3;
    ew_bf16x8* res8 = reinterpret_cast<ew_bf16x8*>(residual + (size_t)row * dim);
    const ew_bf16x8* x8 = reinterpret_cast<const ew_bf16x8*>(x + (size_t)row * dim);
    const ew_bf16x8* g8 = reinterpret_cast<const ew_bf16x8*>(gate + (size_t)row * dim);
    const ew_bf16x8* w8 = reinterpret_cast<const ew_bf16x8*>(weight);
    const __hip_bfloat16* style_row = style + (size_t)row * 3 * dim;
    const ew_bf16x8* sc8 = reinterpret_cast<const ew_bf16x8*>(style_row);
    const ew_bf16x8* sh8 = reinterpret_cast<const ew_bf16x8*>(style_row + dim);
    const ew_bf16x8* gt8 = reinterpret_cast<const ew_bf16x8*>(style_row + 2 * dim);
    ew_fp8x8* out8 = reinterpret_cast<ew_fp8x8*>(out + (size_t)row * dim);
    ew_bf16x8* gate_out8 = reinterpret_cast<ew_bf16x8*>(gate_out + (size_t)row * dim);

    // Prefetch every pass-2 stream (cold in production) up front, plus
    // the scalar scale — all independent of pass 1.
    ew_bf16x8 pw[EW_NORM_MAXC], psc[EW_NORM_MAXC], psh[EW_NORM_MAXC], pgt[EW_NORM_MAXC];
    int cidx[EW_NORM_MAXC];
    int cnt = 0;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        pw[cnt] = w8[c]; psc[cnt] = sc8[c]; psh[cnt] = sh8[c]; pgt[cnt] = gt8[c];
        cidx[cnt] = c;
        ++cnt;
    }
    float inv_scale = 1.0f / (*d_scale);

    extern __shared__ float shared[];
    float local_sum = 0.0f;
    ew_bf16x8 pres[EW_NORM_MAXC];
    for (int j = 0; j < cnt; ++j) {
        int c = cidx[j];
        ew_bf16x8 rv = res8[c];
        ew_bf16x8 xv = x8[c];
        ew_bf16x8 gv = g8[c];
        ew_bf16x8 rounded;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float r0 = to_f32(rv.p[k].x) + to_f32(xv.p[k].x) * to_f32(gv.p[k].x);
            float r1 = to_f32(rv.p[k].y) + to_f32(xv.p[k].y) * to_f32(gv.p[k].y);
            rounded.p[k] = make_packed2<__hip_bfloat16>(
                from_f32<__hip_bfloat16>(r0), from_f32<__hip_bfloat16>(r1));
            local_sum += r0 * r0 + r1 * r1;
        }
        res8[c] = rounded;
        pres[j] = rounded;
    }
    float rms = rsqrtf(block_reduce_sum(local_sum, shared) / dim + eps);

    for (int j = 0; j < cnt; ++j) {
        int c = cidx[j];
        ew_fp8x8 o;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float n0 = to_f32(pres[j].p[k].x) * rms * to_f32(pw[j].p[k].x);
            float n1 = to_f32(pres[j].p[k].y) * rms * to_f32(pw[j].p[k].y);
            float v0 = (n0 * (1.0f + to_f32(psc[j].p[k].x)) + to_f32(psh[j].p[k].x)) * inv_scale;
            float v1 = (n1 * (1.0f + to_f32(psc[j].p[k].y)) + to_f32(psh[j].p[k].y)) * inv_scale;
            o.b[2*k]   = fp8_byte(v0);
            o.b[2*k+1] = fp8_byte(v1);
        }
        out8[c] = o;
        gate_out8[c] = pgt[j];
    }
}

struct NormVariant { const char* name; int kind; int block; };
// kind: 0 legacy, 1 wideblock, 2 waverow, 3 wave4row, 4 wideblock+prefetch
const NormVariant kNormVariants[] = {
    {"n0_legacy_rowblock_b256",  0, 256},
    {"n1_wide_rowblock_b128",    1, 128},
    {"n2_wide_waverow_b64",      2, 64},
    {"n3_wide_wave4row_b256",    3, 256},
    {"n4_wide_rowblock_b256",    1, 256},
    {"n5_wide_prefetch_b256",    4, 256},
    {"n6_wide_prefetch_b128",    4, 128},
};

} // namespace

int ew_tune_norm_variant_count() { return (int)(sizeof(kNormVariants) / sizeof(kNormVariants[0])); }
const char* ew_tune_norm_variant_name(int id) { return kNormVariants[id].name; }

void ew_tune_norm(int variant, __hip_bfloat16* residual, const __hip_bfloat16* x,
                  const __hip_bfloat16* gate, const __hip_bfloat16* weight,
                  const __hip_bfloat16* style, __hip_fp8_e4m3* out,
                  __hip_bfloat16* gate_out, int seq_len, int dim, float eps,
                  const float* d_scale, hipStream_t stream) {
    if (variant < 0 || variant >= ew_tune_norm_variant_count())
        throw std::runtime_error("ew_tune_norm: bad variant " + std::to_string(variant));
    const NormVariant& v = kNormVariants[variant];
    if (v.kind == 0) {
        ew_norm_legacy_kernel<<<seq_len, v.block, 4 * sizeof(float), stream>>>(
            residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
        return;
    }
    if (dim % 8 != 0)
        throw std::runtime_error("ew_tune_norm: wide variant needs dim%8==0");
    if (v.kind == 1) {
        int smem = ((v.block + 63) / 64) * sizeof(float);
        ew_norm_wideblock_kernel<<<seq_len, v.block, smem, stream>>>(
            residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
        return;
    }
    if (v.kind == 4) {
        if (dim > v.block * 8 * EW_NORM_MAXC)
            throw std::runtime_error("ew_tune_norm: prefetch variant dim too large");
        int smem = ((v.block + 63) / 64) * sizeof(float);
        ew_norm_wideblock_prefetch_kernel<<<seq_len, v.block, smem, stream>>>(
            residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
        return;
    }
    if (dim > FVK_WAVE_SIZE * 8 * EW_NORM_MAXC)
        throw std::runtime_error("ew_tune_norm: wave variant needs dim<=4096");
    if (v.kind == 2) {
        ew_norm_waverow_kernel<<<seq_len, 64, 0, stream>>>(
            residual, x, gate, weight, style, out, gate_out, dim, eps, d_scale);
    } else {
        ew_norm_wave4row_kernel<<<(seq_len + 3) / 4, 256, 0, stream>>>(
            residual, x, gate, weight, style, out, gate_out, seq_len, dim, eps, d_scale);
    }
}

// ────────────────────────────────────────────────────────────────
// Family R: fused QKV split + RoPE
// ────────────────────────────────────────────────────────────────

namespace {

// r0 — production geometry: one element per thread, flat 1D grid,
// per-thread div/mod row decomposition. Verbatim copy of rope.hip.
__global__ void ew_rope_legacy_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ rope_weights,
    __hip_bfloat16* __restrict__ Q,
    __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ V,
    int seq, int q_dim, int k_dim, int v_dim, int head_dim) {
    int qkv_dim = q_dim + k_dim + v_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * qkv_dim;
    if (idx >= total) return;

    int row = idx / qkv_dim;
    int col = idx % qkv_dim;

    if (col < q_dim) {
        int q_col = col;
        int head = q_col / head_dim;
        int d_in_head = q_col % head_dim;
        int pair = d_in_head / 2;
        int is_odd = d_in_head & 1;
        int pair_base_qkv = row * qkv_dim + head * head_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);
        int out_idx = row * q_dim + q_col;
        if (is_odd == 0) Q[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        else             Q[out_idx] = __float2bfloat16(x1 * c + x0 * s);
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        int pair = k_col / 2;
        int is_odd = k_col & 1;
        int pair_base_qkv = row * qkv_dim + q_dim + pair * 2;
        float x0 = __bfloat162float(qkv[pair_base_qkv]);
        float x1 = __bfloat162float(qkv[pair_base_qkv + 1]);
        int rope_base = row * head_dim + pair * 2;
        float c = __bfloat162float(rope_weights[rope_base]);
        float s = __bfloat162float(rope_weights[rope_base + 1]);
        int out_idx = row * k_dim + k_col;
        if (is_odd == 0) K[out_idx] = __float2bfloat16(x0 * c - x1 * s);
        else             K[out_idx] = __float2bfloat16(x1 * c + x0 * s);
    } else {
        int v_col = col - q_dim - k_dim;
        V[row * v_dim + v_col] = qkv[idx];
    }
}

// r1/r2 — one rotation pair per thread, 2D grid (blockIdx.y = row):
// no div/mod for the row, one dword load + one dword store per thread,
// each thread computes BOTH rotated outputs of its pair (identical
// float math to the two legacy threads → bit-exact).
__global__ void ew_rope_pair_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const __hip_bfloat16* __restrict__ rope_weights,
    __hip_bfloat16* __restrict__ Q,
    __hip_bfloat16* __restrict__ K,
    __hip_bfloat16* __restrict__ V,
    int q_dim, int k_dim, int v_dim, int head_dim, int qkv_dim) {
    int row = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;   // pair index within row
    int npairs = qkv_dim >> 1;
    if (p >= npairs) return;
    int col = p << 1;

    const __hip_bfloat162* src2 =
        reinterpret_cast<const __hip_bfloat162*>(qkv + (size_t)row * qkv_dim);
    __hip_bfloat162 xv = src2[p];

    if (col < q_dim) {
        int d_in_head = col % head_dim;   // even by construction
        __hip_bfloat162 cs = reinterpret_cast<const __hip_bfloat162*>(
            rope_weights)[((size_t)row * head_dim + d_in_head) >> 1];
        float x0 = __bfloat162float(xv.x), x1 = __bfloat162float(xv.y);
        float c = __bfloat162float(cs.x), s = __bfloat162float(cs.y);
        reinterpret_cast<__hip_bfloat162*>(Q)[((size_t)row * q_dim + col) >> 1] =
            make_packed2<__hip_bfloat16>(
                __float2bfloat16(x0 * c - x1 * s),
                __float2bfloat16(x1 * c + x0 * s));
    } else if (col < q_dim + k_dim) {
        int k_col = col - q_dim;
        __hip_bfloat162 cs = reinterpret_cast<const __hip_bfloat162*>(
            rope_weights)[((size_t)row * head_dim + k_col) >> 1];
        float x0 = __bfloat162float(xv.x), x1 = __bfloat162float(xv.y);
        float c = __bfloat162float(cs.x), s = __bfloat162float(cs.y);
        reinterpret_cast<__hip_bfloat162*>(K)[((size_t)row * k_dim + k_col) >> 1] =
            make_packed2<__hip_bfloat16>(
                __float2bfloat16(x0 * c - x1 * s),
                __float2bfloat16(x1 * c + x0 * s));
    } else {
        int v_col = col - q_dim - k_dim;
        reinterpret_cast<__hip_bfloat162*>(V)[((size_t)row * v_dim + v_col) >> 1] = xv;
    }
}

struct RopeVariant { const char* name; int kind; int block; };
// kind: 0 legacy flat, 1 pair-per-thread 2D
const RopeVariant kRopeVariants[] = {
    {"r0_legacy_elem_flat_b256", 0, 256},
    {"r1_pair_2d_b256",          1, 256},
    {"r2_pair_2d_b64",           1, 64},
};

} // namespace

int ew_tune_rope_variant_count() { return (int)(sizeof(kRopeVariants) / sizeof(kRopeVariants[0])); }
const char* ew_tune_rope_variant_name(int id) { return kRopeVariants[id].name; }

void ew_tune_rope(int variant, const __hip_bfloat16* qkv,
                  const __hip_bfloat16* rope_weights,
                  __hip_bfloat16* Q, __hip_bfloat16* K, __hip_bfloat16* V,
                  int seq, int q_dim, int k_dim, int v_dim, int head_dim,
                  hipStream_t stream) {
    if (variant < 0 || variant >= ew_tune_rope_variant_count())
        throw std::runtime_error("ew_tune_rope: bad variant " + std::to_string(variant));
    const RopeVariant& v = kRopeVariants[variant];
    int qkv_dim = q_dim + k_dim + v_dim;
    if (v.kind == 0) {
        int total = seq * qkv_dim;
        ew_rope_legacy_kernel<<<(total + v.block - 1) / v.block, v.block, 0, stream>>>(
            qkv, rope_weights, Q, K, V, seq, q_dim, k_dim, v_dim, head_dim);
        return;
    }
    if ((q_dim | k_dim | v_dim | head_dim) & 1)
        throw std::runtime_error("ew_tune_rope: pair variant needs even dims");
    int npairs = qkv_dim >> 1;
    dim3 grid((npairs + v.block - 1) / v.block, seq);
    ew_rope_pair_kernel<<<grid, v.block, 0, stream>>>(
        qkv, rope_weights, Q, K, V, q_dim, k_dim, v_dim, head_dim, qkv_dim);
}

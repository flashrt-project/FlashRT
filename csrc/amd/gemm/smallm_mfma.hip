// ================================================================
// FlashRT AMD — MFMA small-M FP8 GEMM (gfx950, wave64)
// See smallm_mfma.h for the contract and design rationale.
//
// Fragment layout of V_MFMA_F32_16X16X32_FP8_FP8 (verified by the
// parity bench, cos=1.0 on all pi05 shapes):
//   a (i64): lane l supplies Amat[m = l&15][k = 32*step + 8*(l>>4) ..+8]
//   b (i64): lane l supplies Bmat[n = l&15][same k window]
//            for Bmat stored row-major (16 rows, K cols)
//   c (f32x4): lane l element i = C[m = 4*(l>>4)+i][n = l&15]
//
// WAVES is the K-split factor inside one workgroup: the narrow-N
// decoder shapes (N=1024 -> only 64 workgroups) cannot reach the
// ~4MB GPU-wide in-flight target with 4 waves; deeper wave counts
// multiply outstanding loads per CU at unchanged grid.
// ================================================================

#include "smallm_mfma.h"
#include <hip/hip_fp8.h>
#include <stdexcept>
#include <string>

namespace {

typedef float floatx4 __attribute__((ext_vector_type(4)));

// S = MFMA steps per wave segment (kseg = 32*S = K/WAVES).
template<int S, int WAVES>
__global__ __launch_bounds__(64 * WAVES)
void smallm_mfma_nt_kernel(
    const unsigned char* __restrict__ A,   // (M,K) fp8 e4m3
    const unsigned char* __restrict__ W,   // (N,K) fp8 e4m3
    __hip_bfloat16* __restrict__ D,        // (M,N) bf16
    int M, int N, int K,
    const float* __restrict__ dsa, const float* __restrict__ dsb) {
    extern __shared__ unsigned char lds[];
    unsigned char* a_lds = lds;                        // M rows x K bytes
    float* red = reinterpret_cast<float*>(lds + (size_t)M * K);

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n0 = blockIdx.x * 16;

    // ── Stage A (M x K fp8) into LDS, cooperative dwordx4 ──
    {
        const uint4* src = reinterpret_cast<const uint4*>(A);
        uint4* dst = reinterpret_cast<uint4*>(a_lds);
        int chunks = (M * K) >> 4;
        for (int c = tid; c < chunks; c += blockDim.x)
            dst[c] = src[c];
    }
    __syncthreads();

    // ── Per-lane fragment geometry ──
    const int fcol = lane & 15;      // n for the b-side / m for the a-side
    const int kgrp = lane >> 4;      // 0..3 -> 8-byte sub-window of each 32k
    const int kbase = wave * (32 * S);

    const bool arow_valid = fcol < M;
    const size_t arow = (size_t)(arow_valid ? fcol : 0) * K;
    const unsigned long long* a8 =
        reinterpret_cast<const unsigned long long*>(a_lds + arow + kbase + kgrp * 8);
    const unsigned long long* w8 = reinterpret_cast<const unsigned long long*>(
        W + (size_t)(n0 + fcol) * K + kbase + kgrp * 8);

    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // Full-segment prefetch: all S weight loads (8B each; the wave
    // collectively covers each 32-byte window across kgrp lanes)
    // issue before any MFMA consumes. Fixed ascending-k order.
    unsigned long long bw[S];
    #pragma unroll
    for (int s = 0; s < S; ++s)
        bw[s] = w8[s << 2];                    // +s*32 bytes
    #pragma unroll
    for (int s = 0; s < S; ++s) {
        unsigned long long av = arow_valid ? a8[s << 2] : 0ull;
        acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            (long)av, (long)bw[s], acc, 0, 0, 0);
    }

    // ── Cross-wave reduce (ascending wave order — deterministic) ──
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        red[(wave << 8) + (lane << 2) + i] = acc[i];
    __syncthreads();
    if (wave != 0) return;

    float sum[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float s = red[(lane << 2) + i];
        for (int wv = 1; wv < WAVES; ++wv)
            s += red[(wv << 8) + (lane << 2) + i];
        sum[i] = s;
    }

    const float s_ab = (*dsa) * (*dsb);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = (lane >> 4) * 4 + i;
        int nl = lane & 15;
        if (m < M)
            D[(size_t)m * N + n0 + nl] = __float2bfloat16(sum[i] * s_ab);
    }
}

// ── LDS-staged variant ──
// The direct kernel's B loads are 8B (dwordx2) — half the bytes per
// outstanding request of the stream_probe winners, and the measured
// per-CU byte rate (~8-10 GB/s) is half the probe's (~26 GB/s at
// dwordx4). This variant decouples the memory pattern from the MFMA
// fragment layout: each wave stages its rows into a PRIVATE LDS
// region with per-lane independent dwordx4 chains (the probe-exact
// recipe), then feeds MFMA fragments from LDS.
template<int S, int WAVES>
__global__ __launch_bounds__(64 * WAVES)
void smallm_mfma_nt_lds_kernel(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ W,
    __hip_bfloat16* __restrict__ D,
    int M, int N, int K,
    const float* __restrict__ dsa, const float* __restrict__ dsb) {
    constexpr int RSTEPS = (S < 8) ? S : 8;      // MFMA steps per round
    constexpr int ROUNDS = S / RSTEPS;
    constexpr int ROWB = RSTEPS * 32;            // staged bytes per row
    constexpr int LCH = (16 * ROWB / 16) / 64;   // 16B chunks per lane

    extern __shared__ unsigned char lds[];
    unsigned char* a_lds = lds;                              // M*K
    unsigned char* b_lds = lds + (size_t)M * K;              // WAVES*16*ROWB
    float* red = reinterpret_cast<float*>(b_lds + (size_t)WAVES * 16 * ROWB);

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n0 = blockIdx.x * 16;

    {
        const uint4* src = reinterpret_cast<const uint4*>(A);
        uint4* dst = reinterpret_cast<uint4*>(a_lds);
        int chunks = (M * K) >> 4;
        for (int c = tid; c < chunks; c += blockDim.x)
            dst[c] = src[c];
    }
    __syncthreads();

    const int fcol = lane & 15;
    const int kgrp = lane >> 4;
    const int kbase = wave * (32 * S);
    const bool arow_valid = fcol < M;
    const size_t arow = (size_t)(arow_valid ? fcol : 0) * K;
    const unsigned long long* a8 =
        reinterpret_cast<const unsigned long long*>(a_lds + arow + kbase + kgrp * 8);
    unsigned char* breg = b_lds + (size_t)wave * 16 * ROWB;
    const unsigned char* wbase = W + (size_t)n0 * K + kbase;

    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int r = 0; r < ROUNDS; ++r) {
        // stage: LCH independent dwordx4 per lane, linear over the
        // wave's 16-row x ROWB tile (row-major in LDS)
        uint4 tmp[LCH];
        #pragma unroll
        for (int j = 0; j < LCH; ++j) {
            int c = lane + j * 64;
            int row = c / (ROWB / 16);
            int off = (c % (ROWB / 16)) * 16;
            tmp[j] = *reinterpret_cast<const uint4*>(
                wbase + (size_t)row * K + r * ROWB + off);
        }
        #pragma unroll
        for (int j = 0; j < LCH; ++j)
            *reinterpret_cast<uint4*>(breg + (size_t)(lane + j * 64) * 16) = tmp[j];
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < RSTEPS; ++s) {
            unsigned long long bv = *reinterpret_cast<const unsigned long long*>(
                breg + (size_t)fcol * ROWB + s * 32 + kgrp * 8);
            unsigned long long av =
                arow_valid ? a8[(r * RSTEPS + s) << 2] : 0ull;
            acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                (long)av, (long)bv, acc, 0, 0, 0);
        }
        if (r + 1 < ROUNDS) __syncthreads();   // before the next overwrite
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        red[(wave << 8) + (lane << 2) + i] = acc[i];
    __syncthreads();
    if (wave != 0) return;

    float sum[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float s = red[(lane << 2) + i];
        for (int wv = 1; wv < WAVES; ++wv)
            s += red[(wv << 8) + (lane << 2) + i];
        sum[i] = s;
    }
    const float s_ab = (*dsa) * (*dsb);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = (lane >> 4) * 4 + i;
        int nl = lane & 15;
        if (m < M)
            D[(size_t)m * N + n0 + nl] = __float2bfloat16(sum[i] * s_ab);
    }
}

// Split-K partial kernel: LDS-staged loads, grid (N/16, splits),
// FP32 unscaled partial output per split (see header). WAVES=4.
template<int S>
__global__ __launch_bounds__(256)
void smallm_mfma_nt_partial_kernel(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ W,
    float* __restrict__ ws,              // (splits, M, N) f32
    int M, int N, int K,
    int kseg_total) {                    // K / splits
    constexpr int WAVES = 4;
    constexpr int RSTEPS = (S < 8) ? S : 8;
    constexpr int ROUNDS = S / RSTEPS;
    constexpr int ROWB = RSTEPS * 32;
    constexpr int LCH = (16 * ROWB / 16) / 64;

    extern __shared__ unsigned char lds[];
    unsigned char* a_lds = lds;
    unsigned char* b_lds = lds + (size_t)M * K;
    float* red = reinterpret_cast<float*>(b_lds + (size_t)WAVES * 16 * ROWB);

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n0 = blockIdx.x * 16;
    const int split = blockIdx.y;

    {
        const uint4* src = reinterpret_cast<const uint4*>(A);
        uint4* dst = reinterpret_cast<uint4*>(a_lds);
        int chunks = (M * K) >> 4;
        for (int c = tid; c < chunks; c += blockDim.x)
            dst[c] = src[c];
    }
    __syncthreads();

    const int fcol = lane & 15;
    const int kgrp = lane >> 4;
    const int kbase = split * kseg_total + wave * (32 * S);
    const bool arow_valid = fcol < M;
    const size_t arow = (size_t)(arow_valid ? fcol : 0) * K;
    const unsigned long long* a8 =
        reinterpret_cast<const unsigned long long*>(a_lds + arow + kbase + kgrp * 8);
    unsigned char* breg = b_lds + (size_t)wave * 16 * ROWB;
    const unsigned char* wbase = W + (size_t)n0 * K + kbase;

    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int r = 0; r < ROUNDS; ++r) {
        uint4 tmp[LCH];
        #pragma unroll
        for (int j = 0; j < LCH; ++j) {
            int c = lane + j * 64;
            int row = c / (ROWB / 16);
            int off = (c % (ROWB / 16)) * 16;
            tmp[j] = *reinterpret_cast<const uint4*>(
                wbase + (size_t)row * K + r * ROWB + off);
        }
        #pragma unroll
        for (int j = 0; j < LCH; ++j)
            *reinterpret_cast<uint4*>(breg + (size_t)(lane + j * 64) * 16) = tmp[j];
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < RSTEPS; ++s) {
            unsigned long long bv = *reinterpret_cast<const unsigned long long*>(
                breg + (size_t)fcol * ROWB + s * 32 + kgrp * 8);
            unsigned long long av =
                arow_valid ? a8[(r * RSTEPS + s) << 2] : 0ull;
            acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                (long)av, (long)bv, acc, 0, 0, 0);
        }
        if (r + 1 < ROUNDS) __syncthreads();
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        red[(wave << 8) + (lane << 2) + i] = acc[i];
    __syncthreads();
    if (wave != 0) return;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float s = red[(lane << 2) + i];
        #pragma unroll
        for (int wv = 1; wv < WAVES; ++wv)
            s += red[(wv << 8) + (lane << 2) + i];
        int m = (lane >> 4) * 4 + i;
        int nl = lane & 15;
        if (m < M)
            ws[((size_t)split * M + m) * N + n0 + nl] = s;
    }
}

// Packed-weight kernel (see header): the weight stream is one linear
// slab per (tile, wave) — pure co-pattern dwordx4 chains, all PAIRS
// chunks in flight per lane before any MFMA consumes.
template<int PAIRS>   // 16B chunk-pairs per lane = (K/128)/... = K/(64*4)/2? = S/2
__global__ __launch_bounds__(256)
void smallm_mfma_nt_packed_kernel(
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ Wp,   // packed (see header)
    __hip_bfloat16* __restrict__ D,
    int M, int N, int K,
    const float* __restrict__ dsa, const float* __restrict__ dsb) {
    constexpr int WAVES = 4;
    extern __shared__ unsigned char lds[];
    unsigned char* a_lds = lds;
    float* red = reinterpret_cast<float*>(lds + (size_t)M * K);

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n0 = blockIdx.x * 16;

    {
        const uint4* src = reinterpret_cast<const uint4*>(A);
        uint4* dst = reinterpret_cast<uint4*>(a_lds);
        int chunks = (M * K) >> 4;
        for (int c = tid; c < chunks; c += blockDim.x)
            dst[c] = src[c];
    }
    __syncthreads();

    const int fcol = lane & 15;
    const int kgrp = lane >> 4;
    const int kbase = wave * (64 * PAIRS);   // = wave * 32 * S
    const bool arow_valid = fcol < M;
    const size_t arow = (size_t)(arow_valid ? fcol : 0) * K;
    const unsigned long long* a8 =
        reinterpret_cast<const unsigned long long*>(a_lds + arow + kbase + kgrp * 8);

    const uint4* wp = reinterpret_cast<const uint4*>(
        Wp + (size_t)blockIdx.x * 16 * K)
        + (size_t)(wave * PAIRS) * 64 + lane;

    uint4 ch[PAIRS];
    #pragma unroll
    for (int j = 0; j < PAIRS; ++j)
        ch[j] = wp[(size_t)j * 64];

    floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int j = 0; j < PAIRS; ++j) {
        unsigned long long bv0 =
            ((unsigned long long)ch[j].y << 32) | ch[j].x;
        unsigned long long bv1 =
            ((unsigned long long)ch[j].w << 32) | ch[j].z;
        unsigned long long av0 = arow_valid ? a8[(2 * j) << 2] : 0ull;
        unsigned long long av1 = arow_valid ? a8[(2 * j + 1) << 2] : 0ull;
        acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            (long)av0, (long)bv0, acc, 0, 0, 0);
        acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            (long)av1, (long)bv1, acc, 0, 0, 0);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i)
        red[(wave << 8) + (lane << 2) + i] = acc[i];
    __syncthreads();
    if (wave != 0) return;

    float sum[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float s = red[(lane << 2) + i];
        #pragma unroll
        for (int wv = 1; wv < WAVES; ++wv)
            s += red[(wv << 8) + (lane << 2) + i];
        sum[i] = s;
    }
    const float s_ab = (*dsa) * (*dsb);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m = (lane >> 4) * 4 + i;
        int nl = lane & 15;
        if (m < M)
            D[(size_t)m * N + n0 + nl] = __float2bfloat16(sum[i] * s_ab);
    }
}

struct MfmaVariant { const char* name; int waves; bool lds; };  // waves 0 = auto
const MfmaVariant kVariants[] = {
    {"m0_auto",   0, false},
    {"m1_w4",     4, false},
    {"m2_w8",     8, false},
    {"m3_w16",   16, false},
    {"m4_lds_w4", 4, true},
    {"m5_lds_w8", 8, true},
};

template<int WAVES>
void launch_w(const unsigned char* A, const unsigned char* W,
              __hip_bfloat16* D, int M, int N, int K,
              const float* dsa, const float* dsb,
              size_t smem, hipStream_t stream) {
    dim3 grid(N / 16);
    int steps = K / (32 * WAVES);
    switch (steps) {
    case 2:  smallm_mfma_nt_kernel<2,  WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    case 4:  smallm_mfma_nt_kernel<4,  WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    case 8:  smallm_mfma_nt_kernel<8,  WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    case 16: smallm_mfma_nt_kernel<16, WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    case 32: smallm_mfma_nt_kernel<32, WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    default:
        throw std::runtime_error("smallm_mfma_nt: unsupported K/WAVES combination");
    }
}

template<int WAVES>
void launch_lds(const unsigned char* A, const unsigned char* W,
                __hip_bfloat16* D, int M, int N, int K,
                const float* dsa, const float* dsb,
                hipStream_t stream) {
    dim3 grid(N / 16);
    int steps = K / (32 * WAVES);
    int rsteps = steps < 8 ? steps : 8;
    size_t smem = (size_t)M * K + (size_t)WAVES * 16 * (rsteps * 32)
                  + (size_t)WAVES * 64 * 4 * sizeof(float);
    if (smem > 64 * 1024)
        throw std::runtime_error("smallm_mfma_nt: lds variant exceeds 64KB LDS");
    switch (steps) {
    case 4:  smallm_mfma_nt_lds_kernel<4,  WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    case 8:  smallm_mfma_nt_lds_kernel<8,  WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    case 16: smallm_mfma_nt_lds_kernel<16, WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    case 32: smallm_mfma_nt_lds_kernel<32, WAVES><<<grid, 64 * WAVES, smem, stream>>>(A, W, D, M, N, K, dsa, dsb); break;
    default:
        throw std::runtime_error("smallm_mfma_nt: unsupported K/WAVES for lds variant");
    }
}

} // namespace

void smallm_mfma_nt_partial(const void* A_fp8, const void* W_fp8,
                            float* ws, int M, int N, int K, int splits,
                            hipStream_t stream) {
    if (M < 1 || M > 16 || N % 16 != 0)
        throw std::runtime_error("smallm_mfma_nt_partial: bad M/N");
    if (splits != 2 && splits != 4)
        throw std::runtime_error("smallm_mfma_nt_partial: splits must be 2 or 4");
    int kseg_total = K / splits;
    if (K % splits != 0 || kseg_total % 128 != 0)
        throw std::runtime_error("smallm_mfma_nt_partial: K not divisible");
    int steps = kseg_total / (32 * 4);     // per-wave steps, WAVES=4
    int rsteps = steps < 8 ? steps : 8;
    size_t smem = (size_t)M * K + 4 * 16 * (rsteps * 32) + 4 * 64 * 4 * sizeof(float);
    if (smem > 64 * 1024)
        throw std::runtime_error("smallm_mfma_nt_partial: exceeds 64KB LDS");
    dim3 grid(N / 16, splits);
    const unsigned char* A = static_cast<const unsigned char*>(A_fp8);
    const unsigned char* W = static_cast<const unsigned char*>(W_fp8);
    switch (steps) {
    case 2:  smallm_mfma_nt_partial_kernel<2><<<grid, 256, smem, stream>>>(A, W, ws, M, N, K, kseg_total); break;
    case 4:  smallm_mfma_nt_partial_kernel<4><<<grid, 256, smem, stream>>>(A, W, ws, M, N, K, kseg_total); break;
    case 8:  smallm_mfma_nt_partial_kernel<8><<<grid, 256, smem, stream>>>(A, W, ws, M, N, K, kseg_total); break;
    case 16: smallm_mfma_nt_partial_kernel<16><<<grid, 256, smem, stream>>>(A, W, ws, M, N, K, kseg_total); break;
    default:
        throw std::runtime_error("smallm_mfma_nt_partial: unsupported K/splits");
    }
}

void smallm_mfma_nt_packed(const void* A_fp8, const void* Wp_fp8,
                           __hip_bfloat16* D, int M, int N, int K,
                           const float* d_scale_a, const float* d_scale_b,
                           hipStream_t stream) {
    if (M < 1 || M > 16 || N % 16 != 0 || K % 1024 != 0)
        throw std::runtime_error("smallm_mfma_nt_packed: bad M/N/K");
    int pairs = K / 256;                    // (K/128 steps) / 2, WAVES=4
    size_t smem = (size_t)M * K + 4 * 64 * 4 * sizeof(float);
    dim3 grid(N / 16);
    const unsigned char* A = static_cast<const unsigned char*>(A_fp8);
    const unsigned char* Wp = static_cast<const unsigned char*>(Wp_fp8);
    switch (pairs) {
    case 4:  smallm_mfma_nt_packed_kernel<4><<<grid, 256, smem, stream>>>(A, Wp, D, M, N, K, d_scale_a, d_scale_b); break;
    case 8:  smallm_mfma_nt_packed_kernel<8><<<grid, 256, smem, stream>>>(A, Wp, D, M, N, K, d_scale_a, d_scale_b); break;
    case 16: smallm_mfma_nt_packed_kernel<16><<<grid, 256, smem, stream>>>(A, Wp, D, M, N, K, d_scale_a, d_scale_b); break;
    default:
        throw std::runtime_error("smallm_mfma_nt_packed: unsupported K");
    }
}

int smallm_mfma_variant_count() { return (int)(sizeof(kVariants) / sizeof(kVariants[0])); }
const char* smallm_mfma_variant_name(int id) { return kVariants[id].name; }

void smallm_mfma_nt(int variant,
                    const void* A_fp8, const void* W_fp8,
                    __hip_bfloat16* D, int M, int N, int K,
                    const float* d_scale_a, const float* d_scale_b,
                    hipStream_t stream) {
    if (variant < 0 || variant >= smallm_mfma_variant_count())
        throw std::runtime_error("smallm_mfma_nt: bad variant " + std::to_string(variant));
    if (M < 1 || M > 16)
        throw std::runtime_error("smallm_mfma_nt: M must be in [1,16]");
    if (N % 16 != 0 || K % 1024 != 0)
        throw std::runtime_error("smallm_mfma_nt: need N%16==0 and K%1024==0");
    if (((uintptr_t)A_fp8 & 15) || ((uintptr_t)W_fp8 & 15))
        throw std::runtime_error("smallm_mfma_nt: pointers must be 16B-aligned");
    int waves = kVariants[variant].waves;
    if (waves == 0) {
        // auto: deepen the K-split until grid x waves covers ~1024 waves
        int wgs = N / 16;
        waves = 4;
        while (waves < 16 && wgs * waves < 1024 && (K / (32 * waves)) > 2)
            waves *= 2;
    }
    const unsigned char* A = static_cast<const unsigned char*>(A_fp8);
    const unsigned char* W = static_cast<const unsigned char*>(W_fp8);
    if (kVariants[variant].lds) {
        switch (waves) {
        case 4: launch_lds<4>(A, W, D, M, N, K, d_scale_a, d_scale_b, stream); break;
        case 8: launch_lds<8>(A, W, D, M, N, K, d_scale_a, d_scale_b, stream); break;
        default:
            throw std::runtime_error("smallm_mfma_nt: lds variant waves must be 4/8");
        }
        return;
    }
    size_t smem = (size_t)M * K + (size_t)waves * 64 * 4 * sizeof(float);
    switch (waves) {
    case 4:  launch_w<4>(A, W, D, M, N, K, d_scale_a, d_scale_b, smem, stream); break;
    case 8:  launch_w<8>(A, W, D, M, N, K, d_scale_a, d_scale_b, smem, stream); break;
    case 16: launch_w<16>(A, W, D, M, N, K, d_scale_a, d_scale_b, smem, stream); break;
    default:
        throw std::runtime_error("smallm_mfma_nt: waves must be 4/16");
    }
}

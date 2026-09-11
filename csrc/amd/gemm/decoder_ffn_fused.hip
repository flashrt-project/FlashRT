#include "decoder_ffn_fused.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cmath>
#include <stdexcept>
#include <string>

// ================================================================
// FlashRT AMD — fused decoder-FFN kernel pair (CDNA4, wave64)
//
// See decoder_ffn_fused.h for the contract and design notes. The
// streaming skeleton (lane-K partition in 16B chunks, sequential
// chunk loop, fixed wave shuffle tree, no atomics) is copied verbatim
// from smallm_fp8_nt_kernel (smallm_fp8.hip) so the fp32 GEMM
// accumulation order — and therefore the pre-epilogue result — is
// bit-identical to smallm_fp8_nt_dev on the same inputs.
//
// ILP=4 weight streaming (stream_probe measurement, see
// FlashRT-amd-dev/notes/stream_craft_gfx950.md): both kernels keep 4
// independent weight dwordx4 chains in flight per lane by ISSUING all
// loads of a group before any convert/FMA consumes them — mega1 via
// the 2*NPW gate/up rows of a chunk, mega2 via NPW columns x CG=4/NPW
// consecutive chunks. Consumption keeps the ORIGINAL fixed order
// (ascending chunk, ascending column, gate before up), so the fp32
// accumulation order is UNCHANGED — the bit-equality contract vs the
// smallm nt chain (test_amd_decoder_ffn_fused.py) still holds with no
// relaxation.
//
// Rounding points (bit-comparability with the unfused chain):
//   * both kernels round the scaled fp32 accumulator to BF16 before
//     the epilogue math, because the unfused chain materializes the
//     GEMM output as BF16 (fp8_nt_dev's D) and the elementwise
//     kernels (gate_geglu_merged_fp8 / gate_mul_residual) read that
//     BF16 buffer;
//   * the geglu polynomial, 1/scale, ±448 clamp and __hip_fp8_e4m3
//     construction are written with the exact same expressions as
//     gate_geglu_merged_fp8 (activation.hip);
//   * the residual update is f32(res) + f32(x_bf16)*f32(gate_bf16)
//     rounded once to BF16, exactly gate_mul_res_kernel
//     (elementwise.hip).
// ================================================================

#define FFN_FUSED_BLOCK 256   // threads per workgroup (4 waves)

// ── fp8 e4m3 byte -> f32 (same as smallm_fp8.hip) ──
__device__ __forceinline__ float ffn_fused_fp8_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

// Unpack 16 fp8 bytes (one dwordx4 load) to f32.
__device__ __forceinline__ void ffn_fused_fp8x16_to_f32(const uint4& v, float* out) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    #pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = ffn_fused_fp8_to_f32(b[j]);
}

// ================================================================
// Kernel 1: gate|up GEMM + GeGLU + FP8 quantize.
//   W stored (2*N_half, K) row-major; geglu output column n streams
//   W rows n (gate) and N_half + n (up) — N_half*K bytes apart —
//   off the same converted A chunk.
//   grid = ceil(N_half / (4*NPW)), block = 256; wave w owns geglu
//   columns nb + w*NPW .. +NPW-1. No LDS, no split-K, no atomics.
// ================================================================
template<int M_T, int NPW>
__global__ __launch_bounds__(FFN_FUSED_BLOCK)
void smallm_fp8_gateup_geglu_kernel(const unsigned char* __restrict__ A,
                                    const unsigned char* __restrict__ W,
                                    __hip_fp8_e4m3* __restrict__ out,
                                    const float* __restrict__ d_scale_a,
                                    const float* __restrict__ d_scale_w,
                                    const float* __restrict__ d_scale_out,
                                    int M, int N_half, int K) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int nb   = blockIdx.x * (4 * NPW) + wave * NPW;

    float acc_g[NPW][M_T];
    float acc_u[NPW][M_T];
    #pragma unroll
    for (int p = 0; p < NPW; ++p)
        #pragma unroll
        for (int m = 0; m < M_T; ++m) { acc_g[p][m] = 0.0f; acc_u[p][m] = 0.0f; }

    // K-chunks of 64 lanes x 16B = 1024 elements; sequential chunk
    // loop per lane keeps accumulation order fixed (deterministic,
    // bit-identical to smallm_fp8_nt_kernel).
    //
    // ILP (stream_craft_gfx950.md): all 2*NPW weight loads of a chunk
    // (gate + up rows per owned column) are ISSUED first into distinct
    // registers — 2*NPW >= 4 independent dwordx4 chains in flight per
    // lane, the measured gfx950 saturation lever — then consumed in
    // the ORIGINAL order (ascending p, gate before up), so each
    // accumulator's fp32 order is unchanged and mega1 stays bit-equal
    // to the smallm nt chain.
    const int chunks = (K + 1023) >> 10;
    for (int c = 0; c < chunks; ++c) {
        const int kw = (c << 10) + (lane << 4);
        if (kw >= K) continue;  // tail guard (K % 16 == 0 keeps loads whole)

        // ── issue the 2*NPW independent gate/up loads (ILP >= 4) ──
        uint4 graw[NPW], uraw[NPW];
        bool  wok[NPW];
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            wok[p] = (n < N_half);
            if (wok[p]) {
                graw[p] = *reinterpret_cast<const uint4*>(
                    W + (size_t)n * K + kw);
                uraw[p] = *reinterpret_cast<const uint4*>(
                    W + (size_t)(N_half + n) * K + kw);
            }
        }

        // A window for this lane's 16 k's, all rows, converted once
        // (A is L1/L2-hot — not the stream).
        float a_f[M_T][16];
        #pragma unroll
        for (int m = 0; m < M_T; ++m) {
            if (m < M) {
                const uint4 av = *reinterpret_cast<const uint4*>(
                    A + (size_t)m * K + kw);
                ffn_fused_fp8x16_to_f32(av, a_f[m]);
            } else {
                #pragma unroll
                for (int j = 0; j < 16; ++j) a_f[m][j] = 0.0f;
            }
        }

        // ── consume: ascending p, gate row then up row ──
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            if (!wok[p]) continue;
            float gf[16];
            ffn_fused_fp8x16_to_f32(graw[p], gf);
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    acc_g[p][m] += a_f[m][j] * gf[j];
            }
            float uf[16];
            ffn_fused_fp8x16_to_f32(uraw[p], uf);
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                #pragma unroll
                for (int j = 0; j < 16; ++j)
                    acc_u[p][m] += a_f[m][j] * uf[j];
            }
        }
    }

    // Fixed-shape wave shuffle tree (order identical every launch).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        #pragma unroll
        for (int p = 0; p < NPW; ++p)
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                acc_g[p][m] += __shfl_down(acc_g[p][m], off);
                acc_u[p][m] += __shfl_down(acc_u[p][m], off);
            }
    }

    if (lane == 0) {
        const float sc = d_scale_a[0] * d_scale_w[0];
        const float inv_scale = 1.0f / (*d_scale_out);
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            if (n >= N_half) continue;
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    // Round to BF16 first: the unfused chain's geglu
                    // reads fp8_nt_dev's BF16 output buffer.
                    const float g = __bfloat162float(
                        __float2bfloat16(acc_g[p][m] * sc));
                    const float u = __bfloat162float(
                        __float2bfloat16(acc_u[p][m] * sc));
                    // EXACT gate_geglu_merged_fp8 epilogue (activation.hip).
                    float gelu = g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
                    float val = gelu * u;
                    val = fminf(fmaxf(val * inv_scale, -448.0f), 448.0f);
                    out[(size_t)m * N_half + n] = __hip_fp8_e4m3(val);
                }
            }
        }
    }
}

// ================================================================
// Kernel 2: down GEMM + gate*x + residual (in place).
//   W stored (N, K) row-major. Same skeleton; the BF16 GEMM output is
//   never materialized — lane 0 folds it straight into the residual.
// ================================================================
template<int M_T, int NPW>
__global__ __launch_bounds__(FFN_FUSED_BLOCK)
void smallm_fp8_down_gateres_kernel(const unsigned char* __restrict__ H,
                                    const unsigned char* __restrict__ W,
                                    __hip_bfloat16* __restrict__ residual,
                                    const __hip_bfloat16* __restrict__ gate,
                                    const float* __restrict__ d_scale_h,
                                    const float* __restrict__ d_scale_w,
                                    int M, int N, int K) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int nb   = blockIdx.x * (4 * NPW) + wave * NPW;

    float acc[NPW][M_T];
    #pragma unroll
    for (int p = 0; p < NPW; ++p)
        #pragma unroll
        for (int m = 0; m < M_T; ++m) acc[p][m] = 0.0f;

    // ILP=4 (stream_craft_gfx950.md): a group of CG = 4/NPW
    // consecutive K-chunks per outer iteration; all NPW*CG == 4
    // independent W dwordx4 loads issued FIRST (4 chains in flight per
    // lane — pi05 site NPW=1, K=4096: the whole weight row preloaded
    // as 4 chains), then consumed ascending chunk / ascending p — the
    // ORIGINAL fixed order, so each accumulator's fp32 order is
    // unchanged and mega2 stays bit-equal to the smallm nt chain.
    constexpr int CG = (NPW >= 4) ? 1 : (4 / NPW);
    const int chunks = (K + 1023) >> 10;
    for (int c0 = 0; c0 < chunks; c0 += CG) {
        // ── issue all NPW x CG independent W loads (ILP4) ──
        uint4 braw[NPW][CG];
        bool  bok[NPW][CG];
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            #pragma unroll
            for (int u = 0; u < CG; ++u) {
                const int kw = ((c0 + u) << 10) + (lane << 4);
                bok[p][u] = (n < N) && (kw < K);  // K % 16 == 0: whole loads
                if (bok[p][u])
                    braw[p][u] = *reinterpret_cast<const uint4*>(
                        W + (size_t)n * K + kw);
            }
        }

        // ── consume: ascending chunk, then ascending column ──
        #pragma unroll
        for (int u = 0; u < CG; ++u) {
            const int kw = ((c0 + u) << 10) + (lane << 4);
            if (kw >= K) continue;  // tail guard

            float a_f[M_T][16];
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    const uint4 av = *reinterpret_cast<const uint4*>(
                        H + (size_t)m * K + kw);
                    ffn_fused_fp8x16_to_f32(av, a_f[m]);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) a_f[m][j] = 0.0f;
                }
            }

            #pragma unroll
            for (int p = 0; p < NPW; ++p) {
                if (!bok[p][u]) continue;
                float bf[16];
                ffn_fused_fp8x16_to_f32(braw[p][u], bf);
                #pragma unroll
                for (int m = 0; m < M_T; ++m) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                        acc[p][m] += a_f[m][j] * bf[j];
                }
            }
        }
    }

    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        #pragma unroll
        for (int p = 0; p < NPW; ++p)
            #pragma unroll
            for (int m = 0; m < M_T; ++m)
                acc[p][m] += __shfl_down(acc[p][m], off);
    }

    if (lane == 0) {
        const float sc = d_scale_h[0] * d_scale_w[0];
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            if (n >= N) continue;
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    const size_t idx = (size_t)m * N + n;
                    // Round to BF16 first: gate_mul_residual reads
                    // the GEMM's BF16 output buffer in the unfused
                    // chain. Then EXACT gate_mul_res_kernel math.
                    const float x = __bfloat162float(
                        __float2bfloat16(acc[p][m] * sc));
                    const float r = __bfloat162float(residual[idx]) +
                                    x * __bfloat162float(gate[idx]);
                    residual[idx] = __float2bfloat16(r);
                }
            }
        }
    }
}

// ================================================================
// Host side
// ================================================================

static void ffn_fused_check(const char* who, int M, int N, int K) {
    if (M < 1 || M > 16)
        throw std::runtime_error(std::string(who) + ": M must be in [1,16], got " +
                                 std::to_string(M));
    if (N < 1)
        throw std::runtime_error(std::string(who) + ": N must be >= 1, got " +
                                 std::to_string(N));
    if (K < 16 || (K % 16) != 0)
        throw std::runtime_error(std::string(who) + ": K must be a multiple of 16, got " +
                                 std::to_string(K));
}

template<int NPW>
static void smallm_fp8_gateup_geglu_impl(const void* A, const void* W, void* out,
                                         const float* d_scale_a,
                                         const float* d_scale_w,
                                         const float* d_scale_out,
                                         int M, int N_half, int K,
                                         hipStream_t stream) {
    ffn_fused_check("smallm_fp8_gateup_geglu", M, N_half, K);
    const int cols_per_wg = 4 * NPW;
    const dim3 grid((unsigned)((N_half + cols_per_wg - 1) / cols_per_wg));
    const dim3 block(FFN_FUSED_BLOCK);
    const unsigned char* a8 = static_cast<const unsigned char*>(A);
    const unsigned char* w8 = static_cast<const unsigned char*>(W);
    __hip_fp8_e4m3* o8 = static_cast<__hip_fp8_e4m3*>(out);

    #define FFN_LAUNCH_GATEUP(MT)                                              \
        hipLaunchKernelGGL(                                                    \
            HIP_KERNEL_NAME(smallm_fp8_gateup_geglu_kernel<MT, NPW>),          \
            grid, block, 0, stream,                                            \
            a8, w8, o8, d_scale_a, d_scale_w, d_scale_out, M, N_half, K)
    if      (M <= 4)  FFN_LAUNCH_GATEUP(4);
    else if (M <= 8)  FFN_LAUNCH_GATEUP(8);
    else if (M <= 10) FFN_LAUNCH_GATEUP(10);
    else              FFN_LAUNCH_GATEUP(16);
    #undef FFN_LAUNCH_GATEUP
}

void smallm_fp8_gateup_geglu(const void* A_fp8, const void* W_gateup,
                             void* OutFp8,
                             const float* d_scale_a, const float* d_scale_w,
                             const float* d_scale_out,
                             int M, int N_half, int K, hipStream_t stream) {
    // NPW=2: 8 geglu columns (16 weight-row streams) per WG ->
    // N_half/8 = 512 WGs at N_half=4096 (2 per MI350X CU); same
    // register footprint as the proven smallm nt NPW=4 config.
    smallm_fp8_gateup_geglu_impl<2>(A_fp8, W_gateup, OutFp8,
                                    d_scale_a, d_scale_w, d_scale_out,
                                    M, N_half, K, stream);
}

void smallm_fp8_gateup_geglu_alt(const void* A_fp8, const void* W_gateup,
                                 void* OutFp8,
                                 const float* d_scale_a, const float* d_scale_w,
                                 const float* d_scale_out,
                                 int M, int N_half, int K, hipStream_t stream) {
    // NPW=4: 256 WGs at N_half=4096, 2x A reuse, higher VGPR pressure.
    smallm_fp8_gateup_geglu_impl<4>(A_fp8, W_gateup, OutFp8,
                                    d_scale_a, d_scale_w, d_scale_out,
                                    M, N_half, K, stream);
}

template<int NPW>
static void smallm_fp8_down_gateres_impl(const void* H, const void* W,
                                         void* residual, const void* gate,
                                         const float* d_scale_h,
                                         const float* d_scale_w,
                                         int M, int N, int K,
                                         hipStream_t stream) {
    ffn_fused_check("smallm_fp8_down_gateres", M, N, K);
    const int cols_per_wg = 4 * NPW;
    const dim3 grid((unsigned)((N + cols_per_wg - 1) / cols_per_wg));
    const dim3 block(FFN_FUSED_BLOCK);
    const unsigned char* h8 = static_cast<const unsigned char*>(H);
    const unsigned char* w8 = static_cast<const unsigned char*>(W);
    __hip_bfloat16* r16 = static_cast<__hip_bfloat16*>(residual);
    const __hip_bfloat16* g16 = static_cast<const __hip_bfloat16*>(gate);

    #define FFN_LAUNCH_DOWN(MT)                                                \
        hipLaunchKernelGGL(                                                    \
            HIP_KERNEL_NAME(smallm_fp8_down_gateres_kernel<MT, NPW>),          \
            grid, block, 0, stream,                                            \
            h8, w8, r16, g16, d_scale_h, d_scale_w, M, N, K)
    if      (M <= 4)  FFN_LAUNCH_DOWN(4);
    else if (M <= 8)  FFN_LAUNCH_DOWN(8);
    else if (M <= 10) FFN_LAUNCH_DOWN(10);
    else              FFN_LAUNCH_DOWN(16);
    #undef FFN_LAUNCH_DOWN
}

void smallm_fp8_down_gateres(const void* H_fp8, const void* W_down,
                             void* residual, const void* gate,
                             const float* d_scale_h, const float* d_scale_w,
                             int M, int N, int K, hipStream_t stream) {
    // NPW=1: 4 columns per WG -> N/4 = 256 WGs at N=1024, exactly one
    // per MI350X CU with a serial 4-chunk K loop (K=4096).
    smallm_fp8_down_gateres_impl<1>(H_fp8, W_down, residual, gate,
                                    d_scale_h, d_scale_w, M, N, K, stream);
}

void smallm_fp8_down_gateres_alt(const void* H_fp8, const void* W_down,
                                 void* residual, const void* gate,
                                 const float* d_scale_h, const float* d_scale_w,
                                 int M, int N, int K, hipStream_t stream) {
    // NPW=2: 128 WGs at N=1024, halves per-byte A-convert ALU cost.
    smallm_fp8_down_gateres_impl<2>(H_fp8, W_down, residual, gate,
                                    d_scale_h, d_scale_w, M, N, K, stream);
}

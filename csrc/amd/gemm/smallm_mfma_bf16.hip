// ================================================================
// FlashRT AMD — MFMA small-M BF16 packed-weight GEMM (gfx950, wave64)
// See smallm_mfma_bf16.h for the contract, pack layout and rationale.
//
// Load ordering inside a round (perf-critical, mirrors the FP8 packed
// kernel's full-in-flight discipline):
//   1. this round's A fragments   (LLC-resident, cheap)
//   2. NEXT round's weight chunks (the DRAM stream — 6 independent
//      dwordx4 per lane, issued before any wait)
//   3. MFMA chain consuming this round's (already landed) weights
// vmcnt retires in issue order, so the MFMA wait covers (1) and the
// PREVIOUS round's weights while leaving (2) outstanding — the weight
// stream never drains across round boundaries.
// ================================================================

#include "smallm_mfma_bf16.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

typedef float floatx4 __attribute__((ext_vector_type(4)));
typedef short shortx8 __attribute__((ext_vector_type(8)));

// Steps consumed per round; every supported S is a multiple.
constexpr int BF16_RSTEPS = 6;
// Reduction scratch stride per (wave, tile): 64 lanes x 4 floats,
// +8 pad so the row stride is not 0 mod 128B.
constexpr int RED_STRIDE = 64 * 4 + 8;

__device__ __forceinline__ float gelu_tanh_f32(float g) {
    // sigmoid form of the tanh approximation:
    //   0.5*g*(1+tanh(0.79788456*(g+0.044715 g^3)))
    // == g * sigmoid(1.59576912 * g * (1 + 0.044715 g^2))
    // (same constants as kernels/activation.hip / hipBLASLt GELU_BIAS)
    return g / (1.0f + __expf(-1.5957691216057308f * g
                              * (1.0f + 0.044715f * g * g)));
}

// S     = MFMA k-steps per wave segment (kseg = 32*S = K/WAVES)
// WAVES = waves per workgroup (K-split factor inside the WG)
// MTPW  = m-tiles owned by this workgroup (1 = split grid.y, 3 = fused)
// EPI   = 0 bias, 1 bias+gelu, 2 bias+residual-accumulate
template<int S, int WAVES, int MTPW, int EPI>
__global__ __launch_bounds__(64 * WAVES)
void smallm_bf16_nn_packed_kernel(
    const __hip_bfloat16* __restrict__ A,     // (M,K) bf16 row-major
    const shortx8* __restrict__ Wp,           // packed 16B chunks (header)
    const __hip_bfloat16* __restrict__ bias,  // (N)
    __hip_bfloat16* __restrict__ D,           // (M,N) bf16
    int M, int N, int K) {
    constexpr int ROUNDS = S / BF16_RSTEPS;
    __shared__ float red[WAVES * MTPW * RED_STRIDE];

    const int tid  = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n0   = blockIdx.x * 16;
    const int gt0  = blockIdx.y * MTPW;   // first global m-tile

    const int fcol = lane & 15;           // n for b / m for a / n for c
    const int kgrp = lane >> 4;           // 8-elem sub-window of each 32k
    const int kbase = wave * (32 * S);

    // A row pointers per m-tile; pad rows (m >= M) clamp to the last
    // real row — their C rows are never stored, only their own
    // accumulators see the duplicate values.
    const __hip_bfloat16* arow[MTPW];
    #pragma unroll
    for (int t = 0; t < MTPW; ++t) {
        int m = 16 * (gt0 + t) + fcol;
        if (m >= M) m = M - 1;
        arow[t] = A + (size_t)m * K + kbase + kgrp * 8;
    }

    // Packed stream: 2*K chunks per n-tile, chunk = step*64 + lane
    // inside the tile, this wave starting at step wave*S.
    const shortx8* wp = Wp + (size_t)blockIdx.x * 2 * K
                           + (size_t)(wave * S) * 64 + lane;

    floatx4 acc[MTPW];
    #pragma unroll
    for (int t = 0; t < MTPW; ++t)
        acc[t] = (floatx4){0.0f, 0.0f, 0.0f, 0.0f};

    shortx8 bw[2][BF16_RSTEPS];
    #pragma unroll
    for (int s = 0; s < BF16_RSTEPS; ++s)
        bw[0][s] = wp[(size_t)s * 64];

    for (int r = 0; r < ROUNDS; ++r) {
        const int cur = r & 1;
        // (1) this round's a-fragments
        shortx8 af[MTPW][BF16_RSTEPS];
        #pragma unroll
        for (int s = 0; s < BF16_RSTEPS; ++s) {
            #pragma unroll
            for (int t = 0; t < MTPW; ++t)
                af[t][s] = *reinterpret_cast<const shortx8*>(
                    arow[t] + (r * BF16_RSTEPS + s) * 32);
        }
        // (2) next round's weight chunks stay in flight through (3)
        if (r + 1 < ROUNDS) {
            #pragma unroll
            for (int s = 0; s < BF16_RSTEPS; ++s)
                bw[cur ^ 1][s] = wp[(size_t)((r + 1) * BF16_RSTEPS + s) * 64];
        }
        // (3) consume: each weight fragment feeds all MTPW accumulators
        #pragma unroll
        for (int s = 0; s < BF16_RSTEPS; ++s) {
            #pragma unroll
            for (int t = 0; t < MTPW; ++t)
                acc[t] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                    af[t][s], bw[cur][s], acc[t], 0, 0, 0);
        }
    }

    // ── Cross-wave reduce (ascending wave order — deterministic) ──
    #pragma unroll
    for (int t = 0; t < MTPW; ++t) {
        float* rp = &red[(wave * MTPW + t) * RED_STRIDE + lane * 4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) rp[i] = acc[t][i];
    }
    __syncthreads();
    if (wave >= MTPW) return;   // wave t finishes tile t

    const int t = wave;
    float sum[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        sum[i] = red[t * RED_STRIDE + lane * 4 + i];
    #pragma unroll
    for (int wv = 1; wv < WAVES; ++wv) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            sum[i] += red[(wv * MTPW + t) * RED_STRIDE + lane * 4 + i];
    }

    const int nl = n0 + fcol;
    const float bv = __bfloat162float(bias[nl]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = 16 * (gt0 + t) + 4 * kgrp + i;
        if (m >= M) continue;
        float v = sum[i] + bv;
        if (EPI == 1) v = gelu_tanh_f32(v);
        __hip_bfloat16* dp = D + (size_t)m * N + nl;
        if (EPI == 2) v += __bfloat162float(*dp);
        *dp = __float2bfloat16(v);
    }
}

// ── Dispatch ──

struct Bf16Variant { const char* name; int waves; bool fused; };
const Bf16Variant kBf16Variants[] = {
    {"auto",     0, false},
    {"w4_fused", 4, true},
    {"w8_fused", 8, true},
    {"w4_split", 4, false},
    {"w8_split", 8, false},
};

template<int S, int WAVES, int MTPW>
void launch_epi(int epi, dim3 grid,
                const __hip_bfloat16* A, const shortx8* Wp,
                const __hip_bfloat16* bias, __hip_bfloat16* D,
                int M, int N, int K, hipStream_t stream) {
    const dim3 block(64 * WAVES);
    switch (epi) {
    case 0: smallm_bf16_nn_packed_kernel<S, WAVES, MTPW, 0>
                <<<grid, block, 0, stream>>>(A, Wp, bias, D, M, N, K); break;
    case 1: smallm_bf16_nn_packed_kernel<S, WAVES, MTPW, 1>
                <<<grid, block, 0, stream>>>(A, Wp, bias, D, M, N, K); break;
    case 2: smallm_bf16_nn_packed_kernel<S, WAVES, MTPW, 2>
                <<<grid, block, 0, stream>>>(A, Wp, bias, D, M, N, K); break;
    default:
        throw std::runtime_error("smallm_mfma_bf16: epilogue must be 0/1/2");
    }
}

template<int WAVES, int MTPW>
void launch_steps(int steps, int epi, dim3 grid,
                  const __hip_bfloat16* A, const shortx8* Wp,
                  const __hip_bfloat16* bias, __hip_bfloat16* D,
                  int M, int N, int K, hipStream_t stream) {
    switch (steps) {
    case 6:  launch_epi<6,  WAVES, MTPW>(epi, grid, A, Wp, bias, D, M, N, K, stream); break;
    case 12: launch_epi<12, WAVES, MTPW>(epi, grid, A, Wp, bias, D, M, N, K, stream); break;
    case 24: launch_epi<24, WAVES, MTPW>(epi, grid, A, Wp, bias, D, M, N, K, stream); break;
    case 48: launch_epi<48, WAVES, MTPW>(epi, grid, A, Wp, bias, D, M, N, K, stream); break;
    default:
        throw std::runtime_error(
            "smallm_mfma_bf16: K/(32*waves) must be 6/12/24/48, got K="
            + std::to_string(K) + " waves=" + std::to_string(WAVES));
    }
}

bool steps_valid(int K, int waves) {
    if (K % (32 * waves) != 0) return false;
    int s = K / (32 * waves);
    return s == 6 || s == 12 || s == 24 || s == 48;
}

} // namespace

int smallm_mfma_bf16_variant_count() {
    return (int)(sizeof(kBf16Variants) / sizeof(kBf16Variants[0]));
}

const char* smallm_mfma_bf16_variant_name(int id) {
    return kBf16Variants[id].name;
}

void smallm_mfma_bf16_nn_packed(int variant, int epilogue,
                                const void* A_bf16, const void* Wp_bf16,
                                const void* bias_bf16, __hip_bfloat16* D,
                                int M, int N, int K, hipStream_t stream) {
    if (variant < 0 || variant >= smallm_mfma_bf16_variant_count())
        throw std::runtime_error("smallm_mfma_bf16: bad variant "
                                 + std::to_string(variant));
    if (M < 1 || M > 48)
        throw std::runtime_error("smallm_mfma_bf16: M must be in [1,48]");
    if (N % 16 != 0)
        throw std::runtime_error("smallm_mfma_bf16: need N % 16 == 0");
    if (((uintptr_t)A_bf16 & 15) || ((uintptr_t)Wp_bf16 & 15) || ((uintptr_t)D & 15))
        throw std::runtime_error("smallm_mfma_bf16: A/Wp/D must be 16B-aligned");
    if (bias_bf16 == nullptr)
        throw std::runtime_error("smallm_mfma_bf16: bias is required");

    const int mt = (M + 15) / 16;
    int waves = kBf16Variants[variant].waves;
    bool fused = kBf16Variants[variant].fused;
    if (waves == 0) {
        // auto: wide N has workgroups to spare — keep the weight
        // stream 1x with fused m-tiles at 4 waves; narrow N is
        // WG-starved — split m-tiles across grid.y and go as deep as
        // the K geometry allows.
        if (N / 16 >= 192 && mt > 1 && steps_valid(K, 4)) {
            waves = 4; fused = true;
        } else if (steps_valid(K, 8)) {
            waves = 8; fused = false;
        } else {
            waves = 4; fused = false;
        }
    }
    if (!steps_valid(K, waves))
        throw std::runtime_error(
            "smallm_mfma_bf16: unsupported K for waves="
            + std::to_string(waves) + " (K=" + std::to_string(K) + ")");

    const int steps = K / (32 * waves);
    const dim3 grid(N / 16, fused ? 1 : mt);
    const __hip_bfloat16* A = static_cast<const __hip_bfloat16*>(A_bf16);
    const shortx8* Wp = static_cast<const shortx8*>(Wp_bf16);
    const __hip_bfloat16* bias = static_cast<const __hip_bfloat16*>(bias_bf16);

    if (fused) {
        if (waves == 4)
            launch_steps<4, 3>(steps, epilogue, grid, A, Wp, bias, D, M, N, K, stream);
        else
            launch_steps<8, 3>(steps, epilogue, grid, A, Wp, bias, D, M, N, K, stream);
    } else {
        if (waves == 4)
            launch_steps<4, 1>(steps, epilogue, grid, A, Wp, bias, D, M, N, K, stream);
        else
            launch_steps<8, 1>(steps, epilogue, grid, A, Wp, bias, D, M, N, K, stream);
    }
}

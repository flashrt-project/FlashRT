// SPDX-License-Identifier: Apache-2.0
// Skinny weight-only INT8 GEMM (W8A16) for the LaWAM flow DiT on sm_120.
//
// partials[s][m][n] = w_scale[n] * sum_{k in split s} A[m][k] * W[n][k]
//
// A is BF16 [M, K] (M <= 32), W INT8 [N, K] (natural nn.Linear layout, one
// FP32 scale per output channel), partials FP32 [K / KC][M][N] — the layout
// of the skinny NVFP4 family, so the DiT consumers read it unchanged (with
// alpha = 1). INT8 values are exact in BF16, the reduction runs on BF16
// mma.sync (m16n8k16, FP32 accumulate).
//
// Each CTA owns BN weight rows x KC of K: its weight slice is loaded into
// registers before the programmatic-dependent-launch wait (the weights do not
// depend on the producer), then the activation chunk is staged in shared
// memory. Within every 64-wide K block, lane quad t owns the 16 consecutive
// k of [16t, 16t + 16): one 16-byte weight load per lane and n8 tile. mma
// step s maps the instruction's k pairs (2t, 2t+1) / (2t+8, 2t+9) to the
// physical k (16t + 4s, +1) / (16t + 4s + 2, +3) for both operands, so the
// reduction is exact up to FP32 summation order.
#include "lawam_kernels.cuh"

#include <cuda_bf16.h>

namespace flash_rt {
namespace lawam {
namespace {

constexpr int kWarps = 4;
constexpr int kRows = 32;            // two m16 tiles

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two signed bytes -> packed BF16 pair (exact).
__device__ __forceinline__ uint32_t i8x2_bf16x2(uint32_t word, int byte) {
    const int lo = static_cast<int8_t>((word >> (8 * byte)) & 0xff);
    const int hi = static_cast<int8_t>((word >> (8 * byte + 8)) & 0xff);
    const __nv_bfloat162 v = __floats2bfloat162_rn(static_cast<float>(lo), static_cast<float>(hi));
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

template <int BN, int KC, bool PDL>
__global__ void __launch_bounds__(kWarps * 32)
w8a16_skinny_kernel(const __nv_bfloat16* __restrict__ A, const int8_t* __restrict__ W,
                    const float* __restrict__ w_scale, float* __restrict__ partials, int M, int N, int K) {
    constexpr int kTiles = BN / (kWarps * 8);   // n8 tiles per warp
    constexpr int kBlocks = KC / 64;            // 64-wide K blocks per CTA
    constexpr int kLd = KC + 8;                 // padded smem row (bf16)
    __shared__ __align__(16) __nv_bfloat16 sa[kRows * kLd];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n_cta = blockIdx.x * BN;
    const int split = blockIdx.y;
    const int k_cta = split * KC;

    // weight slice -> registers (before the dependency wait)
    uint4 wreg[kTiles][kBlocks];
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
        const int n = n_cta + (warp * kTiles + j) * 8 + g;
        const int8_t* wr = W + static_cast<size_t>(n) * K + k_cta + 16 * t;
#pragma unroll
        for (int b = 0; b < kBlocks; ++b) wreg[j][b] = ldg_stream(wr + 64 * b);
    }
    if constexpr (PDL) {
        asm volatile("griddepcontrol.launch_dependents;\n" ::);
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
    }
    // activation chunk [kRows, KC] -> shared (rows >= M zero)
    for (int i = threadIdx.x; i < kRows * (KC / 8); i += kWarps * 32) {
        const int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < M) v = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(r) * K + k_cta + c);
        *reinterpret_cast<uint4*>(sa + r * kLd + c) = v;
    }
    __syncthreads();

    float acc[kTiles][2][4];
#pragma unroll
    for (int j = 0; j < kTiles; ++j)
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[j][mt][e] = 0.f;

#pragma unroll
    for (int b = 0; b < kBlocks; ++b) {
        // this lane's 16 consecutive k of the block, for rows g, g+8, g+16, g+24
        uint32_t ar[4][8];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const __nv_bfloat16* p = sa + (g + 8 * rr) * kLd + 64 * b + 16 * t;
            const uint4 v0 = *reinterpret_cast<const uint4*>(p);
            const uint4 v1 = *reinterpret_cast<const uint4*>(p + 8);
            ar[rr][0] = v0.x; ar[rr][1] = v0.y; ar[rr][2] = v0.z; ar[rr][3] = v0.w;
            ar[rr][4] = v1.x; ar[rr][5] = v1.y; ar[rr][6] = v1.z; ar[rr][7] = v1.w;
        }
#pragma unroll
        for (int j = 0; j < kTiles; ++j) {
            const uint32_t wv[4] = {wreg[j][b].x, wreg[j][b].y, wreg[j][b].z, wreg[j][b].w};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                // physical k 16t + 4s + {0,1} (b0) and {2,3} (b1): bytes 4s .. 4s+3 of the lane's 16
                const uint32_t b0 = i8x2_bf16x2(wv[s], 0);
                const uint32_t b1 = i8x2_bf16x2(wv[s], 2);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    // a0 / a2: row g (+16 mt), a1 / a3: row g + 8 (+16 mt); bf16 pairs 2s, 2s+1
                    mma_bf16(acc[j][mt], ar[2 * mt][2 * s], ar[2 * mt + 1][2 * s], ar[2 * mt][2 * s + 1],
                             ar[2 * mt + 1][2 * s + 1], b0, b1);
                }
            }
        }
    }

    float* out = partials + static_cast<size_t>(split) * M * N;
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
        const int n = n_cta + (warp * kTiles + j) * 8 + 2 * t;
        const float s0 = w_scale[n], s1 = w_scale[n + 1];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int r0 = 16 * mt + g, r1 = r0 + 8;
            if (r0 < M)
                *reinterpret_cast<float2*>(out + static_cast<size_t>(r0) * N + n) =
                    make_float2(acc[j][mt][0] * s0, acc[j][mt][1] * s1);
            if (r1 < M)
                *reinterpret_cast<float2*>(out + static_cast<size_t>(r1) * N + n) =
                    make_float2(acc[j][mt][2] * s0, acc[j][mt][3] * s1);
        }
    }
}

template <int BN, int KC>
int launch(const __nv_bfloat16* A, const int8_t* W, const float* ws, float* partials, int M, int N, int K, bool pdl,
           cudaStream_t stream) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(N / BN, K / KC);
    cfg.blockDim = dim3(kWarps * 32);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    if (pdl)
        return static_cast<int>(cudaLaunchKernelEx(&cfg, w8a16_skinny_kernel<BN, KC, true>, A, W, ws, partials, M, N, K));
    return static_cast<int>(cudaLaunchKernelEx(&cfg, w8a16_skinny_kernel<BN, KC, false>, A, W, ws, partials, M, N, K));
}

}  // namespace

// Tile configurations: (BN, KC); the weight slice of a lane stays within
// 64 registers and the activation chunk within static shared memory.
static const int kCfg[][2] = {{32, 256}, {32, 512}, {64, 256}, {64, 512}, {128, 256}};

int w8a16_skinny_config_count() { return static_cast<int>(sizeof(kCfg) / sizeof(kCfg[0])); }

int w8a16_skinny_config_k_chunk(int cfg) {
    return (cfg >= 0 && cfg < w8a16_skinny_config_count()) ? kCfg[cfg][1] : 0;
}

int w8a16_skinny_gemm(const __nv_bfloat16* A, const int8_t* W, const float* w_scale, float* partials, int M, int N,
                      int K, int cfg, bool pdl, cudaStream_t stream) {
    if (cfg < 0 || cfg >= w8a16_skinny_config_count() || M < 1 || M > kRows) return static_cast<int>(cudaErrorInvalidValue);
    const int bn = kCfg[cfg][0], kc = kCfg[cfg][1];
    if (N % bn != 0 || K % kc != 0) return static_cast<int>(cudaErrorInvalidValue);
    switch (cfg) {
        case 0: return launch<32, 256>(A, W, w_scale, partials, M, N, K, pdl, stream);
        case 1: return launch<32, 512>(A, W, w_scale, partials, M, N, K, pdl, stream);
        case 2: return launch<64, 256>(A, W, w_scale, partials, M, N, K, pdl, stream);
        case 3: return launch<64, 512>(A, W, w_scale, partials, M, N, K, pdl, stream);
        default: return launch<128, 256>(A, W, w_scale, partials, M, N, K, pdl, stream);
    }
}

}  // namespace lawam
}  // namespace flash_rt

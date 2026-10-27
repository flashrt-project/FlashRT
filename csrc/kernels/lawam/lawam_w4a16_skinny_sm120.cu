// SPDX-License-Identifier: Apache-2.0
// Skinny weight-only NVFP4 GEMM (W4A16) for the LaWAM flow DiT on sm_120.
//
// partials[s][m][n] = sum_{k in split s} A[m][k] * e2m1(W[n][k]) * sf[n][k / 16]
//
// A is BF16 [M, K] (M <= 32); W holds E2M1 pairs [N, K / 2] (element 2j in
// the low nibble) with UE4M3 block scales sf [N, K / 16]; the per-tensor FP32
// scale is left to the consumers (their alpha = a_gs * w_gs), so the partials
// have the layout and meaning of lawam_w8a16_skinny_sm120.cu. Every E2M1 value
// times its UE4M3 scale is exact in FP16 (one hardware E2M1 conversion and one
// half2 multiply per pair), the BF16 activations are staged as FP16 (exact for
// the DiT's range) and the reduction runs on FP16 mma.sync (m16n8k16, FP32
// accumulate).
//
// Each CTA owns BN weight rows x KC of K, loaded into registers before the
// programmatic-dependent-launch wait. Within every 128-wide K block lane quad t
// owns the 32 consecutive k of [32t, 32t + 32) (one 16-byte load + two scale
// bytes per n8 tile); mma step s of half h maps the instruction's k pairs
// (2t, 2t+1) / (2t+8, 2t+9) to the physical k (32t + 16h + 4s, +1) /
// (32t + 16h + 4s + 2, +3) for both operands, so the reduction is exact up to
// FP32 summation order.
#include "lawam_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace lawam {
namespace {

constexpr int kWarps = 4;
constexpr int kRows = 32;            // two m16 tiles

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One E2M1 pair (a byte) times a block scale (FP16 pair) -> packed FP16 pair (exact).
__device__ __forceinline__ uint32_t e2m1x2_f16x2(uint32_t byte, __half2 scale) {
    uint32_t h2;
    asm("{\n .reg .b8 b;\n cvt.u8.u32 b, %1;\n cvt.rn.f16x2.e2m1x2 %0, b;\n}\n" : "=r"(h2) : "r"(byte));
    const __half2 v = __hmul2(*reinterpret_cast<const __half2*>(&h2), scale);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldg_u16(const void* p) {
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];\n" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float ue4m3(uint32_t byte) {
    const uint8_t b = static_cast<uint8_t>(byte);
    return static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&b));
}

template <int BN, int KC, bool PDL>
__global__ void __launch_bounds__(kWarps * 32)
w4a16_skinny_kernel(const __nv_bfloat16* __restrict__ A, const uint8_t* __restrict__ W,
                    const uint8_t* __restrict__ sf, float* __restrict__ partials, int M, int N, int K) {
    constexpr int kTiles = BN / (kWarps * 8);   // n8 tiles per warp
    constexpr int kBlocks = KC / 128;           // 128-wide K blocks per CTA
    constexpr int kLd = KC + 8;                 // padded smem row (fp16)
    __shared__ __align__(16) __half sa[kRows * kLd];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int n_cta = blockIdx.x * BN;
    const int split = blockIdx.y;
    const int k_cta = split * KC;

    // weight slice + block scales -> registers (before the dependency wait)
    uint4 wreg[kTiles][kBlocks];
    uint32_t sreg[kTiles][kBlocks];
#pragma unroll
    for (int j = 0; j < kTiles; ++j) {
        const int n = n_cta + (warp * kTiles + j) * 8 + g;
        const uint8_t* wr = W + static_cast<size_t>(n) * (K / 2) + (k_cta + 32 * t) / 2;
        const uint8_t* sr = sf + static_cast<size_t>(n) * (K / 16) + (k_cta + 32 * t) / 16;
#pragma unroll
        for (int b = 0; b < kBlocks; ++b) {
            wreg[j][b] = ldg_stream(wr + 64 * b);
            sreg[j][b] = ldg_u16(sr + 8 * b);
        }
    }
    if constexpr (PDL) {
        asm volatile("griddepcontrol.launch_dependents;\n" ::);
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
    }
    // activation chunk [kRows, KC] -> shared (rows >= M zero)
    for (int i = threadIdx.x; i < kRows * (KC / 8); i += kWarps * 32) {
        const int r = i / (KC / 8), c = (i % (KC / 8)) * 8;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < M) {
            // BF16 -> FP16 (exact for the DiT's |x| < 65504 above the FP16 subnormal range)
            const uint4 u = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(r) * K + k_cta + c);
            const uint32_t uw[4] = {u.x, u.y, u.z, u.w};
            uint32_t hw[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const __half2 h = __float22half2_rn(__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uw[q])));
                hw[q] = *reinterpret_cast<const uint32_t*>(&h);
            }
            v = make_uint4(hw[0], hw[1], hw[2], hw[3]);
        }
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
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            // this lane's 16 consecutive k of the half, for rows g, g+8, g+16, g+24
            uint32_t ar[4][8];
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const __half* p = sa + (g + 8 * rr) * kLd + 128 * b + 32 * t + 16 * h;
                const uint4 v0 = *reinterpret_cast<const uint4*>(p);
                const uint4 v1 = *reinterpret_cast<const uint4*>(p + 8);
                ar[rr][0] = v0.x; ar[rr][1] = v0.y; ar[rr][2] = v0.z; ar[rr][3] = v0.w;
                ar[rr][4] = v1.x; ar[rr][5] = v1.y; ar[rr][6] = v1.z; ar[rr][7] = v1.w;
            }
#pragma unroll
            for (int j = 0; j < kTiles; ++j) {
                // the half's 16 k = 8 E2M1 pairs = word pair (2h, 2h + 1), scale byte h
                const uint32_t w0 = h ? wreg[j][b].z : wreg[j][b].x;
                const uint32_t w1 = h ? wreg[j][b].w : wreg[j][b].y;
                const __half2 sc = __float2half2_rn(ue4m3(sreg[j][b] >> (8 * h)));
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    // physical k 16h + 4s + {0,1} (b0) and {2,3} (b1): bytes 2s, 2s+1 of the half
                    const uint32_t word = s < 2 ? w0 : w1;
                    const uint32_t b0 = e2m1x2_f16x2((word >> (16 * (s & 1))) & 0xffu, sc);
                    const uint32_t b1 = e2m1x2_f16x2((word >> (16 * (s & 1) + 8)) & 0xffu, sc);
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
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
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int r0 = 16 * mt + g, r1 = r0 + 8;
            if (r0 < M)
                *reinterpret_cast<float2*>(out + static_cast<size_t>(r0) * N + n) =
                    make_float2(acc[j][mt][0], acc[j][mt][1]);
            if (r1 < M)
                *reinterpret_cast<float2*>(out + static_cast<size_t>(r1) * N + n) =
                    make_float2(acc[j][mt][2], acc[j][mt][3]);
        }
    }
}

template <int BN, int KC>
int launch(const __nv_bfloat16* A, const uint8_t* W, const uint8_t* sf, float* partials, int M, int N, int K,
           bool pdl, cudaStream_t stream) {
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
        return static_cast<int>(cudaLaunchKernelEx(&cfg, w4a16_skinny_kernel<BN, KC, true>, A, W, sf, partials, M, N, K));
    return static_cast<int>(cudaLaunchKernelEx(&cfg, w4a16_skinny_kernel<BN, KC, false>, A, W, sf, partials, M, N, K));
}

}  // namespace

// Tile configurations: (BN, KC).
static const int kCfg[][2] = {{32, 256}, {32, 512}, {64, 256}, {64, 512}, {128, 256}, {32, 1024}};

int w4a16_skinny_config_count() { return static_cast<int>(sizeof(kCfg) / sizeof(kCfg[0])); }

int w4a16_skinny_config_k_chunk(int cfg) {
    return (cfg >= 0 && cfg < w4a16_skinny_config_count()) ? kCfg[cfg][1] : 0;
}

int w4a16_skinny_gemm(const __nv_bfloat16* A, const uint8_t* W, const uint8_t* sf, float* partials, int M, int N,
                      int K, int cfg, bool pdl, cudaStream_t stream) {
    if (cfg < 0 || cfg >= w4a16_skinny_config_count() || M < 1 || M > kRows || A == nullptr || W == nullptr ||
        sf == nullptr || partials == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    const int bn = kCfg[cfg][0], kc = kCfg[cfg][1];
    if (N % bn != 0 || K % kc != 0) return static_cast<int>(cudaErrorInvalidValue);
    switch (cfg) {
        case 0: return launch<32, 256>(A, W, sf, partials, M, N, K, pdl, stream);
        case 1: return launch<32, 512>(A, W, sf, partials, M, N, K, pdl, stream);
        case 2: return launch<64, 256>(A, W, sf, partials, M, N, K, pdl, stream);
        case 3: return launch<64, 512>(A, W, sf, partials, M, N, K, pdl, stream);
        case 4: return launch<128, 256>(A, W, sf, partials, M, N, K, pdl, stream);
        default: return launch<32, 1024>(A, W, sf, partials, M, N, K, pdl, stream);
    }
}

}  // namespace lawam
}  // namespace flash_rt

// SPDX-License-Identifier: Apache-2.0
//
// Small-M block-scaled NVFP4 GEMM on sm_120a. See the header.
//
// One CTA computes a ROWS x BN output tile over the full K, streaming
// 128-deep K chunks through a double-buffered shared-memory A tile. The MMA
// and scale-factor fragment layouts are the repository's skinny-NVFP4 ones
// (mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64);
// the weights use the skinny pack_weight layout and A/SFA use the prefill
// layout emitted by the attention FP4 epilogue.
//
// Tile choice: ROWS=96 makes the text o-projection (M=692, N=2560) 8 x 20 =
// 160 CTAs, i.e. 4.44 waves at 89% utilization, against 120 CTAs / 3.33
// waves / 83% for the 128-row configured tile. The extra weight passes
// (8 vs 6) are L2-resident.

#include "molmo_smallm_gemm_sm120.cuh"
#include "gemm/fp4/nvfp4_skinny_common.cuh"

#include <cstdint>

namespace flash_rt {
namespace kernels {
namespace {

using namespace flash_rt::nvfp4_skinny;

__device__ __forceinline__ float b2f(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

constexpr int ROWS = 96;        // 6 x m16
constexpr int BN = 128;         // 16 x n8
constexpr int NW = 8;           // warps
constexpr int MT = ROWS / 16;   // 6
constexpr int NT = BN / 8;      // 16
constexpr int NTW = NT / NW;    // 2
constexpr int THREADS = NW * 32;
constexpr int ASTR = 64 + 16;   // packed A bytes per row (128 K) + pad
constexpr int SSTR = 8 + 4;     // scale bytes per row (8 blocks) + pad

__device__ __forceinline__ void cp_async_n(void* dst, const void* src,
                                           bool valid, int bytes) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const int n = valid ? bytes : 0;
    if (bytes == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
                     "l"(src), "r"(n));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d),
                     "l"(src), "r"(n));
}

__device__ __forceinline__ size_t sf_index(int row, int blk, int nb4) {
    return (static_cast<size_t>(row >> 7) * nb4 + (blk >> 2)) * 512 +
           (row & 31) * 16 + ((row >> 5) & 3) * 4 + (blk & 3);
}

__global__ void __launch_bounds__(THREADS, 2)
smallm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ SFA,
              const uint8_t* __restrict__ W, const uint8_t* __restrict__ SFB,
              const __nv_bfloat16* __restrict__ bias,
              const __nv_bfloat16* __restrict__ resid,
              __nv_bfloat16* __restrict__ D, int M, int N, int K,
              const float* __restrict__ alpha_p, float beta) {
    __shared__ __align__(16) uint8_t sA[2][ROWS * ASTR];
    __shared__ __align__(16) uint8_t sS[2][ROWS * SSTR];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int t0 = lane & 3, t1 = lane >> 2;
    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * ROWS;
    const int nb4 = (K / 16 + 3) / 4;
    const int kch = K / 128;

    auto stage = [&](int c, int buf) {
        uint8_t* Ab = sA[buf];
        uint8_t* Sb = sS[buf];
        for (int cid = tid; cid < ROWS * 4; cid += THREADS) {
            const int row = cid >> 2, part = cid & 3;
            const bool ok = (m0 + row) < M;
            cp_async_n(Ab + row * ASTR + part * 16,
                       A + (size_t)(ok ? m0 + row : 0) * (K / 2) + c * 64 +
                           part * 16,
                       ok, 16);
        }
        for (int q = tid; q < ROWS * 2; q += THREADS) {
            const int row = q >> 1, s = q & 1;
            const bool ok = (m0 + row) < M;
            const int gr = ok ? m0 + row : 0;
            cp_async_n(Sb + row * SSTR + s * 4,
                       SFA + sf_index(gr, 8 * c + 4 * s, nb4), ok, 4);
        }
    };

    // prologue: chunk 0 and the first weight tile set
    stage(0, 0);
    asm volatile("cp.async.commit_group;\n" ::);
    uint4 w[NTW];
    uint2 sb[NTW];
#pragma unroll
    for (int t = 0; t < NTW; ++t) {
        const size_t tile =
            (size_t)(n0 / 8 + warp * NTW + t) * kch + 0;
        w[t] = ldg_stream(W + (tile * 32 + lane) * 16);
        sb[t] = ldg_stream8(SFB + (tile * 8 + t1) * 8);
    }

    float acc[MT][NTW][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int t = 0; t < NTW; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[mt][t][i] = 0.f;

    const int sau = (lane & 1) * 8 + (lane >> 2);

    for (int c = 0; c < kch; ++c) {
        const int cur = c & 1, nxt = cur ^ 1;
        uint4 wn[NTW];
        uint2 sbn[NTW];
        if (c + 1 < kch) {
            stage(c + 1, nxt);
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        if (c + 1 < kch) {
#pragma unroll
            for (int t = 0; t < NTW; ++t) {
                const size_t tile =
                    (size_t)(n0 / 8 + warp * NTW + t) * kch + (c + 1);
                wn[t] = ldg_stream(W + (tile * 32 + lane) * 16);
                sbn[t] = ldg_stream8(SFB + (tile * 8 + t1) * 8);
            }
        }

        const uint8_t* Ab = sA[cur];
        const uint8_t* Sb = sS[cur];
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const uint8_t* ra =
                    Ab + (mt * 16 + t1) * ASTR + s * 32 + t0 * 4;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(ra);
                const uint32_t a1 =
                    *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(ra + 16);
                const uint32_t a3 =
                    *reinterpret_cast<const uint32_t*>(ra + 8 * ASTR + 16);
                const uint32_t sfa = *reinterpret_cast<const uint32_t*>(
                    Sb + (mt * 16 + sau) * SSTR + s * 4);
#pragma unroll
                for (int t = 0; t < NTW; ++t) {
                    const uint32_t b0 = s == 0 ? w[t].x : w[t].z;
                    const uint32_t b1 = s == 0 ? w[t].y : w[t].w;
                    const uint32_t sfb = s == 0 ? sb[t].x : sb[t].y;
                    mma_nvf4(acc[mt][t], a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
        }
        if (c + 1 < kch) {
#pragma unroll
            for (int t = 0; t < NTW; ++t) { w[t] = wn[t]; sb[t] = sbn[t]; }
        }
        __syncthreads();
    }

    const float alpha = *alpha_p;
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int t = 0; t < NTW; ++t) {
            const int col = n0 + (warp * NTW + t) * 8 + 2 * t0;
            const int row0 = m0 + mt * 16 + t1;
            const uint32_t bv =
                *reinterpret_cast<const uint32_t*>(bias + col);
            const __nv_bfloat162 bb =
                *reinterpret_cast<const __nv_bfloat162*>(&bv);
            if (row0 < M) {
                float v0 = acc[mt][t][0] * alpha + b2f(bb.x);
                float v1 = acc[mt][t][1] * alpha + b2f(bb.y);
                if (beta != 0.f && resid != nullptr) {
                    const uint32_t rv = *reinterpret_cast<const uint32_t*>(
                        resid + (size_t)row0 * N + col);
                    const __nv_bfloat162 rb =
                        *reinterpret_cast<const __nv_bfloat162*>(&rv);
                    v0 += b2f(rb.x);
                    v1 += b2f(rb.y);
                }
                const __nv_bfloat162 o = __floats2bfloat162_rn(v0, v1);
                *reinterpret_cast<uint32_t*>(D + (size_t)row0 * N + col) =
                    *reinterpret_cast<const uint32_t*>(&o);
            }
            if (row0 + 8 < M) {
                float v0 = acc[mt][t][2] * alpha + b2f(bb.x);
                float v1 = acc[mt][t][3] * alpha + b2f(bb.y);
                if (beta != 0.f && resid != nullptr) {
                    const uint32_t rv = *reinterpret_cast<const uint32_t*>(
                        resid + (size_t)(row0 + 8) * N + col);
                    const __nv_bfloat162 rb =
                        *reinterpret_cast<const __nv_bfloat162*>(&rv);
                    v0 += b2f(rb.x);
                    v1 += b2f(rb.y);
                }
                const __nv_bfloat162 o = __floats2bfloat162_rn(v0, v1);
                *reinterpret_cast<uint32_t*>(D + (size_t)(row0 + 8) * N +
                                            col) =
                    *reinterpret_cast<const uint32_t*>(&o);
            }
        }
    }
}

}  // namespace

int molmo_smallm_gemm_bias_res_bf16out(
    const void* A_packed, const void* SFA, const void* B_packed,
    const void* SFB, const void* bias, const void* residual, void* D,
    int M, int N, int K, const void* alpha, float beta,
    cudaStream_t stream) {
    if (M < 1 || N < 1 || K < 128 || N % BN != 0 || K % 128 != 0 ||
        A_packed == nullptr || SFA == nullptr || B_packed == nullptr ||
        SFB == nullptr || bias == nullptr || D == nullptr || alpha == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    const dim3 grid(N / BN, (M + ROWS - 1) / ROWS);
    smallm_kernel<<<grid, THREADS, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(A_packed),
        reinterpret_cast<const uint8_t*>(SFA),
        reinterpret_cast<const uint8_t*>(B_packed),
        reinterpret_cast<const uint8_t*>(SFB),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<__nv_bfloat16*>(D), M, N, K,
        reinterpret_cast<const float*>(alpha), beta);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace kernels
}  // namespace flash_rt

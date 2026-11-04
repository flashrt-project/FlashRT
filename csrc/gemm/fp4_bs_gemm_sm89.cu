// SPDX-License-Identifier: Apache-2.0
//
// Native Ada (sm_89) NVFP4-weight / FP8-activation block-128 scaled GEMM.
// Header: fp4_bs_gemm_sm89.cuh, device body: fp4_bs_gemm_device.cuh.
//
// The tile grid mirrors the FP8 sibling (fp8_block128_gemm_mma_sm89.cu) so the
// two can be swapped at a call site without re-tuning the dispatcher: the FP8
// tile that M<=32 selects is 16x64x128_w4, and this file's default is the same
// geometry with the NVFP4 staging buffer added.

#include "fp4_bs_gemm_sm89.cuh"
#include "fp4_bs_gemm_device.cuh"

#include <stdexcept>

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

namespace {

template <int BM, int BN, int W, int STAGES, int MIN_BLK, bool EXACT = false,
          bool TILE_MAJOR = false, bool SF128 = false>
int launch_(const void* A, const void* Bq, const void* Bsf, const float* gscale,
            const float* act_scale, const float* w_scale, void* D,
            int M, int N, int K, cudaStream_t s)
{
    constexpr int BK = 128;
    constexpr int SCALE_KTILE = 8;
    int grid_m = (M + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    dim3 grid(grid_m, grid_n, 1);
    dim3 block(W * 32, 1, 1);
    // A cp.async stages (BM*BK each) + raw NVFP4 B stages (BN*BK/2 each) +
    // one expanded FP8 B tile (BN*BK) + staged scale tiles.
    int smem_bytes = STAGES * (BM + BN / 2) * BK
                   + BN * BK
                   + (BM * SCALE_KTILE + SCALE_KTILE) * (int)sizeof(float)
                   + 16 * 4;   // fp16 e2m1 codebook, one entry per bank
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            (const void*)&fp4_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK, EXACT,
                                             TILE_MAJOR, SF128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    fp4_bs_gemm_kernel<BM, BN, W, STAGES, MIN_BLK, EXACT, TILE_MAJOR, SF128>
        <<<grid, block, smem_bytes, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const uint8_t*>(Bq),
        reinterpret_cast<const uint8_t*>(Bsf),
        gscale, act_scale, w_scale,
        reinterpret_cast<__nv_bfloat16*>(D),
        M, N, K);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 1;
}

void check_shape(int N, int K) {
    if (N % 128)
        throw std::runtime_error("fp4_bs_gemm_sm89 requires N multiple of 128");
    if (K % 128)
        throw std::runtime_error("fp4_bs_gemm_sm89 requires K multiple of 128");
}

}  // namespace

#define DEFINE_FP4(NAME, BM, BN, W, S, MB)                                    \
  int NAME(const void* A, const void* Bq, const void* Bsf,                    \
           const float* gscale, const float* act_scale, const float* w_scale, \
           void* D, int M, int N, int K, cudaStream_t s) {                    \
    check_shape(N, K);                                                        \
    if (M == BM && (N % BN) == 0 && (K % 128) == 0)                           \
      return launch_<BM, BN, W, S, MB, true>(A, Bq, Bsf, gscale, act_scale,   \
                                             w_scale, D, M, N, K, s);         \
    return launch_<BM, BN, W, S, MB, false>(A, Bq, Bsf, gscale, act_scale,    \
                                            w_scale, D, M, N, K, s);          \
  }

// Tile-major packed-weight variants (nvfp4_pack_tile_major.cuh). Same
// numerics, same tiles, same stages: only the global B address changes, so the
// FP4 payload streams at the device ceiling instead of 188 GB/s. Only the
// production M<=32 shapes are needed; the EXACT=false arm keeps the dispatcher
// usable for odd shapes.
#define DEFINE_FP4_TM(NAME, BM, BN, W, S, MB)                                 \
  int NAME(const void* A, const void* Bq, const void* Bsf,                    \
           const float* gscale, const float* act_scale, const float* w_scale, \
           void* D, int M, int N, int K, cudaStream_t s) {                    \
    check_shape(N, K);                                                        \
    if (M == BM && (N % BN) == 0 && (K % 128) == 0)                           \
      return launch_<BM, BN, W, S, MB, true, true>(A, Bq, Bsf, gscale,        \
                                                   act_scale, w_scale, D, M,  \
                                                   N, K, s);                  \
    return launch_<BM, BN, W, S, MB, false, true>(A, Bq, Bsf, gscale,         \
                                                  act_scale, w_scale, D, M,   \
                                                  N, K, s);                   \
  }

#define DEFINE_FP4_SF128(NAME, BM, BN, W, S, MB)                              \
  int NAME(const void* A, const void* Bq, const void* Bsf,                    \
           const float* gscale, const float* act_scale, const float* w_scale, \
           void* D, int M, int N, int K, cudaStream_t s) {                    \
    check_shape(N, K);                                                        \
    if (M == BM && (N % BN) == 0 && (K % 128) == 0)                           \
      return launch_<BM, BN, W, S, MB, true, false, true>(A, Bq, Bsf,         \
              gscale, act_scale, w_scale, D, M, N, K, s);                     \
    return launch_<BM, BN, W, S, MB, false, false, true>(A, Bq, Bsf,          \
            gscale, act_scale, w_scale, D, M, N, K, s);                       \
  }

DEFINE_FP4(fp4_bs_gemm_sm89_x64,      16,  64, 4, 2, 4)
DEFINE_FP4(fp4_bs_gemm_sm89_x64_s1,   16,  64, 4, 1, 4)
DEFINE_FP4(fp4_bs_gemm_sm89_x128,     16, 128, 4, 2, 2)
DEFINE_FP4(fp4_bs_gemm_sm89_x32,      16,  32, 2, 2, 8)
DEFINE_FP4(fp4_bs_gemm_sm89_x64_s4,   16,  64, 4, 4, 3)
DEFINE_FP4(fp4_bs_gemm_sm89_x64_s3,   16,  64, 4, 3, 3)
DEFINE_FP4(fp4_bs_gemm_sm89_m64,      64,  64, 4, 2, 2)
DEFINE_FP4(fp4_bs_gemm_sm89_m64n128,  64, 128, 4, 2, 2)

DEFINE_FP4_TM(fp4_bs_gemm_sm89_x64_tm,      16,  64, 4, 2, 4)
DEFINE_FP4_TM(fp4_bs_gemm_sm89_x64_s1_tm,   16,  64, 4, 1, 4)
DEFINE_FP4_TM(fp4_bs_gemm_sm89_x128_tm,     16, 128, 4, 2, 2)
DEFINE_FP4_TM(fp4_bs_gemm_sm89_x32_tm,      16,  32, 2, 2, 8)

// Per-128 SF variants (SF128=true). Same payload/tiles; the kernel reads one
// SF byte per 128-K block from the [N/128, K/128, 128] image.
DEFINE_FP4_SF128(fp4_bs_gemm_sm89_x64_sf128,    16,  64, 4, 2, 4)
DEFINE_FP4_SF128(fp4_bs_gemm_sm89_x64_s1_sf128, 16,  64, 4, 1, 4)

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt

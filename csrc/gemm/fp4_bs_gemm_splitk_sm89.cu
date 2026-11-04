// SPDX-License-Identifier: Apache-2.0
//
// Launcher for the split-K NVFP4-weight GEMM. The tile is the same 16x64x128 /
// 4-warp / 2-stage geometry as the unsplit dispatcher's M<=32 tile; only the
// grid gains a z dimension. `splits` must divide ceil(K/128) so every CTA owns
// a non-empty k range and every partial is written.

#include "fp4_bs_gemm_splitk_sm89.cuh"
#include "fp4_bs_gemm_splitk_device.cuh"

namespace flash_rt {
namespace gemm {
namespace block128_sm89 {

namespace {

template <bool TILE_MAJOR, bool SF128 = false>
int splitk_launch_(const void* A, const void* Bq, const void* Bsf,
                   const float* gscale, const float* act_scale,
                   const float* w_scale, void* D, void* Dpart,
                   int M, int N, int K, int splits, cudaStream_t s,
                   const void* residual = nullptr, const void* bias = nullptr) {
    using namespace flash_rt::gemm::block128_sm89_splitk;
    constexpr int BM = 16, BN = 64, W = 4, ST = 2, MB = 4;
    if (M != BM) return -3;                     // EXACT=true requires M == BLOCK_M
    const int k_iters = (K + 127) / 128;
    if (splits <= 0 || (k_iters % splits) != 0) return -2;
    const int smem = ST * (BM + BN / 2) * 128 + BN * 128
                   + (BM * 8 + 8) * (int)sizeof(float) + 16 * 4 + 256 * 4;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            (const void*)&fp4_bs_gemm_splitk_kernel<BM, BN, W, ST, MB, true,
                                                     TILE_MAJOR, SF128>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, splits);
    dim3 block(W * 32, 1, 1);
    fp4_bs_gemm_splitk_kernel<BM, BN, W, ST, MB, true, TILE_MAJOR, SF128>
        <<<grid, block, smem, s>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A),
        reinterpret_cast<const uint8_t*>(Bq),
        reinterpret_cast<const uint8_t*>(Bsf),
        gscale, act_scale, w_scale,
        reinterpret_cast<__nv_bfloat16*>(D), reinterpret_cast<float*>(Dpart),
        M, N, K, splits);
    const long total = (long)M * N;
    fp4_splitk_reduce_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(
        reinterpret_cast<const float*>(Dpart),
        reinterpret_cast<__nv_bfloat16*>(D), M, N, splits,
        reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(residual)),
        reinterpret_cast<const __nv_bfloat16*>(bias));
    return (int)cudaGetLastError();
}

}  // namespace

int fp4_bs_gemm_splitk_sm89(const void* A, const void* Bq, const void* Bsf,
                            const float* gscale, const float* act_scale,
                            const float* w_scale, void* D, void* Dpart,
                            int M, int N, int K, int splits, cudaStream_t s) {
    return splitk_launch_<false>(A, Bq, Bsf, gscale, act_scale, w_scale, D,
                                 Dpart, M, N, K, splits, s);
}

int fp4_bs_gemm_splitk_sm89_tm(const void* A, const void* Bq, const void* Bsf,
                               const float* gscale, const float* act_scale,
                               const float* w_scale, void* D, void* Dpart,
                               int M, int N, int K, int splits,
                               cudaStream_t s) {
    return splitk_launch_<true>(A, Bq, Bsf, gscale, act_scale, w_scale, D,
                                Dpart, M, N, K, splits, s);
}

int fp4_bs_gemm_splitk_sm89_sf128(const void* A, const void* Bq, const void* Bsf,
                                  const float* gscale, const float* act_scale,
                                  const float* w_scale, void* D, void* Dpart,
                                  int M, int N, int K, int splits,
                                  cudaStream_t s) {
    return splitk_launch_<false, true>(A, Bq, Bsf, gscale, act_scale, w_scale, D,
                                       Dpart, M, N, K, splits, s);
}

int fp4_bs_gemm_splitk_sm89_tm_sf128_res(
    const void* A, const void* Bq, const void* Bsf, const float* gscale,
    const float* act_scale, const float* w_scale, void* Dpart,
    const void* residual, const void* bias, int M, int N, int K, int splits,
    cudaStream_t s) {
    return splitk_launch_<true, true>(A, Bq, Bsf, gscale, act_scale, w_scale,
                                      nullptr, Dpart, M, N, K, splits, s,
                                      residual, bias);
}

int fp4_bs_gemm_splitk_sm89_tm_sf128(const void* A, const void* Bq, const void* Bsf,
                                     const float* gscale, const float* act_scale,
                                     const float* w_scale, void* D, void* Dpart,
                                     int M, int N, int K, int splits,
                                     cudaStream_t s) {
    return splitk_launch_<true, true>(A, Bq, Bsf, gscale, act_scale, w_scale, D,
                                      Dpart, M, N, K, splits, s);
}

}  // namespace block128_sm89
}  // namespace gemm
}  // namespace flash_rt

// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini MoE grouped NVFP4 W4A16 GEMV for sm_120. See header.
//
// The block-scaled mma + swizzled-SF decode helpers below are copied
// verbatim from fp4_w4a4_mma_sm120.cu (file-local anonymous namespace, so
// they cannot be shared without modifying that validated file). The grouped
// kernel body is its full_n_kernel with per-slot base-pointer arithmetic
// added on top; correctness is pinned by a cos=1.0 check vs the per-expert
// fp4_w4a4_mma_sm120_full_n_bf16out loop.

#include "kernels/moe_grouped_gemv_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

namespace flash_rt {
namespace gemm {

namespace {

using AtomType = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t,
    cutlass::float_e2m1_t,
    float,
    cutlass::float_ue4m3_t,
    16>;

constexpr int G_WARPS_PER_BLOCK = 1;
constexpr int G_THREADS_PER_BLOCK = G_WARPS_PER_BLOCK * 32;
constexpr int G_COLS_PER_WARP = 8;
constexpr int G_COLS_PER_BLOCK = G_WARPS_PER_BLOCK * G_COLS_PER_WARP;  // 8

__device__ __forceinline__ uint32_t fast_load_a(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  int row_off = ((reg_idx & 1) ? (t1 + 8) : t1) * 32;
  int col_off = t0 * 4 + ((reg_idx >> 1) & 1) * 16;
  return *reinterpret_cast<const uint32_t*>(sA + row_off + col_off);
}

__device__ __forceinline__ uint32_t fast_load_b(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  int col_off = t0 * 4 + reg_idx * 16;
  return *reinterpret_cast<const uint32_t*>(sB + t1 * 32 + col_off);
}

__device__ __forceinline__ uint32_t fast_load_sfa(
    const uint8_t* sSFA, int unique_row) {
  return *reinterpret_cast<const uint32_t*>(sSFA + unique_row * 4);
}

__device__ __forceinline__ uint32_t fast_load_sfb(
    const uint8_t* sSFB, int unique_col) {
  return *reinterpret_cast<const uint32_t*>(sSFB + unique_col * 4);
}

__device__ __forceinline__ void cp_async_4(
    uint8_t* smem_dst, const uint8_t* gmem_src) {
  uint32_t smem_int = __cvta_generic_to_shared(smem_dst);
  asm volatile(
      "cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n"
      :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group(int N) {
  if (N == 0) {
    asm volatile("cp.async.wait_group 0;\n" ::);
  } else if (N == 1) {
    asm volatile("cp.async.wait_group 1;\n" ::);
  } else {
    asm volatile("cp.async.wait_all;\n" ::);
  }
}

// full_n_kernel body + per-slot (grid.y) base-pointer indexing.
__global__ void grouped_n_kernel(
    const uint8_t* __restrict__ A_stack,
    const uint8_t* __restrict__ B_stack,
    const uint8_t* __restrict__ SFA_stack,
    const uint8_t* __restrict__ SFB_stack,
    __nv_bfloat16* __restrict__ D,
    const float* __restrict__ alpha_stack,
    const int* __restrict__ expert_idx,
    int N, int K,
    long a_stride, long sfa_stride, long w_stride, long sfb_stride) {
  // ── Per-slot base pointers (the only addition over full_n_kernel) ──
  int slot = blockIdx.y;
  int e = expert_idx[slot];
  const uint8_t* A_packed = A_stack + static_cast<long>(slot) * a_stride;
  const uint8_t* SFA = SFA_stack + static_cast<long>(slot) * sfa_stride;
  const uint8_t* B_packed = B_stack + static_cast<long>(e) * w_stride;
  const uint8_t* SFB = SFB_stack + static_cast<long>(e) * sfb_stride;
  __nv_bfloat16* D_e = D + static_cast<long>(slot) * N;
  float alpha = alpha_stack[e];

  __shared__ alignas(16) uint8_t s_A[2][16 * 32];
  __shared__ alignas(16) uint8_t s_SFA[2][16 * 4];
  __shared__ alignas(16) uint8_t s_B_all[2][G_WARPS_PER_BLOCK * 8 * 32];
  __shared__ alignas(16) uint8_t s_SFB_all[2][G_WARPS_PER_BLOCK * 8 * 4];

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  int block_n_off = blockIdx.x * G_COLS_PER_BLOCK;
  int my_n_off = block_n_off + warp * G_COLS_PER_WARP;
  if (my_n_off >= N) return;

  int t0 = lane & 3;
  int t1 = lane >> 2;
  int sfa_unique_row = (lane & 1) * 8 + (lane >> 2);
  int sfb_unique_col = lane >> 2;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  const int K_iters = K / 64;
  const int K_half = K / 2;

  if (lane < 16) {
    int row = lane;
    if (row >= 1 && row <= 15) {
      int4* a0_v = reinterpret_cast<int4*>(s_A[0]);
      int4* a1_v = reinterpret_cast<int4*>(s_A[1]);
      int4 z; z.x = 0; z.y = 0; z.z = 0; z.w = 0;
      a0_v[row * 2 + 0] = z; a0_v[row * 2 + 1] = z;
      a1_v[row * 2 + 0] = z; a1_v[row * 2 + 1] = z;
    }
  }
  if (lane < 4) {
    for (int i = 4 + lane; i < 64; i += 4) {
      s_SFA[0][i] = 0;
      s_SFA[1][i] = 0;
    }
  }

  const int K_blocks = K / 16;
  const int n_col_super = (K_blocks + 3) / 4;

  auto issue_async_load = [&](int buf, int kt) {
    int byte_off = kt * 32;
    if (lane < 8) {
      cp_async_4(s_A[buf] + lane * 4, A_packed + byte_off + lane * 4);
    }
    if (lane == 0) {
      cp_async_4(s_SFA[buf] + 0, SFA + kt * 512);
    }
    {
      uint8_t* my_s_B = s_B_all[buf] + warp * (8 * 32);
      for (int c = 0; c < 2; ++c) {
        int chunk = lane + c * 32;
        int col = chunk >> 3;
        int off = chunk & 7;
        cp_async_4(
            my_s_B + chunk * 4,
            B_packed + (my_n_off + col) * K_half + byte_off + off * 4);
      }
    }
    if (lane < 8) {
      uint8_t* my_s_SFB = s_SFB_all[buf] + warp * (8 * 4);
      int col = my_n_off + lane;
      int rb = col >> 7;
      int ri = col & 127;
      int super_idx = rb * n_col_super + kt;
      int inner_base = (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
      cp_async_4(
          my_s_SFB + lane * 4,
          SFB + super_idx * 512 + inner_base);
    }
  };

  issue_async_load(0, 0);
  cp_async_commit_group();
  if (K_iters > 1) {
    issue_async_load(1, 1);
    cp_async_commit_group();
  }

  for (int kt = 0; kt < K_iters; ++kt) {
    int curr_buf = kt & 1;

    if (kt + 1 < K_iters) {
      cp_async_wait_group(1);
    } else {
      cp_async_wait_group(0);
    }
    __syncwarp();

    uint32_t a0 = fast_load_a(s_A[curr_buf], t0, t1, 0);
    uint32_t a1 = fast_load_a(s_A[curr_buf], t0, t1, 1);
    uint32_t a2 = fast_load_a(s_A[curr_buf], t0, t1, 2);
    uint32_t a3 = fast_load_a(s_A[curr_buf], t0, t1, 3);
    uint8_t* my_s_B = s_B_all[curr_buf] + warp * (8 * 32);
    uint8_t* my_s_SFB = s_SFB_all[curr_buf] + warp * (8 * 4);
    uint32_t b0 = fast_load_b(my_s_B, t0, t1, 0);
    uint32_t b1 = fast_load_b(my_s_B, t0, t1, 1);
    uint32_t sfa = fast_load_sfa(s_SFA[curr_buf], sfa_unique_row);
    uint32_t sfb = fast_load_sfb(my_s_SFB, sfb_unique_col);

    float d0, d1, d2, d3;
    AtomType::fma(d0, d1, d2, d3,
                  a0, a1, a2, a3,
                  b0, b1,
                  c0, c1, c2, c3,
                  sfa, sfb);
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;

    if (kt + 2 < K_iters) {
      issue_async_load(curr_buf, kt + 2);
      cp_async_commit_group();
    }
  }

  int q = lane >> 2;
  int r = lane & 3;
  if (q == 0) {
    int col0 = my_n_off + r * 2;
    int col1 = col0 + 1;
    if (col0 < N) D_e[col0] = __float2bfloat16(c0 * alpha);
    if (col1 < N) D_e[col1] = __float2bfloat16(c1 * alpha);
  }
}

}  // namespace

int moe_grouped_gemv_sm120_bf16(
    const void*  A_stack,
    const void*  B_stack,
    void*        D,
    const void*  SFA_stack,
    const void*  SFB_stack,
    const void*  alpha_stack,
    const void*  expert_idx,
    int          E,
    int          N,
    int          K,
    long         a_stride,
    long         sfa_stride,
    long         w_stride,
    long         sfb_stride,
    cudaStream_t stream) {
  if (!A_stack || !B_stack || !D || !SFA_stack || !SFB_stack ||
      !alpha_stack || !expert_idx) return 1;
  if (K <= 0 || (K % 64) != 0) return 2;
  if (N <= 0 || (N % G_COLS_PER_BLOCK) != 0) return 3;
  if (E <= 0) return 4;

  dim3 block(G_THREADS_PER_BLOCK);
  dim3 grid((N + G_COLS_PER_BLOCK - 1) / G_COLS_PER_BLOCK, E);
  grouped_n_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A_stack),
      reinterpret_cast<const uint8_t*>(B_stack),
      reinterpret_cast<const uint8_t*>(SFA_stack),
      reinterpret_cast<const uint8_t*>(SFB_stack),
      reinterpret_cast<__nv_bfloat16*>(D),
      reinterpret_cast<const float*>(alpha_stack),
      reinterpret_cast<const int*>(expert_idx),
      N, K, a_stride, sfa_stride, w_stride, sfb_stride);
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt

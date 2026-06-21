// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini W16A16 dense GEMM (BF16 activation x BF16 weight), sm120.
//
// The experts-scope dense projections (full-attn q/k/v/o, GDN out_proj, shared
// expert, router) ran as `.float() @ .float().T` -- a full fp32/TF32 tensor-op
// that dominates the prefill profile. cuBLAS bf16 is faster but its split-K /
// heuristic accumulation is non-deterministic (run-to-run logit jitter flips
// near-tie argmaxes -> breaks the token-exact red line, since prefill seeds the
// decode state). This kernel runs the bf16 m16n8k16 tensor-core mma with fp32
// register accumulation in a single pass over K (no split-K, no atomics): it is
// deterministic and, because the accumulate is fp32, matches the .float() path
// argmax while running on full-rate bf16 tensor cores (~2x the TF32 op).
//
// Same multi-warp CTA tile as the W4A16 kernel (BM x BN, 4 warps), but the
// weight is read straight as bf16 into smem -- no fp4 dequant / SF / LUT.
//
// y[M,N] = x[M,K] @ W[N,K]^T, x/W bf16. All add-only.

#include "kernels/w16a16_gemm_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace gemm {

namespace {

constexpr int GM_BM = 64;          // CTA M tile
constexpr int GM_BN = 64;          // CTA N tile
constexpr int GM_BK = 16;          // mma K
constexpr int GM_KT = 64;          // K-tile per cp.async stage (4 mma chunks)
constexpr int GM_KSUB = GM_KT / GM_BK;
constexpr int GM_AS = GM_KT + 8;   // padded smem row stride (kills bank conflict)
constexpr int GM_WARPS = 4;
constexpr int GM_THREADS = GM_WARPS * 32;
constexpr int GM_WN = 2;           // warp grid along N
constexpr int GM_WRB = 2;          // m-blocks per warp (32 rows / 16)
constexpr int GM_WCB = 4;          // n-blocks per warp (32 cols / 8)

__device__ __forceinline__ void cp16(void* d, const void* s) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(d));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(a), "l"(s));
}
__device__ __forceinline__ void cpcommit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cpwait(int n) {
  if (n == 0) asm volatile("cp.async.wait_group 0;\n" ::);
  else asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void mma_m16n8k16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// grid = (ceil(N/BN), ceil(M/BM)). GM_THREADS threads/block.
__global__ __launch_bounds__(GM_THREADS) void w16a16_gemm_kernel(
    const __nv_bfloat16* __restrict__ X,   // (M, K) bf16
    const __nv_bfloat16* __restrict__ W,   // (N, K) bf16
    __nv_bfloat16* __restrict__ Y,         // (M, N) bf16
    float alpha, int M, int N, int K) {
  const int bm = blockIdx.y * GM_BM;
  const int bn = blockIdx.x * GM_BN;

  __shared__ __nv_bfloat16 sA[2][GM_BM * GM_AS];   // activation (bf16, 2-stage)
  __shared__ __nv_bfloat16 sW[2][GM_BN * GM_AS];   // weight     (bf16, 2-stage)

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp / GM_WN;
  const int warp_n = warp % GM_WN;

  float c[GM_WRB][GM_WCB][4];
#pragma unroll
  for (int i = 0; i < GM_WRB; ++i)
#pragma unroll
    for (int j = 0; j < GM_WCB; ++j)
      c[i][j][0] = c[i][j][1] = c[i][j][2] = c[i][j][3] = 0.f;

  const int n_tiles = K / GM_KT;

  auto load = [&](int buf, int kt) {
    const int k0 = kt * GM_KT;
    // sA: BM x KT bf16 via cp.async 16B (8 bf16). 64*64=4096 bf16 = 512 cp16.
#pragma unroll
    for (int j = 0; j < (GM_BM * GM_KT) / (GM_THREADS * 8); ++j) {
      int idx = (tid + j * GM_THREADS) * 8;
      int row = idx / GM_KT;
      int koff = idx % GM_KT;
      int gm = bm + row; if (gm >= M) gm = M - 1;
      cp16(&sA[buf][row * GM_AS + koff], &X[(size_t)gm * K + k0 + koff]);
    }
    // sW: BN x KT bf16 via cp.async 16B (8 bf16). 64*64=4096 bf16 = 512 cp16.
#pragma unroll
    for (int j = 0; j < (GM_BN * GM_KT) / (GM_THREADS * 8); ++j) {
      int idx = (tid + j * GM_THREADS) * 8;
      int col = idx / GM_KT;
      int koff = idx % GM_KT;
      int gn = bn + col; if (gn >= N) gn = N - 1;
      cp16(&sW[buf][col * GM_AS + koff], &W[(size_t)gn * K + k0 + koff]);
    }
  };

  load(0, 0); cpcommit();
  if (n_tiles > 1) { load(1, 1); cpcommit(); }

  const int r = lane >> 2;
  const int kk = (lane & 3) * 2;

  for (int kt = 0; kt < n_tiles; ++kt) {
    int cur = kt & 1;
    cpwait(kt + 1 < n_tiles ? 1 : 0);
    __syncthreads();

#pragma unroll
    for (int ksub = 0; ksub < GM_KSUB; ++ksub) {
      const int kb = ksub * GM_BK;
      // B fragment: thread owns column ncol = warp_n*32 + jb*8 + r, with k =
      // {kb+kk, kb+kk+1} (b0) and {kb+kk+8, kb+kk+9} (b1), read straight from
      // the bf16 weight tile.
      uint32_t bb0[GM_WCB], bb1[GM_WCB];
#pragma unroll
      for (int jb = 0; jb < GM_WCB; ++jb) {
        const int ncol = warp_n * 32 + jb * 8 + r;
        const __nv_bfloat16* wR = &sW[cur][ncol * GM_AS + kb + kk];
        bb0[jb] = *reinterpret_cast<const uint32_t*>(wR);
        bb1[jb] = *reinterpret_cast<const uint32_t*>(wR + 8);
      }
#pragma unroll
      for (int ib = 0; ib < GM_WRB; ++ib) {
        const int mrow = warp_m * 32 + ib * 16;
        const __nv_bfloat16* aR0 = &sA[cur][(mrow + r) * GM_AS + kb + kk];
        const __nv_bfloat16* aR1 = &sA[cur][(mrow + r + 8) * GM_AS + kb + kk];
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(aR0);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(aR1);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(aR0 + 8);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(aR1 + 8);
#pragma unroll
        for (int jb = 0; jb < GM_WCB; ++jb)
          mma_m16n8k16(c[ib][jb][0], c[ib][jb][1], c[ib][jb][2], c[ib][jb][3],
                       a0, a1, a2, a3, bb0[jb], bb1[jb]);
      }
    }
    __syncthreads();
    if (kt + 2 < n_tiles) { load(cur, kt + 2); cpcommit(); }
  }

  // Epilogue. C fragment: c0,c1 -> row (lane/4), col (lane%4)*2 + {0,1};
  //   c2,c3 -> row (lane/4)+8, same cols.
  const int cr = lane >> 2;
  const int cc = (lane & 3) * 2;
#pragma unroll
  for (int ib = 0; ib < GM_WRB; ++ib) {
    const int mrow = bm + warp_m * 32 + ib * 16;
#pragma unroll
    for (int jb = 0; jb < GM_WCB; ++jb) {
      const int ncol = bn + warp_n * 32 + jb * 8 + cc;
      int r0 = mrow + cr, r1 = mrow + cr + 8;
      if (ncol < N) {
        if (r0 < M) Y[(size_t)r0 * N + ncol] =
            __float2bfloat16(c[ib][jb][0] * alpha);
        if (r1 < M) Y[(size_t)r1 * N + ncol] =
            __float2bfloat16(c[ib][jb][2] * alpha);
      }
      if (ncol + 1 < N) {
        if (r0 < M) Y[(size_t)r0 * N + ncol + 1] =
            __float2bfloat16(c[ib][jb][1] * alpha);
        if (r1 < M) Y[(size_t)r1 * N + ncol + 1] =
            __float2bfloat16(c[ib][jb][3] * alpha);
      }
    }
  }
}

}  // namespace

int w16a16_gemm_sm120_bf16(
    const void* X, const void* W, void* Y,
    int M, int N, int K, float alpha, cudaStream_t stream) {
  if (!X || !W || !Y) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || (K % GM_KT) != 0) return 2;
  dim3 grid((N + GM_BN - 1) / GM_BN, (M + GM_BM - 1) / GM_BM);
  dim3 block(GM_THREADS);
  w16a16_gemm_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(X),
      reinterpret_cast<const __nv_bfloat16*>(W),
      reinterpret_cast<__nv_bfloat16*>(Y),
      alpha, M, N, K);
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt

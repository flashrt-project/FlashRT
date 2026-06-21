// SPDX-License-Identifier: Apache-2.0
//
// Nex-N2-mini W4A16 dense GEMM (BF16 activation x NVFP4 weight), sm120.
//
// The prefill dense projections were running as `.float() @ .float().T` -- a
// full fp32/TF32 tensor-op ~10x slower than fp4. The W4A4 fp4 GEMM is fast but
// the fp4 *activation* drops precision (the GDN in_proj is the red line). This
// kernel keeps the activation in BF16 (precise, red-line-safe) and reads the
// weight as 4-bit NVFP4 (4x less weight BW), dequantising it to BF16 in shared
// memory and running the bf16 m16n8k16 tensor-core mma. Multi-warp CTA tile
// (BM x BN, 4 warps) on the sm120 structure -- the activation rows + dequanted
// weight cols are loaded once into smem and shared across warps.
//
// y[M,N] = x[M,K] @ W[N,K]^T, x bf16, W NVFP4 (packed + swizzled UE4M3 SF +
// per-tensor alpha). All add-only.

#include "kernels/nexn2_w4a16_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace flash_rt {
namespace gemm {

namespace {

__device__ __constant__ float c_w4a16_ue4m3[256];

constexpr int GM_BM = 64;          // CTA M tile
constexpr int GM_BN = 64;          // CTA N tile
constexpr int GM_BK = 16;          // mma K
constexpr int GM_KT = 64;          // K-tile per cp.async stage (4 mma chunks)
constexpr int GM_KSUB = GM_KT / GM_BK;
constexpr int GM_AS = GM_KT + 8;   // padded sA row stride (kills A bank conflicts)
constexpr int GM_WARPS = 4;
constexpr int GM_THREADS = GM_WARPS * 32;
constexpr int GM_WN = 2;           // warp grid along N
constexpr int GM_WM = 16;          // rows per mma (m16n8k16)
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

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
  __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&v);
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
__global__ __launch_bounds__(GM_THREADS) void w4a16_gemm_kernel(
    const __nv_bfloat16* __restrict__ X,   // (M, K) bf16
    const uint8_t* __restrict__ W,         // (N, K/2) NVFP4 packed
    const uint8_t* __restrict__ SFB,       // swizzled UE4M3
    __nv_bfloat16* __restrict__ Y,         // (M, N) bf16
    float alpha, int M, int N, int K) {
  const int bm = blockIdx.y * GM_BM;
  const int bn = blockIdx.x * GM_BN;

  __shared__ __nv_bfloat16 sA[2][GM_BM * GM_AS];   // activation (bf16, 2-stage)
  __shared__ uint8_t sWq[2][GM_BN * (GM_KT / 2)];  // weight (NVFP4, 2-stage)
  __shared__ uint8_t sSFB[2][GM_BN * GM_KSUB];     // weight SF (2-stage)

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

  const int K_half = K >> 1;
  const int K_blocks = K >> 4;
  const int n_col_super = (K_blocks + 3) >> 2;
  const int n_tiles = K / GM_KT;
  const int KT_half = GM_KT / 2;        // packed bytes per col per tile

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
    // sWq: BN x KT_half bytes via cp.async 16B. 64*32=2048 B = 128 cp16.
#pragma unroll
    for (int j = 0; j < (GM_BN * KT_half) / (GM_THREADS * 16); ++j) {
      int idx = (tid + j * GM_THREADS) * 16;
      int col = idx / KT_half;
      int boff = idx % KT_half;
      int gn = bn + col; if (gn >= N) gn = N - 1;
      cp16(&sWq[buf][idx], &W[(size_t)gn * K_half + k0 / 2 + boff]);
    }
    // sSFB: BN cols x KSUB sub-blocks (4 contiguous SF bytes / col).
    if (tid < GM_BN) {
      int gn = bn + tid;
      if (gn < N) {
        int rb = gn >> 7, ri = gn & 127;
        int off = (rb * n_col_super + kt) * 512
                  + (ri & 31) * 16 + ((ri >> 5) & 3) * 4;
#pragma unroll
        for (int s = 0; s < GM_KSUB; ++s)
          sSFB[buf][tid * GM_KSUB + s] = __ldg(SFB + off + s);
      } else {
#pragma unroll
        for (int s = 0; s < GM_KSUB; ++s) sSFB[buf][tid * GM_KSUB + s] = 0;
      }
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

    // 4 mma sub-chunks (BK=16) over the loaded K-tile. The B fragment is
    // dequantised inline per-thread straight from sWq (no sW staging buffer):
    // each thread's b0/b1 are the 4 weight values it owns -- byte (ksub*8 +
    // lane%4) holds {w[K],w[K+1]} (cvt -> half2), byte +4 holds {w[K+8],w[K+9]}.
#pragma unroll
    for (int ksub = 0; ksub < GM_KSUB; ++ksub) {
      const int kb = ksub * GM_BK;
      const int byte0 = ksub * 8 + (kk >> 1);   // kk = (lane&3)*2 -> kk>>1 = lane&3
      uint32_t bb0[GM_WCB], bb1[GM_WCB];
#pragma unroll
      for (int jb = 0; jb < GM_WCB; ++jb) {
        const int ncol = warp_n * 32 + jb * 8 + r;
        const uint8_t* wq = &sWq[cur][ncol * KT_half];
        float sf = c_w4a16_ue4m3[sSFB[cur][ncol * GM_KSUB + ksub]] * alpha;
        __half2_raw h0 = __nv_cvt_fp4x2_to_halfraw2(
            static_cast<__nv_fp4x2_storage_t>(wq[byte0]), __NV_E2M1);
        __half2_raw h1 = __nv_cvt_fp4x2_to_halfraw2(
            static_cast<__nv_fp4x2_storage_t>(wq[byte0 + 4]), __NV_E2M1);
        float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
        float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
        bb0[jb] = pack_bf16x2(f0.x * sf, f0.y * sf);
        bb1[jb] = pack_bf16x2(f1.x * sf, f1.y * sf);
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
        if (r0 < M) Y[(size_t)r0 * N + ncol] = __float2bfloat16(c[ib][jb][0]);
        if (r1 < M) Y[(size_t)r1 * N + ncol] = __float2bfloat16(c[ib][jb][2]);
      }
      if (ncol + 1 < N) {
        if (r0 < M) Y[(size_t)r0 * N + ncol + 1] =
            __float2bfloat16(c[ib][jb][1]);
        if (r1 < M) Y[(size_t)r1 * N + ncol + 1] =
            __float2bfloat16(c[ib][jb][3]);
      }
    }
  }
}

void init_w4a16_ue4m3_lut() {
  static bool inited = false;
  if (inited) return;
  inited = true;
  float lut[256];
  for (int i = 0; i < 256; ++i) {
    const int e = (i >> 3) & 0xF;
    const int m = i & 0x7;
    if (e == 0) lut[i] = static_cast<float>(m) * std::ldexp(1.0f, -9);
    else lut[i] = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, e - 7);
  }
  cudaMemcpyToSymbol(c_w4a16_ue4m3, lut, sizeof(lut));
}

}  // namespace

int nexn2_w4a16_gemm_bf16(
    const void* X, const void* W, const void* SFB, void* Y,
    int M, int N, int K, float alpha, cudaStream_t stream) {
  if (!X || !W || !SFB || !Y) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || (K % GM_KT) != 0) return 2;
  init_w4a16_ue4m3_lut();
  dim3 grid((N + GM_BN - 1) / GM_BN, (M + GM_BM - 1) / GM_BM);
  dim3 block(GM_THREADS);
  w4a16_gemm_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(X),
      reinterpret_cast<const uint8_t*>(W),
      reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__nv_bfloat16*>(Y),
      alpha, M, N, K);
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt

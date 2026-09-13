// See nvfp4_m16_gemm_sm110.cuh.
#include "gemm/fp4/nvfp4_m16_gemm_sm110.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace flash_rt {
namespace fp4 {
namespace {

constexpr int NWARP = 8;             // K split factor
constexpr int THREADS = NWARP * 32;
constexpr int CTA_N = 16;            // two m16n8 tiles per warp
constexpr int NT = CTA_N / 8;

// Sm1xxBlockScaledConfig<16> SFA/SFB layout: tiles of 128 rows x 4 blocks
// (512 B), k-tiles fastest; inside a tile (row%32)*16 + ((row/32)%4)*4 + kb%4.
__device__ __forceinline__ int sf_index(int row, int kb, int K) {
  const int ktiles = K >> 6;
  return ((row >> 7) * ktiles + (kb >> 2)) * 512 + (row & 31) * 16 +
         ((row >> 5) & 3) * 4 + (kb & 3);
}

// e2m1 high-byte table (fp16 low byte is always zero).
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

// Four e2m1 nibbles (one 16-bit word: k0 low nibble .. k3 high nibble) ->
// two fp16x2 registers {k0,k1}, {k2,k3}.
__device__ __forceinline__ void dequant4(uint32_t v, uint32_t& lo, uint32_t& hi) {
  constexpr uint32_t LUT_LO = 0x3E3C3800u;   // 0, .5, 1, 1.5
  constexpr uint32_t LUT_HI = 0x46444240u;   // 2, 3, 4, 6
  uint32_t hb = prmt(LUT_LO, LUT_HI, v & 0x7777u);
  // sign: nibble i bit 3 -> byte i bit 7 (prmt a 0x00/0xFF byte mask)
  const uint32_t mask = prmt(0u, 0xFFFFFFFFu, (v & 0x8888u) >> 1);
  hb |= mask & 0x80808080u;
  // hb bytes: [b0 b1 b2 b3] -> lo = {0,b0,0,b1}, hi = {0,b2,0,b3}
  lo = prmt(hb, 0u, 0x1404u);
  hi = prmt(hb, 0u, 0x3424u);
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t b) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(b)));
  return r;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(THREADS)
nvfp4_m16_gemm_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ SFA,
                      const uint8_t* __restrict__ B, const uint8_t* __restrict__ SFB,
                      __half* __restrict__ D, int M, int N, int K) {
  __shared__ float red[NWARP][16][CTA_N + 1];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;        // group: row / column within the 8-wide tile
  const int t = lane & 3;         // k pair
  const int n0 = blockIdx.x * CTA_N;
  const int KB = K >> 4;          // 16-wide k blocks
  const int kb_per_warp = KB / NWARP;
  const int kb0 = warp * kb_per_warp;
  const size_t rowbytes = static_cast<size_t>(K) >> 1;

  const int ra = g, rb = g + 8;
  const bool va = ra < M, vb = rb < M;
  const uint8_t* Aa = A + static_cast<size_t>(va ? ra : 0) * rowbytes;
  const uint8_t* Ab = A + static_cast<size_t>(vb ? rb : 0) * rowbytes;
  const uint8_t* Bn[NT];
  #pragma unroll
  for (int j = 0; j < NT; ++j) Bn[j] = B + static_cast<size_t>(n0 + j * 8 + g) * rowbytes;

  float acc[NT][4];
  #pragma unroll
  for (int j = 0; j < NT; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }

  #pragma unroll 4
  for (int i = 0; i < kb_per_warp; ++i) {
    const int kb = kb0 + i;
    // ---- A fragment: rows g / g+8, k pairs (2t, 2t+1) and (2t+8, 2t+9)
    const uint32_t a_lo0 = va ? Aa[kb * 8 + t] : 0u;
    const uint32_t a_hi0 = va ? Aa[kb * 8 + 4 + t] : 0u;
    const uint32_t a_lo1 = vb ? Ab[kb * 8 + t] : 0u;
    const uint32_t a_hi1 = vb ? Ab[kb * 8 + 4 + t] : 0u;
    const uint32_t sa0 = va ? SFA[sf_index(ra, kb, K)] : 0u;
    const uint32_t sa1 = vb ? SFA[sf_index(rb, kb, K)] : 0u;
    uint32_t afrag[4];
    dequant4(a_lo0 | (a_hi0 << 8), afrag[0], afrag[2]);   // {k2t,k2t+1}, {k2t+8,k2t+9} for row g
    dequant4(a_lo1 | (a_hi1 << 8), afrag[1], afrag[3]);    // row g+8
    const uint32_t sa0h = e4m3x2_to_f16x2(sa0 | (sa0 << 8));
    const uint32_t sa1h = e4m3x2_to_f16x2(sa1 | (sa1 << 8));
    afrag[0] = hmul2(afrag[0], sa0h); afrag[2] = hmul2(afrag[2], sa0h);
    afrag[1] = hmul2(afrag[1], sa1h); afrag[3] = hmul2(afrag[3], sa1h);
    // ---- B fragments per n-tile
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
      const uint2 w = *reinterpret_cast<const uint2*>(Bn[j] + kb * 8);
      const uint32_t b_lo = (w.x >> (8 * t)) & 0xFFu;   // byte t   : k pair (2t, 2t+1)
      const uint32_t b_hi = (w.y >> (8 * t)) & 0xFFu;   // byte 4+t : k pair (2t+8, 2t+9)
      uint32_t b0, b1;
      dequant4(b_lo | (b_hi << 8), b0, b1);
      const uint32_t sb = SFB[sf_index(n0 + j * 8 + g, kb, K)];
      const uint32_t sbh = e4m3x2_to_f16x2(sb | (sb << 8));
      b0 = hmul2(b0, sbh); b1 = hmul2(b1, sbh);
      mma16816(acc[j], afrag, b0, b1);
    }
  }

  // ---- reduce the K split through shared memory
  #pragma unroll
  for (int j = 0; j < NT; ++j) {
    red[warp][g][j * 8 + 2 * t] = acc[j][0];
    red[warp][g][j * 8 + 2 * t + 1] = acc[j][1];
    red[warp][g + 8][j * 8 + 2 * t] = acc[j][2];
    red[warp][g + 8][j * 8 + 2 * t + 1] = acc[j][3];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < 16 * CTA_N; idx += THREADS) {
    const int r = idx / CTA_N, c = idx - r * CTA_N;
    if (r >= M) continue;
    float s = 0.f;
    #pragma unroll
    for (int w = 0; w < NWARP; ++w) s += red[w][r][c];
    D[static_cast<size_t>(r) * N + n0 + c] = __float2half(s);
  }
}

}  // namespace

int nvfp4_m16_gemm_fp16out(const void* A, const void* SFA, const void* B,
                           const void* SFB, void* D, int M, int N, int K,
                           cudaStream_t stream) {
  if (M <= 0 || M > 16 || N % CTA_N != 0 || K % (16 * NWARP) != 0 || K % 64 != 0)
    return -1;
  nvfp4_m16_gemm_kernel<<<N / CTA_N, THREADS, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(SFA),
      reinterpret_cast<const uint8_t*>(B), reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__half*>(D), M, N, K);
  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // namespace fp4
}  // namespace flash_rt

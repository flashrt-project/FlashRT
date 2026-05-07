// SPDX-License-Identifier: Apache-2.0
//
// Tensor-core NVFP4 W4A4 GEMMs for sm_120 — single-tile, multi-K,
// and full-N production entry points. Header at
// fp4_w4a4_mma_sm120.cuh.

#include "fp4_w4a4_mma_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

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

// ── Layout decode helpers ─────────────────────────────────────────
//
// cute Layout decoding convention: layouts map (thread, value) coords
// to a SINGLE flat int. The flat int is interpreted in the OUTPUT
// shape's natural (column-major) layout. For target (M=16, K=64) the
// natural layout is Layout<Shape<_16, _64>> with stride (1, 16) ⇒
// offset = m*1 + k*16, hence m = offset%16, k = offset/16.
// (Row-major m=offset/64 — i.e. the natural reading — would
//  invert M and K for this layout. Use the column-major decoding.)
//
// ALayout = Layout<Shape<Shape<_4,_8>, Shape<_8,_2,_2>>,
//                  Stride<Stride<_128,_1>, Stride<_16,_8,_512>>>
// flat_offset = t0*128 + t1 + v0*16 + v1*8 + v2*512
// (m, k) = (offset%16, offset/16)
//
// Per-lane register packing (cute fragment storage):
//   reg_idx r in 0..3 corresponds to (v1=r&1, v2=(r>>1)&1).
//   byte_in_reg b in 0..7 corresponds to v0=b.
//   Lane t = (t0, t1) with t0 = t%4, t1 = t/4.
//
// Per-register row m is CONSTANT (doesn't depend on b/v0 since
// v0*16 stride is divisible by 16):
//   m = (t1 + 8*v1) % 16
//   k for byte b = t0*8 + b + 32*v2
//
//   * a[0] (v1=0, v2=0): row=t1,    k = t0*8 + b              (b=0..7)
//   * a[1] (v1=1, v2=0): row=t1+8,  k = t0*8 + b
//   * a[2] (v1=0, v2=1): row=t1,    k = t0*8 + b + 32
//   * a[3] (v1=1, v2=1): row=t1+8,  k = t0*8 + b + 32
//
// For M=1, real row 0 appears in reg a[0]/a[2] of lanes with t1=0
// (i.e., lanes 0..3). Reg a[1]/a[3] hold row 8 (= zero in M=1 case).

// BLayout = Layout<Shape<Shape<_4,_8>, Shape<_8,_2>>,
//                  Stride<Stride<_64,_1>, Stride<_8,_256>>>
// flat_offset = t0*64 + t1 + b*8 + v1*256
// Target (N=8, K=64). col-major: n = offset%8, k = offset/8.
//
// Per-register col n is CONSTANT:
//   n = t1
//   k for byte b in reg r = t0*8 + b + r*32
//
//   * b[0] (v1=0): col=t1, k = t0*8 + b              (b=0..7)
//   * b[1] (v1=1): col=t1, k = t0*8 + b + 32

// SFALayout = ((T:(2,2,8), V:64), (stride 8/0/1, 16)).
//   t0 ∈ {0,1}, t1 ∈ {0,1} (REPLICATED - stride 0), t2 ∈ {0..7}
//   Lane t = t0 + t1*2 + t2*4.
//   Unique lane index u = t0*8 + t2  (t1 dropped due to stride 0).
//   Output coord (m, k_pos) with m = offset%16, k_pos = offset/16.
//   Per lane SFA register: 4 bytes, byte b ∈ 0..3 → V = b*16,
//     offset = u + b*256, (m, k_pos) = (u, b*16),
//     k_group = k_pos / 16 = b.
//   ⇒ byte b of lane t holds SFA[row=u, K_group=b].

// SFBLayout = ((T:(4,8), V:64), (stride 0/1, 8)).
//   t0 ∈ {0..3} (REPLICATED), t1 ∈ {0..7}.
//   Lane t = t0 + t1*4.
//   Unique lane index u = t1.
//   Per lane SFB register: 4 bytes, byte b ∈ 0..3 → V = b*16,
//     offset = u + b*128, (n, k_pos) decoded with target stride
//     (n = offset/64, k = offset%64) — actually (n, k_g) = (u, b)
//     by symmetry with SFA.
//   ⇒ byte b of lane t holds SFB[col=u, K_group=b].

// ── Helpers ───────────────────────────────────────────────────────

__device__ __forceinline__ uint8_t a_nibble(
    const uint8_t* sA_bytes, int m, int k) {
  // sA_bytes layout: 16 rows × 32 packed bytes. Each byte = 2 e2m1.
  // For M=1 case: rows 1..15 don't exist in the source; treat as zero.
  if (m != 0) return 0;
  int byte_idx = m * 32 + (k >> 1);
  uint8_t byte = sA_bytes[byte_idx];
  return (k & 1) ? (byte >> 4) : (byte & 0xF);
}

__device__ __forceinline__ uint8_t b_nibble(
    const uint8_t* sB_bytes, int n, int k) {
  // sB_bytes layout: 8 cols × 32 packed bytes (col-major in N).
  // Equivalent to row-major over (n, k) since each col = 64 e2m1.
  int byte_idx = n * 32 + (k >> 1);
  uint8_t byte = sB_bytes[byte_idx];
  return (k & 1) ? (byte >> 4) : (byte & 0xF);
}

__device__ __forceinline__ uint32_t pack_a_reg(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  int v1 = reg_idx & 1;
  int v2 = (reg_idx >> 1) & 1;
  uint32_t reg = 0;
  #pragma unroll
  for (int b = 0; b < 8; ++b) {
    int off = t0 * 128 + t1 + b * 16 + v1 * 8 + v2 * 512;
    int m = off % 16;       // col-major target: m fast, K slow
    int k = off / 16;
    uint8_t nib = a_nibble(sA, m, k);
    reg |= (uint32_t(nib) & 0xF) << (b * 4);
  }
  return reg;
}

__device__ __forceinline__ uint32_t pack_b_reg(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  int v1 = reg_idx;  // 0 or 1
  uint32_t reg = 0;
  #pragma unroll
  for (int b = 0; b < 8; ++b) {
    int off = t0 * 64 + t1 + b * 8 + v1 * 256;
    int n = off % 8;        // col-major target
    int k = off / 8;
    uint8_t nib = b_nibble(sB, n, k);
    reg |= (uint32_t(nib) & 0xF) << (b * 4);
  }
  return reg;
}

__device__ __forceinline__ uint32_t pack_sfa_reg(
    const uint8_t* sSFA, int unique_row) {
  // 4 SF bytes for one row across 4 K-groups.
  // For M=1 case: only unique_row=0 has real data; rows 1..15 = zero.
  uint32_t reg = 0;
  if (unique_row == 0) {
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      uint8_t sf = sSFA[unique_row * 4 + b];
      reg |= uint32_t(sf) << (b * 8);
    }
  }
  // else: zero SF (row contributes 0 to output regardless of A data).
  return reg;
}

__device__ __forceinline__ uint32_t pack_sfb_reg(
    const uint8_t* sSFB, int unique_col) {
  uint32_t reg = 0;
  #pragma unroll
  for (int b = 0; b < 4; ++b) {
    uint8_t sf = sSFB[unique_col * 4 + b];
    reg |= uint32_t(sf) << (b * 8);
  }
  return reg;
}

// ── Kernel ────────────────────────────────────────────────────────
//
// Single-tile single-warp kernel. One MMA. No K loop.
// Block: 32 threads (1 warp). Grid: (1,1,1).

__global__ void single_tile_kernel(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    const uint8_t* __restrict__ SFA,
    const uint8_t* __restrict__ SFB,
    __nv_bfloat16* __restrict__ D,
    float alpha) {
  // Smem stage so all lanes can read with arbitrary indexing.
  __shared__ alignas(16) uint8_t s_A[16 * 32];   // 512 bytes
  __shared__ alignas(16) uint8_t s_B[8 * 32];    // 256 bytes
  __shared__ alignas(16) uint8_t s_SFA[16 * 4];  //  64 bytes
  __shared__ alignas(16) uint8_t s_SFB[8 * 4];   //  32 bytes

  int tid = threadIdx.x;
  if (tid >= 32) return;

  // Cooperative load (one warp).
  // For M=1: A_packed only has row 0 (32 bytes). Zero-fill rows 1..15.
  if (tid < 16) {
    // 32 bytes per row, 16 rows → 16 lanes load 32 bytes each.
    // For row 0: real data. For rows 1..15: zeros.
    for (int b = 0; b < 32; ++b) {
      s_A[tid * 32 + b] = (tid == 0) ? A_packed[b] : 0;
    }
  }
  if (tid < 8) {
    for (int b = 0; b < 32; ++b) {
      s_B[tid * 32 + b] = B_packed[tid * 32 + b];
    }
  }
  // SFA: 16 rows × 4 = 64 bytes. For M=1 only row 0 is real;
  // pad rows 1..15 with zeros.
  if (tid < 64) {
    s_SFA[tid] = (tid < 4) ? SFA[tid] : 0;
  }
  if (tid < 32) {
    s_SFB[tid] = SFB[tid];
  }
  __syncthreads();

  // Decompose lane.
  int t0 = tid & 3;        // 0..3 (for ALayout / BLayout thread axis 0)
  int t1 = tid >> 2;       // 0..7 (for ALayout / BLayout thread axis 1)

  // SFA decompose (different layout): t0_sfa = tid&1, t2_sfa = tid>>2
  // (t1_sfa is replicated — irrelevant).
  int sfa_t0 = tid & 1;
  int sfa_t2 = tid >> 2;
  int sfa_unique_row = sfa_t0 * 8 + sfa_t2;

  // SFB decompose: t0_sfb replicated, t1_sfb = tid >> 2.
  int sfb_unique_col = tid >> 2;

  // Compose fragments.
  uint32_t a0 = pack_a_reg(s_A, t0, t1, 0);
  uint32_t a1 = pack_a_reg(s_A, t0, t1, 1);
  uint32_t a2 = pack_a_reg(s_A, t0, t1, 2);
  uint32_t a3 = pack_a_reg(s_A, t0, t1, 3);
  uint32_t b0 = pack_b_reg(s_B, t0, t1, 0);
  uint32_t b1 = pack_b_reg(s_B, t0, t1, 1);
  uint32_t sfa = pack_sfa_reg(s_SFA, sfa_unique_row);
  uint32_t sfb = pack_sfb_reg(s_SFB, sfb_unique_col);

  // Accumulator starts at 0 (no K accumulation in S1).
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  float d0, d1, d2, d3;

  AtomType::fma(d0, d1, d2, d3,
                a0, a1, a2, a3,
                b0, b1,
                c0, c1, c2, c3,
                sfa, sfb);

  // CLayout = SM80_16x8_Row.
  //   Lane t holds D fragment positions:
  //     d0 = (row=t/4,   col=(t%4)*2)
  //     d1 = (row=t/4,   col=(t%4)*2+1)
  //     d2 = (row=t/4+8, col=(t%4)*2)
  //     d3 = (row=t/4+8, col=(t%4)*2+1)
  //
  // For M=1 we only care about row 0 → lanes with t/4 == 0 (lanes
  // 0..3) carry it in d0, d1.
  int q = tid >> 2;
  int r = tid & 3;
  if (q == 0) {
    int col0 = r * 2;
    int col1 = col0 + 1;
    if (col0 < 8) D[col0] = __float2bfloat16(d0 * alpha);
    if (col1 < 8) D[col1] = __float2bfloat16(d1 * alpha);
  }
}

// ── Multi-K accumulation kernel ────────────────────────────
//
// Same single-warp / N=8 / M=1-padded-to-16 layout as the
// single-tile entry, but loops over K in K_TILE=64 chunks,
// accumulating into the f32 D fragment across all tiles. The C
// operand of each iter's MMA is the previous iter's D, so the
// accumulation is hardware-fused inside the MMA. After the last
// K-tile, alpha is applied and the row 0 portion of D is written
// to gmem (8 bf16 outputs).
//
// The K loop is NOT pipelined here (the production full-N kernel
// below adds cp.async double-buffering). Used as a
// correctness-only oracle in the standalone unit test.

__global__ void multi_k_kernel(
    const uint8_t* __restrict__ A_packed,    // (K/2,)
    const uint8_t* __restrict__ B_packed,    // (8, K/2) row-major
    const uint8_t* __restrict__ SFA,         // (K/16,)
    const uint8_t* __restrict__ SFB,         // (8, K/16)
    __nv_bfloat16* __restrict__ D,           // (8,) bf16
    float alpha,
    int K) {
  __shared__ alignas(16) uint8_t s_A[16 * 32];
  __shared__ alignas(16) uint8_t s_B[8 * 32];
  __shared__ alignas(16) uint8_t s_SFA[16 * 4];
  __shared__ alignas(16) uint8_t s_SFB[8 * 4];

  int tid = threadIdx.x;
  if (tid >= 32) return;

  int t0 = tid & 3;
  int t1 = tid >> 2;
  int sfa_unique_row = (tid & 1) * 8 + (tid >> 2);
  int sfb_unique_col = tid >> 2;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  const int K_TILE_BYTES = 32;          // 64 e2m1 / 2 = 32 packed bytes
  const int K_TILE_SF = 4;              // 64 e2m1 / 16 = 4 SF bytes
  const int K_iters = K / 64;
  const int K_half = K / 2;             // bytes per row of A or B
  const int K_sf = K / 16;              // SFs per row of A or B

  for (int kt = 0; kt < K_iters; ++kt) {
    int byte_off = kt * K_TILE_BYTES;
    int sf_off = kt * K_TILE_SF;

    // ── Load A tile (32 bytes for row 0; rows 1..15 are zero) ──
    if (tid < 16) {
      #pragma unroll
      for (int b = 0; b < 32; ++b) {
        s_A[tid * 32 + b] =
            (tid == 0) ? A_packed[byte_off + b] : uint8_t(0);
      }
    }
    // ── Load B tile (8 cols × 32 bytes) ──
    if (tid < 8) {
      #pragma unroll
      for (int b = 0; b < 32; ++b) {
        s_B[tid * 32 + b] = B_packed[tid * K_half + byte_off + b];
      }
    }
    // ── Load SFA tile (4 bytes for row 0; rows 1..15 are zero) ──
    if (tid < 64) {
      s_SFA[tid] =
          (tid < 4) ? SFA[sf_off + tid] : uint8_t(0);
    }
    // ── Load SFB tile (8 cols × 4 SF bytes) ──
    if (tid < 32) {
      int col = tid >> 2;
      int kg = tid & 3;
      s_SFB[col * 4 + kg] = SFB[col * K_sf + sf_off + kg];
    }
    __syncthreads();

    // Compose fragments and issue MMA, accumulating into c.
    uint32_t a0 = pack_a_reg(s_A, t0, t1, 0);
    uint32_t a1 = pack_a_reg(s_A, t0, t1, 1);
    uint32_t a2 = pack_a_reg(s_A, t0, t1, 2);
    uint32_t a3 = pack_a_reg(s_A, t0, t1, 3);
    uint32_t b0 = pack_b_reg(s_B, t0, t1, 0);
    uint32_t b1 = pack_b_reg(s_B, t0, t1, 1);
    uint32_t sfa = pack_sfa_reg(s_SFA, sfa_unique_row);
    uint32_t sfb = pack_sfb_reg(s_SFB, sfb_unique_col);

    float d0, d1, d2, d3;
    AtomType::fma(d0, d1, d2, d3,
                  a0, a1, a2, a3,
                  b0, b1,
                  c0, c1, c2, c3,
                  sfa, sfb);
    c0 = d0; c1 = d1; c2 = d2; c3 = d3;

    __syncthreads();
  }

  // Lanes 0..3 carry row 0 in c0, c1.
  int q = tid >> 2;
  int r = tid & 3;
  if (q == 0) {
    int col0 = r * 2;
    int col1 = col0 + 1;
    if (col0 < 8) D[col0] = __float2bfloat16(c0 * alpha);
    if (col1 < 8) D[col1] = __float2bfloat16(c1 * alpha);
  }
}

// ── Full-N, multi-warp, multi-K production kernel ─────────────────
//
// Block: 4 warps = 128 threads, gridDim.x = ceil(N / 32).
//   * A and SFA shared across all 4 warps (one block-wide load).
//   * Each warp owns an 8-col N-tile (cols [block*32 + warp*8 :
//     block*32 + warp*8 + 8]) and runs its own K loop.
//   * Each warp loads its B/SFB tile (per-K-tile) into a per-warp
//     section of shared memory before fragment composition.
//
// SF replication (UE4M3 SFA byte 0 = sentinel for "row 0 has data,
// rows 1..15 are zero") relies on the same M=1 padding scheme as
// the single-tile / multi-K paths — A rows 1..15 zero in s_A, SFA
// rows 1..15 zero in s_SFA.

// Occupancy tuning: 1 warp / block (= 8 cols / block).
// Trade-off: A and SFA aren't shared across warps in a block, but
// they're tiny (32 B and 4 B per K-tile) so the duplicated HBM
// traffic is negligible vs the 9.4 MB weight read.
//
// Effect on block grid for production shapes:
//   N=1024  (k/v_proj):    32 → 128 blocks   (4× more SMs busy)
//   N=4096  (q/o/down):  128 → 512 blocks   (4×)
//   N=12288 (mlp_g/up):  384 → 1536 blocks  (4×)
constexpr int S3_WARPS_PER_BLOCK = 1;
constexpr int S3_THREADS_PER_BLOCK = S3_WARPS_PER_BLOCK * 32;
constexpr int S3_COLS_PER_WARP = 8;
constexpr int S3_COLS_PER_BLOCK =
    S3_WARPS_PER_BLOCK * S3_COLS_PER_WARP;   // = 8

// Fast register fragment composition (replaces pack_a_reg etc. for
// the full_n_kernel hot path). Key insight: cute ALayout puts the 8
// nibbles of one register into 4 CONTIGUOUS smem bytes, so a single
// uint32_t load gets the whole register's data with zero shuffling.
//
//   reg a[0] for lane (t0, t1) → row t1, k = t0*8 + 0..7
//                              → smem byte offset t1*32 + t0*4
//                              → 4 contiguous bytes = 1 uint32_t
//
// Byte ordering: the natural little-endian uint32 layout puts byte 0
// of smem in bits 0-7 of the uint32 (containing nibbles for k=0..1
// at bit positions 0-3 and 4-7), byte 1 at bits 8-15 (k=2,3), etc.
// This matches the cute fragment register convention exactly: byte b
// of the register holds nibble v0=b at bit position b*4. No shuffle.
//
// For M=1, A rows 1..15 (i.e., reg a[1] / a[3] which map to row t1+8
// ∈ 8..15) are zero-filled in smem during the cooperative load, so
// the uint32 reads return 0 — equivalent to hard-coding `0` for those
// registers but kept as smem reads so the layout stays uniform.

__device__ __forceinline__ uint32_t fast_load_a(
    const uint8_t* sA, int t0, int t1, int reg_idx) {
  // reg_idx 0,1: m=t1 (or t1+8), k=t0*8..t0*8+7
  // reg_idx 2,3: m=t1 (or t1+8), k=t0*8+32..t0*8+39
  // The (m offset by 8) for reg_idx 1/3 is in *rows*, all zero in
  // M=1 case → the smem loads return 0.
  // Smem byte offset for the 4 contiguous bytes:
  //   row * 32 + (k_start)/2 = row*32 + t0*4 + (reg_idx>>1)*16
  int row_off = ((reg_idx & 1) ? (t1 + 8) : t1) * 32;
  int col_off = t0 * 4 + ((reg_idx >> 1) & 1) * 16;
  return *reinterpret_cast<const uint32_t*>(sA + row_off + col_off);
}

__device__ __forceinline__ uint32_t fast_load_b(
    const uint8_t* sB, int t0, int t1, int reg_idx) {
  // BLayout: col=t1, k = t0*8 + b + reg_idx*32  (reg_idx ∈ {0, 1})
  // Smem byte offset: t1*32 + t0*4 + reg_idx*16
  int col_off = t0 * 4 + reg_idx * 16;
  return *reinterpret_cast<const uint32_t*>(sB + t1 * 32 + col_off);
}

__device__ __forceinline__ uint32_t fast_load_sfa(
    const uint8_t* sSFA, int unique_row) {
  // 4 UE4M3 SF bytes (one per K-group) packed into a uint32_t.
  // Byte b → SFA[unique_row, K_g=b]. Smem stores them contiguously
  // at sSFA[unique_row*4 .. unique_row*4 + 3].
  return *reinterpret_cast<const uint32_t*>(sSFA + unique_row * 4);
}

__device__ __forceinline__ uint32_t fast_load_sfb(
    const uint8_t* sSFB, int unique_col) {
  return *reinterpret_cast<const uint32_t*>(sSFB + unique_col * 4);
}

// ── cp.async helpers ──────────────────────────────────────────────
//
// PTX cp.async.ca.shared.global lets us issue async HBM→smem copies
// that the warp can keep going past while the data is in flight, then
// wait via cp.async.wait_group when we actually need the data. This
// is the basic latency-hiding lever on sm_120 (no Hopper TMA, but
// sm_120 supports the sm_80-era cp.async primitive).
//
// We use 4-byte transfers (smallest, simplest alignment). Each lane
// issues a small handful per K-tile; cp.async.commit_group groups
// them; cp.async.wait_group(N) waits until ≤ N groups remain pending.

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
  // N must be a constexpr int — emit one specialized wait per call site.
  // Using a switch keeps the PTX literal.
  if (N == 0) {
    asm volatile("cp.async.wait_group 0;\n" ::);
  } else if (N == 1) {
    asm volatile("cp.async.wait_group 1;\n" ::);
  } else {
    asm volatile("cp.async.wait_all;\n" ::);
  }
}

__global__ void full_n_kernel(
    const uint8_t* __restrict__ A_packed,    // (K/2,)
    const uint8_t* __restrict__ B_packed,    // (N, K/2)
    const uint8_t* __restrict__ SFA,         // (K/16,)
    const uint8_t* __restrict__ SFB,         // (N, K/16)
    __nv_bfloat16* __restrict__ D,           // (N,)
    float alpha,
    int N, int K) {
  // Double-buffered A / B / SFA / SFB (per warp; kt%2 selects buffer).
  __shared__ alignas(16) uint8_t s_A[2][16 * 32];
  __shared__ alignas(16) uint8_t s_SFA[2][16 * 4];
  __shared__ alignas(16) uint8_t s_B_all[2][
      S3_WARPS_PER_BLOCK * 8 * 32];
  __shared__ alignas(16) uint8_t s_SFB_all[2][
      S3_WARPS_PER_BLOCK * 8 * 4];

  int tid = threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;

  int block_n_off = blockIdx.x * S3_COLS_PER_BLOCK;
  int my_n_off = block_n_off + warp * S3_COLS_PER_WARP;
  if (my_n_off >= N) return;

  int t0 = lane & 3;
  int t1 = lane >> 2;
  int sfa_unique_row = (lane & 1) * 8 + (lane >> 2);
  int sfb_unique_col = lane >> 2;

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  const int K_iters = K / 64;
  const int K_half = K / 2;
  const int K_sf = K / 16;

  // Pre-zero the high rows of s_A and s_SFA in BOTH double-buffer
  // banks (constant for the kernel's lifetime per M=1 padding).
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
    // SFA bytes 4..63 (rows 1..15) → zero in both banks.
    for (int i = 4 + lane; i < 64; i += 4) {
      s_SFA[0][i] = 0;
      s_SFA[1][i] = 0;
    }
  }

  // SFA / SFB swizzle parameters. The loader stores both SFs in the
  // SM120 NVFP4 SF swizzle layout produced by
  // `nvfp4_sf_linear_to_swizzled` (csrc/quantize/) — same scheme the
  // SIMT matvec kernel decodes in
  // csrc/kernels/fp4_w4a4_matvec_sm120.cu.
  //
  // For a single (M=1) NVFP4 GEMM:
  //   K_blocks = K / 16   (number of 16-K-element SF groups)
  //   n_col_super = (K_blocks + 3) / 4
  //
  // For row r at K-group b, the swizzled byte offset is:
  //   rb = r >> 7;   ri = r & 127
  //   cb = b >> 2;   ci = b & 3
  //   super_idx = rb * n_col_super + cb
  //   inner_off = (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + ci
  //   off       = super_idx * 512 + inner_off
  //
  // Critical property: within ONE of our K-tiles (= 64 K, = 4
  // K-groups), all 4 K-groups share the same `cb = kt` because
  // K-tile kt covers K-blocks {kt*4, kt*4+1, kt*4+2, kt*4+3}. So
  // ci varies 0..3 and inner_off is contiguous. The 4 SF bytes for
  // one row × one K-tile are 4 CONSECUTIVE bytes in the swizzled
  // table — readable as one uint32.
  const int K_blocks = K / 16;
  const int n_col_super = (K_blocks + 3) / 4;

  // ── Per-tile async load helper (lambda, captures by reference) ──
  // Issues cp.async loads for K-tile `kt` into buffer `buf`.
  // After all calls, caller issues cp_async_commit_group().
  auto issue_async_load = [&](int buf, int kt) {
    int byte_off = kt * 32;
    // A row 0: 32 bytes = 8 uint32. Lanes 0..7.
    if (lane < 8) {
      cp_async_4(s_A[buf] + lane * 4, A_packed + byte_off + lane * 4);
    }
    // SFA row 0: 4 bytes (one per K-group in this K-tile). Swizzled.
    // For row 0: rb=0, ri=0, super_idx = 0 + kt, inner_off = 0..3 (ci).
    // 4 consecutive bytes at SFA + kt*512.
    if (lane == 0) {
      cp_async_4(s_SFA[buf] + 0, SFA + kt * 512);
    }
    // B (per-warp): 8 cols × 32 bytes = 64 uint32. 32 lanes × 2 each.
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
    // SFB (per-warp): 8 cols × 4 SF bytes. Swizzled — for col c at
    // K-tile kt, the 4 bytes are at (rb*n_col_super + kt)*512 +
    // ri-derived inner_off. ci = 0..3 gives 4 consecutive bytes.
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

  // ── Prologue: prime both double-buffer banks ──
  issue_async_load(0, 0);
  cp_async_commit_group();
  if (K_iters > 1) {
    issue_async_load(1, 1);
    cp_async_commit_group();
  }

  // ── Main loop ──
  // At loop entry for kt, we need buf[kt%2] ready. The cp.async
  // wait pattern: after each iter's MMA, issue load for kt+2 (the
  // far-future tile reusing the just-consumed buffer), then wait
  // until ≤ 1 group is pending → guarantees buf[(kt+1)%2] is ready.
  for (int kt = 0; kt < K_iters; ++kt) {
    int curr_buf = kt & 1;

    // Wait for current buf to be ready.
    // Pending groups at loop entry:
    //   kt = 0: prologue committed K_iters>1 ? 2 : 1 groups
    //   kt > 0: prev iter committed at most 1 new group
    // We want to wait until current buf ready = wait until ≤ 1
    // pending (the future tile, if any) for kt < K_iters-2,
    // or until 0 pending for the last tile.
    if (kt + 1 < K_iters) {
      cp_async_wait_group(1);
    } else {
      cp_async_wait_group(0);
    }
    __syncwarp();

    // Compose fragments + MMA on current buf.
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

    // Issue load for kt+2 (recycle curr_buf since we're done with it).
    if (kt + 2 < K_iters) {
      issue_async_load(curr_buf, kt + 2);
      cp_async_commit_group();
    }
  }

  // Each warp writes its 8 N-cols of row 0 to D[my_n_off : my_n_off+8].
  int q = lane >> 2;
  int r = lane & 3;
  if (q == 0) {
    int col0 = my_n_off + r * 2;
    int col1 = col0 + 1;
    if (col0 < N) D[col0] = __float2bfloat16(c0 * alpha);
    if (col1 < N) D[col1] = __float2bfloat16(c1 * alpha);
  }
}

}  // namespace

// ── Host dispatch ─────────────────────────────────────────────────
int fp4_w4a4_mma_sm120_single_tile_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream) {
  if (!A_packed || !B_packed || !D_bf16 || !SFA || !SFB) return 1;

  dim3 block(32);
  dim3 grid(1);
  single_tile_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A_packed),
      reinterpret_cast<const uint8_t*>(B_packed),
      reinterpret_cast<const uint8_t*>(SFA),
      reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__nv_bfloat16*>(D_bf16),
      alpha);
  return 0;
}

int fp4_w4a4_mma_sm120_multi_k_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    int          K,
    cudaStream_t stream) {
  if (!A_packed || !B_packed || !D_bf16 || !SFA || !SFB) return 1;
  if (K <= 0 || (K % 64) != 0) return 2;

  dim3 block(32);
  dim3 grid(1);
  multi_k_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A_packed),
      reinterpret_cast<const uint8_t*>(B_packed),
      reinterpret_cast<const uint8_t*>(SFA),
      reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__nv_bfloat16*>(D_bf16),
      alpha, K);
  return 0;
}

int fp4_w4a4_mma_sm120_full_n_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int          N,
    int          K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream) {
  if (!A_packed || !B_packed || !D_bf16 || !SFA || !SFB) return 1;
  if (K <= 0 || (K % 64) != 0) return 2;
  if (N <= 0 || (N % S3_COLS_PER_BLOCK) != 0) return 3;

  dim3 block(S3_THREADS_PER_BLOCK);
  dim3 grid((N + S3_COLS_PER_BLOCK - 1) / S3_COLS_PER_BLOCK);
  full_n_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(A_packed),
      reinterpret_cast<const uint8_t*>(B_packed),
      reinterpret_cast<const uint8_t*>(SFA),
      reinterpret_cast<const uint8_t*>(SFB),
      reinterpret_cast<__nv_bfloat16*>(D_bf16),
      alpha, N, K);
  return 0;
}

}  // namespace gemm
}  // namespace flash_rt

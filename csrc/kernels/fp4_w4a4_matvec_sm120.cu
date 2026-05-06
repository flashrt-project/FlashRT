// SPDX-License-Identifier: Apache-2.0
//
// Custom NVFP4 W4A4 M=1 matvec for SM120. See header for design notes.

#include "fp4_w4a4_matvec_sm120.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace flash_rt {
namespace gemm {

namespace {

// ── Device-constant LUTs ──────────────────────────────────────────
// FP4 e2m1 codebook: signed 4-bit nibble → fp32 magnitude+sign.
__device__ __constant__ float c_fp4_codebook[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// UE4M3 (FP8 e4m3 used as non-negative scale) → fp32. 256-entry LUT.
// Bit layout: SEEEEMMM (sign/exp/mantissa). For UE4M3 the sign bit is
// always 0 in the SF byte (block scales are non-negative magnitudes),
// but we still encode the full 256 entries so a stray non-zero sign
// bit doesn't break anything.
//
// Standard FP8 e4m3 with bias=7:
//   E=0:           subnormal, val = M/8 × 2^(-6)
//   1 ≤ E ≤ 14:    normal,    val = (1 + M/8) × 2^(E - 7)
//   E=15, M=0..6:  reserved/saturated (treat as normal extension)
//   E=15, M=7:     NaN (we map to 0 to be safe in matvec)
//
// Initialized via fp4_w4a4_matvec_init_luts() (host code below).
__device__ __constant__ float c_ue4m3_lut[256];

// ── SF swizzle byte offset ────────────────────────────────────────
// Caller passes (row, k_block, n_col_super) — same packing as the
// linear→swizzled converter at csrc/quantize/nvfp4_sf_reshape_sm120.cu.
__device__ __forceinline__ int sf_swz_offset(int row, int k_block,
                                              int n_col_super) {
  int rb = row >> 7;
  int ri = row & 127;
  int cb = k_block >> 2;
  int ci = k_block & 3;
  int super_idx = rb * n_col_super + cb;
  int inner_off = (ri & 31) * 16 + ((ri >> 5) & 3) * 4 + ci;
  return super_idx * 512 + inner_off;
}

// ── Decode helpers ────────────────────────────────────────────────
__device__ __forceinline__ float fp4_decode(uint8_t nibble) {
  return c_fp4_codebook[nibble & 0xF];
}

__device__ __forceinline__ float ue4m3_decode(uint8_t b) {
  return c_ue4m3_lut[b];
}

// ── Matvec kernel ────────────────────────────────────────────────
//
// Grid: (N / ROWS_PER_BLOCK,)
// Block: (32 lanes, ROWS_PER_BLOCK warps) = (32, 32) = 1024 threads
//
// One warp = one output row.
// Each lane handles K_BLOCKS / WARP_SIZE k-blocks of its row.
// After per-lane partial sum, warp shuffle reduce → lane 0 writes D[row].
//
// Activation A (and SFA) are loaded into smem at block start so all
// 32 warps share one HBM read of the activation column.
//
// Per output row HBM read:
//   weights B   :  K/2 bytes
//   weight SFB  :  K/16 bytes
// Per block (32 rows, NOT counting amortized A read):
//   32 × (K/2 + K/16) = 32 × K × 9/16 = 18*K bytes
//
// For K=4096, N=4096: 128 blocks × 18 × 4096 = 9.4 MiB HBM traffic.
// At 1.79 TB/s peak: 5.25 µs ideal. We target ≥ 70% efficiency = ≤ 7.5 µs
// per GEMM (vs CUTLASS 16.4 µs at 30% efficiency = 2.2× speedup).

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;   // 256
constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK;                   // = 8
constexpr int K_BLOCK_BYTES = 8;       // 16 FP4 elts = 8 packed bytes

// v3 design: mirror the qwen36 bf16_matvec pattern that hits ~100% SM
// occupancy on RTX 5090.
//
// Per block:
//   threads          : 256 (8 warps)
//   output rows      : 8 (1 row per warp)
//   smem activation  : K/2 bytes (A_packed) + K_BLOCKS bytes
//                       (de-swizzled SFA linear) — tiny, well below
//                       the 100 KB/SM Blackwell smem budget.
//   gridDim.x        : ceil(N / 8)
//
// Why 8 warps/block (vs the v2 1-warp/block):
//   * 5090 SM has 64 warp slots; 8 blocks of 8 warps each = 64 warps
//     per SM = 100% occupancy. v2 left 50% of warp slots idle.
//   * Activation A is read ONCE per block via int4 (16-byte) bursts
//     across 256 cooperative threads, then 8 warps amortize the
//     read.
//   * Many concurrent blocks → many in-flight HBM transactions →
//     better latency hiding.

template <int K>
__global__ void fp4_w4a4_matvec_kernel(
    const uint8_t* __restrict__ A_packed,       // (K/2,)
    const uint8_t* __restrict__ B_packed,       // (N, K/2)
    const uint8_t* __restrict__ SFA,            // swizzled, (1, K/16)
    const uint8_t* __restrict__ SFB,            // swizzled, (N, K/16)
    __nv_bfloat16* __restrict__ D,              // (N,)
    float alpha,
    int N) {
  static_assert(K % 512 == 0, "K must be a multiple of 16×32 = 512");
  constexpr int K_BLOCKS = K / 16;
  constexpr int K_BLOCKS_PER_LANE = K_BLOCKS / WARP_SIZE;
  static_assert(K_BLOCKS_PER_LANE * WARP_SIZE == K_BLOCKS,
                "K_BLOCKS must be divisible by 32");

  // ── Shared memory: A packed + de-swizzled SFA ────────────────
  __shared__ uint8_t s_A_packed[K / 2];
  __shared__ uint8_t s_A_sf_lin[K_BLOCKS];

  const int tid = threadIdx.x;            // 0..255 (linear in block)
  const int warp_id = tid >> 5;           // 0..7   (warp index in block)
  const int lane = tid & 31;              // 0..31  (lane in warp)

  // ── Cooperative load of A_packed (K/2 bytes) into smem via int4 ──
  // K/2 ∈ {2048, 6144} both divisible by 16 → no tail.
  {
    constexpr int A_BYTES_DIV16 = (K / 2) / 16;
    const int4* a_v = reinterpret_cast<const int4*>(A_packed);
    int4* s_v = reinterpret_cast<int4*>(s_A_packed);
    #pragma unroll
    for (int i = tid; i < A_BYTES_DIV16; i += THREADS_PER_BLOCK) {
      s_v[i] = a_v[i];
    }
  }

  // ── De-swizzle SFA into linear smem layout ─────────────────
  // For M=1, ri=0: SFA bytes for k-block b are at swizzled offset
  //   (b/4) * 512 + (b % 4). Read each by k-block, write linear.
  {
    const int n_col_super = (K_BLOCKS + 3) / 4;
    for (int b = tid; b < K_BLOCKS; b += THREADS_PER_BLOCK) {
      int off = sf_swz_offset(0, b, n_col_super);
      s_A_sf_lin[b] = SFA[off];
    }
  }

  __syncthreads();

  const int row = blockIdx.x * ROWS_PER_BLOCK + warp_id;
  if (row >= N) return;

  const int tid_x = lane;       // alias for the lane inside the K loop
  const int n_col_super = (K_BLOCKS + 3) / 4;

  // Pointer to row's weight bytes: (N × K/2) row-major.
  const uint8_t* B_row = B_packed + row * (K / 2);

  float acc = 0.0f;

  // ── Inner K loop ────────────────────────────────────────────
  // CRITICAL: iterate WARP-WIDE per step so all 32 lanes touch
  // CONSECUTIVE k-blocks. With one row per warp, this turns the
  // B_row weight read into a single 256-byte coalesced burst per
  // iteration (lane t reads bytes [t*8:t*8+8] of the current
  // 256-byte stripe). The previous "per-lane chunk" layout had
  // each lane reading its own 64-byte stride → no cross-lane
  // coalescing → ~5× slower.
  //
  // Per iteration:
  //   warp reads  256 B-bytes  (1 coalesced burst, 32 lanes × 8 B)
  //              + 32 SFB bytes (one per lane; per-row layout has
  //                              4 contiguous bytes for ci=0..3
  //                              within each 4-lane group, plus
  //                              512-byte stride between cb groups
  //                              — somewhat strided, mitigated by
  //                              L2 hits since rows in the same
  //                              super share SFB super-blocks)
  //   reads       8 A-bytes    (smem)
  //              + 1 SFA byte  (smem)
  // Outer iter loop intentionally NOT unrolled — full unroll with the
  // inner j=0..7 unrolled would explode register pressure and spill.
  for (int iter = 0; iter < K_BLOCKS_PER_LANE; ++iter) {
    const int kb = iter * WARP_SIZE + tid_x;

    const int sfb_off = sf_swz_offset(row, kb, n_col_super);
    const float w_sf = ue4m3_decode(__ldg(SFB + sfb_off));
    const float a_sf = ue4m3_decode(s_A_sf_lin[kb]);

    const int b_byte_off = kb * K_BLOCK_BYTES;
    const uint64_t b_pack =
        *reinterpret_cast<const uint64_t*>(B_row + b_byte_off);
    const uint64_t a_pack =
        *reinterpret_cast<const uint64_t*>(s_A_packed + b_byte_off);

    const float scale = w_sf * a_sf;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint8_t a_byte = static_cast<uint8_t>(a_pack >> (j * 8));
      const uint8_t b_byte = static_cast<uint8_t>(b_pack >> (j * 8));
      const float a_lo = fp4_decode(a_byte & 0xF);
      const float a_hi = fp4_decode(a_byte >> 4);
      const float b_lo = fp4_decode(b_byte & 0xF);
      const float b_hi = fp4_decode(b_byte >> 4);
      acc += (a_lo * b_lo + a_hi * b_hi) * scale;
    }
  }

  // ── Warp reduce ─────────────────────────────────────────────
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
  }

  // ── Lane 0 writes ───────────────────────────────────────────
  if (tid_x == 0) {
    const float final_val = acc * alpha;
    D[row] = __float2bfloat16(final_val);
  }
}

}  // namespace

// ── Host: LUT init ────────────────────────────────────────────────
void fp4_w4a4_matvec_init_luts() {
  static bool inited = false;
  if (inited) {
    return;
  }
  inited = true;

  float lut[256];
  for (int i = 0; i < 256; ++i) {
    const int sign = (i >> 7) & 1;          // top bit; UE4M3 inputs have 0
    const int e = (i >> 3) & 0xF;
    const int m = i & 0x7;
    float v;
    if (e == 0) {
      // Subnormal: val = m/8 × 2^(-6) = m × 2^(-9)
      v = static_cast<float>(m) * std::ldexp(1.0f, -9);
    } else if (e == 0xF && m == 7) {
      // Reserved NaN encoding — map to 0 to avoid NaN propagation.
      v = 0.0f;
    } else {
      v = (1.0f + static_cast<float>(m) / 8.0f) *
          std::ldexp(1.0f, e - 7);
    }
    lut[i] = sign ? -v : v;
  }
  cudaMemcpyToSymbol(c_ue4m3_lut, lut, sizeof(lut));
}

// ── Host: dispatch ────────────────────────────────────────────────
int fp4_w4a4_matvec_sm120_bf16out(
    const void*  A_packed,
    const void*  B_packed,
    void*        D_bf16,
    int          N,
    int          K,
    const void*  SFA,
    const void*  SFB,
    float        alpha,
    cudaStream_t stream) {
  // Caller-side validation.
  if (N <= 0 || K <= 0) return 1;
  if (K % 512 != 0) return 2;        // K_BLOCKS_PER_LANE * WARP_SIZE * 16
  // (v2: no N alignment requirement — 1 warp / block)
  (void)0;
  if (!A_packed || !B_packed || !D_bf16 || !SFA || !SFB) return 4;

  // Init LUTs once. Caller may have done it via the binding; idempotent.
  fp4_w4a4_matvec_init_luts();

  // v3: 8 warps / block, gridDim.x = ceil(N / 8) — mirrors qwen36
  // bf16_matvec_qwen36_bf16 layout. 100% SM occupancy on RTX 5090.
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

  if (K == 4096) {
    fp4_w4a4_matvec_kernel<4096><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(A_packed),
        reinterpret_cast<const uint8_t*>(B_packed),
        reinterpret_cast<const uint8_t*>(SFA),
        reinterpret_cast<const uint8_t*>(SFB),
        reinterpret_cast<__nv_bfloat16*>(D_bf16),
        alpha, N);
  } else if (K == 12288) {
    fp4_w4a4_matvec_kernel<12288><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(A_packed),
        reinterpret_cast<const uint8_t*>(B_packed),
        reinterpret_cast<const uint8_t*>(SFA),
        reinterpret_cast<const uint8_t*>(SFB),
        reinterpret_cast<__nv_bfloat16*>(D_bf16),
        alpha, N);
  } else {
    return 5;       // unsupported K (extend specializations as needed)
  }

  return 0;
}

}  // namespace gemm
}  // namespace flash_rt

// ================================================================
//  motus_dev — Hand NVFP4 Conv3d fprop, sm_120a, **v19 + real SF**
//
//  Adds per-K-tile UE4M3 scale factor inputs to v19_skeleton.cu.
//  All other tile geometry / smem layout / cp.async pipeline /
//  Y-major persistent walk unchanged. v19_skeleton.cu is preserved
//  as-is (SF hardcoded to 1.0); this file is additive.
//
//  Phase 7.5.0 probe-verified SF maps (from sf_probe_test.cu):
//
//    SFA u32 byte k → K-block k (1:1)
//    lane L → M-row provider:
//      L%4==0: M = L/4         (rows 0..7)
//      L%4==1: M = L/4 + 8     (rows 8..15)
//      L%4∈{2,3}: ignored
//
//    SFB u32 byte k → K-block k (1:1)
//    lane L → N-col: N = (L%4)*8 + L/4
//    (atom_id = L%4, col_in_atom = L/4)
//
//  Input SF tensor shapes (matches conv input layout):
//    cache_sfa  [B, T_cache, H, W, Ci/16]  UE4M3 bytes
//    new_sfa    [B, T_new,   H, W, Ci/16]  UE4M3 bytes
//    w_sfb      [Co, 3, 3, 3, Ci/16]       UE4M3 bytes
//  (1 SF per 16 FP4 elements; for Ci=64, Ci/16=4 → 4 SF bytes per row,
//   fitting in one u32 — fits the K-tile load pattern below.)
//
//  Smem layout per K-tile (1 stage shown; 2 stages):
//    A_sf_smem [V19_BLOCK_M * 4]  = 512 bytes  (4 SF bytes per M-row)
//    B_sf_smem [V19_BLOCK_N * 4]  = 512 bytes  (4 SF bytes per N-col)
//
//  Load distribution (per K-tile, 256 threads × 4 bytes):
//    threads 0..127 → SFA[m=t][0..3]
//    threads 128..255 → SFB[n=t-128][0..3]
//  Each load is cp.async.4. Source pointer derived from m/co + (kt,kr,ks)
//  at the current K-iter using the same logic as FP4 data ptrs.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

namespace flash_rt {
namespace conv {

constexpr int V19SF_BLOCK_M = 128;
constexpr int V19SF_BLOCK_N = 128;
constexpr int V19SF_BLOCK_K_ELEM = 64;
constexpr int V19SF_N_ATOMS  = V19SF_BLOCK_N / 8;   // 16
constexpr int V19SF_N_GROUPS = V19SF_N_ATOMS / 4;   // 4
constexpr int V19SF_NUM_WARPS = 8;
constexpr int V19SF_THREADS = V19SF_NUM_WARPS * 32;
constexpr int V19SF_STAGES = 2;
constexpr int V19SF_SMEM_K_STRIDE = 48;
constexpr int V19SF_SF_K_PER_ROW = 4;               // 4 SF bytes per row per K-tile (Ci=64)

__device__ __forceinline__
void v19sf_mma_m16n8k64_e2m1_4x(
    float &d0, float &d1, float &d2, float &d3,
    float &d4, float &d5, float &d6, float &d7,
    float &d8, float &d9, float &d10, float &d11,
    float &d12, float &d13, float &d14, float &d15,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3,
    uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7,
    uint32_t sfa, uint32_t sfb)
{
  constexpr uint16_t bidA = 0, tidA = 0, bidB = 0;
  constexpr uint16_t tidB0 = 0, tidB1 = 1, tidB2 = 2, tidB3 = 3;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d0), "+f"(d1), "+f"(d8), "+f"(d9)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(d0), "f"(d1), "f"(d8), "f"(d9),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB0));
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d2), "+f"(d3), "+f"(d10), "+f"(d11)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3),
      "f"(d2), "f"(d3), "f"(d10), "f"(d11),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB1));
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d4), "+f"(d5), "+f"(d12), "+f"(d13)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5),
      "f"(d4), "f"(d5), "f"(d12), "f"(d13),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB2));
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64"
    ".row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "+f"(d6), "+f"(d7), "+f"(d14), "+f"(d15)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7),
      "f"(d6), "f"(d7), "f"(d14), "f"(d15),
      "r"(sfa), "h"(bidA), "h"(tidA),
      "r"(sfb), "h"(bidB), "h"(tidB3));
}

// FP4 data ptrs (verbatim from v19_skeleton.cu, k_global in elements)
__device__ __forceinline__
const uint8_t* v19sf_x_byte_ptr(const uint8_t* cache_x_fp4,
                                const uint8_t* new_x_fp4,
                                int m_global, int k_global,
                                int N, int T_cache, int T_new,
                                int H, int W, int Ci) {
  int K_total = 27 * Ci;
  int M_total = N * T_new * H * W;
  if (k_global >= K_total || m_global >= M_total) return nullptr;
  int spatial = T_new * H * W;
  int b_idx = m_global / spatial;
  int rem   = m_global - b_idx * spatial;
  int t_out = rem / (H * W);
  rem      -= t_out * (H * W);
  int h_out = rem / W;
  int w_out = rem - h_out * W;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3; q /= 3;
  int kr  = q % 3;
  int kt  = q / 3;
  int d_in = t_out + kt;
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  if (d_in < T_cache) {
    int idx_elem = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci + ci0;
    return cache_x_fp4 + (idx_elem >> 1);
  } else {
    int d_new = d_in - T_cache;
    int idx_elem = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci + ci0;
    return new_x_fp4 + (idx_elem >> 1);
  }
}

__device__ __forceinline__
const uint8_t* v19sf_w_byte_ptr(const uint8_t* w_fp4,
                                int co, int k_global, int Co, int Ci) {
  int K_total = 27 * Ci;
  if (co >= Co || k_global >= K_total) return nullptr;
  int q   = k_global / Ci;
  int ci0 = k_global % Ci;
  int ks  = q % 3; q /= 3;
  int kr  = q % 3;
  int kt  = q / 3;
  int idx_elem = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci + ci0;
  return w_fp4 + (idx_elem >> 1);
}

// SFA / SFB ptrs: byte = idx_blk (one SF byte per 16 FP4 elem).
// **k_base is the K-element global offset; we derive kt_iter = k_base / Ci
//   and ci_block_off = (k_base % Ci) >> 4 (offset within (kt,kr,ks) row).**
// Fixed 2026-05-12: original version used only kt_iter and assumed
// BLOCK_K_ELEM == Ci, which silently produced wrong SFs for Ci > 64.
__device__ __forceinline__
const uint8_t* v19sf_sfa_byte_ptr(const uint8_t* cache_sfa,
                                  const uint8_t* new_sfa,
                                  int m_global, int k_base,
                                  int N, int T_cache, int T_new,
                                  int H, int W, int Ci) {
  int Ci_blk = Ci >> 4;                              // Ci / 16
  int M_total = N * T_new * H * W;
  if (m_global >= M_total) return nullptr;
  int spatial = T_new * H * W;
  int b_idx = m_global / spatial;
  int rem   = m_global - b_idx * spatial;
  int t_out = rem / (H * W);
  rem      -= t_out * (H * W);
  int h_out = rem / W;
  int w_out = rem - h_out * W;
  int kt_iter = k_base / Ci;
  int ci_block_off = (k_base % Ci) >> 4;             // SF byte offset within row
  int ks = kt_iter % 3;
  int kr = (kt_iter / 3) % 3;
  int kt = kt_iter / 9;
  int d_in = t_out + kt;
  int h_in = h_out + kr - 1;
  int w_in = w_out + ks - 1;
  if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) return nullptr;
  if (d_in < T_cache) {
    int idx = (((b_idx * T_cache + d_in) * H + h_in) * W + w_in) * Ci_blk + ci_block_off;
    return cache_sfa + idx;
  } else {
    int d_new = d_in - T_cache;
    int idx = (((b_idx * T_new + d_new) * H + h_in) * W + w_in) * Ci_blk + ci_block_off;
    return new_sfa + idx;
  }
}

__device__ __forceinline__
const uint8_t* v19sf_sfb_byte_ptr(const uint8_t* w_sfb,
                                  int co, int k_base, int Co, int Ci) {
  int Ci_blk = Ci >> 4;
  if (co >= Co) return nullptr;
  int kt_iter = k_base / Ci;
  int ci_block_off = (k_base % Ci) >> 4;
  int ks = kt_iter % 3;
  int kr = (kt_iter / 3) % 3;
  int kt = kt_iter / 9;
  int idx = (((co * 3 + kt) * 3 + kr) * 3 + ks) * Ci_blk + ci_block_off;
  return w_sfb + idx;
}

__device__ __forceinline__
void v19sf_cp_async_16(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 16;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__
void v19sf_cp_async_4(uint32_t smem_int_ptr, const uint8_t* src) {
  int src_bytes = (src == nullptr) ? 0 : 4;
  asm volatile(
    "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
    :: "r"(smem_int_ptr), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__
uint32_t v19sf_to_smem_int(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void __launch_bounds__(V19SF_THREADS, 2)
fp4_conv3d_v19sf_kernel(
    const uint8_t* __restrict__ cache_x, const uint8_t* __restrict__ new_x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ cache_sfa, const uint8_t* __restrict__ new_sfa,
    const uint8_t* __restrict__ w_sfb,
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha,
    int M_tiles, int N_tiles)
{
  __shared__ __align__(16) uint8_t A_smem [V19SF_STAGES][V19SF_BLOCK_M * V19SF_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t B_smem [V19SF_STAGES][V19SF_BLOCK_N * V19SF_SMEM_K_STRIDE];
  // SF smem: 4 bytes per row, 128 rows. Pad each row to 4 bytes (same).
  __shared__ __align__(16) uint8_t A_sf_smem[V19SF_STAGES][V19SF_BLOCK_M * V19SF_SF_K_PER_ROW];
  __shared__ __align__(16) uint8_t B_sf_smem[V19SF_STAGES][V19SF_BLOCK_N * V19SF_SF_K_PER_ROW];

  const int t       = threadIdx.x;
  const int warp_id = t / 32;
  const int lane    = t % 32;
  const int l       = lane % 4;
  const int h       = lane / 4;

  const int M_total = N * T_new * H * W;
  const int K_total = 27 * Ci;

  const int ld_row_a   = t / 2;
  const int ld_k_off_a = (t & 1) * 32;  // K elements
  const int ld_row_b   = t / 2;
  const int ld_k_off_b = (t & 1) * 32;

  int tile_idx = blockIdx.x;
  int m_idx  = tile_idx / N_tiles;
  int n_idx  = tile_idx % N_tiles;
  int m_base = m_idx * V19SF_BLOCK_M;
  int co_base = n_idx * V19SF_BLOCK_N;
  if (m_base >= M_total || co_base >= Co) return;

  float dA[V19SF_N_ATOMS] = {0};
  float dB[V19SF_N_ATOMS] = {0};
  float dC[V19SF_N_ATOMS] = {0};
  float dD[V19SF_N_ATOMS] = {0};

  // ── Issue load: FP4 (cp.async.16) for A+B, plus SF (cp.async.4) for SFA+SFB.
  auto issue_load = [&](int stage, int k_base) {
    // FP4 A
    {
      const uint8_t* src = v19sf_x_byte_ptr(cache_x, new_x,
                                            m_base + ld_row_a,
                                            k_base + ld_k_off_a,
                                            N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int = v19sf_to_smem_int(
          &A_smem[stage][ld_row_a * V19SF_SMEM_K_STRIDE + (ld_k_off_a >> 1)]);
      v19sf_cp_async_16(smem_int, src);
    }
    // FP4 B
    {
      const uint8_t* src = v19sf_w_byte_ptr(w, co_base + ld_row_b,
                                            k_base + ld_k_off_b,
                                            Co, Ci);
      uint32_t smem_int = v19sf_to_smem_int(
          &B_smem[stage][ld_row_b * V19SF_SMEM_K_STRIDE + (ld_k_off_b >> 1)]);
      v19sf_cp_async_16(smem_int, src);
    }
    // SF: 4 SF bytes per K-tile per row, indexed by k_base (helper extracts
    // kt_iter and ci_block_off internally).
    if (t < V19SF_BLOCK_M) {
      const uint8_t* src = v19sf_sfa_byte_ptr(cache_sfa, new_sfa,
                                              m_base + t, k_base,
                                              N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int = v19sf_to_smem_int(&A_sf_smem[stage][t * V19SF_SF_K_PER_ROW]);
      v19sf_cp_async_4(smem_int, src);
    } else {
      int n_idx_t = t - V19SF_BLOCK_M;
      const uint8_t* src = v19sf_sfb_byte_ptr(w_sfb, co_base + n_idx_t, k_base, Co, Ci);
      uint32_t smem_int = v19sf_to_smem_int(&B_sf_smem[stage][n_idx_t * V19SF_SF_K_PER_ROW]);
      v19sf_cp_async_4(smem_int, src);
    }
  };

  issue_load(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  int compute_stage = 0;

  for (int k_base = 0; k_base < K_total; k_base += V19SF_BLOCK_K_ELEM) {
    int next_stage = compute_stage ^ 1;
    int k_next = k_base + V19SF_BLOCK_K_ELEM;
    if (k_next < K_total) issue_load(next_stage, k_next);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

    const int warp_M_off = warp_id * 16;
    const int kA0 = 4 * l;
    const int kA2 = 4 * l + 16;

    int rA0 = warp_M_off + h;
    int rA1 = warp_M_off + h + 8;
    uint32_t A0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SF_SMEM_K_STRIDE + kA0]);
    uint32_t A1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SF_SMEM_K_STRIDE + kA0]);
    uint32_t A2 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SF_SMEM_K_STRIDE + kA2]);
    uint32_t A3 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SF_SMEM_K_STRIDE + kA2]);

    // SFA u32 for this lane: 4 SF bytes (4 K-blocks) for the M-row this lane owns.
    //   L%4==0 → M = warp_M_off + L/4         (M_lo, rows 0..7)
    //   L%4==1 → M = warp_M_off + L/4 + 8     (M_hi, rows 8..15)
    //   L%4∈{2,3} → ignored by mma; load any 4-byte value (use L%4==0 row).
    int sfa_m_row;
    if ((lane & 3) == 1) {
      sfa_m_row = warp_M_off + (lane >> 2) + 8;
    } else {
      sfa_m_row = warp_M_off + (lane >> 2);
    }
    uint32_t SFA = *reinterpret_cast<const uint32_t*>(
        &A_sf_smem[compute_stage][sfa_m_row * V19SF_SF_K_PER_ROW]);

    // Per N-group (4 scale_vec::4X calls per warp covering BLOCK_N=128 in 4×32 chunks).
    #pragma unroll
    for (int g = 0; g < V19SF_N_GROUPS; ++g) {
      int base = g * 4;
      int co0 = (base + 0) * 8 + h;
      int co1 = (base + 1) * 8 + h;
      int co2 = (base + 2) * 8 + h;
      int co3 = (base + 3) * 8 + h;
      uint32_t B0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SF_SMEM_K_STRIDE + kA2]);
      uint32_t B2 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B3 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SF_SMEM_K_STRIDE + kA2]);
      uint32_t B4 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B5 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SF_SMEM_K_STRIDE + kA2]);
      uint32_t B6 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B7 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SF_SMEM_K_STRIDE + kA2]);

      // SFB u32 for this lane in this N-group:
      //   N range covered by this scale_vec::4X call = [g*32 .. g*32+31]
      //   Lane L owns N = g*32 + (L%4)*8 + L/4
      int sfb_n = g * 32 + l * 8 + h;       // h == lane/4, l == lane%4
      uint32_t SFB = *reinterpret_cast<const uint32_t*>(
          &B_sf_smem[compute_stage][sfb_n * V19SF_SF_K_PER_ROW]);

      v19sf_mma_m16n8k64_e2m1_4x(
          dA[base+0], dB[base+0],
          dA[base+1], dB[base+1],
          dA[base+2], dB[base+2],
          dA[base+3], dB[base+3],
          dC[base+0], dD[base+0],
          dC[base+1], dD[base+1],
          dC[base+2], dD[base+2],
          dC[base+3], dD[base+3],
          A0, A1, A2, A3,
          B0, B1, B2, B3, B4, B5, B6, B7,
          SFA, SFB);
    }
    compute_stage = next_stage;
  }
  asm volatile("cp.async.wait_all;\n" ::);

  // ── Epilogue store (same as v19_skeleton) ──
  const int warp_M_off = warp_id * 16;
  #pragma unroll
  for (int n_atom = 0; n_atom < V19SF_N_ATOMS; ++n_atom) {
    int co_pair = co_base + n_atom * 8 + 2 * l;
    int row0    = m_base + warp_M_off + h;
    int row1    = m_base + warp_M_off + h + 8;
    float b0 = 0.f, b1 = 0.f;
    if (bias != nullptr && co_pair < Co) {
      b0 = __bfloat162float(bias[co_pair]);
      if (co_pair + 1 < Co) b1 = __bfloat162float(bias[co_pair + 1]);
    }
    if (co_pair + 1 < Co) {
      __nv_bfloat162 packAB;
      packAB.x = __float2bfloat16(dA[n_atom] * alpha + b0);
      packAB.y = __float2bfloat16(dB[n_atom] * alpha + b1);
      __nv_bfloat162 packCD;
      packCD.x = __float2bfloat16(dC[n_atom] * alpha + b0);
      packCD.y = __float2bfloat16(dD[n_atom] * alpha + b1);
      if (row0 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row0 * Co + co_pair]) = packAB;
      }
      if (row1 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row1 * Co + co_pair]) = packCD;
      }
    } else {
      auto store = [&](int row, int co, float v, float bv) {
        if (row < M_total && co < Co) {
          y[row * Co + co] = __float2bfloat16(v * alpha + bv);
        }
      };
      store(row0, co_pair + 0, dA[n_atom], b0);
      store(row0, co_pair + 1, dB[n_atom], b1);
      store(row1, co_pair + 0, dC[n_atom], b0);
      store(row1, co_pair + 1, dD[n_atom], b1);
    }
  }
}

extern "C" int motus_fp4_conv3d_v19sf_ndhwc_bf16out(
    const void* cache_x_fp4,
    const void* new_x_fp4,
    const void* w_fp4,
    const void* cache_sfa, const void* new_sfa, const void* w_sfb,
    void* y_bf16,
    const void* bias_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % V19SF_BLOCK_K_ELEM != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[fp4_conv3d_v19sf] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        V19SF_BLOCK_K_ELEM, Ci, Co);
    return -1;
  }
  if (T_cache != 2) {
    std::fprintf(stderr, "[fp4_conv3d_v19sf] T_cache must be 2 (got %d)\n", T_cache);
    return -3;
  }
  int M = N * T_new * H * W;
  int M_tiles = (M + V19SF_BLOCK_M - 1) / V19SF_BLOCK_M;
  int N_tiles = (Co + V19SF_BLOCK_N - 1) / V19SF_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  dim3 grid(total_tiles);
  dim3 block(V19SF_THREADS);
  fp4_conv3d_v19sf_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(cache_x_fp4),
      reinterpret_cast<const uint8_t*>(new_x_fp4),
      reinterpret_cast<const uint8_t*>(w_fp4),
      reinterpret_cast<const uint8_t*>(cache_sfa),
      reinterpret_cast<const uint8_t*>(new_sfa),
      reinterpret_cast<const uint8_t*>(w_sfb),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      N, T_cache, T_new, H, W, Ci, Co, alpha,
      M_tiles, N_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp4_conv3d_v19sf] launch err: %s\n",
                 cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

// ================================================================
// v2: per-Co outer FP32 scale (Phase 10)
// Goal: break 0.992 cos floor in full-aggressive FP4 by applying a
//   per-output-channel FP32 scale in the epilogue. Weight is divided
//   by outer_w[co] offline before fvk NVFP4 quantization → inner
//   FP4+SF operates on normalized [-6,6] range across all Co channels.
//
// Math:
//   y[m,co] = outer_w[co] * Σ_k FP4_a[k]*SF_a[k]*FP4_w[co,k]*SF_w[co,k]
//           + bias[co]
//
// Kernel body is byte-identical to v1 except for the final epilogue
// multiply (alpha → outer_w[co]). v1 kernel/wrapper unchanged.
// outer_w_fp32 must be non-null (caller guarantees); no branch in hot path.
// ================================================================
__global__ void __launch_bounds__(V19SF_THREADS, 2)
fp4_conv3d_v19sf_kernel_v2(
    const uint8_t* __restrict__ cache_x, const uint8_t* __restrict__ new_x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ cache_sfa, const uint8_t* __restrict__ new_sfa,
    const uint8_t* __restrict__ w_sfb,
    const float* __restrict__ outer_w_fp32,            // [Co], per-Co outer scale
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ bias,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha,                                       // multiplicative gain (typically 1.0)
    int M_tiles, int N_tiles)
{
  __shared__ __align__(16) uint8_t A_smem [V19SF_STAGES][V19SF_BLOCK_M * V19SF_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t B_smem [V19SF_STAGES][V19SF_BLOCK_N * V19SF_SMEM_K_STRIDE];
  __shared__ __align__(16) uint8_t A_sf_smem[V19SF_STAGES][V19SF_BLOCK_M * V19SF_SF_K_PER_ROW];
  __shared__ __align__(16) uint8_t B_sf_smem[V19SF_STAGES][V19SF_BLOCK_N * V19SF_SF_K_PER_ROW];

  const int t       = threadIdx.x;
  const int warp_id = t / 32;
  const int lane    = t % 32;
  const int l       = lane % 4;
  const int h       = lane / 4;

  const int M_total = N * T_new * H * W;
  const int K_total = 27 * Ci;

  const int ld_row_a   = t / 2;
  const int ld_k_off_a = (t & 1) * 32;
  const int ld_row_b   = t / 2;
  const int ld_k_off_b = (t & 1) * 32;

  int tile_idx = blockIdx.x;
  int m_idx  = tile_idx / N_tiles;
  int n_idx  = tile_idx % N_tiles;
  int m_base = m_idx * V19SF_BLOCK_M;
  int co_base = n_idx * V19SF_BLOCK_N;
  if (m_base >= M_total || co_base >= Co) return;

  float dA[V19SF_N_ATOMS] = {0};
  float dB[V19SF_N_ATOMS] = {0};
  float dC[V19SF_N_ATOMS] = {0};
  float dD[V19SF_N_ATOMS] = {0};

  auto issue_load = [&](int stage, int k_base) {
    {
      const uint8_t* src = v19sf_x_byte_ptr(cache_x, new_x,
                                            m_base + ld_row_a,
                                            k_base + ld_k_off_a,
                                            N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int = v19sf_to_smem_int(
          &A_smem[stage][ld_row_a * V19SF_SMEM_K_STRIDE + (ld_k_off_a >> 1)]);
      v19sf_cp_async_16(smem_int, src);
    }
    {
      const uint8_t* src = v19sf_w_byte_ptr(w, co_base + ld_row_b,
                                            k_base + ld_k_off_b,
                                            Co, Ci);
      uint32_t smem_int = v19sf_to_smem_int(
          &B_smem[stage][ld_row_b * V19SF_SMEM_K_STRIDE + (ld_k_off_b >> 1)]);
      v19sf_cp_async_16(smem_int, src);
    }
    if (t < V19SF_BLOCK_M) {
      const uint8_t* src = v19sf_sfa_byte_ptr(cache_sfa, new_sfa,
                                              m_base + t, k_base,
                                              N, T_cache, T_new, H, W, Ci);
      uint32_t smem_int = v19sf_to_smem_int(&A_sf_smem[stage][t * V19SF_SF_K_PER_ROW]);
      v19sf_cp_async_4(smem_int, src);
    } else {
      int n_idx_t = t - V19SF_BLOCK_M;
      const uint8_t* src = v19sf_sfb_byte_ptr(w_sfb, co_base + n_idx_t, k_base, Co, Ci);
      uint32_t smem_int = v19sf_to_smem_int(&B_sf_smem[stage][n_idx_t * V19SF_SF_K_PER_ROW]);
      v19sf_cp_async_4(smem_int, src);
    }
  };

  issue_load(0, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  int compute_stage = 0;

  for (int k_base = 0; k_base < K_total; k_base += V19SF_BLOCK_K_ELEM) {
    int next_stage = compute_stage ^ 1;
    int k_next = k_base + V19SF_BLOCK_K_ELEM;
    if (k_next < K_total) issue_load(next_stage, k_next);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

    const int warp_M_off = warp_id * 16;
    const int kA0 = 4 * l;
    const int kA2 = 4 * l + 16;

    int rA0 = warp_M_off + h;
    int rA1 = warp_M_off + h + 8;
    uint32_t A0 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SF_SMEM_K_STRIDE + kA0]);
    uint32_t A1 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SF_SMEM_K_STRIDE + kA0]);
    uint32_t A2 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA0 * V19SF_SMEM_K_STRIDE + kA2]);
    uint32_t A3 = *reinterpret_cast<const uint32_t*>(
        &A_smem[compute_stage][rA1 * V19SF_SMEM_K_STRIDE + kA2]);

    int sfa_m_row;
    if ((lane & 3) == 1) {
      sfa_m_row = warp_M_off + (lane >> 2) + 8;
    } else {
      sfa_m_row = warp_M_off + (lane >> 2);
    }
    uint32_t SFA = *reinterpret_cast<const uint32_t*>(
        &A_sf_smem[compute_stage][sfa_m_row * V19SF_SF_K_PER_ROW]);

    #pragma unroll
    for (int g = 0; g < V19SF_N_GROUPS; ++g) {
      int base = g * 4;
      int co0 = (base + 0) * 8 + h;
      int co1 = (base + 1) * 8 + h;
      int co2 = (base + 2) * 8 + h;
      int co3 = (base + 3) * 8 + h;
      uint32_t B0 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B1 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co0 * V19SF_SMEM_K_STRIDE + kA2]);
      uint32_t B2 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B3 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co1 * V19SF_SMEM_K_STRIDE + kA2]);
      uint32_t B4 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B5 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co2 * V19SF_SMEM_K_STRIDE + kA2]);
      uint32_t B6 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SF_SMEM_K_STRIDE + kA0]);
      uint32_t B7 = *reinterpret_cast<const uint32_t*>(
          &B_smem[compute_stage][co3 * V19SF_SMEM_K_STRIDE + kA2]);

      int sfb_n = g * 32 + l * 8 + h;
      uint32_t SFB = *reinterpret_cast<const uint32_t*>(
          &B_sf_smem[compute_stage][sfb_n * V19SF_SF_K_PER_ROW]);

      v19sf_mma_m16n8k64_e2m1_4x(
          dA[base+0], dB[base+0],
          dA[base+1], dB[base+1],
          dA[base+2], dB[base+2],
          dA[base+3], dB[base+3],
          dC[base+0], dD[base+0],
          dC[base+1], dD[base+1],
          dC[base+2], dD[base+2],
          dC[base+3], dD[base+3],
          A0, A1, A2, A3,
          B0, B1, B2, B3, B4, B5, B6, B7,
          SFA, SFB);
    }
    compute_stage = next_stage;
  }
  asm volatile("cp.async.wait_all;\n" ::);

  // ── v2 Epilogue: per-Co outer FP32 scale × alpha ──
  const int warp_M_off = warp_id * 16;
  #pragma unroll
  for (int n_atom = 0; n_atom < V19SF_N_ATOMS; ++n_atom) {
    int co_pair = co_base + n_atom * 8 + 2 * l;
    int row0    = m_base + warp_M_off + h;
    int row1    = m_base + warp_M_off + h + 8;
    float b0 = 0.f, b1 = 0.f;
    float ow0 = 0.f, ow1 = 0.f;
    if (co_pair < Co) {
      ow0 = outer_w_fp32[co_pair] * alpha;
      if (bias != nullptr) b0 = __bfloat162float(bias[co_pair]);
    }
    if (co_pair + 1 < Co) {
      ow1 = outer_w_fp32[co_pair + 1] * alpha;
      if (bias != nullptr) b1 = __bfloat162float(bias[co_pair + 1]);
    }
    if (co_pair + 1 < Co) {
      __nv_bfloat162 packAB;
      packAB.x = __float2bfloat16(dA[n_atom] * ow0 + b0);
      packAB.y = __float2bfloat16(dB[n_atom] * ow1 + b1);
      __nv_bfloat162 packCD;
      packCD.x = __float2bfloat16(dC[n_atom] * ow0 + b0);
      packCD.y = __float2bfloat16(dD[n_atom] * ow1 + b1);
      if (row0 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row0 * Co + co_pair]) = packAB;
      }
      if (row1 < M_total) {
        *reinterpret_cast<__nv_bfloat162*>(&y[row1 * Co + co_pair]) = packCD;
      }
    } else {
      auto store = [&](int row, int co, float v, float ow, float bv) {
        if (row < M_total && co < Co) {
          y[row * Co + co] = __float2bfloat16(v * ow + bv);
        }
      };
      store(row0, co_pair + 0, dA[n_atom], ow0, b0);
      store(row0, co_pair + 1, dB[n_atom], ow1, b1);
      store(row1, co_pair + 0, dC[n_atom], ow0, b0);
      store(row1, co_pair + 1, dD[n_atom], ow1, b1);
    }
  }
}

extern "C" int motus_fp4_conv3d_v19sf_ndhwc_bf16out_v2(
    const void* cache_x_fp4,
    const void* new_x_fp4,
    const void* w_fp4,
    const void* cache_sfa, const void* new_sfa, const void* w_sfb,
    const void* outer_w_fp32,                          // [Co] fp32, MUST be non-null
    void* y_bf16,
    const void* bias_bf16,
    int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
    float alpha, cudaStream_t stream)
{
  if (Ci % V19SF_BLOCK_K_ELEM != 0 || Co % 8 != 0) {
    std::fprintf(stderr,
        "[fp4_conv3d_v19sf_v2] Ci%%%d (got %d) or Co%%8 (got %d) bad\n",
        V19SF_BLOCK_K_ELEM, Ci, Co);
    return -1;
  }
  if (T_cache != 2) {
    std::fprintf(stderr, "[fp4_conv3d_v19sf_v2] T_cache must be 2 (got %d)\n", T_cache);
    return -3;
  }
  if (outer_w_fp32 == nullptr) {
    std::fprintf(stderr, "[fp4_conv3d_v19sf_v2] outer_w_fp32 must be non-null\n");
    return -4;
  }
  int M = N * T_new * H * W;
  int M_tiles = (M + V19SF_BLOCK_M - 1) / V19SF_BLOCK_M;
  int N_tiles = (Co + V19SF_BLOCK_N - 1) / V19SF_BLOCK_N;
  int total_tiles = M_tiles * N_tiles;

  dim3 grid(total_tiles);
  dim3 block(V19SF_THREADS);
  fp4_conv3d_v19sf_kernel_v2<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(cache_x_fp4),
      reinterpret_cast<const uint8_t*>(new_x_fp4),
      reinterpret_cast<const uint8_t*>(w_fp4),
      reinterpret_cast<const uint8_t*>(cache_sfa),
      reinterpret_cast<const uint8_t*>(new_sfa),
      reinterpret_cast<const uint8_t*>(w_sfb),
      reinterpret_cast<const float*>(outer_w_fp32),
      reinterpret_cast<__nv_bfloat16*>(y_bf16),
      reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
      N, T_cache, T_new, H, W, Ci, Co, alpha,
      M_tiles, N_tiles);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[fp4_conv3d_v19sf_v2] launch err: %s\n",
                 cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

}  // namespace conv
}  // namespace flash_rt

// SPDX-License-Identifier: Apache-2.0
//
// Routed experts of a prefill chunk (up to kMaxChunk rows) on the SM120 block-scaled tensor-core MMAs: the slots
// (row, k) sorted by expert (expert_sort), then per expert tiles of 128 output rows x 64 slots, one block each
// (8 warps x 16 rows, 8 n-tiles of 8 slots), the weights read once per tile straight from the checkpoint layout.
//   NVFP4: mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64 (E2M1 x E2M1, UE4M3 per 16)
//   MXFP8: mma.sync kind::mxf8f6f4.block_scale.scale_vec::1X m16n8k32 (E4M3 x E4M3, UE8M0 per 32)
// Both MMAs take 32 bytes of each A row and B column per instruction in the same register layout (lane = 4 g + c:
// rows g, g + 8 at bytes 4 c and 16 + 4 c; B column g), so the checkpoint codes and scale rows load directly.
// The quantities are the decode kernels' (block products x weight scale x input scale, the amax scales in the
// epilogue); only the FP32 accumulation order differs.

#include <cub/block/block_radix_sort.cuh>

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kTileRows = 128;            // output rows per block
constexpr int kTileSlots = 64;            // slots per block
constexpr int kMmaThreads = 256;
constexpr int kSortThreads = 1024;
constexpr int kSortItems = kMaxChunk * kTopK / kSortThreads;
static_assert(kSortItems * kSortThreads >= kMaxChunk * kTopK, "sort capacity");

__device__ __forceinline__ void mma_nvfp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                          uint32_t sfb) {
  constexpr uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z),
        "h"(z));
}

__device__ __forceinline__ void mma_mxfp8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                          uint32_t sfb) {
  constexpr uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z),
        "h"(z));
}

__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }

// one block: the slots sorted by (expert, slot) (stable, deterministic), the experts' ranges and 64-slot tiles
__global__ void __launch_bounds__(kSortThreads) expert_sort_kernel(const __grid_constant__ ExpertParams p) {
  using Sort = cub::BlockRadixSort<uint32_t, kSortThreads, kSortItems>;
  __shared__ typename Sort::TempStorage tmp;
  __shared__ int cnt[kExperts];
  pdl_wait();
  pdl_launch();
  const int slots = p.rows * kTopK, t = threadIdx.x;
  const int* ids = after_wait(p.ids);
  if (t < kExperts) cnt[t] = 0;
  uint32_t key[kSortItems];
#pragma unroll
  for (int i = 0; i < kSortItems; ++i) {
    const int s = t * kSortItems + i;
    key[i] = s < slots ? (uint32_t(ids[s]) << 12) | uint32_t(s) : 0xffffffffu;
  }
  __syncthreads();
  Sort(tmp).Sort(key, 0, 19);              // 12 slot bits + 7 expert bits
  int* order = after_wait(p.order);
#pragma unroll
  for (int i = 0; i < kSortItems; ++i) {
    const int s = t * kSortItems + i;
    if (s < slots) {
      order[s] = int(key[i] & 0xfff);
      atomicAdd(&cnt[key[i] >> 12], 1);
    }
  }
  __syncthreads();
  if (t < 32) {                            // exclusive scans over the 128 experts, 4 per lane
    int c[4], ct[4], sc = 0, st = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      c[i] = cnt[4 * t + i];
      ct[i] = (c[i] + kTileSlots - 1) / kTileSlots;
      sc += c[i];
      st += ct[i];
    }
    int ic = sc, it = st;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, ic, o), b = __shfl_up_sync(0xffffffffu, it, o);
      if (t >= o) { ic += a; it += b; }
    }
    ic -= sc;
    it -= st;
    int* off = after_wait(p.offset);
    int* toff = after_wait(p.tile_off);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      off[4 * t + i] = ic;
      toff[4 * t + i] = it;
      ic += c[i];
      it += ct[i];
    }
    if (t == 31) { off[kExperts] = ic; toff[kExperts] = it; }
  }
}

// The block's tile: expert e (the last with tile_off[e] <= blockIdx.y), slots s0 .. s1 - 1 of the sorted order.
__device__ __forceinline__ bool tile_of(const ExpertParams& p, int& e, int& s0, int& s1) {
  const int* toff = after_wait(p.tile_off);
  const int ty = blockIdx.y;
  if (ty >= toff[kExperts]) return false;
  int lo = 0, hi = kExperts;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (toff[mid] <= ty) lo = mid;
    else hi = mid;
  }
  e = lo;
  const int* off = after_wait(p.offset);
  s0 = off[e] + (ty - toff[e]) * kTileSlots;
  s1 = min(off[e + 1], s0 + kTileSlots);
  return true;
}

// acc[j] (n-tile j: slots 8 j .. 8 j + 7 of the tile) over the k chunks: A = 16 weight rows of this warp, B = the
// input rows of the tile's slots (in_row[j]: the B column of this lane in n-tile j).
template <bool MX, int K>
__device__ __forceinline__ void tile_mma(const uint8_t* W, const uint8_t* WS, int f0, const uint8_t* X,
                                         const uint8_t* XS, const int (&in_row)[8], int nt, float (&acc)[8][4]) {
  constexpr int kChunks = K / (MX ? 32 : 64);
  constexpr int rowbytes = MX ? K : K / 2, sbytes = K / (MX ? 32 : 16);
  const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const uint8_t* a0 = W + size_t(f0 + g) * rowbytes + 4 * c;
  const uint8_t* a1 = a0 + 8 * size_t(rowbytes);
  const uint8_t* sa = WS + size_t(f0 + (lane & 1) * 8 + g) * sbytes;
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) acc[j][q] = 0.0f;
#pragma unroll 4
  for (int kc = 0; kc < kChunks; ++kc) {
    const uint32_t a[4] = {ld32(a0 + 32 * kc), ld32(a1 + 32 * kc), ld32(a0 + 32 * kc + 16), ld32(a1 + 32 * kc + 16)};
    const uint32_t sfa = MX ? uint32_t(sa[kc]) : ld32(sa + 4 * kc);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j >= nt) break;
      const uint8_t* xr = X + size_t(in_row[j]) * rowbytes + 32 * kc + 4 * c;
      const uint32_t b0 = ld32(xr), b1 = ld32(xr + 16);
      const uint8_t* xs = XS + size_t(in_row[j]) * sbytes;
      const uint32_t sfb = MX ? uint32_t(xs[kc]) : ld32(xs + 4 * kc);
      if (MX) mma_mxfp8(acc[j], a, b0, b1, sfa, sfb);
      else mma_nvfp4(acc[j], a, b0, b1, sfa, sfb);
    }
  }
}

// w13: rows 2 pair (gate) / 2 pair + 1 (up); act[slot][pair] = bf16(softcapped_swiglu(bf16(g * outer), bf16(u * outer)))
template <bool MX>
__device__ void expert_mma_w13_body(const ExpertParams& p) {
  __shared__ uint16_t tile[kTileSlots][kTileRows / 2];
  int e, s0, s1;
  if (!tile_of(p, e, s0, s1)) return;
  const int ns = s1 - s0, nt = (ns + 7) / 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int f0 = blockIdx.x * kTileRows + warp * 16;
  constexpr int K = kHidden, rowbytes = MX ? K : K / 2, sbytes = K / (MX ? 32 : 16);
  const int* order = after_wait(p.order);
  int in_row[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) in_row[j] = order[s0 + min(8 * j + g, ns - 1)] / kTopK;
  float acc[8][4];
  tile_mma<MX, K>(static_cast<const uint8_t*>(p.codes) + size_t(e) * 2 * kInter * rowbytes,
                  static_cast<const uint8_t*>(p.scales) + size_t(e) * 2 * kInter * sbytes, f0,
                  after_wait(p.xq_codes), after_wait(p.xq_sc8), in_row, nt, acc);
  const float we = MX ? 1.0f : __fdiv_rn(p.amax[e], p.denominator);
  const float* xa = after_wait(p.xq_amax);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (j >= nt) break;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float partner = __shfl_xor_sync(0xffffffffu, acc[j][q], 4);   // row g ^ 1
      const int sl = 8 * j + 2 * c + (q & 1);                             // slot of the tile
      if ((g & 1) == 0 && sl < ns) {
        const int row = s0 + sl, r = order[row] / kTopK;
        const float outer = MX ? 1.0f : __fmul_rn(we, __fmul_rn(xa[r], float(1.0 / 1536.0)));
        const float vg = round_bf16(__fmul_rn(acc[j][q], outer)), vu = round_bf16(__fmul_rn(partner, outer));
        tile[sl][(warp * 16 + g + 8 * (q >> 1)) / 2] = f32_to_bf16_bits(softcapped_swiglu(vg, vu, p.gate_cap, p.up_cap));
      }
    }
  }
  __syncthreads();
  uint16_t* act = after_wait(p.act);
  const int pair0 = blockIdx.x * (kTileRows / 2);
  for (int i = threadIdx.x; i < ns * (kTileRows / 2 / 8); i += kMmaThreads) {
    const int sl = i / (kTileRows / 2 / 8), ch = i % (kTileRows / 2 / 8);
    const int slot = order[s0 + sl];
    reinterpret_cast<uint4*>(act + size_t(slot) * kInter + pair0)[ch] = reinterpret_cast<const uint4*>(tile[sl])[ch];
  }
}

// w2: out[slot][row] = bf16(dot * outer), the slot's quantized activation as B
template <bool MX>
__device__ void expert_mma_w2_body(const ExpertParams& p) {
  __shared__ uint16_t tile[kTileSlots][kTileRows];
  int e, s0, s1;
  if (!tile_of(p, e, s0, s1)) return;
  const int ns = s1 - s0, nt = (ns + 7) / 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int f0 = blockIdx.x * kTileRows + warp * 16;
  constexpr int K = kInter, rowbytes = MX ? K : K / 2, sbytes = K / (MX ? 32 : 16);
  const int* order = after_wait(p.order);
  int in_row[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) in_row[j] = order[s0 + min(8 * j + g, ns - 1)];
  float acc[8][4];
  tile_mma<MX, K>(static_cast<const uint8_t*>(p.codes) + size_t(e) * kHidden * rowbytes,
                  static_cast<const uint8_t*>(p.scales) + size_t(e) * kHidden * sbytes, f0,
                  after_wait(p.xq_codes), after_wait(p.xq_sc8), in_row, nt, acc);
  const float we = MX ? 1.0f : __fdiv_rn(p.amax[e], p.denominator);
  const float* sa = after_wait(p.xq_amax);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    if (j >= nt) break;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int sl = 8 * j + 2 * c + (q & 1);
      if (sl < ns) {
        const float outer = MX ? 1.0f : __fmul_rn(we, __fmul_rn(sa[order[s0 + sl]], float(1.0 / 1536.0)));
        tile[sl][warp * 16 + g + 8 * (q >> 1)] = f32_to_bf16_bits(__fmul_rn(acc[j][q], outer));
      }
    }
  }
  __syncthreads();
  uint16_t* out = after_wait(p.out);
  const int r0 = blockIdx.x * kTileRows;
  for (int i = threadIdx.x; i < ns * (kTileRows / 8); i += kMmaThreads) {
    const int sl = i / (kTileRows / 8), ch = i % (kTileRows / 8);
    const int slot = order[s0 + sl];
    reinterpret_cast<uint4*>(out + size_t(slot) * kHidden + r0)[ch] = reinterpret_cast<const uint4*>(tile[sl])[ch];
  }
}

__global__ void __launch_bounds__(kMmaThreads) expert_mma_w13_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_mma_w13_body<true>(p);
  else expert_mma_w13_body<false>(p);
}

__global__ void __launch_bounds__(kMmaThreads) expert_mma_w2_kernel(const __grid_constant__ ExpertParams p) {
  pdl_wait();
  pdl_launch();
  if (p.mxfp8) expert_mma_w2_body<true>(p);
  else expert_mma_w2_body<false>(p);
}

void check_mma(const ExpertParams& p, const char* what) {
  if (p.rows < 1 || p.rows > kMaxChunk || !p.ids || !p.order || !p.offset || !p.tile_off)
    throw std::invalid_argument(std::string(what) + ": 1..kMaxChunk rows, ids, order, offset, tile_off");
}

}  // namespace

void expert_sort(const ExpertParams& p, cudaStream_t stream) {
  check_mma(p, "expert_sort");
  launch_pdl(expert_sort_kernel, dim3(1), dim3(kSortThreads), 0, stream, p);
}

void expert_mma_w13(const ExpertParams& p, cudaStream_t stream) {
  check_mma(p, "expert_mma_w13");
  if (!p.xq_codes || !p.xq_sc8 || !p.xq_amax || !p.act || p.max_tiles < 1)
    throw std::invalid_argument("expert_mma_w13: xq_codes, xq_sc8, xq_amax, act, max_tiles");
  launch_pdl(expert_mma_w13_kernel, dim3(2 * kInter / kTileRows, p.max_tiles), dim3(kMmaThreads), 0, stream, p);
}

void expert_mma_w2(const ExpertParams& p, cudaStream_t stream) {
  check_mma(p, "expert_mma_w2");
  if (!p.xq_codes || !p.xq_sc8 || !p.xq_amax || !p.out || p.max_tiles < 1)
    throw std::invalid_argument("expert_mma_w2: xq_codes, xq_sc8, xq_amax, out, max_tiles");
  launch_pdl(expert_mma_w2_kernel, dim3(kHidden / kTileRows, p.max_tiles), dim3(kMmaThreads), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt

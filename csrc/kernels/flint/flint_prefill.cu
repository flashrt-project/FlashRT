// SPDX-License-Identifier: Apache-2.0
//
// Flint prefill row kernels (many tokens per launch): embedding, Q/K norm +
// RoPE + KV write, router + top-6, deterministic expert sort with the grouped
// GEMM tile list, and the weighted MoE combine. Numerics match the decode
// kernels row for row.

#include "flint_common.cuh"
#include "flint_kernels.cuh"

#include <cfloat>

namespace flash_rt {
namespace flint {

namespace {
constexpr int kHidden = 2048, kHeadDim = 128, kQHeads = 32, kKVHeads = 4, kExperts = 256, kTopK = 6;
}

// out[r] = table[tokens[*pos + r]], r < n
__global__ void embed_rows_kernel(const int* __restrict__ tokens, const int* __restrict__ pos,
                                  const uint16_t* __restrict__ table, uint16_t* __restrict__ out) {
  const int r = blockIdx.x;
  const int t = tokens[*pos + r];
  reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * kHidden)[threadIdx.x] =
      reinterpret_cast<const uint4*>(table + static_cast<size_t>(t) * kHidden)[threadIdx.x];
}

void embed_rows_bf16(const int* tokens, const int* pos, int n, const void* table, void* out, cudaStream_t s) {
  embed_rows_kernel<<<n, kHidden / 8, 0, s>>>(tokens, pos, static_cast<const uint16_t*>(table),
                                             static_cast<uint16_t*>(out));
}

// grid (n rows), 256 threads: warp w handles heads w, w + 8, ... of the 40 (32 Q, 4 K, 4 V); each head is
// one warp-wide pass (lane holds dims lane + 32j). Row r is at position *pos + r.
__global__ void __launch_bounds__(256)
qk_norm_rope_kv_rows_kernel(const uint16_t* __restrict__ qkv, const float* __restrict__ qw,
                            const float* __restrict__ kw, const float* __restrict__ cos_t,
                            const float* __restrict__ sin_t, const int* __restrict__ pos_p,
                            uint16_t* __restrict__ q_out, uint16_t* __restrict__ cache, int ring, int rope,
                            float eps) {
  const int r = blockIdx.x, lane = threadIdx.x & 31;
  const int pos = *pos_p + r;
  const int slot = ring > 0 ? (pos & (ring - 1)) : pos;
  const uint16_t* row = qkv + static_cast<size_t>(r) * (kQHeads + 2 * kKVHeads) * kHeadDim;
  for (int head = threadIdx.x >> 5; head < kQHeads + 2 * kKVHeads; head += 8) {
    const uint16_t* src;
    uint16_t* dst;
    const float* w = nullptr;
    if (head < kQHeads) {
      src = row + head * kHeadDim; dst = q_out + (static_cast<size_t>(r) * kQHeads + head) * kHeadDim; w = qw;
    } else if (head < kQHeads + kKVHeads) {
      const int h = head - kQHeads;
      src = row + (kQHeads + h) * kHeadDim; dst = cache + static_cast<size_t>(slot) * 1024 + h * kHeadDim; w = kw;
    } else {
      const int h = head - kQHeads - kKVHeads;
      src = row + (kQHeads + kKVHeads + h) * kHeadDim; dst = cache + static_cast<size_t>(slot) * 1024 + 512 + h * kHeadDim;
    }
    float v[4] = {bf16_to_f32(src[lane]), bf16_to_f32(src[lane + 32]), bf16_to_f32(src[lane + 64]),
                  bf16_to_f32(src[lane + 96])};
    if (w != nullptr) {
      float ss = warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
      const float rs = rsqrtf(ss / kHeadDim + eps);
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = round_bf16(v[j] * rs * w[lane + 32 * j]);
      if (rope) {
        const float* ct = cos_t + static_cast<size_t>(pos) * 64;
        const float* st = sin_t + static_cast<size_t>(pos) * 64;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int d = lane + 32 * j;
          const float c = ct[d], sn = st[d], n1 = v[j], n2 = v[j + 2];
          v[j] = fmaf(n1, c, -(n2 * sn));
          v[j + 2] = fmaf(n2, c, n1 * sn);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = f32_to_bf16_bits(v[j]);
  }
}

void qk_norm_rope_kv_rows_bf16(const void* qkv, int n, const float* qw, const float* kw, const float* cos_t,
                               const float* sin_t, const int* pos, void* q_out, void* cache, int ring, int rope,
                               float eps, cudaStream_t s) {
  qk_norm_rope_kv_rows_kernel<<<n, 256, 0, s>>>(
      static_cast<const uint16_t*>(qkv), qw, kw, cos_t, sin_t, pos, static_cast<uint16_t*>(q_out),
      static_cast<uint16_t*>(cache), ring, rope, eps);
}

// Router for 8 tokens per block: thread e computes logit[e] for the block's tokens from the
// K-major router copy wt [2048][256]; one warp per token then selects top-6 (same rule as decode).
__global__ void __launch_bounds__(256)
router_rows_kernel(const uint16_t* __restrict__ xn, int n, const uint16_t* __restrict__ wt,
                   int* __restrict__ ids, float* __restrict__ tw) {
  __shared__ __align__(16) uint16_t xs[8][kHidden];
  __shared__ float lg[8][kExperts];
  const int r0 = blockIdx.x * 8;
  const int nr = min(8, n - r0);
  for (int i = threadIdx.x; i < 8 * kHidden / 8; i += 256) {
    const int rr = i / (kHidden / 8), c = i % (kHidden / 8);
    reinterpret_cast<uint4*>(xs[rr])[c] =
        rr < nr ? reinterpret_cast<const uint4*>(xn + static_cast<size_t>(r0 + rr) * kHidden)[c] : make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  const int e = threadIdx.x;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < kHidden; k += 2) {
    const float w0 = bf16_to_f32(wt[static_cast<size_t>(k) * kExperts + e]);
    const float w1 = bf16_to_f32(wt[static_cast<size_t>(k + 1) * kExperts + e]);
#pragma unroll
    for (int rr = 0; rr < 8; ++rr) {
      const uint32_t xv = *reinterpret_cast<const uint32_t*>(&xs[rr][k]);
      acc[rr] = fmaf(w0, bf16lo(xv), acc[rr]);
      acc[rr] = fmaf(w1, bf16hi(xv), acc[rr]);
    }
  }
#pragma unroll
  for (int rr = 0; rr < 8; ++rr) lg[rr][e] = round_bf16(acc[rr]);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp >= nr) return;
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = lg[warp][lane * 8 + j];
  float sv[kTopK];
  int si[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float bv = -FLT_MAX; int bi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (v[j] > bv) { bv = v[j]; bi = lane * 8 + j; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    sv[k] = bv; si[k] = bi;
    if ((bi >> 3) == lane) v[bi & 7] = -FLT_MAX;
  }
  if (lane == 0) {
    float sg[kTopK], sum = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) { sg[k] = __fdiv_rn(1.0f, 1.0f + expf(-sv[k])); sum += sg[k]; }
    const float den = sum + 1e-20f;
    const int row = r0 + warp;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      ids[row * kTopK + k] = si[k];
      tw[row * kTopK + k] = round_bf16(__fdiv_rn(sg[k], den));
    }
  }
}

void router_rows_bf16(const void* xn, int n, const void* wt, int* ids, float* tw, cudaStream_t s) {
  router_rows_kernel<<<(n + 7) / 8, 256, 0, s>>>(static_cast<const uint16_t*>(xn), n,
                                                 static_cast<const uint16_t*>(wt), ids, tw);
}

// Single-block sort of the n*6 (token, slot) pairs by expert.
//   sorted_tok[i]  : token row of sorted slot i;  inv[t*6+j] : sorted slot of (t, j)
//   tiles[3*k]     : (expert, first sorted slot, rows) for the grouped GEMM, tile_rows per tile
//   n_tiles[0]     : number of tiles
// Within an expert, slots keep (token, j) order, so the layout is deterministic.
__global__ void __launch_bounds__(1024)
moe_sort_kernel(const int* __restrict__ ids, int n, int tile_rows, int* __restrict__ sorted_tok,
                int* __restrict__ inv, int* __restrict__ tiles, int* __restrict__ n_tiles) {
  __shared__ int cnt[kExperts], off[kExperts], toff[kExperts];
  const int tid = threadIdx.x, total = n * kTopK;
  if (tid < kExperts) cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < total; i += blockDim.x) atomicAdd(&cnt[ids[i]], 1);
  __syncthreads();
  if (tid == 0) {
    int a = 0, b = 0;
    for (int e = 0; e < kExperts; ++e) {
      off[e] = a; toff[e] = b;
      a += cnt[e];
      b += (cnt[e] + tile_rows - 1) / tile_rows;
    }
    *n_tiles = b;
  }
  __syncthreads();
  if (tid < kExperts) {
    const int e = tid, c = cnt[e];
    for (int k = 0; k * tile_rows < c; ++k) {
      const int ti = toff[e] + k;
      tiles[3 * ti] = e; tiles[3 * ti + 1] = off[e] + k * tile_rows; tiles[3 * ti + 2] = min(tile_rows, c - k * tile_rows);
    }
  }
  // Stable scatter: each expert's slots in increasing (token, j) order. One warp per expert group of 8.
  const int warp = tid >> 5, lane = tid & 31;
  for (int e = warp; e < kExperts; e += blockDim.x >> 5) {
    int base = off[e];
    for (int i0 = 0; i0 < total; i0 += 32) {
      const int i = i0 + lane;
      const bool hit = i < total && ids[i] == e;
      const unsigned m = __ballot_sync(0xffffffffu, hit);
      if (hit) {
        const int slot = base + __popc(m & ((1u << lane) - 1u));
        sorted_tok[slot] = i / kTopK;
        inv[i] = slot;
      }
      base += __popc(m);
    }
  }
}

void moe_sort(const int* ids, int n, int tile_rows, int* sorted_tok, int* inv, int* tiles, int* n_tiles,
              cudaStream_t s) {
  moe_sort_kernel<<<1, 1024, 0, s>>>(ids, n, tile_rows, sorted_tok, inv, tiles, n_tiles);
}

// Expert offsets from the router's per-expert counts (re-zeroed here for the next layer), the grouped
// GEMM tile list, and the scatter of (token, slot) pairs into expert order. Slot order inside an expert
// is arbitrary: every GEMM row and the combine depend only on their own token, so results are exact
// and deterministic regardless of the placement order.
__global__ void __launch_bounds__(1024)
moe_scan_place_kernel(const int* __restrict__ ids, int n, int tile_rows, int* __restrict__ counts,
                      int* __restrict__ sorted_tok, int* __restrict__ inv, int* __restrict__ tiles,
                      int* __restrict__ n_tiles) {
  __shared__ int off[kExperts], cur[kExperts], wsum[2][kExperts / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int c = 0, nt = 0;
  if (tid < kExperts) {
    c = counts[tid];
    counts[tid] = 0;
    nt = (c + tile_rows - 1) / tile_rows;
    cur[tid] = 0;
  }
  int sc = c, st = nt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int a = __shfl_up_sync(0xffffffffu, sc, o), b = __shfl_up_sync(0xffffffffu, st, o);
    if (lane >= o) { sc += a; st += b; }
  }
  if (tid < kExperts && lane == 31) { wsum[0][warp] = sc; wsum[1][warp] = st; }
  __syncthreads();
  if (tid < kExperts) {
    int bc = 0, bt = 0;
    for (int w = 0; w < warp; ++w) { bc += wsum[0][w]; bt += wsum[1][w]; }
    const int o = bc + sc - c, to = bt + st - nt;
    off[tid] = o;
    if (tid == kExperts - 1) *n_tiles = bt + st;
    for (int k = 0; k < nt; ++k) {
      const int ti = to + k;
      tiles[3 * ti] = tid; tiles[3 * ti + 1] = o + k * tile_rows; tiles[3 * ti + 2] = min(tile_rows, c - k * tile_rows);
    }
  }
  __syncthreads();
  for (int i = tid; i < n * kTopK; i += blockDim.x) {
    const int e = ids[i];
    const int slot = off[e] + atomicAdd(&cur[e], 1);
    sorted_tok[slot] = i / kTopK;
    inv[i] = slot;
  }
}

void moe_scan_place(const int* ids, int n, int tile_rows, int* counts, int* sorted_tok, int* inv, int* tiles,
                    int* n_tiles, cudaStream_t s) {
  moe_scan_place_kernel<<<1, 1024, 0, s>>>(ids, n, tile_rows, counts, sorted_tok, inv, tiles, n_tiles);
}

// out[t] = bf16( bf16(sum_j tw[t,j] * y[inv[t,j]]) + shared[t] )  (y already bf16(acc * gs_e))
__global__ void moe_combine_rows_kernel(const uint16_t* __restrict__ y, const int* __restrict__ inv,
                                        const float* __restrict__ tw, const uint16_t* __restrict__ shared,
                                        uint16_t* __restrict__ out) {
  const int t = blockIdx.x, c0 = threadIdx.x * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    const float w = tw[t * kTopK + j];
    const uint4 v = *reinterpret_cast<const uint4*>(y + static_cast<size_t>(inv[t * kTopK + j]) * kHidden + c0);
    const uint32_t vw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acc[2 * q] = fmaf(w, bf16lo(vw[q]), acc[2 * q]);
      acc[2 * q + 1] = fmaf(w, bf16hi(vw[q]), acc[2 * q + 1]);
    }
  }
  const uint4 sh = *reinterpret_cast<const uint4*>(shared + static_cast<size_t>(t) * kHidden + c0);
  const uint32_t sw[4] = {sh.x, sh.y, sh.z, sh.w};
  uint4 o;
  uint32_t* op = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
  for (int q = 0; q < 4; ++q)
    op[q] = pack_bf16x2(round_bf16(acc[2 * q]) + bf16lo(sw[q]), round_bf16(acc[2 * q + 1]) + bf16hi(sw[q]));
  *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kHidden + c0) = o;
}

void moe_combine_rows_bf16(const void* y, const int* inv, const float* tw, const void* shared, void* out, int n,
                           cudaStream_t s) {
  moe_combine_rows_kernel<<<n, kHidden / 8, 0, s>>>(static_cast<const uint16_t*>(y), inv, tw,
                                                   static_cast<const uint16_t*>(shared), static_cast<uint16_t*>(out));
}

// *pos += n (device-side position advance after a prefill chunk)
__global__ void advance_pos_kernel(int* pos, int n) { *pos += n; }
void advance_pos(int* pos, int n, cudaStream_t s) { advance_pos_kernel<<<1, 1, 0, s>>>(pos, n); }

}  // namespace flint
}  // namespace flash_rt

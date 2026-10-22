// SPDX-License-Identifier: Apache-2.0
//
// Flint prefill row kernels (many tokens per launch): embedding, Q/K norm +
// RoPE + KV write, expert placement with the grouped GEMM tile list, and the
// weighted MoE combine. Numerics match the decode kernels row for row.

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

// Expert offsets from the router's per-expert counts (re-zeroed here for the next layer), the grouped
// GEMM tile list, and the scatter of (token, slot) pairs into expert order. Slot order inside an expert
// is arbitrary: every GEMM row and the combine depend only on their own token, so results are exact
// and deterministic regardless of the placement order. Rows at positions >= *n_end (padding of a
// graph-shaped chunk, not counted by the router) are not placed; their slots point at slot 0.
__global__ void __launch_bounds__(1024)
moe_scan_place_kernel(const int* __restrict__ ids, int n, int tile_rows, int* __restrict__ counts,
                      int* __restrict__ sorted_tok, int* __restrict__ inv, int* __restrict__ tiles,
                      int* __restrict__ n_tiles, const int* __restrict__ pos, const int* __restrict__ n_end) {
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
  const int placed = (pos != nullptr ? min(n, *n_end - *pos) : n) * kTopK;
  for (int i = tid; i < n * kTopK; i += blockDim.x) {
    if (i >= placed) { inv[i] = 0; continue; }
    const int e = ids[i];
    const int slot = off[e] + atomicAdd(&cur[e], 1);
    sorted_tok[slot] = i / kTopK;
    inv[i] = slot;
  }
}

void moe_scan_place(const int* ids, int n, int tile_rows, int* counts, int* sorted_tok, int* inv, int* tiles,
                    int* n_tiles, const int* pos, const int* n_end, cudaStream_t s) {
  moe_scan_place_kernel<<<1, 1024, 0, s>>>(ids, n, tile_rows, counts, sorted_tok, inv, tiles, n_tiles, pos, n_end);
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

// The combine above followed by the next layer's input norm (add_rmsnorm_bf16 with the same 256 x 8 thread
// layout and reduction order, so both outputs are bit-identical to the two-kernel sequence):
//   x = combine(t); s = x + residual[t]; residual[t] = bf16(s); out[t] = bf16(s * rsqrt(mean(s^2) + eps) * w)
__global__ void moe_combine_norm_rows_kernel(const uint16_t* __restrict__ y, const int* __restrict__ inv,
                                             const float* __restrict__ tw, const uint16_t* __restrict__ shared,
                                             uint16_t* __restrict__ residual, const float* __restrict__ w,
                                             uint16_t* __restrict__ out, float eps) {
  const int t = blockIdx.x, c0 = threadIdx.x * 8;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    const float wj = tw[t * kTopK + j];
    const uint4 v = *reinterpret_cast<const uint4*>(y + static_cast<size_t>(inv[t * kTopK + j]) * kHidden + c0);
    const uint32_t vw[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acc[2 * q] = fmaf(wj, bf16lo(vw[q]), acc[2 * q]);
      acc[2 * q + 1] = fmaf(wj, bf16hi(vw[q]), acc[2 * q + 1]);
    }
  }
  const uint4 sh = *reinterpret_cast<const uint4*>(shared + static_cast<size_t>(t) * kHidden + c0);
  const uint4 rv = *reinterpret_cast<const uint4*>(residual + static_cast<size_t>(t) * kHidden + c0);
  const uint32_t sw[4] = {sh.x, sh.y, sh.z, sh.w}, rw[4] = {rv.x, rv.y, rv.z, rv.w};
  float s[8];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const uint32_t x = pack_bf16x2(round_bf16(acc[2 * q]) + bf16lo(sw[q]), round_bf16(acc[2 * q + 1]) + bf16hi(sw[q]));
    s[2 * q] = bf16lo(x) + bf16lo(rw[q]);
    s[2 * q + 1] = bf16hi(x) + bf16hi(rw[q]);
  }
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) ss = fmaf(s[j], s[j], ss);
  ss = warp_sum(ss);
  __shared__ float red[8];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) tot += red[k];
  const float rs = rsqrtf(tot / kHidden + eps);
  const float4 w0 = reinterpret_cast<const float4*>(w + c0)[0];
  const float4 w1 = reinterpret_cast<const float4*>(w + c0)[1];
  const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
  uint4 ro, oo;
  uint32_t* rp = reinterpret_cast<uint32_t*>(&ro);
  uint32_t* op = reinterpret_cast<uint32_t*>(&oo);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    rp[j] = pack_bf16x2(s[2 * j], s[2 * j + 1]);
    op[j] = pack_bf16x2(s[2 * j] * rs * wv[2 * j], s[2 * j + 1] * rs * wv[2 * j + 1]);
  }
  *reinterpret_cast<uint4*>(residual + static_cast<size_t>(t) * kHidden + c0) = ro;
  *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * kHidden + c0) = oo;
}

void moe_combine_norm_rows_bf16(const void* y, const int* inv, const float* tw, const void* shared, void* residual,
                                const float* w, void* out, int n, float eps, cudaStream_t s) {
  moe_combine_norm_rows_kernel<<<n, kHidden / 8, 0, s>>>(static_cast<const uint16_t*>(y), inv, tw,
                                                        static_cast<const uint16_t*>(shared),
                                                        static_cast<uint16_t*>(residual), w,
                                                        static_cast<uint16_t*>(out), eps);
}

// *pos += n (device-side position advance after a prefill chunk)
__global__ void advance_pos_kernel(int* pos, int n) { *pos += n; }
void advance_pos(int* pos, int n, cudaStream_t s) { advance_pos_kernel<<<1, 1, 0, s>>>(pos, n); }

}  // namespace flint
}  // namespace flash_rt

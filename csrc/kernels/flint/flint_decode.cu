// SPDX-License-Identifier: Apache-2.0
//
// Flint decode kernels (one token per step, SM120). Every launch reads the
// token position from device memory, so one captured CUDA graph serves every
// decode step.

#include "flint_common.cuh"
#include "flint_kernels.cuh"

#include <cfloat>

namespace flash_rt {
namespace flint {

constexpr int kHidden = 2048;
constexpr int kHeadDim = 128;
constexpr int kQHeads = 32;
constexpr int kKVHeads = 4;
constexpr int kExperts = 256;
constexpr int kTopK = 6;
constexpr int kExpertInter = 512;
constexpr int kSharedInter = 1024;

// ---------------------------------------------------------------------------
// Embedding of the token at the device position: out = table[tokens[*pos]]
// ---------------------------------------------------------------------------
__global__ void embed_at_pos_kernel(const int* __restrict__ tokens, const int* __restrict__ pos,
                                    const uint16_t* __restrict__ table, uint16_t* __restrict__ out) {
  const int t = tokens[*pos];
  const uint4* src = reinterpret_cast<const uint4*>(table + static_cast<size_t>(t) * kHidden);
  reinterpret_cast<uint4*>(out)[threadIdx.x] = src[threadIdx.x];
}

void embed_at_pos_bf16(const int* tokens, const int* pos, const void* table, void* out, cudaStream_t s) {
  embed_at_pos_kernel<<<1, kHidden / 8, 0, s>>>(tokens, pos, static_cast<const uint16_t*>(table),
                                                static_cast<uint16_t*>(out));
}

// ---------------------------------------------------------------------------
// Fused residual add + RMSNorm (reference rounding):
//   s = f32(x) + f32(residual); residual = bf16(s);
//   out = bf16(s * rsqrt(mean(s^2) + eps) * w)    (variance from the unrounded s)
// first != 0: s = f32(x) (no residual yet; residual = x).
// One block of 256 threads per row, 8 values per thread.
// ---------------------------------------------------------------------------
__global__ void add_rmsnorm_kernel(const uint16_t* __restrict__ x, uint16_t* __restrict__ residual,
                                   const float* __restrict__ w, uint16_t* __restrict__ out,
                                   float eps, int first, int write_res) {
  const int row = blockIdx.x;
  const int i0 = threadIdx.x * 8;
  const uint4 xv = reinterpret_cast<const uint4*>(x + row * kHidden)[threadIdx.x];
  float s[8];
  const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) { s[2 * j] = bf16lo(xw[j]); s[2 * j + 1] = bf16hi(xw[j]); }
  if (!first) {
    const uint4 rv = reinterpret_cast<const uint4*>(residual + row * kHidden)[threadIdx.x];
    const uint32_t rw[4] = {rv.x, rv.y, rv.z, rv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { s[2 * j] += bf16lo(rw[j]); s[2 * j + 1] += bf16hi(rw[j]); }
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
  uint4 ro, oo;
  uint32_t* rp = reinterpret_cast<uint32_t*>(&ro);
  uint32_t* op = reinterpret_cast<uint32_t*>(&oo);
  const float4 w0 = reinterpret_cast<const float4*>(w + i0)[0];
  const float4 w1 = reinterpret_cast<const float4*>(w + i0)[1];
  const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    rp[j] = pack_bf16x2(s[2 * j], s[2 * j + 1]);
    op[j] = pack_bf16x2(s[2 * j] * rs * wv[2 * j], s[2 * j + 1] * rs * wv[2 * j + 1]);
  }
  if (write_res) reinterpret_cast<uint4*>(residual + row * kHidden)[threadIdx.x] = ro;
  reinterpret_cast<uint4*>(out + row * kHidden)[threadIdx.x] = oo;
}

void add_rmsnorm_bf16(const void* x, void* residual, const float* w, void* out, int rows,
                      float eps, int first, cudaStream_t s, int write_res) {
  add_rmsnorm_kernel<<<rows, 256, 0, s>>>(static_cast<const uint16_t*>(x), static_cast<uint16_t*>(residual),
                                          w, static_cast<uint16_t*>(out), eps, first, write_res);
}

// ---------------------------------------------------------------------------
// W4A16 GEMV, M = 1.
//   codes [N, K/2] (element 2i low nibble), scales [N, K/16] UE4M3, x bf16 [K].
// Lane l of a warp owns the 16-byte code chunks at byte offset c*512 + 16*l,
// i.e. elements [c*1024 + 32*l, +32) with scale bytes 2*(c*64 + l) .. +1.
// EPI 0: out bf16 = bf16(acc * gs); 1: out f32 = acc * gs;
// EPI 2: gate/up SwiGLU (dense/shared rule). Gate and up rows are interleaved
//        by 8 (gate a at 16*(a/8) + a%8, up a at +8); each warp computes RPW/2
//        act rows, out[a] = bf16(silu(bf16(g*gs)) * bf16(u*gs)).
// ---------------------------------------------------------------------------
template <int K, int RPW, int WARPS, int EPI>
__global__ void __launch_bounds__(WARPS * 32)
gemv_w4a16_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                  const uint16_t* __restrict__ x, void* __restrict__ out, int N, float gs) {
  constexpr int NCH = K / 1024;  // 16-byte chunks per lane per row
  static_assert(K % 1024 == 0, "K must be a multiple of 1024");
  __shared__ __align__(16) uint16_t xs[K];
  for (int i = threadIdx.x; i < K / 8; i += WARPS * 32)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int rows[RPW];
  if (EPI == 2) {  // gate|up interleaved by 8: gate a -> 16*(a/8) + a%8, up a -> +8
    const int a0 = (blockIdx.x * WARPS + warp) * (RPW / 2);
#pragma unroll
    for (int r = 0; r < RPW / 2; ++r) {
      const int a = a0 + r, ph = ((a >> 3) << 4) + (a & 7);
      rows[r] = a < N / 2 ? ph : N; rows[RPW / 2 + r] = a < N / 2 ? ph + 8 : N;
    }
  } else {
    const int r0 = (blockIdx.x * WARPS + warp) * RPW;
#pragma unroll
    for (int r = 0; r < RPW; ++r) rows[r] = r0 + r;
  }
  const bool active = rows[0] < N;
  if (!active) return;
  uint4 wv[NCH][RPW];
  uint32_t sv[NCH][RPW];
#pragma unroll
  for (int c = 0; c < NCH; ++c)
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int row = rows[r] < N ? rows[r] : rows[0];
      wv[c][r] = ldg_nc_v4(codes + static_cast<size_t>(row) * (K / 2) + c * 512 + lane * 16);
      sv[c][r] = ldg_nc_u16(scales + static_cast<size_t>(row) * (K / 16) + c * 64 + lane * 2);
    }
  float acc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = 0.f;
#pragma unroll
  for (int c = 0; c < NCH; ++c) {
    const uint4* xp = reinterpret_cast<const uint4*>(xs + c * 1024 + lane * 32);
    const uint4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    const uint32_t xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
    const uint32_t xc[4] = {x2.x, x2.y, x2.z, x2.w}, xd[4] = {x3.x, x3.y, x3.z, x3.w};
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const uint32_t s0 = e4m3_to_f16x2(sv[c][r] & 0xff), s1 = e4m3_to_f16x2(sv[c][r] >> 8);
      acc[r] = dot8_e2m1_bf16(wv[c][r].x, s0, xa, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].y, s0, xb, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].z, s1, xc, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].w, s1, xd, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = warp_sum(acc[r]);
  if (lane == 0) {
    if (EPI == 0) {
#pragma unroll
      for (int r = 0; r < RPW; ++r)
        if (rows[r] < N) static_cast<uint16_t*>(out)[rows[r]] = f32_to_bf16_bits(acc[r] * gs);
    } else if (EPI == 1) {
#pragma unroll
      for (int r = 0; r < RPW; ++r)
        if (rows[r] < N) static_cast<float*>(out)[rows[r]] = acc[r] * gs;
    } else {
      const int a0 = (blockIdx.x * WARPS + warp) * (RPW / 2);
#pragma unroll
      for (int r = 0; r < RPW / 2; ++r) {
        if (rows[r] < N) {
          const float g = round_bf16(acc[r] * gs), u = round_bf16(acc[RPW / 2 + r] * gs);
          static_cast<uint16_t*>(out)[a0 + r] = f32_to_bf16_bits(silu_mul_precise(g, u));
        }
      }
    }
  }
}

template <int K, int RPW, int WARPS, int EPI>
static void launch_gemv(const void* codes, const void* scales, const void* x, void* out, int N,
                        float gs, cudaStream_t s) {
  const int rows_per_block = (EPI == 2) ? WARPS * RPW / 2 : WARPS * RPW;
  const int n_out = (EPI == 2) ? N / 2 : N;
  const int blocks = (n_out + rows_per_block - 1) / rows_per_block;
  gemv_w4a16_kernel<K, RPW, WARPS, EPI><<<blocks, WARPS * 32, 0, s>>>(
      static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales),
      static_cast<const uint16_t*>(x), out, N, gs);
}

int gemv_w4a16(const void* codes, const void* scales, const void* x, void* out, int N, int K,
               float gs, int epilogue, cudaStream_t s) {
#define FLINT_GEMV_CASE(KK, RPW, EPI)                                         \
  if (K == KK && epilogue == EPI) {                                           \
    launch_gemv<KK, RPW, 4, EPI>(codes, scales, x, out, N, gs, s);            \
    return 0;                                                                 \
  }
  FLINT_GEMV_CASE(1024, 4, 0)
  FLINT_GEMV_CASE(2048, 4, 0)
  FLINT_GEMV_CASE(4096, 2, 0)
  FLINT_GEMV_CASE(6144, 2, 0)
  FLINT_GEMV_CASE(2048, 4, 1)
  FLINT_GEMV_CASE(2048, 4, 2)
#undef FLINT_GEMV_CASE
  return 1;
}

// ---------------------------------------------------------------------------
// Q/K RMSNorm + NeoX RoPE (sliding layers) + KV-cache write, one token.
//   qkv bf16 [5120] = [q 4096 | k 512 | v 512]; q_out bf16 [32*128].
//   cache layout per slot: [K 4x128 | V 4x128] bf16 (2 KB); slot = pos for
//   global layers, pos & (ring - 1) for sliding layers (ring > 0).
//   cos/sin tables: fp32 [max_pos, 64].
// grid 40 warps (32 q heads, 4 k heads, 4 v heads); block = 1 warp.
// ---------------------------------------------------------------------------
__global__ void qk_norm_rope_kv_kernel(const uint16_t* __restrict__ qkv, const float* __restrict__ qw,
                                       const float* __restrict__ kw, const float* __restrict__ cos_t,
                                       const float* __restrict__ sin_t, const int* __restrict__ pos_p,
                                       uint16_t* __restrict__ q_out, uint16_t* __restrict__ cache,
                                       int ring, int rope, float eps) {
  const int head = blockIdx.x, lane = threadIdx.x;
  const int pos = *pos_p;
  const int slot = ring > 0 ? (pos & (ring - 1)) : pos;
  const uint16_t* src;
  uint16_t* dst;
  const float* w = nullptr;
  if (head < kQHeads) {
    src = qkv + head * kHeadDim; dst = q_out + head * kHeadDim; w = qw;
  } else if (head < kQHeads + kKVHeads) {
    const int h = head - kQHeads;
    src = qkv + kQHeads * kHeadDim + h * kHeadDim; dst = cache + static_cast<size_t>(slot) * 1024 + h * kHeadDim; w = kw;
  } else {
    const int h = head - kQHeads - kKVHeads;
    src = qkv + (kQHeads + kKVHeads) * kHeadDim + h * kHeadDim;
    dst = cache + static_cast<size_t>(slot) * 1024 + 512 + h * kHeadDim;
  }
  // NeoX pairs (d, d+64): lane owns dims l, l+32 (first half) and l+64, l+96 (second half).
  float v[4];
  v[0] = bf16_to_f32(src[lane]); v[1] = bf16_to_f32(src[lane + 32]);
  v[2] = bf16_to_f32(src[lane + 64]); v[3] = bf16_to_f32(src[lane + 96]);
  if (w != nullptr) {
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    ss = warp_sum(ss);
    const float rs = rsqrtf(ss / kHeadDim + eps);
    v[0] = round_bf16(v[0] * rs * w[lane]);
    v[1] = round_bf16(v[1] * rs * w[lane + 32]);
    v[2] = round_bf16(v[2] * rs * w[lane + 64]);
    v[3] = round_bf16(v[3] * rs * w[lane + 96]);
    if (rope) {
      const float* ct = cos_t + static_cast<size_t>(pos) * 64;
      const float* st = sin_t + static_cast<size_t>(pos) * 64;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int d = lane + 32 * j;  // pair (d, d + 64)
        const float c = ct[d], sn = st[d];
        const float n1 = v[j], n2 = v[j + 2];
        v[j] = fmaf(n1, c, -(n2 * sn));
        v[j + 2] = fmaf(n2, c, n1 * sn);
      }
    }
  }
  dst[lane] = f32_to_bf16_bits(v[0]); dst[lane + 32] = f32_to_bf16_bits(v[1]);
  dst[lane + 64] = f32_to_bf16_bits(v[2]); dst[lane + 96] = f32_to_bf16_bits(v[3]);
}

void qk_norm_rope_kv_bf16(const void* qkv, const float* q_norm_w, const float* k_norm_w,
                          const float* cos_t, const float* sin_t, const int* pos, void* q_out,
                          void* cache, int ring, int rope, float eps, cudaStream_t s) {
  qk_norm_rope_kv_kernel<<<kQHeads + 2 * kKVHeads, 32, 0, s>>>(
      static_cast<const uint16_t*>(qkv), q_norm_w, k_norm_w, cos_t, sin_t, pos,
      static_cast<uint16_t*>(q_out), static_cast<uint16_t*>(cache), ring, rope, eps);
}

// ---------------------------------------------------------------------------
// Router (bf16 weights [256, 2048]) + top-6 + sigmoid renormalized weights.
//   logits = bf16(xn . Wr) (fp32 accumulate); top-6 on the bf16 logits, ties
//   to the lowest expert id; s = sigmoid(logit) (precise); tw = bf16(s / (sum + 1e-20)).
// 32 blocks x 8 warps, one warp per expert row; the last block to finish (self
// resetting counter) selects the experts.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
router_topk_kernel(const uint16_t* __restrict__ xn, const uint16_t* __restrict__ wr,
                   float* __restrict__ logits, unsigned* __restrict__ counter,
                   int* __restrict__ ids, float* __restrict__ tw) {
  __shared__ __align__(16) uint16_t xs[kHidden];
  for (int i = threadIdx.x; i < kHidden / 8; i += 256)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(xn)[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.x * 8 + warp;
  const uint16_t* row = wr + static_cast<size_t>(e) * kHidden;
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < kHidden / 256; ++c) {
    const uint4 wv = ldg_nc_v4(row + c * 256 + lane * 8);
    const uint4 xv = *reinterpret_cast<const uint4*>(xs + c * 256 + lane * 8);
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w}, xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc = fmaf(bf16lo(ww[j]), bf16lo(xw[j]), acc);
      acc = fmaf(bf16hi(ww[j]), bf16hi(xw[j]), acc);
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) logits[e] = round_bf16(acc);
  __threadfence();
  __syncthreads();
  __shared__ bool last;
  if (threadIdx.x == 0) last = (atomicAdd(counter, 1u) == gridDim.x - 1);
  __syncthreads();
  if (!last || warp != 0) return;
  __threadfence();
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = __ldcg(logits + lane * 8 + j);
  float sel_v[kTopK];
  int sel_i[kTopK];
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
    sel_v[k] = bv; sel_i[k] = bi;
    if ((bi >> 3) == lane) v[bi & 7] = -FLT_MAX;
  }
  if (lane == 0) {
    float sg[kTopK], sum = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) { sg[k] = __fdiv_rn(1.0f, 1.0f + expf(-sel_v[k])); sum += sg[k]; }
    const float den = sum + 1e-20f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) { ids[k] = sel_i[k]; tw[k] = round_bf16(__fdiv_rn(sg[k], den)); }
    *counter = 0u;
  }
}

void router_topk_bf16(const void* xn, const void* wr, float* logits_scratch, unsigned* counter,
                      int* ids, float* tw, cudaStream_t s) {
  router_topk_kernel<<<kExperts / 8, 256, 0, s>>>(static_cast<const uint16_t*>(xn),
                                                  static_cast<const uint16_t*>(wr), logits_scratch,
                                                  counter, ids, tw);
}

// ---------------------------------------------------------------------------
// MoE gate/up for the 6 routed experts + the shared expert, with SwiGLU.
//   routed (expert e = ids[s]): codes [256][1024 rows, gate|up interleaved by 8][1024 B],
//     scales [256][1024][128], gs[256]. act[s*512 + a] = bf16(silu_approx(g*gs) * (u*gs)) (fp32 g/u)
//   shared: codes [2048 rows, gate|up interleaved by 8][1024 B], scales [2048][128], gs_sh.
//     act[3072 + a] = bf16(silu_precise(bf16(g*gs), bf16(u*gs)))
// Each warp computes 2 act rows (2 gate + 2 up weight rows); block = 8 warps = 16 act rows.
// Blocks [0, 192) routed, [192, 256) shared.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
moe_gate_up_kernel(const uint16_t* __restrict__ xn, const int* __restrict__ ids,
                   const uint8_t* __restrict__ ecodes, const uint8_t* __restrict__ escales,
                   const float* __restrict__ egs, const uint8_t* __restrict__ scodes,
                   const uint8_t* __restrict__ sscales, float sgs, uint16_t* __restrict__ act) {
  __shared__ __align__(16) uint16_t xs[kHidden];
  for (int i = threadIdx.x; i < kHidden / 8; i += 256)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(xn)[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int kRoutedBlocks = kTopK * kExpertInter / 16;
  const bool routed = blockIdx.x < kRoutedBlocks;
  const uint8_t* cb;
  const uint8_t* sb;
  float gs;
  int a0, out0;
  if (routed) {
    const int slot = blockIdx.x / (kExpertInter / 16);
    const int e = ids[slot];
    a0 = (blockIdx.x % (kExpertInter / 16)) * 16 + warp * 2;
    cb = ecodes + static_cast<size_t>(e) * (2 * kExpertInter) * 1024;
    sb = escales + static_cast<size_t>(e) * (2 * kExpertInter) * 128;
    gs = egs[e]; out0 = slot * kExpertInter;
  } else {
    a0 = (blockIdx.x - kRoutedBlocks) * 16 + warp * 2;
    cb = scodes; sb = sscales; gs = sgs; out0 = kTopK * kExpertInter;
  }
  // gate|up interleaved by 8: gate a -> 16*(a/8) + a%8, up a -> +8 (a0 is even)
  const int ph = ((a0 >> 3) << 4) + (a0 & 7);
  const int rows[4] = {ph, ph + 1, ph + 8, ph + 9};
  uint4 wv[2][4];
  uint32_t sv[2][4];
#pragma unroll
  for (int c = 0; c < 2; ++c)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      wv[c][r] = ldg_nc_v4(cb + static_cast<size_t>(rows[r]) * 1024 + c * 512 + lane * 16);
      sv[c][r] = ldg_nc_u16(sb + static_cast<size_t>(rows[r]) * 128 + c * 64 + lane * 2);
    }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const uint4* xp = reinterpret_cast<const uint4*>(xs + c * 1024 + lane * 32);
    const uint4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    const uint32_t xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
    const uint32_t xc[4] = {x2.x, x2.y, x2.z, x2.w}, xd[4] = {x3.x, x3.y, x3.z, x3.w};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const uint32_t s0 = e4m3_to_f16x2(sv[c][r] & 0xff), s1 = e4m3_to_f16x2(sv[c][r] >> 8);
      acc[r] = dot8_e2m1_bf16(wv[c][r].x, s0, xa, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].y, s0, xb, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].z, s1, xc, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].w, s1, xd, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) acc[r] = warp_sum(acc[r]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float a;
      if (routed) a = silu_mul_routed(acc[r] * gs, acc[2 + r] * gs);
      else a = silu_mul_precise(round_bf16(acc[r] * gs), round_bf16(acc[2 + r] * gs));
      act[out0 + a0 + r] = f32_to_bf16_bits(a);
    }
  }
}

void moe_gate_up_bf16(const void* xn, const int* ids, const void* ecodes, const void* escales,
                      const float* egs, const void* scodes, const void* sscales, float sgs, void* act,
                      cudaStream_t s) {
  constexpr int blocks = kTopK * kExpertInter / 16 + kSharedInter / 16;
  moe_gate_up_kernel<<<blocks, 256, 0, s>>>(
      static_cast<const uint16_t*>(xn), ids, static_cast<const uint8_t*>(ecodes),
      static_cast<const uint8_t*>(escales), egs, static_cast<const uint8_t*>(scodes),
      static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(act));
}

// ---------------------------------------------------------------------------
// MoE down projections for the 6 routed experts + shared expert, weighted combine.
//   out[n] = bf16( bf16(sum_j tw_j * bf16(dot_j(n) * gs_j)) + bf16(dot_sh(n) * gs_sh) )
//   routed down: codes [256][2048][256 B], scales [256][2048][32]; shared: [2048][512 B], [2048][64].
// Each warp computes 2 output rows; block = 8 warps = 16 rows; 128 blocks.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
moe_down_combine_kernel(const uint16_t* __restrict__ act, const int* __restrict__ ids,
                        const float* __restrict__ tw, const uint8_t* __restrict__ ecodes,
                        const uint8_t* __restrict__ escales, const float* __restrict__ egs,
                        const uint8_t* __restrict__ scodes, const uint8_t* __restrict__ sscales,
                        float sgs, uint16_t* __restrict__ out) {
  constexpr int kAct = kTopK * kExpertInter + kSharedInter;  // 4096
  __shared__ __align__(16) uint16_t xs[kAct];
  for (int i = threadIdx.x; i < kAct / 8; i += 256)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(act)[i];
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * 16 + warp * 2;
  int e[kTopK];
#pragma unroll
  for (int j = 0; j < kTopK; ++j) e[j] = ids[j];
  uint2 rw[2][kTopK];
  uint32_t rs[2][kTopK];
  uint4 shw[2];
  uint32_t shs[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int n = n0 + r;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
      const size_t rowi = static_cast<size_t>(e[j]) * kHidden + n;
      rw[r][j] = ldg_nc_v2(ecodes + rowi * 256 + lane * 8);
      rs[r][j] = ldg_nc_u8(escales + rowi * 32 + lane);
    }
    shw[r] = ldg_nc_v4(scodes + static_cast<size_t>(n) * 512 + lane * 16);
    shs[r] = ldg_nc_u16(sscales + static_cast<size_t>(n) * 64 + lane * 2);
  }
  float acc[2][kTopK + 1];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j <= kTopK; ++j) acc[r][j] = 0.f;
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    const uint4* xp = reinterpret_cast<const uint4*>(xs + j * kExpertInter + lane * 16);
    const uint4 x0 = xp[0], x1 = xp[1];
    const uint32_t xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint32_t s0 = e4m3_to_f16x2(rs[r][j]);
      acc[r][j] = dot8_e2m1_bf16(rw[r][j].x, s0, xa, acc[r][j]);
      acc[r][j] = dot8_e2m1_bf16(rw[r][j].y, s0, xb, acc[r][j]);
    }
  }
  {
    const uint4* xp = reinterpret_cast<const uint4*>(xs + kTopK * kExpertInter + lane * 32);
    const uint4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    const uint32_t xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
    const uint32_t xc[4] = {x2.x, x2.y, x2.z, x2.w}, xd[4] = {x3.x, x3.y, x3.z, x3.w};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint32_t s0 = e4m3_to_f16x2(shs[r] & 0xff), s1 = e4m3_to_f16x2(shs[r] >> 8);
      float a = 0.f;
      a = dot8_e2m1_bf16(shw[r].x, s0, xa, a);
      a = dot8_e2m1_bf16(shw[r].y, s0, xb, a);
      a = dot8_e2m1_bf16(shw[r].z, s1, xc, a);
      a = dot8_e2m1_bf16(shw[r].w, s1, xd, a);
      acc[r][kTopK] = a;
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j <= kTopK; ++j) acc[r][j] = warp_sum(acc[r][j]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float routed_sum = 0.f;
#pragma unroll
      for (int j = 0; j < kTopK; ++j) routed_sum = fmaf(tw[j], round_bf16(acc[r][j] * egs[e[j]]), routed_sum);
      const float o = round_bf16(routed_sum) + round_bf16(acc[r][kTopK] * sgs);
      out[n0 + r] = f32_to_bf16_bits(o);
    }
  }
}

void moe_down_combine_bf16(const void* act, const int* ids, const float* tw, const void* ecodes,
                           const void* escales, const float* egs, const void* scodes,
                           const void* sscales, float sgs, void* out, cudaStream_t s) {
  moe_down_combine_kernel<<<kHidden / 16, 256, 0, s>>>(
      static_cast<const uint16_t*>(act), ids, tw, static_cast<const uint8_t*>(ecodes),
      static_cast<const uint8_t*>(escales), egs, static_cast<const uint8_t*>(scodes),
      static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(out));
}

// ---------------------------------------------------------------------------
// Greedy argmax over fp32 logits (ties -> lowest index), then advance:
//   idx = *pos + 1; if (n_prompt == nullptr || idx >= *n_prompt) tokens[idx] = argmax;
//   *pos += 1.   (Prompt tokens already in `tokens` are never overwritten.)
// Two passes in one launch: per-block candidates, the last block reduces.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(1024)
argmax_next_kernel(const float* __restrict__ logits, int n, float* __restrict__ cand_v,
                   int* __restrict__ cand_i, unsigned* __restrict__ counter, int* __restrict__ tokens,
                   int* __restrict__ pos, const int* __restrict__ n_prompt) {
  float bv = -FLT_MAX; int bi = 0x7fffffff;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float v = logits[i];
    if (v > bv) { bv = v; bi = i; }
  }
  __shared__ float sv[32]; __shared__ int si[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
  __syncthreads();
  if (warp == 0) {
    bv = lane < (blockDim.x >> 5) ? sv[lane] : -FLT_MAX; bi = lane < (blockDim.x >> 5) ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { cand_v[blockIdx.x] = bv; cand_i[blockIdx.x] = bi; }
  }
  __threadfence();
  __syncthreads();
  __shared__ bool last;
  if (threadIdx.x == 0) last = (atomicAdd(counter, 1u) == gridDim.x - 1);
  __syncthreads();
  if (!last || warp != 0) return;
  __threadfence();
  bv = -FLT_MAX; bi = 0x7fffffff;
  for (int b = lane; b < gridDim.x; b += 32) {
    const float v = __ldcg(cand_v + b); const int i = __ldcg(cand_i + b);
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0) {
    const int idx = *pos + 1;
    if (n_prompt == nullptr || idx >= *n_prompt) tokens[idx] = bi;
    *pos = idx;
    *counter = 0u;
  }
}

void argmax_next_f32(const float* logits, int n, float* cand_v, int* cand_i, unsigned* counter, int* tokens,
                     int* pos, const int* n_prompt, cudaStream_t s) {
  argmax_next_kernel<<<kArgmaxBlocks, 1024, 0, s>>>(logits, n, cand_v, cand_i, counter, tokens, pos, n_prompt);
}

}  // namespace flint
}  // namespace flash_rt

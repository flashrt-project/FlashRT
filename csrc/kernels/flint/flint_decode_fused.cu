// SPDX-License-Identifier: Apache-2.0
//
// Flint fused decode step (one token, SM120): six kernels per MoE layer, all
// launched with programmatic dependent launch.
//
//   qkv_norm_gemv   : residual add + RMSNorm (recomputed per CTA) + QKV GEMV
//   attn_fused      : Q/K RMSNorm + RoPE + KV write + split-KV attention with
//                     sinks + last-CTA combine
//   gemv (o_proj)
//   router_norm     : residual add + RMSNorm + router GEMV + top-6 (last CTA)
//   moe_gate_up     : routed + shared gate/up GEMV with SwiGLU
//   moe_down_combine: routed + shared down GEMV with the weighted combine
//
// PDL contract: before pdl_wait() a kernel only loads static data (weights,
// scales, norm weights); every global write happens after it. The residual
// stream ping-pongs between two buffers so the CTAs of one kernel never race
// on it: a norm site reads one buffer and block 0 writes the other.

#include "flint_common.cuh"
#include "flint_kernels.cuh"

#include <cfloat>

namespace flash_rt {
namespace flint {

namespace {

constexpr int kH = 2048, kHD = 128, kQH = 32, kKVH = 4, kE = 256, kTop = 6, kEI = 512, kSI = 1024;
constexpr float kLog2eF = 1.4426950408889634f;

// Residual add + RMSNorm of the hidden vector into shared memory (every CTA).
//   s = x (+ res_in);  xs = bf16(s * rsqrt(mean(s^2) + eps) * w)
// Block 0 also stores res_out = bf16(s) and, when xn_out != nullptr, the normed vector.
template <int THREADS>
__device__ __forceinline__ void norm_prologue(const uint16_t* __restrict__ x, const uint16_t* __restrict__ res_in,
                                              int first, const float* __restrict__ w, float eps,
                                              uint16_t* __restrict__ res_out, uint16_t* __restrict__ xn_out,
                                              uint16_t* __restrict__ xs, float* __restrict__ red) {
  constexpr int PER = kH / THREADS;      // values per thread (multiple of 8)
  constexpr int NV = PER / 8;
  const int tid = threadIdx.x;
  float s[PER];
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int c = tid + v * THREADS;     // 16-byte chunk index
    const uint4 xv = reinterpret_cast<const uint4*>(x)[c];
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) { s[8 * v + 2 * j] = bf16lo(xw[j]); s[8 * v + 2 * j + 1] = bf16hi(xw[j]); }
    if (!first) {
      const uint4 rv = reinterpret_cast<const uint4*>(res_in)[c];
      const uint32_t rw[4] = {rv.x, rv.y, rv.z, rv.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) { s[8 * v + 2 * j] += bf16lo(rw[j]); s[8 * v + 2 * j + 1] += bf16hi(rw[j]); }
    }
  }
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) ss = fmaf(s[j], s[j], ss);
  ss = warp_sum(ss);
  if ((tid & 31) == 0) red[tid >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int k = 0; k < THREADS / 32; ++k) tot += red[k];
  const float rs = rsqrtf(tot / kH + eps);
#pragma unroll
  for (int v = 0; v < NV; ++v) {
    const int c = tid + v * THREADS;
    const float4 w0 = reinterpret_cast<const float4*>(w + 8 * c)[0];
    const float4 w1 = reinterpret_cast<const float4*>(w + 8 * c)[1];
    const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    uint4 o, r;
    uint32_t* op = reinterpret_cast<uint32_t*>(&o);
    uint32_t* rp = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      op[j] = pack_bf16x2(s[8 * v + 2 * j] * rs * wv[2 * j], s[8 * v + 2 * j + 1] * rs * wv[2 * j + 1]);
      rp[j] = pack_bf16x2(s[8 * v + 2 * j], s[8 * v + 2 * j + 1]);
    }
    reinterpret_cast<uint4*>(xs)[c] = o;
    if (blockIdx.x == 0 && blockIdx.y == 0) {
      reinterpret_cast<uint4*>(res_out)[c] = r;
      if (xn_out != nullptr) reinterpret_cast<uint4*>(xn_out)[c] = o;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// GEMV (M = 1) with optional fused residual add + RMSNorm prologue (K == 2048 only).
// Same lane/row layout and epilogues as gemv_w4a16 (EPI 0 bf16, 1 f32, 2 gate|up SwiGLU).
// ---------------------------------------------------------------------------
template <int K, int RPW, int WARPS, int EPI, bool NORM>
__global__ void __launch_bounds__(WARPS * 32)
gemv2_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const uint16_t* __restrict__ x,
             const uint16_t* __restrict__ res_in, uint16_t* __restrict__ res_out, const float* __restrict__ nw,
             int first, float eps, void* __restrict__ out, int N, float gs) {
  constexpr int NCH = K / 1024;
  static_assert(!NORM || K == kH, "the fused norm prologue is for the hidden size");
  __shared__ __align__(16) uint16_t xs[K];
  __shared__ float red[WARPS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int rows[RPW];
  if (EPI == 2) {
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
  uint4 wv[NCH][RPW];
  uint32_t sv[NCH][RPW];
  if (active) {
#pragma unroll
    for (int c = 0; c < NCH; ++c)
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const int row = rows[r] < N ? rows[r] : rows[0];
        wv[c][r] = ldg_nc_v4(codes + static_cast<size_t>(row) * (K / 2) + c * 512 + lane * 16);
        sv[c][r] = ldg_nc_u16(scales + static_cast<size_t>(row) * (K / 16) + c * 64 + lane * 2);
      }
  }
  pdl_wait();
  if (NORM) {
    norm_prologue<WARPS * 32>(x, res_in, first, nw, eps, res_out, nullptr, xs, red);
  } else {
    for (int i = threadIdx.x; i < K / 8; i += WARPS * 32)
      reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  }
  __syncthreads();
  pdl_launch();
  if (!active) return;
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
      for (int r = 0; r < RPW / 2; ++r)
        if (rows[r] < N) {
          const float g = round_bf16(acc[r] * gs), u = round_bf16(acc[RPW / 2 + r] * gs);
          static_cast<uint16_t*>(out)[a0 + r] = f32_to_bf16_bits(silu_mul_precise(g, u));
        }
    }
  }
}

template <int K, int RPW, int WARPS, int EPI, bool NORM>
static void launch_gemv2(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                         const float* nw, int first, float eps, void* out, int N, float gs, cudaStream_t s) {
  const int rows_per_block = (EPI == 2) ? WARPS * RPW / 2 : WARPS * RPW;
  const int n_out = (EPI == 2) ? N / 2 : N;
  const int blocks = (n_out + rows_per_block - 1) / rows_per_block;
  launch_pdl(gemv2_kernel<K, RPW, WARPS, EPI, NORM>, dim3(blocks), dim3(WARPS * 32), 0, s,
             static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), static_cast<const uint16_t*>(x),
             static_cast<const uint16_t*>(res_in), static_cast<uint16_t*>(res_out), nw, first, eps, out, N, gs);
}

int gemv2_w4a16(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                const float* norm_w, int first, float eps, void* out, int N, int K, float gs, int epilogue,
                cudaStream_t s) {
  const bool norm = norm_w != nullptr;
#define FLINT_G2(KK, RPW, EPI, NORM)                                                                 \
  if (K == KK && epilogue == EPI && norm == NORM) {                                                  \
    launch_gemv2<KK, RPW, 4, EPI, NORM>(codes, scales, x, res_in, res_out, norm_w, first, eps, out, N, \
                                        gs, s);                                                       \
    return 0;                                                                                         \
  }
  FLINT_G2(2048, 4, 0, true)    // qkv with fused input norm
  FLINT_G2(2048, 4, 2, true)    // dense gate|up with fused post-attention norm
  FLINT_G2(4096, 2, 0, false)   // o_proj
  FLINT_G2(6144, 2, 0, false)   // dense down
  FLINT_G2(2048, 4, 1, false)   // lm_head (fp32 logits)
  FLINT_G2(2048, 4, 0, false)
#undef FLINT_G2
  return 1;
}

// ---------------------------------------------------------------------------
// Standalone residual add + RMSNorm with PDL (final norm before the LM head).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256) add_rmsnorm2_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ res_in,
                                                           const float* __restrict__ w, float eps,
                                                           uint16_t* __restrict__ res_out, uint16_t* __restrict__ out) {
  __shared__ __align__(16) uint16_t xs[kH];
  __shared__ float red[8];
  pdl_wait();
  norm_prologue<256>(x, res_in, 0, w, eps, res_out, out, xs, red);
  pdl_launch();
}

void add_rmsnorm2_bf16(const void* x, const void* res_in, const float* w, float eps, void* res_out, void* out,
                       cudaStream_t s) {
  launch_pdl(add_rmsnorm2_kernel, dim3(1), dim3(256), 0, s, static_cast<const uint16_t*>(x),
             static_cast<const uint16_t*>(res_in), w, eps, static_cast<uint16_t*>(res_out), static_cast<uint16_t*>(out));
}

// ---------------------------------------------------------------------------
// Residual add + RMSNorm + router GEMV (bf16 [256, 2048]) + top-6.
// 32 blocks x 8 warps; one warp per expert row; router rows are loaded before pdl_wait.
// Block 0 writes res_out and xn (consumed by the MoE gate/up kernel).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
router_norm_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ res_in, const float* __restrict__ nw,
                   float eps, uint16_t* __restrict__ res_out, uint16_t* __restrict__ xn_out,
                   const uint16_t* __restrict__ wr, float* __restrict__ logits, unsigned* __restrict__ counter,
                   int* __restrict__ ids, float* __restrict__ tw) {
  __shared__ __align__(16) uint16_t xs[kH];
  __shared__ float red[8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.x * 8 + warp;
  const uint16_t* row = wr + static_cast<size_t>(e) * kH;
  uint4 wv[8];
#pragma unroll
  for (int c = 0; c < 8; ++c) wv[c] = ldg_nc_v4(row + c * 256 + lane * 8);
  pdl_wait();
  norm_prologue<256>(x, res_in, 0, nw, eps, res_out, xn_out, xs, red);
  __syncthreads();
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const uint4 xv = *reinterpret_cast<const uint4*>(xs + c * 256 + lane * 8);
    const uint32_t ww[4] = {wv[c].x, wv[c].y, wv[c].z, wv[c].w}, xw[4] = {xv.x, xv.y, xv.z, xv.w};
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
  if (!last) return;
  pdl_launch();
  if (warp != 0) return;
  __threadfence();
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = __ldcg(logits + lane * 8 + j);
  float sel_v[kTop];
  int sel_i[kTop];
#pragma unroll
  for (int k = 0; k < kTop; ++k) {
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
    float sg[kTop], sum = 0.f;
#pragma unroll
    for (int k = 0; k < kTop; ++k) { sg[k] = __fdiv_rn(1.0f, 1.0f + expf(-sel_v[k])); sum += sg[k]; }
    const float den = sum + 1e-20f;
#pragma unroll
    for (int k = 0; k < kTop; ++k) { ids[k] = sel_i[k]; tw[k] = round_bf16(__fdiv_rn(sg[k], den)); }
    *counter = 0u;
  }
}

void router_norm_bf16(const void* x, const void* res_in, const float* nw, float eps, void* res_out, void* xn_out,
                      const void* wr, float* logits, unsigned* counter, int* ids, float* tw, cudaStream_t s) {
  launch_pdl(router_norm_kernel, dim3(kE / 8), dim3(256), 0, s, static_cast<const uint16_t*>(x),
             static_cast<const uint16_t*>(res_in), nw, eps, static_cast<uint16_t*>(res_out),
             static_cast<uint16_t*>(xn_out), static_cast<const uint16_t*>(wr), logits, counter, ids, tw);
}

// ---------------------------------------------------------------------------
// MoE gate/up (routed + shared, SwiGLU) with PDL: shared-expert weights are
// loaded before the wait, routed weights after it (they depend on ids).
// Blocks [0,192) routed (16 act rows each), [192,256) shared.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
moe_gate_up2_kernel(const uint16_t* __restrict__ xn, const int* __restrict__ ids, const uint8_t* __restrict__ ecodes,
                    const uint8_t* __restrict__ escales, const float* __restrict__ egs,
                    const uint8_t* __restrict__ scodes, const uint8_t* __restrict__ sscales, float sgs,
                    uint16_t* __restrict__ act) {
  __shared__ __align__(16) uint16_t xs[kH];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int kRoutedBlocks = kTop * kEI / 16;
  const bool routed = blockIdx.x < kRoutedBlocks;
  int a0, out0, slot = 0;
  if (routed) {
    slot = blockIdx.x / (kEI / 16);
    a0 = (blockIdx.x % (kEI / 16)) * 16 + warp * 2;
    out0 = slot * kEI;
  } else {
    a0 = (blockIdx.x - kRoutedBlocks) * 16 + warp * 2;
    out0 = kTop * kEI;
  }
  const int ph = ((a0 >> 3) << 4) + (a0 & 7);
  const int rows[4] = {ph, ph + 1, ph + 8, ph + 9};
  uint4 wv[2][4];
  uint32_t sv[2][4];
  if (!routed) {
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        wv[c][r] = ldg_nc_v4(scodes + static_cast<size_t>(rows[r]) * 1024 + c * 512 + lane * 16);
        sv[c][r] = ldg_nc_u16(sscales + static_cast<size_t>(rows[r]) * 128 + c * 64 + lane * 2);
      }
  }
  pdl_wait();
  float gs = sgs;
  if (routed) {
    const int e = ids[slot];
    gs = egs[e];
    const uint8_t* cb = ecodes + static_cast<size_t>(e) * (2 * kEI) * 1024;
    const uint8_t* sb = escales + static_cast<size_t>(e) * (2 * kEI) * 128;
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        wv[c][r] = ldg_nc_v4(cb + static_cast<size_t>(rows[r]) * 1024 + c * 512 + lane * 16);
        sv[c][r] = ldg_nc_u16(sb + static_cast<size_t>(rows[r]) * 128 + c * 64 + lane * 2);
      }
  }
  for (int i = threadIdx.x; i < kH / 8; i += 256)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(xn)[i];
  __syncthreads();
  pdl_launch();
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
      const float a = routed ? silu_mul_routed(acc[r] * gs, acc[2 + r] * gs)
                             : silu_mul_precise(round_bf16(acc[r] * gs), round_bf16(acc[2 + r] * gs));
      act[out0 + a0 + r] = f32_to_bf16_bits(a);
    }
  }
}

void moe_gate_up2_bf16(const void* xn, const int* ids, const void* ecodes, const void* escales, const float* egs,
                       const void* scodes, const void* sscales, float sgs, void* act, cudaStream_t s) {
  launch_pdl(moe_gate_up2_kernel, dim3(kTop * kEI / 16 + kSI / 16), dim3(256), 0, s, static_cast<const uint16_t*>(xn),
             ids, static_cast<const uint8_t*>(ecodes), static_cast<const uint8_t*>(escales), egs,
             static_cast<const uint8_t*>(scodes), static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(act));
}

// ---------------------------------------------------------------------------
// MoE down (routed + shared) + weighted combine with PDL: shared-down weights
// before the wait, routed-down weights after it.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
moe_down_combine2_kernel(const uint16_t* __restrict__ act, const int* __restrict__ ids, const float* __restrict__ tw,
                         const uint8_t* __restrict__ ecodes, const uint8_t* __restrict__ escales,
                         const float* __restrict__ egs, const uint8_t* __restrict__ scodes,
                         const uint8_t* __restrict__ sscales, float sgs, uint16_t* __restrict__ out) {
  constexpr int kAct = kTop * kEI + kSI;
  __shared__ __align__(16) uint16_t xs[kAct];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = blockIdx.x * 16 + warp * 2;
  uint4 shw[2];
  uint32_t shs[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    shw[r] = ldg_nc_v4(scodes + static_cast<size_t>(n0 + r) * 512 + lane * 16);
    shs[r] = ldg_nc_u16(sscales + static_cast<size_t>(n0 + r) * 64 + lane * 2);
  }
  // The expert ids were written by the router, two kernels back: this kernel only
  // starts after every gate/up CTA passed its own pdl_wait, i.e. after the router
  // completed, so the ids are final. Read them through L2 (the buffer is reused per
  // layer, an L1 line could be stale) and prefetch the routed rows before our wait.
  int e[kTop];
#pragma unroll
  for (int j = 0; j < kTop; ++j) e[j] = __ldcg(ids + j);
  uint2 rw[2][kTop];
  uint32_t rs[2][kTop];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
      const size_t rowi = static_cast<size_t>(e[j]) * kH + n0 + r;
      rw[r][j] = ldg_nc_v2(ecodes + rowi * 256 + lane * 8);
      rs[r][j] = ldg_nc_u8(escales + rowi * 32 + lane);
    }
  pdl_wait();
  for (int i = threadIdx.x; i < kAct / 8; i += 256)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(act)[i];
  __syncthreads();
  pdl_launch();
  float acc[2][kTop + 1];
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j <= kTop; ++j) acc[r][j] = 0.f;
#pragma unroll
  for (int j = 0; j < kTop; ++j) {
    const uint4* xp = reinterpret_cast<const uint4*>(xs + j * kEI + lane * 16);
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
    const uint4* xp = reinterpret_cast<const uint4*>(xs + kTop * kEI + lane * 32);
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
      acc[r][kTop] = a;
    }
  }
#pragma unroll
  for (int r = 0; r < 2; ++r)
#pragma unroll
    for (int j = 0; j <= kTop; ++j) acc[r][j] = warp_sum(acc[r][j]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float routed_sum = 0.f;
#pragma unroll
      for (int j = 0; j < kTop; ++j) routed_sum = fmaf(tw[j], round_bf16(acc[r][j] * egs[e[j]]), routed_sum);
      out[n0 + r] = f32_to_bf16_bits(round_bf16(routed_sum) + round_bf16(acc[r][kTop] * sgs));
    }
  }
}

void moe_down_combine2_bf16(const void* act, const int* ids, const float* tw, const void* ecodes, const void* escales,
                            const float* egs, const void* scodes, const void* sscales, float sgs, void* out,
                            cudaStream_t s) {
  launch_pdl(moe_down_combine2_kernel, dim3(kH / 16), dim3(256), 0, s, static_cast<const uint16_t*>(act), ids, tw,
             static_cast<const uint8_t*>(ecodes), static_cast<const uint8_t*>(escales), egs,
             static_cast<const uint8_t*>(scodes), static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(out));
}

// ---------------------------------------------------------------------------
// Embedding at the device position (PDL).
// ---------------------------------------------------------------------------
__global__ void embed2_kernel(const int* __restrict__ tokens, const int* __restrict__ pos,
                              const uint16_t* __restrict__ table, uint16_t* __restrict__ out) {
  pdl_wait();
  const int t = tokens[*pos];
  reinterpret_cast<uint4*>(out)[threadIdx.x] =
      reinterpret_cast<const uint4*>(table + static_cast<size_t>(t) * kH)[threadIdx.x];
  pdl_launch();
}

void embed2_bf16(const int* tokens, const int* pos, const void* table, void* out, cudaStream_t s) {
  launch_pdl(embed2_kernel, dim3(1), dim3(kH / 8), 0, s, tokens, pos, static_cast<const uint16_t*>(table),
             static_cast<uint16_t*>(out));
}

}  // namespace flint
}  // namespace flash_rt

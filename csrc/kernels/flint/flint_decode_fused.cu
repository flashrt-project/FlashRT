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
// and after the last layer one lm_head_greedy kernel: final residual add + RMSNorm + LM head (fp32 logits)
// + greedy next token (per-CTA candidates, the last CTA picks and advances the position).
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

constexpr int kH = 2048, kE = 256, kTop = 6, kEI = 512, kSI = 1024;

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
  static_assert(PER % 8 == 0 && NV > 0, "norm_prologue: kH must be a multiple of 8 * THREADS (at most 256 threads)");
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

// Greedy selection state for the LM head (EPI 3): per-CTA candidates, the last CTA's counter ticket,
// and the device-side sequence (same contract as argmax_next_f32).
struct GreedyArgs {
  float* cand_v;
  int* cand_i;
  unsigned* counter;
  int* tokens;
  int* pos;
  const int* n_prompt;
};

__device__ __forceinline__ void better_of(float& bv, int& bi, float v, int i) {
  if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

}  // namespace

// ---------------------------------------------------------------------------
// GEMV (M = 1) with optional fused residual add + RMSNorm prologue (K == 2048 only).
// Same lane/row layout and epilogues as gemv_w4a16 (EPI 0 bf16, 1 f32, 2 gate|up SwiGLU), plus
// EPI 3: fp32 logits and the greedy argmax over all N outputs (ties -> lowest index); the last CTA
// writes tokens[pos + 1] (unless still inside the prompt) and advances pos, like argmax_next_f32.
// ---------------------------------------------------------------------------
template <int K, int RPW, int WARPS, int EPI, bool NORM>
__global__ void __launch_bounds__(WARPS * 32)
gemv_fused_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, const uint16_t* __restrict__ x,
             const uint16_t* __restrict__ res_in, uint16_t* __restrict__ res_out, const float* __restrict__ nw,
             int first, float eps, void* __restrict__ out, int N, float gs, GreedyArgs ga) {
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
  } else if (EPI == 3) {   // idle warps still take part in the block-wide selection
#pragma unroll
    for (int c = 0; c < NCH; ++c)
#pragma unroll
      for (int r = 0; r < RPW; ++r) { wv[c][r] = make_uint4(0u, 0u, 0u, 0u); sv[c][r] = 0u; }
  }
  pdl_wait();
  x = after_wait(x);
  res_in = after_wait(res_in);
  res_out = after_wait(res_out);
  out = after_wait(out);
  if (NORM) {
    norm_prologue<WARPS * 32>(x, res_in, first, nw, eps, res_out, nullptr, xs, red);
  } else {
    for (int i = threadIdx.x; i < K / 8; i += WARPS * 32)
      reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(x)[i];
  }
  __syncthreads();
  pdl_launch();
  if (EPI != 3 && !active) return;
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
  if (lane == 0 && active) {
    if (EPI == 0) {
#pragma unroll
      for (int r = 0; r < RPW; ++r)
        if (rows[r] < N) static_cast<uint16_t*>(out)[rows[r]] = f32_to_bf16_bits(acc[r] * gs);
    } else if (EPI == 1 || EPI == 3) {
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
  if (EPI == 3) {
    __shared__ float cv[WARPS];
    __shared__ int ci[WARPS];
    __shared__ bool last;
    if (lane == 0) {
      float bv = -FLT_MAX;
      int bi = 0x7fffffff;
#pragma unroll
      for (int r = 0; r < RPW; ++r)
        if (active && rows[r] < N) better_of(bv, bi, acc[r] * gs, rows[r]);
      cv[warp] = bv; ci[warp] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float bv = cv[0];
      int bi = ci[0];
      for (int w = 1; w < WARPS; ++w) better_of(bv, bi, cv[w], ci[w]);
      ga.cand_v[blockIdx.x] = bv; ga.cand_i[blockIdx.x] = bi;
      __threadfence();
      last = atomicAdd(ga.counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    float bv = -FLT_MAX;
    int bi = 0x7fffffff;
    for (int b = threadIdx.x; b < static_cast<int>(gridDim.x); b += WARPS * 32)
      better_of(bv, bi, __ldcg(ga.cand_v + b), __ldcg(ga.cand_i + b));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      better_of(bv, bi, __shfl_xor_sync(0xffffffffu, bv, o), __shfl_xor_sync(0xffffffffu, bi, o));
    if (lane == 0) { cv[warp] = bv; ci[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int w = 0; w < WARPS; ++w) better_of(bv, bi, cv[w], ci[w]);
      const int idx = *ga.pos + 1;
      if (ga.n_prompt == nullptr || idx >= *ga.n_prompt) ga.tokens[idx] = bi;
      *ga.pos = idx;
      *ga.counter = 0u;
    }
  }
}

template <int RPW, int WARPS, int EPI>
static int gemv_fused_blocks(int N) {
  const int rows_per_block = (EPI == 2) ? WARPS * RPW / 2 : WARPS * RPW;
  const int n_out = (EPI == 2) ? N / 2 : N;
  return (n_out + rows_per_block - 1) / rows_per_block;
}

template <int K, int RPW, int WARPS, int EPI, bool NORM>
static void launch_gemv_fused(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                         const float* nw, int first, float eps, void* out, int N, float gs, cudaStream_t s,
                         GreedyArgs ga = GreedyArgs{}) {
  const int blocks = gemv_fused_blocks<RPW, WARPS, EPI>(N);
  launch_pdl(gemv_fused_kernel<K, RPW, WARPS, EPI, NORM>, dim3(blocks), dim3(WARPS * 32), 0, s,
             static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), static_cast<const uint16_t*>(x),
             static_cast<const uint16_t*>(res_in), static_cast<uint16_t*>(res_out), nw, first, eps, out, N, gs, ga);
}

static GemvLaunchConfig g_gemv_cfg = {4, 4, 2, 4};

void set_gemv_launch_config(const GemvLaunchConfig& c) { g_gemv_cfg = c; }

int gemv_fused_w4a16(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                const float* norm_w, int first, float eps, void* out, int N, int K, float gs, int epilogue,
                cudaStream_t s) {
  const bool norm = norm_w != nullptr;
  // Tuned shapes for the two per-layer GEMVs (QKV with fused norm, o_proj).
  if (K == 2048 && epilogue == 0 && norm) {
    const int R = g_gemv_cfg.qkv_rows, W = g_gemv_cfg.qkv_warps;
#define FLINT_GQ(RR, WW)                                                                                     \
    if (R == RR && W == WW) {                                                                                \
      launch_gemv_fused<2048, RR, WW, 0, true>(codes, scales, x, res_in, res_out, norm_w, first, eps, out, N, gs, s); \
      return 0;                                                                                              \
    }
    FLINT_GQ(4, 4) FLINT_GQ(2, 4) FLINT_GQ(8, 4) FLINT_GQ(4, 8) FLINT_GQ(2, 8) FLINT_GQ(1, 8) FLINT_GQ(8, 2) FLINT_GQ(4, 2)
#undef FLINT_GQ
  }
  if (K == 4096 && epilogue == 0 && !norm) {
    const int R = g_gemv_cfg.o_rows, W = g_gemv_cfg.o_warps;
#define FLINT_GO(RR, WW)                                                                                     \
    if (R == RR && W == WW) {                                                                                \
      launch_gemv_fused<4096, RR, WW, 0, false>(codes, scales, x, res_in, res_out, norm_w, first, eps, out, N, gs, s); \
      return 0;                                                                                              \
    }
    FLINT_GO(2, 4) FLINT_GO(1, 4) FLINT_GO(4, 4) FLINT_GO(2, 8) FLINT_GO(1, 8) FLINT_GO(4, 2) FLINT_GO(1, 2)
#undef FLINT_GO
  }
#define FLINT_G2(KK, RPW, EPI, NORM)                                                                 \
  if (K == KK && epilogue == EPI && norm == NORM) {                                                  \
    launch_gemv_fused<KK, RPW, 4, EPI, NORM>(codes, scales, x, res_in, res_out, norm_w, first, eps, out, N, \
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

// LM head with the final norm and the greedy pick: 8 warps per block (the norm prologue then sums in the same
// order as add_rmsnorm_pdl_bf16), rows per warp from a measured search.
static int g_lm_rows = 4;

void set_lm_head_rows(int rows) { g_lm_rows = rows; }

int lm_head_greedy_w4a16(const void* codes, const void* scales, const void* x, const void* res_in, void* res_out,
                         const float* norm_w, float eps, float* logits, int N, float gs, float* cand_v, int* cand_i,
                         int cand_cap, unsigned* counter, int* tokens, int* pos, const int* n_prompt, cudaStream_t s) {
  const GreedyArgs ga{cand_v, cand_i, counter, tokens, pos, n_prompt};
#define FLINT_LM(RR)                                                                                          \
  if (g_lm_rows == RR) {                                                                                      \
    if (gemv_fused_blocks<RR, 8, 3>(N) > cand_cap) return 1;                                                  \
    launch_gemv_fused<2048, RR, 8, 3, true>(codes, scales, x, res_in, res_out, norm_w, 0, eps, logits, N, gs, s, ga); \
    return 0;                                                                                                 \
  }
  FLINT_LM(2) FLINT_LM(4) FLINT_LM(8)
#undef FLINT_LM
  return 1;
}

// ---------------------------------------------------------------------------
// Standalone residual add + RMSNorm with PDL (final norm before the LM head).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256) add_rmsnorm_pdl_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ res_in,
                                                           const float* __restrict__ w, float eps,
                                                           uint16_t* __restrict__ res_out, uint16_t* __restrict__ out) {
  __shared__ __align__(16) uint16_t xs[kH];
  __shared__ float red[8];
  pdl_wait();
  x = after_wait(x);
  res_in = after_wait(res_in);
  res_out = after_wait(res_out);
  out = after_wait(out);
  norm_prologue<256>(x, res_in, 0, w, eps, res_out, out, xs, red);
  pdl_launch();
}

void add_rmsnorm_pdl_bf16(const void* x, const void* res_in, const float* w, float eps, void* res_out, void* out,
                       cudaStream_t s) {
  launch_pdl(add_rmsnorm_pdl_kernel, dim3(1), dim3(256), 0, s, static_cast<const uint16_t*>(x),
             static_cast<const uint16_t*>(res_in), w, eps, static_cast<uint16_t*>(res_out), static_cast<uint16_t*>(out));
}

// ---------------------------------------------------------------------------
// Launch shapes of the MoE decode kernels (selected at runtime from a measured search).
// ---------------------------------------------------------------------------
static MoeLaunchConfig g_moe_cfg = {8, 1, 8, 1, 8, 4};   // RTX 5090 decode shape search

void set_moe_launch_config(const MoeLaunchConfig& c) { g_moe_cfg = c; }
MoeLaunchConfig get_moe_launch_config() { return g_moe_cfg; }

// ---------------------------------------------------------------------------
// Residual add + RMSNorm + router GEMV (bf16 [256, 2048]) + top-6.
// WARPS warps per block, RPW expert rows per warp; router rows are loaded before pdl_wait.
// Block 0 writes res_out and xn (consumed by the MoE gate/up kernel); the last block selects.
// ---------------------------------------------------------------------------
template <int WARPS, int RPW>
__global__ void __launch_bounds__(WARPS * 32)
router_norm_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ res_in, const float* __restrict__ nw,
                   float eps, uint16_t* __restrict__ res_out, uint16_t* __restrict__ xn_out,
                   const uint16_t* __restrict__ wr, float* __restrict__ logits, unsigned* __restrict__ counter,
                   int* __restrict__ ids, float* __restrict__ tw) {
  __shared__ __align__(16) uint16_t xs[kH];
  __shared__ float red[WARPS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e0 = (blockIdx.x * WARPS + warp) * RPW;
  uint4 wv[RPW][8];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int c = 0; c < 8; ++c) wv[r][c] = ldg_nc_v4(wr + static_cast<size_t>(e0 + r) * kH + c * 256 + lane * 8);
  uint32_t dep = 0;
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int c = 0; c < 8; ++c) dep ^= wv[r][c].x;
  pdl_wait_after(dep);
  x = after_wait(x);
  res_in = after_wait(res_in);
  res_out = after_wait(res_out);
  xn_out = after_wait(xn_out);
  logits = after_wait(logits);
  counter = after_wait(counter);
  ids = after_wait(ids);
  tw = after_wait(tw);
  norm_prologue<WARPS * 32>(x, res_in, 0, nw, eps, res_out, xn_out, xs, red);
  __syncthreads();
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 xv = *reinterpret_cast<const uint4*>(xs + c * 256 + lane * 8);
      const uint32_t ww[4] = {wv[r][c].x, wv[r][c].y, wv[r][c].z, wv[r][c].w}, xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc = fmaf(bf16lo(ww[j]), bf16lo(xw[j]), acc);
        acc = fmaf(bf16hi(ww[j]), bf16hi(xw[j]), acc);
      }
    }
    acc = warp_sum(acc);
    if (lane == 0) logits[e0 + r] = round_bf16(acc);
  }
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
  const int W = g_moe_cfg.router_warps, R = g_moe_cfg.router_rows;
#define FLINT_RT(WW, RR)                                                                                        \
  if (W == WW && R == RR) {                                                                                     \
    launch_pdl(router_norm_kernel<WW, RR>, dim3(kE / (WW * RR)), dim3(WW * 32), 0, s,                            \
               static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(res_in), nw, eps,                   \
               static_cast<uint16_t*>(res_out), static_cast<uint16_t*>(xn_out), static_cast<const uint16_t*>(wr), \
               logits, counter, ids, tw);                                                                         \
    return;                                                                                                     \
  }
  FLINT_RT(8, 1) FLINT_RT(4, 1) FLINT_RT(2, 1) FLINT_RT(8, 2) FLINT_RT(4, 2) FLINT_RT(2, 2) FLINT_RT(4, 4)
#undef FLINT_RT
  launch_pdl(router_norm_kernel<8, 1>, dim3(kE / 8), dim3(256), 0, s, static_cast<const uint16_t*>(x),
             static_cast<const uint16_t*>(res_in), nw, eps, static_cast<uint16_t*>(res_out),
             static_cast<uint16_t*>(xn_out), static_cast<const uint16_t*>(wr), logits, counter, ids, tw);
}

// ---------------------------------------------------------------------------
// MoE gate/up (routed + shared, SwiGLU) with PDL: shared-expert weights are loaded
// before the wait, routed weights after it (they depend on ids). WARPS warps per
// block, ARPW act rows per warp (2*ARPW weight rows: gate|up interleaved by 8).
// ---------------------------------------------------------------------------
template <int WARPS, int ARPW>
__global__ void __launch_bounds__(WARPS * 32)
moe_gate_up_pdl_kernel(const uint16_t* __restrict__ xn, const int* __restrict__ ids, const uint8_t* __restrict__ ecodes,
                    const uint8_t* __restrict__ escales, const float* __restrict__ egs,
                    const uint8_t* __restrict__ scodes, const uint8_t* __restrict__ sscales, float sgs,
                    uint16_t* __restrict__ act) {
  constexpr int ROWS_B = WARPS * ARPW;               // act rows per block
  constexpr int kRoutedBlocks = kTop * kEI / ROWS_B;
  __shared__ __align__(16) uint16_t xs[kH];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const bool routed = blockIdx.x < kRoutedBlocks;
  int a0, out0, slot = 0;
  if (routed) {
    slot = blockIdx.x / (kEI / ROWS_B);
    a0 = (blockIdx.x % (kEI / ROWS_B)) * ROWS_B + warp * ARPW;
    out0 = slot * kEI;
  } else {
    a0 = (blockIdx.x - kRoutedBlocks) * ROWS_B + warp * ARPW;
    out0 = kTop * kEI;
  }
  const int ph = ((a0 >> 3) << 4) + (a0 & 7);       // a0..a0+ARPW-1 stay in one group of 8
  uint4 wv[2][2 * ARPW];
  uint32_t sv[2][2 * ARPW];
  auto load_w = [&](const uint8_t* cb, const uint8_t* sb) {
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int r = 0; r < 2 * ARPW; ++r) {
        const int row = ph + (r % ARPW) + (r / ARPW) * 8;
        wv[c][r] = ldg_nc_v4(cb + static_cast<size_t>(row) * 1024 + c * 512 + lane * 16);
        sv[c][r] = ldg_nc_u16(sb + static_cast<size_t>(row) * 128 + c * 64 + lane * 2);
      }
  };
  uint32_t dep = 0;
  if (!routed) {
    load_w(scodes, sscales);
#pragma unroll
    for (int c = 0; c < 2; ++c)
#pragma unroll
      for (int r = 0; r < 2 * ARPW; ++r) dep ^= wv[c][r].x ^ sv[c][r];
  }
  pdl_wait_after(dep);
  xn = after_wait(xn);
  ids = after_wait(ids);
  act = after_wait(act);
  float gs = sgs;
  if (routed) {
    const int e = ids[slot];
    gs = egs[e];
    load_w(ecodes + static_cast<size_t>(e) * (2 * kEI) * 1024, escales + static_cast<size_t>(e) * (2 * kEI) * 128);
  }
  for (int i = threadIdx.x; i < kH / 8; i += WARPS * 32)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(xn)[i];
  __syncthreads();
  pdl_launch();
  float acc[2 * ARPW];
#pragma unroll
  for (int r = 0; r < 2 * ARPW; ++r) acc[r] = 0.f;
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const uint4* xp = reinterpret_cast<const uint4*>(xs + c * 1024 + lane * 32);
    const uint4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
    const uint32_t xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
    const uint32_t xc[4] = {x2.x, x2.y, x2.z, x2.w}, xd[4] = {x3.x, x3.y, x3.z, x3.w};
#pragma unroll
    for (int r = 0; r < 2 * ARPW; ++r) {
      const uint32_t s0 = e4m3_to_f16x2(sv[c][r] & 0xff), s1 = e4m3_to_f16x2(sv[c][r] >> 8);
      acc[r] = dot8_e2m1_bf16(wv[c][r].x, s0, xa, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].y, s0, xb, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].z, s1, xc, acc[r]);
      acc[r] = dot8_e2m1_bf16(wv[c][r].w, s1, xd, acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < 2 * ARPW; ++r) acc[r] = warp_sum(acc[r]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ARPW; ++r) {
      const float a = routed ? silu_mul_routed(acc[r] * gs, acc[ARPW + r] * gs)
                             : silu_mul_precise(round_bf16(acc[r] * gs), round_bf16(acc[ARPW + r] * gs));
      act[out0 + a0 + r] = f32_to_bf16_bits(a);
    }
  }
}

void moe_gate_up_pdl_bf16(const void* xn, const int* ids, const void* ecodes, const void* escales, const float* egs,
                       const void* scodes, const void* sscales, float sgs, void* act, cudaStream_t s) {
  const int W = g_moe_cfg.gu_warps, A = g_moe_cfg.gu_rows;
#define FLINT_GU(WW, AA)                                                                                         \
  if (W == WW && A == AA) {                                                                                      \
    launch_pdl(moe_gate_up_pdl_kernel<WW, AA>, dim3((kTop * kEI + kSI) / (WW * AA)), dim3(WW * 32), 0, s,  \
               static_cast<const uint16_t*>(xn), ids, static_cast<const uint8_t*>(ecodes),                        \
               static_cast<const uint8_t*>(escales), egs, static_cast<const uint8_t*>(scodes),                     \
               static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(act));                           \
    return;                                                                                                      \
  }
  FLINT_GU(8, 2) FLINT_GU(8, 1) FLINT_GU(4, 2) FLINT_GU(4, 1) FLINT_GU(4, 4) FLINT_GU(8, 4) FLINT_GU(2, 2) FLINT_GU(2, 4)
  FLINT_GU(16, 1) FLINT_GU(16, 2)
#undef FLINT_GU
  launch_pdl(moe_gate_up_pdl_kernel<8, 2>, dim3((kTop * kEI + kSI) / 16), dim3(256), 0, s, static_cast<const uint16_t*>(xn),
             ids, static_cast<const uint8_t*>(ecodes), static_cast<const uint8_t*>(escales), egs,
             static_cast<const uint8_t*>(scodes), static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(act));
}

// ---------------------------------------------------------------------------
// MoE down (routed + shared) + weighted combine with PDL. All weights (and the ids,
// which are final once every gate/up CTA passed its wait) are loaded before pdl_wait.
// WARPS warps per block, RPW output rows per warp.
// ---------------------------------------------------------------------------
template <int WARPS, int RPW>
__global__ void __launch_bounds__(WARPS * 32)
moe_down_combine_pdl_kernel(const uint16_t* __restrict__ act, const int* __restrict__ ids, const float* __restrict__ tw,
                         const uint8_t* __restrict__ ecodes, const uint8_t* __restrict__ escales,
                         const float* __restrict__ egs, const uint8_t* __restrict__ scodes,
                         const uint8_t* __restrict__ sscales, float sgs, uint16_t* __restrict__ out) {
  constexpr int kAct = kTop * kEI + kSI;
  __shared__ __align__(16) uint16_t xs[kAct];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * WARPS + warp) * RPW;
  uint4 shw[RPW];
  uint32_t shs[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
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
  uint2 rw[RPW][kTop];
  uint32_t rs[RPW][kTop];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int j = 0; j < kTop; ++j) {
      const size_t rowi = static_cast<size_t>(e[j]) * kH + n0 + r;
      rw[r][j] = ldg_nc_v2(ecodes + rowi * 256 + lane * 8);
      rs[r][j] = ldg_nc_u8(escales + rowi * 32 + lane);
    }
  uint32_t dep = 0;
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    dep ^= shw[r].x ^ shs[r];
#pragma unroll
    for (int j = 0; j < kTop; ++j) dep ^= rw[r][j].x ^ rs[r][j];
  }
  pdl_wait_after(dep);
  act = after_wait(act);
  tw = after_wait(tw);
  out = after_wait(out);
  for (int i = threadIdx.x; i < kAct / 8; i += WARPS * 32)
    reinterpret_cast<uint4*>(xs)[i] = reinterpret_cast<const uint4*>(act)[i];
  __syncthreads();
  pdl_launch();
  float acc[RPW][kTop + 1];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int j = 0; j <= kTop; ++j) acc[r][j] = 0.f;
#pragma unroll
  for (int j = 0; j < kTop; ++j) {
    const uint4* xp = reinterpret_cast<const uint4*>(xs + j * kEI + lane * 16);
    const uint4 x0 = xp[0], x1 = xp[1];
    const uint32_t xa[4] = {x0.x, x0.y, x0.z, x0.w}, xb[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
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
    for (int r = 0; r < RPW; ++r) {
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
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int j = 0; j <= kTop; ++j) acc[r][j] = warp_sum(acc[r][j]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      float routed_sum = 0.f;
#pragma unroll
      for (int j = 0; j < kTop; ++j) routed_sum = fmaf(tw[j], round_bf16(acc[r][j] * egs[e[j]]), routed_sum);
      out[n0 + r] = f32_to_bf16_bits(round_bf16(routed_sum) + round_bf16(acc[r][kTop] * sgs));
    }
  }
}

void moe_down_combine_pdl_bf16(const void* act, const int* ids, const float* tw, const void* ecodes, const void* escales,
                            const float* egs, const void* scodes, const void* sscales, float sgs, void* out,
                            cudaStream_t s) {
  const int W = g_moe_cfg.dn_warps, R = g_moe_cfg.dn_rows;
#define FLINT_DN(WW, RR)                                                                                         \
  if (W == WW && R == RR) {                                                                                      \
    launch_pdl(moe_down_combine_pdl_kernel<WW, RR>, dim3(kH / (WW * RR)), dim3(WW * 32), 0, s,                       \
               static_cast<const uint16_t*>(act), ids, tw, static_cast<const uint8_t*>(ecodes),                    \
               static_cast<const uint8_t*>(escales), egs, static_cast<const uint8_t*>(scodes),                     \
               static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(out));                          \
    return;                                                                                                      \
  }
  FLINT_DN(8, 2) FLINT_DN(8, 1) FLINT_DN(4, 2) FLINT_DN(4, 1) FLINT_DN(4, 4) FLINT_DN(8, 4) FLINT_DN(2, 2) FLINT_DN(2, 4)
  FLINT_DN(16, 1) FLINT_DN(16, 2) FLINT_DN(16, 4) FLINT_DN(8, 8)
#undef FLINT_DN
  launch_pdl(moe_down_combine_pdl_kernel<8, 2>, dim3(kH / 16), dim3(256), 0, s, static_cast<const uint16_t*>(act), ids, tw,
             static_cast<const uint8_t*>(ecodes), static_cast<const uint8_t*>(escales), egs,
             static_cast<const uint8_t*>(scodes), static_cast<const uint8_t*>(sscales), sgs, static_cast<uint16_t*>(out));
}

// ---------------------------------------------------------------------------
// Embedding at the device position (PDL).
// ---------------------------------------------------------------------------
__global__ void embed_pdl_kernel(const int* __restrict__ tokens, const int* __restrict__ pos,
                              const uint16_t* __restrict__ table, uint16_t* __restrict__ out) {
  pdl_wait();
  tokens = after_wait(tokens);
  pos = after_wait(pos);
  out = after_wait(out);
  const int t = tokens[*pos];
  reinterpret_cast<uint4*>(out)[threadIdx.x] =
      reinterpret_cast<const uint4*>(table + static_cast<size_t>(t) * kH)[threadIdx.x];
  pdl_launch();
}

void embed_pdl_bf16(const int* tokens, const int* pos, const void* table, void* out, cudaStream_t s) {
  launch_pdl(embed_pdl_kernel, dim3(1), dim3(kH / 8), 0, s, tokens, pos, static_cast<const uint16_t*>(table),
             static_cast<uint16_t*>(out));
}

}  // namespace flint
}  // namespace flash_rt

// SPDX-License-Identifier: Apache-2.0
//
// Batch-one decode attention of a layer visit: the prologue (short convolutions of k / v, L2 clips, value RMS
// norm, log-n scaling, the relative-bias table, the KV write) and split-KV attention with the relative bias and
// the attention sinks (24 query heads over 8 KV heads of dim 128; sliding window 1024 or full). The last block of
// each KV head merges the splits and writes the BF16 output (gated later by the o_proj prologue).

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kHd = 128;
constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kRel = 16;
constexpr int kExtent = 1024;
constexpr int kWindow = 1024;

__device__ __forceinline__ float block_sum128(float v, float* scratch) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
  __syncthreads();
  const float t = scratch[0] + scratch[1] + scratch[2] + scratch[3];
  __syncthreads();
  return t;
}

__device__ __forceinline__ float block_max128(float v, float* scratch) {
  v = warp_max(v);
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
  __syncthreads();
  const float t = fmaxf(fmaxf(scratch[0], scratch[1]), fmaxf(scratch[2], scratch[3]));
  __syncthreads();
  return t;
}

// Static short convolution of the new row, as in inkling2_visit.cu (window and conv in BF16).
__device__ __forceinline__ float short_conv_row(uint16_t* state, const uint16_t* w, int c, int C, float x) {
  const float s0 = bf16_to_f32(state[c]), s1 = bf16_to_f32(state[C + c]), s2 = bf16_to_f32(state[2 * C + c]);
  float conv = bf16_to_f32(w[4 * c]) * s0;
  conv = fmaf(bf16_to_f32(w[4 * c + 1]), s1, conv);
  conv = fmaf(bf16_to_f32(w[4 * c + 2]), s2, conv);
  conv = fmaf(bf16_to_f32(w[4 * c + 3]), x, conv);
  state[c] = state[C + c];
  state[C + c] = state[2 * C + c];
  state[2 * C + c] = f32_to_bf16_bits(x);
  return round_bf16((round_bf16(conv) + x) / sqrtf(5.0f));
}

// Dynamic short convolution (full layers): FP32 weights static[j][c] + dyn[head][j] over the lags j = 0..3
// (j = 0 the new row), then (sum + x) / sqrt(2K + 1).
__device__ __forceinline__ float dyn_conv_row(uint16_t* state, const uint16_t* static_w, const uint16_t* dyn, int c,
                                              int C, float x) {
  const int h = c / kHd;
  const float xs[4] = {x, bf16_to_f32(state[2 * C + c]), bf16_to_f32(state[C + c]), bf16_to_f32(state[c])};
  float out = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float wj = bf16_to_f32(static_w[j * C + c]) + bf16_to_f32(dyn[h * 4 + j]);
    out = __fadd_rn(out, __fmul_rn(xs[j], wj));
  }
  state[c] = state[C + c];
  state[C + c] = state[2 * C + c];
  state[2 * C + c] = f32_to_bf16_bits(x);
  return round_bf16((out + x) / sqrtf(9.0f));
}

__device__ __forceinline__ float l2_clip_scale(float sumsq, float threshold) {
  return fminf(threshold / (sqrtf(sumsq) + 1e-6f), 1.0f);
}

// Blocks 0..7: K heads, 8..15: V heads, 16..39: query heads (with their relative-bias rows). 128 threads.
__global__ void __launch_bounds__(128) attn_prologue_kernel(const __grid_constant__ AttnPrologueParams p) {
  __shared__ float scratch[4];
  if (prefetch_block(p.prefetch, 2 * kKvHeads + kHeads)) return;
  pdl_wait();
  pdl_launch();
  const int t = threadIdx.x;
  const int b = blockIdx.x;
  const int pos = *after_wait(p.pos);
  float tau = 1.0f;
  if (!p.sliding) tau = 1.0f + p.log_alpha * logf(fmaxf((float(pos) + 1.0f) / p.log_floor, 1.0f));
  if (b < 2 * kKvHeads) {
    const bool is_k = b < kKvHeads;
    const int h = is_k ? b : b - kKvHeads;
    const int c = h * kHd + t;
    const float x = bf16_to_f32(after_wait(is_k ? p.k_lin : p.v_lin)[c]);
    uint16_t* st = after_wait(is_k ? p.k_conv_state : p.v_conv_state);
    float y;
    if (p.sliding) y = short_conv_row(st, is_k ? p.k_conv_w : p.v_conv_w, c, kKvHeads * kHd, x);
    else y = dyn_conv_row(st, is_k ? p.k_conv_w : p.v_conv_w, after_wait(is_k ? p.k_dyn : p.v_dyn), c,
                          kKvHeads * kHd, x);
    const float ss = block_sum128(y * y, scratch);
    float out;
    if (is_k) out = y * l2_clip_scale(ss, p.qk_threshold);
    else out = y * (1.0f / sqrtf(ss / float(kHd) + 1e-6f));
    const int slot = p.sliding ? pos % kWindow : pos;
    uint16_t* cache = after_wait(is_k ? p.k_cache : p.v_cache);
    cache[(size_t(slot) * kKvHeads + h) * kHd + t] = f32_to_bf16_bits(out);
    return;
  }
  const int h = b - 2 * kKvHeads;
  const float q = bf16_to_f32(after_wait(p.q_lin)[h * kHd + t]) * bf16_to_f32(p.q_clip[h * kHd + t]);
  const float qs = block_sum128(q * q, scratch);
  float qo = round_bf16(q * l2_clip_scale(qs, p.qk_threshold));
  if (!p.sliding) qo = round_bf16(qo * tau);
  after_wait(p.q_out)[h * kHd + t] = f32_to_bf16_bits(qo);
  // relative states (16 per head): clip, scale, then the bias row r @ proj [16][1024]
  __shared__ float r[kRel];
  float rv = 0.0f;
  if (t < kRel) rv = bf16_to_f32(after_wait(p.r_lin)[h * kRel + t]) * bf16_to_f32(p.rel_clip[h * kRel + t]);
  const float rs = block_sum128(t < kRel ? rv * rv : 0.0f, scratch);
  if (t < kRel) {
    float ro = round_bf16(rv * l2_clip_scale(rs, p.rel_threshold));
    if (!p.sliding) ro = round_bf16(ro * tau);
    r[t] = ro;
  }
  __syncthreads();
  uint16_t* bias = after_wait(p.bias) + size_t(h) * kExtent;
  for (int d = t; d < kExtent; d += 128) {
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < kRel; ++i) acc = fmaf(r[i], bf16_to_f32(p.rel_proj[i * kExtent + d]), acc);
    bias[d] = f32_to_bf16_bits(acc);
  }
}

// Fused decode attention: one block per (KV head h, split of kSplitKeys keys), 128 threads.
//   - the split's cached K / V rows go to shared memory with cp.async (one load round);
//   - the 3 query heads of the group: L2 clip (x scale), log-n scale (full layers), BF16; their relative states the
//     same way, and the bias of exactly this split's distances (r @ proj);
//   - the block holding the new position computes its K / V row from the projections (short convolution, K L2 clip,
//     V RMS norm), writes it to the cache and advances the KV head's conv states;
//   - logits and P V from shared memory; the last block of the head merges the splits with the sinks.
constexpr int kSplitKeys = 64;
constexpr int kAttnThreads = 128;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))),
               "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

__global__ void __launch_bounds__(kAttnThreads) attn_decode_kernel(const __grid_constant__ AttnDecodeParams p) {
  __shared__ __align__(16) uint16_t ks[kSplitKeys][kHd];
  __shared__ __align__(16) uint16_t vs[kSplitKeys][kHd];
  __shared__ float qs[kGroup][kHd];
  __shared__ float rs[kGroup][kRel];
  __shared__ float lg[kGroup][kSplitKeys];
  __shared__ float scratch[4];
  __shared__ float red[kGroup][4];
  __shared__ int last;
  prefetch_share(p.prefetch);
  pdl_wait();
  pdl_launch();
  const int t = threadIdx.x;
  const int h = blockIdx.x;
  const int split = blockIdx.y;
  const int pos = *after_wait(p.pos);
  const int first = p.sliding ? max(0, pos - (kWindow - 1)) : 0;
  const int nkeys = pos - first + 1;
  const int nsplit = (nkeys + kSplitKeys - 1) / kSplitKeys;
  uint16_t* kc = after_wait(p.k_cache);
  uint16_t* vc = after_wait(p.v_cache);
  float* part = after_wait(p.partial);
  if (split < nsplit) {
    const int k0 = first + split * kSplitKeys;
    const int kn = min(kSplitKeys, pos + 1 - k0);
    // cached rows (all but the new position) -> shared memory
    for (int i = t; i < kSplitKeys * 16; i += kAttnThreads) {
      const int key = i >> 4, part16 = i & 15;
      const int kp = k0 + key;
      if (key < kn && kp < pos) {
        const int slot = p.sliding ? kp % kWindow : kp;
        cp_async16(&ks[key][part16 * 8], kc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
        cp_async16(&vs[key][part16 * 8], vc + (size_t(slot) * kKvHeads + h) * kHd + part16 * 8);
      }
    }
    float tau = 1.0f;
    if (!p.sliding) tau = 1.0f + p.log_alpha * logf(fmaxf((float(pos) + 1.0f) / p.log_floor, 1.0f));
    // queries of the group
    const uint16_t* ql = after_wait(p.q_lin);
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const int qh = h * kGroup + g;
      const float q = bf16_to_f32(ql[qh * kHd + t]) * bf16_to_f32(p.q_clip[qh * kHd + t]);
      const float ss = block_sum128(q * q, scratch);
      float qo = round_bf16(q * l2_clip_scale(ss, p.qk_threshold));
      if (!p.sliding) qo = round_bf16(qo * tau);
      qs[g][t] = qo;
    }
    // relative states of the group (16 each): lanes g * 16 + i of the first warps
    {
      const uint16_t* rl = after_wait(p.r_lin);
      const int g = t >> 4, i = t & 15;
      float rv = 0.0f;
      if (g < kGroup) rv = bf16_to_f32(rl[(h * kGroup + g) * kRel + i]) * bf16_to_f32(p.rel_clip[(h * kGroup + g) * kRel + i]);
      float s2 = rv * rv;
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) s2 += __shfl_xor_sync(0xffffffffu, s2, o);
      if (g < kGroup) {
        float ro = round_bf16(rv * l2_clip_scale(s2, p.rel_threshold));
        if (!p.sliding) ro = round_bf16(ro * tau);
        rs[g][i] = ro;
      }
    }
    // the new row (block holding pos): K / V of head h from the projections, cache write, conv states
    if (pos < k0 + kSplitKeys) {
      const int c = h * kHd + t;
      const int key = pos - k0;
      const float xk = bf16_to_f32(after_wait(p.k_lin)[c]), xv = bf16_to_f32(after_wait(p.v_lin)[c]);
      uint16_t* kst = after_wait(p.k_conv_state);
      uint16_t* vst = after_wait(p.v_conv_state);
      float yk, yv;
      if (p.sliding) {
        yk = short_conv_row(kst, p.k_conv_w, c, kKvHeads * kHd, xk);
        yv = short_conv_row(vst, p.v_conv_w, c, kKvHeads * kHd, xv);
      } else {
        yk = dyn_conv_row(kst, p.k_conv_w, after_wait(p.k_dyn), c, kKvHeads * kHd, xk);
        yv = dyn_conv_row(vst, p.v_conv_w, after_wait(p.v_dyn), c, kKvHeads * kHd, xv);
      }
      const float sk = block_sum128(yk * yk, scratch);
      const float sv = block_sum128(yv * yv, scratch);
      const uint16_t kb = f32_to_bf16_bits(yk * l2_clip_scale(sk, p.qk_threshold));
      const uint16_t vb = f32_to_bf16_bits(yv * (1.0f / sqrtf(sv / float(kHd) + 1e-6f)));
      const int slot = p.sliding ? pos % kWindow : pos;
      kc[(size_t(slot) * kKvHeads + h) * kHd + t] = kb;
      vc[(size_t(slot) * kKvHeads + h) * kHd + t] = vb;
      ks[key][t] = kb;
      vs[key][t] = vb;
    }
    cp_async_wait_all();
    __syncthreads();
    // logits: two threads per key, 64 dims each
    {
      const int key = t >> 1, half = t & 1;
      float acc[kGroup] = {0.0f, 0.0f, 0.0f};
      const uint4* kr = reinterpret_cast<const uint4*>(&ks[key][half * 64]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 v = kr[i];
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int d = half * 64 + i * 8 + j * 2;
#pragma unroll
          for (int g = 0; g < kGroup; ++g) {
            acc[g] = fmaf(bf16lo(w[j]), qs[g][d], acc[g]);
            acc[g] = fmaf(bf16hi(w[j]), qs[g][d + 1], acc[g]);
          }
        }
      }
#pragma unroll
      for (int g = 0; g < kGroup; ++g) acc[g] += __shfl_xor_sync(0xffffffffu, acc[g], 1);
      if (half == 0) {
        const int dist = pos - (k0 + key);
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
          float l = -INFINITY;
          if (key < kn) {
            l = acc[g] * (1.0f / sqrtf(float(kHd)));
            if (dist < kExtent) {
              float b = 0.0f;
#pragma unroll
              for (int i = 0; i < kRel; ++i) b = fmaf(rs[g][i], bf16_to_f32(p.rel_proj[i * kExtent + dist]), b);
              l += round_bf16(b);
            }
          }
          lg[g][key] = l;
        }
      }
    }
    __syncthreads();
    // softmax over the split: 64 logits per head, warps 0..2 take one head each
    const int lane = t & 31, warp = t >> 5;
    if (warp < kGroup) {
      const float a = lg[warp][lane], b = lg[warp][lane + 32];
      const float m = warp_max(fmaxf(a, b));
      const float ea = a == -INFINITY ? 0.0f : expf(a - m), eb = b == -INFINITY ? 0.0f : expf(b - m);
      lg[warp][lane] = ea;
      lg[warp][lane + 32] = eb;
      const float s = warp_sum(ea + eb);
      if (lane == 0) { red[warp][0] = m; red[warp][1] = s; }
    }
    __syncthreads();
    // P V: thread = dim
    float acc[kGroup] = {0.0f, 0.0f, 0.0f};
#pragma unroll 8
    for (int key = 0; key < kn; ++key) {
      const float v = bf16_to_f32(vs[key][t]);
#pragma unroll
      for (int g = 0; g < kGroup; ++g) acc[g] = fmaf(lg[g][key], v, acc[g]);
    }
    float* out = part + (size_t(h) * p.max_splits + split) * kGroup * (kHd + 2);
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      out[g * (kHd + 2) + 2 + t] = acc[g];
      if (t == 0) { out[g * (kHd + 2)] = red[g][0]; out[g * (kHd + 2) + 1] = red[g][1]; }
    }
  }
  // the last arriving block of the head merges the splits with the sinks
  __threadfence();
  __syncthreads();
  if (t == 0) last = atomicAdd(after_wait(p.counters) + h, 1) == int(gridDim.y) - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  const uint16_t* sinks = after_wait(p.sinks);
  uint16_t* o = after_wait(p.out);
  const size_t stride = size_t(kGroup) * (kHd + 2);
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    const int qh = h * kGroup + g;
    const float sink = bf16_to_f32(sinks[qh]);
    const float* base = part + size_t(h) * p.max_splits * stride + g * (kHd + 2);
    float mx = sink;
    for (int sp = 0; sp < nsplit; ++sp) mx = fmaxf(mx, base[sp * stride]);
    float den = expf(sink - mx), num = 0.0f;
#pragma unroll 4
    for (int sp = 0; sp < nsplit; ++sp) {
      const float f = expf(base[sp * stride] - mx);
      den = fmaf(base[sp * stride + 1], f, den);
      num = fmaf(base[sp * stride + 2 + t], f, num);
    }
    o[qh * kHd + t] = f32_to_bf16_bits(num / den);
  }
  if (t == 0) after_wait(p.counters)[h] = 0;
}

}  // namespace

void attn_prologue(const AttnPrologueParams& p, cudaStream_t stream) {
  launch_pdl(attn_prologue_kernel, dim3(2 * kKvHeads + kHeads + (p.prefetch.count ? kPrefetchBlocks : 0)), dim3(128), 0,
             stream, p);
}

void attn_decode(const AttnDecodeParams& p, cudaStream_t stream) {
  if (p.max_splits < 1) throw std::invalid_argument("attn_decode: max_splits >= 1");
  launch_pdl(attn_decode_kernel, dim3(kKvHeads, p.max_splits), dim3(kAttnThreads), 0, stream, p);
}

int attn_split_keys() { return kSplitKeys; }

}  // namespace inkling2
}  // namespace flash_rt

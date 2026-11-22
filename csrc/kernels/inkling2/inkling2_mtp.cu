// SPDX-License-Identifier: Apache-2.0
//
// The checkpoint's DSpark drafter (mtp.safetensors) for one sequence: a DFlash block of 8 positions (the anchor's
// normalized embedding and 7 mask embeddings) through 3 bidirectional layers over the drafter context, base logits
// from the target's LM head, and a Markov RNN head walk of 7 greedy steps (SGLang Inkling2ForConditionalGenerationMTP).
// Drafts only propose (the target's verification decides the output), so the drafter computes in FP32 with BF16
// activations between the projections and matches no reference rounding. The projections run on gemv_multi.
//
//   context: committed positions' K / V per layer from a fuse_weight mix of the target features (the hidden state
//            after visits 4 / 22 / 40 and the final normalized state): RMSNorm, k / v context projections, k RMSNorm,
//            NeoX RoPE; the Markov feature from the last committed position.
//   block:   per layer RMSNorm, q / k / v, q / k head RMSNorm + RoPE (the block's K / V into the context cache at its
//            positions), attention over [lo, block start + 8) (lo = start - 1025 on the windowed layers), output
//            projection, RMSNorm, SiLU-gated MLP.
//   walk:    state' / latent from rnn_proj([state, w1[prev], hidden of the position, Markov feature]); the drafted
//            token = argmax over the vocabulary of base logits + fold_w latent + fold_b (fold = W_lm hidden_out).

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

constexpr int kThreads = 256;
constexpr int kHd = 128;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 8;

__device__ __forceinline__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.0f;
#pragma unroll
  for (int i = 0; i < kThreads / 32; ++i) t += red[i];
  __syncthreads();
  return t;
}

// RMSNorm of one 128-value head held 4 per lane (dims 4 lane .. 4 lane + 3), the weight, then NeoX RoPE at `pos`
// (dims d and d + 64 pair up: lanes l and l ^ 16).
__device__ __forceinline__ void head_norm_rope(float (&v)[4], const uint16_t* w, int pos, float theta) {
  const int lane = threadIdx.x & 31;
  float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  ss = warp_sum(ss);
  const float inv = rsqrtf(ss / float(kHd) + 1e-6f);
#pragma unroll
  for (int t = 0; t < 4; ++t) v[t] = v[t] * inv * bf16_to_f32(w[4 * lane + t]);
  float other[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) other[t] = __shfl_xor_sync(0xffffffffu, v[t], 16);
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    const int d = (4 * lane + t) & 63;
    const float freq = powf(theta, -2.0f * float(d) / float(kHd));
    float s, c;
    sincosf(float(pos) * freq, &s, &c);
    v[t] = lane < 16 ? v[t] * c - other[t] * s : v[t] * c + other[t] * s;
  }
}

__device__ __forceinline__ int committed(const MtpParams& p) { return (p.count ? *after_wait(p.count) : 0) + p.count_add; }

// ---------------------------------------------------------------------------------------------- context
// grid (8 rows): rows j < count: x[m][j] = RMSNorm(sum_t feat_t[j] alpha[m][t]) * norm_m (m = 0..2), the last also
// the Markov feature frnn.
__global__ void __launch_bounds__(kThreads) mtp_context_mix_kernel(const __grid_constant__ MtpParams p) {
  __shared__ float red[kThreads / 32];
  __shared__ float mixed[kHidden];
  pdl_wait();
  pdl_launch();
  const int j = blockIdx.x, n = committed(p);
  if (j >= n) return;
  for (int m = 0; m < 4; ++m) {
    if (m == 3 && j != n - 1) break;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kHidden; i += kThreads) {
      float v = 0.0f;
#pragma unroll
      for (int t = 0; t < 4; ++t) v = fmaf(bf16_to_f32(after_wait(p.feat[t])[size_t(j) * kHidden + i]), p.alpha[m * 4 + t], v);
      mixed[i] = v;
      ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block_sum(ss, red) / float(kHidden) + 1e-6f);
    uint16_t* dst = m == 3 ? after_wait(p.frnn) : after_wait(p.x) + (size_t(m) * kMtpBlock + j) * kHidden;
    for (int i = threadIdx.x; i < kHidden; i += kThreads) dst[i] = f32_to_bf16_bits(mixed[i] * inv * bf16_to_f32(p.norm[m][i]));
  }
}

// grid (8 rows, 3 layers), 8 warps = 8 KV heads: k RMSNorm + RoPE, v copied, into kv at pos - count + j.
__global__ void __launch_bounds__(kThreads) mtp_context_write_kernel(const __grid_constant__ MtpParams p) {
  pdl_wait();
  pdl_launch();
  const int j = blockIdx.x, l = blockIdx.y, n = committed(p);
  if (j >= n) return;
  const int pos = *after_wait(p.pos) - n + j;
  const int h = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint16_t* kc = after_wait(p.kvc) + (size_t(l * 2) * kMtpBlock + j) * (kKvHeads * kHd) + h * kHd;
  const uint16_t* vc = after_wait(p.kvc) + (size_t(l * 2 + 1) * kMtpBlock + j) * (kKvHeads * kHd) + h * kHd;
  float k[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) k[t] = bf16_to_f32(kc[4 * lane + t]);
  head_norm_rope(k, p.k_norm[l], pos, p.theta);
  uint16_t* kv = after_wait(p.kv) + size_t(l) * 2 * kKvHeads * p.cap * kHd;
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    kv[(size_t(h) * p.cap + pos) * kHd + 4 * lane + t] = f32_to_bf16_bits(k[t]);
    kv[(size_t(kKvHeads + h) * p.cap + pos) * kHd + 4 * lane + t] = vc[4 * lane + t];
  }
}

// ---------------------------------------------------------------------------------------------- block
// grid (8 rows): dh[0] = the anchor's normalized embedding (as the target's), dh[j >= 1] = the mask embedding.
__global__ void __launch_bounds__(kThreads) mtp_block_embed_kernel(const __grid_constant__ MtpParams p) {
  __shared__ float red[kThreads / 32];
  pdl_wait();
  pdl_launch();
  const int j = blockIdx.x;
  float* dh = after_wait(p.dh) + size_t(j) * kHidden;
  if (j > 0) {
    for (int i = threadIdx.x; i < kHidden; i += kThreads) dh[i] = bf16_to_f32(p.mask_embed[i]);
    return;
  }
  const uint16_t* e = p.embed + size_t(*after_wait(p.anchor)) * kHidden;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < kHidden; i += kThreads) ss = fmaf(bf16_to_f32(e[i]), bf16_to_f32(e[i]), ss);
  const float inv = rsqrtf(block_sum(ss, red) / float(kHidden) + 1e-6f);
  for (int i = threadIdx.x; i < kHidden; i += kThreads) dh[i] = round_bf16(bf16_to_f32(e[i]) * inv * bf16_to_f32(p.embed_norm[i]));
}

// grid (8 rows): dh += add (BF16, optional); x = bf16(RMSNorm(dh) * w).
__global__ void __launch_bounds__(kThreads) mtp_norm_kernel(const __grid_constant__ MtpParams p) {
  __shared__ float red[kThreads / 32];
  pdl_wait();
  pdl_launch();
  const int j = blockIdx.x;
  float* dh = after_wait(p.dh) + size_t(j) * kHidden;
  const uint16_t* add = p.add ? after_wait(p.add) + size_t(j) * kHidden : nullptr;
  const float* pa = p.add_part[0] ? after_wait(p.add_part[0]) + size_t(j) * kHidden : nullptr;
  const float* pb = p.add_part[1] ? after_wait(p.add_part[1]) + size_t(j) * kHidden : nullptr;
  float v[kHidden / kThreads];
  float ss = 0.0f;
#pragma unroll
  for (int u = 0; u < kHidden / kThreads; ++u) {
    const int i = threadIdx.x + u * kThreads;
    v[u] = dh[i];
    if (add) v[u] += bf16_to_f32(add[i]);
    if (pa) v[u] += pa[i] + pb[i];
    if (add || pa) dh[i] = v[u];
    ss = fmaf(v[u], v[u], ss);
  }
  const float inv = rsqrtf(block_sum(ss, red) / float(kHidden) + 1e-6f);
  uint16_t* x = after_wait(p.x) + size_t(j) * kHidden;
#pragma unroll
  for (int u = 0; u < kHidden / kThreads; ++u) {
    const int i = threadIdx.x + u * kThreads;
    x[i] = f32_to_bf16_bits(v[u] * inv * bf16_to_f32(p.norm_w[i]));
  }
}

// grid (8 rows), 8 warps over the 24 q + 8 k + 8 v heads: q / k head RMSNorm + RoPE at pos + j; the block's K / V
// into the cache (layer `layer`).
__global__ void __launch_bounds__(kThreads) mtp_qk_kernel(const __grid_constant__ MtpParams p) {
  pdl_wait();
  pdl_launch();
  const int j = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pos = *after_wait(p.pos) + j;
  uint16_t* kv = after_wait(p.kv) + size_t(p.layer) * 2 * kKvHeads * p.cap * kHd;
  for (int hd = warp; hd < kQHeads + 2 * kKvHeads; hd += kThreads / 32) {
    if (hd >= kQHeads + kKvHeads) {
      const int h = hd - kQHeads - kKvHeads;
      const uint16_t* src = after_wait(p.v_in) + size_t(j) * kKvHeads * kHd + h * kHd;
#pragma unroll
      for (int t = 0; t < 4; ++t) kv[(size_t(kKvHeads + h) * p.cap + pos) * kHd + 4 * lane + t] = src[4 * lane + t];
      continue;
    }
    const bool isq = hd < kQHeads;
    const uint16_t* src = isq ? after_wait(p.q_in) + size_t(j) * kQHeads * kHd + hd * kHd
                              : after_wait(p.k_in) + size_t(j) * kKvHeads * kHd + (hd - kQHeads) * kHd;
    float v[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) v[t] = bf16_to_f32(src[4 * lane + t]);
    head_norm_rope(v, isq ? p.q_norm : p.kb_norm, pos, p.theta);
    if (isq) {
      uint16_t* q = after_wait(p.q_out) + size_t(j) * kQHeads * kHd + hd * kHd;
#pragma unroll
      for (int t = 0; t < 4; ++t) q[4 * lane + t] = f32_to_bf16_bits(v[t]);
    } else {
#pragma unroll
      for (int t = 0; t < 4; ++t) kv[(size_t(hd - kQHeads) * p.cap + pos) * kHd + 4 * lane + t] = f32_to_bf16_bits(v[t]);
    }
  }
}

// grid (8 KV heads, 8 rows): row j's 3 query heads over keys [lo, pos + 8), 8 warps splitting the keys (online
// softmax each), merged in shared memory.
__global__ void __launch_bounds__(kThreads) mtp_attention_kernel(const __grid_constant__ MtpParams p) {
  __shared__ float tm[8][3], tl[8][3];
  __shared__ float ta[8][3][kHd];
  pdl_wait();
  pdl_launch();
  const int h = blockIdx.x, j = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int start = *after_wait(p.pos), hi = start + kMtpBlock;
  const int lo = p.window && start > p.window + 1 ? start - (p.window + 1) : 0;
  const uint16_t* layer = after_wait(p.kv) + size_t(p.layer) * 2 * kKvHeads * p.cap * kHd;
  const uint16_t* K = layer + size_t(h) * p.cap * kHd;
  const uint16_t* V = layer + size_t(kKvHeads + h) * p.cap * kHd;
  const float scale = 0.08838834764831845f;
  float q[3][4], m[3], l[3], acc[3][4];
  const uint16_t* qr = after_wait(p.q_out) + size_t(j) * kQHeads * kHd;
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    const uint2 qv = *reinterpret_cast<const uint2*>(qr + (3 * h + a) * kHd + 4 * lane);
    q[a][0] = bf16lo(qv.x) * scale; q[a][1] = bf16hi(qv.x) * scale;
    q[a][2] = bf16lo(qv.y) * scale; q[a][3] = bf16hi(qv.y) * scale;
    m[a] = -INFINITY;
    l[a] = 0.0f;
#pragma unroll
    for (int t = 0; t < 4; ++t) acc[a][t] = 0.0f;
  }
  for (int k = lo + warp; k < hi; k += 8) {
    const uint2 kk = *reinterpret_cast<const uint2*>(K + size_t(k) * kHd + 4 * lane);
    const uint2 vv = *reinterpret_cast<const uint2*>(V + size_t(k) * kHd + 4 * lane);
    const float kf[4] = {bf16lo(kk.x), bf16hi(kk.x), bf16lo(kk.y), bf16hi(kk.y)};
    const float vf[4] = {bf16lo(vv.x), bf16hi(vv.x), bf16lo(vv.y), bf16hi(vv.y)};
#pragma unroll
    for (int a = 0; a < 3; ++a) {
      float s = q[a][0] * kf[0] + q[a][1] * kf[1] + q[a][2] * kf[2] + q[a][3] * kf[3];
      s = warp_sum(s);
      const float mn = fmaxf(m[a], s);
      const float c = __expf(m[a] - mn), e = __expf(s - mn);
      l[a] = l[a] * c + e;
#pragma unroll
      for (int t = 0; t < 4; ++t) acc[a][t] = acc[a][t] * c + e * vf[t];
      m[a] = mn;
    }
  }
#pragma unroll
  for (int a = 0; a < 3; ++a) {
    if (lane == 0) { tm[warp][a] = m[a]; tl[warp][a] = l[a]; }
#pragma unroll
    for (int t = 0; t < 4; ++t) ta[warp][a][4 * lane + t] = acc[a][t];
  }
  __syncthreads();
  uint16_t* out = after_wait(p.attn_out) + size_t(j) * kQHeads * kHd;
  for (int i = threadIdx.x; i < 3 * kHd; i += kThreads) {
    const int a = i / kHd, d = i % kHd;
    float M = -INFINITY;
#pragma unroll
    for (int s = 0; s < 8; ++s) M = fmaxf(M, tm[s][a]);
    float L = 0.0f, A = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (tm[s][a] == -INFINITY) continue;
      const float c = __expf(tm[s][a] - M);
      L = fmaf(tl[s][a], c, L);
      A = fmaf(ta[s][a][d], c, A);
    }
    out[(3 * h + a) * kHd + d] = f32_to_bf16_bits(A / L);
  }
}

// act = bf16(silu(gate) * up), [8][9216]
__global__ void __launch_bounds__(kThreads) mtp_silu_kernel(const __grid_constant__ MtpParams p) {
  pdl_wait();
  pdl_launch();
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= kMtpBlock * kMtpInter) return;
  const float g = bf16_to_f32(after_wait(p.gate)[i]), u = bf16_to_f32(after_wait(p.up)[i]);
  after_wait(p.act)[i] = f32_to_bf16_bits(g / (1.0f + __expf(-g)) * u);
}

// ---------------------------------------------------------------------------------------------- Markov walk
// one block: rnn input of step `step` = [state 256 | w1[prev] 256 | hidden of block row step + 1 | frnn]
__global__ void __launch_bounds__(kThreads) mtp_rnn_in_kernel(const __grid_constant__ MtpParams p) {
  pdl_wait();
  pdl_launch();
  const int prev = p.step == 0 ? *after_wait(p.anchor) : after_wait(p.drafts)[p.step - 1];
  uint16_t* o = after_wait(p.rnn_in);
  const uint16_t* state = after_wait(p.state);
  const uint16_t* w1 = p.markov_w1 + size_t(prev) * kMtpRank;
  for (int i = threadIdx.x; i < kMtpRank; i += kThreads) { o[i] = state[i]; o[kMtpRank + i] = w1[i]; }
  const uint16_t* hn = after_wait(p.hn) + size_t(p.step + 1) * kHidden;
  const uint16_t* fr = after_wait(p.frnn);
  for (int i = threadIdx.x; i < kHidden; i += kThreads) { o[2 * kMtpRank + i] = hn[i]; o[2 * kMtpRank + kHidden + i] = fr[i]; }
}

// one block: rnn_proj output (+ bias) -> state' (BF16 rounding of each op, as the reference), latent = tanh(out)
__global__ void __launch_bounds__(kThreads) mtp_gates_kernel(const __grid_constant__ MtpParams p) {
  pdl_wait();
  pdl_launch();
  const int c = threadIdx.x;
  const float* proj = after_wait(p.proj);
  const float g = round_bf16(proj[c] + bf16_to_f32(p.rnn_bias[c]));
  const float cand = round_bf16(proj[kMtpRank + c] + bf16_to_f32(p.rnn_bias[kMtpRank + c]));
  const float o = round_bf16(proj[2 * kMtpRank + c] + bf16_to_f32(p.rnn_bias[2 * kMtpRank + c]));
  uint16_t* state = after_wait(p.state);
  const float st = bf16_to_f32(state[c]);
  const float gate = round_bf16(1.0f / (1.0f + expf(-g)));
  const float kept = round_bf16(gate * st);
  const float blend = round_bf16(round_bf16(1.0f - gate) * round_bf16(tanhf(cand)));
  state[c] = f32_to_bf16_bits(kept + blend);
  after_wait(p.latent)[c] = f32_to_bf16_bits(tanhf(o));
}

// The drafted token of step `step`: argmax over v < vocab of logits[step][v] + fold_w[v] . latent + fold_b[v]; a warp
// per vocabulary row (one uint4 of fold_w per lane), the blocks' best merged by the last block.
constexpr int kPickBlocks = 340;
__global__ void __launch_bounds__(kThreads) mtp_pick_kernel(const __grid_constant__ MtpParams p) {
  __shared__ float lat[kMtpRank];
  __shared__ float bv[kThreads / 32];
  __shared__ int bi[kThreads / 32];
  __shared__ int last;
  pdl_wait();
  pdl_launch();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x < kMtpRank) lat[threadIdx.x] = bf16_to_f32(after_wait(p.latent)[threadIdx.x]);
  __syncthreads();
  float lx[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) lx[e] = lat[8 * lane + e];
  const float* lg = after_wait(p.logits) + size_t(p.step) * p.vocab_stride;
  float best = -INFINITY;
  int idx = 1 << 30;
  for (int v = blockIdx.x * (kThreads / 32) + warp; v < p.vocab; v += gridDim.x * (kThreads / 32)) {
    const uint4 w = ldg_stream(reinterpret_cast<const uint4*>(p.fold_w + size_t(v) * kMtpRank) + lane);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    float d = 0.0f;
#pragma unroll
    for (int e = 0; e < 4; ++e) d = fmaf(bf16lo(ww[e]), lx[2 * e], fmaf(bf16hi(ww[e]), lx[2 * e + 1], d));
    d = warp_sum(d);
    const float x = lg[v] + d + bf16_to_f32(p.fold_b[v]);
    if (x > best) { best = x; idx = v; }
  }
  if (lane == 0) { bv[warp] = best; bi[warp] = idx; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kThreads / 32; ++w)
      if (bv[w] > best || (bv[w] == best && bi[w] < idx)) { best = bv[w]; idx = bi[w]; }
    after_wait(p.pick_best)[blockIdx.x] = best;
    after_wait(p.pick_index)[blockIdx.x] = idx;
    __threadfence();
    last = atomicAdd(after_wait(p.pick_counter), 1) == int(gridDim.x) - 1;
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  const float* pb = after_wait(p.pick_best);
  const int* pi = after_wait(p.pick_index);
  best = -INFINITY;
  idx = 1 << 30;
  for (int b = threadIdx.x; b < int(gridDim.x); b += kThreads)
    if (pb[b] > best || (pb[b] == best && pi[b] < idx)) { best = pb[b]; idx = pi[b]; }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ob = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
  }
  if (lane == 0) { bv[warp] = best; bi[warp] = idx; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kThreads / 32; ++w)
      if (bv[w] > best || (bv[w] == best && bi[w] < idx)) { best = bv[w]; idx = bi[w]; }
    after_wait(p.drafts)[p.step] = idx;
    *after_wait(p.pick_counter) = 0;
  }
}

}  // namespace

void mtp_context_mix(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_context_mix_kernel, dim3(kMtpBlock), dim3(kThreads), 0, s, p);
}
void mtp_context_write(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_context_write_kernel, dim3(kMtpBlock, kMtpLayers), dim3(kThreads), 0, s, p);
}
void mtp_block_embed(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_block_embed_kernel, dim3(kMtpBlock), dim3(kThreads), 0, s, p);
}
void mtp_norm(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_norm_kernel, dim3(kMtpBlock), dim3(kThreads), 0, s, p);
}
void mtp_qk(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_qk_kernel, dim3(kMtpBlock), dim3(kThreads), 0, s, p);
}
void mtp_attention(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_attention_kernel, dim3(kKvHeads, kMtpBlock), dim3(kThreads), 0, s, p);
}
void mtp_silu(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_silu_kernel, dim3((kMtpBlock * kMtpInter + kThreads - 1) / kThreads), dim3(kThreads), 0, s, p);
}
void mtp_rnn_in(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_rnn_in_kernel, dim3(1), dim3(kThreads), 0, s, p);
}
void mtp_gates(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_gates_kernel, dim3(1), dim3(kMtpRank), 0, s, p);
}
void mtp_pick(const MtpParams& p, cudaStream_t s) {
  launch_pdl(mtp_pick_kernel, dim3(kPickBlocks), dim3(kThreads), 0, s, p);
}
int mtp_pick_blocks() { return kPickBlocks; }

}  // namespace inkling2
}  // namespace flash_rt

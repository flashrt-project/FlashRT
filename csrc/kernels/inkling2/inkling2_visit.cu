// SPDX-License-Identifier: Apache-2.0
//
// Per-visit vector kernels of the batch-one decode step (one 1024-thread block each: they reduce over the whole
// 2304-wide residual): the visit boundary (expert combination, post-MLP short convolution and residual of the
// previous visit, then the attention-input depth attention and the loop messages of the next one), the
// post-attention step (short convolution, residual, block states, MLP-input depth attention) and the loop
// controllers' gate mix. Semantics of mlx-lm's inkling2 layers: FP32 inside, BF16 where it stores BF16.

#include "kernels/inkling2/inkling2_common.cuh"
#include "kernels/inkling2/inkling2_kernels.cuh"

namespace flash_rt {
namespace inkling2 {

namespace {

// One block of 576 threads; thread t owns the 4 channels 4t .. 4t + 3 of the 2304-wide residual.
constexpr int kThreads = kHidden / 4;
constexpr int kWarps = kThreads / 32;

struct V4 { float x[4]; };
// c ? a : b through selp: the compiler cannot fold a chain of these back into an indexed (local memory) access
__device__ __forceinline__ float opaque_sel(bool c, float a, float b) {
  float r;
  asm("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %1, 0;\n\tselp.f32 %0, %2, %3, p;\n\t}" : "=f"(r) : "r"(int(c)), "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ V4 load_f4(const float* p, int t) {
  const float4 v = reinterpret_cast<const float4*>(p)[t];
  return {{v.x, v.y, v.z, v.w}};
}
__device__ __forceinline__ V4 load_b4(const uint16_t* p, int t) {
  const uint2 v = reinterpret_cast<const uint2*>(p)[t];
  return {{bf16lo(v.x), bf16hi(v.x), bf16lo(v.y), bf16hi(v.y)}};
}
__device__ __forceinline__ void store_f4(float* p, int t, const V4& v) {
  reinterpret_cast<float4*>(p)[t] = make_float4(v.x[0], v.x[1], v.x[2], v.x[3]);
}
__device__ __forceinline__ void store_b4(uint16_t* p, int t, const V4& v) {
  reinterpret_cast<uint2*>(p)[t] = make_uint2(uint32_t(f32_to_bf16_bits(v.x[0])) | (uint32_t(f32_to_bf16_bits(v.x[1])) << 16),
                                              uint32_t(f32_to_bf16_bits(v.x[2])) | (uint32_t(f32_to_bf16_bits(v.x[3])) << 16));
}
__device__ __forceinline__ float sumsq(const V4& v) {
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) s = fmaf(v.x[i], v.x[i], s);
  return s;
}

// Block-wide sums of N per-thread values: warp sums, then warp 0 adds the per-warp partials; all threads get them.
template <int N>
__device__ __forceinline__ void block_sums(float (&v)[N], float (*red)[33]) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float s = warp_sum(v[i]);
    if (lane == 0) red[i][warp] = s;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      float s = lane < kWarps ? red[i][lane] : 0.0f;
      s = warp_sum(s);
      if (lane == 0) red[i][32] = s;
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) v[i] = red[i][32];
  __syncthreads();
}

// block_sums for the values i with live(i) only (the others untouched); each live value's order as in block_sums
template <int N, class Live>
__device__ __forceinline__ void block_sums_live(float (&v)[N], float (*red)[33], Live live) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    if (!live(i)) continue;
    const float s = warp_sum(v[i]);
    if (lane == 0) red[i][warp] = s;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (!live(i)) continue;
      float s = lane < kWarps ? red[i][lane] : 0.0f;
      s = warp_sum(s);
      if (lane == 0) red[i][32] = s;
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i)
    if (live(i)) v[i] = red[i][32];
  __syncthreads();
}

__device__ __forceinline__ float inv_rms(float sumsq_total, int d) { return 1.0f / sqrtf(sumsq_total / float(d) + 1e-6f); }

// Depth attention for this thread's 4 channels: sources src[0..n) (registers), queries q[0..nq) (global FP32
// [nq][kHidden]); out[q] = sum_s softmax_s(src_s . query_q / sqrt(d)) src_s.
template <int MaxN, int MaxQ>
__device__ __forceinline__ void depth_attention_q(const V4 (&src)[MaxN], int n, const V4 (&qv)[MaxQ], int nq, V4 (&out)[MaxQ],
                                                  float (*red)[33]);
template <int MaxN, int MaxQ>
__device__ __forceinline__ void depth_attention(const V4 (&src)[MaxN], int n, const float* query, int nq, V4 (&out)[MaxQ],
                                                float (*red)[33]) {
  const int t = threadIdx.x;
  V4 qv[MaxQ];
#pragma unroll
  for (int q = 0; q < MaxQ; ++q) qv[q] = q < nq ? load_f4(query + q * kHidden, t) : V4{};
  depth_attention_q<MaxN, MaxQ>(src, n, qv, nq, out, red);
}
template <int MaxN, int MaxQ>
__device__ __forceinline__ void depth_attention_q(const V4 (&src)[MaxN], int n, const V4 (&qv)[MaxQ], int nq, V4 (&out)[MaxQ],
                                                  float (*red)[33]) {
  float part[MaxN * MaxQ];
#pragma unroll
  for (int s = 0; s < MaxN; ++s)
#pragma unroll
    for (int q = 0; q < MaxQ; ++q) {
      float a = 0.0f;
      if (s < n && q < nq) {
#pragma unroll
        for (int i = 0; i < 4; ++i) a = fmaf(src[s].x[i], qv[q].x[i], a);
      }
      part[s * MaxQ + q] = a;
    }
  block_sums_live<MaxN * MaxQ>(part, red, [&](int i) { return i / MaxQ < n && i % MaxQ < nq; });
  const float scale = 1.0f / sqrtf(float(kHidden));
#pragma unroll
  for (int q = 0; q < MaxQ; ++q) {
    if (q >= nq) break;
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < MaxN; ++s)
      if (s < n) m = fmaxf(m, part[s * MaxQ + q] * scale);
    float w[MaxN], z = 0.0f;
#pragma unroll
    for (int s = 0; s < MaxN; ++s) {
      w[s] = s < n ? expf(part[s * MaxQ + q] * scale - m) : 0.0f;
      z += w[s];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float acc = 0.0f;
#pragma unroll
      for (int s = 0; s < MaxN; ++s)
        if (s < n) acc = fmaf(w[s] / z, src[s].x[i], acc);
      out[q].x[i] = acc;
    }
  }
}

// Short convolution of one row for 4 channels (mlx ShortConvolution): window (s0, s1, s2 the three previous inputs,
// oldest first, and x) and conv in BF16 (FP32 sum, one rounding), (conv + x) / sqrt(K + 1) in BF16.
// the weights of channels 4t .. 4t + 3 (static: loaded before the dependency wait)
struct ConvW { uint4 a, b; };
__device__ __forceinline__ ConvW load_convw(const uint16_t* w, int t) {
  return {reinterpret_cast<const uint4*>(w)[2 * t], reinterpret_cast<const uint4*>(w)[2 * t + 1]};
}
__device__ __forceinline__ V4 conv4(const V4& s0, const V4& s1, const V4& s2, const V4& x, const ConvW& cw) {
  const uint4 w01 = cw.a, w23 = cw.b;
  const uint32_t ww[8] = {w01.x, w01.y, w01.z, w01.w, w23.x, w23.y, w23.z, w23.w};   // channel c: 4 taps = 2 words
  V4 out;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float conv = bf16lo(ww[2 * i]) * s0.x[i];
    conv = fmaf(bf16hi(ww[2 * i]), s1.x[i], conv);
    conv = fmaf(bf16lo(ww[2 * i + 1]), s2.x[i], conv);
    conv = fmaf(bf16hi(ww[2 * i + 1]), x.x[i], conv);
    out.x[i] = round_bf16((round_bf16(conv) + x.x[i]) / sqrtf(5.0f));
  }
  return out;
}

// Decode: the window from the state, which then shifts in x.
__device__ __forceinline__ V4 short_conv4(uint16_t* __restrict__ state, const ConvW& w, int t, const V4& x) {
  const V4 s0 = load_b4(state, t), s1 = load_b4(state + kHidden, t), s2 = load_b4(state + 2 * kHidden, t);
  const V4 out = conv4(s0, s1, s2, x, w);
  store_b4(state, t, s1);
  store_b4(state + kHidden, t, s2);
  store_b4(state + 2 * kHidden, t, x);
  return out;
}

// Verification row r: previous inputs from this step's rows r - 3 .. r - 1 (prev[3 - j] = row r - j) or the state.
__device__ __forceinline__ V4 staged_conv4(const uint16_t* __restrict__ state, const V4 (&prev)[3], int r,
                                           const ConvW& w, int t, const V4& x) {
  V4 s[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {                 // window slot j holds input r - 3 + j
    const int src = r - 3 + j;
    s[j] = src >= 0 ? prev[j] : load_b4(state + (3 + src) * kHidden, t);
  }
  return conv4(s[0], s[1], s[2], x, w);
}

__global__ void __launch_bounds__(kThreads) visit_boundary_kernel(const __grid_constant__ BoundaryParams p) {
  __shared__ float red[24][33];
  const int R = p.rows < 1 ? 1 : p.rows;
  if (prefetch_block(p.prefetch, R)) return;
  const int t = threadIdx.x;
  const int r = blockIdx.x;                       // the row
  const size_t D1 = kHidden;
  // inputs written two or more launches back (residual: post attention; block states and the message bank: earlier
  // boundaries) and weights: loaded before the dependency wait (the route weights, written by the previous kernel
  // with the fused expert pass, after it)
  V4 res_pre = p.finish ? load_f4(p.residual + r * D1, t) : V4{};
  V4 blk_pre[kMaxBlocks];
#pragma unroll
  for (int s = 0; s < kMaxBlocks; ++s) blk_pre[s] = p.next_hidden && p.nq && s < p.n_blocks ? load_f4(p.blocks[s] + r * D1, t) : V4{};
  V4 q_pre[3];
#pragma unroll
  for (int q = 0; q < 3; ++q) q_pre[q] = p.next_hidden && q < p.nq ? load_f4(p.att_query + q * kHidden, t) : V4{};
  const ConvW mcw = p.finish ? load_convw(p.mlp_conv_w, t) : ConvW{};
  V4 nwp[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) nwp[u] = u < p.norm_n ? load_b4(p.norm_w[u], t) : V4{};
  // loop passes 2 and 3 past a pass's first visit: the messages over the bank (an earlier boundary's normed rows and
  // the pass's bank state, written many launches back) before the wait, stored after it
  V4 msg[4];
  const bool msg_pre = p.messages && p.msg_early;
  if (msg_pre) {
    V4 src[2] = {load_f4(p.bank_block + r * D1, t), load_f4(p.bank_state + r * D1, t)};
    depth_attention<2, 4>(src, 2, p.message_query, 4, msg, red);
    __syncthreads();
  }
  pdl_wait();
  pdl_launch();
  auto mark = [&](int i) {
    if (p.timing && t == 0) {
      unsigned long long g;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
      p.timing[i] = g;
    }
  };
  mark(0);
  V4 h;                                   // the visit output (finish / embed), BF16 values
  bool have_h = false;
  if (p.finish) {
    const uint16_t* __restrict__ y = after_wait(p.expert_out);
    const float* __restrict__ rw = after_wait(p.route_w);
    // the MLP output of row q: bf16(sum_s y_s w_s) in slot order
    auto moe_of = [&](int q) {
      V4 ys[kTopK];
#pragma unroll
      for (int s = 0; s < kTopK; ++s) ys[s] = load_b4(y + (size_t(q) * kTopK + s) * kHidden, t);
      float w[kTopK];
#pragma unroll
      for (int s = 0; s < kTopK; ++s) w[s] = rw[q * kTopK + s];
      V4 moe;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < kTopK; ++s) acc = __fadd_rn(acc, __fmul_rn(ys[s].x[i], w[s]));
        moe.x[i] = round_bf16(acc);
      }
      return moe;
    };
    const V4 moe = moe_of(r);
    const V4 res = res_pre;
    V4 c;
    if (p.stage) {
      V4 prev[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) prev[j] = r - 3 + j >= 0 ? moe_of(r - 3 + j) : V4{};
      c = staged_conv4(after_wait(p.mlp_conv_state), prev, r, mcw, t, moe);
      store_b4(after_wait(p.stage) + r * D1, t, moe);
    } else {
      c = short_conv4(after_wait(p.mlp_conv_state), mcw, t, moe);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) h.x[i] = round_bf16(res.x[i] + c.x[i]);
    store_b4(after_wait(p.hidden_out) + r * D1, t, h);
    have_h = true;
  } else if (p.embed) {
    const int tok = after_wait(p.token)[r];
    const V4 e = load_b4(p.embed_table + size_t(tok) * kHidden, t);
    float v[1] = {sumsq(e)};
    block_sums<1>(v, red);
    const float inv = inv_rms(v[0], kHidden);
    const V4 nw = load_b4(p.embed_norm, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) h.x[i] = round_bf16(e.x[i] * inv * nw.x[i]);
    store_b4(after_wait(p.hidden_out) + r * D1, t, h);
    have_h = true;
  }
  mark(1);
  if (p.bank_partial) {                   // end of a loop pass: the message bank's partial state rms(h)
    float v[1] = {sumsq(h)};
    block_sums<1>(v, red);
    const float inv = inv_rms(v[0], kHidden);
    V4 b;
#pragma unroll
    for (int i = 0; i < 4; ++i) b.x[i] = h.x[i] * inv;
    store_f4(after_wait(p.bank_partial) + r * D1, t, b);
  }
  if (!p.next_hidden) return;
  // the next visit: block residual (the output just computed, or the prelude output at a pass start)
  const V4 x = (have_h && p.next_hidden == p.hidden_out) ? h : load_b4(after_wait(p.next_hidden) + r * D1, t);
  V4 nr;
  {
    float v[1] = {sumsq(x)};
    block_sums<1>(v, red);
    const float inv = inv_rms(v[0], kHidden);
#pragma unroll
    for (int i = 0; i < 4; ++i) nr.x[i] = x.x[i] * inv;
    store_f4(after_wait(p.normed_res) + r * D1, t, nr);
  }
  mark(2);
  float* att = after_wait(p.att_in) + r * 3 * D1;
  V4 att_v[3];
  if (p.nq) {
    V4 src[kMaxBlocks + 1];
#pragma unroll
    for (int s = 0; s < kMaxBlocks; ++s) src[s] = blk_pre[s];
    src[kMaxBlocks] = V4{};
    // the partial (normed residual) is source n_blocks
#pragma unroll
    for (int s = 0; s <= kMaxBlocks; ++s)
#pragma unroll
      for (int i = 0; i < 4; ++i) src[s].x[i] = s == p.n_blocks ? nr.x[i] : src[s].x[i];   // (a select, not an
                                                                                            // indexed store)
    depth_attention_q<kMaxBlocks + 1, 3>(src, p.n_blocks + 1, q_pre, p.nq, att_v, red);
#pragma unroll
    for (int q = 0; q < 3; ++q)              // (compile-time indices: att_v stays in registers)
      if (q < p.nq) store_f4(att + q * kHidden, t, att_v[q]);
  } else {
    att_v[0] = x;
    store_f4(att, t, x);
  }
  if (p.norm_n) {
    // norm_rows' arithmetic on the FP32 row: threads 0..255 sum elements t, t + 256, .. (fmaf), warp sums, the 8
    // warp partials in order; the normalized values bf16(x * inv * w)
    extern __shared__ float xn[];                // kHidden + 8 floats, dynamic: only launches with norms carry it
    float* np = xn + kHidden;
#pragma unroll
    for (int u = 0; u < 3; ++u) {
      if (u >= p.norm_n) break;
      V4 a = att_v[0];
#pragma unroll
      for (int k = 1; k < 3; ++k)
        if (p.nq == 3 && p.norm_src[u] == k) a = att_v[k];
#pragma unroll
      for (int i = 0; i < 4; ++i) xn[4 * t + i] = a.x[i];
      __syncthreads();
      if (t < 256) {
        float ss = 0.0f;
        for (int i = t; i < kHidden; i += 256) ss = fmaf(xn[i], xn[i], ss);
        ss = warp_sum(ss);
        if ((t & 31) == 0) np[t >> 5] = ss;
      }
      __syncthreads();
      float tot = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) tot += np[i];
      const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
      const V4 nw = nwp[u];
      uint16_t* o = after_wait(p.norm_out[u]) + r * D1 + 4 * t;
      ushort4 ov;
      ov.x = f32_to_bf16_bits(a.x[0] * inv * nw.x[0]);
      ov.y = f32_to_bf16_bits(a.x[1] * inv * nw.x[1]);
      ov.z = f32_to_bf16_bits(a.x[2] * inv * nw.x[2]);
      ov.w = f32_to_bf16_bits(a.x[3] * inv * nw.x[3]);
      *reinterpret_cast<ushort4*>(o) = ov;
      __syncthreads();
    }
  }
  mark(3);
  if (!p.messages) return;
  // loop passes 2 and 3: messages over the bank, and the gate inputs concat(rms(h_g), rms(m_g)) in BF16
  {
    if (!msg_pre) {
      V4 src[2] = {load_f4(after_wait(p.bank_block) + r * D1, t), load_f4(after_wait(p.bank_state) + r * D1, t)};
      depth_attention<2, 4>(src, 2, p.message_query, 4, msg, red);
    }
    float* mo = after_wait(p.messages_out) + r * 4 * D1;
#pragma unroll
    for (int q = 0; q < 4; ++q) store_f4(mo + q * kHidden, t, msg[q]);
  }
  float v[6];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const V4 hv = p.nq == 3 ? att_v[g] : att_v[0];
    v[2 * g] = sumsq(hv);
    v[2 * g + 1] = sumsq(msg[g]);
  }
  block_sums<6>(v, red);
  uint16_t* gin = after_wait(p.gate_in);
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const V4 hg = p.nq == 3 ? att_v[g] : att_v[0];
    const float ih = inv_rms(v[2 * g], kHidden), im = inv_rms(v[2 * g + 1], kHidden);
    V4 a, b;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      a.x[i] = hg.x[i] * ih;
      b.x[i] = msg[g].x[i] * im;
    }
    uint16_t* gr = gin + (size_t(g) * R + r) * 2 * kHidden;
    store_b4(gr, t, a);
    store_b4(gr + kHidden, t, b);
  }
  mark(4);
}

__global__ void __launch_bounds__(kThreads) post_attention_kernel(const __grid_constant__ PostAttentionParams p) {
  __shared__ float red[8][33];
  const int R = p.rows < 1 ? 1 : p.rows;
  if (prefetch_block(p.prefetch, R)) return;
  const int t = threadIdx.x;
  const int row = blockIdx.x;
  const size_t D1 = kHidden;
  // written four or more launches back (this visit's boundary, earlier boundaries, the previous step) and weights:
  // loaded before the wait (the output projection)
  const V4 br = load_b4(p.block_residual + row * D1, t);
  V4 src[kMaxBlocks + 1];
#pragma unroll
  for (int s = 0; s < kMaxBlocks; ++s) src[s] = s < p.n_blocks ? load_f4(p.blocks[s] + row * D1, t) : V4{};
  src[kMaxBlocks] = V4{};
  V4 mq[1] = {load_f4(p.mlp_query, t)};
  V4 st[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) st[j] = load_b4(p.attn_conv_state + j * kHidden, t);
  const V4 m_pre = p.message ? load_f4(p.message + row * 4 * D1, t) : V4{};
  const ConvW acw = load_convw(p.attn_conv_w, t);
  pdl_wait();
  pdl_launch();
  const uint16_t* ao = after_wait(p.attn_out);
  const V4 o = load_b4(ao + row * D1, t);
  V4 c;
  if (p.stage) {
    V4 prev[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) prev[j] = row - 3 + j >= 0 ? load_b4(ao + (row - 3 + j) * D1, t) : V4{};
    V4 sw[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      sw[j] = prev[j];
#pragma unroll
      for (int k = 0; k < 3; ++k)            // st[row + j] for row - 3 + j < 0 (selects, compile-time indices)
#pragma unroll
        for (int i = 0; i < 4; ++i) sw[j].x[i] = opaque_sel(row - 3 + j < 0 && row + j == k, st[k].x[i], sw[j].x[i]);
    }
    c = conv4(sw[0], sw[1], sw[2], o, acw);
    store_b4(after_wait(p.stage) + row * D1, t, o);
  } else {
    uint16_t* state = after_wait(p.attn_conv_state);
    c = conv4(st[0], st[1], st[2], o, acw);
    store_b4(state, t, st[1]);
    store_b4(state + kHidden, t, st[2]);
    store_b4(state + 2 * kHidden, t, o);
  }
  V4 r;
#pragma unroll
  for (int i = 0; i < 4; ++i) r.x[i] = p.boundary ? c.x[i] : br.x[i] + c.x[i];
  store_f4(after_wait(p.residual) + row * D1, t, r);
  float v[1] = {sumsq(r)};
  block_sums<1>(v, red);
  const float inv = inv_rms(v[0], kHidden);
  V4 nr;
#pragma unroll
  for (int i = 0; i < 4; ++i) nr.x[i] = r.x[i] * inv;
#pragma unroll
  for (int s = 0; s <= kMaxBlocks; ++s)
#pragma unroll
    for (int i = 0; i < 4; ++i) src[s].x[i] = s == p.n_blocks ? nr.x[i] : src[s].x[i];
  V4 mi[1];
  depth_attention_q<kMaxBlocks + 1, 1>(src, p.n_blocks + 1, mq, 1, mi, red);
  store_f4(after_wait(p.mlp_in) + row * D1, t, mi[0]);
  if (!p.message) return;
  const V4 m = m_pre;
  float u[2] = {sumsq(mi[0]), sumsq(m)};
  block_sums<2>(u, red);
  const float ih = inv_rms(u[0], kHidden), im = inv_rms(u[1], kHidden);
  V4 a, b;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a.x[i] = mi[0].x[i] * ih;
    b.x[i] = m.x[i] * im;
  }
  uint16_t* gin = after_wait(p.gate_in) + row * 2 * D1;
  store_b4(gin, t, a);
  store_b4(gin + kHidden, t, b);
}

// mixed_g = (1 - tanh(F_g s_g)) h_g + tanh(G_g s_g) m_g for each gate g: F_g, G_g [D][64] BF16 weights (FP32 in the
// reference, BF16 values), s_g the 64 down-projected states (BF16), h_g / m_g FP32. Eight lanes per output.
__global__ void __launch_bounds__(256) gate_mix_kernel(const __grid_constant__ GateMixParams p) {
  const int D = kHidden;
  const int lane8 = threadIdx.x & 7;
  const int R = p.rows < 1 ? 1 : p.rows;
  const int j = (blockIdx.x * 256 + threadIdx.x) >> 3;          // output over all gates and rows
  const int g = j / (D * R), rc = j - g * D * R, r = rc / D, c = rc - r * D;
  const bool live = g < p.count;
  const uint4 fw = live ? ldg_stream(static_cast<const uint16_t*>(p.forget[g]) + size_t(c) * 64 + lane8 * 8) : uint4{};
  const uint4 gw = live ? ldg_stream(static_cast<const uint16_t*>(p.gain[g]) + size_t(c) * 64 + lane8 * 8) : uint4{};
  // h (the boundary's / post_attention's rows) and m (the boundary's messages): two launches back, before the wait
  float hpre = 0.0f, mpre = 0.0f;
  if (live && lane8 == 0) {
    hpre = p.h[g][size_t(r) * p.h_stride + c];
    mpre = p.m[g][size_t(r) * p.m_stride + c];
  }
  pdl_wait();
  pdl_launch();
  float f = 0.0f, gn = 0.0f;
  if (live) {
    const uint16_t* s = after_wait(p.states[g]) + r * 64 + lane8 * 8;
    const uint32_t fv[4] = {fw.x, fw.y, fw.z, fw.w}, gv[4] = {gw.x, gw.y, gw.z, gw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float s0 = bf16_to_f32(s[2 * i]), s1 = bf16_to_f32(s[2 * i + 1]);
      f = fmaf(bf16lo(fv[i]), s0, f);
      f = fmaf(bf16hi(fv[i]), s1, f);
      gn = fmaf(bf16lo(gv[i]), s0, gn);
      gn = fmaf(bf16hi(gv[i]), s1, gn);
    }
  }
  f = warp_sum<8>(f);
  gn = warp_sum<8>(gn);
  if (live && lane8 == 0) {
    const float h = hpre;
    const float m = mpre;
    after_wait(p.out[g])[size_t(r) * p.out_stride + c] = (1.0f - tanhf(f)) * h + tanhf(gn) * m;
  }
}

// Prefill chunks: grid (gates, kHidden / 64, row tiles of 64), 4 warps x 16 rows x 64 channels. f / gn on BF16 MMAs
// (the BF16 products exact in FP32), the mix as in gate_mix_kernel.
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__global__ void __launch_bounds__(128) gate_mix_tiles_kernel(const __grid_constant__ GateMixParams p) {
  pdl_wait();
  pdl_launch();
  const int gt = blockIdx.x, c0 = blockIdx.y * 64;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
  const int R = p.rows, r0 = blockIdx.z * 64 + warp * 16;
  if (r0 >= R) return;
  auto u32 = [](const uint16_t* q) { return *reinterpret_cast<const uint32_t*>(q); };
  const uint16_t* S = after_wait(p.states[gt]);
  const int ra = min(r0 + g, R - 1), rb = min(r0 + g + 8, R - 1);
  uint32_t a[4][4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    a[kk][0] = u32(S + ra * 64 + 16 * kk + 2 * c);
    a[kk][1] = u32(S + rb * 64 + 16 * kk + 2 * c);
    a[kk][2] = u32(S + ra * 64 + 16 * kk + 8 + 2 * c);
    a[kk][3] = u32(S + rb * 64 + 16 * kk + 8 + 2 * c);
  }
  float f[8][4], q[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const uint16_t* F = static_cast<const uint16_t*>(p.forget[gt]) + size_t(c0 + 8 * n + g) * 64;
    const uint16_t* G = static_cast<const uint16_t*>(p.gain[gt]) + size_t(c0 + 8 * n + g) * 64;
#pragma unroll
    for (int e = 0; e < 4; ++e) f[n][e] = q[n][e] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      mma_bf16(f[n], a[kk], u32(F + 16 * kk + 2 * c), u32(F + 16 * kk + 8 + 2 * c));
      mma_bf16(q[n], a[kk], u32(G + 16 * kk + 2 * c), u32(G + 16 * kk + 8 + 2 * c));
    }
  }
  const float* h = after_wait(p.h[gt]);
  const float* m = after_wait(p.m[gt]);
  float* out = after_wait(p.out[gt]);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = r0 + g + 8 * (e >> 1), col = c0 + 8 * n + 2 * c + (e & 1);
      if (row < R)
        out[size_t(row) * p.out_stride + col] = (1.0f - tanhf(f[n][e])) * h[size_t(row) * p.h_stride + col] +
                                                tanhf(q[n][e]) * m[size_t(row) * p.m_stride + col];
    }
}

}  // namespace

// msg_gate: the message visits' controllers in one kernel, a cluster of 8 blocks (288 threads) per (gate, pair of
// rows). Block b computes the down-projection outputs 8 b .. 8 b + 7 (warps 0-7, gemv_rows' arithmetic: lane chunks
// lane + 32 u in order, FP32 sums of dot8, the warp sum, BF16); the cluster's 64 states are pulled from the blocks'
// shared memory (DSMEM). Thread j mixes channel 32 b + (j % 32) + 256 (j / 32) with gate_mix's arithmetic (the eight
// lanes' FMA chains, then their butterfly sum in its order). Those are the channels of norm_rows' warp b (its thread
// 32 b + G sums 32 b + G + 256 m in order), so block b's partial is that warp's; the eight partials are pulled and
// summed in order (norm_rows' arithmetic). h / m / the inputs are the previous kernel's output (after the wait).
constexpr int kMsgCluster = 8, kMsgRows = 2, kMsgK8 = 2 * kHidden / 8, kMsgCh = kHidden / kMsgCluster;
constexpr int kMsgThreads = kMsgCh;
static_assert(kMsgK8 == 18 * 32 && kHidden == 9 * 256 && kMsgCh == 9 * 32, "msg_gate shapes");
constexpr size_t kMsgSmem = size_t(kMsgRows) * kMsgK8 * 16 + size_t(kMsgCh) * 16 * 16;
// gemv_rows' eight products of one chunk (inkling2_gemv.cu's dot8)
__device__ __forceinline__ float dot8(const uint4& w, const float* x) {
  float acc = 0.0f;
  const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = fmaf(bf16lo(ww[i]), x[2 * i], acc);
    acc = fmaf(bf16hi(ww[i]), x[2 * i + 1], acc);
  }
  return acc;
}
__device__ __forceinline__ uint32_t cluster_addr(const void* smem, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(r) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem))), "r"(rank));
  return r;
}
__device__ __forceinline__ float4 ld_cluster4(uint32_t addr) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ float ld_cluster(uint32_t addr) {
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// gate_mix's f / gn of one channel: eight 8-FMA chains (its lanes), summed in its butterfly's order
__device__ __forceinline__ float mix_dot(const uint4* w, const float* s) {
  float v[8];
#pragma unroll
  for (int l = 0; l < 8; ++l) {
    const uint32_t wv[4] = {w[l].x, w[l].y, w[l].z, w[l].w};
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      a = fmaf(bf16lo(wv[i]), s[8 * l + 2 * i], a);
      a = fmaf(bf16hi(wv[i]), s[8 * l + 2 * i + 1], a);
    }
    v[l] = a;
  }
  return ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
}
__global__ void __cluster_dims__(kMsgCluster, 1, 1) __launch_bounds__(kMsgThreads) msg_gate_kernel(const __grid_constant__ GateMixParams p) {
  extern __shared__ __align__(16) uint8_t dyn[];
  uint4* xs = reinterpret_cast<uint4*>(dyn);                                 // [2][576] the input rows
  uint4* mw = reinterpret_cast<uint4*>(dyn + size_t(kMsgRows) * kMsgK8 * 16); // [16][288]: forget 0-7, gain 8-15
  __shared__ __align__(16) float own[kMsgRows][8];      // this block's states (pulled by the cluster)
  __shared__ __align__(16) float sst[kMsgRows][64];
  __shared__ float own_part[kMsgRows];
  __shared__ float spart[kMsgRows][kMsgCluster];
  __shared__ float vals[kMsgRows][9][32];
  const int R = p.rows < 1 ? 1 : p.rows, ng = (R + kMsgRows - 1) / kMsgRows;
  const int b = blockIdx.x, g = blockIdx.y / ng, r0 = (blockIdx.y % ng) * kMsgRows;
  const int nr = R - r0 < kMsgRows ? R - r0 : kMsgRows;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31, o = 8 * b + warp;
  const int c = 32 * b + lane + 256 * warp;            // this thread's channel (G = lane, m = warp)
  // static data before the wait: the warp's down-projection row, the channel's mix weights (to shared memory), the
  // norm weight
  uint4 wv[18];
  if (warp < 8) {
    const uint4* wr = reinterpret_cast<const uint4*>(p.down[g]) + size_t(o) * kMsgK8;
#pragma unroll
    for (int u = 0; u < 18; ++u) wv[u] = ldg_stream(wr + lane + 32 * u);
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    cp_async16(mw + k * kMsgCh + t, p.forget[g] + size_t(c) * 64 + k * 8);
    cp_async16(mw + (8 + k) * kMsgCh + t, p.gain[g] + size_t(c) * 64 + k * 8);
  }
  cp_async_commit();
  const bool norm = p.norm_w[0] != nullptr;
  const float nwc = norm ? bf16_to_f32(p.norm_w[g][c]) : 0.0f;
  pdl_wait();
  float hv[kMsgRows], mv[kMsgRows];
#pragma unroll
  for (int r = 0; r < kMsgRows; ++r) {
    hv[r] = r < nr ? after_wait(p.h[g])[size_t(r0 + r) * p.h_stride + c] : 0.0f;
    mv[r] = r < nr ? after_wait(p.m[g])[size_t(r0 + r) * p.m_stride + c] : 0.0f;
  }
  {
    const uint4* src = reinterpret_cast<const uint4*>(after_wait(p.gin[g]) + size_t(r0) * p.gin_stride);
    for (int i = t; i < nr * kMsgK8; i += kMsgThreads)
      xs[i] = src[size_t(i / kMsgK8) * (p.gin_stride / 8) + i % kMsgK8];
  }
  __syncthreads();
  pdl_launch();
  // down projection (rows past nr computed on stale staging, not stored)
  if (warp < 8) {
    float acc[kMsgRows] = {0.0f, 0.0f};
#pragma unroll
    for (int u = 0; u < 18; ++u) {
#pragma unroll
      for (int r = 0; r < kMsgRows; ++r) {
        const uint4 xv = xs[r * kMsgK8 + lane + 32 * u];
        const float xf[8] = {bf16lo(xv.x), bf16hi(xv.x), bf16lo(xv.y), bf16hi(xv.y),
                             bf16lo(xv.z), bf16hi(xv.z), bf16lo(xv.w), bf16hi(xv.w)};
        acc[r] += dot8(wv[u], xf);
      }
    }
#pragma unroll
    for (int r = 0; r < kMsgRows; ++r) {
      const float sv = warp_sum(acc[r]);
      if (lane == 0 && r < nr) {
        const uint16_t sb = f32_to_bf16_bits(sv);
        after_wait(p.states_out[g])[size_t(r0 + r) * 64 + o] = sb;
        own[r][warp] = bf16_to_f32(sb);
      }
    }
  }
  cluster_sync_all();                                  // (also: the cluster's blocks running)
  if (t < 2 * kMsgRows * kMsgCluster) {                // pull the states: (row, block, half)
    const int r = t >> 4, q = (t >> 1) & 7, h2 = t & 1;
    if (r < nr) *reinterpret_cast<float4*>(&sst[r][8 * q + 4 * h2]) = ld_cluster4(cluster_addr(&own[r][4 * h2], q));
  }
  __syncthreads();
  cp_async_wait_all();
  // the mix
  float val[kMsgRows];
  {
    uint4 fw[8], gw[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      fw[k] = mw[k * kMsgCh + t];
      gw[k] = mw[(8 + k) * kMsgCh + t];
    }
#pragma unroll
    for (int r = 0; r < kMsgRows; ++r) {
      val[r] = 0.0f;
      if (r >= nr) break;
      const float f = mix_dot(fw, sst[r]), gn = mix_dot(gw, sst[r]);
      const float h = hv[r], mm = mv[r];
      const float v = (1.0f - tanhf(f)) * h + tanhf(gn) * mm;
      after_wait(p.out[g])[size_t(r0 + r) * p.out_stride + c] = v;
      val[r] = v;
      vals[r][warp][lane] = v;
    }
  }
  if (!norm) {
    asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");   // (pulls done)
    return;
  }
  __syncthreads();
  // norm_rows' arithmetic: thread G of warp 0 is norm_rows' thread 32 b + G
  if (warp == 0) {
#pragma unroll
    for (int r = 0; r < kMsgRows; ++r) {
      if (r >= nr) break;
      float ss = 0.0f;
#pragma unroll
      for (int m = 0; m < 9; ++m) ss = fmaf(vals[r][m][lane], vals[r][m][lane], ss);
      const float x = warp_sum(ss);
      if (lane == 0) own_part[r] = x;
    }
  }
  cluster_sync_all();
  if (t < kMsgRows * kMsgCluster) {
    const int r = t >> 3, q = t & 7;
    if (r < nr) spart[r][q] = ld_cluster(cluster_addr(&own_part[r], q));
  }
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");     // (this block's pulls issued and done below)
  __syncthreads();
#pragma unroll
  for (int r = 0; r < kMsgRows; ++r) {
    if (r >= nr) break;
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < kMsgCluster; ++i) tot += spart[r][i];
    const float inv = 1.0f / sqrtf(tot / float(kHidden) + 1e-6f);
    after_wait(p.norm_out[g])[size_t(r0 + r) * kHidden + c] = f32_to_bf16_bits(val[r] * inv * nwc);
  }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");       // no block exits before the cluster's pulls
}

void visit_boundary(const BoundaryParams& p, cudaStream_t stream) {
  if (p.n_blocks < 0 || p.n_blocks > kMaxBlocks || p.nq < 0 || p.nq > 3)
    throw std::invalid_argument("visit_boundary: bad block or query count");
  const int R = p.rows < 1 ? 1 : p.rows;
  if (R > kMaxChunk) throw std::invalid_argument("visit_boundary: at most kMaxChunk rows");
  if (p.norm_n < 0 || p.norm_n > 3) throw std::invalid_argument("visit_boundary: 0 to 3 norms");
  launch_pdl(visit_boundary_kernel, dim3(R + (p.prefetch.count ? kPrefetchBlocks : 0)), dim3(kThreads),
             p.norm_n ? (kHidden + 8) * sizeof(float) : 0, stream, p);
}

void post_attention(const PostAttentionParams& p, cudaStream_t stream) {
  if (p.n_blocks < 1 || p.n_blocks > kMaxBlocks) throw std::invalid_argument("post_attention: bad block count");
  const int R = p.rows < 1 ? 1 : p.rows;
  if (R > kMaxChunk) throw std::invalid_argument("post_attention: at most kMaxChunk rows");
  launch_pdl(post_attention_kernel, dim3(R + (p.prefetch.count ? kPrefetchBlocks : 0)), dim3(kThreads), 0, stream, p);
}

void msg_gate(const GateMixParams& p, cudaStream_t stream) {
  if (p.count < 1 || p.count > 3 || p.rows > kMaxRows) throw std::invalid_argument("msg_gate: 1 to 3 gates, 1 to 8 rows");
  for (int g = 0; g < p.count; ++g)
    if (!p.down[g] || !p.gin[g] || !p.states_out[g] || (p.norm_w[0] && (!p.norm_w[g] || !p.norm_out[g])))
      throw std::invalid_argument("msg_gate: down / gin / states per gate (norm_w / norm_out per gate when set)");
  if (p.gin_stride % 8) throw std::invalid_argument("msg_gate: gin_stride a multiple of 8");
  static std::atomic<unsigned long long> smem_set{0};
  if (first_use_on_device(smem_set))
    check_cuda(cudaFuncSetAttribute(msg_gate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kMsgSmem)),
               "msg_gate smem");
  const int R = p.rows < 1 ? 1 : p.rows;
  launch_pdl(msg_gate_kernel, dim3(kMsgCluster, p.count * ((R + kMsgRows - 1) / kMsgRows)), dim3(kMsgThreads),
             kMsgSmem, stream, p);
}

void gate_mix(const GateMixParams& p, cudaStream_t stream) {
  if (p.count < 1 || p.count > 3) throw std::invalid_argument("gate_mix: 1 to 3 gates");
  if (p.rows > kMaxRows) {
    launch_pdl(gate_mix_tiles_kernel, dim3(p.count, kHidden / 64, (p.rows + 63) / 64), dim3(128), 0, stream, p);
    return;
  }
  const int outputs = p.count * kHidden * (p.rows < 1 ? 1 : p.rows);
  launch_pdl(gate_mix_kernel, dim3((outputs * 8 + 255) / 256), dim3(256), 0, stream, p);
}

}  // namespace inkling2
}  // namespace flash_rt

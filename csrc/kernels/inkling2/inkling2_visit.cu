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

__device__ __forceinline__ float inv_rms(float sumsq_total, int d) { return 1.0f / sqrtf(sumsq_total / float(d) + 1e-6f); }

// Depth attention for this thread's 4 channels: sources src[0..n) (registers), queries q[0..nq) (global FP32
// [nq][kHidden]); out[q] = sum_s softmax_s(src_s . query_q / sqrt(d)) src_s.
template <int MaxN, int MaxQ>
__device__ __forceinline__ void depth_attention(const V4 (&src)[MaxN], int n, const float* query, int nq, V4 (&out)[MaxQ],
                                                float (*red)[33]) {
  const int t = threadIdx.x;
  V4 qv[MaxQ];
#pragma unroll
  for (int q = 0; q < MaxQ; ++q) qv[q] = q < nq ? load_f4(query + q * kHidden, t) : V4{};
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
  block_sums<MaxN * MaxQ>(part, red);
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
__device__ __forceinline__ V4 conv4(const V4& s0, const V4& s1, const V4& s2, const V4& x, const uint16_t* __restrict__ w,
                                    int t) {
  const uint4 w01 = reinterpret_cast<const uint4*>(w)[2 * t], w23 = reinterpret_cast<const uint4*>(w)[2 * t + 1];
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
__device__ __forceinline__ V4 short_conv4(uint16_t* __restrict__ state, const uint16_t* __restrict__ w, int t, const V4& x) {
  const V4 s0 = load_b4(state, t), s1 = load_b4(state + kHidden, t), s2 = load_b4(state + 2 * kHidden, t);
  const V4 out = conv4(s0, s1, s2, x, w, t);
  store_b4(state, t, s1);
  store_b4(state + kHidden, t, s2);
  store_b4(state + 2 * kHidden, t, x);
  return out;
}

// Verification row r: previous inputs from this step's rows r - 3 .. r - 1 (prev[3 - j] = row r - j) or the state.
__device__ __forceinline__ V4 staged_conv4(const uint16_t* __restrict__ state, const V4 (&prev)[3], int r,
                                           const uint16_t* __restrict__ w, int t, const V4& x) {
  V4 s[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {                 // window slot j holds input r - 3 + j
    const int src = r - 3 + j;
    s[j] = src >= 0 ? prev[j] : load_b4(state + (3 + src) * kHidden, t);
  }
  return conv4(s[0], s[1], s[2], x, w, t);
}

__global__ void __launch_bounds__(kThreads) visit_boundary_kernel(const __grid_constant__ BoundaryParams p) {
  __shared__ float red[24][33];
  const int R = p.rows < 1 ? 1 : p.rows;
  if (prefetch_block(p.prefetch, R)) return;
  pdl_wait();
  pdl_launch();
  const int t = threadIdx.x;
  const int r = blockIdx.x;                       // the row
  const size_t D1 = kHidden;
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
    const V4 res = load_f4(after_wait(p.residual) + r * D1, t);
    V4 c;
    if (p.stage) {
      V4 prev[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) prev[j] = r - 3 + j >= 0 ? moe_of(r - 3 + j) : V4{};
      c = staged_conv4(after_wait(p.mlp_conv_state), prev, r, p.mlp_conv_w, t, moe);
      store_b4(after_wait(p.stage) + r * D1, t, moe);
    } else {
      c = short_conv4(after_wait(p.mlp_conv_state), p.mlp_conv_w, t, moe);
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
    for (int s = 0; s < kMaxBlocks; ++s) src[s] = s < p.n_blocks ? load_f4(after_wait(p.blocks[s]) + r * D1, t) : V4{};
    // the partial (normed residual) is source n_blocks
#pragma unroll
    for (int s = 0; s <= kMaxBlocks; ++s)
      if (s == p.n_blocks) src[s] = nr;
    depth_attention<kMaxBlocks + 1, 3>(src, p.n_blocks + 1, p.att_query, p.nq, att_v, red);
    for (int q = 0; q < p.nq; ++q) store_f4(att + q * kHidden, t, att_v[q]);
  } else {
    att_v[0] = x;
    store_f4(att, t, x);
  }
  mark(3);
  if (!p.messages) return;
  // loop passes 2 and 3: messages over the bank, and the gate inputs concat(rms(h_g), rms(m_g)) in BF16
  V4 msg[4];
  {
    V4 src[2] = {load_f4(after_wait(p.bank_block) + r * D1, t), load_f4(after_wait(p.bank_state) + r * D1, t)};
    depth_attention<2, 4>(src, 2, p.message_query, 4, msg, red);
    float* mo = after_wait(p.messages_out) + r * 4 * D1;
#pragma unroll
    for (int q = 0; q < 4; ++q) store_f4(mo + q * kHidden, t, msg[q]);
  }
  float v[6];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    v[2 * g] = sumsq(p.nq == 3 ? att_v[g] : att_v[0]);
    v[2 * g + 1] = sumsq(msg[g]);
  }
  block_sums<6>(v, red);
  uint16_t* gin = after_wait(p.gate_in);
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const V4& hg = p.nq == 3 ? att_v[g] : att_v[0];
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
  pdl_wait();
  pdl_launch();
  const int t = threadIdx.x;
  const int row = blockIdx.x;
  const size_t D1 = kHidden;
  const uint16_t* ao = after_wait(p.attn_out);
  const V4 o = load_b4(ao + row * D1, t);
  const V4 br = load_b4(after_wait(p.block_residual) + row * D1, t);
  V4 src[kMaxBlocks + 1];
#pragma unroll
  for (int s = 0; s < kMaxBlocks; ++s) src[s] = s < p.n_blocks ? load_f4(after_wait(p.blocks[s]) + row * D1, t) : V4{};
  V4 c;
  if (p.stage) {
    V4 prev[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) prev[j] = row - 3 + j >= 0 ? load_b4(ao + (row - 3 + j) * D1, t) : V4{};
    c = staged_conv4(after_wait(p.attn_conv_state), prev, row, p.attn_conv_w, t, o);
    store_b4(after_wait(p.stage) + row * D1, t, o);
  } else {
    c = short_conv4(after_wait(p.attn_conv_state), p.attn_conv_w, t, o);
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
    if (s == p.n_blocks) src[s] = nr;
  V4 mi[1];
  depth_attention<kMaxBlocks + 1, 1>(src, p.n_blocks + 1, p.mlp_query, 1, mi, red);
  store_f4(after_wait(p.mlp_in) + row * D1, t, mi[0]);
  if (!p.message) return;
  const V4 m = load_f4(after_wait(p.message) + row * 4 * D1, t);
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
    const float h = after_wait(p.h[g])[size_t(r) * p.h_stride + c], m = after_wait(p.m[g])[size_t(r) * p.m_stride + c];
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

void visit_boundary(const BoundaryParams& p, cudaStream_t stream) {
  if (p.n_blocks < 0 || p.n_blocks > kMaxBlocks || p.nq < 0 || p.nq > 3)
    throw std::invalid_argument("visit_boundary: bad block or query count");
  const int R = p.rows < 1 ? 1 : p.rows;
  if (R > kMaxChunk) throw std::invalid_argument("visit_boundary: at most kMaxChunk rows");
  launch_pdl(visit_boundary_kernel, dim3(R + (p.prefetch.count ? kPrefetchBlocks : 0)), dim3(kThreads), 0, stream, p);
}

void post_attention(const PostAttentionParams& p, cudaStream_t stream) {
  if (p.n_blocks < 1 || p.n_blocks > kMaxBlocks) throw std::invalid_argument("post_attention: bad block count");
  const int R = p.rows < 1 ? 1 : p.rows;
  if (R > kMaxChunk) throw std::invalid_argument("post_attention: at most kMaxChunk rows");
  launch_pdl(post_attention_kernel, dim3(R + (p.prefetch.count ? kPrefetchBlocks : 0)), dim3(kThreads), 0, stream, p);
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

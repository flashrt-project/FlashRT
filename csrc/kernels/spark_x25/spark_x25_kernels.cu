// SPDX-License-Identifier: Apache-2.0
//
// Spark-X2.5-specific decode kernels.
//
// Three pieces of the model have no equivalent in the FlashRT kernel library,
// so they live here, in the model repository:
//
//   1. gelu_mul_to_nvfp4_swizzled_bf16
//      Spark's MLP is down_proj(gelu(gate_proj(x)) * up_proj(x)) with *exact
//      erf* GELU, whereas FlashRT ships the SiLU variant. This mirrors
//      FlashRT's silu_mul_to_nvfp4_swizzled byte-for-byte apart from the
//      activation, so the packed values and UE4M3 scales are produced by the
//      same convention the MMA consumes.
//
//   2. qkv_post_rope_kvwrite_bf16
//      Spark applies partial RoPE (rotary factor 0.25 on the 9 full-attention
//      layers, 1.0 on the 27 sliding ones) to Q and K, with no q/k RMSNorm,
//      and writes the KV cache. FlashRT's projections all attach a q/k norm,
//      so none of them can be used unchanged.
//      Sliding layers use a mirrored ring cache: each key/value is written at
//      slot (pos % W) *and* at slot (pos % W) + W. The live window is then
//      always a contiguous run of W entries starting at (pos - W + 1) % W, so
//      an ordinary contiguous attention call over W keys computes the sliding
//      window exactly -- no windowed attention kernel is needed.
//
//   3. attn_out_gate_bf16
//      Spark multiplies the attention output by sigmoid(g_proj(x)) per head.
//
// Numerics follow modeling_spark.py: rotary and the gate are computed in fp32
// and rounded to bf16 at the same points the reference does.

#include "spark_x25_kernels.cuh"

// Shared NVFP4 element/scale converters. Included rather than copied so this
// file and FlashRT's MMA path encode the same wire format from one source.
#include "nvfp4_convert.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt::spark_x25 {

// Exact erf GELU, matching torch's F.gelu(approximate='none') on bf16 inputs:
// the op is widened to fp32, evaluated, and rounded back to bf16.
__device__ __forceinline__ float gelu_f32(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// ─────────────── 1. GELU(gate) * up  ->  NVFP4 (swizzled SF) ───────────────
//
// One block per row, matching FlashRT's silu variant: smem holds the per-block
// scales and the bf16 activation product so pass 2 never re-reads from HBM.
__global__ void gelu_mul_to_nvfp4_swizzled_kernel(
    const __nv_bfloat16* __restrict__ gate,   // (rows, gate_ld)
    const __nv_bfloat16* __restrict__ up,     // (rows, up_ld)
    uint8_t* __restrict__ packed,             // (rows, cols/2)
    uint8_t* __restrict__ sf_swz,
    int cols, int num_blocks, int n_col_blocks,
    int gate_ld, int up_ld, int chunk_cols) {
  const int row = blockIdx.y;
  const int col_begin = blockIdx.x * chunk_cols;
  // Each block owns `chunk_cols` columns of the row. Scale factors are per
  // 16-element block, so as long as the chunk boundaries land on 16 the blocks
  // are fully independent and need no cross-block reduction.
  const __nv_bfloat16* gate_row = gate + (size_t)row * gate_ld + col_begin;
  const __nv_bfloat16* up_row = up + (size_t)row * up_ld + col_begin;
  uint8_t* packed_row = packed + (size_t)row * (cols / 2) + col_begin / 2;
  const int blk0 = col_begin >> 4;          // first 16-element block of this chunk
  const int chunk_blocks = chunk_cols >> 4;

  extern __shared__ __align__(16) uint8_t smem_raw[];
  float* smem_scales = reinterpret_cast<float*>(smem_raw);
  __nv_bfloat16* smem_val =
      reinterpret_cast<__nv_bfloat16*>(smem_raw + chunk_blocks * sizeof(float));

  for (int b = threadIdx.x; b < chunk_blocks; b += blockDim.x) smem_scales[b] = 0.0f;
  __syncthreads();

  // Pass 1: activation + per-block amax. The two bf16 round-trips reproduce
  // exactly what the reference does: gelu rounds to bf16, the product rounds
  // again, because both operands are bf16 in PyTorch.
  for (int i = threadIdx.x; i < chunk_cols; i += blockDim.x) {
    float g = __bfloat162float(gate_row[i]);
    float u = __bfloat162float(up_row[i]);
    __nv_bfloat16 gelu_bf = __float2bfloat16(gelu_f32(g));
    __nv_bfloat16 prod = __float2bfloat16(__bfloat162float(gelu_bf) * u);
    smem_val[i] = prod;
    float val = fabsf(__bfloat162float(prod));
    atomicMax((int*)&smem_scales[i >> 4], __float_as_int(val));
  }
  __syncthreads();

  // Pass 2: amax -> UE4M3 ceil, written straight into swizzled layout.
  const int rb = row / 128;
  const int ri = row % 128;
  for (int b = threadIdx.x; b < chunk_blocks; b += blockDim.x) {
    float amax = __int_as_float(*(int*)&smem_scales[b]);
    uint8_t ue = float_to_ue4m3_ceil(amax / 6.0f);
    const int bg = blk0 + b;
    int cb = bg / 4, ci = bg % 4;
    sf_swz[(rb * n_col_blocks + cb) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + ci] = ue;
    smem_scales[b] = ue4m3_to_float(ue);
  }
  __syncthreads();

  const int half_cols = chunk_cols >> 1;
  for (int p = threadIdx.x; p < half_cols; p += blockDim.x) {
    int i = p * 2;
    float s0 = smem_scales[i >> 4];
    float s1 = smem_scales[(i + 1) >> 4];
    float v0 = __bfloat162float(smem_val[i]) * (s0 > 0.f ? 1.f / s0 : 0.f);
    float v1 = __bfloat162float(smem_val[i + 1]) * (s1 > 0.f ? 1.f / s1 : 0.f);
    packed_row[p] = (uint8_t)((float_to_fp4_e2m1(v1) << 4) | (float_to_fp4_e2m1(v0) & 0x0F));
  }
}

void gelu_mul_to_nvfp4_swizzled_bf16(
    const void* gate, const void* up,
    uint8_t* packed, uint8_t* sf_swz,
    int rows, int cols, int gate_ld, int up_ld, cudaStream_t stream) {
  const int num_blocks = cols / 16;
  const int n_col_blocks = (num_blocks + 3) / 4;
  // One block per row leaves the GPU almost idle on this shape, so the row is
  // split across blocks. Chunks must be multiples of 64 columns so both the
  // 16-element scale blocks and the 32-wide vector writes stay aligned.
  int split = 1;
  while (split < 8 && (cols / (split * 2)) % 64 == 0) split *= 2;
  const int chunk = cols / split;
  const int chunk_blocks = chunk / 16;
  const size_t smem = (size_t)chunk_blocks * sizeof(float)
                    + (size_t)chunk * sizeof(__nv_bfloat16);
  dim3 grid(split, rows);
  gelu_mul_to_nvfp4_swizzled_kernel<<<grid, 256, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<const __nv_bfloat16*>(up),
      packed, sf_swz, cols, num_blocks, n_col_blocks,
      gate_ld, up_ld, chunk);
  (void)n_col_blocks;
  (void)chunk_blocks;
}

// Reference-exact rotation step.
//
// modeling_spark.py writes ``x_rot * c + rotate_half(x_rot) * s``, i.e. two
// separately-rounded fp32 multiplies followed by a rounded add. nvcc contracts
// that into an FMA by default, which rounds once instead of twice and flips the
// odd bf16 result by one ulp. The _rn intrinsics forbid the contraction so the
// kernel reproduces the reference bit for bit.
// ── E4M3 (fp8) KV ────────────────────────────────────────────────────────
//
// The decode attention is bandwidth-bound on the KV bytes, and at 131k those
// bytes are 4.83 GB against a 425.8 GB/s ceiling -- a 11.3 ms floor that is
// already past what 2x Ollama leaves for attention. Halving them is the only
// way through, so K and V are written twice: bf16, which prefill's FA2 still
// reads, and E4M3 with a per-(row, KV head) amax scale, which decode reads.
// E4M3 over E5M2 because the extra mantissa bit is worth more than the extra
// exponent range at these magnitudes.
__device__ __forceinline__ uint8_t float_to_e4m3(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t x) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(x, __NV_E4M3);
  return __half2float(__half(h));
}

// Two E4M3 bytes in one instruction. The naive per-element conversion is what
// made the first fp8 attempt *slower* than bf16 despite halving the bytes
// (pass 1: 788 us at 170 GB/s against bf16's 675 us at 398 GB/s) -- the kernel
// stopped being bandwidth-bound and became conversion-bound.
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t two) {
  const __nv_fp8x2_storage_t raw = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&two);
  return __half22float2(__nv_cvt_fp8x2_to_halfraw2(raw, __NV_E4M3));
}

// Eight E4M3 bytes -> eight bf16, scaled, built in registers so the shared
// memory store stays one 16-byte transaction instead of eight 2-byte ones.
__device__ __forceinline__ uint4 e4m3x8_to_bf16x8_scaled(const uint2 packed,
                                                         float sc) {
  const uint16_t* pairs = reinterpret_cast<const uint16_t*>(&packed);
  __nv_bfloat162 o[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 f = e4m3x2_to_float2(pairs[e]);
    o[e].x = __float2bfloat16_rn(f.x * sc);
    o[e].y = __float2bfloat16_rn(f.y * sc);
  }
  return *reinterpret_cast<const uint4*>(o);
}

// Sixteen E4M3 bytes -> sixteen bf16, scaled. Loading 16 bytes per thread
// instead of 8 is not cosmetic: a warp's 8-byte-per-lane load moves 256 bytes
// and a 16-byte-per-lane one moves 512, so at a fixed wavefront rate the
// narrow load halves the bytes per instruction and cancels the byte saving
// that halving the dtype was supposed to buy.
__device__ __forceinline__ void e4m3x16_to_bf16x16_scaled(const uint4 packed,
                                                          float sc,
                                                          uint4* lo,
                                                          uint4* hi) {
  const uint16_t* pairs = reinterpret_cast<const uint16_t*>(&packed);
  __nv_bfloat162 o[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float2 f = e4m3x2_to_float2(pairs[e]);
    o[e].x = __float2bfloat16_rn(f.x * sc);
    o[e].y = __float2bfloat16_rn(f.y * sc);
  }
  *lo = *reinterpret_cast<const uint4*>(o);
  *hi = *reinterpret_cast<const uint4*>(o + 4);
}

__device__ __forceinline__ float rope_pair(float a, float ca, float b, float sa) {
  return __fadd_rn(__fmul_rn(a, ca), __fmul_rn(b, sa));
}

__device__ __forceinline__ float rope_pair_sub(float a, float ca, float b, float sa) {
  return __fsub_rn(__fmul_rn(a, ca), __fmul_rn(b, sa));
}

// ──────────── 2. QKV split + partial RoPE + KV cache write ────────────
//
// One block per (row, head). Threads sweep head_dim; only the first rope_dim
// elements rotate, and rotate_half pairs element j with j + rope_dim/2.
// pos of row r is pos_start + r.
__global__ void qkv_post_rope_kvwrite_kernel(
    const __nv_bfloat16* __restrict__ qkv,   // (rows, q_heads*hd + 2*kv_heads*hd)
    const float* __restrict__ cos_tab,       // (max_pos, rope_dim/2)
    const float* __restrict__ sin_tab,       // (max_pos, rope_dim/2)
    __nv_bfloat16* __restrict__ q_buf,       // (rows, q_heads*hd)
    __nv_bfloat16* __restrict__ k_cache,     // (slots, kv_heads*hd), linear
    __nv_bfloat16* __restrict__ v_cache,     // (slots, kv_heads*hd), linear
    __nv_bfloat16* __restrict__ k_ring,      // (2*ring_w, kv_heads*hd) or null
    __nv_bfloat16* __restrict__ v_ring,
    uint8_t* __restrict__ k8,                // (slots, kv_heads*hd) E4M3, or null
    uint8_t* __restrict__ v8,
    float* __restrict__ k8_scale,            // (slots, kv_heads)
    float* __restrict__ v8_scale,
    int rows, int q_heads, int kv_heads, int head_dim, int rope_dim,
    const int* pos_dev, int ring_w, int lin_w) {
  const int r = blockIdx.y;
  const int head = blockIdx.x;
  const int t = threadIdx.x;               // element within the head
  const int hd = head_dim;
  // The position is read from device memory, not passed as a launch argument:
  // that is what lets one captured CUDA Graph serve every decode step.
  const int pos = __ldg(pos_dev) + r;
  const int half = rope_dim >> 1;

  const bool is_q = head < q_heads;
  const int local_head = is_q ? head : head - q_heads;

  const __nv_bfloat16* src = qkv + (size_t)r * (q_heads * hd + 2 * kv_heads * hd)
                           + (is_q ? (size_t)local_head * hd
                                   : (size_t)q_heads * hd + (size_t)local_head * hd);
  // V lives after K
  const __nv_bfloat16* src_v = qkv + (size_t)r * (q_heads * hd + 2 * kv_heads * hd)
                             + (size_t)q_heads * hd + (size_t)kv_heads * hd
                             + (size_t)local_head * hd;

  const float* c = cos_tab + (size_t)pos * half;
  const float* s = sin_tab + (size_t)pos * half;

  if (is_q) {
    __nv_bfloat16* dst = q_buf + (size_t)r * (q_heads * hd) + (size_t)local_head * hd;
    float x = __bfloat162float(src[t]);
    float out;
    if (t < half) {
      float x2 = __bfloat162float(src[t + half]);
      out = rope_pair_sub(x, c[t], x2, s[t]);
    } else if (t < rope_dim) {
      int j = t - half;
      float x1 = __bfloat162float(src[j]);
      out = rope_pair(x, c[j], x1, s[j]);
    } else {
      out = x;
    }
    dst[t] = __float2bfloat16(out);
  } else {
    // K: rope, then write to every mirror slot this position maps to.
    float x = __bfloat162float(src[t]);
    float out;
    if (t < half) {
      float x2 = __bfloat162float(src[t + half]);
      out = rope_pair_sub(x, c[t], x2, s[t]);
    } else if (t < rope_dim) {
      int j = t - half;
      float x1 = __bfloat162float(src[j]);
      out = rope_pair(x, c[j], x1, s[j]);
    } else {
      out = x;
    }
    __nv_bfloat16 kbf = __float2bfloat16(out);
    __nv_bfloat16 vbf = src_v[t];
    // Linear store. `lin_w == 0` keeps the historical absolute indexing, which
    // only works when this layer's cache spans the whole context. A sliding
    // layer instead passes `lin_w == prefill_chunk + W`: every prefill window
    // [p-W+1, p+rows-1] is then at most `rows + W - 1 <= lin_w - 1` positions
    // long, so one `lin_w`-period of the ring holds it. The row is written
    // twice, at its slot and at its mirror, so that window is contiguous from
    // wherever it starts and prefill can hand FA2 one linear base pointer.
    {
      // Four independent stores, so a layer writes exactly what it is handed.
      // Sliding layers keep bf16 for both K and V. A full layer keeps bf16 plus
      // an E4M3 mirror when both fit, and E4M3 only (the long-window mode) when
      // they do not. These must stay siblings: nesting the E4M3 store inside a
      // bf16 guard silently wrote nothing for a layer that passes no bf16
      // target, which surfaces as a delivery cosine of 0.63 rather than as
      // anything resembling a precision change.
      const size_t sl = (lin_w > 0) ? (size_t)(pos % lin_w) : (size_t)pos;
      const size_t off = sl * (kv_heads * hd) + (size_t)local_head * hd + t;
      if (k_cache != nullptr) {
        k_cache[off] = kbf;
        if (lin_w > 0) k_cache[off + (size_t)lin_w * (kv_heads * hd)] = kbf;
      }
      if (v_cache != nullptr) {
        v_cache[off] = vbf;
        if (lin_w > 0) v_cache[off + (size_t)lin_w * (kv_heads * hd)] = vbf;
      }
      if (k8 != nullptr || v8 != nullptr) {
        // Row amax over this block's head, then E4M3 at that scale. The scale
        // is per (row, KV head), which is one fp32 per 512 bf16 of KV.
        float mk = fabsf(__bfloat162float(kbf));
        float mv = fabsf(__bfloat162float(vbf));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          mk = fmaxf(mk, __shfl_xor_sync(~0u, mk, o));
          mv = fmaxf(mv, __shfl_xor_sync(~0u, mv, o));
        }
        __shared__ float sk[8], sv[8];
        const int wid = t >> 5, lane = t & 31;
        if (lane == 0) { sk[wid] = mk; sv[wid] = mv; }
        __syncthreads();
        if (t == 0) {
          float a = 0.f, b = 0.f;
#pragma unroll
          for (int w = 0; w < (int)(head_dim / 32); ++w) {
            a = fmaxf(a, sk[w]);
            b = fmaxf(b, sv[w]);
          }
          const float ks = a > 0.f ? a / 448.f : 1.f;
          const float vs = b > 0.f ? b / 448.f : 1.f;
          const size_t ss = (size_t)sl * kv_heads + local_head;
          if (k8_scale != nullptr) k8_scale[ss] = ks;
          if (v8_scale != nullptr) v8_scale[ss] = vs;
          sk[0] = ks;
          sv[0] = vs;
        }
        __syncthreads();
        const float ks = sk[0], vs = sv[0];
        const size_t o8 = (size_t)sl * (kv_heads * hd) + (size_t)local_head * hd + t;
        if (k8 != nullptr) k8[o8] = float_to_e4m3(__bfloat162float(kbf) / ks);
        if (v8 != nullptr) v8[o8] = float_to_e4m3(__bfloat162float(vbf) / vs);
      }
    }
    // Ring store for decode: slot (pos mod W). Decode reads the whole W-slot
    // ring with a fixed pointer; because the W residues each hold their most
    // recent position, the ring holds exactly the last W positions -- and a
    // decode query has no causal mask over them, so their order in the ring
    // does not matter and attention over the set is the sliding window.
    if (k_ring != nullptr) {
      const int sl = pos % ring_w;
      size_t off = (size_t)sl * (kv_heads * hd) + (size_t)local_head * hd + t;
      k_ring[off] = kbf;
      v_ring[off] = vbf;
    }
  }
}

void qkv_post_rope_kvwrite_bf16(
    const void* qkv, const float* cos_tab, const float* sin_tab,
    void* q_buf, void* k_cache, void* v_cache,
    void* k_ring, void* v_ring,
    void* k8, void* v8, void* k8_scale, void* v8_scale,
    int rows, int q_heads, int kv_heads, int head_dim, int rope_dim,
    const int* pos_dev, int ring_w, int lin_w, cudaStream_t stream) {
  dim3 grid(q_heads + kv_heads, rows);
  qkv_post_rope_kvwrite_kernel<<<grid, head_dim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv), cos_tab, sin_tab,
      reinterpret_cast<__nv_bfloat16*>(q_buf),
      reinterpret_cast<__nv_bfloat16*>(k_cache),
      reinterpret_cast<__nv_bfloat16*>(v_cache),
      reinterpret_cast<__nv_bfloat16*>(k_ring),
      reinterpret_cast<__nv_bfloat16*>(v_ring),
      reinterpret_cast<uint8_t*>(k8), reinterpret_cast<uint8_t*>(v8),
      reinterpret_cast<float*>(k8_scale), reinterpret_cast<float*>(v8_scale),
      rows, q_heads, kv_heads, head_dim, rope_dim, pos_dev, ring_w, lin_w);
}

// ──────────── 5. seed a sliding ring from the linear cache ────────────
//
// A ring slot is written by every position congruent to it mod W, so writing a
// whole prefill into the ring would race: the block holding position p and the
// block holding p+W both target slot p%W and nothing orders them. Prefill
// therefore writes only the linear cache, and this kernel copies the surviving
// window in afterwards, one row per block, with no aliasing.
__global__ void seed_ring_kernel(
    const __nv_bfloat16* __restrict__ k_lin, const __nv_bfloat16* __restrict__ v_lin,
    __nv_bfloat16* __restrict__ k_ring, __nv_bfloat16* __restrict__ v_ring,
    int base_pos, int count, int kv_dim, int W, int lin_w) {
  const int r = blockIdx.x;
  if (r >= count) return;
  const int p = base_pos + r;
  const int sl = p % W;
  // `lin_w` mirrors the wrap the KV write used; 0 means absolute indexing.
  const size_t lr = (lin_w > 0) ? (size_t)(p % lin_w) : (size_t)p;
  for (int t = threadIdx.x; t < kv_dim; t += blockDim.x) {
    const __nv_bfloat16 k = k_lin[lr * kv_dim + t];
    const __nv_bfloat16 v = v_lin[lr * kv_dim + t];
    k_ring[(size_t)sl * kv_dim + t] = k;
    v_ring[(size_t)sl * kv_dim + t] = v;
  }
}

void seed_ring_bf16(const void* k_lin, const void* v_lin,
                    void* k_ring, void* v_ring,
                    int base_pos, int count, int kv_dim, int W, int lin_w,
                    cudaStream_t stream) {
  if (count <= 0) return;
  seed_ring_kernel<<<count, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(k_lin),
      reinterpret_cast<const __nv_bfloat16*>(v_lin),
      reinterpret_cast<__nv_bfloat16*>(k_ring),
      reinterpret_cast<__nv_bfloat16*>(v_ring),
      base_pos, count, kv_dim, W, lin_w);
}

// ──────────── 6. decode position / attention-length bookkeeping ────────────
//
// Advances the absolute position and derives the two attention lengths the
// decode step needs, on device, so the step itself has no position-dependent
// launch arguments and a single captured graph can replay at every step:
//   * full-attention layers attend over keys [0, pos]
//   * sliding layers attend over the mirrored ring read at full width once the
//     window is full (2W), and over the linear prefix before that.
__global__ void step_positions_kernel(int* pos, int* full_klen, int* slide_klen,
                                      int64_t* tokens_out, int64_t* token_in,
                                      int W, int first) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  // `first` is set on the graph's opening step, whose position the caller has
  // already armed; later steps advance it on device.
  if (!first) {
    *pos += 1;
    // The token consumed by this step was produced by the step at the previous
    // position, so it belongs at index (*pos - 1).
    if (tokens_out && token_in) tokens_out[*pos - 1] = *token_in;
  }
  const int p = *pos;
  *full_klen = p + 1;
  // Sliding layers read the W-slot ring. Before the window fills, the unwritten
  // slots must be excluded, so the length is the number of positions written.
  *slide_klen = (p + 1 <= W) ? (p + 1) : W;
}

void step_positions_bf16(int* pos, int* full_klen, int* slide_klen,
                         int64_t* tokens_out, int64_t* token_in,
                         int W, int first, cudaStream_t stream) {
  step_positions_kernel<<<1, 32, 0, stream>>>(
      pos, full_klen, slide_klen, tokens_out, token_in, W, first);
}

// ──────────── 4. attention output gate projection (g_proj) ────────────
//
// The gate is a (rows, 2560) x (2560, 16) matmul: tiny N, so the shape is
// latency-bound rather than bandwidth-bound. FlashRT's generic bf16 matmul
// covers it, but its inner loop carries `#pragma unroll 1`, which serialises the
// global loads and leaves the kernel waiting on memory with almost no work in
// flight; on this shape it costs ~29 us per layer, ~1 ms per decoded token.
//
// This kernel keeps the *exact* accumulation order of that reference -- lane L
// sums k = L, L+32, L+64, ... and the warp then reduces with shfl_xor 16..1 --
// so the results are bit-identical, but unrolls the loop so several loads are
// outstanding at once.
__global__ void gproj_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,    // (rows, K)
    const __nv_bfloat16* __restrict__ W,    // (g_dim, K)
    __nv_bfloat16* __restrict__ out,        // (rows, g_dim)
    int rows, int g_dim, int K) {
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int m = blockIdx.y;
  if (n >= g_dim || m >= rows) return;
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* x_row = x + (size_t)m * K;
  const __nv_bfloat16* w_row = W + (size_t)n * K;
  float acc = 0.0f;
  #pragma unroll 8
  for (int j = lane; j < K; j += 32) {
    acc = fmaf(__bfloat162float(x_row[j]), __bfloat162float(w_row[j]), acc);
  }
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffff, acc, off);
  }
  if (lane == 0) out[(size_t)m * g_dim + n] = __float2bfloat16(acc);
}

// Decode variant of the same projection. The single-row call reads 8 output
// rows of weights per block -- 82 KB per layer across the two blocks -- and the
// kernel above reaches them with 16 warps holding 8 outstanding two-byte loads
// each, which is far short of the memory-level parallelism an 82 KB cold read
// needs; the shape is the same one that cost full_n 20% against a plain read
// (see the shape probe). Here the block's 8 weight rows are staged in shared
// memory with 16-byte cooperative loads first, and each lane then keeps
// gproj_bf16's exact accumulation order (k = lane, lane+32, ... with fmaf, then
// a shuffle-xor 16..1 reduction) reading from shared memory, so the result is
// unchanged bit for bit. The normalized row is staged alongside because all 8
// warps read the same copy of it.
__global__ void gproj_bf16_smem_kernel(
    const __nv_bfloat16* __restrict__ x,    // (K,)
    const __nv_bfloat16* __restrict__ W,    // (g_dim, K)
    __nv_bfloat16* __restrict__ out,        // (g_dim,)
    int g_dim, int K) {
  extern __shared__ __nv_bfloat16 s_all[];
  __nv_bfloat16* s_x = s_all;                 // (K,)
  __nv_bfloat16* s_w = s_all + K;             // (8, K)

  const int n0 = blockIdx.x * 8;
  const int nvec = K / 8;                     // 16-byte vectors per row
  const int nrows = min(8, g_dim - n0);

  for (int v = threadIdx.x; v < nrows * nvec; v += (int)blockDim.x) {
    const int r = v / nvec, c = v - r * nvec;
    reinterpret_cast<uint4*>(s_w + (size_t)r * K)[c] =
        reinterpret_cast<const uint4*>(W + (size_t)(n0 + r) * K)[c];
  }
  for (int v = threadIdx.x; v < nvec; v += (int)blockDim.x) {
    reinterpret_cast<uint4*>(s_x)[v] = reinterpret_cast<const uint4*>(x)[v];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int n = n0 + (threadIdx.x >> 5);
  if (n >= g_dim) return;
  const __nv_bfloat16* w_row = s_w + (size_t)(n - n0) * K;
  float acc = 0.0f;
  #pragma unroll 8
  for (int j = lane; j < K; j += 32) {
    acc = fmaf(__bfloat162float(s_x[j]), __bfloat162float(w_row[j]), acc);
  }
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    acc += __shfl_xor_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) out[n] = __float2bfloat16(acc);
}

void gproj_bf16(const void* x, const void* W, void* out,
                int rows, int g_dim, int K, cudaStream_t stream) {
  if (rows == 1) {
    const size_t smem = (size_t)(K + 8 * K) * sizeof(__nv_bfloat16);
    static bool configured = false;
    if (!configured) {
      cudaFuncSetAttribute(gproj_bf16_smem_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem);
      configured = true;
    }
    gproj_bf16_smem_kernel<<<(g_dim + 7) / 8, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<__nv_bfloat16*>(out), g_dim, K);
    return;
  }
  dim3 grid((g_dim + 7) / 8, rows);
  gproj_bf16_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<const __nv_bfloat16*>(W),
      reinterpret_cast<__nv_bfloat16*>(out), rows, g_dim, K);
}

// ──────────── 3. attention output gate (per head) ────────────
// out[row, h, :] = attn[row, h, :] * bf16(sigmoid(float(gate[row, h])))
//
// The reference computes sigmoid in fp32, rounds the result to bf16, then
// multiplies in bf16; this reproduces that exactly.
__global__ void attn_out_gate_kernel(
    const __nv_bfloat16* __restrict__ attn,   // (rows, heads*hd)
    const __nv_bfloat16* __restrict__ gate,   // (rows, heads)
    __nv_bfloat16* __restrict__ out,          // (rows, heads*hd)
    int total, int heads, int head_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int per_row = heads * head_dim;
  const int row = i / per_row;
  const int head = (i - row * per_row) / head_dim;
  float s = 1.0f / (1.0f + expf(-__bfloat162float(gate[row * heads + head])));
  __nv_bfloat16 sbf = __float2bfloat16(s);
  out[i] = __float2bfloat16(__bfloat162float(attn[i]) * __bfloat162float(sbf));
}

void attn_out_gate_bf16(const void* attn, const void* gate, void* out,
                        int rows, int heads, int head_dim, cudaStream_t stream) {
  const int total = rows * heads * head_dim;
  attn_out_gate_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(attn),
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<__nv_bfloat16*>(out), total, heads, head_dim);
}


// ──────────── 3b. attention output gate (per head) -> NVFP4 ────────────
//
// The gated attention row is exactly out_proj's activation, so the NVFP4 pack
// is done here instead of by a second pass over the row: one launch and one
// read instead of two. The arithmetic is identical to attn_out_gate_kernel
// followed by quantize_bf16_to_nvfp4_swizzled.
__global__ void attn_out_gate_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ attn,   // (rows, heads*hd)
    const __nv_bfloat16* __restrict__ gate,   // (rows, heads)
    __nv_bfloat16* __restrict__ out,          // (rows, heads*hd), may be null
    uint8_t* __restrict__ packed,             // (rows, cols/2)
    uint8_t* __restrict__ sf_swz,
    int rows, int heads, int head_dim, int cols, int n_col_blocks,
    int chunk_cols) {
  const int row = blockIdx.y;
  const int col_begin = blockIdx.x * chunk_cols;
  const int per_row = heads * head_dim;
  uint8_t* packed_row = packed + (size_t)row * (cols / 2) + col_begin / 2;
  __nv_bfloat16* out_row = out ? out + (size_t)row * per_row + col_begin : nullptr;

  extern __shared__ __align__(16) uint8_t smem_raw[];
  const int chunk_blocks = chunk_cols >> 4;
  float* smem_scales = reinterpret_cast<float*>(smem_raw);
  __nv_bfloat16* smem_val =
      reinterpret_cast<__nv_bfloat16*>(smem_raw + chunk_blocks * sizeof(float));

  for (int b = threadIdx.x; b < chunk_blocks; b += blockDim.x) smem_scales[b] = 0.0f;
  __syncthreads();

  for (int i = threadIdx.x; i < chunk_cols; i += blockDim.x) {
    const int g = col_begin + i;
    const float a = __bfloat162float(attn[(size_t)row * per_row + g]);
    const float s = 1.0f / (1.0f + expf(-__bfloat162float(gate[row * heads + g / head_dim])));
    // match the reference: sigmoid in fp32, rounded to bf16, then multiplied
    const __nv_bfloat16 prod =
        __float2bfloat16(a * __bfloat162float(__float2bfloat16(s)));
    smem_val[i] = prod;
    if (out_row) out_row[i] = prod;
    atomicMax((int*)&smem_scales[i >> 4], __float_as_int(fabsf(__bfloat162float(prod))));
  }
  __syncthreads();

  const int rb = row / 128;
  const int ri = row % 128;
  const int blk0 = col_begin >> 4;
  for (int b = threadIdx.x; b < chunk_blocks; b += blockDim.x) {
    const float amax = __int_as_float(*(int*)&smem_scales[b]);
    const uint8_t ue = float_to_ue4m3_ceil(amax / 6.0f);
    const int bg = blk0 + b;
    sf_swz[(rb * n_col_blocks + bg / 4) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + (bg % 4)] = ue;
    smem_scales[b] = ue4m3_to_float(ue);
  }
  __syncthreads();

  const int half = chunk_cols >> 1;
  for (int p = threadIdx.x; p < half; p += blockDim.x) {
    const int i = p * 2;
    const float s0 = smem_scales[i >> 4];
    const float s1 = smem_scales[(i + 1) >> 4];
    const float v0 = __bfloat162float(smem_val[i]) * (s0 > 0.f ? 1.f / s0 : 0.f);
    const float v1 = __bfloat162float(smem_val[i + 1]) * (s1 > 0.f ? 1.f / s1 : 0.f);
    packed_row[p] = (uint8_t)((float_to_fp4_e2m1(v1) << 4) | (float_to_fp4_e2m1(v0) & 0x0F));
  }
}

void attn_out_gate_to_nvfp4_bf16(const void* attn, const void* gate, void* out,
                                 uint8_t* packed, uint8_t* sf_swz,
                                 int rows, int heads, int head_dim,
                                 cudaStream_t stream) {
  const int cols = heads * head_dim;
  const int num_blocks = cols / 16;
  const int n_col_blocks = (num_blocks + 3) / 4;
  // one block per row would leave the GPU idle on this shape; split the row
  int split = 1;
  while (split < 8 && (cols / (split * 2)) % 64 == 0) split *= 2;
  const int chunk = cols / split;
  const int chunk_blocks = chunk / 16;
  const size_t smem = (size_t)chunk_blocks * sizeof(float)
                    + (size_t)chunk * sizeof(__nv_bfloat16);
  dim3 grid(split, rows);
  attn_out_gate_to_nvfp4_kernel<<<grid, 256, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(attn),
      reinterpret_cast<const __nv_bfloat16*>(gate),
      reinterpret_cast<__nv_bfloat16*>(out), packed, sf_swz,
      rows, heads, head_dim, cols, n_col_blocks, chunk);
}

// ──────────── 9. scalar position write ────────────
//
// Prefill arms the device position once per call. Doing that with a framework
// fill would put one ATen op inside the equal-scope pipeline, so it is a kernel.
__global__ void set_int32_kernel(int* dst, int value) { *dst = value; }

void set_int32(int* dst, int value, cudaStream_t stream) {
  set_int32_kernel<<<1, 1, 0, stream>>>(dst, value);
}

// ──────────── 8. greedy argmax over the vocabulary ────────────
// The decode step's last op is the greedy sample: argmax over the 131072
// logits. FlashRT's qwen36_argmax_bf16 launches ONE block per row and scans
// with a scalar `col += blockDim.x` stride, so a warp has a single 2-byte load
// in flight at a time. Measured in-graph that costs 42 us/token to read 262 KB
// -- 6 GB/s, against 373 GB/s for a plain read of the same bytes. The scan is
// a pure reduction with no reuse, so it is entirely a memory-level-parallelism
// question, and the fix is to read wider and keep several loads outstanding:
// each thread below reads 16 bytes (8 bf16) per step behind a small unroll,
// which is the same trick gproj_bf16 uses for the same reason.
//
// Tie-break is the reference's: on equal logits the LOWEST index wins. Every
// combine step (in-thread, warp shuffle, cross-warp) applies that same rule,
// so the result is the global lowest-index maximum.
__global__ void argmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,   // (vocab,)
    int64_t* __restrict__ argmax_out,           // (1,)
    int vocab) {
  constexpr int VEC = 8;                        // bf16 per 16-byte load
  __shared__ float s_val[32];
  __shared__ int s_idx[32];

  const int tid = threadIdx.x;
  const int nvec = vocab / VEC;
  float best_v = -INFINITY;
  int best_i = 0;

  #pragma unroll 4
  for (int v = tid; v < nvec; v += blockDim.x) {
    const uint4 raw = reinterpret_cast<const uint4*>(logits)[v];
    const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
    const int base = v * VEC;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float f = __bfloat162float(e[j]);
      if (f > best_v || (f == best_v && base + j < best_i)) {
        best_v = f;
        best_i = base + j;
      }
    }
  }
  // Tail elements, if vocab is not a multiple of VEC.
  for (int i = nvec * VEC + tid; i < vocab; i += blockDim.x) {
    const float f = __bfloat162float(logits[i]);
    if (f > best_v || (f == best_v && i < best_i)) {
      best_v = f;
      best_i = i;
    }
  }

  // Warp reduction, then one cross-warp round through shared memory.
  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_v, off);
    const int oi = __shfl_xor_sync(0xffffffffu, best_i, off);
    if (ov > best_v || (ov == best_v && oi < best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) {
    s_val[warp] = best_v;
    s_idx[warp] = best_i;
  }
  __syncthreads();
  if (warp == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    float v = (lane < nwarps) ? s_val[lane] : -INFINITY;
    int i = (lane < nwarps) ? s_idx[lane] : 0;
    #pragma unroll
    for (int off = 16; off > 0; off /= 2) {
      const float ov = __shfl_xor_sync(0xffffffffu, v, off);
      const int oi = __shfl_xor_sync(0xffffffffu, i, off);
      if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
      }
    }
    if (lane == 0) {
      argmax_out[0] = static_cast<int64_t>(i);
    }
  }
}

void argmax_bf16(const void* logits, void* argmax_out, int vocab,
                 cudaStream_t stream) {
  // 1024 threads over 131072 logits is 16 vector loads each; fewer threads
  // would leave the last few cache lines serialised behind the tail loop.
  argmax_bf16_kernel<<<1, 1024, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits),
      reinterpret_cast<int64_t*>(argmax_out), vocab);
}


// ──────────── 9. residual add + RMSNorm + NVFP4 quantize (decode) ────────────
//
// The decode step's boundary op, and the one kernel it runs 73 times per token
// (two per layer plus the final norm). FlashRT's
// `residual_add_rms_norm_to_nvfp4_swizzled_bf16_v2` does the same arithmetic
// but launches ONE block of 256 threads for the single row, and inside it each
// thread walks the row in 16 scalar strided steps: 32 two-byte global loads,
// then a shared-memory round trip of the normalized row, then a second
// __syncthreads before the quantize pass. Measured in-graph that is 6.43
// us/call for ~20 KB moved.
//
// This is the same computation restructured around the one row:
//   * 16-byte vector loads, one per thread per array, so the row arrives in a
//     single memory round trip instead of sixteen strided ones;
//   * the normalized values stay in registers -- the 16-element scale block is
//     completed by shuffling the neighbouring lane's eight values, which is
//     always in the same warp, so there is no shared-memory staging of `normed`
//     and no second __syncthreads;
//   * the scale factors land in FlashRT's swizzled layout, byte for byte
//     (row 0 of a one-row call, so the swizzle reduces to (block/4)*512 +
//     block%4), and the packed nibbles use this file's own FP4/UE4M3 encoders,
//     which are transcriptions of FlashRT's.
//
// The FLOAT SUM over the row is associated differently from v2's (this sums
// each thread's own eight elements, v2 sums element i, i+256, i+512, ...), so
// `rms` can differ in the last fp32 ulp. tests/test_kernels.py compares the
// packed bytes and swizzled scales against the FlashRT kernel directly rather
// than assuming they match.
__global__ void residual_add_rms_norm_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ h_in,        // (cols,)
    const __nv_bfloat16* __restrict__ attn_proj,   // (cols,)
    __nv_bfloat16* __restrict__ h_post,            // (cols,)
    const __nv_bfloat16* __restrict__ rms_weight,  // (cols,)
    uint8_t* __restrict__ packed,                  // (cols/2,)
    uint8_t* __restrict__ sf_swz,
    int cols, int n_col_blocks, float eps) {
  constexpr int VEC = 8;                 // bf16 per 16-byte load
  __shared__ float s_red[32];
  float s_ssq;

  const int tid = threadIdx.x;
  const uint4 ra = reinterpret_cast<const uint4*>(h_in)[tid];
  const uint4 rb = reinterpret_cast<const uint4*>(attn_proj)[tid];
  const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(&ra);
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&rb);

  // h_post = bf16(h_in + attn_proj), exactly as the reference rounds it, and
  // the same rounded values feed the sum of squares.
  uint4 out;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&out);
  float rbf[VEC];
  float sq = 0.0f;
  #pragma unroll
  for (int j = 0; j < VEC; ++j) {
    const __nv_bfloat16 s = __float2bfloat16(__bfloat162float(a[j]) + __bfloat162float(b[j]));
    o[j] = s;
    rbf[j] = __bfloat162float(s);
    sq = fmaf(rbf[j], rbf[j], sq);
  }
  reinterpret_cast<uint4*>(h_post)[tid] = out;

  #pragma unroll
  for (int off = 16; off > 0; off /= 2) {
    sq += __shfl_xor_sync(0xffffffffu, sq, off);
  }
  if ((tid & 31) == 0) s_red[tid >> 5] = sq;
  __syncthreads();
  // blockDim.x is a multiple of 32, and this kernel is launched with exactly
  // cols/VEC threads, so the number of warps is known from blockDim at run time.
  if (tid < 32) {
    const int nwarps = (int)(blockDim.x >> 5);
    float v = (tid < nwarps) ? s_red[tid] : 0.0f;
    #pragma unroll
    for (int off = 8; off > 0; off /= 2) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (tid == 0) s_red[0] = v;
  }
  __syncthreads();
  const float rms = rsqrtf(s_red[0] / (float)cols + eps);

  // Normalize into registers; the scale factor weights come in as one more
  // vector load.
  const uint4 rw = reinterpret_cast<const uint4*>(rms_weight)[tid];
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&rw);
  __nv_bfloat16 nrm[VEC];
  #pragma unroll
  for (int j = 0; j < VEC; ++j) {
    nrm[j] = __float2bfloat16(rbf[j] * rms * __bfloat162float(w[j]));
  }

  // One 16-element scale block is two lanes' worth. Lane t handles elements
  // [8t, 8t+8) and lane t|1 the other half; they are always adjacent in the
  // same warp, so the even lane collects the pair with one shuffle per packed
  // word and does the block's amax, scale and pack on its own. The shuffles
  // are executed by every lane -- a __shfl_*_sync with a full mask must be
  // reached by all 32 -- and only the even lane consumes the result.
  uint32_t mine[4], theirs[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    mine[j] = *reinterpret_cast<uint32_t*>(&nrm[2 * j]);
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    theirs[j] = __shfl_down_sync(0xffffffffu, mine[j], 1);
  }
  if ((tid & 1) == 0) {
    const __nv_bfloat16* tv = reinterpret_cast<const __nv_bfloat16*>(theirs);
    float vals[16];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      vals[2 * j] = __bfloat162float(nrm[2 * j]);
      vals[2 * j + 1] = __bfloat162float(nrm[2 * j + 1]);
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) vals[8 + j] = __bfloat162float(tv[j]);

    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(vals[j]));
    const uint8_t ue = float_to_ue4m3_ceil(amax / 6.0f);
    const int blk = tid >> 1;
    sf_swz[(blk >> 2) * 512 + (blk & 3)] = ue;

    const float fscale = ue4m3_to_float(ue);
    const float inv = (fscale > 0.0f) ? (1.0f / fscale) : 0.0f;
    uint8_t bytes[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      bytes[j] = (uint8_t)((float_to_fp4_e2m1(vals[2 * j + 1] * inv) << 4)
                           | (float_to_fp4_e2m1(vals[2 * j] * inv) & 0x0F));
    }
    __builtin_memcpy(packed + blk * 8, bytes, 8);
  }
}

void residual_add_rms_norm_to_nvfp4_bf16(
    const void* h_in, const void* attn_proj, void* h_post,
    const void* rms_weight, void* packed, void* sf_swz,
    int cols, float eps, cudaStream_t stream) {
  // One thread per 16-byte vector: the row is 2560 bf16 = 320 vectors, and
  // blockDim.x is both the vector count and the thread count, so the
  // cross-warp reduction knows its own width.
  const int nvec = cols / 8;
  residual_add_rms_norm_to_nvfp4_kernel<<<1, nvec, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(h_in),
      reinterpret_cast<const __nv_bfloat16*>(attn_proj),
      reinterpret_cast<__nv_bfloat16*>(h_post),
      reinterpret_cast<const __nv_bfloat16*>(rms_weight),
      reinterpret_cast<uint8_t*>(packed),
      reinterpret_cast<uint8_t*>(sf_swz),
      cols, (cols / 16 + 3) / 4, eps);
}


// ──────────── 10. Native decode attention ────────────
//
// Why this exists at all. The FA2 decode entry launches one block per
// (query block, KV head, batch). A decode step is a single query row, so that
// grid is 1 x 4 x 1 -- four blocks on a 36-SM part, 32 SMs idle. The entry's
// `num_sms` split recovers some of it but saturates early: swept over its whole
// configuration space at the 131k production shape it reaches 154 GB/s at 64
// blocks and 168 GB/s at 384, against this part's measured 425.8 GB/s streaming
// ceiling. The shape is not the obstacle -- reading the same KV tensor with the
// same (slots, kv_heads, head_dim) layout through a plain GEMM runs at 404 GB/s
// (scripts/splitkv_sweep.py, scripts/attn_pass1_probe.py). The kernel is.
//
// The replacement is two-pass. Pass 1 materialises the whole score row
// S[head][key]; pass 2 is then a plain weighted sum of V rows, with no online
// softmax and no per-split maxima to carry through the accumulation. The cost
// is one extra S round trip -- 4 bytes per head per key, 7% of the KV bytes --
// and what it buys is that every stage becomes a straight streaming reduction
// of the shape that measured 404 and 393 GB/s.
//
// Device-side lengths: like the FA2 call it replaces, every kernel here takes
// the *capacity* as a host argument and reads the live key count from device
// memory, so one captured CUDA Graph serves every decode step.
//
//   state_init -> scores -> softmax -> pv -> pv_combine
//
// `scores` also folds its tile maximum into a device atomicMax, so no separate
// pass over S is needed to find the softmax shift.

#define ATTN_BN 32                       // keys per scores block
#define ATTN_VEC 8                       // bf16 per 16-byte load
#define ATTN_VEC8 16                     // E4M3 per 16-byte load (same width)
#define ATTN_PAD 8                       // smem row pad, in elements
#define ATTN_PV_THREADS 128              // 4 KV heads x 32 dim-octets

__device__ __forceinline__ float bf16x8_dot(const uint4 a, const uint4 b) {
  const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
  const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
  // Two independent chains, so the eight FMAs are not one dependency chain.
  // bf16 -> fp32 is exact, so this accumulates the dot product in full fp32
  // precision -- the same arithmetic the reference does.
  float s0 = 0.f, s1 = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 fa = __bfloat1622float2(pa[i]);
    const float2 fb = __bfloat1622float2(pb[i]);
    s0 = __fmaf_rn(fa.x, fb.x, s0);
    s1 = __fmaf_rn(fa.y, fb.y, s1);
  }
  return s0 + s1;
}

__device__ __forceinline__ void atomic_max_float(float* addr, float value) {
  // Sign-magnitude max: non-negative floats order as signed ints, negative
  // floats order inverted as unsigned, and the two branches never disagree
  // because every non-negative pattern sorts above every negative one.
  if (value >= 0.f)
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
  else
    atomicMin(reinterpret_cast<unsigned*>(addr), __float_as_uint(value));
}

// One block owns one GQA group (4 query heads) and ATTN_BN keys of that group's
// KV head, so the K tile is read once and used 4 times. The shared-memory row
// pitch is padded by ATTN_PAD elements: without it every lane of a warp reads a
// different 512-byte-apart row at the same column, a 32-way bank conflict.
template <bool KV8>
__global__ void attn_scores_kernel(
    const __nv_bfloat16* __restrict__ q,        // (q_heads, head_dim)
    const void* __restrict__ k_cache,           // bf16, or E4M3 when KV8
    const float* __restrict__ k8_scale,         // (slots, kv_heads), KV8 only
    float* __restrict__ s_out,                  // (q_heads, s_stride)
    float* __restrict__ row_max,                // (q_heads)
    const int* __restrict__ klen_dev,
    int s_stride, int q_heads, int kv_heads, int head_dim,
    int group, float scale) {
  const int klen = __ldg(klen_dev);
  const int hd = head_dim;
  const int kv_head = blockIdx.y;
  const int key0 = blockIdx.x * ATTN_BN;
  if (key0 >= klen) return;
  const int qb = kv_head * group;

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* ksm = smem;                              // (BN, hd + PAD)
  __nv_bfloat16* qsm = smem + ATTN_BN * (hd + ATTN_PAD);  // (group, hd)

  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  for (int i = tid; i < group * hd; i += nt) qsm[i] = q[qb * hd + i];

  const int kv_stride = kv_heads * hd;
  if (KV8) {
    // 16 E4M3 bytes in, 16 bf16 out: the tile is dequantized on its way into
    // shared memory, so the dot product below is unchanged and the bytes
    // streamed from DRAM are half.
    //
    // Each thread takes a contiguous run of chunks within ONE row, so the
    // per-row scale is loaded once instead of once per chunk. That matters
    // more than it looks: the bf16 path issues one load per 16 useful bytes,
    // while this path was issuing one K load AND one 4-byte scale load per 16
    // useful bytes -- the same instruction count for half the bytes, which is
    // why E4M3 reaches 315 GB/s where bf16 reaches 405 on identical code.
    const int vecs = hd / ATTN_VEC8;              // 16
    const int tpr = nt / ATTN_BN;                 // threads per row
    const int per = vecs / tpr;                   // chunks per thread
    const int r = tid / tpr;
    const int v0 = (tid - r * tpr) * per;
    {
      const int key = min(key0 + r, klen - 1);
      const float sc = __ldg(k8_scale + (size_t)key * kv_heads + kv_head);
      const size_t base = (size_t)key * kv_stride + (size_t)kv_head * hd;
      __nv_bfloat16* dst = ksm + r * (hd + ATTN_PAD) + v0 * ATTN_VEC8;
#pragma unroll
      for (int c = 0; c < per; ++c) {
        const uint4 packed = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const uint8_t*>(k_cache) + base
            + (size_t)(v0 + c) * ATTN_VEC8);
        uint4 lo, hi;
        e4m3x16_to_bf16x16_scaled(packed, sc, &lo, &hi);
        reinterpret_cast<uint4*>(dst)[2 * c] = lo;
        reinterpret_cast<uint4*>(dst)[2 * c + 1] = hi;
      }
    }
  } else {
    const int vecs = hd / ATTN_VEC;
    for (int i = tid; i < ATTN_BN * vecs; i += nt) {
      const int r = i / vecs, v = i - r * vecs;
      const int key = min(key0 + r, klen - 1);
      const size_t at = (size_t)key * kv_stride + (size_t)kv_head * hd
                      + (size_t)v * ATTN_VEC;
      *reinterpret_cast<uint4*>(ksm + r * (hd + ATTN_PAD) + v * ATTN_VEC) =
          *reinterpret_cast<const uint4*>(
              reinterpret_cast<const __nv_bfloat16*>(k_cache) + at);
    }
  }
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  if (warp >= group) return;
  // ATTN_BN / 32 keys per lane, so the same code serves a 64-key and a 32-key
  // tile. The q vector is loaded once per lane per k-step and shared across all
  // of them; that ratio of FMAs to shared-memory loads is what keeps the loop
  // fed once the tile is small enough not to cap occupancy.
  constexpr int KPL = ATTN_BN / 32;
  const __nv_bfloat16* qrow = qsm + warp * hd;

  float acc[KPL];
#pragma unroll
  for (int k = 0; k < KPL; ++k) acc[k] = 0.f;
  for (int d = 0; d < hd; d += ATTN_VEC) {
    const uint4 qv = *reinterpret_cast<const uint4*>(qrow + d);
#pragma unroll
    for (int k = 0; k < KPL; ++k) {
      const __nv_bfloat16* kp = ksm + (lane + k * 32) * (hd + ATTN_PAD) + d;
      acc[k] += bf16x8_dot(qv, *reinterpret_cast<const uint4*>(kp));
    }
  }

  float* sout = s_out + (size_t)(qb + warp) * s_stride;
  float tmax = -INFINITY;
#pragma unroll
  for (int k = 0; k < KPL; ++k) {
    const int j = key0 + lane + k * 32;
    const float a = acc[k] * scale;
    if (j < klen) { sout[j] = a; tmax = fmaxf(tmax, a); }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(~0u, tmax, o));
  if (lane == 0) atomic_max_float(row_max + qb + warp, tmax);
}

// ── E4M3 KV -> bf16 staging, for prefill in the long-window mode ────────
//
// At 262k a bf16 KV cache and an E4M3 mirror of it do not both fit (measured:
// 14.45 GiB used of 15.47, mirror needs 4.77 more), so above the budget a full
// layer keeps E4M3 only. FA2's prefill still wants bf16, so one layer's prefix
// is expanded into a single reusable staging buffer at a time. What FA2 then
// sees is the quantised cache expanded -- which is the point, and also the
// cost: it is why the long-window mode's long-context logit cosine is ~0.99
// rather than ~0.999.
//
// One warp per two rows, each lane taking a 16-byte chunk, the per-(row, KV
// head) scale loaded once per head per row.
__global__ void kv_dequant_kernel(
    const uint8_t* __restrict__ src, const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ dst, int count, int kv_heads, int head_dim) {
  const int stride = kv_heads * head_dim;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int row = warp * 2 + (lane >> 4);
  if (row >= count) return;
  const int v = lane & 15;
  for (int h = 0; h < kv_heads; ++h) {
    const float sc = __ldg(scale + (size_t)row * kv_heads + h);
    const size_t at = (size_t)row * stride + (size_t)h * head_dim + (size_t)v * 16;
    const uint4 packed = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint8_t*>(src) + at);
    uint4 lo, hi;
    e4m3x16_to_bf16x16_scaled(packed, sc, &lo, &hi);
    __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(dst) + at;
    reinterpret_cast<uint4*>(d)[0] = lo;
    reinterpret_cast<uint4*>(d)[1] = hi;
  }
}

void kv_dequant_bf16(const void* src, const void* scale, void* dst, int count,
                     int kv_heads, int head_dim, cudaStream_t stream) {
  const int rows_per_block = (256 / 32) * 2;
  kv_dequant_kernel<<<(count + rows_per_block - 1) / rows_per_block, 256, 0,
                      stream>>>(
      reinterpret_cast<const uint8_t*>(src),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<__nv_bfloat16*>(dst), count, kv_heads, head_dim);
}

// exp(S - m) into P, with the row sums accumulated for the final normalisation.
// Split over the key range so the pass has enough blocks to fill the part.
__global__ void attn_softmax_kernel(
    const float* __restrict__ s_in, float* __restrict__ p_out,
    const float* __restrict__ row_max, float* __restrict__ row_sum,
    const int* __restrict__ klen_dev,
    int stride, int q_heads, int nchunk) {
  const int klen = __ldg(klen_dev);
  const int head = blockIdx.x;
  const int per = (klen + nchunk - 1) / nchunk;
  const int lo = blockIdx.y * per;
  const int hi = min(klen, lo + per);
  if (lo >= hi) return;
  const float m = __ldg(row_max + head);
  const float* srow = s_in + (size_t)head * stride;
  float* prow = p_out + (size_t)head * stride;

  float acc = 0.f;
  for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
    const float p = expf(srow[j] - m);
    prow[j] = p;
    acc += p;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(row_sum + head, acc);
}

// O_partial[split][head][dim] = sum_j P[head][j] * V[j][kv(head)][dim].
//
// A block owns all kv_heads at once, so the V rows it streams are the full
// 2048-byte row rather than a 512-byte quarter of it. That matters: the
// quarter-row pattern is what caps the naive strided read at ~286 GB/s, and
// full-row streaming is what measured 393 GB/s.
//
// Thread (h, dg) covers KV head h and its 8 dims at 8*dg, for all four query
// heads of that group -- so each V element is loaded exactly once and each P
// value is a warp broadcast.
template <bool KV8>
__global__ void attn_pv_kernel(
    const void* __restrict__ v_cache,           // bf16, or E4M3 when KV8
    const float* __restrict__ v8_scale,         // (slots, kv_heads), KV8 only
    const float* __restrict__ p_in,             // (q_heads, stride)
    float* __restrict__ o_part,                 // (splits, q_heads, head_dim)
    const int* __restrict__ klen_dev,
    int stride, int q_heads, int kv_heads, int head_dim,
    int group, int nsplit) {
  const int klen = __ldg(klen_dev);
  const int per = (klen + nsplit - 1) / nsplit;
  const int lo = blockIdx.x * per;
  const int hi = min(klen, lo + per);
  if (lo >= hi) return;

  const int h = threadIdx.x >> 5;            // KV head
  const int dg = threadIdx.x & 31;           // dim octet
  const int kv_stride = kv_heads * head_dim;
  const int qb = h * group;
  const int d0 = dg * ATTN_VEC;

  float acc[4][ATTN_VEC];
#pragma unroll
  for (int l = 0; l < 4; ++l)
#pragma unroll
    for (int i = 0; i < ATTN_VEC; ++i) acc[l][i] = 0.f;

  const size_t vbase_el = (size_t)h * head_dim + d0;
  for (int j = lo; j < hi; ++j) {
    float vf[ATTN_VEC];
    if (KV8) {
      const uint2 packed = *reinterpret_cast<const uint2*>(
          reinterpret_cast<const uint8_t*>(v_cache)
          + (size_t)j * kv_stride + vbase_el);
      const float sc = __ldg(v8_scale + (size_t)j * kv_heads + h);
      const uint4 bf = e4m3x8_to_bf16x8_scaled(packed, sc);
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&bf);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(b2[e]);
        vf[2 * e] = f.x;
        vf[2 * e + 1] = f.y;
      }
    } else {
      const uint4 vv = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const __nv_bfloat16*>(v_cache)
          + (size_t)j * kv_stride + vbase_el);
      const __nv_bfloat162* pv = reinterpret_cast<const __nv_bfloat162*>(&vv);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pv[i]);
        vf[2 * i] = f.x;
        vf[2 * i + 1] = f.y;
      }
    }
    float p[4];
#pragma unroll
    for (int l = 0; l < 4; ++l) p[l] = __ldg(p_in + (size_t)(qb + l) * stride + j);
#pragma unroll
    for (int l = 0; l < 4; ++l)
#pragma unroll
      for (int i = 0; i < ATTN_VEC; ++i) acc[l][i] = __fmaf_rn(p[l], vf[i], acc[l][i]);
  }

  float* out = o_part + (size_t)blockIdx.x * q_heads * head_dim;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    float* row = out + (size_t)(qb + l) * head_dim + d0;
    *reinterpret_cast<float4*>(row) = make_float4(acc[l][0], acc[l][1], acc[l][2], acc[l][3]);
    *reinterpret_cast<float4*>(row + 4) = make_float4(acc[l][4], acc[l][5], acc[l][6], acc[l][7]);
  }
}

// The E4M3 variant of pass 2. Same result as `attn_pv_kernel<false>`, different
// mapping: 64 threads, each owning 16 dims of one KV head for all four query
// heads of its group. That is what makes the V load 16 bytes wide instead of 8
// -- a warp's 16-byte-per-lane load covers 512 contiguous bytes of the row,
// which is the same wavefront efficiency the bf16 path gets, so the halved
// dtype finally shows up in the time.
#define ATTN_PV8_THREADS 64
#define ATTN_PV8_DIMS 16

__global__ void attn_pv8_kernel(
    const uint8_t* __restrict__ v8, const float* __restrict__ v8_scale,
    const float* __restrict__ p_in, float* __restrict__ o_part,
    const int* __restrict__ klen_dev,
    int stride, int q_heads, int kv_heads, int head_dim,
    int group, int nsplit) {
  const int klen = __ldg(klen_dev);
  const int per = (klen + nsplit - 1) / nsplit;
  const int lo = blockIdx.x * per;
  const int hi = min(klen, lo + per);
  if (lo >= hi) return;

  const int h = threadIdx.x >> 4;         // KV head
  const int dg = threadIdx.x & 15;        // dim group of 16
  const int kv_stride = kv_heads * head_dim;
  const int qb = h * group;
  const int d0 = dg * ATTN_PV8_DIMS;

  float acc[4][ATTN_PV8_DIMS];
#pragma unroll
  for (int l = 0; l < 4; ++l)
#pragma unroll
    for (int i = 0; i < ATTN_PV8_DIMS; ++i) acc[l][i] = 0.f;

  const uint8_t* vbase = v8 + (size_t)h * head_dim + d0;
  for (int j = lo; j < hi; ++j) {
    const uint4 packed = *reinterpret_cast<const uint4*>(
        vbase + (size_t)j * kv_stride);
    const float sc = __ldg(v8_scale + (size_t)j * kv_heads + h);
    const uint16_t* pr = reinterpret_cast<const uint16_t*>(&packed);
    float vf[ATTN_PV8_DIMS];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float2 f = e4m3x2_to_float2(pr[e]);
      vf[2 * e] = f.x * sc;
      vf[2 * e + 1] = f.y * sc;
    }
    float p[4];
#pragma unroll
    for (int l = 0; l < 4; ++l)
      p[l] = __ldg(p_in + (size_t)(qb + l) * stride + j);
#pragma unroll
    for (int l = 0; l < 4; ++l)
#pragma unroll
      for (int i = 0; i < ATTN_PV8_DIMS; ++i)
        acc[l][i] = __fmaf_rn(p[l], vf[i], acc[l][i]);
  }

  float* out = o_part + (size_t)blockIdx.x * q_heads * head_dim;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    float* row = out + (size_t)(qb + l) * head_dim + d0;
#pragma unroll
    for (int q = 0; q < ATTN_PV8_DIMS / 4; ++q)
      reinterpret_cast<float4*>(row)[q] = make_float4(
          acc[l][4 * q], acc[l][4 * q + 1], acc[l][4 * q + 2], acc[l][4 * q + 3]);
  }
}

// Sum the per-split partials and normalise by the softmax denominator.
__global__ void attn_pv_combine_kernel(
    const float* __restrict__ o_part, const float* __restrict__ row_sum,
    __nv_bfloat16* __restrict__ o_out, int nsplit, int q_heads, int head_dim) {
  const int head = blockIdx.x;
  const int d0 = blockIdx.y * blockDim.x + threadIdx.x;
  if (d0 >= head_dim) return;
  const float inv = 1.0f / __ldg(row_sum + head);
  float acc = 0.f;
  for (int s = 0; s < nsplit; ++s)
    acc += o_part[((size_t)s * q_heads + head) * head_dim + d0];
  o_out[(size_t)head * head_dim + d0] = __float2bfloat16(acc * inv);
}

__global__ void attn_state_init_kernel(float* row_max, float* row_sum, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { row_max[i] = -INFINITY; row_sum[i] = 0.f; }
}

void attn_state_init_bf16(void* row_max, void* row_sum, int q_heads,
                          cudaStream_t stream) {
  attn_state_init_kernel<<<(q_heads + 127) / 128, 128, 0, stream>>>(
      reinterpret_cast<float*>(row_max), reinterpret_cast<float*>(row_sum),
      q_heads);
}

void attn_scores_bf16(const void* q, const void* k_cache, const void* k8_scale,
                      void* s_out, void* row_max, const int* klen_dev, int cap,
                      int s_stride, int q_heads, int kv_heads, int head_dim,
                      int group, float scale, int kv8, cudaStream_t stream) {
  dim3 grid((cap + ATTN_BN - 1) / ATTN_BN, kv_heads);
  const int smem = (ATTN_BN * (head_dim + ATTN_PAD) + group * head_dim)
                 * (int)sizeof(__nv_bfloat16);
  const __nv_bfloat16* qq = reinterpret_cast<const __nv_bfloat16*>(q);
  float* so = reinterpret_cast<float*>(s_out);
  float* rm = reinterpret_cast<float*>(row_max);
  const float* ks = reinterpret_cast<const float*>(k8_scale);
  if (kv8)
    attn_scores_kernel<true><<<grid, group * 32, smem, stream>>>(
        qq, k_cache, ks, so, rm, klen_dev, s_stride, q_heads, kv_heads,
        head_dim, group, scale);
  else
    attn_scores_kernel<false><<<grid, group * 32, smem, stream>>>(
        qq, k_cache, ks, so, rm, klen_dev, s_stride, q_heads, kv_heads,
        head_dim, group, scale);
}

void attn_softmax_bf16(const void* s_in, void* p_out, const void* row_max,
                       void* row_sum, const int* klen_dev, int cap, int stride,
                       int q_heads, int nchunk, cudaStream_t stream) {
  attn_softmax_kernel<<<dim3(q_heads, nchunk), 256, 0, stream>>>(
      reinterpret_cast<const float*>(s_in), reinterpret_cast<float*>(p_out),
      reinterpret_cast<const float*>(row_max), reinterpret_cast<float*>(row_sum),
      klen_dev, stride, q_heads, nchunk);
}

void attn_pv_bf16(const void* v_cache, const void* v8_scale, const void* p_in,
                  void* o_part, const int* klen_dev, int stride, int q_heads,
                  int kv_heads, int head_dim, int group, int nsplit, int kv8,
                  cudaStream_t stream) {
  const float* pi = reinterpret_cast<const float*>(p_in);
  float* op = reinterpret_cast<float*>(o_part);
  const float* vs = reinterpret_cast<const float*>(v8_scale);
  if (kv8)
    attn_pv8_kernel<<<nsplit, ATTN_PV8_THREADS, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(v_cache), vs, pi, op, klen_dev,
        stride, q_heads, kv_heads, head_dim, group, nsplit);
  else
    attn_pv_kernel<false><<<nsplit, ATTN_PV_THREADS, 0, stream>>>(
        v_cache, vs, pi, op, klen_dev, stride, q_heads, kv_heads, head_dim,
        group, nsplit);
}

void attn_pv_combine_bf16(const void* o_part, const void* row_sum, void* o_out,
                          int nsplit, int q_heads, int head_dim,
                          cudaStream_t stream) {
  attn_pv_combine_kernel<<<dim3(q_heads, (head_dim + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const float*>(o_part),
      reinterpret_cast<const float*>(row_sum),
      reinterpret_cast<__nv_bfloat16*>(o_out), nsplit, q_heads, head_dim);
}

}  // namespace flash_rt::spark_x25

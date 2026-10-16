// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — DM05 action-expert weight-only int8 GEMM (W8A16) for M <= 16.
//
//   out(M, N) = bf16( (x @ Wq^T)(M, N) * s(N) )
//
// computed transposed as out^T = Wq (N x K) . x^T (K x M) on mma.m16n8k16 with
// bf16 operands and fp32 accumulate: the weight tile is the A operand (16 output
// channels x 16 k), x^T the B operand (16 k x 8 activation rows, two tiles cover
// M <= 16). int8 -> bf16 is exact for |q| <= 127, so the tensor-core product
// equals the scalar int8 x bf16 product and the per-channel scale is applied
// once after the K reduction.
//
// This is W8A16, weight-only: the activations stay bf16. FlashRT's other int8
// path is W8A8 (both operands quantized) and is not built for GPU_ARCH=120, so
// neither it nor the sm87 qwen3_vl M=1 GEMV is a substitute.
//
// The weights are repacked offline into fragment order (see w8a16_pack below):
// for each (16-row block, pair of 16-wide k blocks) the 32 lanes' fragment bytes
// are contiguous, so a lane fetches both k blocks' A fragments with one 16-byte
// load. Blocks split K across warps and reduce the 16 x 16 fp32 partial tile in
// shared memory -- the reduction order a caller calibrated against, so it is part
// of the contract rather than an implementation detail.
//
// See dm05_w8a16.cuh for the interface.

#include "dm05_w8a16.cuh"

#include "dm05_common.cuh"

#include <stdexcept>

namespace flash_rt {
namespace dm05 {
namespace {

// round to bf16 and back, the intermediate rounding the epilogues reproduce
__device__ __forceinline__ float w8_bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

// gelu_tanh in fp32, the form the reference activation uses
__device__ __forceinline__ float w8_gelu_tanh(float x) {
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ uint32_t pack_bf16x2_from_i8(int8_t lo, int8_t hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn((float)lo, (float)hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__global__ void w8a16_pack_kernel(const int8_t* __restrict__ w, int8_t* __restrict__ out, int N, int K) {
  const long long total = (long long)N * K;
  for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x; o < total;
       o += (long long)gridDim.x * blockDim.x) {
    const int byte = (int)(o % 16);
    const int lane = (int)((o / 16) % 32);
    const long long kp = (o / 512) % (K / 32);
    const long long nb = o / (512LL * (K / 32));
    const int kb = (int)(kp * 2 + byte / 8);
    const int e = byte % 8;
    const int g = lane / 4, t = lane % 4;
    const int row = g + ((e & 2) ? 8 : 0);
    const int col = 2 * t + (e & 1) + ((e & 4) ? 8 : 0);
    out[o] = w[(nb * 16 + row) * (long long)K + kb * 16 + col];
  }
}


struct OutSegs { __nv_bfloat16* out[3]; int n0[4]; int ld[3]; int count; };

template <int WARPS, int MODE>
__global__ void __launch_bounds__(WARPS * 32)
w8a16_tc_kernel(const __nv_bfloat16* __restrict__ x, const int8_t* __restrict__ wp,
                const float* __restrict__ ws, OutSegs segs,
                int M, int N, int K) {
  __shared__ float red[WARPS][16][16];
  const int nb = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int kpairs = K / 32;
  float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
  const int4* wv = reinterpret_cast<const int4*>(wp) + (long long)nb * kpairs * 32;
  const bool r0 = g < M, r1 = (g + 8) < M;
  const __nv_bfloat16* x0 = x + (long long)g * K;
  const __nv_bfloat16* x1 = x + (long long)(g + 8) * K;
  for (int kp = warp; kp < kpairs; kp += WARPS) {
    const int4 q = __ldg(wv + (long long)kp * 32 + lane);
    const int8_t* qb = reinterpret_cast<const int8_t*>(&q);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int8_t* f = qb + 8 * h;
      uint32_t a[4] = {pack_bf16x2_from_i8(f[0], f[1]), pack_bf16x2_from_i8(f[2], f[3]),
                       pack_bf16x2_from_i8(f[4], f[5]), pack_bf16x2_from_i8(f[6], f[7])};
      const int k0 = (kp * 2 + h) * 16 + 2 * t;
      const uint32_t b00 = r0 ? __ldg(reinterpret_cast<const uint32_t*>(x0 + k0)) : 0u;
      const uint32_t b01 = r0 ? __ldg(reinterpret_cast<const uint32_t*>(x0 + k0 + 8)) : 0u;
      const uint32_t b10 = r1 ? __ldg(reinterpret_cast<const uint32_t*>(x1 + k0)) : 0u;
      const uint32_t b11 = r1 ? __ldg(reinterpret_cast<const uint32_t*>(x1 + k0 + 8)) : 0u;
      dm05_mma_bf16_16816(c0, a, b00, b01);
      dm05_mma_bf16_16816(c1, a, b10, b11);
    }
  }
  // C fragment: (row n = g | g+8, col m = 2t | 2t+1) for tile 0; col + 8 for tile 1
  red[warp][g][2 * t] = c0[0];      red[warp][g][2 * t + 1] = c0[1];
  red[warp][g + 8][2 * t] = c0[2];  red[warp][g + 8][2 * t + 1] = c0[3];
  red[warp][g][8 + 2 * t] = c1[0];  red[warp][g][8 + 2 * t + 1] = c1[1];
  red[warp][g + 8][8 + 2 * t] = c1[2]; red[warp][g + 8][8 + 2 * t + 1] = c1[3];
  __syncthreads();
  if (MODE == 2) {
    // gated MLP epilogue: block rows 0..7 are gate channels c, rows 8..15 the matching
    // up channels; out[m, c] = bf16( bf16(gelu(bf16(gate))) * bf16(up) ), i.e. the
    // separate gate GEMM, up GEMM and gelu_mul_bf16 roundings.
    for (int i = threadIdx.x; i < 8 * 16; i += WARPS * 32) {
      const int n = i / 16, m = i % 16;
      if (m >= M) continue;
      float g = 0.f, u = 0.f;
#pragma unroll
      for (int w2 = 0; w2 < WARPS; ++w2) { g += red[w2][n][m]; u += red[w2][n + 8][m]; }
      const float gv = w8_bf(g * ws[nb * 16 + n]), uv = w8_bf(u * ws[nb * 16 + n + 8]);
      segs.out[0][(long long)m * segs.ld[0] + nb * 8 + n] = __float2bfloat16(w8_bf(w8_gelu_tanh(gv)) * uv);
    }
    return;
  }
  for (int i = threadIdx.x; i < 16 * 16; i += WARPS * 32) {
    const int n = i / 16, m = i % 16;
    if (m >= M) continue;
    float v = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < WARPS; ++w2) v += red[w2][n][m];
    const int gn = nb * 16 + n;
    if (MODE == 0) {
      segs.out[0][(long long)m * N + gn] = __float2bfloat16(v * ws[gn]);
    } else {
      int sg = 0;
      while (sg + 1 < segs.count && gn >= segs.n0[sg + 1]) ++sg;
      segs.out[sg][(long long)m * segs.ld[sg] + (gn - segs.n0[sg])] = __float2bfloat16(v * ws[gn]);
    }
  }
}


void launch_w8a16(const __nv_bfloat16* xp, const int8_t* wq, const float* sp, OutSegs sg,
                  int M, int N, int K, int warps, cudaStream_t st) {
  if (M > 16 || N % 16 || K % 32) throw std::runtime_error("w8a16_tc: needs M<=16, N%16==0, K%32==0");
  const dim3 grid(N / 16);
#define W8L(W) (sg.count == 0 ? (w8a16_tc_kernel<W, 2><<<grid, W * 32, 0, st>>>(xp, wq, sp, sg, M, N, K), 0) \
                : sg.count > 1 ? (w8a16_tc_kernel<W, 1><<<grid, W * 32, 0, st>>>(xp, wq, sp, sg, M, N, K), 0) \
                               : (w8a16_tc_kernel<W, 0><<<grid, W * 32, 0, st>>>(xp, wq, sp, sg, M, N, K), 0))
  switch (warps) {
    case 2: W8L(2); break;
    case 4: W8L(4); break;
    case 8: W8L(8); break;
    case 16: W8L(16); break;
    default: throw std::runtime_error("w8a16_tc: warps must be 2/4/8/16");
  }
#undef W8L
}

}  // namespace

void w8a16_pack(const int8_t* w, int8_t* out, int N, int K, cudaStream_t stream) {
    if (N <= 0 || K <= 0) return;
    if (!w || !out)
        throw std::runtime_error("dm05_w8a16_pack: null pointer argument");
    if (N % 16 != 0)
        throw std::runtime_error("dm05_w8a16_pack: N must be a multiple of 16");
    if (K % 32 != 0)
        throw std::runtime_error("dm05_w8a16_pack: K must be a multiple of 32");
    const long long total = (long long)N * K;
    int grid = (int)((total + 255) / 256);
    if (grid > 65535) grid = 65535;
    w8a16_pack_kernel<<<grid, 256, 0, stream>>>(w, out, N, K);
}

void w8a16_tc(const __nv_bfloat16* x, const int8_t* wp, const float* ws,
              __nv_bfloat16* out, int M, int N, int K, int warps, cudaStream_t stream) {
    if (!x || !wp || !ws || !out)
        throw std::runtime_error("dm05_w8a16_tc: null pointer argument");
    OutSegs sg{};
    sg.out[0] = out; sg.ld[0] = N; sg.count = 1;
    launch_w8a16(x, wp, ws, sg, M, N, K, warps, stream);
}

void w8a16_tc3(const __nv_bfloat16* x, const int8_t* wp, const float* ws,
               __nv_bfloat16* o0, int n0, int ld0,
               __nv_bfloat16* o1, int n1, int ld1,
               __nv_bfloat16* o2, int n2, int ld2,
               int M, int K, int warps, cudaStream_t stream) {
    if (!x || !wp || !ws)
        throw std::runtime_error("dm05_w8a16_tc3: null pointer argument");
    OutSegs sg{};
    sg.out[0] = o0; sg.out[1] = o1; sg.out[2] = o2;
    sg.n0[0] = 0; sg.n0[1] = n0; sg.n0[2] = n0 + n1; sg.n0[3] = n0 + n1 + n2;
    sg.ld[0] = ld0; sg.ld[1] = ld1; sg.ld[2] = ld2; sg.count = 3;
    launch_w8a16(x, wp, ws, sg, M, n0 + n1 + n2, K, warps, stream);
}

void w8a16_tc_glu(const __nv_bfloat16* x, const int8_t* wp, const float* ws,
                  __nv_bfloat16* out, int ff, int M, int K, int warps, cudaStream_t stream) {
    if (!x || !wp || !ws || !out)
        throw std::runtime_error("dm05_w8a16_tc_glu: null pointer argument");
    OutSegs sg{};
    sg.out[0] = out; sg.ld[0] = ff; sg.count = 0;   // count 0 selects the gated epilogue
    launch_w8a16(x, wp, ws, sg, M, 2 * ff, K, warps, stream);
}

}  // namespace dm05
}  // namespace flash_rt

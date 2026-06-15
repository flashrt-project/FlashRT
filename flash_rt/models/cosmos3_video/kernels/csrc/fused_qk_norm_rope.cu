#include <cuda_bf16.h>
// One warp (32 lanes) per (token,head); D=128 -> 4 elements/lane. Warp-shuffle
// RMS reduction (no block sync, no oversized shared). Multiple warps/block to
// amortize launch. Matches FlashRT rms_norm + partial_rope math (bf16-rounded
// normed before rope, out = bf16(bf16(rot*sin)+x*cos), full rotation).
template<int D>
__global__ void qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ cosb, const __nv_bfloat16* __restrict__ sinb,
    int n, int H, int KV, float eps) {
  constexpr int EPT = D / 32;                       // elems/lane (128/32=4)
  const int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int qwarps = n * H;
  const int total = qwarps + n * KV;
  if (warp >= total) return;
  __nv_bfloat16* X; const __nv_bfloat16* Wt; int heads, row, head;
  if (warp < qwarps) { X = Q; Wt = qw; heads = H; row = warp / H; head = warp % H; }
  else { int b = warp - qwarps; X = K; Wt = kw; heads = KV; row = b / KV; head = b % KV; }
  const size_t base = ((size_t)row * heads + head) * D;

  float xv[EPT]; float ss = 0.f;
  #pragma unroll
  for (int i = 0; i < EPT; ++i) { int c = lane + i * 32; xv[i] = __bfloat162float(X[base + c]); ss += xv[i] * xv[i]; }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const float rms = rsqrtf(ss / D + eps);

  // bf16-rounded normed, kept per-lane; rope needs col +/- D/2 -> exchange via shfl
  float nb[EPT];
  #pragma unroll
  for (int i = 0; i < EPT; ++i) { int c = lane + i * 32;
    nb[i] = __bfloat162float(__float2bfloat16(xv[i] * rms * __bfloat162float(Wt[c]))); }
  const int half = D >> 1;                           // rope_dim==D
  #pragma unroll
  for (int i = 0; i < EPT; ++i) {
    int c = lane + i * 32;
    int rc = (c < half) ? (c + half) : (c - half);   // partner col
    int ri = rc >> 5, rl = rc & 31;                  // partner (elem-idx, lane)
    float rot = __shfl_sync(0xffffffff, nb[ri], rl);
    if (c < half) rot = -rot;
    float cv = __bfloat162float(cosb[(size_t)row * D + c]);
    float sv = __bfloat162float(sinb[(size_t)row * D + c]);
    float rs = __bfloat162float(__float2bfloat16(rot * sv));
    X[base + c] = __float2bfloat16(rs + nb[i] * cv);
  }
}

void qk_norm_rope(int64_t Q, int64_t K, int64_t qw, int64_t kw,
                  int64_t cosb, int64_t sinb,
                  int n, int H, int KV, int D, double eps, int64_t stream) {
  const int warps = n * (H + KV);
  const int wpb = 8;                                 // warps/block
  const int blocks = (warps + wpb - 1) / wpb;
  qk_norm_rope_kernel<128><<<blocks, wpb * 32, 0, (cudaStream_t)stream>>>(
      (__nv_bfloat16*)Q, (__nv_bfloat16*)K,
      (const __nv_bfloat16*)qw, (const __nv_bfloat16*)kw,
      (const __nv_bfloat16*)cosb, (const __nv_bfloat16*)sinb,
      n, H, KV, (float)eps);
}

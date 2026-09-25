// ================================================================
// FlashRT AMD -- fused BF16 QKV split + RoPE (RDNA 3.5, wave32)
//
// Fixed Pi0.5 layout:
//   qkv    [rows, 2560] = Q[8,256] | K[1,256] | V[1,256]
//   cos/sin[position, 128]
//   q_out  [rows, 8, 256], k_out/v_out [rows, 1, 256]
//
// One thread owns one complex pair (2*d, 2*d+1). It loads that position's
// cosine/sine once, applies it to all eight Q heads and the shared K head,
// and copies the V pair. This removes the repeated row decomposition and
// coefficient loads of the scalar-per-output implementation.
//
// Rotation products are explicitly rounded through BF16 before subtraction
// and addition. That reproduces the checkpoint/reference rounding points;
// simplifying the expression to one FP32 FMA changes model-visible BF16 bits.
// The launch uses 256 threads for decoder rows and 128 for long encoder rows,
// as measured on gfx1151. No LDS, atomics, or block synchronization is used.
// ================================================================

#include "common_hip_rdna.h"

namespace {

constexpr int kHeadDim = 256;
constexpr int kQHeads = 8;
constexpr int kQDim = kQHeads * kHeadDim;
constexpr int kKvDim = kHeadDim;
constexpr int kQkvDim = kQDim + 2 * kKvDim;
constexpr int kHeadPairs = kHeadDim / 2;
constexpr int kQPairs = kQDim / 2;
constexpr int kKvPairs = kKvDim / 2;
constexpr int kQkvPairs = kQkvDim / 2;

// ── Pair-owned fused split/rotation kernel ──

__global__ void qkv_rope_rdna_kernel(
    const __hip_bfloat16 *__restrict__ qkv,
    const __hip_bfloat16 *__restrict__ rope_cos,
    const __hip_bfloat16 *__restrict__ rope_sin,
    __hip_bfloat16 *__restrict__ q_out, __hip_bfloat16 *__restrict__ k_out,
    __hip_bfloat16 *__restrict__ v_out, int rows, int position_start) {
  const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pair_count = rows * kHeadPairs;
  if (pair_index >= pair_count) {
    return;
  }

  const int row = pair_index / kHeadPairs;
  const int head_pair = pair_index - row * kHeadPairs;
  const auto *qkv2 = reinterpret_cast<const __hip_bfloat162 *>(qkv);
  const int rope_index = (row + position_start) * kHeadPairs + head_pair;
  const float cosine = __bfloat162float(rope_cos[rope_index]);
  const float sine = __bfloat162float(rope_sin[rope_index]);
  const size_t input_base = static_cast<size_t>(row) * kQkvPairs;
  auto *q_out2 = reinterpret_cast<__hip_bfloat162 *>(q_out);

#pragma unroll
  for (int head = 0; head < kQHeads; ++head) {
    const __hip_bfloat162 value =
        qkv2[input_base + head * kHeadPairs + head_pair];
    const float real = __bfloat162float(value.x);
    const float imaginary = __bfloat162float(value.y);
    const float real_cos = __bfloat162float(__float2bfloat16(real * cosine));
    const float imaginary_sin =
        __bfloat162float(__float2bfloat16(imaginary * sine));
    const float imaginary_cos =
        __bfloat162float(__float2bfloat16(imaginary * cosine));
    const float real_sin = __bfloat162float(__float2bfloat16(real * sine));
    q_out2[static_cast<size_t>(row) * kQPairs + head * kHeadPairs + head_pair] =
        __hip_bfloat162(__float2bfloat16(real_cos - imaginary_sin),
                        __float2bfloat16(imaginary_cos + real_sin));
  }

  const __hip_bfloat162 key_value = qkv2[input_base + kQPairs + head_pair];
  const float key_real = __bfloat162float(key_value.x);
  const float key_imaginary = __bfloat162float(key_value.y);
  auto *k_out2 = reinterpret_cast<__hip_bfloat162 *>(k_out);
  k_out2[static_cast<size_t>(row) * kKvPairs + head_pair] = __hip_bfloat162(
      __float2bfloat16(
          __bfloat162float(__float2bfloat16(key_real * cosine)) -
          __bfloat162float(__float2bfloat16(key_imaginary * sine))),
      __float2bfloat16(
          __bfloat162float(__float2bfloat16(key_imaginary * cosine)) +
          __bfloat162float(__float2bfloat16(key_real * sine))));

  auto *v_out2 = reinterpret_cast<__hip_bfloat162 *>(v_out);
  v_out2[static_cast<size_t>(row) * kKvPairs + head_pair] =
      qkv2[input_base + kQPairs + kKvPairs + head_pair];
}

} // namespace

// ── Host dispatch ──

void qkv_rope_rdna(const __hip_bfloat16 *qkv, const __hip_bfloat16 *rope_cos,
                   const __hip_bfloat16 *rope_sin, __hip_bfloat16 *q_out,
                   __hip_bfloat16 *k_out, __hip_bfloat16 *v_out, int rows,
                   int position_start, hipStream_t stream) {
  const int threads = rows <= 16 ? 256 : 128;
  const int pairs = rows * kHeadPairs;
  qkv_rope_rdna_kernel<<<(pairs + threads - 1) / threads, threads, 0, stream>>>(
      qkv, rope_cos, rope_sin, q_out, k_out, v_out, rows, position_start);
}

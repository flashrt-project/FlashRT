#include "sage2_blackwell.cuh"

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <mutex>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "qattn/qk_int_sv_f8_core.cuh"
#undef PACK_SIZE_QK
#undef PACK_SIZE_V
#undef PACK_SIZE_O
#undef MMA_QK_M
#undef MMA_QK_N
#undef MMA_QK_K
#undef MMA_SV_M
#undef MMA_SV_N
#undef MMA_SV_K
#include "qattn/qk_int_sv_f16_core.cuh"

namespace flashrt_hub::sage2 {
namespace {

constexpr int kHeadDim = 128;
constexpr int kCtaQ = 128;
constexpr int kCtaK = 64;
constexpr int kWarpQ = 32;
constexpr int kWarpK = 64;
constexpr int kPack = 8;

inline int div_up_int(int x, int y) {
  return (x + y - 1) / y;
}

__device__ __forceinline__ int8_t f32_to_i8_sat(float x) {
  x = fminf(127.0f, fmaxf(-127.0f, nearbyintf(x)));
  return static_cast<int8_t>(x);
}

template <int BlockTokens>
__global__ void quant_int8_bf16_nhd_d128_kernel(
    const __nv_bfloat16* __restrict__ x,
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    int seqlen,
    int heads) {
  constexpr int threads_per_token = kHeadDim / kPack;
  const int block_id = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int token_in_block = tid / threads_per_token;
  const int d_pack = tid - token_in_block * threads_per_token;
  const int pos = block_id * BlockTokens + token_in_block;

  float vals[kPack];
  float amax = 1.0e-7f;
  if (pos < seqlen) {
    const __nv_bfloat16* src =
        x + (((long long)b * seqlen + pos) * heads + h) * kHeadDim + d_pack * kPack;
#pragma unroll
    for (int i = 0; i < kPack; ++i) {
      vals[i] = __bfloat162float(src[i]);
      amax = fmaxf(amax, fabsf(vals[i]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < kPack; ++i) {
      vals[i] = 0.0f;
    }
  }

  __shared__ float smem[1024];
  smem[tid] = amax;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale[((long long)b * heads + h) * gridDim.x + block_id] = smem[0] * (1.0f / 127.0f);
  }
  if (pos < seqlen) {
    const float inv_s = 127.0f / smem[0];
    int8_t* dst =
        out + (((long long)b * seqlen + pos) * heads + h) * kHeadDim + d_pack * kPack;
    char4 lo = make_char4(
        f32_to_i8_sat(vals[0] * inv_s),
        f32_to_i8_sat(vals[1] * inv_s),
        f32_to_i8_sat(vals[2] * inv_s),
        f32_to_i8_sat(vals[3] * inv_s));
    char4 hi = make_char4(
        f32_to_i8_sat(vals[4] * inv_s),
        f32_to_i8_sat(vals[5] * inv_s),
        f32_to_i8_sat(vals[6] * inv_s),
        f32_to_i8_sat(vals[7] * inv_s));
    reinterpret_cast<char4*>(dst)[0] = lo;
    reinterpret_cast<char4*>(dst)[1] = hi;
  }
}

__global__ void v_bf16_to_fp16_d128_kernel(
    const __nv_bfloat16* __restrict__ v,
    half* __restrict__ out,
    int seqlen,
    int heads) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b = row / seqlen;
  const int t = row - b * seqlen;
  const __nv_bfloat16* src =
      v + (((long long)b * seqlen + t) * heads + h) * kHeadDim;
  half* dst = out + (((long long)b * seqlen + t) * heads + h) * kHeadDim;
  for (int i = threadIdx.x; i < kHeadDim; i += blockDim.x) {
    dst[i] = __float2half_rn(__bfloat162float(src[i]));
  }
}

__device__ __forceinline__ int sage_v_inv_perm64(int t) {
  const int base = (t >> 4) << 4;
  const int m = t & 15;
  const int inv = (m < 2) ? m :
                  (m < 4) ? (m + 6) :
                  (m < 6) ? (m - 2) :
                  (m < 8) ? (m + 4) :
                  (m < 10) ? (m - 4) :
                  (m < 12) ? (m + 2) :
                  (m < 14) ? (m - 6) : m;
  return base + inv;
}

__global__ void v_bf16_to_fp8_tpp_d128_kernel(
    const __nv_bfloat16* __restrict__ v,
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    int seqlen,
    int heads) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int d = blockIdx.z;
  const int tid = threadIdx.x;
  const int padded = ((seqlen + 63) / 64) * 64;

  float max_v = -INFINITY;
  float min_v = INFINITY;
  for (int t = tid; t < seqlen; t += blockDim.x) {
    const __nv_bfloat16* src =
        v + (((long long)b * seqlen + t) * heads + h) * kHeadDim + d;
    const float x = __bfloat162float(*src);
    max_v = fmaxf(max_v, x);
    min_v = fminf(min_v, x);
  }

  __shared__ float smem_max[256];
  __shared__ float smem_min[256];
  smem_max[tid] = max_v;
  smem_min[tid] = min_v;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem_max[tid] = fmaxf(smem_max[tid], smem_max[tid + stride]);
      smem_min[tid] = fminf(smem_min[tid], smem_min[tid + stride]);
    }
    __syncthreads();
  }
  const float amax = fmaxf(fabsf(smem_max[0]), fabsf(smem_min[0]));
  const float s = fmaxf(amax, 1.0e-7f) * (1.0f / 448.0f);
  if (tid == 0) {
    scale[((long long)b * heads + h) * kHeadDim + d] = s;
  }
  const float inv_s = 448.0f / fmaxf(amax, 1.0e-7f);
  int8_t* out_base = out + (((long long)b * kHeadDim + d) * heads + h) * padded;

  for (int t0 = tid * kPack; t0 < padded; t0 += blockDim.x * kPack) {
    float vals[kPack];
#pragma unroll
    for (int i = 0; i < kPack; ++i) {
      const int out_t = t0 + i;
      const int t = sage_v_inv_perm64(out_t);
      float x = 0.0f;
      if (t < seqlen) {
        const __nv_bfloat16* src =
            v + (((long long)b * seqlen + t) * heads + h) * kHeadDim + d;
        x = __bfloat162float(*src) * inv_s;
      }
      vals[i] = x;
    }
    uint32_t fp8_pack[2];
    floatx4_to_e4m3x4(fp8_pack, vals, vals + 2);
    floatx4_to_e4m3x4(fp8_pack + 1, vals + 4, vals + 6);
    *reinterpret_cast<uint2*>(out_base + t0) = *reinterpret_cast<uint2*>(fp8_pack);
  }
}

template <MaskMode Mask>
int launch_f16(
    const void* q_int8,
    const void* k_int8,
    const void* v_half,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream) {
  if (!q_int8 || !k_int8 || !v_half || !out_bf16 || !q_scale || !k_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return -2;
  }
  const int num_kv_groups = num_q_heads / num_kv_heads;

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_kv_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_bz_v = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim);
  const uint32_t stride_seq_v = static_cast<uint32_t>(num_kv_heads * kHeadDim);
  const uint32_t stride_h_v = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim);

  using Kernel = decltype(&qk_int_sv_f16_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
      float, false, nv_bfloat16, ComputeUnit::kTensorCore, Mask, false, false>);
  Kernel kernel = qk_int_sv_f16_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
      float, false, nv_bfloat16, ComputeUnit::kTensorCore, Mask, false, false>;

  const size_t smem_qkv =
      static_cast<size_t>(kCtaQ * kHeadDim * sizeof(int8_t) +
                          kCtaK * kHeadDim * sizeof(int8_t) +
                          kCtaK * kHeadDim * sizeof(half));
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim * sizeof(half));
  const size_t smem_max = std::max(smem_qkv, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_q_heads, batch);
  dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));
  kernel<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(q_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(k_int8)),
      const_cast<half*>(reinterpret_cast<const half*>(v_half)),
      reinterpret_cast<nv_bfloat16*>(out_bf16),
      nullptr,
      const_cast<float*>(reinterpret_cast<const float*>(q_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(k_scale)),
      nullptr,
      static_cast<uint32_t>(seqlen_q),
      static_cast<uint32_t>(seqlen_k),
      static_cast<uint32_t>(num_kv_groups),
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_seq_v, stride_h_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale);
  return static_cast<int>(cudaGetLastError());
}

template <MaskMode Mask>
int launch_f8_lse(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream,
    int q_offset,
    void* lse_f32) {
  if (!q_int8 || !k_int8 || !v_fp8 || !out_bf16 || !q_scale || !k_scale || !v_scale) {
    return -1;
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return -2;
  }
  const int num_kv_groups = num_q_heads / num_kv_heads;
  const int padded_k = div_up_int(seqlen_k, kCtaK) * kCtaK;

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_kv_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_bz_v = static_cast<uint32_t>(kHeadDim * num_kv_heads * padded_k);
  const uint32_t stride_h_v = static_cast<uint32_t>(padded_k);
  const uint32_t stride_d_v = static_cast<uint32_t>(num_kv_heads * padded_k);
  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim);

  using Kernel = decltype(&qk_int_sv_f8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
      float, false, nv_bfloat16, ComputeUnit::kCudaCore, Mask, true, true, false>);
  Kernel kernel = qk_int_sv_f8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
      float, false, nv_bfloat16, ComputeUnit::kCudaCore, Mask, true, true, false>;

  const size_t smem_qkv = static_cast<size_t>(kCtaQ * kHeadDim + kCtaK * kHeadDim + kCtaK * kHeadDim);
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim * sizeof(half));
  const size_t smem_max = std::max(smem_qkv, smem_o);
  static std::once_flag attr_once_lse;
  std::call_once(attr_once_lse, [&]() {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_q_heads, batch);
  dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));
  kernel<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(q_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(k_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(v_fp8)),
      reinterpret_cast<nv_bfloat16*>(out_bf16),
      reinterpret_cast<float*>(lse_f32),
      const_cast<float*>(reinterpret_cast<const float*>(q_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(k_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(v_scale)),
      nullptr,
      static_cast<uint32_t>(seqlen_q),
      static_cast<uint32_t>(seqlen_k),
      static_cast<uint32_t>(num_kv_groups),
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale, static_cast<uint32_t>(q_offset));
  return static_cast<int>(cudaGetLastError());
}

template <MaskMode Mask>
int launch_f8(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    cudaStream_t stream,
    int q_offset = 0,
    void* lse_f32 = nullptr) {
  if (!q_int8 || !k_int8 || !v_fp8 || !out_bf16 || !q_scale || !k_scale || !v_scale) {
    return -1;
  }
  if (lse_f32 != nullptr) {
    return launch_f8_lse<Mask>(q_int8, k_int8, v_fp8, out_bf16, q_scale, k_scale, v_scale, batch, seqlen_q, seqlen_k,
                               num_q_heads, num_kv_heads, softmax_scale, stream, q_offset, lse_f32);
  }
  if (batch <= 0 || seqlen_q <= 0 || seqlen_k <= 0 ||
      num_q_heads <= 0 || num_kv_heads <= 0 ||
      num_q_heads % num_kv_heads != 0) {
    return -2;
  }
  const int num_kv_groups = num_q_heads / num_kv_heads;
  const int padded_k = div_up_int(seqlen_k, kCtaK) * kCtaK;

  const uint32_t stride_bz_q = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_q = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_q = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_bz_k = static_cast<uint32_t>(seqlen_k * num_kv_heads * kHeadDim);
  const uint32_t stride_seq_k = static_cast<uint32_t>(num_kv_heads * kHeadDim);
  const uint32_t stride_h_k = static_cast<uint32_t>(kHeadDim);
  const uint32_t stride_bz_v = static_cast<uint32_t>(kHeadDim * num_kv_heads * padded_k);
  const uint32_t stride_h_v = static_cast<uint32_t>(padded_k);
  const uint32_t stride_d_v = static_cast<uint32_t>(num_kv_heads * padded_k);
  const uint32_t stride_bz_o = static_cast<uint32_t>(seqlen_q * num_q_heads * kHeadDim);
  const uint32_t stride_seq_o = static_cast<uint32_t>(num_q_heads * kHeadDim);
  const uint32_t stride_h_o = static_cast<uint32_t>(kHeadDim);

  using Kernel = decltype(&qk_int_sv_f8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
      float, false, nv_bfloat16, ComputeUnit::kCudaCore, Mask, false, true, false>);
  Kernel kernel = qk_int_sv_f8_attn_kernel<
      kCtaQ, kCtaK, kWarpQ, kWarpK, kHeadDim,
      DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
      float, false, nv_bfloat16, ComputeUnit::kCudaCore, Mask, false, true, false>;

  const size_t smem_qkv = static_cast<size_t>(kCtaQ * kHeadDim + kCtaK * kHeadDim + kCtaK * kHeadDim);
  const size_t smem_o = static_cast<size_t>(kCtaQ * kHeadDim * sizeof(half));
  const size_t smem_max = std::max(smem_qkv, smem_o);
  static std::once_flag attr_once;
  std::call_once(attr_once, [&]() {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_max));
  });

  dim3 grid(div_up_int(seqlen_q, kCtaQ), num_q_heads, batch);
  dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));
  kernel<<<grid, block, smem_max, stream>>>(
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(q_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(k_int8)),
      const_cast<int8_t*>(reinterpret_cast<const int8_t*>(v_fp8)),
      reinterpret_cast<nv_bfloat16*>(out_bf16),
      nullptr,
      const_cast<float*>(reinterpret_cast<const float*>(q_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(k_scale)),
      const_cast<float*>(reinterpret_cast<const float*>(v_scale)),
      nullptr,
      static_cast<uint32_t>(seqlen_q),
      static_cast<uint32_t>(seqlen_k),
      static_cast<uint32_t>(num_kv_groups),
      stride_bz_q, stride_seq_q, stride_h_q,
      stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v,
      stride_bz_o, stride_seq_o, stride_h_o,
      softmax_scale, static_cast<uint32_t>(q_offset));
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int padded_k64(int seqlen_k) {
  return div_up_int(seqlen_k, kCtaK) * kCtaK;
}

int q_scale_elems(int batch, int seqlen_q, int num_q_heads) {
  return batch * num_q_heads * div_up_int(seqlen_q, kWarpQ);
}

int k_scale_elems(int batch, int seqlen_k, int num_kv_heads) {
  return batch * num_kv_heads * div_up_int(seqlen_k, kWarpK);
}

int v_scale_elems(int batch, int head_dim, int num_kv_heads) {
  return batch * num_kv_heads * head_dim;
}

void quant_per_warp_int8_bf16_d128(
    const void* x_bf16,
    void* out_i8,
    void* scale_f32,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream) {
  if (batch <= 0 || seqlen <= 0 || heads <= 0) return;
  quant_int8_bf16_nhd_d128_kernel<kWarpQ><<<
      dim3(div_up_int(seqlen, kWarpQ), heads, batch),
      kWarpQ * (kHeadDim / kPack),
      0,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<int8_t*>(out_i8),
      reinterpret_cast<float*>(scale_f32),
      seqlen,
      heads);
}

void quant_per_block_int8_bf16_d128(
    const void* x_bf16,
    void* out_i8,
    void* scale_f32,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream) {
  if (batch <= 0 || seqlen <= 0 || heads <= 0) return;
  quant_int8_bf16_nhd_d128_kernel<kWarpK><<<
      dim3(div_up_int(seqlen, kWarpK), heads, batch),
      kWarpK * (kHeadDim / kPack),
      0,
      stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x_bf16),
      reinterpret_cast<int8_t*>(out_i8),
      reinterpret_cast<float*>(scale_f32),
      seqlen,
      heads);
}

void v_bf16_to_fp16_d128(
    const void* v_bf16,
    void* v_half,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream) {
  if (batch <= 0 || seqlen <= 0 || heads <= 0) return;
  v_bf16_to_fp16_d128_kernel<<<dim3(batch * seqlen, heads, 1), 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v_bf16),
      reinterpret_cast<half*>(v_half),
      seqlen,
      heads);
}

void v_bf16_to_fp8_tpp_d128(
    const void* v_bf16,
    void* v_fp8,
    void* v_scale,
    int batch,
    int seqlen,
    int heads,
    cudaStream_t stream) {
  if (batch <= 0 || seqlen <= 0 || heads <= 0) return;
  v_bf16_to_fp8_tpp_d128_kernel<<<dim3(heads, batch, kHeadDim), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v_bf16),
      reinterpret_cast<int8_t*>(v_fp8),
      reinterpret_cast<float*>(v_scale),
      seqlen,
      heads);
}

int sage2_qk_int8_sv_f16_bf16_gqa_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_half,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    bool causal,
    cudaStream_t stream) {
  if (causal) {
    return launch_f16<MaskMode::kCausal>(
        q_int8, k_int8, v_half, out_bf16, q_scale, k_scale,
        batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, softmax_scale, stream);
  }
  return launch_f16<MaskMode::kNone>(
      q_int8, k_int8, v_half, out_bf16, q_scale, k_scale,
      batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, softmax_scale, stream);
}

int sage2_qk_int8_sv_f8_bf16_gqa_d128(
    const void* q_int8,
    const void* k_int8,
    const void* v_fp8,
    void* out_bf16,
    const void* q_scale,
    const void* k_scale,
    const void* v_scale,
    int batch,
    int seqlen_q,
    int seqlen_k,
    int num_q_heads,
    int num_kv_heads,
    float softmax_scale,
    bool causal,
    cudaStream_t stream,
    int q_offset,
    void* lse_f32) {
  if (causal) {
    return launch_f8<MaskMode::kCausal>(
        q_int8, k_int8, v_fp8, out_bf16, q_scale, k_scale, v_scale,
        batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, softmax_scale, stream, q_offset, lse_f32);
  }
  return launch_f8<MaskMode::kNone>(
      q_int8, k_int8, v_fp8, out_bf16, q_scale, k_scale, v_scale,
      batch, seqlen_q, seqlen_k, num_q_heads, num_kv_heads, softmax_scale, stream, q_offset, lse_f32);
}

}  // namespace flashrt_hub::sage2

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include "qattn/qk_int_sv_f16_core.cuh"

void sage_gqa_d128(int64_t q8, int64_t k8, int64_t v16, int64_t out,
                   int64_t qs, int64_t ks,
                   int B, int Lq, int Lk, int Hq, int Hkv,
                   double scale, int64_t stream) {
  constexpr int HD = 128, CTAQ = 128, CTAK = 64, WQ = 32, WK = 64;
  const uint32_t groups = (uint32_t)(Hq / Hkv);
  const uint32_t sbz_q = (uint32_t)(Lq*Hq*HD),  sseq_q = (uint32_t)(Hq*HD),  sh = (uint32_t)HD;
  const uint32_t sbz_k = (uint32_t)(Lk*Hkv*HD), sseq_k = (uint32_t)(Hkv*HD);
  const uint32_t sbz_o = (uint32_t)(Lq*Hq*HD),  sseq_o = (uint32_t)(Hq*HD);
  auto kernel = qk_int_sv_f16_attn_kernel<
      CTAQ, CTAK, WQ, WK, HD, DataType::kInt8,
      QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
      float, false, nv_bfloat16, ComputeUnit::kTensorCore,
      MaskMode::kNone, false, false>;
  size_t smem_qkv = (size_t)(CTAQ*HD + CTAK*HD) + (size_t)CTAK*HD*sizeof(half);
  size_t smem_o = (size_t)CTAQ*HD*sizeof(half);
  size_t smem = std::max(smem_qkv, smem_o);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  dim3 grid((Lq + CTAQ - 1) / CTAQ, Hq, B);
  dim3 block(32, (CTAQ / WQ) * (CTAK / WK));
  kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
      (int8_t*)q8, (int8_t*)k8, (half*)v16, (nv_bfloat16*)out, nullptr,
      (float*)qs, (float*)ks, nullptr,
      (uint32_t)Lq, (uint32_t)Lk, groups,
      sbz_q, sseq_q, sh, sbz_k, sseq_k, sh, sbz_k, sseq_k, sh, sbz_o, sseq_o, sh,
      (float)scale);
}

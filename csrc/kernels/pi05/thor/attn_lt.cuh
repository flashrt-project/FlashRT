// FlashRT — decoder attention chain (QK^T -> softmax -> PV, fp16, fp32 accumulate) through cublasLt with the
// fastest workspace-free heuristic for each of the two shapes, timed once per shape on first (uncaptured) use.
// Drop-in for kernels/attention_cublas.cu attention_qkv_fp16 (same operands and layouts; tile choice may differ).
#pragma once
#include <cuda_runtime.h>
namespace flash_rt {
namespace fp4 {
int attention_qkv_fp16_lt(const void* Q, const void* K, const void* V, void* logits, void* out,
                          int S, int S_kv, int NH, int HD, float attn_scale, cudaStream_t stream);
}  // namespace fp4
}  // namespace flash_rt

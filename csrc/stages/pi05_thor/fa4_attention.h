// FlashAttention-4 forward on FlashRT's AOT-exported CuTe DSL modules
// (csrc/attention/fa4_aot, produced by tools/export_fa4_aot.py). Non-causal fp16.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace flashrt_trt {

// 1/sqrt(head_dim) rounded as FlashRT's FA4 entry computes it (in double,
// then narrowed); the float-domain quotient differs by one ulp for some head
// sizes (72), which changes the attention output bits.
inline float fa4_default_scale(int head_dim) {
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
}

// Loads every module once per process; 0 on success.
int fa4_load();

// GQA, head_dim 256, 1 KV head: q/o [1, Sq, NH, 256], k/v [1, Sk, 1, 256],
// contiguous. Picks the module whose query stage count matches Sq * NH.
int fa4_hd256_gqa(void* q, void* k, void* v, void* o, int32_t sq, int32_t sk, int32_t nh,
                  float scale, cudaStream_t stream);

// MHA, head_dim 72: [B, S, NH, 72]; row strides may differ between tensors
// (Q/K/V as views of one interleaved QKV row). A row stride is in elements.
int fa4_hd72_mha(void* q, int64_t q_row_stride, void* k, int64_t k_row_stride, void* v,
                 int64_t v_row_stride, void* o, int32_t b, int32_t s, int32_t nh, float scale,
                 cudaStream_t stream);

}  // namespace flashrt_trt

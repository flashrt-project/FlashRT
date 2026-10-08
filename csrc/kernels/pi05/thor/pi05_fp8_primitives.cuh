#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" int pi05_cutlass_fp8_sq(void*, void*, void*, int, int, int,
                                  float, float, cudaStream_t);
void pi05_quantize_fp8_static_fp16(const __half*, __nv_fp8_e4m3*, const float*, int, cudaStream_t);
int pi05_qkv_split_rope_kvcache_fp16_vec(const __half*, const __half*, __half*, __half*, __half*,
                                      int, int, int, int, int, long, int, cudaStream_t);

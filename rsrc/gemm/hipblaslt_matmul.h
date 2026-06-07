#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

void hipblaslt_matmul_bf16(const void* a, const void* b, void* out,
                           int64_t m, int64_t n, int64_t k,
                           hipStream_t stream);

void hipblaslt_matmul_fp8_e4m3fnuz_bf16(const void* a, const void* b,
                                        const float* a_scale,
                                        const float* b_scale,
                                        void* out,
                                        int64_t m, int64_t n, int64_t k,
                                        hipStream_t stream);

void hipblaslt_linear_bf16(const void* x, const void* weight, const void* bias,
                           void* out,
                           int64_t m, int64_t n, int64_t k,
                           hipStream_t stream);

void hipblaslt_linear_fp8_e4m3fnuz_bf16(
    const void* x, const void* weight,
    const float* x_scale, const float* weight_scale,
    const void* bias, void* out,
    int64_t m, int64_t n, int64_t k,
    hipStream_t stream);

std::size_t hipblaslt_algo_cache_size();
std::vector<std::string> hipblaslt_algo_cache_keys();
void hipblaslt_algo_cache_clear();
std::size_t hipblaslt_linear_plan_cache_size();
std::vector<std::string> hipblaslt_linear_plan_cache_keys();
void hipblaslt_linear_plan_cache_clear();

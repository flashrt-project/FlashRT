// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
void attention_bshd_bf16_rdna(const __hip_bfloat16 *, const __hip_bfloat16 *,
                              const __hip_bfloat16 *, __hip_bfloat16 *, int,
                              int, int, int, int, int, int64_t, int64_t,
                              int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t, int64_t, bool, hipStream_t);
void normalize_bf16_rdna(const __hip_bfloat16 *, const __hip_bfloat16 *,
                         const __hip_bfloat16 *, const __hip_bfloat16 *,
                         __hip_bfloat16 *, __hip_bfloat16 *, int, int, float,
                         bool, float, bool, hipStream_t);
void rope_bf16_rdna(const __hip_bfloat16 *, const void *, const void *,
                    __hip_bfloat16 *, int64_t, int, int, int64_t, int64_t,
                    int64_t, int64_t, bool, bool, hipStream_t);

// SPDX-License-Identifier: Apache-2.0
//
// FlashRT — shared device helpers for the DM05 kernel module.
//
// These are the primitives the DM05 attention kernels are built from. They are
// deliberately local to csrc/kernels/dm05/ rather than added to the shared
// csrc/kernels/common.cuh: each one encodes an instruction-level choice (the
// fast-math ex2/rcp forms, the cp.async width, the exact mma shapes) that the
// DM05 bit-for-bit gates depend on, and none of it is needed by another model.
//
// The fast-math forms matter. The flash_rt_kernels target builds with
// --ftz=true --prec-div=false --prec-sqrt=false (CMakeLists.txt), so a
// reciprocal or division in the quantize kernels it holds lowers to
// rcp.approx.ftz / div.approx.ftz and their multiplies flush to zero. The
// epilogues in this module spell those instructions out so that a value they
// produce rounds exactly as the standalone FlashRT kernel it replaces would
// round it — IEEE division differs from them on FP8 rounding ties. This module
// does not set those flags itself: it needs IEEE everywhere else, so it opts
// in per instruction instead of per translation unit.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace flash_rt {
namespace dm05 {

// 16-byte cp.async with a predicate. The predicated form is what lets the
// key/value pipeline read past the end of a tile without a branch.
__device__ __forceinline__ void dm05_cp16(void* dst, const void* src, bool pred) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(src), "r"(pred ? 16 : 0));
}

// ex2.approx.ftz — the form --use_fast_math gives exp2f in the FA2 build.
__device__ __forceinline__ float dm05_exp2(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float dm05_rcp(float s) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s));
    return r;
}

__device__ __forceinline__ float dm05_mul(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ void dm05_mma_bf16_16816(float (&c)[4], const uint32_t (&a)[4],
                                                    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void dm05_mma_bf16_1688(float (&c)[4], uint32_t a0, uint32_t a1,
                                                  uint32_t b0) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

}  // namespace dm05
}  // namespace flash_rt

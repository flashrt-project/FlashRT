#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>

#if defined(FLASHRT_AMD_CDNA3)
#include <hipblaslt/hipblaslt_float8.h>
#elif defined(FLASHRT_AMD_CDNA4)
#include <hip/hip_fp8.h>
#endif

// One extension build targets one AMD ISA.  FP8 bytes are not portable
// between CDNA3 E4M3 FNUZ and CDNA4 OCP E4M3, so every producer, consumer,
// and hipBLASLt descriptor must take its format from this header.
#if defined(FLASHRT_AMD_CDNA3) == defined(FLASHRT_AMD_CDNA4)
#error "Define exactly one of FLASHRT_AMD_CDNA3 or FLASHRT_AMD_CDNA4"
#endif

namespace flashrt::amd::arch {

#if defined(FLASHRT_AMD_CDNA3)

// ROCm 6.x exposes the MI300 FNUZ type through hipBLASLt rather than
// <hip/hip_fp8.h>.  Keep the __x byte member used by the existing packed
// stores while delegating conversion to hipblaslt_f8_fnuz.
struct fp8_e4m3 {
    std::uint8_t __x;

    __host__ __device__ fp8_e4m3() = default;
    explicit __host__ __device__ fp8_e4m3(float value) {
        hipblaslt_f8_fnuz converted(value);
#if HIP_VERSION_MAJOR >= 7
        __x = converted.__x;
#else
        __x = converted.data;
#endif
    }
    explicit __host__ __device__ operator float() const {
        hipblaslt_f8_fnuz converted;
#if HIP_VERSION_MAJOR >= 7
        converted.__x = __x;
#else
        converted.data = __x;
#endif
        return static_cast<float>(converted);
    }
};

constexpr float fp8_max_finite = 240.0f;
constexpr hipDataType fp8_hip_type = HIP_R_8F_E4M3_FNUZ;
#if HIP_VERSION_MAJOR >= 7
constexpr hipblasComputeType_t fp8_compute_type =
    HIPBLAS_COMPUTE_32F_FAST_8F_FNUZ;
#else
// ROCm 6.x accepts FNUZ matrix layouts but predates the explicit fast-FNUZ
// compute selector. HIPBLAS_COMPUTE_32F is its ABI-compatible selector.
constexpr hipblasComputeType_t fp8_compute_type = HIPBLAS_COMPUTE_32F;
#endif
constexpr const char* fp8_format = "e4m3fnuz";
constexpr const char* hardware_name = "amd_cdna3";
constexpr bool supports_mxfp4 = false;
constexpr bool supports_packed_fp8_mfma = true;
constexpr bool supports_packed_bf16_mfma = false;
constexpr bool supports_fused_attention_fp8out = true;
constexpr bool supports_aiter = true;

#else
using fp8_e4m3 = __hip_fp8_e4m3;

constexpr float fp8_max_finite = 448.0f;
constexpr hipDataType fp8_hip_type = HIP_R_8F_E4M3;
constexpr hipblasComputeType_t fp8_compute_type = HIPBLAS_COMPUTE_32F;
constexpr const char* fp8_format = "e4m3fn";
constexpr const char* hardware_name = "amd_cdna4";
constexpr bool supports_mxfp4 = true;
constexpr bool supports_packed_fp8_mfma = true;
constexpr bool supports_packed_bf16_mfma = true;
constexpr bool supports_fused_attention_fp8out = true;
constexpr bool supports_aiter = true;

#endif

}  // namespace flashrt::amd::arch

// Preserve the established kernel signatures while centralizing their actual
// byte representation.  This alias is intentionally local to the AMD module.
#if defined(FLASHRT_AMD_CDNA3)
#define __hip_fp8_e4m3 ::flashrt::amd::arch::fp8_e4m3
#endif

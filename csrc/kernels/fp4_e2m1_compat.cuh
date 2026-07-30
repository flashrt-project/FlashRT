// Packed-E2M1 to half2 conversion that does not require <cuda_fp4.h>.
//
// The CUDA header only exists from 12.8 onwards, and on architectures without
// the cvt.rn.f16x2.e2m1x2 instruction it decodes each nibble in software
// anyway. E2M1 has sixteen representable values, so a table gives the same
// result on every target and removes the toolkit dependency: a Jetson image
// pinned to CUDA 12.6 can still build the weight-only 4-bit kernels.
//
// Where the header is present it is used, so the emitted code on those targets
// is unchanged.

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Define FLASHRT_FP4_FORCE_TABLE to take the portable path even where the
// header exists. Used to check the two against each other.
#if !defined(FLASHRT_FP4_FORCE_TABLE) && defined(__has_include)
#if __has_include(<cuda_fp4.h>)
#define FLASHRT_HAVE_CUDA_FP4_HEADER 1
#endif
#endif

#ifdef FLASHRT_HAVE_CUDA_FP4_HEADER
#include <cuda_fp4.h>
#endif

namespace flash_rt {
namespace fp4 {

// Decode one byte holding two E2M1 values, low nibble first.
__device__ __forceinline__ __half2_raw cvt_e2m1x2_to_halfraw2(uint8_t pair) {
#ifdef FLASHRT_HAVE_CUDA_FP4_HEADER
  return __nv_cvt_fp4x2_to_halfraw2(
      static_cast<__nv_fp4x2_storage_t>(pair), __NV_E2M1);
#else
  // The sixteen E2M1 values as raw half bit patterns, indexed by the 4-bit
  // code: one sign bit, two exponent bits, one mantissa bit, giving 0,
  // +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6. Function-local so no
  // translation unit owns a device symbol.
  constexpr unsigned short kAsHalfRaw[16] = {
      0x0000,  // 0.0
      0x3800,  // 0.5
      0x3C00,  // 1.0
      0x3E00,  // 1.5
      0x4000,  // 2.0
      0x4200,  // 3.0
      0x4400,  // 4.0
      0x4600,  // 6.0
      0x8000,  // -0.0
      0xB800,  // -0.5
      0xBC00,  // -1.0
      0xBE00,  // -1.5
      0xC000,  // -2.0
      0xC200,  // -3.0
      0xC400,  // -4.0
      0xC600,  // -6.0
  };
  __half2_raw out;
  out.x = kAsHalfRaw[pair & 0x0F];
  out.y = kAsHalfRaw[(pair >> 4) & 0x0F];
  return out;
#endif
}

}  // namespace fp4
}  // namespace flash_rt

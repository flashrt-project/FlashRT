// ================================================================
// FlashRT AMD -- wave32 BF16 WMMA fragment helpers for RDNA.
//
// Each call is one 16x16x16 BF16 WMMA with FP32 accumulation. The two RDNA
// ISA families distribute the operands differently:
//
//   gfx11 (gfx1151): every lane carries the full 16-element K fragment of
//     row/column lane % 16; lanes 16..31 replicate lanes 0..15.
//     Accumulator element e of lane l holds row 2 * e + l / 16.
//   gfx12 (gfx1201): lane l carries K elements [8 * (l / 16), +8) of
//     row/column l % 16. Accumulator element e of lane l holds row
//     e + 8 * (l / 16).
//
// In both families the accumulator column is lane % 16. The build selects
// the family with FLASHRT_AMD_RDNA4, so one kernel source covers both.
// ================================================================
#pragma once

#include "common_hip_rdna.h"

#if defined(FLASHRT_AMD_RDNA4) && defined(__HIP_DEVICE_COMPILE__) && \
    !defined(__GFX12__)
#error "FLASHRT_AMD_RDNA4 requires a gfx12 device target"
#endif

#if defined(FLASHRT_AMD_RDNA4)
#define FLASHRT_RDNA_WMMA_LANE_K 8
#else
#define FLASHRT_RDNA_WMMA_LANE_K 16
#endif

using RdnaWmmaFragment =
    __bf16 __attribute__((ext_vector_type(FLASHRT_RDNA_WMMA_LANE_K)));
using RdnaWmmaAccumulator = float __attribute__((ext_vector_type(8)));

union RdnaWmmaFragmentWords {
  RdnaWmmaFragment values;
  unsigned long long words[FLASHRT_RDNA_WMMA_LANE_K / 4];
};

// First K element of this lane's fragment inside a 16-element K block.
__device__ __forceinline__ int rdna_wmma_lane_k(int lane) {
#if defined(FLASHRT_AMD_RDNA4)
  return (lane >> 4) * FLASHRT_RDNA_WMMA_LANE_K;
#else
  (void)lane;
  return 0;
#endif
}

__device__ __forceinline__ int rdna_wmma_accumulator_row(int lane,
                                                         int element) {
#if defined(FLASHRT_AMD_RDNA4)
  return element + ((lane >> 4) << 3);
#else
  return 2 * element + (lane >> 4);
#endif
}

// `block` addresses K element 0 of a 16-element block in this lane's
// row/column and must be 8-byte aligned (LDS rows use padded pitches).
__device__ __forceinline__ RdnaWmmaFragment
rdna_wmma_load_fragment(const __bf16 *block, int lane) {
  const unsigned long long *source =
      reinterpret_cast<const unsigned long long *>(block +
                                                   rdna_wmma_lane_k(lane));
  RdnaWmmaFragmentWords fragment;
#pragma unroll
  for (int index = 0; index < FLASHRT_RDNA_WMMA_LANE_K / 4; ++index) {
    fragment.words[index] = source[index];
  }
  return fragment.values;
}

__device__ __forceinline__ RdnaWmmaAccumulator
rdna_wmma_bf16(RdnaWmmaFragment a, RdnaWmmaFragment b,
               RdnaWmmaAccumulator c) {
#if defined(FLASHRT_AMD_RDNA4)
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
}

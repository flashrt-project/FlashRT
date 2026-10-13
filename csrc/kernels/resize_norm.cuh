// Generic uint8 image resize + rescale-to-bf16 preprocessing.
//
// Reproduces the preprocessing a cpp/opencv front-end performs on device, so a
// captured graph can start at raw uint8 frames instead of a host-side resize:
// bilinear resize on uint8 with OpenCV's INTER_LINEAR arithmetic (11-bit
// fixed-point coefficients and its SIMD vertical combine), then rescale and
// normalize to bf16 in the layout a vision tower reads.
//
// The arithmetic is reproduced deliberately rather than approximated. OpenCV's
// scalar FixedPtCast and its bit-exact INTER_LINEAR_EXACT resize differ from
// INTER_LINEAR on roughly 8% of bytes, so a generic bilinear resize is not a
// substitute for a caller whose reference is cv2.
//
//   in:  (nv, H, W, 3) uint8, interleaved HWC
//   out: (nv, 3, OH, OW) bf16, channel-planar CHW
//
// The rescale is the fixed (x - 127.5) / 127.5, i.e. a 1/255 rescale followed by
// a mean-0.5 std-0.5 normalize fused into one step, as a typical image processor
// expresses it.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

void resize_norm_u8_hwc_to_bf16_chw(const uint8_t* in, __nv_bfloat16* out,
                                    int nv, int H, int W, int OH, int OW,
                                    cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

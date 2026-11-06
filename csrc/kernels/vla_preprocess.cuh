// SPDX-License-Identifier: Apache-2.0
// VLA-JEPA boundary preprocessing (see vla_preprocess.cu for the semantics).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace flash_rt {
namespace kernels {

// [views, H, W, C] uint8 frames -> [views, C, oh, ow] fp32, area (adaptive
// average) resize with the /255 rescale folded into the load.
// `ybounds`/`xbounds` are the host-built pooling window tables, one int2
// {start, end} per output index (see the .cu for why they are not computed
// in-kernel).
void vla_frames_area_resize(const uint8_t* frames, float* out,
                            const void* ybounds, const void* xbounds, int views,
                            int H, int W, int C, int oh, int ow,
                            cudaStream_t stream);

// [n, ih, iw] fp32 -> [n, oh, ow] fp32, bicubic (align_corners=false).
void vla_frames_bicubic_resize(const float* src, float* out, int n, int ih,
                               int iw, int oh, int ow, cudaStream_t stream);

// vla_frames_bicubic_resize + vla_frames_patchify in one launch (identical
// output). [views * C, ih, iw] fp32 -> patchified bf16 rows. Returns false
// (nothing launched) outside the instantiated geometry (ps 16, tps 2, m 2).
bool vla_frames_bicubic_patchify(const float* src, __nv_bfloat16* out, int views,
                                 int C, int ih, int iw, int oh, int ow, int tps,
                                 int ps, int merge, float mean, float std_,
                                 cudaStream_t stream);

// [views, C, H, W] fp32 -> patchified bf16 rows (the processor's order).
void vla_frames_patchify(const float* src, __nv_bfloat16* out, int views, int C,
                         int H, int W, int tps, int ps, int merge, float mean,
                         float std_, cudaStream_t stream);

// (raw - mean) / (std + eps) -> bf16.
void vla_state_normalize(const float* raw, const float* mean, const float* std_,
                         float eps, int n, __nv_bfloat16* out,
                         cudaStream_t stream);

// LeRobot action post-processing for the delivered (rows, cols) chunk:
// clip -> gripper pre-snap -> MIN_MAX inverse -> gripper binarize, in fp32.
void vla_action_postprocess(const __nv_bfloat16* actions, float* out,
                            const float* amin, const float* amax, int rows,
                            int cols, int gripper_dim, float gripper_threshold,
                            cudaStream_t stream);

// DiT step input: x[0:head_rows] = head, x[head_rows:head_rows+act_rows] += pos
// (bf16 [rows, dim]); PDL-launched. Returns cudaError_t as int.
int vla_dit_step_rows(__nv_bfloat16* x, const __nv_bfloat16* head, const __nv_bfloat16* pos,
                      int head_rows, int act_rows, int dim, cudaStream_t stream);

}  // namespace kernels
}  // namespace flash_rt

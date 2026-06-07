#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kBlock = 256;
constexpr int kImageHw = 224;
constexpr int kPatch = 14;
constexpr int kPatchesPerSide = 16;
constexpr int kPatchesPerView = kPatchesPerSide * kPatchesPerSide;
constexpr int kChannels = 3;
constexpr int kPatchFlat = kPatch * kPatch * kChannels;

__global__ void patch_im2col_u16_kernel(const uint16_t* input,
                                        uint16_t* output,
                                        int nv) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = nv * kPatchesPerView * kPatchFlat;
  if (idx >= total) {
    return;
  }

  const int patch_idx = idx / kPatchFlat;
  const int feat_idx = idx % kPatchFlat;

  const int batch = patch_idx / kPatchesPerView;
  const int local_patch = patch_idx % kPatchesPerView;
  const int ph = local_patch / kPatchesPerSide;
  const int pw = local_patch % kPatchesPerSide;

  const int pxh = feat_idx / (kPatch * kChannels);
  const int pxw = (feat_idx % (kPatch * kChannels)) / kChannels;
  const int c = feat_idx % kChannels;

  const int row = ph * kPatch + pxh;
  const int col = pw * kPatch + pxw;
  const int src = batch * (kImageHw * kImageHw * kChannels) +
                  row * (kImageHw * kChannels) +
                  col * kChannels + c;
  output[idx] = input[src];
}

__global__ void patch_embed_bias_pos_bf16_kernel(
    hip_bfloat16* output,
    const hip_bfloat16* bias,
    const hip_bfloat16* pos_emb,
    int s,
    int d,
    int s_per_view) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = s * d;
  if (idx >= total) {
    return;
  }

  const int row = idx / d;
  const int col = idx % d;
  const int pos_row = row % s_per_view;
  const float v = static_cast<float>(output[idx]) +
                  static_cast<float>(bias[col]) +
                  static_cast<float>(pos_emb[pos_row * d + col]);
  output[idx] = hip_bfloat16(v);
}

}  // namespace

void launch_patch_im2col_u16(const void* input, void* output, int nv,
                             hipStream_t stream) {
  if (nv <= 0) {
    return;
  }
  const int total = nv * kPatchesPerView * kPatchFlat;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(patch_im2col_u16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<const uint16_t*>(input),
                     static_cast<uint16_t*>(output),
                     nv);
}

void launch_patch_embed_bias_pos_bf16(void* output, const void* bias,
                                      const void* pos_emb, int s, int d,
                                      int s_per_view, hipStream_t stream) {
  if (s <= 0 || d <= 0 || s_per_view <= 0) {
    return;
  }
  const int total = s * d;
  const int blocks = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(patch_embed_bias_pos_bf16_kernel,
                     dim3(blocks), dim3(kBlock), 0, stream,
                     static_cast<hip_bfloat16*>(output),
                     static_cast<const hip_bfloat16*>(bias),
                     static_cast<const hip_bfloat16*>(pos_emb),
                     s, d, s_per_view);
}

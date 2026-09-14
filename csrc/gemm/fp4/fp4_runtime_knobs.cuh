#pragma once
// Process-wide runtime knobs for the forked NVFP4 GEMM mainloops (read at launch, baked into the
// kernel params, so a CUDA graph captures whatever was set before capture).
namespace flash_rt {
namespace fp4 {
void set_weight_evict_first(int on);
int get_weight_evict_first();
}  // namespace fp4
}  // namespace flash_rt

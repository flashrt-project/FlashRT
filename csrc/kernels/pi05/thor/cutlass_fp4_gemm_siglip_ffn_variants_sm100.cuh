// Tile-shape variants of the SigLIP FFN NVFP4 GEMMs (see the base header).
// idx 0 reproduces the base kernels' tiles.
#pragma once
#include <cuda_runtime.h>
namespace flash_rt {
namespace fp4 {
int cutlass_fp4_gemm_bias_gelu_fp4out_v(int idx,
    void const* A_packed, void const* SFA, void const* B_packed, void const* SFB,
    void const* bias_fp16, void* D_packed, void* D_SFD, int M, int N, int K,
    cudaStream_t stream);
int cutlass_fp4_gemm_bias_res_fp16_v(int idx,
    void const* A_packed, void const* SFA, void const* B_packed, void const* SFB,
    void const* bias_fp16, void const* C_fp16, void* D_fp16, int M, int N, int K,
    cudaStream_t stream);
const char* siglip_ffn_variant_name(int which, int idx);   // which: 0 up, 1 down
}  // namespace fp4
}  // namespace flash_rt

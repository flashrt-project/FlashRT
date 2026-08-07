// FlashRT — Hy-VLA single-CTA dynamic per-tensor FP8 quant (Thor SM110).
//
// Collapses quantize_fp8_device's 4 graph nodes (memset + absmax +
// compute_scale + quantize) into ONE single-block kernel. For the expert
// denoise tower (M=41) every activation tensor is <=84K elements, so a single
// CTA holds it: pass 1 reduces the per-tensor amax, pass 2 casts to e4m3. No
// cross-block atomics -> fully deterministic -> trivially CUDA-graph-safe.
//
// Numerics are bit-identical to quantize_fp8_device:
//   amax  = max|x| over n>>1 bf16 pairs   (odd tail dropped, as in reference)
//   scale = max(amax/448, 1e-12)
//   out[i]= e4m3( clamp(x[i] * (1/scale), +-448) )
#include "common.cuh"
#include <cuda_fp8.h>

__global__ void hyvla_quant_fp8_dyn_bf16_kernel(
    const __nv_bfloat16* __restrict__ x, __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scale, int n)
{
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const int n2 = n >> 1;
    __shared__ float red[32];

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        __nv_bfloat162 v = x2[i];
        local_max = fmaxf(local_max,
                          fmaxf(fabsf(to_f32<__nv_bfloat16>(v.x)),
                                fabsf(to_f32<__nv_bfloat16>(v.y))));
    }
    float amax = block_reduce_max(local_max, red);   // broadcast to all lanes
    float sc = fmaxf(amax / 448.0f, 1e-12f);
    if (threadIdx.x == 0) *scale = sc;
    const float inv_s = 1.0f / sc;

    for (int i = threadIdx.x; i < n2; i += blockDim.x) {
        __nv_bfloat162 v = x2[i];
        float v0 = to_f32<__nv_bfloat16>(v.x) * inv_s;
        float v1 = to_f32<__nv_bfloat16>(v.y) * inv_s;
        out[2 * i]     = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        out[2 * i + 1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

extern "C" void hyvla_quant_fp8_dyn_bf16(
    const void* x, void* out, float* scale, int n, cudaStream_t stream)
{
    hyvla_quant_fp8_dyn_bf16_kernel<<<1, 512, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_fp8_e4m3*>(out), scale, n);
}

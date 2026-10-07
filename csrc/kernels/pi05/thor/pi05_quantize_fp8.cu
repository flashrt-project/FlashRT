#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "kernels/pi05/thor/pdl.cuh"

__global__ void quantize_fp8_kernel(const __half* in, __nv_fp8_e4m3* out, const float* descale_ptr, int n) {
    flashrt_pdl_wait_and_trigger();
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) return;
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    const __half2* in2 = reinterpret_cast<const __half2*>(in);
    __half2 vA = in2[i/2], vB = in2[i/2+1];
    float fv[4] = {__half2float(vA.x), __half2float(vA.y),
                   __half2float(vB.x), __half2float(vB.y)};
    __nv_fp8_e4m3 fp8_pack[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        fp8_pack[j] = __nv_fp8_e4m3(fminf(fmaxf(fv[j] * inv_scale, -448.f), 448.f));
    }
    *reinterpret_cast<uint32_t*>(out + i) = *reinterpret_cast<uint32_t*>(fp8_pack);
}

void quantize_fp8_static_fp16(const __half* input, __nv_fp8_e4m3* output,
                               const float* d_scale, int n, cudaStream_t stream) {
    // 4 elem/thread, matching production quant_fp8_static_k
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    flash_rt::fp4::launch_maybe_pdl(quantize_fp8_kernel, dim3(blocks), dim3(threads), 0, stream, input, output, d_scale, n);
}

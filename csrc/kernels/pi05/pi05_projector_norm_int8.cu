// Pi0.5 projector bias + first encoder RMSNorm + dynamic token INT8.
// Derived from csrc/kernels/norm.cu::rms_norm_int8_rowwise_kernel.
// Only visual prefix rows receive bias; language rows remain unchanged.
// Keep the BF16 round-trip and the generic kernel's reduction/math order.
#include "pi05_projector_norm_int8.cuh"
#include "../common.cuh"
#include <stdexcept>
#include <string>

namespace flash_rt {
namespace pi05 {
namespace {
__global__ void pi05_projector_norm_int8_kernel(
        __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ bias,
        const __nv_bfloat16* __restrict__ weight,
        int8_t*  __restrict__ out,
        float*   __restrict__ scales,
        int rows, int cols, int image_rows, float eps) {
    extern __shared__ float smem[];
    float* partial = smem + cols;

    int row = blockIdx.x;
    if (row >= rows) return;

    __nv_bfloat16* xr = x + (int64_t)row * cols;
    int8_t* outr = out + (int64_t)row * cols;

    // Pass 1: load x → smem, accumulate sum of squares
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        __nv_bfloat16 rounded = xr[i];
        if (row < image_rows) {
            // The old add_bias kernel stores BF16 before RMSNorm reads it.
            rounded = __float2bfloat16(
                __bfloat162float(rounded) + __bfloat162float(bias[i]));
            xr[i] = rounded;  // later encoder layers read this residual
        }
        float xi = __bfloat162float(rounded);
        smem[i] = xi;
        sum_sq += xi * xi;
    }
    float rms = rsqrtf(block_reduce_sum(sum_sq, partial) / cols + eps);

    // Pass 2: normalize (reuse smem), accumulate max_abs
    float max_abs = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * rms * __bfloat162float(weight[i]);
        smem[i] = v;
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = fmaxf(block_reduce_max(max_abs, partial) / 127.f, 1e-12f);
    if (threadIdx.x == 0) scales[row] = scale;
    float inv_s = 1.f / scale;

    // Pass 3: write INT8
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = smem[i] * inv_s;
        outr[i] = (int8_t)__float2int_rn(fmaxf(-127.f, fminf(127.f, v)));
    }
}

}  // namespace

void projector_bias_rms_norm_int8(
        __nv_bfloat16* x, const __nv_bfloat16* bias,
        const __nv_bfloat16* weight, int8_t* out, float* scales,
        int rows, int cols, int image_rows, float eps, cudaStream_t stream) {
    if (!x || !bias || !weight || !out || !scales)
        throw std::invalid_argument("pi05_projector_bias_rms_norm_int8: null pointer");
    if (rows <= 0 || cols <= 0 || cols > 8192 || cols % 2 != 0 ||
        image_rows < 0 || image_rows > rows || !(eps > 0.0f))
        throw std::invalid_argument(
            "pi05_projector_bias_rms_norm_int8: require rows>0, even cols in [2,8192], "
            "0<=image_rows<=rows and eps>0");
    pi05_projector_norm_int8_kernel<<<rows,256,(cols+32)*sizeof(float),stream>>>(
        x,bias,weight,out,scales,rows,cols,image_rows,eps);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(
            std::string("pi05_projector_bias_rms_norm_int8: ") + cudaGetErrorString(status));
}

}  // namespace pi05
}  // namespace flash_rt

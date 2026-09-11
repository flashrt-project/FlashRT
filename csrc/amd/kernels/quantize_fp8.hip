// ================================================================
// FlashRT AMD — Quantization kernels (dtype-generic FP8, CDNA wave64)
// FP8 static/device-only quantize + scale-max accumulate.
// Mirrors csrc/kernels/quantize.cu host-wrapper signatures exactly so
// the pipeline layer stays portable text across platforms.
//
// FP8 dtype: OCP e4m3 via <hip/hip_fp8.h> __hip_fp8_e4m3 — the same
// byte format the hipBLASLt GEMM side consumes as HIP_R_8F_E4M3.
// __hip_fp8_e4m3(float) defaults to __HIP_SATFINITE + __HIP_E4M3,
// matching the __nv_fp8_e4m3(float) saturating round-to-nearest-even
// conversion of the CUDA original. Values are pre-clamped to ±448
// (the e4m3 finite max) before conversion, exactly as in quantize.cu.
//
// Deviations from quantize.cu (noted, not silent):
//   - only the deployed pi05 surface is ported: quantize_fp8_static,
//     quantize_fp8_device (+ absmax/compute_scale helpers) and
//     fp8_accumulate_scale_max. The alloc-per-call quantize_fp8,
//     dequant helpers, L2 prefetch and NVFP4 sections are CUDA-only.
//   - absmax_kernel's intra-block reduction is wave64 (CUDA warp32);
//     max-reduction order does not affect the result.
// ================================================================

#include "common_hip.h"
#include <hip/hip_fp8.h>

// ── FP8 Quantize ──

template<typename T>
__global__ void absmax_kernel(const T* __restrict__ x, float* max_val, int n) {
    using T2 = typename packed2<T>::type;
    extern __shared__ float shared[];
    const T2* x2 = reinterpret_cast<const T2*>(x);
    int n2 = n >> 1;
    float local_max = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n2; i += gridDim.x * blockDim.x) {
        T2 val = x2[i];
        local_max = fmaxf(local_max, fmaxf(fabsf(to_f32(val.x)), fabsf(to_f32(val.y))));
    }
    // Wave-level max reduction (wave64)
    local_max = wave_reduce_max(local_max);
    int lane = threadIdx.x & (FVK_WAVE_SIZE - 1);
    int wave_id = threadIdx.x >> 6;
    if (lane == 0) shared[wave_id] = local_max;
    __syncthreads();
    int num_waves = blockDim.x >> 6;
    local_max = (threadIdx.x < num_waves) ? shared[threadIdx.x] : 0.0f;
    if (wave_id == 0) local_max = wave_reduce_max(local_max);
    // absmax >= 0, so int-bit atomicMax orders identically to float max
    if (threadIdx.x == 0) atomicMax((int*)max_val, __float_as_int(local_max));
}

template __global__ void absmax_kernel<__half>(const __half*, float*, int);
template __global__ void absmax_kernel<__hip_bfloat16>(const __hip_bfloat16*, float*, int);

template<typename T>
__global__ void quantize_fp8_kernel_generic(const T* __restrict__ input,
                                     __hip_fp8_e4m3* __restrict__ output,
                                     const float* scale, int n) {
    using T2 = typename packed2<T>::type;
    const T2* in2 = reinterpret_cast<const T2*>(input);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        float inv_s = 1.0f / (*scale);
        T2 val = in2[idx];
        float v0 = to_f32(val.x) * inv_s, v1 = to_f32(val.y) * inv_s;
        output[2*idx]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        output[2*idx+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

template __global__ void quantize_fp8_kernel_generic<__half>(const __half*, __hip_fp8_e4m3*, const float*, int);
template __global__ void quantize_fp8_kernel_generic<__hip_bfloat16>(const __hip_bfloat16*, __hip_fp8_e4m3*, const float*, int);

// ── FP8 Quantize Device-Only (HIP Graph compatible) ──
__global__ void compute_scale_kernel(const float* d_absmax, float* d_scale) {
    float amax = *d_absmax;
    float scale = amax / 448.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    *d_scale = scale;
}

__global__ void fp8_accumulate_scale_max_kernel(const float* src_scale,
                                                float* dst_scale) {
    float scale = *src_scale;
    atomicMax(reinterpret_cast<int*>(dst_scale), __float_as_int(scale));
}

void fp8_accumulate_scale_max(const float* src_scale, float* dst_scale,
                              hipStream_t stream) {
    fp8_accumulate_scale_max_kernel<<<1, 1, 0, stream>>>(src_scale, dst_scale);
}

// ── Wide variant (CDNA retune): dwordx4 load (8 bf16), packed 8-byte
// FP8 store. Same per-element math as quantize_fp8_kernel_generic —
// bit-exact (verified by the ew_tune probe).
struct alignas(16) qz_bf16x8 { __hip_bfloat162 p[4]; };
struct alignas(8) qz_fp8x8 { unsigned char b[8]; };

__global__ void quantize_fp8_kernel_wide8(const qz_bf16x8* __restrict__ in,
                                          qz_fp8x8* __restrict__ out,
                                          const float* __restrict__ scale,
                                          int nchunks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nchunks) return;
    qz_bf16x8 v = in[idx];
    float inv_s = 1.0f / (*scale);
    qz_fp8x8 o;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float v0 = __bfloat162float(v.p[k].x) * inv_s;
        float v1 = __bfloat162float(v.p[k].y) * inv_s;
        o.b[2*k]   = __hip_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f)).__x;
        o.b[2*k+1] = __hip_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f)).__x;
    }
    out[idx] = o;
}

// Static FP8 quantize: uses pre-computed scale on device (no absmax, no reduction)
void quantize_fp8_static(const __hip_bfloat16* input, __hip_fp8_e4m3* output,
                          const float* d_scale, int n, hipStream_t stream) {
    if (n % 8 == 0 && ((uintptr_t)input & 15) == 0 && ((uintptr_t)output & 7) == 0) {
        int nchunks = n >> 3;
        quantize_fp8_kernel_wide8<<<(nchunks + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const qz_bf16x8*>(input),
            reinterpret_cast<qz_fp8x8*>(output), d_scale, nchunks);
        return;
    }
    int n2 = n >> 1;
    int threads = 256;
    int blocks = (n2 + threads - 1) / threads;
    quantize_fp8_kernel_generic<__hip_bfloat16><<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}

void quantize_fp8_device(const __hip_bfloat16* input, __hip_fp8_e4m3* output,
                          float* d_scale, int n, hipStream_t stream) {
    hipMemsetAsync(d_scale, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    int smem = (threads / FVK_WAVE_SIZE) * sizeof(float);
    absmax_kernel<__hip_bfloat16><<<blocks, threads, smem, stream>>>(input, d_scale, n);

    compute_scale_kernel<<<1, 1, 0, stream>>>(d_scale, d_scale);

    int n2 = n >> 1;
    blocks = (n2 + threads - 1) / threads;
    quantize_fp8_kernel_generic<__hip_bfloat16><<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}

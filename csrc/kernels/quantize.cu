// ================================================================
// FlashRT — Quantization kernels (dtype-generic FP8, BF16-only NVFP4)
// FP8 dynamic/static, device-only, L2 prefetch, NVFP4 (SM120+)
// FP8 kernels support: __half (FP16), __nv_bfloat16 (BF16) input
// ================================================================

#include "quantize.cuh"
#include "common.cuh"


// ── FP8 Quantize ──
// Two-pass: 1) find abs max, 2) scale and convert

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
    // Warp-level max reduction
    local_max = warp_reduce_max(local_max);
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) shared[warp_id] = local_max;
    __syncthreads();
    int num_warps = blockDim.x >> 5;
    local_max = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) local_max = warp_reduce_max(local_max);
    if (threadIdx.x == 0) atomicMax((int*)max_val, __float_as_int(local_max));
}

template __global__ void absmax_kernel<__half>(const __half*, float*, int);
template __global__ void absmax_kernel<__nv_bfloat16>(const __nv_bfloat16*, float*, int);

// Verbatim production quant_fp8_static_k: 4 elem/thread, packed uint32 store
__global__ void quantize_fp8_kernel(const __half* in, __nv_fp8_e4m3* out, const float* descale_ptr, int n) {
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

// Keep BF16 template for non-encoder paths
template<typename T>
__global__ void quantize_fp8_kernel_generic(const T* __restrict__ input,
                                     __nv_fp8_e4m3* __restrict__ output,
                                     const float* scale, int n) {
    using T2 = typename packed2<T>::type;
    const T2* in2 = reinterpret_cast<const T2*>(input);
    int n2 = n >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        float inv_s = 1.0f / (*scale);
        T2 val = in2[idx];
        float v0 = to_f32(val.x) * inv_s, v1 = to_f32(val.y) * inv_s;
        output[2*idx]   = __nv_fp8_e4m3(fminf(fmaxf(v0, -448.0f), 448.0f));
        output[2*idx+1] = __nv_fp8_e4m3(fminf(fmaxf(v1, -448.0f), 448.0f));
    }
}

template __global__ void quantize_fp8_kernel_generic<__half>(const __half*, __nv_fp8_e4m3*, const float*, int);
template __global__ void quantize_fp8_kernel_generic<__nv_bfloat16>(const __nv_bfloat16*, __nv_fp8_e4m3*, const float*, int);

float quantize_fp8(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                   float* d_scale, int n, cudaStream_t stream) {
    float* d_max;
    cudaMalloc(&d_max, sizeof(float));
    cudaMemset(d_max, 0, sizeof(float));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    absmax_kernel<__nv_bfloat16><<<blocks, threads, threads * sizeof(float), stream>>>(input, d_max, n);

    float h_max;
    cudaMemcpyAsync(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float scale = h_max / 448.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    cudaMemcpyAsync(d_scale, &scale, sizeof(float), cudaMemcpyHostToDevice, stream);

    int n2 = n >> 1;
    blocks = (n2 + threads - 1) / threads;
    quantize_fp8_kernel_generic<__nv_bfloat16><<<blocks, threads, 0, stream>>>(input, output, d_scale, n);

    cudaFree(d_max);
    return scale;
}

float quantize_fp8_fp16(const __half* input, __nv_fp8_e4m3* output,
                        float* d_scale, int n, cudaStream_t stream) {
    float* d_max;
    cudaMalloc(&d_max, sizeof(float));
    cudaMemset(d_max, 0, sizeof(float));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    absmax_kernel<__half><<<blocks, threads, threads * sizeof(float), stream>>>(input, d_max, n);

    float h_max;
    cudaMemcpyAsync(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    float scale = h_max / 448.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    cudaMemcpyAsync(d_scale, &scale, sizeof(float), cudaMemcpyHostToDevice, stream);

    int n2 = n >> 1;
    blocks = (n2 + threads - 1) / threads;
    quantize_fp8_kernel_generic<__half><<<blocks, threads, 0, stream>>>(input, output, d_scale, n);

    cudaFree(d_max);
    return scale;
}

// ── FP8 Quantize Device-Only (CUDA Graph compatible) ──
__global__ void compute_scale_kernel(const float* d_absmax, float* d_scale) {
    float amax = *d_absmax;
    float scale = amax / 448.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    *d_scale = scale;
}

// Static FP8 quantize: uses pre-computed scale on device (no absmax, no reduction)
void quantize_fp8_static(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                          const float* d_scale, int n, cudaStream_t stream) {
    int n2 = n >> 1;
    int threads = 256;
    int blocks = (n2 + threads - 1) / threads;
    quantize_fp8_kernel_generic<__nv_bfloat16><<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}
void quantize_fp8_static_fp16(const __half* input, __nv_fp8_e4m3* output,
                               const float* d_scale, int n, cudaStream_t stream) {
    // 4 elem/thread, matching production quant_fp8_static_k
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    quantize_fp8_kernel<<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}

// ── L2 Weight Prefetch ──
// Issues PTX prefetch.global.L2 hints for weight data, pulling it into L2 cache.
// Runs before the next GEMM so weights are warm when cuBLASLt reads them.
// Lightweight: no compute, just address hints to memory controller.
__global__ void prefetch_l2_kernel(const char* __restrict__ data, int num_cachelines) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_cachelines) {
        const char* addr = data + idx * 128;  // L2 cacheline = 128 bytes
        asm volatile("prefetch.global.L2 [%0];" :: "l"(addr));
    }
}

void prefetch_l2(const void* data, size_t num_bytes, cudaStream_t stream) {
    int num_cachelines = (num_bytes + 127) / 128;
    int threads = 256;
    int blocks = (num_cachelines + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    prefetch_l2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const char*>(data), num_cachelines);
}

// gate_residual_ada_norm_fp8 is defined in fusion.cu

void quantize_fp8_device(const __nv_bfloat16* input, __nv_fp8_e4m3* output,
                          float* d_scale, int n, cudaStream_t stream) {
    cudaMemsetAsync(d_scale, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    absmax_kernel<__nv_bfloat16><<<blocks, threads, threads * sizeof(float), stream>>>(input, d_scale, n);

    compute_scale_kernel<<<1, 1, 0, stream>>>(d_scale, d_scale);

    int n2 = n >> 1;
    blocks = (n2 + threads - 1) / threads;
    quantize_fp8_kernel_generic<__nv_bfloat16><<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}

void quantize_fp8_device_fp16(const __half* input, __nv_fp8_e4m3* output,
                               float* d_scale, int n, cudaStream_t stream) {
    cudaMemsetAsync(d_scale, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    absmax_kernel<__half><<<blocks, threads, threads * sizeof(float), stream>>>(input, d_scale, n);

    compute_scale_kernel<<<1, 1, 0, stream>>>(d_scale, d_scale);

    int n2 = n >> 1;
    blocks = (n2 + threads - 1) / threads;
    quantize_fp8_kernel_generic<__half><<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}

// ── INT8 dynamic quant + dequant (graph-compatible, BF16 activations) ──

__global__ void compute_scale_int8_kernel(const float* d_absmax, float* d_scale) {
    float amax = *d_absmax;
    float scale = amax / 127.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    *d_scale = scale;
}

template<typename T>
__global__ void quantize_int8_kernel_generic(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float inv_s = 1.0f / fmaxf(*scale, 1e-12f);
    float v = to_f32(input[idx]) * inv_s;
    int q = __float2int_rn(v);
    q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
    output[idx] = static_cast<int8_t>(q);
}

template __global__ void quantize_int8_kernel_generic<__nv_bfloat16>(
    const __nv_bfloat16*, int8_t*, const float*, int);

void quantize_int8_device(const __nv_bfloat16* input, int8_t* output,
                          float* d_scale, int n, cudaStream_t stream) {
    cudaMemsetAsync(d_scale, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    absmax_kernel<__nv_bfloat16><<<blocks, threads, threads * sizeof(float), stream>>>(
        input, d_scale, n);

    compute_scale_int8_kernel<<<1, 1, 0, stream>>>(d_scale, d_scale);
    blocks = (n + threads - 1) / threads;
    quantize_int8_kernel_generic<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        input, output, d_scale, n);
}

// ── Static INT8 quantization (pre-calibrated scale, no amax reduction) ──
//
// Drop-in replacement for quantize_int8_device when d_scale has already
// been calibrated offline. Skips the 2-kernel amax+scale pass and runs
// only the element-wise quantize kernel → 1 launch vs 3.
// CUDA Graph compatible (all ops are pure device-side).
void quantize_int8_static(const __nv_bfloat16* input, int8_t* output,
                           const float* d_scale, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    quantize_int8_kernel_generic<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
        input, output, d_scale, n);
}

__global__ void quantize_int8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* in_row = input + static_cast<size_t>(row) * cols;
    int8_t* out_row = output + static_cast<size_t>(row) * cols;

    float tmax = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        tmax = fmaxf(tmax, fabsf(__bfloat162float(in_row[j])));
    }

    for (int off = 16; off > 0; off >>= 1) {
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
    }

    __shared__ float warp_max[8];
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;
    if (lid == 0) {
        warp_max[wid] = tmax;
    }
    __syncthreads();

    if (wid == 0) {
        tmax = (lid < (blockDim.x >> 5)) ? warp_max[lid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off));
        }
    }

    __shared__ float scale_s;
    if (threadIdx.x == 0) {
        float s = fmaxf(tmax / 127.0f, 1e-10f);
        scales[row] = s;
        scale_s = s;
    }
    __syncthreads();

    float inv_s = 1.0f / scale_s;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __bfloat162float(in_row[j]) * inv_s;
        int q = __float2int_rn(v);
        q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
        out_row[j] = static_cast<int8_t>(q);
    }
}

void quantize_int8_rowwise(const __nv_bfloat16* input, int8_t* output,
                           float* d_scales, int rows, int cols,
                           cudaStream_t stream) {
    int threads = (cols < 256) ? cols : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    quantize_int8_rowwise_kernel<<<rows, threads, 0, stream>>>(
        input, output, d_scales, rows, cols);
}

// ── Static per-row INT8 quantization (pre-calibrated scales, single-pass) ──
//
// Drop-in replacement for quantize_int8_rowwise when the per-row scale
// buffer has been pre-filled at calibration time. Skips the per-row
// amax reduction (warp shuffle + cross-warp shared-memory merge) and
// the second pass over global memory; reads each input element once,
// quantizes against the pre-computed scale, writes once.
//
// Memory traffic per row: 1 BF16 read (cols * 2 B) + 1 INT8 write
// (cols * 1 B) = 3*cols B  vs the dynamic version's 5*cols B.
// Compute per row: 1 fmax (clamp), 1 mul, 1 cvt — no warp shuffles.
//
// Calibration must guarantee the per-row scales bound the per-call
// activation magnitude (max over calibration samples). The frontend
// can either:
//   (a) freeze the dynamic per-row scales from one calibration call
//       (works when row N's distribution is stable across calls — true
//       for prompt rows, approximately true for vision rows on stable
//       camera setups), or
//   (b) fill all rows with a single per-tensor max scalar (loses some
//       per-row precision but trivially safe).
__global__ void quantize_int8_rowwise_static_kernel(
        const __nv_bfloat16* __restrict__ input,
        int8_t*  __restrict__ output,
        const float* __restrict__ scales,   // (rows,) — one scalar per row
        int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* in_row = input + static_cast<size_t>(row) * cols;
    int8_t* out_row = output + static_cast<size_t>(row) * cols;

    // Read scale once per block via __ldg (treated as constant during
    // this kernel call). The 1e-12f floor mirrors the dynamic kernel
    // and guards against degenerate calibration values.
    float inv_s = 1.0f / fmaxf(__ldg(&scales[row]), 1e-12f);

    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = __bfloat162float(in_row[j]) * inv_s;
        int q = __float2int_rn(v);
        q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
        out_row[j] = static_cast<int8_t>(q);
    }
}

void quantize_int8_rowwise_static(const __nv_bfloat16* input, int8_t* output,
                                   const float* d_scales, int rows, int cols,
                                   cudaStream_t stream) {
    int threads = (cols < 256) ? cols : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    quantize_int8_rowwise_static_kernel<<<rows, threads, 0, stream>>>(
        input, output, d_scales, rows, cols);
}

__global__ void dequant_int32_to_bf16_kernel(
    const int32_t* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ d_act_scale,
    const float* __restrict__ d_weight_scale,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float scale = (*d_act_scale) * (*d_weight_scale);
    output[idx] = __float2bfloat16(static_cast<float>(input[idx]) * scale);
}

void dequant_int32_to_bf16(const int32_t* input, __nv_bfloat16* output,
                           const float* d_act_scale, const float* d_weight_scale,
                           int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dequant_int32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        input, output, d_act_scale, d_weight_scale, n);
}

#ifdef ENABLE_NVFP4
// ================================================================
//  NVFP4 (E2M1) Quantization with per-16-block UE4M3 scale factors
//  For cuBLASLt block-scaled GEMM on Blackwell (sm_120)
//
//  Row-major data (rows, cols) is interpreted by cuBLASLt as
//  col-major (cols, rows) with ld=cols. Block scaling is per 16
//  contiguous elements along the leading dim (cols), i.e., within
//  each row of the original row-major data.
//
//  FP4 E2M1 values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
//  UE4M3 (unsigned E4M3): scale factor per 16-element block
// ================================================================

// FP4 E2M1 value table (magnitude only, 3 bits):
//   0b000 = 0.0   (E=0, M=0)
//   0b001 = 0.5   (E=0, M=1, subnormal)
//   0b010 = 1.0   (E=1, M=0)
//   0b011 = 1.5   (E=1, M=1)
//   0b100 = 2.0   (E=2, M=0)
//   0b101 = 3.0   (E=2, M=1)
//   0b110 = 4.0   (E=3, M=0)
//   0b111 = 6.0   (E=3, M=1)

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float v) {
    uint8_t sign = (v < 0.0f) ? 0x8u : 0x0u;
    float a = fabsf(v);
    uint8_t mag;
    if      (a < 0.25f)  mag = 0;  // -> 0.0
    else if (a < 0.75f)  mag = 1;  // -> 0.5
    else if (a < 1.25f)  mag = 2;  // -> 1.0
    else if (a < 1.75f)  mag = 3;  // -> 1.5
    else if (a < 2.5f)   mag = 4;  // -> 2.0
    else if (a < 3.5f)   mag = 5;  // -> 3.0
    else if (a < 5.0f)   mag = 6;  // -> 4.0
    else                 mag = 7;  // -> 6.0
    return sign | mag;
}

// Convert float to UE4M3 (unsigned, 4-bit exponent, 3-bit mantissa)
// Rounds UP (ceil) so that scale >= true_amax / 6.0 (avoids FP4 overflow)
// UE4M3: bias=7, normal = 2^(E-7) * (1 + M/8), subnormal = 2^(-6) * M/8
// Range: [~0.002, 240]
__device__ __forceinline__ uint8_t float_to_ue4m3_ceil(float v) {
    if (v <= 0.0f) return 0;
    if (v > 240.0f) return 0xFE;  // max finite: E=14, M=7 -> 2^7 * 1.875 = 240

    uint32_t bits = __float_as_uint(v);
    int float_exp = ((bits >> 23) & 0xFF) - 127;  // unbiased float exponent
    uint32_t frac = bits & 0x7FFFFF;               // 23-bit float mantissa

    int ue_exp = float_exp + 7;  // UE4M3 bias = 7

    if (ue_exp <= 0) {
        // Subnormal in UE4M3: value = 2^(-6) * M/8
        float scaled = v * 512.0f;  // v / (2^(-6) / 8)
        int m = (int)ceilf(scaled);
        if (m > 7) return (1 << 3) | 0;  // smallest normal: E=1, M=0
        if (m < 1) m = 1;
        return (uint8_t)m;
    }
    if (ue_exp >= 15) return 0xFE;  // clamp to max

    // Extract top 3 mantissa bits, round up
    int m = (int)(frac >> 20);  // top 3 of 23 bits
    if (frac & 0xFFFFF) m++;    // ceil: round up if remaining bits nonzero
    if (m >= 8) { m = 0; ue_exp++; }
    if (ue_exp >= 15) return 0xFE;

    return (uint8_t)((ue_exp << 3) | m);
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
    int e = (v >> 3) & 0xF;
    int m = v & 0x7;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// UE8M0 conversion: 8-bit unsigned exponent, 0 mantissa bits, bias=127
// value = 2^(exp - 127), same as IEEE FP32 exponent extraction
// Used for MX-FP8/MX-FP4 scale factors (CUTLASS mx_float*_t types)
__device__ __forceinline__ uint8_t float_to_ue8m0_ceil(float v) {
    if (v <= 0.0f) return 0;  // zero/negative -> smallest
    uint32_t bits = __float_as_uint(v);
    uint8_t exp = (bits >> 23) & 0xFF;  // extract exponent byte
    uint32_t mant = bits & 0x7FFFFF;    // mantissa
    // Round up: if mantissa > 0 and not at max exponent, increment
    if (mant > 0 && exp < 0xFE) exp++;
    return exp;
}

__device__ __forceinline__ float ue8m0_to_float(uint8_t exp) {
    // Reconstruct as FP32: sign=0, exp=exp, mantissa=0
    uint32_t bits = (uint32_t)exp << 23;
    return __uint_as_float(bits);
}

// Quantize one row of BF16 data to NVFP4 with per-16-block UE4M3 scales
// Each thread block handles one row. 256 threads, each processes dim/256 elements.
// Two passes: (1) compute per-block amax, (2) quantize + pack.
__global__ void quantize_bf16_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,   // row-major (rows, cols)
    uint8_t* __restrict__ fp4_data,            // packed FP4, size = rows*cols/2
    uint8_t* __restrict__ scale_factors,        // UE4M3, row-major (rows, num_blocks)
    int cols, int num_blocks)
{
    int row = blockIdx.x;
    const __nv_bfloat16* row_in = input + (size_t)row * cols;
    uint8_t* row_fp4 = fp4_data + (size_t)row * cols / 2;
    uint8_t* row_sf = scale_factors + (size_t)row * num_blocks;

    // Shared memory: per-block amax values (max num_blocks)
    extern __shared__ float smem[];  // size = num_blocks floats

    // Pass 1: Compute per-16-block amax
    // Initialize shared memory
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
        smem[b] = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = fabsf(__bfloat162float(row_in[i]));
        int blk = i >> 4;  // i / 16
        atomicMax((int*)&smem[blk], __float_as_int(val));  // works for positive floats
    }
    __syncthreads();

    // Pass 1.5: Convert amax to UE4M3 scale factors and store
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float amax = __int_as_float(*(int*)&smem[b]);
        float scale = amax / 6.0f;  // max FP4 E2M1 = 6.0
        uint8_t ue_scale = float_to_ue4m3_ceil(scale);
        row_sf[b] = ue_scale;
        // Store decoded scale back for Pass 2
        smem[b] = ue4m3_to_float(ue_scale);
    }
    __syncthreads();

    // Pass 2: Quantize to FP4 and pack (2 values per byte)
    // Process pairs of elements
    int half_cols = cols >> 1;
    for (int p = threadIdx.x; p < half_cols; p += blockDim.x) {
        int i = p * 2;  // element index
        int blk = i >> 4;
        float scale = smem[blk];
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        float v0 = __bfloat162float(row_in[i]) * inv_scale;
        float v1 = __bfloat162float(row_in[i + 1]) * inv_scale;

        // Handle block boundary: i and i+1 might be in different blocks
        int blk1 = (i + 1) >> 4;
        if (blk1 != blk) {
            float scale1 = smem[blk1];
            float inv1 = (scale1 > 0.0f) ? (1.0f / scale1) : 0.0f;
            v1 = __bfloat162float(row_in[i + 1]) * inv1;
        }

        uint8_t fp4_lo = float_to_fp4_e2m1(v0);
        uint8_t fp4_hi = float_to_fp4_e2m1(v1);
        row_fp4[p] = (fp4_hi << 4) | (fp4_lo & 0x0F);
    }
}

void quantize_bf16_to_nvfp4(const __nv_bfloat16* input, uint8_t* fp4_data,
                              uint8_t* scale_factors, int rows, int cols,
                              cudaStream_t stream) {
    int num_blocks = (cols + 15) / 16;
    int threads = 256;
    int smem_size = num_blocks * sizeof(float);
    quantize_bf16_to_nvfp4_kernel<<<rows, threads, smem_size, stream>>>(
        input, fp4_data, scale_factors, cols, num_blocks);
}

// ── Swizzled variant: writes scale factors directly in cuBLASLt blocked layout ──
__global__ void quantize_bf16_to_nvfp4_swizzled_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ fp4_data,
    uint8_t* __restrict__ scale_factors,  // swizzled output
    int cols, int num_blocks,
    int n_row_blocks, int n_col_blocks)  // padded block counts
{
    int row = blockIdx.x;
    const __nv_bfloat16* row_in = input + (size_t)row * cols;
    uint8_t* row_fp4 = fp4_data + (size_t)row * cols / 2;

    extern __shared__ float smem[];

    // Pass 1: Compute per-16-block amax
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
        smem[b] = 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = fabsf(__bfloat162float(row_in[i]));
        int blk = i >> 4;
        atomicMax((int*)&smem[blk], __float_as_int(val));
    }
    __syncthreads();

    // Pass 1.5: Convert amax to UE4M3 scales and write in swizzled layout
    int rb = row / 128;
    int ri = row % 128;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float amax = __int_as_float(*(int*)&smem[b]);
        float scale = amax / 6.0f;
        uint8_t ue_scale = float_to_ue4m3_ceil(scale);

        int cb = b / 4;
        int ci = b % 4;
        int out_idx = (rb * n_col_blocks + cb) * 512 + (ri % 32) * 16 + (ri / 32) * 4 + ci;
        scale_factors[out_idx] = ue_scale;

        smem[b] = ue4m3_to_float(ue_scale);
    }
    __syncthreads();

    // Pass 2: Quantize to FP4 and pack
    int half_cols = cols >> 1;
    for (int p = threadIdx.x; p < half_cols; p += blockDim.x) {
        int i = p * 2;
        int blk = i >> 4;
        float scale = smem[blk];
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        float v0 = __bfloat162float(row_in[i]) * inv_scale;
        float v1 = __bfloat162float(row_in[i + 1]) * inv_scale;

        int blk1 = (i + 1) >> 4;
        if (blk1 != blk) {
            float scale1 = smem[blk1];
            float inv1 = (scale1 > 0.0f) ? (1.0f / scale1) : 0.0f;
            v1 = __bfloat162float(row_in[i + 1]) * inv1;
        }

        uint8_t fp4_lo = float_to_fp4_e2m1(v0);
        uint8_t fp4_hi = float_to_fp4_e2m1(v1);
        row_fp4[p] = (fp4_hi << 4) | (fp4_lo & 0x0F);
    }
}

void quantize_bf16_to_nvfp4_swizzled(const __nv_bfloat16* input, uint8_t* fp4_data,
                                       uint8_t* scale_factors, int rows, int cols,
                                       cudaStream_t stream) {
    int num_blocks = (cols + 15) / 16;
    int n_row_blocks = (rows + 127) / 128;
    int n_col_blocks = (num_blocks + 3) / 4;
    int threads = 256;
    int smem_size = num_blocks * sizeof(float);
    quantize_bf16_to_nvfp4_swizzled_kernel<<<rows, threads, smem_size, stream>>>(
        input, fp4_data, scale_factors, cols, num_blocks, n_row_blocks, n_col_blocks);
}

// ================================================================
// FUSED: rms_norm(x) + nvfp4 swizzled-SF quant.
// Single kernel replaces (rms_norm_bf16 → quantize_bf16_to_nvfp4_swizzled).
// Reads input bf16 ONCE, writes packed FP4 + swizzled SF directly.
//
// Math (Qwen3.5 RMSNorm with (1+w) precomputed weight):
//   rms     = rsqrt( mean(x^2) + eps )
//   normed  = x * rms * weight                 (per-element)
//   then quantize 16-element blocks of `normed` to NVFP4 e2m1 + UE4M3 SF.
//
// Layout: shared memory holds normed BF16 row + per-block scale (UE4M3 →
// fp32). One thread block per row.
// ================================================================
__global__ void rms_norm_to_nvfp4_swizzled_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ rms_weight,    // (D,)
    uint8_t* __restrict__ packed,                    // (rows, D/2)
    uint8_t* __restrict__ sf_swz,
    int cols, int num_blocks, int n_col_blocks,
    float eps)
{
    const int row = blockIdx.x;
    const __nv_bfloat16* row_in = x + (size_t)row * cols;
    uint8_t* row_fp4 = packed + (size_t)row * cols / 2;

    // smem layout:
    //   [0 .. 31]                       warp-reduction scratch (block_reduce_sum)
    //   [32 .. 32+num_blocks-1]         per-16-element-block fp32 scales
    //   [32+num_blocks ..]              cols/2 bf16x2 packed normed values
    extern __shared__ float smem_dyn[];
    float* warp_red = smem_dyn;
    float* sf_smem = smem_dyn + 32;
    __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(sf_smem + num_blocks);

    // ── Phase 1: sum of squares → RMS ──
    float local_ssq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __bfloat162float(row_in[i]);
        local_ssq += v * v;
    }
    float ssq = block_reduce_sum(local_ssq, warp_red);
    const float rms = rsqrtf(ssq / cols + eps);

    // ── Phase 2: produce normed values into smem (bf16) + per-block amax ──
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
        sf_smem[b] = 0.f;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float xv = __bfloat162float(row_in[i]);
        float wv = __bfloat162float(rms_weight[i]);
        float n_fp32 = xv * rms * wv;
        __nv_bfloat16 nb = __float2bfloat16(n_fp32);
        normed[i] = nb;
        // amax over the bf16-rounded value (matches the unfused path
        // where rms_norm writes BF16 to global, then quantize reads
        // BF16 → fp32 → fabsf; bit-equivalent).
        float n_bf16 = __bfloat162float(nb);
        int blk = i >> 4;
        atomicMax((int*)&sf_smem[blk], __float_as_int(fabsf(n_bf16)));
    }
    __syncthreads();

    // ── Phase 3: per-block UE4M3 scale → swizzled SF write + dequant for pack ──
    int rb = row / 128;
    int ri = row % 128;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float amax = __int_as_float(*(int*)&sf_smem[b]);
        float scale = amax / 6.0f;
        uint8_t ue_scale = float_to_ue4m3_ceil(scale);

        int cb = b / 4;
        int ci = b % 4;
        int out_idx = (rb * n_col_blocks + cb) * 512
                      + (ri % 32) * 16 + (ri / 32) * 4 + ci;
        sf_swz[out_idx] = ue_scale;
        sf_smem[b] = ue4m3_to_float(ue_scale);
    }
    __syncthreads();

    // ── Phase 4: pack normed → FP4 e2m1 (2 per byte) ──
    int half_cols = cols >> 1;
    for (int p = threadIdx.x; p < half_cols; p += blockDim.x) {
        int i = p * 2;
        int blk = i >> 4;
        float scale = sf_smem[blk];
        float inv_scale = (scale > 0.f) ? (1.f / scale) : 0.f;

        float v0 = __bfloat162float(normed[i]) * inv_scale;
        float v1 = __bfloat162float(normed[i + 1]) * inv_scale;
        // Same-block always (16-element blocks, pairs are within block).
        uint8_t lo = float_to_fp4_e2m1(v0);
        uint8_t hi = float_to_fp4_e2m1(v1);
        row_fp4[p] = (hi << 4) | (lo & 0x0F);
    }
}

void rms_norm_to_nvfp4_swizzled_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* rms_weight,
    uint8_t* packed, uint8_t* sf_swz,
    int rows, int cols, float eps,
    cudaStream_t stream)
{
    int num_blocks = (cols + 15) / 16;
    int n_col_blocks = (num_blocks + 3) / 4;
    int threads = 256;
    // smem: 32 fp32 reduction scratch + num_blocks fp32 scales + cols bf16 normed
    size_t smem_size = 32 * sizeof(float)
                       + num_blocks * sizeof(float)
                       + cols * sizeof(__nv_bfloat16);
    rms_norm_to_nvfp4_swizzled_bf16_kernel<<<rows, threads, smem_size, stream>>>(
        x, rms_weight, packed, sf_swz, cols, num_blocks, n_col_blocks, eps);
}

// ================================================================
// FUSED: residual_add(h_in, attn_proj) -> h_post (bf16 written to
// global) -> rms_norm(h_post, weight) -> nvfp4 packed + swizzled SF.
// Replaces the (torch.add + rms_norm + quantize_bf16_to_nvfp4_swizzled)
// 3-launch sequence at every per-layer post-attn transition.
// Layout mirrors rms_norm_to_nvfp4_swizzled_bf16_kernel (same smem
// arrangement; same swizzle math).
// ================================================================
__global__ void residual_add_rms_norm_to_nvfp4_swizzled_bf16_kernel(
    const __nv_bfloat16* __restrict__ h_in,
    const __nv_bfloat16* __restrict__ attn_proj,
    __nv_bfloat16* __restrict__ h_post,
    const __nv_bfloat16* __restrict__ rms_weight,    // (D,) precomputed (1+w)
    uint8_t* __restrict__ packed,                    // (rows, cols/2)
    uint8_t* __restrict__ sf_swz,
    int cols, int num_blocks, int n_col_blocks,
    float eps)
{
    const int row = blockIdx.x;
    const __nv_bfloat16* row_h_in = h_in + (size_t)row * cols;
    const __nv_bfloat16* row_attn = attn_proj + (size_t)row * cols;
    __nv_bfloat16* row_h_post = h_post + (size_t)row * cols;
    uint8_t* row_fp4 = packed + (size_t)row * cols / 2;

    // smem: same layout as rms_norm_to_nvfp4_swizzled_bf16_kernel
    extern __shared__ float smem_dyn[];
    float* warp_red = smem_dyn;
    float* sf_smem = smem_dyn + 32;
    __nv_bfloat16* normed = reinterpret_cast<__nv_bfloat16*>(
        sf_smem + num_blocks);

    // ── Phase 1: residual sum + ssq, write h_post bf16 + accumulate ssq ──
    float local_ssq = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float a = __bfloat162float(row_h_in[i]);
        float b = __bfloat162float(row_attn[i]);
        float r_fp32 = a + b;
        // Write h_post in bf16 (downstream MLP residual reads this; the
        // round here matches the unfused torch.add bf16 output exactly).
        __nv_bfloat16 r_bf = __float2bfloat16(r_fp32);
        row_h_post[i] = r_bf;
        // ssq computed over bf16-rounded value (matches unfused path
        // where rms_norm reads bf16-rounded residual from global).
        float r_bf_fp32 = __bfloat162float(r_bf);
        local_ssq += r_bf_fp32 * r_bf_fp32;
    }
    float ssq = block_reduce_sum(local_ssq, warp_red);
    const float rms = rsqrtf(ssq / cols + eps);

    // ── Phase 2: produce normed (bf16) into smem + per-block amax ──
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x)
        sf_smem[b] = 0.f;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        // Re-read the bf16 residual we just wrote (avoids holding fp32
        // sum in regs across the block-reduction).
        float xv = __bfloat162float(row_h_post[i]);
        float wv = __bfloat162float(rms_weight[i]);
        float n_fp32 = xv * rms * wv;
        __nv_bfloat16 nb = __float2bfloat16(n_fp32);
        normed[i] = nb;
        float n_bf16 = __bfloat162float(nb);
        int blk = i >> 4;
        atomicMax((int*)&sf_smem[blk], __float_as_int(fabsf(n_bf16)));
    }
    __syncthreads();

    // ── Phase 3: per-block UE4M3 SF + swizzled write ──
    int rb = row / 128;
    int ri = row % 128;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float amax = __int_as_float(*(int*)&sf_smem[b]);
        float scale = amax / 6.0f;
        uint8_t ue_scale = float_to_ue4m3_ceil(scale);

        int cb = b / 4;
        int ci = b % 4;
        int out_idx = (rb * n_col_blocks + cb) * 512
                      + (ri % 32) * 16 + (ri / 32) * 4 + ci;
        sf_swz[out_idx] = ue_scale;
        sf_smem[b] = ue4m3_to_float(ue_scale);
    }
    __syncthreads();

    // ── Phase 4: pack normed → FP4 e2m1 (2 per byte) ──
    int half_cols = cols >> 1;
    for (int p = threadIdx.x; p < half_cols; p += blockDim.x) {
        int i = p * 2;
        int blk = i >> 4;
        float scale = sf_smem[blk];
        float inv_scale = (scale > 0.f) ? (1.f / scale) : 0.f;

        float v0 = __bfloat162float(normed[i]) * inv_scale;
        float v1 = __bfloat162float(normed[i + 1]) * inv_scale;
        uint8_t lo = float_to_fp4_e2m1(v0);
        uint8_t hi = float_to_fp4_e2m1(v1);
        row_fp4[p] = (hi << 4) | (lo & 0x0F);
    }
}

void residual_add_rms_norm_to_nvfp4_swizzled_bf16(
    const __nv_bfloat16* h_in,
    const __nv_bfloat16* attn_proj,
    __nv_bfloat16* h_post,
    const __nv_bfloat16* rms_weight,
    uint8_t* packed, uint8_t* sf_swz,
    int rows, int cols, float eps,
    cudaStream_t stream)
{
    int num_blocks = (cols + 15) / 16;
    int n_col_blocks = (num_blocks + 3) / 4;
    int threads = 256;
    size_t smem_size = 32 * sizeof(float)
                       + num_blocks * sizeof(float)
                       + cols * sizeof(__nv_bfloat16);
    residual_add_rms_norm_to_nvfp4_swizzled_bf16_kernel
        <<<rows, threads, smem_size, stream>>>(
            h_in, attn_proj, h_post, rms_weight,
            packed, sf_swz, cols, num_blocks, n_col_blocks, eps);
}

// ================================================================
//  MX-FP8 Quantization with per-32-block UE8M0 scale factors
//  For CUTLASS block-scaled W4A8 GEMM on Blackwell (sm_120)
// ================================================================

__global__ void quantize_bf16_to_mxfp8_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ fp8_data,
    uint8_t* __restrict__ sf_data,
    int rows, int cols,
    int n_k_blocks,
    int n_m_atoms,
    int n_k_atoms)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* row_in = input + (size_t)row * cols;
    __nv_fp8_e4m3* row_fp8 = fp8_data + (size_t)row * cols;

    int m_atom = row / 128;
    int m_local = row % 128;

    extern __shared__ float shared[];

    // Pass 1: compute per-32-block absmax
    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        int base = kb * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32 && (base + j) < cols; j++) {
            float v = fabsf(__bfloat162float(row_in[base + j]));
            amax = fmaxf(amax, v);
        }
        shared[kb] = amax;
    }
    __syncthreads();

    // Pass 2: quantize to FP8 and write scale factors
    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        float amax = shared[kb];
        float scale = amax / 448.0f;
        if (scale < 1e-12f) scale = 1e-12f;

        uint8_t ue8m0_scale = float_to_ue8m0_ceil(scale);

        int k_atom = kb / 4;
        int k_local = kb % 4;
        int sf_atom_offset = (m_atom * n_k_atoms + k_atom) * 512;
        int m_in_32 = m_local % 32;
        int m_group = m_local / 32;
        int sf_idx = sf_atom_offset + m_in_32 * 16 + m_group * 4 + k_local;
        sf_data[sf_idx] = ue8m0_scale;

        float actual_scale = ue8m0_to_float(ue8m0_scale);
        float actual_inv_scale = (actual_scale > 0.0f) ? (1.0f / actual_scale) : 0.0f;

        int base = kb * 32;
        for (int j = 0; j < 32 && (base + j) < cols; j++) {
            float v = __bfloat162float(row_in[base + j]) * actual_inv_scale;
            v = fmaxf(-448.0f, fminf(448.0f, v));
            row_fp8[base + j] = __nv_fp8_e4m3(v);
        }
    }
}

void quantize_bf16_to_mxfp8(const __nv_bfloat16* input, __nv_fp8_e4m3* fp8_data,
                              uint8_t* scale_factors, int rows, int cols,
                              cudaStream_t stream) {
    int n_k_blocks = (cols + 31) / 32;
    int n_m_atoms = (rows + 127) / 128;
    int n_k_atoms = (n_k_blocks + 3) / 4;

    int sf_total = n_m_atoms * n_k_atoms * 512;
    cudaMemsetAsync(scale_factors, 0, sf_total, stream);

    int threads = 256;
    int smem_size = n_k_blocks * sizeof(float);
    quantize_bf16_to_mxfp8_kernel<<<rows, threads, smem_size, stream>>>(
        input, fp8_data, scale_factors, rows, cols, n_k_blocks, n_m_atoms, n_k_atoms);
}

int get_mxfp8_sf_size(int rows, int cols) {
    int n_k_blocks = (cols + 31) / 32;
    int n_m_atoms = (rows + 127) / 128;
    int n_k_atoms = (n_k_blocks + 3) / 4;
    return n_m_atoms * n_k_atoms * 512;
}

// ================================================================
//  MX-FP4 Quantization for CUTLASS W4A8 weight
// ================================================================

__global__ void quantize_bf16_to_mxfp4_cutlass_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ fp4_data,
    uint8_t* __restrict__ sf_data,
    int N, int K,
    int n_k_blocks,
    int n_n_atoms,
    int n_k_atoms)
{
    int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_in = input + (size_t)row * K;
    uint8_t* row_fp4 = fp4_data + (size_t)row * K / 2;

    int n_atom = row / 128;
    int n_local = row % 128;

    extern __shared__ float shared[];

    // Pass 1: compute per-16-block absmax
    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        int base = kb * 16;
        float amax = 0.0f;
        for (int j = 0; j < 16 && (base + j) < K; j++) {
            float v = fabsf(__bfloat162float(row_in[base + j]));
            amax = fmaxf(amax, v);
        }
        shared[kb] = amax;
    }
    __syncthreads();

    // Pass 2: quantize and write scale factors
    for (int kb = threadIdx.x; kb < n_k_blocks; kb += blockDim.x) {
        float amax = shared[kb];
        float scale = amax / 6.0f;
        if (scale < 1e-12f) scale = 1e-12f;
        float inv_scale = 1.0f / scale;

        uint8_t ue8m0_scale = float_to_ue8m0_ceil(scale);

        int k_atom = kb / 4;
        int k_local = kb % 4;
        int sf_atom_offset = (n_atom * n_k_atoms + k_atom) * 512;
        int n_in_32 = n_local % 32;
        int n_group = n_local / 32;
        int sf_idx = sf_atom_offset + n_in_32 * 16 + n_group * 4 + k_local;
        sf_data[sf_idx] = ue8m0_scale;

        float actual_scale = ue8m0_to_float(ue8m0_scale);
        float actual_inv_scale = (actual_scale > 0.0f) ? (1.0f / actual_scale) : 0.0f;

        int base = kb * 16;
        for (int j = 0; j < 16; j += 2) {
            int idx = base + j;
            if (idx + 1 >= K) break;
            float v0 = (idx < K) ? __bfloat162float(row_in[idx]) * actual_inv_scale : 0.0f;
            float v1 = (idx + 1 < K) ? __bfloat162float(row_in[idx + 1]) * actual_inv_scale : 0.0f;
            uint8_t fp4_lo = float_to_fp4_e2m1(v0);
            uint8_t fp4_hi = float_to_fp4_e2m1(v1);
            row_fp4[(idx) / 2] = (fp4_hi << 4) | (fp4_lo & 0x0F);
        }
    }
}

void quantize_bf16_to_mxfp4_cutlass(const __nv_bfloat16* input, uint8_t* fp4_data,
                                      uint8_t* scale_factors, int N, int K,
                                      cudaStream_t stream) {
    int n_k_blocks = (K + 15) / 16;
    int n_n_atoms = (N + 127) / 128;
    int n_k_atoms = (n_k_blocks + 3) / 4;

    int sf_total = n_n_atoms * n_k_atoms * 512;
    cudaMemsetAsync(scale_factors, 0, sf_total, stream);

    int threads = 256;
    int smem_size = n_k_blocks * sizeof(float);
    quantize_bf16_to_mxfp4_cutlass_kernel<<<N, threads, smem_size, stream>>>(
        input, fp4_data, scale_factors, N, K, n_k_blocks, n_n_atoms, n_k_atoms);
}

int get_mxfp4_sf_size(int N, int K) {
    int n_k_blocks = (K + 15) / 16;
    int n_n_atoms = (N + 127) / 128;
    int n_k_atoms = (n_k_blocks + 3) / 4;
    return n_n_atoms * n_k_atoms * 512;
}
#endif  // ENABLE_NVFP4

// ============================================================================
//  FlashRT — vectorized fp16 backbone helpers (SM100/SM110-class).
//
//  16-byte-load rewrites of the small per-layer kernels that dominate the
//  non-GEMM time of FP8 vision/LLM backbones at Thor-class DRAM latency:
//  the scalar originals issue 2-4B accesses and sit at 50-100 GB/s
//  (memory-latency-bound), far under the LPDDR5X floor.
//
//  Element math matches the original kernels exactly; rope / quantize /
//  repeat-interleave are element-independent and therefore bit-exact.
//  The norm reductions accumulate in fp32 like the originals but with a
//  different (deterministic) summation order.
//
//  Additive: new symbols only; the scalar kernels are untouched.
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float block_sum_v(float v, float* sh) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < ((blockDim.x + 31) >> 5)) ? sh[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    const float r = sh[0];
    __syncthreads();
    return r;
}

// ── RMSNorm, warp-per-row (small dim, e.g. per-head 128) ──────────────────
__global__ void rms_norm_fp16_vec_warp_kernel(
    const __half* __restrict__ x, const __half* __restrict__ w,
    __half* __restrict__ out, int rows, int dim, float eps) {
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (row >= rows) return;
    const int lane = threadIdx.x & 31;
    const int n4 = dim >> 3;                       // int4 chunks per row
    const int4* x4 = reinterpret_cast<const int4*>(x + (long)row * dim);
    const int4* w4 = reinterpret_cast<const int4*>(w);
    int4* o4 = reinterpret_cast<int4*>(out + (long)row * dim);

    float ss = 0.f;
    for (int i = lane; i < n4; i += 32) {
        const int4 raw = x4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = __half2float(hh[j]);
            ss += v * v;
        }
    }
    ss = warp_sum(ss);
    const float rms = rsqrtf(ss / dim + eps);

    for (int i = lane; i < n4; i += 32) {
        const int4 raw = x4[i];
        const int4 wr = w4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        const __half* wh = reinterpret_cast<const __half*>(&wr);
        int4 res;
        __half* rh = reinterpret_cast<__half*>(&res);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            rh[j] = __float2half(
                __half2float(hh[j]) * rms * __half2float(wh[j]));
        o4[i] = res;
    }
}

// ── RMSNorm, block-per-row (large dim, e.g. 2048) ─────────────────────────
__global__ void rms_norm_fp16_vec_block_kernel(
    const __half* __restrict__ x, const __half* __restrict__ w,
    __half* __restrict__ out, int dim, float eps) {
    const int row = blockIdx.x;
    const int n4 = dim >> 3;
    const int4* x4 = reinterpret_cast<const int4*>(x + (long)row * dim);
    const int4* w4 = reinterpret_cast<const int4*>(w);
    int4* o4 = reinterpret_cast<int4*>(out + (long)row * dim);
    __shared__ float sh[32];

    float ss = 0.f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = __half2float(hh[j]);
            ss += v * v;
        }
    }
    const float rms = rsqrtf(block_sum_v(ss, sh) / dim + eps);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const int4 wr = w4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        const __half* wh = reinterpret_cast<const __half*>(&wr);
        int4 res;
        __half* rh = reinterpret_cast<__half*>(&res);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            rh[j] = __float2half(
                __half2float(hh[j]) * rms * __half2float(wh[j]));
        o4[i] = res;
    }
}

// ── LayerNorm (gamma/beta), block-per-row ─────────────────────────────────
__global__ void layer_norm_fp16_vec_kernel(
    const __half* __restrict__ x, const __half* __restrict__ w,
    const __half* __restrict__ b, __half* __restrict__ out,
    int dim, float eps) {
    const int row = blockIdx.x;
    const int n4 = dim >> 3;
    const int4* x4 = reinterpret_cast<const int4*>(x + (long)row * dim);
    const int4* w4 = reinterpret_cast<const int4*>(w);
    const int4* b4 = reinterpret_cast<const int4*>(b);
    int4* o4 = reinterpret_cast<int4*>(out + (long)row * dim);
    __shared__ float sh[32];

    float s = 0.f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) s += __half2float(hh[j]);
    }
    const float mean = block_sum_v(s, sh) / dim;

    float var = 0.f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float d = __half2float(hh[j]) - mean;
            var += d * d;
        }
    }
    const float inv_std = rsqrtf(block_sum_v(var, sh) / dim + eps);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const int4 wr = w4[i];
        const int4 br = b4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        const __half* wh = reinterpret_cast<const __half*>(&wr);
        const __half* bh = reinterpret_cast<const __half*>(&br);
        int4 res;
        __half* rh = reinterpret_cast<__half*>(&res);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            rh[j] = __float2half(
                (__half2float(hh[j]) - mean) * inv_std *
                    __half2float(wh[j]) + __half2float(bh[j]));
        o4[i] = res;
    }
}

// ── LayerNorm (gamma/beta) fused with a static FP8 quantize ───────────────
// The norm output feeds only an FP8 GEMM, so emitting fp8 directly removes
// the fp16 intermediate's write and read-back (and one kernel launch).
__global__ void layer_norm_fp8_static_fp16_vec_kernel(
    const __half* __restrict__ x, const __half* __restrict__ w,
    const __half* __restrict__ b, __nv_fp8_e4m3* __restrict__ out,
    const float* __restrict__ descale_ptr, int dim, float eps) {
    const int row = blockIdx.x;
    const int n4 = dim >> 3;
    const int4* x4 = reinterpret_cast<const int4*>(x + (long)row * dim);
    const int4* w4 = reinterpret_cast<const int4*>(w);
    const int4* b4 = reinterpret_cast<const int4*>(b);
    uint2* o2 = reinterpret_cast<uint2*>(out + (long)row * dim);
    __shared__ float sh[32];

    float s = 0.f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) s += __half2float(hh[j]);
    }
    const float mean = block_sum_v(s, sh) / dim;

    float var = 0.f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float d = __half2float(hh[j]) - mean;
            var += d * d;
        }
    }
    const float inv_std = rsqrtf(block_sum_v(var, sh) / dim + eps);
    const float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const int4 raw = x4[i];
        const int4 wr = w4[i];
        const int4 br = b4[i];
        const __half* hh = reinterpret_cast<const __half*>(&raw);
        const __half* wh = reinterpret_cast<const __half*>(&wr);
        const __half* bh = reinterpret_cast<const __half*>(&br);
        uint2 res;
        __nv_fp8_e4m3* rp = reinterpret_cast<__nv_fp8_e4m3*>(&res);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            // fp16 rounding of the norm output keeps this identical to the
            // two-step (fp16 LayerNorm, then static quantize) chain.
            const float v = __half2float(__float2half(
                (__half2float(hh[j]) - mean) * inv_std * __half2float(wh[j]) +
                __half2float(bh[j])));
            rp[j] = __nv_fp8_e4m3(fminf(fmaxf(v * inv_scale, -448.f), 448.f));
        }
        o2[i] = res;
    }
}

// ── Split-half RoPE, 8 pairs per thread ───────────────────────────────────
__global__ void rope_rotate_half_fp16_vec_kernel(
    __half* __restrict__ x, const __half* __restrict__ cos_t,
    const __half* __restrict__ sin_t, int S, int NH, int half_hd) {
    const int n4 = half_hd >> 3;                   // int4 chunks per half-row
    const int total = S * NH * n4;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % n4;
    const int rem = idx / n4;
    const int h = rem % NH;
    const int s = rem / NH;

    const int HD = half_hd * 2;
    __half* base = x + ((long)s * NH + h) * HD;
    int4* lo4 = reinterpret_cast<int4*>(base) + c;
    int4* hi4 = reinterpret_cast<int4*>(base + half_hd) + c;
    const int4* c4 = reinterpret_cast<const int4*>(cos_t + (long)s * HD) + c;
    const int4* s4 = reinterpret_cast<const int4*>(sin_t + (long)s * HD) + c;

    const int4 lo_raw = *lo4;
    const int4 hi_raw = *hi4;
    const int4 c_raw = *c4;
    const int4 s_raw = *s4;
    const __half* lo = reinterpret_cast<const __half*>(&lo_raw);
    const __half* hi = reinterpret_cast<const __half*>(&hi_raw);
    const __half* cc = reinterpret_cast<const __half*>(&c_raw);
    const __half* ss = reinterpret_cast<const __half*>(&s_raw);
    int4 lo_out, hi_out;
    __half* lop = reinterpret_cast<__half*>(&lo_out);
    __half* hip = reinterpret_cast<__half*>(&hi_out);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float xl = __half2float(lo[j]);
        const float xh = __half2float(hi[j]);
        const float cv = __half2float(cc[j]);
        const float sv = __half2float(ss[j]);
        lop[j] = __float2half(xl * cv - xh * sv);
        hip[j] = __float2half(xh * cv + xl * sv);
    }
    *lo4 = lo_out;
    *hi4 = hi_out;
}

// ── Static FP8 quantize, 16 elements per thread ───────────────────────────
__global__ void quantize_fp8_static_fp16_vec_kernel(
    const int4* __restrict__ in, int4* __restrict__ out,
    const float* __restrict__ descale_ptr, int n16) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n16) return;
    const float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    const int4 rawA = in[2 * idx];
    const int4 rawB = in[2 * idx + 1];
    const __half* ha = reinterpret_cast<const __half*>(&rawA);
    const __half* hb = reinterpret_cast<const __half*>(&rawB);
    int4 res;
    __nv_fp8_e4m3* rp = reinterpret_cast<__nv_fp8_e4m3*>(&res);
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        rp[j] = __nv_fp8_e4m3(
            fminf(fmaxf(__half2float(ha[j]) * inv_scale, -448.f), 448.f));
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        rp[8 + j] = __nv_fp8_e4m3(
            fminf(fmaxf(__half2float(hb[j]) * inv_scale, -448.f), 448.f));
    out[idx] = res;
}

// ── Residual add, 8 elements per thread ───────────────────────────────────
__global__ void residual_add_fp16_vec_kernel(
    int4* __restrict__ residual, const int4* __restrict__ x, int n4) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n4) return;
    int4 r = residual[idx];
    const int4 v = x[idx];
    __half* rh = reinterpret_cast<__half*>(&r);
    const __half* vh = reinterpret_cast<const __half*>(&v);
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        rh[j] = __float2half(__half2float(rh[j]) + __half2float(vh[j]));
    residual[idx] = r;
}

// ── GQA repeat-interleave head expand, 8 elements per thread ──────────────
__global__ void repeat_interleave_heads_vec_kernel(
    const int4* __restrict__ src, int4* __restrict__ dst,
    int S, int NH_src, int hd4, int repeat) {
    const int NH_dst = NH_src * repeat;
    const int total = S * NH_dst * hd4;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % hd4;
    const int rem = idx / hd4;
    const int h_dst = rem % NH_dst;
    const int s = rem / NH_dst;
    const int h_src = h_dst / repeat;
    dst[((long)s * NH_dst + h_dst) * hd4 + c] =
        src[((long)s * NH_src + h_src) * hd4 + c];
}

// CTA width for the block-per-row norms: one thread per 16-byte vector,
// rounded to a warp and capped at 256. A 1024-wide row has 128 vectors, so
// a fixed 256-thread block would leave half the CTA idle through both
// reduction passes.
inline int norm_threads(int dim) {
    const int vecs = dim >> 3;
    int t = ((vecs + 31) / 32) * 32;
    if (t < 32) t = 32;
    if (t > 256) t = 256;
    return t;
}

}  // namespace

extern "C" {

int rms_norm_fp16_vec(const __half* x, const __half* w, __half* out,
                      int rows, int dim, float eps, cudaStream_t stream) {
    if (dim % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(w) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 15)) return -1;
    if (dim <= 512) {
        const int warps = 8;
        const int blocks = (rows + warps - 1) / warps;
        rms_norm_fp16_vec_warp_kernel<<<blocks, warps * 32, 0, stream>>>(
            x, w, out, rows, dim, eps);
    } else {
        rms_norm_fp16_vec_block_kernel<<<rows, norm_threads(dim), 0, stream>>>(
            x, w, out, dim, eps);
    }
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int layer_norm_fp16_vec(const __half* x, const __half* w, const __half* b,
                        __half* out, int rows, int dim, float eps,
                        cudaStream_t stream) {
    if (dim % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(w) & 15) ||
        (reinterpret_cast<uintptr_t>(b) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 15)) return -1;
    layer_norm_fp16_vec_kernel<<<rows, norm_threads(dim), 0, stream>>>(
        x, w, b, out, dim, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int layer_norm_fp8_static_fp16_vec(const __half* x, const __half* w,
                                   const __half* b, __nv_fp8_e4m3* out,
                                   const float* d_scale, int rows, int dim,
                                   float eps, cudaStream_t stream) {
    if (dim % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(w) & 15) ||
        (reinterpret_cast<uintptr_t>(b) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 7)) return -1;
    layer_norm_fp8_static_fp16_vec_kernel<<<rows, norm_threads(dim), 0, stream>>>(
        x, w, b, out, d_scale, dim, eps);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int rope_rotate_half_fp16_vec(__half* x, const __half* cos_t,
                              const __half* sin_t, int S, int NH, int HD,
                              cudaStream_t stream) {
    const int half_hd = HD / 2;
    if (half_hd % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(cos_t) & 15) ||
        (reinterpret_cast<uintptr_t>(sin_t) & 15)) return -1;
    const int total = S * NH * (half_hd >> 3);
    const int threads = 256;
    rope_rotate_half_fp16_vec_kernel<<<
        (total + threads - 1) / threads, threads, 0, stream>>>(
        x, cos_t, sin_t, S, NH, half_hd);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int quantize_fp8_static_fp16_vec(const __half* in, __nv_fp8_e4m3* out,
                                 const float* descale_ptr, int n,
                                 cudaStream_t stream) {
    if (n % 16 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(in) & 15) ||
        (reinterpret_cast<uintptr_t>(out) & 15)) return -1;
    const int n16 = n / 16;
    const int threads = 256;
    quantize_fp8_static_fp16_vec_kernel<<<
        (n16 + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(in), reinterpret_cast<int4*>(out),
        descale_ptr, n16);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int residual_add_fp16_vec(__half* residual, const __half* x, int n,
                          cudaStream_t stream) {
    if (n % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(residual) & 15) ||
        (reinterpret_cast<uintptr_t>(x) & 15)) return -1;
    const int n4 = n >> 3;
    const int threads = 256;
    residual_add_fp16_vec_kernel<<<
        (n4 + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<int4*>(residual),
        reinterpret_cast<const int4*>(x), n4);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int gpu_repeat_interleave_heads_vec(const __half* src, __half* dst,
                                    int S, int NH_src, int HD, int repeat,
                                    cudaStream_t stream) {
    if (HD % 8 != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(src) & 15) ||
        (reinterpret_cast<uintptr_t>(dst) & 15)) return -1;
    const int hd4 = HD >> 3;
    const int total = S * NH_src * repeat * hd4;
    const int threads = 256;
    repeat_interleave_heads_vec_kernel<<<
        (total + threads - 1) / threads, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(src), reinterpret_cast<int4*>(dst),
        S, NH_src, hd4, repeat);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

}  // extern "C"

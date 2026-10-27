// FlashRT — X-VLA / Florence-2 DaViT vision tower kernels (SM120).
#include "xvla_vision_sm120.cuh"
#include "pdl_switch.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "nvfp4_convert.cuh"

namespace flash_rt {
namespace xvla {

using bf16 = __nv_bfloat16;

template <typename T> struct Cvt;
template <> struct Cvt<bf16> {
    static __device__ __forceinline__ float to(bf16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ bf16 from(float v) { return __float2bfloat16(v); }
};
template <> struct Cvt<__half> {
    static __device__ __forceinline__ float to(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half from(float v) { return __float2half(v); }
};

// 2-element vector of the token dtype: the window path moves whole head-dim
// pairs, so the real 32 head dims travel as one 32-bit transaction.
template <typename T> struct Pack2T;
template <> struct Pack2T<__half> { using type = __half2; };
template <> struct Pack2T<bf16> { using type = __nv_bfloat162; };
template <typename T> using Pack2 = typename Pack2T<T>::type;

// ── erf GELU + NVFP4 quantize, one thread per 16-element scale block ────────
// The DaViT FFN applies the erf GELU to its 4c-wide midpoint and the next GEMM
// then quantizes that midpoint to NVFP4.  Both steps walk the same (rows, cols)
// tiles and the only statistic is a per-16-element maximum, so folding them is
// bit-identical to running them in sequence -- the maximum does not care about
// reduction order -- and it removes the midpoint's write and re-read.
//
// One thread owns exactly one 16-element scale block: it reads the 32 bytes as
// two 16-byte loads, evaluates the erf GELU on all 16 values, takes the block
// maximum, writes the block's eight packed bytes as one 8-byte store and its
// one scale byte at the swizzled address.  No shared memory, no barriers, and
// the grid is sized by scale blocks instead of rows.
__global__ void gelu_erf_to_nvfp4_block_kernel(
        const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ packed,
        uint8_t* __restrict__ sf_swz, int blocks_per_row, int total_blocks,
        int n_col_blocks) {
    flashrt_pdl_wait_and_trigger();
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_blocks) return;
    const int row = g / blocks_per_row;
    const int blk = g - row * blocks_per_row;

    const uint4* src =
        reinterpret_cast<const uint4*>(x + (size_t)row * blocks_per_row * 16 +
                                       (size_t)blk * 16);
    const uint4 lo = src[0], hi = src[1];
    const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    // The 16 bf16 GELU results, held as eight packed bf16x2 words.
    uint32_t nw[8];
    float m = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(&words[j]);
        const float v0 = __bfloat162float(xv.x);
        const float v1 = __bfloat162float(xv.y);
        const __nv_bfloat16 n0 =
            __float2bfloat16(v0 * 0.5f * (1.0f + erff(v0 * 0.7071067811865476f)));
        const __nv_bfloat16 n1 =
            __float2bfloat16(v1 * 0.5f * (1.0f + erff(v1 * 0.7071067811865476f)));
        const __nv_bfloat162 nv = __nv_bfloat162(n0, n1);
        nw[j] = *reinterpret_cast<const uint32_t*>(&nv);
        m = fmaxf(m, fmaxf(fabsf(__bfloat162float(n0)), fabsf(__bfloat162float(n1))));
    }
    const uint8_t ue = float_to_ue4m3_ceil(m / 6.0f);
    const float s = ue4m3_to_float(ue);
    const float inv = (s > 0.f) ? 1.f / s : 0.f;
    uint64_t enc = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const __nv_bfloat162 nv = *reinterpret_cast<const __nv_bfloat162*>(&nw[j]);
        const uint8_t byte =
            (uint8_t)((float_to_fp4_e2m1_branchless(__bfloat162float(nv.y) * inv) << 4) |
                      (float_to_fp4_e2m1_branchless(__bfloat162float(nv.x) * inv) & 0x0F));
        enc |= (uint64_t)byte << (8 * j);
    }
    *reinterpret_cast<uint64_t*>(packed + (size_t)g * 8) = enc;

    const int rb = row >> 7, ri = row & 127;
    sf_swz[(rb * n_col_blocks + (blk >> 2)) * 512 + (ri % 32) * 16 +
           (ri / 32) * 4 + (blk & 3)] = ue;
}

void vision_gelu_erf_to_nvfp4(const void* x, void* packed, void* sf_swz,
                               int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || (cols & 15)) return;
    const int num_blocks = (cols + 15) / 16;
    const int blocks_per_row = cols >> 4;
    const int total_blocks = rows * blocks_per_row;
    const int grid = (total_blocks + 255) / 256;
    flash_rt::pdl::launch(gelu_erf_to_nvfp4_block_kernel, dim3(grid), dim3(256), 0, stream,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<uint8_t*>(packed),
        reinterpret_cast<uint8_t*>(sf_swz), blocks_per_row, total_blocks,
        (num_blocks + 3) / 4);
}

// ── LayerNorm with an fp16 input and a bf16 output ──────────────────────────
// The DaViT tower runs fp16 and the encoder bf16, so the projector's final
// LayerNorm is the dtype boundary. Emitting bf16 here removes the separate
// fp16->bf16 narrowing kernel that would otherwise follow it.
// One block per row, 128 threads; each thread strides the row so the three
// passes (mean, variance, write) stay coalesced.
__global__ void ln_f16_bf16_kernel(const __half* __restrict__ x,
                                   const __half* __restrict__ w,
                                   const __half* __restrict__ b,
                                   __nv_bfloat16* __restrict__ out, int dim,
                                   float eps) {
    flashrt_pdl_wait_and_trigger();
    extern __shared__ float red[];
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    const __half* xr = x + (size_t)row * dim;
    float s = 0.0f;
    for (int i = t; i < dim; i += nt) s += __half2float(xr[i]);
    red[t] = s;
    __syncthreads();
    for (int off = nt / 2; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        __syncthreads();
    }
    const float mean = red[0] / (float)dim;
    __syncthreads();

    float v = 0.0f;
    for (int i = t; i < dim; i += nt) {
        float d = __half2float(xr[i]) - mean;
        v += d * d;
    }
    red[t] = v;
    __syncthreads();
    for (int off = nt / 2; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)dim + eps);

    __nv_bfloat16* orow = out + (size_t)row * dim;
    for (int i = t; i < dim; i += nt) {
        orow[i] = __float2bfloat16((__half2float(xr[i]) - mean) * inv *
                                   __half2float(w[i]) + __half2float(b[i]));
    }
}

void vision_ln_f16_bf16(const __half* x, const __half* w, const __half* b,
                        __nv_bfloat16* out, int rows, int dim, float eps,
                        cudaStream_t stream) {
    if (rows <= 0 || dim <= 0) return;
    const int nt = 128;
    flash_rt::pdl::launch(ln_f16_bf16_kernel, dim3(rows), dim3(nt), nt * sizeof(float), stream, x, w, b, out, dim,
                                                                 eps);
}

// ── bilinear resize (f32 CHW -> NHWC) ───────────────────────────────────────

template <typename T>
__global__ void resize_bilinear_kernel(const float* __restrict__ src,
                                       T* __restrict__ dst, int cin, int hin,
                                       int win_, int hout, int wout, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % cin;
    int q = idx / cin;
    int w = q % wout;
    q /= wout;
    int h = q % hout;
    int n = q / hout;
    float sh = fmaxf(0.0f, (h + 0.5f) * ((float)hin / hout) - 0.5f);
    float sw = fmaxf(0.0f, (w + 0.5f) * ((float)win_ / wout) - 0.5f);
    int h0 = (int)sh, w0 = (int)sw;
    int h1 = h0 + 1 < hin ? h0 + 1 : hin - 1;
    int w1 = w0 + 1 < win_ ? w0 + 1 : win_ - 1;
    float lh = sh - h0, lw = sw - w0;
    const float* p = src + ((size_t)n * cin + c) * hin * win_;
    float v = (1 - lh) * ((1 - lw) * p[h0 * win_ + w0] + lw * p[h0 * win_ + w1])
            + lh * ((1 - lw) * p[h1 * win_ + w0] + lw * p[h1 * win_ + w1]);
    dst[idx] = Cvt<T>::from(fminf(1.0f, fmaxf(0.0f, v)));
}

template <typename T>
static void resize_launch(T* dst, const float* src, int n, int cin, int hin,
                          int win_, int hout, int wout, cudaStream_t stream) {
    int total = n * cin * hout * wout;
    if (total <= 0) return;
    flash_rt::pdl::launch(resize_bilinear_kernel<T>, dim3((total + 255) / 256), dim3(256), 0, stream,
        src, dst, cin, hin, win_, hout, wout, total);
}

void vision_resize_bilinear_f32(void* dst, const float* src, int n, int cin,
                                int hin, int win_, int hout, int wout, bool half,
                                cudaStream_t stream) {
    if (half) resize_launch<__half>((__half*)dst, src, n, cin, hin, win_, hout, wout, stream);
    else resize_launch<bf16>((bf16*)dst, src, n, cin, hin, win_, hout, wout, stream);
}

// ── im2col (NHWC) ───────────────────────────────────────────────────────────

template <typename T>
__global__ void im2col_kernel(const T* __restrict__ x, T* __restrict__ col,
                              int H, int W, int cin, int k, int stride, int pad,
                              int out_h, int out_w, int K, int nrows, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = idx / K;
    int kk = idx - row * K;
    int img = row / nrows;      // batch: rows are [image][out_h][out_w]
    int r = row - img * nrows;
    int oh = r / out_w;
    int ow = r - oh * out_w;
    // K index matches the flattening of the PyTorch weight [Cout, Cin, kH, kW]:
    // ci is the outer index, then kh, then kw.
    int ci = kk / (k * k);
    int rem = kk - ci * k * k;
    int kh = rem / k;
    int kw = rem - kh * k;
    int ih = oh * stride - pad + kh;
    int iw = ow * stride - pad + kw;
    float v = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = Cvt<T>::to(x[((size_t)img * H * W + ih * W + iw) * cin + ci]);
    col[idx] = Cvt<T>::from(v);
}

// One CTA per output row: the k*k patch pixels are read coalesced into shared
// memory (each pixel's cin channels are contiguous in the NHWC source), then
// the row is written coalesced from the transposed layout.  The values written
// are the elementwise gather's values exactly (including the zero fill for
// out-of-image taps), so the im2col matrix is unchanged; only the access order
// differs.  Used when K is large enough that the read pattern matters.
template <typename T>
__global__ void im2col_smem_kernel(const T* __restrict__ x, T* __restrict__ col,
                                   int H, int W, int cin, int k, int stride,
                                   int pad, int out_h, int out_w, int K, int nrows,
                                   int rows) {
    flashrt_pdl_wait_and_trigger();
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int img = row / nrows;
    const int r = row - img * nrows;
    const int oh = r / out_w;
    const int ow = r - oh * out_w;
    const int kk = k * k;
    const int sm = cin + 1;  // one element of bank padding per pixel
    extern __shared__ float smem_dyn[];  // byte view; same symbol as the other kernels
    T* buf = reinterpret_cast<T*>(smem_dyn);
    for (int idx = threadIdx.x; idx < kk * cin; idx += blockDim.x) {
        const int p = idx / cin;
        const int ci = idx - p * cin;
        const int kh = p / k, kw = p - kh * k;
        const int ih = oh * stride - pad + kh;
        const int iw = ow * stride - pad + kw;
        T v = Cvt<T>::from(0.0f);
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            v = x[((size_t)img * H * W + ih * W + iw) * cin + ci];
        buf[p * sm + ci] = v;
    }
    __syncthreads();
    T* out_row = col + (size_t)row * K;
    for (int t = threadIdx.x; t < K; t += blockDim.x) {
        const int ci = t / kk;
        const int rem = t - ci * kk;
        out_row[t] = buf[rem * sm + ci];
    }
}

template <typename T>
static void im2col_launch(T* col, const T* x, int H, int W, int cin, int k,
                          int stride, int pad, int out_h, int out_w, int batch,
                          cudaStream_t stream) {
    int K = cin * k * k;
    int nrows = out_h * out_w;
    int total = batch * nrows * K;
    if (total <= 0) return;
    const int rows = batch * nrows;
    if (K >= 256) {
        const size_t smem = (size_t)(k * k) * (cin + 1) * sizeof(T);
        flash_rt::pdl::launch(im2col_smem_kernel<T>, dim3(rows), dim3(256), smem, stream, x, col, H, W, cin, k,
                                                           stride, pad, out_h,
                                                           out_w, K, nrows, rows);
        return;
    }
    flash_rt::pdl::launch(im2col_kernel<T>, dim3((total + 255) / 256), dim3(256), 0, stream, x, col, H, W, cin, k,
                                                              stride, pad, out_h,
                                                              out_w, K, nrows, total);
}

void vision_im2col(void* col, const void* x, int H, int W, int cin, int k,
                   int stride, int pad, int out_h, int out_w, int batch, bool half,
                   cudaStream_t stream) {
    if (half) im2col_launch<__half>((__half*)col, (const __half*)x, H, W, cin, k, stride, pad, out_h, out_w, batch, stream);
    else im2col_launch<bf16>((bf16*)col, (const bf16*)x, H, W, cin, k, stride, pad, out_h, out_w, batch, stream);
}

// ── bias add ────────────────────────────────────────────────────────────────

template <typename T>
__global__ void add_bias_kernel(T* __restrict__ x, const T* __restrict__ bias,
                                int cols, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = Cvt<T>::from(Cvt<T>::to(x[idx]) + Cvt<T>::to(bias[idx % cols]));
}

template <typename T>
static void add_bias_launch(T* x, const T* bias, int rows, int cols, cudaStream_t stream) {
    int total = rows * cols;
    if (total <= 0) return;
    flash_rt::pdl::launch(add_bias_kernel<T>, dim3((total + 255) / 256), dim3(256), 0, stream, x, bias, cols, total);
}

void vision_add_bias(void* x, const void* bias, int rows, int cols, bool half,
                     cudaStream_t stream) {
    if (half) add_bias_launch<__half>((__half*)x, (const __half*)bias, rows, cols, stream);
    else add_bias_launch<bf16>((bf16*)x, (const bf16*)bias, rows, cols, stream);
}

// ── depthwise 3x3 + residual (NHWC) ─────────────────────────────────────────

// Each thread produces four consecutive outputs along W for one (img, ih, c),
// so it loads six horizontal positions per row instead of the nine a single
// output would fetch on its own: 18 loads for four outputs against 36.  The
// per-output accumulation stays exactly as it was -- residual plus bias, then
// the taps in (kh, kw) order with out-of-range taps skipped rather than
// zero-filled -- so the result is bit-identical.
template <typename T>
__global__ void dwconv3x3_res_kernel(const T* __restrict__ x,
                                     const T* __restrict__ w,
                                     const T* __restrict__ bias,
                                     T* __restrict__ out, int H, int W, int C,
                                     int groups) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= groups) return;
    int c = idx % C;
    int p = idx / C;
    int img = p / (H * ((W + 3) >> 2));   // batch: tokens are [image][h][w]
    int pr = p - img * (H * ((W + 3) >> 2));
    int iw0 = (pr % ((W + 3) >> 2)) * 4;
    int ih = pr / ((W + 3) >> 2);
    const size_t base = (size_t)img * H * W * C;
    const T* xi = x + base;
    T* oi = out + base;
    const T* wc = w + c * 9;

    float acc[4];
    int live = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iw = iw0 + j;
        if (iw >= W) break;
        acc[j] = Cvt<T>::to(xi[((size_t)ih * W + iw) * C + c]) + Cvt<T>::to(bias[c]);
        ++live;
    }

#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        const int y = ih + kh - 1;
        if (y < 0 || y >= H) continue;
        // the union of this row's taps for the four outputs is six positions
        T row[6];
#pragma unroll
        for (int m = 0; m < 6; ++m) {
            const int xx = iw0 - 1 + m;
            row[m] = (xx >= 0 && xx < W)
                ? xi[((size_t)y * W + xx) * C + c]
                : Cvt<T>::from(0.0f);
        }
#pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j >= live) break;
                const int xx = iw0 + j + kw - 1;
                if (xx < 0 || xx >= W) continue;
                acc[j] += Cvt<T>::to(row[j + kw]) * Cvt<T>::to(wc[kh * 3 + kw]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j >= live) break;
        oi[((size_t)ih * W + iw0 + j) * C + c] = Cvt<T>::from(acc[j]);
    }
}

template <typename T>
static void dwconv_launch(T* out, const T* x, const T* w, const T* bias, int H,
                          int W, int C, int batch, cudaStream_t stream) {
    int groups = batch * H * ((W + 3) >> 2) * C;
    if (groups <= 0) return;
    flash_rt::pdl::launch(dwconv3x3_res_kernel<T>, dim3((groups + 255) / 256), dim3(256), 0, stream, x, w, bias,
                                                                      out, H, W, C, groups);
}

// Defined below, next to the four-channel kernel it dispatches to.
template <typename T>
static void dwconv_c4_launch(T* out, const T* x, const T* w, const T* bias, int H,
                             int W, int C, int batch, cudaStream_t stream);

void vision_dwconv3x3_res(void* out, const void* x, const void* w,
                          const void* bias, int H, int W, int C, int batch,
                          bool half, cudaStream_t stream) {
    if (half) dwconv_c4_launch<__half>((__half*)out, (const __half*)x, (const __half*)w, (const __half*)bias, H, W, C, batch, stream);
    else dwconv_c4_launch<bf16>((bf16*)out, (const bf16*)x, (const bf16*)w, (const bf16*)bias, H, W, C, batch, stream);
}

// ── depthwise 3x3 + residual, four channels per thread ──────────────────────
//
// The scalar kernel spends 18 two-byte loads per four outputs of one channel.
// Measured at the tower's stage shapes (scripts/probe_dwconv.py) it runs at
// ~160 GB/s of unique traffic -- stage 2 (14x14x1024) takes 10.0 us to move
// 802 KB in and 802 KB out -- so the loads, not the bytes, are the limit.
// This variant hands each thread four consecutive channels of the same four
// outputs: the same 18 taps arrive as 8-byte vectors and each output leaves as
// one 8-byte store. The per-element accumulation order (residual + bias, then
// the taps in (kh, kw) order with out-of-range taps skipped) is unchanged, so
// the result is identical to the scalar kernel.
template <typename T>
__device__ __forceinline__ void dw_ld4(const T* p, float (&v)[4]);
template <>
__device__ __forceinline__ void dw_ld4<bf16>(const bf16* p, float (&v)[4]) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(p);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(p + 2);
    v[0] = __bfloat162float(a.x); v[1] = __bfloat162float(a.y);
    v[2] = __bfloat162float(b.x); v[3] = __bfloat162float(b.y);
}
template <>
__device__ __forceinline__ void dw_ld4<__half>(const __half* p, float (&v)[4]) {
    const __half2 a = *reinterpret_cast<const __half2*>(p);
    const __half2 b = *reinterpret_cast<const __half2*>(p + 2);
    v[0] = __half2float(a.x); v[1] = __half2float(a.y);
    v[2] = __half2float(b.x); v[3] = __half2float(b.y);
}
// Four-element copy (8 bytes) for the staging loops; both sides are 4-element
// aligned by construction (group widths and the 3C row stride are multiples of
// four).
template <typename T>
__device__ __forceinline__ void dw_cp4(const T* src, T* dst) {
    *reinterpret_cast<Pack2<T>*>(dst) = *reinterpret_cast<const Pack2<T>*>(src);
    *reinterpret_cast<Pack2<T>*>(dst + 2) = *reinterpret_cast<const Pack2<T>*>(src + 2);
}

template <typename T>
__device__ __forceinline__ void dw_st4(T* p, const float (&v)[4]);
template <>
__device__ __forceinline__ void dw_st4<bf16>(bf16* p, const float (&v)[4]) {
    *reinterpret_cast<__nv_bfloat162*>(p) =
        __nv_bfloat162(__float2bfloat16(v[0]), __float2bfloat16(v[1]));
    *reinterpret_cast<__nv_bfloat162*>(p + 2) =
        __nv_bfloat162(__float2bfloat16(v[2]), __float2bfloat16(v[3]));
}
template <>
__device__ __forceinline__ void dw_st4<__half>(__half* p, const float (&v)[4]) {
    *reinterpret_cast<__half2*>(p) = __floats2half2_rn(v[0], v[1]);
    *reinterpret_cast<__half2*>(p + 2) = __floats2half2_rn(v[2], v[3]);
}

template <typename T>
__global__ void dwconv3x3_res_c4_kernel(const T* __restrict__ x,
                                        const T* __restrict__ w,
                                        const T* __restrict__ bias,
                                        T* __restrict__ out, int H, int W, int C,
                                        int groups) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= groups) return;
    const int C4 = C >> 2;
    const int wi = (W + 3) >> 2;
    const int c4 = idx % C4;
    const int p = idx / C4;
    const int img = p / (H * wi);
    const int pr = p - img * (H * wi);
    const int iw0 = (pr % wi) * 4;
    const int ih = pr / wi;
    const int c0 = c4 * 4;
    const size_t base = (size_t)img * H * W * C;
    const T* xi = x + base;
    T* oi = out + base;

    float acc[4][4];
    int live = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int iw = iw0 + j;
        if (iw >= W) break;
        float v[4];
        dw_ld4<T>(xi + ((size_t)ih * W + iw) * C + c0, v);
#pragma unroll
        for (int dc = 0; dc < 4; ++dc) {
            acc[j][dc] = v[dc] + Cvt<T>::to(bias[c0 + dc]);
        }
        ++live;
    }

#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        const int y = ih + kh - 1;
        if (y < 0 || y >= H) continue;
        float row[6][4];
#pragma unroll
        for (int m = 0; m < 6; ++m) {
            const int xx = iw0 - 1 + m;
            if (xx >= 0 && xx < W) {
                dw_ld4<T>(xi + ((size_t)y * W + xx) * C + c0, row[m]);
            } else {
#pragma unroll
                for (int dc = 0; dc < 4; ++dc) row[m][dc] = 0.0f;
            }
        }
#pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            float wv[4];
#pragma unroll
            for (int dc = 0; dc < 4; ++dc) {
                wv[dc] = Cvt<T>::to(w[(c0 + dc) * 9 + kh * 3 + kw]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j >= live) break;
                const int xx = iw0 + j + kw - 1;
                if (xx < 0 || xx >= W) continue;
#pragma unroll
                for (int dc = 0; dc < 4; ++dc) {
                    acc[j][dc] += row[j + kw][dc] * wv[dc];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (j >= live) break;
        dw_st4<T>(oi + ((size_t)ih * W + iw0 + j) * C + c0, acc[j]);
    }
}

template <typename T>
static void dwconv_c4_launch(T* out, const T* x, const T* w, const T* bias, int H,
                             int W, int C, int batch, cudaStream_t stream) {
    if ((C & 3) != 0) {  // odd channel counts keep the scalar kernel
        dwconv_launch<T>(out, x, w, bias, H, W, C, batch, stream);
        return;
    }
    int groups = batch * H * ((W + 3) >> 2) * (C >> 2);
    if (groups <= 0) return;
    flash_rt::pdl::launch(dwconv3x3_res_c4_kernel<T>, dim3((groups + 255) / 256), dim3(256), 0, stream,
        x, w, bias, out, H, W, C, groups);
}


// ── window partition / reverse ──────────────────────────────────────────────
//
// The windowed QKV is computed on the tokens *in image order* rather than on a
// gathered, zero-padded window copy: `win_qkv_pad` scatters each real token
// into its slot in the FA2 operand and zero-fills the rest, and
// `win_out_unpad` reads each real slot back. The reference runs the QKV and the
// projection on the padded window layout, but a padded row has q = v = 0, so its
// attention output is 0 and its projection is exactly the bias — the row is
// discarded by the unpad either way. Computing it was 2.9x wasted rows at
// stages 2 and 3, and cost a separate gather and scatter pass.

template <typename T>
__global__ void window_gather_kernel(const T* __restrict__ x, T* __restrict__ win,
                                     int H, int W, int C, int ws, int nww,
                                     int nwin, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C;
    int q = idx / C;
    int i = q % (ws * ws);
    int wid = q / (ws * ws);
    int img = wid / nwin, wr = wid - img * nwin;   // batch: windows are [image][w]
    int wh = wr / nww;
    int ww = wr - wh * nww;
    int h = wh * ws + i / ws;
    int w = ww * ws + i % ws;
    // Padded rows/columns are zeros; they are NOT masked out of the attention.
    win[idx] = (h < H && w < W)
                   ? x[((size_t)img * H * W + h * W + w) * C + c]
                   : Cvt<T>::from(0.0f);
}

template <typename T>
__global__ void window_scatter_kernel(const T* __restrict__ win, T* __restrict__ x,
                                      int H, int W, int C, int ws, int nww,
                                      int nwin, int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % C;
    int q = idx / C;
    int i = q % (ws * ws);
    int wid = q / (ws * ws);
    int img = wid / nwin, wr = wid - img * nwin;
    int wh = wr / nww;
    int ww = wr - wh * nww;
    int h = wh * ws + i / ws;
    int w = ww * ws + i % ws;
    if (h < H && w < W) x[((size_t)img * H * W + h * W + w) * C + c] = win[idx];
}

template <typename T>
static void window_gather_launch(T* win, const T* x, int H, int W, int C, int ws,
                                 int batch, cudaStream_t stream) {
    int nwh = (H + ws - 1) / ws, nww = (W + ws - 1) / ws;
    int nwin = nwh * nww;
    int total = batch * nwin * ws * ws * C;
    if (total <= 0) return;
    flash_rt::pdl::launch(window_gather_kernel<T>, dim3((total + 255) / 256), dim3(256), 0, stream, x, win, H, W, C,
                                                                     ws, nww, nwin, total);
}

template <typename T>
static void window_scatter_launch(T* x, const T* win, int H, int W, int C, int ws,
                                  int batch, cudaStream_t stream) {
    int nwh = (H + ws - 1) / ws, nww = (W + ws - 1) / ws;
    int nwin = nwh * nww;
    int total = batch * nwin * ws * ws * C;
    if (total <= 0) return;
    flash_rt::pdl::launch(window_scatter_kernel<T>, dim3((total + 255) / 256), dim3(256), 0, stream, win, x, H, W, C,
                                                                      ws, nww, nwin, total);
}

void vision_window_gather(void* win, const void* x, int H, int W, int C, int ws,
                          int batch, bool half, cudaStream_t stream) {
    if (half) window_gather_launch<__half>((__half*)win, (const __half*)x, H, W, C, ws, batch, stream);
    else window_gather_launch<bf16>((bf16*)win, (const bf16*)x, H, W, C, ws, batch, stream);
}

void vision_window_scatter(void* x, const void* win, int H, int W, int C, int ws,
                           int batch, bool half, cudaStream_t stream) {
    if (half) window_scatter_launch<__half>((__half*)x, (const __half*)win, H, W, C, ws, batch, stream);
    else window_scatter_launch<bf16>((bf16*)x, (const bf16*)win, H, W, C, ws, batch, stream);
}

// ── window attention (head_dim 32) ──────────────────────────────────────────
//
// One block per (window, head). A window is at most 144 tokens of head_dim 32,
// so q/k/v live in shared memory in the token dtype and the score matrix as
// fp16 — a full fp32 score tile would need 138 KB, past the 100 KB limit.

template <typename T>
__global__ void window_attn_kernel(const T* __restrict__ qkv, T* __restrict__ out,
                                   int ws, int C, int hd, float scale) {
    flashrt_pdl_wait_and_trigger();
    extern __shared__ unsigned char smem_raw[];
    const int heads = C / hd;
    const int n = ws * ws;
    const int wid = blockIdx.x / heads;
    const int h = blockIdx.x - wid * heads;
    const int t = threadIdx.x;
    const int nthreads = blockDim.x;

    T* qs = reinterpret_cast<T*>(smem_raw);
    T* ks = qs + n * hd;
    T* vs = ks + n * hd;
    __half* ps = reinterpret_cast<__half*>(vs + n * hd);

    const T* base = qkv + (size_t)wid * n * 3 * C + h * hd;
    for (int i = t; i < n * hd; i += nthreads) {
        int r = i / hd, d = i - r * hd;
        qs[i] = base[(size_t)r * 3 * C + d];
        ks[i] = base[(size_t)r * 3 * C + C + d];
        vs[i] = base[(size_t)r * 3 * C + 2 * C + d];
    }
    __syncthreads();

    for (int i = t; i < n * n; i += nthreads) {
        int r = i / n, c = i - r * n;
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d) acc += Cvt<T>::to(qs[r * hd + d]) * Cvt<T>::to(ks[c * hd + d]);
        ps[i] = __float2half(acc * scale);
    }
    __syncthreads();

    for (int r = t; r < n; r += nthreads) {
        float m = -1e30f;
        for (int c = 0; c < n; ++c) m = fmaxf(m, __half2float(ps[r * n + c]));
        float ssum = 0.0f;
        for (int c = 0; c < n; ++c) {
            float e = __expf(__half2float(ps[r * n + c]) - m);
            ps[r * n + c] = __float2half(e);
            ssum += e;
        }
        float inv = 1.0f / ssum;
        for (int c = 0; c < n; ++c)
            ps[r * n + c] = __float2half(__half2float(ps[r * n + c]) * inv);
    }
    __syncthreads();

    T* obase = out + (size_t)wid * n * C + h * hd;
    for (int i = t; i < n * hd; i += nthreads) {
        int r = i / hd, d = i - r * hd;
        float acc = 0.0f;
        for (int c = 0; c < n; ++c) acc += __half2float(ps[r * n + c]) * Cvt<T>::to(vs[c * hd + d]);
        obase[(size_t)r * C + d] = Cvt<T>::from(acc);
    }
}

template <typename T>
static void window_attn_launch(T* out, const T* qkv, int nw, int ws, int C,
                               int heads, cudaStream_t stream) {
    int hd = C / heads;
    int n = ws * ws;
    size_t smem = (size_t)(3 * n * hd) * sizeof(T) + (size_t)n * n * sizeof(__half);
    // A 12x12 window needs ~68 KB, past the 48 KB default, so opt in once.
    static size_t granted = 48 * 1024;
    if (smem > granted) {
        cudaFuncSetAttribute(window_attn_kernel<T>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        granted = smem;
    }
    flash_rt::pdl::launch(window_attn_kernel<T>, dim3(nw * heads), dim3(256), smem, stream, qkv, out, ws, C,
                                                                   hd, rsqrtf((float)hd));
}

void vision_window_attn(void* out, const void* qkv, int nw, int ws, int C,
                        int heads, bool half, cudaStream_t stream) {
    if (half) window_attn_launch<__half>((__half*)out, (const __half*)qkv, nw, ws, C, heads, stream);
    else window_attn_launch<bf16>((bf16*)out, (const bf16*)qkv, nw, ws, C, heads, stream);
}

// ── window QKV repack for FA2 (head_dim 32 -> padded 64) ───────────────────

template <typename T>
__global__ void win_qkv_pad_kernel(const T* __restrict__ qkv,
                                   const T* __restrict__ bias,
                                   T* __restrict__ q, T* __restrict__ k,
                                   T* __restrict__ v, int H, int W, int nww,
                                   int nwin, int ws, int C, int heads) {
    flashrt_pdl_wait_and_trigger();
    // One block per (window slot, intra-window position): the mapping from the
    // slot back to the image token is block-uniform, so the divisions happen
    // once per 256 elements instead of once per element.
    const int win = blockIdx.x, r = blockIdx.y;
    const int img = win / nwin, wr = win - img * nwin;
    const int row = (wr / nww) * ws + r / ws;
    const int col = (wr % nww) * ws + r % ws;
    const bool real = (row < H && col < W);
    // A window slot past the image edge is a *padded token*: its q/k/v are the
    // block's QKV bias and take part in the softmax.  Those slots are written
    // once by win_qkv_pad_fill_kernel into this block's own region, so the
    // per-call kernel only writes real tokens.
    if (!real) return;
    const T* src = qkv + ((size_t)(img * H * W + row * W + col)) * 3 * C;
    const int base = (win * ws * ws + r) * heads * 32;  // __half2 units
    const int nd = heads * 16;                          // d < 32 pairs only
    Pack2<T>* q2 = reinterpret_cast<Pack2<T>*>(q) + base;
    Pack2<T>* k2 = reinterpret_cast<Pack2<T>*>(k) + base;
    Pack2<T>* v2 = reinterpret_cast<Pack2<T>*>(v) + base;
    for (int i = threadIdx.x; i < nd; i += blockDim.x) {
        const int hid = i >> 4, d = (i & 15) << 1;
        // Head dims 32..63 are the FA2 zero pad: they are never written by any
        // stage, so the buffer's initial zeros stand.
        const int o = hid * 32 + (i & 15);  // each head is 64 halves = 32 half2
        q2[o] = *reinterpret_cast<const Pack2<T>*>(src + hid * 32 + d);
        k2[o] = *reinterpret_cast<const Pack2<T>*>(src + C + hid * 32 + d);
        v2[o] = *reinterpret_cast<const Pack2<T>*>(src + 2 * C + hid * 32 + d);
    }
}

// One-time fill of a window block's padded slots with that block's QKV bias.
// The block's region is private (the per-call pad only writes its real slots),
// so these values persist across replays.
template <typename T>
__global__ void win_qkv_pad_fill_kernel(const T* __restrict__ bias,
                                        T* __restrict__ q, T* __restrict__ k,
                                        T* __restrict__ v, int ws, int C,
                                        int heads) {
    flashrt_pdl_wait_and_trigger();
    const int win = blockIdx.x, r = blockIdx.y;
    const int base = (win * ws * ws + r) * heads * 32;
    const int nd = heads * 16;
    Pack2<T>* q2 = reinterpret_cast<Pack2<T>*>(q) + base;
    Pack2<T>* k2 = reinterpret_cast<Pack2<T>*>(k) + base;
    Pack2<T>* v2 = reinterpret_cast<Pack2<T>*>(v) + base;
    for (int i = threadIdx.x; i < nd; i += blockDim.x) {
        const int hid = i >> 4, d = (i & 15) << 1;
        const int o = hid * 32 + (i & 15);
        q2[o] = *reinterpret_cast<const Pack2<T>*>(bias + hid * 32 + d);
        k2[o] = *reinterpret_cast<const Pack2<T>*>(bias + C + hid * 32 + d);
        v2[o] = *reinterpret_cast<const Pack2<T>*>(bias + 2 * C + hid * 32 + d);
    }
}

void vision_window_qkv_pad(void* q, void* k, void* v, const void* qkv,
                           const void* bias, int H, int W, int ws, int C,
                           int heads, int nwin, int batch, bool half,
                           cudaStream_t stream) {
    int nww = (W + ws - 1) / ws;
    if (batch * nwin * ws * ws <= 0) return;
    dim3 grid(batch * nwin, ws * ws);
    if (half)
        flash_rt::pdl::launch(win_qkv_pad_kernel<__half>, dim3(grid), dim3(256), 0, stream,
            (const __half*)qkv, (const __half*)bias, (__half*)q, (__half*)k,
            (__half*)v, H, W, nww, nwin, ws, C, heads);
    else
        flash_rt::pdl::launch(win_qkv_pad_kernel<bf16>, dim3(grid), dim3(256), 0, stream,
            (const bf16*)qkv, (const bf16*)bias, (bf16*)q, (bf16*)k,
            (bf16*)v, H, W, nww, nwin, ws, C, heads);
}

void vision_window_qkv_pad_fill(void* q, void* k, void* v, const void* bias,
                                   int nw, int ws, int C, int heads, bool half,
                                   cudaStream_t stream) {
    if (nw <= 0) return;
    dim3 grid(nw, ws * ws);
    if (half)
        flash_rt::pdl::launch(win_qkv_pad_fill_kernel<__half>, dim3(grid), dim3(256), 0, stream,
            (const __half*)bias, (__half*)q, (__half*)k, (__half*)v, ws, C, heads);
    else
        flash_rt::pdl::launch(win_qkv_pad_fill_kernel<bf16>, dim3(grid), dim3(256), 0, stream,
            (const bf16*)bias, (bf16*)q, (bf16*)k, (bf16*)v, ws, C, heads);
}

// The window output unpad writes one row per block and the proj GEMM then
// quantizes that row, so the row is written and read back for nothing. This
// stages it in shared memory instead and packs the NVFP4 operand directly: the
// only statistic is a per-16-element maximum, which does not care about
// reduction order, so it is bit-identical to unpad followed by quantize.
template <typename T>
__global__ void win_out_unpad_to_nvfp4_kernel(const T* __restrict__ o,
                                              uint8_t* __restrict__ packed,
                                              uint8_t* __restrict__ sf_swz, int H,
                                              int W, int nww, int nwin, int ws,
                                              int C, int heads,
                                              int groups_per_token,
                                              int total_groups,
                                              int n_col_blocks) {
    flashrt_pdl_wait_and_trigger();
    // One thread per 16-element scale block of the unpadded (token-order) row.
    // A block's 16 channels live inside one head's real 32-wide half, so they
    // are 32 contiguous bytes in the window buffer: two 16-byte loads, the
    // block maximum, one 8-byte packed store and one swizzled scale byte.  The
    // per-block arithmetic is the row-staged kernel's verbatim, so the packed
    // operand and scale factors are unchanged.
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total_groups) return;
    const int t = g / groups_per_token;
    const int blk = g - t * groups_per_token;
    const int img = t / (H * W), rr = t - img * (H * W);
    const int row = rr / W, col = rr % W;
    const int win = img * nwin + (row / ws) * nww + col / ws;
    const int r = (row % ws) * ws + col % ws;
    const int hid = blk >> 1;
    const int d = (blk & 1) << 4;
    const T* src = o + ((size_t)win * ws * ws + r) * heads * 64 + hid * 64 + d;
    const uint4 lo = *reinterpret_cast<const uint4*>(src);
    const uint4 hi = *reinterpret_cast<const uint4*>(src + 8);
    const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    float vals[16];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const Pack2<T> pv = *reinterpret_cast<const Pack2<T>*>(&words[j]);
        vals[2 * j] = Cvt<T>::to(pv.x);
        vals[2 * j + 1] = Cvt<T>::to(pv.y);
        amax = fmaxf(amax, fmaxf(fabsf(vals[2 * j]), fabsf(vals[2 * j + 1])));
    }
    const uint8_t ue = float_to_ue4m3_ceil(amax / 6.0f);
    const int rb = t >> 7, ri = t & 127;
    sf_swz[(rb * n_col_blocks + (blk >> 2)) * 512 + (ri % 32) * 16 +
           (ri / 32) * 4 + (blk & 3)] = ue;
    const float s = ue4m3_to_float(ue);
    const float inv = (s > 0.f) ? 1.f / s : 0.f;
    uint64_t enc = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint8_t byte =
            (uint8_t)((float_to_fp4_e2m1_branchless(vals[2 * j + 1] * inv) << 4) |
                      (float_to_fp4_e2m1_branchless(vals[2 * j] * inv) & 0x0F));
        enc |= (uint64_t)byte << (8 * j);
    }
    *reinterpret_cast<uint64_t*>(packed + (size_t)g * 8) = enc;
}

template <typename T>
static void win_unpad_q_launch(const T* o, uint8_t* packed, uint8_t* sf_swz, int H,
                               int W, int ws, int C, int heads, int batch,
                               cudaStream_t stream) {
    const int nww = (W + ws - 1) / ws;
    const int nwin = ((H + ws - 1) / ws) * nww;
    const int groups_per_token = C >> 4;
    const int total_groups = batch * H * W * groups_per_token;
    flash_rt::pdl::launch(win_out_unpad_to_nvfp4_kernel<T>, dim3((total_groups + 255) / 256), dim3(256), 0, stream,
            o, packed, sf_swz, H, W, nww, nwin, ws, C, heads, groups_per_token,
            total_groups, (groups_per_token + 3) >> 2);
}

void vision_window_out_unpad_to_nvfp4(void* packed, void* sf_swz, const void* o,
                                      int H, int W, int ws, int C, int heads,
                                      int batch, bool half, cudaStream_t stream) {
    if (batch * H * W <= 0 || (C & 15)) return;
    if (half)
        win_unpad_q_launch<__half>((const __half*)o, (uint8_t*)packed,
                                   (uint8_t*)sf_swz, H, W, ws, C, heads, batch,
                                   stream);
    else
        win_unpad_q_launch<bf16>((const bf16*)o, (uint8_t*)packed,
                                 (uint8_t*)sf_swz, H, W, ws, C, heads, batch,
                                 stream);
}

// ── grouped channel attention ───────────────────────────────────────────────
//
// DaViT's channel attention treats the channels *within a group* as the
// attention sequence and the spatial token count N as the feature dimension:
// `Florence2VisionChannelAttention` hands eager_attention_forward a query of
// shape [B, G, hg, N] (heads = G, seq = hg, head_dim = N) with scale N**-0.5,
// so
//     attn[g,i,j] = softmax_j( sum_n q[g,i,n] k[g,j,n] * N**-0.5 )
//     out [g,i,n] = sum_j attn[g,i,j] v[g,j,n]
// Only G*hg*hg = C*hg values come out of an N-element contraction, so the
// stage is a reduction, not a matmul: its entire cost is C*hg*N multiply-adds
// (25.7M at stage 0, 3.2M at stage 3).
//
// The form below splits that reduction over S token chunks with 2x2 register
// tiles and 2x2 vector loads, then folds the cross-chunk sum, the row softmax
// and the apply into one block per (group, token tile). The previous form
// measured 0.22-0.29 TFLOPS at every stage (3.02 ms across the tower) because
// each FMA cost two shared-memory reads, only `groups` blocks ever ran, and
// the apply read attn[i][j] through a 32-way-strided global access.

constexpr int HG32 = 32;  // embed_dim/groups at every X-VLA DaViT stage
constexpr int APPLY_TB = 128;  // tokens per apply block

__device__ __forceinline__ void ld2f(const __half* p, float& a, float& b) {
    const __half2 v = *reinterpret_cast<const __half2*>(p);
    a = __half2float(v.x);
    b = __half2float(v.y);
}

__device__ __forceinline__ void ld2f(const bf16* p, float& a, float& b) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(p);
    a = __bfloat162float(v.x);
    b = __bfloat162float(v.y);
}

// partial[b][g][s][i][j] = sum over image b's tokens [s*chunk, (s+1)*chunk) of
// q[g,i,n] k[g,j,n]. N is one image's token count; `batch` images are stacked
// along the token axis of `qkv`, and every reduction stays inside one image.
template <typename T, int HGD>  // HGD = hg/2; block = HGD*HGD threads
__global__ void channel_scores_split_kernel(const T* __restrict__ qkv,
                                            float* __restrict__ partial, int N,
                                            int C, int S, int batch) {
    flashrt_pdl_wait_and_trigger();
    const int hg = 2 * HGD;
    const int g = blockIdx.x, s = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const int i0 = (t % HGD) * 2, j0 = (t / HGD) * 2;
    qkv += (size_t)b * N * 3 * C;
    const T* qb = qkv + g * hg + i0;
    const T* kb = qkv + C + g * hg + j0;
    const size_t row = (size_t)3 * C;
    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    const int chunk = (N + S - 1) / S;
    const int r1 = min(N, s * chunk + chunk);
    for (int r = s * chunk; r < r1; ++r) {
        float q0, q1, k0, k1;
        ld2f(qb + (size_t)r * row, q0, q1);
        ld2f(kb + (size_t)r * row, k0, k1);
        a00 = fmaf(q0, k0, a00);
        a01 = fmaf(q0, k1, a01);
        a10 = fmaf(q1, k0, a10);
        a11 = fmaf(q1, k1, a11);
    }
    float* d = partial + (((size_t)b * gridDim.x + g) * S + s) * hg * hg + i0 * hg + j0;
    d[0] = a00;
    d[1] = a01;
    d[hg] = a10;
    d[hg + 1] = a11;
}

// Same scores, with the group's q and k staged in shared memory first.
//
// In the direct kernel a thread owns the 2x2 block at (i0, j0) and reads its own
// i0 column of q and j0 column of k across all N tokens, so with a 16x16 thread
// grid every q element is read by the 16 threads that differ only in j0, and
// every k element by the 16 that differ only in i0 -- 16x redundant traffic
// through L2 for the same 3.2 MB of unique data. Staging the chunk once and
// reading it from shared memory removes that without touching the arithmetic:
// the accumulation order over tokens is unchanged, so the result is identical.
template <typename T, int HGD>
__global__ void channel_scores_split_smem_kernel(const T* __restrict__ qkv,
                                                 float* __restrict__ partial,
                                                 int N, int C, int S, int batch) {
    flashrt_pdl_wait_and_trigger();
    extern __shared__ unsigned char smem_raw[];
    T* sq = reinterpret_cast<T*>(smem_raw);
    const int hg = 2 * HGD;
    const int g = blockIdx.x, s = blockIdx.y, b = blockIdx.z, t = threadIdx.x;
    const int i0 = (t % HGD) * 2, j0 = (t / HGD) * 2;
    const T* qkv_b = qkv + (size_t)b * N * 3 * C;
    const size_t row = (size_t)3 * C;
    const int chunk = (N + S - 1) / S;
    const int r0 = s * chunk;
    const int r1 = min(N, r0 + chunk);
    const int rc = r1 - r0;
    T* sk = sq + (size_t)rc * hg;

    // Stage this block's token chunk of q and of k, group g's channels only.
    {
        const T* qg = qkv_b + g * hg;
        const T* kg = qkv_b + C + g * hg;
        const int total = rc * hg;
        if ((hg & 3) == 0 && (row & 3) == 0) {
            // Four elements per thread: the row stride 3C and the 32-wide group
            // both make the 8-byte windows aligned, and this loop is a pure
            // copy, so the staged tile is the same bytes in the same slots.
            const int total4 = total >> 2;
            for (int idx4 = t; idx4 < total4; idx4 += blockDim.x) {
                const int e = idx4 << 2;
                const int r = e / hg, c = e - r * hg;
                const size_t src = (size_t)(r0 + r) * row + c;
                dw_cp4<T>(qg + src, sq + e);
                dw_cp4<T>(kg + src, sk + e);
            }
        } else {
            for (int idx = t; idx < total; idx += blockDim.x) {
                const int r = idx / hg, c = idx - r * hg;
                sq[idx] = qg[(size_t)(r0 + r) * row + c];
                sk[idx] = kg[(size_t)(r0 + r) * row + c];
            }
        }
    }
    __syncthreads();

    float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
    for (int r = 0; r < rc; ++r) {
        float q0, q1, k0, k1;
        ld2f(sq + (size_t)r * hg + i0, q0, q1);
        ld2f(sk + (size_t)r * hg + j0, k0, k1);
        a00 = fmaf(q0, k0, a00);
        a01 = fmaf(q0, k1, a01);
        a10 = fmaf(q1, k0, a10);
        a11 = fmaf(q1, k1, a11);
    }
    float* d = partial + (((size_t)b * gridDim.x + g) * S + s) * hg * hg + i0 * hg + j0;
    d[0] = a00;
    d[1] = a01;
    d[hg] = a10;
    d[hg + 1] = a11;
}

// out[n][g*hg + i] = sum_j softmax_j(partials)[g][i][j] * v[n][g*hg + j]
template <typename T>
__global__ void channel_apply_fused_kernel(const float* __restrict__ partial,
                                           const T* __restrict__ qkv,
                                           T* __restrict__ out, int N, int C,
                                           int S, int batch, float scale) {
    flashrt_pdl_wait_and_trigger();
    constexpr int HG = HG32, TG = 256 / HG;
    const int g = blockIdx.x;
    const int b = blockIdx.z;
    const int n0 = blockIdx.y * APPLY_TB;
    const int t = threadIdx.x;
    const int i = t % HG;   // channel within the group
    const int tg = t / HG;  // token lane
    __shared__ float sa[HG * (HG + 1)];  // +1 pad: the register load below would
                                         // otherwise be a 32-way bank conflict
    qkv += (size_t)b * N * 3 * C;
    out += (size_t)b * N * C;
    const float* pj = partial + ((size_t)b * gridDim.x + g) * S * HG * HG;
    for (int e = t; e < HG * HG; e += HG * TG) {
        float a = 0.f;
        for (int s = 0; s < S; ++s) a += pj[(size_t)s * HG * HG + e];
        sa[e + e / HG] = a;
    }
    __syncthreads();
    if (t < HG) {  // row softmax, one thread per row
        float m = -1e30f;
        for (int j = 0; j < HG; ++j) m = fmaxf(m, sa[t * (HG + 1) + j] * scale);
        float sum = 0.f;
        for (int j = 0; j < HG; ++j) {
            float e = __expf(sa[t * (HG + 1) + j] * scale - m);
            sa[t * (HG + 1) + j] = e;
            sum += e;
        }
        float inv = 1.f / sum;
        for (int j = 0; j < HG; ++j) sa[t * (HG + 1) + j] *= inv;
    }
    __syncthreads();
    float ar[HG];
#pragma unroll
    for (int j = 0; j < HG; ++j) ar[j] = sa[i * (HG + 1) + j];

    const T* vbase = qkv + 2 * C + g * HG;  // vp[j] is the group's channel j
    const int nend = min(N, n0 + APPLY_TB);
    for (int n = n0 + tg; n < nend; n += TG) {
        const T* vp = vbase + (size_t)n * 3 * C;
        float acc = 0.f;
        // Four v channels per load: the inner loop otherwise issues one 2-byte
        // load per fma. Groups are 32 wide and the row stride 3*C is even, so
        // the 8-byte windows are aligned; the accumulation order over j is
        // unchanged, so the result is identical.
#pragma unroll
        for (int j = 0; j < HG; j += 4) {
            float v[4];
            dw_ld4<T>(vp + j, v);
#pragma unroll
            for (int dc = 0; dc < 4; ++dc) acc = fmaf(ar[j + dc], v[dc], acc);
        }
        out[(size_t)n * C + g * HG + i] = Cvt<T>::from(acc);
    }
}

// Per-group channel-attention scores.
//
// A thread-per-(i,j) loop over n would stride 3*C elements per step, so every
// 2-byte load pulls a whole sector and stage 2's channel blocks alone cost
// ~8 GB of traffic. Instead the block stages NT contiguous n-rows of q and k
// into shared memory with coalesced loads, then accumulates the 32x32 tile.
// Retained as the general-width path; X-VLA's DaViT always has hg == 32 and
// takes the split/fused form above.
template <typename T, int NT>
__global__ void channel_scores_kernel(const T* __restrict__ qkv,
                                      float* __restrict__ attn, int N, int C,
                                      int hg, float scale) {
    flashrt_pdl_wait_and_trigger();
    extern __shared__ unsigned char smem_raw[];
    T* qs = reinterpret_cast<T*>(smem_raw);               // [NT, hg]
    T* ks = qs + NT * hg;                                 // [NT, hg]
    float* acc = reinterpret_cast<float*>(ks + NT * hg);  // [hg, hg]

    const int g = blockIdx.x;
    const int t = threadIdx.x;
    const int i = t / hg;
    const int j = t - i * hg;
    if (t < hg * hg) acc[t] = 0.0f;
    __syncthreads();

    const T* qbase = qkv + g * hg;
    const T* kbase = qkv + C + g * hg;
    for (int n0 = 0; n0 < N; n0 += NT) {
        int rows = min(NT, N - n0);
        __syncthreads();
        for (int e = t; e < NT * hg; e += blockDim.x) {
            int r = e / hg, d = e - r * hg;
            if (r < rows) {
                qs[e] = qbase[(size_t)(n0 + r) * 3 * C + d];
                ks[e] = kbase[(size_t)(n0 + r) * 3 * C + d];
            } else {
                qs[e] = Cvt<T>::from(0.0f);
                ks[e] = Cvt<T>::from(0.0f);
            }
        }
        __syncthreads();
        if (t < hg * hg) {
            float a = acc[t];
#pragma unroll 4
            for (int r = 0; r < NT; ++r)
                a += Cvt<T>::to(qs[r * hg + i]) * Cvt<T>::to(ks[r * hg + j]);
            acc[t] = a;
        }
    }
    __syncthreads();

    // Row softmax over j, one thread per row.
    if (t < hg * hg && t % hg == 0) {
        float m = -1e30f;
        for (int c = 0; c < hg; ++c) m = fmaxf(m, acc[i * hg + c] * scale);
        float ssum = 0.0f;
        for (int c = 0; c < hg; ++c) {
            float e = __expf(acc[i * hg + c] * scale - m);
            acc[i * hg + c] = e;
            ssum += e;
        }
        float inv = 1.0f / ssum;
        for (int c = 0; c < hg; ++c) acc[i * hg + c] *= inv;
    }
    __syncthreads();
    for (int k = t; k < hg * hg; k += blockDim.x) attn[g * hg * hg + k] = acc[k];
}

template <typename T>
__global__ void channel_apply_kernel(const float* __restrict__ attn,
                                     const T* __restrict__ qkv,
                                     T* __restrict__ out, int N, int C, int hg,
                                     int total) {
    flashrt_pdl_wait_and_trigger();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int n = idx / C;
    int c = idx - n * C;
    int g = c / hg;
    int i = c - g * hg;
    const float* a = attn + g * hg * hg + i * hg;
    const T* v = qkv + (size_t)n * 3 * C + 2 * C + g * hg;
    float acc = 0.0f;
    for (int j = 0; j < hg; ++j) acc += a[j] * Cvt<T>::to(v[j]);
    out[idx] = Cvt<T>::from(acc);
}

template <typename T>
static void channel_attn_launch(T* out, const T* qkv, float* attn_ws, int N,
                                int C, int groups, int batch, cudaStream_t stream) {
    int hg = C / groups;
    if (hg == HG32 && N > 0) {
        constexpr int HGD = HG32 / 2;
        int S = (N + 511) / 512;  // ~512 tokens per score split
        S = S < 1 ? 1 : (S > 16 ? 16 : S);
        // Stage the chunk in shared memory: the direct kernel's 16x16 thread
        // grid re-reads each q and k element 16 times through L2. chunk <= 512
        // by construction above, so the staging is at most 64 KB.
        const int chunk = (N + S - 1) / S;
        const size_t smem = 2 * (size_t)chunk * HG32 * sizeof(T);
        static bool smem_attr_set = false;
        if (!smem_attr_set && smem > 48 * 1024) {
            cudaFuncSetAttribute(channel_scores_split_smem_kernel<T, HGD>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem);
            smem_attr_set = true;
        }
        flash_rt::pdl::launch(channel_scores_split_smem_kernel<T, HGD>, dim3(groups, S, batch), dim3(HGD * HGD), smem, stream, qkv, attn_ws, N, C,
                                                                  S, batch);
        flash_rt::pdl::launch(channel_apply_fused_kernel<T>, dim3(groups, (N + APPLY_TB - 1) / APPLY_TB, batch), dim3(256), 0, stream,
                attn_ws, qkv, out, N, C, S, batch, rsqrtf((float)N));
        return;
    }
    int threads = hg * hg;
    constexpr int NT = 64;
    size_t smem = (size_t)(2 * NT * hg) * sizeof(T) + (size_t)threads * sizeof(float);
    static size_t granted_scores = 48 * 1024;
    if (smem > granted_scores) {
        cudaFuncSetAttribute(channel_scores_kernel<T, NT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        granted_scores = smem;
    }
    // The scale is N ** -0.5: the number of spatial tokens, not the group width.
    // The general-width path (unused by X-VLA, whose DaViT has hg == 32 at all
    // four stages) runs one image at a time.
    for (int b = 0; b < batch; ++b) {
        flash_rt::pdl::launch(channel_scores_kernel<T, NT>, dim3(groups), dim3(threads), smem, stream,
            qkv + (size_t)b * N * 3 * C, attn_ws + (size_t)b * groups * hg * hg, N,
            C, hg, rsqrtf((float)N));
        int total = N * C;
        flash_rt::pdl::launch(channel_apply_kernel<T>, dim3((total + 255) / 256), dim3(256), 0, stream,
            attn_ws + (size_t)b * groups * hg * hg, qkv + (size_t)b * N * 3 * C,
            out + (size_t)b * N * C, N, C, hg, total);
    }
}

void vision_channel_attn(void* out, const void* qkv, float* attn_ws, int N, int C,
                         int groups, int batch, bool half, cudaStream_t stream) {
    if (half) channel_attn_launch<__half>((__half*)out, (const __half*)qkv, attn_ws, N, C, groups, batch, stream);
    else channel_attn_launch<bf16>((bf16*)out, (const bf16*)qkv, attn_ws, N, C, groups, batch, stream);
}

// ── projector ───────────────────────────────────────────────────────────────

template <typename T>
__global__ void projector_pack_kernel(const T* __restrict__ feat,
                                      const T* __restrict__ pos,
                                      const T* __restrict__ temporal,
                                      T* __restrict__ out, int rows, int dim) {
    flashrt_pdl_wait_and_trigger();
    // One thread per channel; a strided loop walks the rows so the pooled token
    // and the patch tokens come out of a single pass.
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    float sum = 0.0f;
    for (int r = 0; r < rows; ++r) {
        float v = Cvt<T>::to(feat[r * dim + c]) + Cvt<T>::to(pos[r * dim + c]) +
                  Cvt<T>::to(temporal[c]);
        out[(r + 1) * dim + c] = Cvt<T>::from(v);
        sum += v;
    }
    out[c] = Cvt<T>::from(sum / (float)rows);
}

template <typename T>
static void projector_pack_launch(T* out, const T* feat, const T* pos,
                                  const T* temporal, int rows, int dim,
                                  cudaStream_t stream) {
    flash_rt::pdl::launch(projector_pack_kernel<T>, dim3((dim + 255) / 256), dim3(256), 0, stream, feat, pos,
                                                                   temporal, out,
                                                                   rows, dim);
}

void vision_projector_pack(void* out, const void* feat, const void* pos,
                           const void* temporal, int rows, int dim, bool half,
                           cudaStream_t stream) {
    if (half) projector_pack_launch<__half>((__half*)out, (const __half*)feat, (const __half*)pos, (const __half*)temporal, rows, dim, stream);
    else projector_pack_launch<bf16>((bf16*)out, (const bf16*)feat, (const bf16*)pos, (const bf16*)temporal, rows, dim, stream);
}

}  // namespace xvla
}  // namespace flash_rt

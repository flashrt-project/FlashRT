// One-time repack of a ggml GGML_TYPE_NVFP4 weight tensor into the CUTLASS
// block-scaled GEMM's B-side wire format.
//
// ggml block_nvfp4 (36 bytes / 64 elements):
//   uint8 d[4]   - one e4m3 scale per 16-element sub-block; standard e4m3
//                  semantics (ggml's doubled dequant table and halved ue4m3
//                  decode cancel), passed through unmodified
//   uint8 qs[32] - e2m1 codes, sub-block s at qs[s*8..s*8+7], byte j holding
//                  elem[j] in the low nibble and elem[8+j] in the high nibble
//
// Output: packed uint8 [N, K/2] with adjacent-pair nibbles (elem 2i low,
// elem 2i+1 high) and the scale bytes at the Sm1xx atom-layout offsets.
// One thread handles one 16-element sub-block.

#include "fr_kernels.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace ggml_cuda_flashrt {

namespace {

using CfgVec = cutlass::detail::Sm1xxBlockScaledConfig<16>;

constexpr int GGML_NVFP4_BLOCK_BYTES = 36; // 4 scale bytes + 32 data bytes

template <class LayoutSF>
__global__ void kernel_repack(
        const uint8_t * __restrict__ src,  // ggml block_nvfp4 stream
        uint2 * __restrict__ dst_packed,   // [N, K/2] bytes as uint2 per sub-block
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int N, int K16) {                  // K16 = K / 16 sub-blocks per row
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= N || t >= K16) return;

    const int blk = t >> 2;      // 64-element ggml block
    const int sub = t & 3;       // 16-element sub-block within it

    const uint8_t * b = src + (static_cast<int64_t>(row) * (K16 >> 2) + blk) * GGML_NVFP4_BLOCK_BYTES;
    const uint8_t scale = b[sub];
    const uint8_t * qs = b + 4 + sub * 8;

    uint2 out;
    uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        // elements 2p, 2p+1 live in the low nibbles of qs[2p], qs[2p+1]
        ob[p]     = static_cast<uint8_t>((qs[2 * p] & 0x0F)  | ((qs[2 * p + 1] & 0x0F) << 4));
        // elements 8+2p, 8+2p+1 live in the high nibbles of the same bytes
        ob[p + 4] = static_cast<uint8_t>((qs[2 * p] >> 4)    | ((qs[2 * p + 1] & 0xF0)));
    }

    dst_packed[static_cast<int64_t>(row) * K16 + t] = out;
    dst_sf[layout(row, t * 16, 0)] = scale;
}

} // namespace

namespace {

// Pairwise-interleaved variant for the fused GeGLU GEMM: output row 2j is
// gate row j, row 2j+1 is up row j (N_il = 2 * n_ff rows total).
template <class LayoutSF>
__global__ void kernel_repack_pair(
        const uint8_t * __restrict__ gate,
        const uint8_t * __restrict__ up,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int n_ff, int K16) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;             // row within gate/up
    const int which = blockIdx.z;           // 0 = gate, 1 = up
    if (row >= n_ff || t >= K16) return;

    const int blk = t >> 2;
    const int sub = t & 3;
    const uint8_t * src = which ? up : gate;
    const uint8_t * b = src + (static_cast<int64_t>(row) * (K16 >> 2) + blk) * GGML_NVFP4_BLOCK_BYTES;
    const uint8_t scale = b[sub];
    const uint8_t * qs = b + 4 + sub * 8;

    uint2 out;
    uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        ob[p]     = static_cast<uint8_t>((qs[2 * p] & 0x0F) | ((qs[2 * p + 1] & 0x0F) << 4));
        ob[p + 4] = static_cast<uint8_t>((qs[2 * p] >> 4)   | ((qs[2 * p + 1] & 0xF0)));
    }

    const int out_row = 2 * row + which;
    dst_packed[static_cast<int64_t>(out_row) * K16 + t] = out;
    dst_sf[layout(out_row, t * 16, 0)] = scale;
}

} // namespace

int repack_weight_pair_interleaved(const void * gate_blocks, const void * up_blocks,
                                   void * dst_packed, void * dst_sf,
                                   int n_ff, int K, cudaStream_t stream) {
    if (K % 64 != 0) return -1;
    if (reinterpret_cast<uintptr_t>(dst_packed) & 7) return -1;

    const int K16  = K / 16;
    const int N_il = 2 * n_ff;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, n_ff, 2);

    auto shape = cute::make_shape(1, N_il, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N_il, K)) {
        return -3;
    }

    kernel_repack_pair<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(gate_blocks),
        reinterpret_cast<const uint8_t *>(up_blocks),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, n_ff, K16);

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

namespace {

// Rows-padded variant: rows >= N_src emit zero data and zero scales
// (mathematically inert; the extra rows exist only for output alignment).
template <class LayoutSF>
__global__ void kernel_repack_rows_padded(
        const uint8_t * __restrict__ src,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int N_src, int N_pad, int K16) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= N_pad || t >= K16) return;

    uint2 out = make_uint2(0, 0);
    uint8_t scale = 0;
    if (row < N_src) {
        const int blk = t >> 2;
        const int sub = t & 3;
        const uint8_t * b = src + (static_cast<int64_t>(row) * (K16 >> 2) + blk) * GGML_NVFP4_BLOCK_BYTES;
        scale = b[sub];
        const uint8_t * qs = b + 4 + sub * 8;
        uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            ob[p]     = static_cast<uint8_t>((qs[2 * p] & 0x0F) | ((qs[2 * p + 1] & 0x0F) << 4));
            ob[p + 4] = static_cast<uint8_t>((qs[2 * p] >> 4)   | ((qs[2 * p + 1] & 0xF0)));
        }
    }

    dst_packed[static_cast<int64_t>(row) * K16 + t] = out;
    dst_sf[layout(row, t * 16, 0)] = scale;
}

// Group-padded variant: output rows are n_groups groups of group_out rows,
// the first group_in of each group copied from the source (group g, lane l ->
// source row g*group_in + l) and the rest zero. Used to widen per-head
// projections (e.g. SigLIP head_dim 72 -> 80) entirely inside the weights so
// no runtime pad kernel is needed.
template <class LayoutSF>
__global__ void kernel_repack_rows_grouppad(
        const uint8_t * __restrict__ src,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int group_in, int group_out, int n_groups, int K16) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= group_out * n_groups || t >= K16) return;

    const int lane = row % group_out;
    uint2 out = make_uint2(0, 0);
    uint8_t scale = 0;
    if (lane < group_in) {
        const int src_row = (row / group_out) * group_in + lane;
        const int blk = t >> 2;
        const int sub = t & 3;
        const uint8_t * b = src + (static_cast<int64_t>(src_row) * (K16 >> 2) + blk) * GGML_NVFP4_BLOCK_BYTES;
        scale = b[sub];
        const uint8_t * qs = b + 4 + sub * 8;
        uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            ob[p]     = static_cast<uint8_t>((qs[2 * p] & 0x0F) | ((qs[2 * p + 1] & 0x0F) << 4));
            ob[p + 4] = static_cast<uint8_t>((qs[2 * p] >> 4)   | ((qs[2 * p + 1] & 0xF0)));
        }
    }

    dst_packed[static_cast<int64_t>(row) * K16 + t] = out;
    dst_sf[layout(row, t * 16, 0)] = scale;
}

__device__ __forceinline__ uint8_t fr_f32_to_e2m1(float x) {
    uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    float ax = fabsf(x);
    uint8_t m;
    if      (ax <= 0.25f) m = 0u;
    else if (ax <= 0.75f) m = 1u;
    else if (ax <= 1.25f) m = 2u;
    else if (ax <= 1.75f) m = 3u;
    else if (ax <= 2.5f)  m = 4u;
    else if (ax <= 3.5f)  m = 5u;
    else if (ax <= 5.0f)  m = 6u;
    else                  m = 7u;
    return sign | m;
}

// Quantize an fp16 weight [N rows, K_src] to the NVFP4 wire format with the
// K dim zero-padded to K_pad (standard e4m3 amax/6 scales, e2m1 nearest).
template <class LayoutSF>
__global__ void kernel_quantize_weight_f16_padded(
        const __half * __restrict__ src,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int N, int K_src16, int K_pad16) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= N || t >= K_pad16) return;

    uint2 out = make_uint2(0, 0);
    uint8_t sf = 0;
    if (t < K_src16) {
        const __half * xr = src + (int64_t) row * (K_src16 * 16) + t * 16;
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            vals[i] = __half2float(xr[i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }
        float desired = amax / 6.f;
        if (desired < 1e-12f) desired = 1e-12f;
        __nv_fp8_e4m3 q(desired);
        sf = *reinterpret_cast<uint8_t *>(&q);
        const float inv = 1.f / static_cast<float>(q);
        uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint8_t lo = fr_f32_to_e2m1(vals[2 * p]     * inv);
            const uint8_t hi = fr_f32_to_e2m1(vals[2 * p + 1] * inv);
            ob[p] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }

    dst_packed[static_cast<int64_t>(row) * K_pad16 + t] = out;
    dst_sf[layout(row, t * 16, 0)] = sf;
}

} // namespace

int repack_weight_rows_padded(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                              int N_src, int N_pad, int K, cudaStream_t stream) {
    if (K % 64 != 0 || N_pad < N_src) return -1;
    const int K16 = K / 16;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, N_pad);
    auto shape = cute::make_shape(1, N_pad, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N_pad, K)) return -3;
    kernel_repack_rows_padded<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(ggml_blocks),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, N_src, N_pad, K16);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int repack_weight_rows_grouppad(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                                int group_in, int group_out, int n_groups, int K, cudaStream_t stream) {
    if (K % 64 != 0 || group_out < group_in || n_groups <= 0) return -1;
    const int N_pad = group_out * n_groups;
    const int K16 = K / 16;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, N_pad);
    auto shape = cute::make_shape(1, N_pad, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N_pad, K)) return -3;
    kernel_repack_rows_grouppad<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(ggml_blocks),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, group_in, group_out, n_groups, K16);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int quantize_weight_f16_padded(const void * w_f16, void * dst_packed, void * dst_sf,
                               int N, int K_src, int K_pad, cudaStream_t stream) {
    if (K_src % 16 != 0 || K_pad % 64 != 0 || K_pad < K_src) return -1;
    const int K16 = K_pad / 16;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, N);
    auto shape = cute::make_shape(1, N, K_pad, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N, K_pad)) return -3;
    kernel_quantize_weight_f16_padded<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __half *>(w_f16),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, N, K_src / 16, K16);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

namespace {

// Three-tensor row-concat repack for the fused QKV GEMM: output rows are
// [src0 | src1 | src2] stacked (N0 + N1 + N2 rows, same K).
template <class LayoutSF>
__global__ void kernel_repack_concat3(
        const uint8_t * __restrict__ s0, int N0,
        const uint8_t * __restrict__ s1, int N1,
        const uint8_t * __restrict__ s2, int N2,
        uint2 * __restrict__ dst_packed,
        uint8_t * __restrict__ dst_sf,
        LayoutSF layout,
        int K16) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    const int N_tot = N0 + N1 + N2;
    if (row >= N_tot || t >= K16) return;

    const uint8_t * src;
    int src_row;
    if (row < N0)            { src = s0; src_row = row; }
    else if (row < N0 + N1)  { src = s1; src_row = row - N0; }
    else                     { src = s2; src_row = row - N0 - N1; }

    const int blk = t >> 2;
    const int sub = t & 3;
    const uint8_t * b = src + (static_cast<int64_t>(src_row) * (K16 >> 2) + blk) * GGML_NVFP4_BLOCK_BYTES;
    const uint8_t scale = b[sub];
    const uint8_t * qs = b + 4 + sub * 8;

    uint2 out;
    uint8_t * ob = reinterpret_cast<uint8_t *>(&out);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        ob[p]     = static_cast<uint8_t>((qs[2 * p] & 0x0F) | ((qs[2 * p + 1] & 0x0F) << 4));
        ob[p + 4] = static_cast<uint8_t>((qs[2 * p] >> 4)   | ((qs[2 * p + 1] & 0xF0)));
    }

    dst_packed[static_cast<int64_t>(row) * K16 + t] = out;
    dst_sf[layout(row, t * 16, 0)] = scale;
}

} // namespace

int repack_weight_concat3(const void * b0, int N0, const void * b1, int N1,
                          const void * b2, int N2,
                          void * dst_packed, void * dst_sf,
                          int K, cudaStream_t stream) {
    if (K % 64 != 0) return -1;
    const int K16 = K / 16;
    const int N_tot = N0 + N1 + N2;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, N_tot);
    auto shape = cute::make_shape(1, N_tot, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N_tot, K)) return -3;
    kernel_repack_concat3<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(b0), N0,
        reinterpret_cast<const uint8_t *>(b1), N1,
        reinterpret_cast<const uint8_t *>(b2), N2,
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, K16);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

int repack_weight(const void * ggml_blocks, void * dst_packed, void * dst_sf,
                  int N, int K, cudaStream_t stream) {
    if (K % 64 != 0) return -1;
    if (reinterpret_cast<uintptr_t>(dst_packed) & 7) return -1;

    const int K16 = K / 16;
    const int threads = 128;
    dim3 grid((K16 + threads - 1) / threads, N);

    // SFB layout for a [*, N, K] problem; independent of M.
    auto shape = cute::make_shape(1, N, K, 1);
    auto layout = CfgVec::tile_atom_to_shape_SFB(shape);

    // The atom layout may address up to the padded (row, k) extents; the
    // caller allocates sf_bytes(N, K) which must cover the layout codomain.
    if (static_cast<int64_t>(cute::cosize(layout)) > sf_bytes(N, K)) {
        return -3;
    }

    kernel_repack<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(ggml_blocks),
        reinterpret_cast<uint2 *>(dst_packed),
        reinterpret_cast<uint8_t *>(dst_sf),
        layout, N, K16);

    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

namespace {

struct cpy_rows_pair {
    const float * src;
    __half      * dst;
};

struct cpy_rows_args {
    cpy_rows_pair pairs[FR_CPY_ROWS_MAX];
};

// one y-slice of rows per pair; conversion identical to ggml's f32->f16 cpy
__global__ void kernel_cpy_rows_f32_f16(cpy_rows_args args, int hd, int n_rows) {
    const cpy_rows_pair p = args.pairs[blockIdx.x];
    for (int r = blockIdx.y; r < n_rows; r += gridDim.y) {
        const float * s = p.src + (int64_t) r * hd;
        __half      * d = p.dst + (int64_t) r * hd;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            d[i] = __float2half(s[i]);
        }
    }
}

} // namespace

int cpy_rows_f32_f16(const float * const * srcs, void * const * dsts, int n_pairs,
                     int hd, int n_rows, cudaStream_t stream) {
    if (n_pairs < 1 || n_pairs > FR_CPY_ROWS_MAX) {
        return -1;
    }
    cpy_rows_args args;
    for (int i = 0; i < n_pairs; ++i) {
        args.pairs[i].src = srcs[i];
        args.pairs[i].dst = (__half *) dsts[i];
    }
    const int rows_y = n_rows < 64 ? n_rows : 64;
    dim3 grid(n_pairs, rows_y);
    kernel_cpy_rows_f32_f16<<<grid, 128, 0, stream>>>(args, hd, n_rows);
    const cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : -static_cast<int>(e);
}

} // namespace ggml_cuda_flashrt

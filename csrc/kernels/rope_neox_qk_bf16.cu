// ================================================================
// FlashRT — split-half ("rotate_half" / GPT-NeoX) RoPE on Q and K
//
// Applies rotary position embedding using the rotate_half convention
// that pairs element d with d + head_dim/2, matching HuggingFace's
// apply_rotary_pos_emb:
//
//     out[..., d]        = x[..., d]      * cos[d] - x[..., d+half] * sin[d]
//     out[..., d+half]   = x[..., d+half] * cos[d] + x[..., d]      * sin[d]
//
// This is distinct from rope.cu::rope_apply, which uses the interleaved
// (d, d+1) convention. The kernel is model-agnostic and reusable by any
// attention site whose RoPE uses the rotate_half form (Qwen3 backbones,
// SigLIP-style ViT towers, etc.).
//
//   q_in / q_out : (rows, q_heads, head_dim) bf16  (in-place allowed)
//   k_in / k_out : (rows, k_heads, head_dim) bf16  (in-place allowed)
//   cos  / sin   : (rows, head_dim/2)        bf16
//
// One thread handles one (row, head, d<half) triple per tensor; Q and K
// are rotated in the same launch (k_heads may differ from q_heads).
// ================================================================

#include <cuda_bf16.h>
#include "pdl_launch.cuh"

#include "common.cuh"

namespace flash_rt {
namespace kernels {

__global__ void rope_neox_qk_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ cos_tab,
    const __nv_bfloat16* __restrict__ sin_tab,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    int rows, int q_heads, int k_heads, int head_dim) {
    const int half = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int q_total = rows * q_heads * half;
    const int k_total = rows * k_heads * half;

    if (idx < q_total) {
        const int d = idx % half;
        const int tmp = idx / half;
        const int head = tmp % q_heads;
        const int row = tmp / q_heads;
        const int base = (row * q_heads + head) * head_dim;
        const float c = to_f32(cos_tab[row * half + d]);
        const float s = to_f32(sin_tab[row * half + d]);
        const float x0 = to_f32(q_in[base + d]);
        const float x1 = to_f32(q_in[base + d + half]);
        q_out[base + d] = from_f32<__nv_bfloat16>(x0 * c - x1 * s);
        q_out[base + d + half] = from_f32<__nv_bfloat16>(x1 * c + x0 * s);
    }

    if (idx < k_total) {
        const int d = idx % half;
        const int tmp = idx / half;
        const int head = tmp % k_heads;
        const int row = tmp / k_heads;
        const int base = (row * k_heads + head) * head_dim;
        const float c = to_f32(cos_tab[row * half + d]);
        const float s = to_f32(sin_tab[row * half + d]);
        const float x0 = to_f32(k_in[base + d]);
        const float x1 = to_f32(k_in[base + d + half]);
        k_out[base + d] = from_f32<__nv_bfloat16>(x0 * c - x1 * s);
        k_out[base + d + half] = from_f32<__nv_bfloat16>(x1 * c + x0 * s);
    }
}

void rope_neox_qk_bf16(
    const __nv_bfloat16* q_in, const __nv_bfloat16* k_in,
    const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    int rows, int q_heads, int k_heads, int head_dim,
    cudaStream_t stream) {
    const int half = head_dim / 2;
    const int q_total = rows * q_heads * half;
    const int k_total = rows * k_heads * half;
    const int total = q_total > k_total ? q_total : k_total;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    rope_neox_qk_kernel<<<blocks, threads, 0, stream>>>(
        q_in, k_in, cos_tab, sin_tab, q_out, k_out,
        rows, q_heads, k_heads, head_dim);
}

// ---------------------------------------------------------------------------
// Fused qkv split (+per-head bias) + rotate_half RoPE on Q/K + V write.
//
// Replaces the [qkv_split_bias_bf16 -> rope_neox_qk_bf16] pair (two launches,
// ten bf16 megabytes of traffic at the ViT block shape) with one launch that
// streams packed_qkv once. Every unit is a 16-byte (8-element) vector: 8
// consecutive rotary pairs of one (row, head), so the bias add, the bf16
// rounding and the rotation all stay in registers. The biased values are
// rounded through bf16 before the rotation, identical to the two-kernel chain.
//
//   packed_qkv : (rows, (q_heads + k_heads + v_heads) * head_dim) bf16
//   bias       : ((q_heads + k_heads + v_heads) * head_dim,)     bf16
//   cos/sin    : (rows, head_dim / 2)                            bf16
//   q_out      : (rows, q_heads, head_dim) bf16 (rope applied)
//   k_out      : (rows, k_heads, head_dim) bf16 (rope applied)
//   v_out      : (rows, v_heads, head_dim) bf16 (bias only)
namespace {

__device__ __forceinline__ void bias_round(uint4& x, const uint4& b) {
    __nv_bfloat162* xs = reinterpret_cast<__nv_bfloat162*>(&x);
    const __nv_bfloat162* bs = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 xf = __bfloat1622float2(xs[i]);
        const float2 bf = __bfloat1622float2(bs[i]);
        xs[i] = __floats2bfloat162_rn(xf.x + bf.x, xf.y + bf.y);
    }
}

__device__ __forceinline__ void rope_pair(uint4& lo, uint4& hi,
                                          const uint4& c, const uint4& s) {
    __nv_bfloat162* pl = reinterpret_cast<__nv_bfloat162*>(&lo);
    __nv_bfloat162* ph = reinterpret_cast<__nv_bfloat162*>(&hi);
    const __nv_bfloat162* pc = reinterpret_cast<const __nv_bfloat162*>(&c);
    const __nv_bfloat162* ps = reinterpret_cast<const __nv_bfloat162*>(&s);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 x0 = __bfloat1622float2(pl[i]);
        const float2 x1 = __bfloat1622float2(ph[i]);
        const float2 cc = __bfloat1622float2(pc[i]);
        const float2 ss = __bfloat1622float2(ps[i]);
        pl[i] = __floats2bfloat162_rn(x0.x * cc.x - x1.x * ss.x,
                                       x0.y * cc.y - x1.y * ss.y);
        ph[i] = __floats2bfloat162_rn(x1.x * cc.x + x0.x * ss.x,
                                       x1.y * cc.y + x0.y * ss.y);
    }
}

}  // namespace

// Index is long or int (the int form, used whenever the sizes fit, keeps the
// flat-index decomposition off the 64-bit division path; same arithmetic).
template <typename Index>
__global__ void qkv_split_bias_rope_kernel(
    const __nv_bfloat16* __restrict__ packed_qkv,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ cos_tab,
    const __nv_bfloat16* __restrict__ sin_tab,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int rows, int q_heads, int k_heads, int v_heads, int head_dim) {
    flash_rt::pdl::pdl_trigger();
    const int half = head_dim / 2;
    const int vecs = half / 8;                 // 16-byte units per head half
    const int q_dim = q_heads * head_dim;
    const int k_dim = k_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int W = q_dim + k_dim + v_dim;
    const Index q_units = static_cast<Index>(rows) * q_heads * vecs;
    const Index k_units = static_cast<Index>(rows) * k_heads * vecs;
    const Index v_units = static_cast<Index>(rows) * v_dim / 8;
    const Index total = q_units + k_units + v_units;
    const Index stride = static_cast<Index>(gridDim.x) * blockDim.x;

    // One unit per thread when the grid covers the problem (the launcher
    // sizes it so): the bias and RoPE tables are constant, so that unit's
    // loads of them go out before the grid-dependency wait.
    const Index idx0 = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (stride >= total) {
        if (idx0 >= total) { flash_rt::pdl::pdl_wait(); return; }
        const Index idx = idx0;
        const bool is_q = idx < q_units;
        const bool is_k = !is_q && idx < q_units + k_units;
        if (is_q || is_k) {
            const Index u = is_q ? idx : idx - q_units;
            const int heads = is_q ? q_heads : k_heads;
            const int db = static_cast<int>(u % vecs) * 8;
            const Index t = u / vecs;
            const int head = static_cast<int>(t % heads);
            const Index row = t / heads;
            const int col0 = (is_q ? 0 : q_dim) + head * head_dim;
            const Index src = row * W + col0 + db;
            const Index bcol = col0 + db;
            const Index dst = (row * heads + head) * head_dim + db;
            const uint4 b0 = *reinterpret_cast<const uint4*>(bias + bcol);
            const uint4 b1 = *reinterpret_cast<const uint4*>(bias + bcol + half);
            const uint4 c = *reinterpret_cast<const uint4*>(cos_tab + row * half + db);
            const uint4 sn = *reinterpret_cast<const uint4*>(sin_tab + row * half + db);
            flash_rt::pdl::pdl_wait();
            uint4 x0 = *reinterpret_cast<const uint4*>(packed_qkv + src);
            uint4 x1 = *reinterpret_cast<const uint4*>(packed_qkv + src + half);
            bias_round(x0, b0);
            bias_round(x1, b1);
            rope_pair(x0, x1, c, sn);
            __nv_bfloat16* out = is_q ? q_out : k_out;
            *reinterpret_cast<uint4*>(out + dst) = x0;
            *reinterpret_cast<uint4*>(out + dst + half) = x1;
        } else {
            const Index u = (idx - q_units - k_units) * 8;
            const Index row = u / v_dim;
            const int col = static_cast<int>(u - row * v_dim);
            const Index src = row * W + q_dim + k_dim + col;
            const Index bcol = q_dim + k_dim + col;
            const uint4 bb = *reinterpret_cast<const uint4*>(bias + bcol);
            flash_rt::pdl::pdl_wait();
            uint4 x = *reinterpret_cast<const uint4*>(packed_qkv + src);
            bias_round(x, bb);
            *reinterpret_cast<uint4*>(v_out + row * v_dim + col) = x;
        }
        return;
    }
    flash_rt::pdl::pdl_wait();

    for (Index idx = idx0; idx < total; idx += stride) {
        const bool is_q = idx < q_units;
        const bool is_k = !is_q && idx < q_units + k_units;
        if (is_q || is_k) {
            const Index u = is_q ? idx : idx - q_units;
            const int heads = is_q ? q_heads : k_heads;
            const int db = static_cast<int>(u % vecs) * 8;
            const Index t = u / vecs;
            const int head = static_cast<int>(t % heads);
            const Index row = t / heads;
            const int col0 = (is_q ? 0 : q_dim) + head * head_dim;
            const Index src = row * W + col0 + db;
            const Index bcol = col0 + db;
            const Index dst = (row * heads + head) * head_dim + db;
            uint4 x0 = *reinterpret_cast<const uint4*>(packed_qkv + src);
            uint4 x1 = *reinterpret_cast<const uint4*>(packed_qkv + src + half);
            bias_round(x0, *reinterpret_cast<const uint4*>(bias + bcol));
            bias_round(x1, *reinterpret_cast<const uint4*>(bias + bcol + half));
            const uint4 c = *reinterpret_cast<const uint4*>(cos_tab + row * half + db);
            const uint4 s = *reinterpret_cast<const uint4*>(sin_tab + row * half + db);
            rope_pair(x0, x1, c, s);
            __nv_bfloat16* out = is_q ? q_out : k_out;
            *reinterpret_cast<uint4*>(out + dst) = x0;
            *reinterpret_cast<uint4*>(out + dst + half) = x1;
        } else {
            const Index u = (idx - q_units - k_units) * 8;
            const Index row = u / v_dim;
            const int col = static_cast<int>(u - row * v_dim);
            const Index src = row * W + q_dim + k_dim + col;
            const Index bcol = q_dim + k_dim + col;
            uint4 x = *reinterpret_cast<const uint4*>(packed_qkv + src);
            bias_round(x, *reinterpret_cast<const uint4*>(bias + bcol));
            *reinterpret_cast<uint4*>(v_out + row * v_dim + col) = x;
        }
    }
}

void qkv_split_bias_rope_bf16(
    const __nv_bfloat16* packed_qkv, const __nv_bfloat16* bias,
    const __nv_bfloat16* cos_tab, const __nv_bfloat16* sin_tab,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    int rows, int q_heads, int k_heads, int v_heads, int head_dim,
    cudaStream_t stream) {
    const int half = head_dim / 2;
    const long q_units = static_cast<long>(rows) * q_heads * (half / 8);
    const long k_units = static_cast<long>(rows) * k_heads * (half / 8);
    const long v_units = static_cast<long>(rows) * v_heads * head_dim / 8;
    const long total = q_units + k_units + v_units;
    const int threads = 256;
    const long blocks = (total + threads - 1) / threads;
    const long elems = static_cast<long>(rows) * (q_heads + k_heads + v_heads) * head_dim;
    if (elems < (1L << 31)) {
        flash_rt::pdl::launch(qkv_split_bias_rope_kernel<int>, dim3(static_cast<unsigned>(blocks)),
                              dim3(threads), 0, stream, flash_rt::pdl::enabled(),
                              packed_qkv, bias, cos_tab, sin_tab, q_out, k_out, v_out,
                              rows, q_heads, k_heads, v_heads, head_dim);
        return;
    }
    flash_rt::pdl::launch(qkv_split_bias_rope_kernel<long>, dim3(static_cast<unsigned>(blocks)),
                          dim3(threads), 0, stream, flash_rt::pdl::enabled(),
                          packed_qkv, bias, cos_tab, sin_tab, q_out, k_out, v_out,
                          rows, q_heads, k_heads, v_heads, head_dim);
}

}  // namespace kernels
}  // namespace flash_rt

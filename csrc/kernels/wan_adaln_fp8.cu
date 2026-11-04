// Wan2.2 / LingBot-VA AdaLN kernels -- see wan_adaln.cuh for the contract.
//
// Layout note: the reference keeps the conditioning as [B, L, 6, C] and chunks
// it on the 6-axis after adding the block's [6, C] table. Rather than
// materialising that sum (which is 6*S*D fp32 per block per denoise step) this
// kernel adds the two rows it needs on the fly.
#include "wan_adaln.cuh"

namespace flash_rt {
namespace wan_adaln {

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    return smem[0];
}

// block_reduce_sum for two values at once, 256 threads, with the identical
// addition tree: the strides 128, 64 and 32 are combined by warp 0 from one
// shared-memory image (s[t] + s[t + stride], same operands, same order) and
// 16..1 by shuffles, so the two sums are bit-identical to two block_reduce_sum
// calls with one exchange instead of eighteen barriers. smem: 512 floats.
__device__ __forceinline__ float2 block_reduce_sum2_256(float a, float b, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = a;
    smem[256 + tid] = b;
    __syncthreads();
    if (tid < 32) {
        float va[8], vb[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            va[k] = smem[tid + 32 * k];
            vb[k] = smem[256 + tid + 32 * k];
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) { va[k] = va[k] + va[k + 4]; vb[k] = vb[k] + vb[k + 4]; }
#pragma unroll
        for (int k = 0; k < 2; ++k) { va[k] = va[k] + va[k + 2]; vb[k] = vb[k] + vb[k + 2]; }
        float x = va[0] + va[1], y = vb[0] + vb[1];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            x = x + __shfl_down_sync(0xffffffffu, x, o);
            y = y + __shfl_down_sync(0xffffffffu, y, o);
        }
        __syncwarp();
        if (tid == 0) {
            smem[0] = x;
            smem[256] = y;
        }
    }
    __syncthreads();
    return make_float2(smem[0], smem[256]);
}

// Pass 1 for one row: optionally accumulate `x` (scaled by `gate`, if given)
// into `residual`, and return the sum of squares of the row that follows.
// Pass 2 reads the row back, normalises, modulates and quantizes.
//
// Both the residual accumulation and the LayerNorm input are therefore the same
// buffer, which is why the normalisation has to happen after the block-wide
// reduction rather than on registers.
// Same pass-2 arithmetic, emitting the activation already quantised to FP8 in
// 128-wide scale blocks -- the `fp8_per_token_block128_quant_bf16` launch the
// caller used to run over this kernel's output, folded in.
//
// This is exact, and the reason is specific rather than general: a block scale
// is a **max**, and `fmaxf` is associative and commutative, so the reduction
// that produces it carries no ordering constraint -- unlike the sum reductions
// elsewhere in this pipeline, whose tree shape is load-bearing. The per-element
// arithmetic below is the same expression sequence as the loop above, and the
// scale/quantise arithmetic is copied from the standalone kernel
// (`s = max(amax/448, 1e-12)`; `q = clamp(v * (1/s), +-448)`; `__nv_fp8_e4m3`).
//
// One warp owns one 128-wide block, four elements per lane, so the max needs
// only warp shuffles and no `__syncthreads`.
template <typename OutT, typename QuantT, bool BLOCK128_FP8>
__device__ __forceinline__ void adaln_emit_block128_fp8(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ ss_table,
    const __nv_bfloat16* __restrict__ tproj,
    OutT* __restrict__ out,
    float* __restrict__ out_scale,
    long row, long rowidx, long ss_scale, long ss_shift, long tp_scale, long tp_shift,
    float mean, float rstd, int D)
{
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31u);
    const int nblk = D >> 7;
    const int nwarp = (int)(blockDim.x >> 5);
    for (int blk = warp; blk < nblk; blk += nwarp) {
        float vals[4];
        float amax = 0.0f;
        // the lane's four contiguous elements, loaded as vectors
        const int d0 = (blk << 7) + (lane << 2);
        const uint2 vr = *reinterpret_cast<const uint2*>(
            (residual != nullptr ? residual : x) + row + d0);
        const float4 gs = *reinterpret_cast<const float4*>(ss_table + ss_scale + d0);
        const float4 hs = *reinterpret_cast<const float4*>(ss_table + ss_shift + d0);
        const uint2 gt = *reinterpret_cast<const uint2*>(tproj + tp_scale + d0);
        const uint2 ht = *reinterpret_cast<const uint2*>(tproj + tp_shift + d0);
        const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vr);
        const __nv_bfloat16* gb = reinterpret_cast<const __nv_bfloat16*>(&gt);
        const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&ht);
        const float gsv[4] = {gs.x, gs.y, gs.z, gs.w};
        const float hsv[4] = {hs.x, hs.y, hs.z, hs.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float v = __bfloat162float(vb[j]);
            const float g = gsv[j] + __bfloat162float(gb[j]);
            const float sh = hsv[j] + __bfloat162float(hb[j]);
            const float norm_v = (v - mean) * rstd;
            const float mod_v = (1.0f + g) * norm_v + sh;
            const float mod_bf16 = __bfloat162float(__float2bfloat16_rn(mod_v));
            vals[j] = mod_bf16;
            amax = fmaxf(amax, fabsf(mod_bf16));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        const float s = fmaxf(amax / 448.0f, 1.0e-12f);
        const float inv_s = 1.0f / s;
        if (lane == 0) out_scale[rowidx * nblk + blk] = s;
        uint32_t packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float q = fminf(fmaxf(vals[j] * inv_s, -448.0f), 448.0f);
            packed |= (uint32_t)__nv_fp8_e4m3(q).__x << (8 * j);
        }
        *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(out) + row + d0) = packed;
    }
}

// Modulation row of token `tok`: tp_bcast == 0 per token, 1 token 0 for all
// (one timestep for the whole sequence), G > 1 the first token of tok's group of
// G (one timestep per G-token group, e.g. the Picard sweep's 16-token
// replicas). The caller enables 1 / G only when those rows are equal, so every
// form reads the same values.
__device__ __forceinline__ long tp_row_(int tp_bcast, long tok) {
    return tp_bcast == 1 ? 0 : (tp_bcast > 1 ? tok - tok % tp_bcast : tok);
}

template <typename OutT, typename QuantT, bool BLOCK128_FP8 = false>
__global__ void adaln_kernel(
    __nv_bfloat16* __restrict__ residual,       // [B,S,D] or null
    const __nv_bfloat16* __restrict__ x,        // [B,S,D] or null
    const float* __restrict__ gate_vec,         // [S,D] fp32, or null
    const float* __restrict__ ss_table,         // [6,D]
    const __nv_bfloat16* __restrict__ tproj,    // [B,S,6,D]
    OutT* __restrict__ out,
    float* __restrict__ out_scale,              // [B*S, D/128], BLOCK128_FP8 only
    QuantT quant,                               // identity or fp8 quantizer
    int B, int S, int D, int scale_idx, int shift_idx, float eps, int tp_bcast = 0)
{
    int s = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    long row = ((long)b * S + s) * D;
    long ss_scale = (long)scale_idx * D;
    long ss_shift = (long)shift_idx * D;
    // [B,S,6,D] laid out so the six modulation rows of one token are contiguous
    // in the 6 axis; tp_bcast: shared rows, see tp_row_
    const long tok = tp_row_(tp_bcast, (long)b * S + s);
    long tp_scale = tok * 6 * D + ss_scale;
    long tp_shift = tok * 6 * D + ss_shift;

    extern __shared__ float smem[];

    // Pass 1: fold the residual (if any) and accumulate the mean/variance terms.
    float local_sum = 0.0f;
    float local_sq = 0.0f;
    if (residual != nullptr && x == nullptr && D == 3072 && blockDim.x == 256) {
        // Normalise-only at the Wan width: the residual is read, not changed
        // (the loop below writes back the value it read), so issue the twelve
        // loads together and accumulate in the same order.
        float r[12];
#pragma unroll
        for (int k = 0; k < 12; ++k) r[k] = __bfloat162float(residual[row + tid + k * 256]);
#pragma unroll
        for (int k = 0; k < 12; ++k) {
            local_sum += r[k];
            local_sq += r[k] * r[k];
        }
    } else if (residual != nullptr) {
        for (int d = tid; d < D; d += blockDim.x) {
            // x == nullptr means "no addend": normalise the residual as it stands.
            // Reading the residual as its own addend would double it -- invisible in
            // the modulation output (LayerNorm is scale invariant) but destructive to
            // the caller's running residual.
            float r = __bfloat162float(residual[row + d]);
            if (x != nullptr) {
                float v = __bfloat162float(x[row + d]);
                if (gate_vec != nullptr) v *= gate_vec[row + d];
                r += v;
            }
            residual[row + d] = __float2bfloat16_rn(r);
            local_sum += r;
            local_sq += r * r;
        }
    } else {
        for (int d = tid; d < D; d += blockDim.x) {
            float v = __bfloat162float(x[row + d]);
            local_sum += v;
            local_sq += v * v;
        }
    }
    float sum, sq;
    if (blockDim.x == 256) {
        const float2 r = block_reduce_sum2_256(local_sum, local_sq, smem);
        sum = r.x;
        sq = r.y;
    } else {
        sum = block_reduce_sum(local_sum, smem);
        __syncthreads();
        sq = block_reduce_sum(local_sq, smem);
    }
    float mean = sum / (float)D;
    float var = sq / (float)D - mean * mean;
    float rstd = rsqrtf(var + eps);

    if constexpr (BLOCK128_FP8) {
        adaln_emit_block128_fp8<OutT, QuantT, BLOCK128_FP8>(
            residual, x, ss_table, tproj, out, out_scale, row, (long)b * S + s,
            ss_scale, ss_shift, tp_scale, tp_shift, mean, rstd, D);
    } else {
        // Pass 2: normalize + modulate + bf16 round + quantize.
        for (int d = tid; d < D; d += blockDim.x) {
            float v = (residual != nullptr)
                          ? __bfloat162float(residual[row + d])
                          : __bfloat162float(x[row + d]);
            float g = ss_table[ss_scale + d] + __bfloat162float(tproj[tp_scale + d]);
            float sh = ss_table[ss_shift + d] + __bfloat162float(tproj[tp_shift + d]);
            float norm_v = (v - mean) * rstd;
            float mod_v = (1.0f + g) * norm_v + sh;
            // bf16 round before quantizing: matches the eager `.type_as(hidden_states)`.
            float mod_bf16 = __bfloat162float(__float2bfloat16_rn(mod_v));
            out[row + d] = quant(mod_bf16);
        }
    }
}


struct IdentityQuant {
    __device__ __forceinline__ __nv_bfloat16 operator()(float v) const {
        return __float2bfloat16_rn(v);
    }
};

// The static activation scale lives in device memory (same convention as
// fvk.quantize_fp8_static, whose d_scale is a persistent device scalar), so the
// kernel reads it on device -- it must not be dereferenced on the host.
struct Fp8Quant {
    const float* d_scale;
    __device__ __forceinline__ __nv_fp8_e4m3 operator()(float v) const {
        const float inv_scale = 1.0f / __ldg(d_scale);
        const float q = fminf(fmaxf(v * inv_scale, -448.0f), 448.0f);
        return __nv_fp8_e4m3(q);
    }
};

// ---------------------------------------------------------------------------

void wan_adaln_ln_fp8(
    uintptr_t x, uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t s)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    dim3 grid(S, B);
    adaln_kernel<<<grid, 256, 512 * sizeof(float), stream>>>(
        nullptr,
        reinterpret_cast<const __nv_bfloat16*>(x),
        nullptr,
        reinterpret_cast<const float*>(ss_table),
        reinterpret_cast<const __nv_bfloat16*>(tproj),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8), nullptr,
        Fp8Quant{reinterpret_cast<const float*>(act_scale)}, B, S, D, scale_idx, shift_idx, eps);
}

void wan_adaln_residual_ln_fp8(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
    uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t act_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t s)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    dim3 grid(S, B);
    adaln_kernel<<<grid, 256, 512 * sizeof(float), stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const float*>(gate_vec),
        reinterpret_cast<const float*>(ss_table),
        reinterpret_cast<const __nv_bfloat16*>(tproj),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8), nullptr,
        Fp8Quant{reinterpret_cast<const float*>(act_scale)}, B, S, D, scale_idx, shift_idx, eps);
}

void wan_adaln_residual_ln_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
    uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_bf16,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t s)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    dim3 grid(S, B);
    adaln_kernel<<<grid, 256, 512 * sizeof(float), stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const float*>(gate_vec),
        reinterpret_cast<const float*>(ss_table),
        reinterpret_cast<const __nv_bfloat16*>(tproj),
        reinterpret_cast<__nv_bfloat16*>(out_bf16), nullptr,
        IdentityQuant(), B, S, D, scale_idx, shift_idx, eps);
}

__global__ void gated_residual_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,     // [D] or null
    const float* __restrict__ gate_vec,
    int D, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    // The caller used to run `add_bias_bf16(x, bias, S, D)` over this buffer
    // first. Folding it here reproduces that launch exactly -- the same two
    // operands, one fp32 add, one bf16 round -- before the gate multiply.
    if (bias != nullptr)
        v = __bfloat162float(
            __float2bfloat16_rn(__fadd_rn(v, __bfloat162float(bias[i % D]))));
    if (gate_vec != nullptr) v *= gate_vec[i];
    residual[i] = __float2bfloat16_rn(__bfloat162float(residual[i]) + v);
}

// Same arithmetic, with the AdaLN gate formed inline: gate[d] = ss_row[d] +
// tp[row*D + d], both fp32, exactly the `torch.add(S[:,None,:], tp[None,:,:])`
// this replaces -- one fp32 add of the same two operands, so it is bit-exact.
// The caller used to materialise that sum for all 30 blocks at once into a
// [30, L, D] fp32 table (94 MB) twice per forward, then hand each block its
// slice. Forming it here removes the write, and every block now reads the same
// [L, D] tp instead of a distinct slice.
__global__ void gated_residual_sstp_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,     // [D] or null
    const float* __restrict__ ss_row,   // [D]
    const float* __restrict__ tp,       // [B*S, D], same flat index space as residual
    int D, int tp_bcast)
{
    const long row = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    const long i = row * (long)D + d;
    // The bias fold that used to be a separate add_bias_bf16 launch, with the
    // same two operands, one fp32 add and one bf16 round.
    float xb = __bfloat162float(x[i]);
    if (bias != nullptr)
        xb = __bfloat162float(
            __float2bfloat16_rn(__fadd_rn(xb, __bfloat162float(bias[d]))));
    // __fadd_rn/__fmul_rn are not contractible: plain `x * (s + t)` lets nvcc
    // emit an FMA, which rounds once where the torch add plus the fp32 multiply
    // it replaces rounds twice. That is a 1-ulp difference on ~16 of 786432
    // elements -- enough to move the closed loop.
    const float gate = __fadd_rn(ss_row[d], tp[tp_row_(tp_bcast, row) * (long)D + d]);
    const float v = __fmul_rn(xb, gate);
    residual[i] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(residual[i]), v));
}

void wan_gated_residual_sstp_bias_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t bias_bf16, uintptr_t ss_row,
    uintptr_t tp, int B, int S, int D, uintptr_t s, int tp_bcast)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    dim3 grid((unsigned)((D + 255) / 256), (unsigned)((long)B * S));
    gated_residual_sstp_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
        reinterpret_cast<const float*>(ss_row),
        reinterpret_cast<const float*>(tp), D, tp_bcast);
}

void wan_gated_residual_bias_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t bias_bf16, uintptr_t gate_vec,
    int B, int S, int D, uintptr_t s)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    long n = (long)B * S * D;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    gated_residual_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(bias_bf16),
        reinterpret_cast<const float*>(gate_vec), D, n);
}

void wan_gated_residual_bf16(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
    int B, int S, int D, uintptr_t s)
{
    wan_gated_residual_bias_bf16(residual, x, 0, gate_vec, B, S, D, s);
}

// AdaLN + LayerNorm whose output is written straight out as FP8 with 128-wide
// per-token scales, replacing this kernel followed by
// `fp8_per_token_block128_quant_bf16`. The two are bit-identical: the block
// scale is a max, so its reduction order is free, and the per-element
// arithmetic is the same expression sequence with the same rounding.
void wan_adaln_residual_ln_fp8_block128(
    uintptr_t residual, uintptr_t x, uintptr_t gate_vec,
    uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t out_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t s,
    int tp_bcast)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    dim3 grid(S, B);
    adaln_kernel<__nv_fp8_e4m3, IdentityQuant, true><<<grid, 256, 512 * sizeof(float), stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const float*>(gate_vec),
        reinterpret_cast<const float*>(ss_table),
        reinterpret_cast<const __nv_bfloat16*>(tproj),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<float*>(out_scale),
        IdentityQuant(), B, S, D, scale_idx, shift_idx, eps, tp_bcast);
}


// The block's gated residual (gated_residual_sstp_kernel's exact element
// arithmetic) folded into the next AdaLN's pass 1. The AdaLN kernel above
// normalises the residual it reads back as bf16, so feeding its statistics the
// bf16-rounded sum written here keeps both the residual and the FP8 output
// bit-identical to the two launches; the thread-to-element map of pass 1 and
// the reduction tree are unchanged.
template <int ITERS>
__global__ void sstp_adaln_fp8_block128_kernel(
    __nv_bfloat16* __restrict__ residual,       // [B,S,D]
    const __nv_bfloat16* __restrict__ x,        // [B,S,D]
    const __nv_bfloat16* __restrict__ xbias,    // [D] or null
    const float* __restrict__ gate_ss_row,      // [D]
    const float* __restrict__ gate_tp,          // [B*S, D]
    const float* __restrict__ ss_table,         // [6,D]
    const __nv_bfloat16* __restrict__ tproj,    // [B,S,6,D]
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ out_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, int tp_bcast)
{
    int s = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    long row = ((long)b * S + s) * D;
    long ss_scale = (long)scale_idx * D;
    long ss_shift = (long)shift_idx * D;
    const long tok = tp_row_(tp_bcast, (long)b * S + s);
    long tp_scale = tok * 6 * D + ss_scale;
    long tp_shift = tok * 6 * D + ss_shift;
    const long grow = tok * D;     // gate_tp row

    extern __shared__ float smem[];

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    auto upd = [&](int d) -> float {
        const long i = row + d;
        float xb = __bfloat162float(x[i]);
        if (xbias != nullptr)
            xb = __bfloat162float(
                __float2bfloat16_rn(__fadd_rn(xb, __bfloat162float(xbias[d]))));
        const float gate = __fadd_rn(gate_ss_row[d], gate_tp[grow + d]);
        const float v = __fmul_rn(xb, gate);
        const __nv_bfloat16 rb =
            __float2bfloat16_rn(__fadd_rn(__bfloat162float(residual[i]), v));
        residual[i] = rb;
        return __bfloat162float(rb);
    };
    if (ITERS > 0) {
        // the element values are independent: compute them all first (their
        // loads in flight together), then accumulate in pass-1 order
        float r[ITERS > 0 ? ITERS : 1];
#pragma unroll
        for (int k = 0; k < ITERS; ++k) r[k] = upd(tid + k * 256);
#pragma unroll
        for (int k = 0; k < ITERS; ++k) {
            local_sum += r[k];
            local_sq += r[k] * r[k];
        }
    } else {
        for (int d = tid; d < D; d += blockDim.x) {
            const float r = upd(d);
            local_sum += r;
            local_sq += r * r;
        }
    }
    const float2 red = block_reduce_sum2_256(local_sum, local_sq, smem);
    float mean = red.x / (float)D;
    float var = red.y / (float)D - mean * mean;
    float rstd = rsqrtf(var + eps);

    adaln_emit_block128_fp8<__nv_fp8_e4m3, IdentityQuant, true>(
        residual, nullptr, ss_table, tproj, out, out_scale, row, (long)b * S + s,
        ss_scale, ss_shift, tp_scale, tp_shift, mean, rstd, D);
}

void wan_gated_residual_sstp_adaln_fp8_block128(
    uintptr_t residual, uintptr_t x, uintptr_t xbias, uintptr_t gate_ss_row,
    uintptr_t gate_tp, uintptr_t ss_table, uintptr_t tproj,
    uintptr_t out_fp8, uintptr_t out_scale,
    int B, int S, int D, int scale_idx, int shift_idx, float eps, uintptr_t s,
    int tp_bcast)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s);
    dim3 grid(S, B);
    auto kfn = (D == 3072) ? sstp_adaln_fp8_block128_kernel<12>
                           : sstp_adaln_fp8_block128_kernel<0>;
    kfn<<<grid, 256, 512 * sizeof(float), stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(xbias),
        reinterpret_cast<const float*>(gate_ss_row),
        reinterpret_cast<const float*>(gate_tp),
        reinterpret_cast<const float*>(ss_table),
        reinterpret_cast<const __nv_bfloat16*>(tproj),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<float*>(out_scale), B, S, D, scale_idx, shift_idx, eps, tp_bcast);
}

}  // namespace wan_adaln
}  // namespace flash_rt

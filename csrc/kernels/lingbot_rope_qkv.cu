#include "lingbot_common.cuh"

__global__ void lingbot_rope_inplace_bf16_kernel(
    __nv_bfloat16* __restrict__ x,            // [B, S, NH, HD]
    const float* __restrict__ cos_table,      // [B, S, HD/2] fp32
    const float* __restrict__ sin_table,      // [B, S, HD/2] fp32
    int B, int S, int NH, int HD)
{
    // Block: one (b, s, h). Threads cover HD/2 paired elements.
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int half = HD / 2;

    int x_offset = ((b * S + s) * NH + h) * HD;
    int cs_offset = (b * S + s) * half;

    for (int d = tid; d < half; d += blockDim.x) {
        float c = cos_table[cs_offset + d];
        float sn = sin_table[cs_offset + d];
        float x1 = __bfloat162float(x[x_offset + d]);
        float x2 = __bfloat162float(x[x_offset + d + half]);
        x[x_offset + d]        = __float2bfloat16_rn(x1 * c - x2 * sn);
        x[x_offset + d + half] = __float2bfloat16_rn(x2 * c + x1 * sn);
    }
}

void lingbot_rope_inplace_bf16(
    uintptr_t x, uintptr_t cos_table, uintptr_t sin_table,
    int B, int S, int NH, int HD, uintptr_t stream)
{
    int half = HD / 2;
    int block_threads = (half < 128) ? half : 128;
    dim3 grid(NH, S, B);
    dim3 block(block_threads);
    lingbot_rope_inplace_bf16_kernel<<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<__nv_bfloat16*>(x),
        reinterpret_cast<const float*>(cos_table),
        reinterpret_cast<const float*>(sin_table),
        B, S, NH, HD
    );
}

// G33-fix: out-of-place split-half RoPE. Reads ``x`` (source), writes a
// distinct ``out`` buffer. Out-of-place avoids any read-after-write hazard
// between source and destination across CUDA-graph replays, and lets the
// caller keep ``x`` (e.g. for a residual) — fixes the non-determinism that
// disabled the in-place variant.
__global__ void lingbot_rope_to_out_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,      // [B, S, NH, HD]
    __nv_bfloat16* __restrict__ out,          // [B, S, NH, HD]
    const float* __restrict__ cos_table,      // [B, S, HD/2] fp32
    const float* __restrict__ sin_table,      // [B, S, HD/2] fp32
    int B, int S, int NH, int HD)
{
    int b = blockIdx.z;
    int s = blockIdx.y;
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int half = HD / 2;

    int x_offset = ((b * S + s) * NH + h) * HD;
    int cs_offset = (b * S + s) * half;

    for (int d = tid; d < half; d += blockDim.x) {
        float c = cos_table[cs_offset + d];
        float sn = sin_table[cs_offset + d];
        float x1 = __bfloat162float(x[x_offset + d]);
        float x2 = __bfloat162float(x[x_offset + d + half]);
        out[x_offset + d]        = __float2bfloat16_rn(x1 * c - x2 * sn);
        out[x_offset + d + half] = __float2bfloat16_rn(x2 * c + x1 * sn);
    }
}

void lingbot_rope_to_out_bf16(
    uintptr_t x, uintptr_t out, uintptr_t cos_table, uintptr_t sin_table,
    int B, int S, int NH, int HD, uintptr_t stream)
{
    int half = HD / 2;
    int block_threads = (half < 128) ? half : 128;
    dim3 grid(NH, S, B);
    dim3 block(block_threads);
    lingbot_rope_to_out_bf16_kernel<<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(out),
        reinterpret_cast<const float*>(cos_table),
        reinterpret_cast<const float*>(sin_table),
        B, S, NH, HD
    );
}

__device__ __forceinline__ __nv_bfloat16 to_out_t(float v, __nv_bfloat16) {
    return __float2bfloat16_rn(v);
}
__device__ __forceinline__ __half to_out_t(float v, __half) {
    return __float2half_rn(v);
}

template <typename OutT>
__global__ void lingbot_qkv_bias_rope_kernel(
    const __nv_bfloat16* __restrict__ q_raw,  // [M, NHQ*HD]
    const __nv_bfloat16* __restrict__ k_raw,  // [M, NHKV*HD]
    const __nv_bfloat16* __restrict__ v_raw,  // [M, NHKV*HD]
    const __nv_bfloat16* __restrict__ q_bias, // [NHQ*HD] or null
    const __nv_bfloat16* __restrict__ k_bias, // [NHKV*HD] or null
    const __nv_bfloat16* __restrict__ v_bias, // [NHKV*HD] or null
    const float* __restrict__ cos_table,      // [M, HD/2]
    const float* __restrict__ sin_table,      // [M, HD/2]
    OutT* __restrict__ q_out,                  // [M, NHQ*HD]
    OutT* __restrict__ k_out,                  // [M, NHKV*HD]
    OutT* __restrict__ v_out,                  // [M, NHKV*HD]
    int M, int NHQ, int NHKV, int HD)
{
    int slot = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;
    int half = HD / 2;
    int cs_off = m * half;

    const __nv_bfloat16* src;
    const __nv_bfloat16* bias;
    OutT* dst;
    int head, do_rope;
    if (slot < NHQ) {
        head = slot; src = q_raw; bias = q_bias; dst = q_out; do_rope = 1;
        src += m * NHQ * HD + head * HD;  dst += m * NHQ * HD + head * HD;
    } else if (slot < NHQ + NHKV) {
        head = slot - NHQ; src = k_raw; bias = k_bias; dst = k_out; do_rope = 1;
        src += m * NHKV * HD + head * HD; dst += m * NHKV * HD + head * HD;
    } else {
        head = slot - NHQ - NHKV; src = v_raw; bias = v_bias; dst = v_out; do_rope = 0;
        src += m * NHKV * HD + head * HD; dst += m * NHKV * HD + head * HD;
    }
    int bias_off = head * HD;
    OutT tag{};

    if (do_rope) {
        for (int d = tid; d < half; d += blockDim.x) {
            float c = cos_table[cs_off + d];
            float sn = sin_table[cs_off + d];
            float x1 = __bfloat162float(src[d]);
            float x2 = __bfloat162float(src[d + half]);
            if (bias) {
                x1 += __bfloat162float(bias[bias_off + d]);
                x2 += __bfloat162float(bias[bias_off + d + half]);
            }
            dst[d]        = to_out_t(x1 * c - x2 * sn, tag);
            dst[d + half] = to_out_t(x2 * c + x1 * sn, tag);
        }
    } else {
        for (int d = tid; d < HD; d += blockDim.x) {
            float x = __bfloat162float(src[d]);
            if (bias) x += __bfloat162float(bias[bias_off + d]);
            dst[d] = to_out_t(x, tag);
        }
    }
}

void lingbot_qkv_bias_rope_bf16(
    uintptr_t q_raw, uintptr_t k_raw, uintptr_t v_raw,
    uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
{
    int half = HD / 2;
    int block_threads = (half < 128) ? half : 128;
    dim3 grid(NHQ + 2 * NHKV, M);
    dim3 block(block_threads);
    lingbot_qkv_bias_rope_kernel<__nv_bfloat16><<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw),
        reinterpret_cast<const __nv_bfloat16*>(k_raw),
        reinterpret_cast<const __nv_bfloat16*>(v_raw),
        reinterpret_cast<const __nv_bfloat16*>(q_bias),
        reinterpret_cast<const __nv_bfloat16*>(k_bias),
        reinterpret_cast<const __nv_bfloat16*>(v_bias),
        reinterpret_cast<const float*>(cos_table),
        reinterpret_cast<const float*>(sin_table),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<__nv_bfloat16*>(v_out),
        M, NHQ, NHKV, HD);
}

// G49 merged-input variant: q/k/v come from ONE merged GEMM output
// ``qkv`` [M, ROWQKV] where ROWQKV = NHQ*HD + 2*NHKV*HD and each row is
// [q(NHQ*HD) | k(NHKV*HD) | v(NHKV*HD)]. Reads via column offsets (no split
// copy). Writes the three SEPARATE outputs the attention needs. bias stays
// per-section (the merged GEMM is bias-free).
template <typename OutT>
__global__ void lingbot_qkv_bias_rope_merged_kernel(
    const __nv_bfloat16* __restrict__ qkv,    // [M, ROWQKV]
    const __nv_bfloat16* __restrict__ q_bias, // [NHQ*HD] or null
    const __nv_bfloat16* __restrict__ k_bias, // [NHKV*HD] or null
    const __nv_bfloat16* __restrict__ v_bias, // [NHKV*HD] or null
    const float* __restrict__ cos_table,      // [M, HD/2]
    const float* __restrict__ sin_table,      // [M, HD/2]
    OutT* __restrict__ q_out, OutT* __restrict__ k_out, OutT* __restrict__ v_out,
    int M, int NHQ, int NHKV, int HD)
{
    int slot = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;
    int half = HD / 2;
    int cs_off = m * half;
    int ROWQKV = (NHQ + 2 * NHKV) * HD;
    int row_base = m * ROWQKV;
    const __nv_bfloat16* src;
    const __nv_bfloat16* bias;
    OutT* dst;
    int head, do_rope;
    if (slot < NHQ) {
        head = slot; bias = q_bias; dst = q_out; do_rope = 1;
        src = qkv + row_base + head * HD;
        dst += m * NHQ * HD + head * HD;
    } else if (slot < NHQ + NHKV) {
        head = slot - NHQ; bias = k_bias; dst = k_out; do_rope = 1;
        src = qkv + row_base + NHQ * HD + head * HD;
        dst += m * NHKV * HD + head * HD;
    } else {
        head = slot - NHQ - NHKV; bias = v_bias; dst = v_out; do_rope = 0;
        src = qkv + row_base + (NHQ + NHKV) * HD + head * HD;
        dst += m * NHKV * HD + head * HD;
    }
    int bias_off = head * HD;
    OutT tag{};
    if (do_rope) {
        for (int d = tid; d < half; d += blockDim.x) {
            float c = cos_table[cs_off + d];
            float sn = sin_table[cs_off + d];
            float x1 = __bfloat162float(src[d]);
            float x2 = __bfloat162float(src[d + half]);
            if (bias) {
                x1 += __bfloat162float(bias[bias_off + d]);
                x2 += __bfloat162float(bias[bias_off + d + half]);
            }
            dst[d]        = to_out_t(x1 * c - x2 * sn, tag);
            dst[d + half] = to_out_t(x2 * c + x1 * sn, tag);
        }
    } else {
        for (int d = tid; d < HD; d += blockDim.x) {
            float x = __bfloat162float(src[d]);
            if (bias) x += __bfloat162float(bias[bias_off + d]);
            dst[d] = to_out_t(x, tag);
        }
    }
}

void lingbot_qkv_bias_rope_merged_fp16out(
    uintptr_t qkv, uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
{
    int half = HD / 2;
    int block_threads = (half < 128) ? half : 128;
    dim3 grid(NHQ + 2 * NHKV, M);
    dim3 block(block_threads);
    lingbot_qkv_bias_rope_merged_kernel<__half><<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(q_bias),
        reinterpret_cast<const __nv_bfloat16*>(k_bias),
        reinterpret_cast<const __nv_bfloat16*>(v_bias),
        reinterpret_cast<const float*>(cos_table),
        reinterpret_cast<const float*>(sin_table),
        reinterpret_cast<__half*>(q_out),
        reinterpret_cast<__half*>(k_out),
        reinterpret_cast<__half*>(v_out),
        M, NHQ, NHKV, HD);
}

void lingbot_qkv_bias_rope_merged_bf16(
    uintptr_t qkv, uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
{
    int half = HD / 2;
    int block_threads = (half < 128) ? half : 128;
    dim3 grid(NHQ + 2 * NHKV, M);
    dim3 block(block_threads);
    lingbot_qkv_bias_rope_merged_kernel<__nv_bfloat16><<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(q_bias),
        reinterpret_cast<const __nv_bfloat16*>(k_bias),
        reinterpret_cast<const __nv_bfloat16*>(v_bias),
        reinterpret_cast<const float*>(cos_table),
        reinterpret_cast<const float*>(sin_table),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<__nv_bfloat16*>(v_out),
        M, NHQ, NHKV, HD);
}

// G39: fp16-output variant — feeds the fp16 attention island (fmha consumes
// fp16 with no cast). Reads the same bf16 GEMM outputs, writes __half.
void lingbot_qkv_bias_rope_fp16out(
    uintptr_t q_raw, uintptr_t k_raw, uintptr_t v_raw,
    uintptr_t q_bias, uintptr_t k_bias, uintptr_t v_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NHQ, int NHKV, int HD, uintptr_t stream)
{
    int half = HD / 2;
    int block_threads = (half < 128) ? half : 128;
    dim3 grid(NHQ + 2 * NHKV, M);
    dim3 block(block_threads);
    lingbot_qkv_bias_rope_kernel<__half><<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw),
        reinterpret_cast<const __nv_bfloat16*>(k_raw),
        reinterpret_cast<const __nv_bfloat16*>(v_raw),
        reinterpret_cast<const __nv_bfloat16*>(q_bias),
        reinterpret_cast<const __nv_bfloat16*>(k_bias),
        reinterpret_cast<const __nv_bfloat16*>(v_bias),
        reinterpret_cast<const float*>(cos_table),
        reinterpret_cast<const float*>(sin_table),
        reinterpret_cast<__half*>(q_out),
        reinterpret_cast<__half*>(k_out),
        reinterpret_cast<__half*>(v_out),
        M, NHQ, NHKV, HD);
}


// =============================================================================
//  Kernel: ViT fused QKV bias-add + 2-D M-RoPE (per-view ViT attention prep)
// =============================================================================
// The Qwen2.5-VL ViT qkv GEMM output is ONE interleaved buffer
// [M, 3*NH*HD] (type-major: type 0=q, 1=k, 2=v; NH heads each, head_dim HD).
// q/k get (bias add → split-half RoPE); v gets bias only. Writes three
// contiguous [M, NH*HD] outputs ready for the per-view fmha. Replaces, per
// ViT block, the add_bias(qkv) launch + the eager fp32 RoPE storm
// (q.float/k.float, rotate_half cat ×2, mul, add, cast back). The RoPE
// formula is identical to the LLM split-half kernel: with emb=cat([f,f]) the
// two cos/sin halves are equal, so we index the first HD/2 columns of the
// [M, HD] cos/sin tables (row stride HD).
__global__ void lingbot_vit_qkv_bias_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv,      // [M, 3*NH*HD] raw GEMM out
    const __nv_bfloat16* __restrict__ qkv_bias, // [3*NH*HD] or null
    const float* __restrict__ cos_table,        // [M, HD] (uses cols [0,HD/2))
    const float* __restrict__ sin_table,        // [M, HD]
    __nv_bfloat16* __restrict__ q_out,          // [M, NH*HD]
    __nv_bfloat16* __restrict__ k_out,          // [M, NH*HD]
    __nv_bfloat16* __restrict__ v_out,          // [M, NH*HD]
    int M, int NH, int HD)
{
    int slot = blockIdx.x;                      // [0, 3*NH)
    int m = blockIdx.y;
    int tid = threadIdx.x;
    int half = HD / 2;
    int type = slot / NH;                       // 0=q, 1=k, 2=v
    int head = slot - type * NH;
    const __nv_bfloat16* src =
        qkv + ((long)m * 3 * NH + (long)type * NH + head) * HD;
    __nv_bfloat16* dst =
        (type == 0 ? q_out : (type == 1 ? k_out : v_out))
        + ((long)m * NH + head) * HD;
    int bias_off = (type * NH + head) * HD;
    int cs_off = m * HD;
    if (type < 2) {                             // q, k: bias + split-half RoPE
        for (int d = tid; d < half; d += blockDim.x) {
            float c = cos_table[cs_off + d];
            float sn = sin_table[cs_off + d];
            float x1 = __bfloat162float(src[d]);
            float x2 = __bfloat162float(src[d + half]);
            if (qkv_bias) {
                x1 += __bfloat162float(qkv_bias[bias_off + d]);
                x2 += __bfloat162float(qkv_bias[bias_off + d + half]);
            }
            dst[d]        = __float2bfloat16_rn(x1 * c - x2 * sn);
            dst[d + half] = __float2bfloat16_rn(x2 * c + x1 * sn);
        }
    } else {                                    // v: bias only
        for (int d = tid; d < HD; d += blockDim.x) {
            float x = __bfloat162float(src[d]);
            if (qkv_bias) x += __bfloat162float(qkv_bias[bias_off + d]);
            dst[d] = __float2bfloat16_rn(x);
        }
    }
}

void lingbot_vit_qkv_bias_rope_bf16(
    uintptr_t qkv, uintptr_t qkv_bias,
    uintptr_t cos_table, uintptr_t sin_table,
    uintptr_t q_out, uintptr_t k_out, uintptr_t v_out,
    int M, int NH, int HD, uintptr_t stream)
{
    int half = HD / 2;
    int block_threads = (half < 128) ? half : 128;
    dim3 grid(3 * NH, M);
    dim3 block(block_threads);
    lingbot_vit_qkv_bias_rope_kernel<<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(qkv_bias),
        reinterpret_cast<const float*>(cos_table),
        reinterpret_cast<const float*>(sin_table),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<__nv_bfloat16*>(v_out),
        M, NH, HD);
}


// =============================================================================
//  Kernel: fused SwiGLU tail — silu(gate)*up + FP8 static-quant + M-pad
// =============================================================================
//   For the Expert down_proj input. Replaces the eager chain
//   ``F.silu(gate) * up`` (bf16 elementwise, 2 kernels) + ``quantize_fp8``
//   + the linear_fp8 M-pad copy (51->64) with ONE pass that writes a
//   pre-M-padded FP8 buffer the down GEMM reads directly (M=PADM, no copy).
//   Rows [S, PADM) are zero-filled so the sliced-off GEMM output is clean.
//
//   Math matches eager: silu/mul computed in fp32 with a bf16 round on the
//   product (the eager F.silu*up is bf16) before the FP8 quant — same
//   rounding chain as linear_fp8's internal quantize.
//
// Grid:  (PADM,)   one block per output row
// Block: 256 threads, strided over I (intermediate dim)

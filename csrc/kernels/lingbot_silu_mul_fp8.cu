#include "lingbot_common.cuh"

__global__ void lingbot_silu_mul_fp8_mpad_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,  // [S, I]
    const __nv_bfloat16* __restrict__ up,     // [S, I]
    __nv_fp8_e4m3* __restrict__ out_fp8,      // [PADM, I]
    const float* __restrict__ act_scale,      // [1]
    int S, int PADM, int I)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int off = row * I;
    if (row >= S) {
        for (int d = tid; d < I; d += blockDim.x)
            out_fp8[off + d] = __nv_fp8_e4m3(0.0f);
        return;
    }
    float inv_scale = 1.0f / __ldg(act_scale);
    for (int d = tid; d < I; d += blockDim.x) {
        float gv = __bfloat162float(gate[off + d]);
        float uv = __bfloat162float(up[off + d]);
        float silu = gv / (1.0f + __expf(-gv));
        float h = silu * uv;
        // round product through bf16 to match the eager bf16 silu*up tensor
        float h_bf16 = __bfloat162float(__float2bfloat16_rn(h));
        float fp8_v = h_bf16 * inv_scale;
        fp8_v = fmaxf(-448.0f, fminf(448.0f, fp8_v));
        out_fp8[off + d] = __nv_fp8_e4m3(fp8_v);
    }
}

void lingbot_silu_mul_fp8_mpad_bf16(
    uintptr_t gate, uintptr_t up, uintptr_t out_fp8, uintptr_t act_scale,
    int S, int PADM, int I, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    dim3 grid(PADM);
    dim3 block(BLOCK_THREADS);
    lingbot_silu_mul_fp8_mpad_bf16_kernel<<<grid, block, 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        S, PADM, I);
}

// Merged-input variant: gate/up come INTERLEAVED per row in one [S, 2*I] buffer
// (a single merged gate_up GEMM output): row r = [gate(I) | up(I)]. Reads with
// offsets (no split copy) so one GEMM + this kernel replace 2 GEMMs + silu/mul.
__global__ void lingbot_silu_mul_merged_fp8_mpad_bf16_kernel(
    const __nv_bfloat16* __restrict__ gu,    // [S, 2*I] (gate|up per row)
    __nv_fp8_e4m3* __restrict__ out_fp8,      // [PADM, I]
    const float* __restrict__ act_scale,
    int S, int PADM, int I)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int oo = row * I;
    if (row >= S) {
        for (int d = tid; d < I; d += blockDim.x)
            out_fp8[oo + d] = __nv_fp8_e4m3(0.0f);
        return;
    }
    int go = row * 2 * I;        // gate base
    int uo = go + I;             // up base
    float inv_scale = 1.0f / __ldg(act_scale);
    for (int d = tid; d < I; d += blockDim.x) {
        float gv = __bfloat162float(gu[go + d]);
        float uv = __bfloat162float(gu[uo + d]);
        float silu = gv / (1.0f + __expf(-gv));
        float h = silu * uv;
        float h_bf16 = __bfloat162float(__float2bfloat16_rn(h));
        float fp8_v = h_bf16 * inv_scale;
        fp8_v = fmaxf(-448.0f, fminf(448.0f, fp8_v));
        out_fp8[oo + d] = __nv_fp8_e4m3(fp8_v);
    }
}

void lingbot_silu_mul_merged_fp8_mpad_bf16(
    uintptr_t gu, uintptr_t out_fp8, uintptr_t act_scale,
    int S, int PADM, int I, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    lingbot_silu_mul_merged_fp8_mpad_bf16_kernel<<<dim3(PADM), dim3(BLOCK_THREADS), 0, to_stream(stream)>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        S, PADM, I);
}

// G54: fp16-INPUT variant — reads the merged gate_up directly from the FP4 GEMM's
// fp16 output (no bf16 cast). Same silu(gate)*up + FP8 static-quant + M-pad.
__global__ void lingbot_silu_mul_merged_fp8_mpad_fp16in_kernel(
    const __half* __restrict__ gu,            // [S, 2*I] fp16 (gate|up per row)
    __nv_fp8_e4m3* __restrict__ out_fp8,      // [PADM, I]
    const float* __restrict__ act_scale,
    int S, int PADM, int I)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int oo = row * I;
    if (row >= S) {
        for (int d = tid; d < I; d += blockDim.x)
            out_fp8[oo + d] = __nv_fp8_e4m3(0.0f);
        return;
    }
    int go = row * 2 * I;
    int uo = go + I;
    float inv_scale = 1.0f / __ldg(act_scale);
    for (int d = tid; d < I; d += blockDim.x) {
        float gv = __half2float(gu[go + d]);
        float uv = __half2float(gu[uo + d]);
        float silu = gv / (1.0f + __expf(-gv));
        float h = silu * uv;
        float h_bf16 = __bfloat162float(__float2bfloat16_rn(h));
        float fp8_v = h_bf16 * inv_scale;
        fp8_v = fmaxf(-448.0f, fminf(448.0f, fp8_v));
        out_fp8[oo + d] = __nv_fp8_e4m3(fp8_v);
    }
}

void lingbot_silu_mul_merged_fp8_mpad_fp16in(
    uintptr_t gu, uintptr_t out_fp8, uintptr_t act_scale,
    int S, int PADM, int I, uintptr_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    lingbot_silu_mul_merged_fp8_mpad_fp16in_kernel<<<dim3(PADM), dim3(BLOCK_THREADS), 0, to_stream(stream)>>>(
        reinterpret_cast<const __half*>(gu),
        reinterpret_cast<__nv_fp8_e4m3*>(out_fp8),
        reinterpret_cast<const float*>(act_scale),
        S, PADM, I);
}

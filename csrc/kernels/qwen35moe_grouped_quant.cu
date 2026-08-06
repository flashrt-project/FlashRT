// Grouped NVFP4 activation quantisers for the qwen3_5_moe MoE path.
// See qwen35moe_grouped_quant.cuh for the tier this is built under.

#include "qwen35moe_grouped_quant.cuh"

#include "nvfp4_convert.cuh"

#include <cstdint>

// ── Grouped activation quantiser for the MoE grouped GEMM ──
//
// Same math as quantize_bf16_to_nvfp4_swizzled_kernel, block for block; what
// differs is where the scale factors land. The block-scaled GEMM wants each
// group's scales in the Sm1xx atom layout for that group's own row count, and
// that layout blocks rows by 128, so a group beginning at an arbitrary row of a
// jointly-quantised matrix has no contiguous sub-block to point at. Quantising
// per group is correct but costs a launch and a host iteration per expert.
//
// Here a row reads the expert it was sorted by, subtracts its group's first
// row, and indexes its group's own block. Nothing reaches the host, which is
// what lets the surrounding prefill chunk be captured.
__global__ void moe_grouped_quant_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ expert_of_row,
    const int* __restrict__ group_off,
    const int* __restrict__ sfa_off,
    const long* __restrict__ src_row,
    uint8_t* __restrict__ fp4_data,
    uint8_t* __restrict__ scale_factors,
    int cols, int num_blocks, int n_col_blocks)
{
    const int row = blockIdx.x;
    const int e = expert_of_row[row];
    const int local = row - group_off[e];          // row index inside its group
    // Gather while quantising when a permutation is given. Materialising the
    // sorted activation first is a full read and a full write of an (S, HID)
    // matrix per layer -- 14.5 ms of a 2048-token prefill -- for rows this
    // kernel is about to read once anyway.
    const size_t in_row = (src_row == nullptr) ? (size_t)row
                                               : (size_t)src_row[row];
    const __nv_bfloat16* row_in = input + in_row * cols;
    uint8_t* row_fp4 = fp4_data + (size_t)row * cols / 2;
    uint8_t* sf_base = scale_factors + sfa_off[e];

    extern __shared__ float smem[];
    const int tid = threadIdx.x;

    // Per-16-block amax without atomics. One thread takes eight bf16 (a half
    // block), reduces them in registers, and pairs with its neighbour through a
    // shuffle -- j and j^1 land on lanes t and t^1 because the block size is
    // even. The first version of this kernel used one atomicMax per element and
    // ran at 58.3 ms for traffic worth 0.8; the atomics were all of it.
    const int vec8 = cols >> 3;
    for (int j = tid; j < vec8; j += blockDim.x) {
        uint4 v = *reinterpret_cast<const uint4*>(&row_in[j << 3]);
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&v);
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) a = fmaxf(a, fabsf(__bfloat162float(bf[i])));
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
        if ((j & 1) == 0) smem[j >> 1] = a;
    }
    __syncthreads();

    const int rb = local / 128;
    const int ri = local % 128;
    for (int b = tid; b < num_blocks; b += blockDim.x) {
        uint8_t ue_scale = float_to_ue4m3_ceil(smem[b] * (1.0f / 6.0f));
        const int cb = b / 4;
        const int ci = b % 4;
        sf_base[(rb * n_col_blocks + cb) * 512 + (ri % 32) * 16
                + (ri / 32) * 4 + ci] = ue_scale;
        smem[b] = ue4m3_to_float(ue_scale);
    }
    __syncthreads();

    // Pack four bytes at a time: eight bf16 in, one uint32 out, and the eight
    // share a 16-block so the scale is read once.
    const int quads = cols >> 3;
    for (int j = tid; j < quads; j += blockDim.x) {
        uint4 v = *reinterpret_cast<const uint4*>(&row_in[j << 3]);
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&v);
        const float scale = smem[j >> 1];
        const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        uint32_t packed = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t lo = float_to_fp4_e2m1(__bfloat162float(bf[2 * k]) * inv);
            uint32_t hi = float_to_fp4_e2m1(
                __bfloat162float(bf[2 * k + 1]) * inv);
            packed |= ((hi << 4) | (lo & 0xF)) << (k * 8);
        }
        *reinterpret_cast<uint32_t*>(row_fp4 + (j << 2)) = packed;
    }
}

int moe_grouped_quant_nvfp4_bf16(
    const void* A, const void* expert_of_row, const void* group_off,
    const void* sfa_off, const void* src_row, void* out_packed, void* out_sf,
    int slots, int K, cudaStream_t stream)
{
    if (!A || !expert_of_row || !group_off || !sfa_off || !out_packed
        || !out_sf) return 1;
    if (slots <= 0 || K <= 0 || (K & 15) != 0) return 2;
    const int num_blocks = K / 16;
    const int n_col_blocks = (num_blocks + 3) / 4;
    const int threads = 256;
    const size_t smem = (size_t)num_blocks * sizeof(float);
    moe_grouped_quant_nvfp4_kernel<<<slots, threads, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const int*>(expert_of_row),
        reinterpret_cast<const int*>(group_off),
        reinterpret_cast<const int*>(sfa_off),
        reinterpret_cast<const long*>(src_row),
        reinterpret_cast<uint8_t*>(out_packed),
        reinterpret_cast<uint8_t*>(out_sf),
        K, num_blocks, n_col_blocks);
    return 0;
}

// ── Gate and quantise in one pass, for the grouped MoE's down projection ──
//
// The grouped GEMM produces gate and up interleaved in one (slots, 2*inter)
// buffer, and the gate op wants them as two matrices. Slicing columns out of it
// is not free: the halves are strided, so `.contiguous()` copies both -- 67 MB
// a layer at 2048 tokens, to feed an op that then writes another 17 and has it
// read straight back by the quantiser.
//
// Reading the merged buffer directly costs none of that. The silu is computed
// and rounded to bf16 exactly as silu_mul_sm120_bf16 does, so the value that
// reaches the quantiser is the same one it saw before.
__global__ void moe_grouped_silu_quant_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ merged,     // (slots, 2 * inter)
    const int* __restrict__ expert_of_row,
    const int* __restrict__ group_off,
    const int* __restrict__ sfa_off,
    uint8_t* __restrict__ fp4_data,
    uint8_t* __restrict__ scale_factors,
    int inter, int num_blocks, int n_col_blocks)
{
    const int row = blockIdx.x;
    const int e = expert_of_row[row];
    const int local = row - group_off[e];
    const __nv_bfloat16* g_in = merged + (size_t)row * 2 * inter;
    const __nv_bfloat16* u_in = g_in + inter;
    uint8_t* row_fp4 = fp4_data + (size_t)row * inter / 2;
    uint8_t* sf_base = scale_factors + sfa_off[e];

    extern __shared__ float smem[];               // inter gated values, then scales
    float* gated = smem;
    float* scales = smem + inter;

    const int tid = threadIdx.x;
    for (int i = tid; i < inter; i += blockDim.x) {
        const float gv = __bfloat162float(g_in[i]);
        const float uv = __bfloat162float(u_in[i]);
        // Rounded to bf16 here, as the separate gate kernel does, so the
        // quantiser downstream sees the identical value.
        gated[i] = __bfloat162float(
            __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv));
    }
    __syncthreads();

    for (int b = tid; b < num_blocks; b += blockDim.x) {
        float a = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < 16; ++j) a = fmaxf(a, fabsf(gated[b * 16 + j]));
        const uint8_t ue = float_to_ue4m3_ceil(a * (1.0f / 6.0f));
        const int rb = local / 128, ri = local % 128;
        sf_base[(rb * n_col_blocks + (b >> 2)) * 512 + (ri % 32) * 16
                + (ri / 32) * 4 + (b & 3)] = ue;
        scales[b] = ue4m3_to_float(ue);
    }
    __syncthreads();

    const int half = inter >> 1;
    for (int p = tid; p < half; p += blockDim.x) {
        const int i = p * 2;
        const float s = scales[i >> 4];
        const float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
        row_fp4[p] = (uint8_t)((float_to_fp4_e2m1(gated[i + 1] * inv) << 4)
                               | (float_to_fp4_e2m1(gated[i] * inv) & 0x0F));
    }
}

// Warp-per-row form of the same thing.
//
// The block-per-row kernel above gives 256 threads a row of 512 values -- two
// elements each -- behind three barriers and three passes over shared memory,
// so a block reads two kilobytes and then waits. Measured 2.9x off what that
// traffic implies.
//
// Here a warp owns a row and a lane owns one 16-element scale-factor group:
// it reads its own sixteen gate and up values as vectors, gates them, takes
// its own maximum and packs its own eight bytes. Nothing is shared, so there
// are no barriers and no shared memory at all, and each lane has sixteen
// values in flight instead of two.
//
// The arithmetic is the same in the same order, so the output is identical.
__global__ void moe_grouped_silu_quant_nvfp4_warp_kernel(
    const __nv_bfloat16* __restrict__ merged,
    const int* __restrict__ expert_of_row,
    const int* __restrict__ group_off,
    const int* __restrict__ sfa_off,
    uint8_t* __restrict__ fp4_data,
    uint8_t* __restrict__ scale_factors,
    int slots, int inter, int num_blocks, int n_col_blocks)
{
    const int warp_in_blk = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp_in_blk;
    if (row >= slots) return;

    const int e = expert_of_row[row];
    const int local = row - group_off[e];
    const __nv_bfloat16* g_in = merged + (size_t)row * 2 * inter;
    const __nv_bfloat16* u_in = g_in + inter;
    uint8_t* row_fp4 = fp4_data + (size_t)row * inter / 2;
    uint8_t* sf_base = scale_factors + sfa_off[e];
    const int rb = local / 128, ri = local % 128;

    for (int b = lane; b < num_blocks; b += 32) {
        float gated[16];
        const int base = b * 16;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float gv = __bfloat162float(g_in[base + j]);
            const float uv = __bfloat162float(u_in[base + j]);
            gated[j] = __bfloat162float(
                __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv));
        }
        float a = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) a = fmaxf(a, fabsf(gated[j]));

        const uint8_t ue = float_to_ue4m3_ceil(a * (1.0f / 6.0f));
        sf_base[(rb * n_col_blocks + (b >> 2)) * 512 + (ri % 32) * 16
                + (ri / 32) * 4 + (b & 3)] = ue;

        const float sc = ue4m3_to_float(ue);
        const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
        uint8_t* out8 = row_fp4 + (size_t)b * 8;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            out8[p] = (uint8_t)((float_to_fp4_e2m1(gated[2 * p + 1] * inv) << 4)
                                | (float_to_fp4_e2m1(gated[2 * p] * inv) & 0x0F));
        }
    }
}

int moe_grouped_silu_quant_nvfp4_warp_bf16(
    const void* merged, const void* expert_of_row, const void* group_off,
    const void* sfa_off, void* out_packed, void* out_sf,
    int slots, int inter, cudaStream_t stream)
{
    if (!merged || !expert_of_row || !group_off || !sfa_off || !out_packed
        || !out_sf) return 1;
    if (slots <= 0 || inter <= 0 || (inter & 15) != 0) return 2;
    const int num_blocks = inter / 16;
    const int n_col_blocks = (num_blocks + 3) / 4;
    constexpr int kThreads = 256;
    const int rows_per_block = kThreads / 32;
    const int grid = (slots + rows_per_block - 1) / rows_per_block;
    moe_grouped_silu_quant_nvfp4_warp_kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(merged),
        reinterpret_cast<const int*>(expert_of_row),
        reinterpret_cast<const int*>(group_off),
        reinterpret_cast<const int*>(sfa_off),
        reinterpret_cast<uint8_t*>(out_packed),
        reinterpret_cast<uint8_t*>(out_sf),
        slots, inter, num_blocks, n_col_blocks);
    return 0;
}

int moe_grouped_silu_quant_nvfp4_bf16(
    const void* merged, const void* expert_of_row, const void* group_off,
    const void* sfa_off, void* out_packed, void* out_sf,
    int slots, int inter, cudaStream_t stream)
{
    if (!merged || !expert_of_row || !group_off || !sfa_off || !out_packed
        || !out_sf) return 1;
    if (slots <= 0 || inter <= 0 || (inter & 15) != 0) return 2;
    const int num_blocks = inter / 16;
    const int n_col_blocks = (num_blocks + 3) / 4;
    const size_t smem = ((size_t)inter + num_blocks) * sizeof(float);
    moe_grouped_silu_quant_nvfp4_kernel<<<slots, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(merged),
        reinterpret_cast<const int*>(expert_of_row),
        reinterpret_cast<const int*>(group_off),
        reinterpret_cast<const int*>(sfa_off),
        reinterpret_cast<uint8_t*>(out_packed),
        reinterpret_cast<uint8_t*>(out_sf),
        inter, num_blocks, n_col_blocks);
    return 0;
}

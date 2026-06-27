// ================================================================
// FlashRT — Fused CFG combine kernel (BF16). See omnivoice_cfg_combine.cuh.
// ================================================================

#include "omnivoice_cfg_combine.cuh"

#define CC_WARP 32
#define CC_ITERS 48   // 48*32 = 1536 cols max (OmniVoice vocab = 1025)

__global__ void omnivoice_cfg_logsoftmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ c_logits,
    const __nv_bfloat16* __restrict__ u_logits,
    __nv_bfloat16* __restrict__ out,
    int cols, int mask_col, float gs)
{
    const int row = blockIdx.x;
    const int lane = threadIdx.x;

    const __nv_bfloat16* cr = c_logits + row * cols;
    const __nv_bfloat16* ur = u_logits + row * cols;
    __nv_bfloat16* outr = out + row * cols;

    float creg[CC_ITERS], ureg[CC_ITERS];

    // ── Pass 1: load c (mask_col → -inf) + u, find max_c, max_u ──
    float mx_c = -1e30f, mx_u = -1e30f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        int col = it * CC_WARP + lane;
        float cv = (col < cols) ? __bfloat162float(cr[col]) : -1e30f;
        float uv = (col < cols) ? __bfloat162float(ur[col]) : -1e30f;
        if (col == mask_col) cv = -1e30f;   // mask only c (matches PyTorch ref)
        creg[it] = cv; ureg[it] = uv;
        mx_c = fmaxf(mx_c, cv);
        mx_u = fmaxf(mx_u, uv);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx_c = fmaxf(mx_c, __shfl_xor_sync(0xffffffff, mx_c, o));
        mx_u = fmaxf(mx_u, __shfl_xor_sync(0xffffffff, mx_u, o));
    }

    // ── Pass 2: sum exp → lc, lu (log normalizers) ──
    float sc = 0.f, su = 0.f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        sc += __expf(creg[it] - mx_c);
        su += __expf(ureg[it] - mx_u);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sc += __shfl_xor_sync(0xffffffff, sc, o);
        su += __shfl_xor_sync(0xffffffff, su, o);
    }
    const float lc = logf(sc + 1e-12f);
    const float lu = logf(su + 1e-12f);

    // ── Pass 3: combined = c_lp + gs*(c_lp - u_lp), find max_comb ──
    // c_lp = creg - mx_c - lc ; u_lp = ureg - mx_u - lu
    float mx_comb = -1e30f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        float clp = creg[it] - mx_c - lc;
        float ulp = ureg[it] - mx_u - lu;
        float comb = clp + gs * (clp - ulp);
        creg[it] = comb;             // reuse creg[] to store combined (saves a 3rd array)
        mx_comb = fmaxf(mx_comb, comb);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx_comb = fmaxf(mx_comb, __shfl_xor_sync(0xffffffff, mx_comb, o));

    // ── Pass 4: out = log_softmax(combined) ──
    float s_comb = 0.f;
    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it)
        s_comb += __expf(creg[it] - mx_comb);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        s_comb += __shfl_xor_sync(0xffffffff, s_comb, o);
    const float lcomb = logf(s_comb + 1e-12f);

    #pragma unroll
    for (int it = 0; it < CC_ITERS; ++it) {
        int col = it * CC_WARP + lane;
        if (col < cols)
            outr[col] = __float2bfloat16(creg[it] - mx_comb - lcomb);
    }
}

void omnivoice_cfg_logsoftmax_bf16(
    const __nv_bfloat16* c_logits,
    const __nv_bfloat16* u_logits,
    __nv_bfloat16* out,
    int rows, int cols, int mask_col, float guidance_scale,
    cudaStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;
    dim3 grid(rows), block(CC_WARP);
    omnivoice_cfg_logsoftmax_bf16_kernel<<<grid, block, 0, stream>>>(
        c_logits, u_logits, out, cols, mask_col, guidance_scale);
}

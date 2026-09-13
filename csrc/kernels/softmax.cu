// ================================================================
// FlashRT — Softmax kernel (FP16)
// Direct port of pi05 softmax_bf16_kernel.
// 1 warp per row, vectorized __half2 load/store.
// ================================================================

#include "softmax.cuh"
#include "fused_fp4/pdl.cuh"

#define SM_WARP_SIZE 32
#define SM_MAX_COLS 1024
#define SM_ITERS (SM_MAX_COLS / SM_WARP_SIZE)  // 32

__global__ void softmax_fp16_kernel(__half* data, int rows, int cols) {
    flashrt_pdl_wait_and_trigger();
    int lane = threadIdx.x % SM_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    __half* src = data + row * cols;
    int cols2 = cols / 2;
    __half2* src2 = reinterpret_cast<__half2*>(src);

    // Pass 1: load + find max
    float reg[SM_ITERS];
    float mx = -1e30f;

    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __half2 v2 = src2[c2];
            reg[it*2] = __half2float(v2.x);
            reg[it*2+1] = __half2float(v2.y);
            mx = fmaxf(mx, fmaxf(reg[it*2], reg[it*2+1]));
        } else {
            reg[it*2] = -1e30f;
            reg[it*2+1] = -1e30f;
        }
    }
    if ((cols & 1) && lane == 0) {
        float v = __half2float(src[cols-1]);
        reg[SM_ITERS-1] = v;
        mx = fmaxf(mx, v);
    }

    // Warp reduce max
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    // Pass 2: exp + sum
    float sm = 0;
    #pragma unroll
    for (int it = 0; it < SM_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    // Pass 3: normalize + store
    float inv = 1.f / (sm + 1e-8f);
    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __half2 v2;
            v2.x = __float2half(reg[it*2] * inv);
            v2.y = __float2half(reg[it*2+1] * inv);
            src2[c2] = v2;
        }
    }
    if ((cols & 1) && lane == 0) {
        src[cols-1] = __float2half(reg[SM_ITERS-1] * inv);
    }
}

void softmax_fp16(__half* data, int rows, int cols, cudaStream_t stream) {
    // 1 warp per row, 1 row per block (matching pi05)
    flash_rt::fp4::launch_maybe_pdl(softmax_fp16_kernel, dim3(rows), dim3(SM_WARP_SIZE), 0, stream, data, rows, cols);
}

// State-masked softmax with pad handling:
// - Rows [0, mask_rows): mask cols [mask_start, cols) as -inf (state + pad)
// - Rows [mask_rows, rows): mask cols [pad_start, cols) as -inf (pad only)
// pad_start is typically S_kv (the actual key count before padding).
__global__ void softmax_state_masked_fp16_kernel(__half* data, int rows, int cols,
                                                  int mask_rows, int mask_start,
                                                  int pad_start) {
    int lane = threadIdx.x % SM_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    int row_mask_start = (row < mask_rows) ? mask_start : pad_start;
    __half* src = data + row * cols;
    int cols2 = cols / 2;
    __half2* src2 = reinterpret_cast<__half2*>(src);

    // Pass 1: load + find max (with masking)
    float reg[SM_ITERS];
    float mx = -1e30f;

    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            int c_base = c2 * 2;
            __half2 v2 = src2[c2];
            float v0 = __half2float(v2.x);
            float v1 = __half2float(v2.y);
            // Apply mask: columns >= row_mask_start set to -inf
            if (c_base >= row_mask_start) v0 = -1e30f;
            if (c_base + 1 >= row_mask_start) v1 = -1e30f;
            reg[it*2] = v0;
            reg[it*2+1] = v1;
            mx = fmaxf(mx, fmaxf(v0, v1));
        } else {
            reg[it*2] = -1e30f;
            reg[it*2+1] = -1e30f;
        }
    }
    if ((cols & 1) && lane == 0) {
        int c = cols - 1;
        float v = __half2float(src[c]);
        if (c >= row_mask_start) v = -1e30f;
        reg[SM_ITERS-1] = v;
        mx = fmaxf(mx, v);
    }

    // Warp reduce max
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    // Pass 2: exp + sum
    float sm = 0;
    #pragma unroll
    for (int it = 0; it < SM_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    // Pass 3: normalize + store
    float inv = 1.f / (sm + 1e-8f);
    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __half2 v2;
            v2.x = __float2half(reg[it*2] * inv);
            v2.y = __float2half(reg[it*2+1] * inv);
            src2[c2] = v2;
        }
    }
    if ((cols & 1) && lane == 0) {
        src[cols-1] = __float2half(reg[SM_ITERS-1] * inv);
    }
}

void softmax_state_masked_fp16(__half* data, int rows, int cols,
                                int mask_rows, int mask_start, int pad_start,
                                cudaStream_t stream) {
    softmax_state_masked_fp16_kernel<<<rows, SM_WARP_SIZE, 0, stream>>>(
        data, rows, cols, mask_rows, mask_start, pad_start);
}


// Causal softmax — strict upper-triangular masking per-head.
// Layout: (NH * S_q, cols) row-major. For row r, head-local Q index
// q = r % S_q; mask cols j > q AND j >= pad_start.
__global__ void softmax_causal_fp16_kernel(__half* data, int rows, int cols,
                                            int S_q, int pad_start) {
    int lane = threadIdx.x % SM_WARP_SIZE;
    int row = blockIdx.x;
    if (row >= rows) return;

    // Per-head Q index: rows are laid out as (NH * S_q) where the inner
    // axis is S_q. We mask cols j > q for this row.
    int q = row % S_q;
    int row_mask_start = (q + 1 < pad_start) ? (q + 1) : pad_start;

    __half* src = data + row * cols;
    int cols2 = cols / 2;
    __half2* src2 = reinterpret_cast<__half2*>(src);

    // Pass 1: load + mask + find max
    float reg[SM_ITERS];
    float mx = -1e30f;

    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            int c_base = c2 * 2;
            __half2 v2 = src2[c2];
            float v0 = __half2float(v2.x);
            float v1 = __half2float(v2.y);
            if (c_base >= row_mask_start)     v0 = -1e30f;
            if (c_base + 1 >= row_mask_start) v1 = -1e30f;
            reg[it*2]   = v0;
            reg[it*2+1] = v1;
            mx = fmaxf(mx, fmaxf(v0, v1));
        } else {
            reg[it*2]   = -1e30f;
            reg[it*2+1] = -1e30f;
        }
    }
    if ((cols & 1) && lane == 0) {
        int c = cols - 1;
        float v = __half2float(src[c]);
        if (c >= row_mask_start) v = -1e30f;
        reg[SM_ITERS-1] = v;
        mx = fmaxf(mx, v);
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));

    // Pass 2: exp + sum
    float sm = 0;
    #pragma unroll
    for (int it = 0; it < SM_ITERS; it++) {
        reg[it] = __expf(reg[it] - mx);
        sm += reg[it];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sm += __shfl_xor_sync(0xffffffff, sm, o);

    // Pass 3: normalize + store
    float inv = 1.f / (sm + 1e-8f);
    #pragma unroll
    for (int it = 0; it < SM_ITERS / 2; it++) {
        int c2 = it * SM_WARP_SIZE + lane;
        if (c2 < cols2) {
            __half2 v2;
            v2.x = __float2half(reg[it*2]   * inv);
            v2.y = __float2half(reg[it*2+1] * inv);
            src2[c2] = v2;
        }
    }
    if ((cols & 1) && lane == 0) {
        src[cols-1] = __float2half(reg[SM_ITERS-1] * inv);
    }
}

void softmax_causal_fp16(__half* data, int rows, int cols,
                          int S_q, int pad_start,
                          cudaStream_t stream) {
    softmax_causal_fp16_kernel<<<rows, SM_WARP_SIZE, 0, stream>>>(
        data, rows, cols, S_q, pad_start);
}

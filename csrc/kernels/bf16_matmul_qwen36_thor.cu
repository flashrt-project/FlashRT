// Thor-only bf16 matmul kernels for Qwen3.6 — see header for the
// hardware-isolation contract.

#include "bf16_matmul_qwen36_thor.cuh"

namespace flash_rt::kernels {

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;  // 256

// M-tile kernel preserving the generic-chunked fma order. For a given
// (m, n) output, lane t covers j = lane, lane+32, ..., K_FIXED-32+lane
// with single-bf16 reads and one float fma per iteration —
// bit-identical to the shared generic kernel at the same K and one m.
// The across-mt loop carries independent accumulators, so each
// acc[mt] sees the same fma sequence the generic kernel would. The
// gain comes from reading W once per (n, m_tile) block instead of
// once per (n, m), so W bandwidth scales 1 / M_TILE.
template<int K_FIXED, int M_TILE>
__global__ void bf16_matmul_mtile_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ out,
    int M, int N) {
    extern __shared__ __align__(16) __nv_bfloat16 x_sh[];

    const int m0 = blockIdx.y * M_TILE;

    for (int j = threadIdx.x; j < M_TILE * K_FIXED; j += kThreads) {
        int mt = j / K_FIXED;
        int kk = j - mt * K_FIXED;
        int m = m0 + mt;
        x_sh[j] = (m < M) ? x[m * K_FIXED + kk]
                          : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * kWarpsPerBlock + warp_id;
    if (n >= N) return;

    const __nv_bfloat16* w_row = W + n * K_FIXED;

    float acc[M_TILE];
    #pragma unroll
    for (int mt = 0; mt < M_TILE; ++mt) acc[mt] = 0.0f;

    #pragma unroll 1
    for (int j = lane; j < K_FIXED; j += 32) {
        float wv = static_cast<float>(w_row[j]);
        #pragma unroll
        for (int mt = 0; mt < M_TILE; ++mt) {
            float xv = static_cast<float>(x_sh[mt * K_FIXED + j]);
            acc[mt] = fmaf(xv, wv, acc[mt]);
        }
    }

    #pragma unroll
    for (int mt = 0; mt < M_TILE; ++mt) {
        #pragma unroll
        for (int off = 16; off > 0; off /= 2) {
            acc[mt] += __shfl_xor_sync(0xffffffff, acc[mt], off);
        }
        if (lane == 0 && (m0 + mt) < M) {
            out[(m0 + mt) * N + n] = __float2bfloat16(acc[mt]);
        }
    }
}

constexpr int kMtpFcK = 10240;
constexpr int kMtpFcMTile = 8;
constexpr int kMtpFcSmemBytes =
    kMtpFcMTile * kMtpFcK * static_cast<int>(sizeof(__nv_bfloat16));

// Cached probe result: -1 = not probed, 0 = unsupported, 1 = ready.
// Probing is one-shot per process. Atomic on coarse-grained
// transitions is unnecessary because the probe is idempotent.
int g_mtp_fc_ready = -1;

int probe_mtp_fc_support() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;

    int max_optin = 0;
    if (cudaDeviceGetAttribute(
            &max_optin,
            cudaDevAttrMaxSharedMemoryPerBlockOptin,
            dev) != cudaSuccess) {
        return 0;
    }
    if (max_optin < kMtpFcSmemBytes) {
        return 0;
    }

    cudaError_t err = cudaFuncSetAttribute(
        bf16_matmul_mtile_kernel<kMtpFcK, kMtpFcMTile>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kMtpFcSmemBytes);
    if (err != cudaSuccess) {
        // Clear sticky error so subsequent CUDA calls are not
        // misattributed to this probe.
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

}  // namespace

int bf16_matmul_qwen36_thor_mtp_fc_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* W,
    __nv_bfloat16* out,
    int M,
    int N,
    cudaStream_t stream) {
    if (M < 1 || N < 1) return 0;

    if (g_mtp_fc_ready < 0) {
        g_mtp_fc_ready = probe_mtp_fc_support();
    }
    if (g_mtp_fc_ready == 0) {
        // Caller must dispatch the shared kernel instead. Reporting
        // failure rather than silently falling back keeps the call
        // sites explicit about which path executed.
        return 1;
    }

    dim3 grid((N + kWarpsPerBlock - 1) / kWarpsPerBlock,
              (M + kMtpFcMTile - 1) / kMtpFcMTile);
    bf16_matmul_mtile_kernel<kMtpFcK, kMtpFcMTile>
        <<<grid, kThreads, kMtpFcSmemBytes, stream>>>(x, W, out, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Disable the fast path for the rest of the process: the
        // launch is invalid on this device. Caller will retry via
        // the shared kernel.
        g_mtp_fc_ready = 0;
        return 1;
    }
    return 0;
}

}  // namespace flash_rt::kernels

#include "smallm_fp8.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

// ================================================================
// FlashRT AMD — hand-tuned small-M FP8 GEMM kernels (CDNA4, wave64)
//
// See smallm_fp8.h for the contract. Design notes:
//
// The problem is bandwidth: weight bytes (N*K fp8) dwarf everything
// else (A is M*K <= 64KB, D is M*N*2). The floor is one pass over B
// at DRAM rate, so both kernels stream B exactly once with 16-byte
// (dwordx4) lane loads and keep A resident (LDS for nn, registers
// for nt), accumulating FP32 per lane and folding with wave shuffles.
//
// nn ("kn" weights, B[k*N + n], N contiguous per k-row):
//   Workgroup = 256 threads = 4 waves; each wave owns 16 consecutive
//   output columns, so a WG covers a 64-column tile — the 4 waves
//   together touch 64 consecutive bytes of every k-row (a full 64B
//   DRAM sector). Within a wave, the 64 lanes partition K: lane l
//   streams rows k = l, l+64, ... of the WG's k-slice, loading 16
//   bytes of B per row. A is staged transposed in LDS as
//   aT[k][16 bytes] so a lane fetches all M values for its row with
//   one ds_read_b128. Lane partials fold with a fixed shfl_down tree.
//   Grid = (N/64) x S k-slices. n-parallelism alone (N/64 WGs) cannot
//   fill an MI350X, so S slices of K run in parallel and a second
//   kernel reduces the FP32 partials in ascending slice order
//   (deterministic, no atomics — required inside captured graphs).
//   S is the smallest power of two that reaches ~256 WGs (~512 for
//   the _alt config), capped so it divides K/64 and by the workspace
//   bound SMALLM_FP8_MAX_SPLIT. Split-K costs S*M*N*4 bytes written
//   + read back (~1.3MB for the dominant pi05 shapes at S<=16, i.e.
//   ~25-35% of the 4MB weight stream) — the price of occupancy.
//
// nt ("nk" weights, B[n*K + k], K contiguous per output column):
//   The naturally streaming layout: a wave owns whole output columns,
//   lanes tile K in 16-byte chunks (64 lanes x 16B = 1KB of one B row
//   per pass — perfectly coalesced full cache lines), and column
//   parallelism alone fills the GPU with NO split-K, NO workspace, NO
//   LDS. Prefer this path: set the frontend's fp8_layout="nk" so
//   decoder weights are packed (N,K).
//
//   ILP=4 streaming (stream_probe, gfx950 — see
//   FlashRT-amd-dev/notes/stream_craft_gfx950.md): one dwordx4 chain
//   per lane plateaus at ~1373 GB/s @4MB / 3235 @64MB (latency-window
//   bound, Little's law); FOUR independent dwordx4 chains per lane
//   reach 1534-5855 GB/s (co_x4_i4 winners). The kernel therefore
//   keeps NPW * CG == 4 independent B-row loads in flight per lane:
//   NPW simultaneous output columns x CG consecutive K-chunks, all
//   NPW*CG loads ISSUED before any convert/FMA consumes them.
//   Consumption then runs in the ORIGINAL fixed order — ascending
//   chunk, ascending column, ascending j inside the 16-byte packet —
//   so per-column fp32 accumulation order is UNCHANGED and results
//   stay bit-identical to the pre-ILP4 kernel (and to the fused FFN
//   kernels, which copy this skeleton). ILP comes purely from load
//   scheduling, never from splitting accumulators.
//
//   Host config pick (values are bit-identical for ANY (NPW, WAVES):
//   a column's math never depends on which wave computes it):
//     default: 4 waves/WG; largest NPW in {4,2,1} keeping
//       grid = N/(4*NPW) >= 256 WGs (the measured saturation grid) —
//       pi05 site N=1024 -> NPW=1 (256 WGs, CG=4: the whole K=4096
//       row preloaded as 4 chains); N>=4096 -> NPW=4 (A-convert
//       amortized 4x, grid still >= 256).
//     alt: 8 waves/WG (w8 — probe: helps at larger footprints,
//       neutral at small), same wave-count-preserving NPW rule
//       against grid >= 128.
//
// fp8 -> f32 conversion: per-byte through __hip_fp8_e4m3's float
// operator, which lowers to the gfx950 native V_CVT_*_FP8 ops. A
// 256-entry LDS LUT was considered and rejected: the native converts
// are cheap on CDNA4 and a LUT would add an LDS gather (64 scattered
// lane reads) per 16 weight bytes plus LDS pressure.
//
// M is dispatched to a compiled tile height M_T in {4, 8, 10, 16}
// (accumulators are register arrays, so the m-loop must be
// compile-time). M=10 (pi05 action chunk) hits its exact tile.
// ================================================================

#define SMALLM_FP8_BLOCK 256   // threads per workgroup (4 waves)

// ================================================================
// Async global->LDS staging plumbing (gfx9 LDS-DMA).
//
// FRT_LDS_ASYNC (compile-time, default 1) gates the use of
//   __builtin_amdgcn_global_load_lds(gptr_as1, lptr_as3, size, offset, aux)
// which lowers to GLOBAL_LOAD_DWORD ... lds — the hardware reads one
// dword per active lane from the per-lane global address and the LDS-DMA
// path deposits lane l's dword at (wave-uniform lptr) + 4*l, WITHOUT
// touching VGPRs. size/offset/aux must be integer constants; size=4 is
// the portable gfx9 granularity (gfx950 also accepts 12/16 — dwordx3/x4
// per lane, a follow-up A/B knob, see the kernel notes). One issue
// therefore stages 256B per wave; we issue MANY per buffer.
//
// The builtin is a *target* builtin: visible only in the device pass
// (__HIP_DEVICE_COMPILE__), never in the hipcc host pass nor under the
// g++ -fsyntax-only stub harness. So:
//   - FRT_LDS_ASYNC        : host+device agreement on whether the async
//                            variant is real (drives *_async_available()).
//   - FRT_ASYNC_LDS_CODEGEN: whether THIS compilation pass may emit the
//                            builtin call. Host pass compiles the sync
//                            body for the ASYNC instantiation — it never
//                            runs, only the device pass' code does.
//   - FRT_FORCE_ASYNC_LDS  : override for the g++ stub syntax check so
//                            the builtin call sites are type-checked.
// If ROCm lacks the builtin, fail LOUDLY (build with -DFRT_LDS_ASYNC=0
// to get a binary whose "async" variant is the sync fallback) — never
// silently downgrade, so the on-node A/B rows mean what they say.
// ================================================================
#ifndef FRT_LDS_ASYNC
#define FRT_LDS_ASYNC 1
#endif

#if defined(FRT_FORCE_ASYNC_LDS)
#  define FRT_ASYNC_LDS_CODEGEN FRT_FORCE_ASYNC_LDS
#elif defined(__HIP_DEVICE_COMPILE__)
#  if FRT_LDS_ASYNC
#    if !defined(__has_builtin) || !__has_builtin(__builtin_amdgcn_global_load_lds)
#      error "FRT_LDS_ASYNC=1 but __builtin_amdgcn_global_load_lds is unavailable; rebuild with -DFRT_LDS_ASYNC=0"
#    endif
#    define FRT_ASYNC_LDS_CODEGEN 1
#  else
#    define FRT_ASYNC_LDS_CODEGEN 0
#  endif
#else
#  define FRT_ASYNC_LDS_CODEGEN 0
#endif

// Clang address-space qualifier (ignored elsewhere; the g++ stub path
// then sees plain pointers, matching the stubbed builtin signature).
#if defined(__clang__)
#define FRT_AS(n) __attribute__((address_space(n)))
#else
#define FRT_AS(n)
#endif

// The helpers below are ALWAYS defined (their names appear inside
// `if constexpr` branches, which must resolve even when discarded);
// only their bodies are gated. The empty bodies are unreachable: the
// kernels call them only when kAsync (== FRT_ASYNC_LDS_CODEGEN != 0)
// is true, and only the device pass' code ever executes.

// One dword per lane, global -> LDS, no VGPR round-trip. gp is the
// per-lane global address (VGPR); lp must be WAVE-UNIFORM — the DMA
// writes lane l at lp + 4*l. Inactive lanes (EXEC-masked) stage nothing.
// Address-space casts follow composable_kernel's proven pattern
// (amd_direct_load): generic->as(1)/as(3) through uintptr_t — the
// generic LDS aperture bits live above bit 31, so the truncating cast
// to the 32-bit as(3) pointer yields the raw LDS offset.
__device__ __forceinline__ void smallm_async_copy_dword(
        const unsigned char* gp, unsigned char* lp) {
#if FRT_ASYNC_LDS_CODEGEN
    auto* g = reinterpret_cast<const FRT_AS(1) unsigned*>(
        reinterpret_cast<uintptr_t>(gp));
    auto* l = reinterpret_cast<FRT_AS(3) unsigned*>(
        reinterpret_cast<uintptr_t>(lp));
    __builtin_amdgcn_global_load_lds(g, l, /*size=*/4, /*offset=*/0, /*aux=*/0);
#else
    (void)gp; (void)lp;  // unreachable stub (see note above)
#endif
}

// s_waitcnt vmcnt(0), other counters untouched. gfx9 simm16 encoding:
// vmcnt lo [3:0]=0, expcnt [6:4]=7 (ignore), lgkmcnt [11:8]=15 (ignore),
// vmcnt hi [15:14]=0  ->  0x0F70. LDS-DMA completion is tracked by
// vmcnt, and the compiler does NOT model the DMA->ds_read dependency,
// so this wait is mandatory before reading a freshly staged buffer.
__device__ __forceinline__ void smallm_wait_lds_dma() {
#if FRT_ASYNC_LDS_CODEGEN
    __builtin_amdgcn_s_waitcnt(0x0F70);
#endif
}

// ── fp8 e4m3 byte -> f32 ──
__device__ __forceinline__ float smallm_fp8_to_f32(unsigned char b) {
    __hip_fp8_e4m3 v;
    v.__x = b;
    return static_cast<float>(v);
}

// Unpack 16 fp8 bytes (one dwordx4 load) to f32.
__device__ __forceinline__ void smallm_fp8x16_to_f32(const uint4& v, float* out) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    #pragma unroll
    for (int j = 0; j < 16; ++j) out[j] = smallm_fp8_to_f32(b[j]);
}

// Unpack 8 fp8 bytes (one b64 load) to f32 — the LDS-pipeline lane
// granularity (512B chunk / 64 lanes = 8B per lane).
__device__ __forceinline__ void smallm_fp8x8_to_f32(const uint2& v, float* out) {
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = smallm_fp8_to_f32(b[j]);
}

// ================================================================
// nn kernel: B stored (K, N) row-major.
//   grid = (N/64, S), block = 256. Dynamic LDS = k_slice * 16 bytes.
//   S == 1: writes scaled BF16 straight to D.
//   S  > 1: writes unscaled FP32 partials to ws[s][m][n]; the reduce
//           kernel below finishes deterministically.
// ================================================================
template<int M_T>
__global__ __launch_bounds__(SMALLM_FP8_BLOCK)
void smallm_fp8_nn_kernel(const unsigned char* __restrict__ A,
                          const unsigned char* __restrict__ B,
                          __hip_bfloat16* __restrict__ D,
                          float* __restrict__ ws,
                          const float* __restrict__ d_scale_a,
                          const float* __restrict__ d_scale_b,
                          int M, int N, int K, int k_slice) {
    extern __shared__ unsigned char smallm_lds_aT[];  // aT[k_local][16]
    const int tid  = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int n0   = blockIdx.x * 64 + wave * 16;   // wave's 16-column group
    const int s    = blockIdx.y;                    // k-slice index
    const int k0   = s * k_slice;

    // Stage this k-slice of A transposed: aT[k - k0][m], m zero-padded
    // to 16 so a lane reads its row's M values as one 16B LDS load.
    for (int i = tid; i < k_slice * 16; i += SMALLM_FP8_BLOCK) {
        const int m = i & 15, koff = i >> 4;
        smallm_lds_aT[i] =
            (m < M) ? A[(size_t)m * K + (k0 + koff)] : (unsigned char)0;
    }
    __syncthreads();

    float acc[M_T * 16];
    #pragma unroll
    for (int i = 0; i < M_T * 16; ++i) acc[i] = 0.0f;

    // Lane l streams rows k0 + l, k0 + l + 64, ... — fixed sequential
    // order per lane (deterministic accumulation).
    const int iters = k_slice >> 6;
    #pragma unroll 4
    for (int it = 0; it < iters; ++it) {
        const int kl = (it << 6) + lane;  // row within the slice
        const uint4 bv = *reinterpret_cast<const uint4*>(
            B + (size_t)(k0 + kl) * N + n0);
        float bf[16];
        smallm_fp8x16_to_f32(bv, bf);
        const uint4 av = reinterpret_cast<const uint4*>(smallm_lds_aT)[kl];
        const unsigned char* ab = reinterpret_cast<const unsigned char*>(&av);
        #pragma unroll
        for (int m = 0; m < M_T; ++m) {
            const float a = smallm_fp8_to_f32(ab[m]);
            #pragma unroll
            for (int j = 0; j < 16; ++j) acc[m * 16 + j] += a * bf[j];
        }
    }

    // Fixed-shape wave shuffle tree (order identical every launch).
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        #pragma unroll
        for (int i = 0; i < M_T * 16; ++i)
            acc[i] += __shfl_down(acc[i], off);
    }

    if (lane == 0) {
        if (gridDim.y > 1) {
            float* dst = ws + (size_t)s * M * N;
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                        dst[(size_t)m * N + (n0 + j)] = acc[m * 16 + j];
                }
            }
        } else {
            const float sc = d_scale_a[0] * d_scale_b[0];
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                        D[(size_t)m * N + (n0 + j)] =
                            __float2bfloat16(acc[m * 16 + j] * sc);
                }
            }
        }
    }
}

// ================================================================
// Split-K reduce: D[i] = bf16(sum_{s ascending} ws[s][i] * scale).
// One thread per 4 outputs (float4 partial reads). The s-loop runs in
// fixed ascending order — the only cross-workgroup reduction in the
// nn path, and it is deterministic by construction.
// ================================================================
__global__ __launch_bounds__(SMALLM_FP8_BLOCK)
void smallm_fp8_split_reduce_kernel(const float* __restrict__ ws,
                                    __hip_bfloat16* __restrict__ D,
                                    const float* __restrict__ d_scale_a,
                                    const float* __restrict__ d_scale_b,
                                    int split, long long mn) {
    const long long i4 =
        ((long long)blockIdx.x * SMALLM_FP8_BLOCK + threadIdx.x) * 4;
    if (i4 >= mn) return;  // mn is a multiple of 4 (N % 64 == 0)
    float sx = 0.0f, sy = 0.0f, sz = 0.0f, sw = 0.0f;
    for (int s = 0; s < split; ++s) {
        const float4 p =
            *reinterpret_cast<const float4*>(ws + (size_t)s * mn + i4);
        sx += p.x; sy += p.y; sz += p.z; sw += p.w;
    }
    const float sc = d_scale_a[0] * d_scale_b[0];
    D[i4 + 0] = __float2bfloat16(sx * sc);
    D[i4 + 1] = __float2bfloat16(sy * sc);
    D[i4 + 2] = __float2bfloat16(sz * sc);
    D[i4 + 3] = __float2bfloat16(sw * sc);
}

// ================================================================
// nt kernel: B stored (N, K) row-major.
//   grid = ceil(N / (WAVES*NPW)), block = WAVES*64; wave w owns output
//   columns nb + w*NPW .. +NPW-1. Lanes tile K in 16B chunks (1KB per
//   wave pass). No LDS, no split-K, no workspace.
//
//   ILP=4: each outer iteration covers a group of CG = 4/NPW
//   consecutive K-chunks. All NPW*CG == 4 independent B dwordx4 loads
//   for the group are issued FIRST (distinct destination registers ->
//   4 loads in flight per lane, the measured gfx950 saturation
//   lever), then consumed in the ORIGINAL fixed order: ascending
//   chunk u, A window converted once per chunk, then ascending column
//   p, ascending j. Per-accumulator fp32 order is bit-identical to
//   the pre-ILP4 kernel.
// ================================================================
template<int M_T, int NPW, int WAVES>
__global__ __launch_bounds__(WAVES * 64)
void smallm_fp8_nt_kernel(const unsigned char* __restrict__ A,
                          const unsigned char* __restrict__ B,
                          __hip_bfloat16* __restrict__ D,
                          const float* __restrict__ d_scale_a,
                          const float* __restrict__ d_scale_b,
                          int M, int N, int K) {
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int nb   = blockIdx.x * (WAVES * NPW) + wave * NPW;

    // Chunk-group depth: NPW columns x CG chunks = 4 chains in flight.
    constexpr int CG = (NPW >= 4) ? 1 : (4 / NPW);

    float acc[NPW][M_T];
    #pragma unroll
    for (int p = 0; p < NPW; ++p)
        #pragma unroll
        for (int m = 0; m < M_T; ++m) acc[p][m] = 0.0f;

    // K-chunks of 64 lanes x 16B = 1024 elements; the group loop and
    // the in-group consume order are both ascending-chunk, so the
    // accumulation order matches the plain sequential chunk loop.
    const int chunks = (K + 1023) >> 10;
    for (int c0 = 0; c0 < chunks; c0 += CG) {
        // ── issue all NPW x CG independent B loads (ILP4) ──
        uint4 braw[NPW][CG];
        bool  bok[NPW][CG];
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            #pragma unroll
            for (int u = 0; u < CG; ++u) {
                const int kw = ((c0 + u) << 10) + (lane << 4);
                bok[p][u] = (n < N) && (kw < K);  // K % 16 == 0: whole loads
                if (bok[p][u])
                    braw[p][u] = *reinterpret_cast<const uint4*>(
                        B + (size_t)n * K + kw);
            }
        }

        // ── consume: ascending chunk, then ascending column ──
        #pragma unroll
        for (int u = 0; u < CG; ++u) {
            const int kw = ((c0 + u) << 10) + (lane << 4);
            if (kw >= K) continue;  // tail guard

            // A window for this lane's 16 k's, all rows, converted
            // once per chunk (L1/L2-hot — not the stream).
            float a_f[M_T][16];
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                if (m < M) {
                    const uint4 av = *reinterpret_cast<const uint4*>(
                        A + (size_t)m * K + kw);
                    smallm_fp8x16_to_f32(av, a_f[m]);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) a_f[m][j] = 0.0f;
                }
            }

            #pragma unroll
            for (int p = 0; p < NPW; ++p) {
                if (!bok[p][u]) continue;
                float bf[16];
                smallm_fp8x16_to_f32(braw[p][u], bf);
                #pragma unroll
                for (int m = 0; m < M_T; ++m) {
                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                        acc[p][m] += a_f[m][j] * bf[j];
                }
            }
        }
    }

    // Fixed-shape wave shuffle tree.
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        #pragma unroll
        for (int p = 0; p < NPW; ++p)
            #pragma unroll
            for (int m = 0; m < M_T; ++m)
                acc[p][m] += __shfl_down(acc[p][m], off);
    }

    if (lane == 0) {
        const float sc = d_scale_a[0] * d_scale_b[0];
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nb + p;
            if (n < N) {
                #pragma unroll
                for (int m = 0; m < M_T; ++m)
                    if (m < M)
                        D[(size_t)m * N + n] = __float2bfloat16(acc[p][m] * sc);
            }
        }
    }
}

// ================================================================
// nt LDS kernel: B stored (N, K) row-major, double-buffered async
// global->LDS staging (the hipBLASLt craft, rocprof-verified: their
// 4.9-5.8us winners run 18-50KB of LDS wide staging while our
// register-only ILP4 kernel sits at 10.2us / ~410GB/s — the gap is the
// CONSUMER sitting on the load critical path, not raw bandwidth;
// stream_probe already reads 5+TB/s with ILP4 when nothing consumes).
//
//   Work partition (w4 ONLY — 256 threads; every w8 register variant
//   measured priv=76-168B scratch spills, so w8 is dropped from this
//   path entirely):
//     grid = ceil(N / COLS), COLS = 4 waves * NPW columns per wave.
//     Wave w owns output columns nwg + w*NPW .. +NPW-1 for the whole
//     kernel — and stages exactly those columns itself, so the LDS
//     buffers are per-wave-private slices (the WG barrier below is
//     kept for safety/symmetry, not for cross-wave data exchange; a
//     barrier-free per-wave pipeline is a documented follow-up A/B).
//
//   Chunking / LDS budget:
//     KC = 512 B of K per chunk per column. Buffer = COLS * KC:
//       NPW=4: 16 cols -> 8KB/buffer, 16KB total (2 buffers)
//       NPW=2:  8 cols -> 4KB/buffer,  8KB total
//       NPW=1:  4 cols -> 2KB/buffer,  4KB total
//     All well under the 32KB budget => >=4 WGs/CU by LDS. KC=512
//     (not 1024) so the DOMINANT pi05 shape K=1024 still has 2 chunks
//     and a real stage/compute overlap; K=4096 gets 8 pipeline steps.
//
//   Pipeline (double buffer, depth 1):
//     prol   stage S0 -> buf0 ; waitcnt vmcnt(0) ; s_barrier
//     c=0    issue S1 -> buf1 ▶ DMA fills buf1 ║ compute C0 from buf0
//            waitcnt vmcnt(0) ; s_barrier      (S1 landed, C0 retired)
//     c=1    issue S2 -> buf0 ▶                ║ compute C1 from buf1
//            waitcnt vmcnt(0) ; s_barrier
//     ...    (compute never waits on the loads feeding it — those
//             finished a whole compute phase ago; the FMA/convert
//             stream is fed from LDS at LDS rate)
//     Overwrite hazard: S(c+2) targets buf[c&1]; the barrier ending
//     iteration c sits AFTER compute(c), so every read of buf[c&1] has
//     retired before iteration c+1 issues S(c+2) into it.
//
//   Staging (per chunk): the buffer is COLS contiguous 512B column
//   slices, lds[buf][col*KC + k]. It is filled in 256B "units" (one
//   async issue: 64 lanes * 4B); unit u of wave w covers column
//   w*NPW + u/2, half u&1. ASYNC=true issues UPW = 2*NPW
//   global_load_lds dwords per lane back-to-back (8 in flight at
//   NPW=4). ASYNC=false is the register fallback: the same UPW dwords
//   are loaded global->VGPR with all loads hoisted BEFORE compute (ILP
//   like the register kernel) and only deposited to LDS after compute,
//   so the consumer stays decoupled in both variants; the two variants
//   share identical addressing, which is what makes the on-node A/B a
//   single-variable comparison of the LDS-DMA path itself.
//
//   Compute (per chunk): lane l consumes bytes [l*8, l*8+8) of each
//   owned column slice (ds_read_b64, conflict-free: 64 lanes hit 512
//   consecutive bytes), converts A (global, L1/L2-hot — A is tiny and
//   stays on the existing register-convert scheme) and B (LDS) with
//   the native fp8 converts, and FMAs into acc[NPW][M_T].
//
//   Accumulation order (deterministic, graph-replay bit-identical):
//   for a given output element, lane l accumulates k = c*512 + l*8 + j
//   in ascending chunk c, ascending j — a fixed sequential order per
//   lane — then the fixed-shape shuffle tree folds lanes. No atomics.
//   NOTE: the lane<->k partition differs from the register nt kernel
//   (8B windows vs 16B), so results are NOT bit-identical to that
//   kernel — allowed: the gate is cos vs GemmRunner on the same
//   quantized inputs, and each variant is itself deterministic.
//
//   K handling: K % 16 == 0 (checked) => a lane's 8B window and every
//   staged dword are all-in or all-out of K; out-of-range units are
//   simply not staged and out-of-range windows not consumed, so stale
//   LDS bytes are never read. Tail columns (N % COLS) are guarded per
//   column in both stage and compute.
// ================================================================
template<int M_T, int NPW, bool ASYNC>
__global__ __launch_bounds__(SMALLM_FP8_BLOCK)
void smallm_fp8_nt_lds_kernel(const unsigned char* __restrict__ A,
                              const unsigned char* __restrict__ B,
                              __hip_bfloat16* __restrict__ D,
                              const float* __restrict__ d_scale_a,
                              const float* __restrict__ d_scale_b,
                              int M, int N, int K) {
    constexpr int COLS = 4 * NPW;   // output columns per WG (4 waves)
    constexpr int KC   = 512;       // K-bytes per chunk per column
    constexpr int UPW  = 2 * NPW;   // 256B stage units per wave per chunk
    // ASYNC only becomes LDS-DMA when this pass may emit the builtin
    // (device pass / forced stub check); elsewhere it compiles as the
    // sync body, which is fine — that code is never executed.
    constexpr bool kAsync = ASYNC && (FRT_ASYNC_LDS_CODEGEN != 0);

    __shared__ alignas(16) unsigned char lds[2][COLS * KC];

    const int tid  = (int)threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int nwg  = (int)blockIdx.x * COLS;
    const int chunks = (K + KC - 1) / KC;

    float acc[NPW][M_T];
    #pragma unroll
    for (int p = 0; p < NPW; ++p)
        #pragma unroll
        for (int m = 0; m < M_T; ++m) acc[p][m] = 0.0f;

    unsigned sreg[UPW];  // sync-fallback staging registers (dead if kAsync)

    // ── producer: issue chunk c's loads (async: LDS-DMA; sync: ->VGPR) ──
    auto stage_issue = [&](int c) {
        unsigned char* dst = lds[c & 1];
        const int kb = c * KC;
        #pragma unroll
        for (int i = 0; i < UPW; ++i) {
            const int u    = wave * UPW + i;   // wave stages its OWN columns
            const int col  = u >> 1;
            const int off  = (u & 1) * 256 + lane * 4;   // byte in column slice
            const int gk   = kb + off;
            const bool ok  = (nwg + col < N) && (gk < K);  // K%4==0: whole dword
            if constexpr (kAsync) {
                if (ok)
                    smallm_async_copy_dword(B + (size_t)(nwg + col) * K + gk,
                                            dst + col * KC + (u & 1) * 256);
            } else {
                sreg[i] = ok ? *reinterpret_cast<const unsigned*>(
                                   B + (size_t)(nwg + col) * K + gk)
                             : 0u;
            }
        }
    };

    // ── producer: deposit chunk c to LDS (sync fallback only — placed
    //    AFTER compute so the global loads overlap the FMA stream) ──
    auto stage_commit = [&](int c) {
        if constexpr (!kAsync) {
            unsigned char* dst = lds[c & 1];
            #pragma unroll
            for (int i = 0; i < UPW; ++i) {
                const int u = wave * UPW + i;
                const int col = u >> 1;
                const int off = (u & 1) * 256 + lane * 4;
                *reinterpret_cast<unsigned*>(dst + col * KC + off) = sreg[i];
            }
        }
    };

    // ── pipeline barrier: drain this wave's LDS-DMA (vmcnt — the
    //    compiler does not model DMA->ds_read deps), then WG barrier ──
    auto pipe_barrier = [&]() {
        if constexpr (kAsync) smallm_wait_lds_dma();
        __syncthreads();
    };

    // ── consumer: chunk c from lds[c&1] ──
    auto compute = [&](int c) {
        const unsigned char* buf = lds[c & 1];
        const int kb = c * KC;
        const int ko = lane * 8;              // lane's 8B window in the chunk
        if (kb + ko >= K) return;             // K%8==0 => all-in or all-out

        // A window: 8 bytes per row, converted once per chunk (same
        // scheme as the register kernel — A is L1/L2-hot, not the stream).
        float a_f[M_T][8];
        #pragma unroll
        for (int m = 0; m < M_T; ++m) {
            if (m < M) {
                const uint2 av = *reinterpret_cast<const uint2*>(
                    A + (size_t)m * K + kb + ko);
                smallm_fp8x8_to_f32(av, a_f[m]);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) a_f[m][j] = 0.0f;
            }
        }

        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int col = wave * NPW + p;
            if (nwg + col >= N) continue;
            const uint2 bv = *reinterpret_cast<const uint2*>(
                buf + col * KC + ko);
            float bf[8];
            smallm_fp8x8_to_f32(bv, bf);
            #pragma unroll
            for (int m = 0; m < M_T; ++m) {
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[p][m] += a_f[m][j] * bf[j];
            }
        }
    };

    // ── run the pipeline ──
    stage_issue(0);
    stage_commit(0);          // sync fallback prologue stalls once; fine
    pipe_barrier();           // buf0 ready WG-wide
    for (int c = 0; c < chunks; ++c) {
        if (c + 1 < chunks) stage_issue(c + 1);
        compute(c);
        if (c + 1 < chunks) {
            stage_commit(c + 1);
            pipe_barrier();   // next buffer landed; buf[c&1] reads retired
        }
    }

    // ── epilogue: identical to the register nt kernel ──
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) {
        #pragma unroll
        for (int p = 0; p < NPW; ++p)
            #pragma unroll
            for (int m = 0; m < M_T; ++m)
                acc[p][m] += __shfl_down(acc[p][m], off);
    }

    if (lane == 0) {
        const float sc = d_scale_a[0] * d_scale_b[0];
        #pragma unroll
        for (int p = 0; p < NPW; ++p) {
            const int n = nwg + wave * NPW + p;
            if (n < N) {
                #pragma unroll
                for (int m = 0; m < M_T; ++m)
                    if (m < M)
                        D[(size_t)m * N + n] = __float2bfloat16(acc[p][m] * sc);
            }
        }
    }
}

// ================================================================
// Host side
// ================================================================

size_t smallm_fp8_nn_dev_ws_bytes(int M, int N) {
    return (size_t)SMALLM_FP8_MAX_SPLIT * M * N * sizeof(float);
}

static void smallm_fp8_check(const char* who, int M, int N, int K,
                             int n_align, int k_align) {
    if (M < 1 || M > 16)
        throw std::runtime_error(std::string(who) + ": M must be in [1,16], got " +
                                 std::to_string(M));
    if (N < n_align || (N % n_align) != 0)
        throw std::runtime_error(std::string(who) + ": N must be a multiple of " +
                                 std::to_string(n_align) + ", got " + std::to_string(N));
    if (K < k_align || (K % k_align) != 0)
        throw std::runtime_error(std::string(who) + ": K must be a multiple of " +
                                 std::to_string(k_align) + ", got " + std::to_string(K));
}

// Smallest power-of-two split S with (N/64)*S >= target_wgs, subject to
// S | (K/64), S <= SMALLM_FP8_MAX_SPLIT, and LDS staging <= 64KB/slice.
static int smallm_fp8_pick_split(int N, int K, bool have_ws, int target_wgs) {
    const int k64 = K / 64;
    int s = 1;
    if (have_ws) {
        const int n_wgs = N / 64;
        while (n_wgs * s < target_wgs && (s << 1) <= SMALLM_FP8_MAX_SPLIT) s <<= 1;
        while (s > 1 && (s > k64 || (k64 % s) != 0)) s >>= 1;
    }
    // LDS bound: k_slice * 16 bytes <= 64KB  =>  K/S <= 4096.
    while ((size_t)(K / s) * 16 > 65536) {
        const int s2 = s << 1;
        if (!have_ws || s2 > SMALLM_FP8_MAX_SPLIT || s2 > k64 || (k64 % s2) != 0)
            throw std::runtime_error(
                "smallm_fp8_nn_dev: K too large for LDS staging without a "
                "(deeper) split_ws; K=" + std::to_string(K));
        s = s2;
    }
    return s;
}

static void smallm_fp8_nn_dev_impl(const void* A, const void* B, void* D,
                                   const float* d_scale_a, const float* d_scale_b,
                                   int M, int N, int K, float* split_ws,
                                   hipStream_t stream, int target_wgs) {
    smallm_fp8_check("smallm_fp8_nn_dev", M, N, K, 64, 64);
    const int s = smallm_fp8_pick_split(N, K, split_ws != nullptr, target_wgs);
    const int k_slice = K / s;
    const dim3 grid((unsigned)(N / 64), (unsigned)s);
    const dim3 block(SMALLM_FP8_BLOCK);
    const size_t shmem = (size_t)k_slice * 16;
    const unsigned char* a8 = static_cast<const unsigned char*>(A);
    const unsigned char* b8 = static_cast<const unsigned char*>(B);
    __hip_bfloat16* d16 = static_cast<__hip_bfloat16*>(D);

    #define SMALLM_LAUNCH_NN(MT)                                                \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(smallm_fp8_nn_kernel<MT>),           \
                           grid, block, shmem, stream,                          \
                           a8, b8, d16, split_ws, d_scale_a, d_scale_b,         \
                           M, N, K, k_slice)
    if      (M <= 4)  SMALLM_LAUNCH_NN(4);
    else if (M <= 8)  SMALLM_LAUNCH_NN(8);
    else if (M <= 10) SMALLM_LAUNCH_NN(10);
    else              SMALLM_LAUNCH_NN(16);
    #undef SMALLM_LAUNCH_NN

    if (s > 1) {
        const long long mn = (long long)M * N;
        const unsigned rblocks =
            (unsigned)((mn / 4 + SMALLM_FP8_BLOCK - 1) / SMALLM_FP8_BLOCK);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(smallm_fp8_split_reduce_kernel),
                           dim3(rblocks), block, 0, stream,
                           split_ws, d16, d_scale_a, d_scale_b, s, mn);
    }
}

void smallm_fp8_nn_dev(const void* A, const void* B, void* D,
                       const float* d_scale_a, const float* d_scale_b,
                       int M, int N, int K, float* split_ws,
                       hipStream_t stream) {
    smallm_fp8_nn_dev_impl(A, B, D, d_scale_a, d_scale_b, M, N, K,
                           split_ws, stream, /*target_wgs=*/256);
}

void smallm_fp8_nn_dev_alt(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, float* split_ws,
                           hipStream_t stream) {
    smallm_fp8_nn_dev_impl(A, B, D, d_scale_a, d_scale_b, M, N, K,
                           split_ws, stream, /*target_wgs=*/512);
}

template<int NPW, int WAVES>
static void smallm_fp8_nt_dev_impl(const void* A, const void* B, void* D,
                                   const float* d_scale_a, const float* d_scale_b,
                                   int M, int N, int K, hipStream_t stream) {
    smallm_fp8_check("smallm_fp8_nt_dev", M, N, K, 1, 16);
    const int cols_per_wg = WAVES * NPW;
    const dim3 grid((unsigned)((N + cols_per_wg - 1) / cols_per_wg));
    const dim3 block((unsigned)(WAVES * 64));
    const unsigned char* a8 = static_cast<const unsigned char*>(A);
    const unsigned char* b8 = static_cast<const unsigned char*>(B);
    __hip_bfloat16* d16 = static_cast<__hip_bfloat16*>(D);

    #define SMALLM_LAUNCH_NT(MT)                                                \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(smallm_fp8_nt_kernel<MT, NPW, WAVES>),\
                           grid, block, 0, stream,                              \
                           a8, b8, d16, d_scale_a, d_scale_b, M, N, K)
    if      (M <= 4)  SMALLM_LAUNCH_NT(4);
    else if (M <= 8)  SMALLM_LAUNCH_NT(8);
    else if (M <= 10) SMALLM_LAUNCH_NT(10);
    else              SMALLM_LAUNCH_NT(16);
    #undef SMALLM_LAUNCH_NT
}

// Shape-adaptive NPW: the largest NPW in {4, 2, 1} that keeps
// grid = N/(WAVES*NPW) >= min_wgs — the measured gfx950 saturation
// grid (stream_probe: 256 one-shot WGs of 4 waves with ILP4 saturate;
// fewer WGs starve the latency window). Results are bit-identical for
// every (NPW, WAVES): each output column's accumulation never depends
// on which wave computes it, only on the fixed chunk/j order.
template<int WAVES>
static void smallm_fp8_nt_dispatch(const void* A, const void* B, void* D,
                                   const float* d_scale_a, const float* d_scale_b,
                                   int M, int N, int K, int min_wgs,
                                   hipStream_t stream) {
    if (N >= WAVES * 4 * min_wgs)
        smallm_fp8_nt_dev_impl<4, WAVES>(A, B, D, d_scale_a, d_scale_b,
                                         M, N, K, stream);
    else if (N >= WAVES * 2 * min_wgs)
        smallm_fp8_nt_dev_impl<2, WAVES>(A, B, D, d_scale_a, d_scale_b,
                                         M, N, K, stream);
    else
        smallm_fp8_nt_dev_impl<1, WAVES>(A, B, D, d_scale_a, d_scale_b,
                                         M, N, K, stream);
}

template<int NPW, bool ASYNC>
static void smallm_fp8_nt_lds_impl(const void* A, const void* B, void* D,
                                   const float* d_scale_a, const float* d_scale_b,
                                   int M, int N, int K, hipStream_t stream) {
    constexpr int cols = 4 * NPW;   // w4 only — every w8 variant spills
    const dim3 grid((unsigned)((N + cols - 1) / cols));
    const dim3 block(SMALLM_FP8_BLOCK);
    const unsigned char* a8 = static_cast<const unsigned char*>(A);
    const unsigned char* b8 = static_cast<const unsigned char*>(B);
    __hip_bfloat16* d16 = static_cast<__hip_bfloat16*>(D);

    #define SMALLM_LAUNCH_NT_LDS(MT)                                            \
        hipLaunchKernelGGL(                                                     \
            HIP_KERNEL_NAME(smallm_fp8_nt_lds_kernel<MT, NPW, ASYNC>),          \
            grid, block, 0, stream, a8, b8, d16, d_scale_a, d_scale_b, M, N, K)
    if      (M <= 4)  SMALLM_LAUNCH_NT_LDS(4);
    else if (M <= 8)  SMALLM_LAUNCH_NT_LDS(8);
    else if (M <= 10) SMALLM_LAUNCH_NT_LDS(10);
    else              SMALLM_LAUNCH_NT_LDS(16);
    #undef SMALLM_LAUNCH_NT_LDS
}

// Same shape-adaptive NPW rule as the register path (grid >= 256 WGs,
// the measured gfx950 saturation grid): N>=4096 -> NPW=4 (16 cols/WG,
// 16KB LDS), N>=2048 -> NPW=2 (8KB), else NPW=1 (4KB).
template<bool ASYNC>
static void smallm_fp8_nt_lds_dispatch(const void* A, const void* B, void* D,
                                       const float* d_scale_a,
                                       const float* d_scale_b,
                                       int M, int N, int K, hipStream_t stream) {
    if (N >= 4 * 4 * 256)
        smallm_fp8_nt_lds_impl<4, ASYNC>(A, B, D, d_scale_a, d_scale_b,
                                         M, N, K, stream);
    else if (N >= 4 * 2 * 256)
        smallm_fp8_nt_lds_impl<2, ASYNC>(A, B, D, d_scale_a, d_scale_b,
                                         M, N, K, stream);
    else
        smallm_fp8_nt_lds_impl<1, ASYNC>(A, B, D, d_scale_a, d_scale_b,
                                         M, N, K, stream);
}

int smallm_fp8_nt_lds_async_available() {
    return FRT_LDS_ASYNC ? 1 : 0;
}

void smallm_fp8_nt_lds_dev(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, int use_async,
                           hipStream_t stream) {
    smallm_fp8_check("smallm_fp8_nt_lds_dev", M, N, K, 1, 16);
    // use_async selects between the two compiled variants at runtime so
    // one bench process can A/B them. If the binary was built with
    // -DFRT_LDS_ASYNC=0 the async instantiation is the sync body; route
    // to sync explicitly so the choice is honest either way (query
    // smallm_fp8_nt_lds_async_available() to annotate bench rows).
    if (use_async != 0 && FRT_LDS_ASYNC)
        smallm_fp8_nt_lds_dispatch<true>(A, B, D, d_scale_a, d_scale_b,
                                         M, N, K, stream);
    else
        smallm_fp8_nt_lds_dispatch<false>(A, B, D, d_scale_a, d_scale_b,
                                          M, N, K, stream);
}

void smallm_fp8_nt_dev(const void* A, const void* B, void* D,
                       const float* d_scale_a, const float* d_scale_b,
                       int M, int N, int K, float* split_ws,
                       hipStream_t stream) {
    (void)split_ws;  // nt never splits K — accepted for signature symmetry
    // Default is now the LDS double-buffered pipeline (async staging
    // unless FRT_LDS_ASYNC=0 in the env or at build time). The legacy
    // register-only ILP4 path stays behind FRT_SMALLM_REG=1 for A/B.
    // NOTE: env vars are read on the HOST at launch/capture time — a
    // captured graph keeps whatever the env said at capture; flip the
    // env BEFORE capturing, or use smallm_fp8_nt_lds_dev directly.
    const char* reg = std::getenv("FRT_SMALLM_REG");
    if (reg && reg[0] == '1' && reg[1] == '\0') {
        // Legacy: 4 waves/WG, grid target >= 256 WGs. pi05 site N=1024
        // -> NPW=1 (256 WGs, CG=4); N>=2048 -> NPW=2; N>=4096 -> NPW=4.
        smallm_fp8_nt_dispatch<4>(A, B, D, d_scale_a, d_scale_b, M, N, K,
                                  /*min_wgs=*/256, stream);
        return;
    }
    const char* as = std::getenv("FRT_LDS_ASYNC");
    const int use_async = (as && as[0] == '0' && as[1] == '\0') ? 0 : 1;
    smallm_fp8_nt_lds_dev(A, B, D, d_scale_a, d_scale_b, M, N, K,
                          use_async, stream);
}

void smallm_fp8_nt_dev_alt(const void* A, const void* B, void* D,
                           const float* d_scale_a, const float* d_scale_b,
                           int M, int N, int K, float* split_ws,
                           hipStream_t stream) {
    (void)split_ws;
    // Alt: w8 (512-thread WGs — probe: 8 waves helps at larger
    // footprints, neutral at small), same total wave count as the
    // default via a halved grid target.
    smallm_fp8_nt_dispatch<8>(A, B, D, d_scale_a, d_scale_b, M, N, K,
                              /*min_wgs=*/128, stream);
}

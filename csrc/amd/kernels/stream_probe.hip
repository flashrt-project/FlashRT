#include "stream_probe.h"
#include "common_hip.h"

#include <stdexcept>
#include <string>

// ================================================================
// FlashRT AMD — weight-streaming read-bandwidth probe (CDNA4, wave64)
//
// See stream_probe.h for the contract. Design notes:
//
// The physics under test is Little's law: sustained read bandwidth =
// bytes in flight / memory latency. A wave with one dwordx4 load
// outstanding contributes 16B per lane = 1KB in flight; whether the
// machine reaches the DRAM rate depends entirely on how many such
// loads the whole grid keeps outstanding at once. The three levers
// are ILP (independent loads per thread before the first s_waitcnt),
// TLP (waves resident per CU), and grid (CUs covered + waves per CU).
// The probe holds everything else constant — one pass over the
// buffer, u32 adds only — so the measured GB/s isolates the levers.
//
// ILP is kept honest: every outer loop is `#pragma unroll 1`, so a
// variant's chain depth is exactly its template parameter. At ILP=N
// the N loads have independent addresses and no consumer between
// them, so the compiler clusters them ahead of one s_waitcnt block —
// N*16B in flight per lane. (ILP=1 still allows the scheduler to
// overlap the loop-carried next-address computation, mirroring what
// the real kernels' serial K loops get for free — that is the point:
// variant 0 is the existing kernels' pattern, measured.)
//
// The strided pattern reproduces the small-M GEMM nn access shape:
// each lane serially walks its own 1KB row, so at any instant the 64
// lanes of a wave touch 64 different cache lines 1KB apart. Per-lane
// the stream is contiguous, per-instruction it is maximally scattered
// — the DRAM-sector-utilization question the GEMM raised.
//
// The nontemporal variants use __builtin_nontemporal_load, which on
// gfx950 sets the load's temporal-hint bits so streamed weights do
// not evict L2 working set. For a pure streaming pass it also tests
// whether skipping L2 allocation changes the achievable rate.
//
// Checksum: u32 lane accumulate -> wave shfl tree -> LDS across
// waves -> one word per WG. Fixed order is irrelevant here (integer
// add is associative/commutative); the sum exists only to keep the
// loads live and to let the host cross-check that every variant read
// every byte exactly once.
// ================================================================

// ── 16B / 8B load vector types ──
// clang ext_vector_type lowers to single dwordx4 / dwordx2 loads and
// is accepted by __builtin_nontemporal_load. The plain-struct branch
// exists only for host-side syntax tooling; hipcc is always clang.
#if defined(__clang__)
typedef unsigned int probe_u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int probe_u32x2 __attribute__((ext_vector_type(2)));
#else
struct probe_u32x4 { unsigned x, y, z, w; };
struct probe_u32x2 { unsigned x, y; };
#endif

namespace {

constexpr int PROBE_ROW_BYTES = 1024;   // strided pattern: 1KB row per lane
constexpr int PROBE_MAX_WAVES = 16;     // 1024-thread WG ceiling

template<int LOAD> struct probe_vec { using type = probe_u32x4; };
template<>         struct probe_vec<2> { using type = probe_u32x2; };

__device__ __forceinline__ unsigned vsum(probe_u32x4 v) { return v.x + v.y + v.z + v.w; }
__device__ __forceinline__ unsigned vsum(probe_u32x2 v) { return v.x + v.y; }

template<int LOAD, typename V>
__device__ __forceinline__ V probe_load(const V* p) {
#if defined(__clang__)
    if constexpr (LOAD == 1) return __builtin_nontemporal_load(p);
#endif
    return *p;
}

// u32 block reduce: wave shfl tree, then LDS across waves; thread 0
// writes the WG's word. One word per WG — negligible write traffic.
__device__ __forceinline__ void probe_reduce_store(unsigned acc,
                                                   unsigned* __restrict__ out) {
    __shared__ unsigned wsum[PROBE_MAX_WAVES];
    const int lane = threadIdx.x & (FVK_WAVE_SIZE - 1);
    const int wave = threadIdx.x >> 6;
    #pragma unroll
    for (int off = FVK_WAVE_SIZE / 2; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (lane == 0) wsum[wave] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned total = 0;
        const int nwaves = blockDim.x >> 6;
        for (int w = 0; w < nwaves; ++w) total += wsum[w];
        out[blockIdx.x] = total;
    }
}

template<int ILP, int LOAD, bool STRIDED, bool PERSISTENT>
__global__ void stream_probe_kernel(const unsigned* __restrict__ src,
                                    size_t n_bytes,
                                    unsigned* __restrict__ out) {
    using V = typename probe_vec<LOAD>::type;
    constexpr size_t VB = sizeof(V);
    const V* vsrc = reinterpret_cast<const V*>(src);
    unsigned acc = 0;

    if constexpr (!STRIDED) {
        // ── lane-coalesced streaming ──
        // Chain element k sits one thread-stride further along, so at
        // every k the full WG (or grid) front is contiguous in memory.
        const size_t n_vec = n_bytes / VB;
        size_t base, end, tstride;
        if constexpr (PERSISTENT) {
            base    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
            end     = n_vec;
            tstride = (size_t)gridDim.x * blockDim.x;
        } else {
            const size_t slab = (n_vec + gridDim.x - 1) / gridDim.x;
            const size_t beg  = (size_t)blockIdx.x * slab;
            base    = beg + threadIdx.x;
            end     = beg + slab < n_vec ? beg + slab : n_vec;
            tstride = blockDim.x;
        }
        const size_t chain_span = (size_t)(ILP - 1) * tstride;
        size_t i = base;
        #pragma unroll 1
        while (i + chain_span < end) {          // hot loop: chain fully in bounds
            V v[ILP];
            #pragma unroll
            for (int k = 0; k < ILP; ++k)
                v[k] = probe_load<LOAD>(vsrc + i + (size_t)k * tstride);
            #pragma unroll
            for (int k = 0; k < ILP; ++k) acc += vsum(v[k]);
            i += (size_t)ILP * tstride;
        }
        #pragma unroll 1
        for (; i < end; i += tstride)           // tail: remaining chain positions
            acc += vsum(probe_load<LOAD>(vsrc + i));
    } else {
        // ── row-strided: each lane serially walks its own 1KB row ──
        // (small-M GEMM nn shape: adjacent lanes 1KB apart, so every
        // load instruction touches 64 distinct cache lines per wave.)
        constexpr int ROW_VECS = PROBE_ROW_BYTES / (int)VB;
        static_assert(ROW_VECS % ILP == 0, "row must divide into ILP chains");
        const size_t n_rows = n_bytes / PROBE_ROW_BYTES;
        size_t rbase, rend, rstride;
        if constexpr (PERSISTENT) {
            rbase   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
            rend    = n_rows;
            rstride = (size_t)gridDim.x * blockDim.x;
        } else {
            const size_t slab = (n_rows + gridDim.x - 1) / gridDim.x;
            const size_t beg  = (size_t)blockIdx.x * slab;
            rbase   = beg + threadIdx.x;
            rend    = beg + slab < n_rows ? beg + slab : n_rows;
            rstride = blockDim.x;
        }
        #pragma unroll 1
        for (size_t r = rbase; r < rend; r += rstride) {
            const V* row = vsrc + r * ROW_VECS;
            #pragma unroll 1
            for (int j = 0; j < ROW_VECS; j += ILP) {
                V v[ILP];
                #pragma unroll
                for (int k = 0; k < ILP; ++k)
                    v[k] = probe_load<LOAD>(row + j + k);
                #pragma unroll
                for (int k = 0; k < ILP; ++k) acc += vsum(v[k]);
            }
        }
    }
    probe_reduce_store(acc, out);
}

// ── Variant matrix ──
// One-factor-at-a-time from the baseline (variant 0 = the shared
// pattern of the existing hand kernels), plus combined candidates
// for the expected winners on each axis pairing, plus the strided
// family that mirrors the small-M GEMM nn access shape.
//
// name grammar: <pattern>_<load>_i<ilp>_w<waves>_<g|p><grid>
//   co/rs = coalesced / row-strided; x4/nt/x2 = dwordx4 / dwordx4
//   nontemporal / dwordx2; g = slab partition, p = persistent
//   grid-stride.
const StreamProbeVariant kVariants[] = {
    //  name                  ilp waves grid load strided persistent
    { "co_x4_i1_w4_g256",      1,  4,  256, 0, 0, 0 },  // 0: baseline = existing kernels
    { "co_x4_i4_w4_g256",      4,  4,  256, 0, 0, 0 },  // 1: ILP axis
    { "co_x4_i8_w4_g256",      8,  4,  256, 0, 0, 0 },  // 2
    { "co_x4_i1_w8_g256",      1,  8,  256, 0, 0, 0 },  // 3: TLP axis
    { "co_x4_i1_w16_g256",     1, 16,  256, 0, 0, 0 },  // 4
    { "co_x4_i4_w8_g256",      4,  8,  256, 0, 0, 0 },  // 5: ILP x TLP
    { "co_x4_i1_w4_g64",       1,  4,   64, 0, 0, 0 },  // 6: grid axis
    { "co_x4_i1_w4_g1024",     1,  4, 1024, 0, 0, 0 },  // 7
    { "co_x4_i4_w4_g64",       4,  4,   64, 0, 0, 0 },  // 8: can ILP substitute grid?
    { "co_x4_i4_w4_g1024",     4,  4, 1024, 0, 0, 0 },  // 9
    { "co_x4_i4_w8_g1024",     4,  8, 1024, 0, 0, 0 },  // 10: max in-flight candidate
    { "co_x4_i8_w8_g1024",     8,  8, 1024, 0, 0, 0 },  // 11
    { "co_x4_i1_w4_p256",      1,  4,  256, 0, 0, 1 },  // 12: slab vs grid-stride @256
    { "co_x4_i4_w4_p256",      4,  4,  256, 0, 0, 1 },  // 13
    { "co_x4_i4_w8_p256",      4,  8,  256, 0, 0, 1 },  // 14
    { "co_nt_i1_w4_g256",      1,  4,  256, 1, 0, 0 },  // 15: nontemporal axis
    { "co_nt_i4_w8_g1024",     4,  8, 1024, 1, 0, 0 },  // 16: nt on winner candidate
    { "co_x2_i1_w4_g256",      1,  4,  256, 2, 0, 0 },  // 17: dwordx2 axis
    { "co_x2_i8_w4_g256",      8,  4,  256, 2, 0, 0 },  // 18: x2, bytes-in-flight = x4 i4
    { "rs_x4_i1_w4_g256",      1,  4,  256, 0, 1, 0 },  // 19: strided baseline (GEMM nn)
    { "rs_x4_i4_w4_g256",      4,  4,  256, 0, 1, 0 },  // 20: does ILP rescue strided?
    { "rs_x4_i1_w8_g1024",     1,  8, 1024, 0, 1, 0 },  // 21: does occupancy?
    { "rs_nt_i4_w8_g1024",     4,  8, 1024, 1, 1, 0 },  // 22: strided all-in
};
constexpr int kVariantCount = (int)(sizeof(kVariants) / sizeof(kVariants[0]));

template<int ILP, int LOAD>
void probe_launch(const StreamProbeVariant& v, const unsigned* src,
                  size_t n_bytes, unsigned* out, hipStream_t stream) {
    const dim3 grid((unsigned)v.grid);
    const dim3 block((unsigned)(v.waves * FVK_WAVE_SIZE));
    #define STREAM_PROBE_LAUNCH(S, P)                                          \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(stream_probe_kernel<ILP, LOAD, S, P>), \
                           grid, block, 0, stream, src, n_bytes, out)
    if      (!v.strided && !v.persistent) STREAM_PROBE_LAUNCH(false, false);
    else if (!v.strided &&  v.persistent) STREAM_PROBE_LAUNCH(false, true);
    else if ( v.strided && !v.persistent) STREAM_PROBE_LAUNCH(true,  false);
    else                                  STREAM_PROBE_LAUNCH(true,  true);
    #undef STREAM_PROBE_LAUNCH
}

}  // namespace

int stream_probe_variant_count() { return kVariantCount; }

const StreamProbeVariant& stream_probe_variant(int id) {
    if (id < 0 || id >= kVariantCount)
        throw std::runtime_error("stream_probe_variant: bad id " + std::to_string(id));
    return kVariants[id];
}

void stream_probe(int variant_id, const void* src, size_t nbytes,
                  unsigned* out, hipStream_t stream) {
    if (variant_id < 0 || variant_id >= kVariantCount)
        throw std::runtime_error("stream_probe: bad variant id " +
                                 std::to_string(variant_id));
    if (nbytes == 0 || nbytes % PROBE_ROW_BYTES != 0)
        throw std::runtime_error("stream_probe: nbytes must be a nonzero "
                                 "multiple of " + std::to_string(PROBE_ROW_BYTES));
    if ((reinterpret_cast<uintptr_t>(src) & 15) != 0)
        throw std::runtime_error("stream_probe: src must be 16-byte aligned");

    const StreamProbeVariant& v = kVariants[variant_id];
    const unsigned* s = static_cast<const unsigned*>(src);
    switch (v.ilp * 10 + v.load) {
        case 10: probe_launch<1, 0>(v, s, nbytes, out, stream); break;
        case 11: probe_launch<1, 1>(v, s, nbytes, out, stream); break;
        case 12: probe_launch<1, 2>(v, s, nbytes, out, stream); break;
        case 40: probe_launch<4, 0>(v, s, nbytes, out, stream); break;
        case 41: probe_launch<4, 1>(v, s, nbytes, out, stream); break;
        case 42: probe_launch<4, 2>(v, s, nbytes, out, stream); break;
        case 80: probe_launch<8, 0>(v, s, nbytes, out, stream); break;
        case 81: probe_launch<8, 1>(v, s, nbytes, out, stream); break;
        case 82: probe_launch<8, 2>(v, s, nbytes, out, stream); break;
        default:
            throw std::runtime_error("stream_probe: unhandled ilp/load combo for '" +
                                     std::string(v.name) + "'");
    }
}

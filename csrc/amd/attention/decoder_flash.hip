// ================================================================
// FlashRT AMD — decoder cross-attention, split-KV flash (CDNA wave64)
// v2: thread-parallel dot products, zero cross-lane ops in hot loops.
// v3: fused single kernel — no workspace, no reduce launch (below).
//
// ── v3 (default path, Skv <= 2048): one kernel per (h, q) ──
// Iteration-2 measurement (rocprof, real captured pi05 graph):
// partial<32> 21.4us + reduce<32> 12.4us = 33.8us/call. The reduce
// kernel spends 12us on ~0.5us of real work — fp32 workspace
// round-trip (2.5 MiB write + read) plus an 80-WG launch tail. v3
// deletes both: grid (Hq, Sq) = 80 WGs, 256 threads, each WG owns one
// (head, query) row end-to-end. K/V have ONE head (GQA Hq:1), so all
// 80 WGs stream the SAME ~850 KB — it lives in L2/LLC after the first
// touch and every later read is a cache hit; per-XCD L2 (4 MB) holds
// it whole. No K/V LDS staging anywhere: LDS staging buys coalescing,
// but here the data is cache-resident, so staging would only add
// barriers and an LDS round-trip.
//   Phase 0: Q(h,q) row -> LDS as fp32 (D floats; every later read is
//     a same-address wave broadcast).
//   Phase A (scores): thread t owns KV rows j = t, t+256, ... (~4 rows
//     at skv 876). Full D-dim dot per row, K read straight from
//     global as 16 B packets (8 bf16) x D/8, q side read as float4
//     pairs from LDS; 8 fp32 accumulators, fixed combine
//     ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)). Lanes sit on different
//     rows -> uncoalesced by design, L2-resident so latency-tolerant
//     (unrolled independent 16 B loads). Scores -> s_lds[j].
//   Phase B (softmax): grid-stride local max per thread (same fixed
//     row map), fixed radix-2 wave tree, 4 wave partials -> LDS, every
//     thread combines the 4 serially (max is order-free). Then exp in
//     place + local sum (ascending j per thread), wave tree, 4
//     partials combined in fixed order (r0+r1)+(r2+r3) -> L, invL.
//     3 barriers total, no workspace.
//   Phase C (PV): thread t owns dim-pair dp = t % (D/2) and j-half
//     t / (D/2) — 128 pairs x 2 halves = 256 threads at D=256. For a
//     fixed j the wave's lanes read CONSECUTIVE dims of V[j]: a
//     contiguous coalesced 4 B/lane burst (it is phase A, not C, that
//     is column-strided). Each thread accumulates its half [0,skv/2)
//     or [skv/2,skv) ascending; high half parks its partial in LDS;
//     low half combines fixed-order (low + high), scales by invL,
//     emits bf16. Direct-global chosen over the 64-row LDS-tile
//     alternative: same global traffic (every WG reads V exactly
//     once), but zero extra barriers (tile version: ~13 tiles x 2
//     barriers + 32 KB LDS round-trip) and the coalescing a tile
//     stage would buy is already there.
//   Optional fused FP8 epilogue (attention_decoder_gqa_fp8out): after
//     the bf16 store, re-expand the ROUNDED bf16 value, multiply by
//     inv = 1/(*d_scale) (device scalar, same per-thread single
//     division as quantize_fp8_static), clamp to +-448, convert
//     __hip_fp8_e4m3 — bit-identical to running quantize_fp8_static
//     on the bf16 O, so the standalone quantize launch (~180/graph)
//     disappears. O bf16 is still written unchanged for parity
//     tooling and bf16 consumers.
//   Occupancy: 80 WGs on 256 CUs = 31% of CUs, 4 waves each = 1
//     wave/SIMD on the busy CUs — latency hiding comes from ILP
//     (unrolled independent loads), not TLP. LDS ~10.3 KB/WG.
//   Determinism: fixed thread<->row and thread<->dim maps, fixed-order
//     serial + radix-2 tree reductions, no atomics. Bit-stable for a
//     given (shape, skv_eff).
//   Wrapper: fused path when Skv <= 2048 and D % 8 == 0 (always true
//     at the pi05 site); v2 split path otherwise, and
//     FRT_ATTN_FORCE_SPLIT=1 forces v2 for A/B benches (read per call,
//     frozen into a graph at capture). partial_ws is accepted and
//     ignored by the fused path. Signatures unchanged.
//
// ── v2 (fallback path) — original notes below ──
//
// One fixed production site: tiny-M GQA cross-attention
//     Q (Sq, Hq, D) bf16   — Sq ~ 10 query tokens, Hq query heads
//     K (Skv, D)    bf16   — single KV head (GQA Hq:1), Skv ~ 590-876
//     V (Skv, D)    bf16
//     O (Sq, Hq, D) bf16   — bidirectional, no mask
//
// Why v1 was slow (76 us in-graph vs 49 us sdpa): v1 computed each
// score with a wave_reduce_sum — per workgroup ~10 queries x ~26 KV
// rows = ~260 serial 6-step shuffle trees, each a latency chain the
// wave sits through. The site is ~34 MFLOP / ~850 KB — pure
// structure/overhead, so shuffle latency dominated everything.
//
// v2 restructure — every hot phase is one THREAD per output value,
// dots run serially in registers from LDS, no shuffles at all in
// kernel 1:
//   Phase 0 (stage):   Q (Sq x D, fp32) and this split's K tile
//                      (chunk x D, bf16 pairs) into LDS, coalesced.
//                      K/V staging batches 4 independent global loads
//                      per thread before storing (ILP=4 — measured
//                      gfx950 streaming recipe; pure copy, bytes and
//                      mapping unchanged: attn_stage_tile_ilp4).
//   Phase A (scores):  one thread per (query, kv-row) pair —
//                      Sq*chunk pairs over 256 threads (chunk=26 at
//                      the site -> 260 pairs, ~1 loop iteration).
//                      Each thread runs the full D-dim dot serially
//                      from LDS into 4 fp32 accumulators (fixed
//                      combine order), writes one score to LDS.
//   Phase A'(V stage): K tile is dead after A; the same LDS buffer
//                      is refilled with the split's V tile.
//   Phase B (softmax): one thread per query — serial ascending-j
//                      max / exp / sum over <= chunk scores in LDS
//                      (<= ~3*64 LDS ops on <= 16 threads; cheaper
//                      than extra barriers + reduction round-trips).
//   Phase C (PV):      one thread per (query, dim-pair) — Sq*D/2
//                      units (1280 at site) over 256 threads = 5
//                      serial rows each, accumulating over chunk V
//                      rows from LDS, ascending j.
//
// Split-KV decomposition (unchanged from v1):
//   kernel 1 (partial): grid (NSPLIT, Hq). Workgroup (i, h) owns KV
//     rows [i*chunk, min((i+1)*chunk, skv)), chunk = ceil(skv/NSPLIT),
//     and emits an online-softmax partial for ALL Sq queries: m[q],
//     l[q], unnormalized acc[q][d] -> fp32 workspace.
//   kernel 2 (reduce): grid (Sq, Hq) — combines the NSPLIT partials
//     with the standard flash rescale and emits bf16 O. v2 change:
//     the per-split m/l loads run on a full wave (lane i <-> split i)
//     with fixed radix-2 shuffle trees, instead of v1's thread-0
//     serial loop; the acc combine stays one thread per output dim,
//     with the NSPLIT split loads batched 4-wide (ILP4) ahead of the
//     ascending-split accumulate (sum order unchanged — see the
//     in-kernel comment). Env FRT_ATTN_REDUCE_ALT=1 swaps in a
//     512-thread alternate (2 (q,h) pairs per WG, grid halved to
//     ceil(Sq*Hq/2)) with bit-identical outputs — read per call on
//     the host, frozen into a graph at capture time.
//     Optional fused FP8 epilogue (same contract as the v3 fused
//     kernel's): when O_fp8 != nullptr, after the bf16 O store each
//     thread re-expands its ROUNDED bf16 value, multiplies by
//     inv = 1/(*d_scale), pre-clamps to +-448 and converts
//     __hip_fp8_e4m3 — byte-identical to a standalone
//     quantize_fp8_static launch on the bf16 O with the same device
//     scale. This makes attention_decoder_gqa_fp8out fully fused on
//     BOTH wrapper paths (the production v2 split path included),
//     deleting the ~180 standalone quantize launches per inference.
//   Two kernels on one stream: plain stream ordering, no atomics, no
//   cooperative counters — trivially HIP-graph capturable. A fused
//   single-kernel variant would need a last-WG-done flag (atomics —
//   forbidden for determinism); two in-graph launches are ~2 x
//   ~2.5 us and fit the 10 us budget.
//
// NSPLIT selection (compile-time template, instantiated 8/16/32):
//   Kernel-1 latency scales with chunk (score dots per thread =
//   ceil(Sq*chunk/256); PV inner trip = chunk), while the reduce
//   kernel's cost in NSPLIT is a wave-parallel handful of loads —
//   so smaller chunks win until launch/reduce overhead bites.
//   Host heuristic keeps chunk ~<= 32:
//       skv <= 256 -> NSPLIT 8;  skv <= 512 -> 16;  else -> 32.
//   Capacity: skv <= NSPLIT * CHUNK_MAX(64), i.e. 512 / 1024 / 2048;
//   the guard falls back to NSPLIT=32 when the pick can't hold Skv.
//   Env FRT_ATTN_NSPLIT=8|16|32 overrides (read per call on the
//   host, so a bench can sweep variants; inside a captured graph the
//   choice is frozen at capture time — flipping the env later does
//   NOT re-route an existing graph). In fixed/seqused mode the pick
//   uses the host Skv (the padded capacity bound, >= seqused[0]).
//
// Occupancy (MI350X, 256 CUs = 8 XCDs x 32):
//   kernel 1 WGs = NSPLIT x Hq(8): 256 WGs at NSPLIT=32 -> 1 WG/CU,
//   every CU busy, per-WG serial work minimal (chunk 26 at skv 826:
//   score = 1-2 dots x 128 packed iters/thread; PV = 5 x 26 FMAs).
//   NSPLIT=16 -> 128 WGs on 256 CUs: half the CUs idle but each WG
//   is still latency- not throughput-bound, so it's competitive only
//   when skv is small enough that chunk stays ~<= 32; the heuristic
//   encodes exactly that. NSPLIT=8 -> 64 WGs, only for tiny skv.
//   kernel 2 = Sq x Hq = 80 tiny WGs (D threads live) — launch bound
//   by design. LDS 54 KB/WG caps residency at 1 WG/CU on a 64 KB
//   budget (2 on gfx950's 160 KB) — irrelevant: the grid itself is
//   <= 1 WG/CU.
//
// LDS layout (kernel 1, ~54 KB static) and bank reasoning
// (CDNA banks are 4 B; row pads below keep row strides odd in
// 4 B units, so distinct rows -> distinct banks for any power-of-2
// bank count):
//   kv_lds [CHUNK_MAX][D/2 + 1] bf16-pairs (4 B units), pitch 129:
//     holds K for phase A, then V for phase C.
//     Phase A: consecutive threads = consecutive kv rows (pair index
//     p = q*chunk + r) reading element i -> bank (r*129 + i) % NB =
//     (r + i) % NB: distinct rows hit distinct banks; threads from
//     different q-groups reading the SAME row read the same address
//     -> hardware broadcast, no serialization. Rows r and r+NB
//     collide 2-way at chunk > NB (chunk 52 at NSPLIT=16) — accepted.
//     Phase C: consecutive threads = consecutive dim-pairs of one
//     query reading v[j*129 + dp] -> consecutive banks, conflict-free.
//     The matching global K/V loads are linear 4 B/lane bursts
//     (rows are contiguous), fully coalesced.
//   q_lds [SQ_MAX][D + 2] fp32, pitch 258 (even, float2-aligned):
//     Phase A reads q_lds[q*258 + 2i] as float2; lanes in the same
//     q-group read the same address (broadcast); distinct q differ
//     by 258*dq -> bank pair offset 2*dq % NB, distinct for q < 16.
//   s_lds  [SQ_MAX][CHUNK_MAX + 1] fp32, pitch 65 (odd):
//     Phase A writes one score/thread (consecutive r -> consecutive
//     banks); phase C reads s[q*65 + j] at one address per q-group
//     (broadcast).
//
// fp32 math order (deterministic, fixed for a given shape + NSPLIT):
//   - scores: single thread, serial ascending d in 4 accumulators
//     (d%4 lanes), combined (a0+a1)+(a2+a3); scale applied once.
//   - softmax: serial ascending j for max, exp (__expf, rel err
//     ~2^-22, far below bf16 input noise), and sum — one thread per q.
//   - acc: serial ascending j per (q, dim-pair), fp32 FMA.
//   - reduce: fixed radix-2 wave trees over a fixed lane<->split map
//     (max is order-free; the l-sum tree order is fixed), then serial
//     ascending split index for the acc combine. No atomics anywhere.
//
// Workspace (partial_ws, caller-provided, NO allocation here), sized
// for NSPLIT_MAX = 32 regardless of the runtime pick:
//   32 * Hq * Sq * (D + 2) floats  (pi05 site: 32*8*10*258*4
//   = 2,641,920 B ~ 2.52 MiB). Per-(split,head) block layout:
//   [Sq*D unnormalized acc][Sq m][Sq l]. Never needs zeroing:
//   empty tail slices mark l = 0 and the reducer skips them.
//
// Sequence length modes (matches qkv_split_rope_devpos):
//   exact mode:  seqused == nullptr, Skv host int is the length.
//   fixed mode:  seqused != nullptr, kernel reads seqused[0] on
//     device; host Skv is only the capacity bound for the NSPLIT
//     pick. Rows >= seqused[0] are NEVER touched (all loops bound by
//     the device value), so a fixed-shape graph can run any valid
//     prefix length and padding rows may hold garbage/NaN.
//
// Limits (fixed-site kernel; wrapper documents, nothing asserts):
//   Sq <= 16, D <= 256, D % 4 == 0, skv <= NSPLIT*64 (in-kernel
//   clamped), single KV head (GQA Hq:1).
// ================================================================

#include "../kernels/common_hip.h"

#include <hip/hip_fp8.h>

#include <cfloat>
#include <cstdlib>

#define ATTN_NSPLIT_MAX 32   // workspace sizing bound (test allocates for 32)
#define ATTN_NTHREADS   256  // 4 waves
#define ATTN_SQ_MAX     16   // LDS sizing bound for Sq
#define ATTN_D_MAX      256  // LDS sizing bound for D
#define ATTN_CHUNK_MAX  64   // max KV rows per split -> skv <= NSPLIT*64
#define ATTN_KV_PITCH   (ATTN_D_MAX / 2 + 1)   // 129 bf16-pairs: odd stride
#define ATTN_Q_PITCH    (ATTN_D_MAX + 2)       // 258 floats: even, f2-aligned
#define ATTN_S_PITCH    (ATTN_CHUNK_MAX + 1)   // 65 floats: odd stride

#define ATTN_FUSED_SKV_MAX 2048  // fused-path score buffer bound (8 KB LDS)

// ── ILP=4 GLOBAL->LDS K/V tile stage (v2 split path) ──
// stream_probe (gfx950, stream_craft_gfx950.md): one outstanding load
// per lane is latency-window bound (~1373 GB/s @4MB footprints); four
// INDEPENDENT loads issued before any consumer quadruple the bytes in
// flight (co_x4_i4 winners). Here the per-thread stage loop was a
// serial 4 B load chain — batch four grid-stride iterations so the
// four global loads issue back-to-back before the LDS stores. Pure
// copy: element mapping, staged bytes, and barrier placement are
// identical to the plain loop, so kernel math is bit-unchanged.
// src must be pre-offset to the slice (K/V base + start*dh, bf16 pairs);
// the global side is then linear in idx; the LDS side keeps the
// ATTN_KV_PITCH row pitch.
__device__ __forceinline__ void attn_stage_tile_ilp4(
    __hip_bfloat162* __restrict__ dst,          // kv_lds
    const __hip_bfloat162* __restrict__ src,    // slice base (row `start`)
    int cn, int dh, int tid) {
    const int total = cn * dh;
    int idx = tid;
    for (; idx + 3 * ATTN_NTHREADS < total; idx += 4 * ATTN_NTHREADS) {
        const int i0 = idx;
        const int i1 = idx + ATTN_NTHREADS;
        const int i2 = idx + 2 * ATTN_NTHREADS;
        const int i3 = idx + 3 * ATTN_NTHREADS;
        // four independent global loads, distinct destination registers
        const __hip_bfloat162 v0 = src[i0];
        const __hip_bfloat162 v1 = src[i1];
        const __hip_bfloat162 v2 = src[i2];
        const __hip_bfloat162 v3 = src[i3];
        const int r0 = i0 / dh, e0 = i0 - r0 * dh;
        const int r1 = i1 / dh, e1 = i1 - r1 * dh;
        const int r2 = i2 / dh, e2 = i2 - r2 * dh;
        const int r3 = i3 / dh, e3 = i3 - r3 * dh;
        dst[r0 * ATTN_KV_PITCH + e0] = v0;
        dst[r1 * ATTN_KV_PITCH + e1] = v1;
        dst[r2 * ATTN_KV_PITCH + e2] = v2;
        dst[r3 * ATTN_KV_PITCH + e3] = v3;
    }
    for (; idx < total; idx += ATTN_NTHREADS) {
        const int r = idx / dh, e = idx - r * dh;
        dst[r * ATTN_KV_PITCH + e] = src[idx];
    }
}

// ── Kernel 1: per-slice flash partials ──
// grid (NSPLIT, Hq), block ATTN_NTHREADS.
template <int NSPLIT>
__global__ __launch_bounds__(ATTN_NTHREADS)
void attention_decoder_gqa_partial_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const __hip_bfloat16* __restrict__ K,
    const __hip_bfloat16* __restrict__ V,
    float* __restrict__ ws,
    const int* __restrict__ seqused,
    int Sq, int Skv, int Hq, int D, float scale) {
    const int split = blockIdx.x;
    const int h     = blockIdx.y;
    const int tid   = threadIdx.x;

    // K tile for phase A, then (reused) V tile for phase C.
    alignas(16) __shared__ __hip_bfloat162 kv_lds[ATTN_CHUNK_MAX * ATTN_KV_PITCH];
    alignas(16) __shared__ float q_lds[ATTN_SQ_MAX * ATTN_Q_PITCH];
    alignas(16) __shared__ float s_lds[ATTN_SQ_MAX * ATTN_S_PITCH];

    // Effective sequence length: device seqused[0] wins (fixed-shape
    // graph mode); otherwise the host int (exact mode).
    int skv_eff = (seqused != nullptr) ? seqused[0] : Skv;
    if (skv_eff > NSPLIT * ATTN_CHUNK_MAX)
        skv_eff = NSPLIT * ATTN_CHUNK_MAX;  // documented hard cap

    const int chunk = (skv_eff + NSPLIT - 1) / NSPLIT;
    const int start = split * chunk;
    int end = start + chunk;
    if (end > skv_eff) end = skv_eff;
    const int cn = end - start;  // rows in this slice (may be <= 0)
    const int dh = D >> 1;       // bf16-pairs per row

    // Workspace block for (split, head): [Sq*D acc][Sq m][Sq l]
    const size_t base = (size_t)(split * Hq + h) * (size_t)Sq * (size_t)(D + 2);
    float* acc_ws = ws + base;
    float* m_ws   = ws + base + (size_t)Sq * D;
    float* l_ws   = m_ws + Sq;

    if (cn <= 0) {
        // Empty tail slice: l = 0 tells the reducer to skip this block
        // (its acc region is never read), so ws needs no pre-zeroing.
        if (tid < Sq) { m_ws[tid] = -FLT_MAX; l_ws[tid] = 0.0f; }
        return;
    }

    // ── Phase 0: stage Q (fp32) and the K tile (bf16 pairs) ──
    for (int idx = tid; idx < Sq * D; idx += ATTN_NTHREADS) {
        const int q = idx / D;
        const int d = idx - q * D;
        q_lds[q * ATTN_Q_PITCH + d] =
            __bfloat162float(Q[((size_t)q * Hq + h) * D + d]);
    }
    const __hip_bfloat162* K2 = reinterpret_cast<const __hip_bfloat162*>(K);
    attn_stage_tile_ilp4(kv_lds, K2 + (size_t)start * dh, cn, dh, tid);
    __syncthreads();

    // ── Phase A: one thread per (query, kv-row) pair ──
    // Full D-dim dot serially from LDS; 4 accumulators break the FMA
    // dependency chain to ~D/4; fixed combine order (a0+a1)+(a2+a3).
    const int npairs = Sq * cn;
    for (int p = tid; p < npairs; p += ATTN_NTHREADS) {
        const int q = p / cn;
        const int r = p - q * cn;
        const __hip_bfloat162* kr = &kv_lds[r * ATTN_KV_PITCH];
        const float2* qr =
            reinterpret_cast<const float2*>(&q_lds[q * ATTN_Q_PITCH]);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        if ((dh & 7) == 0) {
            // 8-deep explicit load batching (LDS latency hiding at the
            // kernel's 1-wave/SIMD occupancy). Per-chain accumulation
            // order matches the plain loop — bit-exact.
            for (int i = 0; i < dh; i += 8) {
                __hip_bfloat162 kk[8];
                float2 qq[8];
                #pragma unroll
                for (int t = 0; t < 8; ++t) { kk[t] = kr[i + t]; qq[t] = qr[i + t]; }
                #pragma unroll
                for (int t = 0; t < 8; t += 2) {
                    a0 = fmaf(__bfloat162float(kk[t].x),     qq[t].x,     a0);
                    a1 = fmaf(__bfloat162float(kk[t].y),     qq[t].y,     a1);
                    a2 = fmaf(__bfloat162float(kk[t + 1].x), qq[t + 1].x, a2);
                    a3 = fmaf(__bfloat162float(kk[t + 1].y), qq[t + 1].y, a3);
                }
            }
        } else {
            #pragma unroll 4
            for (int i = 0; i < dh; i += 2) {  // dh even (D % 4 == 0)
                const __hip_bfloat162 k0 = kr[i];
                const __hip_bfloat162 k1 = kr[i + 1];
                const float2 q0 = qr[i];
                const float2 q1 = qr[i + 1];
                a0 = fmaf(__bfloat162float(k0.x), q0.x, a0);
                a1 = fmaf(__bfloat162float(k0.y), q0.y, a1);
                a2 = fmaf(__bfloat162float(k1.x), q1.x, a2);
                a3 = fmaf(__bfloat162float(k1.y), q1.y, a3);
            }
        }
        s_lds[q * ATTN_S_PITCH + r] = ((a0 + a1) + (a2 + a3)) * scale;
    }
    __syncthreads();  // scores done; K tile is dead from here

    // ── Phase A': refill the tile with V (all threads) ──
    const __hip_bfloat162* V2 = reinterpret_cast<const __hip_bfloat162*>(V);
    attn_stage_tile_ilp4(kv_lds, V2 + (size_t)start * dh, cn, dh, tid);

    // ── Phase B: per-query serial softmax (interleaved with A') ──
    // Serial ascending j: max, then exp in place + sum — fixed order,
    // <= 3*chunk LDS ops on Sq threads; s_lds is untouched by A'.
    // ILP4 softmax: 4 independent chains hide the serial LDS latency
    // (only Sq threads are active here). max is order-free; the exp
    // sum combines its chains in fixed order (l0+l1)+(l2+l3).
    if (tid < Sq) {
        float* sp = &s_lds[tid * ATTN_S_PITCH];
        float m0 = -FLT_MAX, m1 = -FLT_MAX, m2 = -FLT_MAX, m3 = -FLT_MAX;
        int j = 0;
        for (; j + 3 < cn; j += 4) {
            m0 = fmaxf(m0, sp[j]);
            m1 = fmaxf(m1, sp[j + 1]);
            m2 = fmaxf(m2, sp[j + 2]);
            m3 = fmaxf(m3, sp[j + 3]);
        }
        for (; j < cn; ++j) m0 = fmaxf(m0, sp[j]);
        const float m = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
        float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
        j = 0;
        for (; j + 3 < cn; j += 4) {
            const float e0 = __expf(sp[j] - m);
            const float e1 = __expf(sp[j + 1] - m);
            const float e2 = __expf(sp[j + 2] - m);
            const float e3 = __expf(sp[j + 3] - m);
            sp[j] = e0; sp[j + 1] = e1; sp[j + 2] = e2; sp[j + 3] = e3;
            l0 += e0; l1 += e1; l2 += e2; l3 += e3;
        }
        for (; j < cn; ++j) {
            const float e = __expf(sp[j] - m);
            sp[j] = e;
            l0 += e;
        }
        m_ws[tid] = m;
        l_ws[tid] = (l0 + l1) + (l2 + l3);
    }
    __syncthreads();  // V staged + probabilities final

    // ── Phase C: one thread per (query, dim-pair) ──
    // acc[q][2dp..2dp+1] = sum_j p[q][j] * V[j][2dp..2dp+1], ascending j.
    // ILP8 PV: 8 independent accumulator chains hide the serial LDS
    // latency; chains combine in a fixed pairwise order.
    const int nunits = Sq * dh;
    for (int u = tid; u < nunits; u += ATTN_NTHREADS) {
        const int q  = u / dh;
        const int dp = u - q * dh;
        const float* sp = &s_lds[q * ATTN_S_PITCH];
        const __hip_bfloat162* vp = &kv_lds[dp];
        float axs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float ays[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int j = 0;
        for (; j + 7 < cn; j += 8) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                const __hip_bfloat162 v = vp[(j + t) * ATTN_KV_PITCH];
                const float pj = sp[j + t];
                axs[t] = fmaf(pj, __bfloat162float(v.x), axs[t]);
                ays[t] = fmaf(pj, __bfloat162float(v.y), ays[t]);
            }
        }
        for (; j < cn; ++j) {
            const __hip_bfloat162 v = vp[j * ATTN_KV_PITCH];
            const float pj = sp[j];
            axs[0] = fmaf(pj, __bfloat162float(v.x), axs[0]);
            ays[0] = fmaf(pj, __bfloat162float(v.y), ays[0]);
        }
        const float ax = ((axs[0] + axs[1]) + (axs[2] + axs[3]))
                       + ((axs[4] + axs[5]) + (axs[6] + axs[7]));
        const float ay = ((ays[0] + ays[1]) + (ays[2] + ays[3]))
                       + ((ays[4] + ays[5]) + (ays[6] + ays[7]));
        float* out = acc_ws + (size_t)q * D + 2 * dp;
        out[0] = ax;   // adjacent 4 B stores; consecutive lanes are
        out[1] = ay;   // consecutive pairs -> coalesced 8 B/lane
    }
}

// ── Kernel 2: combine the NSPLIT partials per (q, h) ──
// grid (Sq, Hq), block ATTN_NTHREADS (thread t = output dim t).
// Standard flash rescale: gm = max_i m_i, coeff_i = exp(m_i - gm),
// O = (sum_i coeff_i * acc_i) / (sum_i coeff_i * l_i).
// v2: wave 0 loads all m/l in parallel (lane i <-> split i) and
// reduces with fixed shuffle trees; v1 looped serially on thread 0.
// Optional FP8 epilogue: O_fp8/d_scale as in the fused kernel —
// nullptr O_fp8 disables it (plain path, zero extra work).
template <int NSPLIT>
__global__ __launch_bounds__(ATTN_NTHREADS)
void attention_decoder_gqa_reduce_kernel(
    const float* __restrict__ ws,
    __hip_bfloat16* __restrict__ O,
    __hip_fp8_e4m3* __restrict__ O_fp8,  // optional epilogue (nullptr = off)
    const float* __restrict__ d_scale,   // device scale, read iff O_fp8
    int Sq, int Hq, int D) {
    const int q    = blockIdx.x;
    const int h    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid & (FVK_WAVE_SIZE - 1);
    const int wave = tid >> 6;

    __shared__ float coeff[NSPLIT];
    __shared__ float s_invL;

    const size_t blk = (size_t)Sq * (size_t)(D + 2);  // floats per (split,head)
    const size_t ml_off = (size_t)Sq * D;

    static_assert(NSPLIT <= FVK_WAVE_SIZE, "one wave covers all splits");
    if (wave == 0) {
        float mi = -FLT_MAX, li = 0.0f;
        if (lane < NSPLIT) {
            const size_t b = (size_t)(lane * Hq + h) * blk;
            li = ws[b + ml_off + Sq + q];
            mi = ws[b + ml_off + q];
        }
        // Global max: order-free; empty slices (li == 0) excluded.
        float gm = wave_reduce_max((li > 0.0f) ? mi : -FLT_MAX);
        gm = __shfl(gm, 0);  // tree result lives in lane 0 — broadcast
        const float c = (lane < NSPLIT && li > 0.0f) ? __expf(mi - gm) : 0.0f;
        if (lane < NSPLIT) coeff[lane] = c;
        // L: fixed radix-2 tree over the fixed lane<->split map.
        const float L = wave_reduce_sum(c * li);
        if (lane == 0) s_invL = (L > 0.0f) ? 1.0f / L : 0.0f;  // skv>=1 => L>0
    }
    __syncthreads();

    if (tid < D) {
        // ILP4 acc combine (stream_probe: one outstanding 4 B load per
        // lane is latency-window bound; four INDEPENDENT loads issued
        // before any consumer quadruple the bytes in flight). The old
        // loop's guarded load-then-FMA compiled as one serial chain of
        // NSPLIT latency windows per thread. Batch FOUR split loads
        // into registers first, then accumulate — the ADDS stay in
        // ascending split order, so the fixed deterministic fp32 sum
        // order documented in the header is preserved; only the LOADS
        // are hoisted. Loads are now unconditional: the workspace is
        // allocated for all NSPLIT splits (never-written empty-tail acc
        // may hold garbage/NaN), and the c == 0 SELECT on the
        // accumulate still excludes them — select, not multiply, since
        // 0 * NaN would poison the sum.
        static_assert(NSPLIT % 4 == 0, "ILP4 combine assumes 4 | NSPLIT");
        const size_t sstride = (size_t)Hq * blk;  // floats, split i -> i+1
        const float* ap = ws + (size_t)h * blk + (size_t)q * D + tid;
        // Full-depth prefetch: all NSPLIT split loads issue before any
        // consumer (NSPLIT <= 32 registers; occupancy is not the
        // constraint here). The accumulate keeps the ascending-split
        // fixed order — bit-identical to the 4-batch version.
        float vv[NSPLIT];
        #pragma unroll
        for (int i = 0; i < NSPLIT; ++i)
            vv[i] = ap[(size_t)i * sstride];
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < NSPLIT; ++i) {
            const float c = coeff[i];
            s = (c != 0.0f) ? fmaf(c, vv[i], s) : s;
        }
        const size_t o_off = ((size_t)q * Hq + h) * D + tid;
        const __hip_bfloat16 b = __float2bfloat16(s * s_invL);
        O[o_off] = b;
        if (O_fp8 != nullptr) {
            // Bit-exact quantize_fp8_static semantics: quantize the
            // ROUNDED bf16 value, per-thread inv = 1/(*d_scale)
            // (single division), pre-clamp +-448, e4m3 convert.
            const float inv = 1.0f / (*d_scale);
            const float f =
                fminf(fmaxf(__bfloat162float(b) * inv, -448.0f), 448.0f);
            O_fp8[o_off] = __hip_fp8_e4m3(f);
        }
    }
}

// ── Optional reduce alternate (env FRT_ATTN_REDUCE_ALT=1) ──
// Same per-(q, h) math as attention_decoder_gqa_reduce_kernel, but 512
// threads per WG covering TWO (q, h) pairs (half 0 = waves 0-3, half 1
// = waves 4-7), so the grid halves from Sq*Hq tiny WGs to
// ceil(Sq*Hq/2) — amortizing launch/WG setup, the dominant cost of a
// kernel whose real work is ~0.5 us. Register pressure is tiny here
// (unlike the partial kernel's w8 spill precedent), so doubling the
// block is free. Pair p = q*Hq + h maps halves to ADJACENT output
// D-blocks -> one WG stores 2*D contiguous bf16. The load/accumulate
// code per pair is identical to the default kernel (same ILP4 batching,
// same ascending-split fixed-order sum, same fp8 epilogue), so outputs
// are bit-identical. Guarded per-phase for odd Sq*Hq (idle half still
// reaches __syncthreads).
template <int NSPLIT>
__global__ __launch_bounds__(2 * ATTN_NTHREADS)
void attention_decoder_gqa_reduce_alt_kernel(
    const float* __restrict__ ws,
    __hip_bfloat16* __restrict__ O,
    __hip_fp8_e4m3* __restrict__ O_fp8,  // optional epilogue (nullptr = off)
    const float* __restrict__ d_scale,   // device scale, read iff O_fp8
    int Sq, int Hq, int D) {
    const int hf   = (int)(threadIdx.x >> 8);            // half 0 / 1
    const int tid  = (int)(threadIdx.x & (ATTN_NTHREADS - 1));
    const int lane = tid & (FVK_WAVE_SIZE - 1);
    const int wave = tid >> 6;                           // wave within half
    const int np   = Sq * Hq;
    const int p    = (int)blockIdx.x * 2 + hf;           // pair = q*Hq + h
    const int q    = p / Hq;
    const int h    = p - q * Hq;

    __shared__ float coeff[2][NSPLIT];
    __shared__ float s_invL[2];

    const size_t blk = (size_t)Sq * (size_t)(D + 2);  // floats per (split,head)
    const size_t ml_off = (size_t)Sq * D;

    static_assert(NSPLIT <= FVK_WAVE_SIZE, "one wave covers all splits");
    if (wave == 0 && p < np) {  // wave 0 of each half owns its pair's m/l
        float mi = -FLT_MAX, li = 0.0f;
        if (lane < NSPLIT) {
            const size_t b = (size_t)(lane * Hq + h) * blk;
            li = ws[b + ml_off + Sq + q];
            mi = ws[b + ml_off + q];
        }
        float gm = wave_reduce_max((li > 0.0f) ? mi : -FLT_MAX);
        gm = __shfl(gm, 0);
        const float c = (lane < NSPLIT && li > 0.0f) ? __expf(mi - gm) : 0.0f;
        if (lane < NSPLIT) coeff[hf][lane] = c;
        const float L = wave_reduce_sum(c * li);
        if (lane == 0) s_invL[hf] = (L > 0.0f) ? 1.0f / L : 0.0f;
    }
    __syncthreads();

    if (tid < D && p < np) {
        static_assert(NSPLIT % 4 == 0, "ILP4 combine assumes 4 | NSPLIT");
        const size_t sstride = (size_t)Hq * blk;  // floats, split i -> i+1
        const float* ap = ws + (size_t)h * blk + (size_t)q * D + tid;
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < NSPLIT; i += 4) {
            const float v0 = ap[(i + 0) * sstride];
            const float v1 = ap[(i + 1) * sstride];
            const float v2 = ap[(i + 2) * sstride];
            const float v3 = ap[(i + 3) * sstride];
            const float c0 = coeff[hf][i + 0], c1 = coeff[hf][i + 1];
            const float c2 = coeff[hf][i + 2], c3 = coeff[hf][i + 3];
            s = (c0 != 0.0f) ? fmaf(c0, v0, s) : s;
            s = (c1 != 0.0f) ? fmaf(c1, v1, s) : s;
            s = (c2 != 0.0f) ? fmaf(c2, v2, s) : s;
            s = (c3 != 0.0f) ? fmaf(c3, v3, s) : s;
        }
        const size_t o_off = ((size_t)q * Hq + h) * D + tid;
        const __hip_bfloat16 b = __float2bfloat16(s * s_invL[hf]);
        O[o_off] = b;
        if (O_fp8 != nullptr) {
            const float inv = 1.0f / (*d_scale);
            const float f =
                fminf(fmaxf(__bfloat162float(b) * inv, -448.0f), 448.0f);
            O_fp8[o_off] = __hip_fp8_e4m3(f);
        }
    }
}

template <int NSPLIT>
static void launch_decoder_gqa(const __hip_bfloat16* Q,
                               const __hip_bfloat16* K,
                               const __hip_bfloat16* V,
                               __hip_bfloat16* O,
                               __hip_fp8_e4m3* O_fp8,
                               const float* d_scale,
                               float* partial_ws,
                               int Sq, int Skv, int Hq, int D,
                               const int* seqused,
                               float softmax_scale,
                               hipStream_t stream) {
    dim3 grid1(NSPLIT, Hq);
    attention_decoder_gqa_partial_kernel<NSPLIT><<<grid1, ATTN_NTHREADS, 0, stream>>>(Q, K, V, partial_ws, seqused, Sq, Skv, Hq, D, softmax_scale);
    // FRT_ATTN_REDUCE_ALT=1 -> w8 alternate (2 pairs/WG, half the
    // grid). Read per call on the host, like FRT_ATTN_NSPLIT: frozen
    // into a captured graph at capture time, env flips later do NOT
    // re-route an existing graph.
    bool alt = false;
    if (const char* e = std::getenv("FRT_ATTN_REDUCE_ALT")) alt = std::atoi(e) != 0;
    if (alt) {
        dim3 grid2((Sq * Hq + 1) / 2);
        attention_decoder_gqa_reduce_alt_kernel<NSPLIT><<<grid2, 2 * ATTN_NTHREADS, 0, stream>>>(partial_ws, O, O_fp8, d_scale, Sq, Hq, D);
    } else {
        dim3 grid2(Sq, Hq);
        attention_decoder_gqa_reduce_kernel<NSPLIT><<<grid2, ATTN_NTHREADS, 0, stream>>>(partial_ws, O, O_fp8, d_scale, Sq, Hq, D);
    }
}

// ================================================================
// v3 fused single kernel — see the v3 section of the file header.
// grid (Hq, Sq), block 256 (4 waves). One WG owns O(h, q, :) whole:
// scores -> softmax -> PV, all in one launch, no workspace.
// ================================================================

// 16 B global-load packet: 8 bf16 (one dwordx4). Rows are 16 B
// aligned whenever D % 8 == 0 and the tensor base is 16 B aligned.
struct alignas(16) attn_bf16x8 {
    __hip_bfloat162 p0, p1, p2, p3;
};

__global__ __launch_bounds__(ATTN_NTHREADS)
void attention_decoder_gqa_fused_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const __hip_bfloat16* __restrict__ K,
    const __hip_bfloat16* __restrict__ V,
    __hip_bfloat16* __restrict__ O,
    __hip_fp8_e4m3* __restrict__ O_fp8,  // optional epilogue (nullptr = off)
    const float* __restrict__ d_scale,   // device scale, read iff O_fp8
    const int* __restrict__ seqused,
    int Skv, int Hq, int D, float scale) {
    static_assert(ATTN_NTHREADS == 256, "combine trees hardcode 4 waves");
    const int h    = blockIdx.x;
    const int q    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int lane = tid & (FVK_WAVE_SIZE - 1);
    const int wave = tid >> 6;

    alignas(16) __shared__ float q_lds[ATTN_D_MAX];        // fp32 query row
    alignas(16) __shared__ float s_lds[ATTN_FUSED_SKV_MAX];// scores -> probs
    __shared__ float red_lds[8];                           // [0..3] max, [4..7] sum
    __shared__ float c_lds[ATTN_D_MAX];                    // phase-C high-half partials

    int skv = (seqused != nullptr) ? seqused[0] : Skv;
    if (skv > ATTN_FUSED_SKV_MAX) skv = ATTN_FUSED_SKV_MAX;  // documented cap

    // ── Phase 0: stage Q(h, q, :) as fp32 (later reads broadcast) ──
    if (tid < D)
        q_lds[tid] = __bfloat162float(Q[((size_t)q * Hq + h) * D + tid]);
    __syncthreads();

    // ── Phase A: scores — thread t owns rows j = t, t+256, ... ──
    // K straight from global (L2-resident, shared by all 80 WGs);
    // uncoalesced across lanes by design, hidden by unrolled
    // independent 16 B loads. Fixed 8-acc combine order.
    const int d8 = D >> 3;  // 16 B packets per row
    for (int j = tid; j < skv; j += ATTN_NTHREADS) {
        const attn_bf16x8* kr =
            reinterpret_cast<const attn_bf16x8*>(K + (size_t)j * D);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < d8; ++i) {
            const attn_bf16x8 kb = kr[i];
            const float4* qp =
                reinterpret_cast<const float4*>(&q_lds[i << 3]);
            const float4 qa = qp[0];
            const float4 qb = qp[1];
            a0 = fmaf(__bfloat162float(kb.p0.x), qa.x, a0);
            a1 = fmaf(__bfloat162float(kb.p0.y), qa.y, a1);
            a2 = fmaf(__bfloat162float(kb.p1.x), qa.z, a2);
            a3 = fmaf(__bfloat162float(kb.p1.y), qa.w, a3);
            a4 = fmaf(__bfloat162float(kb.p2.x), qb.x, a4);
            a5 = fmaf(__bfloat162float(kb.p2.y), qb.y, a5);
            a6 = fmaf(__bfloat162float(kb.p3.x), qb.z, a6);
            a7 = fmaf(__bfloat162float(kb.p3.y), qb.w, a7);
        }
        s_lds[j] = (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) * scale;
    }
    __syncthreads();

    // ── Phase B1: global max (order-free) ──
    float lm = -FLT_MAX;
    for (int j = tid; j < skv; j += ATTN_NTHREADS) lm = fmaxf(lm, s_lds[j]);
    lm = wave_reduce_max(lm);
    if (lane == 0) red_lds[wave] = lm;
    __syncthreads();
    const float gm =
        fmaxf(fmaxf(red_lds[0], red_lds[1]), fmaxf(red_lds[2], red_lds[3]));

    // ── Phase B2: exp in place + fixed-order sum ──
    // Per-thread serial ascending j over the fixed row map, then the
    // fixed radix-2 wave tree, then (r0+r1)+(r2+r3) — deterministic
    // for a given (shape, skv). Every thread computes L redundantly
    // (identical bits) — saves a broadcast barrier.
    float ls = 0.0f;
    for (int j = tid; j < skv; j += ATTN_NTHREADS) {
        const float e = __expf(s_lds[j] - gm);
        s_lds[j] = e;   // own rows only — no race
        ls += e;
    }
    ls = wave_reduce_sum(ls);
    if (lane == 0) red_lds[4 + wave] = ls;
    __syncthreads();  // probs final + sum partials visible
    const float L = (red_lds[4] + red_lds[5]) + (red_lds[6] + red_lds[7]);
    const float invL = (L > 0.0f) ? 1.0f / L : 0.0f;

    // ── Phase C: PV — thread t owns dim-pair (t % dh) in j-half (t / dh) ──
    // For fixed j the wave reads consecutive dims of V[j]: contiguous
    // coalesced 4 B/lane. s_lds[j] is a same-address broadcast.
    const int dh   = D >> 1;            // dim-pairs; <= 128
    const int half = tid / dh;          // 0 = low j-half, 1 = high; >= 2 idle
    const int dp   = tid - half * dh;
    const int jmid = skv >> 1;
    float ax = 0.0f, ay = 0.0f;
    if (half < 2) {
        const int j0 = half ? jmid : 0;
        const int j1 = half ? skv : jmid;
        const __hip_bfloat162* vp =
            reinterpret_cast<const __hip_bfloat162*>(V) + dp;
        #pragma unroll 4
        for (int j = j0; j < j1; ++j) {
            const __hip_bfloat162 v = vp[(size_t)j * dh];
            const float pj = s_lds[j];
            ax = fmaf(pj, __bfloat162float(v.x), ax);
            ay = fmaf(pj, __bfloat162float(v.y), ay);
        }
    }
    if (half == 1) {          // SoA park: consecutive lanes, consecutive banks
        c_lds[dp] = ax;
        c_lds[dh + dp] = ay;
    }
    __syncthreads();
    if (half == 0) {
        const float sx = (ax + c_lds[dp]) * invL;       // fixed: low + high
        const float sy = (ay + c_lds[dh + dp]) * invL;
        const __hip_bfloat16 bx = __float2bfloat16(sx);
        const __hip_bfloat16 by = __float2bfloat16(sy);
        const size_t o_off = ((size_t)q * Hq + h) * D + 2 * (size_t)dp;
        *reinterpret_cast<__hip_bfloat162*>(O + o_off) =
            __halves2bfloat162(bx, by);
        if (O_fp8 != nullptr) {
            // Bit-exact quantize_fp8_static semantics: quantize the
            // ROUNDED bf16 value, per-thread inv = 1/(*d_scale)
            // (single division), pre-clamp +-448, e4m3 convert.
            const float inv = 1.0f / (*d_scale);
            const float fx =
                fminf(fmaxf(__bfloat162float(bx) * inv, -448.0f), 448.0f);
            const float fy =
                fminf(fmaxf(__bfloat162float(by) * inv, -448.0f), 448.0f);
            O_fp8[o_off]     = __hip_fp8_e4m3(fx);
            O_fp8[o_off + 1] = __hip_fp8_e4m3(fy);
        }
    }
}

// ── Host wrappers ──
// partial_ws: caller-provided device fp32 scratch of at least
//   ATTN_NSPLIT_MAX(32) * Hq * Sq * (D + 2) floats (pi05 decoder
//   site: 32*8*10*258*4 = 2,641,920 bytes) — sized for the largest
//   split variant. The fused path accepts and IGNORES it (kept so the
//   signature and callers are unchanged). No allocation, no memset,
//   no host sync — safe inside HIP graph capture.
// seqused: nullptr => exact mode (Skv host int is the length);
//   else device int32, seqused[0] read in-kernel and the host Skv is
//   only the padded capacity bound (fixed-shape graph mode; padding
//   rows beyond seqused[0] are never read).
// Path pick (host, per call — frozen into a graph at capture time):
//   fused single kernel when Skv <= 2048 and D % 8 == 0 (K/V rows
//   16 B aligned); v2 split otherwise. FRT_ATTN_FORCE_SPLIT=1 forces
//   the v2 path for A/B benches. On the v2 path the NSPLIT heuristic
//   and FRT_ATTN_NSPLIT=8|16|32 behave as before.
// Constraints (fixed-site kernel, not checked): D <= 256, D % 4 == 0
//   (fused additionally wants D % 8 == 0), seq length <= 2048, single
//   KV head (K/V are (Skv, D)). v2 also needs Sq <= 16; the fused
//   path has no Sq cap (one WG per query).

static bool attn_use_fused(int Skv, int D) {
    // Default is the v2 split path: measured in-graph 36.5us vs the fused
    // single kernel's 40.6us at the pi05 site, and the fused kernel's
    // larger per-call L2 footprint degrades neighboring GEMMs in the real
    // captured graph (E2E +2ms). FRT_ATTN_FUSED=1 opts back in for A/B.
    if (const char* e = std::getenv("FRT_ATTN_FUSED"))
        if (std::atoi(e) != 0)
            return Skv <= ATTN_FUSED_SKV_MAX && D <= ATTN_D_MAX && (D % 8) == 0;
    if (const char* e = std::getenv("FRT_ATTN_FORCE_SPLIT"))
        (void)e;  // legacy switch, now the default
    return false;
}

static void launch_decoder_gqa_split(const __hip_bfloat16* Q,
                                     const __hip_bfloat16* K,
                                     const __hip_bfloat16* V,
                                     __hip_bfloat16* O,
                                     __hip_fp8_e4m3* O_fp8,
                                     const float* d_scale,
                                     float* partial_ws,
                                     int Sq, int Skv, int Hq, int D,
                                     const int* seqused,
                                     float softmax_scale,
                                     hipStream_t stream) {
    int ns = 0;
    if (const char* e = std::getenv("FRT_ATTN_NSPLIT")) ns = std::atoi(e);
    if (ns != 8 && ns != 16 && ns != 32)
        ns = (Skv <= 256) ? 8 : (Skv <= 512) ? 16 : 32;
    if (Skv > ns * ATTN_CHUNK_MAX) ns = 32;  // capacity guard
    switch (ns) {
        case 8:
            launch_decoder_gqa<8>(Q, K, V, O, O_fp8, d_scale, partial_ws, Sq, Skv, Hq, D, seqused, softmax_scale, stream);
            break;
        case 16:
            launch_decoder_gqa<16>(Q, K, V, O, O_fp8, d_scale, partial_ws, Sq, Skv, Hq, D, seqused, softmax_scale, stream);
            break;
        default:
            launch_decoder_gqa<32>(Q, K, V, O, O_fp8, d_scale, partial_ws, Sq, Skv, Hq, D, seqused, softmax_scale, stream);
            break;
    }
}

void attention_decoder_gqa(const __hip_bfloat16* Q,
                           const __hip_bfloat16* K,
                           const __hip_bfloat16* V,
                           __hip_bfloat16* O,
                           float* partial_ws,
                           int Sq, int Skv, int Hq, int D,
                           const int* seqused,
                           float softmax_scale,
                           hipStream_t stream) {
    if (attn_use_fused(Skv, D)) {
        dim3 grid(Hq, Sq);
        attention_decoder_gqa_fused_kernel<<<grid, ATTN_NTHREADS, 0, stream>>>(
            Q, K, V, O, /*O_fp8=*/nullptr, /*d_scale=*/nullptr, seqused,
            Skv, Hq, D, softmax_scale);
        return;
    }
    launch_decoder_gqa_split(Q, K, V, O, /*O_fp8=*/nullptr,
                             /*d_scale=*/nullptr, partial_ws,
                             Sq, Skv, Hq, D, seqused, softmax_scale, stream);
}

// Fused-epilogue variant: same attention, plus OCP e4m3 output
// O_fp8 (Sq, Hq*D) row-major — byte-for-byte what quantize_fp8_static
// would produce from the bf16 O with the same device scale. O (bf16)
// is still written, identical to the plain path. BOTH wrapper paths
// are fully fused: the v3 fused kernel carries the epilogue inline,
// and the v2 split path (the production default) carries it in the
// reduce kernel — no standalone quantize launch anywhere.
void attention_decoder_gqa_fp8out(const __hip_bfloat16* Q,
                                  const __hip_bfloat16* K,
                                  const __hip_bfloat16* V,
                                  __hip_bfloat16* O,
                                  __hip_fp8_e4m3* O_fp8,
                                  const float* d_scale,
                                  float* partial_ws,
                                  int Sq, int Skv, int Hq, int D,
                                  const int* seqused,
                                  float softmax_scale,
                                  hipStream_t stream) {
    if (attn_use_fused(Skv, D)) {
        dim3 grid(Hq, Sq);
        attention_decoder_gqa_fused_kernel<<<grid, ATTN_NTHREADS, 0, stream>>>(
            Q, K, V, O, O_fp8, d_scale, seqused, Skv, Hq, D, softmax_scale);
        return;
    }
    launch_decoder_gqa_split(Q, K, V, O, O_fp8, d_scale, partial_ws,
                             Sq, Skv, Hq, D, seqused, softmax_scale, stream);
}

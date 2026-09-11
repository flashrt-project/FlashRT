#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

// ================================================================
// FlashRT AMD — weight-streaming read-bandwidth probe (CDNA4, wave64)
//
// A parameterized family of pure-read kernels that answers one
// question: which combination of in-flight bytes (ILP), waves per
// workgroup (TLP), grid size, load width/hint, and access pattern
// saturates HBM/L2 read bandwidth on this part. The hand-written
// weight-streaming kernels (small-M GEMM, fused FFN, fused decoder
// attention) all share one pattern — 256-thread WG, per-lane serial
// K loop of dwordx4 loads — and all plateau far below the D2D copy
// rate. This probe sweeps each axis of that pattern in isolation so
// the winning recipe is measured, not guessed.
//
// Each variant streams the source buffer exactly once, accumulates a
// u32 checksum per thread (so loads cannot be dead-code-eliminated;
// integer adds are far too cheap to ever bound the loop), reduces it
// within the workgroup, and writes one word per WG to `out`. The sum
// of out[0..grid) equals the u32 sum of the whole buffer (mod 2^32)
// for every variant — a cross-variant correctness check for free.
//
// Axes (see kVariants in stream_probe.hip for the instantiated set):
//   ilp        independent chained loads per thread per loop iteration
//              (1 / 4 / 8) — outer loops are unroll-1 so the compiler
//              cannot silently widen the axis
//   waves      waves per WG: 4 (256T) / 8 (512T) / 16 (1024T)
//   grid       workgroups: 64 / 256 / 1024
//   load       0 = global dwordx4, 1 = dwordx4 nontemporal (streaming
//              hint, bypasses L2 retention), 2 = global dwordx2
//   strided    0 = lane-coalesced streaming; 1 = each lane serially
//              walks its own 1KB row (the small-M GEMM nn pattern:
//              per lane load touches a distinct cache line)
//   persistent 0 = each WG owns one contiguous slab of the buffer
//              (serial walk, like a per-lane K loop at WG scale);
//              1 = grid-stride over the whole buffer
// ================================================================

struct StreamProbeVariant {
    const char* name;
    int ilp;        // independent chain loads per thread per iteration
    int waves;      // waves per WG (blockDim.x = waves * 64)
    int grid;       // workgroups launched
    int load;       // 0 = dwordx4, 1 = dwordx4 nontemporal, 2 = dwordx2
    int strided;    // 0 = lane-coalesced, 1 = 1KB row per lane
    int persistent; // 0 = per-WG contiguous slab, 1 = grid-stride
};

// Number of instantiated variants.
int stream_probe_variant_count();

// Descriptor for variant `id` (0 <= id < stream_probe_variant_count()).
const StreamProbeVariant& stream_probe_variant(int id);

// Run one variant: read `nbytes` from `src`, write one u32 checksum
// word per WG into `out` (must hold >= variant.grid words). Throws
// std::runtime_error on a bad id, nbytes not a multiple of 1024
// (the row size — keeps every variant covering identical bytes), or
// src not 16-byte aligned.
void stream_probe(int variant_id, const void* src, size_t nbytes,
                  unsigned* out, hipStream_t stream);

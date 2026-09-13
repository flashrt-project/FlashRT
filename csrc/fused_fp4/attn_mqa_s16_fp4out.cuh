// ============================================================================
//  FlashRT — MQA attention for a handful of query rows with the NVFP4
//  activation quantize fused into the split merge (sm_80+ mma.sync, fp16).
//
//  Q  : (S, NH*HD) fp16, RoPE already applied      (S <= 16, HD == 256)
//  K,V: (T, HD) fp16, one shared KV head
//  out: packed e2m1 (S, NH*HD/2) + UE4M3 SFA bytes in the CUTLASS Sm1xx
//       layout for an (S, NH*HD) A operand — the same bytes the
//       quantize_fp4_dynamic_sfa_fp16_vec kernel would produce from the fp16
//       attention output (fp16-rounded before quantization).
//  One CTA per (key split, head pair); the last split of a head pair merges
//  the partial (m, l, O) triples and quantizes. Replaces the cuBLAS QK^T /
//  softmax / PV chain plus the quantize launch.
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash_rt {
namespace fp4 {

// Workspace bytes (partials + one counter per head pair); must be zero-filled
// once at allocation (the kernel resets its counters after every use).
size_t attn_mqa_s16_fp4out_ws_bytes(int NH);

int attn_mqa_s16_fp4out(const void* Q, const void* K, const void* V, void* ws,
                        void* packed, void* sfa, int S, int T, int NH, int HD,
                        float attn_scale, cudaStream_t stream, int variant = 0, int dbg = 0);
// variant: 0 = 4 splits/2 stages, 1 = 8/2, 2 = 16/2, 3 = 8/3, 4 = 4/3; dbg bit0 skips the MMAs, bit1 the tile loads (timing probes).

}  // namespace fp4
}  // namespace flash_rt

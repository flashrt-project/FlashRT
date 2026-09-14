// Fused Pi0.5 decoder attention over one shared KV head (fp16, HD=256):
// RoPE on the fresh q/k rows, KV-cache append of the fresh k/v rows, QK^T,
// softmax, PV over the whole cache with the key range split across 16 CTAs,
// an in-kernel merge of the split partials and the NVFP4 quantize of the
// context rows (packed e2m1 + CUTLASS SFA bytes), all in one launch.
// Replaces qkv_split_rope_kvcache_fp16_vec + cuBLAS QK^T + softmax + PV +
// quantize_fp4_dynamic_sfa_fp16_vec on the decoder path.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace flash_rt {
namespace fp4 {

// Workspace bytes (split partials + arrival counter); zero-fill once at allocation.
size_t attn_mqa_fused_ws_bytes();

// qkv: (S, qkv_stride) fp16 rows [q (NH*HD) | k (HD) | v (HD)], un-rotated.
// rope: (S, HD) fp16 (cos, sin) pairs for the S fresh positions.
// Kc/Vc: this layer's cache base, (enc_seq + S, HD) fp16; rows [enc_seq, enc_seq+S) are written here.
// packed/sfa: NVFP4 context rows (S, NH*HD) in the CUTLASS blocked SFA layout.
int attn_mqa_fused_fp4out(const void* qkv, const void* rope, void* Kc, void* Vc, void* ws,
                          void* packed, void* sfa, int S, int enc_seq, int NH, int HD,
                          int qkv_stride, float attn_scale, cudaStream_t stream, int dbg);

}  // namespace fp4
}  // namespace flash_rt

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// ================================================================
// FlashRT AMD — MFMA small-M BF16 packed-weight GEMM (gfx950, wave64)
//
// BF16 sibling of the FP8 packed kernel (smallm_mfma.h): a weight-
// streaming GEMM for the M<=48 "NN" projection sites where the whole
// call is bounded by reading the (K,N) bf16 weight once. Target sites
// are the GROOT N1.7 DiT projections at M=41:
//   (41, 1536, 1536)  q/k/v/o        -> epilogue bias (o: bias_res)
//   (41, 6144, 1536)  ffn up         -> epilogue bias_gelu
//   (41, 1536, 6144)  ffn down       -> epilogue bias_res
//
// Math (all epilogues, fp32 accumulate, bf16 store):
//   acc(M,N)  = A_bf16(M,K) @ W_bf16(K,N) + bias(N)     [fp32]
//   bias      : D = bf16(acc)
//   bias_gelu : D = bf16(gelu_tanh(acc))     (tanh-approx GELU, the
//               hipBLASLt GELU_BIAS / activation.hip semantics)
//   bias_res  : D = bf16(float(D) + acc)     (accumulate into D)
//
// Fragment layout of V_MFMA_F32_16X16X32_BF16 (same mapping the
// encoder flash kernel uses, verified by its parity gate):
//   a (shortx8): lane l supplies Amat[m = l&15][k = 32*step + 8*(l>>4) ..+8]
//   b (shortx8): lane l supplies Bmat[n = l&15][same k window]
//   c (floatx4): lane l element i = C[m = 4*(l>>4)+i][n = l&15]
//
// Packed weight layout — weights are static, so they are repacked
// ONCE at setup into exact per-lane consumption order: for each
// 16-column n-tile the stream is a linear slab of 16-byte chunks,
// chunk index (step*64 + lane) inside the tile, chunk contents
//   W[32*step + 8*(lane>>4) .. +8][n_tile*16 + (lane&15)]
// so a wave reading "chunk = base + s*64 + lane" walks DRAM linearly
// (the stream_probe co-pattern; every lane an independent dwordx4
// chain). From the (K, N) row-major bf16 weight, in torch:
//
//   Wp = W.view(K//32, 4, 8, N//16, 16).permute(3, 0, 1, 4, 2).contiguous()
//
// (dims after permute: (N/16 tile, K/32 step, k-group, n-lane, k-elem);
// flattened 16B-chunk index = tile*(2*K) + step*64 + lane, with
// lane = (k-group << 4) | n-lane.)
//
// Structure: grid.x = N/16 column tiles, WAVES waves per workgroup
// split K into WAVES fixed segments (S = K/(32*WAVES) MFMA steps
// each), consumed in rounds of 6 steps with the next round's weight
// chunks prefetched behind the current round's MFMAs (6 independent
// dwordx4 in flight per lane = the gfx950 streaming recipe). Partials
// are reduced across waves through LDS in ascending wave order — no
// atomics, graph replay is bit-identical.
//
// M > 16 needs ceil(M/16) m-tiles; two placements are provided:
//   fused : one workgroup owns all m-tiles (MTPW accumulators per
//           lane share each weight fragment — DRAM traffic stays 1x).
//           Best when grid.x alone gives enough workgroups (wide N).
//   split : grid.y = ceil(M/16), one m-tile per workgroup. Triples
//           the workgroup count for the WG-starved narrow-N shapes
//           (N=1536 -> 96 column tiles); the 2nd/3rd readers of each
//           weight line ride the LLC.
// The A operand (<= 48 rows) is NOT staged in LDS: per-fragment 16B
// global reads hit the LLC (A is KBs vs MBs of weight) and staging
// 41xK bf16 rows would blow the LDS at K=6144.
//
// `variant`: 0 = auto (heuristic below), 1 = w4_fused, 2 = w8_fused,
// 3 = w4_split, 4 = w8_split. Auto picks w4_fused when N/16 >= 192
// and M > 16, else the deepest valid split form. The parity/bench
// gate drives all variants; production pins the measured winner.
//
// Constraints: 1 <= M <= 48, N % 16 == 0, and K/(32*waves) must land
// in {6, 12, 24, 48} for the selected wave depth (waves 4: K in
// {768, 1536, 3072, 6144}; waves 8: K in {1536, 3072, 6144, 12288}).
// A, Wp, D 16-byte aligned; bias is a dense bf16 vector of length N.
// ================================================================

int smallm_mfma_bf16_variant_count();
const char* smallm_mfma_bf16_variant_name(int id);

// epilogue: 0 = bias, 1 = bias + tanh-GELU, 2 = bias + residual (D +=)
void smallm_mfma_bf16_nn_packed(int variant, int epilogue,
                                const void* A_bf16, const void* Wp_bf16,
                                const void* bias_bf16, __hip_bfloat16* D,
                                int M, int N, int K, hipStream_t stream);

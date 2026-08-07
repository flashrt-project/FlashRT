// FlashRT — Hy-VLA denoise FFN megakernel declaration (Thor SM110, plain FP8 MMA).
#pragma once
#include <cuda_runtime.h>

// A: gu-GEMM (K->2*Nout) with gate/up dual-accumulator + silu_mul -> bf16 act (M,Nout).
//    descale = (*sx) * sgu ;  gu_w is fp8 (2*Nout, K): rows[0:Nout)=gate, [Nout:2Nout)=up.
extern "C" void hyvla_ffn_gu_silu_bf16(
    const void* x_fp8, const void* gu_w_fp8, void* act_bf16,
    int M, int K, int Nout, const void* sx_ptr, float sgu, cudaStream_t stream);

// B: dn-GEMM (K->N) + residual -> bf16 y (M,N).  descale = (*sa) * sdn ; dn_w fp8 (N,K).
extern "C" void hyvla_ffn_dn_res_bf16(
    const void* act_fp8, const void* dn_w_fp8, const void* residual, void* y_bf16,
    int M, int K, int N, const void* sa_ptr, float sdn, cudaStream_t stream);

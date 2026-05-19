// ================================================================
// FlashRT — Elementwise kernel declarations
// Residual add, gate multiply, bias residual
// Supports: __half (FP16), __nv_bfloat16 (BF16)
// ================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ── BF16 (original signatures, backward compatible) ──

void gate_mul_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                       const __nv_bfloat16* gate, int n,
                       cudaStream_t stream = 0);

void gate_mul_residual_out_bf16_g1d(const __nv_bfloat16* residual,
                                    const __nv_bfloat16* x,
                                    const __nv_bfloat16* gate_1d,
                                    __nv_bfloat16* out,
                                    int seq_len, int dim,
                                    cudaStream_t stream = 0);

void gate_mul_residual_out_bf16(const __nv_bfloat16* residual,
                                const __nv_bfloat16* x,
                                const __nv_bfloat16* gate,
                                __nv_bfloat16* out, int n,
                                cudaStream_t stream = 0);

void gate_mul_residual_out_bf16_gate_fp8(const __nv_bfloat16* residual,
                                         const __nv_bfloat16* x,
                                         const __nv_fp8_e4m3* gate,
                                         const float* gate_scale,
                                         __nv_bfloat16* out, int n,
                                         cudaStream_t stream = 0);

void bias_residual(__nv_bfloat16* residual, const __nv_bfloat16* x,
                   const __nv_bfloat16* bias, int seq_len, int dim,
                   cudaStream_t stream = 0);

void bias_residual_out_bf16(const __nv_bfloat16* residual,
                            const __nv_bfloat16* x,
                            const __nv_bfloat16* bias,
                            __nv_bfloat16* out,
                            int seq_len, int dim,
                            cudaStream_t stream = 0);

void residual_add(__nv_bfloat16* residual, const __nv_bfloat16* x, int n,
                  cudaStream_t stream = 0);

void add_bf16_out(const __nv_bfloat16* a, const __nv_bfloat16* b,
                  __nv_bfloat16* out, int n,
                  cudaStream_t stream = 0);

void euler_step_bf16_out(const __nv_bfloat16* latent,
                         const __nv_bfloat16* velocity,
                         __nv_bfloat16* out,
                         float dt, int n,
                         cudaStream_t stream = 0);

void teacher_force_first_frame_bf16(__nv_bfloat16* video_latent,
                                    const __nv_bfloat16* cond_latent,
                                    int B, int C, int T, int H, int W,
                                    cudaStream_t stream = 0);

void motus_decode_postprocess_bf16_to_fp32(const __nv_bfloat16* decoded,
                                           float* out,
                                           int B, int C, int T_in,
                                           int H, int W,
                                           cudaStream_t stream = 0);

void cast_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n,
                       cudaStream_t stream = 0);

void ncdhw_to_blc_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                       int B, int C, int T, int H, int W,
                       cudaStream_t stream = 0);

void dup_up3d_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                   int B, int Cin, int Cout, int T, int H, int W,
                   int factor_t, int factor_s, int repeats,
                   int first_chunk, cudaStream_t stream = 0);

void time_unshuffle2_bf16(const __nv_bfloat16* x, __nv_bfloat16* out,
                          int B, int C, int T, int H, int W,
                          cudaStream_t stream = 0);

void add_bias_ncdhw_bf16(__nv_bfloat16* x, const __nv_bfloat16* bias,
                         int B, int C, int T, int H, int W,
                         cudaStream_t stream = 0);

void update_cache2_ncdhw_bf16(const __nv_bfloat16* cur,
                              const __nv_bfloat16* prev,
                              __nv_bfloat16* out,
                              int B, int C, int T, int H, int W,
                              cudaStream_t stream = 0);

void adaln_modulation6_bf16(const float* adaln_params,
                            const float* layer_modulation,
                            __nv_bfloat16* out0,
                            __nv_bfloat16* out1,
                            __nv_bfloat16* out2,
                            __nv_bfloat16* out3,
                            __nv_bfloat16* out4,
                            __nv_bfloat16* out5,
                            int B, int S, int D,
                            cudaStream_t stream = 0);

void concat3_qkv_bf16(const __nv_bfloat16* q0,
                      const __nv_bfloat16* q1,
                      const __nv_bfloat16* q2,
                      const __nv_bfloat16* k0,
                      const __nv_bfloat16* k1,
                      const __nv_bfloat16* k2,
                      const __nv_bfloat16* v0,
                      const __nv_bfloat16* v1,
                      const __nv_bfloat16* v2,
                      __nv_bfloat16* q_out,
                      __nv_bfloat16* k_out,
                      __nv_bfloat16* v_out,
                      int B, int L0, int L1, int L2,
                      int H, int D,
                      long long q0s0, long long q0s1, long long q0s2,
                      long long q1s0, long long q1s1, long long q1s2,
                      long long q2s0, long long q2s1, long long q2s2,
                      long long k0s0, long long k0s1, long long k0s2,
                      long long k1s0, long long k1s1, long long k1s2,
                      long long k2s0, long long k2s1, long long k2s2,
                      long long v0s0, long long v0s1, long long v0s2,
                      long long v1s0, long long v1s1, long long v1s2,
                      long long v2s0, long long v2s1, long long v2s2,
                      cudaStream_t stream = 0);

void concat3_qkv_bf16_fast(const __nv_bfloat16* q0,
                           const __nv_bfloat16* q1,
                           const __nv_bfloat16* q2,
                           const __nv_bfloat16* k0,
                           const __nv_bfloat16* k1,
                           const __nv_bfloat16* k2,
                           const __nv_bfloat16* v0,
                           const __nv_bfloat16* v1,
                           const __nv_bfloat16* v2,
                           __nv_bfloat16* q_out,
                           __nv_bfloat16* k_out,
                           __nv_bfloat16* v_out,
                           int B, int L0, int L1, int L2, int H, int D,
                           long long q0s0, long long q0s1,
                           long long q1s0, long long q1s1,
                           long long q2s0, long long q2s1,
                           long long k0s0, long long k0s1,
                           long long k1s0, long long k1s1,
                           long long k2s0, long long k2s1,
                           long long v0s0, long long v0s1,
                           long long v1s0, long long v1s1,
                           long long v2s0, long long v2s1,
                           cudaStream_t stream = 0);

void concat3_qk_int8_v_fp16_d128(const __nv_bfloat16* q0,
                                 const __nv_bfloat16* q1,
                                 const __nv_bfloat16* q2,
                                 const __nv_bfloat16* k0,
                                 const __nv_bfloat16* k1,
                                 const __nv_bfloat16* k2,
                                 const __nv_bfloat16* v0,
                                 const __nv_bfloat16* v1,
                                 const __nv_bfloat16* v2,
                                 int8_t* q_out,
                                 int8_t* k_out,
                                 __half* v_out,
                                 float* q_scale,
                                 float* k_scale,
                                 int B, int L0, int L1, int L2, int H,
                                 long long q0s0, long long q0s1,
                                 long long q1s0, long long q1s1,
                                 long long q2s0, long long q2s1,
                                 long long k0s0, long long k0s1,
                                 long long k1s0, long long k1s1,
                                 long long k2s0, long long k2s1,
                                 long long v0s0, long long v0s1,
                                 long long v1s0, long long v1s1,
                                 long long v2s0, long long v2s1,
                                 cudaStream_t stream = 0);

void concat3_qk_int8_v_fp8_d128(const __nv_bfloat16* q0,
                                const __nv_bfloat16* q1,
                                const __nv_bfloat16* q2,
                                const __nv_bfloat16* k0,
                                const __nv_bfloat16* k1,
                                const __nv_bfloat16* k2,
                                const __nv_bfloat16* v0,
                                const __nv_bfloat16* v1,
                                const __nv_bfloat16* v2,
                                int8_t* q_out,
                                int8_t* k_out,
                                int8_t* v_fp8_out,
                                float* q_scale,
                                float* k_scale,
                                float* v_scale,
                                int B, int L0, int L1, int L2, int H,
                                long long q0s0, long long q0s1,
                                long long q1s0, long long q1s1,
                                long long q2s0, long long q2s1,
                                long long k0s0, long long k0s1,
                                long long k1s0, long long k1s1,
                                long long k2s0, long long k2s1,
                                long long v0s0, long long v0s1,
                                long long v1s0, long long v1s1,
                                long long v2s0, long long v2s1,
                                cudaStream_t stream = 0);

void concat3_v_transpose_pad_permute_bf16_d128(
                                const __nv_bfloat16* v0,
                                const __nv_bfloat16* v1,
                                const __nv_bfloat16* v2,
                                __nv_bfloat16* v_tpp_out,
                                int B, int L0, int L1, int L2, int H,
                                long long v0s0, long long v0s1,
                                long long v1s0, long long v1s1,
                                long long v2s0, long long v2s1,
                                cudaStream_t stream = 0);

void v_tpp_bf16_quant_fp8_d128(const __nv_bfloat16* v_tpp,
                               int8_t* v_fp8,
                               float* v_scale,
                               int B, int L, int H,
                               cudaStream_t stream = 0);

void concat3_v_tpp_bf16_amax_d128(const __nv_bfloat16* v0,
                                  const __nv_bfloat16* v1,
                                  const __nv_bfloat16* v2,
                                  __nv_bfloat16* v_tpp_out,
                                  float* tile_amax,
                                  int B, int L0, int L1, int L2, int H,
                                  long long v0s0, long long v0s1,
                                  long long v1s0, long long v1s1,
                                  long long v2s0, long long v2s1,
                                  cudaStream_t stream = 0);

void v_tpp_bf16_quant_fp8_amax_d128(const __nv_bfloat16* v_tpp,
                                    const float* tile_amax,
                                    int8_t* v_fp8,
                                    float* v_scale,
                                    int B, int L, int H,
                                    cudaStream_t stream = 0);

void quant_per_warp_int8_bf16_d128(const __nv_bfloat16* x,
                                   int8_t* out,
                                   float* scale,
                                   int B, int L, int H,
                                   cudaStream_t stream = 0);

void quant_per_block_int8_bf16_d128(const __nv_bfloat16* x,
                                    int8_t* out,
                                    float* scale,
                                    int B, int L, int H,
                                    cudaStream_t stream = 0);

// ── FP16 variants ──

void gate_mul_residual_fp16(__half* residual, const __half* x,
                            const __half* gate, int n,
                            cudaStream_t stream = 0);

void bias_residual_fp16(__half* residual, const __half* x,
                        const __half* bias, int seq_len, int dim,
                        cudaStream_t stream = 0);

// G6.7: residual += (x + bias) * gate, in-place
void bias_gate_mul_residual_bf16(__nv_bfloat16* residual,
                                 const __nv_bfloat16* x,
                                 const __nv_bfloat16* bias,
                                 const __nv_bfloat16* gate,
                                 int seq_len, int dim,
                                 cudaStream_t stream = 0);

void bias_gate_mul_residual_out_bf16(const __nv_bfloat16* residual,
                                     const __nv_bfloat16* x,
                                     const __nv_bfloat16* bias,
                                     const __nv_bfloat16* gate,
                                     __nv_bfloat16* out,
                                     int seq_len, int dim,
                                     cudaStream_t stream = 0);

void bias_gate_mul_residual_out_bf16_g1d(const __nv_bfloat16* residual,
                                         const __nv_bfloat16* x,
                                         const __nv_bfloat16* bias,
                                         const __nv_bfloat16* gate_1d,
                                         __nv_bfloat16* out,
                                         int seq_len, int dim,
                                         cudaStream_t stream = 0);

void bias_gate_mul_residual_out_bf16_gate_fp8(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* x,
    const __nv_bfloat16* bias,
    const __nv_fp8_e4m3* gate,
    const float* gate_scale,
    __nv_bfloat16* out,
    int seq_len, int dim,
    cudaStream_t stream = 0);

void motus_joint_residual3_out_bf16(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_bfloat16* v_gate,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_bias,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream = 0);

void motus_joint_residual3_out_bf16_vgate_fp8(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_fp8_e4m3* v_gate,
    const float* v_gate_scale,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_bias,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream = 0);

void motus_joint_residual3_out_bf16_action_nobias(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_bfloat16* v_gate,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream = 0);

void motus_joint_residual3_out_bf16_g1d_action_nobias(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_bfloat16* v_gate_1d,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_gate_1d,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream = 0);

void motus_joint_residual3_out_bf16_vgate_fp8_action_nobias(
    const __nv_bfloat16* v_residual,
    const __nv_bfloat16* v_x,
    const __nv_bfloat16* v_bias,
    const __nv_fp8_e4m3* v_gate,
    const float* v_gate_scale,
    __nv_bfloat16* v_out,
    int v_n, int v_dim,
    const __nv_bfloat16* a_residual,
    const __nv_bfloat16* a_x,
    const __nv_bfloat16* a_gate,
    __nv_bfloat16* a_out,
    int a_n, int a_dim,
    const __nv_bfloat16* u_residual,
    const __nv_bfloat16* u_x,
    __nv_bfloat16* u_out,
    int u_n, int u_dim,
    cudaStream_t stream = 0);

void bias_gate_mul_residual_fp16(__half* residual,
                                 const __half* x,
                                 const __half* bias,
                                 const __half* gate,
                                 int seq_len, int dim,
                                 cudaStream_t stream = 0);

void residual_add_fp16(__half* residual, const __half* x, int n,
                       cudaStream_t stream = 0);

// ── Classifier-Free Guidance combine ──
// In-place: noise[i] += v_uncond[i] + beta * (v_cond[i] - v_uncond[i])
//         = noise[i] + (1 - beta) * v_uncond[i] + beta * v_cond[i]
// Used by Pi05CFGPipeline per denoising step (arXiv:2511.14759 App. E).

void cfg_combine_into_residual(__nv_bfloat16* residual,
                               const __nv_bfloat16* v_cond,
                               const __nv_bfloat16* v_uncond,
                               float beta, int n,
                               cudaStream_t stream = 0);

void cfg_combine_into_residual_fp16(__half* residual,
                                    const __half* v_cond,
                                    const __half* v_uncond,
                                    float beta, int n,
                                    cudaStream_t stream = 0);

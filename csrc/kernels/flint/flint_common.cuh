// SPDX-License-Identifier: Apache-2.0
//
// Shared device helpers for the Flint (SM120) kernels.
//
// Numerics contract (W4A16, official checkpoint semantics):
//  * NVFP4 weights are dequantized exactly: e2m1(code) * e4m3(block scale) has
//    at most 6 significant bits, so it is exact in fp16/bf16/fp32.
//  * Activations are BF16, every matmul accumulates in fp32, and the f32
//    per-tensor (or per-expert) global scale multiplies the fp32 accumulator
//    before the single rounding of the output.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include <string>

namespace flash_rt {
namespace flint {

inline void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}


__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

// Round-to-nearest-even fp32 -> bf16 bits (NaN-safe enough for activations).
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
  __nv_bfloat16 h = __float2bfloat16_rn(f);
  return *reinterpret_cast<uint16_t*>(&h);
}

__device__ __forceinline__ float round_bf16(float f) {
  return bf16_to_f32(f32_to_bf16_bits(f));
}

// Low / high bf16 of a packed pair as fp32.
__device__ __forceinline__ float bf16lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  return static_cast<uint32_t>(f32_to_bf16_bits(lo)) |
         (static_cast<uint32_t>(f32_to_bf16_bits(hi)) << 16);
}

// Four bytes of E2M1 codes (element 2i in the low nibble of byte i) -> four
// f16x2 (exact). out[i] holds elements 2i (low half) and 2i+1 (high half).
__device__ __forceinline__ void e2m1x8_to_f16x2(uint32_t codes, uint32_t (&out)[4]) {
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
      : "r"(codes));
}

// One E4M3 scale byte -> f16x2 broadcast (exact; bit 7 must be clear).
__device__ __forceinline__ uint32_t e4m3_to_f16x2(uint32_t byte) {
  uint32_t out;
  const uint16_t pair = static_cast<uint16_t>(byte | (byte << 8));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(pair));
  return out;
}

__device__ __forceinline__ float2 h2_to_f2(uint32_t h2) {
  __half2 v = *reinterpret_cast<__half2*>(&h2);
  return __half22float2(v);
}

__device__ __forceinline__ uint32_t hmul2_u32(uint32_t a, uint32_t b) {
  __half2 r = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

// Dot product of 8 packed E2M1 codes (one u32, element 2i in the low nibble of
// byte i) with 8 bf16 activations (4 packed u32), weights scaled by the f16x2
// block scale `s2`. Exact weights, fp32 FMA accumulation.
__device__ __forceinline__ float dot8_e2m1_bf16(uint32_t codes, uint32_t s2, const uint32_t (&x)[4], float acc) {
  uint32_t h[4];
  e2m1x8_to_f16x2(codes, h);
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const float2 w = h2_to_f2(hmul2_u32(h[b], s2));
    acc = fmaf(w.x, bf16lo(x[b]), acc);
    acc = fmaf(w.y, bf16hi(x[b]), acc);
  }
  return acc;
}

template <int W = 32>
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = W / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

template <int W = 32>
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = W / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ uint4 ldg_nc_v4(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ldg_nc_v2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
               : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ldg_nc_u16(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t ldg_nc_u8(const void* p) {
  unsigned short r;
  asm volatile("ld.global.nc.L1::no_allocate.u8 %0, [%1];" : "=h"(r) : "l"(p));
  return r;
}

// Precise SiLU-and-mul of bf16-rounded inputs (dense / shared-expert rule).
__device__ __forceinline__ float silu_mul_precise(float g, float u) {
  return g / (1.0f + expf(-g)) * u;
}

// Approximate SiLU on fp32 gate/up (routed-expert rule: ex2/rcp approximations).
__device__ __forceinline__ float silu_mul_routed(float g, float u) {
  float e, r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(-g * 1.4426950408889634f));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(1.0f + e));
  return g * r * u;
}

}  // namespace flint
}  // namespace flash_rt

// ---------------------------------------------------------------------------
// Programmatic dependent launch (PDL). Kernels launched with launch_pdl may
// start while the previous kernel in the stream is still running; each kernel
// therefore only reads static data (weights, scales) before pdl_wait() and
// performs no global writes before it. pdl_wait() is a no-op for kernels
// launched without the attribute.
// ---------------------------------------------------------------------------
#include <cstdlib>
#include <utility>

namespace flash_rt {
namespace flint {

// Bulk prefetch of a contiguous global range into L2 (size multiple of 16 bytes).
__device__ __forceinline__ void prefetch_l2_bulk(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// pdl_wait() ordered after the loads that produced `dep` (fold one word of every pre-wait load into it).
// ptxas otherwise schedules griddepcontrol.wait ahead of independent pre-wait loads, which then miss the
// overlap with the previous kernel. Storing `dep` to a shared scratch word right before the wait (one asm
// block) keeps those loads in front of the wait; the stored value is never read.
__device__ __forceinline__ void pdl_wait_after(uint32_t dep) {
  __shared__ uint32_t pdl_sink;
  asm volatile("{\n\tst.shared.u32 [%0], %1;\n\tgriddepcontrol.wait;\n\t}"
               :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&pdl_sink))), "r"(dep) : "memory");
}

// Pointer launder for data produced or consumed by other grids. A "memory" clobber on
// griddepcontrol.wait does not stop the compiler from hoisting loads through
// const __restrict__ pointers above it (restrict promises nobody else writes the data
// during the kernel, which PDL deliberately violates). Every pointer to such data is
// passed through this opaque move right after pdl_wait(), so its loads/stores depend on
// a value that only exists after the wait.
template <typename T>
__device__ __forceinline__ T* after_wait(T* p) {
  asm volatile("mov.b64 %0, %0;" : "+l"(p)::"memory");
  return p;
}
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" :::); }

inline bool pdl_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("FLINT_PDL");
    return !(v && v[0] == '0');
  }();
  return on;
}

inline bool attn_cluster_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("FLINT_ATTN_CLUSTER");
    return !(v && v[0] == '0');
  }();
  return on;
}

template <typename... KArgs, typename... Args>
inline void launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args&&... args) {
  if (!pdl_enabled()) {
    kernel<<<grid, block, smem, s>>>(std::forward<Args>(args)...);
    return;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  check_cuda(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...), "cudaLaunchKernelEx");
}

}  // namespace flint
}  // namespace flash_rt

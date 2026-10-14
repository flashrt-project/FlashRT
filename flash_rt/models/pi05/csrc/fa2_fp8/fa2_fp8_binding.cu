// Raw-pointer binding for the FA2 causal bf16 head_dim-256 path with the FP8
// attention-output epilogue. Mirrors flash_rt_fa2.fwd_bf16_causal and adds the
// FP8 destination, so the prefix o_proj GEMM reads its input straight from the
// attention kernel instead of from a separate quantize pass.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "fa2_fp8_wrapper.h"

static void fwd_bf16(
    uintptr_t q, uintptr_t k, uintptr_t v, uintptr_t o, uintptr_t lse,
    uintptr_t lse_accum, uintptr_t o_accum,
    int batch, int seqlen_q, int seqlen_k, int nheads_q, int nheads_kv, int head_dim,
    std::tuple<int, int, int> qs, std::tuple<int, int, int> ks,
    std::tuple<int, int, int> vs, std::tuple<int, int, int> os,
    double scale, int num_sms, uintptr_t stream,
    uintptr_t o_fp8, uintptr_t o_fp8_scale,
    std::tuple<int, int, int> o8s, bool skip_bf16, bool causal) {
  auto entry = causal ? &dm05_fa2_fp8_fwd_bf16_causal : &dm05_fa2_fp8_fwd_bf16_noncausal;
  entry(
      (const void*)q, (const void*)k, (const void*)v, (void*)o, (void*)lse,
      (void*)lse_accum, (void*)o_accum,
      batch, seqlen_q, seqlen_k, nheads_q, nheads_kv, head_dim,
      std::get<0>(qs), std::get<1>(qs), std::get<2>(qs),
      std::get<0>(ks), std::get<1>(ks), std::get<2>(ks),
      std::get<0>(vs), std::get<1>(vs), std::get<2>(vs),
      std::get<0>(os), std::get<1>(os), std::get<2>(os),
      (float)scale, num_sms, (cudaStream_t)stream,
      (void*)o_fp8, (const float*)o_fp8_scale,
      std::get<0>(o8s), std::get<1>(o8s), std::get<2>(o8s), skip_bf16 ? 1 : 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd_bf16", &fwd_bf16,
        "FA2 bf16 hdim256 (causal or full) with an optional FP8 attention-output epilogue");
}

// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#if defined(CUDA_KERNEL)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "registration.h"
#include "sage2_blackwell.cuh"

namespace {

constexpr int64_t kHeadDim = 128;

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_dtype(torch::Tensor const& tensor, c10::ScalarType dtype, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
}

void check_same_device(torch::Tensor const& a, torch::Tensor const& b, const char* an, const char* bn) {
  TORCH_CHECK(a.get_device() == b.get_device(), an, " and ", bn, " must be on the same CUDA device");
}

void check_bhd128(torch::Tensor const& x, const char* name, c10::ScalarType dtype) {
  check_dtype(x, dtype, name);
  TORCH_CHECK(x.dim() == 4, name, " must have shape (batch, seqlen, heads, 128)");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && x.size(3) == kHeadDim,
              name, " must have shape (batch, seqlen, heads, 128)");
}

void check_scale(torch::Tensor const& scale, int64_t expected, const char* name) {
  check_dtype(scale, torch::kFloat32, name);
  TORCH_CHECK(scale.numel() >= expected, name, " is too small");
}

int64_t div_up_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

}  // namespace

int64_t padded_k64(int64_t seqlen_k) {
  TORCH_CHECK(seqlen_k > 0, "seqlen_k must be positive");
  return flashrt_hub::sage2::padded_k64(static_cast<int>(seqlen_k));
}

int64_t q_scale_elems(int64_t batch, int64_t seqlen_q, int64_t q_heads) {
  TORCH_CHECK(batch > 0 && seqlen_q > 0 && q_heads > 0, "shape values must be positive");
  return flashrt_hub::sage2::q_scale_elems(static_cast<int>(batch), static_cast<int>(seqlen_q), static_cast<int>(q_heads));
}

int64_t k_scale_elems(int64_t batch, int64_t seqlen_k, int64_t kv_heads) {
  TORCH_CHECK(batch > 0 && seqlen_k > 0 && kv_heads > 0, "shape values must be positive");
  return flashrt_hub::sage2::k_scale_elems(static_cast<int>(batch), static_cast<int>(seqlen_k), static_cast<int>(kv_heads));
}

int64_t v_scale_elems(int64_t batch, int64_t kv_heads) {
  TORCH_CHECK(batch > 0 && kv_heads > 0, "shape values must be positive");
  return flashrt_hub::sage2::v_scale_elems(static_cast<int>(batch), static_cast<int>(kHeadDim), static_cast<int>(kv_heads));
}

void quantize_q_bf16_d128(torch::Tensor const& q, torch::Tensor& q_i8, torch::Tensor& q_scale) {
  check_bhd128(q, "q", torch::kBFloat16);
  check_bhd128(q_i8, "q_i8", torch::kInt8);
  TORCH_CHECK(q_i8.sizes() == q.sizes(), "q_i8 shape must match q");
  check_scale(q_scale, q_scale_elems(q.size(0), q.size(1), q.size(2)), "q_scale");
  check_same_device(q, q_i8, "q", "q_i8");
  check_same_device(q, q_scale, "q", "q_scale");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  flashrt_hub::sage2::quant_per_warp_int8_bf16_d128(
      q.data_ptr(), q_i8.data_ptr(), q_scale.data_ptr(),
      static_cast<int>(q.size(0)), static_cast<int>(q.size(1)), static_cast<int>(q.size(2)), stream);
#else
  TORCH_CHECK(false, "sageattention2-blackwell was not built with CUDA support");
#endif
}

void quantize_k_bf16_d128(torch::Tensor const& k, torch::Tensor& k_i8, torch::Tensor& k_scale) {
  check_bhd128(k, "k", torch::kBFloat16);
  check_bhd128(k_i8, "k_i8", torch::kInt8);
  TORCH_CHECK(k_i8.sizes() == k.sizes(), "k_i8 shape must match k");
  check_scale(k_scale, k_scale_elems(k.size(0), k.size(1), k.size(2)), "k_scale");
  check_same_device(k, k_i8, "k", "k_i8");
  check_same_device(k, k_scale, "k", "k_scale");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k.device());
  auto stream = at::cuda::getCurrentCUDAStream(k.get_device()).stream();
  flashrt_hub::sage2::quant_per_block_int8_bf16_d128(
      k.data_ptr(), k_i8.data_ptr(), k_scale.data_ptr(),
      static_cast<int>(k.size(0)), static_cast<int>(k.size(1)), static_cast<int>(k.size(2)), stream);
#else
  TORCH_CHECK(false, "sageattention2-blackwell was not built with CUDA support");
#endif
}

void quantize_v_fp16_bf16_d128(torch::Tensor const& v, torch::Tensor& v_half) {
  check_bhd128(v, "v", torch::kBFloat16);
  check_bhd128(v_half, "v_half", torch::kFloat16);
  TORCH_CHECK(v_half.sizes() == v.sizes(), "v_half shape must match v");
  check_same_device(v, v_half, "v", "v_half");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(v.device());
  auto stream = at::cuda::getCurrentCUDAStream(v.get_device()).stream();
  flashrt_hub::sage2::v_bf16_to_fp16_d128(
      v.data_ptr(), v_half.data_ptr(),
      static_cast<int>(v.size(0)), static_cast<int>(v.size(1)), static_cast<int>(v.size(2)), stream);
#else
  TORCH_CHECK(false, "sageattention2-blackwell was not built with CUDA support");
#endif
}

void quantize_v_fp8_bf16_d128(torch::Tensor const& v, torch::Tensor& v_fp8_tpp, torch::Tensor& v_scale) {
  check_bhd128(v, "v", torch::kBFloat16);
  check_dtype(v_fp8_tpp, torch::kInt8, "v_fp8_tpp");
  TORCH_CHECK(v_fp8_tpp.dim() == 4, "v_fp8_tpp must have shape (batch, 128, kv_heads, padded_seqlen)");
  TORCH_CHECK(v_fp8_tpp.size(0) == v.size(0) && v_fp8_tpp.size(1) == kHeadDim &&
              v_fp8_tpp.size(2) == v.size(2) && v_fp8_tpp.size(3) >= padded_k64(v.size(1)),
              "v_fp8_tpp shape must be (batch, 128, kv_heads, padded_seqlen)");
  check_scale(v_scale, v_scale_elems(v.size(0), v.size(2)), "v_scale");
  check_same_device(v, v_fp8_tpp, "v", "v_fp8_tpp");
  check_same_device(v, v_scale, "v", "v_scale");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(v.device());
  auto stream = at::cuda::getCurrentCUDAStream(v.get_device()).stream();
  flashrt_hub::sage2::v_bf16_to_fp8_tpp_d128(
      v.data_ptr(), v_fp8_tpp.data_ptr(), v_scale.data_ptr(),
      static_cast<int>(v.size(0)), static_cast<int>(v.size(1)), static_cast<int>(v.size(2)), stream);
#else
  TORCH_CHECK(false, "sageattention2-blackwell was not built with CUDA support");
#endif
}

void sage2_qk_int8_sv_f16_bf16_d128(
    torch::Tensor const& q_i8,
    torch::Tensor const& k_i8,
    torch::Tensor const& v_half,
    torch::Tensor const& q_scale,
    torch::Tensor const& k_scale,
    double softmax_scale,
    bool causal,
    torch::Tensor& out) {
  check_bhd128(q_i8, "q_i8", torch::kInt8);
  check_bhd128(k_i8, "k_i8", torch::kInt8);
  check_bhd128(v_half, "v_half", torch::kFloat16);
  check_bhd128(out, "out", torch::kBFloat16);
  TORCH_CHECK(q_i8.size(0) == k_i8.size(0) && q_i8.size(0) == v_half.size(0), "batch mismatch");
  TORCH_CHECK(k_i8.size(1) == v_half.size(1), "K/V seqlen mismatch");
  TORCH_CHECK(k_i8.size(2) == v_half.size(2), "K/V heads mismatch");
  TORCH_CHECK(q_i8.size(2) % k_i8.size(2) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(out.sizes() == q_i8.sizes(), "out shape must match q_i8");
  check_scale(q_scale, q_scale_elems(q_i8.size(0), q_i8.size(1), q_i8.size(2)), "q_scale");
  check_scale(k_scale, k_scale_elems(k_i8.size(0), k_i8.size(1), k_i8.size(2)), "k_scale");
  check_same_device(q_i8, k_i8, "q_i8", "k_i8");
  check_same_device(q_i8, v_half, "q_i8", "v_half");
  check_same_device(q_i8, out, "q_i8", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q_i8.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_i8.get_device()).stream();
  int rc = flashrt_hub::sage2::sage2_qk_int8_sv_f16_bf16_gqa_d128(
      q_i8.data_ptr(), k_i8.data_ptr(), v_half.data_ptr(), out.data_ptr(),
      q_scale.data_ptr(), k_scale.data_ptr(),
      static_cast<int>(q_i8.size(0)), static_cast<int>(q_i8.size(1)), static_cast<int>(k_i8.size(1)),
      static_cast<int>(q_i8.size(2)), static_cast<int>(k_i8.size(2)),
      static_cast<float>(softmax_scale), causal, stream);
  TORCH_CHECK(rc == 0, "sage2_qk_int8_sv_f16_bf16_d128 failed with cuda error ", rc);
#else
  TORCH_CHECK(false, "sageattention2-blackwell was not built with CUDA support");
#endif
}

void sage2_qk_int8_sv_f8_bf16_d128(
    torch::Tensor const& q_i8,
    torch::Tensor const& k_i8,
    torch::Tensor const& v_fp8_tpp,
    torch::Tensor const& q_scale,
    torch::Tensor const& k_scale,
    torch::Tensor const& v_scale,
    double softmax_scale,
    bool causal,
    torch::Tensor& out) {
  check_bhd128(q_i8, "q_i8", torch::kInt8);
  check_bhd128(k_i8, "k_i8", torch::kInt8);
  check_dtype(v_fp8_tpp, torch::kInt8, "v_fp8_tpp");
  check_bhd128(out, "out", torch::kBFloat16);
  TORCH_CHECK(v_fp8_tpp.dim() == 4, "v_fp8_tpp must have shape (batch, 128, kv_heads, padded_seqlen)");
  TORCH_CHECK(q_i8.size(0) == k_i8.size(0) && q_i8.size(0) == v_fp8_tpp.size(0), "batch mismatch");
  TORCH_CHECK(v_fp8_tpp.size(1) == kHeadDim && v_fp8_tpp.size(2) == k_i8.size(2) &&
              v_fp8_tpp.size(3) >= padded_k64(k_i8.size(1)),
              "v_fp8_tpp shape must be (batch, 128, kv_heads, padded_seqlen)");
  TORCH_CHECK(q_i8.size(2) % k_i8.size(2) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(out.sizes() == q_i8.sizes(), "out shape must match q_i8");
  check_scale(q_scale, q_scale_elems(q_i8.size(0), q_i8.size(1), q_i8.size(2)), "q_scale");
  check_scale(k_scale, k_scale_elems(k_i8.size(0), k_i8.size(1), k_i8.size(2)), "k_scale");
  check_scale(v_scale, v_scale_elems(k_i8.size(0), k_i8.size(2)), "v_scale");
  check_same_device(q_i8, k_i8, "q_i8", "k_i8");
  check_same_device(q_i8, v_fp8_tpp, "q_i8", "v_fp8_tpp");
  check_same_device(q_i8, out, "q_i8", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q_i8.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_i8.get_device()).stream();
  int rc = flashrt_hub::sage2::sage2_qk_int8_sv_f8_bf16_gqa_d128(
      q_i8.data_ptr(), k_i8.data_ptr(), v_fp8_tpp.data_ptr(), out.data_ptr(),
      q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
      static_cast<int>(q_i8.size(0)), static_cast<int>(q_i8.size(1)), static_cast<int>(k_i8.size(1)),
      static_cast<int>(q_i8.size(2)), static_cast<int>(k_i8.size(2)),
      static_cast<float>(softmax_scale), causal, stream, 0, nullptr);
  TORCH_CHECK(rc == 0, "sage2_qk_int8_sv_f8_bf16_d128 failed with cuda error ", rc);
#else
  TORCH_CHECK(false, "sageattention2-blackwell was not built with CUDA support");
#endif
}

void sage2_qk_int8_sv_f8_lse_bf16_d128(
    torch::Tensor const& q_i8,
    torch::Tensor const& k_i8,
    torch::Tensor const& v_fp8_tpp,
    torch::Tensor const& q_scale,
    torch::Tensor const& k_scale,
    torch::Tensor const& v_scale,
    double softmax_scale,
    bool causal,
    int64_t q_offset,
    torch::Tensor& out,
    torch::Tensor& lse) {
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32 && lse.is_contiguous() &&
              lse.numel() == q_i8.size(0) * q_i8.size(2) * q_i8.size(1), "lse must be float32 [batch, q_heads, seqlen_q]");
  check_bhd128(q_i8, "q_i8", torch::kInt8);
  check_bhd128(k_i8, "k_i8", torch::kInt8);
  check_dtype(v_fp8_tpp, torch::kInt8, "v_fp8_tpp");
  check_bhd128(out, "out", torch::kBFloat16);
  TORCH_CHECK(v_fp8_tpp.dim() == 4, "v_fp8_tpp must have shape (batch, 128, kv_heads, padded_seqlen)");
  TORCH_CHECK(q_i8.size(0) == k_i8.size(0) && q_i8.size(0) == v_fp8_tpp.size(0), "batch mismatch");
  TORCH_CHECK(v_fp8_tpp.size(1) == kHeadDim && v_fp8_tpp.size(2) == k_i8.size(2) &&
              v_fp8_tpp.size(3) >= padded_k64(k_i8.size(1)),
              "v_fp8_tpp shape must be (batch, 128, kv_heads, padded_seqlen)");
  TORCH_CHECK(q_i8.size(2) % k_i8.size(2) == 0, "q_heads must be divisible by kv_heads");
  TORCH_CHECK(out.sizes() == q_i8.sizes(), "out shape must match q_i8");
  check_scale(q_scale, q_scale_elems(q_i8.size(0), q_i8.size(1), q_i8.size(2)), "q_scale");
  check_scale(k_scale, k_scale_elems(k_i8.size(0), k_i8.size(1), k_i8.size(2)), "k_scale");
  check_scale(v_scale, v_scale_elems(k_i8.size(0), k_i8.size(2)), "v_scale");
  check_same_device(q_i8, k_i8, "q_i8", "k_i8");
  check_same_device(q_i8, v_fp8_tpp, "q_i8", "v_fp8_tpp");
  check_same_device(q_i8, out, "q_i8", "out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q_i8.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_i8.get_device()).stream();
  int rc = flashrt_hub::sage2::sage2_qk_int8_sv_f8_bf16_gqa_d128(
      q_i8.data_ptr(), k_i8.data_ptr(), v_fp8_tpp.data_ptr(), out.data_ptr(),
      q_scale.data_ptr(), k_scale.data_ptr(), v_scale.data_ptr(),
      static_cast<int>(q_i8.size(0)), static_cast<int>(q_i8.size(1)), static_cast<int>(k_i8.size(1)),
      static_cast<int>(q_i8.size(2)), static_cast<int>(k_i8.size(2)),
      static_cast<float>(softmax_scale), causal, stream, static_cast<int>(q_offset), lse.data_ptr());
  TORCH_CHECK(rc == 0, "sage2_qk_int8_sv_f8_lse_bf16_d128 failed with cuda error ", rc);
#else
  TORCH_CHECK(false, "sageattention2-blackwell was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("padded_k64(int seqlen_k) -> int");
  ops.def("q_scale_elems(int batch, int seqlen_q, int q_heads) -> int");
  ops.def("k_scale_elems(int batch, int seqlen_k, int kv_heads) -> int");
  ops.def("v_scale_elems(int batch, int kv_heads) -> int");
  ops.def("quantize_q_bf16_d128(Tensor q, Tensor! q_i8, Tensor! q_scale) -> ()");
  ops.def("quantize_k_bf16_d128(Tensor k, Tensor! k_i8, Tensor! k_scale) -> ()");
  ops.def("quantize_v_fp16_bf16_d128(Tensor v, Tensor! v_half) -> ()");
  ops.def("quantize_v_fp8_bf16_d128(Tensor v, Tensor! v_fp8_tpp, Tensor! v_scale) -> ()");
  ops.def("sage2_qk_int8_sv_f16_bf16_d128(Tensor q_i8, Tensor k_i8, Tensor v_half, Tensor q_scale, Tensor k_scale, float softmax_scale, bool causal, Tensor! out) -> ()");
  ops.def("sage2_qk_int8_sv_f8_bf16_d128(Tensor q_i8, Tensor k_i8, Tensor v_fp8_tpp, Tensor q_scale, Tensor k_scale, Tensor v_scale, float softmax_scale, bool causal, Tensor! out) -> ()");
  ops.def("sage2_qk_int8_sv_f8_lse_bf16_d128(Tensor q_i8, Tensor k_i8, Tensor v_fp8_tpp, Tensor q_scale, Tensor k_scale, Tensor v_scale, float softmax_scale, bool causal, int q_offset, Tensor! out, Tensor! lse) -> ()");
  ops.impl("padded_k64", torch::kCPU, &padded_k64);
  ops.impl("q_scale_elems", torch::kCPU, &q_scale_elems);
  ops.impl("k_scale_elems", torch::kCPU, &k_scale_elems);
  ops.impl("v_scale_elems", torch::kCPU, &v_scale_elems);
  ops.impl("quantize_q_bf16_d128", torch::kCUDA, &quantize_q_bf16_d128);
  ops.impl("quantize_k_bf16_d128", torch::kCUDA, &quantize_k_bf16_d128);
  ops.impl("quantize_v_fp16_bf16_d128", torch::kCUDA, &quantize_v_fp16_bf16_d128);
  ops.impl("quantize_v_fp8_bf16_d128", torch::kCUDA, &quantize_v_fp8_bf16_d128);
  ops.impl("sage2_qk_int8_sv_f16_bf16_d128", torch::kCUDA, &sage2_qk_int8_sv_f16_bf16_d128);
  ops.impl("sage2_qk_int8_sv_f8_bf16_d128", torch::kCUDA, &sage2_qk_int8_sv_f8_bf16_d128);
  ops.impl("sage2_qk_int8_sv_f8_lse_bf16_d128", torch::kCUDA, &sage2_qk_int8_sv_f8_lse_bf16_d128);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

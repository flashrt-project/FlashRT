// SPDX-License-Identifier: Apache-2.0
// FP16-accumulate GEMM with the bias + GELU epilogue fused (CUTLASS SM80
// tensor-op tiles, run on sm_120).
//
//   D[M,N] = act(fp16(A[M,K] @ B[K,N]) + bias[N])      A, B, D row-major FP16
//   (act identity with an optional bias, tanh-GELU or erf-GELU)
//
// The mainloop accumulates in FP16 like the cuBLASLt COMPUTE_16F plans of
// lawam_gemm_f16acc.cu, with the same h16816 tile family and the same serial
// split-K (each K slice accumulates its own FP16 sum, the slices are added in
// order into D) that cuBLASLt selects for the long-K shapes. The epilogue adds
// the bias and applies the GELU in FP32 exactly as lawam::bias_gelu does on the
// stored GEMM output, so the separate elementwise pass (or cuBLASLt's
// stand-alone GELU epilogue kernel) disappears. The tile and split count are
// timed per shape on the first call.
#include "lawam_kernels.cuh"

#include <map>
#include <memory>
#include <mutex>
#include <tuple>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"

#include "45_dual_gemm/device/dual_gemm.h"

namespace flash_rt {
namespace lawam {
namespace {

template <typename T>
struct Identity {
    CUTLASS_HOST_DEVICE T operator()(T const& x) const { return x; }
};

template <typename T>
struct GeluErf {
    CUTLASS_HOST_DEVICE T operator()(T const& x) const { return T(0.5f * x * (1.0f + erff(x * 0.70710678118654752f))); }
};

template <int N>
struct GeluErf<cutlass::Array<float, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& v) const {
        cutlass::Array<float, N> y;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) y[i] = 0.5f * v[i] * (1.0f + erff(v[i] * 0.70710678118654752f));
        return y;
    }
};

template <typename T>
struct GeluTanh {
    CUTLASS_HOST_DEVICE T operator()(T const& x) const {
        return T(0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))));
    }
};

template <int N>
struct GeluTanh<cutlass::Array<float, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& v) const {
        cutlass::Array<float, N> y;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            // 0.5 x (1 + tanh(u)) = x / (1 + exp(-2u)): one exp and one reciprocal instead of tanhf
            const float x = v[i];
#if defined(__CUDA_ARCH__)
            y[i] = __fdividef(x, 1.0f + __expf(-1.5957691216f * (x + 0.044715f * x * x * x)));
#else
            y[i] = x / (1.0f + expf(-1.5957691216f * (x + 0.044715f * x * x * x)));
#endif
        }
        return y;
    }
};

using half_t = cutlass::half_t;
using Row = cutlass::layout::RowMajor;
using cutlass::gemm::GemmShape;
using cutlass::epilogue::thread::ScaleType;

// D = act(1 * acc + bias): the bias enters as the source operand with a zero
// row stride. Without a bias: D = 1 * acc + 0 * D (the source is read only by
// the later serial split-K slices, whose beta becomes 1).
template <template <typename> class Act, bool kBias>
using Epilogue = cutlass::epilogue::thread::LinearCombinationGeneric<
    Act, half_t, 8, half_t, float, kBias ? ScaleType::NoBetaScaling : ScaleType::Default>;

template <template <typename> class Act, bool kBias, class TB, class Warp, int Stages>
using GemmT = cutlass::gemm::device::Gemm<half_t, Row, half_t, Row, half_t, Row, half_t,
                                          cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp,
                                          GemmShape<16, 8, 16>, Epilogue<Act, kBias>,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, 8, 8,
                                          /*SplitKSerial=*/true>;

// Stream-K tiles: the K iterations of the output tiles spread evenly over the
// SMs (36 on the RTX 5060 Ti, where the data-parallel grids of the M ~ 219 and
// 512 shapes leave SMs idle in the last wave); partial tiles meet in a
// workspace in fixed (turnstile) order, so results are reproducible.
template <template <typename> class Act, bool kBias, class TB, class Warp, int Stages>
using GemmSK = cutlass::gemm::device::GemmUniversal<half_t, Row, half_t, Row, half_t, Row, half_t,
                                                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp,
                                                    GemmShape<16, 8, 16>, Epilogue<Act, kBias>,
                                                    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages, 8,
                                                    8>;

// Serial split-K semaphores: one zeroed buffer per stream (every slice
// sequence leaves its tile semaphore at 0 again).
constexpr size_t kSemaphoreBytes = 1 << 16;

struct Ctx {
    std::mutex mu;
    std::map<std::tuple<int, int, int, int>, int> plan;   // (M, N, K, epilogue) -> tile * 16 + split
    std::map<cudaStream_t, void*> semaphores;
};

Ctx& ctx() {
    static Ctx c;
    return c;
}

void* semaphores_for(cudaStream_t stream) {
    Ctx& c = ctx();
    std::lock_guard<std::mutex> lock(c.mu);
    auto it = c.semaphores.find(stream);
    if (it != c.semaphores.end()) return it->second;
    void* w = nullptr;
    if (cudaMalloc(&w, kSemaphoreBytes) != cudaSuccess) return nullptr;
    if (cudaMemset(w, 0, kSemaphoreBytes) != cudaSuccess) return nullptr;
    c.semaphores[stream] = w;
    return w;
}

// The operator of each (shape, split, stream) is initialized once (the
// initialization clears the semaphores, so it must not run inside a graph
// capture); later calls only rebind the pointers.
template <class G>
int run(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, int split,
        cudaStream_t stream) {
    const half_t* src = reinterpret_cast<const half_t*>(bias != nullptr ? bias : D);
    typename G::Arguments args({M, N, K}, {reinterpret_cast<const half_t*>(A), K},
                               {reinterpret_cast<const half_t*>(B), N}, {src, bias != nullptr ? 0 : N},
                               {reinterpret_cast<half_t*>(D), N}, {1.0f, 0.0f}, split);
    void* sem = split > 1 ? semaphores_for(stream) : nullptr;
    if (split > 1 && (sem == nullptr || G::get_workspace_size(args) > kSemaphoreBytes))
        return static_cast<int>(cudaErrorMemoryAllocation);
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, int, cudaStream_t>, std::unique_ptr<G>> ops;
    G* op = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto& slot = ops[std::make_tuple(M, N, K, split, stream)];
        if (!slot) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap);
            if (cap != cudaStreamCaptureStatusNone) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
            auto fresh = std::make_unique<G>();
            if (fresh->can_implement(args) != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorInvalidValue);
            if (fresh->initialize(args, sem, stream) != cutlass::Status::kSuccess)
                return static_cast<int>(cudaErrorUnknown);
            slot = std::move(fresh);
        } else if (slot->update(args, sem) != cutlass::Status::kSuccess) {
            return static_cast<int>(cudaErrorUnknown);
        }
        op = slot.get();
    }
    return op->run(stream) == cutlass::Status::kSuccess ? 0 : static_cast<int>(cudaErrorLaunchFailure);
}

// Stream-K operator of each (shape, stream): initialized once outside a
// capture (workspace allocated and zeroed; the barriers reset themselves after
// every run), later calls rebind the pointers.
template <class G>
int run_sk(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, cudaStream_t stream) {
    const half_t* src = reinterpret_cast<const half_t*>(bias != nullptr ? bias : D);
    typename G::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, {1.0f, 0.0f},
                               reinterpret_cast<const half_t*>(A), reinterpret_cast<const half_t*>(B), src,
                               reinterpret_cast<half_t*>(D), 0, 0, 0, 0, K, N, bias != nullptr ? 0 : N, N);
    struct Slot {
        std::unique_ptr<G> op;
        void* ws = nullptr;
    };
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, cudaStream_t>, Slot> ops;
    G* op = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto& slot = ops[std::make_tuple(M, N, K, stream)];
        if (!slot.op) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap);
            if (cap != cudaStreamCaptureStatusNone) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
            auto fresh = std::make_unique<G>();
            if (fresh->can_implement(args) != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorInvalidValue);
            const size_t bytes = G::get_workspace_size(args);
            if (bytes > 0) {
                if (cudaMalloc(&slot.ws, bytes) != cudaSuccess) return static_cast<int>(cudaErrorMemoryAllocation);
                if (cudaMemset(slot.ws, 0, bytes) != cudaSuccess) return static_cast<int>(cudaErrorUnknown);
            }
            if (fresh->initialize(args, slot.ws, stream) != cutlass::Status::kSuccess)
                return static_cast<int>(cudaErrorUnknown);
            slot.op = std::move(fresh);
        } else if (slot.op->update(args) != cutlass::Status::kSuccess) {
            return static_cast<int>(cudaErrorUnknown);
        }
        op = slot.op.get();
    }
    return op->run(stream) == cutlass::Status::kSuccess ? 0 : static_cast<int>(cudaErrorLaunchFailure);
}

// Tile configurations (threadblock M x N x K, warp tile, stages); 8.. stream-K.
constexpr int kTiles = 11;
constexpr int kMaxSplit = 8;

template <template <typename> class Act, bool kBias>
int run_tile(int tile, int split, const __half* A, const __half* B, __half* D, const __half* bias, int M, int N,
             int K, cudaStream_t stream) {
    switch (tile) {
        case 0: return run<GemmT<Act, kBias, GemmShape<128, 128, 32>, GemmShape<64, 64, 32>, 5>>(A, B, D, bias, M, N, K, split, stream);
        case 1: return run<GemmT<Act, kBias, GemmShape<64, 128, 64>, GemmShape<32, 64, 64>, 4>>(A, B, D, bias, M, N, K, split, stream);
        case 2: return run<GemmT<Act, kBias, GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 3>>(A, B, D, bias, M, N, K, split, stream);
        case 3: return run<GemmT<Act, kBias, GemmShape<128, 64, 64>, GemmShape<64, 32, 64>, 4>>(A, B, D, bias, M, N, K, split, stream);
        case 4: return run<GemmT<Act, kBias, GemmShape<64, 64, 64>, GemmShape<32, 32, 64>, 5>>(A, B, D, bias, M, N, K, split, stream);
        case 5: return run<GemmT<Act, kBias, GemmShape<128, 256, 32>, GemmShape<64, 64, 32>, 3>>(A, B, D, bias, M, N, K, split, stream);
        case 6: return run<GemmT<Act, kBias, GemmShape<256, 128, 32>, GemmShape<64, 64, 32>, 3>>(A, B, D, bias, M, N, K, split, stream);
        case 7: return run<GemmT<Act, kBias, GemmShape<64, 256, 32>, GemmShape<32, 64, 32>, 4>>(A, B, D, bias, M, N, K, split, stream);
        case 8: return run_sk<GemmSK<Act, kBias, GemmShape<128, 128, 32>, GemmShape<64, 64, 32>, 5>>(A, B, D, bias, M, N, K, stream);
        case 9: return run_sk<GemmSK<Act, kBias, GemmShape<128, 256, 32>, GemmShape<64, 64, 32>, 3>>(A, B, D, bias, M, N, K, stream);
        default: return run_sk<GemmSK<Act, kBias, GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 3>>(A, B, D, bias, M, N, K, stream);
    }
}

// act 0: identity (+ bias when given), 1 tanh-GELU, 2 erf-GELU
int run_act(int act, int tile, int split, const __half* A, const __half* B, __half* D, const __half* bias, int M,
            int N, int K, cudaStream_t stream) {
    if (act == 1) return run_tile<GeluTanh, true>(tile, split, A, B, D, bias, M, N, K, stream);
    if (act == 2) return run_tile<GeluErf, true>(tile, split, A, B, D, bias, M, N, K, stream);
    if (bias != nullptr) return run_tile<Identity, true>(tile, split, A, B, D, bias, M, N, K, stream);
    return run_tile<Identity, false>(tile, split, A, B, D, bias, M, N, K, stream);
}

// ---- SwiGLU: silu(A @ Bg) * (A @ Bu) as one dual GEMM ------------------------
//
// CUTLASS example 45 (DualGemm): both products share the A tiles; the gate and
// up sums are formed like the single GEMM above (same serial split-K, the
// slices stored to Dg / Du), and the last slice's epilogue writes
// D = silu(g) * u in FP32 exactly as lawam::silu_mul does.

template <typename T, int N>
struct SiluMulOp {
    using ElementOutput = T;
    using ElementAccumulator = T;
    using ElementCompute = float;
    static int const kCount = N;
    using FragmentOutput = cutlass::Array<T, N>;
    using FragmentAccumulator = cutlass::Array<T, N>;
    struct Params {};
    CUTLASS_HOST_DEVICE SiluMulOp(Params const&) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_HOST_DEVICE FragmentOutput operator()(FragmentAccumulator const& g, FragmentAccumulator const& u) const {
        FragmentOutput y;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            const float x = float(g[i]);
            y[i] = T((x / (1.0f + expf(-x))) * float(u[i]));
        }
        return y;
    }
};

// slice sums: D = 1 * acc + beta * source (beta 0 on the first slice, whose
// source is a zero row, 1 on the later ones, whose source is D)
using SliceOp = cutlass::epilogue::thread::LinearCombination<half_t, 8, half_t, float>;

template <class TB, class Warp, int Stages>
using DualT = cutlass::gemm::device::DualGemm<half_t, Row, half_t, Row, Row, half_t, Row, half_t,
                                              cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp,
                                              GemmShape<16, 8, 16>, SliceOp, SliceOp, SiluMulOp<half_t, 8>,
                                              cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
                                              true, true, /*SplitKSerial=*/true, 8, 8>;

constexpr size_t kZeroBytes = 1 << 16;

const half_t* zero_row() {
    static void* z = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (cudaMalloc(&z, kZeroBytes) == cudaSuccess) cudaMemset(z, 0, kZeroBytes);
    });
    return static_cast<const half_t*>(z);
}

template <class G>
int run_dual(const __half* A, const __half* Bg, const __half* Bu, int ldb, __half* Dg, __half* Du, int ldd, __half* D,
             int M, int N, int K, int split, cudaStream_t stream) {
    const half_t* z = zero_row();
    if (z == nullptr || static_cast<size_t>(N) * sizeof(half_t) > kZeroBytes)
        return static_cast<int>(cudaErrorMemoryAllocation);
    typename G::Arguments args(cutlass::gemm::DualGemmMode::kGemm, {M, N, K},
                               {reinterpret_cast<const half_t*>(A), K}, {reinterpret_cast<const half_t*>(Bg), ldb},
                               {z, 0}, {reinterpret_cast<half_t*>(Dg), ldd},
                               {reinterpret_cast<const half_t*>(Bu), ldb}, {z, 0},
                               {reinterpret_cast<half_t*>(Du), ldd}, {reinterpret_cast<half_t*>(D), N},
                               {1.0f, 0.0f}, {1.0f, 0.0f}, {}, split);
    void* sem = split > 1 ? semaphores_for(stream) : nullptr;
    if (split > 1 && (sem == nullptr || G::get_workspace_size(args) > kSemaphoreBytes))
        return static_cast<int>(cudaErrorMemoryAllocation);
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, int, cudaStream_t>, std::unique_ptr<G>> ops;
    G* op = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto& slot = ops[std::make_tuple(M, N, K, split, stream)];
        if (!slot) {
            cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(stream, &cap);
            if (cap != cudaStreamCaptureStatusNone) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
            auto fresh = std::make_unique<G>();
            if (fresh->can_implement(args) != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorInvalidValue);
            if (fresh->initialize(args, sem, stream) != cutlass::Status::kSuccess)
                return static_cast<int>(cudaErrorUnknown);
            slot = std::move(fresh);
        } else if (slot->update(args, sem) != cutlass::Status::kSuccess) {
            return static_cast<int>(cudaErrorUnknown);
        }
        op = slot.get();
    }
    return op->run(stream) == cutlass::Status::kSuccess ? 0 : static_cast<int>(cudaErrorLaunchFailure);
}

constexpr int kDualTiles = 4;

int run_dual_tile(int tile, int split, const __half* A, const __half* Bg, const __half* Bu, int ldb, __half* Dg,
                  __half* Du, int ldd, __half* D, int M, int N, int K, cudaStream_t stream) {
    switch (tile) {
        case 0: return run_dual<DualT<GemmShape<128, 64, 32>, GemmShape<64, 32, 32>, 3>>(A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, split, stream);
        case 1: return run_dual<DualT<GemmShape<128, 128, 32>, GemmShape<64, 64, 32>, 3>>(A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, split, stream);
        case 2: return run_dual<DualT<GemmShape<64, 128, 32>, GemmShape<32, 64, 32>, 4>>(A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, split, stream);
        default: return run_dual<DualT<GemmShape<128, 64, 64>, GemmShape<64, 32, 64>, 3>>(A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, split, stream);
    }
}

bool args_ok(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, int act) {
    return A != nullptr && B != nullptr && D != nullptr && act >= 0 && act <= 2 && (act == 0 || bias != nullptr) &&
           M >= 1 && N % 8 == 0 && K % 8 == 0;
}


// FP16 operands, FP16 accumulation, BF16 output with a BF16 row bias (the
// source operand with a zero row stride): the flow head's cross K / V
// projections, whose consumers read BF16.
using bf16_t = cutlass::bfloat16_t;
template <class TB, class Warp, int Stages>
using GemmBf16Out = cutlass::gemm::device::Gemm<
    half_t, Row, half_t, Row, bf16_t, Row, half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp,
    GemmShape<16, 8, 16>, cutlass::epilogue::thread::LinearCombination<bf16_t, 8, half_t, float, ScaleType::NoBetaScaling>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, 8, 8>;

template <class G>
int run_bf16out(const __half* A, const __half* B, __nv_bfloat16* D, const __nv_bfloat16* bias, int M, int N, int K,
                cudaStream_t stream) {
    typename G::Arguments args({M, N, K}, {reinterpret_cast<const half_t*>(A), K},
                               {reinterpret_cast<const half_t*>(B), N}, {reinterpret_cast<const bf16_t*>(bias), 0},
                               {reinterpret_cast<bf16_t*>(D), N}, {1.0f, 0.0f});
    G op;
    if (op.can_implement(args) != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorInvalidValue);
    if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorUnknown);
    return op.run(stream) == cutlass::Status::kSuccess ? 0 : static_cast<int>(cudaErrorLaunchFailure);
}

}  // namespace

int gemm_f16acc_bf16out(const __half* A, const __half* B, __nv_bfloat16* D, const __nv_bfloat16* bias, int M, int N,
                        int K, int tile, cudaStream_t stream) {
    if (A == nullptr || B == nullptr || D == nullptr || bias == nullptr || M < 1 || N % 8 != 0 || K % 8 != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    switch (tile) {
        case 0: return run_bf16out<GemmBf16Out<GemmShape<128, 128, 32>, GemmShape<64, 64, 32>, 5>>(A, B, D, bias, M, N, K, stream);
        case 1: return run_bf16out<GemmBf16Out<GemmShape<128, 256, 32>, GemmShape<64, 64, 32>, 3>>(A, B, D, bias, M, N, K, stream);
        case 2: return run_bf16out<GemmBf16Out<GemmShape<64, 256, 32>, GemmShape<32, 64, 32>, 4>>(A, B, D, bias, M, N, K, stream);
        case 3: return run_bf16out<GemmBf16Out<GemmShape<128, 128, 64>, GemmShape<64, 64, 64>, 3>>(A, B, D, bias, M, N, K, stream);
        default: return static_cast<int>(cudaErrorInvalidValue);
    }
}

int gemm_f16acc_act(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, int act,
                    int split, cudaStream_t stream) {
    if (!args_ok(A, B, D, bias, M, N, K, act) || split > kMaxSplit || split * 64 > K)
        return static_cast<int>(cudaErrorInvalidValue);
    Ctx& c = ctx();
    const auto key = std::make_tuple(M, N, K, (act * 2 + (bias != nullptr)) * 16 + (split > 0 ? split : 0));
    int plan = -1;
    {
        std::lock_guard<std::mutex> lock(c.mu);
        const auto it = c.plan.find(key);
        if (it != c.plan.end()) plan = it->second;
    }
    if (plan < 0) {
        // first call (before graph capture): time every tile (and split count,
        // unless fixed) on the caller's buffers
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float best = 1e30f;
        for (int tile = 0; tile < kTiles; ++tile) {
            for (int sp = 1; sp <= kMaxSplit; ++sp) {
                if ((split > 0 && sp != split) || sp * 64 > K || (tile >= 8 && sp > 1)) continue;
                bool ok = true;
                for (int r = 0; r < 3 && ok; ++r) ok = run_act(act, tile, sp, A, B, D, bias, M, N, K, stream) == 0;
                if (!ok) continue;
                cudaEventRecord(e0, stream);
                for (int r = 0; r < 10; ++r) run_act(act, tile, sp, A, B, D, bias, M, N, K, stream);
                cudaEventRecord(e1, stream);
                cudaEventSynchronize(e1);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) {
                    best = ms;
                    plan = tile * 16 + sp;
                }
            }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        if (plan < 0) return static_cast<int>(cudaErrorNotSupported);
        std::lock_guard<std::mutex> lock(c.mu);
        c.plan[key] = plan;
    }
    return run_act(act, plan / 16, plan % 16, A, B, D, bias, M, N, K, stream);
}

int gemm_f16acc_act_cfg(const __half* A, const __half* B, __half* D, const __half* bias, int M, int N, int K, int act,
                        int tile, int split, cudaStream_t stream) {
    if (!args_ok(A, B, D, bias, M, N, K, act) || tile < 0 || tile >= kTiles || split < 1 || split > kMaxSplit)
        return static_cast<int>(cudaErrorInvalidValue);
    return run_act(act, tile, split, A, B, D, bias, M, N, K, stream);
}

int gemm_f16acc_swiglu(const __half* A, const __half* Bg, const __half* Bu, int ldb, __half* Dg, __half* Du, int ldd,
                       __half* D, int M, int N, int K, int split, cudaStream_t stream) {
    if (A == nullptr || Bg == nullptr || Bu == nullptr || Dg == nullptr || Du == nullptr || D == nullptr || M < 1 ||
        N % 8 != 0 || K % 8 != 0 || ldb % 8 != 0 || ldd % 8 != 0 || split > kMaxSplit || split * 64 > K)
        return static_cast<int>(cudaErrorInvalidValue);
    Ctx& c = ctx();
    const auto key = std::make_tuple(M, N, K, -1 - (split > 0 ? split : 0));
    int plan = -1;
    {
        std::lock_guard<std::mutex> lock(c.mu);
        const auto it = c.plan.find(key);
        if (it != c.plan.end()) plan = it->second;
    }
    if (plan < 0) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cap);
        if (cap != cudaStreamCaptureStatusNone) return static_cast<int>(cudaErrorStreamCaptureUnsupported);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float best = 1e30f;
        for (int t = 0; t < kDualTiles; ++t) {
            for (int sp = 1; sp <= kMaxSplit; ++sp) {
                if ((split > 0 && sp != split) || sp * 64 > K) continue;
                bool ok = true;
                for (int r = 0; r < 3 && ok; ++r)
                    ok = run_dual_tile(t, sp, A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, stream) == 0;
                if (!ok) continue;
                cudaEventRecord(e0, stream);
                for (int r = 0; r < 10; ++r) run_dual_tile(t, sp, A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, stream);
                cudaEventRecord(e1, stream);
                cudaEventSynchronize(e1);
                float ms = 0.f;
                cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) {
                    best = ms;
                    plan = t * 16 + sp;
                }
            }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        if (plan < 0) return static_cast<int>(cudaErrorNotSupported);
        std::lock_guard<std::mutex> lock(c.mu);
        c.plan[key] = plan;
    }
    return run_dual_tile(plan / 16, plan % 16, A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, stream);
}

int gemm_f16acc_swiglu_cfg(const __half* A, const __half* Bg, const __half* Bu, int ldb, __half* Dg, __half* Du,
                           int ldd, __half* D, int M, int N, int K, int tile, int split, cudaStream_t stream) {
    if (tile < 0 || tile >= kDualTiles || split < 1 || split > kMaxSplit) return static_cast<int>(cudaErrorInvalidValue);
    return run_dual_tile(tile, split, A, Bg, Bu, ldb, Dg, Du, ldd, D, M, N, K, stream);
}

}  // namespace lawam
}  // namespace flash_rt

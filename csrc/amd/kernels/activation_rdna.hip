// ================================================================
// FlashRT AMD -- BF16 activation kernels (RDNA 3.5, wave32)
//
// Implements the activation surface used by the Pi0.5 BF16 pipeline:
//   gelu:             out[i] = GELU_tanh(x[i])
//   gelu_mul:         out[i] = GELU_tanh(gate[i]) * up[i]
//   gelu_mul_merged:  same operation from row-major [gate | up]
//   silu:             out[i] = x[i] / (1 + exp(-x[i]))
//
// Inputs and outputs are BF16; transcendental and multiply operations use
// FP32 intermediates and round once at the output. Even-sized tensors use
// packed BF16x2 or BF16x4 accesses. Scalar kernels are retained for odd-size
// correctness rather than imposing alignment constraints on the raw ABI.
//
// These kernels are bandwidth/exp-pipeline bound and use no LDS, atomics, or
// cross-wave synchronization. The merged GeGLU form consumes the contiguous
// [rows, 2*width] result of the fused gate/up GEMM without materializing two
// independent activation tensors.
// ================================================================

#include "common_hip_rdna.h"

namespace {

struct alignas(8) RdnaBf16x4 {
  __hip_bfloat162 lo;
  __hip_bfloat162 hi;
};

// PyTorch's tanh approximation rewritten through exp. This preserves the
// approximation formula while avoiding the expensive device tanhf sequence
// emitted by the gfx11 toolchain.
__device__ __forceinline__ float gelu_tanh_rdna(float value) {
  // Algebraically identical to the tanh approximation used by PyTorch:
  // 0.5*x*(1+tanh(a)) == x/(1+exp(-2*a)). The native exponential maps to
  // a substantially cheaper gfx11 instruction sequence than device tanhf.
  const float exponent =
      -1.5957691216057308f * value * (1.0f + 0.044715f * value * value);
  return value / (1.0f + __expf(exponent));
}

// ── GELU ──

__global__ void gelu_rdna_kernel(const __hip_bfloat16 *__restrict__ input,
                                 __hip_bfloat16 *__restrict__ output,
                                 int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    output[index] =
        __float2bfloat16(gelu_tanh_rdna(__bfloat162float(input[index])));
  }
}

__global__ void
gelu_packed_rdna_kernel(const __hip_bfloat162 *__restrict__ input,
                        __hip_bfloat162 *__restrict__ output, int pairs) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < pairs) {
    const __hip_bfloat162 value = input[pair];
    output[pair] = __hip_bfloat162(
        __float2bfloat16(gelu_tanh_rdna(__bfloat162float(value.x))),
        __float2bfloat16(gelu_tanh_rdna(__bfloat162float(value.y))));
  }
}

// ── GeGLU: separate gate/up inputs ──

__global__ void gelu_mul_rdna_kernel(const __hip_bfloat16 *__restrict__ gate,
                                     const __hip_bfloat16 *__restrict__ up,
                                     __hip_bfloat16 *__restrict__ output,
                                     int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const float activated = gelu_tanh_rdna(__bfloat162float(gate[index]));
    output[index] = __float2bfloat16(activated * __bfloat162float(up[index]));
  }
}

__global__ void gelu_mul_packed_rdna_kernel(const RdnaBf16x4 *__restrict__ gate,
                                            const RdnaBf16x4 *__restrict__ up,
                                            RdnaBf16x4 *__restrict__ output,
                                            int items) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= items) {
    return;
  }
  const RdnaBf16x4 gate_value = gate[item];
  const RdnaBf16x4 up_value = up[item];
  RdnaBf16x4 result;
  result.lo = __hip_bfloat162(
      __float2bfloat16(gelu_tanh_rdna(__bfloat162float(gate_value.lo.x)) *
                       __bfloat162float(up_value.lo.x)),
      __float2bfloat16(gelu_tanh_rdna(__bfloat162float(gate_value.lo.y)) *
                       __bfloat162float(up_value.lo.y)));
  result.hi = __hip_bfloat162(
      __float2bfloat16(gelu_tanh_rdna(__bfloat162float(gate_value.hi.x)) *
                       __bfloat162float(up_value.hi.x)),
      __float2bfloat16(gelu_tanh_rdna(__bfloat162float(gate_value.hi.y)) *
                       __bfloat162float(up_value.hi.y)));
  output[item] = result;
}

// ── GeGLU: merged [gate | up] row layout ──

__global__ void
gelu_mul_merged_rdna_kernel(const __hip_bfloat16 *__restrict__ gate_up,
                            __hip_bfloat16 *__restrict__ output, int rows,
                            int width) {
  const int row = blockIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && column < width) {
    const size_t base = static_cast<size_t>(row) * 2 * width + column;
    const float gate = __bfloat162float(gate_up[base]);
    const float up = __bfloat162float(gate_up[base + width]);
    output[static_cast<size_t>(row) * width + column] =
        __float2bfloat16(gelu_tanh_rdna(gate) * up);
  }
}

__global__ void
gelu_mul_merged_packed_rdna_kernel(const __hip_bfloat162 *__restrict__ gate_up,
                                   __hip_bfloat162 *__restrict__ output,
                                   int rows, int width_pairs) {
  const int row = blockIdx.y;
  const int column_pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && column_pair < width_pairs) {
    const size_t base = static_cast<size_t>(row) * 2 * width_pairs;
    const __hip_bfloat162 gate = gate_up[base + column_pair];
    const __hip_bfloat162 up = gate_up[base + width_pairs + column_pair];
    output[static_cast<size_t>(row) * width_pairs + column_pair] =
        __hip_bfloat162(
            __float2bfloat16(gelu_tanh_rdna(__bfloat162float(gate.x)) *
                             __bfloat162float(up.x)),
            __float2bfloat16(gelu_tanh_rdna(__bfloat162float(gate.y)) *
                             __bfloat162float(up.y)));
  }
}

// ── SiLU ──

__global__ void silu_rdna_kernel(const __hip_bfloat16 *__restrict__ input,
                                 __hip_bfloat16 *__restrict__ output,
                                 int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    const float value = __bfloat162float(input[index]);
    output[index] = __float2bfloat16(value / (1.0f + __expf(-value)));
  }
}

__global__ void
silu_packed_rdna_kernel(const __hip_bfloat162 *__restrict__ input,
                        __hip_bfloat162 *__restrict__ output, int pairs) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair < pairs) {
    const __hip_bfloat162 packed = input[pair];
    const float first = __bfloat162float(packed.x);
    const float second = __bfloat162float(packed.y);
    output[pair] =
        __hip_bfloat162(__float2bfloat16(first / (1.0f + __expf(-first))),
                        __float2bfloat16(second / (1.0f + __expf(-second))));
  }
}

} // namespace

// ── Host dispatch ──
// Prefer packed paths whenever the tensor extent permits an aligned complete
// traversal; otherwise dispatch the scalar tail-safe implementation.

void gelu_rdna(const __hip_bfloat16 *input, __hip_bfloat16 *output, int size,
               hipStream_t stream) {
  constexpr int threads = 256;
  if ((size & 1) == 0) {
    const int pairs = size / 2;
    gelu_packed_rdna_kernel<<<(pairs + threads - 1) / threads, threads, 0,
                              stream>>>(
        reinterpret_cast<const __hip_bfloat162 *>(input),
        reinterpret_cast<__hip_bfloat162 *>(output), pairs);
    return;
  }
  gelu_rdna_kernel<<<(size + threads - 1) / threads, threads, 0, stream>>>(
      input, output, size);
}

void gelu_mul_rdna(const __hip_bfloat16 *gate, const __hip_bfloat16 *up,
                   __hip_bfloat16 *output, int size, hipStream_t stream) {
  constexpr int threads = 256;
  if ((size & 3) == 0) {
    const int items = size / 4;
    gelu_mul_packed_rdna_kernel<<<(items + threads - 1) / threads, threads, 0,
                                  stream>>>(
        reinterpret_cast<const RdnaBf16x4 *>(gate),
        reinterpret_cast<const RdnaBf16x4 *>(up),
        reinterpret_cast<RdnaBf16x4 *>(output), items);
    return;
  }
  gelu_mul_rdna_kernel<<<(size + threads - 1) / threads, threads, 0, stream>>>(
      gate, up, output, size);
}

void gelu_mul_merged_rdna(const __hip_bfloat16 *gate_up, __hip_bfloat16 *output,
                          int rows, int width, hipStream_t stream) {
  constexpr int threads = 256;
  if ((width & 1) == 0) {
    const int width_pairs = width / 2;
    const dim3 grid((width_pairs + threads - 1) / threads, rows);
    gelu_mul_merged_packed_rdna_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __hip_bfloat162 *>(gate_up),
        reinterpret_cast<__hip_bfloat162 *>(output), rows, width_pairs);
    return;
  }
  const dim3 grid((width + threads - 1) / threads, rows);
  gelu_mul_merged_rdna_kernel<<<grid, threads, 0, stream>>>(gate_up, output,
                                                            rows, width);
}

void silu_rdna(const __hip_bfloat16 *input, __hip_bfloat16 *output, int size,
               hipStream_t stream) {
  constexpr int threads = 256;
  if ((size & 1) == 0) {
    const int pairs = size / 2;
    silu_packed_rdna_kernel<<<(pairs + threads - 1) / threads, threads, 0,
                              stream>>>(
        reinterpret_cast<const __hip_bfloat162 *>(input),
        reinterpret_cast<__hip_bfloat162 *>(output), pairs);
    return;
  }
  silu_rdna_kernel<<<(size + threads - 1) / threads, threads, 0, stream>>>(
      input, output, size);
}

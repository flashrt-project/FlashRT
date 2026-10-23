# Flint Tensor operators on SM120

This optional producer exposes prequantized NVFP4 projections, dense SwiGLU,
routed/shared FFNs and paged decode attention as Torch Tensor operations.
[FlashRT Structures](https://github.com/flashrt-project/FlashRT-Structures)
owns host discovery, weight residency, attachment and rollback; vLLM retains
request scheduling and KV allocation. The existing
[standalone Flint frontend](flint_usage.md) remains a separate consumer.

## Build and install

Linux, SM120 / RTX 5090 and a CUDA-enabled PyTorch installation are required.
The measured cell uses CUDA 13.0 and PyTorch 2.13.0+cu130. Bring a matching
CUDA compiler and follow [INSTALL.md](INSTALL.md) for CUTLASS.

```bash
pip install -e ".[flint-ops]"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGPU_ARCH=120 -DFLASHRT_ENABLE_FLINT=ON
cmake --build build --target flash_rt_flint -j
# Optional full-attention prefill producer; explicitly build before an engine is created.
python -m flash_rt.native_kernels.sage2 --build
```

`FLASHRT_ENABLE_FLINT` is OFF by default and accepts only `GPU_ARCH=120`.
The producer does not alter `flash_rt_kernels`. Sage2 compilation is an
explicit setup command, never a side effect of import, attachment or forward.
Its binary is installed beside `flash_rt`; `--source-root <csrc/sage2>` may
identify the checked-out source during an explicit build. A stale or missing
binary refuses with a rebuild instruction.

## API and contracts

```python
from flash_rt.native_kernels.nvfp4 import PackedMatrix, Linear
matrix = PackedMatrix(codes, scales, global_scale)
projection = Linear(matrix)  # resolves ABI/device/symbols and validates weights
y = projection(x)
```

| Surface | Contract |
|---|---|
| `PackedMatrix` | CUDA contiguous uint8 codes `[N,K/2]`, adjacent E2M1 nibbles; uint8 E4M3 scale bytes `[N,K/16]`; finite positive global scale; N multiple of 128, K multiple of 64 |
| `Linear` | BF16 contiguous `[M,K]` → BF16 `[M,N]` |
| `DenseFfn` | Interleaved gate/up (groups of gate8/up8), precise SiLU, then down; shape relationships validated at bind |
| `RoutedSharedFfn` | Coupled sigmoid/top6 router, 256 experts, hidden 2048, expert intermediate 512, ungated shared intermediate 1024; capacity 1–8192 rows |
| `LmHead` | Packed NVFP4 vocabulary projection with FP32 logits |
| `sage2.attention_fp8` | BF16 Q `[Tq,Hq,128]`, K/V `[Tk,Hk,128]`, `Tk>=Tq`, `Hq % Hk == 0`; causal prefix offset `Tk-Tq`, optional per-head sink; BF16 result |

ABI 1 and the hardware/profile metadata live in `flash_rt_flint.TENSOR_API_METADATA`.
The Python producer validates all required symbols before the first execution.
`Linear`, `DenseFfn`, `LmHead` and `RoutedSharedFfn` reject calls unless the
activation and every weight tensor share the same CUDA device. Routed/shared
per-expert global scales and scratch must also be on that device. Device
mismatches raise `ValueError` before entering CUDA execution.
Input owners, packed weights and graph buffers must outlive every replay.
Sharing scratch supports one sequential execution lane; independent concurrent
lanes require separate scratch. Packing and dynamic activation quantization
belong to the producer, not to a host adapter's calibration traversal.

## Execution forms

The default W4A16 path retains checkpoint NVFP4 codes and scales. Optional
W4A8 uses int8 Tensor Cores and dynamic per-32 activation scales; the M=1
path remains W4A16. Dense FP8 and grouped FP4 prefill forms introduce further
rounding and require the matching vLLM binary operators. They are not described
as lossless transformations. FP8 KV uses the host's unit-scale cache contract.

The supported serving recipe and measured batch 1–8 cell are in the companion
[Flint/vLLM guide](https://github.com/flashrt-project/FlashRT-Structures/blob/feat/flint-w4a8-lm-head/docs/flint_vllm.md).
Configuration is fixed before importing this module. The entire vLLM route is
opt-in; importing `flash_rt` or leaving the plugin disabled does not select it.

| Producer switch | Default | Meaning |
|---|---:|---|
| `FLASHRT_NATIVE_A8` | 0 | Enable W4A8 where the row/shape policy admits it |
| `FLASHRT_NATIVE_A8_MIN_M` / `A8_DENSE_MIN_M` | 2 / 192 | Minimum W4A8 activation/dense rows |
| `FLASHRT_NATIVE_A8_GU_MIN_M` | 1000000000 | Grouped gate/up W4A8 threshold; effectively disabled until selected |
| `FLASHRT_NATIVE_FP4_MOE_MIN_M` | 1000000000 | Grouped FP4 prefill threshold |
| `FLASHRT_NATIVE_DENSE_FP8_MIN_M` | 1000000000 | Dense FP8 prefill threshold |
| `FLASHRT_NATIVE_FP4_SF_CACHE` | 1 | Cache packed weight scale-factor layouts at bind; disable for the measured memory-constrained recipe |
| `FLASHRT_NATIVE_GEMV_MMA_MAX` | 16 | Exact few-row W4A16 MMA band |
| `FLASHRT_NATIVE_RESIDENT` | 1 | Exact resident-activation policy for qualified 9–16-row shapes |
| `FLASHRT_NATIVE_SIDE_STREAM` | 1 | Overlap the shared expert on a producer-owned stream |

Other tuning and fake-quantization switches in `nvfp4.py` are experimental,
not part of the recommended serving recipe. In particular `STREAM_MAX_M`,
`SPLITK`, `FAKE_A4_*`, `FAKE_DENSE_FP8_MIN_M`, `A8_PER_ROW` and diagnostic
dump/check flags remain disabled unless explicitly selected. Experimental
packed K-FP8/V-FP4 cache bindings are not qualified by the FP8-KV measurements.

## Validation

```bash
PYTHONPATH=. pytest -q tests/test_flint_contracts.py tests/test_flint_tensor_api.py
PYTHONPATH=. pytest -q tests/test_flint_kernels.py
FLASHRT_FLINT_GGUF="<official-nvfp4.gguf>" PYTHONPATH=. pytest -q tests/test_flint_pipeline.py
```

The Tensor tests cover stale ABI/symbol/device refusal, W4A16/W4A8 numerical
references, changed-input graph replay, and Sage2 causal-prefix/sink attention.
The checkpoint suite also exercises the existing standalone frontend. Tests
that require CUDA or a checkpoint skip explicitly when those are absent.

## Sources and credit

- [RecursiveSI Flint](https://huggingface.co/RecursiveSI/flint) and the
  [official NVFP4 checkpoint](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf): model and weights.
- [FlashRT](https://github.com/flashrt-project/FlashRT) and
  [FlashRT Structures](https://github.com/flashrt-project/FlashRT-Structures): producers and host integration.
- [SageAttention](https://github.com/thu-ml/SageAttention) and
  [FlashRT-HF-kernels](https://github.com/flashrt-project/FlashRT-HF-kernels/tree/059d28393239c8fec04699bccaed6f4c88d25f24/sageattention2-blackwell): adapted attention cores; see [vendor notices](../csrc/sage2/VENDOR.md).
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer),
  [CUTLASS](https://github.com/NVIDIA/cutlass), [Triton](https://github.com/triton-lang/triton)
  and [vLLM](https://github.com/vllm-project/vllm): attention helpers, GEMM primitives, Tensor producers and serving.

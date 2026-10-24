# Qwen3-4B BF16 prefill

Batch-1 causal prefill of [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
to logits, in BF16, matching HuggingFace `Qwen3ForCausalLM`. This is a text
decoder, so it is constructed directly rather than through `load_model`.

Scope: batch 1, prefill to logits (all rows, or the last `logits_to_keep`),
BF16 weights and activations. No decode loop, KV-cache output, sampling or
quantization.

| | Qwen3-4B (this doc) | Qwen3-8B NVFP4 (`qwen3_8b_nvfp4.md`) |
|---|---:|---:|
| hidden / intermediate | 2560 / 9728 | 4096 / 12288 |
| layers | 36 | 36 |
| GQA | 32q / 8kv / hd=128 | same |
| weights | BF16 | NVFP4 W4A4 |

## Targets

One layer sequence (`flash_rt/models/qwen3/prefill_bf16.py`) serves every
device. A target fixes the device generation it was validated on and binds
the attention backend; each has its own frontend class and `_PIPELINE_MAP`
entry, and a frontend refuses any other device.

| target | frontend | attention | build |
|---|---|---|---|
| `rtx_sm120` (RTX 50-series) | `qwen3_4b_rtx.Qwen3_4BTorchFrontendRtx` | FA2 causal (`flash_rt_fa2.fwd_bf16_causal`, split-KV heuristic) | `GPU_ARCH=120`: `flash_rt_kernels`, `flash_rt_fa2` |
| `rtx_sm89` (RTX 40-series) | `qwen3_4b_rtx.Qwen3_4BTorchFrontendRtxSm89` | FA2 causal, as SM120 | `GPU_ARCH=89`: `flash_rt_kernels`, `flash_rt_fa2`; with CUDA < 12.8 add `-DFA2_ARCH_NATIVE_ONLY=ON` |
| `thor` (Jetson AGX Thor) | `qwen3_4b_thor.Qwen3_4BTorchFrontendThor` | FlashAttention-4 forward (`hardware/thor/fa4_backend.py`) | `GPU_ARCH=110`: `flash_rt_kernels`; FA4 runtime deps from the `thor-fa4` extra |

The norm, RoPE and SiLU kernels launch with programmatic dependent launch on
SM120 and Thor; SM89 has no `griddepcontrol` and launches them plainly.

Construction checks the device's compute capability, every
`flash_rt_kernels` entry the sequence launches, and loads and runs the
attention backend once; a missing module, symbol or instantiation is an
error there, never a fallback to another implementation.

## Usage

```bash
cmake -B build -S . -DGPU_ARCH=120          # 89 on RTX 40-series, 110 on Thor
cmake --build build --target flash_rt_kernels flash_rt_fa2 -j   # Thor: flash_rt_kernels
```

```python
from flash_rt.frontends.torch.qwen3_4b_rtx import Qwen3_4BTorchFrontendRtx
# RTX 40-series: Qwen3_4BTorchFrontendRtxSm89 from the same module
# Thor: from flash_rt.frontends.torch.qwen3_4b_thor import Qwen3_4BTorchFrontendThor

fe = Qwen3_4BTorchFrontendRtx("<Qwen3-4B checkpoint dir>", device="cuda:0", max_seq=2048)
logits = fe.forward_prefill(input_ids)                    # [S, vocab] bf16, new tensor
last = fe.forward_prefill(input_ids, logits_to_keep=1)    # [1, vocab]
hidden = fe.forward_hidden(input_ids)                     # [S, hidden], after the final norm
```

* `input_ids`: `[S]` or `[1, S]`. A CUDA int64/int32 tensor on the instance's
  device is copied on the device (no host round-trip, so ids are not
  range-checked); a CPU tensor or a list is checked against the vocabulary.
  `S <= max_seq`.
* Work runs in the instance's device context on that device's current stream.
* Weights, activations and attention scratch are allocated once for
  `max_seq`. With `cuda_graphs=True` (default) each prompt length is captured
  once (after one eager pass that builds the GEMM plans) and replayed; the
  last `graph_cache_size` lengths are kept. The LM head runs outside the
  graph into a fresh output tensor.
* `rope_inv_freq` names the HF baseline's RoPE frequency precision (below).
* `autotune_gemms=True` (default `False`) lets each GEMM shape run a faster
  cuBLASLt candidate that matched the heuristic first choice bit for bit on
  random probe operands (activations and weights drawn by the kernel,
  independent of the model) and then on the call's own operands. Split-K
  and other re-ordered reductions fail that check, but it is a check over
  the probes, not a proof for every input: the strict default is what the
  bit-for-bit statement below covers.

## Numerical contract

Every stage except attention reproduces the HF result bit for bit:

* **GEMMs.** q, k, v, gate and up are separate projections, as in HF.
  `bf16_matmul_cublaslt_ref_bf16` runs cuBLASLt's heuristic first choice --
  the algorithm torch's matmul runs at these shapes -- so the GEMMs are HF's
  bit for bit. HF and FlashRT share the process's `libcublasLt` (torch's
  copy, loaded first).
* **RMSNorm / residual / q-k norm + RoPE / SiLU gate**: the `*_hf_bf16`
  kernels round where HF rounds (`w * bf16(x * rsqrt(var + eps))`; each RoPE
  product rounded before the add; `bf16(bf16(silu(g)) * u)`; the norm reduces
  the bf16 residual) and sum squares in PyTorch's reduction order, including
  the wider block torch uses below 16 rows. PyTorch 2.10 changed that order
  (shuffle direction, vectorized 128-wide rows); the tree is chosen from the
  running PyTorch.
* **RoPE tables** are computed with HF's expression (`inv_freq` built on the
  CPU in fp32, `inv_freq @ positions` in fp32 on the device, cos / sin cast to
  bf16). `rope_inv_freq="fp32"` matches `from_pretrained(dtype=torch.bfloat16)`;
  `"bf16"` matches a model cast with `.to(torch.bfloat16)`, which rounds the
  `inv_freq` buffer and moves every angle at long positions.
* **Attention** is the one stage that is close to rather than equal to the
  reference: FA2 with the split-KV heuristic of torch's flash SDPA on SM120
  and SM89, FA4 on Thor.

Declared HF baseline: `Qwen3ForCausalLM.from_pretrained(path, dtype=torch.bfloat16)`
moved to the device with `.to()`, default attention implementation (`sdpa`,
which selects torch's flash backend on every target), fp32 matmul precision
left at its default.

Full-logits cosine is input-dependent. A strongly repetitive prompt (one
sentence repeated) makes the BF16 reference itself fragile: on SM120,
replacing only HF's attention with FA2 already gives 0.9976 at S=256 and
0.9953 at S=1024 there, while top-1 predictions still agree. Natural text and
random token ids stay above 0.9997.

## Validation

`tests/test_qwen3_4b_hf_cosine.py` and `tests/test_qwen3_4b_hf_checkpoint.py`
(gate: full-logits cosine >= 0.999 vs HF), `tests/test_rms_norm_hf_bf16.py`
and `tests/test_bf16_matmul_cublaslt_ref.py` (bitwise vs the torch ops):

```bash
PYTHONPATH=. pytest -q tests/test_qwen3_4b_hf_cosine.py tests/test_rms_norm_hf_bf16.py \
    tests/test_bf16_matmul_cublaslt_ref.py
QWEN3_4B_CHECKPOINT=<dir> PYTHONPATH=. pytest -q tests/test_qwen3_4b_hf_checkpoint.py
```

Full-logits cosine vs HF, CUDA graphs on:

| S | SM89 random | SM89 ckpt | SM120 random | SM120 ckpt | Thor random | Thor ckpt |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 8 | 0.999945 | 0.999903 | 0.999910 | 0.999848 | 0.999717 | 0.999851 |
| 64 | 0.999526 | 0.999652 | 0.999652 | 0.999668 | 0.999587 | 0.999764 |
| 256 | 0.999397 | 0.999793 | 0.999511 | 0.999832 | 0.999378 | 0.999847 |
| 1024 | 0.999501 | 0.999837 | 0.999532 | 0.999837 | 0.999209 | 0.999826 |
| 2048 | 0.999475 | 0.999544 | 0.999517 | 0.999708 | 0.999171 | 0.999721 |

Random: `Qwen3ForCausalLM(config).to(bf16)`, seed 0, random token ids
(`rope_inv_freq="bf16"`), minimum over 3 id draws. Checkpoint:
Qwen/Qwen3-4B, natural-text prompts (`rope_inv_freq="fp32"`), minimum over
2 prompts. Each length is captured once and replayed for the later draws;
replay equals eager bit for bit. S=1 is bit-identical to HF. The Thor margin
at long prompts is FA4's: with every other stage exact, the cosine equals that
of HF with only its attention replaced by FA4.

| | SM89 | SM120 | Thor |
|---|---|---|---|
| GPU | GeForce RTX 4090 | GeForce RTX 5090 | Jetson AGX Thor (L4T R39.2.1) |
| PyTorch | 2.5.1+cu124 | 2.9.0a0 (CUDA 13.0) | 2.14.0+cu130 |
| transformers | 4.55.0 | 5.3.0 | 4.55.4 |
| cuBLASLt (in process) | 12.4.5 (PyTorch wheel) | 13.1.0 | 13.1.1 (PyTorch wheel) |
| build | `GPU_ARCH=89`, `FA2_ARCH_NATIVE_ONLY=ON` (CUDA 12.4) | `GPU_ARCH=120` | `GPU_ARCH=110`; FA4: nvidia-cutlass-dsl 4.5.1, quack-kernels 0.4.1 |

## Performance

`benchmarks/qwen3_4b_latency.py`: same checkpoint on both sides, full logits
(embed + 36 layers + final norm + LM head over every row), CUDA events,
median of 20 after 5 warm-up calls, ms.

HF / FlashRT with the default (strict) GEMMs, speedup:

| S | RTX 4090 (SM89) | RTX 5090 (SM120) | Jetson Thor |
|---:|---:|---:|---:|
| 8 | 30.17 / 10.50 (2.87x) | 12.95 / 7.87 (1.64x) | 39.68 / 33.95 (1.17x) |
| 128 | 31.37 / 12.12 (2.59x) | 13.49 / 7.32 (1.84x) | 50.01 / 38.45 (1.30x) |
| 512 | 37.28 / 29.92 (1.25x) | 28.71 / 21.73 (1.32x) | 82.19 / 49.55 (1.66x) |
| 1024 | 68.16 / 57.05 (1.19x) | 52.43 / 43.70 (1.20x) | 165.21 / 94.35 (1.75x) |
| 2048 | 139.16 / 119.55 (1.16x) | 101.20 / 87.01 (1.16x) | 338.41 / 163.92 (2.06x) |

With `autotune_gemms=True`, Jetson Thor runs 33.90 / 38.09 / 50.68 / 73.88 /
162.33 ms at S = 8 / 128 / 512 / 1024 / 2048 (2.24x HF at S=1024).

Where the time goes: from S=512 on SM120 the GEMMs run at ~200 TFLOPS (and
~150 TFLOPS on SM89 at S=1024), so the prefill is GEMM-bound and both sides
run the same GEMM algorithms. Below that the GEMMs stream weights at cuBLAS's
small-M efficiency; FlashRT's gain there is the fused norm / RoPE / SiLU
kernels, FA2's split-KV and graph replay, and on hosts where HF's eager
dispatch is slow (the SM89 machine above) the graph replay dominates. On the
RTX targets `autotune_gemms=True` changes nothing measurable. On Thor, FA4
halves attention time at long prompts and the norm and SiLU passes run at
Thor's DRAM bandwidth; cuBLASLt's first choice around S=1024 runs at ~104
TFLOPS, which `autotune_gemms=True` lifts to ~141 TFLOPS with probe-checked
rather than guaranteed GEMM bits. On SM120 and Thor the norm, RoPE and SiLU
kernels launch with programmatic dependent launch; the GEMMs they follow
(cuBLASLt) expose no PDL trigger, so the overlap it buys is within
run-to-run noise here.

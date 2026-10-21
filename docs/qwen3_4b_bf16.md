# Qwen3-4B BF16 on FlashRT

FlashRT BF16 text path for [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).

This is **not** the Qwen3-8B NVFP4 path (`docs/qwen3_8b_nvfp4.md`) and **not**
Qwen3.5-VLA. Geometry:

| | Qwen3-4B (this doc) | Qwen3-8B NVFP4 |
|---|---:|---:|
| hidden | 2560 | 4096 |
| intermediate | 9728 | 12288 |
| layers | 36 | 36 |
| GQA | 32q / 8kv / hd=128 | same |
| quant | BF16 | NVFP4 W4A4 |

## Build

Reuse a Python env with matching CUDA (example: conda `py310`). On RTX 4090:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
pip install -e . --no-deps   # or pip install -e ".[torch]" if deps resolve
cmake -B build -S . -DGPU_ARCH=89 -DFA2_ARCH_NATIVE_ONLY=ON \
  -DPython3_EXECUTABLE=$(which python)
cmake --build build -j$(nproc)
```

`FA2_ARCH_NATIVE_ONLY=ON` avoids compiling Blackwell `sm_121` FA2 slices that
need a newer nvcc than some CUDA 12.8 installs provide.

## How to run

From the `flashrt/` repo root (CUDA + built kernels required):

```bash
export PYTHONPATH=$PWD:$PYTHONPATH

# HF cosine parity (random weights, no Hub download): S=8 and S=1024
pytest tests/test_qwen3_4b_hf_cosine.py -q -s

# Prefill latency vs HuggingFace (random shared state_dict)
python benchmarks/qwen3_4b_latency.py --seq 1024 --warmup 3 --runs 10

# Optional: also time the all-fvk path
python benchmarks/qwen3_4b_latency.py --seq 1024 --warmup 3 --runs 10 --all-fvk
python benchmarks/qwen3_4b_latency.py --seq 1024 --warmup 3 --runs 10 --compare-paths
```

Smoke with a real checkpoint directory (HF layout):

```python
from flash_rt.frontends.torch.qwen3_4b_rtx import Qwen3_4BTorchFrontendRtx

fe = Qwen3_4BTorchFrontendRtx("/path/to/Qwen3-4B", device="cuda:0")
logits = fe.forward_prefill([1, 2, 3, 4])   # [S, vocab] bf16
```

## HF cosine (random weights)

The cosine test builds `Qwen3ForCausalLM.from_config`, copies the random
`state_dict` into `Qwen3_4BTorchFrontendRtx`, runs prefill at seq=8 and
seq=1024, and asserts full-sequence logits cosine ≥ 0.999 vs HuggingFace.

Measured on RTX 4090 after the RoPE / split-QKV fix (default path). Full
write-up: [`qwen3_4b_bf16_precision_fix.md`](qwen3_4b_bf16_precision_fix.md).

| seq | full-seq cosine vs HF | gate (≥ 0.999) |
|---:|---|:---:|
| 8 | ≥ 0.999 | pass |
| 64 / 128 | ≥ 0.999 | pass |
| 256 | ≥ 0.999 (10 seeds, min ≈ 0.99921) | pass |
| 512 / 1024 / 2048 | ≥ 0.999 (1024 can reach 1.0) | pass |

Latency for this default path is in
[`qwen3_4b_e2e_latency_hf_vs_flashrt.md`](qwen3_4b_e2e_latency_hf_vs_flashrt.md)
(RTX 4090, S=512 1.04×, S=1024 1.01× vs HF). Thor SDPA numbers
(S=8/128/800 cosine and S=128/800 latency) are in
[`qwen3_4b_thor.md`](qwen3_4b_thor.md).

### Precision-sensitive ops

Leave-one-out at S=256/1024 (with bf16-aligned RoPE tables) showed only
`use_hf_qkn_rope` and `use_sep_qkv` are required for cosine ≥ 0.999.
Other toggles default to the FlashRT fast path:

| Flag | Default | Path |
|---|---|---|
| `use_sdpa` | `False` | FA2 `fwd_bf16_causal` |
| `use_hf_norm` | `False` | fvk `rms_norm` / `residual_add_rms_norm` |
| `use_hf_qkn_rope` | **`True`** | HF q/k RMSNorm + `rotate_half` RoPE |
| `use_sep_gateup` | `False` | fused `gu` + `silu_mul_qwen36_bf16` |
| `use_sep_qkv` | **`True`** | separate q/k/v GEMMs |

Embedding + o/down/lm_head GEMM stay on FlashRT.

RoPE tables mirror HF after `model.to(bfloat16)`: `inv_freq` is
round-tripped through bf16 before building cos/sin. Without this, long-seq
cosine vs HF collapses (~0.96 at S=1024). Details:
`docs/qwen3_4b_bf16_precision_fix.md`.

```python
# All-fvk speed path (may drop mid/long-seq cosine below 0.999):
fe_fast = Qwen3_4BTorchFrontendRtx(
    "/path/to/Qwen3-4B", device="cuda:0",
    use_hf_qkn_rope=False, use_sep_qkv=False,
)
```

Registered as `("qwen3_4b", "torch", "rtx_sm89"|"rtx_sm120")` in
`flash_rt.hardware._PIPELINE_MAP`.

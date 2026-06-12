# Qwen3.6-27B NVFP4 on DGX Spark

This document covers the FlashRT Spark path for Qwen3.6-27B NVFP4 on
NVIDIA GB10 / SM121. It is an additive frontend over the RTX Qwen3.6
path: existing RTX and Thor frontends keep their own dispatch and kernel
defaults.

For the general Qwen3.6 NVFP4 model contract, K selection background,
and parameter reference, see [`qwen36_nvfp4.md`](qwen36_nvfp4.md) and
[`qwen36_usage.md`](qwen36_usage.md).

## Requirements

- DGX Spark / GB10 GPU, compute capability SM121.
- FlashRT built with `GPU_ARCH=121`.
- Qwen3.6-27B NVFP4 main checkpoint.
- Paired Qwen3.6 FP8 MTP checkpoint containing `mtp.safetensors`.
- Single-GPU inference. Batch size is still 1.

The measured path uses the public NVFP4 main checkpoint plus a paired
FP8 MTP checkpoint converted to NVFP4 at load time. The Spark frontend
adds the NVFP4 MTP prompt-tail K/V prefill needed by this checkpoint
layout; it does not require BF16 shadow MTP projection weights.

## Build

```bash
cmake -B build-spark-sm121 -S . -DGPU_ARCH=121
cmake --build build-spark-sm121 -j
pip install -e ".[torch]"
```

The SM121 build enables the same NVFP4/CUTLASS, FA2, FP8-KV XQA, and
decode GEMV objects used by the Qwen3.6 path, compiled for Spark.

## Direct Frontend

```python
import os
import torch

from flash_rt.frontends.torch.qwen36_spark import Qwen36TorchFrontendSpark

os.environ["FLASHRT_QWEN36_MTP_CKPT_DIR"] = "/models/Qwen3.6-27B-FP8-MTP"
os.environ["FLASHRT_QWEN36_LONG_KV_CACHE"] = "fp8"

fe = Qwen36TorchFrontendSpark(
    "/models/Qwen3.6-27B-NVFP4",
    quant="nvfp4",
    max_seq=32768,
)

prompt = "Explain CUDA graphs and speculative decoding briefly."
ids = fe._tokenizer(prompt, return_tensors="pt").input_ids.to(fe.device)

out = fe.generate_own_speculative_KN_nvfp4(
    ids,
    max_new_tokens=64,
    K=6,
)
print(fe._tokenizer.decode(out[0, ids.shape[1]:].tolist()))
```

## Agent Server

The OpenAI-compatible agent server auto-selects the Spark frontend on
SM121:

```bash
pip install -e ".[torch,server]"

export FLASHRT_QWEN36_MTP_CKPT_DIR=/models/Qwen3.6-27B-FP8-MTP
export FLASHRT_QWEN36_LONG_KV_CACHE=fp8

python -m serving.qwen36_agent.server \
  --checkpoint /models/Qwen3.6-27B-NVFP4 \
  --model-name qwen36-27b \
  --max-seq 32768 \
  --port 8000
```

On SM121 the server leaves the SM120 hand-tuned fastgemm/warpsplit
decode kernels opt-in by default, because Spark profiling favored the
default CUTLASS NVFP4 path. Exact-position long-decode graphs are also
disabled by default for agent serving, matching the existing agent
latency policy.

## Tuned Spark Policy

Spark uses a measured policy for the long-context speculative route:

| prompt ctx | effective K | MTP tail | FP8-KV XQA |
|---:|---:|---:|:---:|
| 128 | 6 | 128 | on |
| 2 K | 6 | 2048 | on |
| 8 K | 7 | 4096 | off |
| 16 K | 6 | 4096 | on |
| 24 K | 7 | 2048 | on |
| 32 K | 7 | 2048 | on |

The policy is encoded in `Qwen36TorchFrontendSpark`; callers can still
override `FLASHRT_QWEN36_TQ_SPEC_K`,
`FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL`, or FP8-XQA env vars for
experiments.

## Performance

Measured on a single DGX Spark / GB10 (SM121), Qwen3.6-27B NVFP4 main
checkpoint plus FP8 MTP checkpoint, `max_new_tokens=64`, repeated text
prompt, warm-state decode, long KV cache in FP8. Decode tok/s excludes
prefill.

| prompt ctx | effective K | MTP tail | TTFT / prefill | decode tok/s | AL |
|---:|---:|---:|---:|---:|---:|
| 128 | 6 | 128 | 170.1 ms | 40.42 | 3.57 |
| 2 K | 6 | 2048 | 1.784 s | 40.11 | 3.64 |
| 8 K | 7 | 4096 | 5.160 s | 36.14 | 3.50 |
| 16 K | 6 | 4096 | 8.545 s | 54.94 | 5.30 |
| 24 K | 7 | 2048 | 11.077 s | 57.73 | 6.22 |
| 32 K | 7 | 2048 | 15.110 s | 57.32 | 6.67 |

The 8 K bucket is acceptance-length limited on this prompt distribution:
changing only XQA or the MTP tail does not recover the 16 K / 24 K AL.
The checked-in Spark policy keeps the best measured 8 K setting while
preserving the stronger long-context buckets.

## Reproducing the Table

```bash
PYTHONPATH=. python benchmarks/qwen36_spark_al_sweep.py \
  --frontend spark \
  --model /models/Qwen3.6-27B-NVFP4 \
  --mtp-dir /models/Qwen3.6-27B-FP8-MTP \
  --ctx 128,2048,8192,16384 \
  --max-new 64 \
  --K auto \
  --tails auto \
  --tail-kv-only 1 \
  --prompts repeat \
  --warmup 1 \
  --reps 2 \
  --max-seq 32768 \
  --long-kv-cache fp8 \
  --out qwen36_spark_al_repeat_128_16k_auto_policy.csv
```

For 24 K / 32 K:

```bash
PYTHONPATH=. python benchmarks/qwen36_spark_al_sweep.py \
  --frontend spark \
  --model /models/Qwen3.6-27B-NVFP4 \
  --mtp-dir /models/Qwen3.6-27B-FP8-MTP \
  --ctx 24576,32768 \
  --max-new 64 \
  --K auto \
  --tails auto \
  --tail-kv-only 1 \
  --prompts repeat \
  --warmup 1 \
  --reps 1 \
  --max-seq 65536 \
  --long-kv-cache fp8 \
  --out qwen36_spark_al_24k_32k_auto_policy.csv
```

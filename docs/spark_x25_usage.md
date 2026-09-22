# Spark-X2.5-4B — Parameter Reference

Spark-X2.5-4B is a hybrid-attention chat model: 36 layers, 27 with a 512-token
sliding window and 9 full-attention, NVFP4 weights, 1M-token position range.
Batch is 1 and every steady-state shape is fixed, so the whole decode loop is
captured into one CUDA Graph.

See `docs/spark_x25_rtx.md` for the measurements, the KV residency modes and
their correctness cost.

## Installation

The kernels build into their own module, `flash_rt_sparkx25`, gated on SM120:

```bash
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
cmake -B build -S . -DGPU_ARCH=120
cmake --build build --target flash_rt_kernels flash_rt_fa2 flash_rt_sparkx25 -j4
```

`flash_rt_kernels` supplies the NVFP4 W4A4 GEMMs and `flash_rt_fa2` the prefill
attention; both are required. The build prints
`Spark-X2.5-4B kernels: ENABLED (separate module flash_rt_sparkx25)` when the
third target is configured. On any other `GPU_ARCH` it is skipped and the
frontend will not resolve.

## Constructor

```python
from flash_rt.frontends.torch.spark_x25_rtx import SparkX25TorchFrontendRtx

fe = SparkX25TorchFrontendRtx("/models/Spark-X2.5-4B-NVFP4", max_seq=131072)
```

| argument | default | meaning |
|---|---|---|
| `checkpoint` | — | checkpoint directory; validated for 36 layers, head_dim 256, a 4:1 GQA group and a 512-token window |
| `max_seq` | `32768` | KV capacity and the largest position. Sizes the caches and **chooses the KV residency mode** (below) |
| `prefill_cap` | `min(max_seq, 8192)` | largest single prefill forward; the prompt is walked in chunks of `prefill_chunk` |
| `prefill_chunk` | `min(prefill_cap, 2048)` | rows per chunk. Sizes the activation working set and every sliding layer's linear cache |
| `device` | `"cuda"` | |

The frontend is constructed directly rather than through `flash_rt.api.load_model`:
`load_model` wraps models in the VLA surface, which a text decoder does not have.
Qwen3.6 and Qwen3-VL are constructed the same way.

## Generation

```python
import torch
ids = fe.tokenizer("用一句话解释什么是量子纠缠。", return_tensors="pt")["input_ids"][0]
out = fe.generate(ids, max_new_tokens=128)          # prompt + new token ids
text = fe.tokenizer.decode(out[len(ids):], skip_special_tokens=True)

# or, rendering the chat template for you:
print(fe.generate_text("用一句话解释什么是量子纠缠。", max_new_tokens=128))
```

`generate` prefills, captures a decode graph of `max_new_tokens` steps, replays
it, and returns the ids. `graph_steps=` overrides the captured step count.
There is no sampling: the model's own `argmax_bf16` kernel runs on device, so
nothing synchronises the host inside the loop.

## KV residency and `max_seq`

Two full-attention KV layouts exist and the constructor picks between them from
the free device memory:

| mode | full layers hold | prefill reads | long-context logit cosine |
|---|---|---|---|
| `bf16+E4M3` | bf16 K+V plus an E4M3 copy for decode | exact bf16 | ~0.999 |
| `E4M3 only` | E4M3 K+V | a bf16 staging buffer expanded from E4M3 | ~0.99 |

`fe.kv8_only` reports which one is active. At 131 072 tokens on a 16 GB part the
first is chosen (12.1 GiB resident); at 262 144 the second is (11.0 GiB, 4.2
free) because the bf16 cache and its E4M3 copy do not both fit — the mirror
alone needs 4.77 GiB more than is free. The second mode is what makes a 262k
window run at 46.7 tok/s instead of 31.9; it costs the long-context cosine.

**E4M3 is only used for decode in the first mode.** Decode-time quantisation is
free against the frozen delivery metric, which is computed from the prefill's
last row; quantising what *prefill* reads is not, which is why the two modes
differ at all.

## Greedy decode rate

Measured on one RTX 5060 Ti 16 GB (SM120), batch 1, repeated-text prompt,
128 decode steps in a captured graph
(`benchmarks/spark_x25_rtx_latency.py`):

| prompt ctx | 128 | 512 | 2 048 | 32 k | 64 k | 128 k | 256 k |
|---|---|---|---|---|---|---|---|
| TTFT (ms) | 14 | 29 | 114 | 3 457 | 10 628 | 36 105 | 132 762 |
| decode (tok/s) | 138.7 | 136.4 | 135.9 | 111.3 | 92.7 | 69.7 | 46.7 |

## Known limits

- **SM120 only.** The module is gated on `GPU_ARCH=120`; the tile sizes were
  chosen against that part's shared-memory budget and the KV writer quantises
  with `__nv_cvt_float_to_fp8`.
- **Batch 1.** The capture is one query row wide. A batch would need a second
  capture at that batch.
- **No speculative decoding.** The checkpoint has no MTP or draft head and
  FlashRT ships no drafter for it, so long-context decode cannot amortise its
  once-per-step KV read over several tokens the way Qwen3.6's MTP path does.
  See `docs/spark_x25_rtx.md` for what that costs on the context curve.
- **The E4M3 load path is ~18% off the bf16 one.** Bounded and measured; the
  receipt is in `docs/spark_x25_rtx.md`.

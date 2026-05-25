# Qwen3.6-27B NVFP4 on RTX 5090

This document covers the FlashRT NVFP4 inference path for Qwen3.6-27B,
including model dependencies, the speculative-decode `K` selection, real
measured throughput data, and reproduction commands. Numbers in this
doc were measured on RTX 5090 (sm_120, 32 GB HBM, BW 1.79 TB/s).

For the full per-parameter API reference (constructor, generate args,
env vars), see [`qwen36_usage.md`](qwen36_usage.md). For an OpenAI-API
compatible HTTP server, see
[`examples/qwen36_openai_server.py`](../examples/qwen36_openai_server.py).

## 0. Quickstart

The minimum to run Qwen3.6-27B NVFP4 with K=6 speculative decode at
~134 tok/s decode on the short standard prompt. Step 1 is one-time; step 2 is the
inference call.

Install the Torch frontend extra before building/running Qwen. The
long-context runtime uses native FlashRT CUDA/CUTLASS kernels and does
not require Triton/FLA Python kernels:

```bash
pip install -e ".[torch]"
# Add the server extra too if you use examples/qwen36_openai_server.py:
# pip install -e ".[torch,server]"
```

```python
# 1) Build the kernels (one-time, from the FlashRT repo root)
#    cmake -S . -B build && cmake --build build -j --target flash_rt_kernels
#    flash_rt_kernels*.so lands directly in flash_rt/ — no manual cp.

# 2) Inference
import os, torch
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

# Tell the frontend where to find the FP8 ckpt's mtp.safetensors
# (see §1 below on why).
os.environ['FLASHRT_QWEN36_MTP_CKPT_DIR'] = '<path-to-FP8-ckpt-dir>'

fe = Qwen36TorchFrontendRtx(
    '<path-to-NVFP4-ckpt-dir>',   # prithivMLmods/Qwen3.6-27B-NVFP4
    quant='nvfp4',
)

prompt = 'Explain quantum entanglement in one short paragraph.'
input_ids = fe._tokenizer(prompt, return_tensors='pt').input_ids.cuda()

out = fe.generate_own_speculative_KN_nvfp4(
    input_ids, max_new_tokens=256, K=6,
)

text = fe._tokenizer.decode(out[0, input_ids.shape[1]:].tolist())
print(text)
```

`K=6` is the recommended default (peaks at ~134 tok/s decode on RTX
5090 for short prompts). For sustained / long generations, drop to
`K=5` or `K=3` — see §3 for the full curve.

## 1. Model dependencies

The NVFP4 inference path needs **two** checkpoints:

| Role | Format | Source |
|---|---|---|
| Main model | NVFP4 W4A16 (`compressed-tensors` `nvfp4-pack-quantized`) | [`prithivMLmods/Qwen3.6-27B-NVFP4`](https://huggingface.co/prithivMLmods/Qwen3.6-27B-NVFP4) |
| MTP head | FP8 e4m3 block-128 or community BF16/native `mtp.safetensors` | paired Qwen3.6-Next-27B MTP ckpt |

Pass the main NVFP4 ckpt directory as the `checkpoint_path` argument
to `Qwen36TorchFrontendRtx`. The NVFP4 ckpt does **not** ship an MTP
module — `compressed-tensors` strips it — so the ckpt directory that
contains `mtp.safetensors` is loaded separately via the
`FLASHRT_QWEN36_MTP_CKPT_DIR` environment variable. FP8-source MTP is
converted FP8 → BF16 → NVFP4 once at load (no FP8 in the hot path).
BF16/native MTP checkpoints keep BF16 projection weights by default for
better drafter alignment; set `FLASHRT_QWEN36_MTP_KEEP_BF16=0` to force
the lower-memory NVFP4-converted path. Without the env var, MTP is None
and speculative decode is disabled (pure single-token decode still
works at ~36 tok/s).

### Can I use a different NVFP4 Qwen3.6 checkpoint?

Yes, **as long as** all four conditions hold:

1. **Architecture** = Qwen3.6-Next 27B (`num_hidden_layers=64`,
   `num_v_heads=48`, `hidden_size=5120`, head_dim=128, vocab 248,320).
   The frontend hard-codes these for buffer allocation.
2. **Quantization** = `compressed-tensors` `nvfp4-pack-quantized` with
   the standard schema:
   - `<prefix>.weight_packed` u8 `(out, in/2)` (FP4 e2m1, 2-per-byte)
   - `<prefix>.weight_scale` fp8_e4m3 `(out, in/16)` (per-block-16 SF)
   - `<prefix>.weight_global_scale` fp32 scalar (= `448 / amax`)
3. **Quant scope** = MLP gate/up/down + full-attn q/k/v/o.
   Linear-attn projections (`in_proj_qkv`, `in_proj_z`, `out_proj`,
   `in_proj_a`, `in_proj_b`, `conv1d`) **must stay BF16** — that's
   what the lin-attn kernels expect.
4. **MTP head from a paired FP8 ckpt**. Without it, speculative decode
   is unavailable; pure-decode (no spec) still works at ~36 tok/s.

If any of those is different (e.g. AWQ, GGUF, GPTQ, MXFP4, full-tensor
NVFP4 of lin-attn), the loader will reject or produce wrong outputs.

The loader source of truth is [`flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py`](../flash_rt/frontends/torch/_qwen36_rtx_nvfp4_weights.py)
— see the module docstring for the exact key list.

## 2. Headline numbers (decode tok/s)

Decode tok/s = `(N_OUT - 1) × 1000 / decode_time_ms`. **Excludes**
prefill (TTFT). Same metric vLLM and TensorRT-LLM report.

Single representative prompt (`"Explain quantum entanglement in one
short paragraph."`, 11 tokens), max_new_tokens=128, default `K=6`:

```
TTFT (prefill)        :   ~233 ms      (one-shot, doesn't recur)
TPOT                  :   ~7.45 ms/token
★ decode tok/s        :   134.18

spec stats: K=6  attempts=29  p_full=0.345  p_ind=0.575  AL=4.38
```

`AL=4.10` means each spec cycle emits 4.1 tokens on average; `p_full`
is the fraction of cycles where the full draft chain is accepted.

## 3. Choosing `K` (speculative chain length)

`K` is the MTP draft chain length per spec cycle. Verify processes
`K+1` tokens, the spec loop accepts the longest matching prefix. The
right `K` depends on prompt distribution and target output length.

### Measured K-curve (single prompt, NTOK=128)

```
 K   decode tok/s   AL    p_ind
 3   119.23         3.17  0.733
 4   112.68         3.17  0.556   (drafter trough)
 5   124.15         3.74  0.559
 6   129.44         4.10  0.522   ★ peak (NTOK=128)
 7   119.16         3.97  0.429   (rolls off)
```

Why `K=6` wins at NTOK=128: AL keeps growing through `K=6` faster than
verify cost grows; at `K=7` drafter `p_ind` drops far enough that AL
plateaus while verify cost dominates. `K=4` is a local minimum because
`p_ind` crashes from 0.733 to 0.556 at the new "deepest" position
without enough total AL gain to offset.

### Length sensitivity

Drafter quality decays as generation goes on (drift from the original
prompt distribution). Longer outputs → smaller `K` becomes safer.

```
                    NTOK=128   NTOK=256   NTOK=512
 K=3  decode tok/s  119.23     113.66     113.98
 K=5  decode tok/s  124.15     117.37     114.40
 K=6  decode tok/s  129.44     109.68     112.34   ← peak shifts
 K=7  decode tok/s  119.16     107.28     110.65
```

Peak shifts from `K=6` (NTOK=128) to `K=5` (NTOK=256), and by NTOK=512
all values converge around 113. For workloads with mostly short outputs
(< 256 tokens), `K=6` is best; for sustained generation, `K=5` is more
robust.

### Per-prompt variance (5 prompts × 2 NTOK)

Speculative decode is sensitive to prompt-text distribution. The
drafter aligns better with structured prompts (math, code) than
free-form ones (creative writing).

```
    prompt  NTOK   K   prompt_len   decode tok/s     AL  p_full
   explain   128   3      11             119.11   3.17   0.575
   two_sum   128   3      41             110.28   2.95   0.558
      sort   128   3      22             115.85   3.10   0.512
      math   128   3      17             122.01   3.26   0.538
   summary   128   3      19              94.90   2.54   0.220
   explain   128   6      11             128.87   4.10   0.290
   two_sum   128   6      41             110.69   3.53   0.139
      sort   128   6      22             102.29   3.26   0.051
      math   128   6      17             117.32   3.74   0.000
   summary   128   6      19              83.15   2.65   0.000
   explain   256   3      11             113.41   3.04   0.512
   two_sum   256   3      41             119.08   3.19   0.613
      sort   256   3      22             123.82   3.31   0.623
      math   256   3      17             130.58   3.49   0.671
   summary   256   3      19             103.49   2.77   0.348
   explain   256   6      11             109.48   3.49   0.123
   two_sum   256   6      41             121.05   3.86   0.152
      sort   256   6      22             115.83   3.70   0.101
      math   256   6      17             131.09   4.18   0.098
   summary   256   6      19              96.28   3.07   0.060
```

Aggregate (mean ± CV across 5 prompts):

| NTOK | K | min | median | max | mean | CV |
|---|---|---:|---:|---:|---:|---:|
| 128 | 3 | 94.90 | 115.85 | 122.01 | 112.43 | 9.5% |
| 256 | 3 | 103.49 | 119.08 | 130.58 | 118.07 | 8.7% |
| 128 | 6 | 83.15 | 110.69 | 128.87 | 108.46 | 15.8% |
| 256 | 6 | 96.28 | 115.83 | 131.09 | 114.74 | 11.3% |

**Reading the table:**
- `K=6` peak (131.09 on math/256) > `K=3` peak (130.58) — captures the
  best case, which is the headline-worthy number.
- `K=6` mean ≈ `K=3` mean across diverse prompts. The expected speedup
  for "any prompt" is roughly even.
- `K=6` is **more sensitive** to prompt distribution (CV 11-16% vs
  K=3's 9%). Predictable workloads (single prompt class) favor K=6;
  varied workloads favor K=3.
- "summary"-style creative prompts are 1.6× slower than math/code
  prompts at the same K. This is normal speculative-decoding
  variance — not a bug.

### Recommendation

| Workload | Suggested K |
|---|:---:|
| Mostly short generations (≤ 256 tokens), single prompt class | **6** (default) |
| Mixed workloads, longer generations | **5** |
| Ultra-conservative (tightest variance) | **3** |

Set via `TEST_K=<n>` env var when running `standard_bench`, or pass
`K=<n>` to `generate_own_speculative_KN_nvfp4`.

## 4. Long-context throughput

When `max_seq` is above the long-context threshold, the frontend keeps
a small BF16 KV/spec working window and allocates a compressed KV cache
for requests routed beyond it. The default compressed route is FP8 KV;
TurboQuant remains available for memory/accuracy bisection. The default
long-route threshold is 512 prompt tokens, preserving the peak BF16/spec
decode path for very short prompts while using chunked FP8-KV for
512-token and larger prompts. Short prompts route to FP8-KV only if the
full request exceeds the retained BF16 window.

Spec decode is wired to the long-ctx compressed-KV path. Short requests
that fit in the retained BF16 window still use the normal CUDA-Graph
MTP spec path; longer requests use MTP draft plus compressed-KV verify.
The default long-context route is `FLASHRT_QWEN36_LONG_KV_CACHE=fp8`,
matching the community-style e4m3 FP8 KV serving direction. That path
stores persistent full-attn KV as FP8. On SM120, long verify attention
uses the vendored FlashInfer XQA native FP8-KV kernel once the KV length
passes the measured `FLASHRT_QWEN36_FP8_XQA_MIN_CTX=auto` bucket policy,
and keeps the older FP8->BF16-stage + FA2 bridge in buckets where that
path measured faster.
Set `FLASHRT_QWEN36_LONG_KV_CACHE=tq` to use the TurboQuant
packed path for memory/accuracy bisection.
The current measured FP8-KV warm decode table, including 256K, is in
the TTFT section below.
The long TQ/spec path uses measured K buckets by default: `3` below 6K
prompt tokens, `4` from 6K to 12K, `5` from 12K to 24K, `4` from 24K
to 48K, `7` from 48K to below 160K, and `6` elsewhere, then
adaptively drops from K≥4 to `K=3` inside a request when early accept
stats show a low-hit prompt; set
`FLASHRT_QWEN36_TQ_SPEC_K` to force a fixed K. TQ verify and the MTP
draft chain are CUDA-Graph captured per `(cur_pos, K)` in warm state.
Long-context MTP prompt-tail prefill seeds the drafter's private K/V
cache with 1024 prompt-tail rows for 12K+ prompts by default, which
keeps long-context draft acceptance materially higher without full MTP
prompt prefill.
Long-context prefill uses the same TQ S=K forward in chunks (default
chunk size = `MAX_Q_SEQ`, currently 2048) instead of one full forward
per prompt token; full-attention layers use the vendored FA2 causal
hdim=256 path for one q_seq=S attention call per chunk, and
linear-attention layers use the native FlashRT WY/cuBLASLt Gated
DeltaNet scan for prefill chunks. Long-context NVFP4 defaults to the
fused MLP gate/up GEMM when the checkpoint's gate/up scales allow it;
the separate non-widen tile remains available by setting
`FLASHRT_QWEN36_FUSE_MLP_GATE_UP=0`.
Linear-attention A/B projections use a deterministic fused AB96 BF16
kernel in prefill chunks, and the default Gated DeltaNet prefill route
uses `FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt` with the f32 GEMM
chunk-state update. Intermediate prompt chunks skip final-norm/lm-head
logits entirely, and the final chunk computes logits only for the last
prompt row. The large K-row logits workspace is allocated lazily only
for explicit all-logits diagnostic calls. Set
`FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=native` to force the direct-conv
FlashRT chunk scan for bisection.

## 5. TTFT (prefill latency)

The recommended full-context serving profile keeps the 2048-row BF16
working window for chunk size, uses FP8 persistent KV, and routes
512-token and larger prompts through the chunked compressed-KV path:

```bash
export FLASHRT_QWEN36_LONG_KV_CACHE=fp8
export FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ=512
```

Warm measurements on RTX 5090, `max_new_tokens=64`, repeated text
prompt, one same-shape warmup generation followed by one timed
generation. The 128-token row uses BF16/spec because it is still the
highest decode-throughput route there; 512 tokens and above use FP8-KV.

| prompt ctx | route | K | MTP tail | TTFT / prefill | prefill tok/s | decode ms | decode tok/s | spec attempts / accepts / full |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | BF16/spec | 6 | full | 2.686 s | 47.7 | 440.3 | 145.4 | 14 / 49 / 7 |
| 512 | FP8-KV | 4 | 512 | 72.6 ms | 7,057 | 530.4 | 120.7 | 19 / 45 / 9 |
| 1 K | FP8-KV | 5 | 2048 | 131.9 ms | 7,762 | 603.8 | 106.0 | 20 / 44 / 6 |
| 2 K | FP8-KV | 6 | 2048 | 253.8 ms | 8,070 | 388.2 | 164.9 | 12 / 52 / 6 |
| 4 K | FP8-KV | 3 | 512 | 350.9 ms | 11,673 | 494.0 | 129.6 | 19 / 45 / 11 |
| 8 K | FP8-KV | 5 | 2048 | 774.2 ms | 10,581 | 482.4 | 132.7 | 16 / 48 / 4 |
| 16 K | FP8-KV | 7 | 2048 | 1.570 s | 10,437 | 364.9 | 175.4 | 10 / 54 / 4 |
| 32 K | FP8-KV | 6 | 2048 | 3.553 s | 9,223 | 425.4 | 150.4 | 13 / 51 / 5 |
| 64 K | FP8-KV | 7 | 2048 | 9.133 s | 7,176 | 424.3 | 150.9 | 12 / 51 / 2 |
| 128 K | FP8-KV | 7 | 2048 | 26.731 s | 4,903 | 405.0 | 158.0 | 11 / 52 / 3 |
| 200 K | FP8-KV | 6 | 2048 | 56.921 s | 3,598 | 576.0 | 111.1 | 15 / 49 / 4 |
| 256 K | FP8-KV | 6 | 2048 | 87.976 s | 2,980 | 442.7 | 144.6 | 11 / 52 / 4 |

`decode tok/s` excludes TTFT and is the TPOT-style LLM serving metric.
The low-TTFT FP8-KV route is available below 512 tokens, but it trades
away decode throughput on the measured prompt distribution.

FP8 KV also improves prefill versus the TurboQuant cache, but the gain
is modest until very large contexts because prefill is dominated by the
full prompt forward. Local FP8-vs-TQ prefill deltas:

| ctx | FP8 prefill | TQ prefill | FP8 gain |
|---:|---:|---:|---:|
| 4 K | 320.6 ms | 329.9 ms | 2.8% |
| 16 K | 1.506 s | 1.557 s | 3.3% |
| 64 K | 9.073 s | 9.451 s | 4.0% |
| 128 K | 26.70 s | 27.78 s | 3.9% |
| 200 K | 56.75 s | 63.60 s | 10.8% |
| 256 K | 87.55 s | 99.63 s | 12.1% |

The table is measured with the public frontend API below: run one
same-shape warmup generation, then time a second generation with CUDA
events around the prefill and decode windows. Developer-only micro-bench
probes live outside the tracked public tree.

## 6. Reproduction

Build (from the FlashRT repo root):

```bash
cmake -S . -B build
cmake --build build -j --target flash_rt_kernels
# flash_rt_kernels*.so lands in flash_rt/ via CMake's
# LIBRARY_OUTPUT_DIRECTORY — no manual cp needed.
```

Run (replace `<NVFP4_CKPT>` with the `prithivMLmods/Qwen3.6-27B-NVFP4`
directory and `<FP8_CKPT>` with the directory that contains the FP8
ckpt's `mtp.safetensors`):

```python
import torch
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

import os
os.environ['FLASHRT_QWEN36_MTP_CKPT_DIR'] = '<FP8_CKPT>'
fe = Qwen36TorchFrontendRtx('<NVFP4_CKPT>', quant='nvfp4')

prompt = 'Explain quantum entanglement in one short paragraph.'
input_ids = fe._tokenizer(prompt, return_tensors='pt').input_ids.cuda()
out = fe.generate_own_speculative_KN_nvfp4(
    input_ids, max_new_tokens=128, K=6)
print(fe._tokenizer.decode(out[0, input_ids.shape[1]:].tolist()))
```

Recommended runtime env vars:

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Developer-only micro-bench probes are intentionally kept out of the
tracked public tree.

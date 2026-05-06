# Qwen3-8B-NVFP4 on RTX 5090

FlashRT inference path for [Qwen3-8B-Instruct](https://huggingface.co/Qwen/Qwen3-8B-Instruct)
in NVFP4 W4A4 on a single RTX 5090 (sm_120, 32 GB), with an
OpenAI-compatible HTTP server on top.

Designed for: low-latency single-stream chat / agentic / tool-calling
workloads where time-to-first-token (TTFT) under 1k-token prompts
and steady decode throughput both matter.

For the framework-level intro see [`../README.md`](../README.md);
for general install + Pi0 / Pi0.5 setup see
[`../USAGE.md`](../USAGE.md).

---

## 1. Headline performance

```
RTX 5090 / sm_120 / 32 GB · driver 580.x · NVFP4 W4A4 ckpt
ckpt: JunHowie/Qwen3-8B-Instruct-2512-SFT-NVFP4
```

| Metric | HF SDPA baseline | FlashRT |
|---|---:|---:|
| TTFT  P=64    (graph)   | 280 ms | **9.1 ms** |
| TTFT  P=256   (graph)   | 295 ms | **11.1 ms** |
| TTFT  P=512   (graph)   | 315 ms | **14.2 ms** |
| TTFT  P=1024  (graph)   | 366 ms | **24.8 ms** |
| Decode warm graph       | 3.6 tok/s | **150 tok/s** |
| OAI server warm tok/s   | n/a | **150 tok/s** |
| VRAM @ P=1024 + N=256   | 5.99 GiB | 7.30 GiB |

The OAI server warm tok/s lands at the engine's standalone bench —
the FastAPI / uvicorn / asyncio / SSE / per-token decode layers add
no measurable overhead at this throughput class.

---

## 2. Quick start

```bash
# 1. Start the server (after the FlashRT install in ../USAGE.md)
python examples/qwen3_openai_server.py \
    --checkpoint /path/to/Qwen3-8B-Instruct-2512-SFT-NVFP4 \
    --port 8000

# 2. Call it like any OpenAI v1 endpoint
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "qwen3-8b-nvfp4",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "max_tokens": 128
    }'
```

Startup is ~9 s (3 s checkpoint load + 6 s graph warmup). The server
is ready when uvicorn logs `Uvicorn running on http://0.0.0.0:8000`.

```python
# Or via the OpenAI Python SDK
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8000/v1', api_key='not-required')

resp = client.chat.completions.create(
    model='qwen3-8b-nvfp4',
    messages=[{'role': 'user', 'content': 'Hi'}],
    max_tokens=128,
)
print(resp.choices[0].message.content)
```

The Qwen3-8B path is **text-only** — no images, no tokenizer file
to install separately. Pi0 / Pi0.5 paths have additional one-time
prereqs; see [`../USAGE.md`](../USAGE.md).

---

## 3. Inference architecture

```
                    Qwen3-8B-Instruct-2512-SFT-NVFP4
                       36 layers · all full-attn
                       hidden 4096 · head_dim 128
                       GQA 32Q / 8KV · interm 12288
                       vocab 151,936

  FlashRT frontend                       Backend kernels
  ─────────────────                      ─────────────────
  embed_tokens (BF16)
  for L in 36:
    ┌─ rms_norm + nvfp4 quant                  (one fused launch)
    ├─ qkv NVFP4 W4A4 GEMM, fused N=6144       (one launch, M=1 decode)
    ├─ q_norm + RoPE + Q_buf write             (fused, head-parallel)
    ├─ k_norm + RoPE + K/V cache write         (fused, head-parallel)
    ├─ FA2 attention (decode q_seq=1 / prefill causal q_seq=N)
    ├─ o_proj NVFP4 W4A4 GEMM                  (one launch)
    ├─ residual + post_norm + nvfp4 quant      (one fused launch)
    ├─ gate+up NVFP4 W4A4 GEMM, fused N=24576  (one launch)
    ├─ silu(gate) * up + nvfp4 quant           (one fused launch)
    ├─ down NVFP4 W4A4 GEMM                    (one launch)
    └─ residual_2 + (next-layer) input_norm    (fused at boundary)
  final RMSNorm + lm_head BF16 mat-vec
  ↓
  logits → sampler  (greedy / top-k+top-p multinomial / seeded)
```

Decode hot-path uses **5 NVFP4 W4A4 GEMMs per layer + 1 BF16
mat-vec lm_head + 1 attention** (vs 7 NVFP4 GEMMs naive). Prefill
(S=N) uses CUTLASS NVFP4 W4A16 instead of the M=1 W4A4 MMA — that's
the right tile economics at S>1.

Per-layer ops outside GEMMs are aggressively fused:

- Pre-projection `rms_norm + activation NVFP4 quant`: one kernel.
- Post-attention `residual + RMSNorm + activation NVFP4 quant`:
  one kernel; reused at the layer boundary (post-MLP residual_2 +
  next layer's input_norm + quant) so the inter-layer transition
  is also a single launch.
- Per-head `q_norm + RoPE + Q_buf write` and
  `k_norm + RoPE + K_cache + V_cache write` are each one kernel —
  collapsing what was previously a 14-op chain (RMSNorm + 6-op
  RoPE × 2 + 3 KV/Q copies) into 2 launches per layer.
- `silu(gate) * up + nvfp4 swizzled quant`: one kernel, no bf16
  intermediate round-trip through HBM.

`lm_head` stays BF16 by design — see §7.

---

## 4. Required checkpoint schema

The path consumes a `compressed-tensors` `nvfp4-pack-quantized`
checkpoint with the following per-linear schema:

```
self_attn.{q,k,v,o}_proj            NVFP4 W4A4
mlp.{gate,up,down}_proj             NVFP4 W4A4
input_layernorm                     BF16  (plain RMSNorm)
post_attention_layernorm            BF16
self_attn.q_norm / k_norm           BF16  (per-head RMSNorm)
embed_tokens / model.norm           BF16
lm_head                             BF16  (in ckpt's `ignore` list)
```

Per-linear NVFP4 fields the loader expects:

```
weight_packed         u8        (out, in/2)
weight_scale          fp8_e4m3  (out, in/16)            linear layout
weight_global_scale   fp32      (1,)
input_global_scale    fp32      (1,)        captured but unused at runtime
```

The fused QKV / fused gate+up GEMM dispatch relies on the
calibration producing bit-identical `weight_global_scale` for
{q,k,v} and {gate,up} within every layer — verified across all 36
layers of the JunHowie ckpt (per-set max relative diff = 0). The
loader has a homogeneity check and falls back to per-linear
weights if a future ckpt breaks the invariant.

---

## 5. OpenAI-compatible HTTP server

`examples/qwen3_openai_server.py`. Drop-in replacement for an
OpenAI v1 base URL.

### Endpoints

- `POST /v1/chat/completions` (non-stream + `stream: true` SSE)
- `GET /v1/models` (returns one canonical model id)
- `GET /health`

### Request fields

| Field | Notes |
|---|---|
| `model` | accepted but not enforced (server hosts a single model) |
| `messages` | required; roles `system` / `user` / `assistant` / `tool` |
| `tools` | OpenAI tools spec — passed through to the chat template |
| `tool_choice` | model-driven (Qwen3 template decides) |
| `max_tokens` | default 256 |
| `temperature` | default 0.0 (greedy + deterministic) |
| `top_p` | default 1.0 |
| `top_k` | default 0 (off) |
| `seed` | optional, deterministic when set |
| `stop` | string or list; matched on the model output and stripped |
| `stream` | boolean — when true, response is SSE chunks |

### Tool calling

Native via the Qwen3 chat template. The tokenizer's
`apply_chat_template(messages, tools=...)` injects the tool schema
into the system prompt; the model emits `<tool_call>...</tool_call>`
blocks whose contents are JSON of shape
`{"name": "...", "arguments": {...}}`.

`StreamParser` splits the assistant stream into:

- free-text content deltas, and
- complete tool-call dicts in OpenAI shape:
  `{index, id, type:"function", function:{name, arguments(string)}}`.

A `tool_call` is emitted only after the closing `</tool_call>` is
observed AND the JSON parses successfully — partial JSON is never
streamed out.

### Tool-result round-trip

Standard OpenAI flow works:

```python
# Round 1 — model emits tool_call
resp1 = client.chat.completions.create(
    model='qwen3-8b-nvfp4',
    messages=[{'role': 'user', 'content': 'Weather in Tokyo?'}],
    tools=[...],
)
tc = resp1.choices[0].message.tool_calls[0]

# Run the tool, then send the result back
resp2 = client.chat.completions.create(
    model='qwen3-8b-nvfp4',
    messages=[
        {'role': 'user', 'content': 'Weather in Tokyo?'},
        {'role': 'assistant', 'content': None,
         'tool_calls': [tc.model_dump()]},
        {'role': 'tool', 'tool_call_id': tc.id,
         'content': '{"temp_c": 22, "condition": "sunny"}'},
    ],
    tools=[...],
)
# resp2.choices[0].message.content
#   → "The weather in Tokyo is 22°C and sunny."
```

### Server flags

```bash
python examples/qwen3_openai_server.py \
    --checkpoint /path/to/ckpt \
    --port 8000 \
    --host 0.0.0.0 \
    --max-seq 2048 \
    --max-q-seq 128 \
    --device cuda:0 \
    --model-name qwen3-8b-nvfp4 \
    --warmup 32:128,128:256
```

| Flag | Default | Notes |
|---|---|---|
| `--checkpoint` | (required) | Path to NVFP4 ckpt directory |
| `--port` | 8000 | |
| `--host` | 0.0.0.0 | |
| `--max-seq` | 2048 | Total prompt + generation length budget |
| `--max-q-seq` | 128 | Max prefill chunk |
| `--warmup` | `32:128,128:256` | Comma-separated `prompt_len:max_tokens` pairs to pre-capture at startup. Set to `""` to skip. |

Pre-startup `--warmup` pre-captures CUDA Graphs for the listed
shapes plus all six prefill buckets (32 / 64 / 128 / 256 / 512 /
1024) — so the first real request hits warm graphs immediately. A
prompt that lands outside the listed `(prompt_len, max_tokens)`
shapes pays a one-time graph capture (~10-20 ms) on the first hit
at that shape; subsequent hits at the same shape are warm.

### Concurrency

Single-process, single-GPU, batch = 1. Concurrent HTTP requests
are serialised by an `asyncio.Lock`. Multi-tenant fan-out belongs
in an external router (e.g., several FlashRT instances behind a
load balancer).

VRAM at default `max_seq=2048` is ~7-8 GiB, fitting comfortably
on a 32 GiB 5090.

---

## 6. Correctness

Every release of this path is gated on an automated test suite
covering both the model body and the OAI HTTP surface:

| Check | Floor | Scope |
|---|---|---|
| Layer-0 hidden cosine vs HF reference | ≥ 0.999 | NVFP4 W4A4 first-layer numerics |
| Logits cosine vs HF reference | ≥ 0.985 | full 36-layer cumulative + lm_head |
| Greedy first-token argmax MATCH | required | argmax decisions match HF SDPA |
| Greedy 32-token byte match | ≥ 16 / 32 | first ~8 tokens deterministic, then synonym drift OK |
| Prefill ≡ S=1 ingest loop | argmax MATCH | the captured prefill graph and the per-token loop yield identical first sampled token |
| TTFT eager / graph buckets | listed in §1 | per `prompt_len` bucket |
| Decode tok/s warm graph | listed in §1 | low run-to-run variance (±2 tok/s) |
| OpenAI surface | non-stream + SSE | both arms hit; `Content-Type` / `Cache-Control` headers OK |
| Tool calling | valid JSON + `finish_reason='tool_calls'` | model emits well-formed `tool_calls[]` deltas |
| CUDA Graph determinism | `max_diff = 0.0` | 100 replays at fixed `cur_pos` |

---

## 7. Notes & limitations

- **`lm_head` stays BF16.** A native NVFP4 W4A4 lm_head was
  evaluated and reverted: at the 152K-class output argmax, the W4A4
  noise compounds over decode and dropped greedy 32-token match
  noticeably with no measurable BW saving on this kernel layout. A
  custom large-N small-K matvec specifically tuned for the lm_head
  shape (M=1, N≈152K, K=4096) could potentially recover the BW gap,
  but the headroom is small (~3-5 %) given the current path is
  already at ~91 % HBM BW efficiency on this op.
- **Prompts longer than the largest prefill bucket (1024)** fall
  back to eager prefill — still correct, ~10-15 % slower TTFT. To
  serve longer prompts at warm-graph speed, extend the bucket
  ladder:
  `--prefill-buckets 32,64,128,256,512,1024,2048`.
- **No `prompt_logprobs` / `logprobs`**, no `n > 1`, no
  `logit_bias` / `presence_penalty` / `frequency_penalty`. Greedy
  or sampled-token output only.
- **No multi-modal.** Text in, text out.
- **No KV-cache offload.** Prompt + generation must fit in
  `max_seq`.
- **No automatic continuous batching.** This path is built for
  single-stream low-latency; multi-tenant batching belongs in an
  external router.

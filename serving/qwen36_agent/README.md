# serving/qwen36_agent

Production-oriented Qwen3.6-27B NVFP4 serving example for long-running agent
sessions.

This directory is the **policy layer** above the FlashRT execution contract. It
owns session cache, exact token-prefix reuse, OpenAI-compatible tool calling,
streaming, and request scheduling. It must not add session or KV verbs to
`exec/`; the contract remains Buffer / Graph / Plan / Event / ShapeKey.

The execution-state **capsule** feature (cold-prefill a shared prefix once, then
restore instead of re-prefill on later turns) is documented in
[`capsules.md`](capsules.md); this server exposes session prefix reuse, and the
capsule API lives on the frontend.

## Quickstart (end-to-end, reproducible)

**Prerequisites**

- A CUDA GPU (developed on RTX 5090, sm_120) and the FlashRT runtime built/installed
  (`pip install -e ".[torch]"`, then the CMake build — see the repo `docs/INSTALL.md`).
- The Qwen3.6 NVFP4 checkpoint directory (the model weights) and, for speculative
  decode, the MTP checkpoint. Point the server at the NVFP4 directory.
- Server-only Python deps: `pip install fastapi uvicorn`.

**1. Start the server**

```bash
python -m serving.qwen36_agent.server \
  --checkpoint /path/to/qwen36_nvfp4 \
  --model-name qwen36-27b \
  --host 127.0.0.1 --port 8000
# startup loads the model, then logs: Uvicorn running on http://127.0.0.1:8000
```

**2. Check it is up**

```bash
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/health      # model, max_seq, live sessions
```

**3. A chat completion (OpenAI-compatible)**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen36-27b",
  "messages": [{"role": "user", "content": "Write a Python one-liner to reverse a string."}],
  "max_tokens": 128,
  "flashrt_session_id": "demo"
}'
```

The response is an OpenAI `chat.completion` with an extra `flashrt` block of serving
telemetry (see [Response fields](#response-fields)).

**4. Streaming (Server-Sent Events)**

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen36-27b", "stream": true,
  "messages": [{"role": "user", "content": "Explain a hash map in two sentences."}],
  "max_tokens": 128, "flashrt_session_id": "demo"
}'
# emits `data: {chat.completion.chunk}` lines, then `data: [DONE]`
```

Tokens are streamed only after they are committed to the session state (committed
decode), so the visible transcript never runs ahead of the GPU state.

## Design target

- 256K context on the existing Qwen3.6 long-context FP8-KV/TQ kernel path.
- Latency-first, single-stream hot session by default.
- Exact token-prefix reuse for coding-agent turns: cold prefill once, then only
  prefill appended user/tool/diff/log tokens.
- Direct long-decode kernel launch by default for live agent sessions, so a
  growing conversation does not pay exact-position CUDA Graph capture in the
  first real request. Fixed-shape graph replay remains opt-in for demos and
  benchmarks.
- True SSE streaming at speculative-decode accept boundaries.
- Streamed tokens are session-committed tokens only. The old stateless
  full-generate shortcut of over-verifying and trimming output is forbidden in
  this host because it would leave hidden KV state ahead of the client-visible
  transcript.
- OpenAI-compatible tool calls without leaking partial `<tool_call>` JSON.
- Interfaces that can later grow into paged/offloaded KV, batched decode, or
  multi-GPU routing without changing the `exec` contract.

## v1 cache policy

The first backend is contiguous and session-first because that matches the
current fastest Qwen3.6 single-stream kernel path. A request can reuse the hot
frontend state when its tokenized prompt exactly extends the cached session
prefix.

For OpenAI-style clients that resend full visible history, the service also
tracks the visible message journal. If the token journal contains hidden
Qwen-only tokens that the client does not resend, the service recognizes the
message-list append and prefills only the serialized suffix after the previous
assistant turn. Divergent prompts rebuild or restore at a future checkpoint
boundary. Truncation also rebuilds in v1: the frontend cannot roll the hot GPU
state back to a shorter prefix until checkpoint/rollback support lands.

For OpenAI-style clients that resend the full message list every turn, prefix
reuse requires the history to include the assistant content/tool call emitted by
the previous response. If a client sends only the new user/tool message without
the assistant turn, the token stream has diverged and the server must rebuild or
restore from a checkpoint.

This intentionally differs from paged/block serving frameworks: those are good
for high-concurrency batch serving, but the first FlashRT agent target is one
interactive long session on a consumer GPU.

## Implementation phases

1. CPU-only meta validation for prefix planning and tool-call streaming.
2. Split Qwen3.6 frontend generation into prefill and spec-decode steps.
3. Add the FastAPI host that maps OpenAI requests to session-aware generation.
4. Add checkpoint/rollback and eviction policy.
5. Benchmark: cold 128K/200K/256K plus incremental 2K/8K/16K turns.

## Current backend gate

`Qwen36FrontendAgentEngine` is wired to the real Qwen3.6 frontend for the
short-context committed split:

- cold short prefill: `prefill_own_speculative_nvfp4_agent`
- hot contiguous short append: `append_own_speculative_nvfp4_agent`
- cold long prefill: `prefill_long_ctx_nvfp4_agent`
- hot contiguous long append: `append_long_ctx_nvfp4_agent`
- committed streaming decode:
  `decode_own_speculative_nvfp4_committed_stream` or
  `decode_long_ctx_nvfp4_committed_stream`

Long-context append-prefill is limited to the currently hot contiguous session.
Non-hot sessions still rebuild/restore at the policy layer rather than reporting
a fake cache hit.
Exact same-length prompts continue from the current hot boundary; shorter
prompts rebuild until rollback/checkpoint support lands.

## Server parameters

`python -m serving.qwen36_agent.server [flags]`:

| flag | default | meaning |
| --- | --- | --- |
| `--checkpoint` | (required) | Qwen3.6 NVFP4 checkpoint directory |
| `--model-name` | `qwen36-27b` | id reported by `/v1/models` and echoed in responses |
| `--device` | `cuda` | torch device |
| `--max-seq` | `262208` | max sequence length (prompt + generation) |
| `--route-min-seq` | `0` | min prompt length sent to the chunked long-context FP8-KV path; `0` routes even short real prompts there to avoid request-time per-position graph capture |
| `--graph-cache-max` | auto | per-cache CUDA-graph LRU bound for opt-in exact graph replay; the production agent default uses direct long-decode kernels instead of exact-position decode graphs |
| `--warmup-preset` | `none` | startup warmup shapes: `none` / `agent` / `short` / `long` / `all`; production agent serving does not need graph warmup by default |
| `--warmup` | `""` | extra warmup shapes, comma-separated `prompt_len:max_tokens` |
| `--warmup-K` | `6` | speculative K used during warmup |
| `--warmup-committed-max-prompt` | `1024` | run real committed-stream warmup up to this prompt length; larger long-context shapes use graph-only warmup |
| `--warm-long-prefill-graphs` | off | also capture long-context prefill chunk graphs at startup |
| `--capsule-budget-mb` | `0` | GPU byte budget (MB) for pinned shared-prefix capsules; `0` disables pinning. See [Capsule pinning](#capsule-pinning-shared-prefix-reuse-that-survives-eos). |
| `--default-max-tokens` | `2048` | generated-token budget used when a request omits both `max_tokens` and `max_completion_tokens` |
| `--max-output-tokens` | `8192` | hard generated-token cap; requests above this return HTTP 400 instead of being silently truncated |
| `--default-session-id` | unset | fallback session id for requests that omit `flashrt_session_id` / `session_id`; intended only for single-client local agent demos or trusted one-user deployments |
| `--host` / `--port` | `127.0.0.1` / `8000` | bind address |
| `--log-level` | `info` | uvicorn log level |
| `--access-log` | off | enable uvicorn per-request access logs; off by default to avoid benchmark jitter |

Capsule pin/restore in this server is a production long-route feature. If a
request supplies `flashrt_pin_prefix`, the server requires a positive capsule
budget plus the long FP8-KV route (`--route-min-seq 0` and a long-context
`--max-seq`). It fails fast instead of silently falling back to the legacy short
prefill path.

Startup warmup is optional. The production agent default avoids exact-position
decode graphs on the live path, so arbitrary coding-agent sessions do not need
minutes of synthetic warmup. Use `--warmup-preset agent/all` only when you are
intentionally preparing fixed-shape graph-replay demos or benchmarks.

Startup logs print every queued warmup shape and then a `startup warmup done
i/N` line as each shape finishes. Per-request logs use the same metric fields for
both buffered and streaming responses:

```text
complete sid=... prompt=... completion=... prefill_ms=... first_delta_ms=... decode_ms=... decode_tok/s=...
stream   sid=... prompt=... completion=... prefill_ms=... first_delta_ms=... decode_ms=... decode_tok/s=... stream_wall_tok/s=...
```

For streaming responses, `decode_tok/s` measures backend decode-active time;
`stream_wall_tok/s` includes SSE/client backpressure and is the user-visible
streaming wall-time rate.

On SM120, the server defaults to the optimized decode kernels used by the
benchmark path (`FLASHRT_QWEN36_DECODE_FASTGEMM=1` and
`FLASHRT_QWEN36_VERIFY_WARPSPLIT=1`). It also defaults
`FLASHRT_QWEN36_TQ_VERIFY_GRAPH=0` and
`FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH=0` for agent serving, because exact-position
decode graphs are keyed by the live `cur_pos` and hurt first-use latency in a
continually growing session. Set those graph flags to `1` before startup only
for fixed-shape warmed benchmark runs. `/health` reports the kernel flags.

## HTTP surface and request fields

OpenAI-compatible: `GET /v1/models`, `GET /health`, `POST /v1/chat/completions`,
`POST /v1/sessions`, `DELETE /v1/sessions/{id}`. Standard OpenAI request fields
(`messages`, `max_tokens` / `max_completion_tokens`, `stream`, `tools`) plus
FlashRT extensions:

If neither `max_tokens` nor `max_completion_tokens` is supplied, the agent
server defaults to 2048 generated tokens so coding-agent tool turns are not cut
off by a chat-sized output cap. Production deployments should set
`--default-max-tokens` for their client mix and keep `--max-output-tokens` as a
hard safety bound; oversized requests fail with HTTP 400 instead of being
silently shortened.

- `flashrt_session_id` (or `session_id`): stable session key for prefix reuse.
- `flashrt_cache_salt`: optional namespace separator for different prompt policies.
- `flashrt_K`: speculative decode K for this request (default 6).
- `enable_thinking`: passed to the Qwen chat template (default false).
- `flashrt_pin_prefix`: pin this request's shared prefix as a capsule for reuse —
  an integer (pin that many leading prompt tokens) or `true` (pin the whole
  prompt's chunk-aligned head). Inert unless the server was started with
  `--capsule-budget-mb > 0`. See [Capsule pinning](#capsule-pinning-shared-prefix-reuse-that-survives-eos).

For OpenAI-compatible clients that cannot pass FlashRT extension fields, a
single-client deployment may set `--default-session-id <id>` so omitted session
ids still hit the local prefix/session cache. Leave it unset for multi-client
serving.

## Response fields

On top of the standard `chat.completion` (`choices[].message`, `usage`), each
response carries a `flashrt` telemetry block:

| field | meaning |
| --- | --- |
| `session_id` | the session this request used |
| `cached_tokens` | prompt tokens reused from the hot session (the prefix-reuse win) |
| `new_prefill_tokens` | prompt tokens actually prefilled this turn |
| `prefill_ms` | prefill / append time |
| `first_delta_ms` | time to first emitted delta (TTFT-like) |
| `decode_ms`, `decode_tok_per_s` | decode time and throughput |
| `prefix_action` | how the session was reused: `exact` / `append` / `message_append` / `restore` / `pin` / `truncate` / `rebuild` / `activate_rebuild` |

## Measured (RTX 5090, in-container)

Single RTX 5090 (sm_120), `qwen36_nvfp4` (25 GB) + MTP, `--route-min-seq 0`,
FP8-KV. Numbers are the serving path (real `/v1/chat/completions`), measured to
substantiate the two design claims below; this is not a throughput-serving
benchmark (single stream, latency-first).

**1. Session prefix reuse keeps prefill flat as a conversation grows.** A 4-turn
coding-agent session (same `flashrt_session_id`, full history resent each turn):

| turn | `prefix_action` | `cached_tokens` | `new_prefill_tokens` | `prefill_ms` |
| --- | --- | ---: | ---: | ---: |
| 1 | append (cold) | 0 | 352 | 14.5 |
| 2 | message_append | 416 | 23 | 12.4 |
| 3 | message_append | 503 | 22 | 12.7 |
| 4 | message_append | 589 | 20 | 12.5 |

Each turn prefills only the ~20 new tokens and reuses the growing cached prefix
(416 → 589), so prefill stays ~12 ms instead of growing with the transcript. A
server without prefix reuse re-prefills the full prompt every turn (589 tokens on
turn 4). This is the `append` / `message_append` path; correctness is gated
token-exact by `tests/test_qwen36_agent_gpu_split.py`.

> **Honest scope of contiguous append.** The committed stream stops at the
> visible stop-token boundary, so the hot session remains reusable when the next
> OpenAI request resends the prior assistant/tool turn. If a client rewrites or
> omits prior visible history, the token stream is no longer append-only and the
> server rebuilds or restores from a capsule instead of reporting a fake hit.

**2. Capsule restore replaces a shared-prefix cold prefill with a flat copy.**
Snapshot a shared prefix once, then restore + append the new suffix instead of
re-prefilling it (see [`capsules.md`](capsules.md) for the API and the full
table). Long FP8-KV route, chunk-aligned prefix, cold vs capsule TTFT:

| shared prefix | cold TTFT | capsule TTFT | speedup | token-exact |
| ---: | ---: | ---: | ---: | --- |
| 2048 | 259.6 ms | 111.0 ms | 2.3x | yes |
| 4096 | 358.5 ms | 46.5 ms | 7.7x | yes |
| 8192 | 775.6 ms | 111.0 ms | 7.0x | yes |

Cold TTFT grows with prefix length; capsule restore is a bandwidth-bound copy and
stays roughly flat, so the gap widens with the shared-prefix length a coding
agent resends each turn. Validated token-exact in
`tests/test_qwen36_agent_capsule.py`.

### Honest framing vs vLLM / SGLang (prefix reuse is *not* our differentiator)

Shared-prefix reuse itself is table stakes — vLLM and SGLang both have it, and
vLLM's Automatic Prefix Caching even reuses this hybrid model's GDN/mamba state
("Mamba cache mode = align"). Measured on the same checkpoint/GPU (vLLM 0.22,
`enable_prefix_caching=True`, base, prefix + 24-token suffix), vLLM saves a
comparable fraction of TTFT on a cached prefix:

| shared prefix | vLLM cold | vLLM APC reuse | vLLM saved |
| ---: | ---: | ---: | ---: |
| 2048 | 481 ms | 143 ms | 70% |
| 4096 | 549 ms | 76 ms | 86% |
| 8192 | 1076 ms | 120 ms | 89% |

So we do **not** claim a better prefix-reuse *mechanism*. What is actually
different here:

- **Lower absolute latency** from full-graph CUDA-graph replay + hand-tuned
  NVFP4 kernels (the cold prefill above is ~1.4-1.9x faster than vLLM's; clean
  same-method single-stream TTFT is ~1.5-1.9x and decode with MTP ~2x — see the
  decode-kernel docs). The *reuse ratio* is comparable; the *floor* is lower.
- **Capsule is an explicit, host-controlled, bit-exact primitive**
  (`snapshot` / `restore` / `fork` / restore-to-an-earlier-checkpoint), not an
  implicit block pool — it lets the host fork one prefill into N branches and
  roll a session back to a committed boundary deterministically.
- **One mechanism across LLM + VLA + robot** under the same execution contract
  (vLLM/SGLang are LLM-only); the robot side uses the identical Buffer
  snapshot/restore (`serving/robot_recap`, cosine 1.0).

In short: for high-concurrency multi-tenant LLM serving, the paged/radix engines
lead; FlashRT's target is latency-first single/few-session work on consumer/edge
hardware, hybrid models, and cross-domain (LLM/VLA/robot) — where the lower
latency floor + the bit-exact cross-domain capsule are the real edge, not prefix
reuse per se.

**Decode throughput is unchanged by either feature** (they touch prefill / TTFT
only): warm steady-state matches the frontend's documented decode number; the
serving policy adds no measurable decode overhead.

**3. Decode is stable across task types** (real `/v1/chat/completions`,
median of 3 runs in the fixed-shape warmed benchmark mode):

| scenario | ctx | fixed-shape decode tok/s |
| --- | ---: | ---: |
| code (merge sort) | 20 | 159.0 |
| reasoning (bat & ball) | 41 | 150.3 |
| code (two-sum) | 26 | 138.2 |
| math (word problem) | 38 | 128.9 |
| chat (explain) | 23 | 119.8 |
| long generation (512 tok) | 22 | 115.9 |
| doc-QA / RAG | 3023 | 90.6 |

Run-to-run variance < 2%. Decode tok/s varies with the task's speculative
accept-length (predictable code/reasoning highest; long-context attention pulls
the 3K-context RAG case down) — not with the serving path. These rows are the
fixed-shape warmed graph-replay benchmark mode, not the production agent
default.

To reproduce these rows, keep the same serving envelope and measurement
discipline:

```bash
export FLASHRT_QWEN36_MTP_CKPT_DIR=/path/to/qwen36_mtp_ckpt
export FLASHRT_QWEN36_LONG_KV_CACHE=fp8
export FLASHRT_QWEN36_TQ_VERIFY_GRAPH=1
export FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH=1
python -m serving.qwen36_agent.server \
  --checkpoint /path/to/qwen36_nvfp4 \
  --max-seq 32768 \
  --route-min-seq 0 \
  --warmup-preset agent \
  --host 127.0.0.1 --port 8000
```

The production agent path should be measured without those graph flags: it uses
direct verify/MTP-chain kernel launch, so an arbitrary new prompt length no
longer falls to cold graph-capture throughput. On RTX 5090, first-use live
requests measured ~122 tok/s for a short text stream and ~130 tok/s for a
tool-shaped stream with K=6.

## Session prefix reuse (walkthrough)

Reuse the same `flashrt_session_id` across turns and resend the full message list
(including the previous assistant turn). The server tokenizes the new prompt,
finds the longest exact token-prefix match against the hot session, and prefills
only the appended suffix:

```bash
# turn 1 (cold): flashrt.cached_tokens == 0, prefix_action == "rebuild"
curl -s :8000/v1/chat/completions -d '{"model":"qwen36-27b","flashrt_session_id":"s1",
 "messages":[{"role":"user","content":"List three sorting algorithms."}],"max_tokens":128}'

# turn 2 (warm): append the prior assistant reply + a new user message;
# flashrt.cached_tokens > 0, prefix_action == "append" / "message_append"
curl -s :8000/v1/chat/completions -d '{"model":"qwen36-27b","flashrt_session_id":"s1",
 "messages":[{"role":"user","content":"List three sorting algorithms."},
             {"role":"assistant","content":"<prior reply>"},
             {"role":"user","content":"Now give the time complexity of each."}],"max_tokens":128}'
```

If a client sends only the new message without the prior assistant turn, or a
shorter/divergent prompt, the token stream has diverged and the server rebuilds
or restores at a checkpoint boundary (it reports `rebuild`, never a fake hit).

## Capsule pinning (shared-prefix reuse that survives EOS)

A coding agent resends a large stable prefix every turn — system prompt, tool
schemas, repo index/summary — then a small new user/tool suffix, and each turn
ends on a stop token. Because an EOS-terminated turn invalidates contiguous append
(above), the way to reuse that prefix is to **pin it as an execution-state
capsule** and *restore* a clean committed boundary on every later turn/session,
re-prefilling only the suffix. This is FlashRT's graph-replay-native prefix reuse
(see [`capsules.md`](capsules.md) and [`../../docs/serving_design.md`](../../docs/serving_design.md)).

Enable it at startup with a GPU byte budget, then pin per request:

```bash
python -m serving.qwen36_agent.server --checkpoint /path/to/qwen36_nvfp4 \
  --max-seq 32768 --route-min-seq 0 --capsule-budget-mb 4096
```

```bash
# First turn: pin the stable prefix (e.g. its first 6000 tokens). The server
# cold-prefills + snapshots a chunk-aligned capsule, then serves normally.
curl -s :8000/v1/chat/completions -d '{"model":"qwen36-27b",
 "messages":[{"role":"system","content":"<system + tool schemas + repo index>"},
             {"role":"user","content":"First task"}],
 "flashrt_pin_prefix": 6000, "max_tokens": 256}'
# -> flashrt.prefix_action == "pin"

# Later turns / fresh sessions that share that prefix: restore + suffix only.
curl -s :8000/v1/chat/completions -d '{"model":"qwen36-27b",
 "messages":[{"role":"system","content":"<same system + tools + repo index>"},
             {"role":"user","content":"Second task"}],
 "flashrt_pin_prefix": 6000, "max_tokens": 256}'
# -> flashrt.prefix_action == "restore", cached_tokens == the chunk-aligned
#    boundary, new_prefill_tokens == only the suffix after it
```

Semantics and bounds:

- `flashrt_pin_prefix` is an int (pin that many leading prompt tokens) or `true`
  (pin the whole prompt's aligned head). The pin boundary is floored to the long
  prefill chunk size (2048), so a prefix shorter than one chunk is not pinned.
- A capsule is keyed by the digest of its chunk-aligned prefix tokens, so any
  later request — same session or not — whose prompt starts with that exact prefix
  restores it. Restore is **token-identical to a cold full prefill** (gated by
  `tests/test_qwen36_agent_capsule_serving.py`); tool conversations benefit too,
  because the stable system+tools head is pinned and only the changing suffix is
  re-prefilled.
- `--capsule-budget-mb` bounds GPU footprint. Capsules are LRU-evicted to fit and
  a single capsule larger than the whole budget is rejected (the request is served
  cold — never an OOM, never a false hit). Each capsule's KV grows with the pin
  length (≈230 MB for a 4096-token aligned prefix here), and competes with the
  model + KV cache for VRAM, so size the budget to the headroom you have.
- `/health` reports `capsules` (count, bytes, budget). Default budget is `0`
  (pinning off, serving path byte-identical).

## Validation

Fast policy and HTTP checks:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_qwen36_agent_serving_policy.py \
  tests/test_qwen36_server_warmup.py \
  tests/test_qwen36_agent_gpu_split.py
```

The GPU split test is skipped unless both checkpoint variables are present.  To
validate real Qwen3.6 short/long split and long append equivalence:

```bash
FLASHRT_QWEN36_NVFP4_CKPT_DIR=CHECKPOINT_DIR \
FLASHRT_QWEN36_MTP_CKPT_DIR=MTP_CHECKPOINT_DIR \
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q tests/test_qwen36_agent_gpu_split.py \
  tests/test_qwen36_agent_capsule.py \
  tests/test_qwen36_agent_capsule_serving.py -s
```

`test_qwen36_agent_capsule_serving.py` gates the serving-layer pin/restore policy:
a restored pinned prefix produces the same tokens as a cold full prefill.

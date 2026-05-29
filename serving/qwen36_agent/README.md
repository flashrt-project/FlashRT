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
# startup runs graph warmup, then logs: Uvicorn running on http://127.0.0.1:8000
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
- Startup committed-stream warmup so the first real request does not pay
  CUDA Graph capture for common agent prompt shapes.
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
current fastest Qwen3.6 CUDA-graph replay path. A request can reuse the hot
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
| `--graph-cache-max` | `128` | per-cache CUDA-graph LRU bound on the frontend |
| `--warmup-preset` | `agent` | startup warmup shapes: `agent` / `short` / `long` / `all` / `none` |
| `--warmup` | `""` | extra warmup shapes, comma-separated `prompt_len:max_tokens` |
| `--warmup-K` | `6` | speculative K used during warmup |
| `--warmup-committed-max-prompt` | `1024` | run real committed-stream warmup up to this prompt length; larger long-context shapes use graph-only warmup |
| `--warm-long-prefill-graphs` | off | also capture long-context prefill chunk graphs at startup |
| `--host` / `--port` | `127.0.0.1` / `8000` | bind address |
| `--log-level` | `info` | uvicorn log level |

Startup warmup moves CUDA-graph capture out of the first request: it runs real
committed-stream warmup for short/medium shapes and graph-only warmup for larger
long-context shapes.

## HTTP surface and request fields

OpenAI-compatible: `GET /v1/models`, `GET /health`, `POST /v1/chat/completions`,
`POST /v1/sessions`, `DELETE /v1/sessions/{id}`. Standard OpenAI request fields
(`messages`, `max_tokens` / `max_completion_tokens`, `stream`, `tools`) plus
FlashRT extensions:

- `flashrt_session_id` (or `session_id`): stable session key for prefix reuse.
- `flashrt_cache_salt`: optional namespace separator for different prompt policies.
- `flashrt_K`: speculative decode K for this request (default 6).
- `enable_thinking`: passed to the Qwen chat template (default false).

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
| `prefix_action` | how the session was reused: `exact` / `append` / `message_append` / `truncate` / `rebuild` / `activate_rebuild` |

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
pytest -q tests/test_qwen36_agent_gpu_split.py -s
```

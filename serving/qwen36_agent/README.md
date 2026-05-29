# serving/qwen36_agent

Production-oriented Qwen3.6-27B NVFP4 serving example for long-running agent
sessions.

This directory is the **policy layer** above the FlashRT execution contract. It
owns session cache, exact token-prefix reuse, OpenAI-compatible tool calling,
streaming, and request scheduling. It must not add session or KV verbs to
`exec/`; the contract remains Buffer / Graph / Plan / Event / ShapeKey.

## Design target

- 256K context on the existing Qwen3.6 long-context FP8-KV/TQ kernel path.
- Latency-first, single-stream hot session by default.
- Exact token-prefix reuse for coding-agent turns: cold prefill once, then only
  prefill appended user/tool/diff/log tokens.
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
prefix. Divergent prompts rebuild or restore at a future checkpoint boundary.

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

# examples/archive — retired examples

These files are kept for reference only. They are superseded by a maintained
path and are **not** the recommended way to run the model.

## `qwen36_openai_server.py` — retired

Replaced by the production agent server at
[`serving/qwen36_agent/`](../../serving/qwen36_agent/) (see its
[`README.md`](../../serving/qwen36_agent/README.md) for the run instructions and
measured performance).

The agent server supersedes this single-file example on every axis:

- committed-stream decode (streams only session-committed tokens) instead of the
  example's one-shot stream;
- session / exact-token-prefix reuse + message-append across turns;
- execution-state capsules (snapshot / restore / fork) for shared prefixes;
- `/v1/sessions` management, OpenAI tool-call streaming, true SSE;
- auto-scaled CUDA-graph cache, access-log-off-by-default, per-completion metric
  line, explicit MTP/speculative status, and hardened request parsing.

Use `python -m serving.qwen36_agent.server --checkpoint <ckpt>` instead.

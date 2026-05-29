# Qwen3.6 frontend split notes

The agent-serving host cannot wrap `generate_own_speculative_KN_nvfp4()` as-is:
that method resets state and prefills the full prompt on every call.  The
serving path needs split operations so a session can cold-prefill once and then
append only new tokens.

## Required frontend surface

```python
prefill_empty(input_ids) -> PrefillBoundary
prefill_append(input_ids_suffix, start_pos, boundary) -> PrefillBoundary
decode_spec_stream(max_new_tokens, K, boundary) -> Iterator[DecodeChunk]
truncate_or_restore(pos) -> PrefillBoundary
```

The first implementation may keep a single hot session and rebuild on
divergence.  It still needs `prefill_append`: without that, code-agent turns
fall back to cold long-context prefill.

## State that defines a boundary

- token position / `cur_pos`
- current token for the next decode step
- last hidden used to seed MTP
- linear-attention recurrent state
- linear-attention conv state
- full-attention persistent KV valid range
- long-context FP8/TQ dequant-stage valid end
- MTP compact cache base and valid range
- graph-cache key coverage for the early decode range

This state remains owned by the Qwen frontend.  The serving layer only stores
metadata and decides whether a request can reuse the hot frontend state.

## Streaming correctness rule

The old full-generate loop may verify `K+1` tokens and return only the prefix
needed to satisfy `max_new_tokens`. That is acceptable for a stateless response,
but it is not acceptable for an agent session cache: the GPU state would contain
tokens that were never sent to the client.

The split API therefore has a hard rule:

```text
every streamed token must be committed to the frontend state
```

If the decode loop needs lookahead, it must keep that lookahead explicitly and
either:

- commit it before yielding it; or
- roll back/truncate to the last yielded token before returning control to the
  session host.

For Qwen3.6 this means the first token predicted by prompt prefill cannot be
blindly yielded as a session boundary unless the main model state has also
processed that token. A correct implementation can use a one-token commit step
or a small lookahead buffer, but the serving layer must only see committed
chunks.

## Long-context path

For 512+ token prompts and the 128-token FP8-KV exception, the split must reuse
the existing long path:

- `_prefill_long_ctx_tq_chunked(input_ids)` becomes a range prefill:
  `prefill_long_ctx_range(input_ids, start_pos, end_pos, logits_mode)`.
- `_generate_long_ctx_speculative_KN_nvfp4()` becomes:
  1. cold/append prefill;
  2. MTP tail update for only the new suffix or configured tail window;
  3. spec decode loop that yields accepted chunks after each verify.
- graph warmup remains bucket-based and lives outside the request path.

## Short-context path

The legacy short path walks prompt tokens with per-position S=1 graphs and then
runs MTP prefill.  It should be split the same way:

- prefill suffix with `_ensure_graph_for_pos_nvfp4(pos)` and `_replay_pos_graph`;
- write `_prefill_h_cache` for suffix rows needed by MTP;
- update MTP cache for appended rows;
- stream accepted chunks from the existing captured MTP chain + verify loop.

## First correctness gates

1. full-generate output equals split prefill+decode output for a fixed short
   prompt, `K=6`, greedy.
2. long route full-generate output equals split prefill+decode output for 128,
   512, and 4K prompt buckets.
3. append test: `prefill_empty(A+B)` equals `prefill_empty(A)` +
   `prefill_append(B)` at logits and generated token sequence.
4. divergent prompt rebuilds and matches the old full-generate path.
5. stream chunks concatenate to the same text/token ids as non-stream mode.

# Qwen3.6 NVFP4 — Parameter Reference

Per-parameter reference for the v1 NVFP4 inference path. For the
high-level intro / quickstart / measured throughput, see
[`qwen36_nvfp4.md`](qwen36_nvfp4.md). Only the **NVFP4** path is
documented here (FP8 path exists but is not the v1 surface).

## Constructor

```python
from flash_rt.frontends.torch.qwen36_rtx import Qwen36TorchFrontendRtx

fe = Qwen36TorchFrontendRtx(
    checkpoint_path,            # required, str
    *,                          # everything below is keyword-only
    device='cuda:0',
    max_seq=2048,
    alloc_own_forward_buffers=True,
    quant='nvfp4',              # for the v1 path, set to 'nvfp4'
)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `checkpoint_path` | `str` | (required) | Directory of the NVFP4 main ckpt. Must contain `compressed-tensors` `nvfp4-pack-quantized` safetensors **and** the tokenizer files (`tokenizer.json` / `tokenizer_config.json` / etc). The HuggingFace ckpt `prithivMLmods/Qwen3.6-27B-NVFP4` ships these together. |
| `device` | `str` | `'cuda:0'` | CUDA device string. Single-GPU only; multi-GPU not supported in v1. |
| `max_seq` | `int` | `2048` | Max output sequence length the KV cache + per-token scratch is sized for. **Increase** if you plan to generate (or feed) more than 2048 tokens — the long-ctx grid in `qwen36_nvfp4.md` was measured at `max_seq=8192..262144`. Larger `max_seq` raises baseline VRAM proportionally to KV cache size (~10 MB / 1K tokens). |
| `alloc_own_forward_buffers` | `bool` | `True` | Pre-allocate every per-step buffer the own-forward / spec decode path consumes (zero per-call alloc; required for stable CUDA Graph capture). Set `False` only for memory-introspection unit tests. |
| `quant` | `str` | `'fp8'` | Set to `'nvfp4'` to get the v1 NVFP4 path. The default `'fp8'` is the legacy FP8 baseline path documented separately. |

The constructor performs the entire one-time setup: weight loading,
NVFP4 swizzle, MTP head conversion (if `FLASHRT_QWEN36_MTP_CKPT_DIR` is
set), and buffer allocation. After it returns, the model is ready for
inference. Wall time on RTX 5090: ~10-20 s, dominated by safetensors
read of the 17 GB NVFP4 weights.

VRAM after init (NVFP4 path, max_seq=2048): **~30 GB** total —
27 GB ckpt + ~1.5 GB MTP head + ~1.5 GB scratch (per-step state save
buffers, K_save_max=8). Fits comfortably in 32 GB on RTX 5090.

## Speculative decode

```python
output_ids = fe.generate_own_speculative_KN_nvfp4(
    input_ids,                # required, (1, prompt_len) cuda long
    *,                        # everything below is keyword-only
    max_new_tokens,           # required
    K=5,
)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `input_ids` | `torch.LongTensor` of shape `(1, prompt_len)` on CUDA | (required) | Tokenized prompt. Use `fe._tokenizer(prompt, return_tensors='pt').input_ids.cuda()`. Batch size must be `1`; multi-batch not supported in v1. |
| `max_new_tokens` | `int` | (required) | Number of tokens to generate. The output tensor is `(1, prompt_len + max_new_tokens)`. |
| `K` | `int` | `5` | MTP draft chain length per spec cycle. Verify processes `K+1` tokens at once. Valid range: `1 ≤ K ≤ MAX_Q_SEQ - 1` (MAX_Q_SEQ defaults to 16). The recommended value is `K=6` for short generations (≤ 256 output tokens) — see `qwen36_nvfp4.md` §3. |

Greedy-only in v1 — no `temperature`, `top_p`, or `top_k`. Returns a
deterministic argmax sequence.

If `FLASHRT_QWEN36_MTP_CKPT_DIR` was not set at construction, the MTP
head is not loaded and this method raises `RuntimeError`. Use
[`forward_own_decode_nvfp4`](#single-token-decode) for non-spec decode
in that case.

## Single-token decode

If you don't have an MTP head ckpt (or want to bypass spec for
correctness debugging), you can call the per-step forward directly:

```python
fe.reset_state()
if not hasattr(fe, '_rope_cos_table'):
    fe._build_rope_table()

cur_pos = 0
prompt_len = int(input_ids.shape[1])
generated = []
for p in range(prompt_len + max_new_tokens):
    if p < prompt_len:
        tok = input_ids[:, p:p+1]
    else:
        tok = generated[-1]
    fe._static_token_id.copy_(tok)
    cos, sin = fe._rope_cos_sin(cur_pos)
    fe.forward_own_decode_nvfp4(
        fe._static_token_id, cos, sin, cur_pos)
    if p >= prompt_len - 1:
        next_tok = fe._logits_buf.argmax(dim=-1, keepdim=True).view(1, 1)
        generated.append(next_tok)
    cur_pos += 1
```

This path tops out at ~36 tok/s decode (vs spec K=6's ~129 tok/s) but
needs only the NVFP4 ckpt — no MTP head dependency.

## Environment variables

All variables are read once at construction; setting them after the
frontend is built has no effect.

| Env var | Required? | Default | Meaning |
|---|---|---|---|
| `FLASHRT_QWEN36_MTP_CKPT_DIR` | Required for spec decode | unset | Directory containing `mtp.safetensors` (FP8 e4m3 block-128) from a paired Qwen3.6-Next-27B-FP8 ckpt. Loaded once at construction and converted FP8 → BF16 → NVFP4. If unset, MTP is `None` and `generate_own_speculative_KN_nvfp4` raises; pure-decode still works. |
| `FLASHRT_QWEN36_HF_PATCH` | Optional | unset | Path to a HF FP8 dispatch monkey-patch script. Only consulted by the legacy FP8 path; the NVFP4 path doesn't need it. If unset or path doesn't exist, the patch step is silently skipped. |
| `FLASHRT_QWEN36_DFLASH_CKPT_DIR` | Optional | unset | Drafter ckpt directory for the DFlash add-on path. Required only if you call `init_dflash_drafter()`; raises a clear error if unset and `ckpt_dir` is also not passed. |
| `FLASHRT_NVFP4_LOAD_DEBUG` | Optional | `0` | Set to `1` for verbose VRAM-tracking prints during NVFP4 weight load. |
| `FLASHRT_DFLASH_LOAD_DEBUG` | Optional | `0` | Same, for DFlash drafter load. |
| `PYTORCH_CUDA_ALLOC_CONF` | Recommended | system default | Set to `expandable_segments:True` to avoid fragmentation when the long-ctx grid pushes past 30 GB. The standard bench was run with this. |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | Recommended | unset | Set to `1` if you've already downloaded the ckpt locally — saves ~1-2 s of network probe at construction. |

## Tokenizer

The constructor loads the tokenizer from `checkpoint_path` via
`AutoTokenizer.from_pretrained`. It's stored as `fe._tokenizer` and is
the standard HuggingFace `PreTrainedTokenizerFast` instance — call
`.encode()`, `.decode()`, `.apply_chat_template()`, etc. directly.

Example chat-style prompt (Qwen3.6 uses the `qwen` chat template):

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum entanglement briefly."},
]
prompt = fe._tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
input_ids = fe._tokenizer(prompt, return_tensors='pt').input_ids.cuda()
```

## OpenAI server tool calling

The bundled OpenAI-compatible server accepts OpenAI-shaped `tools` on
`/v1/chat/completions`. Qwen's chat template injects the function
schema into the prompt, and the server parses model-emitted
`<tool_call>...</tool_call>` blocks into OpenAI `tool_calls`.

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:8000/v1', api_key='-')

tools = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the current weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {'type': 'string'},
            },
            'required': ['city'],
        },
    },
}]

resp1 = client.chat.completions.create(
    model='qwen3.6-27b-nvfp4',
    messages=[{'role': 'user', 'content': 'What is the weather in Tokyo?'}],
    tools=tools,
    max_tokens=128,
)

tool_call = resp1.choices[0].message.tool_calls[0]

resp2 = client.chat.completions.create(
    model='qwen3.6-27b-nvfp4',
    messages=[
        {'role': 'user', 'content': 'What is the weather in Tokyo?'},
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [tool_call.model_dump()],
        },
        {
            'role': 'tool',
            'tool_call_id': tool_call.id,
            'content': '{"city":"Tokyo","temp_c":22,"condition":"sunny"}',
        },
    ],
    tools=tools,
    max_tokens=128,
)

print(resp2.choices[0].message.content)
```

For `stream=True`, the v1 server still emits a single response chunk
rather than token-by-token deltas, but any parsed `tool_calls` are
returned as OpenAI-style SSE `delta.tool_calls` entries before the
final chunk.

## Cold-start vs warm-state

The headline 90-130 tok/s decode rate is the **warm-state** number —
what you measure after CUDA Graphs for the relevant `cur_pos` range
have been captured. The **first call** at a previously-unseen
`(prompt_len, max_new_tokens)` shape pays a one-time graph-capture
cost of roughly 5-25 s (proportional to `prompt_len + max_new_tokens`
and to whether spec-K verify graphs at those positions are also new),
manifesting as ~20-40 tok/s for that first call only.

This is a property of the CUDA Graph capture/replay model, shared
with SGLang and vLLM's compile mode. TensorRT-LLM avoids it by AOT
engine compilation (paid at deploy time instead). vLLM eager and TGI
avoid it by not capturing graphs (paying per-launch overhead per
step instead).

For a server deployment, run a dummy generation at startup over the
prompt_len/max_tokens shapes you expect to see — this populates the
graph cache before live traffic arrives. The bundled OpenAI server
example does this automatically via `--warmup` (default
`32:128,128:256`); see [`examples/qwen36_openai_server.py`](../examples/qwen36_openai_server.py).

After warmup, requests at the same shape stay warm. Requests at
different shapes will still pay capture cost on the parts of their
`cur_pos` range not yet covered.

## Known limits in v1

- **Batch size 1 only.** Multi-batch / continuous batching not in v1.
- **Greedy decode only.** No temperature, top-p, top-k, repetition
  penalty. The token sequence is deterministic given the prompt.
- **No streaming.** `generate_own_speculative_KN_nvfp4` returns the
  full output tensor at the end. The OpenAI server example wraps this
  with chunked SSE for `stream=True` requests by waiting for the full
  response and emitting it in one chunk; true token-by-token streaming
  needs a frontend modification.
- **Single GPU.** Multi-GPU tensor parallel not supported.
- **K ≤ 7** at K_save_max=8. Bumping K_save_max trades ~75 MB VRAM
  per slot for the ability to use larger K — but the K-curve plateaus
  past K=6 anyway (see `qwen36_nvfp4.md` §3).

## Choosing K — quick rule of thumb

| Output length | Recommended K | Why |
|---|:---:|---|
| ≤ 128 tokens | **6** | Peak measured (129 tok/s on the standard prompt). |
| 128–256 tokens | **5** | K=6 starts losing acceptance past ~150 tokens; K=5 is more robust. |
| ≥ 512 tokens | **3** | All K values converge near 113 tok/s by NTOK=512; K=3 has the lowest CV across prompts. |

The full K-sweep (K=3..7 × NTOK=128/256/512 × 5 prompt classes) is
in `qwen36_nvfp4.md` §3.

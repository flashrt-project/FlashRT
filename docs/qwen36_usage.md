# Qwen3.6 NVFP4 — Parameter Reference

Per-parameter reference for the v1 NVFP4 inference path. For the
high-level intro / quickstart / measured throughput, see
[`qwen36_nvfp4.md`](qwen36_nvfp4.md). Only the **NVFP4** path is
documented here (FP8 path exists but is not the v1 surface).

## Installation

Install the Torch frontend extra from the repository root:

```bash
pip install -e ".[torch]"
```

The Qwen3.6 long-context path uses native FlashRT CUDA/CUTLASS
kernels; it does not require Triton/FLA Python kernels.
For the OpenAI-compatible server, install:

```bash
pip install -e ".[torch,server]"
```

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
| `max_seq` | `int` | `2048` | Max output sequence length. For NVFP4, values above the long-context threshold allocate a compressed KV cache for long requests while retaining a small BF16/spec window. Increase this if you plan to generate or feed more than 2048 tokens; requests above `FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ` use MTP draft plus compressed-KV verify. |
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
    K=6,
)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `input_ids` | `torch.LongTensor` of shape `(1, prompt_len)` on CUDA | (required) | Tokenized prompt. Use `fe._tokenizer(prompt, return_tensors='pt').input_ids.cuda()`. Batch size must be `1`; multi-batch not supported in v1. |
| `max_new_tokens` | `int` | (required) | Number of tokens to generate. The output tensor is `(1, prompt_len + max_new_tokens)`. |
| `K` | `int` | `6` | MTP draft chain length per spec cycle. Verify processes `K+1` tokens at once. Valid range: `1 ≤ K ≤ 15` in the public path. `K=6` is the default for short generations (≤ 256 output tokens) — see `qwen36_nvfp4.md` §3. |

Greedy-only in v1 — no `temperature`, `top_p`, or `top_k`. Returns a
deterministic argmax sequence.

When `quant='nvfp4'` is constructed with a large `max_seq`, this method
auto-routes per request: short requests that fit inside the retained
BF16/spec window still run MTP speculative decode, while larger
requests use MTP speculative decode with the compressed-KV verify path.
Long-context prefill is chunked with the same S=K
forward (`FLASHRT_QWEN36_TQ_PREFILL_CHUNK`, capped by `MAX_Q_SEQ`;
default cap 2048), and full-attention prefill chunks use the vendored FA2
causal hdim=256 path. Linear-attention prefill chunks use chunked
causal-conv and the native FlashRT WY/cuBLASLt Gated DeltaNet backend
by default (`FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt`). Set
`FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=native` to force the direct-conv
FlashRT recurrent scan for bisection. `FVK_QWEN36_CHUNK_CONV_PARALLEL=0`
forces the older serial chunk conv update. Experimental fused gate/up
and cuBLAS AB paths are
available behind environment variables but default off because they
were either slower or not elementwise-stable in local checks. The
default linear-attention A/B path uses a deterministic AB96 kernel that
is bit-identical to the previous two-matmul path while saving a small
amount of prefill time. During long prefill, intermediate chunks skip
lm-head logits and the final chunk computes only the last prompt row's
logits; verify/spec decode still computes all required logits. The
large all-row logits workspace is allocated lazily only for explicit
diagnostic calls, so the default long-context working set stays smaller.
The long-context verify path and MTP draft chain are CUDA-Graph captured
in warm state when the corresponding graph env vars are left enabled.

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

This path tops out at ~36 tok/s decode (vs spec K=6's ~134 tok/s on the
short standard prompt) but
needs only the NVFP4 ckpt — no MTP head dependency.

## Environment variables

All variables are read once at construction; setting them after the
frontend is built has no effect.

| Env var | Required? | Default | Meaning |
|---|---|---|---|
| `FLASHRT_QWEN36_MTP_CKPT_DIR` | Required for spec decode | unset | Directory containing `mtp.safetensors` (FP8 e4m3 block-128) from a paired Qwen3.6-Next-27B-FP8 ckpt. Loaded once at construction and converted FP8 → BF16 → NVFP4. If unset, MTP is `None` and `generate_own_speculative_KN_nvfp4` raises; pure-decode still works. |
| `FLASHRT_QWEN36_MTP_KEEP_BF16` | Optional | BF16-source MTP: `1`; FP8-source MTP: n/a | For community BF16/native MTP checkpoints, keep BF16 projection weights and use them in the drafter hot path. This improves MTP alignment at the cost of extra VRAM. Set `0` to force the lower-memory NVFP4-converted MTP path. |
| `FLASHRT_QWEN36_HF_PATCH` | Optional | unset | Path to a HF FP8 dispatch monkey-patch script. Only consulted by the legacy FP8 path; the NVFP4 path doesn't need it. If unset or path doesn't exist, the patch step is silently skipped. |
| `FLASHRT_QWEN36_DFLASH_CKPT_DIR` | Optional | unset | Drafter ckpt directory for the DFlash add-on path. Required only if you call `init_dflash_drafter()`; raises a clear error if unset and `ckpt_dir` is also not passed. |
| `FLASHRT_QWEN36_MAX_Q_SEQ` | Optional | `2048` | Maximum S=K working-set rows for verify/prefill buffers. Long prefill chunking is additionally capped by the retained BF16 working window. |
| `FLASHRT_QWEN36_LONG_CTX_BF16_WINDOW` | Optional | `min(2048, MAX_Q_SEQ)` | Retained BF16 working-window rows in long-context mode. Raising this can enable larger prompt chunks but costs substantial VRAM. |
| `FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ` | Optional | `512` in long-ctx mode | Prompt length at or above which a long-context frontend routes through the chunked compressed-KV path. Short prompts stay on BF16/spec unless the full request exceeds the retained BF16 window. |
| `FLASHRT_QWEN36_LONG_KV_CACHE` | Optional | `fp8` | Long-context persistent KV format. `fp8` uses an e4m3 FP8 KV cache. On SM120, long verify attention uses the vendored FlashInfer XQA FP8-KV kernel above the XQA threshold and falls back to BF16 FA2 staging below it. Set `tq` to use the TurboQuant packed path for memory/accuracy bisection. |
| `FLASHRT_QWEN36_FP8_XQA` | Optional | `1` | Enable the SM120 FlashInfer XQA native FP8-KV verify path for long-context FP8 KV. Set `0` to force the previous FP8->BF16-stage + FA2 path. |
| `FLASHRT_QWEN36_FP8_XQA_MIN_CTX` | Optional | `auto` | XQA gating for FP8-KV verify. `auto` uses measured buckets: off below 6K, on from 6K to 12K, off from 12K to 24K, and on from 24K upward. Set a number to force the older minimum-KV-length threshold. |
| `FLASHRT_QWEN36_FP8_XQA_SCRATCH_MB` | Optional | `256` | Scratch workspace reserved for XQA multi-block reductions. |
| `FLASHRT_QWEN36_TQ_SPEC_K` | Optional | unset | Override the effective speculative K for long-context TQ/spec requests. If unset, long TQ/spec uses measured buckets: `4` around 512 tokens, `5` around 1K/8K, `6` around 2K/32K/200K+, `3` around 4K, and `7` around 16K/64K/128K. Passing K below 6 keeps that lower caller cap. Short BF16/spec requests keep the caller K unchanged. |
| `FLASHRT_QWEN36_TQ_ADAPTIVE_K` | Optional | `1` | When long TQ/spec uses the default K≥4 policy, drop to K=3 inside a request if the early accept statistics show a low-hit prompt. Explicit `FLASHRT_QWEN36_TQ_SPEC_K` disables this adaptation. |
| `FLASHRT_QWEN36_LONG_MTP_PREFILL_TAIL` | Optional | `auto` | Long-context MTP prompt-tail prefill. `auto` uses measured KV-only buckets: disabled below 512 tokens, 512 rows around 512/4K, and 2048 rows for 1K-2K and 8K+. Set `0` to disable or a positive value to force a fixed tail length. |
| `FLASHRT_QWEN36_LONG_MTP_TAIL_KV_ONLY` | Optional | `1` | When prompt-tail prefill is enabled and the MTP checkpoint has BF16 projection weights, populate only the MTP K/V cache rows needed by the drafter. Set `0` to force the older full-MTP-head tail loop for bisection. |
| `FLASHRT_QWEN36_TQ_STRICT_NEXT` | Optional | `0` | Debug/validation mode that recomputes the correction or bonus token on the sequential target path after batched TQ verify. This preserves greedy next-token invariance for tail-prefill experiments but is much slower than the default batched verify path. |
| `FLASHRT_QWEN36_TQ_STRICT_NEXT_GRAPH` | Optional | `1` | Use per-position K=1 TQ verify graphs for the strict-next recompute. Only consulted when `FLASHRT_QWEN36_TQ_STRICT_NEXT=1`. |
| `FLASHRT_QWEN36_TQ_VERIFY_EXACT_GATING` | Unsupported | `0` | Legacy torch-style GDN gating bisection path. The kernel-only Qwen3.6 route rejects `1`; use the fused FlashRT gating kernel. |
| `FLASHRT_QWEN36_TQ_VERIFY_GRAPH` | Optional | `1` | Capture/replay the long-context TQ verify forward as per-`(cur_pos, K)` CUDA Graphs. This is the fastest warm path. Set `0` only when optimizing first-request latency without prewarm or debugging graph capture. |
| `FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH` | Optional | `1` | Capture/replay the long-context MTP draft chain. This is the fastest warm path. Set `0` only when optimizing first-request latency without prewarm or debugging graph capture. |
| `FLASHRT_QWEN36_LONG_WARMUP_MIN_FREE_MB` | Optional | `1024` | Stop long-context startup graph warmup once free VRAM falls below this waterline. This prevents 200K+ buckets from over-capturing graphs and leaving too little memory for the first real request. |
| `FLASHRT_QWEN36_LONG_GRAPH_MIN_FREE_MB` | Optional | `768` | During a real long-context decode, skip new TQ verify graph capture and run eager verify when free VRAM is below this waterline. Already-warmed graphs are still replayed. |
| `FLASHRT_QWEN36_TQ_PER_LAYER_STAGE_MAX_SEQ` | Optional | `132000` | Maximum TQ cache length eligible for per-layer BF16 KV staging. Auto uses 16 staged full-attn layers up to ~64K and 8 layers around 128K; larger servers fall back to shared staging to avoid 32 GB OOM. |
| `FLASHRT_QWEN36_TQ_PER_LAYER_STAGE_LAYERS` | Optional | `auto` | Override the number of full-attn layers with persistent BF16 KV stage (`0..16`). More layers reduce repeated TQ dequant in long decode but cost about `prompt_len * 4 KB` per layer at BF16 K+V. |
| `FLASHRT_QWEN36_TQ_HOT_STAGE_LAYERS` | Optional | `auto` | Extra 128K-tier BF16 staging layers for servers sized to 200K+. Auto keeps this conservative so CUDA Graph capture still has free VRAM. Set an integer to force a more aggressive hot tier for benchmarking. |
| `FLASHRT_QWEN36_TQ_HOT_STAGE_RESERVE_MB` | Optional | `1536` | Free-memory reserve used when auto-sizing the 128K hot stage. Increase for safer serving; lower only for controlled benchmarking. |
| `FLASHRT_QWEN36_FP8_STAGE_LAYERS` | Optional | `auto` | Extra per-layer BF16 stage count for the FP8-KV bridge. Auto keeps one 200K-cap layer on 32GB cards to avoid repeated full-prefix FP8 dequant while preserving CUDA Graph memory headroom. |
| `FLASHRT_QWEN36_FP8_HOT_STAGE_LAYERS` | Optional | `auto` | Extra 128K-tier FP8-KV stage count. Auto keeps one hot layer when a larger 200K stage is active. Higher values can block CUDA Graph capture on 32GB GPUs. |
| `FLASHRT_QWEN36_FP8_STAGE_RESERVE_MB` / `FLASHRT_QWEN36_FP8_HOT_STAGE_RESERVE_MB` | Optional | `1024` | Free-memory reserves used by FP8 stage auto-sizing. Lower only for controlled benchmarking. |
| `FVK_QWEN36_TQ_CUTLASS` | Optional | `auto` | Use CUTLASS fused TQ dequant for shared staging. `auto` enables it up to the 128K profile and leaves 256K on the lower-memory path. Set `0`/`1` to force. |
| `FLASHRT_QWEN36_TQ_KERNEL_WRITE` | Required | `1` | Use explicit cuBLASLt/cuBLAS wrappers for TurboQuant write-side GEMMs. The older `torch.matmul` write path is removed from the kernel-only route. |
| `FLASHRT_QWEN36_FUSE_MLP_GATE_UP` | Optional | long NVFP4: `1`; otherwise `0` | Runs MLP gate/up as one fused NVFP4 GEMM when the checkpoint has homogeneous gate/up scales. This is the default long-context path because it improves warm TQ/spec decode and reduces scratch memory; set `0` to force the older two-GEMM path. |
| `FLASHRT_QWEN36_FUSE_SILU_MUL_QUANT` | Optional | `0` | Experimental fused `silu(gate)*up -> NVFP4` activation path. Forced on when fused gate/up is enabled for correct merged-buffer stride handling; otherwise default off because it was slower locally. |
| `FLASHRT_QWEN36_LIN_AB96_KERNEL` | Optional | `1` | Deterministic fused kernel for the tiny linear-attention A/B projections in long-prefill chunks. Bit-identical to the old two-call BF16 path; set `0` to force the previous path. |
| `FLASHRT_QWEN36_LIN_AB_TORCH_MM_MIN_K` | Unsupported | `0` | Legacy Torch matmul bisection path for linear-attention A/B projections. The kernel-only route rejects values above `0`; use `FLASHRT_QWEN36_LIN_AB96_KERNEL=1`. |
| `FLASHRT_QWEN36_FULL_GATE_SIGMOID_MUL` | Optional | `0` | Experimental fused full-attention output gate `sigmoid(gate) * attn` kernel. Bit-identical in random checks but did not improve the 1024-token chunk benchmark, so it defaults off. |
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
Qwen thinking mode is disabled by default so ordinary chat responses do
not start inside `<think>`. Pass `"enable_thinking": true` in the JSON
request body if you want the model's thinking-mode template.

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

The headline decode rate is the **warm-state** number -- what you
measure after CUDA Graphs for the relevant `cur_pos` range have been
captured. The first call at a previously unseen
`(prompt_len, max_new_tokens)` shape pays a one-time graph-capture cost
that can dominate decode latency. This is a property of the CUDA Graph
capture/replay model: fastest steady-state decode requires paying
capture either during warmup or on the first live request.

For server deployment, run dummy generations at startup over the
prompt_len/max_tokens buckets you expect to see. This populates graph
cache, allocator state, kernel state, and library plans before live
traffic. The bundled OpenAI server example does this with
`--warmup-preset auto` by default; add explicit buckets with `--warmup`
when your traffic includes larger contexts:

```bash
export FLASHRT_QWEN36_LONG_KV_CACHE=fp8
python examples/qwen36_openai_server.py \
  --checkpoint /path/to/qwen36_nvfp4 \
  --max-seq 262208 \
  --warmup-preset all \
  --warmup "262144:16"
```

The default long-context route threshold is 512 prompt tokens: very
short prompts stay on BF16/spec for peak decode unless the requested
completion exceeds the retained BF16 window, while 512-token and larger
prompts use the tuned chunked FP8-KV path. `--warmup-preset auto`
warms 64-token short-chat buckets plus 2K/4K/8K/16K/32K/64K/128K/256K
buckets that fit inside `--max-seq`. Add explicit `--warmup`
`prompt_len:max_tokens` entries for longer completion caps.
The 256K prompt bucket requires `--max-seq` larger than `262144` by at
least the requested completion length. See
[`examples/qwen36_openai_server.py`](../examples/qwen36_openai_server.py).

If first-request latency matters more than warm decode throughput, set
`FLASHRT_QWEN36_TQ_VERIFY_GRAPH=0` and
`FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH=0`. That avoids per-position graph
capture, but the warm decode rate is lower.

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
| ≤ 128 tokens | **6** | Peak measured (134 tok/s on the standard prompt). |
| 128–256 tokens | **5** | K=6 starts losing acceptance past ~150 tokens; K=5 is more robust. |
| ≥ 512 tokens | **3** | All K values converge near 113 tok/s by NTOK=512; K=3 has the lowest CV across prompts. |

The full K-sweep (K=3..7 × NTOK=128/256/512 × 5 prompt classes) is
in `qwen36_nvfp4.md` §3.

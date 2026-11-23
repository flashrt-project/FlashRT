# Inkling 2 Mini on RTX 5090 (SM120)

FlashRT runs the text decoder of `thinkingmachines/Inkling-2-Mini` on the RTX
5090 with its own CUDA kernels (`flash_rt_inkling2`, a separate module behind
`FLASHRT_ENABLE_INKLING2`), and drives speculative decoding with the
checkpoint's DSpark drafter (`mtp.safetensors`). The checkpoint is used at its
released precision: BF16 attention, gates, embedding and LM head, FP32
depth-attention queries; NVFP4 experts (E2M1 codes, E4M3 scales per 16 inputs,
per-expert global scale) in layers 0-19 and MXFP8 experts (E4M3 codes, E8M0
scales per 32) in layers 20-23, multiplied with the expert inputs quantized as
`hf_quant_config.json` requires (NVFP4 with the `fouroversix_mae` scale rule,
MXFP8). The model's weights are loaded as stored; nothing is re-quantized. (The
drafter's draft-only vocabulary products read E4M3 copies of the LM head and of
its folded correction; they choose drafts, never the verified tokens.)

Every forward is a recorded program: a fixed list of kernel launches over
device pointers that reads the position and the input tokens from device
memory, captured once as a CUDA graph (with programmatic dependent launch) and
replayed. Text only (the vision tower is not loaded), greedy, one sequence.

## Requirements

| | |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` directory (`config.json`, `model-*.safetensors`, `mtp.safetensors` for speculative decoding, `tokenizer.json`, `chat_template.jinja`) |
| GPU | RTX 5090 or another SM120 GPU with at least 24 GB (about 20 GB with the drafter at `max_seq=4096`) |
| Software | CUDA 12.8 or newer (block-scaled FP4 / FP8 `mma.sync` of sm_120a), PyTorch with CUDA, `safetensors`; `tokenizers` + `jinja2` for the tokenizer and chat template |

```bash
cmake -B build -DGPU_ARCH=120 -DFLASHRT_ENABLE_INKLING2=ON
cmake --build build --target flash_rt_inkling2 -j     # flash_rt/flash_rt_inkling2*.so
```

A checkpoint with another geometry, layer layout or drafter configuration is
refused before any weight is loaded; expert formats are checked while loading.
`INKLING2_PDL=0` launches without programmatic dependent launch (for
per-kernel profiling).

## Usage

```python
from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx

fe = Inkling2TorchFrontendRtx("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True)
print(fe.chat([{"role": "user", "content": "Write a quicksort in Python."}],
              max_new_tokens=512, reasoning_effort="none"))

# Token-level streaming; speculative and prefill mode per call.
ids = fe.tokenizer.encode(fe.apply_chat_template([{"role": "user", "content": "Hello"}]))
for block in fe.stream(ids, max_new_tokens=256, speculative=False, prefill="exact"):
    print(fe.tokenizer.decode(block), end="", flush=True)
```

| Argument | Meaning |
|---|---|
| `max_seq` | KV capacity of the global-attention layers (sliding-window layers keep 1024 slots): prompt plus generated tokens plus a margin, 1 position for plain decoding and `(check_rounds + 1) * draft_width + 8` (28 by default) for speculative decoding |
| `speculative` | load the DSpark drafter and decode speculatively by default; `stream(..., speculative=False)` decodes plainly |
| `draft_width` | verified rows per speculative step: the anchor and `draft_width - 1` drafts (2..8, default 4) |
| `prefill` | `"fast"`: tensor-core chunks (default); `"exact"`: per-row chunks, bit-identical to decoding the prompt token by token |
| `check_block`, `check_rounds` | decode steps / speculative steps between host reads of the generated tokens (a stop token is seen that late) |

Programs are recorded and captured on first use; `fe.warmup(prompt_len)`
captures the ones a prompt of that length and the decode loop need. Each
`stream` call starts a new sequence.

## Components

| Part | Where |
|---|---|
| Frontend: validation, streaming, chat template | `flash_rt/frontends/torch/inkling2_rtx.py` |
| Runtime: configuration checks, weights, buffers, program recording, prefill planning | `flash_rt/frontends/torch/_inkling2_rtx_runtime.py` |
| DSpark drafter programs | `flash_rt/frontends/torch/_inkling2_rtx_mtp.py` |
| Kernels | `csrc/kernels/inkling2/`, bindings `csrc/inkling2_bindings.cpp` |

The kernels by program:

- **Decode** (one row, about 480 launches): multi-output GEMV with RMSNorm and
  gate prologues, the layer-visit boundary (depth attention over the block
  residuals, message gates between passes of the looped layers), split-key
  attention with relative position bias, L2 clips, sinks and the output gate,
  fused routing + expert kernels (input quantization behind the first weight
  loads, exact FP4 / FP8 block products), the LM head with the greedy argmax.
  Weights of the next kernels are prefetched into L2 before each kernel waits
  on its predecessor.
- **Verification** (2..8 rows): the same per-row arithmetic over staged rows,
  flat expert passes over the distinct experts of the rows, greedy targets,
  acceptance, and a commit of the accepted rows' KV and convolution states.
- **Prompt chunks** (16..512 rows): projections on cuBLAS (BF16; the router
  with FP32 sums on TF32 tensor cores, exact for its BF16-valued operands, set
  explicitly whatever the process defaults), grouped block-scaled MMA experts (`mma.sync` `kind::mxf4nvf4`
  `m16n8k64` for NVFP4, `kind::mxf8f6f4` `m16n8k32` for MXFP8; the checkpoint's
  codes and scale rows load as stored) over the slots sorted by expert, a
  tensor-core attention (BF16 Q K, the relative bias from MMA tables, P V with
  P split into two FP16 parts), MMA message gates. Prompts run as chunks of
  512, 256, ..., 16 rows, then per-row chunks of 8, then decode steps.
- **DSpark**: the drafter's context rows (written from the target's features
  after every committed row), the block of eight positions over three layers,
  the Markov RNN head with the rank-256 correction folded into the vocabulary;
  the draft-only vocabulary products read E4M3 copies with per-row scales. One
  CUDA graph per speculative step: verify, commit, context, next drafts.

## Numerics

- **Verification rows are bit-identical to decode steps.** Each row of a
  verification runs the decode step's arithmetic (same reductions, same
  rounding); the targets of W rows are the tokens W decode steps produce, and
  the committed state equals theirs. Speculative decoding therefore returns
  exactly the plain greedy output.
- **Exact prefill is bit-identical to token-by-token decoding** (KV,
  convolution states, logits, drafter context), past the sliding window.
- **Fast prefill is equivalent, not bit-identical**: tensor-core products and
  attention accumulate in another order. With the experts' FP4 rounding the
  checkpoint's logits are sensitive to such reorderings; the reference's own
  perturbations (its quantizer written as the kernel's arithmetic, or W4A16
  instead of W4A4) move the greedy continuation of the benchmark prompt after
  16-19 tokens. Measured against a reference implementation written from the
  upstream model definition, teacher-forced over 48 positions after the
  512-token prompt:

  | Prefill | argmax agreement | KL median | KL p90 | KL max |
  |---|---:|---:|---:|---:|
  | token steps / exact chunks | 48 / 48 | 1.1e-4 | 1.8e-2 | 8.6e-2 |
  | fast chunks | 47 / 48 | 1.3e-4 | 1.5e-2 | 5.9e-2 |
  | fast chunks, router product on FP32 CUDA cores | 47 / 48 | 1.3e-4 | 4.3e-2 | 1.8e-1 |

  The last row only sums the router's exact products in another order; that
  moves single positions by up to 0.1-0.2 in KL (a routing near-tie resolved
  the other way changes the prompt's KV), the spread of any reordering of the
  fast path.

- **KV cache in BF16** (the checkpoint's attention precision); no FP8 KV.

## Validation

```bash
# anywhere (no GPU): configuration and drafter checks, chunk planning, routing
python -m pytest tests/test_inkling2_rtx_contracts.py
# SM120 GPU, module built: GEMV rows against one-row launches (bit for bit), MMA gate tiles, row limits
python -m pytest tests/test_inkling2_rtx_kernels.py
# SM120 GPU + checkpoint
FLASHRT_INKLING2_DIR=/path/to/Inkling-2-Mini python -m pytest tests/test_inkling2_rtx_pipeline.py
```

The pipeline file checks verification rows against decode steps (logits bit for
bit, acceptance, committed state with a correct and a wrong draft), exact
prefill against token steps past the sliding window (bit for bit), fast
prefill against exact prefill for several chunk plans, the grouped MMA experts
against the flat passes, speculative against plain greedy decoding (both
prefill modes, widths 2, 4 and 8), stop tokens and lengths, input checks and
the chat path.

## RTX 5090 numbers

RTX 5090 (170 SMs, 32 GB), batch 1, greedy. Reference: KERN 0.5.0
(`pegainfer-project/kern-inkling-2mini`), the fastest Inkling 2 Mini runtime
for this GPU we know of, on the same GPU with its own benchmark driver: the
512-token prompt below, 512 new tokens, plain decoding and DSpark (KERN's
8-row speculative mode), three runs each. FlashRT: the same token ids, plain
and DSpark (width 4) interleaved, three runs each after a warm-up request, 30 s
between runs and 0.2 s of decode steps before each timed request (the GPU at
its clocks rather than idle). Decode tokens/s = (new tokens - 1) / time from
the first token to the last; prefill = prompt to first token (with DSpark
including the drafter context). Medians:

| | KERN 0.5.0 | FlashRT | |
|---|---:|---:|---:|
| Prefill, 512 tokens | 117-142 ms | **42.3 ms** (DSpark 42.9 ms) | 2.8-3.4x |
| Plain decode | 155.8 tok/s | **179.1 tok/s** | 1.15x |
| DSpark decode | 206.8 tok/s (2.17 tokens / step) | **221.7 tok/s** (2.18 tokens / step) | 1.07x |

Every DSpark run returned exactly the plain greedy tokens; the three runs of
each mode are within 0.1 %. PyTorch's peak allocation is 20.3 GB. The prompt
is a review task rendered with the checkpoint's chat template
(`reasoning_effort="none"`), its first 64 and last 448 token ids; the accepted
length depends on the continuation (and so on the prefill's numerics: 2.06 to
2.26 tokens per step across the fast-path variants measured on this prompt),
so the DSpark rate differs on other text.

A decode step reads about 5.6 GB: BF16 attention, projections, gates and LM
head 3.8 GB, the eight experts of each of the 48 layer visits 1.8 GB. At the
5090's 1.79 TB/s that is 3.1 ms; a step takes 5.58 ms.

```bash
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --new 512 --reps 3
```

## Limitations

- Text only, greedy, one sequence per runtime (no sampling, no batching).
- SM120 only; the kernels are written for this checkpoint's geometry.
- Fast prefill is equivalent rather than bit-identical (see *Numerics*); use
  `prefill="exact"` (about 0.7 s for 512 tokens) when the continuation must
  match token-by-token decoding.

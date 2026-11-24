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

A checkpoint with another geometry, layer layout or drafter configuration, or
with a tensor of another dtype or shape than the kernels index (read from the
safetensors headers), is refused before any weight is loaded. `INKLING2_PDL=0`
launches without programmatic dependent launch (for per-kernel profiling).

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
- **Prompt chunks** (16..512 rows): the BF16 projections and the router's
  FP32 logits on the module's own cuBLAS GEMM (`gemm`: BF16 inputs, FP32
  accumulation, reduced-precision reductions disallowed, a workspace per
  runtime; PyTorch's matmul settings are never read or changed), grouped
  block-scaled MMA experts (`mma.sync` `kind::mxf4nvf4`
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
  checkpoint's logits are sensitive to such reorderings, so the comparison is
  teacher-forced (the reference's tokens fed at every step) against an FP32
  reference implementation written from `mlx-lm`'s `inkling2_loop` model with
  the checkpoint's W4A4 / W8A8 expert inputs, next to the reference's own
  perturbation (its quantizer evaluated in the kernels' order of operations):

  | Prompt | Steps | Path | argmax agreement | KL median | KL max |
  |---|---:|---|---:|---:|---:|
  | review task, 512 tokens | 48 | token steps / exact chunks | 48 / 48 | 1.1e-4 | 8.6e-2 |
  | | | fast chunks | 48 / 48 | 9.7e-5 | 8.8e-2 |
  | | | reference against its own variant | 47 / 48 | 2.5e-4 | 5.0e-2 |
  | Python source, 8192 tokens | 16 | exact chunks | 15 / 16 | 4.4e-4 | 1.1e-1 |
  | | | fast chunks | 15 / 16 | 2.4e-3 | 1.7e-1 |
  | | | reference against its own variant | 14 / 16 | 1.5e-3 | 2.2e-1 |
  | Python source, 32768 tokens | 16 | exact chunks | 16 / 16 | 8.5e-5 | 3.3e-2 |
  | | | fast chunks | 16 / 16 | 5.0e-5 | 2.6e-2 |
  | | | reference against its own variant | 15 / 16 | 1.3e-4 | 2.7e-2 |

  The first token after each prompt equals the reference's on every path. Past
  8192 positions the global layers' log-scaled queries take effect (the
  32768-token row). Reordering only the router's FP32 sums of exact products
  moves the 512-token fast row between KL p90 1.5e-2 / max 5.9e-2 (TF32 tensor
  cores) and 4.3e-2 / 1.8e-1 (FP32 CUDA cores), against 4.0e-2 / 8.8e-2 here:
  the spread of any reordering of the fast path.
- **KV cache in BF16** (the checkpoint's attention precision); no FP8 KV.

## Validation

```bash
# anywhere (no GPU): configuration, drafter and tensor checks, chunk planning, routing
python -m pytest tests/test_inkling2_rtx_contracts.py
# SM120 GPU, module built: GEMV rows against one-row launches (bit for bit), MMA gate tiles, the GEMM
python -m pytest tests/test_inkling2_rtx_kernels.py
# SM120 GPU + checkpoint
FLASHRT_INKLING2_DIR=/path/to/Inkling-2-Mini python -m pytest tests/test_inkling2_rtx_pipeline.py
```

The kernel file also checks the GEMM against FP64 (BF16 and FP32 outputs,
strided inputs), that its results do not depend on PyTorch's matmul settings,
and that threads on their own streams get the sequential results bit for bit.
The pipeline file checks verification rows against decode steps (logits bit for
bit, acceptance, committed state with a correct and a wrong draft), exact
prefill against token steps past the sliding window (bit for bit), fast
prefill against exact prefill for several chunk plans, the grouped MMA experts
against the flat passes, that a prefill gives the same logits whatever
PyTorch's matmul settings are and leaves them unchanged, that a checkpoint
with a tensor of another dtype or shape is refused before loading, speculative
against plain greedy decoding (both prefill modes, widths 2, 4 and 8), stop
tokens and lengths, input checks and the chat path.

## RTX 5090 results

RTX 5090 (170 SMs, 32 GB), PyTorch 2.9, CUDA 13; batch 1, greedy, DSpark width
4. Each row is one process (`examples/inkling2_rtx_benchmark.py`): an untimed
request, then plain and DSpark interleaved (three runs at 512 tokens, two
above), 30-60 s between runs and 0.2 s of decode steps before each timed
request (the GPU at its clocks rather than idle); runs of a row differ by less
than 0.1 %. Prefill = prompt to first token (with DSpark the drafter context
is written too: +1.2 ms at 512 tokens, under 1 % from 4096); decode tokens/s =
(new tokens - 1) / time from the first token to the last, 512 new tokens
without stop tokens. The review prompt is the default one; the Python-source
prompts ask to explain the interpreter's standard library (Python 3.12) cut to
the length (`--prompt code --isl N`).

| Prompt | Prefill | Decode, tok/s | DSpark, tok/s (tokens / step) |
|---|---:|---:|---:|
| review task, 512 tokens | 41.5 ms | 179.3 | 214.3 (2.11) |
| Python source, 4096 tokens | 392 ms | 172.4 | 230.2 (2.42) |
| Python source, 8192 tokens | 839 ms | 166.2 | 239.9 (2.66) |
| Python source, 32768 tokens | 4.57 s | 136.3 | 176.1 (2.88) |
| Python source, 131072 tokens | 37.0 s | 71.9 | 73.2 (2.51) |

Sixteen short requests of each kind (`--suite code` / `--suite general`:
19-41 prompt tokens, up to 512 new tokens or the end-of-generation token),
aggregate decode rate and DSpark's accepted tokens per step (range over the
requests):

| Requests | Generated tokens | Decode, tok/s | DSpark, tok/s (tokens / step) |
|---|---:|---:|---:|
| code (functions and classes in Python, C, C++, Rust, Go, SQL, ...) | 7241 | 181.3 | 273.2 (2.62; 2.21-3.26) |
| general (explanations, summaries, arithmetic, letters, stories, ...) | 5454 | 179.5 | 215.4 (2.05; 1.71-2.75) |

Every DSpark run returned exactly the plain greedy tokens.

- **Decode.** At 512 tokens of context a step reads about 5.6 GB: BF16
  attention, projections, gates and LM head 3.8 GB (the looped layers are read
  once per pass), the eight experts of each of the 48 layer visits 1.8 GB. At
  the 5090's 1.79 TB/s that is 3.1 ms; a step takes 5.58 ms. With context the
  global layers' KV adds 32 KiB per token per step (4.3 GB at 128K, a 5.5 ms
  floor against the step's 13.9 ms).
- **DSpark.** The accepted length depends on the text, and on one prompt on the
  continuation the prefill's rounding leads to: reordering only the prefill's
  sums moved the 4096-token prompt from 3.14 to 2.42 tokens per step (the
  suites average over sixteen requests). A speculative step costs 1.8 plain
  steps at 512 tokens of context and 2.5 at 128K (34.3 ms against 13.9 ms),
  where DSpark only matches plain decoding: the verification's attention reads
  the context's KV once per verified row, and the drafter's full-attention
  layer once per block position.
- **Prefill.** 512 tokens: 19.9 ms in the expert MMAs, 10.2 ms in the GEMMs,
  3.5 ms in attention, 3.3 ms in the layer-visit boundaries, 1.4 ms after
  attention, 3.4 ms in the rest (eager per-kernel times). Long prompts are
  bound by the global layers' attention (12.3K prompt tokens/s at 512, 3.5K at
  128K). The exact path takes 0.71 s for 512 tokens, 12.6 s for 8192 and 61 s
  for 32768.

## Memory

| | |
|---|---:|
| Text decoder weights (non-expert BF16 / FP32 3.0 GiB, NVFP4 / MXFP8 experts 14.6 GiB) | 17.6 GiB |
| DSpark drafter weights | 0.6 GiB |
| Derived copies: E4M3 LM head, folded drafter correction | 0.6 GiB |
| KV per token of `max_seq`: global layers 32 KiB, drafter 12 KiB | 44 KiB |
| Sliding-window rings (40 layer visits x 1024 slots), fixed | 160 MiB |
| PyTorch peak allocation, `max_seq` 4096 / 36864 / 132096 | 20.3 / 21.5 / 25.7 GiB |

```bash
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --new 512 --reps 3
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --prompt code --isl 32768 --max-seq 36864 --reps 2
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --suite code --cooldown 5
```

## Limitations

- Text only, greedy, one sequence per runtime (no sampling, no batching).
- SM120 only; the kernels are written for this checkpoint's geometry.
- Fast prefill is equivalent rather than bit-identical (see *Numerics*); use
  `prefill="exact"` (about 0.7 s for 512 tokens) when the continuation must
  match token-by-token decoding.

# Inkling 2 Mini on RTX 5090 (SM120)

FlashRT runs the text decoder of `thinkingmachines/Inkling-2-Mini` on the RTX
5090 with its own CUDA kernels (`flash_rt_inkling2`, a separate module behind
`FLASHRT_ENABLE_INKLING2`), and drives speculative decoding with the
checkpoint's DSpark drafter (`mtp.safetensors`) or the DFlash 2 drafter
([incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)).
The checkpoint is used at its
released precision: BF16 attention, gates, embedding and LM head, FP32
depth-attention queries; NVFP4 experts (E2M1 codes, E4M3 scales per 16 inputs,
per-expert global scale) in layers 0-19 and MXFP8 experts (E4M3 codes, E8M0
scales per 32) in layers 20-23, multiplied with the expert inputs quantized as
`hf_quant_config.json` requires (NVFP4 with the `fouroversix_mae` scale rule,
MXFP8). The model's weights are loaded as stored; nothing is re-quantized. (The
drafters' draft-only products read derived copies: DSpark E4M3 copies of the LM
head and of its folded correction, DFlash 2 E4M3 copies of its layer products
and an NVFP4 copy of the LM head; they choose drafts, never the verified
tokens.) `fp8=True` is an opt-in variant of the target itself: decode steps and
verification rows read E4M3 copies of the attention / output projections and
the LM head (see *Numerics*).

Every forward is a recorded program: a fixed list of kernel launches over
device pointers that reads the position and the input tokens from device
memory, captured once as a CUDA graph (with programmatic dependent launch) and
replayed. Text only (the vision tower is not loaded), greedy, one sequence.

## Requirements

| | |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` directory (`config.json`, `model-*.safetensors`, `mtp.safetensors` for DSpark, `tokenizer.json`, `chat_template.jinja`) |
| DFlash 2 drafter (optional) | `incoai/Inkling-2-Mini-DFlash2` directory (`config.json`, `model.safetensors`) |
| GPU | RTX 5090 or another SM120 GPU with at least 24 GB (PyTorch peak at `max_seq=4096`: 20.3 GiB with DSpark, 21.6 GiB with DFlash 2 and `fp8=True`) |
| Software | CUDA 12.8 or newer (block-scaled FP4 / FP8 `mma.sync` of sm_120a), PyTorch with CUDA, `safetensors`; `tokenizers` + `jinja2` for the tokenizer and chat template |

```bash
cmake -B build -DGPU_ARCH=120 -DFLASHRT_ENABLE_INKLING2=ON
cmake --build build --target flash_rt_inkling2 -j     # flash_rt/flash_rt_inkling2*.so
```

A checkpoint or drafter with another geometry, layer layout or drafter
configuration, or with a tensor of another dtype or shape than the kernels
index (read from the safetensors headers), is refused before any weight is
loaded. `INKLING2_PDL=0`
launches without programmatic dependent launch (for per-kernel profiling).

## Usage

```python
from flash_rt.frontends.torch.inkling2_rtx import Inkling2TorchFrontendRtx

fe = Inkling2TorchFrontendRtx("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True)
print(fe.chat([{"role": "user", "content": "Write a quicksort in Python."}],
              max_new_tokens=512, reasoning_effort="none"))

# The DFlash 2 drafter instead of DSpark (adaptive widths by default).
fe = Inkling2TorchFrontendRtx("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True,
                              dflash2="/path/to/Inkling-2-Mini-DFlash2")

# Token-level streaming; speculative and prefill mode per call.
ids = fe.tokenizer.encode(fe.apply_chat_template([{"role": "user", "content": "Hello"}]))
for block in fe.stream(ids, max_new_tokens=256, speculative=False, prefill="exact"):
    print(fe.tokenizer.decode(block), end="", flush=True)
```

| Argument | Meaning |
|---|---|
| `max_seq` | KV capacity of the global-attention layers (sliding-window layers keep 1024 slots): prompt plus generated tokens plus a margin, 1 position for plain decoding and `(check_rounds + 1) * width + 8` for speculative decoding (28 with DSpark's default width 4, 48 with DFlash 2's adaptive widths) |
| `speculative` | load the drafter and decode speculatively by default; `stream(..., speculative=False)` decodes plainly |
| `dflash2` | the DFlash 2 drafter's directory (with `speculative=True`) instead of DSpark |
| `adaptive` | DFlash 2: each step verifies the number of rows (2..8) with the most expected tokens per unit of time (default); `False`: `draft_width` rows |
| `draft_width` | verified rows per speculative step: the anchor and `draft_width - 1` drafts (2..8, default 4) |
| `fp8` | decode steps and verification rows read E4M3 copies of the attention / output projections and the LM head (default `False`) |
| `prefill` | `"fast"`: tensor-core chunks (default); `"exact"`: per-row chunks, bit-identical to decoding the prompt token by token |
| `check_block`, `check_rounds` | decode steps / speculative steps between host reads of the generated tokens (a stop token is seen that late) |

Programs are recorded and captured on first use; `fe.warmup(prompt_len)`
captures the ones a prompt of that length and the decode loop need (with
DFlash 2's adaptive widths, the round of every width). Each `stream` call
starts a new sequence.

## Components

| Part | Where |
|---|---|
| Frontend: validation, streaming, chat template | `flash_rt/frontends/torch/inkling2_rtx.py` |
| Runtime: configuration checks, weights, buffers, program recording, prefill planning | `flash_rt/frontends/torch/_inkling2_rtx_runtime.py` |
| DSpark drafter programs | `flash_rt/frontends/torch/_inkling2_rtx_mtp.py` |
| DFlash 2 drafter programs, adaptive widths | `flash_rt/frontends/torch/_inkling2_rtx_dflash2.py` |
| Kernels | `csrc/kernels/inkling2/`, bindings `csrc/inkling2_bindings.cpp` |

The kernels by program:

- **Decode** (one row, 434 launches): multi-output GEMV with RMSNorm and
  gate prologues, the layer-visit boundary (depth attention over the block
  residuals, the loop messages between passes of the looped layers), the
  message gates' down projections, mixing and input norms in one cluster
  kernel (`msg_gate`), split-key attention with relative position bias, L2
  clips, sinks and the output gate, fused routing + expert kernels (input
  quantization behind the first weight loads, exact FP4 / FP8 block products;
  w2 redoes the routing so that its first weight loads run under w13), the LM
  head with the greedy argmax. Weights of the next kernels are prefetched into
  L2 before each kernel waits on its predecessor.
- **Verification** (2..8 rows, 437 launches for 8): the same per-row
  arithmetic over staged rows.
  The visit boundary normalizes the rows' attention inputs, the attention
  blocks compute the new rows' K / V themselves (up to 4 rows), the router GEMV
  runs inside the routing pass and w13 + w2 in one flat kernel over the
  distinct experts of the rows (dynamic chunks; the w2 phase waits on per-group
  counts of finished w13 items instead of a kernel boundary and quantizes its
  inputs itself); greedy targets, acceptance, and a commit of the accepted
  rows' KV and convolution states.
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
- **DFlash 2** (SGLang's `DFlash2DraftModel` with the Inkling integration): the
  drafter's own context K / V from four target features per committed position
  (the hidden states entering visits 5, 23 and 41 and the final normalized row:
  `fc`, `hidden_norm`, per-layer K / V with `k_norm` and RoPE); a block of the
  anchor and seven mask tokens through five bidirectional layers (32 query / 8
  KV heads over a 2047-position window and the block, two-tap dynamic grouped
  convolutions around attention and MLP, SwiGLU), attention with each KV head's
  keys read once on tensor cores; the target's LM head on the mask rows (NVFP4
  copy, tensor cores), their 16 largest logits and the candidate selector's
  greedy path (predecessor / successor codebooks of rank 256, softcap 30 after
  the correction). The layer products read E4M3 copies (`gemv_fp8`); prompt
  chunks write the context from BF16 GEMMs. One CUDA graph per round; with
  adaptive widths the host reads the round's accepted count and the next
  drafts' confidences (pinned memory written by the graph) and picks the next
  round's rows from per-confidence-bin agreement rates learned online.

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
- **`fp8=True` is not the checkpoint's precision.** Decode steps and
  verification rows read E4M3 copies (FP32 scale per row) of the attention
  projections, the output projections and the LM head on FP16 tensor cores;
  prompt chunks keep the BF16 weights. Verification rows stay bit-identical to
  this variant's decode steps, so speculative decoding returns its greedy
  output, which can differ from the BF16 checkpoint's. Teacher-forced against
  the FP32 reference on the review prompt (47 steps after the first token), its
  argmax is the reference token at 46 steps (BF16: 47), KL median / p90 / max
  5.2e-5 / 3.9e-2 / 8.7e-2 (BF16: 9.7e-5 / 3.0e-2 / 8.8e-2); on the BF16 greedy
  continuations of the review and Python-source prompts (128 steps each) its
  argmax agrees with BF16's at 126 and 120 steps.
- **Drafts never change the output.** Both drafters only decide how many
  tokens a step accepts; their derived copies (E4M3 / NVFP4) and FP32 sums in
  another order than the reference implementations move the acceptance, not
  the tokens.
- **KV cache in BF16** (the checkpoint's attention precision); no FP8 KV.

## Validation

```bash
# anywhere (no GPU): configuration, drafter and tensor checks, chunk planning, routing
python -m pytest tests/test_inkling2_rtx_contracts.py
# SM120 GPU, module built: GEMV rows against one-row launches (bit for bit), MMA gate tiles, the GEMM
python -m pytest tests/test_inkling2_rtx_kernels.py
# SM120 GPU + checkpoint
FLASHRT_INKLING2_DIR=/path/to/Inkling-2-Mini python -m pytest tests/test_inkling2_rtx_pipeline.py \
    tests/test_inkling2_rtx_experts.py tests/test_inkling2_rtx_row_attention.py
# SM120 GPU + checkpoint + DFlash 2 drafter (BF16 and fp8=True)
FLASHRT_INKLING2_DIR=/path/to/Inkling-2-Mini FLASHRT_INKLING2_DFLASH2_DIR=/path/to/Inkling-2-Mini-DFlash2 \
    python -m pytest tests/test_inkling2_rtx_dflash2.py
```

The kernel file also checks the GEMM against FP64 (BF16 and FP32 outputs,
strided inputs), that its results do not depend on PyTorch's matmul settings,
and that threads on their own streams get the sequential results bit for bit;
the tensor-core GEMVs (BF16, E4M3, NVFP4 weights) against one-row launches (bit
for bit) and the weights they represent, the fp8 norm prologue against
`norm_rows`, `msg_gate` against the three kernels it replaces (bit for bit),
and the DFlash 2 top-16 and attention kernels against torch / FP64 (the
attention with decisive keys at each row's window edge).
The pipeline file checks verification rows against decode steps (logits bit for
bit, acceptance, committed state with a correct and a wrong draft), exact
prefill against token steps past the sliding window (bit for bit), fast
prefill against exact prefill for several chunk plans, the grouped MMA experts
against the flat passes, that a prefill gives the same logits whatever
PyTorch's matmul settings are and leaves them unchanged, that a checkpoint
with a tensor of another dtype or shape is refused before loading, speculative
against plain greedy decoding (both prefill modes, widths 2, 4 and 8), stop
tokens and lengths, input checks and the chat path. The experts file checks
the verification rows' fused expert kernel against the split passes
(activations, routing and outputs of 2..8 rows bit for bit, an NVFP4 and an
MXFP8 layer) and that it completes with the whole GPU's output on a green
context of 16 SMs (most of its blocks waiting for an SM); the row-attention
file prompt chunks on the per-row attention kernels (`flash_prefill=False`)
across runs and against exact prefill. The DFlash 2 file checks,
for BF16 and `fp8=True`, verification rows against decode steps (widths 2, 5,
8), speculative against plain greedy decoding (adaptive and fixed widths, both
prefill modes, stop tokens), and the kernels' drafter context against the
PyTorch reference path; the contracts file the drafter's configuration and
tensors.

## RTX 5090 results

RTX 5090 (170 SMs, 32 GB), PyTorch 2.9, CUDA 13; batch 1, greedy, DSpark width
4, DFlash 2 with adaptive widths. Each row is one process
(`examples/inkling2_rtx_benchmark.py`): an untimed request, then plain and
speculative decoding interleaved (three runs, two at 128K), 30 s between runs
and 0.2 s of decode steps before each timed request (the GPU at its clocks
rather than idle); runs of a row differ by at most 0.3 %. Prefill = prompt to
first token (plain; writing the drafter context adds 1 ms at 512 tokens, under
1 % from 4096); decode tokens/s = (new tokens - 1) / time from the first
token to the last, 512 new tokens without stop tokens. The review prompt is the
default one; the Python-source prompts ask to explain the interpreter's
standard library (Python 3.12) cut to the length (`--prompt code --isl N`).

| Prompt | Prefill | Decode, tok/s | DSpark, tok/s (tokens / step) | DFlash 2, tok/s (tokens / step) |
|---|---:|---:|---:|---:|
| review task, 512 tokens | 40.1 ms | 219.6 | 274.3 (2.11) | 381.2 (2.86) |
| Python source, 512 tokens | 40.6 ms | 219.4 | 338.1 (2.62) | 441.9 (3.36) |
| Python source, 4096 tokens | 382 ms | 207.9 | 287.0 (2.42) | 345.7 (2.80) |
| Python source, 8192 tokens | 818 ms | 200.4 | 293.8 (2.66) | 327.1 (2.66) |
| Python source, 32768 tokens | 4.48 s | 157.7 | 203.6 (2.88) | 254.9 (3.03) |
| Python source, 131072 tokens | 36.8 s | 78.2 | 78.0 (2.51) | 100.4 (2.38) |

With `fp8=True` (decode steps and verification rows on the E4M3 copies; its
own greedy output, so its accepted lengths differ):

| Prompt | Decode, tok/s | DSpark, tok/s (tokens / step) | DFlash 2, tok/s (tokens / step) |
|---|---:|---:|---:|
| review task, 512 tokens | 266.2 | 304.9 (2.13) | 415.8 (2.67) |
| Python source, 512 tokens | 266.2 | | 520.3 (3.59) |
| Python source, 4096 tokens | 250.7 | | 346.3 (2.32) |
| Python source, 32768 tokens | 181.7 | | 265.8 (2.77) |

Sixteen short requests of each kind (`--suite code` / `--suite general`:
19-41 prompt tokens, up to 512 new tokens or the end-of-generation token),
aggregate decode rate and the accepted tokens per step (range over the
requests):

| Requests | Generated tokens | Decode, tok/s | DSpark, tok/s (tokens / step) | DFlash 2, tok/s (tokens / step) |
|---|---:|---:|---:|---:|
| code (functions and classes in Python, C, C++, Rust, Go, SQL, ...) | 7241 | 218.7 | 352.4 (2.62; 2.21-3.26) | 518.5 (3.96; 2.74-5.65) |
| general (explanations, summaries, arithmetic, letters, stories, ...) | 5454 | 216.2 | 278.3 (2.05; 1.71-2.75) | 397.2 (2.80; 2.29-4.25) |

Every speculative run returned exactly the plain greedy tokens of its variant.

- **Decode.** At 512 tokens of context a step reads about 5.6 GB: BF16
  attention, projections, gates and LM head 3.8 GB (the looped layers are read
  once per pass), the eight experts of each of the 48 layer visits 1.8 GB. At
  the 5090's 1.79 TB/s that is 3.1 ms; a step takes 4.55 ms (3.76 ms with
  `fp8=True`). With context the global layers' KV adds 32 KiB per token per
  step (4.3 GB at 128K).
- **Verification.** At 512 tokens of context a DFlash 2 round (verification,
  context of the accepted rows, next drafts) takes 5.9 ms with 2 rows and 9.4
  ms with 8 (`fp8=True`: 5.0 and 8.3 ms), against 4.55 ms for a decode step: the
  rows share every weight read, and each verified row adds the attention over
  its context and the distinct experts it routes to. The adaptive widths verify
  4.5 rows per step on the review prompt.
- **Long contexts.** The verification's attention reads the context's KV once
  per verified row, and the drafters' attention their own context: at 128K
  DSpark (a full-attention drafter layer) only matches plain decoding, DFlash 2
  (its drafter attends a window of 2047 keys) stays 28 % ahead.
- **Prefill.** Long prompts are bound by the global layers' attention (12.7K
  prompt tokens/s at 512, 3.6K at 128K). The exact path (per-row chunks of 8)
  takes 0.55 s for 512 tokens and 9.8 s for 8192.

## Memory

| | |
|---|---:|
| Text decoder weights (non-expert BF16 / FP32 3.0 GiB, NVFP4 / MXFP8 experts 14.6 GiB) | 17.6 GiB |
| DSpark drafter weights; derived copies (E4M3 LM head, folded drafter correction) | 0.6 GiB; 0.6 GiB |
| DFlash 2 drafter: BF16 (fc, K / V rows, codebooks, norms), E4M3 copies of its products, NVFP4 copy of the LM head | 0.30 GiB, 0.39 GiB, 0.24 GiB |
| `fp8=True`: E4M3 copies of the attention / output projections and the LM head | 1.0 GiB |
| KV per token of `max_seq`: global layers 32 KiB, DSpark 12 KiB, DFlash 2 20 KiB | |
| Sliding-window rings (40 layer visits x 1024 slots), fixed | 160 MiB |
| PyTorch peak allocation, `max_seq` 4096: DSpark / DFlash 2 / DFlash 2 with `fp8=True` | 20.3 / 20.6 / 21.6 GiB |
| PyTorch peak allocation, `max_seq` 132096: DSpark / DFlash 2 | 25.7 / 26.9 GiB |

```bash
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --new 512 --reps 3
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --dflash2 /path/to/Inkling-2-Mini-DFlash2 [--fp8]
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --prompt code --isl 32768 --max-seq 36864 --reps 2 [--dflash2 ...]
python examples/inkling2_rtx_benchmark.py --checkpoint /path/to/Inkling-2-Mini --suite code --cooldown 5 [--dflash2 ...]
```

## Limitations

- Text only, greedy, one sequence per runtime (no sampling, no batching).
- SM120 only; the kernels are written for this checkpoint's geometry (and the
  DFlash 2 drafter's).
- The flat expert passes of verification need a GPU of at least 64 SMs. Their
  w13 + w2 kernel waits only on w13 chunks that running blocks have claimed,
  so it also completes under an SM partition (MPS limits, green contexts),
  more slowly.
- Fast prefill is equivalent rather than bit-identical (see *Numerics*); use
  `prefill="exact"` (about 0.55 s for 512 tokens) when the continuation must
  match token-by-token decoding.

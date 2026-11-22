# Inkling 2 Mini on Apple Silicon (M4 Pro)

FlashRT runs the text decoder of `thinkingmachines/Inkling-2-Mini` natively on
Apple Silicon through the Apple Metal plan (`cpp/apple`): single-sequence
decoding is one resident dispatch that runs up to eight tokens, prefill and
batched decoding run as row programs recorded once as indirect command buffers,
and the checkpoint's DSpark drafter (`mtp.safetensors`) drives speculative
decoding. The checkpoint is used in its released precisions: BF16 attention,
gates, embedding and LM head; NVFP4 experts (E2M1 codes, E4M3 scales per 16
inputs, a per-expert FP32 global scale) in layers 0-19 and MXFP8 experts (E4M3
codes, E8M0 scales per 32 inputs) in layers 20-23. Weights are copied verbatim
into one Metal buffer; nothing is re-quantized. The runtime needs numpy and the
plan library only (no PyTorch, no MLX).

The arithmetic contract for a one-token step is `mlx-lm`'s (the reference
engine below, with Inkling 2 support): a step reproduces its logits bit for bit
(see *Numerics*). This entry is text-only (the vision tower is not loaded).

## Requirements

| | |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` directory (`config.json`, `generation_config.json`, `model-*.safetensors`, `mtp.safetensors` for speculative decoding, `tokenizer.json`, `chat_template.jinja`); validated revision `ed40e7da625f594fc9c4990f5f66be24873c22d7` |
| Hardware | Apple M4 Pro / M4 Max (Metal GPU family Apple9), 20+ GPU cores (the decode step keeps 36 threadgroups co-resident), 48 GB unified memory or more (19.5 GB weights; KV and conv state per sequence) |
| Software | macOS 26.2 or newer (validated on 26.7.3), Xcode command line tools to build the plan library |
| Python | 3.10+, numpy; `tokenizers` + `jinja2` for the tokenizer and chat template |

```bash
bash scripts/apple/build_native.sh     # .cache/apple-native/libflashrt_apple_plan.dylib
```

Kernels are compiled from embedded Metal source when the runtime is created;
`FLASHRT_APPLE_PLAN_LIBRARY` overrides the library path. A checkpoint with
another geometry, other hard-coded constants or another drafter is refused
before anything is compiled.

## Usage

```python
from flash_rt.frontends.torch.inkling2_mps import Inkling2TorchFrontendMps

fe = Inkling2TorchFrontendMps("/path/to/Inkling-2-Mini", max_seq=8192)
text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512,
               reasoning_effort="low")

# Token-level streaming (blocks of up to eight tokens), greedy or top-k / top-p sampling.
ids = fe.tokenizer.encode(fe.apply_chat_template(messages, reasoning_effort="medium"))
for block in fe.stream(ids, max_new_tokens=256, temperature=1.0, top_p=0.95, top_k=20, seed=7):
    ...

# DSpark speculative decoding: the drafter proposes, the model verifies greedily.
spec = Inkling2TorchFrontendMps("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True)
```

`stream` stops after an end-of-generation token (`generation_config.json`'s
`eos_token_id`; `stop_ids=()` disables stopping). Prompts of any length are
prefilled in chunks of 1024, 512, ..., 1 rows. `temperature=0`, the default, is
greedy (on-device argmax, lowest id on ties). With a temperature the step
samples on the GPU: the `top_k` (1..32) largest logits, cut to the nucleus
`top_p` of the full-vocabulary probabilities, then the maximum of
log-probability / temperature plus Gumbel noise drawn from `seed` and the
token position (the first token after the prompt applies the same rule on the
host). The checkpoint's `generation_config.json` asks for temperature 1 without
a top-k; this sampler always keeps at most 32 candidates. `apply_chat_template`
renders the checkpoint's template, passing keyword arguments such as
`reasoning_effort` through; `chat(..., sampling={...})` takes the sampling
arguments. Speculative decoding is greedy.

Several sequences (the batched decode step and batched speculative decoding)
are reached through the runtime:

```python
from flash_rt.frontends.torch._inkling2_mps_runtime import Inkling2Runtime

rt = Inkling2Runtime(path, max_seq=2048, sequences=16, row_capacity=1024,
                     mtp_batches=[16], mtp_prefill=(1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1))
out = rt.mtp_generate(prompts, new_tokens=1024)        # 16 prompts, greedy, DSpark
```

`examples/apple/inkling2_benchmark.py` shows the plain batched loop (one
row-program step per token for all sequences).

## Components

| Layer | Path |
|---|---|
| Checkpoint reader (numpy, zero-copy safetensors), geometry and drafter checks | `flash_rt/models/inkling2/checkpoint.py` |
| Runtime (weights, buffers, decode step, row programs, chunked prefill) | `flash_rt/frontends/torch/_inkling2_mps_runtime.py` |
| DSpark drafter and the speculative loop | `flash_rt/frontends/torch/_inkling2_mtp.py` |
| Frontend (tokenizer, chat template, streaming) | `flash_rt/frontends/torch/inkling2_mps.py` |
| Kernels | `csrc/apple/inkling2/`: shared device functions, resident decode step, per-row kernels, TensorOps products, drafter |
| Benchmark drivers | `examples/apple/inkling2_benchmark.py`, `examples/apple/inkling2_mlx_benchmark.py` (reference engine) |

The model: 24 stored layers in four blocks of six; layers 6-17 run three
passes, each visit with its own caches, so a token visits 48 layers. Hidden
2304; 24 query / 8 KV heads of 128 with a relative-position bias, per-head sink
logits, a sigmoid output gate and L2-clipped queries / keys; 20 sliding-window
layers (window 1024) and 4 global ones; short convolutions on K, V, the
attention output and the MLP output; block attention residuals (a softmax over
the earlier blocks' states feeds every layer); message gates between passes;
128 experts with the top 8 by sigmoid score and a softcapped SwiGLU.

**Decode (one sequence).** One dispatch of 36 co-resident threadgroups runs up
to eight tokens with device-scope barriers between phases. A token reads about
5.8 GB at 2K context (attention and LM head weights, eight experts per visit,
the KV it attends to).

**Row programs (prefill, batched decode, verification).** Each layer visit is a
sequence of dispatches over R rows (prefill positions or sequences): per-row
kernels for norms, block attention residuals, convolutions, attention, routing
and the expert combination (the decode step's device functions), and products:

- up to 4 rows, multi-row matrix-vector products and a per-expert multi-row
  quantized product that decodes each NVFP4 block once for all of the expert's
  rows; each row's arithmetic is the decode step's, so a row's result does not
  depend on the batch;
- more rows, TensorOps products (the independent products of a visit in one
  dispatch) and expert tiles grouped by expert, with the experts that received
  at most 3 rows on the per-expert quantized product.

**Speculative decoding.** The drafter follows SGLang's
`Inkling2ForConditionalGenerationMTP` (a DFlash block drafter with DSpark's RNN
Markov head): the model's hidden state entering visits 5, 23 and 41 and its
final normalized state are mixed per drafter layer into context K / V; a block
of 8 positions (the last accepted token and 7 mask embeddings) runs 3
bidirectional layers; each drafted position takes the LM head's logits plus
the Markov head's bias (its `hidden_out` folded into the LM head) and the
greedy token feeds the next position. The model verifies the anchor and the
first W - 1 drafts of every sequence in one row program (W = 8 for 1-4
sequences, 4 for up to 16, 3 beyond: verification grows with the rows while the
accepted length saturates). Verification stages the conv states and
sliding-window K / V of its rows; only the accepted rows are committed.

## Numerics

- **One-token steps are bit-identical to `mlx-lm`.** The decode step follows
  MLX 0.32 operation by operation for a single position: BF16 and FP32
  matrix-vector products, the NVFP4 / MXFP8 quantized products, RMSNorm, the
  depth-attention and routing kernels `mlx-lm` compiles, the vector attention
  (32 key-strided partitions below 1024 keys, two passes with BF16 partials
  from 1024) with sinks and an additive BF16 bias, the convolutions and every
  elementwise rounding. The exponential is the fast one inside the kernels
  `mlx-lm` compiles at run time and the precise one elsewhere.
- **Few-row programs** (up to 4 rows) give every row the decode step's hidden
  states at position 0 and are batch-invariant; across positions their
  attention partitions differ from the step's.
- **Prefill and larger batches are equivalent, not bit-identical**: products
  as TensorOps tiles (FP32 accumulation), FlashRT's attention partitions. For a
  1024-token prompt the last position's logits differ from `mlx-lm`'s
  one-token evaluation by KL 9.0e-6, below the reference engine's own
  difference between its batched and stepwise evaluation (KL 4.7e-5); the
  first token agrees.
- **Sampling** keeps the tokens `mlx-lm`'s sampler keeps for the same top-k and
  top-p (the nucleus over the full distribution intersected with the top-k) and
  draws by Gumbel-max at the temperature with FlashRT's counter-based noise:
  sampled sequences are reproducible per seed but are not the reference
  engine's sequences.
- **Speculative decoding** commits the tokens the verification selects: the
  model's greedy continuation in the row programs' rounding. It follows the
  one-token greedy output until the two orders of summation round two leading
  candidates differently (observed once in the first 128 tokens of a code
  prompt: two raw logits within one BF16 step became an exact tie).

## Validation

### Tests

```bash
# anywhere (no GPU): configuration and drafter checks, refusals, routing, chunking, helpers
python -m pytest tests/test_inkling2_apple_contracts.py
# on the Mac, after scripts/apple/build_native.sh:
FLASHRT_INKLING2_DIR=/path/to/Inkling-2-Mini python -m pytest tests/test_inkling2_apple_pipeline.py
```

The Mac file checks the one-row program against the decode step (every
visit's hidden state, bit for bit), batch invariance of a four-row decode step,
chunked against one-chunk prefill, speculative against plain greedy decoding,
sampling (top-k 1 equals greedy, seeds reproduce), the input checks and the
chat path.

### Equivalence with the reference engine (M4 Pro, macOS 26)

- **Decode steps.** 2100 consecutive one-token steps from an empty context
  (Python standard-library text; positions 0-2099, across the 1024-key switch
  of the attention and the sliding-window wrap): the FP32 logits of every step
  are bit-identical to `mlx-lm`'s (0 of 2100 differ), as are the hidden states
  after every one of the 48 layer visits at the 12 steps where they were
  recorded.
- **Prefill.** Last-position logits after the same text's first 1024 tokens as
  one chunk: KL(reference || FlashRT) = 9.0e-6, largest logit difference 0.68;
  the reference engine's batched and one-token evaluations of the same prompt
  differ by KL 4.7e-5. The first token is the same.

## M4 Pro numbers

Mac mini M4 Pro (14-core CPU, 20-core GPU, 64 GB), macOS 26.7.3. Reference
engine: `mlx-lm` with Inkling 2 support (the `inkling2_loop` model of mlx-lm
pull request 2086) on MLX 0.32.4, the same checkpoint and the same token ids:
the Python standard-library sources tokenized with the checkpoint's tokenizer
(`examples/apple/inkling2_benchmark.py --make-corpus`). The model card's
benchmark shape: C concurrent sequences, each a 1024-token prompt and exactly
1024 generated tokens (no stop token), greedy. Both engines run in the
foreground, interleaved per concurrency (reference, FlashRT, FlashRT with
DSpark, reference, two-minute pauses); every driver runs three times on
different prompt slices and reports the median of the last two.

### Decode (tokens/s over the C sequences, after every sequence has its first token)

| C | mlx-lm, two runs | FlashRT | vs mlx-lm | FlashRT + DSpark | vs mlx-lm | accepted tokens per step |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.0 / 26.0 (`stream_generate` 28.4 / 28.5) | **35.2** | 1.24x | **54.2** (45.9-62.5) | 1.91x | 4.8-6.6 of 8 |
| 4 | 52.2 / 52.2 | **64.0** | 1.23x | **81.3** (75.7-86.9) | 1.56x | 5.3-5.4 of 8 |
| 16 | 71.2 / 69.9 | **101.9** | 1.44x | **131.3** (126.4-136.2) | 1.86x | 3.5-3.6 of 4 |
| 64 | 113.1 / 116.7 | **169.1** | 1.47x | **185.6** (180.3-190.9) | 1.62x | 2.8 of 3 |

mlx-lm runs these concurrency levels on its BatchGenerator (the engine of
`mlx_lm.server`); at C = 1 its single-sequence `stream_generate` is faster and
is the comparison given in parentheses. FlashRT's plain decoding is the resident
step at C = 1 and the batched row program above. DSpark's ranges are the two
measured prompt slices: the accepted length depends on the text. From C = 16
the M4 Pro's matrix units, not its bandwidth, bound a step, so verifying more
rows buys less.

### Prefill (seconds until every sequence has its first token)

| C | mlx-lm | FlashRT |
|---:|---:|---:|
| 1 | 2.17-2.25 | 2.16 |
| 4 | 9.26-9.33 | 9.45 |
| 16 | 36.4-36.7 | 39.4 |
| 64 | 151.5-153.1 | 155.2 |

FlashRT prefills the prompts one after another (chunks of up to 1024 rows);
mlx-lm prefills up to eight prompts together.

### A one-sequence decode step

At 1-2K context a token reads about 5.8 GB: the attention, output-projection and
LM-head weights (3.7 GB), the eight experts of each of the 48 visits (1.8 GB)
and the KV it attends to. At the M4 Pro's measured 254 GB/s streaming rate that
is 22.8 ms; the step takes 28.4 ms (35.2 tokens/s). The projections, output
projection and LM head stream at that rate, the expert products at 85-94 % of
it; the remaining time is attention (about 1.5 ms) and the device-scope
barriers between the step's phases (about 2 ms).

### The model card's numbers

The model card reports the same benchmark shape on one NVIDIA B300 with SGLang:
33.6 / 126.8 / 434.5 / 1319.9 output tokens/s at C = 1 / 4 / 16 / 64 without
the drafter and 147.8 / 534.6 / 1620.5 / 4058.8 with it (29.7 / 31.4 / 36.1 /
47.5 ms and 5.2 / 6.1 / 7.3 / 10.9 ms per token). Those runs measure SGLang's
output throughput over the whole benchmark, prefill included, on a GPU with
about 30 times the M4 Pro's memory bandwidth. At C = 1 without the drafter, a
FlashRT token on the M4 Pro takes 28.4 ms.

### Reproduce

```bash
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --make-corpus corpus.i32
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1,4,16,64
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1,4,16,64 --mtp
python examples/apple/inkling2_mlx_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1,4,16,64
python examples/apple/inkling2_mlx_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --stream
```

## Limitations

- Text only. Sampling keeps at most 32 candidates (top-k 1..32); speculative
  decoding is greedy.
- The frontend serves one sequence; several sequences go through the runtime.
- The resident decode step keeps 36 threadgroups co-resident (20-core GPU);
  smaller GPUs are not supported.
- Verification widths are tuned on the M4 Pro; the accepted length depends on
  the text (the numbers above use source code).

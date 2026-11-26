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
into one Metal buffer; nothing is re-quantized (the row programs add a copy of
the NVFP4 experts with the same codes and scales in another order). The runtime needs numpy and the
plan library only (no PyTorch, no MLX).

The arithmetic contract for a one-token step is `mlx-lm`'s (the reference
engine below, with Inkling 2 support): a step reproduces its logits bit for bit
(see *Numerics*). This entry is text-only (the vision tower is not loaded).

## Requirements

| | |
|---|---|
| Checkpoint | `thinkingmachines/Inkling-2-Mini` directory (`config.json`, `generation_config.json`, `model-*.safetensors`, `mtp.safetensors` for speculative decoding, `tokenizer.json`, `chat_template.jinja`); validated revision `ed40e7da625f594fc9c4990f5f66be24873c22d7` |
| Optional drafter | `incoai/Inkling-2-Mini-DFlash2` directory (`config.json`, `model.safetensors`; Apache-2.0) for `dflash2=` |
| Hardware | Apple M4 Pro / M4 Max (Metal GPU family Apple9), 20+ GPU cores (the resident steps keep 36-40 threadgroups co-resident), 48 GB unified memory or more (19.5 GB weights, 11.5 GB re-ordered NVFP4 experts for the row programs; KV and conv state per sequence) |
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
# verify="exact": every verified row repeats the decode step's arithmetic (output = greedy decode, bit for bit)
exact = Inkling2TorchFrontendMps("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True, verify="exact")
# q8=True: attention projections and LM head from Q8_0 copies made at load (faster; not the checkpoint's numerics)
q8 = Inkling2TorchFrontendMps("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True, q8=True)
# adaptive=False: the fast verification always checks 8 rows (default: 2-8 per step from the drafts' probabilities,
# 12 or 16 with lookup drafts from the text; lookup=False: at most 8)
fixed = Inkling2TorchFrontendMps("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True, adaptive=False)
# dflash2=...: draft with the DFlash 2 drafter (block diffusion + candidate selector) instead of DSpark
df2 = Inkling2TorchFrontendMps("/path/to/Inkling-2-Mini", max_seq=8192, speculative=True,
                               dflash2="/path/to/Inkling-2-Mini-DFlash2")
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
| DFlash 2 drafter | `flash_rt/frontends/torch/_inkling2_dflash2.py`, `csrc/apple/inkling2/inkling2_dflash2_bf16.h` |
| Frontend (tokenizer, chat template, streaming) | `flash_rt/frontends/torch/inkling2_mps.py` |
| Kernels | `csrc/apple/inkling2/`: shared device functions, resident decode step, resident verification step (`inkling2_verify_resident_bf16.h`), per-row kernels, TensorOps products, drafter; `csrc/apple/gemm/q8_0_rows.h`: Q8_0 copies and few-row products |
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
  dispatch); the NVFP4 experts on SIMD-group matrix multiplies over a copy of
  their weights re-ordered on the GPU at load (11.5 GB): 8 x 8 fragments of
  activations times weights in FP32, the weight fragment decoded in registers,
  the inputs of each 128-input block permuted so that a lane loads 8 adjacent
  activations at once. Up to 64 rows one threadgroup runs all 32-row passes of
  an expert; above, every pass has its threadgroup (the expert scan's row
  tiles), so the large experts of a prefill chunk spread over the GPU. Experts
  that received at most 2 rows take the per-expert quantized product. The
  MXFP8 experts run TensorOps tiles grouped by expert (experts with at most 3
  rows on the quantized product). `Inkling2Runtime.expert_mma = 0` keeps every
  expert on TensorOps tiles and skips the copy. Up to 32 rows
  (`expert_overlap_rows`) the quantized-product experts and the others of a
  stage write disjoint rows and run as concurrent dispatches (no barrier
  between them).

**Speculative decoding.** The drafter follows SGLang's
`Inkling2ForConditionalGenerationMTP` (a DFlash block drafter with DSpark's RNN
Markov head): the model's hidden state entering visits 5, 23 and 41 and its
final normalized state are mixed per drafter layer into context K / V; a block
of 8 positions (the last accepted token and 7 mask embeddings) runs 3
bidirectional layers; each drafted position takes the LM head's logits plus
the Markov head's bias (its `hidden_out` folded into the LM head) and the
greedy token feeds the next position. Drafts only decide how many tokens a step
accepts, never which, so for one sequence the drafter runs from Q8_0 copies of
its layers and of the LM head (its base logits), and the Markov head ranks the
256 largest base logits (with the fold bias) instead of the whole vocabulary.

The model verifies the anchor and up to W - 1 drafts (W = 16 for one
sequence, 8 for 2-4, 4 for up to 16, 3 beyond). One sequence is verified by a
resident step: one dispatch over up to 16 rows (40 threadgroups from 4 rows, 36 below; device-scope
barriers), the weights read once for all rows, the experts of the rows
deduplicated (32 distinct experts per visit on average for 8 rows, 15-47) and
taken by
threadgroups from a counter, heavy and light experts alternately, each expert's
down rows as soon as its gate / up rows are done. Two variants share it:
`exact` gives every row the decode step's arithmetic (its products, lane order
and roundings), `fast` (the default) sums in its own order: expert products in
half precision (the codes decoded by shifts and masks into pairs of halves, the
inputs scaled by a power of two per row so the products of BF16 inputs and
E2M1 codes are exact halves, sums of 16 in half, FP32 beyond), coalesced
16-byte code loads with two output rows per lane, attention by blocks of key
positions for up to four rows at once, and beyond 8 rows the BF16 projections,
output projection and LM head as 8x8 matrix multiply-accumulates (the k order
inside a 64-wide chunk permuted so a lane's fragment elements are consecutive
weights and inputs; with Q8_0 the projections up to 12 rows and the LM head
stay on the Q8_0 copies, whose multi-row products run in 128-row units with the
fewest staged chunks). With the fast step a step verifies
2-8 rows: the rows that maximize the expected accepted tokens minus the
measured tokens per millisecond times the step's cost by rows (the drafts'
probabilities mapped to measured agreement rates; the cost table scaled by the
measured times). Several sequences are verified by the row programs.
Verification stages the conv states and sliding-window K / V of its rows; only
the accepted rows are committed, in the command buffer of the next draft. The
agreement rates of the adaptive rows follow the text: per probability bin the
share of reached drafted positions the target accepted, with the measured table
as a prior and older steps decayed.

**Lookup drafts** (one sequence, fast verification). Generated text repeats
itself (code, lists, quoted passages, greedy loops), and the continuation of an
earlier occurrence predicts it. After the drafter's 7 drafts the step can add
the 8 tokens that followed the latest earlier occurrence of the last 4 drafted
tokens in the committed text (a near occurrence repeats with its period) and
verify 12 or 16 rows; their agreement rate is learned per bin of the match
length (how many tokens before them equal the text before the occurrence).
Where the committed text itself matches a long stretch, a step can skip the
drafter: its context still runs, and the step verifies the text's own
continuation as all 15 drafts. It does when its expected gain (agreement learned
per match-length bin) beats the measured gain of the drafter's steps in the
same bin. With DFlash 2 and Q8_0 on corpus slices 1 and 12, the steps after a
committed match of 48 tokens or more accept 15.2-15.5 of 16 rows, 0.145 tokens
per millisecond against 0.09-0.12 for the drafter's steps.

**DFlash 2 drafter** (`dflash2=`, one sequence). The drafter follows SGLang's
`DFlash2DraftModel` with the Inkling integration: each committed position's
four target features go through `fc` and `hidden_norm` into per-layer context
K / V (`k_norm`, RoPE) in the drafter's own cache; a block of the anchor and 7
mask tokens (the target's embedding and `embed_norm`) runs 5 bidirectional
layers (32 query / 8 KV heads over a 2047-position window and the block, two-tap
dynamic grouped convolutions before and after attention and MLP, SwiGLU); the
LM head's 16 largest logits per mask row over the real vocabulary (200058 rows)
feed the candidate selector's greedy path (predecessor and successor codebooks
of rank 256, the score softcapped after the correction). Its products run from
Q8_0 copies (the contexts of prefill chunks from the BF16 weights); the step's
drafts take 6.0 ms.

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
  as TensorOps tiles or SIMD-group matrix multiplies (FP32 accumulation in a
  different order), FlashRT's attention partitions. For a 1024-token prompt the
  last position's logits differ from `mlx-lm`'s one-token evaluation by KL
  1.5e-5 (9.0e-6 with every expert on TensorOps tiles), below the reference
  engine's own difference between its batched and stepwise evaluation (KL
  4.7e-5); the first token agrees. A batched decode step's logits differ
  between the two expert paths by a mean KL of 0.5-1.2e-4, the spread of
  moving the few-row experts between the quantized product and the tiles.
- **Sampling** keeps the tokens `mlx-lm`'s sampler keeps for the same top-k and
  top-p (the nucleus over the full distribution intersected with the top-k) and
  draws by Gumbel-max at the temperature with FlashRT's counter-based noise:
  sampled sequences are reproducible per seed but are not the reference
  engine's sequences.
- **Speculative decoding** commits the tokens the verification selects. With
  `verify="exact"` that is the greedy decode's continuation bit for bit (1024 of
  1024 tokens on three benchmark slices). The fast verification and the row
  programs sum in other orders: they follow the greedy decode until the two
  orders round two leading candidates differently. For 8 rows at four places of
  the corpus (after 37, 1018, 1500 and 3000 tokens) the fast verification's
  logits differ from the exact step's by a mean KL of 1.9e-4 to 9.0e-4 per place
  (largest row 2.6e-3), the row programs' by 1.1e-2 to 2.7e-2 (largest 0.19).
- **Q8_0 (`q8=True`)** re-quantizes the BF16 attention projections (q, k, v,
  relative, gate, output) and the LM head to blocks of 32 int8 codes with an
  FP16 scale at load (in a separate buffer; prefill keeps the BF16 weights).
  Teacher forced on the BF16 greedy continuation, the decode step's logits
  differ by a mean KL of 2.5-8e-4 (largest 9e-3) and pick the same token in 125
  of 128 steps.

## Validation

### Tests

```bash
# anywhere (no GPU): configuration and drafter checks, refusals, routing, chunking, helpers
python -m pytest tests/test_inkling2_apple_contracts.py
# on the Mac, after scripts/apple/build_native.sh:
FLASHRT_INKLING2_DIR=/path/to/Inkling-2-Mini python -m pytest tests/test_inkling2_apple_pipeline.py
# with FLASHRT_INKLING2_DFLASH2_DIR=/path/to/Inkling-2-Mini-DFlash2 also the DFlash 2 drafter
```

The Mac file checks the one-row program against the decode step (every
visit's hidden state, bit for bit), batch invariance of a four-row decode step,
chunked against one-chunk prefill, speculative (exact verification) against
plain greedy decoding and the fast verification up to a near-tie, the exact
resident verification of 8 rows against 8 decode steps (logits bit for bit)
with the fast one (8, 5 and 2 rows) closer to it than the row program, the
same for 16 rows, lookup steps on a repeated text (16-row steps, the fast
verification following greedy up to a near-tie), the
DFlash 2 drafter (exact verification still the greedy output), the Q8_0 decode step
against the BF16 one, sampling (top-k 1 equals greedy, seeds reproduce), the
input checks, the chat path, the re-ordered expert copy against its numpy
reference and the tiled expert passes against the per-expert ones (bit for bit).

### Equivalence with the reference engine (M4 Pro, macOS 26)

- **Decode steps.** 2100 consecutive one-token steps from an empty context
  (Python standard-library text; positions 0-2099, across the 1024-key switch
  of the attention and the sliding-window wrap): the FP32 logits of every step
  are bit-identical to `mlx-lm`'s (0 of 2100 differ), as are the hidden states
  after every one of the 48 layer visits at the 12 steps where they were
  recorded.
- **Prefill.** Last-position logits after the same text's first 1024 tokens as
  one chunk: KL(reference || FlashRT) = 1.5e-5, largest logit difference 0.67
  (every expert on TensorOps tiles: 9.0e-6, 0.68); the reference engine's
  batched and one-token evaluations of the same prompt differ by KL 4.7e-5.
  The first token is the same.

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
| 1 | 26.0 / 26.1 (`stream_generate` 28.4 / 28.5) | **35.3** | 1.24x | **89.2** (87.5-90.9) | 3.14x | 5.4-5.5 of 6.4-6.5 |
| 4 | 53.0 / 52.2 | **64.3** | 1.22x | **86.4** (84.1-88.7) | 1.64x | 5.1 of 8 |
| 16 | 72.5 / 72.6 | **116.3** | 1.60x | **154.3** (148.7-159.9) | 2.13x | 3.5-3.6 of 4 |
| 64 | 120.0 / 120.3 | **210.7** | 1.75x | **212.8** (207.4-218.2) | 1.77x | 2.8 of 3 |

mlx-lm runs these concurrency levels on its BatchGenerator (the engine of
`mlx_lm.server`); at C = 1 its single-sequence `stream_generate` is faster and
is the comparison given in parentheses. FlashRT's plain decoding is the resident
step at C = 1 and the batched row program above. DSpark verifies with the fast
resident step at C = 1 (2-8 rows per step, 6.4-6.5 on average) and the row
programs above. DSpark's ranges are the two measured prompt slices: the accepted
length depends on the text and the continuation, which the verification's
summation order and rows can change (with the exact verification, C = 1 runs
61.6 tokens/s in 89 ms steps of 8 rows). Over twelve slices the spread is wider
(see below). From C = 16
the M4 Pro's matrix units, not its bandwidth, bound a step, so verifying more
rows buys less. The DSpark columns are one measuring session (C = 1 and above);
the row programs of C > 1 run as in the previous release, which reruns within
0.5 % in the same session, while sessions differ by up to 5 %. C = 1 here is
without lookup drafts (as `lookup=False` runs); the next table has both.

With `q8=True` (C = 1): plain decoding 45.6 tokens/s, DSpark 98.2 (84.2-112.2;
the accepted length follows the continuation, which differs from the BF16 one).

### DSpark and DFlash 2 at C = 1

| Drafter | Attention / LM head | Lookup drafts | Slices 1-12 | Slices 1-2: median (range) | Tokens per step | Rows per step |
|---|---|---|---:|---:|---:|---:|
| DSpark | BF16 | off | 93.4 | 98.9 (91.4-106.5) | 5.55 | 6.45 |
| DSpark | BF16 | on | **104.7** | 117.4 (115.6-119.3) | 6.79 | 8.26 |
| DFlash 2 | BF16 | off | 97.1 | 92.2 (80.7-103.6) | 5.91 | 6.44 |
| DFlash 2 | BF16 | on | **107.8** | 108.5 (87.9-129.2) | 7.26 | 8.59 |
| DSpark | Q8_0 | off | 98.3 | 94.5 (83.8-105.2) | 5.20 | 6.06 |
| DSpark | Q8_0 | on | **100.2** | 108.1 (88.8-127.4) | 5.63 | 7.14 |
| DFlash 2 | Q8_0 | off | 103.7 | 98.4 (90.8-106.1) | 5.70 | 6.22 |
| DFlash 2 | Q8_0 | on | **110.5** | 113.1 (90.3-135.9) | 6.65 | 7.81 |

Tokens/s of decoding, one measuring session; slices 1-12 is all generated
tokens over all decode time of twelve slices of the same protocol (1024-token
prompts at 7919-token steps), tokens and rows per step over the same steps.
With lookup drafts 15-30 % of the steps verify 12 or 16 rows and 12-17 % skip
the drafter. Per slice the rate ranges from 76 to 136 tokens/s: the
continuation, and with it the accepted length, changes with the verification's
rows and summation order, so configurations compare over many slices (DSpark
with lookup drafts: Q8_0 100.2 against BF16 104.7 at 5.63 against 6.79 tokens
per step, on different continuations).


### A speculative step (one sequence)

With DFlash 2 the step drafts in 6.0 ms after a step of at most 8 rows (the
drafter context over those rows; 6.45 ms after a 12- or 16-row step): the five
layers' products 2.4 ms, the Q8_0 LM head for 7 positions 2.2 ms, attention over
the 2047-position window 0.5 ms, context, candidates and selector the rest. With
DSpark the step drafts in about 5 ms (the drafter's three layers over the block
from their Q8_0 copies 1.4 ms, the Q8_0 LM head for 7 positions 2.2 ms, the 256
candidates per position 0.35 ms, the Markov walk, the drafter's attention and
context about 1.2 ms). The fast verification takes 35 ms for 2 rows, 44 for 4,
52 for 6, 59 for 8, 75 for 12 and 87 for 16 (Q8_0: 29, 38, 46.6, 54.9, 74 and 86),
the exact step 83-88 ms for 8. In the fast 8-row Q8_0 step (fixed routing of 32
distinct experts per visit, 36 threadgroups) the expert passes take 33.4 ms
(gate / up 19.0 ms at the streaming rate, down 11.6 ms), the Q8_0 projections
7.3 ms, the LM head 2.9 ms, the output projection 2.2 ms, attention 4.0 ms
(scores 1.6, values 1.9, merge 0.5), the message gates 2.6 ms and the per-row
vector phases about 4 ms; the device-scope barriers alone 1.5 ms.

### Prefill (seconds until every sequence has its first token)

| C | mlx-lm, two runs | FlashRT |
|---:|---:|---:|
| 1 | 2.17 / 2.17 | 2.08 |
| 4 | 8.85 / 8.91 | 9.12 |
| 16 | 35.7 / 35.7 | 37.2 |
| 64 | 149.9 / 148.5 | 147.8 |

FlashRT prefills the prompts one after another (chunks of up to 1024 rows);
mlx-lm prefills up to eight prompts together.

### A one-sequence decode step

At 1-2K context a token reads about 5.8 GB: the attention, output-projection and
LM-head weights (3.7 GB), the eight experts of each of the 48 visits (1.8 GB)
and the KV it attends to. At the M4 Pro's measured 254 GB/s streaming rate that
is 22.8 ms; the step takes 28.3 ms (35.3 tokens/s). The projections, output
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
FlashRT token on the M4 Pro takes 28.3 ms.

### Reproduce

```bash
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --make-corpus corpus.i32
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1,4,16,64
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1,4,16,64 --mtp
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1 --mtp --verify exact
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1 --mtp --fixed-rows
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1 --mtp --q8
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1 --q8
# one sequence over twelve slices: DSpark, then DFlash 2 (each also with --q8 and with --no-lookup)
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1 --mtp --reps 12
python examples/apple/inkling2_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1 --mtp --reps 12 \
    --dflash2 /path/to/Inkling-2-Mini-DFlash2
python examples/apple/inkling2_mlx_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --conc 1,4,16,64
python examples/apple/inkling2_mlx_benchmark.py --model /path/to/Inkling-2-Mini --corpus corpus.i32 --stream
```

The twelve-slice rates are the tokens generated in runs 1-12 over their decode
seconds (run 0 is the warm-up; the driver prints every run and the median).

## Limitations

- Text only. Sampling keeps at most 32 candidates (top-k 1..32); speculative
  decoding is greedy.
- The frontend serves one sequence; several sequences go through the runtime.
- The resident decode step keeps 36 threadgroups co-resident (20-core GPU);
  smaller GPUs are not supported.
- The DFlash 2 drafter serves one sequence (`mtp_batches=[1]`).
- Verification widths, the adaptive rows' cost table and the drafts' agreement
  rates are measured on the M4 Pro (the cost table is rescaled by the measured
  times at run time); the accepted length depends on the text (the numbers above
  use source code). The resident verification
  serves one sequence; `verify="exact"` is not available with `q8=True` (it
  repeats the BF16 decode step).

# Qwen3.6-35B-A3B MLX int4 on Apple Silicon (M4 Pro / M4 Max)

FlashRT runs the text decoder of the MLX affine int4 export of Qwen3.6-35B-A3B
(`mlx-community/Qwen3.6-35B-A3B-4bit`, the files `mlx-lm` loads; "the reference
engine" below) natively on Apple Silicon through the Apple Metal plan
(`cpp/apple`): the decode step is one resident dispatch that runs up to eight
tokens, and prefill runs in chunks recorded once as indirect command buffers.
Weights are copied verbatim from the safetensors files into one Metal buffer;
nothing is re-quantized. The runtime needs numpy and the plan library only (no
PyTorch, no MLX).

The arithmetic contract is the reference engine's: affine int4 (group 64)
projections, experts, embedding and LM head, affine int8 router and
shared-expert gate, BF16 norms, activations, residual stream and KV cache, FP32
Gated DeltaNet state. A one-token step reproduces `mlx-lm`'s logits bit for bit
(see *Numerics*).

This entry is text-only: the vision tower of the checkpoint is not loaded.

## Requirements

| | |
|---|---|
| Checkpoint | `mlx-community/Qwen3.6-35B-A3B-4bit` directory (`config.json`, `model-*.safetensors`, `tokenizer.json`, `chat_template.jinja`); validated revision `38740b847e4cb78f352aba30aa41c76e08e6eb46` |
| Hardware | Apple M4 Pro / M4 Max (Metal GPU family Apple9), 20+ GPU cores, 48 GB unified memory or more (18.2 GiB weights + state) |
| Software | macOS 26.2 or newer (validated on 26.7.3), Xcode command line tools to build the plan library |
| Python | 3.10+, numpy; `tokenizers` + `jinja2` for chat and token fixtures |

```bash
bash scripts/apple/build_native.sh     # .cache/apple-native/libflashrt_apple_plan.dylib
```

Kernels are compiled from embedded Metal source when the runtime is created
(a few seconds); `FLASHRT_APPLE_PLAN_LIBRARY` overrides the library path. A
checkpoint with another geometry or quantization layout is refused before
anything is compiled.

## Usage

```python
from flash_rt.frontends.torch.qwen36_moe_mps import Qwen36MoeTorchFrontendMps

fe = Qwen36MoeTorchFrontendMps("/path/to/Qwen3.6-35B-A3B-4bit", max_seq=32768)
text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

# Token-level streaming, context reuse across turns, greedy or top-k/top-p sampling.
ids = fe.tokenizer.encode(fe.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
for block in fe.stream(ids, max_new_tokens=256, reuse_prefix=True, temperature=1.0, top_p=0.95, top_k=20, seed=7):
    ...
```

`stream` yields blocks of up to eight tokens and stops after an
end-of-generation token. `temperature=0` is greedy (on-device argmax, lowest id
on ties); otherwise top-k (1..32) / top-p sampling with Gumbel noise drawn on
the GPU from `seed` and the token position. Arguments outside these ranges
raise `ValueError`.

`reuse_prefix=True` prefills only the new tokens when the prompt extends
everything the model state already holds (the previous prompt and its generated
tokens: the next turn of a chat). The Gated DeltaNet layers keep a recurrent
state that exists only at the last position, so any other prompt (an edited
history, a regenerate) is recomputed from the start. Reused positions keep the
state the decode step gave them, so a reused continuation can differ in
rounding from a full prefill of the same tokens.

`max_seq` sizes the KV cache of the ten attention layers (2 KB per position
and layer); the DeltaNet state is 63 MB regardless of length.

## Components

| Layer | Path |
|---|---|
| Checkpoint reader (numpy, zero-copy safetensors) | `flash_rt/models/qwen36_moe/mlx_checkpoint.py` |
| Programs (decode, chunked prefill, LM head, noise) | `flash_rt/models/qwen36_moe/pipeline_mps.py` |
| Runtime (weights, buffers, state, parameter blocks) | `flash_rt/frontends/torch/_qwen36_moe_mps_runtime.py` |
| Frontend (tokenizer, streaming, context reuse) | `flash_rt/frontends/torch/qwen36_moe_mps.py` |
| Geometry-specialized kernels | `csrc/apple/qwen35moe/`: resident decode step, prefill rows (DeltaNet, Q/K, router, experts) |
| Model-independent kernels | `csrc/apple/gemm/w4a16_affine_*`, `csrc/apple/attention/gqa8_d256_causal_gated_prefill_mma_bf16.h`, `csrc/apple/kernels/gumbel_noise_f32.h` |
| Benchmark drivers | `examples/apple/qwen36_moe_benchmark.py`, `examples/apple/qwen36_moe_mlx_benchmark.py` (reference engine, optional) |

The model: 40 layers, hidden 2048. Every fourth layer is softmax attention (16
query / 2 KV heads of 256, RoPE on the first 64 dimensions, a sigmoid output
gate); the others are Gated DeltaNet layers (a 4-tap depthwise conv over the
q/k/v projections, 16 key and 32 value heads of 128, a recurrent FP32 state of
32 x 128 x 128 per layer updated by the delta rule, a gated RMSNorm). Every
layer has a routed MoE: 256 experts, the top 8 by softmax probability with
renormalized weights, plus a shared expert scaled by a sigmoid gate.

**Decode.** One dispatch of 36 co-resident threadgroups (256 threads each) runs
up to eight tokens: embedding, all layers, LM head and token selection, with
device-scope barriers between phases. A token reads 1.65 GB of weights and
reads and writes the 63 MB of DeltaNet state. Within a layer the
matrix-vector products are spread over all 288 SIMD groups; each of the 32
value heads' delta rule runs in its own threadgroup while the remaining z
projection rows are claimed by ticket; the nine expert products (eight routed,
one shared) take four threadgroups each.

**Prefill.** A prompt is processed in chunks of up to 2048 positions (the
reference engine's step). Per layer: RMSNorm rows, the projections as affine
int4 GEMMs, then either the DeltaNet rows (conv + gates, the delta-rule
recurrence over the chunk's rows in order, gated norm) or Q/K norm + RoPE with
K/V publication and causal attention on SIMD-group MMAs; then the output
projection, router rows, the routed experts as GEMMs grouped by expert, the
shared expert and the combination. Chunks below 513 rows use a second program
that adds the split-K and row-vector product paths MLX takes for few rows.

## Numerics

- **One-token steps are bit-identical to `mlx-lm`.** Every operation of the
  decode step follows MLX 0.32 for a single position: the quantized
  matrix-vector products (per-lane FP32 accumulation over 512-input blocks, one
  reduction per row), RMSNorm (leaf sums of 128, `rsqrt`, two BF16 roundings),
  the depthwise conv, SiLU and SwiGLU, the DeltaNet gates and delta rule, RoPE,
  the vector attention (32 key-strided partitions below 1024 keys, two passes
  with BF16 partial outputs from 1024), the softmax router with MLX's top-8
  order (ascending by probability and index), BF16 sums in order, and BF16
  rounding after every BF16 operation. The exponential is the precise one where
  MLX runs a built-in kernel and the fast one inside the functions `mlx-lm`
  compiles at run time.
- **A block of eight steps equals eight single steps**, tokens and logits.
- **Prefill** uses the same row arithmetic for norms, conv, gates, the delta
  rule, the router and the expert combination, MLX's quantized-matmul
  accumulation for the GEMMs, and FlashRT's own attention kernel (FP32
  accumulation, one BF16 rounding). `mlx-lm` itself evaluates a prompt with
  different kernels when it is fed in chunks or token by token, and its two
  results differ by reordering noise; the chunked prefill differs from either
  by the same amount (see *Equivalence*).
- **Sampling** applies the reference rule (top-k of the nucleus computed over
  the full distribution, then Gumbel-max with the temperature) with FlashRT's
  counter-based noise, so sampled sequences are reproducible per seed but are
  not the reference engine's sequences.

## Validation

### Tests

```bash
# anywhere (no GPU): configuration, geometry refusal, chunking, parameter layout, selection rule
python -m pytest tests/test_qwen36_moe_apple_contracts.py
# on the Mac, after scripts/apple/build_native.sh:
FLASHRT_QWEN36_MOE_MLX_DIR=/path/to/Qwen3.6-35B-A3B-4bit python -m pytest tests/test_qwen36_moe_apple_pipeline.py
```

The end-to-end file checks: 1040 one-token steps against `mlx-lm` (logits
compared bit for bit at 32 of them, below and above the 1024-key switch; needs
`mlx-lm` installed, skipped otherwise), block decode against single steps,
determinism of the small-, single- and multi-chunk prefill programs, a prefill
continued at a chunk boundary, chunked against stepwise prefill, state restore,
seeded sampling from the first token on, a stop token inside a block and
context reuse.

### Equivalence with the reference engine (M4 Pro, macOS 26)

- **Decode steps.** 1300 consecutive one-token steps from an empty context
  (Python standard-library text): the BF16 logits of every step are
  bit-identical to `mlx-lm`'s (0 of 1300 differ), as is the residual stream
  after the embedding and after every layer at the 12 steps where it was
  recorded.
- **Prefill.** Last-position logits after a prompt of N tokens, for 22 values
  of N from 16 to 5000, against `mlx-lm`'s batched prefill. The reference
  engine's own two evaluation orders (batched chunks, and one token at a time,
  which FlashRT's stepwise path reproduces exactly) give the noise floor:

| Prompt | KL(batched ‖ FlashRT chunked) | KL(batched ‖ one-token steps) | First token |
|---|---|---|---|
| 24 | 0.107 | 0.172 | same |
| 40 | 0.074 | 0.0037 | same |
| 256 | 0.0040 | 0.054 | same |
| 600 | 5.0e-6 | 2.7e-6 | same |
| 1024 | 1.2e-3 | 6.6e-5 | same |
| 2048 | 1.2e-4 | 6.5e-5 | same |
| 3000 | 1.9e-3 | 4.7e-3 | same |
| 5000 | 1.8e-6 | 1.2e-5 | same |

  Over the 22 lengths the two columns have the same order of magnitude and
  neither is consistently smaller; the first token equals the batched
  reference's at 21 of them (at 512 and 513 the reference's own two orders
  disagree with each other as well). The 4-bit model is sensitive to rounding
  (an expert choice can flip), which is why single positions show KL up to 0.1
  between two evaluations of the same engine.

## M4 Pro numbers

Mac mini M4 Pro (14-core CPU, 20-core GPU, 64 GB), macOS 26.7.3, batch 1.
Reference engine: `mlx-lm` 0.31.3 on MLX 0.32.0, same checkpoint, same token
fixtures (Python standard-library sources tokenized with the checkpoint's
tokenizer). Both engines run in the foreground; the 1K, 4K and 16K rows are an
interleaved session (reference, FlashRT, reference, two-minute pauses), medians
of two timed runs after a warm-up, 1024 generated tokens.

### Decode (tokens/s, prefill excluded)

| Context | Greedy, mlx-lm | Greedy, FlashRT | Default sampling, mlx-lm | Default sampling, FlashRT |
|---|---|---|---|---|
| 1K | 86.6-86.9 | 110.0 (+27%) | 82.6-82.8 | 109.6 (+32%) |
| 4K | 83.6-83.8 | 106.3 (+27%) | 79.5-79.6 | 106.0 (+33%) |
| 16K | 72.2 | 91.8 (+27%) | 69.2-69.3 | 91.6 (+32%) |
| 8K * | 79.1 | 100.3 (+27%) | 75.7 | |
| 32K * | 61.5 | 76.3 (+24%) | 59.6 | |
| 64K * | 48.4 | 58.1 (+20%) | 47.4 | |

Default sampling is temperature 1, top-p 0.95, top-k 20 (the checkpoint's
`generation_config.json`); FlashRT selects the token on the GPU inside the
resident step. Rows marked * are single passes of each engine in separate
sessions with 512 generated tokens.

### Prefill (seconds to the first token)

| Prompt | mlx-lm | FlashRT | Speedup |
|---|---|---|---|
| 1,024 | 1.13 | 1.00 | 1.13x |
| 4,096 | 4.44 | 3.80 | 1.17x |
| 16,384 | 21.5-21.7 | 18.6 | 1.16x |
| 8,192 * | 10.2 | 8.32 | 1.23x |
| 32,768 * | 53.2 | 49.4 | 1.08x |
| 65,536 * | 136.7 | 140.4 | 0.97x |

At 64K the quadratic attention term dominates and the two engines are level.

### Where a decode step's time goes (1K context, 9.05 ms per token)

A token reads 1.65 GB of weights and reads and writes 63 MB of state; at the
254 GB/s this GPU streams, that is about 7.0 ms. Measured by skipping one phase
at a time: DeltaNet input projections 1.9 ms, expert gate/up 1.7 ms, LM head
1.1 ms, output projections 0.9 ms, expert down 0.85 ms, DeltaNet state update
0.8 ms (bandwidth: the state is read and written once per token), attention
projections 0.5 ms, attention 0.3 ms, router 0.2 ms, norms, barriers and the
expert combination 0.5 ms. The expert products and the LM head run at the
streaming rate; the remainder is spread over barriers, short stages and the
attention of short contexts.

### Reproduce

```bash
PY=python3   # an interpreter with numpy, tokenizers; mlx + mlx-lm for the reference driver
M=/path/to/Qwen3.6-35B-A3B-4bit
PYTHONPATH=. $PY examples/apple/qwen36_moe_benchmark.py --mlx-dir $M --make-tokens tokens
PYTHONPATH=. $PY examples/apple/qwen36_moe_benchmark.py --mlx-dir $M --tokens tokens/prompt-1024.i32 tokens/prompt-4096.i32
$PY examples/apple/qwen36_moe_mlx_benchmark.py --mlx-dir $M --tokens tokens/prompt-1024.i32 tokens/prompt-4096.i32
```

Run benchmarks in the foreground (a shell job started with `&` runs at a lower
priority on macOS and loses several percent) and leave pauses between heavy
runs: both engines slow down as the machine warms up.

## Limitations

- Text only (no vision tower), batch 1.
- Geometry and quantization layout are fixed in the kernels: the 4-bit MLX
  export of Qwen3.6-35B-A3B. Other bit widths or mixed-precision exports are
  refused.
- Context reuse needs a prompt that extends the held context; there is no
  recurrent-state checkpoint for edited histories.
- The multi-token-prediction head of the original checkpoint is not part of
  this MLX export and is not used.

# Qwen3.6-35B-A3B MLX int4 on Apple Silicon (M4 Pro / M4 Max)

FlashRT runs the text decoder of the MLX affine int4 export of Qwen3.6-35B-A3B
(`mlx-community/Qwen3.6-35B-A3B-4bit`) natively on Apple Silicon through the
Apple Metal plan (`cpp/apple`): the decode step is one resident dispatch that
runs up to eight tokens, and prefill runs in chunks recorded once as indirect
command buffers. The checkpoint is the one `mlx-lm` loads; `mlx-lm` is "the
reference engine" below. Weights are copied verbatim from the safetensors files
into one Metal buffer and nothing is re-quantized (the only derived data is a
BF16 copy of each layer's router, 40 MB in total, for the prefill's router
product). The runtime needs numpy and the plan library only (no PyTorch, no
MLX).

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
end-of-generation token (the decode step stops by itself at the first four stop
ids; further ones are checked per block). `temperature=0`, the default, is
greedy (on-device argmax, lowest id on ties). With a temperature the step
samples on the GPU: the `top_k` (1..32) largest logits, cut to the nucleus
`top_p` using full-vocabulary probabilities, then Gumbel noise drawn from
`seed` and the token position. The checkpoint's `generation_config.json`
settings (temperature 1, top-p 0.95, top-k 20) are in `fe.sampling_defaults`
and are not applied unless passed. Arguments outside these ranges raise
`ValueError`.

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
| Geometry-specialized kernels | `csrc/apple/qwen35moe/`: resident decode step, prefill rows (DeltaNet, Q/K, experts), prefill router |
| Model-independent kernels | `csrc/apple/gemm/w4a16_affine_*` (incl. the `*_direct*` entries), `csrc/apple/attention/gqa8_d256_causal_gated_prefill_mma_bf16.h`, `csrc/apple/kernels/gumbel_noise_f32.h` |
| Kernels shared with the Flint prefill | `flint_rms_rows_bf16` (RMSNorm rows, hidden 2048), `flint_expert_scan` (expert block offsets) in `csrc/apple/flint/` |
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
matrix-vector products are spread over all 288 SIMD groups; the delta rule and
gated norm of each of the 32 value heads run in a threadgroup of their own; the
nine expert products (eight routed, one shared) take four threadgroups each.

**Prefill.** A prompt is processed in chunks of up to 2048 positions (the
reference engine's step). Per layer: RMSNorm rows, the projections as affine
int4 GEMMs, then either the DeltaNet rows (conv + gates, the delta-rule
recurrence over the chunk's rows in order, gated norm) or Q/K norm + RoPE with
K/V publication and causal attention on SIMD-group MMAs; then the output
projection, the router (a TensorOps product over the BF16 router copy, softmax,
top 8), the routed experts as GEMMs grouped by expert, the shared expert and
the combination. A last chunk below 513 rows uses a second program whose
products run as one accumulation chain for any row count (the `*_direct*`
entries), with MLX's row-vector kernels below its vector bound (10-18 rows).

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
- **Prefill is equivalent, not bit-identical.** It uses the decode step's row
  arithmetic for norms, conv, gates, the delta rule, the top-8 selection and
  the expert combination; the products are GEMMs over weight tiles dequantized
  to BF16 as MLX's quantized matmul does, with FP32 accumulation in one chain
  (MLX splits the chain for few rows or narrow outputs; FlashRT does not); the
  attention is FlashRT's own kernel (FP32 accumulation, one BF16 rounding).
  `mlx-lm` itself evaluates a prompt with different kernels when it is fed in
  chunks or token by token, and its two results differ by reordering noise; the
  chunked prefill differs from either by the same amount (see *Equivalence*).
- **Sampling** keeps the tokens `mlx-lm`'s sampler keeps (the nucleus over the
  full distribution intersected with the top-k) and draws by Gumbel-max at the
  temperature, with FlashRT's counter-based noise: sampled sequences are
  reproducible per seed but are not the reference engine's sequences.

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
`mlx-lm` installed and about 40 GB for the two engines, skipped otherwise),
block decode against single steps,
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
- **Prefill.** Last-position logits after a prompt of N tokens, for 21 values
  of N from 16 to 5000, against `mlx-lm`'s batched prefill. The reference
  engine's own two evaluation orders (batched chunks, and one token at a time,
  which FlashRT's stepwise path reproduces exactly) give the noise floor:

| Prompt | KL(batched ‖ FlashRT chunked) | KL(batched ‖ one-token steps) | First token |
|---|---|---|---|
| 24 | 0.071 | 0.172 | same |
| 40 | 0.0036 | 0.0037 | same |
| 256 | 0.129 | 0.054 | same |
| 600 | 1.5e-6 | 2.7e-6 | same |
| 1024 | 2.8e-3 | 6.6e-5 | same |
| 2048 | 1.3e-5 | 6.5e-5 | same |
| 3000 | 1.6e-3 | 4.7e-3 | same |
| 5000 | 1.0e-5 | 1.2e-5 | same |

  Over the 21 lengths the chunked prefill is the closer of the two to the
  batched reference at 11 and the farther at 10, and its first token equals
  the batched reference's at 20 (at 513 it differs; at 512 the reference's own
  one-token evaluation differs from its batched one). The 4-bit model is
  sensitive to rounding (an expert choice can flip), which is why single
  positions show KL up to 0.1 between two evaluations by the same engine.

## M4 Pro numbers

Mac mini M4 Pro (14-core CPU, 20-core GPU, 64 GB), macOS 26.7.3, batch 1.
Reference engine: `mlx-lm` 0.31.3 on MLX 0.32.0, same checkpoint, same token
fixtures (Python standard-library sources tokenized with the checkpoint's
tokenizer). Both engines run in the foreground, interleaved per prompt
(reference, FlashRT, reference, with two-minute pauses); the two reference
values bracket the thermal drift. Decode: 1024 generated tokens, no stop token,
tokens/s = (tokens - 1) / time after the first token, median of two timed runs
after a warm-up (one timed run of the reference from 32K). Prefill: wall time
to the first token, median of two runs after a warm-up (one from 32K).

### Decode (tokens/s, prefill excluded)

| Context | Greedy, mlx-lm | Greedy, FlashRT | Default sampling, mlx-lm | Default sampling, FlashRT |
|---|---|---|---|---|
| 1K | 87.0-87.1 | 110.3 (+27%) | 82.6 | 109.6 (+33%) |
| 2K | 85.8-85.9 | 109.0 (+27%) | 81.5-81.9 | 108.5 (+33%) |
| 4K | 83.6-83.8 | 106.5 (+27%) | 79.5-79.6 | 106.0 (+33%) |
| 8K | 78.8 | 100.8 (+28%) | 75.2 | 99.9 (+33%) |
| 16K | 72.2 | 92.2 (+28%) | 69.2-69.3 | 91.8 (+32-33%) |
| 32K | 62.5-62.6 | 78.7 (+26%) | 59.1-60.2 | 78.3 (+30-32%) |
| 64K | 49.4-49.6 | 60.0 (+21%) | 48.1-48.3 | 59.8 (+24%) |

"Default sampling" passes the checkpoint's `generation_config.json` settings
(temperature 1, top-p 0.95, top-k 20) to both engines; FlashRT selects the
token on the GPU inside the resident step.

### Prefill (seconds to the first token)

| Prompt | mlx-lm | FlashRT | Speedup |
|---|---|---|---|
| 1,024 | 1.13-1.16 | 0.95 | 1.20-1.23x |
| 2,048 | 2.20 | 1.75 | 1.26x |
| 4,096 | 4.43-4.48 | 3.62 | 1.22-1.24x |
| 8,192 | 9.39-9.41 | 7.70 | 1.22x |
| 16,384 | 21.7-22.0 | 17.6 | 1.23-1.25x |
| 32,768 | 50.4-50.7 | 44.7 | 1.13x |
| 65,536 | 132.4-133.0 | 126.5 | 1.05x |

The attention of the ten softmax layers is quadratic in the prompt length and
takes over from about 32K, where the two engines' attention kernels are closer
than their products.

Short prompts (single runs, not interleaved): 40 tokens 0.14 s (mlx-lm
0.11 s), 64 tokens 0.16 s (0.15 s), 128 tokens 0.23 s (0.25 s), 256 tokens
0.35 s (0.43 s), 512 tokens 0.52 s (0.77 s). Below about 64 tokens the
reference's row-vector path is faster: FlashRT's smallest GEMM tile is 64 rows.

### Where a decode step's time goes (1K context, 9.0 ms per token)

A token reads 1.65 GB of weights and reads and writes 63 MB of state; at the
254 GB/s this GPU streams, that is about 7.0 ms. Measured by skipping one phase
at a time: DeltaNet input projections 1.7 ms, expert gate/up 1.7 ms, LM head
1.1 ms, output projections 0.9 ms, expert down 0.8 ms, DeltaNet state update
0.6 ms (bandwidth: the state is read and written once per token), attention
projections 0.5 ms, expert combination 0.4 ms, attention 0.2 ms, router 0.2 ms,
norms and barriers 0.4 ms. The weight products together run at the streaming
rate; what remains is the short phases between them. With the context the
attention grows (2.0 ms at 16K, 7.9 ms at 64K): it is MLX's exact vector
schedule at every length, whose keys must be visited in strided order.

### Reproduce

```bash
PY=python3   # an interpreter with numpy, tokenizers; mlx + mlx-lm for the reference driver
M=/path/to/Qwen3.6-35B-A3B-4bit
PYTHONPATH=. $PY examples/apple/qwen36_moe_benchmark.py --mlx-dir $M --make-tokens tokens
PYTHONPATH=. $PY examples/apple/qwen36_moe_benchmark.py --mlx-dir $M --tokens tokens/prompt-1024.i32 tokens/prompt-4096.i32
$PY examples/apple/qwen36_moe_mlx_benchmark.py --mlx-dir $M --tokens tokens/prompt-1024.i32 tokens/prompt-4096.i32
```

`--make-tokens` writes `prompt-<n>.i32` fixtures for 1K to 64K from the
interpreter's standard-library sources (`--corpus` selects other text files);
both drivers print one JSON line per run and a median per prompt. Run
benchmarks in the foreground (a shell job started with `&` runs at a lower
priority on macOS and loses several percent) and leave pauses between heavy
runs: both engines slow down as the machine warms up.

## Limitations

- Text only (no vision tower), batch 1, M4 Pro / M4 Max class GPUs (the
  resident step needs its 36 threadgroups running at the same time; it reports
  a barrier timeout otherwise).
- Geometry and quantization layout are fixed in the kernels: the 4-bit MLX
  export of Qwen3.6-35B-A3B. Other bit widths or mixed-precision exports are
  refused.
- Context reuse needs a prompt that extends the held context; there is no
  recurrent-state checkpoint for edited histories.
- The multi-token-prediction head of the original checkpoint is not part of
  this MLX export and is not used.

# Flint MLX int4 on Apple Silicon (M4 Pro / M4 Max)

FlashRT runs Flint's MLX affine int4 checkpoint (`RecursiveSI/flint-int4`, the
files that Recursive's MLX engine [rec](https://github.com/recursive-oss/rec)
loads; "the reference engine" below) natively on Apple Silicon through the
Apple Metal plan (`cpp/apple`): the decode step is one resident dispatch that
runs up to eight tokens, and prefill runs in chunks recorded once as indirect
command buffers. Weights are copied verbatim from the safetensors files into one
Metal buffer; nothing is re-quantized. The runtime needs numpy and the plan
library only (no PyTorch, no MLX).

The arithmetic contract is the reference engine's: affine int4 (group 64)
projections, experts and LM head, int8 embedding, BF16 router, norms, sinks,
activations and KV cache, with every operation reproducing the reference
engine's rounding and reduction order (see *Numerics*).

## Requirements

| | |
|---|---|
| Checkpoint | `RecursiveSI/flint-int4` directory (`config.json`, `model*.safetensors`, `tokenizer.json`, chat template); validated revision `3b93f9053b530488405ae9fca51d4900d6ce21f0` |
| Hardware | Apple M4 Pro / M4 Max (Metal GPU family Apple9; MLX architecture `applegpu_g16s`), 20+ GPU cores, 48 GB unified memory or more (21 GiB weights + KV) |
| Software | macOS 26.2 or newer (Metal Shading Language 4.0 BF16 TensorOps; validated on 26.7.3), Xcode command line tools to build the plan library (validated with 26.6) |
| Python | 3.10+, numpy; `tokenizers` + `jinja2` for chat and token fixtures; `fastapi` + `uvicorn` only for the example server |

```bash
bash scripts/apple/build_native.sh     # .cache/apple-native/libflashrt_apple_plan.dylib
```

Kernels are compiled from embedded Metal source when the runtime is created
(a few seconds); `FLASHRT_APPLE_PLAN_LIBRARY` overrides the library path.

## Usage

```python
from flash_rt.frontends.torch.flint_mps import FlintTorchFrontendMps

fe = FlintTorchFrontendMps("/path/to/flint-int4", max_seq=32768)
text = fe.chat([{"role": "user", "content": "Write a quicksort in Python."}], max_new_tokens=512)

# Token-level streaming, prefix reuse across turns, greedy or top-k/top-p sampling.
ids = fe.tokenizer.encode(fe.tokenizer.apply_chat_template(messages, add_generation_prompt=True))
for block in fe.stream(ids, max_new_tokens=256, reuse_prefix=True, temperature=1.0, top_p=0.95, top_k=20, seed=7):
    ...
```

`stream` yields blocks of up to eight tokens and stops after an
end-of-generation token. `temperature=0` is greedy (on-device argmax, lowest id
on ties); otherwise top-k (1..32, default 20) / top-p sampling with Gumbel noise
drawn on the GPU from `seed` and the token position. Arguments outside these
ranges raise `ValueError` (the server answers 400).

`reuse_prefix=True` keeps the KV cache of the longest common prefix (at least
64 tokens) with the previous sequence, prompt and generated tokens, and
prefills only the rest as a continued chunk. When the sliding-window history
the reuse position needs has been overwritten (a regenerate after decoding past
it, for example) the prompt is recomputed from the start. Reused positions keep
their KV as computed then (by prefill, or by the decode step for generated
tokens), so a reused continuation can differ in rounding from a full prefill of
the same tokens; the reference engine's prompt cache instead re-prefills
generated tokens in the background, so the two caches are not bit-identical
either.

Lookup decoding (greedy only): build the frontend with `lookup_rows=3` and pass
`lookup=True` to `stream` / `generate`. Draft tokens come from earlier
occurrences of the sequence's last 6..4 tokens in the prompt and the generated
text; one verification step computes the last token and up to `lookup_rows - 1`
drafts as consecutive positions and keeps the drafts that equal greedy's
choice. The tokens are exactly plain greedy decoding's (see *Lookup decoding*
under *M4 Pro numbers*).

### OpenAI-compatible server

```bash
pip install fastapi uvicorn
PYTHONPATH=. python examples/flint_openai_server.py --mlx-dir /path/to/flint-int4 --port 8000 --max-seq 131072
```

The same server as the RTX path (`/v1/chat/completions` stream and
non-stream, `/v1/models`, `/health`, `ignore_eos`, `cache_prompt`, `timings`);
with `--mlx-dir` it also accepts `temperature`, `top_p`, `top_k` and `seed`.

## Components

| Layer | Path |
|---|---|
| Checkpoint reader (numpy, zero-copy safetensors) | `flash_rt/models/flint/mlx_checkpoint.py` |
| Programs (decode, chunked prefill, LM head, noise) | `flash_rt/models/flint/pipeline_mps.py` |
| Runtime (weights, buffers, KV, parameter blocks) | `flash_rt/frontends/torch/_flint_mps_runtime.py` |
| Frontend (tokenizer, streaming, prefix reuse) | `flash_rt/frontends/torch/flint_mps.py` |
| Lookup decoding (n-gram drafts, verification policy) | `flash_rt/models/flint/lookup.py` |
| Flint kernels (geometry-specialized) | `csrc/apple/flint/`: resident decode step, prefill rows, router and expert placement |
| Model-independent kernels | `csrc/apple/gemm/w4a16_affine_*`, `csrc/apple/attention/gqa8_causal_prefill_mma_bf16.h` (and the TensorOps variant `gqa8_causal_prefill_bf16.h`), `csrc/apple/kernels/gumbel_noise_f32.h` |
| Benchmark drivers | `examples/apple/flint_benchmark.py`, `examples/apple/flint_rec_benchmark.py` (reference engine, optional) |

Decode: one resident dispatch of 36 co-resident threadgroups runs embedding, 49
layers, LM head and token selection for up to eight consecutive tokens between
device-scope barriers (`flint_decode_resident_bf16`). Three programs: `greedy`
(production), `sampled` (fused top-k/top-p) and `reference` (the reference
engine's attention schedule, used for bit-exact qualification; it also writes
the residual stream after every layer, `FlintRuntime.trace`).

Prefill: chunks of up to 2048 positions (the reference engine's chunk size on
M4 Pro/Max). Up to four consecutive full chunks share one program of up to 8192
rows: its row-wise operations (norms, projections, routing, expert products,
whose values do not depend on the row count above 512 rows) run once over all
rows, while attention is dispatched per chunk with that chunk's position, rows
and sliding window, so the result equals separate chunks bit for bit and the
larger expert blocks make the expert products faster. Per layer: RMSNorm rows,
Q/K/V GEMMs, Q/K norm + RoPE with KV publication, GQA-8 attention with sinks
(sliding layers through a linear scratch of the 513 preceding positions),
o-proj with residual, router top-6, expert counting/placement, expert-grouped
gate/up (SwiGLU) and down GEMMs, the shared expert and the weighted
combination. After the last chunk, the final norm and LM head of the last
position give the first token's logits. Attention runs on SIMD-group matrix
multiplies (`gqa8_causal_prefill_mma_bf16`, one SIMD group per position's eight
query heads); `FlintRuntime(..., prefill_attention="tensor_ops")` selects the
TensorOps kernel with the same arithmetic. Each program is recorded once; a
chunk's start, row count and ring layout are read from a state buffer.

## Numerics

* Decode, `reference` program: bit-identical logits and residual stream to the
  reference engine's resident kernel at every step.
* Prefill: bit-identical to the reference engine's prefill (MLX forward with
  the LM head on the last row): logits of the last prompt position, all 49
  layers' residual stream and all KV. Each operation reproduces MLX's
  arithmetic for the chunk's row count: RMSNorm reduction order, attention in
  log2 units with the running maximum advanced every 16 keys, the router's
  split-K accumulation, quantized matmul split-K partitions and vector paths
  (`qmv_fast`, `qmv_wide`) for small chunks, expert products on the vector path
  below 1024 assignments, and the sliding-ring layout that the decode step's slot
  rule expects after a prefill.
* Exception: chunks of at most 8 rows (prompts of at most 8 tokens, or a last
  chunk of 1..8 positions) are computed but not bit-identical, because MLX
  switches to its vector attention path there.
* Decode, `greedy` / `sampled` programs: identical to `reference` up to 1024
  keys; global layers with more keys use an MMA attention schedule whose
  reordering stays within the reference engine's own reordering band (see
  *Validation*).
* Verification step (lookup decoding): every row's logits are bit-identical to
  the `greedy` step at its position. Sliding-layer K/V of draft rows stay in a
  side buffer until accepted, so rejected drafts leave the cache as `greedy`
  would. The step rounds to BF16 in integer arithmetic (equal to the hardware
  conversion for every FP32 arithmetic result, which this GPU flushes to zero
  below the normal range) because the compiler packs the hardware conversion's
  16-bit results of different rows into one register and corrupts one.

## Validation

### Tests

```bash
PYTHONPATH=. python -m pytest -q tests/test_flint_contracts.py tests/test_flint_server.py tests/test_apple_native_binding.py
# On the Mac, after scripts/apple/build_native.sh:
.cache/apple-native/flashrt_apple_kernel_registry_smoke                                    # compiles every Metal entry
PYTHONPATH=. python -m pytest -q tests/test_apple_w4a16_affine.py                          # kernels vs NumPy
FLASHRT_FLINT_MLX_DIR="<path-to-flint-int4>" PYTHONPATH=. python -m pytest -q tests/test_flint_apple_pipeline.py
```

The CPU tests cover the routing entry, the framework-free runtime import, the
config and checkpoint refusals (including the geometry the Apple kernels
hard-code), the MLX affine reader (low nibble first, `w = s*q + b`), MLX's
vector-path row bounds, the first-token selection rule, sampling argument
checks, sliding-window bookkeeping for prefix reuse and the server's request
validation. The Metal tests compare the w4a16 affine products with a NumPy
reference and exercise the multi-program plan API. The checkpoint tests check
that the production decode program equals the `reference` program below 1024
keys, that prefill is deterministic for small, single and multi-chunk prompts,
that chunked prefill agrees with stepping the prompt through the decode step,
that a prefill continued at a chunk boundary equals the full prefill, prefix
reuse and its recompute rule, seeded sampling and stop tokens.

### Equivalence with the reference engine (M4 Pro, macOS 26)

Method: the reference engine and FlashRT read the same checkpoint and the same
token ids, taken from the standard-library corpus of *Reproduce*.

* Prefill logits of the last prompt position were compared bit for bit for 35
  prompt lengths from 12 to 4100 tokens, including two- and three-chunk prompts.
  34 are identical in all 150,144 logits. The exception is 4100 tokens, whose last
  chunk has 4 rows (*Numerics*); its KL is 2e-4.
* Prefill was also isolated per layer by feeding the reference residual stream
  into each FlashRT layer. All 49 layers match for every row. The full 49-layer
  residual stream and all KV caches match the reference forward bit for bit.
* End to end, 128 greedy tokens after prefill (FlashRT prefill + `reference`
  decode) are identical to the reference engine for prompts of 100, 600, 1300 and
  2600 tokens.
* Production decode was teacher-forced: 256 steps along the reference greedy
  sequence, one token per dispatch, KL(reference || FlashRT) per step. The
  noise band comes from the reference engine's own attention with different
  partition counts.

| Context | `reference` KL / top-1 | `greedy` KL / top-1 | Reference engine's own reorderings, KL / top-1 |
|---|---|---|---|
| 1K | <= 8e-7 / 100% | 0.137 / 93.8% | 0.108 / 94.5%, 0.062 / 93.4% |
| 16K | <= 6e-7 / 100% | 0.012 / 98.8% | 0.0055 / 98.4%, 0.0177 / 98.4% |
| 64K | <= 8e-7 / 100% | 0.0012 / 99.2% | 0.0009 / 98.8%, 0 / 100% |

The `reference` residue is the comparison itself (FP32 reference log-softmax
against an FP64 one of the same logits). Decode is chaotic at short context,
so a single reordering moves the 1K KL by about 0.1.

## M4 Pro numbers

Setup: Mac mini, M4 Pro (20-core GPU), 64 GB, macOS 26.7.3. Reference engine:
rec v0.1.1 (release build, MLX 0.32.0, its bundled Python 3.12.12). Both
engines read the same checkpoint files (revision above) and the same token ids
(see *Reproduce*), batch 1, both in the foreground at normal priority (as a
background job at reduced priority, zsh's default for `&` jobs, FlashRT's 4K
decode was about 7% slower and the reference engine's 2-3%). Reference engine
runs were interleaved with FlashRT's: reference, FlashRT, reference for the
prompts up to 8K (the two reference values bracket a thermal drift of up to
about 10% between passes); the same order at 16K with a two-minute pause before
each run; three rounds at 32K; one run of each engine, reference first, at 64K
and 128K.

### Prefill

Wall time from the token upload to the first token's logits (FlashRT also
includes the first token's selection), one warmup, median of three runs; prompts
of 16K tokens and longer: one run.

| Prompt | Reference engine | FlashRT | Speedup |
|---|---|---|---|
| 1,024 | 1.11-1.28 s | 0.96 s | 1.16-1.33x |
| 2,048 | 2.44-2.67 s | 1.82 s | 1.34-1.46x |
| 4,096 | 5.39-5.79 s | 3.87 s | 1.39-1.50x |
| 8,192 | 11.94-12.27 s | 8.18 s | 1.46-1.50x |
| 16,384 | 22.96-22.98 s | 16.45 s | 1.40x |
| 32,768 | 58.4-66.1 s | 48.3-50.6 s | 1.19-1.36x |
| 65,536 | 178.9 s | 145.9 s | 1.23x |
| 131,072 | 603.1 s | 473.1 s | 1.27x |

Single runs at long prompts vary by up to about 12% within a session (32K
prefill, three rounds: reference engine 58.4, 65.5 and 66.1 s, FlashRT 48.3 to
50.6 s), so the 32K speedups (prefill and decode) are given per adjacent pair.
The machine's thermal state moves both engines, FlashRT's decode more: at 16K,
run right after the shorter prompts, the reference engine prefilled in 26.8 s
and decoded at 91.5 tok/s, FlashRT in 20.4 s and at 96.2 tok/s; after
two-minute pauses (the table), 23.0 and 16.4 s, 99.4 and 111.4 tok/s. Absolute
values also move between sessions, so compare the engines within one session.

### Decode

The reference engine's published method: 1024 output tokens, prefill excluded,
tokens/s = (tokens - 1) / time after the first token, no stop token, median of
two runs. "Default" is temperature 1, top-p 0.95, top-k 20. The reference
engine reports its own `decode_tokens_per_second`; FlashRT is timed from the
first token to the last.

| Context | Greedy, reference engine | Greedy, FlashRT | Default, reference engine | Default, FlashRT |
|---|---|---|---|---|
| 1K | 131.2-133.5 | 137.0 (+3-4%) | 121.2-124.4 | 132.8 (+7-10%) |
| 2K | 125.9-130.4 | 131.7 (+1-5%) | 118.5-121.6 | 131.5 (+8-11%) |
| 4K | 119.4-124.2 | 127.9 (+3-7%) | 113.6-116.0 | 127.3 (+10-12%) |
| 8K | 108.1-111.4 | 120.9 (+9-12%) | 104.6-105.6 | 120.2 (+14-15%) |
| 16K | 99.4-99.5 | 111.4 (+12%) | 94.7 | 111.0 (+17%) |
| 32K | 69.1-75.9 | 86.9-89.5 (+18-26%) | 66.5-73.1 | 87.0-88.9 (+22-31%) |
| 64K | 50.1 | 63.5 (+27%) | 49.3 | 64.0 (+30%) |
| 128K | 29.7 | 42.0 (+41%) | 29.5 | 41.9 (+42%) |

### Lookup decoding

Greedy decoding with verification steps (`--lookup 3` in the FlashRT driver:
up to two drafts per step), same session and prompts as plain greedy, 1024
output tokens; the generated tokens' SHA256 equals plain greedy's at every
prompt.

| Context | Greedy, FlashRT | Lookup, FlashRT | Drafts accepted (1K) |
|---|---|---|---|
| 1K | 138.3 | 158.2 (+14%) | 510 of 607 |
| 4K | 131.5 | 154.3 (+17%) | |
| 8K | 124.2 | 141.5 (+14%) | |
| 16K | 111.5 | 122.2 (+10%) | |
| 32K | 91.6 | 102.5 (+12%) | |

The gain depends on how often the continuation repeats earlier text: these
prompts are Python source, whose greedy continuations repeat a lot (50-95% of
drafts accepted). Without matching n-grams the policy runs plain resident steps
(1, 2, 4, then 8 per dispatch), so non-repeating text decodes at plain greedy's
speed. A step costs more than one greedy step because every row reads its own
six routed experts and attends over its own keys (1K: 1.6 greedy steps for two
rows, 2.2 for three, 3.0 for four); the policy grows the draft only after a
fully accepted step.

### Reproduce

Build the plan library (*Requirements*) and create the token fixtures once:
the checkpoint tokenizer's ids of the Python standard-library sources (`*.py`,
then `*/*.py`, each sorted by path, every file tokenized on its own, ids
concatenated), cut to 1K..128K tokens. With Python 3.12.12 the 131,072-token
fixture has SHA256
`9c1b9307707854928486cb524339407c4d40d7f9f9f070f1fc6769eeeab3182c`.

```bash
PYTHONPATH=. python3.12 examples/apple/flint_benchmark.py --mlx-dir "<path-to-flint-int4>" \
    --make-tokens build/flint-tokens
```

FlashRT, then the reference engine through the optional reference driver,
which must run under rec's own interpreter (`rec_mlx` and `mlx` importable;
`~/.local/share/rec/runtimes/*/python/bin/python3.12` for a default install).
rec's model directory (`~/.local/share/rec/models/*/`, with `flint-int4.json`)
holds the same checkpoint files, so it can serve as `--mlx-dir` too. Each
command prints every repetition as a JSON line and a median per prompt and
sampling mode.

```bash
P="build/flint-tokens/prompt"
PYTHONPATH=. python examples/apple/flint_benchmark.py --mlx-dir "<path-to-flint-int4>" \
    --tokens $P-1024.i32 $P-2048.i32 $P-4096.i32 $P-8192.i32
"<rec-python>" examples/apple/flint_rec_benchmark.py --model-dir "<rec-model-dir>" \
    --tokens $P-1024.i32 $P-2048.i32 $P-4096.i32 $P-8192.i32
# 16K and longer: one prefill per prompt.
PYTHONPATH=. python examples/apple/flint_benchmark.py --mlx-dir "<path-to-flint-int4>" \
    --prefill-reps 0 --tokens $P-16384.i32 $P-32768.i32 $P-65536.i32 $P-131072.i32
# Lookup decoding next to plain greedy (decode only).
PYTHONPATH=. python examples/apple/flint_benchmark.py --mlx-dir "<path-to-flint-int4>" \
    --phase decode --sampling greedy --lookup 3 --tokens $P-1024.i32 $P-4096.i32 $P-8192.i32
"<rec-python>" examples/apple/flint_rec_benchmark.py --model-dir "<rec-model-dir>" \
    --prefill-reps 0 --tokens $P-16384.i32 $P-32768.i32 $P-65536.i32 $P-131072.i32
```

Run the two engines in separate processes, in the foreground, and alternate
them (reference, FlashRT, reference) so thermal drift shows up as the spread of
the reference values. The reference driver replaces the stop tokens with the reserved id
150143, discards each prompt's first decode run, which pays the prefill, and
sets `MLX_METAL_FAST_SYNCH=1`, the default of rec's own launcher
(`--fast-sync`). The FlashRT driver restores the prompt's sliding-window state
before every decode run, so each run continues the same prompt.

## Limitations

* Batch size 1; one cached sequence for prefix reuse.
* Lookup decoding: greedy only (sampling would need draft acceptance by
  probability, not equality); one verification step per dispatch.
* Checkpoint: the kernels hard-code Flint's geometry; any other geometry is
  rejected with `FlintConfigError` before compilation.
* Device: the decode step needs its 36 threadgroups of 256 threads resident at
  once (M4 Pro / M4 Max). On a GPU that cannot hold them, the first decode
  raises `RuntimeError` (resident barrier timeout) after the spin limit.
* Device profile: the dispatch rules reproduced for prefill are those of the
  Apple GPU 's' size class (M4 Pro, M4 Max, M5); other classes use other MLX
  thresholds, so prefill there runs but is not bit-identical to the reference.
* Chunks of at most 8 rows are not bit-identical to the reference prefill.
* macOS only: the runtime raises `RuntimeError` on other platforms.

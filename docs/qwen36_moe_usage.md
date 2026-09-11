# Qwen3.6-35B-A3B text inference

FlashRT runs the language backbone from the official
`Qwen/Qwen3.6-35B-A3B` BF16 checkpoint on an RTX 5090 (SM120) and on Jetson AGX
Thor (SM110). The checkpoint uses the same `qwen3_5_moe` text architecture as
Nex-N2-mini, so both models share the same weight loader, prefill, attention,
MoE, recurrent-state, and CUDA Graph decode implementation, and both
architectures run the same frontend -- what differs is which kernel tiers the
build has.

This entry is text-only: it does not load the vision tower, and image or video
input is not part of this interface.

It does execute the checkpoint's MTP draft head, but only when asked. Pass
`load_mtp=True` to the constructor and call `generate_spec()`; see *Speculative
decode* below. `generate()` never reads the head, and without `load_mtp=True`
the loader does not read it either.

## Requirements

| | |
|---|---|
| Checkpoint | `Qwen/Qwen3.6-35B-A3B` BF16 safetensors |
| Hardware | RTX 5090 / SM120, Jetson AGX Thor / SM110 |
| GPU memory | 32 GB (SM120); unified memory on Thor |
| Framework | PyTorch |
| Runtime quantization | NVFP4 |

Configure and build the gated `qwen3_5_moe` kernels. **The two targets take
different flags** -- `FLASHRT_ENABLE_QWEN35MOE` turns on the block-scaled 4-bit
MMA tier as well, which sm_110 refuses at configure time, so Thor names the two
tiers it can compile:

RTX 5090 (SM120):

```bash
cmake -S . -B build \
  -DGPU_ARCH=120 \
  -DFLASHRT_ENABLE_QWEN35MOE=ON
cmake --build build -j
pip install -e ".[torch]"
```

Jetson AGX Thor (SM110):

```bash
cmake -S . -B build \
  -DGPU_ARCH=110 \
  -DFLASHRT_ENABLE_QWEN35MOE_CORE=ON \
  -DFLASHRT_ENABLE_QWEN35MOE_W4A16=ON \
  -DFLASHRT_ENABLE_THOR_FA2=ON
cmake --build build -j 2
pip install -e ".[torch]"
```

`FLASHRT_ENABLE_THOR_FA2` is what builds the vendored FA2 kernels on sm_110;
see *Attention differs by target* below for what it buys and why it is opt-in.
Without it the model still runs, and produces the same tokens, but a long
prefill is far slower.

### Kernel tiers

`FLASHRT_ENABLE_QWEN35MOE=ON` is a convenience switch for all three tiers
below. Targets that cannot run a tier can select the remainder explicitly.

| Flag | Kernels | Requires |
|---|---|---|
| `FLASHRT_ENABLE_QWEN35MOE_CORE` | QKV layout/split, bf16 matvec, router top-k, SiLU/sigmoid fusion, GDN recurrence, weighted-sum reducer, bf16 GEMM | SM80 and newer |
| `FLASHRT_ENABLE_QWEN35MOE_W4A16` | weight-only 4-bit matvec, grouped matvec, GEMM | SM80 and newer; hardware operand conversion from SM89 |
| `FLASHRT_ENABLE_QWEN35MOE_W4A4` | block-scaled 4-bit MMA: grouped GEMV, M16/M64/block-tile MMA | sm_120a / sm_121a |

The upper tiers depend on the core tier, so enabling either turns it on. SM120
runs all three; sm_110 runs the first two, and the block-scaled tier refuses to
configure there rather than building kernels that fail at run time.

**These three tiers are not the whole dependency set.** Walking every `fvk`
call the pipeline makes and resolving each to the preprocessor guard active
where it is defined gives 32 kernels across seven gates:

| gate | kernels |
|---|---:|
| `FLASHRT_HAVE_QWEN36_KERNELS` | 12 |
| `FLASHRT_HAVE_QWEN35MOE_CORE` | 10 |
| `FLASHRT_HAVE_QWEN35MOE_W4A16` | 3 |
| `ENABLE_CUTLASS_SM120_NVFP4_W4A16` | 2 |
| `FLASHRT_HAVE_QWEN35MOE_W4A4` | 2 |
| `FLASHRT_HAVE_NVFP4_SWIZZLE` | 1 |
| ungated | 1 |

The twelve under `FLASHRT_HAVE_QWEN36_KERNELS` are the linear-attention path:
causal convolution and its update, the gated-DeltaNet recurrence, the WY chunk
stack, the fused RMSNorm-gated-SiLU, partial RoPE and argmax. They are shared
with the rest of the Qwen3.6 family and are gated on `NOT FLASHRT_SLIM_BUILD`,
not on architecture — so `-DFLASHRT_SLIM_BUILD=ON` removes them and the
frontend then refuses to start, naming what is missing. Do not use a slim build
for this model.

Selecting tiers by reading the source's own grouping is therefore not enough to
know what a target needs; the call sites are what decide.

Two further kernels are **optional** and are not part of that required set,
because the frontend resolves each through `getattr` and falls back to the
kernel it replaces when a build does not carry it:

| symbol | gate | replaces | why |
|---|---|---|---|
| `gated_deltanet_recurrent_edge_qwen36_bf16` | `FLASHRT_HAVE_QWEN36_KERNELS` | `gated_deltanet_recurrent_qwen36_bf16` | same arithmetic without the local-memory round trip for the state column |
| `moe_router_topk_warp_sm120_bf16` | `FLASHRT_HAVE_QWEN35MOE_CORE` | `moe_router_topk_sm120_bf16` | same selection in one warp instead of `k` rounds of block-wide barriers |
| `moe_shared_gate_combine_edge_bf16` | `FLASHRT_HAVE_QWEN35MOE_CORE` | a five-launch tensor chain | one kernel, same arithmetic in the same order, rounded once at the store |
| `moe_grouped_gemm_nvfp4_sm100_bf16out` | `FLASHRT_HAVE_QWEN35MOE_GROUPED_SM100` (sm_110 + `_W4A16`) | the per-expert GEMM loop, or the grouped GEMV | every routed expert of a layer in one launch, with the per-group shapes read from device memory |

All produce output identical to the path they stand in for, so the fallback is
a performance difference and never a numerical one. The edge recurrence is
shape-specialized to a head dim of 128 and raises for anything else rather than
leaving the output buffer undefined.

Which of each pair actually runs is a `KernelPolicy` field, not the answer to
"is the symbol there" — see *Runtime controls*.

### Attention differs by target

The ten full-attention layers do not use the same kernel everywhere:

| target | attention | how it is built |
|---|---|---|
| SM120 / SM89 / SM87 | vendored FA2 | automatic: the SM80-family source, which `__CUDA_ARCH__ >= 800` admits |
| Thor SM110 | vendored FA2, or the decomposed reference | opt-in: `-DFLASHRT_ENABLE_THOR_FA2=ON` |
| Thor SM110 | FA4 | its SM100-class CuTe-DSL kernel needs Blackwell tensor memory; ships as the `thor-fa4` pip extra, not compiled into `flash_rt_kernels` |

FA2 was originally excluded from sm_110 on the grounds that Thor has its own
attention path and FA2 would add about 10 MB of `.so` for nothing. That holds
for the models that use the decomposed path, and it does not hold for a long
prefill of this one: the decomposed path materialises an `(S * heads, S_kv)`
score buffer -- 3.4 GB per layer at ten thousand tokens -- and this model needs
one instantiation (bf16, head_dim 256), not the twelve the size estimate
assumed. Enabling it takes the chunking penalty of a chunked prefill from 64%
to 4%.

It stays opt-in because every other Thor model still uses its own attention
path and would only be paying the compile time and the binary size. A build
without it runs this model correctly and emits the same tokens; only a long
prefill is slower. Treat a missing FA2 as a signal to fall back, not as a build
error.

At the *decode* shape the two are the same answer to bf16 precision -- measured
against an fp32 reference, 2.0e-3 relative for both at kv=64, 2.2e-3 against
2.1e-3 at kv=2048 -- so decode on sm_110 keeps the reference path the golden
fixture was recorded through, and takes FA2 only for prefill.
`FLASHRT_NEXN2_DECODE_FA2=1` overrides that.

The attention backend probes its kernel at construction: it runs one case
through the same launch the hot path uses and compares against
`scaled_dot_product_attention`, falling back if they disagree. That is not
belt-and-braces. Three times in this work a kernel compiled, linked and loaded
while being unable to run — the block-scaled 4-bit tier substitutes an invalid
control path off its own architecture, the lm_head kernel is simply absent
outside GPU_ARCH 120/121, and the vendored FA2 on an SM110 part printed a
complaint and returned without writing its output. That last one still produced
15 of 16 reference tokens, because ten of forty layers contributing nothing is
survivable for a residual stream — which is precisely why a symbol check is not
a capability check.

`_W4A4` refuses to configure on a target without block-scaled MMA. CUTLASS
still compiles those translation units elsewhere, but substitutes
`CUTE_INVALID_CONTROL_PATH` for the MMA, so the build would succeed and then
fail at run time. The explicit gate turns that into a configure-time error.

## Usage

```python
from flash_rt.frontends.torch.qwen36_moe import Qwen36MoeTextFrontend

frontend = Qwen36MoeTextFrontend(
    "/models/Qwen3.6-35B-A3B",
    device="cuda:0",
    max_seq=4096,
    quant_scope="experts",
)
frontend.set_prompt("Explain why deterministic reductions matter.")
token_ids = frontend.generate(max_new_tokens=64)
print(frontend.tokenizer.decode(token_ids))
```

`set_prompt()` accepts already-rendered text. For chat requests, render the
checkpoint's own template first:

```python
messages = [{"role": "user", "content": "Write a CUDA reduction checklist."}]
prompt = frontend.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
frontend.set_prompt(prompt)
```

The direct frontend is intentional. `flash_rt.load_model()` wraps VLA models
with a `predict(images, ...)` API and therefore redirects
`config="qwen36_moe"` to the class above.

The frontend defaults to `kernelized=True`. Setting `kernelized=False` is
rejected because it would select the parent Transformers reference path, which
loads the complete multimodal BF16 model rather than this text-only NVFP4
runtime.

## Checkpoint contract

Before allocating GPU memory, the frontend checks:

- the exact 40-layer `qwen3_5_moe` text geometry, including MoE widths,
  convolution width, gated attention, normalization epsilon, and RoPE
  parameters;
- the 30 linear-attention / 10 full-attention schedule;
- the exact shapes of all 693 text-backbone tensors consumed by the shared
  pipeline;
- the exact shapes of all 19 official MTP tensors;
- every safetensors shard referenced by the index.

Extra vision tensors are allowed and ignored by this text-only entry. The
validation can be run without loading model weights:

```bash
PYTHONPATH=. python - <<'PY'
from flash_rt.frontends.torch.qwen36_moe import (
    validate_qwen36_moe_checkpoint,
)
print(validate_qwen36_moe_checkpoint("/models/Qwen3.6-35B-A3B"))
PY
```

## Runtime controls

The shared architecture uses:

- `FLASHRT_QWEN35MOE_PREFILL_CHUNK` — chunked-prefill block size, default
  `8192`; `0` disables chunking.
- `FLASHRT_QWEN35MOE_GRAPH_CACHE_MAX` — decode CUDA Graph LRU capacity,
  default `256`.
- `FLASHRT_QWEN35MOE_SPEC_GRAPH_CACHE_MAX` — speculative-window CUDA Graph LRU
  capacity, default `16`. Bounded separately and far lower than the decode
  cache: a speculative graph covers `k+1` positions through the whole stack, so
  its memory pool is several times a decode step's. The constructor argument
  `spec_graph_cache_max` sets it per frontend.

The older `FLASHRT_NEXN2_PREFILL_CHUNK` and
`FLASHRT_NEXN2_GRAPH_CACHE_MAX` names remain compatible aliases.

Which of several interchangeable kernels each step calls is a
`KernelPolicy` (`flash_rt.frontends.torch._nexn2_rtx_forward`), not a symbol
lookup: every field selects between implementations checked against each other
with `torch.equal`, so a field decides speed and never output. The environment
variables above and `NEXN2_WY_GDN`, `NEXN2_ROUTE_KERNEL`,
`NEXN2_DENSE_CUBLASLT`, `FLASHRT_QWEN35MOE_W4A16_EDGE` and
`FLASHRT_QWEN35MOE_VERIFY_K_ROWS` are its defaults. A policy must not be
changed between a CUDA graph capture and its replay.

## Validation

### Build and symbol matrix

A build with every `qwen3_5_moe` option off must compile the same sources and
export the same symbols it did before the tiers existed. That is a property of
the gates, so it is checked by reading them:

```bash
python scripts/qwen35moe_build_matrix.py           # print sources + symbols per tier
python scripts/qwen35moe_build_matrix.py --check   # exit 1 if any tier leaks
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q -p no:cacheprovider tests/test_qwen35moe_build_matrix.py
```

The five configurations behind it, each configure-only:

| configuration | flags | result |
|---|---|---|
| baseline SM120 | `-DGPU_ARCH=120` | `FA2 ENABLED`; no `qwen3_5_moe` source or symbol |
| baseline SM110 | `-DGPU_ARCH=110` | `FA2 DISABLED`; no `qwen3_5_moe` source or symbol |
| SM110 supported | `-DGPU_ARCH=110 -DFLASHRT_ENABLE_QWEN35MOE_CORE=ON -DFLASHRT_ENABLE_QWEN35MOE_W4A16=ON -DFLASHRT_ENABLE_THOR_FA2=ON` | core + weight-only tiers, grouped MoE GEMM, FA2 at `hdim={256} x dtype={bf16}` |
| SM120 supported | `-DGPU_ARCH=120 -DFLASHRT_ENABLE_QWEN35MOE=ON` | all three tiers |
| SM110 block-scaled | `-DGPU_ARCH=110 -DFLASHRT_ENABLE_QWEN35MOE_W4A4=ON` | **configure fails**, naming the two tiers that do apply |

The last row is the point of the explicit gate: CUTLASS would otherwise compile
those translation units on sm_110 with the MMA replaced by an invalid control
path, and the failure would arrive at run time instead.

### Tests

The repository smoke test is checkpoint-independent:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q -p no:cacheprovider tests/test_qwen36_moe_smoke.py
```

Speculative decode has its own file. The constructor contract and the graph
cache's eviction policy run anywhere; the equivalence tests -- K=1 and K=2
against plain greedy, the window's logits and recurrent, conv and KV state
against the decode steps they stand in for, the rejected-tail rewind, and the
boundary token counts -- need a GPU and a checkpoint and skip without them:

```bash
FLASHRT_QWEN36_MOE_CKPT_DIR=/models/Qwen3.6-35B-A3B \
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q -p no:cacheprovider tests/test_qwen36_moe_spec_decode.py
```

Set `FLASHRT_QWEN36_MOE_CKPT_DIR` to include the official checkpoint contract
test. Performance and precision numbers must be measured on Qwen3.6 weights;
Nex-N2-mini measurements are not interchangeable even though the compute
pipeline is shared.

The optional GPU suite checks the shared weighted-sum reducer against the
former Torch reduction, repeats it eagerly and through CUDA Graph replay, then
loads the checkpoint, checks finite logits, and compares cold and warm CUDA
Graph generation with an official Transformers BF16 greedy-token fixture:

```bash
FLASHRT_QWEN36_MOE_CKPT_DIR=/models/Qwen3.6-35B-A3B \
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q -p no:cacheprovider tests/test_qwen36_moe_gpu.py
```

First-light data on an RTX 5090, PyTorch 2.9.1, CUDA 12.8 runtime, and the
official BF16 checkpoint:

| Measurement | Result |
|---|---:|
| Runtime weight load | 47.96 s |
| Resident allocated memory after load | 21.44 GiB |
| Peak allocated memory during load | 22.94 GiB |
| First 21-token prefill, including warmup | 230.95 ms |
| Subsequent 20–45-token prefill | 28.99–35.12 ms |
| 64-token prompt, 32-token eager decode | 48.14 tok/s |
| 64-token prompt, 32-token warm CUDA Graph decode | 195.49 tok/s |

These are the original first-light figures, measured with a 32-token
generation. The current SM120 reference for this path is the same-shape
measurement at `P=64, N=64`, warm CUDA Graph:

| Measurement | Result |
|---|---:|
| Prefill | 40.42 ms |
| Decode, median of 8 runs | 257.95 tok/s |
| Decode, range across 8 runs | 256.90–258.22 tok/s |
| Repeated sequences identical | 8 / 8 |

Quote that one, not the 32-token row above: the two use different generation
lengths and are not interchangeable.

The decode work described under *Speculative decode* was tuned and measured on
Thor, and its one tuning constant is scoped to that architecture, so the SM120
kernels are byte-identical to the ones these numbers were taken on.

The eager, first-capture, and warm-graph runs produced the same 32 token IDs.
These numbers are a first-light correctness run, not a context-length sweep.

Reproduce with:

```bash
PYTHONPATH=. python benchmarks/qwen36_moe_edge_decode.py \
    --checkpoint /path/to/Qwen3.6-35B-A3B \
    --prompt-tokens 64 --max-new-tokens 32
```

The benchmark refuses to report throughput if the eager and captured paths
disagree on any token, because a rate for a path that emits different text is
not a rate for the same work.

The table above predates the decode work described under *Speculative decode*
below; the correctness gate (`tests/test_qwen36_moe_gpu.py`) passes on the
current tree, but the SM120 latency figures have not been re-measured since.

Four chat prompts from 12 to 45 tokens were also compared with the official
Transformers BF16 implementation:

| Precision check | Result |
|---|---:|
| Last-token logit cosine, minimum | 0.95635 |
| Last-token logit cosine, mean | 0.96455 |
| First-token argmax matches | 4 / 4 |
| Greedy generation matches | 64 / 64 tokens |

The logit cosine is lower than the Nex-N2-mini measurement, but the tested
greedy sequences were token-exact for 16 generated tokens on all four prompts.

## Jetson AGX Thor numbers

Measured on Jetson AGX Thor (sm_110), unified memory, the same BF16 checkpoint
with runtime NVFP4 conversion, against vLLM 0.26.0 on the same part with the
same pre-tokenized prompts and the same protocol on both sides: TTFT is the
wall time of a one-token generate, decode is the rest of a 64-token generate
with that TTFT subtracted, best of three after a warm-up. One prompt length per
process on both sides. The vision tower is off on both, since this frontend is
text-only.

| prompt | TTFT | vLLM TTFT | |
|---:|---:|---:|---|
| 20 | **89.5 ms** | 102.3 ms | +14% |
| 256 | **104.7 ms** | 214.5 ms | +105% |
| 512 | **144.5 ms** | 251.1 ms | +74% |
| 1024 | **216.0 ms** | 319.4 ms | +48% |
| 2048 | **379.6 ms** | 495.0 ms | +30% |
| 4096 | **748.6 ms** | 867.4 ms | +16% |
| 10240 | **1890.7 ms** | 2144.5 ms | +13% |
| 32768 | **7207.5 ms** | 7231.8 ms | +0.3% |

Where the prefill is dominated by per-layer work -- projections, routing, the
linear-attention scan -- the kernels win, and win by more the shorter the
prompt. Where it is dominated by attention's O(S^2) the lead narrows, because
attention is the one component still reached through torch.

Decode. Each row carries the tree it was taken on, because the decode round
below moved the step 15.3% and the vLLM sweep predates it:

| measurement | tree | decode | vLLM decode | |
|---|---|---:|---:|---|
| @1024, from the sweep above | before the decode round | 87.1 tok/s | 31.6 tok/s | 2.8x |
| @2048 | before the decode round | 86.3 tok/s | 31.5 tok/s | 2.7x |
| @4096 | before the decode round | 85.1 tok/s | 31.2 tok/s | 2.7x |
| captured steady step @20 | before the decode round | 89.0 tok/s | | |
| captured steady step @20 | **current** | **102.6 tok/s** | | |

The two protocols agree on the same tree -- the sweep reads 87.1 at 1024, the
captured step 89.0 at 20, a 2% spread -- so the distance from 87 to 102.6 is
the decode round, not the way it was timed. vLLM's side is a different binary
and nothing here changes it, which makes 2.7-2.8x a lower bound on the current
ratio; the current tree has not been swept at 1024-4096, so no figure is quoted
for it.

Context reaches 128 K on this board, at 2470 tok/s of prefill. It could not
before FA2 was available here: the default configuration chunks past 8192
tokens and every chunk asked for a non-square causal window, which had no fused
backend, so the scores were materialised.

### The decode round

Five changes, each bit-identical to what it replaced, measured in the captured
steady step at a 20-token prompt.

| | step | tok/s |
|---|---:|---:|
| round start | 11.238 ms | 89.0 |
| gating constants derived once, not per step | 11.059 ms | 90.4 |
| MoE tail fused into one kernel | 10.827 ms | 92.4 |
| spill-free gated-DeltaNet recurrence | 10.379 ms | 96.4 |
| single-warp router top-8 | 10.297 ms | 97.1 |
| grouped GEMV `kUnroll` 4 -> 2 | **9.743 ms** | **102.6** |

The last row is a build-time constant and is bit-identical by construction: the
main loop advances by `32*kUnroll` and the tail takes the remainder, so a lane
visits the same k-blocks in the same order for any value. The sweep is not
monotone -- 1: 99.7, **2: 102.6**, 3: 96.5, 4: 96.3 -- so two is a genuine
optimum, and it is scoped to sm_110 in CMake because it was measured on a
20-SM part.

## Speculative decode

The MTP head ships with the checkpoint and is loaded on request. It is a
DeepSeek-V3-style single module: it reads the pre-final-norm hidden state of the
previous position and the token emitted at this one, and predicts the next.
Drafts are chained, so acceptance decays with each additional draft.

The window is verified through the decode kernels at `K+1` rows, over the
weights the decode step caches, so a verified row is the decode step it stands
in for -- bit for bit, not approximately. That is what allows the emitted text
to be plain greedy's, and it is checked directly: logits rows, per-token
recurrent and conv snapshots, and the KV rows written are all compared with
`torch.equal` against a decode step run over the same tokens.

Plain and speculative are measured in the same process, so every ratio is
paired with a baseline from its own run. The absolute rates move a few percent
with what else the board is doing; the ratio is the stable part. Every row
emitted plain greedy's sequence token for token, with the 16-token golden
fixture passing on the same build.

| tree | plain | K=2 | |
|---|---:|---:|---:|
| after the fused MoE tail | 91.18 | 97.34 | 1.07x |
| after the single-warp router | 91.98 | 98.57 | 1.07x |
| after `kUnroll` 4 -> 2 | 96.64 | 100.66 | 1.04x |
| current | **100.35** | **106.74** | **1.06x** |

K=1 on the current tree reads 105.22 against the same 100.35.

`K=2` is the operating point. Above it the window costs more than the extra
accepted tokens return: each additional verified row re-reads the routed
experts, which do not amortise across a window the way the dense weights do,
and each additional draft pays a full-vocabulary projection.

Acceptance rises with context -- 2.60 tokens kept per window at a 20-token
prompt, 2.72 at 512, 2.74 at 2048 -- while the ratio does not, because the
verify runs `K+1` separate single-query attention passes. That is the price of
keeping the window bit-exact, and it is the largest remaining lever on this
path.

For scale, vLLM 0.26.0 supports `Qwen3_5MoeMTP` -- the same head -- and with
`num_speculative_tokens=2` reaches 55.00 tok/s at a 20-token prompt against its
own 31.59, a 1.74x gain. The larger gain rests on a step three times heavier: a
fixed per-draft cost is proportionally three times cheaper against 31.7 ms than
against 9.7 ms. Plain greedy decoding here is faster than that speculative
figure by 1.8x, with no speculation at all.

Enable it with `load_mtp=True` on the constructor; the window width is the `k`
argument to `generate_spec`:

```python
frontend = Qwen36MoeTextFrontend(
    "/models/Qwen3.6-35B-A3B",
    device="cuda:0",
    max_seq=2048,
    load_mtp=True,
    spec_graph_cache_max=16,
)
frontend.set_prompt("Explain why deterministic reductions matter.")
token_ids = frontend.generate_spec(max_new_tokens=128, k=2)
print(frontend.tokenizer.decode(token_ids))
```

`FLASHRT_QWEN35MOE_VERIFY_K_ROWS=0` falls back to verifying through the prefill
forward, which is slower and produces the same tokens.

## Limitations

- Text only; the vision tower is not loaded.
- The kernelized runtime NVFP4 path is required.
- Greedy decode only. Speculative decode is greedy as well: it emits the
  sequence plain greedy decoding would emit, token for token, or it is a bug.
- Only the BF16 source checkpoint with runtime NVFP4 conversion is supported.
- Sampling, batching, and beam search are not implemented.

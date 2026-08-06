# Qwen3.6-35B-A3B text inference

FlashRT runs the language backbone from the official
`Qwen/Qwen3.6-35B-A3B` BF16 checkpoint on an RTX 5090. The checkpoint uses the
same `qwen3_5_moe` text architecture as Nex-N2-mini, so both models share the
same weight loader, prefill, attention, MoE, recurrent-state, and CUDA Graph
decode implementation.

This entry is text-only. It does not load the vision tower and it validates but
does not execute the checkpoint's MTP head. Image/video input and speculative
decode are not part of this interface.

## Requirements

| | |
|---|---|
| Checkpoint | `Qwen/Qwen3.6-35B-A3B` BF16 safetensors |
| Hardware | RTX 5090 / SM120, Jetson AGX Thor / SM110 |
| GPU memory | 32 GB (SM120); unified memory on Thor |
| Framework | PyTorch |
| Runtime quantization | NVFP4 |
| Build flags | `-DGPU_ARCH=120 -DFLASHRT_ENABLE_QWEN35MOE=ON` (SM120), `-DGPU_ARCH=110 ...` (Thor) |

Configure and build the gated `qwen3_5_moe` kernels:

```bash
cmake -S . -B build \
  -DGPU_ARCH=120 \
  -DFLASHRT_ENABLE_QWEN35MOE=ON
cmake --build build -j
pip install -e ".[torch]"
```

### Kernel tiers

`FLASHRT_ENABLE_QWEN35MOE=ON` is a convenience switch for all three tiers
below. Targets that cannot run a tier can select the remainder explicitly.

| Flag | Kernels | Requires |
|---|---|---|
| `FLASHRT_ENABLE_QWEN35MOE_CORE` | QKV layout/split, bf16 matvec, router top-k, SiLU/sigmoid fusion, GDN recurrence, weighted-sum reducer, bf16 GEMM | SM80 and newer |
| `FLASHRT_ENABLE_QWEN35MOE_W4A16` | weight-only 4-bit matvec, grouped matvec, GEMM | SM80 and newer; hardware operand conversion from SM89 |
| `FLASHRT_ENABLE_QWEN35MOE_W4A4` | block-scaled 4-bit MMA: grouped GEMV, M16/M64/block-tile MMA | sm_120a / sm_121a |

The upper tiers depend on the core tier, so enabling either turns it on. The
SM120 text runtime documented here needs all three.

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

Both produce output identical to the kernel they stand in for, so the fallback
is a performance difference and never a numerical one. The edge recurrence is
shape-specialized to a head dim of 128 and raises for anything else rather than
leaving the output buffer undefined.

### Attention differs by target, by design

The ten full-attention layers do not use the same kernel everywhere, and the
arch lists reflect that rather than overlooking it:

| target | attention | why |
|---|---|---|
| SM120 / SM89 / SM87 | vendored FA2 | the SM80-family source, which `__CUDA_ARCH__ >= 800` admits |
| Thor SM110 | FA4 | its SM100-class CuTe-DSL kernel needs Blackwell tensor memory; ships as the `thor-fa4` pip extra, not compiled into `flash_rt_kernels` |

So FA2 is deliberately absent from the Thor build, and FA4 cannot serve
Ampere-class SM87. Treat a missing FA2 as a signal to fall back, not as a
build error.

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
from flash_rt.frontends.torch.qwen36_moe_rtx import (
    Qwen36MoeTextFrontendRtx,
)

frontend = Qwen36MoeTextFrontendRtx(
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
from flash_rt.frontends.torch.qwen36_moe_rtx import (
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

The older `FLASHRT_NEXN2_PREFILL_CHUNK` and
`FLASHRT_NEXN2_GRAPH_CACHE_MAX` names remain compatible aliases.

## Validation

The repository smoke test is checkpoint-independent:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q -p no:cacheprovider tests/test_qwen36_moe_smoke.py
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
| Resident allocated memory after load | 21.44 GiB |
| Peak allocated memory during load | 22.94 GiB |
| Subsequent 64-token prefill | 34.58–35.73 ms |
| 64-token prompt, 32-token eager decode | 107.84 tok/s |
| 64-token prompt, 32-token warm CUDA Graph decode | 238.55 tok/s |

Against the original first-light run on the same card and checkpoint, warm
CUDA-graph decode moved 195.49 -> 238.55 tok/s and the eager path 48.14 ->
107.84. Resident and peak allocation are unchanged to the megabyte. Two rows of
that first-light table are dropped rather than compared: weight load time is
dominated by page-cache state, and its "first prefill including warmup" was
measured by a different harness with a different notion of warmup.

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

Measured on Jetson AGX Thor, 20-token prompt, 128 generated tokens, one process
per point, best of five:

| | tok/s | vs plain |
|---|---:|---:|
| plain greedy | 100.35 | |
| speculative, K=1 | 105.22 | 1.09x |
| speculative, K=2 | **106.74** | 1.06x |

`K=2` is the operating point. Above it the window costs more than the extra
accepted tokens return: each additional verified row re-reads the routed
experts, which do not amortise across a window the way the dense weights do,
and each additional draft pays a full-vocabulary projection.

Enable it with `_load_mtp = True` on the frontend subclass; the window width is
the `k` argument to `generate_spec`. `FLASHRT_QWEN35MOE_VERIFY_K_ROWS=0` falls
back to verifying through the prefill forward.

## Limitations

- Text only; the vision tower is not loaded.
- The kernelized runtime NVFP4 path is required.
- Greedy decode only. Speculative decode is greedy as well: it emits the
  sequence plain greedy decoding would emit, token for token, or it is a bug.
- Only the BF16 source checkpoint with runtime NVFP4 conversion is supported.
- Sampling, batching, and beam search are not implemented.

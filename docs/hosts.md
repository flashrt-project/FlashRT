# Which host you have, which door to use, what to expect

This page answers three questions in order: *does this layer apply to my
host*, *what do I call*, and *what should I see afterwards*. For the
mechanism behind any of it see [`structures.md`](structures.md); for the
serving engines specifically see
[`serving_engines.md`](serving_engines.md).

## What the layer does

It replaces sub-blocks of a model you did not write with qualified
implementations, in place, without forking the host. It does not
reimplement the model, does not own the loop, and does not require the
host to know it exists. The host keeps its weights loading, its memory
planning, its graph capture and its scheduler.

Two consequences worth stating up front, because they set expectations
better than any number:

- **A seam it cannot claim stays with the host, exactly.** Not
  approximately — the host's own module runs, so that region is
  bit-identical to not using this at all. "Refused" is an outcome the
  layer is built to produce, not a failure it is trying to avoid.
- **Kernels come from the kernel hub, per host.** The distribution is
  pure Python. What is available to you depends on your architecture and
  your torch version, which is why the same call gives different seat
  counts on different machines and why the refusal trail matters more
  than the speedup.

## Which door

| your host | door | what it seats |
|---|---|---|
| a model library tree you can reach (`transformers`-style) | `structures.attach(model, forward)` | whatever discovery claims: projections, feed-forwards, norms, attention, gated-delta |
| the same, but you want to choose the seams yourself | `structures.get(name)` + `swap.attach` | exactly what you list |
| a diffusion transformer (`diffusers`-style) | `structures.attach`, adapters recognise the attention and feed-forward shapes | attention, feed-forward, modulation chains |
| **vLLM** | `vllm_engine.install_load_hook()` | dense projections, routed expert banks, LM head |
| **SGLang** | `sglang_engine.install()` | dense projections |
| an already-quantized checkpoint | `structures.adopt_prequantized(model, fmt=...)` | converts packed projections into structure impls |
| a decode loop you want captured whole | `structures.decode_loop(...)` | the step, as one graphed region |

`structures.explain(plan)` prints what a pass decided, including every
refusal, before anything is swapped. On an unfamiliar host that is the
first thing to run.

## The three states, and how to tell them apart

Almost every confusing result is one of these, and they need different
fixes. Distinguishing them costs one line each.

| state | what you see | what it means |
|---|---|---|
| **seated** | `N seats, M refused` in the log; `handle.summary()["seams"] > 0` | working; read `M` before reading the speedup |
| **refused** | seats reported, `refused` non-empty, each with a reason | the layer looked and declined — a coverage or boundary fact, not a bug |
| **never ran** | no log line at all; `handle` never created | the door did not fire. On vLLM this is the start-method trap; see [`serving_engines.md`](serving_engines.md) |

The third is the dangerous one because it looks like "no benefit". It
produces no error, and a strict mode cannot catch it, because the code
that would be strict never executes. Assert that the door fired.

## Reading a refusal

Refusals name the form and the shape. They are not "this cannot be
bound"; they are "this shape, declined for this reason".

```python
handle.notes["refused"]                       # (site, reason) per seat
from flash_rt.structures import impls
impls.unavailable_report()                    # per kernel package
```

Common ones and what each is telling you:

| reason | what to change |
|---|---|
| the kernel client is not installed | `pip install 'flash-rt[hub]'` |
| no build variant for this host | usually the torch version, not the package — see the README |
| `does not match system CPU` | the resolved *revision* has no build for your architecture, even if a newer one does |
| a required slot is absent on this host | the structure needs a weight this host does not carry; a binding can supply it |
| the boundary includes a norm / a mask form | the host's region is genuinely outside the declared boundary |

A large refused count concentrated on one seam family is a coverage
report. If it is the family that carries your gain — expert banks on a
routed-MoE host, for instance — the number you measure afterwards is
about that, not about the layer.

## What to expect

Measured, on the hosts named. These are receipts from specific machines,
not promises about yours: the band, the gates and the refusal trail exist
precisely so that your host decides rather than this table.

| host | model | measured |
|---|---|---|
| vLLM, consumer GPU (sm_120) | Qwen3-8B | 99.4 → 157.0 tok/s, 144 seats + 5 head slabs |
| SGLang, consumer GPU (sm_120) | Qwen3-8B | 101.0 → 201.5 tok/s, 144 seats |
| vLLM, edge module (sm_110) | 35B-A3B MoE | 37.2 → 76.0 tok/s, 200 seats + 4 head slabs |

Single-stream steady decode. Concurrency on the edge module with the MoE
model: **2.05× / 2.31× / 2.46× / 1.65×** at 1 / 4 / 8 / 16 — peaking at 8
and still positive at 16, which is not what a dense model does.

For a worked model-library example with its own receipts, see
[`adopt_in_20_lines.md`](adopt_in_20_lines.md).

Three things shape whether you see numbers like these:

- **Batch.** A dense host's gain thins as the batch grows — one weight
  read serves every row. A routed-MoE host's does not, because each
  token picks its own experts. Expect different curves, and do not carry
  a number from one family to the other.
- **Which seams got claimed.** More seats is not better. Small
  projections can cost more in per-operator overhead than they save in
  bytes; large sparse ones are where the gain is. A pass that claims 160
  extra seams and gets slower is a real, observed outcome.
- **What your host already does well.** This layer is worth most where
  per-operator fixed costs dominate — small batch, latency-sensitive,
  edge and consumer parts. At large batch the host's own tuned path is
  usually already in its element.

## What it does not do

- It is not a replacement for the engine, the scheduler or the loop.
- It does not tune for you: the band constants and the ladder order are
  measured defaults, and a host that differs should re-measure them
  rather than inherit them.
- It does not ship kernels. The pure-Python distribution carries
  specifications, implementations and adapters; the compiled halves come
  from the kernel hub, or from a local build for the native path.

## Verifying on your own host

Nothing here needs a benchmark harness to check:

```python
import flash_rt.structures as S
S.list_structures()          # the catalog this install carries
plan = S.auto_swaps(model, forward)
print(S.explain(plan))       # what would be claimed, and every refusal
```

Then attach, read `summary()` and `notes["refused"]`, and compare a
paired run — same process, same prompts, prefill and steady decode
reported separately. An end-to-end average hides which of the two moved.
